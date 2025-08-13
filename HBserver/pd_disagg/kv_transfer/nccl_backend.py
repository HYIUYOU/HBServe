"""
NCCL backend for KV transfer
"""
import asyncio
import torch
import torch.distributed as dist
import logging
from typing import Dict, Any, Optional
import pickle
import struct

from .base import KVTransferBase, KVBuffer, KVLookupBuffer


class NcclKVTransfer(KVTransferBase):
    """基于NCCL的KV传输实现"""
    
    def __init__(self, config: Dict[str, Any], rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(f"cuda:{rank}")
        
        # 查找缓冲区
        self.lookup_buffer = KVLookupBuffer(max_buffer_size=config.get("max_buffer_size", 100))
        
        # 传输状态跟踪
        self.pending_transfers: Dict[str, Dict[str, Any]] = {}
        self.completed_transfers: Dict[str, bool] = {}
        
        # 通信组
        self.process_group = None
        
    async def initialize(self):
        """初始化NCCL后端"""
        try:
            # 初始化分布式环境
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.get("backend", "nccl"),
                    init_method=self.config.get("init_method", "tcp://127.0.0.1:12349"),
                    world_size=self.world_size,
                    rank=self.rank
                )
                
            # 创建通信组
            self.process_group = dist.new_group(list(range(self.world_size)))
            
            # 设置CUDA设备
            torch.cuda.set_device(self.device)
            
            self.is_initialized = True
            self.logger.info(f"NCCL KV传输后端初始化完成，rank: {self.rank}")
            
        except Exception as e:
            self.logger.error(f"NCCL初始化失败: {e}")
            raise
            
    async def close(self):
        """关闭NCCL后端"""
        try:
            if self.is_initialized and dist.is_initialized():
                dist.destroy_process_group(self.process_group)
                
            self.lookup_buffer.clear()
            self.pending_transfers.clear()
            self.completed_transfers.clear()
            
            self.is_initialized = False
            self.logger.info("NCCL KV传输后端已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭NCCL后端失败: {e}")
            
    async def send_kv_cache(self, 
                           request_id: str,
                           kv_buffer: KVBuffer,
                           target_rank: int) -> bool:
        """发送KV缓存"""
        if not self.is_initialized:
            raise RuntimeError("NCCL后端未初始化")
            
        if not self.validate_kv_buffer(kv_buffer):
            self.logger.error(f"无效的KV缓存格式: {request_id}")
            return False
            
        try:
            # 将KV缓存移到GPU
            kv_buffer = kv_buffer.to(self.device)
            
            # 序列化元数据
            metadata = {
                "request_id": request_id,
                "key_shape": list(kv_buffer.key_cache.shape),
                "value_shape": list(kv_buffer.value_cache.shape),
                "token_ids_shape": list(kv_buffer.token_ids.shape),
                "sequence_lengths_shape": list(kv_buffer.sequence_lengths.shape),
                "dtype": str(kv_buffer.key_cache.dtype),
                "has_block_indices": kv_buffer.block_indices is not None
            }
            
            if kv_buffer.block_indices is not None:
                metadata["block_indices_shape"] = list(kv_buffer.block_indices.shape)
                
            metadata_bytes = pickle.dumps(metadata)
            metadata_size = len(metadata_bytes)
            
            # 发送元数据大小
            size_tensor = torch.tensor([metadata_size], dtype=torch.int64, device=self.device)
            dist.send(size_tensor, dst=target_rank, group=self.process_group)
            
            # 发送元数据
            metadata_tensor = torch.frombuffer(metadata_bytes, dtype=torch.uint8).to(self.device)
            dist.send(metadata_tensor, dst=target_rank, group=self.process_group)
            
            # 发送KV缓存数据
            dist.send(kv_buffer.key_cache, dst=target_rank, group=self.process_group)
            dist.send(kv_buffer.value_cache, dst=target_rank, group=self.process_group)
            dist.send(kv_buffer.token_ids, dst=target_rank, group=self.process_group)
            dist.send(kv_buffer.sequence_lengths, dst=target_rank, group=self.process_group)
            
            if kv_buffer.block_indices is not None:
                dist.send(kv_buffer.block_indices, dst=target_rank, group=self.process_group)
                
            # 标记传输完成
            self.completed_transfers[request_id] = True
            
            self.logger.info(f"成功发送KV缓存: {request_id} -> rank {target_rank}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送KV缓存失败 {request_id}: {e}")
            self.completed_transfers[request_id] = False
            return False
            
    async def receive_kv_cache(self, 
                              request_id: str,
                              source_rank: int) -> Optional[KVBuffer]:
        """接收KV缓存"""
        if not self.is_initialized:
            raise RuntimeError("NCCL后端未初始化")
            
        # 先检查查找缓冲区
        kv_buffer = self.lookup_buffer.lookup(request_id)
        if kv_buffer is not None:
            return self.lookup_buffer.remove(request_id)
            
        try:
            # 接收元数据大小
            size_tensor = torch.zeros(1, dtype=torch.int64, device=self.device)
            dist.recv(size_tensor, src=source_rank, group=self.process_group)
            metadata_size = size_tensor.item()
            
            # 接收元数据
            metadata_tensor = torch.zeros(metadata_size, dtype=torch.uint8, device=self.device)
            dist.recv(metadata_tensor, src=source_rank, group=self.process_group)
            metadata_bytes = metadata_tensor.cpu().numpy().tobytes()
            metadata = pickle.loads(metadata_bytes)
            
            # 解析元数据
            received_request_id = metadata["request_id"]
            key_shape = metadata["key_shape"]
            value_shape = metadata["value_shape"]
            token_ids_shape = metadata["token_ids_shape"]
            sequence_lengths_shape = metadata["sequence_lengths_shape"]
            dtype = getattr(torch, metadata["dtype"].split('.')[-1])
            has_block_indices = metadata["has_block_indices"]
            
            # 接收KV缓存数据
            key_cache = torch.zeros(key_shape, dtype=dtype, device=self.device)
            value_cache = torch.zeros(value_shape, dtype=dtype, device=self.device)
            token_ids = torch.zeros(token_ids_shape, dtype=torch.long, device=self.device)
            sequence_lengths = torch.zeros(sequence_lengths_shape, dtype=torch.long, device=self.device)
            
            dist.recv(key_cache, src=source_rank, group=self.process_group)
            dist.recv(value_cache, src=source_rank, group=self.process_group)
            dist.recv(token_ids, src=source_rank, group=self.process_group)
            dist.recv(sequence_lengths, src=source_rank, group=self.process_group)
            
            block_indices = None
            if has_block_indices:
                block_indices_shape = metadata["block_indices_shape"]
                block_indices = torch.zeros(block_indices_shape, dtype=torch.long, device=self.device)
                dist.recv(block_indices, src=source_rank, group=self.process_group)
                
            # 创建KV缓存对象
            received_kv_buffer = KVBuffer(
                key_cache=key_cache,
                value_cache=value_cache,
                token_ids=token_ids,
                sequence_lengths=sequence_lengths,
                block_indices=block_indices
            )
            
            # 如果是目标请求，直接返回；否则存入查找缓冲区
            if received_request_id == request_id:
                self.logger.info(f"成功接收KV缓存: {request_id} <- rank {source_rank}")
                return received_kv_buffer
            else:
                self.lookup_buffer.insert(received_request_id, received_kv_buffer)
                self.logger.info(f"接收KV缓存并存入缓冲区: {received_request_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"接收KV缓存失败 {request_id}: {e}")
            return None
            
    async def check_transfer_status(self, request_id: str) -> Dict[str, Any]:
        """检查传输状态"""
        if request_id in self.completed_transfers:
            success = self.completed_transfers[request_id]
            return {
                "status": "completed" if success else "failed",
                "request_id": request_id,
                "success": success
            }
        elif request_id in self.pending_transfers:
            return {
                "status": "pending",
                "request_id": request_id,
                "success": False
            }
        else:
            return {
                "status": "not_found",
                "request_id": request_id,
                "success": False
            }
            
    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        return {
            "lookup_buffer_size": self.lookup_buffer.size(),
            "lookup_buffer_memory": self.lookup_buffer.get_memory_usage(),
            "pending_transfers": len(self.pending_transfers),
            "completed_transfers": len(self.completed_transfers),
            "max_buffer_size": self.lookup_buffer.max_buffer_size
        }
        
    def _start_background_receiver(self):
        """启动后台接收器（用于处理乱序到达的KV缓存）"""
        async def background_receiver():
            while self.is_initialized:
                try:
                    # 尝试接收任何来源的数据
                    for source_rank in range(self.world_size):
                        if source_rank == self.rank:
                            continue
                            
                        # 非阻塞检查是否有待接收的数据
                        # 这里需要使用NCCL的非阻塞接口
                        # 暂时使用简单的轮询方式
                        await asyncio.sleep(0.01)
                        
                except Exception as e:
                    self.logger.warning(f"后台接收器异常: {e}")
                    await asyncio.sleep(0.1)
                    
        # 启动后台任务
        asyncio.create_task(background_receiver())
