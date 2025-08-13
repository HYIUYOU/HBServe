"""
Mooncake backend for KV transfer
"""
import asyncio
import torch
import logging
import json
import struct
import pickle
from typing import Dict, Any, Optional
import zmq.asyncio

from .base import KVTransferBase, KVBuffer, KVLookupBuffer


class MooncakeKVTransfer(KVTransferBase):
    """基于Mooncake的KV传输实现"""
    
    def __init__(self, config: Dict[str, Any], rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(f"cuda:{rank}")
        
        # Mooncake配置
        self.prefill_url = config.get("prefill_url", "127.0.0.1:12346")
        self.decode_url = config.get("decode_url", "127.0.0.1:12347") 
        self.metadata_server = config.get("metadata_server", "127.0.0.1:12348")
        self.protocol = config.get("protocol", "tcp")
        self.device_name = config.get("device_name", "")
        
        # 查找缓冲区
        self.lookup_buffer = KVLookupBuffer(max_buffer_size=config.get("max_buffer_size", 100))
        
        # 传输状态跟踪
        self.pending_transfers: Dict[str, Dict[str, Any]] = {}
        self.completed_transfers: Dict[str, bool] = {}
        
        # Mooncake引擎和ZMQ连接
        self.transfer_engine = None
        self.zmq_context = None
        self.sender_socket = None
        self.receiver_socket = None
        
    async def initialize(self):
        """初始化Mooncake后端"""
        try:
            # 导入Mooncake模块
            try:
                from mooncake.engine import TransferEngine
                self.transfer_engine = TransferEngine()
            except ImportError as e:
                self.logger.error("无法导入Mooncake，请按照说明安装Mooncake")
                raise ImportError(
                    "请按照 https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                    "的说明安装Mooncake以运行Mooncake KV传输") from e
                    
            # 初始化Mooncake引擎
            if self.rank == 0:  # prefill instance
                local_url = self.prefill_url
                remote_url = self.decode_url
            else:  # decode instance  
                local_url = self.decode_url
                remote_url = self.prefill_url
                
            # 配置Mooncake引擎
            self.transfer_engine.initialize(
                local_url=local_url,
                metadata_server=self.metadata_server,
                protocol=self.protocol,
                device_name=self.device_name
            )
            
            # 初始化ZMQ连接（用于元数据传输）
            self.zmq_context = zmq.asyncio.Context()
            
            if self.rank == 0:  # prefill实例作为sender
                self.sender_socket = self.zmq_context.socket(zmq.PUSH)
                sender_port = 15000 + self.rank
                self.sender_socket.bind(f"tcp://*:{sender_port}")
            else:  # decode实例作为receiver
                self.receiver_socket = self.zmq_context.socket(zmq.PULL)
                receiver_port = 15000 + 0  # 连接到prefill实例
                self.receiver_socket.connect(f"tcp://127.0.0.1:{receiver_port}")
                
            # 设置CUDA设备
            torch.cuda.set_device(self.device)
            
            self.is_initialized = True
            self.logger.info(f"Mooncake KV传输后端初始化完成，rank: {self.rank}")
            
        except Exception as e:
            self.logger.error(f"Mooncake初始化失败: {e}")
            raise
            
    async def close(self):
        """关闭Mooncake后端"""
        try:
            if self.transfer_engine:
                self.transfer_engine.close()
                
            if self.sender_socket:
                self.sender_socket.close()
            if self.receiver_socket:
                self.receiver_socket.close()
            if self.zmq_context:
                self.zmq_context.term()
                
            self.lookup_buffer.clear()
            self.pending_transfers.clear()
            self.completed_transfers.clear()
            
            self.is_initialized = False
            self.logger.info("Mooncake KV传输后端已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭Mooncake后端失败: {e}")
            
    async def send_kv_cache(self, 
                           request_id: str,
                           kv_buffer: KVBuffer,
                           target_rank: int) -> bool:
        """发送KV缓存"""
        if not self.is_initialized:
            raise RuntimeError("Mooncake后端未初始化")
            
        if not self.validate_kv_buffer(kv_buffer):
            self.logger.error(f"无效的KV缓存格式: {request_id}")
            return False
            
        try:
            # 将KV缓存移到GPU
            kv_buffer = kv_buffer.to(self.device)
            
            # 注册GPU内存到Mooncake引擎
            key_ptr = kv_buffer.key_cache.data_ptr()
            value_ptr = kv_buffer.value_cache.data_ptr()
            key_size = kv_buffer.key_cache.numel() * kv_buffer.key_cache.element_size()
            value_size = kv_buffer.value_cache.numel() * kv_buffer.value_cache.element_size()
            
            self.transfer_engine.register(key_ptr, key_size)
            self.transfer_engine.register(value_ptr, value_size)
            
            # 创建Mooncake会话
            session_id = f"{request_id}_{self.rank}_{target_rank}"
            
            # 准备传输列表
            transfer_blocks = [
                (key_ptr, 0, key_size),    # (src_addr, dst_offset, size)
                (value_ptr, key_size, value_size)  # value紧跟在key后面
            ]
            
            # 执行数据传输
            status = await self._transfer_data_async(session_id, transfer_blocks)
            
            if status == 0:  # 成功
                # 发送元数据
                await self._send_metadata(request_id, kv_buffer)
                
                self.completed_transfers[request_id] = True
                self.logger.info(f"成功发送KV缓存: {request_id} -> rank {target_rank}")
                return True
            else:
                self.logger.error(f"Mooncake传输失败，状态码: {status}")
                self.completed_transfers[request_id] = False
                return False
                
        except Exception as e:
            self.logger.error(f"发送KV缓存失败 {request_id}: {e}")
            self.completed_transfers[request_id] = False
            return False
            
    async def receive_kv_cache(self, 
                              request_id: str,
                              source_rank: int) -> Optional[KVBuffer]:
        """接收KV缓存"""
        if not self.is_initialized:
            raise RuntimeError("Mooncake后端未初始化")
            
        # 先检查查找缓冲区
        kv_buffer = self.lookup_buffer.lookup(request_id)
        if kv_buffer is not None:
            return self.lookup_buffer.remove(request_id)
            
        try:
            # 接收元数据
            metadata = await self._receive_metadata()
            if not metadata:
                return None
                
            received_request_id = metadata["request_id"]
            
            # 根据元数据分配内存
            key_shape = metadata["key_shape"]
            value_shape = metadata["value_shape"]
            dtype = getattr(torch, metadata["dtype"].split('.')[-1])
            
            key_cache = torch.zeros(key_shape, dtype=dtype, device=self.device)
            value_cache = torch.zeros(value_shape, dtype=dtype, device=self.device)
            
            # 注册接收缓冲区
            key_ptr = key_cache.data_ptr()
            value_ptr = value_cache.data_ptr()
            key_size = key_cache.numel() * key_cache.element_size()
            value_size = value_cache.numel() * value_cache.element_size()
            
            self.transfer_engine.register(key_ptr, key_size)
            self.transfer_engine.register(value_ptr, value_size)
            
            # 等待Mooncake传输完成
            session_id = f"{received_request_id}_{source_rank}_{self.rank}"
            await self._wait_for_transfer(session_id)
            
            # 重建其他张量
            token_ids = torch.tensor(metadata["token_ids"], dtype=torch.long, device=self.device)
            sequence_lengths = torch.tensor(metadata["sequence_lengths"], dtype=torch.long, device=self.device)
            
            block_indices = None
            if metadata.get("has_block_indices", False):
                block_indices = torch.tensor(metadata["block_indices"], dtype=torch.long, device=self.device)
                
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
            "max_buffer_size": self.lookup_buffer.max_buffer_size,
            "backend": "mooncake"
        }
        
    async def _transfer_data_async(self, session_id: str, transfer_blocks: list) -> int:
        """异步执行Mooncake数据传输"""
        loop = asyncio.get_event_loop()
        
        def _sync_transfer():
            try:
                src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
                return self.transfer_engine.batch_transfer_sync(
                    session_id, list(src_addrs), list(dst_addrs), list(lengths)
                )
            except Exception as e:
                self.logger.error(f"Mooncake传输异常: {e}")
                return -1
                
        # 在线程池中执行同步传输
        return await loop.run_in_executor(None, _sync_transfer)
        
    async def _send_metadata(self, request_id: str, kv_buffer: KVBuffer):
        """发送元数据"""
        if not self.sender_socket:
            return
            
        metadata = {
            "request_id": request_id,
            "key_shape": list(kv_buffer.key_cache.shape),
            "value_shape": list(kv_buffer.value_cache.shape),
            "dtype": str(kv_buffer.key_cache.dtype),
            "token_ids": kv_buffer.token_ids.cpu().tolist(),
            "sequence_lengths": kv_buffer.sequence_lengths.cpu().tolist(),
            "has_block_indices": kv_buffer.block_indices is not None
        }
        
        if kv_buffer.block_indices is not None:
            metadata["block_indices"] = kv_buffer.block_indices.cpu().tolist()
            
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        await self.sender_socket.send(metadata_bytes)
        
    async def _receive_metadata(self) -> Optional[Dict[str, Any]]:
        """接收元数据"""
        if not self.receiver_socket:
            return None
            
        try:
            # 非阻塞接收，超时返回None
            metadata_bytes = await asyncio.wait_for(
                self.receiver_socket.recv(), 
                timeout=0.1
            )
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            return metadata
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.warning(f"接收元数据失败: {e}")
            return None
            
    async def _wait_for_transfer(self, session_id: str, timeout: float = 30.0):
        """等待Mooncake传输完成"""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"传输超时: {session_id}")
                
            # 检查传输状态（这里需要Mooncake提供状态查询接口）
            # 暂时使用简单的等待
            await asyncio.sleep(0.01)
            
            # 假设传输已完成（实际需要查询Mooncake状态）
            break
