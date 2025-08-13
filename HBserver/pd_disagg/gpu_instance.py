"""
GPU实例，负责实际的模型推理和KV缓存管理
"""
import asyncio
import torch
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from hbserve.config import Config
from hbserve.engine.model_runner import ModelRunner
from hbserve.engine.sequence import Sequence, SequenceStatus
from hbserve.sampling_params import SamplingParams

from .config import PdDisaggConfig, InstanceType
from .kv_transfer import KVTransferManager, KVBuffer
from .rpc_client import RpcServer, RpcMessage
from .chunked_prefill import ChunkedPrefillManager


class InstanceStatus(Enum):
    """实例状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class GpuRequest:
    """GPU实例上的请求"""
    request_id: str
    token_ids: List[int]
    sampling_params: SamplingParams
    sequence: Optional[Sequence] = None
    status: str = "pending"
    kv_cache_key: Optional[str] = None
    completion_tokens: List[int] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)
    
    @property
    def is_finished(self) -> bool:
        return self.status in ["completed", "failed"]


class MessageHandler:
    """RPC消息处理器"""
    
    def __init__(self, gpu_instance: 'GpuInstance'):
        self.gpu_instance = gpu_instance
        self.logger = logging.getLogger(__name__)
        
    async def handle_message(self, message: RpcMessage) -> Dict[str, Any]:
        """处理RPC消息"""
        try:
            handler_method = getattr(self, f"_handle_{message.message_type}", None)
            if handler_method:
                result = await handler_method(message.data)
                return {"status": "success", "data": result}
            else:
                return {"status": "error", "data": {"error": f"未知消息类型: {message.message_type}"}}
                
        except Exception as e:
            self.logger.error(f"处理消息失败 {message.message_type}: {e}")
            return {"status": "error", "data": {"error": str(e)}}
            
    async def _handle_get_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取实例状态"""
        return self.gpu_instance.get_status()
        
    async def _handle_prefill(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理prefill请求"""
        request_id = data["request_id"]
        token_ids = data["token_ids"]
        sampling_params = data["sampling_params"]
        
        result = await self.gpu_instance.prefill(request_id, token_ids, sampling_params)
        return result
        
    async def _handle_transfer_kv_cache(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理KV缓存传输请求"""
        request_id = data["request_id"]
        target_instance = data["target_instance"]
        
        result = await self.gpu_instance.transfer_kv_cache(request_id, target_instance)
        return result
        
    async def _handle_get_transfer_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取传输状态"""
        request_id = data["request_id"]
        
        result = await self.gpu_instance.get_transfer_status(request_id)
        return result
        
    async def _handle_start_decode(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """开始decode"""
        request_id = data["request_id"]
        sampling_params = data["sampling_params"]
        
        result = await self.gpu_instance.start_decode(request_id, sampling_params)
        return result
        
    async def _handle_get_request_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取请求状态"""
        request_id = data["request_id"]
        
        result = self.gpu_instance.get_request_status(request_id)
        return result
        
    async def _handle_chunked_prefill(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理分块prefill请求"""
        request_id = data["request_id"]
        token_ids = data["token_ids"]
        chunk_size = data["chunk_size"]
        sampling_params = data["sampling_params"]
        
        result = await self.gpu_instance.chunked_prefill(
            request_id, token_ids, chunk_size, sampling_params
        )
        return result
        
    async def _handle_get_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取统计信息"""
        return self.gpu_instance.get_stats()


class GpuInstance:
    """GPU实例"""
    
    def __init__(self, config: PdDisaggConfig, instance_config: Dict[str, Any]):
        self.config = config
        self.instance_config = instance_config
        self.instance_id = instance_config["instance_id"]
        self.instance_type = instance_config["instance_type"]
        self.gpu_id = instance_config["gpu_id"]
        self.host = instance_config["host"]
        self.port = instance_config["port"]
        self.kv_rank = instance_config.get("kv_rank", 0)
        
        self.logger = logging.getLogger(__name__)
        self.status = InstanceStatus.INITIALIZING
        
        # 设置CUDA设备
        self.device = torch.device(f"cuda:{self.gpu_id}")
        torch.cuda.set_device(self.device)
        
        # 模型相关
        self.model_runner = None
        self.nano_config = None
        
        # 请求管理
        self.active_requests: Dict[str, GpuRequest] = {}
        self.request_queue: List[str] = []
        
        # KV传输管理器
        self.kv_transfer_manager = KVTransferManager(config, self.kv_rank)
        
        # 分块prefill管理器
        self.chunked_prefill_manager = None
        if config.chunked_prefill_enabled:
            self.chunked_prefill_manager = ChunkedPrefillManager(config)
        
        # RPC服务器
        self.message_handler = MessageHandler(self)
        self.rpc_server = RpcServer(self.host, self.port, self.message_handler)
        
        # 统计信息
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        
    async def start(self):
        """启动GPU实例"""
        try:
            self.logger.info(f"启动GPU实例 {self.instance_id}")
            
            # 初始化模型
            await self._init_model()
            
            # 初始化KV传输
            await self.kv_transfer_manager.initialize()
            
            # 初始化分块prefill管理器
            if self.chunked_prefill_manager:
                await self.chunked_prefill_manager.initialize(self.model_runner)
            
            # 启动RPC服务器
            asyncio.create_task(self.rpc_server.start())
            
            # 启动处理循环
            asyncio.create_task(self._processing_loop())
            
            self.status = InstanceStatus.READY
            self.logger.info(f"GPU实例 {self.instance_id} 启动完成")
            
        except Exception as e:
            self.status = InstanceStatus.ERROR
            self.logger.error(f"启动GPU实例失败: {e}")
            raise
            
    async def stop(self):
        """停止GPU实例"""
        try:
            self.logger.info(f"停止GPU实例 {self.instance_id}")
            
            # 停止RPC服务器
            await self.rpc_server.stop()
            
            # 关闭KV传输
            await self.kv_transfer_manager.close()
            
            # 清理资源
            self.active_requests.clear()
            
            self.status = InstanceStatus.ERROR
            self.logger.info(f"GPU实例 {self.instance_id} 已停止")
            
        except Exception as e:
            self.logger.error(f"停止GPU实例失败: {e}")
            
    async def _init_model(self):
        """初始化模型"""
        # 创建hbserve配置
        self.nano_config = Config(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens
        )
        
        # 初始化模型运行器
        self.model_runner = ModelRunner(self.nano_config, self.gpu_id, [])
        
        self.logger.info(f"模型初始化完成: {self.config.model_path}")
        
    def get_status(self) -> Dict[str, Any]:
        """获取实例状态"""
        available = (self.status == InstanceStatus.READY and 
                    len(self.active_requests) < self.config.max_num_seqs)
        
        return {
            "instance_id": self.instance_id,
            "instance_type": self.instance_type.value,
            "status": self.status.value,
            "available": available,
            "active_requests": len(self.active_requests),
            "device": str(self.device),
            "gpu_id": self.gpu_id
        }
        
    async def prefill(self, request_id: str, token_ids: List[int], 
                     sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行prefill"""
        try:
            if request_id in self.active_requests:
                return {"success": False, "error": "请求已存在"}
                
            # 创建请求
            sp = SamplingParams(**sampling_params)
            request = GpuRequest(
                request_id=request_id,
                token_ids=token_ids,
                sampling_params=sp
            )
            
            # 创建序列
            request.sequence = Sequence(token_ids, sp)
            request.status = "prefilling"
            
            self.active_requests[request_id] = request
            self.request_queue.append(request_id)
            self.total_requests += 1
            
            # 如果启用了分块prefill，使用分块处理
            if (self.chunked_prefill_manager and 
                len(token_ids) > self.config.chunk_size):
                
                kv_cache_key = await self.chunked_prefill_manager.chunked_prefill(
                    request.sequence
                )
            else:
                # 标准prefill
                kv_cache_key = await self._standard_prefill(request.sequence)
                
            request.kv_cache_key = kv_cache_key
            request.status = "prefill_completed"
            
            self.logger.info(f"Prefill完成: {request_id}")
            
            return {
                "success": True,
                "kv_cache_key": kv_cache_key,
                "request_id": request_id
            }
            
        except Exception as e:
            self.logger.error(f"Prefill失败 {request_id}: {e}")
            if request_id in self.active_requests:
                self.active_requests[request_id].status = "failed"
            return {"success": False, "error": str(e)}
            
    async def _standard_prefill(self, sequence: Sequence) -> str:
        """标准prefill处理"""
        # 调用模型运行器执行prefill
        # 这里需要适配hbserve的接口
        seqs = [sequence]
        is_prefill = True
        
        # 调用模型推理
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # 生成KV缓存键
        kv_cache_key = f"kv_{sequence.seq_id}_{int(time.time() * 1000)}"
        
        return kv_cache_key
        
    async def transfer_kv_cache(self, request_id: str, 
                               target_instance: str) -> Dict[str, Any]:
        """传输KV缓存"""
        try:
            if request_id not in self.active_requests:
                return {"success": False, "error": "请求不存在"}
                
            request = self.active_requests[request_id]
            if not request.kv_cache_key:
                return {"success": False, "error": "KV缓存未准备好"}
                
            # 创建KV缓存对象（简化版）
            # 实际需要从模型runner中提取真实的KV缓存
            kv_buffer = self._create_kv_buffer(request)
            
            # 发送KV缓存
            success = await self.kv_transfer_manager.send_kv_cache(
                request_id, kv_buffer, target_instance
            )
            
            if success:
                request.status = "transferred"
                self.logger.info(f"KV缓存传输成功: {request_id} -> {target_instance}")
            
            return {"success": success}
            
        except Exception as e:
            self.logger.error(f"KV缓存传输失败 {request_id}: {e}")
            return {"success": False, "error": str(e)}
            
    def _create_kv_buffer(self, request: GpuRequest) -> KVBuffer:
        """创建KV缓存对象（简化版）"""
        # 这里需要从实际的模型状态中提取KV缓存
        # 为了演示，创建模拟的KV缓存
        batch_size = 1
        num_layers = 12  # 假设12层
        num_heads = 8
        seq_len = len(request.token_ids)
        head_dim = 64
        
        key_cache = torch.randn(
            batch_size, num_layers, num_heads, seq_len, head_dim,
            device=self.device, dtype=torch.float16
        )
        value_cache = torch.randn(
            batch_size, num_layers, num_heads, seq_len, head_dim,
            device=self.device, dtype=torch.float16
        )
        
        token_ids = torch.tensor(request.token_ids, device=self.device, dtype=torch.long)
        sequence_lengths = torch.tensor([seq_len], device=self.device, dtype=torch.long)
        
        return KVBuffer(
            key_cache=key_cache,
            value_cache=value_cache,
            token_ids=token_ids,
            sequence_lengths=sequence_lengths
        )
        
    async def get_transfer_status(self, request_id: str) -> Dict[str, Any]:
        """获取传输状态"""
        status = await self.kv_transfer_manager.check_transfer_status(request_id)
        
        # 检查是否接收到KV缓存
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            if request.status == "waiting_kv" and status.get("status") == "completed":
                request.status = "ready"
                return {"status": "ready", "request_id": request_id}
                
        return status
        
    async def start_decode(self, request_id: str, 
                          sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """开始decode"""
        try:
            # 首先尝试从KV传输中接收缓存
            kv_buffer = await self.kv_transfer_manager.receive_kv_cache(
                request_id, source_instance_id="prefill-0"  # 简化：假设来自prefill-0
            )
            
            if not kv_buffer:
                # 标记为等待KV缓存
                if request_id not in self.active_requests:
                    sp = SamplingParams(**sampling_params)
                    request = GpuRequest(
                        request_id=request_id,
                        token_ids=[],  # decode阶段token_ids由KV缓存确定
                        sampling_params=sp,
                        status="waiting_kv"
                    )
                    self.active_requests[request_id] = request
                    
                return {"success": True, "status": "waiting_kv"}
                
            # 创建decode请求
            sp = SamplingParams(**sampling_params)
            request = GpuRequest(
                request_id=request_id,
                token_ids=kv_buffer.token_ids.cpu().tolist(),
                sampling_params=sp,
                status="decoding"
            )
            
            # 从KV缓存恢复序列
            request.sequence = Sequence(request.token_ids, sp)
            
            self.active_requests[request_id] = request
            self.request_queue.append(request_id)
            
            self.logger.info(f"开始decode: {request_id}")
            
            return {"success": True, "status": "decoding"}
            
        except Exception as e:
            self.logger.error(f"开始decode失败 {request_id}: {e}")
            return {"success": False, "error": str(e)}
            
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """获取请求状态"""
        if request_id not in self.active_requests:
            return {"status": "not_found"}
            
        request = self.active_requests[request_id]
        
        return {
            "status": request.status,
            "request_id": request_id,
            "completion_tokens": request.completion_tokens,
            "new_tokens": [],  # 这里应该返回新生成的token
            "is_finished": request.is_finished
        }
        
    async def chunked_prefill(self, request_id: str, token_ids: List[int],
                             chunk_size: int, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """分块prefill处理"""
        try:
            if not self.chunked_prefill_manager:
                return await self.prefill(request_id, token_ids, sampling_params)
                
            # 创建序列
            sp = SamplingParams(**sampling_params)
            sequence = Sequence(token_ids, sp)
            
            # 执行分块prefill
            kv_cache_key = await self.chunked_prefill_manager.chunked_prefill(sequence)
            
            # 创建请求记录
            request = GpuRequest(
                request_id=request_id,
                token_ids=token_ids,
                sampling_params=sp,
                sequence=sequence,
                kv_cache_key=kv_cache_key,
                status="prefill_completed"
            )
            
            self.active_requests[request_id] = request
            self.total_requests += 1
            
            self.logger.info(f"分块prefill完成: {request_id}")
            
            return {
                "success": True,
                "kv_cache_key": kv_cache_key,
                "request_id": request_id,
                "chunks_processed": len(token_ids) // chunk_size + 1
            }
            
        except Exception as e:
            self.logger.error(f"分块prefill失败 {request_id}: {e}")
            return {"success": False, "error": str(e)}
            
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        kv_stats = self.kv_transfer_manager.get_buffer_stats()
        
        return {
            "instance_id": self.instance_id,
            "instance_type": self.instance_type.value,
            "status": self.status.value,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "active_requests": len(self.active_requests),
            "device": str(self.device),
            "kv_transfer_stats": kv_stats
        }
        
    async def _processing_loop(self):
        """处理循环"""
        while self.status != InstanceStatus.ERROR:
            try:
                await self._process_requests()
                await asyncio.sleep(0.01)  # 10ms间隔
            except Exception as e:
                self.logger.error(f"处理循环异常: {e}")
                await asyncio.sleep(0.1)
                
    async def _process_requests(self):
        """处理请求队列"""
        # 清理已完成的请求
        to_remove = []
        for request_id, request in self.active_requests.items():
            if request.is_finished:
                to_remove.append(request_id)
                if request.status == "completed":
                    self.completed_requests += 1
                else:
                    self.failed_requests += 1
                    
        for request_id in to_remove:
            del self.active_requests[request_id]
            if request_id in self.request_queue:
                self.request_queue.remove(request_id)
                
        # 处理decode请求
        decode_requests = [
            req for req in self.active_requests.values()
            if req.status == "decoding"
        ]
        
        if decode_requests:
            await self._process_decode_requests(decode_requests)
            
    async def _process_decode_requests(self, decode_requests: List[GpuRequest]):
        """处理decode请求"""
        try:
            # 简化的decode处理
            for request in decode_requests:
                if request.sequence:
                    # 模拟token生成
                    new_token = 100  # 简化：生成固定token
                    request.completion_tokens.append(new_token)
                    
                    # 检查是否完成
                    if (len(request.completion_tokens) >= request.sampling_params.max_tokens or
                        new_token == self.nano_config.eos):
                        request.status = "completed"
                        
        except Exception as e:
            self.logger.error(f"处理decode请求失败: {e}")
            for request in decode_requests:
                request.status = "failed"
