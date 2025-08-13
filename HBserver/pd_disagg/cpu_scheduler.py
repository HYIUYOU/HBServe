"""
CPU端的调度器，负责管理请求队列和分发任务
"""
import asyncio
import logging
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from hbserve.sampling_params import SamplingParams
from .config import PdDisaggConfig, InstanceType
from .rpc_client import RpcClient


class RequestStatus(Enum):
    """请求状态"""
    WAITING = "waiting"
    PREFILLING = "prefilling"
    TRANSFERRING = "transferring" 
    DECODING = "decoding"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class PdRequest:
    """pd分离架构中的请求"""
    request_id: str
    prompt: str
    token_ids: List[int]
    sampling_params: SamplingParams
    status: RequestStatus = RequestStatus.WAITING
    prefill_instance_id: Optional[str] = None
    decode_instance_id: Optional[str] = None
    kv_cache_key: Optional[str] = None
    created_time: float = field(default_factory=time.time)
    prefill_start_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    completion_tokens: List[int] = field(default_factory=list)
    
    @property
    def prompt_length(self) -> int:
        return len(self.token_ids)
    
    @property
    def completion_length(self) -> int:
        return len(self.completion_tokens)
    
    @property
    def is_finished(self) -> bool:
        return self.status in [RequestStatus.FINISHED, RequestStatus.FAILED]


class CpuScheduler:
    """CPU端调度器"""
    
    def __init__(self, config: PdDisaggConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 请求队列
        self.waiting_queue: deque[PdRequest] = deque()
        self.active_requests: Dict[str, PdRequest] = {}
        
        # RPC客户端
        self.rpc_clients: Dict[str, RpcClient] = {}
        self._init_rpc_clients()
        
        # 统计信息
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        
    def _init_rpc_clients(self):
        """初始化RPC客户端"""
        for instance_config in self.config.gpu_instances:
            instance_id = instance_config["instance_id"]
            host = instance_config["host"]
            port = instance_config["port"]
            
            client = RpcClient(
                instance_id=instance_id,
                host=host,
                port=port,
                timeout=self.config.rpc_timeout
            )
            self.rpc_clients[instance_id] = client
            
    async def start(self):
        """启动调度器"""
        self.logger.info("启动CPU调度器...")
        
        # 启动RPC客户端连接
        for client in self.rpc_clients.values():
            await client.connect()
            
        # 启动调度循环
        asyncio.create_task(self._schedule_loop())
        self.logger.info("CPU调度器启动完成")
        
    async def stop(self):
        """停止调度器"""
        self.logger.info("停止CPU调度器...")
        
        # 关闭RPC客户端
        for client in self.rpc_clients.values():
            await client.close()
            
        self.logger.info("CPU调度器已停止")
        
    def add_request(self, prompt: str, token_ids: List[int], 
                   sampling_params: SamplingParams) -> str:
        """添加新请求"""
        request_id = f"req_{self.total_requests}"
        self.total_requests += 1
        
        request = PdRequest(
            request_id=request_id,
            prompt=prompt,
            token_ids=token_ids,
            sampling_params=sampling_params
        )
        
        self.waiting_queue.append(request)
        self.active_requests[request_id] = request
        
        self.logger.info(f"添加请求 {request_id}, prompt长度: {request.prompt_length}")
        return request_id
        
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """获取请求状态"""
        if request_id not in self.active_requests:
            return None
            
        request = self.active_requests[request_id]
        return {
            "request_id": request_id,
            "status": request.status.value,
            "prompt_length": request.prompt_length,
            "completion_length": request.completion_length,
            "completion_tokens": request.completion_tokens,
            "prefill_instance": request.prefill_instance_id,
            "decode_instance": request.decode_instance_id
        }
        
    async def _schedule_loop(self):
        """主调度循环"""
        while True:
            try:
                await self._schedule_step()
                await asyncio.sleep(0.01)  # 10ms调度间隔
            except Exception as e:
                self.logger.error(f"调度循环异常: {e}")
                await asyncio.sleep(0.1)
                
    async def _schedule_step(self):
        """单步调度"""
        # 处理等待队列中的请求
        await self._schedule_waiting_requests()
        
        # 检查正在处理的请求状态
        await self._check_active_requests()
        
    async def _schedule_waiting_requests(self):
        """调度等待中的请求"""
        if not self.waiting_queue:
            return
            
        # 找到可用的prefill实例
        available_prefill_instances = await self._get_available_prefill_instances()
        
        while self.waiting_queue and available_prefill_instances:
            request = self.waiting_queue.popleft()
            instance_id = available_prefill_instances.pop(0)
            
            # 分配请求到prefill实例
            await self._assign_to_prefill(request, instance_id)
            
    async def _get_available_prefill_instances(self) -> List[str]:
        """获取可用的prefill实例"""
        available = []
        
        for instance_config in self.config.get_prefill_instances():
            instance_id = instance_config["instance_id"]
            client = self.rpc_clients[instance_id]
            
            try:
                # 检查实例状态
                status = await client.get_status()
                if status.get("available", False):
                    available.append(instance_id)
            except Exception as e:
                self.logger.warning(f"检查实例 {instance_id} 状态失败: {e}")
                
        return available
        
    async def _assign_to_prefill(self, request: PdRequest, instance_id: str):
        """分配请求到prefill实例"""
        try:
            request.prefill_instance_id = instance_id
            request.status = RequestStatus.PREFILLING
            request.prefill_start_time = time.time()
            
            client = self.rpc_clients[instance_id]
            
            # 发送prefill请求
            result = await client.prefill(
                request_id=request.request_id,
                token_ids=request.token_ids,
                sampling_params=request.sampling_params.__dict__
            )
            
            if result.get("success", False):
                request.kv_cache_key = result.get("kv_cache_key")
                self.logger.info(f"请求 {request.request_id} 开始prefill")
            else:
                request.status = RequestStatus.FAILED
                self.logger.error(f"请求 {request.request_id} prefill失败")
                
        except Exception as e:
            request.status = RequestStatus.FAILED
            self.logger.error(f"分配请求 {request.request_id} 到prefill实例失败: {e}")
            
    async def _check_active_requests(self):
        """检查活跃请求的状态"""
        to_remove = []
        
        for request_id, request in self.active_requests.items():
            if request.is_finished:
                continue
                
            try:
                if request.status == RequestStatus.PREFILLING:
                    await self._check_prefill_status(request)
                elif request.status == RequestStatus.TRANSFERRING:
                    await self._check_transfer_status(request)
                elif request.status == RequestStatus.DECODING:
                    await self._check_decode_status(request)
                    
                # 清理已完成的请求
                if request.is_finished:
                    to_remove.append(request_id)
                    if request.status == RequestStatus.FINISHED:
                        self.completed_requests += 1
                    else:
                        self.failed_requests += 1
                        
            except Exception as e:
                self.logger.error(f"检查请求 {request_id} 状态失败: {e}")
                request.status = RequestStatus.FAILED
                to_remove.append(request_id)
                
        # 移除已完成的请求
        for request_id in to_remove:
            del self.active_requests[request_id]
            
    async def _check_prefill_status(self, request: PdRequest):
        """检查prefill状态"""
        if not request.prefill_instance_id:
            return
            
        client = self.rpc_clients[request.prefill_instance_id]
        status = await client.get_request_status(request.request_id)
        
        if status.get("status") == "completed":
            # Prefill完成，开始KV传输
            request.status = RequestStatus.TRANSFERRING
            await self._start_kv_transfer(request)
        elif status.get("status") == "failed":
            request.status = RequestStatus.FAILED
            
    async def _start_kv_transfer(self, request: PdRequest):
        """开始KV缓存传输"""
        # 选择decode实例
        decode_instances = self.config.get_decode_instances()
        if not decode_instances:
            request.status = RequestStatus.FAILED
            self.logger.error("没有可用的decode实例")
            return
            
        # 简单选择第一个decode实例
        decode_instance = decode_instances[0]
        request.decode_instance_id = decode_instance["instance_id"]
        
        try:
            # 通知prefill实例传输KV缓存
            prefill_client = self.rpc_clients[request.prefill_instance_id]
            await prefill_client.transfer_kv_cache(
                request_id=request.request_id,
                target_instance=request.decode_instance_id
            )
            
            self.logger.info(f"请求 {request.request_id} 开始KV传输")
            
        except Exception as e:
            request.status = RequestStatus.FAILED
            self.logger.error(f"请求 {request.request_id} KV传输失败: {e}")
            
    async def _check_transfer_status(self, request: PdRequest):
        """检查KV传输状态"""
        if not request.decode_instance_id:
            return
            
        decode_client = self.rpc_clients[request.decode_instance_id]
        status = await decode_client.get_transfer_status(request.request_id)
        
        if status.get("status") == "ready":
            # KV传输完成，开始decode
            request.status = RequestStatus.DECODING
            request.decode_start_time = time.time()
            
            await decode_client.start_decode(
                request_id=request.request_id,
                sampling_params=request.sampling_params.__dict__
            )
            
            self.logger.info(f"请求 {request.request_id} 开始decode")
            
    async def _check_decode_status(self, request: PdRequest):
        """检查decode状态"""
        if not request.decode_instance_id:
            return
            
        decode_client = self.rpc_clients[request.decode_instance_id]
        status = await decode_client.get_request_status(request.request_id)
        
        if status.get("status") == "generating":
            # 获取新生成的token
            new_tokens = status.get("new_tokens", [])
            request.completion_tokens.extend(new_tokens)
            
        elif status.get("status") == "completed":
            # Decode完成
            final_tokens = status.get("completion_tokens", [])
            request.completion_tokens = final_tokens
            request.status = RequestStatus.FINISHED
            
            self.logger.info(f"请求 {request.request_id} 完成，生成长度: {len(final_tokens)}")
            
        elif status.get("status") == "failed":
            request.status = RequestStatus.FAILED
            
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "waiting_requests": len(self.waiting_queue),
            "active_requests": len(self.active_requests),
            "success_rate": self.completed_requests / max(1, self.total_requests)
        }
