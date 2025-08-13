"""
KV传输的基础抽象类
"""
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class KVBuffer:
    """KV缓存数据结构"""
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    token_ids: torch.Tensor
    sequence_lengths: torch.Tensor
    block_indices: Optional[torch.Tensor] = None
    
    def to(self, device: torch.device):
        """移动到指定设备"""
        return KVBuffer(
            key_cache=self.key_cache.to(device),
            value_cache=self.value_cache.to(device), 
            token_ids=self.token_ids.to(device),
            sequence_lengths=self.sequence_lengths.to(device),
            block_indices=self.block_indices.to(device) if self.block_indices is not None else None
        )
    
    @property
    def device(self) -> torch.device:
        return self.key_cache.device
    
    @property
    def batch_size(self) -> int:
        return self.key_cache.shape[0]
    
    @property
    def num_layers(self) -> int:
        return self.key_cache.shape[1]
    
    @property 
    def num_heads(self) -> int:
        return self.key_cache.shape[2]
    
    @property
    def head_dim(self) -> int:
        return self.key_cache.shape[4]
    
    def get_memory_usage(self) -> int:
        """获取内存使用量（字节）"""
        total_bytes = 0
        total_bytes += self.key_cache.numel() * self.key_cache.element_size()
        total_bytes += self.value_cache.numel() * self.value_cache.element_size()
        total_bytes += self.token_ids.numel() * self.token_ids.element_size()
        total_bytes += self.sequence_lengths.numel() * self.sequence_lengths.element_size()
        if self.block_indices is not None:
            total_bytes += self.block_indices.numel() * self.block_indices.element_size()
        return total_bytes


class KVTransferBase(ABC):
    """KV传输基础类"""
    
    def __init__(self, config: Dict[str, Any], rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_initialized = False
        
    @abstractmethod
    async def initialize(self):
        """初始化传输后端"""
        pass
        
    @abstractmethod
    async def close(self):
        """关闭传输后端"""
        pass
        
    @abstractmethod
    async def send_kv_cache(self, 
                           request_id: str,
                           kv_buffer: KVBuffer,
                           target_rank: int) -> bool:
        """发送KV缓存"""
        pass
        
    @abstractmethod
    async def receive_kv_cache(self, 
                              request_id: str,
                              source_rank: int) -> Optional[KVBuffer]:
        """接收KV缓存"""
        pass
        
    @abstractmethod
    async def check_transfer_status(self, request_id: str) -> Dict[str, Any]:
        """检查传输状态"""
        pass
        
    @abstractmethod
    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        pass
        
    def validate_kv_buffer(self, kv_buffer: KVBuffer) -> bool:
        """验证KV缓存格式"""
        try:
            # 检查基本维度
            if kv_buffer.key_cache.dim() != 5:  # [batch, layer, head, seq, head_dim]
                return False
            if kv_buffer.value_cache.dim() != 5:
                return False
                
            # 检查维度一致性
            if kv_buffer.key_cache.shape != kv_buffer.value_cache.shape:
                return False
                
            # 检查设备一致性
            if kv_buffer.key_cache.device != kv_buffer.value_cache.device:
                return False
                
            # 检查数据类型
            if kv_buffer.key_cache.dtype != kv_buffer.value_cache.dtype:
                return False
                
            return True
            
        except Exception:
            return False


class KVLookupBuffer:
    """KV查找缓冲区，用于处理乱序请求"""
    
    def __init__(self, max_buffer_size: int = 100):
        self.max_buffer_size = max_buffer_size
        self.buffer: Dict[str, KVBuffer] = {}
        self.request_queue: List[str] = []
        
    def insert(self, request_id: str, kv_buffer: KVBuffer):
        """插入KV缓存"""
        if len(self.buffer) >= self.max_buffer_size:
            # 移除最早的请求
            oldest_request = self.request_queue.pop(0)
            if oldest_request in self.buffer:
                del self.buffer[oldest_request]
                
        self.buffer[request_id] = kv_buffer
        if request_id not in self.request_queue:
            self.request_queue.append(request_id)
            
    def lookup(self, request_id: str) -> Optional[KVBuffer]:
        """查找KV缓存"""
        return self.buffer.get(request_id)
        
    def remove(self, request_id: str) -> Optional[KVBuffer]:
        """移除并返回KV缓存"""
        kv_buffer = self.buffer.pop(request_id, None)
        if request_id in self.request_queue:
            self.request_queue.remove(request_id)
        return kv_buffer
        
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.request_queue.clear()
        
    def size(self) -> int:
        """获取缓冲区大小"""
        return len(self.buffer)
        
    def get_memory_usage(self) -> int:
        """获取总内存使用量"""
        total_bytes = 0
        for kv_buffer in self.buffer.values():
            total_bytes += kv_buffer.get_memory_usage()
        return total_bytes
