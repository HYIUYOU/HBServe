"""
pd分离架构的配置类
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class KVTransferBackend(Enum):
    """KV传输后端类型"""
    NCCL = "nccl"
    MOONCAKE = "mooncake"


class InstanceType(Enum):
    """实例类型"""
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class PdDisaggConfig:
    """pd分离架构配置"""
    
    # 基础配置
    model_path: str
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    
    # KV传输配置
    kv_transfer_backend: KVTransferBackend = KVTransferBackend.NCCL
    kv_buffer_size: int = 1024 * 1024 * 1024  # 1GB
    kv_ip: str = "127.0.0.1"
    kv_port: int = 12345
    
    # RPC配置
    rpc_port_base: int = 50051
    rpc_timeout: float = 30.0
    
    # Chunked prefill配置
    chunked_prefill_enabled: bool = True
    chunk_size: int = 512
    max_chunk_prefill_tokens: int = 4096
    
    # CUDA graph配置
    cuda_graph_enabled: bool = True
    cuda_graph_max_seq_len: int = 2048
    
    # GPU实例配置
    gpu_instances: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.gpu_instances is None:
            # 默认配置：1个prefill实例，1个decode实例
            self.gpu_instances = [
                {
                    "instance_id": "prefill-0",
                    "instance_type": InstanceType.PREFILL,
                    "gpu_id": 0,
                    "host": "localhost",
                    "port": self.rpc_port_base,
                    "kv_rank": 0
                },
                {
                    "instance_id": "decode-0", 
                    "instance_type": InstanceType.DECODE,
                    "gpu_id": 1,
                    "host": "localhost", 
                    "port": self.rpc_port_base + 1,
                    "kv_rank": 1
                }
            ]
    
    def get_prefill_instances(self) -> List[Dict[str, Any]]:
        """获取prefill实例配置"""
        return [inst for inst in self.gpu_instances 
                if inst["instance_type"] == InstanceType.PREFILL]
    
    def get_decode_instances(self) -> List[Dict[str, Any]]:
        """获取decode实例配置"""
        return [inst for inst in self.gpu_instances 
                if inst["instance_type"] == InstanceType.DECODE]


@dataclass 
class MooncakeConfig:
    """Mooncake配置"""
    prefill_url: str = "127.0.0.1:12346"
    decode_url: str = "127.0.0.1:12347"
    metadata_server: str = "127.0.0.1:12348"
    protocol: str = "tcp"
    device_name: str = ""
    metadata_backend: Optional[str] = None


@dataclass
class NcclConfig:
    """NCCL配置"""
    world_size: int = 2
    init_method: str = "tcp://127.0.0.1:12349"
    backend: str = "nccl"
