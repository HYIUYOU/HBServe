"""
KV传输管理器，统一管理不同backend
"""
import logging
from typing import Dict, Any, Optional

from ..config import PdDisaggConfig, KVTransferBackend
from .base import KVTransferBase, KVBuffer
from .nccl_backend import NcclKVTransfer
from .mooncake_backend import MooncakeKVTransfer


class KVTransferManager:
    """KV传输管理器"""
    
    def __init__(self, config: PdDisaggConfig, instance_rank: int):
        self.config = config
        self.instance_rank = instance_rank
        self.logger = logging.getLogger(__name__)
        
        # 确定world_size
        self.world_size = len(config.gpu_instances)
        
        # 创建对应的KV传输后端
        self.kv_transfer = self._create_kv_transfer_backend()
        
    def _create_kv_transfer_backend(self) -> KVTransferBase:
        """根据配置创建KV传输后端"""
        backend_config = {
            "kv_buffer_size": self.config.kv_buffer_size,
            "kv_ip": self.config.kv_ip,
            "kv_port": self.config.kv_port,
            "max_buffer_size": 100
        }
        
        if self.config.kv_transfer_backend == KVTransferBackend.NCCL:
            backend_config.update({
                "backend": "nccl",
                "init_method": f"tcp://{self.config.kv_ip}:{self.config.kv_port}"
            })
            return NcclKVTransfer(
                config=backend_config,
                rank=self.instance_rank,
                world_size=self.world_size
            )
            
        elif self.config.kv_transfer_backend == KVTransferBackend.MOONCAKE:
            backend_config.update({
                "prefill_url": f"{self.config.kv_ip}:{self.config.kv_port}",
                "decode_url": f"{self.config.kv_ip}:{self.config.kv_port + 1}",
                "metadata_server": f"{self.config.kv_ip}:{self.config.kv_port + 2}",
                "protocol": "tcp",
                "device_name": ""
            })
            return MooncakeKVTransfer(
                config=backend_config,
                rank=self.instance_rank,
                world_size=self.world_size
            )
            
        else:
            raise ValueError(f"不支持的KV传输后端: {self.config.kv_transfer_backend}")
            
    async def initialize(self):
        """初始化KV传输管理器"""
        try:
            await self.kv_transfer.initialize()
            self.logger.info(f"KV传输管理器初始化完成，backend: {self.config.kv_transfer_backend.value}")
        except Exception as e:
            self.logger.error(f"KV传输管理器初始化失败: {e}")
            raise
            
    async def close(self):
        """关闭KV传输管理器"""
        try:
            await self.kv_transfer.close()
            self.logger.info("KV传输管理器已关闭")
        except Exception as e:
            self.logger.error(f"关闭KV传输管理器失败: {e}")
            
    async def send_kv_cache(self, 
                           request_id: str,
                           kv_buffer: KVBuffer,
                           target_instance_id: str) -> bool:
        """发送KV缓存到目标实例"""
        # 根据实例ID找到对应的rank
        target_rank = self._get_rank_by_instance_id(target_instance_id)
        if target_rank is None:
            self.logger.error(f"找不到目标实例: {target_instance_id}")
            return False
            
        return await self.kv_transfer.send_kv_cache(request_id, kv_buffer, target_rank)
        
    async def receive_kv_cache(self, 
                              request_id: str,
                              source_instance_id: str) -> Optional[KVBuffer]:
        """从源实例接收KV缓存"""
        # 根据实例ID找到对应的rank
        source_rank = self._get_rank_by_instance_id(source_instance_id)
        if source_rank is None:
            self.logger.error(f"找不到源实例: {source_instance_id}")
            return None
            
        return await self.kv_transfer.receive_kv_cache(request_id, source_rank)
        
    async def check_transfer_status(self, request_id: str) -> Dict[str, Any]:
        """检查传输状态"""
        return await self.kv_transfer.check_transfer_status(request_id)
        
    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        stats = self.kv_transfer.get_buffer_stats()
        stats.update({
            "backend": self.config.kv_transfer_backend.value,
            "instance_rank": self.instance_rank,
            "world_size": self.world_size
        })
        return stats
        
    def _get_rank_by_instance_id(self, instance_id: str) -> Optional[int]:
        """根据实例ID获取rank"""
        for i, instance_config in enumerate(self.config.gpu_instances):
            if instance_config["instance_id"] == instance_id:
                return instance_config.get("kv_rank", i)
        return None
        
    def _get_instance_id_by_rank(self, rank: int) -> Optional[str]:
        """根据rank获取实例ID"""
        for instance_config in self.config.gpu_instances:
            if instance_config.get("kv_rank", -1) == rank:
                return instance_config["instance_id"]
        return None
        
    def validate_config(self) -> bool:
        """验证配置"""
        try:
            # 检查实例配置
            if not self.config.gpu_instances:
                self.logger.error("没有配置GPU实例")
                return False
                
            # 检查rank配置
            ranks = set()
            for instance_config in self.config.gpu_instances:
                rank = instance_config.get("kv_rank", -1)
                if rank in ranks:
                    self.logger.error(f"重复的kv_rank: {rank}")
                    return False
                ranks.add(rank)
                
            # 检查端口配置
            if self.config.kv_transfer_backend == KVTransferBackend.MOONCAKE:
                # Mooncake需要多个端口
                required_ports = 3  # prefill_url, decode_url, metadata_server
                # 这里可以添加端口冲突检查
                
            return True
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return False
