#!/usr/bin/env python3
"""
启动pd分离架构的脚本
"""
import argparse
import asyncio
import logging
import os
import signal
import sys
import yaml
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from hbserve.pd_disagg.config import PdDisaggConfig, KVTransferBackend, InstanceType
from hbserve.pd_disagg.disaggregated_llm import DisaggregatedLLM
from hbserve.pd_disagg.gpu_instance import GpuInstance


class PdDisaggServer:
    """pd分离架构服务器"""
    
    def __init__(self, config_path: str, instance_id: str = None):
        self.config_path = config_path
        self.instance_id = instance_id
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config = self._load_config()
        
        # 如果指定了instance_id，只启动单个实例
        if instance_id:
            self.single_instance = True
            self.gpu_instance = self._create_single_instance(instance_id)
            self.disagg_llm = None
        else:
            self.single_instance = False
            self.gpu_instance = None
            self.disagg_llm = DisaggregatedLLM(self.config)
            
        # 运行状态
        self.running = False
        
    def _load_config(self) -> PdDisaggConfig:
        """加载配置文件"""
        if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
                
            # 转换配置格式
            config_args = {
                "model_path": os.path.expanduser(config_dict["model"]["path"]),
                "tensor_parallel_size": config_dict["model"]["tensor_parallel_size"],
                "max_num_seqs": config_dict["performance"]["max_num_seqs"],
                "max_num_batched_tokens": config_dict["performance"]["max_num_batched_tokens"],
                
                "kv_transfer_backend": KVTransferBackend(config_dict["kv_transfer"]["backend"]),
                "kv_buffer_size": config_dict["kv_transfer"]["buffer_size"],
                "kv_ip": config_dict["kv_transfer"]["ip"],
                "kv_port": config_dict["kv_transfer"]["port"],
                
                "rpc_port_base": config_dict["rpc"]["port_base"],
                "rpc_timeout": config_dict["rpc"]["timeout"],
                
                "chunked_prefill_enabled": config_dict["chunked_prefill"]["enabled"],
                "chunk_size": config_dict["chunked_prefill"]["chunk_size"],
                "max_chunk_prefill_tokens": config_dict["chunked_prefill"]["max_chunk_prefill_tokens"],
                
                "cuda_graph_enabled": config_dict["cuda_graph"]["enabled"],
                "cuda_graph_max_seq_len": config_dict["cuda_graph"]["max_seq_len"],
            }
            
            # 转换GPU实例配置
            gpu_instances = []
            for instance_config in config_dict["gpu_instances"]:
                gpu_instances.append({
                    "instance_id": instance_config["instance_id"],
                    "instance_type": InstanceType(instance_config["instance_type"]),
                    "gpu_id": instance_config["gpu_id"],
                    "host": instance_config["host"],
                    "port": instance_config["port"],
                    "kv_rank": instance_config["kv_rank"]
                })
            config_args["gpu_instances"] = gpu_instances
            
            return PdDisaggConfig(**config_args)
        else:
            raise ValueError(f"不支持的配置文件格式: {self.config_path}")
            
    def _create_single_instance(self, instance_id: str) -> GpuInstance:
        """创建单个GPU实例"""
        instance_config = None
        for config in self.config.gpu_instances:
            if config["instance_id"] == instance_id:
                instance_config = config
                break
                
        if not instance_config:
            raise ValueError(f"找不到实例配置: {instance_id}")
            
        return GpuInstance(self.config, instance_config)
        
    async def start(self):
        """启动服务器"""
        try:
            self.logger.info(f"启动pd分离架构服务器...")
            
            if self.single_instance:
                # 启动单个实例
                self.logger.info(f"启动单个GPU实例: {self.instance_id}")
                await self.gpu_instance.start()
            else:
                # 启动完整系统
                self.logger.info("启动完整pd分离系统")
                await self.disagg_llm.start()
                
            self.running = True
            self.logger.info("pd分离架构服务器启动完成")
            
            # 等待信号
            await self._wait_for_signal()
            
        except Exception as e:
            self.logger.error(f"启动服务器失败: {e}")
            raise
            
    async def stop(self):
        """停止服务器"""
        try:
            self.logger.info("停止pd分离架构服务器...")
            
            if self.single_instance:
                await self.gpu_instance.stop()
            else:
                await self.disagg_llm.stop()
                
            self.running = False
            self.logger.info("pd分离架构服务器已停止")
            
        except Exception as e:
            self.logger.error(f"停止服务器失败: {e}")
            
    async def _wait_for_signal(self):
        """等待退出信号"""
        import signal
        
        stop_event = asyncio.Event()
        
        def signal_handler(signum, frame):
            self.logger.info(f"接收到信号 {signum}，准备退出...")
            stop_event.set()
            
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 等待信号
        await stop_event.wait()
        
        # 停止服务器
        await self.stop()


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pd_disagg.log')
        ]
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="pd分离架构服务器")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="配置文件路径"
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        help="启动单个GPU实例（可选）"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)
        
    # 创建服务器
    server = PdDisaggServer(args.config, args.instance_id)
    
    try:
        # 启动服务器
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n服务器被用户中断")
    except Exception as e:
        print(f"服务器运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
