"""
统一的分离式LLM接口
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

from .config import PdDisaggConfig
from .cpu_scheduler import CpuScheduler
from .gpu_instance import GpuInstance
from hbserve.sampling_params import SamplingParams


class DisaggregatedLLM:
    """分离式LLM主类"""
    
    def __init__(self, config: PdDisaggConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True)
        
        # CPU调度器
        self.cpu_scheduler = CpuScheduler(config)
        
        # GPU实例列表
        self.gpu_instances: Dict[str, GpuInstance] = {}
        self._create_gpu_instances()
        
        # 运行状态
        self.is_running = False
        
    def _create_gpu_instances(self):
        """创建GPU实例"""
        for instance_config in self.config.gpu_instances:
            instance_id = instance_config["instance_id"]
            instance = GpuInstance(self.config, instance_config)
            self.gpu_instances[instance_id] = instance
            
    async def start(self):
        """启动分离式LLM"""
        try:
            self.logger.info("启动分离式LLM...")
            
            # 启动所有GPU实例
            start_tasks = []
            for instance in self.gpu_instances.values():
                start_tasks.append(instance.start())
                
            await asyncio.gather(*start_tasks)
            
            # 等待实例就绪
            await asyncio.sleep(2)
            
            # 启动CPU调度器
            await self.cpu_scheduler.start()
            
            self.is_running = True
            self.logger.info("分离式LLM启动完成")
            
        except Exception as e:
            self.logger.error(f"启动分离式LLM失败: {e}")
            raise
            
    async def stop(self):
        """停止分离式LLM"""
        try:
            self.logger.info("停止分离式LLM...")
            
            # 停止CPU调度器
            await self.cpu_scheduler.stop()
            
            # 停止所有GPU实例
            stop_tasks = []
            for instance in self.gpu_instances.values():
                stop_tasks.append(instance.stop())
                
            await asyncio.gather(*stop_tasks)
            
            self.is_running = False
            self.logger.info("分离式LLM已停止")
            
        except Exception as e:
            self.logger.error(f"停止分离式LLM失败: {e}")
            
    def add_request(self, prompt: str, sampling_params: SamplingParams) -> str:
        """添加生成请求"""
        if not self.is_running:
            raise RuntimeError("分离式LLM未运行")
            
        # 编码prompt
        token_ids = self.tokenizer.encode(prompt)
        
        # 添加到调度器
        request_id = self.cpu_scheduler.add_request(prompt, token_ids, sampling_params)
        
        return request_id
        
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """获取请求状态"""
        return self.cpu_scheduler.get_request_status(request_id)
        
    async def generate(self, 
                      prompts: List[str],
                      sampling_params: SamplingParams | List[SamplingParams],
                      return_incremental: bool = False) -> List[Dict[str, Any]]:
        """批量生成"""
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            
        # 提交所有请求
        request_ids = []
        for prompt, sp in zip(prompts, sampling_params):
            request_id = self.add_request(prompt, sp)
            request_ids.append(request_id)
            
        # 等待所有请求完成
        results = []
        while True:
            all_finished = True
            current_results = []
            
            for request_id in request_ids:
                status = self.get_request_status(request_id)
                if not status:
                    current_results.append(None)
                    continue
                    
                if status["status"] in ["finished", "failed"]:
                    # 解码completion tokens
                    completion_text = ""
                    if status["completion_tokens"]:
                        completion_text = self.tokenizer.decode(
                            status["completion_tokens"], 
                            skip_special_tokens=True
                        )
                        
                    current_results.append({
                        "text": completion_text,
                        "token_ids": status["completion_tokens"],
                        "status": status["status"],
                        "request_id": request_id
                    })
                else:
                    all_finished = False
                    if return_incremental:
                        # 返回增量结果
                        completion_text = ""
                        if status["completion_tokens"]:
                            completion_text = self.tokenizer.decode(
                                status["completion_tokens"], 
                                skip_special_tokens=True
                            )
                        current_results.append({
                            "text": completion_text,
                            "token_ids": status["completion_tokens"],
                            "status": status["status"],
                            "request_id": request_id
                        })
                    else:
                        current_results.append(None)
                        
            if return_incremental:
                results = current_results
                if all_finished:
                    break
            else:
                if all_finished:
                    results = current_results
                    break
                    
            await asyncio.sleep(0.1)  # 100ms轮询间隔
            
        return results
        
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            "cpu_scheduler": self.cpu_scheduler.get_stats(),
            "gpu_instances": {},
            "config": {
                "kv_transfer_backend": self.config.kv_transfer_backend.value,
                "chunked_prefill_enabled": self.config.chunked_prefill_enabled,
                "cuda_graph_enabled": self.config.cuda_graph_enabled,
                "chunk_size": self.config.chunk_size
            }
        }
        
        for instance_id, instance in self.gpu_instances.items():
            stats["gpu_instances"][instance_id] = instance.get_stats()
            
        return stats


# 简化的同步接口（类似于原始的hbserve接口）
class LLM:
    """同步LLM接口，兼容原始hbserve"""
    
    def __init__(self, model_path: str, **kwargs):
        # 创建配置
        config_args = {
            "model_path": model_path,
            **kwargs
        }
        self.config = PdDisaggConfig(**config_args)
        
        # 创建分离式LLM
        self.disagg_llm = DisaggregatedLLM(self.config)
        
        # 事件循环
        self.loop = None
        self._start_event_loop()
        
    def _start_event_loop(self):
        """启动异步事件循环"""
        import threading
        
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
            
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # 等待循环启动
        import time
        time.sleep(0.1)
        
        # 启动分离式LLM
        future = asyncio.run_coroutine_threadsafe(
            self.disagg_llm.start(), self.loop
        )
        future.result()  # 等待启动完成
        
    def generate(self, 
                prompts: List[str], 
                sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        """同步生成接口"""
        if not isinstance(prompts, list):
            prompts = [prompts]
            
        # 在事件循环中执行异步生成
        future = asyncio.run_coroutine_threadsafe(
            self.disagg_llm.generate(prompts, sampling_params), 
            self.loop
        )
        
        return future.result()
        
    def __del__(self):
        """清理资源"""
        if self.loop and self.loop.is_running():
            # 停止分离式LLM
            future = asyncio.run_coroutine_threadsafe(
                self.disagg_llm.stop(), self.loop
            )
            try:
                future.result(timeout=5)
            except:
                pass
                
            # 停止事件循环
            self.loop.call_soon_threadsafe(self.loop.stop)
