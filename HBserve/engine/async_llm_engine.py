import asyncio
import uuid
from typing import Dict, List, Optional, AsyncIterator
from dataclasses import dataclass

from HBserve.engine.llm_engine import LLMEngine
from HBserve.sampling_params import SamplingParams


@dataclass
class RequestOutput:
    request_id: str
    prompt: str
    text: str
    token_ids: List[int]
    finished: bool


class AsyncLLMEngine:
    """异步 LLM 引擎，支持在线推理"""
    
    def __init__(self, model: str, **kwargs):
        self.engine = LLMEngine(model, **kwargs)
        self.request_outputs: Dict[str, asyncio.Queue] = {}
        self.request_prompts: Dict[str, str] = {}
        self.seq_id_to_request_id: Dict[str, str] = {}  # seq_id -> request_id 映射
        self._running = False
        self._background_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """启动后台引擎循环"""
        if self._running:
            return
        self._running = True
        self._background_task = asyncio.create_task(self._engine_loop())
        print("[AsyncEngine] Background loop started")
        
    async def stop(self):
        """停止引擎"""
        self._running = False
        if self._background_task:
            await self._background_task
        self.engine.exit()
        print("[AsyncEngine] Stopped")
    
    async def _engine_loop(self):
        """后台持续运行的推理循环"""
        while self._running:
            # 检查是否有待处理的请求
            if self.engine.is_finished():
                await asyncio.sleep(0.01)  # 空闲时短暂休眠
                continue
            
            # 在线程池中执行同步的 step
            loop = asyncio.get_event_loop()
            outputs, num_tokens = await loop.run_in_executor(
                None, 
                self.engine.step
            )
            
            # 将结果发送到对应的队列
            for seq_id, token_ids in outputs:
                request_id = self.seq_id_to_request_id.get(seq_id)
                if request_id and request_id in self.request_outputs:
                    output = RequestOutput(
                        request_id=request_id,
                        prompt=self.request_prompts.get(request_id, ""),
                        text=self.engine.tokenizer.decode(token_ids),
                        token_ids=token_ids,
                        finished=True
                    )
                    await self.request_outputs[request_id].put(output)
                    # 清理映射
                    del self.seq_id_to_request_id[seq_id]
            
            # 让出控制权
            await asyncio.sleep(0)
    
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None
    ) -> RequestOutput:
        """异步生成单个请求的结果"""
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # 创建输出队列
        output_queue = asyncio.Queue()
        self.request_outputs[request_id] = output_queue
        self.request_prompts[request_id] = prompt
        
        # 提交请求（在线程池中执行）
        loop = asyncio.get_event_loop()
        
        # 添加请求并获取内部 seq_id
        def add_request_sync():
            if isinstance(prompt, str):
                token_ids = self.engine.tokenizer.encode(prompt)
            else:
                token_ids = prompt
            
            from HBserve.engine.sequence import Sequence
            seq = Sequence(token_ids, sampling_params)
            self.engine.scheduler.add(seq)
            return seq.seq_id  # 返回内部 seq_id
        
        seq_id = await loop.run_in_executor(None, add_request_sync)
        
        # 建立 seq_id 到 request_id 的映射
        self.seq_id_to_request_id[seq_id] = request_id
        
        # 等待结果
        output = await output_queue.get()
        
        # 清理
        del self.request_outputs[request_id]
        del self.request_prompts[request_id]
        
        return output
    
    async def generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None
    ) -> AsyncIterator[RequestOutput]:
        """流式生成（当前简化版本：一次性返回）"""
        # 未来可以实现真正的 token-by-token 流式输出
        output = await self.generate(prompt, sampling_params, request_id)
        yield output