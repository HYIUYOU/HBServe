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
        print("[AsyncEngine] Initializing engine...")
        self.engine = LLMEngine(model, **kwargs)
        self.request_outputs: Dict[str, asyncio.Queue] = {}
        self.request_prompts: Dict[str, str] = {}
        self.seq_id_to_request_id: Dict[str, str] = {}
        self._running = False
        self._background_task: Optional[asyncio.Task] = None
        print("[AsyncEngine] Engine initialized")
        
    async def start(self):
        """启动后台引擎循环"""
        if self._running:
            print("[AsyncEngine] Already running")
            return
        self._running = True
        self._background_task = asyncio.create_task(self._engine_loop())
        print("[AsyncEngine] Background loop task created")
        
    async def stop(self):
        """停止引擎"""
        print("[AsyncEngine] Stopping engine...")
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        self.engine.exit()
        print("[AsyncEngine] Engine stopped")
    
    async def _engine_loop(self):
        """后台持续运行的推理循环"""
        print("[AsyncEngine] 🚀 Engine loop started")
        loop = asyncio.get_event_loop()
        step_count = 0
        
        while self._running:
            try:
                # 检查是否有待处理的请求
                if self.engine.is_finished():
                    # 没有任务时短暂休眠
                    await asyncio.sleep(0.001)
                    continue
                
                # 在线程池中执行同步的 step
                step_count += 1
                if step_count % 100 == 1:  # 每100步打印一次
                    print(f"[AsyncEngine] Executing step {step_count}...")
                
                outputs, num_tokens = await loop.run_in_executor(
                    None, 
                    self.engine.step
                )
                
                if outputs:
                    print(f"[AsyncEngine] Step {step_count} completed: {len(outputs)} outputs, {num_tokens} tokens")
                
                # 将结果发送到对应的队列
                for seq_id, token_ids in outputs:
                    request_id = self.seq_id_to_request_id.get(seq_id)
                    if request_id and request_id in self.request_outputs:
                        try:
                            text = self.engine.tokenizer.decode(token_ids, skip_special_tokens=True)
                            output = RequestOutput(
                                request_id=request_id,
                                prompt=self.request_prompts.get(request_id, ""),
                                text=text,
                                token_ids=token_ids,
                                finished=True
                            )
                            print(f"[AsyncEngine] ✅ Sending output for {request_id}: {text[:50]}...")
                            await self.request_outputs[request_id].put(output)
                            
                            # 清理映射
                            if seq_id in self.seq_id_to_request_id:
                                del self.seq_id_to_request_id[seq_id]
                        except Exception as e:
                            print(f"[AsyncEngine] Error processing output for {request_id}: {e}")
                            import traceback
                            traceback.print_exc()
                
                # 让出控制权，避免阻塞其他协程
                await asyncio.sleep(0)
                
            except asyncio.CancelledError:
                print("[AsyncEngine] Engine loop cancelled")
                break
            except Exception as e:
                print(f"[AsyncEngine] ❌ Error in engine loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.1)
        
        print("[AsyncEngine] Engine loop exited")
    
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None
    ) -> RequestOutput:
        """异步生成单个请求的结果"""
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        print(f"[AsyncEngine] 📥 Generating for request {request_id}: {prompt[:50]}...")
        
        # 创建输出队列
        output_queue = asyncio.Queue()
        self.request_outputs[request_id] = output_queue
        self.request_prompts[request_id] = prompt
        
        # 提交请求（在线程池中执行）
        loop = asyncio.get_event_loop()
        
        def add_request_sync():
            print(f"[AsyncEngine] Adding request {request_id} to scheduler")
            try:
                if isinstance(prompt, str):
                    token_ids = self.engine.tokenizer.encode(prompt)
                else:
                    token_ids = prompt
                
                print(f"[AsyncEngine] Tokenized prompt for {request_id}: {len(token_ids)} tokens")
                
                from HBserve.engine.sequence import Sequence
                seq = Sequence(token_ids, sampling_params)
                self.engine.scheduler.add(seq)
                
                print(f"[AsyncEngine] ✅ Request {request_id} added to scheduler, seq_id={seq.seq_id}")
                return seq.seq_id
            except Exception as e:
                print(f"[AsyncEngine] ❌ Error adding request {request_id}: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        try:
            # 在线程池中添加请求
            seq_id = await loop.run_in_executor(None, add_request_sync)
            
            # 建立 seq_id 到 request_id 的映射
            self.seq_id_to_request_id[seq_id] = request_id
            print(f"[AsyncEngine] Mapped seq_id {seq_id} -> request_id {request_id}")
            
            print(f"[AsyncEngine] ⏳ Waiting for output of request {request_id}...")
            
            # 等待结果（设置超时）
            output = await asyncio.wait_for(output_queue.get(), timeout=300.0)
            
            print(f"[AsyncEngine] 📤 Got output for request {request_id}: {output.text[:50]}...")
            
            return output
            
        except asyncio.TimeoutError:
            print(f"[AsyncEngine] ⏰ Timeout waiting for request {request_id}")
            raise
        except Exception as e:
            print(f"[AsyncEngine] ❌ Error generating request {request_id}: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 清理
            if request_id in self.request_outputs:
                del self.request_outputs[request_id]
            if request_id in self.request_prompts:
                del self.request_prompts[request_id]
            print(f"[AsyncEngine] Cleaned up request {request_id}")
    
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
