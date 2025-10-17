import argparse
import asyncio
import json
import time
from typing import AsyncGenerator, Optional, List, Dict, Any
import uuid
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from HBserve.sampling_params import SamplingParams
from HBserve.engine.async_llm_engine import AsyncLLMEngine
from transformers import AutoTokenizer


# ============= Request/Response Models =============

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 64
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | List[str]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 64
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class FinishNotification(BaseModel):
    total_requests: int


# ============= Performance Monitoring =============

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_latency = 0
        self.request_history = deque(maxlen=1000)
        self.start_time = time.time()
        self.active_requests = 0
        self.max_active_requests = 0
        
    def record_request(self, latency: float, success: bool = True):
        """记录请求"""
        self.request_count += 1
        self.total_latency += latency
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        self.request_history.append({
            'timestamp': time.time(),
            'latency': latency,
            'success': success
        })
    
    def increment_active(self):
        """增加活跃请求数"""
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
    
    def decrement_active(self):
        """减少活跃请求数"""
        self.active_requests = max(0, self.active_requests - 1)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        uptime = time.time() - self.start_time
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        
        # 计算最近的 RPS
        recent_requests = [r for r in self.request_history 
                          if r['timestamp'] > time.time() - 60]
        recent_rps = len(recent_requests) / 60 if recent_requests else 0
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / self.request_count if self.request_count > 0 else 0,
            'average_latency': avg_latency,
            'current_rps': recent_rps,
            'overall_rps': self.request_count / uptime if uptime > 0 else 0,
            'active_requests': self.active_requests,
            'max_active_requests': self.max_active_requests
        }


# ============= Global State =============

engine: Optional[AsyncLLMEngine] = None
tokenizer: Optional[AutoTokenizer] = None
model_name: str = ""
model_config: Optional[Dict] = None  # 👈 全局配置
monitor = PerformanceMonitor()

# 并发控制
MAX_CONCURRENT_REQUESTS = 100
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


# ============= Lifespan Management =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期"""
    # Startup
    if engine is not None:
        print("[Server] Starting engine...")
        await engine.start()
        print("[Server] Engine background loop started")
        print(f"[Server] Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
        
        # 等待引擎完全启动
        await asyncio.sleep(0.5)
        
        # 验证引擎是否正常工作
        print("[Server] Testing engine with a simple request...")
        try:
            test_output = await asyncio.wait_for(
                engine.generate(
                    "Hello", 
                    SamplingParams(temperature=1.0, max_tokens=5)
                ),
                timeout=30.0
            )
            print(f"[Server] ✅ Engine test successful: {test_output.text[:50]}...")
        except Exception as e:
            print(f"[Server] ⚠️ Engine test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 启动监控任务
    asyncio.create_task(monitor_task())
    
    yield  # 应用运行中
    
    # Shutdown
    if engine is not None:
        print("[Server] Stopping engine...")
        await engine.stop()
        print("[Server] Engine stopped")
        
        # 打印最终统计
        stats = monitor.get_stats()
        print("\n" + "="*60)
        print("📊 Final Statistics")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        print("="*60)


async def monitor_task():
    """定期打印监控信息"""
    while True:
        await asyncio.sleep(30)
        stats = monitor.get_stats()
        print(f"\n[Monitor] Active: {stats['active_requests']}, "
              f"Total: {stats['total_requests']}, "
              f"RPS: {stats['current_rps']:.2f}, "
              f"Avg Latency: {stats['average_latency']:.3f}s")


# ============= FastAPI App =============

app = FastAPI(
    title="HBserve OpenAI-Compatible API",
    lifespan=lifespan
)

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Middleware =============

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加处理时间监控"""
    start_time = time.time()
    monitor.increment_active()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # 记录成功请求
        if response.status_code == 200:
            monitor.record_request(process_time, success=True)
        else:
            monitor.record_request(process_time, success=False)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        monitor.record_request(process_time, success=False)
        raise
        
    finally:
        monitor.decrement_active()


# ============= Helper Functions =============

def format_chat_completion_response(output, model: str, request_id: str = None) -> Dict:
    """Format response in OpenAI chat completion format"""
    if request_id is None:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output.text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(output.token_ids),
            "total_tokens": len(output.token_ids)
        }
    }


def format_completion_response(outputs: List[Any], model: str) -> Dict:
    """Format response in OpenAI completion format"""
    choices = []
    for i, output in enumerate(outputs):
        choices.append({
            "index": i,
            "text": output.text,
            "finish_reason": "stop"
        })
    
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices
    }


async def stream_chat_completion_async(
    prompt: str,
    sampling_params: SamplingParams,
    model: str,
    request_id: str = None
) -> AsyncGenerator[str, None]:
    """异步流式输出"""
    if request_id is None:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    try:
        # 发送角色
        role_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"
        
        # 流式生成内容
        async for output in engine.generate_stream(prompt, sampling_params):
            content_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": output.text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"
        
        # 发送结束标记
        finish_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(finish_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except asyncio.CancelledError:
        print(f"[Stream] Client disconnected for request {request_id}")
        raise
    except Exception as e:
        print(f"[Stream] Error in stream for request {request_id}: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============= API Endpoints =============

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hbserve"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    async with request_semaphore:
        try:
            print(f"[API] Processing request {request_id}: {request.messages[-1].content[:50]}...")
            
            # 转换消息为 prompt
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 🔧 强制限制：最大输入 256 tokens
            MAX_INPUT_TOKENS = 256
            
            # 检查并截断输入
            prompt_tokens = tokenizer.encode(prompt)
            original_len = len(prompt_tokens)
            
            if original_len > MAX_INPUT_TOKENS:
                print(f"[API] ⚠️ Input too long ({original_len} tokens), truncating to {MAX_INPUT_TOKENS}")
                # 截断（保留最后部分）
                prompt_tokens = prompt_tokens[-MAX_INPUT_TOKENS:]
                prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
                print(f"[API] Truncated prompt length: {len(prompt_tokens)} tokens")
            else:
                print(f"[API] Prompt tokens: {original_len}")
            
            # 🔧 安全检查：确保 token_ids 不为空
            if len(prompt_tokens) == 0:
                print(f"[API] ⚠️ Empty prompt after truncation, using fallback")
                prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
                prompt_tokens = tokenizer.encode(prompt)
            
            print(f"[API] Final prompt length: {len(prompt_tokens)} tokens")
            
            # 创建采样参数
            sampling_params = SamplingParams(
                temperature=request.temperature if request.temperature else 1.0,
                max_tokens=request.max_tokens if request.max_tokens else 64,
                ignore_eos=False
            )
            
            # 动态调整超时
            timeout = max(60, request.max_tokens * 0.1)
            
            # 异步生成
            if request.stream:
                print(f"[API] Starting stream for {request_id}")
                return StreamingResponse(
                    stream_chat_completion_async(prompt, sampling_params, request.model, request_id),
                    media_type="text/event-stream"
                )
            else:
                print(f"[API] Generating non-stream response for {request_id}")
                output = await asyncio.wait_for(
                    engine.generate(prompt, sampling_params, request_id),
                    timeout=timeout
                )
                print(f"[API] Got output for {request_id}: {output.text[:50]}...")
                return format_chat_completion_response(output, request.model, request_id)
                
        except asyncio.TimeoutError:
            print(f"[API] Timeout for request {request_id}")
            monitor.record_request(timeout, success=False)
            raise HTTPException(status_code=504, detail=f"Request timeout after {timeout}s")
            
        except Exception as e:
            print(f"[API] Error processing request {request_id}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible text completion endpoint"""
    async with request_semaphore:
        try:
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
            
            if len(prompts) > 1 and request.stream:
                raise HTTPException(
                    status_code=400,
                    detail="Streaming only supports single prompt"
                )
            
            # 🔧 强制限制：最大输入 256 tokens
            MAX_INPUT_TOKENS = 256
            
            # 检查并截断输入
            processed_prompts = []
            for prompt in prompts:
                prompt_tokens = tokenizer.encode(prompt)
                original_len = len(prompt_tokens)
                
                if original_len > MAX_INPUT_TOKENS:
                    print(f"[API] Truncating prompt from {original_len} to {MAX_INPUT_TOKENS} tokens")
                    prompt_tokens = prompt_tokens[-MAX_INPUT_TOKENS:]
                    prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
                
                # 确保不为空
                if len(prompt_tokens) == 0:
                    prompt = "Hello"
                
                processed_prompts.append(prompt)
            
            sampling_params = SamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            timeout = max(60, request.max_tokens * 0.1)
            
            if request.stream:
                # 流式处理
                async def stream_completion():
                    output = await asyncio.wait_for(
                        engine.generate(processed_prompts[0], sampling_params),
                        timeout=timeout
                    )
                    chunk = {
                        "id": f"cmpl-{uuid.uuid4().hex}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "text": output.text,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    stream_completion(),
                    media_type="text/event-stream"
                )
            else:
                # 并发处理多个 prompt
                outputs = await asyncio.gather(*[
                    asyncio.wait_for(
                        engine.generate(prompt, sampling_params),
                        timeout=timeout
                    )
                    for prompt in processed_prompts
                ])
                return format_completion_response(outputs, request.model)
                
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    stats = monitor.get_stats()
    
    # 检查引擎状态
    engine_status = "unknown"
    if engine is not None:
        if hasattr(engine, '_running'):
            engine_status = "running" if engine._running else "stopped"
        else:
            engine_status = "initialized"
    else:
        engine_status = "not_initialized"
    
    return {
        "status": "ok",
        "engine_status": engine_status,
        "model": model_name,
        "config": model_config,
        "stats": {
            "active_requests": stats['active_requests'],
            "total_requests": stats['total_requests'],
            "current_rps": round(stats['current_rps'], 2),
        }
    }


@app.get("/stats")
async def get_stats():
    """Get detailed statistics"""
    return monitor.get_stats()


@app.post("/finish")
async def finish_notification(notification: FinishNotification):
    """Handle finish notification from client"""
    stats = monitor.get_stats()
    
    print("\n" + "="*60)
    print("📊 Server Statistics Report")
    print("="*60)
    print(f"Client reported: {notification.total_requests} requests")
    print(f"Server processed: {stats['total_requests']} requests")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"Average latency: {stats['average_latency']:.3f}s")
    print(f"Overall RPS: {stats['overall_rps']:.2f}")
    print(f"Max concurrent: {stats['max_active_requests']}")
    print("="*60)
    
    return {"status": "ok", "stats": stats}


# ============= Server Initialization =============

def initialize_model(model_path: str, max_concurrent: int = 100, **kwargs):
    """Initialize the LLM engine and tokenizer"""
    global engine, tokenizer, model_name, model_config, request_semaphore, MAX_CONCURRENT_REQUESTS
    
    MAX_CONCURRENT_REQUESTS = max_concurrent
    request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    print(f"[Init] Loading model from {model_path}...")
    
    # 检查显存
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[Init] CUDA available: True")
            print(f"[Init] CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3
                print(f"[Init]   Device {i}: {torch.cuda.get_device_name(i)}, {total_memory:.2f} GB")
    except Exception as e:
        print(f"[Init] Warning: {e}")
    
    # 设置默认参数
    if 'num_kvcache_blocks' not in kwargs:
        # A10 24GB，根据模型大小估算
        kwargs['num_kvcache_blocks'] = 2000
        print(f"[Init] Setting default num_kvcache_blocks: {kwargs['num_kvcache_blocks']}")
    
    if 'max_model_len' not in kwargs:
        kwargs['max_model_len'] = 8192
        print(f"[Init] Setting default max_model_len: {kwargs['max_model_len']}")
    
    if 'max_num_seqs' not in kwargs:
        kwargs['max_num_seqs'] = 256
        print(f"[Init] Setting default max_num_seqs: {kwargs['max_num_seqs']}")
    
    if 'max_num_batched_tokens' not in kwargs:
        kwargs['max_num_batched_tokens'] = kwargs['max_model_len']
        print(f"[Init] Setting default max_num_batched_tokens: {kwargs['max_num_batched_tokens']}")
    
    if 'enforce_eager' not in kwargs:
        kwargs['enforce_eager'] = True
        print(f"[Init] Setting default enforce_eager: {kwargs['enforce_eager']}")
    
    # 👈 保存配置到全局变量
    model_config = {
        'max_model_len': kwargs.get('max_model_len', 8192),
        'max_num_batched_tokens': kwargs.get('max_num_batched_tokens', 16384),
        'max_num_seqs': kwargs.get('max_num_seqs', 256),
        'num_kvcache_blocks': kwargs.get('num_kvcache_blocks', 2000),
        'gpu_memory_utilization': kwargs.get('gpu_memory_utilization', 0.9),
        'tensor_parallel_size': kwargs.get('tensor_parallel_size', 1),
        'enforce_eager': kwargs.get('enforce_eager', True),
    }
    
    print(f"\n[Init] ============ Model Configuration ============")
    for key, value in model_config.items():
        print(f"[Init]   {key}: {value}")
    print(f"[Init] =============================================\n")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"[Init] ✅ Tokenizer loaded")
    
    # 创建引擎
    engine = AsyncLLMEngine(model_path, **kwargs)
    print(f"[Init] ✅ Engine created")
    
    model_name = model_path.split("/")[-1]
    print(f"[Init] ✅ Model loaded: {model_name}")


# ============= Main =============

def main():
    parser = argparse.ArgumentParser(
        description="HBserve OpenAI-Compatible API Server"
    )
    parser.add_argument(
        "--model-path", type=str, default="../Qwen3-0.6B",
        help="Path to model"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind (default: 8000)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=100,
        help="Maximum concurrent requests (default: 100)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Tensor parallel size (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="GPU memory utilization (default: 0.9)"
    )
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="Enforce eager execution"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Maximum model sequence length (default: 8192)"
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=None,
        help="Maximum batched tokens (default: same as max-model-len)"
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=256,
        help="Maximum number of sequences (default: 256)"
    )
    parser.add_argument(
        "--num-kvcache-blocks", type=int, default=2000,
        help="Number of KV cache blocks (default: 2000)"
    )
    
    args = parser.parse_args()
    
    # 准备参数
    engine_kwargs = {
        'tensor_parallel_size': args.tensor_parallel_size,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'enforce_eager': args.enforce_eager,
        'max_model_len': args.max_model_len,
        'max_num_seqs': args.max_num_seqs,
        'num_kvcache_blocks': args.num_kvcache_blocks,
    }
    
    # 设置 max_num_batched_tokens
    if args.max_num_batched_tokens is not None:
        engine_kwargs['max_num_batched_tokens'] = args.max_num_batched_tokens
    else:
        engine_kwargs['max_num_batched_tokens'] = args.max_model_len
    
    # Initialize model
    initialize_model(
        args.model_path,
        max_concurrent=args.max_concurrent,
        **engine_kwargs
    )
    
    # Start server
    print(f"\n{'='*60}")
    print(f"🚀 HBserve OpenAI-Compatible API Server")
    print(f"{'='*60}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Model: {model_name}")
    print(f"Max Concurrent: {args.max_concurrent}")
    print(f"\n📍 Endpoints:")
    print(f"  📝 Chat Completions: POST /v1/chat/completions")
    print(f"  📄 Text Completions: POST /v1/completions")
    print(f"  📋 List Models:      GET  /v1/models")
    print(f"  ❤️  Health Check:     GET  /health")
    print(f"  📊 Statistics:       GET  /stats")
    print(f"  ✅ Finish Notify:    POST /finish")
    print(f"\n✨ Features:")
    print(f"  ✅ Async/Await support")
    print(f"  ✅ Concurrent request control")
    print(f"  ✅ Performance monitoring")
    print(f"  ✅ Automatic input truncation")
    print(f"  ✅ Dynamic timeout")
    print(f"  ✅ Streaming output")
    print(f"  ✅ OpenAI compatible")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
