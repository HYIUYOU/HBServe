import argparse
import asyncio
import json
import time
from typing import AsyncGenerator, Optional, List, Dict, Any
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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


# ============= Global State =============

engine: Optional[AsyncLLMEngine] = None
tokenizer: Optional[AutoTokenizer] = None
model_name: str = ""


# ============= Lifespan Management =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期"""
    # Startup
    if engine is not None:
        await engine.start()
        print("[Server] Engine background loop started")
    
    yield  # 应用运行中
    
    # Shutdown
    if engine is not None:
        await engine.stop()
        print("[Server] Engine stopped")


# ============= FastAPI App =============

app = FastAPI(
    title="HBserve OpenAI-Compatible API",
    lifespan=lifespan
)


# ============= Helper Functions =============

def format_chat_completion_response(output, model: str) -> Dict:
    """Format response in OpenAI chat completion format"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
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
    model: str
) -> AsyncGenerator[str, None]:
    """异步流式输出"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    
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


async def stream_completion_async(
    prompt: str,
    sampling_params: SamplingParams,
    model: str
) -> AsyncGenerator[str, None]:
    """异步流式文本补全"""
    request_id = f"cmpl-{uuid.uuid4().hex}"
    
    async for output in engine.generate_stream(prompt, sampling_params):
        chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "text": output.text,
                "finish_reason": "stop" if output.finished else None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"


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
    try:
        # 转换消息为 prompt
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 创建采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        # 异步生成
        if request.stream:
            return StreamingResponse(
                stream_chat_completion_async(prompt, sampling_params, request.model),
                media_type="text/event-stream"
            )
        else:
            # 使用异步引擎
            output = await engine.generate(prompt, sampling_params)
            return format_chat_completion_response(output, request.model)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible text completion endpoint"""
    try:
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        
        if len(prompts) > 1 and request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming only supports single prompt"
            )
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        if request.stream:
            return StreamingResponse(
                stream_completion_async(prompts[0], sampling_params, request.model),
                media_type="text/event-stream"
            )
        else:
            # 并发处理多个 prompt
            outputs = await asyncio.gather(*[
                engine.generate(prompt, sampling_params)
                for prompt in prompts
            ])
            return format_completion_response(outputs, request.model)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "engine_running": engine is not None}


# ============= Server Initialization =============

def initialize_model(model_path: str, **kwargs):
    """Initialize the LLM engine and tokenizer"""
    global engine, tokenizer, model_name
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    engine = AsyncLLMEngine(model_path, **kwargs)
    model_name = model_path.split("/")[-1]
    print(f"Model loaded: {model_name}")


# ============= Main =============

def main():
    parser = argparse.ArgumentParser(
        description="HBserve OpenAI-Compatible API Server"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
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
    
    args = parser.parse_args()
    
    # Initialize model
    initialize_model(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager
    )
    
    # Start server
    print(f"\n{'='*60}")
    print(f"🚀 HBserve OpenAI-Compatible API Server (Async)")
    print(f"{'='*60}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Model: {model_name}")
    print(f"\nEndpoints:")
    print(f"  📝 Chat Completions: POST /v1/chat/completions")
    print(f"  📄 Text Completions: POST /v1/completions")
    print(f"  📋 List Models:      GET  /v1/models")
    print(f"  ❤️  Health Check:     GET  /health")
    print(f"\nFeatures:")
    print(f"  ✅ Async/Await support")
    print(f"  ✅ Concurrent request handling")
    print(f"  ✅ Streaming output")
    print(f"  ✅ OpenAI compatible")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()