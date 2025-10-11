import argparse
import asyncio
import json
import time
from typing import AsyncGenerator, Optional, List, Dict, Any
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer


# ============= Request/Response Models =============

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    n: Optional[int] = 1
    stop: Optional[List[str]] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str | List[str]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


# ============= Global State =============

app = FastAPI(title="HBserve OpenAI-Compatible API")

llm: Optional[LLM] = None
tokenizer: Optional[AutoTokenizer] = None
model_name: str = ""


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
        # Convert messages to prompt
        prompt = tokenizer.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in request.messages],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
            n=request.n
        )
        
        # Generate
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(prompt, sampling_params, request.model),
                media_type="text/event-stream"
            )
        else:
            outputs = llm.generate([prompt], sampling_params)
            return format_chat_completion_response(outputs[0], request.model)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible text completion endpoint"""
    try:
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop
        )
        
        if request.stream:
            if len(prompts) > 1:
                raise HTTPException(status_code=400, detail="Streaming only supports single prompt")
            return StreamingResponse(
                stream_completion(prompts[0], sampling_params, request.model),
                media_type="text/event-stream"
            )
        else:
            outputs = llm.generate(prompts, sampling_params)
            return format_completion_response(outputs, request.model)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


# ============= Helper Functions =============

def format_chat_completion_response(output: Dict[str, Any], model: str) -> Dict:
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
                    "content": output["text"]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(output.get("prompt_token_ids", [])),
            "completion_tokens": len(output.get("token_ids", [])),
            "total_tokens": len(output.get("prompt_token_ids", [])) + len(output.get("token_ids", []))
        }
    }

def format_completion_response(outputs: List[Dict[str, Any]], model: str) -> Dict:
    """Format response in OpenAI completion format"""
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": i,
                "text": output["text"],
                "finish_reason": "stop"
            }
            for i, output in enumerate(outputs)
        ]
    }

async def stream_chat_completion(
    prompt: str,
    sampling_params: SamplingParams,
    model: str
) -> AsyncGenerator[str, None]:
    """Stream chat completion in SSE format"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    # Note: HBserve needs to support streaming. This is a placeholder.
    # You'll need to modify HBserve to support async streaming.
    outputs = llm.generate([prompt], sampling_params)
    
    # Simulate streaming by sending the complete response
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": outputs[0]["text"]
                },
                "finish_reason": None
            }
        ]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send finish chunk
    finish_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"

async def stream_completion(
    prompt: str,
    sampling_params: SamplingParams,
    model: str
) -> AsyncGenerator[str, None]:
    """Stream text completion in SSE format"""
    request_id = f"cmpl-{uuid.uuid4().hex}"
    
    outputs = llm.generate([prompt], sampling_params)
    
    chunk = {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": outputs[0]["text"],
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ============= Server Initialization =============

def initialize_model(model_path: str, **kwargs):
    """Initialize the LLM and tokenizer"""
    global llm, tokenizer, model_name
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, **kwargs)
    model_name = model_path.split("/")[-1]
    print(f"Model loaded: {model_name}")


# ============= Main =============

def main():
    parser = argparse.ArgumentParser(description="HBserve OpenAI-Compatible API Server")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--enforce-eager", action="store_true", help="Enforce eager execution")
    
    args = parser.parse_args()
    
    # Initialize model
    initialize_model(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager
    )
    
    # Start server
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API endpoints:")
    print(f"  - Chat: POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  - Completion: POST http://{args.host}:{args.port}/v1/completions")
    print(f"  - Models: GET http://{args.host}:{args.port}/v1/models")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()