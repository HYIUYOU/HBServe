#!/usr/bin/env python3
"""
vLLM Server with performance metrics
"""
import argparse
import asyncio
import time
from typing import AsyncGenerator
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.utils import random_uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import json
from collections import defaultdict
import threading

app = FastAPI()

model_path = "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

# Global metrics
class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.latencies = []
        self.start_time = time.time()
        
    def add_request(self, prompt_tokens: int, completion_tokens: int, latency: float):
        with self.lock:
            self.total_requests += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens
            self.latencies.append(latency)
    
    def get_stats(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "throughput_tokens_per_sec": self.total_tokens / elapsed if elapsed > 0 else 0,
                "throughput_requests_per_sec": self.total_requests / elapsed if elapsed > 0 else 0,
                "avg_latency": sum(self.latencies) / len(self.latencies) if self.latencies else 0,
                "p50_latency": sorted(self.latencies)[len(self.latencies)//2] if self.latencies else 0,
                "p95_latency": sorted(self.latencies)[int(len(self.latencies)*0.95)] if self.latencies else 0,
                "p99_latency": sorted(self.latencies)[int(len(self.latencies)*0.99)] if self.latencies else 0,
                "elapsed_time": elapsed
            }
    
    def reset(self):
        with self.lock:
            self.total_requests = 0
            self.total_tokens = 0
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.latencies = []
            self.start_time = time.time()

metrics = Metrics()
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    # Initialize vLLM engine
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"vLLM Engine initialized with model: {args.model}")

@app.post("/v1/completions")
async def completions(request: Request):
    """OpenAI-compatible completions endpoint"""
    request_dict = await request.json()
    prompt = request_dict.get("prompt", "")
    max_tokens = request_dict.get("max_tokens", 256)
    temperature = request_dict.get("temperature", 0.7)
    top_p = request_dict.get("top_p", 1.0)
    
    start_time = time.time()
    request_id = random_uuid()
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # Generate
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    # Collect all results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    # Calculate metrics
    latency = time.time() - start_time
    prompt_tokens = len(final_output.prompt_token_ids)
    completion_tokens = sum(len(output.token_ids) for output in final_output.outputs)
    
    metrics.add_request(prompt_tokens, completion_tokens, latency)
    
    # Return response
    return JSONResponse({
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": args.model,
        "choices": [{
            "text": final_output.outputs[0].text,
            "index": 0,
            "finish_reason": final_output.outputs[0].finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint"""
    request_dict = await request.json()
    messages = request_dict.get("messages", [])
    max_tokens = request_dict.get("max_tokens", 256)
    temperature = request_dict.get("temperature", 0.7)
    top_p = request_dict.get("top_p", 1.0)
    
    # Convert messages to prompt (simple concatenation)
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    start_time = time.time()
    request_id = random_uuid()
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    latency = time.time() - start_time
    prompt_tokens = len(final_output.prompt_token_ids)
    completion_tokens = sum(len(output.token_ids) for output in final_output.outputs)
    
    metrics.add_request(prompt_tokens, completion_tokens, latency)
    
    return JSONResponse({
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": args.model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": final_output.outputs[0].text,
            },
            "index": 0,
            "finish_reason": final_output.outputs[0].finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return metrics.get_stats()

@app.post("/metrics/reset")
async def reset_metrics():
    """Reset metrics"""
    metrics.reset()
    return {"status": "reset"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=model_path, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)