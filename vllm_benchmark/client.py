#!/usr/bin/env python3
"""
Benchmark client with dynamic RPS
"""
import argparse
import asyncio
import aiohttp
import time
import json
from typing import List, Dict
from data_loader import DatasetLoader
import numpy as np
from collections import defaultdict

class BenchmarkClient:
    def __init__(self, server_url: str, dataset_loader: DatasetLoader):
        self.server_url = server_url
        self.dataset_loader = dataset_loader
        self.results = []
        self.start_time = None
        
    async def send_request(self, session: aiohttp.ClientSession, request_data: Dict, request_id: int):
        """Send a single request"""
        start_time = time.time()
        
        try:
            # 验证 prompt 不为空
            if not request_data.get("prompt", "").strip():
                print(f"Request {request_id} skipped: empty prompt")
                self.results.append({
                    "request_id": request_id,
                    "timestamp": start_time - self.start_time,
                    "success": False,
                    "error": "Empty prompt"
                })
                return
            
            # Determine endpoint based on data type
            if request_data["type"] == "sharegpt" and "messages" in request_data:
                endpoint = f"{self.server_url}/v1/chat/completions"
                payload = {
                    "messages": request_data["messages"][:1],  # Only first message
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                }
            else:
                endpoint = f"{self.server_url}/v1/completions"
                payload = {
                    "prompt": request_data["prompt"],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                }
            
            # 长文本需要更长的超时时间
            timeout_seconds = 600 if request_data.get("type") == "longbench" else 300
            
            async with session.post(
                endpoint, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as response:
                result = await response.json()
                end_time = time.time()
                
                latency = end_time - start_time
                tokens = result.get("usage", {})
                
                self.results.append({
                    "request_id": request_id,
                    "latency": latency,
                    "prompt_tokens": tokens.get("prompt_tokens", 0),
                    "completion_tokens": tokens.get("completion_tokens", 0),
                    "total_tokens": tokens.get("total_tokens", 0),
                    "timestamp": start_time - self.start_time,
                    "success": True,
                    "task": request_data.get("task", ""),
                    "original_length": request_data.get("original_length", 0),
                })
        
        except asyncio.TimeoutError:
            end_time = time.time()
            print(f"Request {request_id} timeout after {end_time - start_time:.1f}s")
            self.results.append({
                "request_id": request_id,
                "latency": end_time - start_time,
                "timestamp": start_time - self.start_time,
                "success": False,
                "error": "Timeout"
            })
        except Exception as e:
            end_time = time.time()
            print(f"Request {request_id} failed: {e}")
            self.results.append({
                "request_id": request_id,
                "latency": end_time - start_time,
                "timestamp": start_time - self.start_time,
                "success": False,
                "error": str(e)
            })
    
    async def run_benchmark(self, duration: int, rps_schedule: List[tuple]):
        """
        Run benchmark with dynamic RPS
        
        Args:
            duration: Total duration in seconds
            rps_schedule: List of (time, rps) tuples, e.g., [(0, 10), (30, 20), (60, 30)]
        """
        self.start_time = time.time()
        self.results = []
        
        connector = aiohttp.TCPConnector(limit=1000)
        async with aiohttp.ClientSession(connector=connector) as session:
            request_id = 0
            tasks = []
            
            # Sort schedule by time
            rps_schedule = sorted(rps_schedule, key=lambda x: x[0])
            schedule_idx = 0
            current_rps = rps_schedule[0][1] if rps_schedule else 10
            
            while time.time() - self.start_time < duration:
                current_time = time.time() - self.start_time
                
                # Update RPS based on schedule
                while schedule_idx < len(rps_schedule) - 1 and current_time >= rps_schedule[schedule_idx + 1][0]:
                    schedule_idx += 1
                    current_rps = rps_schedule[schedule_idx][1]
                    print(f"[{current_time:.1f}s] Switching to {current_rps} RPS")
                
                # Calculate inter-request delay
                if current_rps > 0:
                    delay = 1.0 / current_rps
                else:
                    delay = 1.0
                
                # Send request
                request_data = self.dataset_loader.get_request()
                task = asyncio.create_task(
                    self.send_request(session, request_data, request_id)
                )
                tasks.append(task)
                request_id += 1
                
                # Wait for delay
                await asyncio.sleep(delay)
            
            # Wait for all pending requests
            print("Waiting for pending requests...")
            await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"Benchmark completed. Sent {request_id} requests.")
    
    def print_statistics(self):
        """Print benchmark statistics"""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        if not successful:
            print("No successful requests!")
            return
        
        latencies = [r["latency"] for r in successful]
        total_tokens = sum(r["total_tokens"] for r in successful)
        prompt_tokens = sum(r["prompt_tokens"] for r in successful)
        completion_tokens = sum(r["completion_tokens"] for r in successful)
        
        elapsed = max(r["timestamp"] + r["latency"] for r in successful)
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Total Requests: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Success Rate: {len(successful)/len(self.results)*100:.2f}%")
        print(f"\nDuration: {elapsed:.2f}s")
        print(f"Throughput: {len(successful)/elapsed:.2f} requests/s")
        print(f"Token Throughput: {total_tokens/elapsed:.2f} tokens/s")
        print(f"\nTokens:")
        print(f"  Total: {total_tokens}")
        print(f"  Prompt: {prompt_tokens}")
        print(f"  Completion: {completion_tokens}")
        print(f"\nLatency (seconds):")
        print(f"  Mean: {np.mean(latencies):.3f}")
        print(f"  Median (P50): {np.median(latencies):.3f}")
        print(f"  P95: {np.percentile(latencies, 95):.3f}")
        print(f"  P99: {np.percentile(latencies, 99):.3f}")
        print(f"  Min: {np.min(latencies):.3f}")
        print(f"  Max: {np.max(latencies):.3f}")
        print("="*60)
        
        # 只在指定 --save-results 时保存文件
        if args.save_results:
            output_file = f"benchmark_results_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "summary": {
                        "total_requests": len(self.results),
                        "successful": len(successful),
                        "failed": len(failed),
                        "duration": elapsed,
                        "throughput_rps": len(successful)/elapsed,
                        "throughput_tokens_per_sec": total_tokens/elapsed,
                        "total_tokens": total_tokens,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "latency_mean": float(np.mean(latencies)),
                        "latency_p50": float(np.median(latencies)),
                        "latency_p95": float(np.percentile(latencies, 95)),
                        "latency_p99": float(np.percentile(latencies, 99)),
                    },
                    "detailed_results": self.results
                }, f, indent=2)
            print(f"\n✅ Detailed results saved to: {output_file}")
        else:
            print(f"\n💡 Use --save-results to save detailed results to JSON file")

def parse_rps_schedule(schedule_str: str) -> List[tuple]:
    """
    Parse RPS schedule string
    Format: "0:10,30:20,60:30" means 10 RPS at 0s, 20 RPS at 30s, 30 RPS at 60s
    """
    if not schedule_str:
        return [(0, 10)]  # Default: 10 RPS
    
    schedule = []
    for pair in schedule_str.split(','):
        time_str, rps_str = pair.split(':')
        schedule.append((int(time_str), int(rps_str)))
    return schedule


# python client.py --dataset sharegpt --dataset-path ../data/sharegpt_data.json --duration 60 --rps-schedule "0:10,30:20" --save-results

# 调试 LongBench，不保存
# python client.py --dataset longbench --dataset-path ../data/longbench/narrativeqa.jsonl --max-input-length 2048 --duration 30 --rps-schedule "0:5"

# 正式测试 LongBench，保存结果
# python client.py  --dataset longbench --dataset-path ../data/longbench/narrativeqa.jsonl --max-input-length 4096 --tokenizer "Qwen/Qwen-7B" --duration 60 --rps-schedule "0:5" --save-results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    parser.add_argument("--dataset", type=str, 
                        choices=["alpaca", "sharegpt", "longbench"], 
                        required=True)
    parser.add_argument("--dataset-path", type=str, help="Path to local dataset file")
    parser.add_argument("--duration", type=int, default=60, 
                        help="Benchmark duration in seconds")
    parser.add_argument("--rps-schedule", type=str, default="0:10",
                        help="RPS schedule, format: '0:10,30:20,60:30'")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    # LongBench 专用参数
    parser.add_argument("--max-input-length", type=int, default=None,
                        help="Max input length in tokens (for truncation)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name for accurate truncation")
    
    # 控制是否保存结果文件
    parser.add_argument("--save-results", action="store_true", default=False,
                        help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset_loader = DatasetLoader(
        args.dataset, 
        args.dataset_path,
        max_length=args.max_input_length,
        tokenizer_name=args.tokenizer
    )
    
    # Parse RPS schedule
    rps_schedule = parse_rps_schedule(args.rps_schedule)
    print(f"RPS Schedule: {rps_schedule}")
    
    # Run benchmark
    client = BenchmarkClient(args.server_url, dataset_loader)
    asyncio.run(client.run_benchmark(args.duration, rps_schedule))
    
    # Print statistics
    client.print_statistics()