import time
import asyncio
from openai import OpenAI, AsyncOpenAI

def benchmark_sync(n=5):
    """同步基准测试"""
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    
    print(f"\n📊 Synchronous Benchmark ({n} requests)")
    start = time.time()
    
    for i in range(n):
        response = client.chat.completions.create(
            model="Qwen3-0.6B",
            messages=[{"role": "user", "content": f"Say number {i}"}],
            max_tokens=10
        )
    
    elapsed = time.time() - start
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Avg per request: {elapsed/n:.2f}s")
    print(f"   Throughput: {n/elapsed:.2f} req/s")
    return elapsed

async def benchmark_async(n=5):
    """异步基准测试"""
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    
    print(f"\n📊 Asynchronous Benchmark ({n} requests)")
    start = time.time()
    
    tasks = [
        client.chat.completions.create(
            model="Qwen3-0.6B",
            messages=[{"role": "user", "content": f"Say number {i}"}],
            max_tokens=10
        )
        for i in range(n)
    ]
    
    await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Avg per request: {elapsed/n:.2f}s")
    print(f"   Throughput: {n/elapsed:.2f} req/s")
    return elapsed

async def main():
    print("="*60)
    print("  HBserve API Performance Benchmark")
    print("="*60)
    
    # 同步测试
    sync_time = benchmark_sync(5)
    
    # 异步测试
    async_time = await benchmark_async(5)
    
    # 对比
    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}")
    print(f"  Synchronous:  {sync_time:.2f}s")
    print(f"  Asynchronous: {async_time:.2f}s")
    print(f"  Speedup:      {sync_time/async_time:.2f}x")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(main())