import asyncio
import time
from openai import AsyncOpenAI

# 创建异步客户端
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

def print_section(title: str):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

async def test_chat_completion():
    """测试 Chat Completion"""
    print("🔹 Testing Chat Completion...")
    start = time.time()
    
    response = await client.chat.completions.create(
        model="Qwen3-0.6B",
        messages=[
            {"role": "user", "content": "Hello! Who are you?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    elapsed = time.time() - start
    print(f"✅ Response ({elapsed:.2f}s):")
    print(f"   {response.choices[0].message.content}")

async def test_streaming_chat():
    """测试流式 Chat"""
    print("\n🔹 Testing Streaming Chat...")
    start = time.time()
    
    stream = await client.chat.completions.create(
        model="Qwen3-0.6B",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True,
        max_tokens=50
    )

    print("✅ Streaming output:")
    print("   ", end="")
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    elapsed = time.time() - start
    print(f"\n   Time: {elapsed:.2f}s")

async def test_concurrent_requests():
    """测试并发请求（异步的优势）"""
    print("\n🔹 Testing Concurrent Requests...")
    
    # 创建多个并发请求
    tasks = [
        client.chat.completions.create(
            model="Qwen3-0.6B",
            messages=[{"role": "user", "content": f"What is {i} + {i}?"}],
            max_tokens=20
        )
        for i in range(1, 4)
    ]
    
    start = time.time()
    # 并发执行所有请求
    responses = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    print(f"✅ Completed {len(responses)} requests concurrently in {elapsed:.2f}s")
    for i, response in enumerate(responses, 1):
        content = response.choices[0].message.content
        print(f"   Request {i}: {content[:50]}...")

async def test_concurrent_vs_sequential():
    """对比并发 vs 顺序执行"""
    print("\n🔹 Comparing Concurrent vs Sequential...")
    
    # 顺序执行
    print("  ⏳ Sequential execution...")
    start = time.time()
    for i in range(3):
        await client.chat.completions.create(
            model="Qwen3-0.6B",
            messages=[{"role": "user", "content": f"Say {i}"}],
            max_tokens=10
        )
    sequential_time = time.time() - start
    
    # 并发执行
    print("  ⏳ Concurrent execution...")
    start = time.time()
    tasks = [
        client.chat.completions.create(
            model="Qwen3-0.6B",
            messages=[{"role": "user", "content": f"Say {i}"}],
            max_tokens=10
        )
        for i in range(3)
    ]
    await asyncio.gather(*tasks)
    concurrent_time = time.time() - start
    
    print(f"\n✅ Results:")
    print(f"   Sequential: {sequential_time:.2f}s")
    print(f"   Concurrent: {concurrent_time:.2f}s")
    print(f"   Speedup: {sequential_time/concurrent_time:.2f}x")

async def test_batch_processing():
    """测试批量处理"""
    print("\n🔹 Testing Batch Processing...")
    
    prompts = [
        "What is AI?",
        "What is machine learning?",
        "What is deep learning?",
        "What is neural network?",
        "What is NLP?"
    ]
    
    start = time.time()
    tasks = [
        client.chat.completions.create(
            model="Qwen3-0.6B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30
        )
        for prompt in prompts
    ]
    
    responses = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    print(f"✅ Processed {len(prompts)} prompts in {elapsed:.2f}s")
    print(f"   Average: {elapsed/len(prompts):.2f}s per request")
    print(f"   Throughput: {len(prompts)/elapsed:.2f} req/s")

async def main():
    print_section("🚀 HBserve OpenAI-Compatible API Tests (Async)")
    
    try:
        # 测试健康检查（使用 httpx 或 aiohttp）
        import httpx
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get("http://localhost:8000/health")
            health = response.json()
            print(f"✅ Server Status: {health['status']}")
            print(f"   Engine Running: {health.get('engine_running', False)}")
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return
    
    # 运行测试
    print_section("Test 1: Chat Completion")
    await test_chat_completion()
    
    print_section("Test 2: Streaming Chat")
    await test_streaming_chat()
    
    print_section("Test 3: Concurrent Requests")
    await test_concurrent_requests()
    
    print_section("Test 4: Concurrent vs Sequential")
    await test_concurrent_vs_sequential()
    
    print_section("Test 5: Batch Processing")
    await test_batch_processing()
    
    print_section("✨ All Async Tests Completed!")

if __name__ == "__main__":
    # 安装依赖: pip install httpx
    asyncio.run(main())