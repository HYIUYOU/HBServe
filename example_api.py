from openai import OpenAI
import time
# 同步
# 创建客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # HBserve doesn't require auth
)

def print_section(title: str):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_chat_completion():
    """测试 Chat Completion"""
    print("🔹 Testing Chat Completion...")
    start = time.time()
    
    response = client.chat.completions.create(
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
    print(f"   Tokens: {response.usage.completion_tokens}")

def test_streaming_chat():
    """测试流式 Chat"""
    print("\n🔹 Testing Streaming Chat...")
    start = time.time()
    
    stream = client.chat.completions.create(
        model="Qwen3-0.6B",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True,
        max_tokens=50
    )

    print("✅ Streaming output:")
    print("   ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    elapsed = time.time() - start
    print(f"\n   Time: {elapsed:.2f}s")

def test_text_completion():
    """测试文本补全"""
    print("\n🔹 Testing Text Completion...")
    start = time.time()
    
    response = client.completions.create(
        model="Qwen3-0.6B",
        prompt="Once upon a time",
        max_tokens=50
    )
    
    elapsed = time.time() - start
    print(f"✅ Response ({elapsed:.2f}s):")
    print(f"   {response.choices[0].text}")

def test_multiple_messages():
    """测试多轮对话"""
    print("\n🔹 Testing Multi-turn Conversation...")
    
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"}
    ]
    
    start = time.time()
    response = client.chat.completions.create(
        model="Qwen3-0.6B",
        messages=messages,
        temperature=0.7,
        max_tokens=50
    )
    
    elapsed = time.time() - start
    print(f"✅ Response ({elapsed:.2f}s):")
    print(f"   {response.choices[0].message.content}")

def test_error_handling():
    """测试错误处理"""
    print("\n🔹 Testing Error Handling...")
    
    try:
        # 测试无效的 model
        response = client.chat.completions.create(
            model="invalid-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}")
        print(f"   Message: {str(e)[:100]}...")

def main():
    print_section("🚀 HBserve OpenAI-Compatible API Tests")
    
    try:
        # 测试健康检查
        import requests
        health = requests.get("http://localhost:8000/health").json()
        print(f"✅ Server Status: {health['status']}")
        print(f"   Engine Running: {health.get('engine_running', False)}")
        
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return
    
    # 运行测试
    print_section("Test 1: Chat Completion")
    test_chat_completion()
    
    print_section("Test 2: Streaming Chat")
    test_streaming_chat()
    
    print_section("Test 3: Text Completion")
    test_text_completion()
    
    print_section("Test 4: Multi-turn Conversation")
    test_multiple_messages()
    
    print_section("Test 5: Error Handling")
    test_error_handling()
    
    print_section("✨ All Tests Completed!")

if __name__ == "__main__":
    main()