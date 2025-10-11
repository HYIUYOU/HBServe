from openai import OpenAI

# 创建客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # HBserve doesn't require auth
)

# Chat Completion
response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "Hello! Who are you?"}
    ],
    temperature=0.7,
    max_tokens=100
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    max_tokens=200
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")