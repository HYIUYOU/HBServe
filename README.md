# 💻 HBserve

基于vLLM实现的推理引擎，支持跨GPU层管理和并行执行优化。

## 与其他服务对比

| 特性 | HBserve | vLLM | Text-generation-inference |
|-----|---------|------|--------------------------|
| OpenAI API | ✅ | ✅ | ✅ |
| 异步支持 | ✅ | ✅ | ✅ |
| 层复制优化 | ✅ | ❌ | ❌ |
| 跨GPU管理 | ✅ | ❌ | ❌ |
| 自适应调优 | ✅ | ❌ | ❌ |



## 🔜 快速开始

### 命令行 cli
```bash
# 1. 安装
cd HBserve && pip install -e . 

# 2. 下载模型
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False

# 3. 运行
python example.py
```
## 🌐 OpenAI 兼容 API 服务

HBserve 提供 OpenAI 兼容的 REST API，支持异步并发处理和流式输出。

### 启动服务器

```bash
python openai_api_server.py \
    --model-path ~/huggingface/Qwen3-0.6B \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --enforce-eager
```

**启动参数：**
- `--model-path`: 模型路径（必需）
- `--host`: 绑定地址（默认：0.0.0.0）
- `--port`: 端口号（默认：8000）
- `--tensor-parallel-size`: 张量并行大小（默认：1）
- `--gpu-memory-utilization`: GPU 内存利用率（默认：0.9）
- `--enforce-eager`: 强制 eager 执行

### API 端点

| 端点 | 方法 | 说明 | 流式 |
|-----|------|------|-----|
| `/v1/chat/completions` | POST | Chat 对话补全 | ✅ |
| `/v1/completions` | POST | 文本补全 | ✅ |
| `/v1/models` | GET | 列出可用模型 | - |
| `/health` | GET | 健康检查 | - |

### 功能特性

- ✅ **异步处理**：支持高并发请求
- ✅ **流式输出**：Server-Sent Events (SSE) 格式
- ✅ **OpenAI 兼容**：无缝对接 OpenAI SDK
- ✅ **批量处理**：支持并发多个请求
- ✅ **生命周期管理**：优雅启动和关闭

### 客户端示例

#### 同步调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # HBserve 不需要认证
)

# Chat 补全
response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=100
)
print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### 异步调用（高并发）

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

async def main():
    # 并发处理多个请求
    tasks = [
        client.chat.completions.create(
            model="Qwen3-0.6B",
            messages=[{"role": "user", "content": f"What is {i}+{i}?"}],
            max_tokens=50
        )
        for i in range(5)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Request {i}: {response.choices[0].message.content}")

asyncio.run(main())
```

#### 使用 curl

```bash
# Chat 补全
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# 流式输出
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

### 测试脚本

我们提供了完整的测试示例：

```bash
# 基础测试（同步）
python example_api.py

# 异步并发测试
python example_api_async.py

# 性能基准测试
python benchmark_api.py
```

**测试输出示例：**
```
============================================================
  🚀 HBserve OpenAI-Compatible API Tests (Async)
============================================================

✅ Server Status: ok
   Engine Running: True

============================================================
  Test 1: Chat Completion
============================================================

🔹 Testing Chat Completion...
✅ Response (0.63s):
   <think>
Okay, the user asked...
```

### 支持的参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `temperature` | float | 1.0 | 采样温度 |
| `max_tokens` | int | 64 | 最大生成长度 |
| `stream` | bool | false | 是否流式输出 |
| `messages` | array | - | 对话消息列表 |
| `prompt` | string | - | 文本提示（completions） |

⚠️ **注意**：`top_p`、`n`、`stop` 等参数会被接受但不生效，因为底层引擎暂不支持。


### 架构说明

```
客户端请求 → FastAPI 异步端点 → AsyncLLMEngine
                                      ↓
                              后台推理循环（持续运行）
                                      ↓
                              LLMEngine.step()
                                      ↓
                              返回结果到客户端
```

**关键特性**：
- 请求提交和推理执行解耦
- 后台持续运行推理循环
- 通过 `asyncio.Queue` 传递结果
- 支持动态添加请求


## 📦 核心功能

### 🔧 层设备管理（Layer Device Management）
将不同层分配到不同GPU，解决单卡内存不足问题。

```python
model = llm.model.model

# 单层移动
model.move_layer_to_device(9, 'cuda:1')

# 批量分配
model.set_layer_device_distribution({
    9: 'cuda:1',
    10: 'cuda:1',
    15: 'cuda:2'
})
```

### ⚡ 层复制并行（Layer Replication）
将瓶颈层复制到另一GPU，batch切分并行执行，提升吞吐量。

```python
# 基础用法
model.replicate_layer_to_device(9, 'cuda:1', split_ratio=0.5)

# 启用自适应调优（推荐）
model.enable_replication_autotune(9, beta=0.3, min_ratio=0.2, max_ratio=0.8)

# 查看调优日志
# export HB_REPLICA_LOG=1
```

**工作原理**：将batch切分到原始层（GPU0）和副本层（GPU1）并行计算，自动根据两侧耗时调整切分比例。

### 🧠 Attention Offload（注意力卸载）
针对Attention层的特殊优化，支持两种卸载策略，解决大模型注意力计算的内存和计算瓶颈。

#### 1. Batch Offload（批次卸载）
将attention计算按batch维度切分到不同GPU，适合大batch场景。

```python
# 基础用法
model.attention_offload_by_batch(
    layer_id=9,
    offload_device='cuda:1',
    split_ratio=0.5,
    enable_autotune=False
)

# 启用自适应调优
model.attention_offload_by_batch(
    layer_id=9,
    offload_device='cuda:1',
    split_ratio=0.5,
    enable_autotune=True
)
```

#### 2. KV Head Split（KV头分离）
将attention的K、V头分配到不同GPU，适合多头注意力场景。

```python
# 基础用法
model.attention_offload_by_kv_head(
    layer_id=9,
    offload_device='cuda:1',
    split_kv_head_idx=None,  # None = 均分
    enable_autotune=False
)

# 指定头分离
model.attention_offload_by_kv_head(
    layer_id=9,
    offload_device='cuda:1',
    split_kv_head_idx=[0, 1, 2],  # 指定哪些头在offload设备
    enable_autotune=True
)
```

**工作原理**：
- **Batch Offload**：将输入batch按比例分配到两个GPU，并行计算attention
- **KV Head Split**：将attention的K、V矩阵按头维度分离，减少单GPU内存占用

## 使用示例

```python
import os
from HBserve import LLM, SamplingParams

os.environ['HB_REPLICA_LOG'] = '1'  # 启用日志
llm = LLM(model_path, enforce_eager=True)
model = llm.model.model

# 方案1: 层分布（解决内存问题）
model.set_layer_device_distribution({0: 'cuda:0', 12: 'cuda:1'})

# 方案2: 层复制（提升吞吐量）
model.replicate_layer_to_device(9, 'cuda:1', split_ratio=0.5)
model.enable_replication_autotune(9, beta=0.3)

# 方案3: 组合使用
model.set_layer_device_distribution({15: 'cuda:1'})
model.replicate_layer_to_device(9, 'cuda:2', split_ratio=0.5)

# 方案4: Attention Offload
model.attention_offload_by_batch(9, 'cuda:1', split_ratio=0.5)
model.attention_offload_by_kv_head(10, 'cuda:2', split_kv_head_idx=None)

# 正常推理
outputs = llm.generate(prompts, sampling_params)
```

## 性能调优

### 自适应参数选择

| Beta值 | 适用场景 |
|--------|---------|
| 0.1-0.2 | 稳定工作负载 |
| 0.3-0.5 | 大多数场景（推荐） |
| 0.6-1.0 | 动态变化负载 |

### 最佳实践

1. **识别瓶颈层**：通常是Attention层和大型MLP层
2. **副本设备选择**：确保与原层不在同一GPU
3. **监控日志**：观察`time_a`和`time_b`，判断是否均衡
4. **适用场景**：Prefill阶段效果最佳，大batch效果更明显

## API参考

### 层设备管理
- `move_layer_to_device(layer_id, device)` - 移动单层
- `set_layer_device_distribution(layer_device_map)` - 批量设置
- `get_layer_device(layer_id)` - 查询设备

### 层复制并行
- `replicate_layer_to_device(layer_id, device, split_ratio=0.5)` - 创建副本
- `enable_replication_autotune(layer_id, beta=0.2, min_ratio=0.1, max_ratio=0.9)` - 启用自适应
- `update_replication_split_ratio(layer_id, split_ratio)` - 手动调整比例
- `clear_layer_replication(layer_id=None)` - 清除配置
- `disable_replication_autotune(layer_id)` - 禁用自适应

### Attention Offload
- `attention_offload_by_batch(layer_id, offload_device, split_ratio=0.5, enable_autotune=False)` - 批次卸载
- `attention_offload_by_kv_head(layer_id, offload_device, split_kv_head_idx=None, enable_autotune=False)` - KV头分离
- `clear_attention_offload(layer_id)` - 清除attention offload配置

## ⚠️ 注意事项

- **内存开销**：层复制会占用额外显存
- **KV Cache同步**：Decode阶段有同步开销
- **设备选择**：副本必须在不同GPU才有效果
- **Batch大小**：Batch越大，并行效果越明显
- **Attention Offload**：仅适用于attention层，需要至少2张GPU
- **调试日志**：设置`HB_ATTN_OFFLOAD_LOG=1`查看详细日志


## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📚 参考

- [nanovllm](https://github.com/GeeeekExplorer/nanovllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [FastAPI](https://fastapi.tiangolo.com/)
- [AsyncOpenAI](https://github.com/openai/openai-python#async-usage)
