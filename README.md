
<h1 align="center">💻 HBserve</h1>
<p align="center">基于vLLM实现的轻量级推理引擎，支持跨GPU层管理、并行执行优化和注意力卸载。</p>


## ✨ 核心特性

- 🚀 **高性能推理** - 与vLLM相当的推理速度，支持Flash Attention和CUDA Graph优化
- 🔧 **跨GPU层管理** - 智能将不同层分配到不同GPU，解决单卡内存不足问题
- ⚡ **层复制并行** - 瓶颈层复制到多GPU并行计算，显著提升吞吐量
- 🧠 **注意力卸载** - 支持Batch和KV Head两种注意力卸载策略
- 🔄 **自适应调优** - 根据实际负载自动调整并行比例
- 🌐 **OpenAI兼容** - 完整的REST API，支持流式输出和异步处理

## 📊 性能对比

| 特性 | HBserve | vLLM | Text-generation-inference |
|-----|---------|------|--------------------------|
| OpenAI API | ✅ | ✅ | ✅ |
| 异步支持 | ✅ | ✅ | ✅ |
| 层复制优化 | ✅ | ❌ | ❌ |
| 跨GPU管理 | ✅ | ❌ | ❌ |
| 自适应调优 | ✅ | ❌ | ❌ |
| 注意力卸载 | ✅ | ❌ | ❌ |
| 轻量级实现 | ✅ | ❌ | ❌ |

## 🚀 快速开始

### 安装

```bash
# 从源码安装
git clone https://github.com/HYIUYOU/HBServe.git
cd HBServe
pip install -e .

# 或直接安装
pip install git+https://github.com/HYIUYOU/HBServe.git
```

### 依赖要求

- Python >= 3.10, < 3.13
- PyTorch >= 2.4.0
- CUDA >= 11.8
- 其他依赖见 `pyproject.toml`

### 基础使用

```bash
# 1. 下载模型
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False

# 2. 运行基础示例
python example.py

# 3. 运行层管理示例
python quick_start_layer_management.py
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

## 💡 使用示例

### 基础推理

```python
import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer

# 加载模型
model_path = "~/huggingface/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

# 准备输入
prompts = [
    "Hello, how are you?",
    "What is machine learning?"
]

# 应用聊天模板
formatted_prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    for prompt in prompts
]

# 生成回复
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(formatted_prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"问题: {prompt}")
    print(f"回答: {output['text']}\n")
```

### 高级优化配置

```python
import os
from HBserve import LLM, SamplingParams

# 启用调试日志
os.environ['HB_REPLICA_LOG'] = '1'
os.environ['HB_ATTN_OFFLOAD_LOG'] = '1'

llm = LLM(model_path, enforce_eager=True)
model = llm.model.model

# 方案1: 层分布（解决内存问题）
model.set_layer_device_distribution({
    0: 'cuda:0',    # 前几层在GPU 0
    12: 'cuda:1',   # 中间层在GPU 1
    20: 'cuda:2'    # 后几层在GPU 2
})

# 方案2: 层复制（提升吞吐量）
model.replicate_layer_to_device(9, 'cuda:1', split_ratio=0.5)
model.enable_replication_autotune(9, beta=0.3, min_ratio=0.2, max_ratio=0.8)

# 方案3: Attention Offload
model.attention_offload_by_batch(9, 'cuda:1', split_ratio=0.5, enable_autotune=True)
model.attention_offload_by_kv_head(10, 'cuda:2', split_kv_head_idx=None, enable_autotune=True)

# 方案4: 组合使用（推荐）
model.set_layer_device_distribution({15: 'cuda:1'})
model.replicate_layer_to_device(9, 'cuda:2', split_ratio=0.5)
model.attention_offload_by_batch(11, 'cuda:3', split_ratio=0.5)

# 开始推理
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)
```

### 最佳实践

1. **识别瓶颈层**：使用`HB_DEBUG=1`监控各层耗时，通常是Attention层和大型MLP层
2. **副本设备选择**：确保与原层不在同一GPU，避免内存竞争
3. **监控日志**：观察`time_a`和`time_b`，判断是否均衡
4. **适用场景**：Prefill阶段效果最佳，大batch效果更明显
5. **内存管理**：合理设置`gpu_memory_utilization`，避免OOM

## 📚 API参考

### 层设备管理

| 方法 | 参数 | 说明 |
|------|------|------|
| `move_layer_to_device(layer_id, device)` | `layer_id: int`<br>`device: str\|torch.device` | 将指定层移动到目标设备 |
| `set_layer_device_distribution(layer_device_map)` | `layer_device_map: dict[int, str]` | 批量设置层设备分布 |
| `get_layer_device(layer_id)` | `layer_id: int` | 查询指定层的当前设备 |

**示例：**
```python
model = llm.model.model

# 单层移动
model.move_layer_to_device(9, 'cuda:1')

# 批量设置
model.set_layer_device_distribution({
    0: 'cuda:0',
    12: 'cuda:1',
    20: 'cuda:2'
})

# 查询设备
device = model.get_layer_device(9)
```

### 层复制并行

| 方法 | 参数 | 说明 |
|------|------|------|
| `replicate_layer_to_device(layer_id, device, split_ratio=0.5)` | `layer_id: int`<br>`device: str`<br>`split_ratio: float` | 创建层副本并设置切分比例 |
| `enable_replication_autotune(layer_id, beta=0.3, min_ratio=0.1, max_ratio=0.9)` | `layer_id: int`<br>`beta: float`<br>`min_ratio: float`<br>`max_ratio: float` | 启用自适应调优 |
| `update_replication_split_ratio(layer_id, split_ratio)` | `layer_id: int`<br>`split_ratio: float` | 手动调整切分比例 |
| `clear_layer_replication(layer_id=None)` | `layer_id: int\|None` | 清除复制配置 |
| `disable_replication_autotune(layer_id)` | `layer_id: int` | 禁用自适应调优 |

**示例：**
```python
# 创建副本
model.replicate_layer_to_device(9, 'cuda:1', split_ratio=0.5)

# 启用自适应调优
model.enable_replication_autotune(9, beta=0.3, min_ratio=0.2, max_ratio=0.8)

# 手动调整比例
model.update_replication_split_ratio(9, 0.6)
```

### Attention Offload

| 方法 | 参数 | 说明 |
|------|------|------|
| `attention_offload_by_batch(layer_id, offload_device, split_ratio=0.5, enable_autotune=False)` | `layer_id: int`<br>`offload_device: str`<br>`split_ratio: float`<br>`enable_autotune: bool` | 按batch维度卸载attention |
| `attention_offload_by_kv_head(layer_id, offload_device, split_kv_head_idx=None, enable_autotune=False)` | `layer_id: int`<br>`offload_device: str`<br>`split_kv_head_idx: list\|None`<br>`enable_autotune: bool` | 按KV头维度卸载attention |
| `clear_attention_offload(layer_id)` | `layer_id: int` | 清除attention offload配置 |

**示例：**
```python
# Batch卸载
model.attention_offload_by_batch(9, 'cuda:1', split_ratio=0.5, enable_autotune=True)

# KV头分离
model.attention_offload_by_kv_head(10, 'cuda:2', split_kv_head_idx=[0,1,2])

# 清除配置
model.clear_attention_offload(9)
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `HB_DEBUG` | 启用调试日志 | `0` |
| `HB_REPLICA_LOG` | 启用层复制日志 | `0` |
| `HB_ATTN_OFFLOAD_LOG` | 启用attention offload日志 | `0` |

## ⚠️ 注意事项

- **内存开销**：层复制会占用额外显存
- **KV Cache同步**：Decode阶段有同步开销
- **设备选择**：副本必须在不同GPU才有效果
- **Batch大小**：Batch越大，并行效果越明显
- **Attention Offload**：仅适用于attention层，需要至少2张GPU
- **调试日志**：设置`HB_ATTN_OFFLOAD_LOG=1`查看详细日志


## 🏗️ 项目结构

```
HBServe/
├── HBserve/                    # 核心库
│   ├── engine/                 # 推理引擎
│   │   ├── async_llm_engine.py # 异步引擎
│   │   ├── llm_engine.py       # 主引擎
│   │   ├── model_runner.py     # 模型运行器
│   │   └── scheduler.py         # 调度器
│   ├── layers/                 # 神经网络层
│   │   ├── attention.py        # 注意力层
│   │   ├── linear.py          # 线性层
│   │   └── rotary_embedding.py # 旋转位置编码
│   ├── models/                 # 模型定义
│   │   └── qwen3.py           # Qwen3模型
│   └── utils/                  # 工具函数
├── example.py                   # 基础示例
├── example_api.py              # API同步示例
├── example_api_async.py        # API异步示例
├── banchmark.py               # 性能基准测试
├── openai_api_server.py       # OpenAI API服务器
└── quick_start_layer_management.py # 层管理快速开始
```

## 🤝 贡献

我们欢迎各种形式的贡献！

### 贡献方式

1. **报告问题**：在GitHub Issues中报告bug或提出功能请求
2. **提交代码**：Fork项目并提交Pull Request
3. **改进文档**：完善README、注释或示例代码
4. **性能优化**：优化推理性能或内存使用

### 开发环境

```bash
# 克隆项目
git clone https://github.com/HYIUYOU/HBServe.git
cd HBServe

# 安装开发依赖
pip install -e .

# 运行测试
python example.py
python example_api_async.py
```

## 📚 参考

- [nanovllm](https://github.com/GeeeekExplorer/nanovllm) - 轻量级vLLM实现
- [vLLM](https://github.com/vllm-project/vllm) - 高性能LLM推理引擎
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - KV Cache优化
- [OpenAI API](https://platform.openai.com/docs/api-reference) - OpenAI API规范
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Python Web框架
- [AsyncOpenAI](https://github.com/openai/openai-python#async-usage) - OpenAI异步客户端

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**

[GitHub](https://github.com/HYIUYOU/HBServe) • [Issues](https://github.com/HYIUYOU/HBServe/issues) • [Pull Requests](https://github.com/HYIUYOU/HBServe/pulls)

</div>
