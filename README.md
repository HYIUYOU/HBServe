# 💻 HBserve

基于vLLM实现的推理引擎，支持跨GPU层管理和并行执行优化。

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
### OpenAI 兼容 API 服务
HBserve 支持 OpenAI 兼容的 API 格式，方便集成现有应用。

#### 启动服务器
```bash
python openai_api_server.py \
    --model-path ../Qwen3-0.6B \
    --port 8000 \
    --gpu-memory-utilization 0.6
```
#### API 端点

- POST /v1/chat/completions - Chat 补全
- POST /v1/completions - 文本补全
- GET /v1/models - 列出模型
- GET /health - 健康检查

#### 客户端

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

或者你可以使用我们提供的用例：

```bash
python example_api.py
```


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

## ⚠️ 注意事项

- **内存开销**：层复制会占用额外显存
- **KV Cache同步**：Decode阶段有同步开销
- **设备选择**：副本必须在不同GPU才有效果
- **Batch大小**：Batch越大，并行效果越明显


## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📚 参考

- [nanovllm](https://github.com/GeeeekExplorer/nanovllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)
