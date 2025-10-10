# HBserve

基于vLLM实现的推理引擎

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖
cd HBserve
pip install -e . 

```

### 2. 下载模型

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### 3. 运行示例


```bash
python example.py

```

# HBServe 动态层设备管理

这个功能允许您在HBServe的实际使用场景中将模型的不同层分配到不同的GPU设备上，实现跨GPU的层执行和动态设备调整。

## 功能特性

- **单层设备移动**: 将指定层移动到目标GPU
- **批量设备分布**: 一次性设置多个层的设备位置
- **运行时动态调整**: 在推理过程中动态改变层的设备位置
- **自动tensor传输**: 自动处理跨设备的数据传输
- **设备位置跟踪**: 记录和查询每层的当前设备位置

## 主要方法

### 1. `move_layer_to_device(layer_id, device)`
将指定层移动到目标设备。

**参数:**
- `layer_id` (int): 层的索引，从0开始
- `device` (str | torch.device): 目标设备，如 'cuda:1'

**示例:**
```python
# 将第10层移动到GPU 1
model.move_layer_to_device(9, 'cuda:1')
```

### 2. `set_layer_device_distribution(layer_device_map)`
批量设置层的设备分布。

**参数:**
- `layer_device_map` (dict): 字典，键为层索引，值为目标设备

**示例:**
```python
# 设置多个层的设备分布
layer_device_map = {
    9: 'cuda:1',   # 第10层在GPU 1
    10: 'cuda:1',  # 第11层在GPU 1
    15: 'cuda:2',  # 第16层在GPU 2
}
model.set_layer_device_distribution(layer_device_map)
```

### 3. `get_layer_device(layer_id)`
获取指定层的当前设备。

**参数:**
- `layer_id` (int): 层的索引

**返回:**
- `torch.device`: 层当前所在的设备

## 使用场景

### 1. 负载均衡
将模型层均匀分布到多个GPU上，实现负载均衡：

```python
# 将24层均匀分布到3个GPU
num_gpus = 3
for layer_id in range(len(model.layers)):
    gpu_id = layer_id % num_gpus
    model.move_layer_to_device(layer_id, f'cuda:{gpu_id}')
```

### 2. 特定层优化
将计算密集的层（如注意力层）分配到性能更好的GPU：

```python
# 将注意力层分配到高性能GPU
attention_layers = [9, 10, 11, 12]  # 假设这些是注意力层
for layer_id in attention_layers:
    model.move_layer_to_device(layer_id, 'cuda:1')  # 高性能GPU
```

### 3. 内存优化
将部分层移动到不同的GPU以节省内存：

```python
# 将后半部分层移动到GPU 1
num_layers = len(model.layers)
for layer_id in range(num_layers // 2, num_layers):
    model.move_layer_to_device(layer_id, 'cuda:1')
```

### 4. 运行时动态调整
根据运行时条件动态调整层的设备位置：

```python
# 根据内存使用情况动态调整
if gpu_0_memory_usage > threshold:
    # 将一些层移动到GPU 1
    model.move_layer_to_device(10, 'cuda:1')
    model.move_layer_to_device(11, 'cuda:1')
```

## 实际使用场景

### 在HBServe中使用动态层设备管理

```python
import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer

# 加载模型
model_path = "/path/to/your/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

# 访问底层模型进行层设备管理
if hasattr(llm, 'model') and hasattr(llm.model, 'model'):
    model = llm.model.model
    
    # 将第10层移动到GPU 1
    model.move_layer_to_device(9, 'cuda:1')
    
    # 批量设置层设备分布
    layer_device_map = {
        9: 'cuda:1',   # 第10层
        10: 'cuda:1',  # 第11层
        15: 'cuda:2',  # 第16层
    }
    model.set_layer_device_distribution(layer_device_map)

# 正常进行推理
sampling_params = SamplingParams(temperature=0.6, max_tokens=100)
prompts = ["Tell me a story about AI"]
formatted_prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    for prompt in prompts
]
outputs = llm.generate(formatted_prompts, sampling_params)
```

## 完整示例

运行 `example_layer_device_management.py` 查看完整的使用示例，包括：

- 基本使用示例
- 动态层设备管理
- 运行时动态调整
- 内存优化策略

## 性能建议

1. **减少设备切换**: 尽量将相邻的层放在同一设备上
2. **合理分配**: 根据层的计算复杂度合理分配设备
3. **监控内存**: 注意跨设备传输的内存开销
4. **测试性能**: 在实际工作负载下测试性能影响

## 注意事项

1. **设备兼容性**: 确保目标GPU设备存在且可用
2. **内存管理**: 跨设备传输会增加内存使用和传输开销
3. **性能影响**: 频繁的设备切换可能影响性能
4. **tensor同步**: 系统会自动处理tensor的设备同步



## 贡献

欢迎提交Issue和Pull Request！

## 参考

- [nanovllm](https://github.com/GeeeekExplorer/nanovllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)
