# HBserve pd分离架构

基于HBserve实现的简易pd(prefill-decode)分离架构，支持KV缓存传输、分块prefill和CUDA graph优化。

## 功能特性

### 🚀 核心功能
- **pd分离架构**: 独立的prefill和decode实例
- **KV传输**: 支持NCCL和Mooncake两种backend
- **RPC通信**: CPU调度器与GPU实例间的高效通信
- **分块Prefill**: 支持长序列的分块处理
- **CUDA Graph**: 自动优化重复计算模式

### 📋 技术特点
- **异步调度**: CPU端异步调度，支持高并发
- **内存优化**: KV缓存查找缓冲区，处理乱序请求
- **兼容接口**: 保持与原始HBserve的接口兼容
- **灵活部署**: 支持单机多GPU和多机部署

## 架构设计

```
[CPU调度器] --> RPC --> [Prefill实例(GPU-0)]
                              |
                           KV传输 (NCCL/Mooncake)
                              |
                              v
                        [Decode实例(GPU-1)]
```

### 组件说明

1. **CPU调度器** (`CpuScheduler`)
   - 管理请求队列和分发
   - 协调prefill和decode流程
   - 监控实例状态

2. **GPU实例** (`GpuInstance`)
   - Prefill实例：处理prompt编码
   - Decode实例：处理token生成
   - 支持chunked prefill和CUDA graph

3. **KV传输** (`KVTransferManager`)
   - NCCL backend：基于PyTorch分布式
   - Mooncake backend：高性能RDMA传输
   - 查找缓冲区处理乱序请求

4. **RPC通信** (`RpcClient/RpcServer`)
   - 轻量级TCP协议
   - 异步非阻塞通信
   - 支持多种消息类型

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

-----------------------------------

```python
import asyncio
from HBserve.pd_disagg import DisaggregatedLLM, PdDisaggConfig, KVTransferBackend
from HBserve.sampling_params import SamplingParams

# 创建配置
config = PdDisaggConfig(
    model_path="~/huggingface/Qwen3-0.6B/",
    kv_transfer_backend=KVTransferBackend.NCCL,
    chunked_prefill_enabled=True,
    cuda_graph_enabled=True
)

async def main():
    # 创建分离式LLM
    llm = DisaggregatedLLM(config)
    
    try:
        # 启动系统
        await llm.start()
        
        # 生成文本
        prompts = ["介绍一下人工智能", "什么是机器学习？"]
        sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        
        results = await llm.generate(prompts, sampling_params)
        
        for prompt, result in zip(prompts, results):
            print(f"Prompt: {prompt}")
            print(f"Response: {result['text']}")
            
    finally:
        await llm.stop()

asyncio.run(main())
```

### 4. 兼容接口

```python
# 与原始HBserve兼容的同步接口
from HBserve.pd_disagg import LLM
from HBserve.sampling_params import SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B/", 
          kv_transfer_backend="nccl",
          chunked_prefill_enabled=True)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, pd-disagg HBserve!"]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

## 配置说明

### 基础配置

```python
config = PdDisaggConfig(
    # 模型配置
    model_path="~/huggingface/Qwen3-0.6B/",
    tensor_parallel_size=1,
    max_num_seqs=256,
    max_num_batched_tokens=2048,
    
    # KV传输配置
    kv_transfer_backend=KVTransferBackend.NCCL,  # 或 MOONCAKE
    kv_buffer_size=1024*1024*1024,  # 1GB
    
    # 分块prefill配置
    chunked_prefill_enabled=True,
    chunk_size=512,
    
    # CUDA graph配置
    cuda_graph_enabled=True,
    cuda_graph_max_seq_len=2048
)
```

### YAML配置文件

```yaml
# configs/pd_disagg_config.yaml
model:
  path: "~/huggingface/Qwen3-0.6B/"
  tensor_parallel_size: 1

kv_transfer:
  backend: "nccl"  # "nccl" 或 "mooncake"
  buffer_size: 1073741824  # 1GB
  ip: "127.0.0.1"
  port: 12345

chunked_prefill:
  enabled: true
  chunk_size: 512

cuda_graph:
  enabled: true
  max_seq_len: 2048

gpu_instances:
  - instance_id: "prefill-0"
    instance_type: "prefill"
    gpu_id: 0
    host: "localhost"
    port: 50051
    kv_rank: 0
    
  - instance_id: "decode-0"
    instance_type: "decode"
    gpu_id: 1
    host: "localhost"
    port: 50052
    kv_rank: 1
```

## 部署方式

### 单机部署

```bash
# 启动完整系统
python scripts/start_pd_disagg.py --config configs/pd_disagg_config.yaml

# 或者分别启动实例
python scripts/start_pd_disagg.py --config configs/pd_disagg_config.yaml --instance-id prefill-0
python scripts/start_pd_disagg.py --config configs/pd_disagg_config.yaml --instance-id decode-0
```

### 多机部署

```bash
# 在prefill节点
python scripts/start_pd_disagg.py --config configs/pd_disagg_config.yaml --instance-id prefill-0

# 在decode节点  
python scripts/start_pd_disagg.py --config configs/pd_disagg_config.yaml --instance-id decode-0

# 在调度节点
python scripts/start_cpu_scheduler.py --config configs/pd_disagg_config.yaml
```

## KV传输Backend

### NCCL Backend

- **适用场景**: 单机多GPU或InfiniBand网络
- **优势**: 原生PyTorch支持，配置简单
- **配置**: 自动使用PyTorch分布式

```python
config = PdDisaggConfig(
    kv_transfer_backend=KVTransferBackend.NCCL,
    kv_ip="127.0.0.1",
    kv_port=12345
)
```

### Mooncake Backend

- **适用场景**: 高性能RDMA网络
- **优势**: 更低延迟，更高带宽
- **依赖**: 需要安装Mooncake库

```python
config = PdDisaggConfig(
    kv_transfer_backend=KVTransferBackend.MOONCAKE,
    kv_ip="10.0.0.1",
    kv_port=12346
)
```

```bash
# 设置Mooncake配置
export MOONCAKE_CONFIG_PATH=configs/mooncake_config.json
```

## 分块Prefill

### 工作原理

长序列自动分块处理，每个分块独立进行prefill，支持：

- **自适应分块**: 根据序列长度自动分块
- **CUDA Graph优化**: 固定大小分块使用CUDA Graph
- **内存优化**: 避免长序列的内存峰值

### 配置参数

```python
config = PdDisaggConfig(
    chunked_prefill_enabled=True,
    chunk_size=512,                    # 分块大小
    max_chunk_prefill_tokens=4096,     # 最大分块tokens
    cuda_graph_enabled=True            # 启用CUDA Graph
)
```

## 性能优化

### CUDA Graph

- **自动缓存**: 相同shape的计算图自动缓存
- **适用场景**: 固定batch size和序列长度
- **性能提升**: 减少GPU kernel启动开销

### 内存优化

- **KV缓存复用**: 高效的KV缓存管理
- **分块处理**: 避免长序列内存溢出
- **查找缓冲区**: 处理乱序请求，提高吞吐量

### 批处理优化

- **动态批处理**: 自动组合小批次
- **异步调度**: CPU和GPU并行处理
- **负载均衡**: 多实例间负载分配

## 监控和调试

### 系统统计

```python
# 获取详细统计信息
stats = llm.get_system_stats()

print("CPU调度器统计:")
print(f"  总请求数: {stats['cpu_scheduler']['total_requests']}")
print(f"  完成请求数: {stats['cpu_scheduler']['completed_requests']}")
print(f"  成功率: {stats['cpu_scheduler']['success_rate']:.2%}")

print("GPU实例统计:")
for instance_id, instance_stats in stats["gpu_instances"].items():
    print(f"  {instance_id}:")
    print(f"    状态: {instance_stats['status']}")
    print(f"    活跃请求: {instance_stats['active_requests']}")
```

### 日志配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 调试模式

```bash
# 启用调试日志
python scripts/start_pd_disagg.py --config configs/pd_disagg_config.yaml --log-level DEBUG
```

## 故障排查

### 常见问题

1. **KV传输失败**
   - 检查网络连接和端口
   - 验证NCCL/Mooncake安装
   - 确认GPU设备可见性

2. **RPC连接超时**
   - 检查实例启动顺序
   - 验证端口配置
   - 确认防火墙设置

3. **内存不足**
   - 减少batch size或chunk size
   - 启用分块prefill
   - 调整KV缓存大小

### 性能调优

1. **优化分块大小**
   ```python
   # 根据GPU内存调整
   config.chunk_size = 256  # 减少内存使用
   config.chunk_size = 1024 # 提高效率
   ```

2. **调整缓冲区大小**
   ```python
   # 平衡内存和性能
   config.kv_buffer_size = 512 * 1024 * 1024  # 512MB
   ```

3. **CUDA Graph配置**
   ```python
   # 限制CUDA Graph使用范围
   config.cuda_graph_max_seq_len = 1024
   ```

## 扩展开发

### 添加新的KV传输Backend

```python
from HBserve.pd_disagg.kv_transfer.base import KVTransferBase

class CustomKVTransfer(KVTransferBase):
    async def send_kv_cache(self, request_id, kv_buffer, target_rank):
        # 实现自定义传输逻辑
        pass
        
    async def receive_kv_cache(self, request_id, source_rank):
        # 实现自定义接收逻辑
        pass
```

### 自定义调度策略

```python
from HBserve.pd_disagg.cpu_scheduler import CpuScheduler

class CustomScheduler(CpuScheduler):
    async def _schedule_waiting_requests(self):
        # 实现自定义调度逻辑
        pass
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交Issue和Pull Request！

## 参考

- [nanovllm](https://github.com/GeeeekExplorer/nanovllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)
