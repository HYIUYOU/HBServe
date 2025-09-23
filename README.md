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

## 贡献

欢迎提交Issue和Pull Request！

## 参考

- [nanovllm](https://github.com/GeeeekExplorer/nanovllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)
