# vLLM 性能测试工具

一个功能完善的 vLLM 推理性能测试工具，支持动态 RPS 调度和多种数据集。

## ✨ 特性

- 🚀 **动态 RPS 调度**：通过时间节点控制请求速率，模拟真实流量模式
- 📊 **多数据集支持**：
  - Alpaca（指令遵循）
  - ShareGPT（对话型）
  - LongBench（长文本，支持自动截断）
- 📈 **全面的性能指标**：
  - 吞吐量（请求/秒，token/秒）
  - 延迟统计（P50、P95、P99）
  - Token 使用情况（输入、输出、总计）
  - 成功/失败率
- 🎯 **智能截断**：基于 tokenizer 的输入长度控制
- 💾 **可选 JSON 导出**：保存详细结果用于后续分析
- ⚡ **高并发支持**：支持数千并发请求

## 📋 环境要求
```bash
Python 3.8+
vllm >= 0.5.0
fastapi
uvicorn
aiohttp
numpy
transformers
torch
```

## 🔧 安装
```bash
# 安装依赖
pip install vllm fastapi uvicorn aiohttp numpy transformers torch

# 克隆仓库
git https://github.com/HYIUYOU/HBServe.git
cd vllm-benchmark
```

## 🚀 快速开始

### 1. 启动 vLLM 服务端
```bash
# 基础启动
python server.py --model meta-llama/Llama-2-7b-hf

# 多 GPU 启动
python server.py \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256
```

### 2. 运行性能测试
```bash
# ShareGPT 数据集快速测试
python client.py \
    --dataset sharegpt \
    --dataset-path data/sharegpt_data.json \
    --duration 60 \
    --rps-schedule "0:10"

# 保存结果到 JSON
python client.py \
    --dataset sharegpt \
    --dataset-path data/sharegpt_data.json \
    --duration 60 \
    --rps-schedule "0:10" \
    --save-results
```

## 📖 使用指南

### 服务端参数
```bash
python server.py [选项]

选项：
  --model TEXT                    模型名称或路径（必需）
  --host TEXT                     服务器地址 [默认: 0.0.0.0]
  --port INTEGER                  服务器端口 [默认: 8000]
  --tensor-parallel-size INTEGER  GPU 数量 [默认: 1]
  --max-model-len INTEGER         最大序列长度 [默认: 4096]
  --gpu-memory-utilization FLOAT  GPU 显存使用率 [默认: 0.9]
  --max-num-seqs INTEGER          最大并发序列数 [默认: 256]
```

### 客户端参数
```bash
python client.py [选项]

必需参数：
  --dataset [alpaca|sharegpt|longbench]  数据集类型
  --dataset-path TEXT                    数据集文件路径

可选参数：
  --server-url TEXT           服务器地址 [默认: http://localhost:8000]
  --duration INTEGER          测试时长（秒）[默认: 60]
  --rps-schedule TEXT         RPS 调度计划 [默认: "0:10"]
  --max-tokens INTEGER        最大输出 token 数 [默认: 256]
  --temperature FLOAT         采样温度 [默认: 0.7]
  --save-results             保存详细结果到 JSON 文件
  
LongBench 专用参数：
  --max-input-length INTEGER  最大输入 token 数（用于截断）
  --tokenizer TEXT            Tokenizer 名称（精确截断）
```

## 📝 数据集使用示例

### Alpaca 数据集
```bash
python client.py \
    --dataset alpaca \
    --dataset-path data/alpaca_data.json \
    --duration 60 \
    --rps-schedule "0:10"
```

### ShareGPT 数据集
```bash
# 固定 RPS
python client.py \
    --dataset sharegpt \
    --dataset-path data/sharegpt_data.json \
    --duration 60 \
    --rps-schedule "0:10"

# 动态 RPS（逐步增加）
python client.py \
    --dataset sharegpt \
    --dataset-path data/sharegpt_data.json \
    --duration 120 \
    --rps-schedule "0:10,30:20,60:30,90:15" \
    --save-results
```

### LongBench 数据集
```bash
# 使用字符估算截断
python client.py \
    --dataset longbench \
    --dataset-path data/longbench/narrativeqa.jsonl \
    --max-input-length 2048 \
    --duration 30 \
    --rps-schedule "0:5"

# 使用 tokenizer 精确截断（推荐）
python client.py \
    --dataset longbench \
    --dataset-path data/longbench/narrativeqa.jsonl \
    --max-input-length 4096 \
    --tokenizer "Qwen/Qwen-7B" \
    --duration 60 \
    --rps-schedule "0:5" \
    --save-results
```

## 🎯 RPS 调度详解

### 什么是 RPS？

**RPS (Requests Per Second，每秒请求数)** 控制测试的负载强度。

- `RPS = 10` → 每秒发送 10 个请求（每隔 0.1 秒发送一个）
- `RPS = 5` → 每秒发送 5 个请求（每隔 0.2 秒发送一个）
- `RPS = 1` → 每秒发送 1 个请求

### 调度格式
```bash
--rps-schedule "时间1:RPS1,时间2:RPS2,时间3:RPS3"
```

### 常见模式

#### 1. 固定负载
```bash
--rps-schedule "0:10"
# 整个测试期间保持 10 RPS
```

#### 2. 预热模式
```bash
--rps-schedule "0:5,30:10"
# 0-30秒：5 RPS（预热阶段）
# 30秒-结束：10 RPS（正常负载）
```

#### 3. 压力测试（逐步加压）
```bash
--rps-schedule "0:5,20:10,40:20,60:30"
# 逐步增加负载，找到性能边界
#
# 0-20秒：  5 RPS（基准）
# 20-40秒：10 RPS（正常）
# 40-60秒：20 RPS（高负载）
# 60秒-结束：30 RPS（峰值）
```

#### 4. 真实流量模拟
```bash
--rps-schedule "0:10,30:20,60:15,90:25,120:10"
# 模拟真实业务流量波动
#
# 0-30秒：  10 RPS（正常）
# 30-60秒： 20 RPS（高峰期）
# 60-90秒： 15 RPS（回落）
# 90-120秒：25 RPS（第二波高峰）
# 120秒-结束：10 RPS（恢复正常）
```

### RPS 选择指南

| 场景 | 推荐 RPS | 说明 |
|------|---------|------|
| 🐛 **调试阶段** | `0:1` 或 `0:2` | 方便观察日志 |
| 🧪 **开发测试** | `0:5` | 快速验证功能 |
| 📊 **性能测试** | `0:10` 到 `0:20` | 标准负载 |
| 🔥 **压力测试** | `0:50` 到 `0:100` | 找性能瓶颈 |
| 📏 **长文本测试** | `0:2` 到 `0:5` | 处理速度较慢 |

### 性能关系
```
✅ 系统能承受时：
RPS ↑  →  吞吐量 (tokens/s) ↑
RPS ↑  →  GPU 利用率 ↑
RPS ↑  →  并发请求数 ↑

❌ 系统过载时：
RPS ↑↑ →  延迟 ↑↑↑
RPS ↑↑ →  队列堆积 ↑
RPS ↑↑ →  失败率 ↑
RPS ↑↑ →  吞吐量不再增加（饱和）
```

## 📊 输出示例
```
============================================================
性能测试结果
============================================================
总请求数: 1200
成功: 1195
失败: 5
成功率: 99.58%

测试时长: 120.45秒
吞吐量: 9.92 请求/秒
Token 吞吐量: 2,534.12 tokens/秒

Tokens 统计:
  总计: 305,234
  输入: 125,430
  输出: 179,804

延迟（秒）:
  平均: 0.856
  中位数 (P50): 0.723
  P95: 1.845
  P99: 2.341
  最小: 0.234
  最大: 3.567
============================================================

💡 使用 --save-results 保存详细结果到 JSON 文件
```

使用 `--save-results` 后：
```
✅ 详细结果已保存到: benchmark_results_1234567890.json
```

## 🔍 服务端监控

查询实时性能指标：
```bash
# 获取当前指标
curl http://localhost:8000/metrics

# 响应示例：
{
  "total_requests": 1250,
  "total_tokens_generated": 320450,
  "avg_tokens_per_second": 2847.3,
  "avg_latency_seconds": 0.734,
  "requests_per_second": 10.2
}

# 重置指标
curl -X POST http://localhost:8000/metrics/reset
```

## 🧪 完整测试流程

### 1. 基础验证（低 RPS）
```bash
python client.py \
    --dataset sharegpt \
    --dataset-path data/sharegpt_data.json \
    --duration 10 \
    --rps-schedule "0:1"
```

### 2. 寻找最佳 RPS
```bash
python client.py \
    --dataset sharegpt \
    --dataset-path data/sharegpt_data.json \
    --duration 60 \
    --rps-schedule "0:5,20:10,40:15,60:20"
```

### 3. 稳定性测试
```bash
python client.py \
    --dataset sharegpt \
    --dataset-path data/sharegpt_data.json \
    --duration 300 \
    --rps-schedule "0:10" \
    --save-results
```

### 4. 极限压测
```bash
python client.py \
    --dataset sharegpt \
    --dataset-path data/sharegpt_data.json \
    --duration 60 \
    --rps-schedule "0:50" \
    --save-results
```

## 📁 项目结构
```
vllm-benchmark/
├── server.py           # vLLM 服务端（带性能监控）
├── client.py           # 测试客户端（支持动态 RPS）
├── data_loader.py      # 数据集加载器（Alpaca/ShareGPT/LongBench）
├── metrics.py          # 性能指标收集
├── README.md           # 本文件
└── data/              # 数据集目录（不包含在仓库中）
    ├── alpaca_data.json
    ├── sharegpt_data.json
    └── longbench/
        ├── narrativeqa.jsonl
        ├── qasper.jsonl
        └── ...
```

## 🎓 最佳实践

### 短文本测试（< 2K tokens）
- 从 RPS 10-20 开始
- 使用固定或逐步增加的调度
- 重点关注 P99 延迟

### 长文本测试（> 4K tokens）
- 使用较低的 RPS（2-5）
- 启用 `--max-input-length` 截断
- 使用 tokenizer 进行精确截断
- 预期更长的超时时间

### 生产环境模拟
- 使用动态 RPS 调度
- 测试时长至少 5-10 分钟
- 使用 `--save-results` 保存结果
- 监控失败率

### 压力测试
- 从低 RPS 逐步加压
- 观察延迟突增点
- 找到系统性能边界
- 测试恢复能力

## 🐛 常见问题

### 服务端无法启动
```bash
# 检查端口是否被占用
netstat -tlnp | grep 8000

# 使用其他端口
python server.py --model your-model --port 8001
```

### 连接被拒绝
```bash
# 验证服务端是否运行
curl http://localhost:8000/health

# 应该返回：{"status":"healthy"}
```

### 延迟过高
- 降低 RPS
- 减少 `--max-tokens`
- 检查 GPU 利用率
- 增加服务端的 `--max-num-seqs`

### 显存不足
- 减少 `--max-model-len`
- 降低 `--gpu-memory-utilization`
- 减少 `--max-num-seqs`
- 使用更小的批处理大小

## 📄 输出文件

使用 `--save-results` 时，会创建包含以下内容的 JSON 文件：
```json
{
  "summary": {
    "total_requests": 600,
    "successful": 598,
    "failed": 2,
    "duration": 60.45,
    "throughput_rps": 9.89,
    "throughput_tokens_per_sec": 2534.12,
    "total_tokens": 153234,
    "latency_mean": 0.856,
    "latency_p50": 0.723,
    "latency_p95": 1.845,
    "latency_p99": 2.341
  },
  "detailed_results": [...]
}
```

## 🤝 贡献

欢迎提交 Pull Request 和 Issue！

## 📝 开源协议

MIT License

## 🙏 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [LongBench](https://github.com/THUDM/LongBench) - 长文本评测数据集
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) - 指令微调数据集

## 📧 联系方式

如有问题或建议，请在 GitHub 上提交 Issue。

---

**祝测试顺利！ 🚀**
