# 使用方法

### 启动 Server

```bash
# 基本启动
python server.py --model meta-llama/Llama-2-7b-hf

# 多GPU启动
python server.py \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256
```

### 运行 Client 测试

```bash
# Alpaca 数据集，固定 10 RPS
python client.py \
    --server-url http://localhost:8000 \
    --dataset alpaca \
    --duration 60 \
    --rps-schedule "0:10"

# ShareGPT 数据集，动态 RPS（0秒:10 RPS, 30秒:20 RPS, 60秒:30 RPS）
python client.py \
    --server-url http://localhost:8000 \
    --dataset sharegpt \
    --duration 120 \
    --rps-schedule "0:10,30:20,60:30,90:15"

# 使用本地数据集
python client.py \
    --server-url http://localhost:8000 \
    --dataset alpaca \
    --dataset-path ./alpaca_data.json \
    --duration 60 \
    --rps-schedule "0:5,20:10,40:20"
```

### 查看实时指标

```bash
# 获取server端指标
curl http://localhost:8000/metrics

# 重置指标
curl -X POST http://localhost:8000/metrics/reset
```

### 输出示例

```
============================================================
BENCHMARK RESULTS
============================================================
Total Requests: 1200
Successful: 1195
Failed: 5
Success Rate: 99.58%

Duration: 120.45s
Throughput: 9.92 requests/s
Token Throughput: 2534.12 tokens/s

Tokens:
  Total: 305234
  Prompt: 125430
  Completion: 179804

Latency (seconds):
  Mean: 0.856
  Median (P50): 0.723
  P95: 1.845
  P99: 2.341
  Min: 0.234
  Max: 3.567
============================================================

Detailed results saved to: benchmark_results_1698765432.json
```

## RPS 详解

**RPS = Requests Per Second（每秒请求数）**

这是衡量系统负载和性能的关键指标。

## 基本概念

```
RPS = 5  →  每秒发送 5 个请求
RPS = 10 →  每秒发送 10 个请求
RPS = 20 →  每秒发送 20 个请求
```

**举例：**
- RPS = 10 意味着：每隔 0.1 秒（100毫秒）发送一个请求
- RPS = 5 意味着：每隔 0.2 秒（200毫秒）发送一个请求
- RPS = 1 意味着：每隔 1 秒发送一个请求

## RPS Schedule（动态 RPS 调度）

在你的测试中，`--rps-schedule` 允许**动态改变 RPS**，模拟真实场景的流量变化。

### 格式说明

```bash
--rps-schedule "时间:RPS,时间:RPS,时间:RPS"
```

### 实例解析

#### 示例 1: 固定 RPS
```bash
--rps-schedule "0:10"
```
- **含义**：从第 0 秒开始，保持 10 RPS 直到测试结束
- **效果**：恒定负载，每秒 10 个请求

#### 示例 2: 两阶段测试
```bash
--rps-schedule "0:5,30:10"
```
- **0-30秒**：5 RPS（低负载预热）
- **30秒-结束**：10 RPS（正常负载）

#### 示例 3: 压力测试（逐步增加）
```bash
--rps-schedule "0:5,20:10,40:20,60:30"
```
时间线：
```
0s  ───→ 20s:  5 RPS   (低负载)
20s ───→ 40s: 10 RPS   (中等负载)
40s ───→ 60s: 20 RPS   (高负载)
60s ───→ 结束: 30 RPS   (峰值负载)
```

#### 示例 4: 模拟真实流量波动
```bash
--rps-schedule "0:10,30:20,60:15,90:25,120:10"
```
```
时间段      RPS    说明
0-30s      10     正常流量
30-60s     20     高峰期（流量翻倍）
60-90s     15     高峰后回落
90-120s    25     再次高峰
120s-结束  10     恢复正常
```

## 如何选择 RPS？

### 1. 调试阶段（低 RPS）
```bash
--rps-schedule "0:1"   # 每秒 1 个请求，方便观察日志
--rps-schedule "0:2"   # 每秒 2 个请求，快速验证
```

### 2. 预热阶段（逐步增加）
```bash
--rps-schedule "0:1,10:5,20:10"
# 先慢后快，让模型预热
```

### 3. 性能测试（固定中等负载）
```bash
--rps-schedule "0:10"   # 持续 10 RPS
--rps-schedule "0:20"   # 持续 20 RPS
```

### 4. 压力测试（高负载）
```bash
--rps-schedule "0:50"   # 高压力
--rps-schedule "0:100"  # 极限测试
```

### 5. 长文本测试（低 RPS）
```bash
# LongBench 长文本处理慢，降低 RPS
--rps-schedule "0:2"
--rps-schedule "0:5"
```

## RPS 与系统性能的关系

```
低 RPS (1-5)     → 系统轻松，延迟低，适合调试
中 RPS (10-30)   → 正常负载，测试稳定性
高 RPS (50-100)  → 压力测试，找性能瓶颈
超高 RPS (100+)  → 极限测试，可能导致请求失败
```

## 实际测试建议

### 短文本（Alpaca/ShareGPT）
```bash
# 调试
python client.py --dataset sharegpt --dataset-path data.json \
    --duration 10 --rps-schedule "0:2"

# 正常测试
python client.py --dataset sharegpt --dataset-path data.json \
    --duration 60 --rps-schedule "0:10"

# 压力测试（逐步增加）
python client.py --dataset sharegpt --dataset-path data.json \
    --duration 120 --rps-schedule "0:10,30:20,60:30,90:40"
```

### 长文本（LongBench）
```bash
# 调试（长文本处理慢）
python client.py --dataset longbench --dataset-path data.jsonl \
    --max-input-length 2048 --duration 30 --rps-schedule "0:1"

# 正常测试
python client.py --dataset longbench --dataset-path data.jsonl \
    --max-input-length 4096 --duration 60 --rps-schedule "0:5"

# 压力测试（谨慎提高）
python client.py --dataset longbench --dataset-path data.jsonl \
    --max-input-length 8192 --duration 60 --rps-schedule "0:2,30:5"
```

## RPS 与其他指标的关系

```python
# 如果系统能承受：
RPS ↑  →  吞吐量 (tokens/s) ↑
RPS ↑  →  并发请求数 ↑
RPS ↑  →  GPU 利用率 ↑

# 如果超过系统能力：
RPS ↑↑ →  延迟 (latency) ↑↑↑
RPS ↑↑ →  队列堆积 ↑
RPS ↑↑ →  失败率 ↑
RPS ↑↑ →  吞吐量不再增加（饱和）
```

## 典型的测试流程

```bash
# 1. 先用低 RPS 确认系统正常
python client.py --dataset sharegpt --dataset-path data.json \
    --duration 10 --rps-schedule "0:1"

# 2. 找到系统的最佳 RPS（延迟低，吞吐高）
python client.py --dataset sharegpt --dataset-path data.json \
    --duration 60 --rps-schedule "0:5,20:10,40:15,60:20"

# 3. 测试峰值能力
python client.py --dataset sharegpt --dataset-path data.json \
    --duration 60 --rps-schedule "0:30"

# 4. 长时间稳定性测试
python client.py --dataset sharegpt --dataset-path data.json \
    --duration 300 --rps-schedule "0:10" --save-results
```

## 查看实时 RPS

测试过程中会打印切换信息：

```
Loading sharegpt dataset...
Loaded 59005 valid samples from sharegpt
RPS Schedule: [(0, 5), (30, 10), (60, 20)]
[30.1s] Switching to 10 RPS
[60.2s] Switching to 20 RPS
Waiting for pending requests...
Benchmark completed. Sent 900 requests.
```

## 总结

**RPS 参数的作用：**
1. 📊 控制测试负载强度
2. 🔄 模拟真实流量变化
3. 🎯 找到系统性能边界
4. 🐛 调试时用低 RPS，正式测试用高 RPS

**推荐起始值：**
- 调试：`--rps-schedule "0:1"`
- 短文本测试：`--rps-schedule "0:10"`
- 长文本测试：`--rps-schedule "0:5"`
- 压力测试：`--rps-schedule "0:10,30:20,60:30"`

