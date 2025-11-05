# 性能对比工具套件 📊

全面的优化策略性能测试和对比工具，帮助你找到最适合的加速方案。

## 🎯 工具概览

| 工具 | 功能 | 输出 |
|------|------|------|
| `benchmark_optimizations.py` | 核心测试脚本 | JSON结果 + 终端报告 |
| `visualize_benchmark.py` | 结果可视化 | PNG图表 + Markdown报告 |
| `run_benchmark.sh` | 一键测试脚本 | 交互式执行 |
| `性能对比使用指南.md` | 详细使用说明 | 文档 |

## 🚀 快速开始（3步）

### 1. 一键测试

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

选择测试模式，自动完成测试和可视化。

### 2. 查看结果

```bash
# 查看报告
cat BENCHMARK_REPORT.md

# 或在浏览器中查看图表
```

### 3. 应用最优策略

根据报告中的加速比选择策略，修改代码应用。

## 📊 测试的优化策略

### 1. Baseline（基准）
- **描述**: 单设备执行，无优化
- **用途**: 对比基准
- **内存**: 最小
- **延迟**: 最慢

### 2. Layer Replication（层复制）
- **描述**: 数据并行，两个设备各执行一半数据
- **优势**: 接近2倍加速（大batch）
- **内存**: 约2倍
- **适用**: Prefill阶段，大batch

### 3. Attention Offload（注意力卸载）
- **描述**: 只有Attention部分卸载到第二设备
- **优势**: 内存使用较少
- **内存**: 约1.5倍
- **适用**: 中等batch，注意力占比大

### 4. Continuous Replication（连续层复制）
- **描述**: 流水线并行，层分布在不同设备
- **优势**: 深层网络效果好
- **内存**: 约2倍
- **适用**: 深层网络，固定batch

## 📈 输出示例

### 终端输出

```
╔════════════════════════════════════════════════════════════════════════════╗
║                        优化策略性能综合对比测试                              ║
╚════════════════════════════════════════════════════════════════════════════╝

================================================================================
性能测试摘要
================================================================================

+------------------------+--------------+---------------+---------+-----------+
| 策略                   | 平均延迟(ms) | 吞吐量(tok/s) | 加速比  | 内存(GB)  |
+========================+==============+===============+=========+===========+
| Baseline               | 245.76       | 66,700        | 1.00x   | 4.52      |
| Layer Replication      | 130.45       | 125,600       | 1.88x✅ | 8.94      |
| Attention Offload      | 155.23       | 105,600       | 1.58x✅ | 6.73      |
| Continuous Replication | 140.89       | 116,300       | 1.74x✅ | 8.21      |
+------------------------+--------------+---------------+---------+-----------+

最佳性能:
  最低延迟: Layer Replication (130.45 ms)
  最高吞吐: Layer Replication (125,600 tokens/s)
```

### 可视化图表

测试完成后自动生成：

1. **延迟对比柱状图** (`latency_comparison.png`)
   - 横向对比各策略在不同配置下的延迟

2. **加速比热力图** (`speedup_heatmap.png`)
   - 直观显示最优配置

3. **吞吐量折线图** (`throughput_comparison.png`)
   - 展示随batch size变化的吞吐量趋势

4. **内存使用对比** (`memory_usage.png`)
   - 各策略的内存开销

### Markdown报告

完整的 `BENCHMARK_REPORT.md` 包含：
- 所有配置的详细结果
- 可视化图表
- 最佳策略推荐
- 适用场景分析

## 🔧 使用方法

### 方法1: 交互式（推荐新手）

```bash
./run_benchmark.sh
```

### 方法2: 命令行（推荐高级用户）

```bash
# 单个配置测试
python benchmark_optimizations.py \
    --batch_size 16 \
    --seq_len 1024 \
    --num_layers 8

# 综合测试（多种配置）
python benchmark_optimizations.py --comprehensive

# 生成可视化
python visualize_benchmark.py
```

### 方法3: Python API（推荐集成）

```python
from benchmark_optimizations import BenchmarkConfig, OptimizationBenchmark

# 创建配置
config = BenchmarkConfig(
    batch_size=32,
    seq_len=2048,
    num_layers=16
)

# 运行测试
benchmark = OptimizationBenchmark(config)
benchmark.run_all_benchmarks()
benchmark.print_summary()
benchmark.save_results("my_results.json")
```

## 📊 结果解读

### 加速比评级

| 加速比 | 标记 | 评价 | 建议 |
|--------|------|------|------|
| ≥ 1.8x | ✅✅✅ | 优秀 | 强烈推荐，接近理论上限 |
| 1.5-1.8x | ✅✅ | 很好 | 推荐使用，显著提升 |
| 1.2-1.5x | ✅ | 有效 | 可以使用，有改进 |
| 1.0-1.2x | ⚠️ | 边界 | 慎用，效果不明显 |
| < 1.0x | ❌ | 变慢 | 不推荐，检查配置 |

### 选择指南

**场景1: 高吞吐量推理（批量处理）**
- ✅ 推荐: Layer Replication
- 配置: batch_size ≥ 16, seq_len ≥ 1024
- 预期: 1.7-2.0x 加速

**场景2: 中等batch推理**
- ✅ 推荐: Attention Offload
- 配置: batch_size 8-16
- 预期: 1.4-1.7x 加速

**场景3: 深层网络**
- ✅ 推荐: Continuous Replication
- 配置: num_layers ≥ 16
- 预期: 1.5-1.8x 加速

**场景4: 低延迟推理（在线服务）**
- ⚠️ 推荐: Baseline（不优化）
- 原因: 小batch下优化开销大于收益

## 🔍 性能调优技巧

### 1. 如果加速比 < 1.2x

```bash
# 增大workload
python benchmark_optimizations.py --batch_size 32 --seq_len 2048

# 检查NVLink
nvidia-smi nvlink --status

# 启用调试日志
export HB_NVLINK_LOG=1
```

### 2. 如果内存不足

```bash
# 减小batch或sequence length
python benchmark_optimizations.py --batch_size 8 --seq_len 512

# 减少层数测试
python benchmark_optimizations.py --num_layers 4
```

### 3. 对比NVLink vs PCIe

```bash
# NVLink (cuda:0 <-> cuda:1)
python benchmark_optimizations.py --device_a cuda:0 --device_b cuda:1

# PCIe (不同NUMA节点)
python benchmark_optimizations.py --device_a cuda:0 --device_b cuda:2
```

## 📁 生成的文件

测试完成后会生成：

```
benchmark_results_b4_s128.json         # 小batch测试结果
benchmark_results_b8_s512.json         # 中batch测试结果
benchmark_results_b16_s1024.json       # 大batch测试结果
benchmark_results_b32_s2048.json       # 超大batch测试结果
latency_comparison.png                 # 延迟对比图
speedup_heatmap.png                    # 加速比热力图
throughput_comparison.png              # 吞吐量图
memory_usage.png                       # 内存使用图
BENCHMARK_REPORT.md                    # 完整报告
```

## 🐛 常见问题

### Q1: 只有1个GPU怎么办？

```bash
# 使用CPU作为第二设备（仅用于功能测试）
python benchmark_optimizations.py --device_b cpu
```
⚠️ 注意：CPU会非常慢，结果仅供参考

### Q2: 测试时间太长？

```bash
# 使用快速模式（少量迭代）
# 修改代码中的 num_iterations 参数
```

### Q3: 结果不稳定？

- 增加预热次数
- 增加测试迭代次数
- 确保GPU空闲（关闭其他程序）

### Q4: 如何保存历史结果？

```bash
# 重命名结果文件
mv benchmark_results.json results_$(date +%Y%m%d_%H%M%S).json
mv BENCHMARK_REPORT.md report_$(date +%Y%m%d_%H%M%S).md
```

## 📚 相关文档

| 文档 | 内容 |
|------|------|
| `性能对比使用指南.md` | 详细使用教程 |
| `NVLINK_OPTIMIZATION.md` | NVLink优化指南 |
| `快速开始_NVLINK优化.md` | NVLink快速上手 |
| `OPTIMIZATION_FIX.md` | 优化问题修复 |
| `docs/optimization_analysis.md` | 性能问题深入分析 |

## 🎓 最佳实践

1. ✅ **先运行综合测试**，了解全貌
2. ✅ **关注加速比热力图**，找到sweet spot
3. ✅ **结合内存使用**，选择合适方案
4. ✅ **在真实模型上验证**，benchmark仅供参考
5. ✅ **定期重新测试**，硬件/软件更新后重测

## 🤝 贡献

如果你有新的优化策略想要加入对比：

1. 在 `benchmark_optimizations.py` 中添加新的测试方法
2. 按照现有格式返回 `BenchmarkResult`
3. 更新可视化脚本以包含新策略

## 📮 反馈

遇到问题或有建议？
- 查看 `性能对比使用指南.md` 的故障排查部分
- 检查生成的日志文件
- 验证GPU状态：`nvidia-smi`

## 🎯 总结

这套工具帮助你：

✅ 快速对比多种优化策略  
✅ 找到最适合你场景的方案  
✅ 生成专业的性能报告  
✅ 可视化分析瓶颈  
✅ 指导实际部署决策  

**开始测试，找到最优方案！** 🚀

---

**快速命令备忘**:
```bash
# 快速测试
./run_benchmark.sh

# 查看结果
cat BENCHMARK_REPORT.md

# 查看帮助
cat 性能对比使用指南.md
```

