#!/bin/bash
# 一键运行性能对比测试

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              优化策略性能对比 - 一键测试脚本                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 检查依赖
echo "检查依赖..."
python3 -c "import torch; import tabulate; import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少依赖，正在安装..."
    pip install torch tabulate matplotlib numpy -q
fi

# 检查GPU
echo "检查GPU..."
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "⚠️  警告: 只检测到 $GPU_COUNT 个GPU"
    echo "   将使用 cuda:0 和 cpu 进行测试（性能会受影响）"
    read -p "   是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 显示NVLink状态
echo ""
echo "NVLink状态:"
nvidia-smi nvlink --status 2>/dev/null || echo "  (NVLink信息不可用)"
echo ""

# 选择测试模式
echo "请选择测试模式:"
echo "  1) 快速测试 (单个配置)"
echo "  2) 标准测试 (中等配置)"
echo "  3) 综合测试 (多种配置，耗时较长)"
echo "  4) 自定义配置"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "运行快速测试 (batch=8, seq_len=512)..."
        python3 benchmark_optimizations.py --batch_size 8 --seq_len 512 --num_layers 4
        ;;
    2)
        echo ""
        echo "运行标准测试 (batch=16, seq_len=1024)..."
        python3 benchmark_optimizations.py --batch_size 16 --seq_len 1024 --num_layers 8
        ;;
    3)
        echo ""
        echo "运行综合测试 (4种配置)..."
        echo "预计耗时: 5-15分钟"
        read -p "确认继续? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 benchmark_optimizations.py --comprehensive
        else
            echo "已取消"
            exit 0
        fi
        ;;
    4)
        echo ""
        read -p "Batch Size: " batch_size
        read -p "Sequence Length: " seq_len
        read -p "Num Layers: " num_layers
        echo ""
        echo "运行自定义测试..."
        python3 benchmark_optimizations.py \
            --batch_size $batch_size \
            --seq_len $seq_len \
            --num_layers $num_layers
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

# 检查测试是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 测试失败"
    exit 1
fi

echo ""
echo "✅ 测试完成！"
echo ""

# 生成可视化
echo "是否生成可视化报告? (y/n)"
read -p "> " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "生成可视化..."
    python3 visualize_benchmark.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 可视化完成！"
        echo ""
        echo "生成的文件:"
        ls -lh latency_comparison.png speedup_heatmap.png \
           throughput_comparison.png memory_usage.png \
           BENCHMARK_REPORT.md 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
        echo ""
        echo "查看报告:"
        echo "  cat BENCHMARK_REPORT.md"
        echo ""
        echo "查看图表:"
        echo "  在文件浏览器中打开 *.png 文件"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "测试完成！"
echo ""
echo "下一步:"
echo "  1. 查看 BENCHMARK_REPORT.md 了解详细结果"
echo "  2. 查看 *.png 图表分析性能"
echo "  3. 根据结果选择最优策略"
echo ""
echo "需要帮助?"
echo "  cat 性能对比使用指南.md"
echo "════════════════════════════════════════════════════════════════════════════"

