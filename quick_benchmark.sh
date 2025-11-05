#!/bin/bash
# 快速性能测试脚本 - 真实模型版本

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                  真实模型优化策略性能测试                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 默认配置
MODEL_PATH="../Qwen3-0.6B"
BATCH_SIZE=4
MAX_TOKENS=256
NUM_ITERS=5
LAYERS=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --iters)
            NUM_ITERS="$2"
            shift 2
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model PATH    模型路径 (默认: ../Qwen3-0.6B)"
            echo "  --batch SIZE    Batch size (默认: 4)"
            echo "  --tokens NUM    最大生成tokens (默认: 256)"
            echo "  --iters NUM     测试迭代次数 (默认: 5)"
            echo "  --layers IDS    测试层ID，逗号分隔 (默认: 自动选择)"
            echo "  --help          显示帮助"
            echo ""
            echo "示例:"
            echo "  $0 --batch 8 --tokens 512"
            echo "  $0 --layers 5,6,7 --iters 10"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查GPU
echo "检查GPU..."
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "❌ 错误: 需要至少2个GPU"
    echo "   当前检测到: $GPU_COUNT 个GPU"
    exit 1
fi
echo "✅ 检测到 $GPU_COUNT 个GPU"
echo ""

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    echo "   请使用 --model 指定正确的路径"
    exit 1
fi
echo "✅ 模型路径: $MODEL_PATH"
echo ""

# 显示NVLink状态
echo "NVLink状态:"
nvidia-smi nvlink --status 2>/dev/null | head -5 || echo "  (无法获取NVLink信息)"
echo ""

# 显示测试配置
echo "测试配置:"
echo "  模型: $MODEL_PATH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Tokens: $MAX_TOKENS"
echo "  迭代次数: $NUM_ITERS"
if [ -n "$LAYERS" ]; then
    echo "  测试层: $LAYERS"
else
    echo "  测试层: 自动选择"
fi
echo ""

# 预估时间
ESTIMATED_TIME=$((NUM_ITERS * 4))  # 每个迭代约30秒，4个策略
echo "预计耗时: ~${ESTIMATED_TIME}秒"
echo ""

read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 构建命令
CMD="python benchmark_real_model.py"
CMD="$CMD --model_path \"$MODEL_PATH\""
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_tokens $MAX_TOKENS"
CMD="$CMD --num_iterations $NUM_ITERS"

if [ -n "$LAYERS" ]; then
    CMD="$CMD --layers $LAYERS"
fi

# 运行测试
echo ""
echo "运行测试..."
echo "命令: $CMD"
echo ""

eval $CMD

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "✅ 测试完成！"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "结果已保存到: real_model_benchmark_results.json"
    echo ""
    echo "查看结果:"
    echo "  cat real_model_benchmark_results.json | python -m json.tool"
    echo ""
    echo "下一步:"
    echo "  1. 分析加速比，选择最佳策略"
    echo "  2. 在实际应用中验证性能"
    echo "  3. 调整配置优化效果"
else
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "❌ 测试失败"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "可能的原因:"
    echo "  1. GPU内存不足 → 尝试: --batch 2"
    echo "  2. 模型路径错误 → 检查: $MODEL_PATH"
    echo "  3. 依赖缺失 → 安装: pip install transformers"
    echo ""
    echo "查看详细日志:"
    echo "  export HB_DEBUG=1"
    echo "  $0 $@"
fi

