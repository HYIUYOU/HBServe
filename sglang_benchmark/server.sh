#!/bin/bash
# SGLang 服务端启动脚本
# model_path = "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

MODEL=${1:-"/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"}
PORT=${2:-8000}
TP_SIZE=${3:-1}

echo "========================================="
echo "启动 SGLang 服务端"
echo "========================================="
echo "模型: $MODEL"
echo "端口: $PORT"
echo "TP Size: $TP_SIZE"
echo ""

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    --tp-size $TP_SIZE \
    --mem-fraction-static 0.9
