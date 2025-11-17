#!/bin/bash

# run_experiments.sh

# 方法1: 使用 seq 命令生成序列（推荐）
for samples in $(seq 3 1 30)
do
    echo "Running with max_samples=$samples"
    python example.py --max_samples $samples &> res_long/out_baseline_rps_${samples}.txt
    echo "Completed max_samples=$samples"
done

echo "All experiments completed!"
