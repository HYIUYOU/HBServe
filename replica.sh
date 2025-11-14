#!/bin/bash

# 创建结果目录
RESULT_DIR="result"
mkdir -p "$RESULT_DIR"


# 外层循环：不同的层数
layer_counts=(50)

for num_layers in "${layer_counts[@]}"
do
    echo "--- Testing with $num_layers replica layers ---"
    
    # 计算结束层（从第10层开始）
    start_layer=10
    end_layer=$((start_layer + num_layers))
    
    # 内层循环：不同的样本数
    for samples in $(seq 5 1 24)
    do
        echo "Running with max_samples=$samples, layers=$num_layers ($start_layer-$end_layer)"
        python example_replica.py --max_samples $samples \
            --dp_start_layer $start_layer --dp_end_layer $end_layer \
            &> "$RESULT_DIR/out_rep_rps_${samples}_l_${num_layers}.txt"
        echo "✓ Completed max_samples=$samples, layers=$num_layers"

        echo "Waiting 10 seconds before next experiment..."

        sleep 10
    done
    
    echo "Finished testing $num_layers layers"
    echo ""
done

echo "All experiments completed!"
echo "Results saved in: $RESULT_DIR"
