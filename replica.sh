#!/bin/bash

# 创建结果目录
RESULT_DIR="result_coco"
mkdir -p "$RESULT_DIR"

# 定义device组合
device_configs=("1" "1 2" "1 2 3")
# device_configs=("1")
# 外层循环：不同的层数
layer_counts=(10 20 30 40 50)

for num_layers in "${layer_counts[@]}"
do
    echo "--- Testing with $num_layers replica layers ---"
    
    # 计算结束层（从第10层开始）
    start_layer=10
    end_layer=$((start_layer + num_layers))
    
    # 循环遍历device配置
    for device_config in "${device_configs[@]}"
    do
        # 获取最后一个device编号作为d
        d=$(($(echo "$device_config" | awk '{print $NF}') + 1))
        
        echo "=== Testing with dp_devices=$device_config (d=$d) ==="
        
        # 内层循环：不同的样本数
        for samples in $(seq 5 5 30)
        do
            echo "Running with max_samples=$samples, layers=$num_layers ($start_layer-$end_layer), devices=$device_config"
            
            python example_replica.py \
                --max_samples $samples \
                --dp_devices $device_config \
                --dp_start_layer $start_layer \
                --dp_end_layer $end_layer \
                &> "$RESULT_DIR/out_rep_rps_${samples}_l_${num_layers}_d_${d}.txt"
            
            echo "✓ Completed max_samples=$samples, layers=$num_layers, d=$d"
            echo "Waiting 10 seconds before next experiment..."
            sleep 10
        done
        
        echo "Finished testing with device config: $device_config"
        echo ""
    done
    
    echo "Finished testing $num_layers layers"
    echo ""
done

echo "All experiments completed!"
echo "Results saved in: $RESULT_DIR"
