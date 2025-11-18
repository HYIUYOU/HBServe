#!/bin/bash

# 创建结果目录
RESULT_DIR="result_coco_nsys"
mkdir -p "$RESULT_DIR"

# 定义device组合
device_configs=("1" "1 2" "1 2 3")
# device_configs=("1")
# 外层循环：不同的层数
layer_counts=(30)

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
            
            # 定义基础文件名（不含扩展名）
            base_name="$RESULT_DIR/out_rep_rps_${samples}_l_${num_layers}_d_${d}"
            
            # 执行nsys性能分析
            # nsys输出: ${base_name}.qdrep
            # 程序日志: ${base_name}.txt
            nsys profile \
                -o "$base_name" \
                --stats=true \
                --force-overwrite true \
                python -u example_replica.py \
                    --max_samples $samples \
                    --dp_devices $device_config \
                    --dp_start_layer $start_layer \
                    --dp_end_layer $end_layer \
                    &> "${base_name}.txt"
            
            echo "✓ Completed max_samples=$samples, layers=$num_layers, d=$d"
            echo "  - nsys report: ${base_name}.qdrep"
            echo "  - program log: ${base_name}.txt"
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
echo "To view nsys reports:"
echo "  nsys-ui $RESULT_DIR/*.qdrep"