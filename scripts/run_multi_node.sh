#!/bin/bash
# 多节点启动脚本

set -e

# 配置
CONFIG_FILE="../configs/pd_disagg_config.yaml"
LOG_LEVEL="INFO"

# 节点配置
PREFILL_NODES=("node-1:50051" "node-1:50053")
DECODE_NODES=("node-2:50052" "node-2:50054")

echo "启动pd分离架构多节点部署"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 函数：启动远程实例
start_remote_instance() {
    local node_host=$1
    local instance_id=$2
    local script_path=$3
    
    echo "在 $node_host 启动实例 $instance_id"
    
    ssh "$node_host" "cd $script_path && python3 start_pd_disagg.py --config $CONFIG_FILE --instance-id $instance_id --log-level $LOG_LEVEL" &
    
    # 记录进程ID
    echo $! >> /tmp/pd_disagg_pids.txt
}

# 函数：启动本地完整系统
start_local_system() {
    echo "启动本地完整系统"
    python3 start_pd_disagg.py --config "$CONFIG_FILE" --log-level "$LOG_LEVEL" &
    echo $! >> /tmp/pd_disagg_pids.txt
}

# 清理函数
cleanup() {
    echo "正在停止所有实例..."
    
    if [ -f /tmp/pd_disagg_pids.txt ]; then
        while read pid; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "停止进程 $pid"
                kill "$pid"
            fi
        done < /tmp/pd_disagg_pids.txt
        rm -f /tmp/pd_disagg_pids.txt
    fi
    
    echo "所有实例已停止"
}

# 注册清理函数
trap cleanup EXIT INT TERM

# 清理旧的PID文件
rm -f /tmp/pd_disagg_pids.txt

# 部署模式选择
case "${1:-local}" in
    "multi-node")
        echo "多节点模式"
        
        # 启动prefill实例
        for node in "${PREFILL_NODES[@]}"; do
            IFS=':' read -r host port <<< "$node"
            instance_id="prefill-$(echo $port | tail -c 2)"
            start_remote_instance "$host" "$instance_id" "/path/to/scripts"
        done
        
        # 启动decode实例
        for node in "${DECODE_NODES[@]}"; do
            IFS=':' read -r host port <<< "$node"
            instance_id="decode-$(echo $port | tail -c 2)"
            start_remote_instance "$host" "$instance_id" "/path/to/scripts"
        done
        
        # 启动CPU调度器（在当前节点）
        echo "启动CPU调度器"
        python3 start_cpu_scheduler.py --config "$CONFIG_FILE" --log-level "$LOG_LEVEL" &
        echo $! >> /tmp/pd_disagg_pids.txt
        ;;
        
    "single-node")
        echo "单节点多GPU模式"
        
        # 启动prefill实例
        python3 start_pd_disagg.py --config "$CONFIG_FILE" --instance-id "prefill-0" --log-level "$LOG_LEVEL" &
        echo $! >> /tmp/pd_disagg_pids.txt
        
        # 等待prefill实例启动
        sleep 5
        
        # 启动decode实例
        python3 start_pd_disagg.py --config "$CONFIG_FILE" --instance-id "decode-0" --log-level "$LOG_LEVEL" &
        echo $! >> /tmp/pd_disagg_pids.txt
        
        # 等待decode实例启动
        sleep 5
        
        # 启动CPU调度器
        echo "启动CPU调度器"
        python3 start_cpu_scheduler.py --config "$CONFIG_FILE" --log-level "$LOG_LEVEL" &
        echo $! >> /tmp/pd_disagg_pids.txt
        ;;
        
    "local"|*)
        echo "本地完整系统模式"
        start_local_system
        ;;
esac

echo "所有实例启动完成"
echo "按 Ctrl+C 停止所有实例"

# 等待所有后台进程
wait
