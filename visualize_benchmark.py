#!/usr/bin/env python3
"""
性能测试结果可视化工具

读取 benchmark_results.json 并生成可视化图表
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(result_files: List[str]) -> List[Dict]:
    """加载测试结果"""
    results = []
    for file in result_files:
        if os.path.exists(file):
            with open(file, 'r') as f:
                results.append(json.load(f))
    return results


def plot_latency_comparison(results: List[Dict], output_file: str = "latency_comparison.png"):
    """绘制延迟对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategies = []
    latencies = []
    colors = []
    
    for result_set in results:
        config = result_set['config']
        label = f"{config['batch_size']}×{config['seq_len']}"
        
        for result in result_set['results']:
            if result['success']:
                strategy = result['strategy']
                latency = result['avg_latency_ms']
                
                if strategy not in strategies:
                    strategies.append(strategy)
                
                latencies.append({
                    'strategy': strategy,
                    'config': label,
                    'latency': latency
                })
    
    # 按配置分组
    configs = list(set(item['config'] for item in latencies))
    configs.sort(key=lambda x: int(x.split('×')[0]))  # 按batch size排序
    
    x = np.arange(len(strategies))
    width = 0.15
    
    for i, config in enumerate(configs):
        config_data = [item['latency'] for item in latencies 
                      if item['config'] == config and item['strategy'] in strategies]
        
        if len(config_data) == len(strategies):
            offset = width * (i - len(configs) / 2)
            ax.bar(x + offset, config_data, width, label=config)
    
    ax.set_xlabel('优化策略', fontsize=12)
    ax.set_ylabel('延迟 (ms)', fontsize=12)
    ax.set_title('不同优化策略的延迟对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 延迟对比图已保存: {output_file}")
    plt.close()


def plot_speedup_heatmap(results: List[Dict], output_file: str = "speedup_heatmap.png"):
    """绘制加速比热力图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取数据
    configs = []
    strategies = []
    speedup_matrix = []
    
    for result_set in results:
        config = result_set['config']
        config_label = f"{config['batch_size']}×{config['seq_len']}"
        configs.append(config_label)
        
        speedups = []
        for result in result_set['results']:
            if result['success']:
                strategy = result['strategy']
                if strategy not in strategies and strategy != 'Baseline':
                    strategies.append(strategy)
                
                if strategy != 'Baseline':
                    speedups.append(result['speedup_vs_baseline'])
        
        if speedups:
            speedup_matrix.append(speedups)
    
    if not speedup_matrix:
        print("⚠️  没有足够的数据绘制热力图")
        return
    
    speedup_matrix = np.array(speedup_matrix)
    
    im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.0)
    
    # 设置刻度
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_yticklabels(configs)
    
    # 添加数值标注
    for i in range(len(configs)):
        for j in range(len(strategies)):
            text = ax.text(j, i, f'{speedup_matrix[i, j]:.2f}x',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('加速比热力图 (相对于Baseline)', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='加速比')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 加速比热力图已保存: {output_file}")
    plt.close()


def plot_throughput_comparison(results: List[Dict], output_file: str = "throughput_comparison.png"):
    """绘制吞吐量对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategies = []
    data_by_strategy = {}
    
    for result_set in results:
        config = result_set['config']
        config_label = f"{config['batch_size']}×{config['seq_len']}"
        
        for result in result_set['results']:
            if result['success']:
                strategy = result['strategy']
                throughput = result['throughput_tokens_per_sec']
                
                if strategy not in data_by_strategy:
                    data_by_strategy[strategy] = {'configs': [], 'throughputs': []}
                
                data_by_strategy[strategy]['configs'].append(config_label)
                data_by_strategy[strategy]['throughputs'].append(throughput)
    
    # 为每个策略绘制折线
    for strategy, data in data_by_strategy.items():
        ax.plot(data['configs'], data['throughputs'], marker='o', 
                linewidth=2, markersize=8, label=strategy)
    
    ax.set_xlabel('配置 (Batch×SeqLen)', fontsize=12)
    ax.set_ylabel('吞吐量 (tokens/s)', fontsize=12)
    ax.set_title('不同配置下的吞吐量对比', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 吞吐量对比图已保存: {output_file}")
    plt.close()


def plot_memory_usage(results: List[Dict], output_file: str = "memory_usage.png"):
    """绘制内存使用对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategies = set()
    for result_set in results:
        for result in result_set['results']:
            if result['success']:
                strategies.add(result['strategy'])
    
    strategies = sorted(list(strategies))
    x = np.arange(len(strategies))
    width = 0.15
    
    for i, result_set in enumerate(results):
        config = result_set['config']
        config_label = f"{config['batch_size']}×{config['seq_len']}"
        
        memories = []
        for strategy in strategies:
            result = next((r for r in result_set['results'] 
                          if r['strategy'] == strategy and r['success']), None)
            memories.append(result['memory_allocated_gb'] if result else 0)
        
        offset = width * (i - len(results) / 2)
        ax.bar(x + offset, memories, width, label=config_label)
    
    ax.set_xlabel('优化策略', fontsize=12)
    ax.set_ylabel('内存使用 (GB)', fontsize=12)
    ax.set_title('不同策略的内存使用对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 内存使用图已保存: {output_file}")
    plt.close()


def generate_markdown_report(results: List[Dict], output_file: str = "BENCHMARK_REPORT.md"):
    """生成Markdown格式的测试报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 优化策略性能测试报告\n\n")
        f.write(f"生成时间: {os.popen('date').read().strip()}\n\n")
        
        f.write("## 测试配置\n\n")
        for i, result_set in enumerate(results, 1):
            config = result_set['config']
            f.write(f"### 配置 {i}\n\n")
            f.write(f"- Batch Size: {config['batch_size']}\n")
            f.write(f"- Sequence Length: {config['seq_len']}\n")
            f.write(f"- Total Tokens: {config['batch_size'] * config['seq_len']}\n")
            f.write(f"- Hidden Size: {config['hidden_size']}\n")
            f.write(f"- Num Layers: {config['num_layers']}\n")
            f.write(f"- 设备: {config['device_a']} <-> {config['device_b']}\n\n")
        
        f.write("## 性能对比\n\n")
        
        for i, result_set in enumerate(results, 1):
            config = result_set['config']
            f.write(f"### 配置 {i}: {config['batch_size']}×{config['seq_len']}\n\n")
            
            f.write("| 策略 | 延迟 (ms) | 吞吐量 (tok/s) | 加速比 | 内存 (GB) |\n")
            f.write("|------|-----------|----------------|--------|----------|\n")
            
            for result in result_set['results']:
                if result['success']:
                    strategy = result['strategy']
                    latency = result['avg_latency_ms']
                    throughput = result['throughput_tokens_per_sec']
                    speedup = result['speedup_vs_baseline']
                    memory = result['memory_allocated_gb']
                    
                    # 添加表情符号
                    if speedup >= 1.5:
                        emoji = "✅✅"
                    elif speedup >= 1.2:
                        emoji = "✅"
                    elif speedup >= 1.0:
                        emoji = "⚠️"
                    else:
                        emoji = "❌"
                    
                    f.write(f"| {strategy} | {latency:.2f} | {throughput:.0f} | {speedup:.2f}x {emoji} | {memory:.2f} |\n")
            
            f.write("\n")
        
        f.write("## 可视化\n\n")
        f.write("![延迟对比](latency_comparison.png)\n\n")
        f.write("![加速比热力图](speedup_heatmap.png)\n\n")
        f.write("![吞吐量对比](throughput_comparison.png)\n\n")
        f.write("![内存使用](memory_usage.png)\n\n")
        
        f.write("## 结论\n\n")
        f.write("图例:\n")
        f.write("- ✅✅ : 加速 ≥ 1.5x (强烈推荐)\n")
        f.write("- ✅  : 加速 ≥ 1.2x (推荐)\n")
        f.write("- ⚠️  : 加速 ≥ 1.0x (边界情况)\n")
        f.write("- ❌  : 加速 < 1.0x (不推荐)\n\n")
        
        # 分析最佳策略
        f.write("### 最佳策略分析\n\n")
        for result_set in results:
            config = result_set['config']
            config_label = f"{config['batch_size']}×{config['seq_len']}"
            
            best_speedup = max((r for r in result_set['results'] if r['success']), 
                              key=lambda r: r['speedup_vs_baseline'])
            
            f.write(f"**{config_label}**: {best_speedup['strategy']} "
                   f"({best_speedup['speedup_vs_baseline']:.2f}x 加速)\n\n")
    
    print(f"✅ Markdown报告已保存: {output_file}")


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    性能测试结果可视化工具                                    ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # 查找所有测试结果文件
    result_files = list(Path('.').glob('benchmark_results*.json'))
    
    if not result_files:
        print("❌ 错误: 未找到测试结果文件")
        print("   请先运行: python benchmark_optimizations.py")
        return
    
    print(f"找到 {len(result_files)} 个测试结果文件:")
    for f in result_files:
        print(f"  - {f}")
    print()
    
    # 加载结果
    results = load_results([str(f) for f in result_files])
    
    if not results:
        print("❌ 错误: 无法加载测试结果")
        return
    
    print(f"成功加载 {len(results)} 组测试结果\n")
    
    # 生成可视化
    print("生成可视化图表...")
    plot_latency_comparison(results)
    plot_speedup_heatmap(results)
    plot_throughput_comparison(results)
    plot_memory_usage(results)
    
    # 生成报告
    print("\n生成测试报告...")
    generate_markdown_report(results)
    
    print(f"\n{'='*80}")
    print("✅ 可视化完成！")
    print(f"{'='*80}\n")
    print("生成的文件:")
    print("  - latency_comparison.png (延迟对比图)")
    print("  - speedup_heatmap.png (加速比热力图)")
    print("  - throughput_comparison.png (吞吐量对比图)")
    print("  - memory_usage.png (内存使用图)")
    print("  - BENCHMARK_REPORT.md (完整报告)")
    print()


if __name__ == "__main__":
    main()

