import re
import os
from pathlib import Path
import csv
import pandas as pd

def parse_filename(filename):
    """
    解析文件名，提取参数信息
    - out_baseline_rps_10.txt -> {'type': 'baseline', 'rps': 10}
    - out_rep_rps_2_l_10.txt -> {'type': 'replica', 'rps': 2, 'replica_layers': 10}
    """
    info = {}
    
    if 'baseline' in filename:
        info['type'] = 'baseline'
        info['use_replica'] = False
        match = re.search(r'rps_(\d+)', filename)
        if match:
            info['rps'] = int(match.group(1))
        info['replica_layers'] = 0
    elif 'rep' in filename:
        info['type'] = 'replica'
        info['use_replica'] = True
        rps_match = re.search(r'rps_(\d+)', filename)
        l_match = re.search(r'l_(\d+)', filename)
        if rps_match:
            info['rps'] = int(rps_match.group(1))
        if l_match:
            info['replica_layers'] = int(l_match.group(1))
    
    return info

def extract_throughput_data(file_path):
    """
    从单个文件中提取prefill和decode的吞吐量数据
    """
    prefill_values = []
    decode_values = []
    seen_pairs = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Generating:' not in line:
                    continue
                
                prefill_match = re.search(r'Prefill=(\d+)tok/s', line)
                decode_match = re.search(r'Decode=(\d+)tok/s', line)
                
                if prefill_match and decode_match:
                    prefill_val = int(prefill_match.group(1))
                    decode_val = int(decode_match.group(1))
                    
                    if decode_val > 0:
                        pair = (prefill_val, decode_val)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            prefill_values.append(prefill_val)
                            decode_values.append(decode_val)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], []
    
    return prefill_values, decode_values

def extract_summary_metrics(file_path):
    """
    提取文件末尾的汇总指标
    """
    metrics = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            samples_match = re.search(r'max_samples:\s*(\d+)', content)
            if samples_match:
                metrics['max_samples'] = int(samples_match.group(1))
            
            latency_match = re.search(r'latency:\s*([\d.]+)s', content)
            if latency_match:
                metrics['latency'] = float(latency_match.group(1))
            
            throughput_match = re.search(r'throughput:\s*([\d.]+)\s*requests/s', content)
            if throughput_match:
                metrics['throughput'] = float(throughput_match.group(1))
    
    except Exception as e:
        print(f"Error extracting metrics from {file_path}: {e}")
    
    return metrics

def calculate_stats(values):
    """
    计算统计信息
    """
    if not values:
        return None
    
    return {
        'count': len(values),
        'avg': sum(values) / len(values),
        'max': max(values),
        'min': min(values)
    }

def process_all_files(directory):
    """
    处理目录下的所有txt文件
    """
    results = []
    
    txt_files = sorted(Path(directory).glob('out_*.txt'))
    
    print(f"找到 {len(txt_files)} 个文件\n")
    
    for file_path in txt_files:
        print(f"处理: {file_path.name}", end=" ... ")
        
        file_info = parse_filename(file_path.name)
        prefill_values, decode_values = extract_throughput_data(file_path)
        summary_metrics = extract_summary_metrics(file_path)
        
        prefill_stats = calculate_stats(prefill_values)
        decode_stats = calculate_stats(decode_values)
        
        result = {
            'filename': file_path.name,
            **file_info,
            **summary_metrics
        }
        
        if prefill_stats:
            result['prefill_avg'] = prefill_stats['avg']
            result['prefill_max'] = prefill_stats['max']
            result['prefill_min'] = prefill_stats['min']
        
        if decode_stats:
            result['decode_avg'] = decode_stats['avg']
            result['decode_max'] = decode_stats['max']
            result['decode_min'] = decode_stats['min']
        
        results.append(result)
        print("✓")
    
    return results

def print_matrix_tables(results):
    """
    打印矩阵式表格：Replica Layers(行) x RPS(列)
    """
    print("\n" + "="*120)
    print("性能数据矩阵 (Replica Layers x RPS)")
    print("="*120)
    
    # 分离baseline和replica结果
    baseline_results = [r for r in results if r.get('type') == 'baseline']
    rep_results = [r for r in results if r.get('type') == 'replica']
    
    # 获取所有的RPS值和Replica Layers值
    all_rps = sorted(set([r.get('rps') for r in results if r.get('rps') is not None]))
    all_layers = sorted(set([r.get('replica_layers') for r in rep_results if r.get('replica_layers') is not None]))
    
    # 1. Decode平均值矩阵
    print("\n【Decode平均值 (tok/s)】")
    print(f"{'Layers':<10}", end="")
    for rps in all_rps:
        print(f"RPS={rps:<6}", end="")
    print()
    print("-" * (10 + len(all_rps) * 11))
    
    # Baseline行
    print(f"{'Baseline':<10}", end="")
    for rps in all_rps:
        baseline = next((r for r in baseline_results if r.get('rps') == rps), None)
        if baseline and baseline.get('decode_avg'):
            print(f"{baseline['decode_avg']:<10.2f} ", end="")
        else:
            print(f"{'N/A':<10} ", end="")
    print()
    
    # Replica行
    for layers in all_layers:
        print(f"{'L=' + str(layers):<10}", end="")
        for rps in all_rps:
            rep = next((r for r in rep_results 
                       if r.get('rps') == rps and r.get('replica_layers') == layers), None)
            if rep and rep.get('decode_avg'):
                print(f"{rep['decode_avg']:<10.2f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")
        print()
    
    # 2. Throughput矩阵
    print("\n【Throughput (req/s)】")
    print(f"{'Layers':<10}", end="")
    for rps in all_rps:
        print(f"RPS={rps:<6}", end="")
    print()
    print("-" * (10 + len(all_rps) * 11))
    
    # Baseline行
    print(f"{'Baseline':<10}", end="")
    for rps in all_rps:
        baseline = next((r for r in baseline_results if r.get('rps') == rps), None)
        if baseline and baseline.get('throughput'):
            print(f"{baseline['throughput']:<10.4f} ", end="")
        else:
            print(f"{'N/A':<10} ", end="")
    print()
    
    # Replica行
    for layers in all_layers:
        print(f"{'L=' + str(layers):<10}", end="")
        for rps in all_rps:
            rep = next((r for r in rep_results 
                       if r.get('rps') == rps and r.get('replica_layers') == layers), None)
            if rep and rep.get('throughput'):
                print(f"{rep['throughput']:<10.4f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")
        print()
    
    # 3. Latency矩阵
    print("\n【Latency (s)】")
    print(f"{'Layers':<10}", end="")
    for rps in all_rps:
        print(f"RPS={rps:<6}", end="")
    print()
    print("-" * (10 + len(all_rps) * 11))
    
    # Baseline行
    print(f"{'Baseline':<10}", end="")
    for rps in all_rps:
        baseline = next((r for r in baseline_results if r.get('rps') == rps), None)
        if baseline and baseline.get('latency'):
            print(f"{baseline['latency']:<10.2f} ", end="")
        else:
            print(f"{'N/A':<10} ", end="")
    print()
    
    # Replica行
    for layers in all_layers:
        print(f"{'L=' + str(layers):<10}", end="")
        for rps in all_rps:
            rep = next((r for r in rep_results 
                       if r.get('rps') == rps and r.get('replica_layers') == layers), None)
            if rep and rep.get('latency'):
                print(f"{rep['latency']:<10.2f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")
        print()
    
    # 4. Prefill平均值矩阵
    print("\n【Prefill平均值 (tok/s)】")
    print(f"{'Layers':<10}", end="")
    for rps in all_rps:
        print(f"RPS={rps:<6}", end="")
    print()
    print("-" * (10 + len(all_rps) * 11))
    
    # Baseline行
    print(f"{'Baseline':<10}", end="")
    for rps in all_rps:
        baseline = next((r for r in baseline_results if r.get('rps') == rps), None)
        if baseline and baseline.get('prefill_avg'):
            print(f"{baseline['prefill_avg']:<10.2f} ", end="")
        else:
            print(f"{'N/A':<10} ", end="")
    print()
    
    # Replica行
    for layers in all_layers:
        print(f"{'L=' + str(layers):<10}", end="")
        for rps in all_rps:
            rep = next((r for r in rep_results 
                       if r.get('rps') == rps and r.get('replica_layers') == layers), None)
            if rep and rep.get('prefill_avg'):
                print(f"{rep['prefill_avg']:<10.2f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")
        print()

def print_improvement_matrix(results):
    """
    打印性能提升矩阵（相对于baseline的提升百分比）
    """
    print("\n" + "="*120)
    print("性能提升矩阵 (相对于Baseline的提升百分比)")
    print("="*120)
    
    baseline_results = [r for r in results if r.get('type') == 'baseline']
    rep_results = [r for r in results if r.get('type') == 'replica']
    
    all_rps = sorted(set([r.get('rps') for r in results if r.get('rps') is not None]))
    all_layers = sorted(set([r.get('replica_layers') for r in rep_results if r.get('replica_layers') is not None]))
    
    # Decode提升百分比
    print("\n【Decode性能提升 (%)】")
    print(f"{'Layers':<10}", end="")
    for rps in all_rps:
        print(f"RPS={rps:<8}", end="")
    print()
    print("-" * (10 + len(all_rps) * 13))
    
    for layers in all_layers:
        print(f"{'L=' + str(layers):<10}", end="")
        for rps in all_rps:
            baseline = next((r for r in baseline_results if r.get('rps') == rps), None)
            rep = next((r for r in rep_results 
                       if r.get('rps') == rps and r.get('replica_layers') == layers), None)
            
            if baseline and rep and baseline.get('decode_avg') and rep.get('decode_avg'):
                improvement = ((rep['decode_avg'] - baseline['decode_avg']) / baseline['decode_avg'] * 100)
                print(f"{improvement:+10.2f}%  ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
        print()
    
    # Throughput提升百分比
    print("\n【Throughput提升 (%)】")
    print(f"{'Layers':<10}", end="")
    for rps in all_rps:
        print(f"RPS={rps:<8}", end="")
    print()
    print("-" * (10 + len(all_rps) * 13))
    
    for layers in all_layers:
        print(f"{'L=' + str(layers):<10}", end="")
        for rps in all_rps:
            baseline = next((r for r in baseline_results if r.get('rps') == rps), None)
            rep = next((r for r in rep_results 
                       if r.get('rps') == rps and r.get('replica_layers') == layers), None)
            
            if baseline and rep and baseline.get('throughput') and rep.get('throughput'):
                improvement = ((rep['throughput'] - baseline['throughput']) / baseline['throughput'] * 100)
                print(f"{improvement:+10.2f}%  ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
        print()

def save_matrix_to_csv(results, output_dir='output'):
    """
    保存矩阵数据到CSV文件
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_results = [r for r in results if r.get('type') == 'baseline']
    rep_results = [r for r in results if r.get('type') == 'replica']
    
    all_rps = sorted(set([r.get('rps') for r in results if r.get('rps') is not None]))
    all_layers = sorted(set([r.get('replica_layers') for r in rep_results if r.get('replica_layers') is not None]))
    
    # 保存各个指标的矩阵
    metrics = {
        'decode_avg': 'Decode Average (tok/s)',
        'prefill_avg': 'Prefill Average (tok/s)',
        'throughput': 'Throughput (req/s)',
        'latency': 'Latency (s)'
    }
    
    for metric_key, metric_name in metrics.items():
        # 创建矩阵数据
        matrix_data = []
        
        # Baseline行
        baseline_row = {'Layers': 'Baseline'}
        for rps in all_rps:
            baseline = next((r for r in baseline_results if r.get('rps') == rps), None)
            if baseline and baseline.get(metric_key):
                baseline_row[f'RPS_{rps}'] = round(baseline[metric_key], 4)
            else:
                baseline_row[f'RPS_{rps}'] = None
        matrix_data.append(baseline_row)
        
        # Replica行
        for layers in all_layers:
            row = {'Layers': f'L={layers}'}
            for rps in all_rps:
                rep = next((r for r in rep_results 
                           if r.get('rps') == rps and r.get('replica_layers') == layers), None)
                if rep and rep.get(metric_key):
                    row[f'RPS_{rps}'] = round(rep[metric_key], 4)
                else:
                    row[f'RPS_{rps}'] = None
            matrix_data.append(row)
        
        # 保存到CSV
        df = pd.DataFrame(matrix_data)
        output_file = os.path.join(output_dir, f'matrix_{metric_key}.csv')
        df.to_csv(output_file, index=False)
        print(f"已保存: {output_file}")
    
    # 保存提升百分比矩阵
    for metric_key, metric_name in [('decode_avg', 'Decode'), ('throughput', 'Throughput')]:
        improvement_data = []
        
        for layers in all_layers:
            row = {'Layers': f'L={layers}'}
            for rps in all_rps:
                baseline = next((r for r in baseline_results if r.get('rps') == rps), None)
                rep = next((r for r in rep_results 
                           if r.get('rps') == rps and r.get('replica_layers') == layers), None)
                
                if baseline and rep and baseline.get(metric_key) and rep.get(metric_key):
                    improvement = ((rep[metric_key] - baseline[metric_key]) / baseline[metric_key] * 100)
                    row[f'RPS_{rps}'] = round(improvement, 2)
                else:
                    row[f'RPS_{rps}'] = None
            improvement_data.append(row)
        
        df = pd.DataFrame(improvement_data)
        output_file = os.path.join(output_dir, f'improvement_{metric_key}.csv')
        df.to_csv(output_file, index=False)
        print(f"已保存: {output_file}")

def save_all_data_csv(results, output_file='all_results.csv'):
    """
    保存所有原始数据到一个CSV文件
    """
    fieldnames = ['filename', 'type', 'rps', 'replica_layers', 
                  'prefill_avg', 'prefill_max', 'prefill_min',
                  'decode_avg', 'decode_max', 'decode_min',
                  'throughput', 'latency', 'max_samples']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"已保存所有数据: {output_file}")

def generate_best_config_report(results):
    """
    生成最优配置报告
    """
    print("\n" + "="*120)
    print("最优配置推荐")
    print("="*120)
    
    baseline_results = [r for r in results if r.get('type') == 'baseline']
    rep_results = [r for r in results if r.get('type') == 'replica']
    
    all_rps = sorted(set([r.get('rps') for r in results if r.get('rps') is not None]))
    
    print(f"\n{'RPS':<6} {'最优配置':<15} {'Decode提升':<15} {'Throughput提升':<18} {'推荐理由':<30}")
    print("-" * 100)
    
    best_configs = []
    
    for rps in all_rps:
        baseline = next((r for r in baseline_results if r.get('rps') == rps), None)
        replicas = [r for r in rep_results if r.get('rps') == rps]
        
        if not baseline or not replicas:
            continue
        
        # 找到throughput最高的配置
        best_replica = max(replicas, key=lambda x: x.get('throughput', 0))
        
        baseline_decode = baseline.get('decode_avg', 0)
        baseline_throughput = baseline.get('throughput', 0)
        best_decode = best_replica.get('decode_avg', 0)
        best_throughput = best_replica.get('throughput', 0)
        
        decode_improvement = ((best_decode - baseline_decode) / baseline_decode * 100) if baseline_decode else 0
        throughput_improvement = ((best_throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput else 0
        
        # 判断是否值得使用replica
        if throughput_improvement > 5:
            reason = "显著提升，推荐使用"
        elif throughput_improvement > 0:
            reason = "有提升，可考虑使用"
        else:
            reason = "无明显提升，使用baseline"
        
        print(f"{rps:<6} {'L=' + str(best_replica.get('replica_layers')):<15} "
              f"{decode_improvement:+.2f}%{'':<9} "
              f"{throughput_improvement:+.2f}%{'':<12} "
              f"{reason:<30}")
        
        best_configs.append({
            'rps': rps,
            'best_layers': best_replica.get('replica_layers'),
            'decode_improvement': decode_improvement,
            'throughput_improvement': throughput_improvement
        })
    
    return best_configs

def generate_visualization_script():
    """
    生成可视化脚本
    """
    script_content = '''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# 读取矩阵数据
decode_df = pd.read_csv('output/matrix_decode_avg.csv')
throughput_df = pd.read_csv('output/matrix_throughput.csv')
improvement_decode_df = pd.read_csv('output/improvement_decode_avg.csv')
improvement_throughput_df = pd.read_csv('output/improvement_throughput.csv')

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Decode性能热力图
decode_matrix = decode_df.set_index('Layers')
decode_matrix.columns = [col.replace('RPS_', '') for col in decode_matrix.columns]
sns.heatmap(decode_matrix, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 0], cbar_kws={'label': 'tok/s'})
axes[0, 0].set_title('Decode Average Performance (tok/s)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('RPS', fontsize=12)
axes[0, 0].set_ylabel('Configuration', fontsize=12)

# 2. Throughput性能热力图
throughput_matrix = throughput_df.set_index('Layers')
throughput_matrix.columns = [col.replace('RPS_', '') for col in throughput_matrix.columns]
sns.heatmap(throughput_matrix, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0, 1], cbar_kws={'label': 'req/s'})
axes[0, 1].set_title('Throughput Performance (req/s)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('RPS', fontsize=12)
axes[0, 1].set_ylabel('Configuration', fontsize=12)

# 3. Decode提升百分比热力图
improvement_decode_matrix = improvement_decode_df.set_index('Layers')
improvement_decode_matrix.columns = [col.replace('RPS_', '') for col in improvement_decode_matrix.columns]
sns.heatmap(improvement_decode_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
            ax=axes[1, 0], cbar_kws={'label': '% improvement'})
axes[1, 0].set_title('Decode Improvement vs Baseline (%)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('RPS', fontsize=12)
axes[1, 0].set_ylabel('Replica Layers', fontsize=12)

# 4. Throughput提升百分比热力图
improvement_throughput_matrix = improvement_throughput_df.set_index('Layers')
improvement_throughput_matrix.columns = [col.replace('RPS_', '') for col in improvement_throughput_matrix.columns]
sns.heatmap(improvement_throughput_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=axes[1, 1], cbar_kws={'label': '% improvement'})
axes[1, 1].set_title('Throughput Improvement vs Baseline (%)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('RPS', fontsize=12)
axes[1, 1].set_ylabel('Replica Layers', fontsize=12)

plt.tight_layout()
plt.savefig('performance_heatmaps.png', dpi=300, bbox_inches='tight')
print("热力图已保存: performance_heatmaps.png")
plt.show()

# 创建线图：不同Replica Layers下的性能曲线
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

# 读取原始数据用于绘制线图
all_data = pd.read_csv('all_results.csv')

# 5. Decode性能曲线
baseline_data = all_data[all_data['type'] == 'baseline'].sort_values('rps')
axes2[0].plot(baseline_data['rps'], baseline_data['decode_avg'], 
             marker='o', linewidth=2, markersize=8, label='Baseline', color='black')

replica_data = all_data[all_data['type'] == 'replica']
for layers in sorted(replica_data['replica_layers'].unique()):
    layer_data = replica_data[replica_data['replica_layers'] == layers].sort_values('rps')
    axes2[0].plot(layer_data['rps'], layer_data['decode_avg'], 
                 marker='o', linewidth=2, markersize=6, label=f'L={layers}', alpha=0.7)

axes2[0].set_xlabel('RPS', fontsize=12)
axes2[0].set_ylabel('Decode Average (tok/s)', fontsize=12)
axes2[0].set_title('Decode Performance vs RPS', fontsize=14, fontweight='bold')
axes2[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes2[0].grid(True, alpha=0.3)

# 6. Throughput性能曲线
axes2[1].plot(baseline_data['rps'], baseline_data['throughput'], 
             marker='o', linewidth=2, markersize=8, label='Baseline', color='black')

for layers in sorted(replica_data['replica_layers'].unique()):
    layer_data = replica_data[replica_data['replica_layers'] == layers].sort_values('rps')
    axes2[1].plot(layer_data['rps'], layer_data['throughput'], 
                 marker='o', linewidth=2, markersize=6, label=f'L={layers}', alpha=0.7)

axes2[1].set_xlabel('RPS', fontsize=12)
axes2[1].set_ylabel('Throughput (req/s)', fontsize=12)
axes2[1].set_title('Throughput vs RPS', fontsize=14, fontweight='bold')
axes2[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_curves.png', dpi=300, bbox_inches='tight')
print("性能曲线图已保存: performance_curves.png")
plt.show()

print("\\n所有图表已生成完成！")
'''
    
    with open('generate_plots.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n可视化脚本已生成: generate_plots.py")

def main():
    # 设置目录路径
    directory = '/home/admin/workspace/aop_lab/app_source/HBServe/result'
    # directory = '.'  # 使用当前目录
    
    print("="*120)
    print("Replica性能分析工具")
    print("="*120)
    print("\n配置说明:")
    print("  - Baseline: 不使用replica的基准配置")
    print("  - Replica (L=X): 使用X层layer复制的配置")
    print("  - RPS: Requests Per Second (每秒请求数)")
    print()
    
    # 处理所有文件
    print("步骤 1/5: 读取并解析文件...")
    results = process_all_files(directory)
    
    # 打印矩阵表格
    print("\n步骤 2/5: 生成性能矩阵...")

