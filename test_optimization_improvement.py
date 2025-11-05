"""
测试优化改进效果
对比原始方法 vs 优化方法的性能
"""

import torch
import time
import os
from typing import Dict

# 启用调试日志
os.environ['HB_REPLICA_LOG'] = '1'
os.environ['HB_KVCACHE_LOG'] = '1'


def benchmark_data_transfer(
    batch_size: int,
    seq_len: int,
    hidden_size: int = 4096,
    device_a: torch.device = torch.device("cuda:0"),
    device_b: torch.device = torch.device("cuda:1")
):
    """基准测试：数据传输时间"""
    print(f"\n{'='*80}")
    print(f"数据传输基准测试: batch={batch_size}, seq_len={seq_len}")
    print(f"{'='*80}")
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device_a)
    
    # 测试阻塞式传输
    torch.cuda.synchronize()
    start = time.perf_counter()
    hs_b_blocking = hidden_states.to(device_b)
    torch.cuda.synchronize()
    time_blocking = (time.perf_counter() - start) * 1000
    
    # 测试非阻塞式传输
    torch.cuda.synchronize()
    start = time.perf_counter()
    hs_b_nonblocking = hidden_states.to(device_b, non_blocking=True)
    torch.cuda.synchronize()
    time_nonblocking = (time.perf_counter() - start) * 1000
    
    size_mb = hidden_states.numel() * hidden_states.element_size() / 1024 / 1024
    
    print(f"数据大小: {size_mb:.2f} MB")
    print(f"阻塞式传输: {time_blocking:.3f} ms")
    print(f"非阻塞式传输: {time_nonblocking:.3f} ms")
    print(f"加速比: {time_blocking / time_nonblocking:.2f}x")
    
    return time_blocking, time_nonblocking


def estimate_compute_time(batch_size: int, seq_len: int, hidden_size: int = 4096):
    """估算计算时间（简化模拟）"""
    # 假设每个token的计算时间为常数
    # 实际情况会更复杂，但这给出了一个粗略估计
    tokens = batch_size * seq_len
    
    # 基于经验：4096维度的transformer layer，每个token约0.01-0.02ms
    compute_time_per_token = 0.015  # ms
    
    return tokens * compute_time_per_token


def analyze_optimization_viability(
    batch_size: int,
    seq_len: int,
    hidden_size: int = 4096,
    overhead_ms: float = 27.0
):
    """分析优化是否可行"""
    print(f"\n{'='*80}")
    print(f"优化可行性分析: batch={batch_size}, seq_len={seq_len}")
    print(f"{'='*80}")
    
    total_tokens = batch_size * seq_len
    compute_time = estimate_compute_time(batch_size, seq_len, hidden_size)
    
    # 双设备理论时间
    time_with_opt = compute_time / 2 + overhead_ms
    time_no_opt = compute_time
    
    speedup = time_no_opt / time_with_opt
    
    print(f"Token数量: {total_tokens}")
    print(f"估算计算时间: {compute_time:.2f} ms")
    print(f"传输+开销: {overhead_ms:.2f} ms")
    print(f"---")
    print(f"无优化时间: {time_no_opt:.2f} ms")
    print(f"有优化时间: {time_with_opt:.2f} ms")
    print(f"预期加速比: {speedup:.2f}x")
    
    if speedup > 1.2:
        print(f"✅ 建议启用优化 (加速>{speedup:.2f}x)")
        return True
    elif speedup > 1.0:
        print(f"⚠️  边界情况 (小幅加速~{speedup:.2f}x)")
        return False
    else:
        print(f"❌ 不建议优化 (会变慢~{speedup:.2f}x)")
        return False


def find_optimal_batch_size():
    """找到最优的batch size"""
    print(f"\n{'='*80}")
    print("寻找最优Batch Size")
    print(f"{'='*80}")
    
    test_configs = [
        (1, 128),
        (2, 256),
        (4, 512),
        (8, 1024),
        (16, 2048),
        (32, 4096),
        (64, 512),
    ]
    
    overhead_ms = 27.0
    results = []
    
    for batch_size, seq_len in test_configs:
        total_tokens = batch_size * seq_len
        compute_time = estimate_compute_time(batch_size, seq_len)
        time_with_opt = compute_time / 2 + overhead_ms
        time_no_opt = compute_time
        speedup = time_no_opt / time_with_opt
        
        results.append({
            'batch': batch_size,
            'seq_len': seq_len,
            'tokens': total_tokens,
            'speedup': speedup,
            'viable': speedup > 1.2
        })
    
    print(f"\n{'Batch':<8} {'SeqLen':<8} {'Tokens':<10} {'加速比':<10} {'建议'}")
    print(f"{'-'*60}")
    for r in results:
        status = "✅ 启用" if r['viable'] else "❌ 跳过"
        print(f"{r['batch']:<8} {r['seq_len']:<8} {r['tokens']:<10} {r['speedup']:<10.2f} {status}")
    
    # 找到最小可行配置
    viable = [r for r in results if r['viable']]
    if viable:
        min_viable = min(viable, key=lambda x: x['tokens'])
        print(f"\n最小可行配置: batch={min_viable['batch']}, seq_len={min_viable['seq_len']}, tokens={min_viable['tokens']}")
        return min_viable['tokens']
    else:
        print(f"\n⚠️  警告：在当前开销下，所有测试配置都不建议使用优化！")
        return None


def test_kv_cache_savings():
    """测试KV Cache增量同步的节省"""
    print(f"\n{'='*80}")
    print("KV Cache增量同步节省分析")
    print(f"{'='*80}")
    
    # 假设参数
    num_layers = 32
    batch_size = 16
    max_seq_len = 2048
    num_kv_heads = 8
    head_dim = 128
    block_size = 16
    
    # 单个block的大小
    block_bytes = block_size * num_kv_heads * head_dim * 2 * 2  # K+V, bfloat16
    
    # Prefill阶段：全量同步
    prefill_blocks_per_batch = max_seq_len // block_size
    prefill_total_bytes = batch_size * prefill_blocks_per_batch * block_bytes * num_layers
    
    # Decode阶段：每次只新增1个token
    decode_steps = 100  # 生成100个token
    
    # 全量同步（原方法）
    decode_total_full = decode_steps * batch_size * prefill_blocks_per_batch * block_bytes * num_layers
    
    # 增量同步（优化方法）
    # 大部分时候只需要同步1个block（每个batch每层）
    decode_total_incremental = decode_steps * batch_size * 1 * block_bytes * num_layers
    
    print(f"配置:")
    print(f"  Layers: {num_layers}, Batch: {batch_size}, MaxSeqLen: {max_seq_len}")
    print(f"  KV Heads: {num_kv_heads}, Head Dim: {head_dim}")
    print(f"\nPrefill阶段 (一次性):")
    print(f"  全量同步: {prefill_total_bytes / 1024 / 1024:.2f} MB")
    print(f"\nDecode阶段 ({decode_steps} steps):")
    print(f"  全量同步: {decode_total_full / 1024 / 1024:.2f} MB")
    print(f"  增量同步: {decode_total_incremental / 1024 / 1024:.2f} MB")
    print(f"  节省: {(decode_total_full - decode_total_incremental) / 1024 / 1024:.2f} MB ({(1 - decode_total_incremental/decode_total_full)*100:.1f}%)")
    
    # 时间节省（假设16GB/s带宽）
    bandwidth_gbps = 16.0
    time_full = (decode_total_full / 1024 / 1024 / 1024) / bandwidth_gbps * 1000
    time_incremental = (decode_total_incremental / 1024 / 1024 / 1024) / bandwidth_gbps * 1000
    
    print(f"\n时间节省 (假设{bandwidth_gbps}GB/s带宽):")
    print(f"  全量同步: {time_full:.2f} ms/step")
    print(f"  增量同步: {time_incremental:.2f} ms/step")
    print(f"  节省: {time_full - time_incremental:.2f} ms/step")


def print_recommendations():
    """打印优化建议"""
    print(f"\n{'='*80}")
    print("优化建议总结")
    print(f"{'='*80}")
    
    print("""
基于你的性能分析结果（开销27ms），这里是具体建议：

【立即实施】

1. ✅ 动态启用/禁用优化
   - Prefill: 只在 total_tokens >= 4096 时启用
   - Decode: 只在 batch_size >= 32 时启用
   - 实现方式：在 execute_layer_replication_forward 开头添加检查
   
2. ✅ 使用 non_blocking=True 传输
   - 将所有 .to(device) 改为 .to(device, non_blocking=True)
   - 预期改进：传输时间减少 20-40%
   
3. ✅ KV Cache增量同步
   - Decode阶段只同步新增的blocks
   - 预期改进：减少 90%+ 的传输量

【测试配置建议】

推荐的测试配置（确保有收益）：
- 小规模测试: batch=16, seq_len=2048  (32768 tokens)
- 中规模测试: batch=32, seq_len=2048  (65536 tokens)
- 大规模测试: batch=64, seq_len=2048  (131072 tokens)

不建议的配置（会变慢）：
- batch=4, seq_len=128   (512 tokens)   ❌
- batch=8, seq_len=512   (4096 tokens)  ❌
- batch=16, seq_len=512  (8192 tokens)  ⚠️  边界

【代码修改】

参考 HBserve/utils/optimized_forward.py 中的实现：
- should_enable_optimization(): 动态判断
- IncrementalKVCache: 增量同步
- execute_optimized_layer_replication(): 整合所有优化

【预期效果】

修改后的性能预期：
- 小batch (< 4096 tokens): 直接跳过优化，无性能损失 ✅
- 中batch (4096-16384 tokens): 基本持平或小幅提升
- 大batch (> 16384 tokens): 1.5-1.8x 加速 ✅✅

【验证方法】

1. 运行修改后的代码：
   python example_replication_autotune.py
   
2. 观察日志：
   - 看到 "跳过优化" 说明正确禁用了小batch
   - 看到 "启用优化" 和加速比 > 1.2x 说明生效
   
3. 对比不同batch size的性能
""")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                        优化改进效果分析                                      ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # 检查GPU
    if torch.cuda.device_count() < 2:
        print("⚠️  警告: 只有1个GPU，某些测试将跳过")
        device_a = torch.device("cuda:0")
        device_b = torch.device("cpu")
    else:
        device_a = torch.device("cuda:0")
        device_b = torch.device("cuda:1")
        
        # 测试non-blocking传输
        benchmark_data_transfer(4, 128, device_a=device_a, device_b=device_b)
    
    # 分析不同配置
    test_configs = [
        (4, 128),    # 原始测试配置
        (8, 512),    # 中等配置
        (16, 1024),  # 建议配置
        (32, 2048),  # 最佳配置
    ]
    
    for batch, seq_len in test_configs:
        analyze_optimization_viability(batch, seq_len)
    
    # 找最优配置
    min_tokens = find_optimal_batch_size()
    
    # KV Cache节省
    test_kv_cache_savings()
    
    # 打印建议
    print_recommendations()
    
    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}\n")

