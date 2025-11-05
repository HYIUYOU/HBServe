"""
性能分析工具：量化优化操作的各个环节开销
用于诊断为什么优化反而变慢
"""

import torch
import time
from typing import Dict, Optional
from contextlib import contextmanager

class OptimizationProfiler:
    """优化操作性能分析器"""
    
    def __init__(self, device_a, device_b):
        self.device_a = device_a
        self.device_b = device_b
        self.timings = {}
        self.enabled = True
        
    @contextmanager
    def profile(self, name: str):
        """性能分析上下文管理器"""
        if not self.enabled:
            yield
            return
            
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        yield
        
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
    
    def print_summary(self):
        """打印性能分析摘要"""
        print("\n" + "="*80)
        print("性能分析摘要 (单位: ms)")
        print("="*80)
        
        total_time = 0
        for name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            total = sum(times)
            total_time += total
            print(f"{name:50s}: 平均={avg:8.3f}ms, 总计={total:8.3f}ms, 次数={len(times)}")
        
        print("="*80)
        print(f"{'总时间':50s}: {total_time:8.3f}ms")
        print("="*80)
        
        # 计算百分比
        print("\n占比分析:")
        for name, times in sorted(self.timings.items(), key=lambda x: sum(x[1]), reverse=True):
            total = sum(times)
            percentage = (total / total_time) * 100
            print(f"{name:50s}: {percentage:6.2f}%")
        print("="*80)
    
    def reset(self):
        """重置统计"""
        self.timings = {}


def analyze_data_transfer_overhead(
    batch_size: int = 4,
    seq_len: int = 128,
    hidden_size: int = 4096,
    device_a: torch.device = torch.device("cuda:0"),
    device_b: torch.device = torch.device("cuda:1")
):
    """分析数据传输开销"""
    print("\n" + "="*80)
    print(f"数据传输开销分析")
    print(f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
    print(f"device_a={device_a}, device_b={device_b}")
    print("="*80)
    
    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device_a)
    positions = torch.arange(seq_len, device=device_a).unsqueeze(0).expand(batch_size, -1)
    
    data_size_mb = hidden_states.numel() * hidden_states.element_size() / (1024 * 1024)
    
    # 测试传输时间
    n_iters = 10
    
    # A -> B
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        data_b = hidden_states.to(device_b)
        torch.cuda.synchronize()
    elapsed_a_to_b = (time.perf_counter() - start) / n_iters * 1000
    
    # B -> A
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        data_a = data_b.to(device_a)
        torch.cuda.synchronize()
    elapsed_b_to_a = (time.perf_counter() - start) / n_iters * 1000
    
    # 计算带宽
    bandwidth_a_to_b = data_size_mb / (elapsed_a_to_b / 1000)
    bandwidth_b_to_a = data_size_mb / (elapsed_b_to_a / 1000)
    
    print(f"\n数据大小: {data_size_mb:.2f} MB")
    print(f"A -> B 传输时间: {elapsed_a_to_b:.3f} ms (带宽: {bandwidth_a_to_b:.2f} GB/s)")
    print(f"B -> A 传输时间: {elapsed_b_to_a:.3f} ms (带宽: {bandwidth_b_to_a:.2f} GB/s)")
    print(f"往返总时间: {elapsed_a_to_b + elapsed_b_to_a:.3f} ms")
    
    return elapsed_a_to_b + elapsed_b_to_a


def analyze_split_merge_overhead(
    batch_size: int = 4,
    seq_len: int = 128,
    hidden_size: int = 4096,
    split_ratio: float = 0.5,
    device: torch.device = torch.device("cuda:0")
):
    """分析切分和合并开销"""
    print("\n" + "="*80)
    print("切分/合并操作开销分析")
    print("="*80)
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    split_idx = int(batch_size * split_ratio)
    
    n_iters = 100
    
    # 测试切分
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        hs_a = hidden_states[:split_idx].contiguous()
        hs_b = hidden_states[split_idx:].contiguous()
        torch.cuda.synchronize()
    elapsed_split = (time.perf_counter() - start) / n_iters * 1000
    
    hs_a = hidden_states[:split_idx].contiguous()
    hs_b = hidden_states[split_idx:].contiguous()
    
    # 测试合并
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        merged = torch.cat([hs_a, hs_b], dim=0)
        torch.cuda.synchronize()
    elapsed_merge = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"切分操作: {elapsed_split:.3f} ms")
    print(f"合并操作: {elapsed_merge:.3f} ms")
    print(f"总开销: {elapsed_split + elapsed_merge:.3f} ms")
    
    return elapsed_split + elapsed_merge


def analyze_context_split_overhead(
    batch_size: int = 4,
    seq_len: int = 128,
    split_ratio: float = 0.5,
    device: torch.device = torch.device("cuda:0")
):
    """分析context切分开销"""
    print("\n" + "="*80)
    print("Context切分开销分析")
    print("="*80)
    
    # 模拟context数据
    cu_seqlens_q = torch.cumsum(torch.tensor([0] + [seq_len] * batch_size), dim=0).to(device)
    slot_mapping = torch.arange(batch_size * seq_len, device=device)
    context_lens = torch.full((batch_size,), seq_len, device=device)
    block_tables = torch.arange(batch_size * 10, device=device).view(batch_size, 10)
    
    split_idx = int(batch_size * split_ratio)
    token_split_idx = cu_seqlens_q[split_idx].item()
    
    n_iters = 100
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iters):
        # 切分所有context
        cu_seqlens_q_a = cu_seqlens_q[:split_idx+1].contiguous()
        cu_seqlens_q_b = cu_seqlens_q[split_idx:].clone().contiguous()
        cu_seqlens_q_b = cu_seqlens_q_b - cu_seqlens_q_b[0]
        
        slot_mapping_a = slot_mapping[:token_split_idx].contiguous()
        slot_mapping_b = slot_mapping[token_split_idx:].contiguous()
        
        context_lens_a = context_lens[:split_idx].contiguous()
        context_lens_b = context_lens[split_idx:].contiguous()
        
        block_tables_a = block_tables[:split_idx].contiguous()
        block_tables_b = block_tables[split_idx:].contiguous()
        
        torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"Context切分总开销: {elapsed:.3f} ms")
    print(f"  - cu_seqlens: 2个切片 + 1个clone + 1个减法")
    print(f"  - slot_mapping: 2个切片")
    print(f"  - context_lens: 2个切片")
    print(f"  - block_tables: 2个切片")
    print(f"  总计: ~8-10个小张量操作")
    
    return elapsed


def analyze_stream_overhead(
    device_a: torch.device = torch.device("cuda:0"),
    device_b: torch.device = torch.device("cuda:1")
):
    """分析stream创建和同步开销"""
    print("\n" + "="*80)
    print("Stream开销分析")
    print("="*80)
    
    n_iters = 100
    
    # 创建开销
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        stream_a = torch.cuda.Stream(device=device_a)
        stream_b = torch.cuda.Stream(device=device_b)
    elapsed_create = (time.perf_counter() - start) / n_iters * 1000
    
    # 同步开销（空stream）
    stream_a = torch.cuda.Stream(device=device_a)
    stream_b = torch.cuda.Stream(device=device_b)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        stream_a.synchronize()
        stream_b.synchronize()
    elapsed_sync = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"创建2个stream: {elapsed_create:.3f} ms")
    print(f"同步2个stream (空): {elapsed_sync:.3f} ms")
    print(f"总开销: {elapsed_create + elapsed_sync:.3f} ms")
    
    return elapsed_create + elapsed_sync


def estimate_optimization_overhead(
    batch_size: int = 4,
    seq_len: int = 128,
    hidden_size: int = 4096,
    device_a: torch.device = torch.device("cuda:0"),
    device_b: torch.device = torch.device("cuda:1")
):
    """估算单次优化操作的总开销"""
    print("\n" + "="*80)
    print("单次优化操作总开销估算")
    print("="*80)
    
    overhead = {}
    
    # 数据传输
    overhead['data_transfer'] = analyze_data_transfer_overhead(
        batch_size, seq_len, hidden_size, device_a, device_b
    )
    
    # 切分合并
    overhead['split_merge'] = analyze_split_merge_overhead(
        batch_size, seq_len, hidden_size, device=device_a
    )
    
    # Context切分
    overhead['context_split'] = analyze_context_split_overhead(
        batch_size, seq_len, device=device_a
    )
    
    # Stream开销
    overhead['stream'] = analyze_stream_overhead(device_a, device_b)
    
    total_overhead = sum(overhead.values())
    
    print("\n" + "="*80)
    print("总开销汇总:")
    print("="*80)
    for name, value in overhead.items():
        print(f"{name:30s}: {value:8.3f} ms ({value/total_overhead*100:5.1f}%)")
    print(f"{'总计':30s}: {total_overhead:8.3f} ms")
    print("="*80)
    
    print("\n结论:")
    print(f"  每次优化操作的固定开销约为 {total_overhead:.2f} ms")
    print(f"  只有当并行计算节省的时间 > {total_overhead:.2f} ms 时，优化才有效")
    print(f"  例如：如果单设备计算需要 20ms，则分片后每个设备需要 < {(20-total_overhead)/2:.2f} ms")
    
    return total_overhead


if __name__ == "__main__":
    # 检查设备
    if torch.cuda.device_count() < 2:
        print("警告: 只有1个GPU，某些测试将跳过")
        device_a = torch.device("cuda:0")
        device_b = torch.device("cpu")
    else:
        device_a = torch.device("cuda:0")
        device_b = torch.device("cuda:1")
    
    print(f"\n使用设备: device_a={device_a}, device_b={device_b}")
    
    # 运行分析
    estimate_optimization_overhead(
        batch_size=4,
        seq_len=128,
        hidden_size=4096,
        device_a=device_a,
        device_b=device_b
    )

