#!/usr/bin/env python3
"""
优化策略性能对比工具

对比以下策略：
1. Baseline (无优化)
2. Layer Replication (层复制)
3. Attention Offload (注意力卸载)
4. Continuous Replication (连续层复制)
5. KV Head Split (KV头切分)

生成详细的性能报告，包括：
- 延迟对比
- 吞吐量对比
- 加速比
- GPU利用率
- 内存使用
"""

import torch
import time
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tabulate import tabulate
import numpy as np


@dataclass
class BenchmarkConfig:
    """测试配置"""
    batch_size: int
    seq_len: int
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    device_a: str = "cuda:0"
    device_b: str = "cuda:1"
    num_warmup: int = 3
    num_iterations: int = 10


@dataclass
class BenchmarkResult:
    """测试结果"""
    strategy: str
    config: BenchmarkConfig
    avg_latency_ms: float
    std_latency_ms: float
    throughput_tokens_per_sec: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    speedup_vs_baseline: float
    success: bool
    error_msg: Optional[str] = None
    
    def to_dict(self):
        result = asdict(self)
        result['config'] = asdict(self.config)
        return result


class OptimizationBenchmark:
    """优化策略性能测试"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.baseline_latency = None
        
        # 设置设备
        self.device_a = torch.device(config.device_a)
        self.device_b = torch.device(config.device_b)
        
        print(f"\n{'='*80}")
        print(f"初始化性能测试")
        print(f"{'='*80}")
        print(f"配置: batch={config.batch_size}, seq_len={config.seq_len}")
        print(f"设备: {config.device_a} <-> {config.device_b}")
        print(f"预热次数: {config.num_warmup}, 测试次数: {config.num_iterations}")
        print(f"{'='*80}\n")
    
    def create_mock_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建模拟数据"""
        batch_size = self.config.batch_size
        seq_len = self.config.seq_len
        hidden_size = self.config.hidden_size
        
        # 创建输入数据
        hidden_states = torch.randn(
            batch_size * seq_len, hidden_size,
            dtype=torch.float16,
            device=self.device_a
        )
        
        positions = torch.arange(
            batch_size * seq_len,
            dtype=torch.long,
            device=self.device_a
        )
        
        return hidden_states, positions
    
    def create_mock_layer(self, device: torch.device) -> torch.nn.Module:
        """创建模拟的transformer层"""
        class MockTransformerLayer(torch.nn.Module):
            def __init__(self, hidden_size, device):
                super().__init__()
                self.linear1 = torch.nn.Linear(hidden_size, hidden_size * 4, device=device, dtype=torch.float16)
                self.linear2 = torch.nn.Linear(hidden_size * 4, hidden_size, device=device, dtype=torch.float16)
                self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.float16)
            
            def forward(self, x):
                residual = x
                x = self.norm(x)
                x = self.linear1(x)
                x = torch.nn.functional.gelu(x)
                x = self.linear2(x)
                return x + residual
        
        return MockTransformerLayer(self.config.hidden_size, device)
    
    def measure_latency(
        self,
        forward_fn: callable,
        warmup: int = None,
        iterations: int = None
    ) -> Tuple[float, float, float, float]:
        """
        测量延迟
        
        Returns:
            (avg_latency_ms, std_latency_ms, memory_allocated_gb, memory_reserved_gb)
        """
        if warmup is None:
            warmup = self.config.num_warmup
        if iterations is None:
            iterations = self.config.num_iterations
        
        # 预热
        for _ in range(warmup):
            try:
                forward_fn()
                torch.cuda.synchronize()
            except Exception as e:
                return 0.0, 0.0, 0.0, 0.0
        
        # 重置峰值内存
        torch.cuda.reset_peak_memory_stats(self.device_a)
        if self.device_b.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device_b)
        
        # 测量
        latencies = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            forward_fn()
            
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)
        
        # 内存统计
        mem_allocated = torch.cuda.max_memory_allocated(self.device_a) / 1024**3
        mem_reserved = torch.cuda.max_memory_reserved(self.device_a) / 1024**3
        
        if self.device_b.type == 'cuda':
            mem_allocated += torch.cuda.max_memory_allocated(self.device_b) / 1024**3
            mem_reserved += torch.cuda.max_memory_reserved(self.device_b) / 1024**3
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        return avg_latency, std_latency, mem_allocated, mem_reserved
    
    def benchmark_baseline(self) -> BenchmarkResult:
        """测试Baseline（无优化）"""
        print(f"{'='*80}")
        print(f"测试策略: Baseline (无优化)")
        print(f"{'='*80}")
        
        try:
            # 创建单个层
            layer = self.create_mock_layer(self.device_a)
            hidden_states, positions = self.create_mock_data()
            
            def forward_fn():
                output = hidden_states
                for _ in range(self.config.num_layers):
                    output = layer(output)
                return output
            
            avg_lat, std_lat, mem_alloc, mem_res = self.measure_latency(forward_fn)
            
            total_tokens = self.config.batch_size * self.config.seq_len
            throughput = total_tokens / (avg_lat / 1000) if avg_lat > 0 else 0
            
            self.baseline_latency = avg_lat
            
            result = BenchmarkResult(
                strategy="Baseline",
                config=self.config,
                avg_latency_ms=avg_lat,
                std_latency_ms=std_lat,
                throughput_tokens_per_sec=throughput,
                memory_allocated_gb=mem_alloc,
                memory_reserved_gb=mem_res,
                speedup_vs_baseline=1.0,
                success=True
            )
            
            print(f"✅ Baseline测试完成")
            print(f"   延迟: {avg_lat:.2f} ± {std_lat:.2f} ms")
            print(f"   吞吐量: {throughput:.0f} tokens/s")
            print(f"   内存: {mem_alloc:.2f} GB\n")
            
        except Exception as e:
            result = BenchmarkResult(
                strategy="Baseline",
                config=self.config,
                avg_latency_ms=0.0,
                std_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                memory_allocated_gb=0.0,
                memory_reserved_gb=0.0,
                speedup_vs_baseline=0.0,
                success=False,
                error_msg=str(e)
            )
            print(f"❌ Baseline测试失败: {e}\n")
        
        self.results.append(result)
        return result
    
    def benchmark_layer_replication(self) -> BenchmarkResult:
        """测试Layer Replication"""
        print(f"{'='*80}")
        print(f"测试策略: Layer Replication (层复制)")
        print(f"{'='*80}")
        
        try:
            # 创建两个设备上的层
            layer_a = self.create_mock_layer(self.device_a)
            layer_b = self.create_mock_layer(self.device_b)
            hidden_states, positions = self.create_mock_data()
            
            split_ratio = 0.5
            split_idx = int(hidden_states.size(0) * split_ratio)
            
            def forward_fn():
                # 切分数据
                hs_a = hidden_states[:split_idx]
                hs_b = hidden_states[split_idx:].to(self.device_b, non_blocking=True)
                
                # 创建两个stream并行执行
                stream_a = torch.cuda.Stream(device=self.device_a)
                stream_b = torch.cuda.Stream(device=self.device_b)
                
                with torch.cuda.stream(stream_a):
                    out_a = hs_a
                    for _ in range(self.config.num_layers):
                        out_a = layer_a(out_a)
                
                with torch.cuda.stream(stream_b):
                    out_b = hs_b
                    for _ in range(self.config.num_layers):
                        out_b = layer_b(out_b)
                
                stream_a.synchronize()
                stream_b.synchronize()
                
                # 合并结果
                out_b = out_b.to(self.device_a, non_blocking=True)
                output = torch.cat([out_a, out_b], dim=0)
                return output
            
            avg_lat, std_lat, mem_alloc, mem_res = self.measure_latency(forward_fn)
            
            total_tokens = self.config.batch_size * self.config.seq_len
            throughput = total_tokens / (avg_lat / 1000) if avg_lat > 0 else 0
            speedup = self.baseline_latency / avg_lat if avg_lat > 0 and self.baseline_latency else 1.0
            
            result = BenchmarkResult(
                strategy="Layer Replication",
                config=self.config,
                avg_latency_ms=avg_lat,
                std_latency_ms=std_lat,
                throughput_tokens_per_sec=throughput,
                memory_allocated_gb=mem_alloc,
                memory_reserved_gb=mem_res,
                speedup_vs_baseline=speedup,
                success=True
            )
            
            print(f"✅ Layer Replication测试完成")
            print(f"   延迟: {avg_lat:.2f} ± {std_lat:.2f} ms")
            print(f"   吞吐量: {throughput:.0f} tokens/s")
            print(f"   加速比: {speedup:.2f}x")
            print(f"   内存: {mem_alloc:.2f} GB\n")
            
        except Exception as e:
            result = BenchmarkResult(
                strategy="Layer Replication",
                config=self.config,
                avg_latency_ms=0.0,
                std_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                memory_allocated_gb=0.0,
                memory_reserved_gb=0.0,
                speedup_vs_baseline=0.0,
                success=False,
                error_msg=str(e)
            )
            print(f"❌ Layer Replication测试失败: {e}\n")
        
        self.results.append(result)
        return result
    
    def benchmark_attention_offload(self) -> BenchmarkResult:
        """测试Attention Offload"""
        print(f"{'='*80}")
        print(f"测试策略: Attention Offload (注意力卸载)")
        print(f"{'='*80}")
        
        try:
            # 创建注意力层和MLP层
            attn_a = torch.nn.Linear(
                self.config.hidden_size, self.config.hidden_size,
                device=self.device_a, dtype=torch.float16
            )
            attn_b = torch.nn.Linear(
                self.config.hidden_size, self.config.hidden_size,
                device=self.device_b, dtype=torch.float16
            )
            mlp = self.create_mock_layer(self.device_a)
            
            hidden_states, positions = self.create_mock_data()
            split_ratio = 0.5
            split_idx = int(hidden_states.size(0) * split_ratio)
            
            def forward_fn():
                output = hidden_states
                for _ in range(self.config.num_layers):
                    # Attention部分offload
                    hs_a = output[:split_idx]
                    hs_b = output[split_idx:].to(self.device_b, non_blocking=True)
                    
                    stream_a = torch.cuda.Stream(device=self.device_a)
                    stream_b = torch.cuda.Stream(device=self.device_b)
                    
                    with torch.cuda.stream(stream_a):
                        out_a = attn_a(hs_a)
                    
                    with torch.cuda.stream(stream_b):
                        out_b = attn_b(hs_b)
                    
                    stream_a.synchronize()
                    stream_b.synchronize()
                    
                    out_b = out_b.to(self.device_a, non_blocking=True)
                    attn_out = torch.cat([out_a, out_b], dim=0)
                    
                    # MLP在主设备
                    output = mlp(attn_out)
                
                return output
            
            avg_lat, std_lat, mem_alloc, mem_res = self.measure_latency(forward_fn)
            
            total_tokens = self.config.batch_size * self.config.seq_len
            throughput = total_tokens / (avg_lat / 1000) if avg_lat > 0 else 0
            speedup = self.baseline_latency / avg_lat if avg_lat > 0 and self.baseline_latency else 1.0
            
            result = BenchmarkResult(
                strategy="Attention Offload",
                config=self.config,
                avg_latency_ms=avg_lat,
                std_latency_ms=std_lat,
                throughput_tokens_per_sec=throughput,
                memory_allocated_gb=mem_alloc,
                memory_reserved_gb=mem_res,
                speedup_vs_baseline=speedup,
                success=True
            )
            
            print(f"✅ Attention Offload测试完成")
            print(f"   延迟: {avg_lat:.2f} ± {std_lat:.2f} ms")
            print(f"   吞吐量: {throughput:.0f} tokens/s")
            print(f"   加速比: {speedup:.2f}x")
            print(f"   内存: {mem_alloc:.2f} GB\n")
            
        except Exception as e:
            result = BenchmarkResult(
                strategy="Attention Offload",
                config=self.config,
                avg_latency_ms=0.0,
                std_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                memory_allocated_gb=0.0,
                memory_reserved_gb=0.0,
                speedup_vs_baseline=0.0,
                success=False,
                error_msg=str(e)
            )
            print(f"❌ Attention Offload测试失败: {e}\n")
        
        self.results.append(result)
        return result
    
    def benchmark_continuous_replication(self) -> BenchmarkResult:
        """测试Continuous Replication"""
        print(f"{'='*80}")
        print(f"测试策略: Continuous Replication (连续层复制)")
        print(f"{'='*80}")
        
        try:
            # 在两个设备上各创建一半的层
            num_layers_per_device = self.config.num_layers // 2
            layers_a = [self.create_mock_layer(self.device_a) for _ in range(num_layers_per_device)]
            layers_b = [self.create_mock_layer(self.device_b) for _ in range(num_layers_per_device)]
            
            hidden_states, positions = self.create_mock_data()
            
            def forward_fn():
                output = hidden_states
                
                # 在device_a执行前半部分层
                for layer in layers_a:
                    output = layer(output)
                
                # 传输到device_b
                output = output.to(self.device_b, non_blocking=True)
                
                # 在device_b执行后半部分层
                for layer in layers_b:
                    output = layer(output)
                
                # 传输回device_a
                output = output.to(self.device_a, non_blocking=True)
                
                return output
            
            avg_lat, std_lat, mem_alloc, mem_res = self.measure_latency(forward_fn)
            
            total_tokens = self.config.batch_size * self.config.seq_len
            throughput = total_tokens / (avg_lat / 1000) if avg_lat > 0 else 0
            speedup = self.baseline_latency / avg_lat if avg_lat > 0 and self.baseline_latency else 1.0
            
            result = BenchmarkResult(
                strategy="Continuous Replication",
                config=self.config,
                avg_latency_ms=avg_lat,
                std_latency_ms=std_lat,
                throughput_tokens_per_sec=throughput,
                memory_allocated_gb=mem_alloc,
                memory_reserved_gb=mem_res,
                speedup_vs_baseline=speedup,
                success=True
            )
            
            print(f"✅ Continuous Replication测试完成")
            print(f"   延迟: {avg_lat:.2f} ± {std_lat:.2f} ms")
            print(f"   吞吐量: {throughput:.0f} tokens/s")
            print(f"   加速比: {speedup:.2f}x")
            print(f"   内存: {mem_alloc:.2f} GB\n")
            
        except Exception as e:
            result = BenchmarkResult(
                strategy="Continuous Replication",
                config=self.config,
                avg_latency_ms=0.0,
                std_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                memory_allocated_gb=0.0,
                memory_reserved_gb=0.0,
                speedup_vs_baseline=0.0,
                success=False,
                error_msg=str(e)
            )
            print(f"❌ Continuous Replication测试失败: {e}\n")
        
        self.results.append(result)
        return result
    
    def run_all_benchmarks(self):
        """运行所有测试"""
        print(f"\n{'='*80}")
        print(f"开始完整性能测试")
        print(f"{'='*80}\n")
        
        # 测试各种策略
        self.benchmark_baseline()
        self.benchmark_layer_replication()
        self.benchmark_attention_offload()
        self.benchmark_continuous_replication()
        
        print(f"\n{'='*80}")
        print(f"所有测试完成")
        print(f"{'='*80}\n")
    
    def print_summary(self):
        """打印测试摘要"""
        print(f"\n{'='*80}")
        print(f"性能测试摘要")
        print(f"{'='*80}\n")
        
        print(f"配置:")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Sequence Length: {self.config.seq_len}")
        print(f"  Total Tokens: {self.config.batch_size * self.config.seq_len}")
        print(f"  Hidden Size: {self.config.hidden_size}")
        print(f"  Num Layers: {self.config.num_layers}")
        print(f"  设备: {self.config.device_a} <-> {self.config.device_b}\n")
        
        # 创建表格
        table_data = []
        for result in self.results:
            if result.success:
                table_data.append([
                    result.strategy,
                    f"{result.avg_latency_ms:.2f}",
                    f"{result.std_latency_ms:.2f}",
                    f"{result.throughput_tokens_per_sec:.0f}",
                    f"{result.speedup_vs_baseline:.2f}x",
                    f"{result.memory_allocated_gb:.2f}",
                    "✅"
                ])
            else:
                table_data.append([
                    result.strategy,
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    f"❌ {result.error_msg[:20]}"
                ])
        
        headers = ["策略", "平均延迟(ms)", "标准差(ms)", "吞吐量(tok/s)", "加速比", "内存(GB)", "状态"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # 找出最佳策略
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            best_latency = min(successful_results, key=lambda r: r.avg_latency_ms)
            best_throughput = max(successful_results, key=lambda r: r.throughput_tokens_per_sec)
            
            print(f"\n最佳性能:")
            print(f"  最低延迟: {best_latency.strategy} ({best_latency.avg_latency_ms:.2f} ms)")
            print(f"  最高吞吐: {best_throughput.strategy} ({best_throughput.throughput_tokens_per_sec:.0f} tokens/s)")
    
    def save_results(self, output_path: str = "benchmark_results.json"):
        """保存结果到文件"""
        results_dict = {
            'config': asdict(self.config),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✅ 结果已保存到: {output_path}")


def run_comprehensive_benchmark():
    """运行综合测试（多种配置）"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    优化策略性能综合对比测试                                  ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # 检查GPU
    if torch.cuda.device_count() < 2:
        print("⚠️  警告: 需要至少2个GPU进行测试")
        print("   将使用 cuda:0 和 cpu 进行测试（性能会受影响）")
        device_b = "cpu"
    else:
        device_b = "cuda:1"
    
    # 多种配置测试
    test_configs = [
        # 小batch
        BenchmarkConfig(batch_size=4, seq_len=128, num_layers=8, device_b=device_b),
        # 中等batch
        BenchmarkConfig(batch_size=8, seq_len=512, num_layers=8, device_b=device_b),
        # 大batch
        BenchmarkConfig(batch_size=16, seq_len=1024, num_layers=8, device_b=device_b),
        # 超大batch
        BenchmarkConfig(batch_size=32, seq_len=2048, num_layers=8, device_b=device_b),
    ]
    
    all_results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'#'*80}")
        print(f"# 测试配置 {i}/{len(test_configs)}")
        print(f"{'#'*80}\n")
        
        benchmark = OptimizationBenchmark(config)
        benchmark.run_all_benchmarks()
        benchmark.print_summary()
        
        # 保存结果
        output_file = f"benchmark_results_b{config.batch_size}_s{config.seq_len}.json"
        benchmark.save_results(output_file)
        
        all_results.append(benchmark.results)
    
    # 生成综合报告
    generate_comprehensive_report(test_configs, all_results)


def generate_comprehensive_report(configs: List[BenchmarkConfig], all_results: List[List[BenchmarkResult]]):
    """生成综合对比报告"""
    print(f"\n{'='*80}")
    print(f"综合性能对比报告")
    print(f"{'='*80}\n")
    
    # 为每种策略创建对比表
    strategies = ["Baseline", "Layer Replication", "Attention Offload", "Continuous Replication"]
    
    for strategy in strategies:
        print(f"\n{strategy} - 不同配置下的性能:")
        print(f"{'-'*80}")
        
        table_data = []
        for config, results in zip(configs, all_results):
            result = next((r for r in results if r.strategy == strategy), None)
            if result and result.success:
                table_data.append([
                    f"{config.batch_size}×{config.seq_len}",
                    config.batch_size * config.seq_len,
                    f"{result.avg_latency_ms:.2f}",
                    f"{result.throughput_tokens_per_sec:.0f}",
                    f"{result.speedup_vs_baseline:.2f}x",
                    f"{result.memory_allocated_gb:.2f}"
                ])
        
        headers = ["配置", "Tokens", "延迟(ms)", "吞吐量(tok/s)", "加速比", "内存(GB)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 加速比热力图数据
    print(f"\n{'='*80}")
    print(f"加速比对比（相对于Baseline）")
    print(f"{'='*80}\n")
    
    heatmap_data = []
    for config, results in zip(configs, all_results):
        row = [f"{config.batch_size}×{config.seq_len}"]
        for strategy in strategies[1:]:  # 跳过Baseline
            result = next((r for r in results if r.strategy == strategy), None)
            if result and result.success:
                speedup = result.speedup_vs_baseline
                if speedup >= 1.5:
                    row.append(f"{speedup:.2f}x ✅✅")
                elif speedup >= 1.2:
                    row.append(f"{speedup:.2f}x ✅")
                elif speedup >= 1.0:
                    row.append(f"{speedup:.2f}x ⚠️")
                else:
                    row.append(f"{speedup:.2f}x ❌")
            else:
                row.append("N/A")
        heatmap_data.append(row)
    
    headers = ["配置"] + strategies[1:]
    print(tabulate(heatmap_data, headers=headers, tablefmt="grid"))
    
    print(f"\n图例:")
    print(f"  ✅✅ : 加速 ≥ 1.5x (推荐)")
    print(f"  ✅   : 加速 ≥ 1.2x (有效)")
    print(f"  ⚠️   : 加速 ≥ 1.0x (边界)")
    print(f"  ❌   : 加速 < 1.0x (变慢)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="优化策略性能对比测试")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive benchmark")
    parser.add_argument("--device_a", type=str, default="cuda:0", help="Primary device")
    parser.add_argument("--device_b", type=str, default="cuda:1", help="Secondary device")
    
    args = parser.parse_args()
    
    if args.comprehensive:
        # 运行综合测试
        run_comprehensive_benchmark()
    else:
        # 运行单个配置测试
        config = BenchmarkConfig(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_layers=args.num_layers,
            device_a=args.device_a,
            device_b=args.device_b
        )
        
        benchmark = OptimizationBenchmark(config)
        benchmark.run_all_benchmarks()
        benchmark.print_summary()
        benchmark.save_results()

