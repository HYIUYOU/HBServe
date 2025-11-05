#!/usr/bin/env python3
"""
真实模型性能对比工具

使用实际的LLM模型测试不同优化策略的性能
基于 HBserve.LLM 和真实的transformer模型
"""

import os
import torch
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tabulate import tabulate
from transformers import AutoTokenizer

from HBserve import LLM, SamplingParams


@dataclass
class BenchmarkConfig:
    """测试配置"""
    model_path: str
    batch_size: int
    max_tokens: int = 256
    temperature: float = 0.6
    num_warmup: int = 2
    num_iterations: int = 5
    gpu_memory_utilization: float = 0.6
    tensor_parallel_size: int = 1


@dataclass
class BenchmarkResult:
    """测试结果"""
    strategy: str
    config: BenchmarkConfig
    avg_latency_ms: float
    std_latency_ms: float
    throughput_tokens_per_sec: float
    prefill_time_ms: float
    decode_time_ms: float
    total_tokens_generated: int
    memory_allocated_gb: float
    speedup_vs_baseline: float
    success: bool
    error_msg: Optional[str] = None
    layer_ids: Optional[List[int]] = None  # 应用优化的层ID
    
    def to_dict(self):
        result = asdict(self)
        result['config'] = asdict(self.config)
        return result


class RealModelBenchmark:
    """真实模型性能测试"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.baseline_latency = None
        self.llm = None
        self.tokenizer = None
        self.test_prompts = None
        
        print(f"\n{'='*80}")
        print(f"初始化真实模型性能测试")
        print(f"{'='*80}")
        print(f"模型: {config.model_path}")
        print(f"Batch Size: {config.batch_size}")
        print(f"Max Tokens: {config.max_tokens}")
        print(f"预热次数: {config.num_warmup}, 测试次数: {config.num_iterations}")
        print(f"{'='*80}\n")
    
    def load_model(self):
        """加载模型"""
        print("加载模型和tokenizer...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        # 加载模型
        self.llm = LLM(
            self.config.model_path,
            enforce_eager=True,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization
        )
        
        print(f"✅ 模型加载完成\n")
    
    def prepare_prompts(self):
        """准备测试prompts"""
        # 使用多样化的prompts
        base_prompts = [
            "Introduce yourself briefly.",
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "What are the benefits of deep learning?",
            "How does neural network work?",
            "What is natural language processing?",
            "Describe computer vision applications.",
            "What is reinforcement learning?",
        ]
        
        # 根据batch_size选择prompts
        selected_prompts = base_prompts[:self.config.batch_size]
        if len(selected_prompts) < self.config.batch_size:
            # 如果不够，重复使用
            selected_prompts = selected_prompts * (self.config.batch_size // len(selected_prompts) + 1)
            selected_prompts = selected_prompts[:self.config.batch_size]
        
        # 应用chat template
        self.test_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            for prompt in selected_prompts
        ]
        
        print(f"准备了 {len(self.test_prompts)} 个测试prompts")
    
    def measure_inference(
        self,
        sampling_params: SamplingParams = None
    ) -> Tuple[float, float, float, float, int]:
        """
        测量推理性能
        
        Returns:
            (total_time_ms, prefill_time_ms, decode_time_ms, memory_gb, total_tokens)
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        
        # 重置内存统计
        torch.cuda.reset_peak_memory_stats()
        
        # 测量时间
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        outputs = self.llm.generate(self.test_prompts, sampling_params)
        
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) * 1000  # ms
        
        # 统计生成的token数
        total_tokens = sum(len(out.get('token_ids', [])) for out in outputs)
        
        # 内存使用
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        
        # 注意：这里无法精确分离prefill和decode时间，使用总时间
        # 如果需要更精确的测量，需要修改LLM内部逻辑
        prefill_time = total_time * 0.3  # 估算：30%为prefill
        decode_time = total_time * 0.7   # 估算：70%为decode
        
        return total_time, prefill_time, decode_time, memory_gb, total_tokens
    
    def benchmark_baseline(self) -> BenchmarkResult:
        """测试Baseline（无优化）"""
        print(f"{'='*80}")
        print(f"测试策略: Baseline (无优化)")
        print(f"{'='*80}")
        
        try:
            # 确保没有启用任何优化
            model = self.llm.model_runner.model.model
            model.clear_all_optimizations()
            
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # 预热
            print("预热中...")
            for _ in range(self.config.num_warmup):
                _ = self.llm.generate(self.test_prompts[:1], sampling_params)
                torch.cuda.synchronize()
            
            # 测量
            print("测量中...")
            latencies = []
            total_tokens_list = []
            
            for i in range(self.config.num_iterations):
                total_time, prefill_time, decode_time, memory_gb, total_tokens = \
                    self.measure_inference(sampling_params)
                latencies.append(total_time)
                total_tokens_list.append(total_tokens)
                print(f"  迭代 {i+1}/{self.config.num_iterations}: {total_time:.2f} ms")
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            avg_tokens = np.mean(total_tokens_list)
            throughput = avg_tokens / (avg_latency / 1000)
            
            self.baseline_latency = avg_latency
            
            result = BenchmarkResult(
                strategy="Baseline",
                config=self.config,
                avg_latency_ms=avg_latency,
                std_latency_ms=std_latency,
                throughput_tokens_per_sec=throughput,
                prefill_time_ms=prefill_time,
                decode_time_ms=decode_time,
                total_tokens_generated=int(avg_tokens),
                memory_allocated_gb=memory_gb,
                speedup_vs_baseline=1.0,
                success=True
            )
            
            print(f"✅ Baseline测试完成")
            print(f"   延迟: {avg_latency:.2f} ± {std_latency:.2f} ms")
            print(f"   吞吐量: {throughput:.0f} tokens/s")
            print(f"   内存: {memory_gb:.2f} GB\n")
            
        except Exception as e:
            print(f"❌ Baseline测试失败: {e}\n")
            import traceback
            traceback.print_exc()
            result = BenchmarkResult(
                strategy="Baseline",
                config=self.config,
                avg_latency_ms=0.0,
                std_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                prefill_time_ms=0.0,
                decode_time_ms=0.0,
                total_tokens_generated=0,
                memory_allocated_gb=0.0,
                speedup_vs_baseline=0.0,
                success=False,
                error_msg=str(e)
            )
        
        self.results.append(result)
        return result
    
    def benchmark_layer_replication(
        self,
        layer_ids: List[int] = None,
        replica_device: str = 'cuda:1',
        split_ratio: float = 0.5,
        enable_autotune: bool = False
    ) -> BenchmarkResult:
        """测试Layer Replication"""
        print(f"{'='*80}")
        print(f"测试策略: Layer Replication (层复制)")
        print(f"{'='*80}")
        
        try:
            model = self.llm.model_runner.model.model
            
            # 清除之前的配置
            model.clear_all_optimizations()
            
            # 确定要复制的层
            if layer_ids is None:
                num_layers = len(model.layers)
                # 默认复制中间几层
                layer_ids = list(range(num_layers // 3, 2 * num_layers // 3))
            
            print(f"在层 {layer_ids} 上启用 Layer Replication...")
            print(f"  replica_device: {replica_device}")
            print(f"  split_ratio: {split_ratio}")
            
            # 配置layer replication
            for layer_id in layer_ids:
                model.layer_replication(
                    layer_id=layer_id,
                    replica_device=replica_device,
                    split_ratio=split_ratio,
                    enable_autotune=enable_autotune
                )
            
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # 预热
            print("预热中...")
            for _ in range(self.config.num_warmup):
                _ = self.llm.generate(self.test_prompts[:1], sampling_params)
                torch.cuda.synchronize()
            
            # 测量
            print("测量中...")
            latencies = []
            total_tokens_list = []
            
            for i in range(self.config.num_iterations):
                total_time, prefill_time, decode_time, memory_gb, total_tokens = \
                    self.measure_inference(sampling_params)
                latencies.append(total_time)
                total_tokens_list.append(total_tokens)
                print(f"  迭代 {i+1}/{self.config.num_iterations}: {total_time:.2f} ms")
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            avg_tokens = np.mean(total_tokens_list)
            throughput = avg_tokens / (avg_latency / 1000)
            speedup = self.baseline_latency / avg_latency if avg_latency > 0 and self.baseline_latency else 1.0
            
            result = BenchmarkResult(
                strategy="Layer Replication",
                config=self.config,
                avg_latency_ms=avg_latency,
                std_latency_ms=std_latency,
                throughput_tokens_per_sec=throughput,
                prefill_time_ms=prefill_time,
                decode_time_ms=decode_time,
                total_tokens_generated=int(avg_tokens),
                memory_allocated_gb=memory_gb,
                speedup_vs_baseline=speedup,
                success=True,
                layer_ids=layer_ids
            )
            
            print(f"✅ Layer Replication测试完成")
            print(f"   延迟: {avg_latency:.2f} ± {std_latency:.2f} ms")
            print(f"   吞吐量: {throughput:.0f} tokens/s")
            print(f"   加速比: {speedup:.2f}x")
            print(f"   内存: {memory_gb:.2f} GB\n")
            
            # 清理配置
            for layer_id in layer_ids:
                model.clear_layer_replication(layer_id)
            
        except Exception as e:
            print(f"❌ Layer Replication测试失败: {e}\n")
            import traceback
            traceback.print_exc()
            result = BenchmarkResult(
                strategy="Layer Replication",
                config=self.config,
                avg_latency_ms=0.0,
                std_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                prefill_time_ms=0.0,
                decode_time_ms=0.0,
                total_tokens_generated=0,
                memory_allocated_gb=0.0,
                speedup_vs_baseline=0.0,
                success=False,
                error_msg=str(e),
                layer_ids=layer_ids
            )
        
        self.results.append(result)
        return result
    
    def benchmark_attention_offload_batch(
        self,
        layer_ids: List[int] = None,
        offload_device: str = 'cuda:1',
        split_ratio: float = 0.5,
        enable_autotune: bool = False
    ) -> BenchmarkResult:
        """测试Attention Offload (Batch维度切分)"""
        print(f"{'='*80}")
        print(f"测试策略: Attention Offload - Batch Split")
        print(f"{'='*80}")
        
        try:
            model = self.llm.model_runner.model.model
            model.clear_all_optimizations()
            
            if layer_ids is None:
                num_layers = len(model.layers)
                layer_ids = [num_layers // 2]  # 默认中间一层
            
            print(f"在层 {layer_ids} 上启用 Attention Offload (Batch)...")
            
            for layer_id in layer_ids:
                model.attention_offload_by_batch(
                    layer_id=layer_id,
                    offload_device=offload_device,
                    split_ratio=split_ratio,
                    enable_autotune=enable_autotune
                )
            
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # 预热
            print("预热中...")
            for _ in range(self.config.num_warmup):
                _ = self.llm.generate(self.test_prompts[:1], sampling_params)
                torch.cuda.synchronize()
            
            # 测量
            print("测量中...")
            latencies = []
            total_tokens_list = []
            
            for i in range(self.config.num_iterations):
                total_time, prefill_time, decode_time, memory_gb, total_tokens = \
                    self.measure_inference(sampling_params)
                latencies.append(total_time)
                total_tokens_list.append(total_tokens)
                print(f"  迭代 {i+1}/{self.config.num_iterations}: {total_time:.2f} ms")
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            avg_tokens = np.mean(total_tokens_list)
            throughput = avg_tokens / (avg_latency / 1000)
            speedup = self.baseline_latency / avg_latency if avg_latency > 0 and self.baseline_latency else 1.0
            
            result = BenchmarkResult(
                strategy="Attention Offload (Batch)",
                config=self.config,
                avg_latency_ms=avg_latency,
                std_latency_ms=std_latency,
                throughput_tokens_per_sec=throughput,
                prefill_time_ms=prefill_time,
                decode_time_ms=decode_time,
                total_tokens_generated=int(avg_tokens),
                memory_allocated_gb=memory_gb,
                speedup_vs_baseline=speedup,
                success=True,
                layer_ids=layer_ids
            )
            
            print(f"✅ Attention Offload (Batch)测试完成")
            print(f"   延迟: {avg_latency:.2f} ± {std_latency:.2f} ms")
            print(f"   加速比: {speedup:.2f}x\n")
            
            # 清理
            for layer_id in layer_ids:
                model.clear_attention_offload(layer_id)
            
        except Exception as e:
            print(f"❌ Attention Offload (Batch)测试失败: {e}\n")
            import traceback
            traceback.print_exc()
            result = BenchmarkResult(
                strategy="Attention Offload (Batch)",
                config=self.config,
                avg_latency_ms=0.0,
                std_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                prefill_time_ms=0.0,
                decode_time_ms=0.0,
                total_tokens_generated=0,
                memory_allocated_gb=0.0,
                speedup_vs_baseline=0.0,
                success=False,
                error_msg=str(e),
                layer_ids=layer_ids
            )
        
        self.results.append(result)
        return result
    
    def benchmark_kv_head_split(
        self,
        layer_ids: List[int] = None,
        offload_device: str = 'cuda:1',
        split_kv_head_idx: int = None
    ) -> BenchmarkResult:
        """测试KV Head Split"""
        print(f"{'='*80}")
        print(f"测试策略: KV Head Split")
        print(f"{'='*80}")
        
        try:
            model = self.llm.model_runner.model.model
            model.clear_all_optimizations()
            
            if layer_ids is None:
                num_layers = len(model.layers)
                layer_ids = [num_layers // 2]
            
            print(f"在层 {layer_ids} 上启用 KV Head Split...")
            
            for layer_id in layer_ids:
                model.attention_offload_by_kv_head(
                    layer_id=layer_id,
                    offload_device=offload_device,
                    split_kv_head_idx=split_kv_head_idx
                )
            
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # 预热
            print("预热中...")
            for _ in range(self.config.num_warmup):
                _ = self.llm.generate(self.test_prompts[:1], sampling_params)
                torch.cuda.synchronize()
            
            # 测量
            print("测量中...")
            latencies = []
            total_tokens_list = []
            
            for i in range(self.config.num_iterations):
                total_time, prefill_time, decode_time, memory_gb, total_tokens = \
                    self.measure_inference(sampling_params)
                latencies.append(total_time)
                total_tokens_list.append(total_tokens)
                print(f"  迭代 {i+1}/{self.config.num_iterations}: {total_time:.2f} ms")
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            avg_tokens = np.mean(total_tokens_list)
            throughput = avg_tokens / (avg_latency / 1000)
            speedup = self.baseline_latency / avg_latency if avg_latency > 0 and self.baseline_latency else 1.0
            
            result = BenchmarkResult(
                strategy="KV Head Split",
                config=self.config,
                avg_latency_ms=avg_latency,
                std_latency_ms=std_latency,
                throughput_tokens_per_sec=throughput,
                prefill_time_ms=prefill_time,
                decode_time_ms=decode_time,
                total_tokens_generated=int(avg_tokens),
                memory_allocated_gb=memory_gb,
                speedup_vs_baseline=speedup,
                success=True,
                layer_ids=layer_ids
            )
            
            print(f"✅ KV Head Split测试完成")
            print(f"   延迟: {avg_latency:.2f} ± {std_latency:.2f} ms")
            print(f"   加速比: {speedup:.2f}x\n")
            
            # 清理
            for layer_id in layer_ids:
                model.clear_attention_offload(layer_id)
            
        except Exception as e:
            print(f"❌ KV Head Split测试失败: {e}\n")
            import traceback
            traceback.print_exc()
            result = BenchmarkResult(
                strategy="KV Head Split",
                config=self.config,
                avg_latency_ms=0.0,
                std_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                prefill_time_ms=0.0,
                decode_time_ms=0.0,
                total_tokens_generated=0,
                memory_allocated_gb=0.0,
                speedup_vs_baseline=0.0,
                success=False,
                error_msg=str(e),
                layer_ids=layer_ids
            )
        
        self.results.append(result)
        return result
    
    def run_all_benchmarks(self, layer_ids: List[int] = None):
        """运行所有测试"""
        print(f"\n{'='*80}")
        print(f"开始完整性能测试")
        print(f"{'='*80}\n")
        
        # 加载模型
        self.load_model()
        self.prepare_prompts()
        
        # 测试各种策略
        self.benchmark_baseline()
        self.benchmark_layer_replication(layer_ids=layer_ids)
        self.benchmark_attention_offload_batch(layer_ids=layer_ids)
        self.benchmark_kv_head_split(layer_ids=layer_ids)
        
        print(f"\n{'='*80}")
        print(f"所有测试完成")
        print(f"{'='*80}\n")
    
    def print_summary(self):
        """打印测试摘要"""
        print(f"\n{'='*80}")
        print(f"性能测试摘要")
        print(f"{'='*80}\n")
        
        print(f"配置:")
        print(f"  模型: {self.config.model_path}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Max Tokens: {self.config.max_tokens}")
        print(f"  Temperature: {self.config.temperature}\n")
        
        # 创建表格
        table_data = []
        for result in self.results:
            if result.success:
                speedup_str = f"{result.speedup_vs_baseline:.2f}x"
                if result.speedup_vs_baseline >= 1.5:
                    speedup_str += " ✅✅"
                elif result.speedup_vs_baseline >= 1.2:
                    speedup_str += " ✅"
                elif result.speedup_vs_baseline >= 1.0:
                    speedup_str += " ⚠️"
                else:
                    speedup_str += " ❌"
                
                table_data.append([
                    result.strategy,
                    str(result.layer_ids) if result.layer_ids else "N/A",
                    f"{result.avg_latency_ms:.2f}",
                    f"{result.throughput_tokens_per_sec:.0f}",
                    speedup_str,
                    f"{result.memory_allocated_gb:.2f}",
                    "✅"
                ])
            else:
                table_data.append([
                    result.strategy,
                    str(result.layer_ids) if result.layer_ids else "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    f"❌"
                ])
        
        headers = ["策略", "应用层", "延迟(ms)", "吞吐量(tok/s)", "加速比", "内存(GB)", "状态"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # 找出最佳策略
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            best_speedup = max(successful_results, key=lambda r: r.speedup_vs_baseline)
            
            print(f"\n最佳策略: {best_speedup.strategy}")
            print(f"  加速比: {best_speedup.speedup_vs_baseline:.2f}x")
            print(f"  延迟: {best_speedup.avg_latency_ms:.2f} ms")
            print(f"  应用层: {best_speedup.layer_ids}")
    
    def save_results(self, output_path: str = "real_model_benchmark_results.json"):
        """保存结果"""
        results_dict = {
            'config': asdict(self.config),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✅ 结果已保存到: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="真实模型性能对比测试")
    parser.add_argument("--model_path", type=str, 
                       default=os.path.expanduser("../Qwen3-0.6B"),
                       help="模型路径")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_tokens", type=int, default=256, help="最大生成tokens")
    parser.add_argument("--num_iterations", type=int, default=5, help="测试迭代次数")
    parser.add_argument("--layers", type=str, default=None, 
                       help="测试的层ID，逗号分隔，如: 5,6,7")
    
    args = parser.parse_args()
    
    # 解析layer_ids
    layer_ids = None
    if args.layers:
        layer_ids = [int(x.strip()) for x in args.layers.split(',')]
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    真实模型优化策略性能对比测试                               ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    config = BenchmarkConfig(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        num_iterations=args.num_iterations
    )
    
    benchmark = RealModelBenchmark(config)
    benchmark.run_all_benchmarks(layer_ids=layer_ids)
    benchmark.print_summary()
    benchmark.save_results()


if __name__ == "__main__":
    main()

