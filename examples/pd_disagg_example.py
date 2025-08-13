"""
pd分离架构示例代码
"""
import asyncio
import logging
import os
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 添加父目录到路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from hbserve.pd_disagg.config import PdDisaggConfig, KVTransferBackend, InstanceType
from hbserve.pd_disagg.disaggregated_llm import DisaggregatedLLM, LLM
from hbserve.sampling_params import SamplingParams


async def async_example():
    """异步接口示例"""
    print("=== 异步pd分离架构示例 ===")
    
    # 创建配置
    config = PdDisaggConfig(
        model_path=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        tensor_parallel_size=1,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        
        # KV传输配置
        kv_transfer_backend=KVTransferBackend.NCCL,  # 或 MOONCAKE
        kv_buffer_size=1024 * 1024 * 1024,  # 1GB
        
        # 分块prefill配置
        chunked_prefill_enabled=True,
        chunk_size=512,
        
        # CUDA graph配置
        cuda_graph_enabled=True,
        cuda_graph_max_seq_len=2048,
        
        # GPU实例配置
        gpu_instances=[
            {
                "instance_id": "prefill-0",
                "instance_type": InstanceType.PREFILL,
                "gpu_id": 0,
                "host": "localhost",
                "port": 50051,
                "kv_rank": 0
            },
            {
                "instance_id": "decode-0",
                "instance_type": InstanceType.DECODE,
                "gpu_id": 0,  # 单GPU模拟多实例
                "host": "localhost", 
                "port": 50052,
                "kv_rank": 1
            }
        ]
    )
    
    # 创建分离式LLM
    llm = DisaggregatedLLM(config)
    
    try:
        # 启动系统
        print("启动分离式LLM系统...")
        await llm.start()
        
        # 准备测试数据
        prompts = [
            "介绍一下人工智能的发展历史",
            "请列举一些机器学习的应用场景",
            "解释什么是深度学习",
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=256,
            top_p=0.9
        )
        
        # 批量生成
        print(f"开始生成 {len(prompts)} 个请求...")
        results = await llm.generate(prompts, sampling_params)
        
        # 显示结果
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\n--- 请求 {i+1} ---")
            print(f"Prompt: {prompt}")
            print(f"Response: {result['text']}")
            print(f"Status: {result['status']}")
            print(f"Tokens: {len(result['token_ids'])}")
            
        # 显示系统统计
        print("\n=== 系统统计信息 ===")
        stats = llm.get_system_stats()
        
        print(f"CPU调度器统计:")
        cpu_stats = stats["cpu_scheduler"]
        print(f"  总请求数: {cpu_stats['total_requests']}")
        print(f"  完成请求数: {cpu_stats['completed_requests']}")
        print(f"  成功率: {cpu_stats['success_rate']:.2%}")
        
        print(f"GPU实例统计:")
        for instance_id, instance_stats in stats["gpu_instances"].items():
            print(f"  {instance_id}:")
            print(f"    状态: {instance_stats['status']}")
            print(f"    总请求数: {instance_stats['total_requests']}")
            print(f"    完成请求数: {instance_stats['completed_requests']}")
            
            # KV传输统计
            kv_stats = instance_stats.get("kv_transfer_stats", {})
            if kv_stats:
                print(f"    KV传输:")
                print(f"      后端: {kv_stats.get('backend', 'unknown')}")
                print(f"      缓冲区大小: {kv_stats.get('lookup_buffer_size', 0)}")
                print(f"      完成传输: {kv_stats.get('completed_transfers', 0)}")
        
    finally:
        # 停止系统
        print("停止分离式LLM系统...")
        await llm.stop()


def sync_example():
    """同步接口示例（兼容原始hbserve）"""
    print("\n=== 同步pd分离架构示例 ===")
    
    # 使用同步接口，参数与原始hbserve兼容
    llm = LLM(
        model=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        tensor_parallel_size=1,
        
        # pd分离特有配置
        kv_transfer_backend=KVTransferBackend.NCCL,
        chunked_prefill_enabled=True,
        cuda_graph_enabled=True
    )
    
    try:
        # 准备测试数据
        prompts = [
            "什么是量子计算？",
            "请简要介绍区块链技术"
        ]
        
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=128
        )
        
        # 生成（与原始hbserve接口兼容）
        print("开始同步生成...")
        results = llm.generate(prompts, sampling_params)
        
        # 显示结果
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\n--- 同步请求 {i+1} ---")
            print(f"Prompt: {prompt}")
            print(f"Response: {result['text']}")
            
    except Exception as e:
        print(f"同步示例出错: {e}")


def chunked_prefill_example():
    """分块prefill示例"""
    print("\n=== 分块Prefill示例 ===")
    
    # 创建长prompt测试分块prefill
    long_prompt = "请详细介绍人工智能的发展历程，包括：" + \
                 "1. 早期的符号主义AI研究；" * 20 + \
                 "2. 连接主义和神经网络的兴起；" * 20 + \
                 "3. 深度学习革命；" * 20 + \
                 "4. 大语言模型时代。" * 20
    
    config = PdDisaggConfig(
        model_path=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        chunked_prefill_enabled=True,
        chunk_size=256,  # 较小的分块大小
        cuda_graph_enabled=True
    )
    
    async def run_chunked_example():
        llm = DisaggregatedLLM(config)
        
        try:
            await llm.start()
            
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=100
            )
            
            print(f"长prompt长度: {len(llm.tokenizer.encode(long_prompt))} tokens")
            print("使用分块prefill处理...")
            
            results = await llm.generate([long_prompt], sampling_params)
            
            print(f"分块prefill结果: {results[0]['text'][:200]}...")
            
            # 显示分块统计
            stats = llm.get_system_stats()
            for instance_id, instance_stats in stats["gpu_instances"].items():
                if "chunked_prefill" in instance_stats:
                    chunked_stats = instance_stats["chunked_prefill"]
                    print(f"分块统计 ({instance_id}):")
                    print(f"  处理的分块数: {chunked_stats.get('total_chunks_processed', 0)}")
                    print(f"  CUDA Graph命中率: {chunked_stats.get('cuda_graph_hit_rate', 0):.2%}")
                    
        finally:
            await llm.stop()
    
    asyncio.run(run_chunked_example())


def mooncake_backend_example():
    """Mooncake后端示例"""
    print("\n=== Mooncake后端示例 ===")
    
    config = PdDisaggConfig(
        model_path=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        kv_transfer_backend=KVTransferBackend.MOONCAKE,
        kv_ip="127.0.0.1",
        kv_port=12346
    )
    
    async def run_mooncake_example():
        llm = DisaggregatedLLM(config)
        
        try:
            print("启动Mooncake后端...")
            await llm.start()
            
            prompts = ["使用Mooncake后端的测试请求"]
            sampling_params = SamplingParams(temperature=0.6, max_tokens=50)
            
            results = await llm.generate(prompts, sampling_params)
            print(f"Mooncake后端结果: {results[0]['text']}")
            
        except Exception as e:
            print(f"Mooncake后端示例失败（可能需要安装Mooncake）: {e}")
        finally:
            await llm.stop()
    
    asyncio.run(run_mooncake_example())


def performance_comparison():
    """性能对比示例"""
    print("\n=== 性能对比示例 ===")
    
    import time
    
    # 标准配置
    standard_config = PdDisaggConfig(
        model_path=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        chunked_prefill_enabled=False,
        cuda_graph_enabled=False
    )
    
    # 优化配置
    optimized_config = PdDisaggConfig(
        model_path=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        chunked_prefill_enabled=True,
        cuda_graph_enabled=True,
        chunk_size=512
    )
    
    test_prompts = [
        "请解释机器学习的基本概念" * 10,  # 长prompt测试
        "简单问题",  # 短prompt测试
    ]
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=100)
    
    async def benchmark_config(config, name):
        llm = DisaggregatedLLM(config)
        
        try:
            await llm.start()
            
            start_time = time.time()
            results = await llm.generate(test_prompts, sampling_params)
            end_time = time.time()
            
            total_tokens = sum(len(r['token_ids']) for r in results)
            throughput = total_tokens / (end_time - start_time)
            
            print(f"{name}:")
            print(f"  总时间: {end_time - start_time:.2f}s")
            print(f"  吞吐量: {throughput:.2f} tokens/s")
            
            return end_time - start_time, throughput
            
        finally:
            await llm.stop()
    
    async def run_benchmark():
        print("基准测试中...")
        
        standard_time, standard_throughput = await benchmark_config(
            standard_config, "标准配置"
        )
        
        optimized_time, optimized_throughput = await benchmark_config(
            optimized_config, "优化配置"
        )
        
        print(f"\n性能提升:")
        print(f"  时间减少: {(1 - optimized_time/standard_time)*100:.1f}%")
        print(f"  吞吐量提升: {(optimized_throughput/standard_throughput - 1)*100:.1f}%")
    
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    print("pd分离架构示例")
    print("注意：需要先下载模型到 ~/huggingface/Qwen3-0.6B/")
    
    # 检查模型是否存在
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        print("请先下载模型:")
        print("huggingface-cli download --resume-download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
        exit(1)
    
    try:
        # 运行异步示例
        asyncio.run(async_example())
        
        # 运行同步示例
        sync_example()
        
        # 运行分块prefill示例
        chunked_prefill_example()
        
        # 运行Mooncake示例（可选）
        mooncake_backend_example()
        
        # 运行性能对比（可选）
        performance_comparison()
        
    except KeyboardInterrupt:
        print("\n示例被用户中断")
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()
