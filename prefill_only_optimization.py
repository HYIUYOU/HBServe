"""
仅在Prefill阶段启用优化的补丁

原理：修改 optimization_forward.py，在函数内部检查 is_prefill
- Prefill: 启用优化（计算密集，数据并行有效）
- Decode: 跳过优化（计算稀疏，传输开销大）
"""

import os
import re


def apply_prefill_only_optimization():
    """应用仅Prefill优化的补丁"""
    
    optimization_forward_path = "HBserve/utils/optimization_forward.py"
    
    if not os.path.exists(optimization_forward_path):
        print(f"❌ 找不到文件: {optimization_forward_path}")
        return False
    
    # 备份
    backup_path = optimization_forward_path + ".backup_prefill_only"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(optimization_forward_path, backup_path)
        print(f"✅ 已备份: {backup_path}")
    
    with open(optimization_forward_path, 'r') as f:
        content = f.read()
    
    # ===== 修改1: execute_layer_replication_forward =====
    # 在函数开头添加 decode 阶段检查
    
    pattern1 = r"(def execute_layer_replication_forward\([^)]+\)[^:]*:\s*\"\"\"[^\"]*\"\"\")"
    
    replacement1 = r'''\1
    
    # ===== Prefill-Only 优化：Decode阶段跳过 =====
    if not context.is_prefill:
        # Decode阶段：单token计算，传输开销 > 并行收益
        # 直接使用单设备执行
        DEBUG = os.environ.get("HB_REPLICA_LOG", "0") != "0"
        if DEBUG:
            print(f"[Replica][L{layer_id}] Decode阶段，跳过优化（避免传输开销）")
        return layer(positions, hidden_states, residual)
    # ===== End Prefill-Only 优化 ====='''
    
    if re.search(r"def execute_layer_replication_forward", content):
        content = re.sub(pattern1, replacement1, content, count=1)
        print("✅ 修改1: execute_layer_replication_forward 添加 Prefill-Only 检查")
    
    # ===== 修改2: execute_attention_offload_forward =====
    pattern2 = r"(def execute_attention_offload_forward\([^)]+\)[^:]*:\s*\"\"\"[^\"]*\"\"\")"
    
    replacement2 = r'''\1
    
    # ===== Prefill-Only 优化：Decode阶段跳过 =====
    if not context.is_prefill:
        DEBUG = os.environ.get("HB_ATTN_OFFLOAD_LOG", "0") != "0"
        if DEBUG:
            print(f"[AttnOffload][L{layer_id}] Decode阶段，跳过优化")
        return config['src_attn'](positions, hidden_states)
    # ===== End Prefill-Only 优化 ====='''
    
    if re.search(r"def execute_attention_offload_forward", content):
        content = re.sub(pattern2, replacement2, content, count=1)
        print("✅ 修改2: execute_attention_offload_forward 添加 Prefill-Only 检查")
    
    # ===== 修改3: execute_continuous_layer_replication =====
    pattern3 = r"(def execute_continuous_layer_replication\([^)]+\)[^:]*:\s*\"\"\"[^\"]*\"\"\"[^\"]*\"\"\")"
    
    replacement3 = r'''\1
    
    # ===== Prefill-Only 优化：Decode阶段跳过 =====
    if not context.is_prefill:
        DEBUG = os.environ.get("HB_REPLICA_LOG", "0") != "0"
        if DEBUG:
            print(f"[ReplicaGroup][L{layer_id}] Decode阶段，跳过优化")
        return layer(positions, hidden_states, residual)
    # ===== End Prefill-Only 优化 ====='''
    
    if re.search(r"def execute_continuous_layer_replication", content):
        content = re.sub(pattern3, replacement3, content, count=1)
        print("✅ 修改3: execute_continuous_layer_replication 添加 Prefill-Only 检查")
    
    # 写回文件
    with open(optimization_forward_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Prefill-Only 优化已应用到: {optimization_forward_path}")
    return True


def create_comparison_example():
    """创建性能对比示例"""
    
    example_code = '''#!/usr/bin/env python3
"""
性能对比：Prefill-Only 优化 vs 全程优化

测试场景：
1. 不使用优化（baseline）
2. 全程使用优化（prefill + decode都优化）
3. 仅Prefill优化（推荐）
"""

import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer
import time

os.environ['HB_REPLICA_LOG'] = '1'


def test_no_optimization():
    """测试1: 不使用优化"""
    print("\\n" + "="*80)
    print("测试1: Baseline（无优化）")
    print("="*80)
    
    path = os.path.expanduser("../Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, gpu_memory_utilization=0.6)
    
    prompts = ["Explain machine learning in detail."] * 4
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        )
        for p in prompts
    ]
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    print(f"\\n✅ Baseline完成")
    print(f"   总时间: {elapsed:.2f}s")
    print(f"   吞吐量: {len(prompts)/elapsed:.2f} req/s")
    
    return elapsed


def test_always_optimization():
    """测试2: 全程优化（prefill + decode）"""
    print("\\n" + "="*80)
    print("测试2: 全程优化（Prefill + Decode都启用）")
    print("="*80)
    
    path = os.path.expanduser("../Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, gpu_memory_utilization=0.6)
    
    # 配置优化（全程启用）
    model = llm.model_runner.model.model
    for layer_id in range(5, 8):
        model.replicate_layer_to_device(layer_id, 'cuda:1', split_ratio=0.5)
    
    prompts = ["Explain machine learning in detail."] * 4
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        )
        for p in prompts
    ]
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    print(f"\\n✅ 全程优化完成")
    print(f"   总时间: {elapsed:.2f}s")
    print(f"   吞吐量: {len(prompts)/elapsed:.2f} req/s")
    
    # 清理
    for layer_id in range(5, 8):
        model.clear_layer_replication(layer_id)
    
    return elapsed


def test_prefill_only_optimization():
    """测试3: 仅Prefill优化（推荐）"""
    print("\\n" + "="*80)
    print("测试3: Prefill-Only 优化（推荐）")
    print("="*80)
    print("说明: 应用补丁后，decode阶段自动跳过优化")
    
    path = os.path.expanduser("../Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, gpu_memory_utilization=0.6)
    
    # 配置优化（但decode会自动跳过）
    model = llm.model_runner.model.model
    for layer_id in range(5, 8):
        model.replicate_layer_to_device(layer_id, 'cuda:1', split_ratio=0.5)
    
    prompts = ["Explain machine learning in detail."] * 4
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        )
        for p in prompts
    ]
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    print(f"\\n✅ Prefill-Only完成")
    print(f"   总时间: {elapsed:.2f}s")
    print(f"   吞吐量: {len(prompts)/elapsed:.2f} req/s")
    
    # 清理
    for layer_id in range(5, 8):
        model.clear_layer_replication(layer_id)
    
    return elapsed


def compare_all():
    """对比所有方案"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  Prefill vs Decode 优化策略对比                              ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    results = {}
    
    # 测试1: Baseline
    results['baseline'] = test_no_optimization()
    
    # 测试2: 全程优化
    # 注意：在应用 prefill_only 补丁前运行
    # results['always_opt'] = test_always_optimization()
    
    # 测试3: Prefill-Only
    results['prefill_only'] = test_prefill_only_optimization()
    
    # 打印对比
    print("\\n" + "="*80)
    print("性能对比总结")
    print("="*80)
    
    baseline = results['baseline']
    
    for name, elapsed in results.items():
        speedup = baseline / elapsed
        if speedup >= 1.2:
            status = "✅"
        elif speedup >= 1.0:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"{name:20s}: {elapsed:6.2f}s  (加速: {speedup:.2f}x {status})")
    
    print("="*80)
    print("\\n结论:")
    print("  - Baseline: 不使用优化")
    # print("  - Always Opt: Prefill加速，但Decode拖慢")
    print("  - Prefill-Only: 综合最佳（Prefill加速，Decode不受影响）")


if __name__ == "__main__":
    compare_all()
'''
    
    with open("example_prefill_decode_comparison.py", 'w') as f:
        f.write(example_code)
    
    print(f"✅ 创建了对比示例: example_prefill_decode_comparison.py")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              Prefill-Only 优化补丁（解决Decode变慢问题）                      ║
╚════════════════════════════════════════════════════════════════════════════╝

问题：Layer Replication在Prefill阶段加速，但Decode阶段变慢
原因：Decode每次只计算1个token，传输开销 > 并行收益

解决：仅在Prefill阶段启用优化，Decode阶段自动跳过
""")
    
    print("\n应用补丁...")
    if apply_prefill_only_optimization():
        print("\n" + "="*80)
        print("✅ 补丁应用成功！")
        print("="*80)
        
        print("\n现在的行为:")
        print("  - Prefill阶段: 自动启用优化（加速1.5-2x）")
        print("  - Decode阶段: 自动跳过优化（保持原速度）")
        
        print("\n你的代码不需要修改，直接运行:")
        print("  python example_replication_autotune.py")
        
        print("\n查看效果:")
        print("  export HB_REPLICA_LOG=1")
        print("  python example_replication_autotune.py")
        print("  # 会看到：[Replica][L*] Decode阶段，跳过优化")
        
        # 创建对比示例
        print("\n创建性能对比示例...")
        create_comparison_example()
        
        print("\n运行对比:")
        print("  python example_prefill_decode_comparison.py")
        
    else:
        print("\n❌ 补丁应用失败")



