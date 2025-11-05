"""
测试 KV Head Split 修复
验证 OPT 模型的 KV Head Split 功能是否正常工作
"""

import os
os.environ["HB_DEBUG"] = "1"

from transformers import AutoTokenizer
from HBserve import LLM, SamplingParams


def main():
    print("=" * 80)
    print("测试 KV Head Split 修复")
    print("=" * 80)
    
    # 使用 OPT-13B 模型
    path = "/root/llm-resource/Models/opt-13b"
    
    print(f"\n1. 加载模型: {path}")
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    
    print("\n2. 配置 KV Head Split...")
    # 在多个层上启用 KV Head Split
    for layer_id in [10, 11, 12]:
        print(f"   配置层 {layer_id}...")
        llm.model_runner.model.model.attention_offload_by_kv_head(
            layer_id=layer_id,
            offload_device="cuda:1",
            split_kv_head_idx=None  # 默认从中间切分
        )
    
    print("\n3. 准备测试输入...")
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
    prompts = [
        "The capital of France is",
        "Python is a programming language that",
    ]
    
    print("\n4. 开始推理...")
    print("-" * 80)
    
    try:
        outputs = llm.generate(prompts, sampling_params)
        
        print("\n✓ 推理成功完成！")
        print("=" * 80)
        print("生成结果:")
        print("=" * 80)
        
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"\n[{i+1}] Prompt: {prompt}")
            print(f"    Output: {output['text']}")
        
        print("\n" + "=" * 80)
        print("✓ KV Head Split 功能正常工作！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

