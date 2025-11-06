"""
调试 KV Head Split 的脚本
帮助定位 CUDA device-side assert 的原因
"""

import torch
import os

# 设置环境变量以获得更详细的错误信息
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["HB_DEBUG"] = "1"

from transformers import AutoTokenizer, AutoConfig
from HBserve import LLM, SamplingParams


def main():
    # 使用 OPT-13B 模型
    path = "/root/llm-resource/Models/opt-13b"
    
    # 加载配置和tokenizer
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    
    print("=" * 80)
    print("配置信息:")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print("=" * 80)
    
    # 初始化模型
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    
    # 配置 KV Head Split - 只在第10层测试
    print("\n配置 KV Head Split (只在第10层)...")
    llm.model_runner.model.model.attention_offload_by_kv_head(
        layer_id=10,
        offload_device="cuda:1",
        split_kv_head_idx=None  # 默认从中间切分
    )
    print("✓ KV Head Split 已启用\n")
    
    # 准备测试数据 - 使用非常简单的输入
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    prompts = ["Hello, how are you?"]
    
    print("开始推理...")
    print(f"输入: {prompts[0]}")
    print("=" * 80)
    
    try:
        outputs = llm.generate(prompts, sampling_params)
        print("\n✓ 推理成功!")
        print(f"输出: {outputs[0]['text']}")
    except RuntimeError as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

