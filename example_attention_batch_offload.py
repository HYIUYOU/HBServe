import os
import torch
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ['HB_ATTN_OFFLOAD_LOG'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步执行，便于调试
os.environ['HB_DEBUG'] = '1'  # 启用 attention 调试日志

def main():
    model_path = os.path.expanduser("../Qwen3-0.6B")
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("❌ 需要至少 2 张 GPU")
        return
    
    print("="*60)
    print("  简化的 Attention Offload 测试")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6
    )
    
    model = llm.model_runner.model.model
    
    # 配置 Attention Offload
    print("\n配置 Attention Offload...")
    model.attention_offload_by_batch(
        layer_id=9,
        offload_device='cuda:1',
        split_ratio=0.5,
        enable_autotune=False  # 先禁用 autotune
    )
    
    # 简单测试：单个短 prompt
    print("\n测试 1: 单个短 prompt")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    prompts = ["Hello"]
    
    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        print(f"✅ 测试 1 通过: {outputs[0]['text'][:50]}")
    except Exception as e:
        print(f"❌ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        model.clear_attention_offload(9)
        return
    
    # 测试 2: 多个 prompt
    print("\n测试 2: 多个 prompts")
    prompts = ["Hello", "Hi", "Hey"]
    
    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        print(f"✅ 测试 2 通过")
        for i, output in enumerate(outputs):
            print(f"   [{i}] {output['text'][:30]}")
    except Exception as e:
        print(f"❌ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
    
    model.clear_attention_offload(9)
    print("\n测试完成")

if __name__ == "__main__":
    main()