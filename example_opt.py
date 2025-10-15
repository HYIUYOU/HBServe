# example_opt.py

import os
from transformers import AutoTokenizer
from HBserve import LLM, SamplingParams

def main():
    path = os.path.expanduser("/home/admin/workspace/aop_lab/app_data/.cache/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.9)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # OPT 使用普通文本提示
    prompts = [
        "Once upon a time",
        "The capital of France is",
        "In a galaxy far far away",
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    # ========== 正确的访问方式 ==========
    for i, output in enumerate(outputs):
        prompt = prompts[i]  # ← 从原始 prompts 列表获取
        generated_text = output['text']  # ← 从 output 字典获取生成的文本
        token_ids = output['token_ids']  # ← token IDs（可选）
        
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print(f"Tokens: {len(token_ids)}")
        print("-" * 60)

if __name__ == "__main__":
    main()