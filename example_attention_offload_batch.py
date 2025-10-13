import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer
os.environ['HB_ATTN_OFFLOAD_LOG'] = '0'
os.environ['HB_DEBUG'] = '0'  # 不启用 attention 调试日志

def main():
    path = os.path.expanduser("../Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.6)
    
    # === 配置 attention batch offload ===
    model = llm.model_runner.model.model
    print("配置层 9 使用 attention batch offload...")
    model.attention_offload_by_batch(
        layer_id=9,
        offload_device='cuda:1',
        split_ratio=0.5,
        enable_autotune=False  # 先禁用 autotune
    )
    print("✓attention batch offload 已启用\n")
    
    # === 推理 ===
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    
    # === 清理 ===
    model.clear_attention_offload(9)
    print("\n✓ 已清除 attention batch offload 配置")

    model.clear_attention_offload(9)
    print("\n测试完成")

if __name__ == "__main__":
    main()