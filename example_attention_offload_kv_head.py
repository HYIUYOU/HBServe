import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer
os.environ['HB_ATTN_OFFLOAD_LOG'] = '0'
os.environ['HB_DEBUG'] = '0'  # 不启用 attention 调试日志

def main():
    path = os.path.expanduser("/home/admin/workspace/aop_lab/app_data/.cache/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    
    # === 配置 KV Head Split ===
    model = llm.model_runner.model.model
    for i in range(10,16):
        print("配置层,", i ,"使用 KV Head Split...")
        model.attention_offload_by_kv_head(
            layer_id=i,
            offload_device='cuda:1',
            split_kv_head_idx=None,  # None = 均分
        )
    print("✓ KV Head Split 已启用\n")
    
    # === 推理 ===
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    from HBserve.utils.loader import load_longbench_prompts
    jsonl_file = "/home/admin/workspace/aop_lab/app_source/data/longbench/2wikimqa_e.jsonl"
    prompts = load_longbench_prompts(jsonl_file, max_samples=50) 
    # prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #         enable_thinking=True
    #     )
    #     for prompt in prompts
    # ]
    import time
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print("latency: ", time.time() - t1, " throughput: ", len(prompts) /(time.time() - t1) )

    # for prompt, output in zip(prompts, outputs):
    #     print("\n")
    #     print(f"Prompt: {prompt!r}")
    #     print(f"Completion: {output['text']!r}")
    
    # === 清理 ===
    model.clear_attention_offload(9)
    print("\n✓ 已清除 KV Head Split 配置")


if __name__ == "__main__":
    main()