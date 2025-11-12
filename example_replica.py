import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-32B/snapshots/ba1f828c09458ab0ae83d42eaacc2cf8720c7957"
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        # Local DP（连续层副本）
        local_dp_start_layer=9,    # 第10层（含）
        local_dp_end_layer=20,     # 第20层（不含）
        local_dp_device=1          # 复制到 GPU1
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "describe the benefits of model parallelism",
        # "introduce yourself",
        # "list all prime numbers within 100",
        # "introduce yourself",
        # "list all prime numbers within 100",
        # "introduce yourself",
        # "list all prime numbers within 100",
        # "introduce yourself",
        # "list all prime numbers within 100",
        # "introduce yourself",
        # "list all prime numbers within 100",
        # "introduce yourself",
        # "list all prime numbers within 100",
        # "introduce yourself",
        # "list all prime numbers within 100",
    ]
    from HBserve.utils.loader import load_longbench_prompts
    jsonl_file = "/home/admin/workspace/aop_lab/app_source/data/longbench/2wikimqa_e.jsonl"
    prompts = load_longbench_prompts(jsonl_file, max_samples=10)  
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


if __name__ == "__main__":
    main()
