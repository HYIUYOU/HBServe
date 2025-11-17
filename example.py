import os
import argparse
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Run LLM inference with configurable parameters')
    parser.add_argument('--max_samples', type=int, default=7, 
                        help='Maximum number of samples to load (default: 15)')
    args = parser.parse_args()
    
    path = os.path.expanduser("/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-32B/snapshots/ba1f828c09458ab0ae83d42eaacc2cf8720c7957")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.9)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "describe the benefits of model parallelism",
        
    ]*200
    from HBserve.utils.loader import load_longbench_prompts
    jsonl_file = "/home/admin/workspace/aop_lab/app_source/data/longbench/2wikimqa_e.jsonl"
    prompts = load_longbench_prompts(jsonl_file, max_samples=args.max_samples, tokenizer = tokenizer, max_length = 1024)  
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    # print("prompts:",prompts)
    import time
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    latency = time.time() - t1
    throughput = len(prompts) / latency
    print(f"latency: {latency}s, throughput: {throughput} requests/s")

    # for prompt, output in zip(prompts, outputs):
    #     print("\n")
    #     print(f"Prompt: {prompt!r}")
    #     print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
