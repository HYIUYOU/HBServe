import os
import argparse
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer
import warnings
import torch
os.environ['PYTHONUNBUFFERED'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
torch._dynamo.config.cache_size_limit = 64  # 增加到64

def main():
    parser = argparse.ArgumentParser(description='Run LLM inference with configurable parameters')
    parser.add_argument('--max_samples', type=int, default=10, 
                        help='Maximum number of samples to load (default: 30)')
    parser.add_argument('--dp_start_layer', type=int, default=10,
                        help='Start layer for local DP (inclusive, default: 10)')
    parser.add_argument('--dp_end_layer', type=int, default=20,
                        help='End layer for local DP (exclusive, default: 20)')
    parser.add_argument('--dp_devices', type=int, nargs='+', default=[1],
                        help='Target GPU device IDs for DP replicas (e.g., 1 2 3)')

    args = parser.parse_args()

    Qwen_8B = "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    Qwen_32B = "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-32B/snapshots/ba1f828c09458ab0ae83d42eaacc2cf8720c7957"

    path = Qwen_32B
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        # Local DP（连续层副本）
        local_dp_start_layer=args.dp_start_layer,
        local_dp_end_layer=args.dp_end_layer,
        local_dp_devices=args.dp_devices,  # 核心修改：传入选修列表
        use_cuda_graph=True
    )
 

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "describe the benefits of model parallelism",
        
    ]*100
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
    import time
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    latency  = time.time() - t1
    print("latency: ",latency , " throughput: ", len(prompts) /(latency) )

    # for prompt, output in zip(prompts, outputs):
    #     print("\n")
    #     print(f"Prompt: {prompt!r}")
    #     print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
