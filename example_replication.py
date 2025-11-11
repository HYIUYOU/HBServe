# HB_REPLICA_LOG = 1 python example_replication.py
import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from HBserve.utils.loader import load_longbench_prompts
os.environ['HB_ATTN_OFFLOAD_LOG'] = '0'
os.environ['HB_DEBUG'] = '0'  # 不启用 attention 调试日志
os.environ['HB_REPLICA_LOG'] = '1'
os.environ['HB_FLASH_LOG'] = '1'

def main():
    # 与 example.py 保持一致的加载方式
    path = os.path.expanduser("/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-32B/snapshots/ba1f828c09458ab0ae83d42eaacc2cf8720c7957")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1,gpu_memory_utilization=0.9)

    # 配置第10层复制到GPU1，并开启自适应（需要>=2张GPU）
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            # 访问底层 Qwen3Model
            if hasattr(llm, 'model_runner') and hasattr(llm.model_runner, 'model') and hasattr(llm.model_runner.model, 'model'):
                model = llm.model_runner.model.model
                # 第10层索引为 9
                for i in range(10,20):
                    model.replicate_layer_to_device(i, 'cuda:1', split_ratio=0.5)
                   
                
        else:
            print("警告: 需要至少2张GPU来演示复制并行，自适应功能将跳过。")
    except Exception as e:
        print(f"配置复制/自适应失败: {e}")

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
    jsonl_file = "/home/admin/workspace/aop_lab/app_source/data/longbench/2wikimqa_e.jsonl"
    prompts = load_longbench_prompts(jsonl_file, max_samples=10)  
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
    print("latency: ", time.time() - t1, " throughput: ", len(prompts) /(time.time() - t1) )
    # for prompt, output in zip(prompts, outputs):
    #     print("\n")
    #     print(f"Prompt: {prompt!r}")
    #     print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
