# HB_REPLICA_LOG = 1 python example_replication_autotune.py
import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
os.environ['HB_ATTN_OFFLOAD_LOG'] = '0'
os.environ['HB_DEBUG'] = '0'  # 不启用 attention 调试日志
os.environ['HB_REPLICA_LOG'] = '0'

def main():
    # 与 example.py 保持一致的加载方式
    path = os.path.expanduser("/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1,gpu_memory_utilization=0.6)

    # 配置第10层复制到GPU1，并开启自适应（需要>=2张GPU）
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            # 访问底层 Qwen3Model
            if hasattr(llm, 'model_runner') and hasattr(llm.model_runner, 'model') and hasattr(llm.model_runner.model, 'model'):
                model = llm.model_runner.model.model
                # 单层移动
                # model.move_layer_to_device(9, 'cuda:1')

                # 批量分配
                model.set_layer_device_distribution({
                    9: 'cuda:1',
                    10: 'cuda:1',
                    11: 'cuda:1'
                })
                
        else:
            print("警告: 需要至少2张GPU来演示复制并行，自适应功能将跳过。")
    except Exception as e:
        print(f"配置复制/自适应失败: {e}")

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "describe the benefits of model parallelism",
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


if __name__ == "__main__":
    main()
