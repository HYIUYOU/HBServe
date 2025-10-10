import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer
import torch


def main():
    # 与 example.py 保持一致的加载方式
    path = os.path.expanduser("../Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 配置第10层复制到GPU1，并开启自适应（需要>=2张GPU）
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            # 访问底层 Qwen3Model
            if hasattr(llm, 'model_runner') and hasattr(llm.model_runner, 'model') and hasattr(llm.model_runner.model, 'model'):
                model = llm.model_runner.model.model
                # 第10层索引为 9
                model.replicate_layer_to_device(9, 'cuda:1', split_ratio=0.5)
                model.enable_replication_autotune(9, beta=0.3, min_ratio=0.2, max_ratio=0.8)
                # 可选日志：启用自适应日志打印
                os.environ.setdefault("HB_REPLICA_LOG", "1")
        else:
            print("警告: 需要至少2张GPU来演示复制并行，自适应功能将跳过。")
    except Exception as e:
        print(f"配置复制/自适应失败: {e}")

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
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
