import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.6)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "describe the benefits of model parallelism",
        "introduce yourself",
        "list all prime numbers within 100",
        "introduce yourself",
        "list all prime numbers within 100",
        "introduce yourself",
        "list all prime numbers within 100",
        "introduce yourself",
        "list all prime numbers within 100",
        "introduce yourself",
        "list all prime numbers within 100",
        "introduce yourself",
        "list all prime numbers within 100",
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
