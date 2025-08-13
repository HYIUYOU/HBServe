import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from hbserve.config import Config
from hbserve.sampling_params import SamplingParams
from hbserve.engine.sequence import Sequence
from hbserve.engine.scheduler import Scheduler
from hbserve.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn") # spawn 模式，确保子进程完全独立，避免CUDA context 冲突
        for i in range(1, config.tensor_parallel_size): # 创建多个进程，每个进程负责一个GPU
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event)) # rank1,2,3...tp_size-1
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) # rank0,主进程
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config) # 创建调度器（全局只有一个）
        # 调度器中包含：waiting队列（prefill），running队列（decode），block_manager
        # max_num_batched_tokens，max_num_seqs，eos
        atexit.register(self.exit) # 注册退出函数，主要是释放资源和退出model runner

    def exit(self):
        self.model_runner.call("exit") # 调用model_runner中的exit方法
        del self.model_runner # 删除自己的model_runner
        for p in self.ps:
            p.join() # 等待所有子进程退出

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt) # 将prompt转换为token_ids
        seq = Sequence(prompt, sampling_params) # 创建序列,(block table,num_cached_tokens)
        self.scheduler.add(seq) # 加入到waiting队列等待prefill调度

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list): # 如果sampling_params不是列表，则将其扩展为与prompts长度相同的列表
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params): # 将prompts和sampling_params一一对应
            self.add_request(prompt, sp) # 添加请求到waiting队列等待prefill
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
