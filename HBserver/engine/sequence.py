from copy import copy
from enum import Enum, auto
from itertools import count

from hbserve.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256 # 一个block中有多少个token
    counter = count() # 计数器，用于生成序列ID

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter) # 生成序列ID
        self.status = SequenceStatus.WAITING # 最开始为WAITING状态，需要进行prefill
        self.token_ids = copy(token_ids) # 这里需要复制，这是因为在推理系统中，同一个prompt可能被用于创建多个不同的序列（比如beam search或多个采样），每个序列都应该有自己独立的token列表
        self.last_token = token_ids[-1] # 最后一个token
        self.num_tokens = len(self.token_ids) # 总token数
        self.num_prompt_tokens = len(token_ids) # 提示token数
        self.num_cached_tokens = 0 #
        self.block_table = [] # block_table 注意用于paged attention
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self): # 这里是计算有多少个block，注意这里是向上取整
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i): # 获取第i个block的token_ids,这个是一个list
        assert 0 <= i < self.num_blocks # 由于计算block数量是向上取整，如果是最后一个没有填满
        return self.token_ids[i*self.block_size: (i+1)*self.block_size] # Python自动截断到实际长度，不会出错

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
