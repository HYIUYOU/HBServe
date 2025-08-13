from collections import deque

from hbserve.config import Config
from hbserve.engine.sequence import Sequence, SequenceStatus
from hbserve.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self): # waiting和running全为空才结束scheduler
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 注意这里是按batch调度，也就是prefill 是bs > 1的
        while self.waiting and num_seqs < self.max_num_seqs: # 如果waiting队列不为空，且当前序列数小于最大序列数
            seq = self.waiting[0] # 取出waiting队列的第一个序列（和后面的popleft())配合
            # 如果当前序列数加上新序列长度大于最大序列数，或者block_manager不能分配，则跳出循环
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break # 跳出循环进入decode调度
            num_seqs += 1 # 当前序列数加1
            self.block_manager.allocate(seq) # 给当前的prefill seq 分配block，并且进行prefix caching优化
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq) # 等待decode调度
            scheduled_seqs.append(seq) # 从prefill等待队列中取出进行prefill
        if scheduled_seqs:
            return scheduled_seqs, True # 返回的是batch

        # decode
        # 注意只要当前有需要进行prefill的，都要优先进行prefill，decode只能等prefill全部结束才能被调度
        # 或者当前的prefill已经满了，无法被调度，才会进行decode调度
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft() # 取出running队列的第一个序列
            # 如果当前的block_manager不能append，则需要preempt
            while not self.block_manager.can_append(seq):
                if self.running: # 如果有等待的，则驱逐最后一个，重新进行prefill
                    self.preempt(self.running.pop()) 
                else: # 这种情况说明无法进行decode，需要将当前的seq调度到prefill重新计算
                    self.preempt(seq) # 驱逐自己，删除自己的block，重新进行prefill
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq) # 重新进行prefill

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
