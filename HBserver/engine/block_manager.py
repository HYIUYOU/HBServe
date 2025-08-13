from collections import deque
import xxhash
import numpy as np

from hbserve.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1 # 被使用一次
        self.hash = -1 # 没有prefix
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 拥有所有的block
        self.hash_to_block_id: dict[int, int] = dict() # 用于存储hash值到block_id的映射
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 用于存储空闲的block_id
        self.used_block_ids: set[int] = set() # 用于存储被使用的block_id

    # 这里计算的是当前的block中的token_ids的hash值
    # 比如当前的block内容为 [1,2,3,4]   
    # 这里会把[1,2,3,4]转换为bytes，然后更新hash
    # 也就是一个list 得到一个hash 的值
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes()) # 计算在这个list 二进制化后的hash值
        return h.intdigest() # 返回一个int 的hash值

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id] # 获取block
        assert block.ref_count == 0 # 确保block的引用计数为0，也就是没有被使用
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id) # 在这里表示被使用了，用于prefix caching 的判断
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks
    """
    这里的hash计算需要仔细理解! 这个链式hash用来实现Prefix Caching!

    举个例子：

    序列1 =  [ [1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    序列2 =  [ [1,2,3],[4,5,6],[10,12,13],[20,21,22]]

    序列1的hash值为：
    comput_hash([1,2,3],-1) = x1   通过 hash_to_block_id 对应 block 0
    comput_hash([4,5,6],x1) = x2  对应 block 1
    comput_hash([7,8,9],x2) = x3  对应 block 2
    comput_hash([10,11,12],x3) = x4  对应 block 3

    序列2的hash值为：
    comput_hash([1,2,3],-1) = x1  对应 block 0
    comput_hash([4,5,6],x1) = x2  对应 block 1
    comput_hash([10,12,13],x2) = x5  对应 block 6
    comput_hash([20,21,22],x5) = x6  对应 block 7

    序列2的prefix caching 为  [block0,block1]

    """
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i) #得到block i 中存的 token_ids,这里是一个list
            # 如果当前的block是满的，则计算hash，否则hash为-1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 如果hash值在hash_to_block_id中存在，则获取对应的block_id，否则为-1
            # 如果找到了当前的token_ids的block，并不能说明prefix caching命中了
            # 只能说明当前的block在之前出现过，还有没有命中，需要判断
            block_id = self.hash_to_block_id.get(h, -1)
            # 如果block_id为-1，或者block_id对应的block的token_ids与当前block的token_ids不同，
            # 则表示cache miss，需要重新分配block
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0] # 如果cache miss，则分配一个空闲的block
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                # 如果block_id在used_block_ids中，则表示block被使用过，则增加引用计数
                # 否则说明当前的block被deallocate过，则需要重新分配一个空闲的block
                if block_id in self.used_block_ids: #used_block_ids 会在 _allocate_block 和 _deallocate_block 中被更新
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        # 倒着删除主要是为了更好的内存管理和缓存局部性
        # 如果从前往后删除，则会导致缓存失效，因为删除的block可能被其他序列使用
        for block_id in reversed(seq.block_table): 
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # 注意这个只会在decode阶段使用，当len(seq) % self.block_size == 1时，
        # 说明此前是满的，也就是需要重新分配一个block
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1 # 填满的block的hash应该已经计算过了
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id) # 新增一个block
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0: # 刚好填满
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1) # 获取最后一个block的token_ids
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix) # 使用前一个block的hash更新当前的block的hash
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1 # 没填满的block的hash不应该被计算了
