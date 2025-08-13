import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from hbserve.utils.context import get_context

# key.shape = [N,num_heads,head_dim] ==> key.stride(0) = num_heads * head_dim == D
# store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), 
#                                 k_cache, v_cache, slot_mapping, D)

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride, # key.stride(0) = num_heads * head_dim == D 一个key的size
    value_ptr, # 新计算得到的V
    value_stride, # value.stride(0) = num_heads * head_dim == D
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0) # 表示当前的线程id
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets) # 一次加载D个元素
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D) # 定位对于的KV应该存在哪个位置
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value) # 一次存D个元素


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape # N表示需要存储的token的数量，也就是slot_mapping的长度
    D = num_heads * head_dim # D 表示每个K，V的大小，因为K 的形状是[N,num_heads,head_dim]
    # tensor.stride() 表示tensor的stride，也就是tensor的每个维度之间的距离
    # 比如key的形状是[N,num_heads,head_dim]，那么key.stride(-1) = 1，表示每个token的head_dim之间的距离为1
    # key.stride(1) = head_dim，表示每个token的num_heads之间的距离为head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N # slot_mapping是一个list，表示每个token应该存储在KV Cache 的哪个位置
    
    # Triton Grid配置的语法
    # 1. 1D grid  ==> Kernel[(N,)](args)  表示N个线程并行
    # 2. 2D grid  ==> Kernel[(M,N)](args)  表示M*N个线程并行
    # 3. 3D grid  ==> Kernel[(P,M,N)](args)  表示P*M*N个线程并行
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), 
                                k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim) # 新计算得到的Q
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim) # 新计算得到的KV
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache # 拿到KV Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping) # 将新计算得到的KV存储到KV Cache中
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache # 使用KV Cache
            # q => 新的token，不包含prefix caching
            # k,v => 包含prefix caching的KV
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables) 
        else:    # decode
            # TODO：check q，k，v shape
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
