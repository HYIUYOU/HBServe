import os
import torch
from torch import nn
import triton
import triton.language as tl
import torch.cuda.nvtx as nvtx

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from HBserve.utils.context import get_context

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
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) # 表示当前的线程id
    
    # 使用 BLOCK_SIZE 代替 D，并添加 mask 来处理非 2 的幂次方的 D
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0) # 一次加载D个元素
    
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + offsets # 定位对于的KV应该存在哪个位置
    
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask) # 一次存D个元素


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
    
    # 计算最接近 D 的 2 的幂次方（向上取整）
    # 例如：D=2560 -> BLOCK_SIZE=4096
    import math
    BLOCK_SIZE = 2 ** math.ceil(math.log2(D))
    
    # Triton Grid配置的语法
    # 1. 1D grid  ==> Kernel[(N,)](args)  表示N个线程并行
    # 2. 2D grid  ==> Kernel[(M,N)](args)  表示M*N个线程并行
    # 3. 3D grid  ==> Kernel[(P,M,N)](args)  表示P*M*N个线程并行
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), 
                                k_cache, v_cache, slot_mapping, D, BLOCK_SIZE)


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
        debug_flag = os.environ.get("HB_DEBUG", "0") != "0"
        flash_debug_flag = os.environ.get("HB_FLASH_LOG", "0") != "0"
        DEBUG = debug_flag or flash_debug_flag
        def log(msg):
            if DEBUG:
                print(f"[HB-Debug][Attention] {msg}")
        def flash_log(msg):
            if flash_debug_flag:
                print(f"[HB-Flash][Attention] {msg}")
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        # 统一上下文与缓存到当前计算设备，避免Triton/Flash-Attn跨设备访问
        dev = q.device
        # 确保当前CUDA上下文与张量所在设备一致（Triton/FlashAttn依赖当前device）
        torch.cuda.set_device(dev)
        log(f"set current device -> {dev}")
        log(f"q.dev={q.device} k.dev={k.device} v.dev={v.device} q.is_cuda={q.is_cuda}")
        if context.slot_mapping is not None and context.slot_mapping.device != dev:
            context.slot_mapping = context.slot_mapping.to(dev, non_blocking=True)
        if context.block_tables is not None and context.block_tables.device != dev:
            context.block_tables = context.block_tables.to(dev, non_blocking=True)
        if hasattr(context, "cu_seqlens_q") and context.cu_seqlens_q is not None and context.cu_seqlens_q.device != dev:
            context.cu_seqlens_q = context.cu_seqlens_q.to(dev, non_blocking=True)
        if hasattr(context, "cu_seqlens_k") and context.cu_seqlens_k is not None and context.cu_seqlens_k.device != dev:
            context.cu_seqlens_k = context.cu_seqlens_k.to(dev, non_blocking=True)
        if hasattr(context, "context_lens") and context.context_lens is not None and context.context_lens.device != dev:
            context.context_lens = context.context_lens.to(dev, non_blocking=True)

        k_cache, v_cache = self.k_cache, self.v_cache  # 拿到KV Cache
        k_cache_dev = k_cache
        v_cache_dev = v_cache
        migrated = False
        if k_cache.numel() and k_cache.device != dev:
            k_cache_dev = k_cache.to(dev, non_blocking=True)
            migrated = True
            flash_log(f"migrate k_cache from {k_cache.device} to {dev}")
        if v_cache.numel() and v_cache.device != dev:
            v_cache_dev = v_cache.to(dev, non_blocking=True)
            migrated = True
            flash_log(f"migrate v_cache from {v_cache.device} to {dev}")
        # 保证传入triton/flash-attn的数据在同一设备且连续
        if context.slot_mapping is not None:
            context.slot_mapping = context.slot_mapping.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if k_cache_dev.numel() and not k_cache_dev.is_contiguous():
            k_cache_dev = k_cache_dev.contiguous()
            migrated = True
        if v_cache_dev.numel() and not v_cache_dev.is_contiguous():
            v_cache_dev = v_cache_dev.contiguous()
            migrated = True
        if migrated:
            # 将 cache 固定在当前执行设备，避免后续重复跨卡复制
            if k_cache_dev.numel():
                self.k_cache = k_cache_dev.detach()
            if v_cache_dev.numel():
                self.v_cache = v_cache_dev.detach()
        if DEBUG:
            log(f"slot_mapping.dev={getattr(context.slot_mapping,'device',None)} is_cuda={getattr(context.slot_mapping,'is_cuda',False)}")
            log(f"k_cache.dev={k_cache_dev.device if k_cache_dev.numel() else None} v_cache.dev={v_cache_dev.device if v_cache_dev.numel() else None}")
            log(f"contig q={q.is_contiguous()} k={k.is_contiguous()} v={v.is_contiguous()}")

        # 额外保障：所有传入triton的张量必须是cuda张量
        if k_cache_dev.numel() and v_cache_dev.numel() and context.slot_mapping is not None:
            if not (k.is_cuda and v.is_cuda and k_cache_dev.is_cuda and v_cache_dev.is_cuda and context.slot_mapping.is_cuda):
                k = k.to(dev, non_blocking=True)
                v = v.to(dev, non_blocking=True)
                k_cache_dev = k_cache_dev.to(dev, non_blocking=True)
                v_cache_dev = v_cache_dev.to(dev, non_blocking=True)
                context.slot_mapping = context.slot_mapping.to(dev, non_blocking=True)
            log("call store_kvcache")
            store_kvcache(k, v, k_cache_dev, v_cache_dev, context.slot_mapping) # 将新计算得到的KV存储到KV Cache中
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache_dev, v_cache_dev # 使用KV Cache（确保与q同设备）
            # q => 新的token，不包含prefix caching
            # k,v => 包含prefix caching的KV
            flash_log(
                "call flash_attn_varlen_func "
                f"layer={getattr(context, 'layer_id', getattr(context, '_layer_id', 'unknown'))} "
                f"q_dev={q.device} k_dev={k.device} v_dev={v.device}"
            )
            # 确保可变长输入相关张量连续
            if hasattr(context, "cu_seqlens_q") and context.cu_seqlens_q is not None:
                context.cu_seqlens_q = context.cu_seqlens_q.contiguous()
            if hasattr(context, "cu_seqlens_k") and context.cu_seqlens_k is not None:
                context.cu_seqlens_k = context.cu_seqlens_k.contiguous()
            if context.block_tables is not None:
                context.block_tables = context.block_tables.contiguous()
            nvtx_range_name = (
                f"flash_attn_varlen[layer={getattr(context, 'layer_id', getattr(context, '_layer_id', 'unknown'))}]"
                if flash_debug_flag else None
            )
            if nvtx_range_name:
                nvtx.range_push(nvtx_range_name)
            try:
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            finally:
                if nvtx_range_name:
                    nvtx.range_pop()
        else:    # decode
            # TODO：check q，k，v shape
            flash_log(
                "call flash_attn_with_kvcache "
                f"layer={getattr(context, 'layer_id', getattr(context, '_layer_id', 'unknown'))} "
                f"q_dev={q.device} k_cache_dev={k_cache_dev.device if k_cache_dev.numel() else None}"
            )
            nvtx_range_name = (
                f"flash_attn_kvcache[layer={getattr(context, 'layer_id', getattr(context, '_layer_id', 'unknown'))}]"
                if flash_debug_flag else None
            )
            if nvtx_range_name:
                nvtx.range_push(nvtx_range_name)
            try:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache_dev,
                    v_cache_dev,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
            finally:
                if nvtx_range_name:
                    nvtx.range_pop()
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
