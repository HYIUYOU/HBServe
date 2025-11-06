"""
优化前向传播的执行逻辑
将复杂的执行逻辑从模型类中分离出来
"""

import torch
import os
from torch import nn
from typing import Dict, Optional, Callable
from HBserve.utils.context import get_context, set_context, Context




# ============================================================================
# NVLink优化：增量KV Cache同步
# ============================================================================

class _NVLinkKVCacheSync:
    """利用NVLink高带宽的增量KV Cache同步"""
    
    def __init__(self):
        self.synced_lengths = {}
    
    def sync_incremental(
        self,
        layer_id: int,
        src_k: torch.Tensor,
        src_v: torch.Tensor,
        dst_k: torch.Tensor,
        dst_v: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        start_batch_idx: int,
        block_size: int = 16
    ):
        """只同步新增的KV blocks，使用non_blocking充分利用NVLink"""
        DEBUG = os.environ.get("HB_NVLINK_LOG", "0") != "0"
        
        for batch_idx in range(start_batch_idx, len(context_lens)):
            current_len = context_lens[batch_idx].item()
            key = (layer_id, batch_idx)
            last_len = self.synced_lengths.get(key, 0)
            
            if current_len <= last_len:
                continue
            
            start_block = last_len // block_size
            end_block = (current_len + block_size - 1) // block_size
            blocks = block_tables[batch_idx]
            
            synced = 0
            for blk_idx in range(start_block, end_block):
                if blk_idx >= len(blocks):
                    break
                phys_blk = blocks[blk_idx].item()
                dst_k[phys_blk].copy_(src_k[phys_blk], non_blocking=True)
                dst_v[phys_blk].copy_(src_v[phys_blk], non_blocking=True)
                synced += 1
            
            self.synced_lengths[key] = current_len
            
            if DEBUG and synced > 0:
                print(f"[NVLink-KVCache][L{layer_id}][B{batch_idx}] "
                      f"同步 {synced}/{end_block} blocks")

# 全局KV Cache管理器
_global_kv_cache_sync = _NVLinkKVCacheSync()


# ============================================================================
# NVLink优化：动态启用检查（更激进的阈值）
# ============================================================================

def _should_enable_nvlink_optimization(hidden_states, context, min_tokens=1024):
    """
    NVLink下的优化启用策略
    
    由于NVLink传输开销极小（<0.5ms），可以在更小的batch上启用优化
    """
    total_tokens = hidden_states.size(0)
    
    if context.is_prefill:
        # Prefill: 1024+ tokens就启用（PCIe需要4096+）
        if total_tokens >= min_tokens:
            return True, f"Prefill，tokens={total_tokens} (NVLink)"
        return False, f"tokens太少 ({total_tokens} < {min_tokens})"
    else:
        # Decode: 8+ batch就启用（PCIe需要32+）
        batch_size = hidden_states.size(0)
        min_batch = 8
        if batch_size >= min_batch:
            return True, f"Decode，batch={batch_size} (NVLink)"
        return False, f"batch太小 ({batch_size} < {min_batch})"


def execute_kv_head_split_forward(
    layer_id: int,
    layer: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    context: Context,
    config: Dict
) -> torch.Tensor:
    """
    执行按 KV Head 切分的 Attention 计算
    
    Args:
        layer_id: 层索引
        layer: 当前层
        positions: 位置张量
        hidden_states: 隐藏状态
        context: 上下文
        config: KV Head Split 配置
    
    Returns:
        Attention 输出
    """
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    from HBserve.layers.attention import store_kvcache
    
    DEBUG = os.environ.get("HB_DEBUG", "0") != "0"
    
    src_device = config['src_device']
    offload_device = config['offload_device']
    split_q_head_idx = config['split_q_head_idx']
    split_kv_head_idx = config['split_kv_head_idx']
    
    # 确保输入在原设备
    if hidden_states.device != src_device:
        hidden_states = hidden_states.to(src_device, non_blocking=True)
    if positions.device != src_device:
        positions = positions.to(src_device, non_blocking=True)
    
    # 处理输入维度
    is_prefill = context.is_prefill
    if hidden_states.dim() == 2:
        batch_size = hidden_states.size(0)
        seq_len = 1
        hidden_size = hidden_states.size(1)
        hidden_states = hidden_states.unsqueeze(1)
    elif hidden_states.dim() == 3:
        batch_size, seq_len, hidden_size = hidden_states.shape
    else:
        raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
    
    # 初始化分片 cache（如果还没初始化的话）
    # 注意：需要在原始 attention 的 cache 已经创建之后才能初始化
    if not config['cache_initialized']:
        src_attn_module = config['src_attn'].attn
        if src_attn_module.k_cache.numel() > 0:
            _init_split_kv_cache(layer_id, config)
    
    # === QKV Projection ===
    qkv_0 = torch.nn.functional.linear(
        hidden_states, config['qkv_weight_0'], config['qkv_bias_0']
    )
    hs_1 = hidden_states.to(offload_device, non_blocking=True)
    qkv_1 = torch.nn.functional.linear(
        hs_1, config['qkv_weight_1'], config['qkv_bias_1']
    )
    
    # === 分离 Q, K, V ===
    num_heads_0 = split_q_head_idx
    num_kv_heads_0 = split_kv_head_idx
    num_heads_1 = config['src_attn'].num_heads - split_q_head_idx
    num_kv_heads_1 = config['num_kv_heads_1']
    head_dim = config['head_dim']
    
    q_size_0 = num_heads_0 * head_dim
    kv_size_0 = num_kv_heads_0 * head_dim
    q_size_1 = num_heads_1 * head_dim
    kv_size_1 = num_kv_heads_1 * head_dim
    
    # Device 0
    q_0, k_0, v_0 = qkv_0.split([q_size_0, kv_size_0, kv_size_0], dim=-1)
    q_0 = q_0.view(batch_size, seq_len, num_heads_0, head_dim)
    k_0 = k_0.view(batch_size, seq_len, num_kv_heads_0, head_dim)
    v_0 = v_0.view(batch_size, seq_len, num_kv_heads_0, head_dim)
    
    # Device 1
    q_1, k_1, v_1 = qkv_1.split([q_size_1, kv_size_1, kv_size_1], dim=-1)
    q_1 = q_1.view(batch_size, seq_len, num_heads_1, head_dim)
    k_1 = k_1.view(batch_size, seq_len, num_kv_heads_1, head_dim)
    v_1 = v_1.view(batch_size, seq_len, num_kv_heads_1, head_dim)
    
    # === RMS Norm ===
    q_norm_weight = config['q_norm_weight'].to(src_device, non_blocking=True)
    k_norm_weight = config['k_norm_weight'].to(src_device, non_blocking=True)
    
    q_0 = torch.nn.functional.rms_norm(q_0, (head_dim,), q_norm_weight, eps=1e-6)
    k_0 = torch.nn.functional.rms_norm(k_0, (head_dim,), k_norm_weight, eps=1e-6)
    
    q_norm_weight_1 = q_norm_weight.to(offload_device, non_blocking=True)
    k_norm_weight_1 = k_norm_weight.to(offload_device, non_blocking=True)
    q_1 = torch.nn.functional.rms_norm(q_1, (head_dim,), q_norm_weight_1, eps=1e-6)
    k_1 = torch.nn.functional.rms_norm(k_1, (head_dim,), k_norm_weight_1, eps=1e-6)
    
    # === RoPE ===
    rotary_emb = config['rotary_emb']
    
    q_0 = q_0.view(batch_size * seq_len, num_heads_0, head_dim)
    k_0 = k_0.view(batch_size * seq_len, num_kv_heads_0, head_dim)
    q_0, k_0 = rotary_emb(positions, q_0, k_0)
    
    positions_1 = positions.to(offload_device, non_blocking=True)
    q_1 = q_1.view(batch_size * seq_len, num_heads_1, head_dim)
    k_1 = k_1.view(batch_size * seq_len, num_kv_heads_1, head_dim)
    q_1, k_1 = rotary_emb(positions_1, q_1, k_1)
    
    v_0 = v_0.view(batch_size * seq_len, num_kv_heads_0, head_dim)
    v_1 = v_1.view(batch_size * seq_len, num_kv_heads_1, head_dim)
    
    # === Attention 计算 ===
    stream_0 = torch.cuda.Stream(device=src_device) if src_device.type == 'cuda' else None
    stream_1 = torch.cuda.Stream(device=offload_device) if offload_device.type == 'cuda' else None
    
    # 并行计算
    if stream_0 is not None:
        with torch.cuda.stream(stream_0):
            o_0 = _compute_split_attention(
                q_0, k_0, v_0,
                config['k_cache_0'], config['v_cache_0'],
                context, src_device, layer_id, True
            )
    else:
        o_0 = _compute_split_attention(
            q_0, k_0, v_0,
            config['k_cache_0'], config['v_cache_0'],
            context, src_device, layer_id, True
        )
    
    if stream_1 is not None:
        with torch.cuda.stream(stream_1):
            o_1 = _compute_split_attention(
                q_1, k_1, v_1,
                config['k_cache_1'], config['v_cache_1'],
                context, offload_device, layer_id, False
            )
    else:
        o_1 = _compute_split_attention(
            q_1, k_1, v_1,
            config['k_cache_1'], config['v_cache_1'],
            context, offload_device, layer_id, False
        )
    
    # 移除同步操作以提高性能（NVLink优化）
    
    # === Output Projection ===
    o_0 = o_0.view(batch_size * seq_len, num_heads_0 * head_dim)
    o_1 = o_1.view(batch_size * seq_len, num_heads_1 * head_dim)
    
    if o_1.device != src_device:
        o_1 = o_1.to(src_device, non_blocking=True)
    
    out_0 = torch.nn.functional.linear(o_0, config['o_weight_0'], bias=None)
    o_weight_1 = config['o_weight_1'].to(src_device, non_blocking=True) if config['o_weight_1'].device != src_device else config['o_weight_1']
    out_1 = torch.nn.functional.linear(o_1, o_weight_1, bias=None)
    
    output = out_0 + out_1
    
    if seq_len == 1:
        output = output.view(batch_size, hidden_size)
    else:
        output = output.view(batch_size, seq_len, hidden_size)
    
    return output


def _init_split_kv_cache(layer_id: int, config: Dict) -> None:
    """初始化分片 KV cache"""
    DEBUG = os.environ.get("HB_DEBUG", "0") != "0"
    
    if config['cache_initialized']:
        return
    
    src_attn_module = config['src_attn'].attn
    src_k_cache = src_attn_module.k_cache
    src_v_cache = src_attn_module.v_cache
    
    if src_k_cache.numel() == 0:
        if DEBUG:
            print(f"[KVHeadSplit][layer {layer_id}] Original cache not initialized yet")
        return
    
    num_blocks, block_size, num_kv_heads, head_dim = src_k_cache.shape
    split_kv_head_idx = config['split_kv_head_idx']
    
    src_device = config['src_device']
    offload_device = config['offload_device']
    
    # Device 0
    config['k_cache_0'] = torch.empty(
        (num_blocks, block_size, split_kv_head_idx, head_dim),
        dtype=src_k_cache.dtype,
        device=src_device
    )
    config['v_cache_0'] = torch.empty(
        (num_blocks, block_size, split_kv_head_idx, head_dim),
        dtype=src_v_cache.dtype,
        device=src_device
    )
    
    config['k_cache_0'].copy_(src_k_cache[:, :, :split_kv_head_idx, :])
    config['v_cache_0'].copy_(src_v_cache[:, :, :split_kv_head_idx, :])
    
    # Device 1
    config['k_cache_1'] = torch.empty(
        (num_blocks, block_size, num_kv_heads - split_kv_head_idx, head_dim),
        dtype=src_k_cache.dtype,
        device=offload_device
    )
    config['v_cache_1'] = torch.empty(
        (num_blocks, block_size, num_kv_heads - split_kv_head_idx, head_dim),
        dtype=src_v_cache.dtype,
        device=offload_device
    )
    
    config['k_cache_1'].copy_(src_k_cache[:, :, split_kv_head_idx:, :].to(offload_device, non_blocking=True))
    config['v_cache_1'].copy_(src_v_cache[:, :, split_kv_head_idx:, :].to(offload_device, non_blocking=True))
    
    config['cache_initialized'] = True


def _compute_split_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    context: Context,
    device: torch.device,
    layer_id: int,
    is_device_0: bool
) -> torch.Tensor:
    """在指定设备上计算 attention（使用独立的分片 cache）"""
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    from HBserve.layers.attention import store_kvcache
    
    torch.cuda.set_device(device)
    
    # 确保所有张量在同一设备且连续
    q = q.to(device, non_blocking=True).contiguous()
    k = k.to(device, non_blocking=True).contiguous()
    v = v.to(device, non_blocking=True).contiguous()
    
    # 移动 context 到当前设备
    slot_mapping = None
    if context.slot_mapping is not None:
        slot_mapping = context.slot_mapping.to(device, non_blocking=True).contiguous()
    
    block_tables = None
    if context.block_tables is not None:
        block_tables = context.block_tables.to(device, non_blocking=True).contiguous()
    
    context_lens = None
    if context.context_lens is not None:
        context_lens = context.context_lens.to(device, non_blocking=True).contiguous()
    
    # 计算 attention
    scaling = (q.shape[-1]) ** -0.5
    
    if context.is_prefill:
        cu_seqlens_q = None
        cu_seqlens_k = None
        max_seqlen_q = context.max_seqlen_q if hasattr(context, 'max_seqlen_q') else None
        max_seqlen_k = context.max_seqlen_k if hasattr(context, 'max_seqlen_k') else None
        
        if hasattr(context, "cu_seqlens_q") and context.cu_seqlens_q is not None:
            cu_seqlens_q = context.cu_seqlens_q.to(device, non_blocking=True).contiguous()
        
        if hasattr(context, "cu_seqlens_k") and context.cu_seqlens_k is not None:
            cu_seqlens_k = context.cu_seqlens_k.to(device, non_blocking=True).contiguous()
        
        # 存储 KV 到分片 cache（如果 cache 已初始化）
        if k_cache is not None and v_cache is not None and slot_mapping is not None:
            k_contiguous = k.contiguous()
            v_contiguous = v.contiguous()
            store_kvcache(k_contiguous, v_contiguous, k_cache, v_cache, slot_mapping)
        
        # Prefill 阶段直接使用当前的 k, v 进行attention计算
        k_use = k.contiguous()
        v_use = v.contiguous()
        
        o = flash_attn_varlen_func(
            q, k_use, v_use,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scaling,
            causal=True,
            block_table=None  # Prefill 阶段不使用 block_table
        )
    else:
        if k_cache is None or k_cache.numel() == 0:
            raise RuntimeError(f"KV cache not initialized for decode mode")
        
        o = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache, v_cache,
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=scaling,
            causal=True
        )
        o = o.squeeze(1)
    
    return o


def execute_attention_offload_forward(
    layer_id: int,
    layer: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    context: Context,
    config: Dict,
    split_context_fn: Callable,
    sync_cache_fn: Callable
) -> torch.Tensor:
    """
    执行 Attention offload 的核心逻辑
    
    Args:
        layer_id: 层索引
        layer: 当前层
        positions: 位置张量
        hidden_states: 隐藏状态
        context: 上下文
        config: Attention offload 配置
        split_context_fn: 切分 context 的函数
        sync_cache_fn: 同步 cache 的函数
    
    Returns:
        Attention 输出
    """
    
    # ===== NVLink优化：动态启用检查 =====
    should_enable, reason = _should_enable_nvlink_optimization(hidden_states, context)
    DEBUG = os.environ.get("HB_NVLINK_LOG", "0") != "0"
    
    if not should_enable:
        if DEBUG:
            print(f"[NVLink][execute_attention_offload_forward] 跳过优化: {reason}")
        # 直接使用原始执行路径
        
        return config['src_attn'](positions, hidden_states)

    elif DEBUG:
        print(f"[NVLink][execute_attention_offload_forward] 启用优化: {reason}")
    
    src_attn = layer.self_attn
    offload_attn = config['offload_attn']
    src_device = config['src_device']
    offload_device = config['offload_device']
    ratio = config['split_ratio']
    is_prefill = context.is_prefill
    
    # 计算切分点
    if is_prefill:
        if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
            batch_size = len(context.cu_seqlens_q) - 1
            split_idx = int(round(batch_size * ratio))
            split_idx = max(1, min(split_idx, batch_size - 1))
            token_split_idx = context.cu_seqlens_q[split_idx].item()
        else:
            total_tokens = hidden_states.size(0)
            token_split_idx = int(round(total_tokens * ratio))
            token_split_idx = max(1, min(token_split_idx, total_tokens - 1))
            split_idx = token_split_idx
    else:
        batch_size = hidden_states.size(0)
        split_idx = int(round(batch_size * ratio))
        split_idx = max(1, min(split_idx, batch_size - 1))
        token_split_idx = split_idx
    
    if token_split_idx == 0 or token_split_idx >= hidden_states.size(0):
        return src_attn(positions, hidden_states)
    
    # Decode 阶段：同步 KV Cache
    if not is_prefill:
        sync_cache_fn(src_attn, offload_attn, split_idx, context.block_tables)
    
    # 切分输入
    hs_a = hidden_states[:token_split_idx]
    hs_b = hidden_states[token_split_idx:]
    pos_a = positions[:token_split_idx]
    pos_b = positions[token_split_idx:]
    
    # 切分 Context
    ctx_a = split_context_fn(context, 0, split_idx, token_split_idx)
    ctx_b = split_context_fn(context, split_idx, None, token_split_idx)
    
    # 移动到各自设备
    if hs_a.device != src_device:
        hs_a = hs_a.to(src_device, non_blocking=True)
        pos_a = pos_a.to(src_device, non_blocking=True)
    if hs_b.device != offload_device:
        hs_b = hs_b.to(offload_device, non_blocking=True)
        pos_b = pos_b.to(offload_device, non_blocking=True)
    
    # 并行执行
    stream_a = torch.cuda.Stream(device=src_device) if src_device.type == 'cuda' else None
    stream_b = torch.cuda.Stream(device=offload_device) if offload_device.type == 'cuda' else None
    
    start_a = end_a = start_b = end_b = None
    if src_device.type == 'cuda':
        start_a = torch.cuda.Event(enable_timing=True)
        end_a = torch.cuda.Event(enable_timing=True)
    if offload_device.type == 'cuda':
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)
    
    # 执行 A
    if stream_a is not None:
        with torch.cuda.stream(stream_a):
            if start_a is not None:
                start_a.record(stream_a)
            set_context(**ctx_a)
            out_a = src_attn(pos_a, hs_a)
            if end_a is not None:
                end_a.record(stream_a)
    else:
        set_context(**ctx_a)
        out_a = src_attn(pos_a, hs_a)
    
    # 执行 B
    if stream_b is not None:
        with torch.cuda.stream(stream_b):
            if start_b is not None:
                start_b.record(stream_b)
            set_context(**ctx_b)
            out_b = offload_attn(pos_b, hs_b)
            if end_b is not None:
                end_b.record(stream_b)
    else:
        set_context(**ctx_b)
        out_b = offload_attn(pos_b, hs_b)
    
    # 移除同步操作以提高性能（NVLink优化）
    
    # 恢复原始 context
    set_context(
        is_prefill=context.is_prefill,
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=context.max_seqlen_q,
        max_seqlen_k=context.max_seqlen_k,
        slot_mapping=context.slot_mapping,
        context_lens=context.context_lens,
        block_tables=context.block_tables
    )
    
    # 合并结果
    if out_b.device != src_device:
        out_b = out_b.to(src_device, non_blocking=True)
    
    output = torch.cat([out_a, out_b], dim=0)
    
    # 自适应调优
    if config['enable_autotune'] and start_a and end_a and start_b and end_b:
        _update_attention_offload_ratio(
            layer_id, config, ratio,
            start_a, end_a, start_b, end_b,
            src_device, offload_device
        )
    
    return output


def _update_attention_offload_ratio(
    layer_id: int,
    config: Dict,
    old_ratio: float,
    start_a, end_a, start_b, end_b,
    src_device, offload_device
) -> None:
    """更新 Attention offload 的切分比例"""
    time_a = start_a.elapsed_time(end_a) if src_device.type == 'cuda' else 0.0
    time_b = start_b.elapsed_time(end_b) if offload_device.type == 'cuda' else 0.0
    total = time_a + time_b
    
    if total > 0:
        target_ratio = time_b / total
        beta = config['autotune_beta']
        new_ratio = (1.0 - beta) * old_ratio + beta * target_ratio
        
        stats = config['autotune_stats']
        new_ratio = max(stats['min_ratio'], min(new_ratio, stats['max_ratio']))
        
        config['split_ratio'] = new_ratio
        
        if os.environ.get("HB_ATTN_OFFLOAD_LOG", "0") != "0":
            print(
                f"[AttnOffload][layer {layer_id}] "
                f"time_a={time_a:.3f}ms time_b={time_b:.3f}ms "
                f"ratio: {old_ratio:.3f} -> {new_ratio:.3f} (target={target_ratio:.3f})"
            )


def execute_layer_replication_forward(
    layer_id: int,
    layer: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    context: Context,
    replica: nn.Module,
    replica_device: torch.device,
    split_ratio: float,
    autotune_config: Optional[Dict],
    layer_device: torch.device,
    sync_kv_cache_fn: Callable
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    执行层复制的前向传播
    
    Args:
        layer_id: 层索引
        layer: 原始层
        positions: 位置张量
        hidden_states: 隐藏状态
        residual: 残差
        context: 上下文
        replica: 复制的层
        replica_device: 复制层的设备
        split_ratio: 切分比例
        autotune_config: 自适应调优配置
        layer_device: 原始层设备
        sync_kv_cache_fn: 同步 KV cache 的函数
    
    Returns:
        (hidden_states, residual)
    """
    
    # ===== NVLink优化：动态启用检查 =====
    should_enable, reason = _should_enable_nvlink_optimization(hidden_states, context)
    DEBUG = os.environ.get("HB_NVLINK_LOG", "0") != "0"
    
    if not should_enable:
        if DEBUG:
            print(f"[NVLink][execute_layer_replication_forward] 跳过优化: {reason}")
        # 直接使用原始执行路径
        
        return layer(positions, hidden_states, residual)

    elif DEBUG:
        print(f"[NVLink][execute_layer_replication_forward] 启用优化: {reason}")
    
    is_prefill = context.is_prefill
    
    # 保存原始 context
    orig_ctx = Context(
        is_prefill=context.is_prefill,
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=context.max_seqlen_q,
        max_seqlen_k=context.max_seqlen_k,
        slot_mapping=context.slot_mapping,
        context_lens=context.context_lens,
        block_tables=context.block_tables
    )
    
    # ===== 修复1: 改进切分点计算 =====
    if is_prefill:
        if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
            batch_size = len(context.cu_seqlens_q) - 1
            split_idx = int(round(batch_size * split_ratio))
            # 确保至少有1个batch在每一侧
            split_idx = max(1, min(split_idx, batch_size - 1))
            
            # 获取token切分点
            token_split_idx = context.cu_seqlens_q[split_idx].item()
        else:
            total_tokens = hidden_states.size(0)
            token_split_idx = int(round(total_tokens * split_ratio))
            token_split_idx = max(1, min(token_split_idx, total_tokens - 1))
            split_idx = token_split_idx
    else:
        batch_size = hidden_states.size(0)
        split_idx = int(round(batch_size * split_ratio))
        split_idx = max(1, min(split_idx, batch_size - 1))
        token_split_idx = split_idx
    
    # ===== 修复2: 增强边界检查 =====
    if split_idx <= 0 or split_idx >= batch_size:
        if os.environ.get("HB_REPLICA_LOG", "0") != "0":
            print(f"[Replica][layer {layer_id}] Invalid split_idx={split_idx}, batch_size={batch_size}, fallback to single device")
        return layer(positions, hidden_states, residual)
    
    if token_split_idx <= 0 or token_split_idx >= hidden_states.size(0):
        if os.environ.get("HB_REPLICA_LOG", "0") != "0":
            print(f"[Replica][layer {layer_id}] Invalid token_split_idx={token_split_idx}, total_tokens={hidden_states.size(0)}, fallback to single device")
        return layer(positions, hidden_states, residual)
    
    # Decode 阶段需要同步 KV cache
    if not is_prefill:
        sync_kv_cache_fn(layer, replica, split_idx, context.block_tables)
    
    # 切分输入
    hs_a = hidden_states[:token_split_idx].contiguous()
    hs_b = hidden_states[token_split_idx:].contiguous()
    pos_a = positions[:token_split_idx].contiguous()
    pos_b = positions[token_split_idx:].contiguous()
    res_a = None if residual is None else residual[:token_split_idx].contiguous()
    res_b = None if residual is None else residual[token_split_idx:].contiguous()
    
    # ===== 修复3: 正确切分 cu_seqlens =====
    cu_seqlens_q_a = None
    cu_seqlens_k_a = None
    cu_seqlens_q_b = None
    cu_seqlens_k_b = None
    
    if context.cu_seqlens_q is not None:
        cu_seqlens_q_a = context.cu_seqlens_q[:split_idx+1].contiguous()
        
        # 正确处理 B 部分：需要重新归零
        cu_seqlens_q_b = context.cu_seqlens_q[split_idx:].clone().contiguous()
        if len(cu_seqlens_q_b) > 0:
            offset = cu_seqlens_q_b[0].item()
            cu_seqlens_q_b = cu_seqlens_q_b - offset
    
    if context.cu_seqlens_k is not None:
        cu_seqlens_k_a = context.cu_seqlens_k[:split_idx+1].contiguous()
        
        cu_seqlens_k_b = context.cu_seqlens_k[split_idx:].clone().contiguous()
        if len(cu_seqlens_k_b) > 0:
            offset = cu_seqlens_k_b[0].item()
            cu_seqlens_k_b = cu_seqlens_k_b - offset
    
    # ===== 修复4: 确保 slot_mapping 和其他张量的正确切分 =====
    slot_mapping_a = None
    slot_mapping_b = None
    if context.slot_mapping is not None:
        if token_split_idx < len(context.slot_mapping):
            slot_mapping_a = context.slot_mapping[:token_split_idx].contiguous()
            slot_mapping_b = context.slot_mapping[token_split_idx:].contiguous()
        else:
            # 边界情况：token_split_idx 超出范围
            slot_mapping_a = context.slot_mapping.contiguous()
            slot_mapping_b = None
    
    context_lens_a = None
    context_lens_b = None
    if context.context_lens is not None:
        if split_idx < len(context.context_lens):
            context_lens_a = context.context_lens[:split_idx].contiguous()
            context_lens_b = context.context_lens[split_idx:].contiguous()
        else:
            context_lens_a = context.context_lens.contiguous()
            context_lens_b = None
    
    block_tables_a = None
    block_tables_b = None
    if context.block_tables is not None:
        if split_idx < len(context.block_tables):
            block_tables_a = context.block_tables[:split_idx].contiguous()
            block_tables_b = context.block_tables[split_idx:].contiguous()
        else:
            block_tables_a = context.block_tables.contiguous()
            block_tables_b = None
    
    # 移动到各自设备
    if hs_a.device != layer_device:
        hs_a = hs_a.to(layer_device, non_blocking=True)
        pos_a = pos_a.to(layer_device, non_blocking=True)
        if res_a is not None:
            res_a = res_a.to(layer_device, non_blocking=True)
    
    if hs_b.device != replica_device:
        hs_b = hs_b.to(replica_device, non_blocking=True)
        pos_b = pos_b.to(replica_device, non_blocking=True)
        if res_b is not None:
            res_b = res_b.to(replica_device, non_blocking=True)
    
    # 准备stream和计时
    stream_a = torch.cuda.Stream(device=layer_device) if layer_device.type == 'cuda' else None
    stream_b = torch.cuda.Stream(device=replica_device) if replica_device.type == 'cuda' else None
    
    start_a = end_a = start_b = end_b = None
    if layer_device.type == 'cuda':
        start_a = torch.cuda.Event(enable_timing=True)
        end_a = torch.cuda.Event(enable_timing=True)
    if replica_device.type == 'cuda':
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)
    
    # ===== 修复5: 添加调试日志 =====
    if os.environ.get("HB_REPLICA_LOG", "0") != "0":
        print(f"[Replica][layer {layer_id}] split_idx={split_idx}, token_split_idx={token_split_idx}")
        print(f"  hs_a.shape={hs_a.shape}, hs_b.shape={hs_b.shape}")
        if cu_seqlens_q_a is not None:
            print(f"  cu_seqlens_q_a={cu_seqlens_q_a.tolist()}")
        if cu_seqlens_q_b is not None:
            print(f"  cu_seqlens_q_b={cu_seqlens_q_b.tolist()}")
    
    # 并行执行
    if stream_a is not None:
        with torch.cuda.stream(stream_a):
            if start_a is not None:
                start_a.record(stream_a)
            set_context(
                is_prefill=orig_ctx.is_prefill,
                cu_seqlens_q=cu_seqlens_q_a,
                cu_seqlens_k=cu_seqlens_k_a,
                max_seqlen_q=orig_ctx.max_seqlen_q,
                max_seqlen_k=orig_ctx.max_seqlen_k,
                slot_mapping=slot_mapping_a,
                context_lens=context_lens_a,
                block_tables=block_tables_a
            )
            out_a, res_out_a = layer(pos_a, hs_a, res_a)
            if end_a is not None:
                end_a.record(stream_a)
    else:
        set_context(
            is_prefill=orig_ctx.is_prefill,
            cu_seqlens_q=cu_seqlens_q_a,
            cu_seqlens_k=cu_seqlens_k_a,
            max_seqlen_q=orig_ctx.max_seqlen_q,
            max_seqlen_k=orig_ctx.max_seqlen_k,
            slot_mapping=slot_mapping_a,
            context_lens=context_lens_a,
            block_tables=block_tables_a
        )
        out_a, res_out_a = layer(pos_a, hs_a, res_a)
    
    if stream_b is not None:
        with torch.cuda.stream(stream_b):
            if start_b is not None:
                start_b.record(stream_b)
            set_context(
                is_prefill=orig_ctx.is_prefill,
                cu_seqlens_q=cu_seqlens_q_b,
                cu_seqlens_k=cu_seqlens_k_b,
                max_seqlen_q=orig_ctx.max_seqlen_q,
                max_seqlen_k=orig_ctx.max_seqlen_k,
                slot_mapping=slot_mapping_b,
                context_lens=context_lens_b,
                block_tables=block_tables_b
            )
            out_b, res_out_b = replica(pos_b, hs_b, res_b)
            if end_b is not None:
                end_b.record(stream_b)
    else:
        set_context(
            is_prefill=orig_ctx.is_prefill,
            cu_seqlens_q=cu_seqlens_q_b,
            cu_seqlens_k=cu_seqlens_k_b,
            max_seqlen_q=orig_ctx.max_seqlen_q,
            max_seqlen_k=orig_ctx.max_seqlen_k,
            slot_mapping=slot_mapping_b,
            context_lens=context_lens_b,
            block_tables=block_tables_b
        )
        out_b, res_out_b = replica(pos_b, hs_b, res_b)
    
    # 移除同步操作以提高性能（NVLink优化）
    
    # 恢复context
    set_context(
        is_prefill=orig_ctx.is_prefill,
        cu_seqlens_q=orig_ctx.cu_seqlens_q,
        cu_seqlens_k=orig_ctx.cu_seqlens_k,
        max_seqlen_q=orig_ctx.max_seqlen_q,
        max_seqlen_k=orig_ctx.max_seqlen_k,
        slot_mapping=orig_ctx.slot_mapping,
        context_lens=orig_ctx.context_lens,
        block_tables=orig_ctx.block_tables
    )
    
    # 移回layer_device
    if out_b.device != layer_device:
        out_b = out_b.to(layer_device, non_blocking=True)
    if res_out_b is not None and res_out_b.device != layer_device:
        res_out_b = res_out_b.to(layer_device, non_blocking=True)
    
    # 合并结果
    hidden_states = torch.cat([out_a, out_b], dim=0)
    if res_out_a is None and res_out_b is None:
        residual = None
    elif res_out_a is None:
        residual = torch.cat([torch.zeros_like(out_a), res_out_b], dim=0)
    elif res_out_b is None:
        residual = torch.cat([res_out_a, torch.zeros_like(out_b)], dim=0)
    else:
        residual = torch.cat([res_out_a, res_out_b], dim=0)
    
    # Autotune
    if autotune_config and start_a and end_a and start_b and end_b:
        time_a = start_a.elapsed_time(end_a) if layer_device.type == 'cuda' else 0.0
        time_b = start_b.elapsed_time(end_b) if replica_device.type == 'cuda' else 0.0
        total = time_a + time_b
        if total > 0:
            target_ratio = time_b / total
            beta = autotune_config["beta"]
            new_ratio = (1.0 - beta) * split_ratio + beta * target_ratio
            new_ratio = max(autotune_config["min"], min(new_ratio, autotune_config["max"]))
            
            if os.environ.get("HB_REPLICA_LOG", "0") != "0":
                print(
                    f"[Replica][layer {layer_id}] time_a={time_a:.3f}ms time_b={time_b:.3f}ms "
                    f"ratio(old)={split_ratio:.3f} -> ratio(new)={new_ratio:.3f}"
                )
    
    return hidden_states, residual


def execute_continuous_layer_replication(
    layer_id: int,
    layer: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    context: Context,
    replica: nn.Module,
    replica_device: torch.device,
    split_ratio: float,
    autotune_config: Optional[Dict],
    layer_device: torch.device,
    sync_kv_cache_fn: Callable,
    is_first_in_group: bool,
    is_last_in_group: bool
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    执行连续复制层组的前向传播（高性能优化版）
    
    关键优化设计：
    - 第一层：切分 -> 并行计算 -> **不合并** + 保存分片状态（避免跨设备传输）
    - 中间层：直接用分片数据计算 -> **不合并** + 更新分片状态（数据保持在各自设备）
    - 最后一层：直接用分片数据计算 -> **仅在此合并到原始设备** + 清除分片状态
    
    **性能优势**：中间层完全避免跨设备数据传输和合并开销，充分利用NVLink并行性
    """
    
    # ===== NVLink优化：动态启用检查 =====
    should_enable, reason = _should_enable_nvlink_optimization(hidden_states, context)
    DEBUG = os.environ.get("HB_NVLINK_LOG", "0") != "0"
    
    if not should_enable:
        if DEBUG:
            print(f"[NVLink][execute_continuous_layer_replication] 跳过优化: {reason}")
        # 直接使用原始执行路径
        
        return layer(positions, hidden_states, residual)

    elif DEBUG:
        print(f"[NVLink][execute_continuous_layer_replication] 启用优化: {reason}")
    
    is_prefill = context.is_prefill
    DEBUG = os.environ.get("HB_REPLICA_LOG", "0") != "0"
    
    # 检查是否有上一层保存的分片状态
    split_state = _load_split_state_from_context(context)
    has_split_state = (split_state is not None) and (not is_first_in_group)
    
    # ===== 情况1：需要切分输入（第一层或没有分片状态） =====
    if is_first_in_group or not has_split_state:
        if DEBUG:
            print(f"[ReplicaGroup][layer {layer_id}] Splitting input")
        
        # 保存原始 context
        orig_ctx = Context(
            is_prefill=context.is_prefill,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            slot_mapping=context.slot_mapping,
            context_lens=context.context_lens,
            block_tables=context.block_tables
        )
        
        # 计算切分点
        split_idx, token_split_idx, batch_size = _compute_split_indices(
            hidden_states, context, split_ratio, is_prefill
        )
        
        # 边界检查
        if split_idx <= 0 or split_idx >= batch_size or \
           token_split_idx <= 0 or token_split_idx >= hidden_states.size(0):
            if DEBUG:
                print(f"[ReplicaGroup][layer {layer_id}] Invalid split, fallback")
            # 不使用分片，直接执行
            result = layer(positions, hidden_states, residual)
            if not is_last_in_group:
                _clear_split_state_from_context(context)  # 确保清除可能存在的旧状态
            return result
        
        # 同步 KV cache（decode 阶段）
        if not is_prefill:
            sync_kv_cache_fn(layer, replica, split_idx, context.block_tables)
        
        # 切分所有输入
        split_data = _split_inputs_for_replication(
            hidden_states, positions, residual, context,
            split_idx, token_split_idx
        )
        
        # 移动到各自设备
        split_data = _move_split_data_to_devices(
            split_data, layer_device, replica_device
        )
        
        # 并行执行
        out_a, res_a, out_b, res_b = _parallel_execute_split_layer_no_sync(
            layer, replica, split_data, orig_ctx,
            layer_device, replica_device
        )
        
        # **性能优化**：只在最后一层合并，中间层保持分片状态
        if is_last_in_group:
            # 最后一层：合并到原始设备
            hidden_states, residual = _merge_split_outputs(
                out_a, out_b, res_a, res_b, layer_device
            )
            if DEBUG:
                print(f"[ReplicaGroup][layer {layer_id}] Last layer - merged outputs")
            return hidden_states, residual
        else:
            # 第一层/中间层：保存分片状态，不合并（避免跨设备传输）
            _save_split_state_to_context(
                context,
                split_idx=split_idx,
                token_split_idx=token_split_idx,
                out_a=out_a,
                res_a=res_a,
                out_b=out_b,
                res_b=res_b,
                pos_a=split_data['pos_a'],
                pos_b=split_data['pos_b'],
                ctx_a=split_data['ctx_a'],
                ctx_b=split_data['ctx_b'],
                orig_ctx=orig_ctx,
                device_a=layer_device,
                device_b=replica_device
            )
            # 返回占位符（告知外部逻辑数据已分片）
            return _create_split_output_placeholder(out_a, out_b, res_a, res_b, layer_device)
    
    # ===== 情况2：使用上一层的分片状态（中间层和最后一层） =====
    else:
        if DEBUG:
            print(f"[ReplicaGroup][layer {layer_id}] Using split state from previous layer")
        
        # 同步 KV cache
        if not is_prefill:
            sync_kv_cache_fn(layer, replica, split_state['split_idx'], context.block_tables)
        
        # 从分片状态获取数据（这些是上一层的输出）
        hs_a = split_state['hs_a']
        hs_b = split_state['hs_b']
        pos_a = split_state['pos_a']
        pos_b = split_state['pos_b']
        res_a = split_state['res_a']
        res_b = split_state['res_b']
        ctx_a = split_state['ctx_a']
        ctx_b = split_state['ctx_b']
        orig_ctx = split_state['orig_ctx']
        
        # 确保数据在正确的设备上
        if hs_a.device != layer_device:
            hs_a = hs_a.to(layer_device, non_blocking=True)
            if res_a is not None:
                res_a = res_a.to(layer_device, non_blocking=True)
        
        if hs_b.device != replica_device:
            hs_b = hs_b.to(replica_device, non_blocking=True)
            if res_b is not None:
                res_b = res_b.to(replica_device, non_blocking=True)
        
        # 构建 split_data
        split_data = {
            'hs_a': hs_a,
            'hs_b': hs_b,
            'pos_a': pos_a,
            'pos_b': pos_b,
            'res_a': res_a,
            'res_b': res_b,
            'ctx_a': ctx_a,
            'ctx_b': ctx_b
        }
        
        # 并行执行（无同步版本）
        out_a, res_out_a, out_b, res_out_b = _parallel_execute_split_layer_no_sync(
            layer, replica, split_data, orig_ctx,
            layer_device, replica_device
        )
        
        # **性能优化**：只在最后一层合并，中间层保持分片状态
        if is_last_in_group:
            # 最后一层：合并到原始设备并清除分片状态
            hidden_states, residual = _merge_split_outputs(
                out_a, out_b, res_out_a, res_out_b, layer_device
            )
            _clear_split_state_from_context(context)
            if DEBUG:
                print(f"[ReplicaGroup][layer {layer_id}] Last layer - merged and cleared split state")
            return hidden_states, residual
        else:
            # 中间层：更新分片状态，不合并（避免跨设备传输）
            _save_split_state_to_context(
                context,
                split_idx=split_state['split_idx'],
                token_split_idx=split_state['token_split_idx'],
                out_a=out_a,
                res_a=res_out_a,
                out_b=out_b,
                res_b=res_out_b,
                pos_a=pos_a,
                pos_b=pos_b,
                ctx_a=ctx_a,
                ctx_b=ctx_b,
                orig_ctx=orig_ctx,
                device_a=layer_device,
                device_b=replica_device
            )
            if DEBUG:
                print(f"[ReplicaGroup][layer {layer_id}] Middle layer - updated split state (no merge)")
            # 返回占位符
            return _create_split_output_placeholder(out_a, out_b, res_out_a, res_out_b, layer_device)


# ===== 辅助函数 =====

def _compute_split_indices(
    hidden_states: torch.Tensor,
    context: Context,
    split_ratio: float,
    is_prefill: bool
) -> tuple[int, int, int]:
    """计算切分索引"""
    if is_prefill:
        if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
            batch_size = len(context.cu_seqlens_q) - 1
            split_idx = int(round(batch_size * split_ratio))
            split_idx = max(1, min(split_idx, batch_size - 1))
            token_split_idx = context.cu_seqlens_q[split_idx].item()
        else:
            total_tokens = hidden_states.size(0)
            token_split_idx = int(round(total_tokens * split_ratio))
            token_split_idx = max(1, min(token_split_idx, total_tokens - 1))
            split_idx = token_split_idx
            batch_size = total_tokens
    else:
        batch_size = hidden_states.size(0)
        split_idx = int(round(batch_size * split_ratio))
        split_idx = max(1, min(split_idx, batch_size - 1))
        token_split_idx = split_idx
    
    return split_idx, token_split_idx, batch_size


def _split_inputs_for_replication(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    residual: Optional[torch.Tensor],
    context: Context,
    split_idx: int,
    token_split_idx: int
) -> Dict:
    """切分输入数据"""
    # 切分张量
    hs_a = hidden_states[:token_split_idx].contiguous()
    hs_b = hidden_states[token_split_idx:].contiguous()
    pos_a = positions[:token_split_idx].contiguous()
    pos_b = positions[token_split_idx:].contiguous()
    res_a = None if residual is None else residual[:token_split_idx].contiguous()
    res_b = None if residual is None else residual[token_split_idx:].contiguous()
    
    # 切分 context（使用之前修复的逻辑）
    ctx_a, ctx_b = _split_context_for_replication(context, split_idx, token_split_idx)
    
    return {
        'hs_a': hs_a, 'hs_b': hs_b,
        'pos_a': pos_a, 'pos_b': pos_b,
        'res_a': res_a, 'res_b': res_b,
        'ctx_a': ctx_a, 'ctx_b': ctx_b,
        'split_idx': split_idx,
        'token_split_idx': token_split_idx,
        'orig_ctx': context
    }


def _split_context_for_replication(
    context: Context,
    split_idx: int,
    token_split_idx: int
) -> tuple[Dict, Dict]:
    """切分 context（修复版）"""
    # Context A
    cu_seqlens_q_a = None
    cu_seqlens_k_a = None
    if context.cu_seqlens_q is not None:
        cu_seqlens_q_a = context.cu_seqlens_q[:split_idx+1].contiguous()
    if context.cu_seqlens_k is not None:
        cu_seqlens_k_a = context.cu_seqlens_k[:split_idx+1].contiguous()
    
    slot_mapping_a = None if context.slot_mapping is None else \
        context.slot_mapping[:token_split_idx].contiguous()
    context_lens_a = None if context.context_lens is None else \
        context.context_lens[:split_idx].contiguous()
    block_tables_a = None if context.block_tables is None else \
        context.block_tables[:split_idx].contiguous()
    
    # Context B
    cu_seqlens_q_b = None
    cu_seqlens_k_b = None
    if context.cu_seqlens_q is not None:
        cu_seqlens_q_b = context.cu_seqlens_q[split_idx:].clone().contiguous()
        if len(cu_seqlens_q_b) > 0:
            offset = cu_seqlens_q_b[0].item()
            cu_seqlens_q_b = cu_seqlens_q_b - offset
    
    if context.cu_seqlens_k is not None:
        cu_seqlens_k_b = context.cu_seqlens_k[split_idx:].clone().contiguous()
        if len(cu_seqlens_k_b) > 0:
            offset = cu_seqlens_k_b[0].item()
            cu_seqlens_k_b = cu_seqlens_k_b - offset
    
    slot_mapping_b = None if context.slot_mapping is None else \
        context.slot_mapping[token_split_idx:].contiguous()
    context_lens_b = None if context.context_lens is None else \
        context.context_lens[split_idx:].contiguous()
    block_tables_b = None if context.block_tables is None else \
        context.block_tables[split_idx:].contiguous()
    
    ctx_a = {
        'cu_seqlens_q': cu_seqlens_q_a,
        'cu_seqlens_k': cu_seqlens_k_a,
        'slot_mapping': slot_mapping_a,
        'context_lens': context_lens_a,
        'block_tables': block_tables_a
    }
    
    ctx_b = {
        'cu_seqlens_q': cu_seqlens_q_b,
        'cu_seqlens_k': cu_seqlens_k_b,
        'slot_mapping': slot_mapping_b,
        'context_lens': context_lens_b,
        'block_tables': block_tables_b
    }
    
    return ctx_a, ctx_b


def _move_split_data_to_devices(
    split_data: Dict,
    device_a: torch.device,
    device_b: torch.device
) -> Dict:
    """将分片数据移动到对应设备"""
    split_data['hs_a'] = split_data['hs_a'].to(device_a, non_blocking=True)
    split_data['pos_a'] = split_data['pos_a'].to(device_a, non_blocking=True)
    if split_data['res_a'] is not None:
        split_data['res_a'] = split_data['res_a'].to(device_a, non_blocking=True)
    
    split_data['hs_b'] = split_data['hs_b'].to(device_b, non_blocking=True)
    split_data['pos_b'] = split_data['pos_b'].to(device_b, non_blocking=True)
    if split_data['res_b'] is not None:
        split_data['res_b'] = split_data['res_b'].to(device_b, non_blocking=True)
    
    return split_data


def _parallel_execute_split_layer(
    layer_a: nn.Module,
    layer_b: nn.Module,
    split_data: Dict,
    orig_ctx: Context,
    device_a: torch.device,
    device_b: torch.device
) -> tuple:
    """并行执行分片层"""
    stream_a = torch.cuda.Stream(device=device_a) if device_a.type == 'cuda' else None
    stream_b = torch.cuda.Stream(device=device_b) if device_b.type == 'cuda' else None
    
    # 执行 A
    if stream_a is not None:
        with torch.cuda.stream(stream_a):
            set_context(
                is_prefill=orig_ctx.is_prefill,
                cu_seqlens_q=split_data['ctx_a']['cu_seqlens_q'],
                cu_seqlens_k=split_data['ctx_a']['cu_seqlens_k'],
                max_seqlen_q=orig_ctx.max_seqlen_q,
                max_seqlen_k=orig_ctx.max_seqlen_k,
                slot_mapping=split_data['ctx_a']['slot_mapping'],
                context_lens=split_data['ctx_a']['context_lens'],
                block_tables=split_data['ctx_a']['block_tables']
            )
            out_a, res_a = layer_a(split_data['pos_a'], split_data['hs_a'], split_data['res_a'])
    else:
        set_context(
            is_prefill=orig_ctx.is_prefill,
            cu_seqlens_q=split_data['ctx_a']['cu_seqlens_q'],
            cu_seqlens_k=split_data['ctx_a']['cu_seqlens_k'],
            max_seqlen_q=orig_ctx.max_seqlen_q,
            max_seqlen_k=orig_ctx.max_seqlen_k,
            slot_mapping=split_data['ctx_a']['slot_mapping'],
            context_lens=split_data['ctx_a']['context_lens'],
            block_tables=split_data['ctx_a']['block_tables']
        )
        out_a, res_a = layer_a(split_data['pos_a'], split_data['hs_a'], split_data['res_a'])
    
    # 执行 B
    if stream_b is not None:
        with torch.cuda.stream(stream_b):
            set_context(
                is_prefill=orig_ctx.is_prefill,
                cu_seqlens_q=split_data['ctx_b']['cu_seqlens_q'],
                cu_seqlens_k=split_data['ctx_b']['cu_seqlens_k'],
                max_seqlen_q=orig_ctx.max_seqlen_q,
                max_seqlen_k=orig_ctx.max_seqlen_k,
                slot_mapping=split_data['ctx_b']['slot_mapping'],
                context_lens=split_data['ctx_b']['context_lens'],
                block_tables=split_data['ctx_b']['block_tables']
            )
            out_b, res_b = layer_b(split_data['pos_b'], split_data['hs_b'], split_data['res_b'])
    else:
        set_context(
            is_prefill=orig_ctx.is_prefill,
            cu_seqlens_q=split_data['ctx_b']['cu_seqlens_q'],
            cu_seqlens_k=split_data['ctx_b']['cu_seqlens_k'],
            max_seqlen_q=orig_ctx.max_seqlen_q,
            max_seqlen_k=orig_ctx.max_seqlen_k,
            slot_mapping=split_data['ctx_b']['slot_mapping'],
            context_lens=split_data['ctx_b']['context_lens'],
            block_tables=split_data['ctx_b']['block_tables']
        )
        out_b, res_b = layer_b(split_data['pos_b'], split_data['hs_b'], split_data['res_b'])
    
    # 移除同步操作以提高性能（NVLink优化）
    
    # 恢复原始 context
    set_context(
        is_prefill=orig_ctx.is_prefill,
        cu_seqlens_q=orig_ctx.cu_seqlens_q,
        cu_seqlens_k=orig_ctx.cu_seqlens_k,
        max_seqlen_q=orig_ctx.max_seqlen_q,
        max_seqlen_k=orig_ctx.max_seqlen_k,
        slot_mapping=orig_ctx.slot_mapping,
        context_lens=orig_ctx.context_lens,
        block_tables=orig_ctx.block_tables
    )
    
    return out_a, res_a, out_b, res_b




def _load_split_state_from_context(context: Context) -> Optional[Dict]:
    """从 context 加载分片状态"""
    return getattr(context, '_replica_split_state', None)


def _clear_split_state_from_context(context: Context):
    """清除分片状态"""
    if hasattr(context, '_replica_split_state'):
        delattr(context, '_replica_split_state')

def _create_split_output_placeholder(
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    res_a: Optional[torch.Tensor],
    res_b: Optional[torch.Tensor],
    target_device: torch.device
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    创建占位符输出（用于中间层，数据保持分片状态）
    
    **NVLink优化**：返回轻量级占位符，避免跨设备数据传输和合并开销
    真实数据保持在各自设备上（context._replica_split_state），利用NVLink异步传输
    """
    # 创建一个小的占位符张量（而不是空张量，避免某些代码检查维度时出错）
    # 使用1个token的占位符，确保下游代码能正常处理维度
    placeholder_hs = torch.zeros((1, out_a.size(-1) if out_a.dim() > 1 else out_a.size(0)), 
                                  dtype=out_a.dtype, device=target_device)
    placeholder_res = None
    if res_a is not None or res_b is not None:
        res_ref = res_a if res_a is not None else res_b
        placeholder_res = torch.zeros((1, res_ref.size(-1) if res_ref.dim() > 1 else res_ref.size(0)),
                                      dtype=res_ref.dtype, device=target_device)
    
    return placeholder_hs, placeholder_res

def _compute_split_indices(
    hidden_states: torch.Tensor,
    context: Context,
    split_ratio: float,
    is_prefill: bool
) -> tuple[int, int, int]:
    """计算切分索引"""
    if is_prefill:
        if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
            batch_size = len(context.cu_seqlens_q) - 1
            split_idx = int(round(batch_size * split_ratio))
            split_idx = max(1, min(split_idx, batch_size - 1))
            token_split_idx = context.cu_seqlens_q[split_idx].item()
        else:
            total_tokens = hidden_states.size(0)
            token_split_idx = int(round(total_tokens * split_ratio))
            token_split_idx = max(1, min(token_split_idx, total_tokens - 1))
            split_idx = token_split_idx
            batch_size = total_tokens
    else:
        batch_size = hidden_states.size(0)
        split_idx = int(round(batch_size * split_ratio))
        split_idx = max(1, min(split_idx, batch_size - 1))
        token_split_idx = split_idx
    
    return split_idx, token_split_idx, batch_size


def _split_inputs_for_replication(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    residual: Optional[torch.Tensor],
    context: Context,
    split_idx: int,
    token_split_idx: int
) -> Dict:
    """切分输入数据"""
    # 切分张量
    hs_a = hidden_states[:token_split_idx].contiguous()
    hs_b = hidden_states[token_split_idx:].contiguous()
    pos_a = positions[:token_split_idx].contiguous()
    pos_b = positions[token_split_idx:].contiguous()
    res_a = None if residual is None else residual[:token_split_idx].contiguous()
    res_b = None if residual is None else residual[token_split_idx:].contiguous()
    
    # 切分 context
    ctx_a, ctx_b = _split_context_for_replication(context, split_idx, token_split_idx)
    
    return {
        'hs_a': hs_a, 'hs_b': hs_b,
        'pos_a': pos_a, 'pos_b': pos_b,
        'res_a': res_a, 'res_b': res_b,
        'ctx_a': ctx_a, 'ctx_b': ctx_b,
        'split_idx': split_idx,
        'token_split_idx': token_split_idx
    }


def _split_context_for_replication(
    context: Context,
    split_idx: int,
    token_split_idx: int
) -> tuple[Dict, Dict]:
    """切分 context（修复版 - 增强边界检查）"""
    import torch
    DEBUG = os.environ.get("HB_REPLICA_LOG", "0") != "0"
    
    # Context A
    cu_seqlens_q_a = None
    cu_seqlens_k_a = None
    if context.cu_seqlens_q is not None:
        # 边界检查
        if split_idx + 1 <= len(context.cu_seqlens_q):
            cu_seqlens_q_a = context.cu_seqlens_q[:split_idx+1].contiguous()
        else:
            cu_seqlens_q_a = context.cu_seqlens_q.contiguous()
            if DEBUG:
                print(f"[FIX] cu_seqlens_q_a: split_idx+1={split_idx+1} > len={len(context.cu_seqlens_q)}")
    
    if context.cu_seqlens_k is not None:
        if split_idx + 1 <= len(context.cu_seqlens_k):
            cu_seqlens_k_a = context.cu_seqlens_k[:split_idx+1].contiguous()
        else:
            cu_seqlens_k_a = context.cu_seqlens_k.contiguous()
    
    slot_mapping_a = None
    if context.slot_mapping is not None:
        if token_split_idx <= len(context.slot_mapping):
            slot_mapping_a = context.slot_mapping[:token_split_idx].contiguous()
        else:
            slot_mapping_a = context.slot_mapping.contiguous()
            if DEBUG:
                print(f"[FIX] slot_mapping_a: token_split_idx={token_split_idx} > len={len(context.slot_mapping)}")
    
    context_lens_a = None
    if context.context_lens is not None:
        if split_idx <= len(context.context_lens):
            context_lens_a = context.context_lens[:split_idx].contiguous()
        else:
            context_lens_a = context.context_lens.contiguous()
    
    block_tables_a = None
    if context.block_tables is not None:
        if split_idx <= len(context.block_tables):
            block_tables_a = context.block_tables[:split_idx].contiguous()
        else:
            block_tables_a = context.block_tables.contiguous()
    
    # Context B（修复：正确处理偏移 + 边界检查）
    cu_seqlens_q_b = None
    cu_seqlens_k_b = None
    if context.cu_seqlens_q is not None:
        if split_idx < len(context.cu_seqlens_q):
            cu_seqlens_q_b = context.cu_seqlens_q[split_idx:].clone().contiguous()
            if len(cu_seqlens_q_b) > 0:
                offset = cu_seqlens_q_b[0].item()
                cu_seqlens_q_b = cu_seqlens_q_b - offset
        else:
            # 如果split_idx过大，创建一个空的但有效的张量
            cu_seqlens_q_b = torch.tensor([0], dtype=context.cu_seqlens_q.dtype, device=context.cu_seqlens_q.device)
            if DEBUG:
                print(f"[FIX] cu_seqlens_q_b: split_idx={split_idx} >= len={len(context.cu_seqlens_q)}, using empty")
    
    if context.cu_seqlens_k is not None:
        if split_idx < len(context.cu_seqlens_k):
            cu_seqlens_k_b = context.cu_seqlens_k[split_idx:].clone().contiguous()
            if len(cu_seqlens_k_b) > 0:
                offset = cu_seqlens_k_b[0].item()
                cu_seqlens_k_b = cu_seqlens_k_b - offset
        else:
            cu_seqlens_k_b = torch.tensor([0], dtype=context.cu_seqlens_k.dtype, device=context.cu_seqlens_k.device)
    
    slot_mapping_b = None if context.slot_mapping is None else \
        context.slot_mapping[token_split_idx:].contiguous()
    context_lens_b = None if context.context_lens is None else \
        context.context_lens[split_idx:].contiguous()
    block_tables_b = None if context.block_tables is None else \
        context.block_tables[split_idx:].contiguous()
    
    ctx_a = {
        'cu_seqlens_q': cu_seqlens_q_a,
        'cu_seqlens_k': cu_seqlens_k_a,
        'slot_mapping': slot_mapping_a,
        'context_lens': context_lens_a,
        'block_tables': block_tables_a
    }
    
    ctx_b = {
        'cu_seqlens_q': cu_seqlens_q_b,
        'cu_seqlens_k': cu_seqlens_k_b,
        'slot_mapping': slot_mapping_b,
        'context_lens': context_lens_b,
        'block_tables': block_tables_b
    }
    
    return ctx_a, ctx_b


def _move_split_data_to_devices(
    split_data: Dict,
    device_a: torch.device,
    device_b: torch.device
) -> Dict:
    """将分片数据移动到对应设备"""
    split_data['hs_a'] = split_data['hs_a'].to(device_a, non_blocking=True)
    split_data['pos_a'] = split_data['pos_a'].to(device_a, non_blocking=True)
    if split_data['res_a'] is not None:
        split_data['res_a'] = split_data['res_a'].to(device_a, non_blocking=True)
    
    split_data['hs_b'] = split_data['hs_b'].to(device_b, non_blocking=True)
    split_data['pos_b'] = split_data['pos_b'].to(device_b, non_blocking=True)
    if split_data['res_b'] is not None:
        split_data['res_b'] = split_data['res_b'].to(device_b, non_blocking=True)
    
    return split_data


def _parallel_execute_split_layer_no_sync(
    layer_a: nn.Module,
    layer_b: nn.Module,
    split_data: Dict,
    orig_ctx: Context,
    device_a: torch.device,
    device_b: torch.device
) -> tuple:
    """并行执行分片层（无同步优化版，用于连续层复制）"""
    stream_a = torch.cuda.Stream(device=device_a) if device_a.type == 'cuda' else None
    stream_b = torch.cuda.Stream(device=device_b) if device_b.type == 'cuda' else None
    
    # 执行 A
    if stream_a is not None:
        with torch.cuda.stream(stream_a):
            set_context(
                is_prefill=orig_ctx.is_prefill,
                cu_seqlens_q=split_data['ctx_a']['cu_seqlens_q'],
                cu_seqlens_k=split_data['ctx_a']['cu_seqlens_k'],
                max_seqlen_q=orig_ctx.max_seqlen_q,
                max_seqlen_k=orig_ctx.max_seqlen_k,
                slot_mapping=split_data['ctx_a']['slot_mapping'],
                context_lens=split_data['ctx_a']['context_lens'],
                block_tables=split_data['ctx_a']['block_tables']
            )
            out_a, res_a = layer_a(split_data['pos_a'], split_data['hs_a'], split_data['res_a'])
    else:
        set_context(
            is_prefill=orig_ctx.is_prefill,
            cu_seqlens_q=split_data['ctx_a']['cu_seqlens_q'],
            cu_seqlens_k=split_data['ctx_a']['cu_seqlens_k'],
            max_seqlen_q=orig_ctx.max_seqlen_q,
            max_seqlen_k=orig_ctx.max_seqlen_k,
            slot_mapping=split_data['ctx_a']['slot_mapping'],
            context_lens=split_data['ctx_a']['context_lens'],
            block_tables=split_data['ctx_a']['block_tables']
        )
        out_a, res_a = layer_a(split_data['pos_a'], split_data['hs_a'], split_data['res_a'])
    
    # 执行 B
    if stream_b is not None:
        with torch.cuda.stream(stream_b):
            set_context(
                is_prefill=orig_ctx.is_prefill,
                cu_seqlens_q=split_data['ctx_b']['cu_seqlens_q'],
                cu_seqlens_k=split_data['ctx_b']['cu_seqlens_k'],
                max_seqlen_q=orig_ctx.max_seqlen_q,
                max_seqlen_k=orig_ctx.max_seqlen_k,
                slot_mapping=split_data['ctx_b']['slot_mapping'],
                context_lens=split_data['ctx_b']['context_lens'],
                block_tables=split_data['ctx_b']['block_tables']
            )
            out_b, res_b = layer_b(split_data['pos_b'], split_data['hs_b'], split_data['res_b'])
    else:
        set_context(
            is_prefill=orig_ctx.is_prefill,
            cu_seqlens_q=split_data['ctx_b']['cu_seqlens_q'],
            cu_seqlens_k=split_data['ctx_b']['cu_seqlens_k'],
            max_seqlen_q=orig_ctx.max_seqlen_q,
            max_seqlen_k=orig_ctx.max_seqlen_k,
            slot_mapping=split_data['ctx_b']['slot_mapping'],
            context_lens=split_data['ctx_b']['context_lens'],
            block_tables=split_data['ctx_b']['block_tables']
        )
        out_b, res_b = layer_b(split_data['pos_b'], split_data['hs_b'], split_data['res_b'])
    
    # **关键NVLink优化**：不恢复context，不同步
    # 数据保持在各自设备上，利用NVLink异步传输，最大化并行性
    
    return out_a, res_a, out_b, res_b


def _parallel_execute_split_layer(
    layer_a: nn.Module,
    layer_b: nn.Module,
    split_data: Dict,
    orig_ctx: Context,
    device_a: torch.device,
    device_b: torch.device
) -> tuple:
    """并行执行分片层（用于非连续层复制）"""
    stream_a = torch.cuda.Stream(device=device_a) if device_a.type == 'cuda' else None
    stream_b = torch.cuda.Stream(device=device_b) if device_b.type == 'cuda' else None
    
    # 执行 A
    if stream_a is not None:
        with torch.cuda.stream(stream_a):
            set_context(
                is_prefill=orig_ctx.is_prefill,
                cu_seqlens_q=split_data['ctx_a']['cu_seqlens_q'],
                cu_seqlens_k=split_data['ctx_a']['cu_seqlens_k'],
                max_seqlen_q=orig_ctx.max_seqlen_q,
                max_seqlen_k=orig_ctx.max_seqlen_k,
                slot_mapping=split_data['ctx_a']['slot_mapping'],
                context_lens=split_data['ctx_a']['context_lens'],
                block_tables=split_data['ctx_a']['block_tables']
            )
            out_a, res_a = layer_a(split_data['pos_a'], split_data['hs_a'], split_data['res_a'])
    else:
        set_context(
            is_prefill=orig_ctx.is_prefill,
            cu_seqlens_q=split_data['ctx_a']['cu_seqlens_q'],
            cu_seqlens_k=split_data['ctx_a']['cu_seqlens_k'],
            max_seqlen_q=orig_ctx.max_seqlen_q,
            max_seqlen_k=orig_ctx.max_seqlen_k,
            slot_mapping=split_data['ctx_a']['slot_mapping'],
            context_lens=split_data['ctx_a']['context_lens'],
            block_tables=split_data['ctx_a']['block_tables']
        )
        out_a, res_a = layer_a(split_data['pos_a'], split_data['hs_a'], split_data['res_a'])
    
    # 执行 B
    if stream_b is not None:
        with torch.cuda.stream(stream_b):
            set_context(
                is_prefill=orig_ctx.is_prefill,
                cu_seqlens_q=split_data['ctx_b']['cu_seqlens_q'],
                cu_seqlens_k=split_data['ctx_b']['cu_seqlens_k'],
                max_seqlen_q=orig_ctx.max_seqlen_q,
                max_seqlen_k=orig_ctx.max_seqlen_k,
                slot_mapping=split_data['ctx_b']['slot_mapping'],
                context_lens=split_data['ctx_b']['context_lens'],
                block_tables=split_data['ctx_b']['block_tables']
            )
            out_b, res_b = layer_b(split_data['pos_b'], split_data['hs_b'], split_data['res_b'])
    else:
        set_context(
            is_prefill=orig_ctx.is_prefill,
            cu_seqlens_q=split_data['ctx_b']['cu_seqlens_q'],
            cu_seqlens_k=split_data['ctx_b']['cu_seqlens_k'],
            max_seqlen_q=orig_ctx.max_seqlen_q,
            max_seqlen_k=orig_ctx.max_seqlen_k,
            slot_mapping=split_data['ctx_b']['slot_mapping'],
            context_lens=split_data['ctx_b']['context_lens'],
            block_tables=split_data['ctx_b']['block_tables']
        )
        out_b, res_b = layer_b(split_data['pos_b'], split_data['hs_b'], split_data['res_b'])
    
    # 移除同步操作以提高性能（NVLink优化）
    
    # 恢复原始 context
    set_context(
        is_prefill=orig_ctx.is_prefill,
        cu_seqlens_q=orig_ctx.cu_seqlens_q,
        cu_seqlens_k=orig_ctx.cu_seqlens_k,
        max_seqlen_q=orig_ctx.max_seqlen_q,
        max_seqlen_k=orig_ctx.max_seqlen_k,
        slot_mapping=orig_ctx.slot_mapping,
        context_lens=orig_ctx.context_lens,
        block_tables=orig_ctx.block_tables
    )
    
    return out_a, res_a, out_b, res_b


def _merge_split_outputs(
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    res_a: Optional[torch.Tensor],
    res_b: Optional[torch.Tensor],
    target_device: torch.device
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    合并分片输出（NVLink优化版）
    
    **性能优化**：
    1. 使用 non_blocking=True 充分利用 NVLink 异步传输
    2. 避免不必要的同步操作
    3. 只在最后一层调用此函数
    """
    # NVLink优化：使用非阻塞传输
    if out_b.device != target_device:
        out_b = out_b.to(target_device, non_blocking=True)
    if res_b is not None and res_b.device != target_device:
        res_b = res_b.to(target_device, non_blocking=True)
    
    # 合并张量
    hidden_states = torch.cat([out_a, out_b], dim=0)
    
    if res_a is None and res_b is None:
        residual = None
    else:
        residual = torch.cat([
            res_a if res_a is not None else torch.zeros_like(out_a),
            res_b if res_b is not None else torch.zeros_like(out_b)
        ], dim=0)
    
    return hidden_states, residual


def _save_split_state_to_context(
    context: Context,
    split_idx: int,
    token_split_idx: int,
    out_a: torch.Tensor,
    res_a: Optional[torch.Tensor],
    out_b: torch.Tensor,
    res_b: Optional[torch.Tensor],
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    ctx_a: Dict,
    ctx_b: Dict,
    orig_ctx: Context,
    device_a: torch.device,
    device_b: torch.device
):
    """保存分片状态到 context"""
    context._replica_split_state = {
        'split_idx': split_idx,
        'token_split_idx': token_split_idx,
        'hs_a': out_a,
        'res_a': res_a,
        'hs_b': out_b,
        'res_b': res_b,
        'pos_a': pos_a,
        'pos_b': pos_b,
        'ctx_a': ctx_a,
        'ctx_b': ctx_b,
        'orig_ctx': orig_ctx,
        'device_a': device_a,
        'device_b': device_b
    }

