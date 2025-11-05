"""
NVLink优化版前向传播
专门针对NVLink连接的GPU优化，充分利用高带宽低延迟特性

关键优化：
1. 全部使用 non_blocking=True（NVLink下传输延迟<1ms）
2. 更激进的优化启用策略（因为传输开销小）
3. 异步传输+计算overlap
4. 增量KV Cache同步
"""

import torch
import os
from torch import nn
from typing import Dict, Optional, Callable, Tuple
from HBserve.utils.context import Context, set_context


# ============================================================================
# NVLink下的动态启用策略（更激进）
# ============================================================================

def should_enable_nvlink_optimization(
    hidden_states: torch.Tensor,
    context: Context,
    min_tokens_threshold: int = 1024  # NVLink下大幅降低阈值
) -> Tuple[bool, str]:
    """
    判断是否启用优化（NVLink版本）
    
    NVLink传输开销约0.1-0.5ms，远小于PCIe的27ms
    因此可以在更小的batch上启用优化
    """
    total_tokens = hidden_states.size(0)
    
    # NVLink下，即使小batch也可能有收益
    if context.is_prefill:
        # Prefill: 1024+ tokens就可以尝试
        if total_tokens >= min_tokens_threshold:
            return True, f"Prefill阶段，tokens={total_tokens} (NVLink)"
        else:
            return False, f"tokens太少 ({total_tokens} < {min_tokens_threshold})"
    
    # Decode阶段
    else:
        batch_size = hidden_states.size(0)
        # Decode: 8+ batch就可以尝试
        min_batch_decode = 8
        
        if batch_size >= min_batch_decode:
            return True, f"Decode阶段，batch={batch_size} (NVLink)"
        else:
            return False, f"batch太小 ({batch_size} < {min_batch_decode})"


# ============================================================================
# NVLink增量KV Cache同步
# ============================================================================

class NVLinkKVCache:
    """利用NVLink高带宽的增量KV Cache管理"""
    
    def __init__(self):
        self.synced_lengths = {}
        self.stats = {
            'total_synced_mb': 0,
            'total_saved_mb': 0,
            'sync_count': 0
        }
    
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
        """增量同步，使用non_blocking充分利用NVLink带宽"""
        DEBUG = os.environ.get("HB_NVLINK_LOG", "0") != "0"
        
        for batch_idx in range(start_batch_idx, len(context_lens)):
            current_len = context_lens[batch_idx].item()
            key = (layer_id, batch_idx)
            last_len = self.synced_lengths.get(key, 0)
            
            if current_len <= last_len:
                continue
            
            # 计算需要同步的block
            start_block = last_len // block_size
            end_block = (current_len + block_size - 1) // block_size
            
            blocks = block_tables[batch_idx]
            
            synced_blocks = 0
            for block_idx in range(start_block, end_block):
                if block_idx >= len(blocks):
                    break
                    
                phys_block = blocks[block_idx].item()
                
                # 关键：使用non_blocking=True，NVLink下几乎无延迟
                dst_k[phys_block].copy_(src_k[phys_block], non_blocking=True)
                dst_v[phys_block].copy_(src_v[phys_block], non_blocking=True)
                synced_blocks += 1
            
            self.synced_lengths[key] = current_len
            
            # 统计
            if synced_blocks > 0:
                block_mb = src_k[0].numel() * src_k.element_size() * 2 / 1024 / 1024
                self.stats['total_synced_mb'] += synced_blocks * block_mb
                self.stats['sync_count'] += 1
                
                if start_block > 0:
                    self.stats['total_saved_mb'] += start_block * block_mb
            
            if DEBUG:
                print(
                    f"[NVLink-KVCache][L{layer_id}][B{batch_idx}] "
                    f"同步 {synced_blocks}/{end_block} blocks (len:{last_len}->{current_len})"
                )
    
    def reset_batch(self, batch_idx: int):
        """重置某个batch"""
        keys_to_remove = [k for k in self.synced_lengths.keys() if k[1] == batch_idx]
        for key in keys_to_remove:
            del self.synced_lengths[key]
    
    def print_stats(self):
        """打印统计"""
        if self.stats['sync_count'] == 0:
            return
        
        total = self.stats['total_synced_mb'] + self.stats['total_saved_mb']
        saved_pct = (self.stats['total_saved_mb'] / total * 100) if total > 0 else 0
        
        print(f"\n[NVLink-KVCache] 统计:")
        print(f"  同步次数: {self.stats['sync_count']}")
        print(f"  传输数据: {self.stats['total_synced_mb']:.2f} MB")
        print(f"  节省数据: {self.stats['total_saved_mb']:.2f} MB ({saved_pct:.1f}%)")


# ============================================================================
# NVLink优化的Layer Replication
# ============================================================================

def execute_nvlink_layer_replication(
    layer_id: int,
    layer: nn.Module,
    replica: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    context: Context,
    layer_device: torch.device,
    replica_device: torch.device,
    split_ratio: float = 0.5,
    kv_cache_manager: Optional[NVLinkKVCache] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    NVLink优化的Layer Replication
    
    关键优化：
    1. 所有数据传输使用non_blocking=True
    2. 更激进的启用策略
    3. Stream overlap最大化
    """
    DEBUG = os.environ.get("HB_NVLINK_LOG", "0") != "0"
    
    # ===== 1. 动态启用检查（NVLink版本）=====
    should_opt, reason = should_enable_nvlink_optimization(hidden_states, context)
    
    if not should_opt:
        if DEBUG:
            print(f"[NVLink-Replica][L{layer_id}] 跳过: {reason}")
        return layer(positions, hidden_states, residual)
    
    if DEBUG:
        print(f"[NVLink-Replica][L{layer_id}] 启用: {reason}")
    
    # ===== 2. 计算切分点 =====
    is_prefill = context.is_prefill
    
    if is_prefill:
        if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
            batch_size = len(context.cu_seqlens_q) - 1
            split_idx = max(1, min(int(batch_size * split_ratio), batch_size - 1))
            token_split_idx = context.cu_seqlens_q[split_idx].item()
        else:
            total_tokens = hidden_states.size(0)
            token_split_idx = max(1, min(int(total_tokens * split_ratio), total_tokens - 1))
            split_idx = token_split_idx
            batch_size = total_tokens
    else:
        batch_size = hidden_states.size(0)
        split_idx = max(1, min(int(batch_size * split_ratio), batch_size - 1))
        token_split_idx = split_idx
    
    # ===== 3. 增量KV Cache同步（Decode）=====
    if not is_prefill and kv_cache_manager is not None:
        # 获取attention模块
        src_attn = getattr(layer, 'self_attn', layer)
        dst_attn = getattr(replica, 'self_attn', replica)
        
        if hasattr(src_attn, 'attn'):
            src_attn = src_attn.attn
            dst_attn = dst_attn.attn
        
        if hasattr(src_attn, 'k_cache') and hasattr(dst_attn, 'k_cache'):
            if src_attn.k_cache.numel() > 0:
                kv_cache_manager.sync_incremental(
                    layer_id,
                    src_attn.k_cache,
                    src_attn.v_cache,
                    dst_attn.k_cache,
                    dst_attn.v_cache,
                    context.block_tables,
                    context.context_lens,
                    split_idx
                )
    
    # ===== 4. 切分数据 =====
    hs_a = hidden_states[:token_split_idx].contiguous()
    hs_b = hidden_states[token_split_idx:].contiguous()
    pos_a = positions[:token_split_idx].contiguous()
    pos_b = positions[token_split_idx:].contiguous()
    res_a = None if residual is None else residual[:token_split_idx].contiguous()
    res_b = None if residual is None else residual[token_split_idx:].contiguous()
    
    # ===== 5. 切分Context =====
    ctx_a, ctx_b = _split_context_nvlink(context, split_idx, token_split_idx)
    
    # ===== 6. 异步传输到replica设备（关键！non_blocking=True）=====
    # 注意：即使设备相同也显式调用，确保数据在正确设备
    hs_b_dev = hs_b.to(replica_device, non_blocking=True)
    pos_b_dev = pos_b.to(replica_device, non_blocking=True)
    res_b_dev = res_b.to(replica_device, non_blocking=True) if res_b is not None else None
    
    # 同时将A的数据确保在layer_device上
    if hs_a.device != layer_device:
        hs_a = hs_a.to(layer_device, non_blocking=True)
        pos_a = pos_a.to(layer_device, non_blocking=True)
        if res_a is not None:
            res_a = res_a.to(layer_device, non_blocking=True)
    
    # ===== 7. 并行计算（Stream overlap）=====
    stream_a = torch.cuda.Stream(device=layer_device) if layer_device.type == 'cuda' else None
    stream_b = torch.cuda.Stream(device=replica_device) if replica_device.type == 'cuda' else None
    
    # 计时（用于autotune）
    start_a = end_a = start_b = end_b = None
    if layer_device.type == 'cuda':
        start_a = torch.cuda.Event(enable_timing=True)
        end_a = torch.cuda.Event(enable_timing=True)
    if replica_device.type == 'cuda':
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)
    
    # 保存原始context
    orig_ctx = _save_context(context)
    
    # 执行A
    if stream_a is not None:
        with torch.cuda.stream(stream_a):
            if start_a:
                start_a.record(stream_a)
            _set_context_dict(ctx_a)
            out_a, res_out_a = layer(pos_a, hs_a, res_a)
            if end_a:
                end_a.record(stream_a)
    else:
        _set_context_dict(ctx_a)
        out_a, res_out_a = layer(pos_a, hs_a, res_a)
    
    # 执行B（在独立stream中，与A并行）
    if stream_b is not None:
        with torch.cuda.stream(stream_b):
            if start_b:
                start_b.record(stream_b)
            _set_context_dict(ctx_b)
            out_b, res_out_b = replica(pos_b_dev, hs_b_dev, res_b_dev)
            if end_b:
                end_b.record(stream_b)
    else:
        _set_context_dict(ctx_b)
        out_b, res_out_b = replica(pos_b_dev, hs_b_dev, res_b_dev)
    
    # 同步两个stream
    if stream_a:
        stream_a.synchronize()
    if stream_b:
        stream_b.synchronize()
    
    # 恢复context
    _restore_context(orig_ctx)
    
    # ===== 8. 异步传输结果回主设备（non_blocking）=====
    if out_b.device != layer_device:
        out_b = out_b.to(layer_device, non_blocking=True)
    if res_out_b is not None and res_out_b.device != layer_device:
        res_out_b = res_out_b.to(layer_device, non_blocking=True)
    
    # ===== 9. 合并结果 =====
    hidden_states = torch.cat([out_a, out_b], dim=0)
    
    if res_out_a is None and res_out_b is None:
        residual = None
    else:
        residual = torch.cat([
            res_out_a if res_out_a is not None else torch.zeros_like(out_a),
            res_out_b if res_out_b is not None else torch.zeros_like(out_b)
        ], dim=0)
    
    # ===== 10. 打印性能（可选）=====
    if DEBUG and start_a and end_a and start_b and end_b:
        torch.cuda.synchronize()  # 确保所有操作完成
        time_a = start_a.elapsed_time(end_a)
        time_b = start_b.elapsed_time(end_b)
        max_time = max(time_a, time_b)
        efficiency = (time_a + time_b) / (2 * max_time) * 100
        
        print(
            f"[NVLink-Replica][L{layer_id}] "
            f"A={time_a:.2f}ms B={time_b:.2f}ms "
            f"并行效率={efficiency:.1f}% "
            f"(理论加速={(time_a+time_b)/max_time:.2f}x)"
        )
    
    return hidden_states, residual


# ============================================================================
# NVLink优化的Attention Offload
# ============================================================================

def execute_nvlink_attention_offload(
    layer_id: int,
    layer: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    context: Context,
    src_attn: nn.Module,
    offload_attn: nn.Module,
    src_device: torch.device,
    offload_device: torch.device,
    split_ratio: float = 0.5
) -> torch.Tensor:
    """NVLink优化的Attention Offload"""
    DEBUG = os.environ.get("HB_NVLINK_LOG", "0") != "0"
    
    # 动态启用检查
    should_opt, reason = should_enable_nvlink_optimization(hidden_states, context)
    
    if not should_opt:
        if DEBUG:
            print(f"[NVLink-AttnOffload][L{layer_id}] 跳过: {reason}")
        return src_attn(positions, hidden_states)
    
    is_prefill = context.is_prefill
    
    # 计算切分点
    if is_prefill:
        if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
            batch_size = len(context.cu_seqlens_q) - 1
            split_idx = max(1, min(int(batch_size * split_ratio), batch_size - 1))
            token_split_idx = context.cu_seqlens_q[split_idx].item()
        else:
            total_tokens = hidden_states.size(0)
            token_split_idx = max(1, min(int(total_tokens * split_ratio), total_tokens - 1))
            split_idx = token_split_idx
    else:
        batch_size = hidden_states.size(0)
        split_idx = max(1, min(int(batch_size * split_ratio), batch_size - 1))
        token_split_idx = split_idx
    
    # 切分
    hs_a = hidden_states[:token_split_idx].contiguous()
    hs_b = hidden_states[token_split_idx:].contiguous()
    pos_a = positions[:token_split_idx].contiguous()
    pos_b = positions[token_split_idx:].contiguous()
    
    ctx_a, ctx_b = _split_context_nvlink(context, split_idx, token_split_idx)
    
    # 异步传输
    hs_b_dev = hs_b.to(offload_device, non_blocking=True)
    pos_b_dev = pos_b.to(offload_device, non_blocking=True)
    
    # 并行执行
    stream_a = torch.cuda.Stream(device=src_device) if src_device.type == 'cuda' else None
    stream_b = torch.cuda.Stream(device=offload_device) if offload_device.type == 'cuda' else None
    
    orig_ctx = _save_context(context)
    
    if stream_a is not None:
        with torch.cuda.stream(stream_a):
            _set_context_dict(ctx_a)
            out_a = src_attn(pos_a, hs_a)
    else:
        _set_context_dict(ctx_a)
        out_a = src_attn(pos_a, hs_a)
    
    if stream_b is not None:
        with torch.cuda.stream(stream_b):
            _set_context_dict(ctx_b)
            out_b = offload_attn(pos_b_dev, hs_b_dev)
    else:
        _set_context_dict(ctx_b)
        out_b = offload_attn(pos_b_dev, hs_b_dev)
    
    if stream_a:
        stream_a.synchronize()
    if stream_b:
        stream_b.synchronize()
    
    _restore_context(orig_ctx)
    
    # 传输结果
    if out_b.device != src_device:
        out_b = out_b.to(src_device, non_blocking=True)
    
    output = torch.cat([out_a, out_b], dim=0)
    
    return output


# ============================================================================
# 辅助函数
# ============================================================================

def _split_context_nvlink(context: Context, split_idx: int, token_split_idx: int) -> Tuple[Dict, Dict]:
    """切分context（使用contiguous确保性能）"""
    ctx_a = {
        'is_prefill': context.is_prefill,
        'max_seqlen_q': context.max_seqlen_q,
        'max_seqlen_k': context.max_seqlen_k,
    }
    
    ctx_b = {
        'is_prefill': context.is_prefill,
        'max_seqlen_q': context.max_seqlen_q,
        'max_seqlen_k': context.max_seqlen_k,
    }
    
    if context.cu_seqlens_q is not None:
        ctx_a['cu_seqlens_q'] = context.cu_seqlens_q[:split_idx+1].contiguous()
        cu_q_b = context.cu_seqlens_q[split_idx:].clone()
        ctx_b['cu_seqlens_q'] = (cu_q_b - cu_q_b[0]).contiguous()
    
    if context.cu_seqlens_k is not None:
        ctx_a['cu_seqlens_k'] = context.cu_seqlens_k[:split_idx+1].contiguous()
        cu_k_b = context.cu_seqlens_k[split_idx:].clone()
        ctx_b['cu_seqlens_k'] = (cu_k_b - cu_k_b[0]).contiguous()
    
    if context.slot_mapping is not None:
        ctx_a['slot_mapping'] = context.slot_mapping[:token_split_idx].contiguous()
        ctx_b['slot_mapping'] = context.slot_mapping[token_split_idx:].contiguous()
    
    if context.context_lens is not None:
        ctx_a['context_lens'] = context.context_lens[:split_idx].contiguous()
        ctx_b['context_lens'] = context.context_lens[split_idx:].contiguous()
    
    if context.block_tables is not None:
        ctx_a['block_tables'] = context.block_tables[:split_idx].contiguous()
        ctx_b['block_tables'] = context.block_tables[split_idx:].contiguous()
    
    return ctx_a, ctx_b


def _save_context(context: Context) -> Dict:
    """保存context"""
    return {
        'is_prefill': context.is_prefill,
        'cu_seqlens_q': context.cu_seqlens_q,
        'cu_seqlens_k': context.cu_seqlens_k,
        'max_seqlen_q': context.max_seqlen_q,
        'max_seqlen_k': context.max_seqlen_k,
        'slot_mapping': context.slot_mapping,
        'context_lens': context.context_lens,
        'block_tables': context.block_tables
    }


def _set_context_dict(ctx: Dict):
    """设置context"""
    set_context(**ctx)


def _restore_context(ctx: Dict):
    """恢复context"""
    set_context(**ctx)

