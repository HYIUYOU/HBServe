"""
优化版前向传播：解决性能问题
核心策略：大幅减少数据传输，只在确实有收益时才启用优化
"""

import torch
import os
from torch import nn
from typing import Dict, Optional, Callable, Tuple
from HBserve.utils.context import Context


# ============================================================================
# 策略1: 动态启用/禁用优化
# ============================================================================

def should_enable_optimization(
    hidden_states: torch.Tensor,
    context: Context,
    optimization_type: str,
    min_tokens_threshold: int = 4096
) -> Tuple[bool, str]:
    """
    判断是否应该启用优化
    
    根据性能分析：开销约27ms，因此需要计算时间 > 54ms 才有收益
    """
    total_tokens = hidden_states.size(0)
    
    # 策略1: token数量阈值
    if total_tokens < min_tokens_threshold:
        return False, f"token数量太少 ({total_tokens} < {min_tokens_threshold})"
    
    # 策略2: Prefill阶段更适合优化（计算密集）
    if context.is_prefill:
        if total_tokens >= min_tokens_threshold:
            return True, f"Prefill阶段，token数量充足 ({total_tokens})"
        else:
            return False, f"Prefill阶段，但token数量不足"
    
    # 策略3: Decode阶段需要更高的阈值（计算量小）
    else:
        batch_size = hidden_states.size(0)
        min_batch_decode = 32  # 根据27ms开销，需要至少32个batch
        
        if batch_size < min_batch_decode:
            return False, f"Decode阶段batch太小 ({batch_size} < {min_batch_decode})"
        else:
            return True, f"Decode阶段，batch充足 ({batch_size})"


# ============================================================================
# 策略2: 增量KV Cache同步（减少90%+的传输量）
# ============================================================================

class IncrementalKVCache:
    """增量KV Cache管理"""
    
    def __init__(self):
        self.synced_lengths = {}  # (layer_id, batch_idx) -> synced_length
        self.total_saved_mb = 0
    
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
        """只同步新增的KV"""
        DEBUG = os.environ.get("HB_KVCACHE_LOG", "0") != "0"
        
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
            
            # 只同步新增block
            synced_blocks = 0
            for block_idx in range(start_block, end_block):
                if block_idx >= len(blocks):
                    break
                    
                phys_block = blocks[block_idx].item()
                dst_k[phys_block].copy_(src_k[phys_block], non_blocking=True)
                dst_v[phys_block].copy_(src_v[phys_block], non_blocking=True)
                synced_blocks += 1
            
            self.synced_lengths[key] = current_len
            
            # 统计节省
            if start_block > 0:
                saved_mb = start_block * block_size * src_k.shape[-1] * src_k.shape[-2] * src_k.element_size() * 2 / 1024 / 1024
                self.total_saved_mb += saved_mb
            
            if DEBUG:
                print(f"[KVCache][L{layer_id}][B{batch_idx}] 同步 {synced_blocks} blocks (total {end_block})")


# ============================================================================
# 策略3: 减少数据传输的Layer Replication
# ============================================================================

def execute_optimized_layer_replication(
    layer_id: int,
    layer: nn.Module,
    replica: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    context: Context,
    layer_device: torch.device,
    replica_device: torch.device,
    split_ratio: float,
    kv_cache_manager: Optional[IncrementalKVCache] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    优化版Layer Replication
    
    关键优化：
    1. 检查是否应该启用
    2. 使用non_blocking传输
    3. 增量KV Cache同步
    """
    DEBUG = os.environ.get("HB_REPLICA_LOG", "0") != "0"
    
    # ===== 优化1: 动态启用检查 =====
    should_opt, reason = should_enable_optimization(
        hidden_states, context, "layer_replication"
    )
    
    if not should_opt:
        if DEBUG:
            print(f"[Replica][L{layer_id}] 跳过优化: {reason}")
        return layer(positions, hidden_states, residual)
    
    if DEBUG:
        print(f"[Replica][L{layer_id}] 启用优化: {reason}")
    
    # ===== 优化2: 计算切分点 =====
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
    
    # ===== 优化3: 增量KV Cache同步（Decode阶段）=====
    if not is_prefill and kv_cache_manager is not None:
        src_attn = layer.self_attn.attn if hasattr(layer.self_attn, 'attn') else layer.self_attn
        dst_attn = replica.self_attn.attn if hasattr(replica.self_attn, 'attn') else replica.self_attn
        
        if hasattr(src_attn, 'k_cache') and hasattr(dst_attn, 'k_cache'):
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
    
    # ===== 优化4: 切分数据 =====
    hs_a = hidden_states[:token_split_idx].contiguous()
    hs_b = hidden_states[token_split_idx:].contiguous()
    pos_a = positions[:token_split_idx].contiguous()
    pos_b = positions[token_split_idx:].contiguous()
    res_a = None if residual is None else residual[:token_split_idx].contiguous()
    res_b = None if residual is None else residual[token_split_idx:].contiguous()
    
    # ===== 优化5: 切分Context =====
    ctx_a, ctx_b = _split_context(context, split_idx, token_split_idx)
    
    # ===== 优化6: 使用non_blocking传输 =====
    if hs_b.device != replica_device:
        hs_b = hs_b.to(replica_device, non_blocking=True)
        pos_b = pos_b.to(replica_device, non_blocking=True)
        if res_b is not None:
            res_b = res_b.to(replica_device, non_blocking=True)
    
    # ===== 优化7: 并行计算 =====
    stream_a = torch.cuda.Stream(device=layer_device) if layer_device.type == 'cuda' else None
    stream_b = torch.cuda.Stream(device=replica_device) if replica_device.type == 'cuda' else None
    
    # 计时
    start_a = end_a = start_b = end_b = None
    if layer_device.type == 'cuda':
        start_a = torch.cuda.Event(enable_timing=True)
        end_a = torch.cuda.Event(enable_timing=True)
    if replica_device.type == 'cuda':
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)
    
    # 执行A
    if stream_a is not None:
        with torch.cuda.stream(stream_a):
            if start_a:
                start_a.record(stream_a)
            _set_context_from_dict(context, ctx_a)
            out_a, res_out_a = layer(pos_a, hs_a, res_a)
            if end_a:
                end_a.record(stream_a)
    else:
        _set_context_from_dict(context, ctx_a)
        out_a, res_out_a = layer(pos_a, hs_a, res_a)
    
    # 执行B
    if stream_b is not None:
        with torch.cuda.stream(stream_b):
            if start_b:
                start_b.record(stream_b)
            _set_context_from_dict(context, ctx_b)
            out_b, res_out_b = replica(pos_b, hs_b, res_b)
            if end_b:
                end_b.record(stream_b)
    else:
        _set_context_from_dict(context, ctx_b)
        out_b, res_out_b = replica(pos_b, hs_b, res_b)
    
    # 同步
    if stream_a:
        stream_a.synchronize()
    if stream_b:
        stream_b.synchronize()
    
    # 恢复context
    _restore_original_context(context)
    
    # ===== 优化8: 使用non_blocking传输结果 =====
    if out_b.device != layer_device:
        out_b = out_b.to(layer_device, non_blocking=True)
    if res_out_b is not None and res_out_b.device != layer_device:
        res_out_b = res_out_b.to(layer_device, non_blocking=True)
    
    # 合并
    hidden_states = torch.cat([out_a, out_b], dim=0)
    
    if res_out_a is None and res_out_b is None:
        residual = None
    else:
        residual = torch.cat([
            res_out_a if res_out_a is not None else torch.zeros_like(out_a),
            res_out_b if res_out_b is not None else torch.zeros_like(out_b)
        ], dim=0)
    
    # 打印时间
    if DEBUG and start_a and end_a and start_b and end_b:
        time_a = start_a.elapsed_time(end_a)
        time_b = start_b.elapsed_time(end_b)
        print(f"[Replica][L{layer_id}] time_a={time_a:.2f}ms, time_b={time_b:.2f}ms")
    
    return hidden_states, residual


def _split_context(context: Context, split_idx: int, token_split_idx: int) -> Tuple[Dict, Dict]:
    """切分context"""
    # Context A
    ctx_a = {
        'is_prefill': context.is_prefill,
        'max_seqlen_q': context.max_seqlen_q,
        'max_seqlen_k': context.max_seqlen_k,
    }
    
    if context.cu_seqlens_q is not None:
        ctx_a['cu_seqlens_q'] = context.cu_seqlens_q[:split_idx+1].contiguous()
    if context.cu_seqlens_k is not None:
        ctx_a['cu_seqlens_k'] = context.cu_seqlens_k[:split_idx+1].contiguous()
    if context.slot_mapping is not None:
        ctx_a['slot_mapping'] = context.slot_mapping[:token_split_idx].contiguous()
    if context.context_lens is not None:
        ctx_a['context_lens'] = context.context_lens[:split_idx].contiguous()
    if context.block_tables is not None:
        ctx_a['block_tables'] = context.block_tables[:split_idx].contiguous()
    
    # Context B
    ctx_b = {
        'is_prefill': context.is_prefill,
        'max_seqlen_q': context.max_seqlen_q,
        'max_seqlen_k': context.max_seqlen_k,
    }
    
    if context.cu_seqlens_q is not None:
        cu_q_b = context.cu_seqlens_q[split_idx:].clone()
        ctx_b['cu_seqlens_q'] = (cu_q_b - cu_q_b[0]).contiguous()
    if context.cu_seqlens_k is not None:
        cu_k_b = context.cu_seqlens_k[split_idx:].clone()
        ctx_b['cu_seqlens_k'] = (cu_k_b - cu_k_b[0]).contiguous()
    if context.slot_mapping is not None:
        ctx_b['slot_mapping'] = context.slot_mapping[token_split_idx:].contiguous()
    if context.context_lens is not None:
        ctx_b['context_lens'] = context.context_lens[split_idx:].contiguous()
    if context.block_tables is not None:
        ctx_b['block_tables'] = context.block_tables[split_idx:].contiguous()
    
    return ctx_a, ctx_b


def _set_context_from_dict(context: Context, ctx_dict: Dict):
    """从字典设置context"""
    from HBserve.utils.context import set_context
    set_context(**ctx_dict)


def _restore_original_context(context: Context):
    """恢复原始context"""
    from HBserve.utils.context import set_context
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


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("""
使用方法：

1. 在模型的forward中：

```python
from HBserve.utils.optimized_forward import (
    should_enable_optimization,
    execute_optimized_layer_replication,
    IncrementalKVCache
)

class OptimizedModel:
    def __init__(self):
        self.kv_cache_manager = IncrementalKVCache()
    
    def forward(self, positions, hidden_states, residual=None):
        # 对每一层
        for i, layer in enumerate(self.layers):
            if hasattr(layer, '_replica'):
                # 使用优化版
                hidden_states, residual = execute_optimized_layer_replication(
                    layer_id=i,
                    layer=layer,
                    replica=layer._replica,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    context=get_context(),
                    layer_device=layer.device,
                    replica_device=layer._replica.device,
                    split_ratio=0.5,
                    kv_cache_manager=self.kv_cache_manager
                )
            else:
                hidden_states, residual = layer(positions, hidden_states, residual)
        
        return hidden_states
```

2. 设置合适的batch size：
   - Prefill: batch_size * seq_len >= 4096
   - Decode: batch_size >= 32

3. 启用调试日志：
   export HB_REPLICA_LOG=1
   export HB_KVCACHE_LOG=1
""")

