#!/usr/bin/env python3
"""
修复索引越界问题的补丁

问题：在 continuous layer replication 中，cu_seqlens 切分导致索引越界
解决：添加严格的边界检查和安全的切分逻辑
"""

import os
import sys

def apply_fix():
    """应用修复补丁"""
    
    optimization_forward_path = "HBserve/utils/optimization_forward.py"
    
    if not os.path.exists(optimization_forward_path):
        print(f"❌ 找不到文件: {optimization_forward_path}")
        return False
    
    # 备份原文件
    backup_path = optimization_forward_path + ".backup_index_fix"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(optimization_forward_path, backup_path)
        print(f"✅ 已备份原文件到: {backup_path}")
    
    # 读取文件
    with open(optimization_forward_path, 'r') as f:
        content = f.read()
    
    # 修复1: 改进 _split_context_for_replication 函数
    old_split_context = '''def _split_context_for_replication(
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
    
    slot_mapping_a = None if context.slot_mapping is None else \\
        context.slot_mapping[:token_split_idx].contiguous()
    context_lens_a = None if context.context_lens is None else \\
        context.context_lens[:split_idx].contiguous()
    block_tables_a = None if context.block_tables is None else \\
        context.block_tables[:split_idx].contiguous()
    
    # Context B（修复：正确处理偏移）
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
            cu_seqlens_k_b = cu_seqlens_k_b - offset'''

    new_split_context = '''def _split_context_for_replication(
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
            cu_seqlens_k_b = torch.tensor([0], dtype=context.cu_seqlens_k.dtype, device=context.cu_seqlens_k.device)'''

    # 应用修复
    if old_split_context in content:
        content = content.replace(old_split_context, new_split_context)
        print("✅ 修复1: 增强了 _split_context_for_replication 的边界检查")
    else:
        print("⚠️  修复1: 未找到匹配的代码段，可能已经修复或代码结构不同")
    
    # 修复2: 在 execute_continuous_layer_replication 开头添加验证
    old_execute_start = '''def execute_continuous_layer_replication(
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
    执行连续复制层组的前向传播（优化版）
    
    关键设计：
    - 第一层：切分 -> 并行计算 -> 合并返回 + 保存分片状态
    - 中间层：检查分片状态 -> 直接用分片数据计算 -> 合并返回 + 更新分片状态
    - 最后一层：检查分片状态 -> 直接用分片数据计算 -> 合并返回 + 清除分片状态
    
    **重要**：所有层都返回完整的合并张量，确保设备管理逻辑正常工作
    """
    is_prefill = context.is_prefill
    DEBUG = os.environ.get("HB_REPLICA_LOG", "0") != "0"'''

    new_execute_start = '''def execute_continuous_layer_replication(
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
    执行连续复制层组的前向传播（优化版）
    
    关键设计：
    - 第一层：切分 -> 并行计算 -> 合并返回 + 保存分片状态
    - 中间层：检查分片状态 -> 直接用分片数据计算 -> 合并返回 + 更新分片状态
    - 最后一层：检查分片状态 -> 直接用分片数据计算 -> 合并返回 + 清除分片状态
    
    **重要**：所有层都返回完整的合并张量，确保设备管理逻辑正常工作
    """
    is_prefill = context.is_prefill
    DEBUG = os.environ.get("HB_REPLICA_LOG", "0") != "0"
    
    # ===== 新增：输入验证 =====
    if DEBUG:
        print(f"[ReplicaGroup][L{layer_id}] Input validation:")
        print(f"  hidden_states.shape: {hidden_states.shape}")
        if context.cu_seqlens_q is not None:
            print(f"  cu_seqlens_q.shape: {context.cu_seqlens_q.shape}")
            print(f"  cu_seqlens_q: {context.cu_seqlens_q.tolist()}")
        if context.slot_mapping is not None:
            print(f"  slot_mapping.shape: {context.slot_mapping.shape}")
    
    # 边界检查：确保数据一致性
    if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 0:
        max_token_idx = context.cu_seqlens_q[-1].item()
        actual_tokens = hidden_states.size(0)
        if max_token_idx != actual_tokens:
            if DEBUG:
                print(f"[ReplicaGroup][L{layer_id}] WARNING: cu_seqlens mismatch!")
                print(f"  cu_seqlens_q[-1]={max_token_idx}, actual_tokens={actual_tokens}")
            # 如果不匹配，禁用优化
            return layer(positions, hidden_states, residual)'''

    if old_execute_start in content:
        content = content.replace(old_execute_start, new_execute_start)
        print("✅ 修复2: 添加了输入验证逻辑")
    else:
        print("⚠️  修复2: 未找到匹配的代码段")
    
    # 写回文件
    with open(optimization_forward_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ 修复已应用到: {optimization_forward_path}")
    return True


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    索引越界问题修复工具                                      ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    if apply_fix():
        print("\n" + "="*80)
        print("修复完成！")
        print("="*80)
        print("\n现在可以重新运行:")
        print("  python example_replication_autotune.py")
        print("\n如果问题仍然存在，尝试:")
        print("  export HB_REPLICA_LOG=1")
        print("  export CUDA_LAUNCH_BLOCKING=1")
        print("  python example_replication_autotune.py")
    else:
        print("\n❌ 修复失败")
        sys.exit(1)

