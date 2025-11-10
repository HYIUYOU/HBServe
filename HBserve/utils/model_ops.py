"""
模型优化 Mixin 类
将层迁移、层复制、Attention Offload 等功能模块化
"""

import torch
import copy
from torch import nn
from typing import Dict, Optional, Tuple, Any
from HBserve.utils.context import get_context, set_context, Context


class LayerDeviceManagementMixin:
    """层设备管理 Mixin - 负责将层迁移到不同设备"""
    
    def __init__(self):
        super().__init__()
        # 跟踪每层的设备位置
        self.layer_devices: Dict[int, torch.device] = {}
    
    def move_layer_to_device(self, layer_id: int, device: str | torch.device) -> None:
        """将指定层移动到目标设备"""
        if layer_id < 0 or layer_id >= len(self.layers):
            raise ValueError(f"层索引 {layer_id} 超出范围 [0, {len(self.layers)-1}]")
        
        if isinstance(device, str):
            device = torch.device(device)
        
        self.layers[layer_id] = self.layers[layer_id].to(device)
        self.layer_devices[layer_id] = device
        print(f"层 {layer_id} 已移动到设备 {device}")
    
    def get_layer_device(self, layer_id: int) -> torch.device:
        """获取指定层的当前设备"""
        if layer_id in self.layer_devices:
            return self.layer_devices[layer_id]
        else:
            return next(self.layers[layer_id].parameters()).device
    
    def set_layer_device_distribution(self, layer_device_map: Dict[int, str | torch.device]) -> None:
        """批量设置层的设备分布"""
        for layer_id, device in layer_device_map.items():
            self.move_layer_to_device(layer_id, device)


class LayerReplicationMixin:
    """层复制 Mixin - 负责层复制和自适应调优"""
    
    def __init__(self):
        super().__init__()
        # 复制执行：记录被复制的层及其副本和设备
        self.replicas: Dict[int, nn.Module] = {}
        self.replica_devices: Dict[int, torch.device] = {}
        self.replica_split_ratio: Dict[int, float] = {}
    
    def replicate_layer_to_device(
        self, 
        layer_id: int, 
        device: str | torch.device, 
        split_ratio: float = 0.5,
        layer_class: type = None
    ) -> None:
        """
        将指定层复制一个副本到目标GPU设备，用于批次切分并行执行该层。
        
        Args:
            layer_id: 层索引
            device: 目标设备
            split_ratio: 原设备处理的batch比例（0-1之间）
            layer_class: 层的类（用于创建新实例），如果为None则从config推断
        """
        if layer_id < 0 or layer_id >= len(self.layers):
            raise ValueError(f"层索引 {layer_id} 超出范围 [0, {len(self.layers)-1}]")
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio 需在 (0, 1) 区间内")
        if isinstance(device, str):
            device = torch.device(device)
        
        # 获取原始层的dtype和设备
        src_layer = self.layers[layer_id]
        src_dtype = next(src_layer.parameters()).dtype
        src_device = next(src_layer.parameters()).device
        
        # 更安全的复制：先转到CPU，保持dtype
        src_state = {k: v.detach().cpu() for k, v in src_layer.state_dict().items()}
        
        # 创建副本 - 如果提供了layer_class就用它，否则调用子类实现的方法
        if layer_class is not None:
            replica = layer_class(self.config)
        else:
            replica = self._create_decoder_layer()
        
        replica = replica.to(device=device, dtype=src_dtype)
        replica.load_state_dict(src_state, strict=True)
        
        self.replicas[layer_id] = replica
        self.replica_devices[layer_id] = device
        self.replica_split_ratio[layer_id] = float(split_ratio)
        
        print(f"层 {layer_id} 已复制：{src_device}({src_dtype}) -> {device}({src_dtype})，切分比例: {split_ratio:.2f}")
    
    def clear_layer_replication(self, layer_id: Optional[int] = None) -> None:
        """清除指定层或全部层的复制副本"""
        if layer_id is None:
            self.replicas.clear()
            self.replica_devices.clear()
            self.replica_split_ratio.clear()
            print("已清除所有层的复制配置")
        else:
            self.replicas.pop(layer_id, None)
            self.replica_devices.pop(layer_id, None)
            self.replica_split_ratio.pop(layer_id, None)
            print(f"已清除层 {layer_id} 的复制配置")
    
    def update_replication_split_ratio(self, layer_id: int, split_ratio: float) -> None:
        """更新已复制层的切分比例（原设备比例）"""
        if layer_id not in self.replicas:
            raise ValueError(f"层 {layer_id} 未配置复制，无法更新split_ratio")
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio 需在 (0, 1) 区间内")
        self.replica_split_ratio[layer_id] = float(split_ratio)
        print(f"层 {layer_id} 切分比例已更新为: {split_ratio:.2f}")
    
    
    def _sync_kv_cache_for_decode(
        self, 
        src_layer: nn.Module, 
        dst_layer: nn.Module, 
        split_idx: int,
        block_tables: Optional[torch.Tensor]
    ) -> None:
        """在 decode 阶段同步 KV cache 到副本层"""
        src_attn = src_layer.self_attn.attn
        dst_attn = dst_layer.self_attn.attn
        
        if src_attn.k_cache.numel() == 0:
            return
        
        dst_device = next(dst_layer.parameters()).device
        
        if dst_attn.k_cache.numel() == 0 or dst_attn.k_cache.shape != src_attn.k_cache.shape:
            dst_attn.k_cache = src_attn.k_cache.to(dst_device, non_blocking=True)
            dst_attn.v_cache = src_attn.v_cache.to(dst_device, non_blocking=True)
        else:
            dst_attn.k_cache.copy_(src_attn.k_cache, non_blocking=True)
            dst_attn.v_cache.copy_(src_attn.v_cache, non_blocking=True)
    
    def _create_decoder_layer(self):
        """
        子类需要实现此方法，返回一个新的decoder layer实例
        例如：return Qwen3DecoderLayer(self.config)
        """
        raise NotImplementedError("子类必须实现 _create_decoder_layer 方法")


class AttentionOffloadMixin:
    """Attention Offload Mixin - 负责 Attention 卸载相关功能"""
    
    def __init__(self):
        super().__init__()
        # Attention offload 配置
        self.attention_offload: Dict[int, dict] = {}
    
    def attention_offload_by_batch(
        self,
        layer_id: int,
        offload_device: str | torch.device,
        split_ratio: float = 0.5,
        attention_class: type = None
    ) -> None:
        """将指定层的 Attention 模块 offload 到另一个 GPU，按 batch 切分并行计算"""
        if layer_id < 0 or layer_id >= len(self.layers):
            raise ValueError(f"层索引 {layer_id} 超出范围")
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio 需在 (0, 1) 区间内")
        
        if isinstance(offload_device, str):
            offload_device = torch.device(offload_device)
        
        src_layer = self.layers[layer_id]
        src_attn = src_layer.self_attn
        src_device = next(src_attn.parameters()).device
        src_dtype = next(src_attn.parameters()).dtype
        
        # 创建 attention 副本
        if attention_class is not None:
            offload_attn = attention_class(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                max_position=self.config.max_position_embeddings,
                rms_norm_eps=self.config.rms_norm_eps,
                qkv_bias=getattr(self.config, 'attention_bias', False),
                head_dim=getattr(self.config, 'head_dim', None),
                rope_theta=getattr(self.config, "rope_theta", 1000000),
                rope_scaling=getattr(self.config, "rope_scaling", None),
            )
        else:
            offload_attn = self._create_attention_module()
        
        # 复制权重到 offload 设备
        src_state = {k: v.detach().cpu() for k, v in src_attn.state_dict().items()}
        offload_attn = offload_attn.to(device=offload_device, dtype=src_dtype)
        offload_attn.load_state_dict(src_state, strict=True)
        
        self.attention_offload[layer_id] = {
            'type': 'batch_split',
            'offload_attn': offload_attn,
            'offload_device': offload_device,
            'src_device': src_device,
            'split_ratio': float(split_ratio),
        }
        
        print(f"Attention Offload: 层 {layer_id} Attention 已 offload：")
        print(f"  原设备: {src_device} ({src_dtype})")
        print(f"  目标设备: {offload_device} ({src_dtype})")
        print(f"  切分比例: {split_ratio:.2f}")
    
    def attention_offload_by_kv_head(
        self,
        layer_id: int,
        offload_device: str | torch.device,
        split_kv_head_idx: Optional[int] = None,
    ) -> None:
        """按 KV Head 切分 Attention 到两个 GPU"""
        if layer_id < 0 or layer_id >= len(self.layers):
            raise ValueError(f"层索引 {layer_id} 超出范围")
        
        if isinstance(offload_device, str):
            offload_device = torch.device(offload_device)
        
        src_layer = self.layers[layer_id]
        src_attn = src_layer.self_attn
        src_device = next(src_attn.parameters()).device
        src_dtype = next(src_attn.parameters()).dtype
        
        num_heads = src_attn.num_heads
        num_kv_heads = src_attn.num_kv_heads
        head_dim = src_attn.head_dim
        
        if split_kv_head_idx is None:
            split_kv_head_idx = num_kv_heads // 2
        
        if split_kv_head_idx <= 0 or split_kv_head_idx >= num_kv_heads:
            raise ValueError(f"split_kv_head_idx={split_kv_head_idx} 必须在 (0, {num_kv_heads}) 范围内")
        
        heads_per_kv_head = num_heads // num_kv_heads
        split_q_head_idx = split_kv_head_idx * heads_per_kv_head
        
        # 提取和分片原始权重
        qkv_weight = src_attn.qkv_proj.weight.data
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        
        q_weight = qkv_weight[:q_size, :]
        k_weight = qkv_weight[q_size:q_size+kv_size, :]
        v_weight = qkv_weight[q_size+kv_size:, :]
        
        # Device 0 的权重
        q_weight_0 = q_weight[:split_q_head_idx * head_dim, :]
        k_weight_0 = k_weight[:split_kv_head_idx * head_dim, :]
        v_weight_0 = v_weight[:split_kv_head_idx * head_dim, :]
        qkv_weight_0 = torch.cat([q_weight_0, k_weight_0, v_weight_0], dim=0)
        
        # Device 1 的权重
        q_weight_1 = q_weight[split_q_head_idx * head_dim:, :]
        k_weight_1 = k_weight[split_kv_head_idx * head_dim:, :]
        v_weight_1 = v_weight[split_kv_head_idx * head_dim:, :]
        qkv_weight_1 = torch.cat([q_weight_1, k_weight_1, v_weight_1], dim=0)
        
        # 处理 bias
        qkv_bias_0 = qkv_bias_1 = None
        if src_attn.qkv_proj.bias is not None:
            qkv_bias = src_attn.qkv_proj.bias.data
            q_bias = qkv_bias[:q_size]
            k_bias = qkv_bias[q_size:q_size+kv_size]
            v_bias = qkv_bias[q_size+kv_size:]
            
            q_bias_0 = q_bias[:split_q_head_idx * head_dim]
            k_bias_0 = k_bias[:split_kv_head_idx * head_dim]
            v_bias_0 = v_bias[:split_kv_head_idx * head_dim]
            qkv_bias_0 = torch.cat([q_bias_0, k_bias_0, v_bias_0], dim=0)
            
            q_bias_1 = q_bias[split_q_head_idx * head_dim:]
            k_bias_1 = k_bias[split_kv_head_idx * head_dim:]
            v_bias_1 = v_bias[split_kv_head_idx * head_dim:]
            qkv_bias_1 = torch.cat([q_bias_1, k_bias_1, v_bias_1], dim=0)
        
        # 分片 output projection 权重
        o_weight = src_attn.o_proj.weight.data
        o_weight_0 = o_weight[:, :split_q_head_idx * head_dim].contiguous()
        o_weight_1 = o_weight[:, split_q_head_idx * head_dim:].contiguous()
        
        # 保存配置
        self.attention_offload[layer_id] = {
            'type': 'kv_head_split',
            'src_attn': src_attn,
            'src_device': src_device,
            'offload_device': offload_device,
            'split_kv_head_idx': split_kv_head_idx,
            'split_q_head_idx': split_q_head_idx,
            'num_kv_heads_0': split_kv_head_idx,
            'num_kv_heads_1': num_kv_heads - split_kv_head_idx,
            'head_dim': head_dim,
            'qkv_weight_0': qkv_weight_0.to(src_device),
            'qkv_bias_0': qkv_bias_0.to(src_device) if qkv_bias_0 is not None else None,
            'o_weight_0': o_weight_0.to(src_device),
            'qkv_weight_1': qkv_weight_1.to(offload_device),
            'qkv_bias_1': qkv_bias_1.to(offload_device) if qkv_bias_1 is not None else None,
            'o_weight_1': o_weight_1.to(offload_device),
            'q_norm_weight': src_attn.q_norm.weight.data.clone(),
            'k_norm_weight': src_attn.k_norm.weight.data.clone(),
            'rotary_emb': src_attn.rotary_emb,
            'cache_initialized': False,
            'k_cache_0': None,
            'v_cache_0': None,
            'k_cache_1': None,
            'v_cache_1': None,
        }
        
        print(f"KV Head Split: 层 {layer_id} Attention 已按 KV Head 切分：")
        print(f"  原设备 {src_device}: Q heads [0:{split_q_head_idx}], KV heads [0:{split_kv_head_idx}]")
        print(f"  目标设备 {offload_device}: Q heads [{split_q_head_idx}:{num_heads}], KV heads [{split_kv_head_idx}:{num_kv_heads}]")
    
    def clear_attention_offload(self, layer_id: Optional[int] = None) -> None:
        """清除 Attention offload 配置"""
        if layer_id is None:
            self.attention_offload.clear()
            print("已清除所有 Attention offload 配置")
        else:
            self.attention_offload.pop(layer_id, None)
            print(f"已清除层 {layer_id} 的 Attention offload 配置")
    
    def _create_attention_module(self):
        """子类需要实现此方法，返回一个新的attention模块实例"""
        raise NotImplementedError("子类必须实现 _create_attention_module 方法")
    
    def _split_context_for_attention(
        self,
        context: Context,
        batch_start: int,
        batch_end: Optional[int],
        token_offset: int
    ) -> dict:
        """为 attention 切分 context"""
        is_prefill = context.is_prefill
        
        if batch_end is None:
            cu_seqlens_q = context.cu_seqlens_q[batch_start:] - token_offset if context.cu_seqlens_q is not None else None
            cu_seqlens_k = context.cu_seqlens_k[batch_start:] - token_offset if context.cu_seqlens_k is not None else None
            
            if is_prefill:
                slot_mapping = context.slot_mapping[token_offset:] if context.slot_mapping is not None else None
            else:
                slot_mapping = context.slot_mapping[batch_start:] if context.slot_mapping is not None else None
            
            context_lens = context.context_lens[batch_start:] if context.context_lens is not None else None
            block_tables = context.block_tables[batch_start:] if context.block_tables is not None else None
        else:
            cu_seqlens_q = context.cu_seqlens_q[:batch_end+1] if context.cu_seqlens_q is not None else None
            cu_seqlens_k = context.cu_seqlens_k[:batch_end+1] if context.cu_seqlens_k is not None else None
            
            if is_prefill:
                slot_mapping = context.slot_mapping[:token_offset] if context.slot_mapping is not None else None
            else:
                slot_mapping = context.slot_mapping[:batch_end] if context.slot_mapping is not None else None
            
            context_lens = context.context_lens[:batch_end] if context.context_lens is not None else None
            block_tables = context.block_tables[:batch_end] if context.block_tables is not None else None
        
        return {
            'is_prefill': context.is_prefill,
            'cu_seqlens_q': cu_seqlens_q,
            'cu_seqlens_k': cu_seqlens_k,
            'max_seqlen_q': context.max_seqlen_q,
            'max_seqlen_k': context.max_seqlen_k,
            'slot_mapping': slot_mapping,
            'context_lens': context_lens,
            'block_tables': block_tables
        }
    
    def _sync_attention_kv_cache(
        self,
        src_attn: nn.Module,
        dst_attn: nn.Module,
        split_idx: int,
        block_tables: Optional[torch.Tensor]
    ) -> None:
        """同步 Attention 的 KV Cache"""
        src_attn_module = src_attn.attn
        dst_attn_module = dst_attn.attn
        
        if src_attn_module.k_cache.numel() == 0:
            return
        
        dst_device = next(dst_attn.parameters()).device
        src_shape = src_attn_module.k_cache.shape
        
        if dst_attn_module.k_cache.numel() == 0:
            dst_attn_module.k_cache = src_attn_module.k_cache.to(dst_device, non_blocking=True)
            dst_attn_module.v_cache = src_attn_module.v_cache.to(dst_device, non_blocking=True)
        elif dst_attn_module.k_cache.shape != src_shape:
            dst_attn_module.k_cache = src_attn_module.k_cache.to(dst_device, non_blocking=True)
            dst_attn_module.v_cache = src_attn_module.v_cache.to(dst_device, non_blocking=True)
        else:
            dst_attn_module.k_cache.copy_(src_attn_module.k_cache, non_blocking=True)
            dst_attn_module.v_cache.copy_(src_attn_module.v_cache, non_blocking=True)
        
        if dst_device.type == 'cuda':
            torch.cuda.synchronize(dst_device)


class ModelOptimizationMixin(
    LayerDeviceManagementMixin,
    LayerReplicationMixin,
    AttentionOffloadMixin
):
    """组合所有优化功能的 Mixin"""
    
    def __init__(self):
        super().__init__()