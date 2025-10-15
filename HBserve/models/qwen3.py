"""
Qwen3 模型实现 - 使用 Mixin 模式重构
支持层迁移、层复制、Attention Offload 等优化功能
"""

import torch
import copy
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config
from typing import Tuple, Optional

from HBserve.layers.activation import SiluAndMul
from HBserve.layers.attention import Attention
from HBserve.layers.layernorm import RMSNorm
from HBserve.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from HBserve.layers.rotary_embedding import get_rope
from HBserve.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from HBserve.utils.context import get_context, set_context, Context

# 导入优化 Mixin 和执行逻辑
from HBserve.utils.model_ops import ModelOptimizationMixin
from HBserve.utils.optimization_forward import (
    execute_kv_head_split_forward,
    execute_attention_offload_forward,
    execute_layer_replication_forward
)


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states) # 注意这里的hidden_states 不包含prefix caching命中的部分
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # q，k，v都不包含prefix caching命中的部分
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k) # 根据position进行旋转位置编码，因为RoPE为绝对位置编码，因此可以这么做
        o = self.attn(q, k, v) # 将q，k，v（不包含prefix caching命中的部分）传入attention计算
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None: # 第一次计算，没有残差
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states) # 这里的hidden_states 不包含prefix caching命中的部分
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states) # 将positions传入attention计算
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(ModelOptimizationMixin, nn.Module):
    """
    Qwen3 模型 - 使用 Mixin 模式重构
    
    继承 ModelOptimizationMixin 后，自动获得以下优化功能：
    1. 层设备管理（move_layer_to_device, get_layer_device 等）
    2. 层复制（replicate_layer_to_device, clear_layer_replication 等）
    3. Attention Offload（attention_offload_by_batch, attention_offload_by_kv_head 等）
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        # 先初始化 nn.Module
        nn.Module.__init__(self)
        # 再初始化 Mixin（这会初始化所有优化相关的字典）
        ModelOptimizationMixin.__init__(self)
        
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # ========== 实现 Mixin 需要的抽象方法 ==========
    
    def _create_decoder_layer(self):
        """创建一个新的 decoder layer 实例（LayerReplicationMixin 需要）"""
        return Qwen3DecoderLayer(self.config)
    
    def _create_attention_module(self):
        """创建一个新的 attention 模块实例（AttentionOffloadMixin 需要）"""
        return Qwen3Attention(
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

    # ========== 前向传播 ==========

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 获取第一层的设备
        first_layer_device = self.get_layer_device(0)
        
        # 2. 确保 embed_tokens 在第一层设备
        if self.embed_tokens.weight.device != first_layer_device:
            self.embed_tokens = self.embed_tokens.to(first_layer_device)
        
        # 3. 确保输入在第一层设备
        if input_ids.device != first_layer_device:
            input_ids = input_ids.to(first_layer_device)
        if positions.device != first_layer_device:
            positions = positions.to(first_layer_device)
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        
        # 获取 context
        context = get_context()
        is_prefill = context.is_prefill
        
        for layer_id, layer in enumerate(self.layers):
            # ===== 1. 设备管理 =====
            layer_device = self.get_layer_device(layer_id)
            current_device = hidden_states.device
            
            if layer_device != current_device:
                hidden_states = hidden_states.to(layer_device)
                positions = positions.to(layer_device)
                if residual is not None:
                    residual = residual.to(layer_device)
            
            # ===== 2. 检查并应用优化策略 =====
            
            # 优先级 1: KV Head Split（最细粒度的优化）
            if layer_id in self.attention_offload and \
               self.attention_offload[layer_id].get('type') == 'kv_head_split':
                hidden_states, residual = self._forward_with_kv_head_split(
                    layer_id, layer, positions, hidden_states, residual, context
                )
            
            # 优先级 2: Attention Offload by Batch
            elif layer_id in self.attention_offload:
                hidden_states, residual = self._forward_with_attention_offload(
                    layer_id, layer, positions, hidden_states, residual, context
                )
            
            # 优先级 3: Layer Replication
            elif layer_id in self.replicas:
                hidden_states, residual = self._forward_with_layer_replication(
                    layer_id, layer, positions, hidden_states, residual, context
                )
            
            # 默认: 正常前向传播
            else:
                hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 4. 获取最后一层的设备
        last_layer_device = self.get_layer_device(len(self.layers) - 1)
        
        # 5. 确保 norm 在最后一层设备
        if self.norm.weight.device != last_layer_device:
            self.norm = self.norm.to(last_layer_device)
        
        # 6. 确保输出在最后一层设备
        if hidden_states.device != last_layer_device:
            hidden_states = hidden_states.to(last_layer_device)
        if residual is not None and residual.device != last_layer_device:
            residual = residual.to(last_layer_device)
        # 最后的 norm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # ========== 各种优化策略的执行逻辑 ==========

    def _forward_with_kv_head_split(
        self,
        layer_id: int,
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        context: Context
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 KV Head Split 的前向传播"""
        # Pre-attention norm
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        
        # 执行分片的 attention（调用外部实现）
        hidden_states = execute_kv_head_split_forward(
            layer_id, layer, positions, hidden_states, context,
            self.attention_offload[layer_id]
        )
        
        # Post-attention norm 和 MLP
        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        hidden_states = layer.mlp(hidden_states)
        
        return hidden_states, residual

    def _forward_with_attention_offload(
        self,
        layer_id: int,
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        context: Context
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 Attention Offload 的前向传播"""
        # Pre-attention norm
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        
        # 执行 offload 的 attention（调用外部实现）
        hidden_states = execute_attention_offload_forward(
            layer_id, layer, positions, hidden_states, context,
            self.attention_offload[layer_id],
            self._split_context_for_attention,
            self._sync_attention_kv_cache
        )
        
        # Post-attention norm 和 MLP
        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        hidden_states = layer.mlp(hidden_states)
        
        return hidden_states, residual

    def _forward_with_layer_replication(
        self,
        layer_id: int,
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        context: Context
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用层复制的前向传播"""
        # 调用外部实现（包含完整的层执行，含 norm 和 MLP）
        return execute_layer_replication_forward(
            layer_id, layer, positions, hidden_states, residual, context,
            self.replicas[layer_id],
            self.replica_devices[layer_id],
            self.replica_split_ratio[layer_id],
            self.replica_autotune.get(layer_id),
            self.get_layer_device(layer_id),
            self._sync_kv_cache_for_decode
        )

class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions) # 注意这里的input_ids不包含prefix caching命中的部分
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_device = hidden_states.device
        if self.lm_head.weight.device != hidden_device:
            self.lm_head = self.lm_head.to(hidden_device)
        logits = self.lm_head(hidden_states)
        return logits


# # ========== 便捷函数：用于模型优化配置 ==========

# def configure_model_optimization(
#     model: Qwen3ForCausalLM,
#     device_map: Optional[dict] = None,
#     replication_config: Optional[dict] = None,
#     attention_offload_config: Optional[dict] = None
# ) -> None:
#     """
#     便捷函数：配置模型优化
    
#     Args:
#         model: Qwen3ForCausalLM 模型
#         device_map: 层到设备的映射，例如 {0: 'cuda:0', 1: 'cuda:1'}
#         replication_config: 层复制配置，例如 {5: {'device': 'cuda:1', 'ratio': 0.6}}
#         attention_offload_config: Attention offload 配置
    
#     Example:
#         >>> model = Qwen3ForCausalLM(config)
#         >>> configure_model_optimization(
#         ...     model,
#         ...     device_map={0: 'cuda:0', 1: 'cuda:0', 2: 'cuda:1'},
#         ...     replication_config={5: {'device': 'cuda:2', 'ratio': 0.6}},
#         ...     attention_offload_config={10: {'device': 'cuda:3', 'type': 'kv_head'}}
#         ... )
#     """
#     qwen_model = model.model
    
#     # 1. 配置设备分布
#     if device_map:
#         qwen_model.set_layer_device_distribution(device_map)
    
#     # 2. 配置层复制
#     if replication_config:
#         for layer_id, cfg in replication_config.items():
#             qwen_model.replicate_layer_to_device(
#                 layer_id=layer_id,
#                 device=cfg['device'],
#                 split_ratio=cfg.get('ratio', 0.5)
#             )
#             if cfg.get('autotune', False):
#                 qwen_model.enable_replication_autotune(
#                     layer_id=layer_id,
#                     beta=cfg.get('beta', 0.2),
#                     min_ratio=cfg.get('min_ratio', 0.1),
#                     max_ratio=cfg.get('max_ratio', 0.9)
#                 )
    
#     # 3. 配置 Attention offload
#     if attention_offload_config:
#         for layer_id, cfg in attention_offload_config.items():
#             offload_type = cfg.get('type', 'batch')
#             if offload_type == 'kv_head':
#                 qwen_model.attention_offload_by_kv_head(
#                     layer_id=layer_id,
#                     offload_device=cfg['device'],
#                     split_kv_head_idx=cfg.get('split_idx'),
#                     enable_autotune=cfg.get('autotune', False),
#                     autotune_beta=cfg.get('beta', 0.3)
#                 )
#             else:  # batch
#                 qwen_model.attention_offload_by_batch(
#                     layer_id=layer_id,
#                     offload_device=cfg['device'],
#                     split_ratio=cfg.get('ratio', 0.5),
#                     enable_autotune=cfg.get('autotune', False),
#                     autotune_beta=cfg.get('beta', 0.3)
#                 )


# # ========== 使用示例 ==========

# if __name__ == "__main__":
#     # 创建模型
#     config = Qwen3Config()
#     model = Qwen3ForCausalLM(config)
    
#     # 方式 1: 直接使用 API
#     model.model.move_layer_to_device(0, 'cuda:0')
#     model.model.replicate_layer_to_device(5, 'cuda:1', split_ratio=0.6)
#     model.model.attention_offload_by_kv_head(10, 'cuda:2')
    
#     # 方式 2: 使用配置函数
#     configure_model_optimization(
#         model,
#         device_map={
#             0: 'cuda:0',
#             1: 'cuda:0',
#             2: 'cuda:1',
#             3: 'cuda:1',
#         },
#         replication_config={
#             5: {
#                 'device': 'cuda:2',
#                 'ratio': 0.6,
#                 'autotune': True,
#                 'beta': 0.2
#             }
#         },
#         attention_offload_config={
#             10: {
#                 'device': 'cuda:3',
#                 'type': 'kv_head',
#                 'split_idx': 4,
#                 'autotune': True
#             }
#         }
#     )
    
#     print("Qwen3 模型配置完成！")
#     print(f"层数: {len(model.model.layers)}")
#     print(f"复制的层: {list(model.model.replicas.keys())}")
#     print(f"Attention offload 的层: {list(model.model.attention_offload.keys())}")