"""
Llama 模型实现 - 使用 Mixin 模式重构
支持层迁移、层复制、Attention Offload 等优化功能
"""

import torch
import copy
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
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

from HBserve.models import register_model  # ← 导入装饰器


class LlamaAttention(nn.Module):

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
        use_qk_norm: bool = False,  # ← 新增：是否使用 Q/K norm
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
        self.use_qk_norm = use_qk_norm  # ← 保存配置

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
        
        # ========== 修改：Q/K norm 改为可选 ==========
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # ========== 修改：Q/K norm 改为可选 ==========
        if self.use_qk_norm:
            q_by_head = q.view(-1, self.num_heads, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
        
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class LlamaMLP(nn.Module):

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


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            use_qk_norm=getattr(config, 'use_qk_norm', False),  # ← 新增：从配置读取
        )
        self.mlp = LlamaMLP(
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
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(ModelOptimizationMixin, nn.Module):
    """
    Llama 模型 - 使用 Mixin 模式重构
    
    继承 ModelOptimizationMixin 后，自动获得以下优化功能：
    1. 层设备管理（move_layer_to_device, get_layer_device 等）
    2. 层复制（replicate_layer_to_device, clear_layer_replication 等）
    3. Attention Offload（attention_offload_by_batch, attention_offload_by_kv_head 等）
    """

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        # 先初始化 nn.Module
        nn.Module.__init__(self)
        # 再初始化 Mixin（这会初始化所有优化相关的字典）
        ModelOptimizationMixin.__init__(self)
        
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # ========== 实现 Mixin 需要的抽象方法 ==========
    
    def _create_decoder_layer(self):
        """创建一个新的 decoder layer 实例（LayerReplicationMixin 需要）"""
        return LlamaDecoderLayer(self.config)
    
    def _create_attention_module(self):
        """创建一个新的 attention 模块实例（AttentionOffloadMixin 需要）"""
        return LlamaAttention(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            max_position=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            qkv_bias=getattr(self.config, 'attention_bias', False),
            head_dim=getattr(self.config, 'head_dim', None),
            rope_theta=getattr(self.config, "rope_theta", 10000),
            rope_scaling=getattr(self.config, "rope_scaling", None),
            use_qk_norm=getattr(self.config, 'use_qk_norm', False),
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
            self.get_layer_device(layer_id),
            self._sync_kv_cache_for_decode
        )


# ========== 修复：装饰器改为 "llama" ==========
@register_model("llama")
class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig
    ) -> None:
        super().__init__()
        
        # ========== 新增：标准化配置 ==========
        self._standardize_config(config)
        
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    @staticmethod
    def _standardize_config(config: LlamaConfig):
        """
        标准化 Llama 配置，添加缺失的属性
        """
        print("🔧 标准化 Llama 配置...")
        
        # 1. head_dim（通常 Llama 配置中有，但为了保险）
        if not hasattr(config, 'head_dim'):
            config.head_dim = config.hidden_size // config.num_attention_heads
            print(f"  ✓ 计算 head_dim = {config.head_dim}")
        
        # 2. attention_bias（标准 Llama 不使用 bias）
        if not hasattr(config, 'attention_bias'):
            config.attention_bias = False
            print(f"  ✓ 设置 attention_bias = {config.attention_bias}")
        
        # 3. rope_theta（Llama 1/2 是 10000，Llama 3 是 500000）
        if not hasattr(config, 'rope_theta'):
            config.rope_theta = 10000.0
            print(f"  ✓ 设置 rope_theta = {config.rope_theta}")
        
        # 4. use_qk_norm（标准 Llama 不使用）
        if not hasattr(config, 'use_qk_norm'):
            config.use_qk_norm = False
            print(f"  ✓ 设置 use_qk_norm = {config.use_qk_norm} (标准 Llama 不使用)")
        
        print("✅ Llama 配置标准化完成")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
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