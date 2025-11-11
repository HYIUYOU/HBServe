"""
OPT 模型实现 - 使用 Mixin 模式重构
支持层迁移、层复制、Attention Offload 等优化功能
"""

import torch
import copy
from torch import nn
import torch.distributed as dist
from transformers import OPTConfig
from typing import Tuple, Optional

from HBserve.layers.activation import SiluAndMul
from HBserve.layers.attention import Attention
from HBserve.layers.layernorm import RMSNorm
from HBserve.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from HBserve.layers.rotary_embedding import get_rope
from HBserve.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from HBserve.utils.context import get_context, set_context, Context


from HBserve.models import register_model  # ← 导入装饰器


class OPTAttention(nn.Module):

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
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
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


class OPTMLP(nn.Module):

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
        assert hidden_act == "relu"  # ← OPT 使用 relu，不是 silu
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
    ) -> None:
        super().__init__()
        self.self_attn = OPTAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = OPTMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.ffn_dim,  # ← OPT 用 ffn_dim，不是 intermediate_size
            hidden_act=config.activation_function,  # ← OPT 用 activation_function
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


class OPTModel(nn.Module):
    """
    OPT 模型
    """

    def __init__(
        self,
        config: OPTConfig,
    ) -> None:
        nn.Module.__init__(self)
        
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            OPTDecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # ========== 前向传播 ==========

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_model("opt")
class OPTForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: OPTConfig
    ) -> None:
        super().__init__()
        
        # ========== 关键添加：标准化 OPT 配置 ==========
        self._standardize_config(config)
        
        self.config = config
        self.model = OPTModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    @staticmethod
    def _standardize_config(config: OPTConfig):
        """
        标准化 OPT 配置，添加缺失的属性
        使其与 Qwen3 等模型的配置格式兼容
        """
        print("🔧 标准化 OPT 配置...")
        
        # 1. num_key_value_heads (OPT 使用 MHA，没有这个属性)
        if not hasattr(config, 'num_key_value_heads'):
            config.num_key_value_heads = config.num_attention_heads
            print(f"  ✓ 设置 num_key_value_heads = {config.num_key_value_heads} (MHA)")
        
        # 2. head_dim
        if not hasattr(config, 'head_dim'):
            config.head_dim = config.hidden_size // config.num_attention_heads
            print(f"  ✓ 计算 head_dim = {config.head_dim}")
        
        # 3. intermediate_size (OPT 使用 ffn_dim)
        if not hasattr(config, 'intermediate_size'):
            config.intermediate_size = config.ffn_dim
            print(f"  ✓ 设置 intermediate_size = {config.intermediate_size} (from ffn_dim)")
        
        # 4. hidden_act (OPT 使用 activation_function)
        if not hasattr(config, 'hidden_act'):
            config.hidden_act = config.activation_function
            print(f"  ✓ 设置 hidden_act = {config.hidden_act}")
        
        # 5. rope_theta (OPT 可能不使用 RoPE，但为了兼容性添加)
        if not hasattr(config, 'rope_theta'):
            config.rope_theta = 10000.0
            print(f"  ✓ 设置 rope_theta = {config.rope_theta} (默认值)")
        
        # 6. rope_scaling
        if not hasattr(config, 'rope_scaling'):
            config.rope_scaling = None
        
        # 7. rms_norm_eps (OPT 可能使用 layer_norm_eps)
        if not hasattr(config, 'rms_norm_eps'):
            # OPT 通常没有这个字段，使用默认值
            config.rms_norm_eps = 1e-6
            print(f"  ✓ 设置 rms_norm_eps = {config.rms_norm_eps} (默认值)")
        
        # 8. attention_bias (OPT 使用 enable_bias)
        if not hasattr(config, 'attention_bias'):
            config.attention_bias = getattr(config, 'enable_bias', False)
            print(f"  ✓ 设置 attention_bias = {config.attention_bias}")
        
        print("✅ OPT 配置标准化完成")

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