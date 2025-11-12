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

from HBserve.models import register_model  # ← 导入装饰器

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


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.local_dp_start = getattr(config, "local_dp_start_layer", None)
        self.local_dp_end = getattr(config, "local_dp_end_layer", None)
        self.local_dp_device = getattr(config, "local_dp_device", None)
        self.local_dp_enabled = (
            self.local_dp_start is not None and
            self.local_dp_end is not None and
            self.local_dp_device is not None and
            0 <= self.local_dp_start < self.local_dp_end <= len(self.layers)
        )
        if self.local_dp_enabled:
            target_device = self.local_dp_device
            if isinstance(target_device, int):
                target_device = f"cuda:{target_device}"
            self._dp_target_device = torch.device(target_device)
            dp_layers = []
            for i in range(self.local_dp_start, self.local_dp_end):
                replica = copy.deepcopy(self.layers[i]).to(self._dp_target_device)
                for m in replica.modules():
                    if hasattr(m, "k_cache") and hasattr(m, "v_cache"):
                        setattr(m, "is_replica", True)
                        setattr(m, "replica_device", self._dp_target_device)
                dp_layers.append(replica)
            self.dp_layers = nn.ModuleList(dp_layers)
        else:
            self.dp_layers = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        import torch.cuda.nvtx as nvtx
        
        if not self.local_dp_enabled:
            for layer in self.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        
        start = self.local_dp_start
        end = self.local_dp_end
        
        # 前面的层
        
        for i in range(start):
            nvtx.range_push("pre_parallel_layers")
            hidden_states, residual = self.layers[i](positions, hidden_states, residual)
            nvtx.range_pop()
        
        ctx = get_context()
        dev0 = hidden_states.device
        
        # 创建两个 CUDA stream
        stream0 = torch.cuda.Stream(device=dev0)
        stream1 = torch.cuda.Stream(device=self._dp_target_device)
        
        if ctx.is_prefill:
            nvtx.range_push("prefill_data_prepare")
            B = ctx.cu_seqlens_q.numel() - 1
            mid = B // 2
            n0 = ctx.cu_seqlens_q[mid].item()
            
            # 分割数据
            pos0, pos1 = positions[:n0], positions[n0:]
            hs0, hs1 = hidden_states[:n0], hidden_states[n0:]
            res0 = None if residual is None else residual[:n0]
            res1 = None if residual is None else residual[n0:]
            
            # 分割 context
            cuq0 = ctx.cu_seqlens_q[:mid+1] - ctx.cu_seqlens_q[0]
            cuk0 = ctx.cu_seqlens_k[:mid+1] - ctx.cu_seqlens_k[0]
            cuq1 = ctx.cu_seqlens_q[mid:] - ctx.cu_seqlens_q[mid]
            cuk1 = ctx.cu_seqlens_k[mid:] - ctx.cu_seqlens_k[mid]
            sm0 = ctx.slot_mapping[:n0]
            sm1 = ctx.slot_mapping[n0:]
            bt0 = ctx.block_tables[:mid] if ctx.block_tables is not None else None
            bt1 = ctx.block_tables[mid:] if ctx.block_tables is not None else None
            
            ctx_orig = (ctx.is_prefill, ctx.cu_seqlens_q, ctx.cu_seqlens_k, 
                        ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.slot_mapping, 
                        ctx.context_lens, ctx.block_tables)
            nvtx.range_pop()
            
            nvtx.range_push("prefill_parallel_execution")
            
            # Stream 0: 处理 shard 0 (原始 layers)
            with torch.cuda.stream(stream0):
                nvtx.range_push("stream0_prefill_shard0")
                nvtx.range_push("stream0_set_context")
                set_context(True, cuq0, cuk0, ctx.max_seqlen_q, ctx.max_seqlen_k, sm0, None, bt0)
                nvtx.range_pop()
                
                hs0_d, res0_d = hs0, res0
                for j in range(start, end):
                    nvtx.range_push(f"stream0_dp_layer_{j}")
                    hs0_d, res0_d = self.layers[j](pos0, hs0_d, res0_d)
                    nvtx.range_pop()
                nvtx.range_pop()  # stream0_prefill_shard0
            
            # Stream 1: 处理 shard 1 (dp_layers) 
            with torch.cuda.stream(stream1):
                nvtx.range_push("stream1_prefill_shard1")
                
                nvtx.range_push("stream1_h2d_transfer")
                # 数据传输到 dp device
                cuq1_d = cuq1.to(self._dp_target_device, non_blocking=True)
                cuk1_d = cuk1.to(self._dp_target_device, non_blocking=True)
                sm1_d = sm1.to(self._dp_target_device, non_blocking=True)
                bt1_d = bt1.to(self._dp_target_device, non_blocking=True) if bt1 is not None else None
                pos1_d = pos1.to(self._dp_target_device, non_blocking=True)
                hs1_d = hs1.to(self._dp_target_device, non_blocking=True)
                res1_d = None if res1 is None else res1.to(self._dp_target_device, non_blocking=True)
                nvtx.range_pop()
                
                nvtx.range_push("stream1_set_context")
                set_context(True, cuq1_d, cuk1_d, ctx.max_seqlen_q, ctx.max_seqlen_k, sm1_d, None, bt1_d)
                nvtx.range_pop()
                
                for j in range(start, end):
                    nvtx.range_push(f"stream1_dp_layer_{j}")
                    hs1_d, res1_d = self.dp_layers[j - start](pos1_d, hs1_d, res1_d)
                    nvtx.range_pop()
                nvtx.range_pop()  # stream1_prefill_shard1
            
            # 等待两个 stream 完成
            nvtx.range_push("wait_streams_sync")
            stream0.synchronize()
            stream1.synchronize()
            nvtx.range_pop()
            
            # 合并结果
            nvtx.range_push("merge_results_d2h")
            hs1_back = hs1_d.to(dev0, non_blocking=False)
            res1_back = None if res1_d is None else res1_d.to(dev0, non_blocking=False)
            
            hidden_states = torch.cat([hs0_d, hs1_back], dim=0)
            residual = None if res0_d is None else torch.cat([res0_d, res1_back], dim=0)
            nvtx.range_pop()
            
            # 恢复 context
            nvtx.range_push("restore_context")
            set_context(*ctx_orig)
            nvtx.range_pop()
            
            nvtx.range_pop()  # prefill_parallel_execution
            
        else:  # decode 阶段
            nvtx.range_push("decode_data_prepare")
            B = positions.size(0)
            mid = B // 2
            
            pos0, pos1 = positions[:mid], positions[mid:]
            hs0, hs1 = hidden_states[:mid], hidden_states[mid:]
            res0 = None if residual is None else residual[:mid]
            res1 = None if residual is None else residual[mid:]
            
            sm0 = ctx.slot_mapping[:mid]
            sm1 = ctx.slot_mapping[mid:]
            cl0 = ctx.context_lens[:mid]
            cl1 = ctx.context_lens[mid:]
            bt0 = ctx.block_tables[:mid] if ctx.block_tables is not None else None
            bt1 = ctx.block_tables[mid:] if ctx.block_tables is not None else None
            
            ctx_orig = (ctx.is_prefill, None, None, 0, 0, ctx.slot_mapping, 
                        ctx.context_lens, ctx.block_tables)
            nvtx.range_pop()
            
            nvtx.range_push("decode_parallel_execution")
            
            # Stream 0
            with torch.cuda.stream(stream0):
                nvtx.range_push("stream0_decode_shard0")
                nvtx.range_push("stream0_set_context")
                set_context(False, slot_mapping=sm0, context_lens=cl0, block_tables=bt0)
                nvtx.range_pop()
                
                hs0_d, res0_d = hs0, res0
                for j in range(start, end):
                    nvtx.range_push(f"stream0_layer_{j}")
                    hs0_d, res0_d = self.layers[j](pos0, hs0_d, res0_d)
                    nvtx.range_pop()
                nvtx.range_pop()  # stream0_decode_shard0
            
            # Stream 1
            with torch.cuda.stream(stream1):
                nvtx.range_push("stream1_decode_shard1")
                
                nvtx.range_push("stream1_h2d_transfer")
                sm1_d = sm1.to(self._dp_target_device, non_blocking=True)
                cl1_d = cl1.to(self._dp_target_device, non_blocking=True)
                bt1_d = bt1.to(self._dp_target_device, non_blocking=True) if bt1 is not None else None
                pos1_d = pos1.to(self._dp_target_device, non_blocking=True)
                hs1_d = hs1.to(self._dp_target_device, non_blocking=True)
                res1_d = None if res1 is None else res1.to(self._dp_target_device, non_blocking=True)
                nvtx.range_pop()
                
                nvtx.range_push("stream1_set_context")
                set_context(False, slot_mapping=sm1_d, context_lens=cl1_d, block_tables=bt1_d)
                nvtx.range_pop()
                
                for j in range(start, end):
                    nvtx.range_push(f"stream1_dp_layer_{j}")
                    hs1_d, res1_d = self.dp_layers[j - start](pos1_d, hs1_d, res1_d)
                    nvtx.range_pop()
                nvtx.range_pop()  # stream1_decode_shard1
            
            nvtx.range_push("wait_streams_sync")
            stream0.synchronize()
            stream1.synchronize()
            nvtx.range_pop()
            
            nvtx.range_push("merge_results_d2h")
            hs1_back = hs1_d.to(dev0, non_blocking=False)
            res1_back = None if res1_d is None else res1_d.to(dev0, non_blocking=False)
            
            hidden_states = torch.cat([hs0_d, hs1_back], dim=0)
            residual = None if res0_d is None else torch.cat([res0_d, res1_back], dim=0)
            nvtx.range_pop()
            
            nvtx.range_push("restore_context")
            set_context(*ctx_orig)
            nvtx.range_pop()
            
            nvtx.range_pop()  # decode_parallel_execution
        
        # 后面的层
        
        for i in range(end, len(self.layers)):
            nvtx.range_push("post_parallel_layers")
            hidden_states, residual = self.layers[i](positions, hidden_states, residual)
            nvtx.range_pop()
        
        nvtx.range_push("final_norm")
        hidden_states, _ = self.norm(hidden_states, residual)
        nvtx.range_pop()
        
        return hidden_states

        
    
@register_model("qwen3") 
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
        self.local_dp_enabled = getattr(self.model, "local_dp_enabled", False)

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
