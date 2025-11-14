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
        # ===== 添加空张量检查 =====
        if hidden_states.size(0) == 0:
            # 返回正确形状的空张量
            return torch.empty_like(hidden_states)
        # ==========================
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

# ==== CUDA Graph ====
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
        
        # CUDA Graph 相关
        self.use_cuda_graph = getattr(config, "use_cuda_graph", True)
        self.decode_graph_cache = {}  # 缓存不同 batch size 的 graph
        self.graph_pool_handle = None
        
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
            
            # 创建持久化的 streams
            self.stream0 = torch.cuda.Stream(device=self.layers[0].self_attn.qkv_proj.weight.device)
            self.stream1 = torch.cuda.Stream(device=self._dp_target_device)
        else:
            self.dp_layers = None
            self.stream0 = None
            self.stream1 = None

    def _capture_decode_graph(self, B, positions, hidden_states, residual, ctx):
        """捕获 decode 阶段的 CUDA Graph"""
        import torch.cuda.nvtx as nvtx
        # ===== 添加检查 =====
        if B < 2:
            print(f"Batch size {B} too small for graph capture, need at least 2")
            return None
        # ====================
        start = self.local_dp_start
        end = self.local_dp_end
        mid = B // 2
        dev0 = hidden_states.device
        
        # 分割数据
        pos0, pos1 = positions[:mid], positions[mid:]
        hs0, hs1 = hidden_states[:mid], hidden_states[mid:]
        res0 = None if residual is None else residual[:mid]
        res1 = None if residual is None else residual[mid:]
        
        sm0 = ctx.slot_mapping[:mid].clone()
        sm1 = ctx.slot_mapping[mid:].clone()
        cl0 = ctx.context_lens[:mid].clone()
        cl1 = ctx.context_lens[mid:].clone()
        
        # 获取最大 block 表大小并创建固定大小的 buffer
        if ctx.block_tables is not None:
            cur_max_blocks = ctx.block_tables.shape[1]
            max_blocks = int(cur_max_blocks * 10)
            # 创建固定大小的 block_tables buffer
            bt0 = torch.zeros((mid, max_blocks), dtype=ctx.block_tables.dtype, device=ctx.block_tables.device)
            bt1 = torch.zeros((B - mid, max_blocks), dtype=ctx.block_tables.dtype, device=ctx.block_tables.device)
            # 复制当前有效数据
            bt0[:, :ctx.block_tables.shape[1]].copy_(ctx.block_tables[:mid])
            bt1[:, :ctx.block_tables.shape[1]].copy_(ctx.block_tables[mid:])
        else:
            max_blocks = None
            bt0 = None
            bt1 = None
        
        # 准备静态 buffer
        static_pos0 = pos0.clone()
        static_hs0 = hs0.clone()
        static_res0 = res0.clone() if res0 is not None else None
        
        static_pos1 = pos1.clone()
        static_hs1 = hs1.clone()
        static_res1 = res1.clone() if res1 is not None else None
        
        static_sm1 = sm1.clone()
        static_cl1 = cl1.clone()
        static_bt1 = bt1.clone() if bt1 is not None else None
        
        # Stream1 的 GPU 侧 buffer
        static_pos1_d = static_pos1.to(self._dp_target_device)
        static_hs1_d = static_hs1.to(self._dp_target_device)
        static_res1_d = static_res1.to(self._dp_target_device) if static_res1 is not None else None
        static_sm1_d = static_sm1.to(self._dp_target_device)
        static_cl1_d = static_cl1.to(self._dp_target_device)
        static_bt1_d = static_bt1.to(self._dp_target_device) if static_bt1 is not None else None
        
        # Warmup (至少 3 次)
        print(f"Warming up CUDA Graph for batch size {B} with max_blocks={max_blocks}...")
        for _ in range(3):
            with torch.cuda.stream(self.stream0):
                set_context(False, slot_mapping=sm0, context_lens=cl0, block_tables=bt0)
                tmp_hs0, tmp_res0 = static_hs0.clone(), static_res0.clone() if static_res0 is not None else None
                for j in range(start, end):
                    tmp_hs0, tmp_res0 = self.layers[j](static_pos0, tmp_hs0, tmp_res0)
            
            with torch.cuda.stream(self.stream1):
                static_hs1_d.copy_(static_hs1, non_blocking=True)
                if static_res1_d is not None:
                    static_res1_d.copy_(static_res1, non_blocking=True)
                
                set_context(False, slot_mapping=static_sm1_d, context_lens=static_cl1_d, block_tables=static_bt1_d)
                tmp_hs1_d, tmp_res1_d = static_hs1_d.clone(), static_res1_d.clone() if static_res1_d is not None else None
                for j in range(start, end):
                    tmp_hs1_d, tmp_res1_d = self.dp_layers[j - start](static_pos1_d, tmp_hs1_d, tmp_res1_d)
            
            self.stream0.synchronize()
            self.stream1.synchronize()
        
        torch.cuda.synchronize()
        
        # 录制 Graph0 (stream0)
        print("Capturing graph0...")
        graph0 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph0, stream=self.stream0, pool=self.graph_pool_handle):
            set_context(False, slot_mapping=sm0, context_lens=cl0, block_tables=bt0)
            out_hs0 = static_hs0
            out_res0 = static_res0
            for j in range(start, end):
                out_hs0, out_res0 = self.layers[j](static_pos0, out_hs0, out_res0)
        
        # 录制 Graph1 (stream1)
        print("Capturing graph1...")
        graph1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph1, stream=self.stream1, pool=self.graph_pool_handle):
            set_context(False, slot_mapping=static_sm1_d, context_lens=static_cl1_d, block_tables=static_bt1_d)
            out_hs1_d = static_hs1_d
            out_res1_d = static_res1_d
            for j in range(start, end):
                out_hs1_d, out_res1_d = self.dp_layers[j - start](static_pos1_d, out_hs1_d, out_res1_d)
        
        torch.cuda.synchronize()
        print(f"Graph capture completed for batch size {B}")
        
        return {
            'graph0': graph0,
            'graph1': graph1,
            'max_blocks': max_blocks,  # 保存最大 block 数
            'static_inputs': {
                'pos0': static_pos0,
                'hs0': static_hs0,
                'res0': static_res0,
                'pos1': static_pos1,
                'hs1': static_hs1,
                'res1': static_res1,
                'hs1_d': static_hs1_d,
                'res1_d': static_res1_d,
                'sm0': sm0,
                'cl0': cl0,
                'bt0': bt0,
            },
            'static_outputs': {
                'hs0': out_hs0,
                'res0': out_res0,
                'hs1_d': out_hs1_d,
                'res1_d': out_res1_d,
            }
        }

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
            hidden_states, residual = self.layers[i](positions, hidden_states, residual)
        
        ctx = get_context()
        dev0 = hidden_states.device
        
        if ctx.is_prefill:
            nvtx.range_push("prefill_data_prepare")
            B = ctx.cu_seqlens_q.numel() - 1

            # ===== 添加空batch检查 =====
            if B == 0:
                nvtx.range_pop()
                # 跳过 replica 处理，直接执行后续层
                for i in range(end, len(self.layers)):
                    hidden_states, residual = self.layers[i](positions, hidden_states, residual)
                hidden_states, _ = self.norm(hidden_states, residual)
                return hidden_states
            # ==========================
            
            # ===== 确保 mid 不会导致空shard =====
            mid = max(1, min(B - 1, B // 2))  # 确保 mid 在 [1, B-1] 范围内
            # ====================================
            
            n0 = ctx.cu_seqlens_q[mid].item()
            
            # ===== 添加token数检查 =====
            if n0 == 0 or n0 == positions.size(0):
                # 如果分割导致某一边为空，退化为单路执行
                nvtx.range_pop()
                for j in range(start, end):
                    hidden_states, residual = self.layers[j](positions, hidden_states, residual)
                for i in range(end, len(self.layers)):
                    hidden_states, residual = self.layers[i](positions, hidden_states, residual)
                hidden_states, _ = self.norm(hidden_states, residual)
                return hidden_states
            # ==========================

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
            
            # 在 default stream 记录开始事件
            start_event = torch.cuda.Event()
            start_event.record(torch.cuda.current_stream(dev0))
            
            # Stream 0: 处理 shard 0 (原始 layers) on dev0
            with torch.cuda.stream(self.stream0):
                self.stream0.wait_event(start_event)
                
                torch.cuda.nvtx.range_push("stream0_shard0_total")
                set_context(True, cuq0, cuk0, ctx.max_seqlen_q, ctx.max_seqlen_k, sm0, None, bt0)
                
                hs0_d, res0_d = hs0, res0
                for j in range(start, end):
                    hs0_d, res0_d = self.layers[j](pos0, hs0_d, res0_d)
                
                torch.cuda.nvtx.range_pop()
            
            # Stream 1: 处理 shard 1 (dp_layers) on dp_device
            with torch.cuda.stream(self.stream1):
                self.stream1.wait_event(start_event)
                
                torch.cuda.nvtx.range_push("stream1_shard1_total")
                
                torch.cuda.nvtx.range_push("stream1_p2p_transfer")
                cuq1_d = cuq1.to(self._dp_target_device, non_blocking=True)
                cuk1_d = cuk1.to(self._dp_target_device, non_blocking=True)
                sm1_d = sm1.to(self._dp_target_device, non_blocking=True)
                bt1_d = bt1.to(self._dp_target_device, non_blocking=True) if bt1 is not None else None
                pos1_d = pos1.to(self._dp_target_device, non_blocking=True)
                hs1_d = hs1.to(self._dp_target_device, non_blocking=True)
                res1_d = None if res1 is None else res1.to(self._dp_target_device, non_blocking=True)
                torch.cuda.nvtx.range_pop()
                
                set_context(True, cuq1_d, cuk1_d, ctx.max_seqlen_q, ctx.max_seqlen_k, sm1_d, None, bt1_d)
                
                for j in range(start, end):
                    hs1_d, res1_d = self.dp_layers[j - start](pos1_d, hs1_d, res1_d)
                
                torch.cuda.nvtx.range_push("stream1_p2p_transfer_back")
                hs1_back = hs1_d.to(dev0, non_blocking=True)
                res1_back = None if res1_d is None else res1_d.to(dev0, non_blocking=True)
                torch.cuda.nvtx.range_pop()
                
                torch.cuda.nvtx.range_pop()  # stream1_shard1_total
            
            # 等待两个 stream 完成
            nvtx.range_push("wait_both_streams")
            self.stream0.synchronize()
            self.stream1.synchronize()
            nvtx.range_pop()
            
            # 合并结果
            nvtx.range_push("merge_results")
            hidden_states = torch.cat([hs0_d, hs1_back], dim=0)
            residual = None if res0_d is None else torch.cat([res0_d, res1_back], dim=0)
            nvtx.range_pop()
            
            # 恢复 context
            set_context(*ctx_orig)
            
            nvtx.range_pop()  # prefill_parallel_execution
            
        else:  # decode 阶段 - 使用 CUDA Graph
            B = positions.size(0)
            # mid = B // 2
            # ===== 添加空batch检查 =====
            if B == 0:
                # 空batch，直接返回
                for i in range(end, len(self.layers)):
                    hidden_states, residual = self.layers[i](positions, hidden_states, residual)
                hidden_states, _ = self.norm(hidden_states, residual)
                return hidden_states
            # ==========================
            
            # ===== 确保 mid 不会导致空shard =====
            if B == 1:
                # 只有1个样本，不分割
                for j in range(start, end):
                    hidden_states, residual = self.layers[j](positions, hidden_states, residual)
                for i in range(end, len(self.layers)):
                    hidden_states, residual = self.layers[i](positions, hidden_states, residual)
                hidden_states, _ = self.norm(hidden_states, residual)
                return hidden_states
            
            mid = max(1, min(B - 1, B // 2))  # 确保 mid 在 [1, B-1] 范围内
            # ====================================
            if self.use_cuda_graph:
                nvtx.range_push("decode_graph_execution")
                
                # 检查是否已经有该 batch size 的 graph
                if B not in self.decode_graph_cache:
                    nvtx.range_push("capture_graph")
                    self.decode_graph_cache[B] = self._capture_decode_graph(
                        B, positions, hidden_states, residual, ctx
                    )
                    nvtx.range_pop()
                
                graph_data = self.decode_graph_cache[B]
                
                # 保存原始 context
                ctx_orig = (ctx.is_prefill, None, None, 0, 0, 
                            ctx.slot_mapping, ctx.context_lens, ctx.block_tables)
                
                # 更新静态输入 buffer
                nvtx.range_push("update_graph_inputs")
                
                # Shard 0 inputs
                graph_data['static_inputs']['pos0'].copy_(positions[:mid])
                graph_data['static_inputs']['hs0'].copy_(hidden_states[:mid])
                if residual is not None:
                    graph_data['static_inputs']['res0'].copy_(residual[:mid])
                
                # Shard 1 inputs
                graph_data['static_inputs']['pos1'].copy_(positions[mid:])
                graph_data['static_inputs']['hs1'].copy_(hidden_states[mid:])
                if residual is not None:
                    graph_data['static_inputs']['res1'].copy_(residual[mid:])
                
                # 更新 Shard 1 的 GPU 侧 buffer
                with torch.cuda.stream(self.stream1):
                    graph_data['static_inputs']['hs1_d'].copy_(
                        graph_data['static_inputs']['hs1'], non_blocking=True
                    )
                    if graph_data['static_inputs']['res1_d'] is not None:
                        graph_data['static_inputs']['res1_d'].copy_(
                            graph_data['static_inputs']['res1'], non_blocking=True
                        )
                
                # 更新 context metadata
                sm0 = ctx.slot_mapping[:mid]
                sm1 = ctx.slot_mapping[mid:]
                cl0 = ctx.context_lens[:mid]
                cl1 = ctx.context_lens[mid:]
                
                # 处理 block_tables（支持大小变化）
                if ctx.block_tables is not None:
                    bt0_current = ctx.block_tables[:mid]
                    bt1_current = ctx.block_tables[mid:]
                    
                    # 检查是否需要重新捕获 graph（block 数量增加超过预留空间）
                    if bt0_current.shape[1] > graph_data['max_blocks']:
                        print(f"Block table size increased from {graph_data['max_blocks']} to {bt0_current.shape[1]}, recapturing graph...")
                        nvtx.range_pop()  # update_graph_inputs
                        nvtx.range_pop()  # decode_graph_execution
                        
                        # 重新捕获 graph
                        self.decode_graph_cache[B] = self._capture_decode_graph(
                            B, positions, hidden_states, residual, ctx
                        )
                        graph_data = self.decode_graph_cache[B]
                        
                        nvtx.range_push("decode_graph_execution")
                        nvtx.range_push("update_graph_inputs")
                        
                        # 重新准备输入
                        graph_data['static_inputs']['pos0'].copy_(positions[:mid])
                        graph_data['static_inputs']['hs0'].copy_(hidden_states[:mid])
                        if residual is not None:
                            graph_data['static_inputs']['res0'].copy_(residual[:mid])
                        
                        graph_data['static_inputs']['pos1'].copy_(positions[mid:])
                        graph_data['static_inputs']['hs1'].copy_(hidden_states[mid:])
                        if residual is not None:
                            graph_data['static_inputs']['res1'].copy_(residual[mid:])
                        
                        with torch.cuda.stream(self.stream1):
                            graph_data['static_inputs']['hs1_d'].copy_(
                                graph_data['static_inputs']['hs1'], non_blocking=True
                            )
                            if graph_data['static_inputs']['res1_d'] is not None:
                                graph_data['static_inputs']['res1_d'].copy_(
                                    graph_data['static_inputs']['res1'], non_blocking=True
                                )
                        
                        bt0_current = ctx.block_tables[:mid]
                        bt1_current = ctx.block_tables[mid:]
                    
                    # 复制 block_tables（只复制有效部分，其余保持为 0）
                    bt0 = graph_data['static_inputs']['bt0']
                    bt0.zero_()  # 先清零
                    bt0[:, :bt0_current.shape[1]].copy_(bt0_current)
                else:
                    bt0 = None
                
                graph_data['static_inputs']['sm0'].copy_(sm0)
                graph_data['static_inputs']['cl0'].copy_(cl0)
                
                nvtx.range_pop()
                
                # Replay graphs - 真正的并行！
                nvtx.range_push("replay_graphs")
                graph_data['graph0'].replay()
                graph_data['graph1'].replay()
                nvtx.range_pop()
                
                # 在 replay 后手动执行 P2P 传输
                nvtx.range_push("p2p_transfer_back")
                with torch.cuda.stream(self.stream1):
                    hs1_back = graph_data['static_outputs']['hs1_d'].to(dev0, non_blocking=True)
                    res1_back = graph_data['static_outputs']['res1_d'].to(dev0, non_blocking=True) if graph_data['static_outputs']['res1_d'] is not None else None
                nvtx.range_pop()
                
                # 等待 graphs 和传输完成
                nvtx.range_push("wait_graphs")
                self.stream0.synchronize()
                self.stream1.synchronize()
                nvtx.range_pop()
                
                # 从静态输出 buffer 和传输结果合并
                nvtx.range_push("merge_results")
                hidden_states = torch.cat([
                    graph_data['static_outputs']['hs0'],
                    hs1_back
                ], dim=0)
                
                if graph_data['static_outputs']['res0'] is not None:
                    residual = torch.cat([
                        graph_data['static_outputs']['res0'],
                        res1_back
                    ], dim=0)
                else:
                    residual = None
                nvtx.range_pop()
                
                # 恢复原始 context
                nvtx.range_push("restore_context")
                set_context(*ctx_orig)
                nvtx.range_pop()
                
                nvtx.range_pop()  # decode_graph_execution
                
            else:  # Fallback: 不使用 CUDA Graph
                nvtx.range_push("decode_data_prepare")
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
                
                start_event = torch.cuda.Event()
                start_event.record(torch.cuda.current_stream(dev0))
                
                # Stream 0
                with torch.cuda.stream(self.stream0):
                    self.stream0.wait_event(start_event)
                    
                    torch.cuda.nvtx.range_push("stream0_shard0_total")
                    set_context(False, slot_mapping=sm0, context_lens=cl0, block_tables=bt0)
                    
                    hs0_d, res0_d = hs0, res0
                    for j in range(start, end):
                        hs0_d, res0_d = self.layers[j](pos0, hs0_d, res0_d)
                    
                    torch.cuda.nvtx.range_pop()
                
                # Stream 1
                with torch.cuda.stream(self.stream1):
                    self.stream1.wait_event(start_event)
                    
                    torch.cuda.nvtx.range_push("stream1_shard1_total")
                    
                    torch.cuda.nvtx.range_push("stream1_p2p_transfer")
                    sm1_d = sm1.to(self._dp_target_device, non_blocking=True)
                    cl1_d = cl1.to(self._dp_target_device, non_blocking=True)
                    bt1_d = bt1.to(self._dp_target_device, non_blocking=True) if bt1 is not None else None
                    pos1_d = pos1.to(self._dp_target_device, non_blocking=True)
                    hs1_d = hs1.to(self._dp_target_device, non_blocking=True)
                    res1_d = None if res1 is None else res1.to(self._dp_target_device, non_blocking=True)
                    torch.cuda.nvtx.range_pop()
                    
                    set_context(False, slot_mapping=sm1_d, context_lens=cl1_d, block_tables=bt1_d)
                    
                    for j in range(start, end):
                        hs1_d, res1_d = self.dp_layers[j - start](pos1_d, hs1_d, res1_d)
                    
                    # 在 stream1 内部传回
                    torch.cuda.nvtx.range_push("stream1_p2p_transfer_back")
                    hs1_back = hs1_d.to(dev0, non_blocking=True)
                    res1_back = None if res1_d is None else res1_d.to(dev0, non_blocking=True)
                    torch.cuda.nvtx.range_pop()
                    
                    torch.cuda.nvtx.range_pop()
                
                nvtx.range_push("wait_both_streams")
                self.stream0.synchronize()
                self.stream1.synchronize()
                nvtx.range_pop()
                
                nvtx.range_push("merge_results")
                hidden_states = torch.cat([hs0_d, hs1_back], dim=0)
                residual = None if res0_d is None else torch.cat([res0_d, res1_back], dim=0)
                nvtx.range_pop()
                
                set_context(*ctx_orig)
                
                nvtx.range_pop()  # decode_parallel_execution
        
        # 后面的层
        for i in range(end, len(self.layers)):
            hidden_states, residual = self.layers[i](positions, hidden_states, residual)
        
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