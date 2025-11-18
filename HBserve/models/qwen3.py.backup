"""
Qwen3 模型实现 - 使用 Mixin 模式重构
支持层迁移、层复制、Attention Offload 等优化功能
"""

import torch
import copy
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config,AutoConfig
from typing import Tuple, Optional
import torch.cuda.nvtx as nvtx

from HBserve.layers.activation import SiluAndMul
from HBserve.layers.attention import Attention
from HBserve.layers.layernorm import RMSNorm
from HBserve.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from HBserve.layers.rotary_embedding import get_rope
from HBserve.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from HBserve.utils.context import get_context, set_context, Context

# 导入优化 Mixin 和执行逻辑

from HBserve.models import register_model  # ← 导入装饰器

def fix_qwen3_config_if_big_model(config: Qwen3Config) -> Qwen3Config:
    H = getattr(config, 'hidden_size', None)
    if H is None:
        return config

    # number of transformer layers
    L = getattr(config, 'num_hidden_layers', None) or getattr(config, 'num_layers', None)
    if L is None:
        return config

    # intermediate size (fallback to 4*H if not provided)
    inter = getattr(config, 'intermediate_size', 4 * H)

    # approximate vocab contribution
    vocab = getattr(config, 'vocab_size', 0) or 0

    # estimate parameters
    per_layer = 4 * int(H) * int(H) + 3 * int(H) * int(inter)
    total_params = int(per_layer) * int(L) + 2 * int(H) * int(vocab)

    # apply fixes only for models >= 14 billion parameters
    if total_params >= 13_000_000_000:
        if hasattr(config, 'hidden_size'):
            config.hidden_size = 5120
        if hasattr(config, 'max_position_embeddings'):
            config.max_position_embeddings = 40960

    return config

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
        config = fix_qwen3_config_if_big_model(config)
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

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # ===== 修改DP设备配置解析 =====
        # 支持 local_dp_devices 列表（优先）或单个 local_dp_device（向后兼容）
        dp_device = getattr(config, "local_dp_device", None)
        dp_devices = getattr(config, "local_dp_devices", None)
        
        if dp_devices is not None:
            self.local_dp_devices = dp_devices if isinstance(dp_devices, list) else [dp_devices]
        elif dp_device is not None:
            self.local_dp_devices = [dp_device]
        else:
            self.local_dp_devices = []
        
        self.local_dp_start = getattr(config, "local_dp_start_layer", None)
        self.local_dp_end = getattr(config, "local_dp_end_layer", None)
        self.local_dp_degree = len(self.local_dp_devices)
        
        # 更新启用判断条件
        self.local_dp_enabled = (
            self.local_dp_start is not None and
            self.local_dp_end is not None and
            self.local_dp_degree > 0 and
            0 <= self.local_dp_start < self.local_dp_end <= len(self.layers)
        )
        # ============================
        
        # CUDA Graph相关
        self.use_cuda_graph = getattr(config, "use_cuda_graph", True)
        self.decode_graph_cache = {}
        self.graph_pool_handle = None
        
        if self.local_dp_enabled:
            # ===== 修改为每个DP设备创建副本和stream =====
            target_devices = []
            for dev in self.local_dp_devices:
                if isinstance(dev, int):
                    target_devices.append(torch.device(f"cuda:{dev}"))
                else:
                    target_devices.append(torch.device(dev))
            
            self._dp_target_devices = target_devices
            
            # 为每个设备创建独立的dp_layers
            nvtx.range_push("replica layers")
            self.dp_layers = nn.ModuleList()
            for idx, target_device in enumerate(target_devices):
                device_layers = nn.ModuleList()
                for i in range(self.local_dp_start, self.local_dp_end):
                    replica = copy.deepcopy(self.layers[i]).to(target_device)
                    # 标记副本属性
                    for m in replica.modules():
                        if hasattr(m, "k_cache") and hasattr(m, "v_cache"):
                            setattr(m, "is_replica", True)
                            setattr(m, "replica_device", target_device)
                            setattr(m, "replica_id", idx)
                    device_layers.append(replica)
                self.dp_layers.append(device_layers)
            nvtx.range_pop()
            # 创建streams（主设备stream + 每个DP设备stream）
            self.stream0 = torch.cuda.Stream(device=self.layers[0].self_attn.qkv_proj.weight.device)
            self.dp_streams = [torch.cuda.Stream(device=dev) for dev in target_devices]
            # ==========================================
        else:
            self.dp_layers = None
            self.dp_streams = None
            self.stream0 = None
            self._dp_target_devices = []

    def _capture_decode_graph(self, B, positions, hidden_states, residual, ctx):
        """捕获 decode 阶段的 CUDA Graph（支持多设备）"""
        
        # 确保batch足够大
        num_shards = self.local_dp_degree + 1
        if B < num_shards:
            print(f"Batch size {B} too small for {num_shards} shards, need at least {num_shards}")
            return None
        
        start, end = self.local_dp_start, self.local_dp_end
        dev0 = hidden_states.device
        
        # 分割batch
        shard_sizes = [B // num_shards] * num_shards
        for i in range(B % num_shards):
            shard_sizes[i] += 1
        
        offsets = [0]
        for size in shard_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        offsets.append(B)
        
        # 准备每个shard的静态buffer
        static_inputs = []
        static_outputs = []
        graphs = []
        
        # 获取最大block表大小
        max_blocks = None
        if ctx.block_tables is not None:
            cur_max_blocks = ctx.block_tables.shape[1]
            max_blocks = int(cur_max_blocks * 1.5)  # 预留50%空间
        
        # Warmup和捕获每个shard的graph
        for i in range(num_shards):
            s, e = offsets[i], offsets[i+1]
            is_main_shard = (i == 0)
            
            # 准备该shard的数据
            pos = positions[s:e]
            hs = hidden_states[s:e]
            res = residual[s:e] if residual is not None else None
            sm = ctx.slot_mapping[s:e]
            cl = ctx.context_lens[s:e]
            bt = ctx.block_tables[s:e] if ctx.block_tables is not None else None
            
            if is_main_shard:
                stream = self.stream0
                layers = self.layers  # 主设备使用完整的layers
                device = dev0
            else:
                device = self._dp_target_devices[i-1]
                stream = self.dp_streams[i-1]
                layers = self.dp_layers[i-1]  # DP设备使用其对应的副本layers
                # 将数据移动到目标设备
                pos = pos.to(device)
                hs = hs.to(device)
                res = res.to(device) if res is not None else None
                sm = sm.to(device)
                cl = cl.to(device)
                bt = bt.to(device) if bt is not None else None
            
            # 为block_tables创建固定大小的buffer
            if bt is not None:
                fixed_bt = torch.zeros((len(pos), max_blocks), dtype=bt.dtype, device=bt.device)
                fixed_bt[:, :bt.shape[1]] = bt
                bt = fixed_bt
            
            # Warmup
            for _ in range(3):
                with torch.cuda.stream(stream):
                    set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
                    tmp_hs, tmp_res = hs, res
                    # 修复：正确处理layer索引
                    for j in range(start, end):
                        layer_idx = j if is_main_shard else j - start
                        tmp_hs, tmp_res = layers[layer_idx](pos, tmp_hs, tmp_res)
            
            # 捕获Graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=self.graph_pool_handle):
                set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
                out_hs, out_res = hs, res
                # 修复：正确处理layer索引
                for j in range(start, end):
                    layer_idx = j if is_main_shard else j - start
                    out_hs, out_res = layers[layer_idx](pos, out_hs, out_res)
            
            graphs.append(graph)
            static_inputs.append({
                'positions': pos,
                'hidden_states': hs,
                'residual': res,
                'slot_mapping': sm,
                'context_lens': cl,
                'block_tables': bt,
            })
            static_outputs.append({
                'hidden_states': out_hs,
                'residual': out_res,
            })
        
        torch.cuda.synchronize()
        print(f"Graph capture completed for batch size {B} with {num_shards} shards")
        
        return {
            'graphs': graphs,
            'max_blocks': max_blocks,
            'static_inputs': static_inputs,
            'static_outputs': static_outputs,
            'shard_sizes': shard_sizes,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        
        if not self.local_dp_enabled:
            for layer in self.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        
        start = self.local_dp_start
        end = self.local_dp_end
        
        # 前面的层
        for i in range(start):
            nvtx.range_push("norm layer")
            hidden_states, residual = self.layers[i](positions, hidden_states, residual)
            nvtx.range_pop()
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
                
                nvtx.range_push("dp_shard0_total")
                set_context(True, cuq0, cuk0, ctx.max_seqlen_q, ctx.max_seqlen_k, sm0, None, bt0)
                
                hs0_d, res0_d = hs0, res0
                for j in range(start, end):
                    hs0_d, res0_d = self.layers[j](pos0, hs0_d, res0_d)
                
                nvtx.range_pop()
            
            # 修复：使用 self.dp_streams[0] 而不是 self.stream1
            with torch.cuda.stream(self.dp_streams[0]):
                self.dp_streams[0].wait_event(start_event)
                
                nvtx.range_push("dp_shard1_total")
                
                nvtx.range_push("dp_p2p_transfer")
                cuq1_d = cuq1.to(self._dp_target_devices[0], non_blocking=True)
                cuk1_d = cuk1.to(self._dp_target_devices[0], non_blocking=True)
                sm1_d = sm1.to(self._dp_target_devices[0], non_blocking=True)
                bt1_d = bt1.to(self._dp_target_devices[0], non_blocking=True) if bt1 is not None else None
                pos1_d = pos1.to(self._dp_target_devices[0], non_blocking=True)
                hs1_d = hs1.to(self._dp_target_devices[0], non_blocking=True)
                res1_d = None if res1 is None else res1.to(self._dp_target_devices[0], non_blocking=True)
                nvtx.range_pop()
                
                set_context(True, cuq1_d, cuk1_d, ctx.max_seqlen_q, ctx.max_seqlen_k, sm1_d, None, bt1_d)
                
                for j in range(start, end):
                    hs1_d, res1_d = self.dp_layers[0][j - start](pos1_d, hs1_d, res1_d)
                
                nvtx.range_push("dp_p2p_transfer_back")
                hs1_back = hs1_d.to(dev0, non_blocking=True)
                res1_back = None if res1_d is None else res1_d.to(dev0, non_blocking=True)
                nvtx.range_pop()
                
                nvtx.range_pop()
            
            # 等待两个 stream 完成
            nvtx.range_push("wait_both_streams")
            self.stream0.synchronize()
            self.dp_streams[0].synchronize()
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
            num_shards = self.local_dp_degree + 1
            
            # ===== 添加空batch检查 =====
            if B == 0:
                # 空batch，直接执行后续层
                for i in range(end, len(self.layers)):
                    hidden_states, residual = self.layers[i](positions, hidden_states, residual)
                hidden_states, _ = self.norm(hidden_states, residual)
                return hidden_states
            
            # ===== 确保 batch 足够大以支持多路并行 =====
            if B < num_shards:
                # batch太小，退化为串行执行
                for j in range(start, end):
                    hidden_states, residual = self.layers[j](positions, hidden_states, residual)
                for i in range(end, len(self.layers)):
                    hidden_states, residual = self.layers[i](positions, hidden_states, residual)
                hidden_states, _ = self.norm(hidden_states, residual)
                return hidden_states
            
            mid = B // 2  # 目前只实现两路shard，后续可扩展
            
            if self.use_cuda_graph:
                nvtx.range_push("decode_graph_execution")
                
                # 检查是否已有该 batch size 的 graph
                if B not in self.decode_graph_cache:
                    nvtx.range_push("capture decode graph")
                    import time
                    t1 = time.time()
                    self.decode_graph_cache[B] = self._capture_decode_graph(
                        B, positions, hidden_states, residual, ctx
                    )
                    print("Scale UP time: ", time.time() - t1)
                    nvtx.range_pop()
                
                graph_data = self.decode_graph_cache[B]
                
                # 分割数据
                shard_sizes = graph_data['shard_sizes']
                offsets = [0]
                for size in shard_sizes[:-1]:
                    offsets.append(offsets[-1] + size)
                
                # 更新所有shard的输入
                nvtx.range_push("update_graph_inputs")
                for i in range(num_shards):
                    s, e = offsets[i], offsets[i] + shard_sizes[i]
                    static_in = graph_data['static_inputs'][i]  # 列表索引
                    
                    # 更新输入数据
                    if i == 0:  # 主设备
                        static_in['positions'].copy_(positions[s:e])
                        static_in['hidden_states'].copy_(hidden_states[s:e])
                        if residual is not None:
                            static_in['residual'].copy_(residual[s:e])
                    else:  # DP设备
                        # 先在CPU侧准备数据，然后通过非阻塞传输
                        with torch.cuda.stream(self.dp_streams[i-1]):
                            static_in['positions'].copy_(positions[s:e], non_blocking=True)
                            static_in['hidden_states'].copy_(hidden_states[s:e], non_blocking=True)
                            if residual is not None and static_in['residual'] is not None:
                                static_in['residual'].copy_(residual[s:e], non_blocking=True)
                    
                    # 更新context数据
                    static_in['slot_mapping'].copy_(ctx.slot_mapping[s:e])
                    static_in['context_lens'].copy_(ctx.context_lens[s:e])
                    
                    # 处理block_tables（大小可能变化）
                    if ctx.block_tables is not None and static_in['block_tables'] is not None:
                        bt_current = ctx.block_tables[s:e]
                        if bt_current.shape[1] > static_in['block_tables'].shape[1]:
                            # block表大小超出预留空间，重新捕获graph
                            nvtx.range_pop()  # update_graph_inputs
                            nvtx.range_pop()  # decode_graph_execution
                            
                            print(f"Block table size increased from {static_in['block_tables'].shape[1]} to {bt_current.shape[1]}, recapturing graph for batch {B}...")
                            import time
                            t1 = time.time()
                            nvtx.range_push("capture decode graph")
                            self.decode_graph_cache[B] = self._capture_decode_graph(
                                B, positions, hidden_states, residual, ctx
                            )
                            nvtx.range_pop()
                            print("Scale UP time: ", time.time() - t1)
                            graph_data = self.decode_graph_cache[B]
                            nvtx.range_push("decode_graph_execution")
                            nvtx.range_push("update_graph_inputs")
                            
                            # 重新准备输入（会重新循环）
                        else:
                            # 只复制有效部分，其余保持为0
                            static_in['block_tables'].zero_()
                            static_in['block_tables'][:, :bt_current.shape[1]].copy_(bt_current)
                
                nvtx.range_pop()
                
                # 并行replay所有graphs
                nvtx.range_push("replay_graphs")
                for i, graph in enumerate(graph_data['graphs']):
                    if i == 0:
                        graph.replay()
                    else:
                        # 确保数据传输完成
                        self.dp_streams[i-1].synchronize()
                        graph.replay()
                nvtx.range_pop()
                
                # 等待所有stream完成
                nvtx.range_push("wait_all_streams")
                self.stream0.synchronize()
                for stream in self.dp_streams:
                    stream.synchronize()
                nvtx.range_pop()
                
                # 合并结果
                nvtx.range_push("merge_results")
                all_hidden_states = []
                all_residuals = []
                
                for i in range(num_shards):
                    out_hs = graph_data['static_outputs'][i]['hidden_states']
                    out_res = graph_data['static_outputs'][i]['residual']
                    
                    if i == 0:
                        all_hidden_states.append(out_hs)
                        if out_res is not None:
                            all_residuals.append(out_res)
                    else:
                        # DP设备的结果需要传回主设备
                        with torch.cuda.stream(self.dp_streams[i-1]):
                            hs_back = out_hs.to(dev0, non_blocking=True)
                            res_back = out_res.to(dev0, non_blocking=True) if out_res is not None else None
                        self.dp_streams[i-1].synchronize()
                        all_hidden_states.append(hs_back)
                        if res_back is not None:
                            all_residuals.append(res_back)
                
                hidden_states = torch.cat(all_hidden_states, dim=0)
                residual = torch.cat(all_residuals, dim=0) if all_residuals else None
                nvtx.range_pop()
                
                # 恢复原始 context
                ctx_orig = (False, None, None, 0, 0,
                        ctx.slot_mapping, ctx.context_lens, ctx.block_tables)
                set_context(*ctx_orig)
                
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
                    
                    nvtx.range_push("stream0_shard0_total")
                    set_context(False, slot_mapping=sm0, context_lens=cl0, block_tables=bt0)
                    
                    hs0_d, res0_d = hs0, res0
                    for j in range(start, end):
                        hs0_d, res0_d = self.layers[j](pos0, hs0_d, res0_d)
                    
                    nvtx.range_pop()
                
                # 修复：使用 self.dp_streams[0]
                with torch.cuda.stream(self.dp_streams[0]):
                    self.dp_streams[0].wait_event(start_event)
                    
                    nvtx.range_push("stream1_shard1_total")
                    
                    nvtx.range_push("stream1_p2p_transfer")
                    sm1_d = sm1.to(self._dp_target_devices[0], non_blocking=True)
                    cl1_d = cl1.to(self._dp_target_devices[0], non_blocking=True)
                    bt1_d = bt1.to(self._dp_target_devices[0], non_blocking=True) if bt1 is not None else None
                    pos1_d = pos1.to(self._dp_target_devices[0], non_blocking=True)
                    hs1_d = hs1.to(self._dp_target_devices[0], non_blocking=True)
                    res1_d = None if res1 is None else res1.to(self._dp_target_devices[0], non_blocking=True)
                    nvtx.range_pop()
                    
                    set_context(False, slot_mapping=sm1_d, context_lens=cl1_d, block_tables=bt1_d)
                    
                    # 修复：正确索引 dp_layers
                    for j in range(start, end):
                        hs1_d, res1_d = self.dp_layers[0][j - start](pos1_d, hs1_d, res1_d)
                    
                    # 在 stream1 内部传回
                    nvtx.range_push("stream1_p2p_transfer_back")
                    hs1_back = hs1_d.to(dev0, non_blocking=True)
                    res1_back = None if res1_d is None else res1_d.to(dev0, non_blocking=True)
                    nvtx.range_pop()
                    
                    nvtx.range_pop()
                
                nvtx.range_push("wait_both_streams")
                self.stream0.synchronize()
                self.dp_streams[0].synchronize()
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
        config = fix_qwen3_config_if_big_model(config)
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
        nvtx.range_push("model forward")
        hidden_states = self.model(input_ids, positions) # 注意这里的input_ids不包含prefix caching命中的部分
        nvtx.range_pop()
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        nvtx.range_push("compute logits")
        hidden_device = hidden_states.device
        if self.lm_head.weight.device != hidden_device:
            self.lm_head = self.lm_head.to(hidden_device)
        logits = self.lm_head(hidden_states)
        nvtx.range_pop()
        return logits