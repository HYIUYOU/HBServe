import torch
import copy
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from HBserve.layers.activation import SiluAndMul
from HBserve.layers.attention import Attention
from HBserve.layers.layernorm import RMSNorm
from HBserve.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from HBserve.layers.rotary_embedding import get_rope
from HBserve.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from HBserve.utils.context import get_context, set_context, Context


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
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 跟踪每层的设备位置
        self.layer_devices = {}
        # 复制执行：记录被复制的层及其副本和设备
        self.replicas: dict[int, nn.Module] = {}
        self.replica_devices: dict[int, torch.device] = {}
        self.replica_split_ratio: dict[int, float] = {}
        # 自动调参：按两侧耗时自适应比例
        self.replica_autotune: dict[int, dict] = {}
        # Attention offload 配置
        self.attention_offload: dict[int, dict] = {}  # layer_id -> offload config

    # ===== LB: move layer =====
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

    def set_layer_device_distribution(self, layer_device_map: dict[int, str | torch.device]) -> None:
        """批量设置层的设备分布"""
        for layer_id, device in layer_device_map.items():
            self.move_layer_to_device(layer_id, device)

    # ===== scale up: replicate layer =====
    def replicate_layer_to_device(self, layer_id: int, device: str | torch.device, split_ratio: float = 0.5) -> None:
        """
        将指定层复制一个副本到目标GPU设备，用于批次切分并行执行该层。
        split_ratio 决定在原设备上处理的batch比例（0-1之间），其余在副本设备上处理。
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
        
        # 创建副本并移动到目标设备，同时设置正确的dtype
        replica = Qwen3DecoderLayer(self.config)
        replica = replica.to(device=device, dtype=src_dtype)
        replica.load_state_dict(src_state, strict=True)
        
        self.replicas[layer_id] = replica
        self.replica_devices[layer_id] = device
        self.replica_split_ratio[layer_id] = float(split_ratio)
        
        print(f"层 {layer_id} 已复制：{src_device}({src_dtype}) -> {device}({src_dtype})，切分比例: {split_ratio:.2f}")
    
    # ==== scale down : clear layer replication ====
    def clear_layer_replication(self, layer_id: int | None = None) -> None:
        """清除指定层或全部层的复制副本"""
        if layer_id is None:
            self.replicas.clear()
            self.replica_devices.clear()
            self.replica_split_ratio.clear()
            self.replica_autotune.clear()
            print("已清除所有层的复制配置")
        else:
            self.replicas.pop(layer_id, None)
            self.replica_devices.pop(layer_id, None)
            self.replica_split_ratio.pop(layer_id, None)
            self.replica_autotune.pop(layer_id, None)
            print(f"已清除层 {layer_id} 的复制配置")

    def update_replication_split_ratio(self, layer_id: int, split_ratio: float) -> None:
        """更新已复制层的切分比例（原设备比例）"""
        if layer_id not in self.replicas:
            raise ValueError(f"层 {layer_id} 未配置复制，无法更新split_ratio")
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio 需在 (0, 1) 区间内")
        self.replica_split_ratio[layer_id] = float(split_ratio)
        print(f"层 {layer_id} 切分比例已更新为: {split_ratio:.2f}")

    def enable_replication_autotune(self, layer_id: int, beta: float = 0.2, min_ratio: float = 0.1, max_ratio: float = 0.9) -> None:
        """
        启用复制层的比例自适应：依据原/副本两侧耗时，逐步逼近更均衡的split_ratio。
        beta: 指数平滑系数 (0,1]；min_ratio/max_ratio: 原设备比例上下界。
        """
        if layer_id not in self.replicas:
            raise ValueError(f"层 {layer_id} 未配置复制，无法启用autotune")
        if not (0.0 < beta <= 1.0):
            raise ValueError("beta 需在 (0, 1] 区间内")
        if not (0.0 < min_ratio < max_ratio < 1.0):
            raise ValueError("min_ratio/max_ratio 需满足 0<min<max<1")
        
        self.replica_autotune[layer_id] = {
            "beta": float(beta), 
            "min": float(min_ratio), 
            "max": float(max_ratio)
        }
        print(f"层 {layer_id} 已启用自适应调优，beta={beta}, 范围=[{min_ratio}, {max_ratio}]")

    def disable_replication_autotune(self, layer_id: int) -> None:
        """禁用指定层的自适应调优"""
        self.replica_autotune.pop(layer_id, None)
        print(f"层 {layer_id} 已禁用自适应调优")

    # === LB: attention offload by kv head ====
    def attention_offload_by_kv_head(
        self,
        layer_id: int,
        offload_device: str | torch.device,
        split_kv_head_idx: int | None = None,
        enable_autotune: bool = False,
        autotune_beta: float = 0.3
    ) -> None:
        """
        按 KV Head 切分 Attention 到两个 GPU。
        
        相比 batch 切分，这种方式：
        1. 更细粒度的并行
        2. KV Cache 可以按 head 分片，减少内存占用和同步开销
        3. 适合 GQA 模型
        
        Args:
            layer_id: 层索引
            offload_device: offload 目标设备
            split_kv_head_idx: 在哪个 KV head 索引处切分（None 表示中间切分）
            enable_autotune: 是否启用自适应调优
            autotune_beta: 自适应调优的平滑系数
        """
        if layer_id < 0 or layer_id >= len(self.layers):
            raise ValueError(f"层索引 {layer_id} 超出范围")
        
        if isinstance(offload_device, str):
            offload_device = torch.device(offload_device)
        
        # 获取原始层和 attention 模块
        src_layer = self.layers[layer_id]
        src_attn = src_layer.self_attn
        src_device = next(src_attn.parameters()).device
        src_dtype = next(src_attn.parameters()).dtype
        
        # 获取 head 配置
        num_heads = src_attn.num_heads
        num_kv_heads = src_attn.num_kv_heads
        head_dim = src_attn.head_dim
        
        # 确定切分点
        if split_kv_head_idx is None:
            split_kv_head_idx = num_kv_heads // 2
        
        if split_kv_head_idx <= 0 or split_kv_head_idx >= num_kv_heads:
            raise ValueError(f"split_kv_head_idx={split_kv_head_idx} 必须在 (0, {num_kv_heads}) 范围内")
        
        # 计算每个设备的 Q heads
        heads_per_kv_head = num_heads // num_kv_heads
        split_q_head_idx = split_kv_head_idx * heads_per_kv_head
        
        # 提取和分片原始权重
        qkv_weight = src_attn.qkv_proj.weight.data
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        
        # 分离 Q, K, V 权重
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
            # 权重
            'qkv_weight_0': qkv_weight_0.to(src_device),
            'qkv_bias_0': qkv_bias_0.to(src_device) if qkv_bias_0 is not None else None,
            'o_weight_0': o_weight_0.to(src_device),
            'qkv_weight_1': qkv_weight_1.to(offload_device),
            'qkv_bias_1': qkv_bias_1.to(offload_device) if qkv_bias_1 is not None else None,
            'o_weight_1': o_weight_1.to(offload_device),
            # Norm 权重
            'q_norm_weight': src_attn.q_norm.weight.data.clone(),
            'k_norm_weight': src_attn.k_norm.weight.data.clone(),
            'rotary_emb': src_attn.rotary_emb,
            # 独立的 cache（关键！）
            'cache_initialized': False,
            'k_cache_0': None,
            'v_cache_0': None,
            'k_cache_1': None,
            'v_cache_1': None,
            # 性能调优
            'enable_autotune': enable_autotune,
            'autotune_beta': autotune_beta if enable_autotune else None,
        }
        
        print(f"KV Head Split: 层 {layer_id} Attention 已按 KV Head 切分：")
        print(f"  原设备 {src_device}: Q heads [0:{split_q_head_idx}], KV heads [0:{split_kv_head_idx}]")
        print(f"  目标设备 {offload_device}: Q heads [{split_q_head_idx}:{num_heads}], KV heads [{split_kv_head_idx}:{num_kv_heads}]")
        print(f"  KV Cache: 按 head 分片存储（减少内存占用 {split_kv_head_idx}/{num_kv_heads} = {split_kv_head_idx/num_kv_heads*100:.0f}%）")
        print(f"  自适应调优: {'启用' if enable_autotune else '禁用'}")

    def _init_split_kv_cache(
        self,
        layer_id: int,
        config: dict
    ) -> None:
        """从原始 cache 初始化分片 cache（只在第一次 decode 时调用）"""
        import os
        DEBUG = os.environ.get("HB_DEBUG", "0") != "0"
        
        if config['cache_initialized']:
            return
        
        src_attn_module = config['src_attn'].attn
        src_k_cache = src_attn_module.k_cache
        src_v_cache = src_attn_module.v_cache
        
        if src_k_cache.numel() == 0:
            if DEBUG:
                print(f"[KVHeadSplit][layer {layer_id}] Original cache not initialized yet")
            return
        
        # Cache 形状: [num_blocks, block_size, num_kv_heads, head_dim]
        num_blocks, block_size, num_kv_heads, head_dim = src_k_cache.shape
        split_kv_head_idx = config['split_kv_head_idx']
        
        src_device = config['src_device']
        offload_device = config['offload_device']
        
        # Device 0: 分配并初始化 cache
        config['k_cache_0'] = torch.empty(
            (num_blocks, block_size, split_kv_head_idx, head_dim),
            dtype=src_k_cache.dtype,
            device=src_device
        )
        config['v_cache_0'] = torch.empty(
            (num_blocks, block_size, split_kv_head_idx, head_dim),
            dtype=src_v_cache.dtype,
            device=src_device
        )
        
        # 从原始 cache 复制数据
        config['k_cache_0'].copy_(src_k_cache[:, :, :split_kv_head_idx, :])
        config['v_cache_0'].copy_(src_v_cache[:, :, :split_kv_head_idx, :])
        
        # Device 1: 分配并初始化 cache
        config['k_cache_1'] = torch.empty(
            (num_blocks, block_size, num_kv_heads - split_kv_head_idx, head_dim),
            dtype=src_k_cache.dtype,
            device=offload_device
        )
        config['v_cache_1'] = torch.empty(
            (num_blocks, block_size, num_kv_heads - split_kv_head_idx, head_dim),
            dtype=src_v_cache.dtype,
            device=offload_device
        )
        
        # 从原始 cache 复制数据
        config['k_cache_1'].copy_(src_k_cache[:, :, split_kv_head_idx:, :].to(offload_device))
        config['v_cache_1'].copy_(src_v_cache[:, :, split_kv_head_idx:, :].to(offload_device))
        
        config['cache_initialized'] = True
        
        if DEBUG:
            print(f"[KVHeadSplit][layer {layer_id}] Cache initialized:")
            print(f"  Device 0: {config['k_cache_0'].shape} on {src_device}")
            print(f"  Device 1: {config['k_cache_1'].shape} on {offload_device}")
            print(f"  Memory reduction: {(1 - split_kv_head_idx/num_kv_heads)*100:.1f}% less on device 1")

    def _execute_kv_head_split_attention(
        self,
        layer_id: int,
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: Context
    ) -> torch.Tensor:
        """
        执行按 KV Head 切分的 Attention 计算
        
        核心优化：
        1. 权重已预先切分，直接计算
        2. KV Cache 按 head 分片，只同步需要的部分
        3. 两个设备完全独立计算，最后合并
        """
        import os
        config = self.attention_offload[layer_id]
        
        DEBUG = os.environ.get("HB_DEBUG", "0") != "0"
        def log(msg):
            if DEBUG:
                print(f"[KVHeadSplit][layer {layer_id}] {msg}")
        
        src_device = config['src_device']
        offload_device = config['offload_device']
        split_q_head_idx = config['split_q_head_idx']
        split_kv_head_idx = config['split_kv_head_idx']
        
        # 确保输入在原设备
        if hidden_states.device != src_device:
            hidden_states = hidden_states.to(src_device)
        if positions.device != src_device:
            positions = positions.to(src_device)
        
        # 处理 2D 和 3D 输入
        is_prefill = context.is_prefill
        if hidden_states.dim() == 2:
            batch_size = hidden_states.size(0)
            seq_len = 1
            hidden_size = hidden_states.size(1)
            hidden_states = hidden_states.unsqueeze(1)
        elif hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
        else:
            raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
        
        log(f"Input shape: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}, is_prefill={is_prefill}")
        
        # 初始化分片 cache（仅在第一次 decode 时）
        if not is_prefill and not config['cache_initialized']:
            self._init_split_kv_cache(layer_id, config)
        
        # === 1. QKV Projection ===
        qkv_0 = torch.nn.functional.linear(hidden_states, config['qkv_weight_0'], config['qkv_bias_0'])
        hs_1 = hidden_states.to(offload_device)
        qkv_1 = torch.nn.functional.linear(hs_1, config['qkv_weight_1'], config['qkv_bias_1'])
        
        # === 2. 分离 Q, K, V ===
        num_heads_0 = split_q_head_idx
        num_kv_heads_0 = split_kv_head_idx
        num_heads_1 = config['src_attn'].num_heads - split_q_head_idx
        num_kv_heads_1 = config['num_kv_heads_1']
        head_dim = config['head_dim']
        
        q_size_0 = num_heads_0 * head_dim
        kv_size_0 = num_kv_heads_0 * head_dim
        q_size_1 = num_heads_1 * head_dim
        kv_size_1 = num_kv_heads_1 * head_dim
        
        # Device 0
        q_0, k_0, v_0 = qkv_0.split([q_size_0, kv_size_0, kv_size_0], dim=-1)
        q_0 = q_0.view(batch_size, seq_len, num_heads_0, head_dim)
        k_0 = k_0.view(batch_size, seq_len, num_kv_heads_0, head_dim)
        v_0 = v_0.view(batch_size, seq_len, num_kv_heads_0, head_dim)
        
        # Device 1
        q_1, k_1, v_1 = qkv_1.split([q_size_1, kv_size_1, kv_size_1], dim=-1)
        q_1 = q_1.view(batch_size, seq_len, num_heads_1, head_dim)
        k_1 = k_1.view(batch_size, seq_len, num_kv_heads_1, head_dim)
        v_1 = v_1.view(batch_size, seq_len, num_kv_heads_1, head_dim)
        
        # === 3. RMS Norm ===
        q_norm_weight = config['q_norm_weight'].to(src_device)
        k_norm_weight = config['k_norm_weight'].to(src_device)
        
        q_0 = torch.nn.functional.rms_norm(q_0, (head_dim,), q_norm_weight, eps=1e-6)
        k_0 = torch.nn.functional.rms_norm(k_0, (head_dim,), k_norm_weight, eps=1e-6)
        
        q_norm_weight_1 = q_norm_weight.to(offload_device)
        k_norm_weight_1 = k_norm_weight.to(offload_device)
        q_1 = torch.nn.functional.rms_norm(q_1, (head_dim,), q_norm_weight_1, eps=1e-6)
        k_1 = torch.nn.functional.rms_norm(k_1, (head_dim,), k_norm_weight_1, eps=1e-6)
        
        # === 4. RoPE ===
        rotary_emb = config['rotary_emb']
        
        positions_0 = positions if positions.device == src_device else positions.to(src_device)
        q_0 = q_0.view(batch_size * seq_len, num_heads_0, head_dim)
        k_0 = k_0.view(batch_size * seq_len, num_kv_heads_0, head_dim)
        q_0, k_0 = rotary_emb(positions_0, q_0, k_0)
        
        positions_1 = positions.to(offload_device)
        q_1 = q_1.view(batch_size * seq_len, num_heads_1, head_dim)
        k_1 = k_1.view(batch_size * seq_len, num_kv_heads_1, head_dim)
        q_1, k_1 = rotary_emb(positions_1, q_1, k_1)
        
        v_0 = v_0.view(batch_size * seq_len, num_kv_heads_0, head_dim)
        v_1 = v_1.view(batch_size * seq_len, num_kv_heads_1, head_dim)
        
        # === 5. Attention 计算 ===
        log(f"Computing attention with split KV cache")
        
        # 准备 streams
        stream_0 = torch.cuda.Stream(device=src_device) if src_device.type == 'cuda' else None
        stream_1 = torch.cuda.Stream(device=offload_device) if offload_device.type == 'cuda' else None
        
        # Device 0
        if stream_0 is not None:
            with torch.cuda.stream(stream_0):
                o_0 = self._compute_split_attention(
                    q_0, k_0, v_0,
                    config['k_cache_0'], config['v_cache_0'],
                    context, src_device, layer_id, is_device_0=True
                )
        else:
            o_0 = self._compute_split_attention(
                q_0, k_0, v_0,
                config['k_cache_0'], config['v_cache_0'],
                context, src_device, layer_id, is_device_0=True
            )
        
        # Device 1
        if stream_1 is not None:
            with torch.cuda.stream(stream_1):
                o_1 = self._compute_split_attention(
                    q_1, k_1, v_1,
                    config['k_cache_1'], config['v_cache_1'],
                    context, offload_device, layer_id, is_device_0=False
                )
        else:
            o_1 = self._compute_split_attention(
                q_1, k_1, v_1,
                config['k_cache_1'], config['v_cache_1'],
                context, offload_device, layer_id, is_device_0=False
            )
        
        # 同步
        if stream_0 is not None:
            stream_0.synchronize()
        if stream_1 is not None:
            stream_1.synchronize()
        
        # === 6. Output Projection ===
        log(f"Computing output projection")
        
        o_0 = o_0.view(batch_size * seq_len, num_heads_0 * head_dim)
        o_1 = o_1.view(batch_size * seq_len, num_heads_1 * head_dim)
        
        if o_1.device != src_device:
            o_1 = o_1.to(src_device)
        
        out_0 = torch.nn.functional.linear(o_0, config['o_weight_0'], bias=None)
        o_weight_1 = config['o_weight_1'].to(src_device) if config['o_weight_1'].device != src_device else config['o_weight_1']
        out_1 = torch.nn.functional.linear(o_1, o_weight_1, bias=None)
        
        output = out_0 + out_1
        
        if seq_len == 1:
            output = output.view(batch_size, hidden_size)
        else:
            output = output.view(batch_size, seq_len, hidden_size)
        
        log(f"Attention computation done, output shape: {output.shape}")
        return output

    def _sync_kv_head_cache(
        self,
        layer_id: int,
        config: dict,
        context: Context
    ) -> None:
        """
        优化的 KV Cache 同步（按 head 分片）
        
        关键优化：
        1. 只同步各自负责的 head 的 cache
        2. 增量同步：只传输新增的 tokens
        3. 按需同步：只同步当前 batch 需要的 blocks
        """
        src_attn = config['src_attn'].attn
        src_k_cache = src_attn.k_cache
        src_v_cache = src_attn.v_cache
        
        if src_k_cache.numel() == 0:
            return
        
        src_device = config['src_device']
        offload_device = config['offload_device']
        split_kv_head_idx = config['split_kv_head_idx']
        
        # Cache 形状: [num_blocks, block_size, num_kv_heads, head_dim]
        num_blocks, block_size, num_kv_heads, head_dim = src_k_cache.shape
        
        # 分片索引
        # Device 0: kv_heads [0:split_kv_head_idx]
        # Device 1: kv_heads [split_kv_head_idx:num_kv_heads]
        
        # 提取分片（这里使用切片，不需要复制整个 cache）
        k_cache_0 = src_k_cache[:, :, :split_kv_head_idx, :]
        v_cache_0 = src_v_cache[:, :, :split_kv_head_idx, :]
        
        k_cache_1 = src_k_cache[:, :, split_kv_head_idx:, :]
        v_cache_1 = src_v_cache[:, :, split_kv_head_idx:, :]
        
        # 更新 cache（只复制切片，大幅减少传输量）
        config['k_cache_0'] = k_cache_0.contiguous()
        config['v_cache_0'] = v_cache_0.contiguous()
        
        # 只传输 device 1 的部分
        if config['k_cache_1'].numel() == 0 or config['k_cache_1'].shape != k_cache_1.shape:
            config['k_cache_1'] = k_cache_1.to(offload_device, non_blocking=True)
            config['v_cache_1'] = v_cache_1.to(offload_device, non_blocking=True)
        else:
            config['k_cache_1'].copy_(k_cache_1, non_blocking=True)
            config['v_cache_1'].copy_(v_cache_1, non_blocking=True)
        
        import os
        if os.environ.get("HB_DEBUG", "0") != "0":
            print(f"[KVHeadSplit] Cache synced: ")
            print(f"  Device 0: {k_cache_0.shape} (on {src_device})")
            print(f"  Device 1: {k_cache_1.shape} -> {offload_device}")
            print(f"  Reduction: {(1 - split_kv_head_idx/num_kv_heads)*100:.1f}% less data transferred")

    def _compute_split_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        context: Context,
        device: torch.device,
        layer_id: int,
        is_device_0: bool
    ) -> torch.Tensor:
        """在指定设备上计算 attention（使用独立的分片 cache）"""
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        import os
        
        DEBUG = os.environ.get("HB_DEBUG", "0") != "0"
        def log(msg):
            if DEBUG:
                device_name = "dev0" if is_device_0 else "dev1"
                print(f"[KVHeadSplit][layer {layer_id}][{device_name}] {msg}")
        
        # 确保所有张量在同一设备
        torch.cuda.set_device(device)
        
        # 确保 Q, K, V 在正确的设备且连续
        if q.device != device:
            q = q.to(device)
        if k.device != device:
            k = k.to(device)
        if v.device != device:
            v = v.to(device)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        log(f"q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
        log(f"q.device={q.device}, k.device={k.device}, v.device={v.device}")
        log(f"k_cache={'None' if k_cache is None else k_cache.shape}")
        
        # 移动 context 到当前设备并确保连续
        slot_mapping = None
        if context.slot_mapping is not None:
            slot_mapping = context.slot_mapping.to(device, non_blocking=True).contiguous()
        
        block_tables = None
        if context.block_tables is not None:
            block_tables = context.block_tables.to(device, non_blocking=True).contiguous()
        
        context_lens = None
        if context.context_lens is not None:
            context_lens = context.context_lens.to(device, non_blocking=True).contiguous()
        
        # 存储 KV 到独立的分片 cache
        if k_cache is not None and v_cache is not None and slot_mapping is not None:
            from HBserve.layers.attention import store_kvcache
            log(f"Storing KV to split cache")
            store_kvcache(k, v, k_cache, v_cache, slot_mapping)
        
        # 计算 attention
        scaling = (q.shape[-1]) ** -0.5
        
        if context.is_prefill:
            log(f"Prefill mode")
            
            # 准备 cu_seqlens
            cu_seqlens_q = None
            cu_seqlens_k = None
            max_seqlen_q = context.max_seqlen_q if hasattr(context, 'max_seqlen_q') else None
            max_seqlen_k = context.max_seqlen_k if hasattr(context, 'max_seqlen_k') else None
            
            if hasattr(context, "cu_seqlens_q") and context.cu_seqlens_q is not None:
                cu_seqlens_q = context.cu_seqlens_q.to(device, non_blocking=True).contiguous()
                log(f"cu_seqlens_q: {cu_seqlens_q.tolist()}")
            
            if hasattr(context, "cu_seqlens_k") and context.cu_seqlens_k is not None:
                cu_seqlens_k = context.cu_seqlens_k.to(device, non_blocking=True).contiguous()
                log(f"cu_seqlens_k: {cu_seqlens_k.tolist()}")
            
            log(f"max_seqlen_q={max_seqlen_q}, max_seqlen_k={max_seqlen_k}")
            
            # Prefill: 使用 cache 或直接的 K/V
            if block_tables is not None and k_cache is not None and k_cache.numel() > 0:
                k_use, v_use = k_cache, v_cache
                log(f"Using split cache for prefill")
            else:
                k_use, v_use = k, v
                log(f"Using direct K/V for prefill")
            
            # 确保 k_use, v_use 在正确设备且连续
            k_use = k_use.contiguous()
            v_use = v_use.contiguous()
            
            log(f"Calling flash_attn_varlen_func with q={q.shape}, k_use={k_use.shape}, v_use={v_use.shape}")
            
            # 调用 flash attention
            try:
                o = flash_attn_varlen_func(
                    q, k_use, v_use,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=scaling,
                    causal=True,
                    block_table=block_tables
                )
            except Exception as e:
                log(f"ERROR in flash_attn_varlen_func: {e}")
                log(f"  q: shape={q.shape}, device={q.device}, dtype={q.dtype}, contiguous={q.is_contiguous()}")
                log(f"  k_use: shape={k_use.shape}, device={k_use.device}, dtype={k_use.dtype}, contiguous={k_use.is_contiguous()}")
                log(f"  v_use: shape={v_use.shape}, device={v_use.device}, dtype={v_use.dtype}, contiguous={v_use.is_contiguous()}")
                if cu_seqlens_q is not None:
                    log(f"  cu_seqlens_q: shape={cu_seqlens_q.shape}, device={cu_seqlens_q.device}, dtype={cu_seqlens_q.dtype}")
                if cu_seqlens_k is not None:
                    log(f"  cu_seqlens_k: shape={cu_seqlens_k.shape}, device={cu_seqlens_k.device}, dtype={cu_seqlens_k.dtype}")
                if block_tables is not None:
                    log(f"  block_tables: shape={block_tables.shape}, device={block_tables.device}, dtype={block_tables.dtype}")
                raise
        else:
            # Decode 模式
            log(f"Decode mode")
            
            if k_cache is None or k_cache.numel() == 0:
                raise RuntimeError(f"KV cache not initialized for decode mode on {'dev0' if is_device_0 else 'dev1'}")
            
            log(f"Using flash_attn_with_kvcache")
            log(f"  k_cache.shape={k_cache.shape}, k_cache.device={k_cache.device}")
            log(f"  v_cache.shape={v_cache.shape}, v_cache.device={v_cache.device}")
            if context_lens is not None:
                log(f"  context_lens: {context_lens.tolist() if context_lens.numel() < 20 else f'shape={context_lens.shape}'}")
            if block_tables is not None:
                log(f"  block_tables.shape={block_tables.shape}")
            
            try:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),  # [batch, 1, num_heads, head_dim]
                    k_cache, v_cache,
                    cache_seqlens=context_lens,
                    block_table=block_tables,
                    softmax_scale=scaling,
                    causal=True
                )
                o = o.squeeze(1)  # [batch, num_heads, head_dim]
            except Exception as e:
                log(f"ERROR in flash_attn_with_kvcache: {e}")
                log(f"  q: shape={q.shape}, device={q.device}, dtype={q.dtype}")
                log(f"  k_cache: shape={k_cache.shape}, device={k_cache.device}, dtype={k_cache.dtype}")
                log(f"  v_cache: shape={v_cache.shape}, device={v_cache.device}, dtype={v_cache.dtype}")
                if context_lens is not None:
                    log(f"  context_lens: shape={context_lens.shape}, device={context_lens.device}, dtype={context_lens.dtype}")
                if block_tables is not None:
                    log(f"  block_tables: shape={block_tables.shape}, device={block_tables.device}, dtype={block_tables.dtype}")
                raise
        
        log(f"Attention output shape: {o.shape}")
        return o

    # === LB: attention offload by batch ====
    def attention_offload_by_batch(
        self,
        layer_id: int,
        offload_device: str | torch.device,
        split_ratio: float = 0.5,
        enable_autotune: bool = False,
        autotune_beta: float = 0.3
    ) -> None:
        """
        将指定层的 Attention 模块 offload 到另一个 GPU，按 batch 切分并行计算。
        
        Args:
            layer_id: 层索引
            offload_device: offload 目标设备
            split_ratio: 原设备处理的 batch 比例 (0-1)
            enable_autotune: 是否启用自适应调优
            autotune_beta: 自适应调优的平滑系数
        """
        if layer_id < 0 or layer_id >= len(self.layers):
            raise ValueError(f"层索引 {layer_id} 超出范围")
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio 需在 (0, 1) 区间内")
        
        if isinstance(offload_device, str):
            offload_device = torch.device(offload_device)
        
        # 获取原始层和 attention 模块
        src_layer = self.layers[layer_id]
        src_attn = src_layer.self_attn
        src_device = next(src_attn.parameters()).device
        src_dtype = next(src_attn.parameters()).dtype
        
        # 创建 attention 副本
        offload_attn = Qwen3Attention(
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
        
        # 复制权重到 offload 设备
        src_state = {k: v.detach().cpu() for k, v in src_attn.state_dict().items()}
        offload_attn = offload_attn.to(device=offload_device, dtype=src_dtype)
        offload_attn.load_state_dict(src_state, strict=True)
        
        # 保存配置
        self.attention_offload[layer_id] = {
            'offload_attn': offload_attn,
            'offload_device': offload_device,
            'src_device': src_device,
            'split_ratio': float(split_ratio),
            'enable_autotune': enable_autotune,
            'autotune_beta': autotune_beta,
            'autotune_stats': {'min_ratio': 0.1, 'max_ratio': 0.9}
        }
        
        print(f"Attention Offload: 层 {layer_id} Attention 已 offload：")
        print(f"  原设备: {src_device} ({src_dtype})")
        print(f"  目标设备: {offload_device} ({src_dtype})")
        print(f"  切分比例: {split_ratio:.2f}")
        print(f"  自适应调优: {'启用' if enable_autotune else '禁用'}")

    def clear_attention_offload(self, layer_id: int | None = None) -> None:
        """清除 Attention offload 配置"""
        if layer_id is None:
            self.attention_offload.clear()
            print("已清除所有 Attention offload 配置")
        else:
            self.attention_offload.pop(layer_id, None)
            print(f"已清除层 {layer_id} 的 Attention offload 配置")

    def _execute_attention_offload(
        self,
        layer_id: int,
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: Context
    ) -> torch.Tensor:
        """
        执行 Attention offload 的核心逻辑
        
        Returns:
            attention 输出
        """
        config = self.attention_offload[layer_id]
        src_attn = layer.self_attn
        offload_attn = config['offload_attn']
        src_device = config['src_device']
        offload_device = config['offload_device']
        ratio = config['split_ratio']
        is_prefill = context.is_prefill
        
        # 计算切分点
        if is_prefill:
            if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
                batch_size = len(context.cu_seqlens_q) - 1
                split_idx = int(round(batch_size * ratio))
                split_idx = max(1, min(split_idx, batch_size - 1))
                token_split_idx = context.cu_seqlens_q[split_idx].item()
            else:
                total_tokens = hidden_states.size(0)
                token_split_idx = int(round(total_tokens * ratio))
                token_split_idx = max(1, min(token_split_idx, total_tokens - 1))
                split_idx = token_split_idx
        else:
            # Decode: 按 batch 切分
            batch_size = hidden_states.size(0)
            split_idx = int(round(batch_size * ratio))
            split_idx = max(1, min(split_idx, batch_size - 1))
            token_split_idx = split_idx
        
        # 边界检查
        if token_split_idx == 0 or token_split_idx >= hidden_states.size(0):
            # 不切分，使用原设备
            return src_attn(positions, hidden_states)
        
        # === Decode 阶段：同步 KV Cache ===
        if not is_prefill:
            self._sync_attention_kv_cache(src_attn, offload_attn, split_idx, context.block_tables)
        
        # === 切分输入 ===
        hs_a = hidden_states[:token_split_idx]
        hs_b = hidden_states[token_split_idx:]
        pos_a = positions[:token_split_idx]
        pos_b = positions[token_split_idx:]
        
        # === 切分 Context ===
        ctx_a = self._split_context_for_attention(context, 0, split_idx, token_split_idx)
        ctx_b = self._split_context_for_attention(context, split_idx, None, token_split_idx)
        
        # === 移动到各自设备 ===
        if hs_a.device != src_device:
            hs_a = hs_a.to(src_device)
            pos_a = pos_a.to(src_device)
        if hs_b.device != offload_device:
            hs_b = hs_b.to(offload_device)
            pos_b = pos_b.to(offload_device)
        
        # === 并行执行 ===
        stream_a = torch.cuda.Stream(device=src_device) if src_device.type == 'cuda' else None
        stream_b = torch.cuda.Stream(device=offload_device) if offload_device.type == 'cuda' else None
        
        # 计时事件
        start_a = end_a = start_b = end_b = None
        if src_device.type == 'cuda':
            start_a = torch.cuda.Event(enable_timing=True)
            end_a = torch.cuda.Event(enable_timing=True)
        if offload_device.type == 'cuda':
            start_b = torch.cuda.Event(enable_timing=True)
            end_b = torch.cuda.Event(enable_timing=True)
        
        # 执行 A (原设备)
        if stream_a is not None:
            with torch.cuda.stream(stream_a):
                if start_a is not None:
                    start_a.record(stream_a)
                set_context(**ctx_a)
                out_a = src_attn(pos_a, hs_a)
                if end_a is not None:
                    end_a.record(stream_a)
        else:
            set_context(**ctx_a)
            out_a = src_attn(pos_a, hs_a)
        
        # 执行 B (offload 设备)
        if stream_b is not None:
            with torch.cuda.stream(stream_b):
                if start_b is not None:
                    start_b.record(stream_b)
                set_context(**ctx_b)
                out_b = offload_attn(pos_b, hs_b)
                if end_b is not None:
                    end_b.record(stream_b)
        else:
            set_context(**ctx_b)
            out_b = offload_attn(pos_b, hs_b)
        
        # === 同步 ===
        if stream_a is not None:
            stream_a.synchronize()
        if stream_b is not None:
            stream_b.synchronize()
        
        # 恢复原始 context
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
        
        # === 合并结果 ===
        if out_b.device != src_device:
            out_b = out_b.to(src_device)
        
        output = torch.cat([out_a, out_b], dim=0)
        
        # === 自适应调优 ===
        if config['enable_autotune'] and start_a and end_a and start_b and end_b:
            self._update_attention_offload_ratio(
                layer_id, config, ratio,
                start_a, end_a, start_b, end_b,
                src_device, offload_device
            )
        
        return output

    def _split_context_for_attention(
        self,
        context: Context,
        batch_start: int,
        batch_end: int | None,
        token_offset: int
    ) -> dict:
        """为 attention 切分 context"""
        is_prefill = context.is_prefill
        
        if batch_end is None:
            # 第二部分：从 batch_start 到结束
            cu_seqlens_q = context.cu_seqlens_q[batch_start:] - token_offset if context.cu_seqlens_q is not None else None
            cu_seqlens_k = context.cu_seqlens_k[batch_start:] - token_offset if context.cu_seqlens_k is not None else None
            
            # Decode 阶段不切分 slot_mapping，保持全局索引
            if is_prefill:
                slot_mapping = context.slot_mapping[token_offset:] if context.slot_mapping is not None else None
            else:
                # Decode 阶段：保持完整的 slot_mapping，因为它包含全局 cache 索引
                slot_mapping = context.slot_mapping[batch_start:] if context.slot_mapping is not None else None
            
            context_lens = context.context_lens[batch_start:] if context.context_lens is not None else None
            block_tables = context.block_tables[batch_start:] if context.block_tables is not None else None
        else:
            # 第一部分：从 0 到 batch_end
            cu_seqlens_q = context.cu_seqlens_q[:batch_end+1] if context.cu_seqlens_q is not None else None
            cu_seqlens_k = context.cu_seqlens_k[:batch_end+1] if context.cu_seqlens_k is not None else None
            
            # Decode 阶段不切分 slot_mapping
            if is_prefill:
                slot_mapping = context.slot_mapping[:token_offset] if context.slot_mapping is not None else None
            else:
                # Decode 阶段：保持完整的 slot_mapping
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
        block_tables: torch.Tensor | None
    ) -> None:
        """同步 Attention 的 KV Cache（改进版）"""
        src_attn_module = src_attn.attn
        dst_attn_module = dst_attn.attn
        
        # 检查源 cache 是否为空
        if src_attn_module.k_cache.numel() == 0:
            return
        
        dst_device = next(dst_attn.parameters()).device
        
        # 确保目标 cache 有足够的空间
        src_shape = src_attn_module.k_cache.shape
        
        # 复制完整 KV Cache
        if dst_attn_module.k_cache.numel() == 0:
            # 第一次：分配新的 cache
            dst_attn_module.k_cache = src_attn_module.k_cache.to(dst_device, non_blocking=True)
            dst_attn_module.v_cache = src_attn_module.v_cache.to(dst_device, non_blocking=True)
            print(f"  [Sync] 首次同步 KV cache 到 {dst_device}, shape={src_shape}")
        elif dst_attn_module.k_cache.shape != src_shape:
            # 形状不匹配：重新分配
            print(f"  [Sync] Cache 形状不匹配，重新分配: {dst_attn_module.k_cache.shape} -> {src_shape}")
            dst_attn_module.k_cache = src_attn_module.k_cache.to(dst_device, non_blocking=True)
            dst_attn_module.v_cache = src_attn_module.v_cache.to(dst_device, non_blocking=True)
        else:
            # 已有 cache 且形状匹配，只需复制数据
            dst_attn_module.k_cache.copy_(src_attn_module.k_cache, non_blocking=True)
            dst_attn_module.v_cache.copy_(src_attn_module.v_cache, non_blocking=True)
        
        # 同步，确保复制完成
        if dst_device.type == 'cuda':
            torch.cuda.synchronize(dst_device)

    def _update_attention_offload_ratio(
        self,
        layer_id: int,
        config: dict,
        old_ratio: float,
        start_a, end_a, start_b, end_b,
        src_device, offload_device
    ) -> None:
        """更新 Attention offload 的切分比例"""
        time_a = start_a.elapsed_time(end_a) if src_device.type == 'cuda' else 0.0
        time_b = start_b.elapsed_time(end_b) if offload_device.type == 'cuda' else 0.0
        total = time_a + time_b
        
        if total > 0:
            target_ratio = time_b / total
            beta = config['autotune_beta']
            new_ratio = (1.0 - beta) * old_ratio + beta * target_ratio
            
            # 限制范围
            stats = config['autotune_stats']
            new_ratio = max(stats['min_ratio'], min(new_ratio, stats['max_ratio']))
            
            config['split_ratio'] = new_ratio
            
            # 日志
            import os
            if os.environ.get("HB_ATTN_OFFLOAD_LOG", "0") != "0":
                print(
                    f"[AttnOffload][layer {layer_id}] "
                    f"time_a={time_a:.3f}ms time_b={time_b:.3f}ms "
                    f"ratio: {old_ratio:.3f} -> {new_ratio:.3f} (target={target_ratio:.3f})"
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) 
        residual = None
        
        # 获取 context
        context = get_context()
        is_prefill = context.is_prefill
        
        for layer_id, layer in enumerate(self.layers):
            layer_device = self.get_layer_device(layer_id)
            current_device = hidden_states.device
            
            if layer_device != current_device:
                hidden_states = hidden_states.to(layer_device)
                positions = positions.to(layer_device)
                if residual is not None:
                    residual = residual.to(layer_device)
            
            # === 检查是否有 KV Head Split 配置 ===
            if layer_id in self.attention_offload and self.attention_offload[layer_id].get('type') == 'kv_head_split':
                # 使用 KV Head Split 执行
                if residual is None:
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = layer.input_layernorm(hidden_states, residual)
                
                # 执行分片的 attention
                hidden_states = self._execute_kv_head_split_attention(
                    layer_id, layer, positions, hidden_states, context
                )
                
                # 继续执行后续部分
                hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
                hidden_states = layer.mlp(hidden_states)
            
            # === 检查是否有 Attention Offload 配置 ===
            elif layer_id in self.attention_offload:
                # 使用 Attention offload
                if residual is None:
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = layer.input_layernorm(hidden_states, residual)
                
                # 执行 offload 的 attention
                hidden_states = self._execute_attention_offload(
                    layer_id, layer, positions, hidden_states, context
                )
                
                # 继续执行后续部分
                hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
                hidden_states = layer.mlp(hidden_states)
            
            # === 层复制逻辑 ===
            # 使用层复制（prefill 和 decode 都支持）
            elif layer_id in self.replicas:
                replica = self.replicas[layer_id]
                replica_device = self.replica_devices[layer_id]
                
                # 获取原始context
                orig_ctx = Context(
                    is_prefill=context.is_prefill,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=context.max_seqlen_k,
                    slot_mapping=context.slot_mapping,
                    context_lens=context.context_lens,
                    block_tables=context.block_tables
                )
                
                ratio = self.replica_split_ratio.get(layer_id, 0.5)
                
                # 计算切分点
                if is_prefill:
                    # Prefill: 根据 cu_seqlens_q 切分
                    if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
                        batch_size = len(context.cu_seqlens_q) - 1
                        split_idx = int(round(batch_size * ratio))
                        split_idx = max(1, min(split_idx, batch_size - 1))
                        token_split_idx = context.cu_seqlens_q[split_idx].item()
                    else:
                        total_tokens = hidden_states.size(0)
                        token_split_idx = int(round(total_tokens * ratio))
                        token_split_idx = max(1, min(token_split_idx, total_tokens - 1))
                        split_idx = token_split_idx
                else:
                    # Decode: 每个 token 对应一个序列，直接按 batch 切分
                    batch_size = hidden_states.size(0)
                    split_idx = int(round(batch_size * ratio))
                    split_idx = max(1, min(split_idx, batch_size - 1))
                    token_split_idx = split_idx
                
                if token_split_idx == 0 or token_split_idx >= hidden_states.size(0):
                    hidden_states, residual = layer(positions, hidden_states, residual)
                else:
                    # === KV Cache 同步（关键部分）===
                    if not is_prefill:
                        # Decode 阶段需要同步 KV cache
                        self._sync_kv_cache_for_decode(layer, replica, split_idx, context.block_tables)
                    
                    # 切分输入张量
                    hs_a = hidden_states[:token_split_idx]
                    hs_b = hidden_states[token_split_idx:]
                    pos_a = positions[:token_split_idx]
                    pos_b = positions[token_split_idx:]
                    res_a = None if residual is None else residual[:token_split_idx]
                    res_b = None if residual is None else residual[token_split_idx:]
                    
                    # 准备context_a
                    cu_seqlens_q_a = context.cu_seqlens_q[:split_idx+1] if context.cu_seqlens_q is not None else None
                    cu_seqlens_k_a = context.cu_seqlens_k[:split_idx+1] if context.cu_seqlens_k is not None else None
                    slot_mapping_a = context.slot_mapping[:token_split_idx] if context.slot_mapping is not None else None
                    context_lens_a = context.context_lens[:split_idx] if context.context_lens is not None else None
                    block_tables_a = context.block_tables[:split_idx] if context.block_tables is not None else None
                    
                    # 准备context_b
                    if context.cu_seqlens_q is not None:
                        cu_seqlens_q_b = context.cu_seqlens_q[split_idx:] - token_split_idx
                    else:
                        cu_seqlens_q_b = None
                    if context.cu_seqlens_k is not None:
                        cu_seqlens_k_b = context.cu_seqlens_k[split_idx:] - token_split_idx
                    else:
                        cu_seqlens_k_b = None
                    slot_mapping_b = context.slot_mapping[token_split_idx:] if context.slot_mapping is not None else None
                    context_lens_b = context.context_lens[split_idx:] if context.context_lens is not None else None
                    block_tables_b = context.block_tables[split_idx:] if context.block_tables is not None else None
                    
                    # 移动到各自设备
                    if hs_a.device != layer_device:
                        hs_a = hs_a.to(layer_device)
                        pos_a = pos_a.to(layer_device)
                        if res_a is not None:
                            res_a = res_a.to(layer_device)
                    
                    if hs_b.device != replica_device:
                        hs_b = hs_b.to(replica_device)
                        pos_b = pos_b.to(replica_device)
                        if res_b is not None:
                            res_b = res_b.to(replica_device)
                    
                    # 准备stream和计时
                    stream_a = torch.cuda.Stream(device=layer_device) if layer_device.type == 'cuda' else None
                    stream_b = torch.cuda.Stream(device=replica_device) if replica_device.type == 'cuda' else None
                    
                    start_a = end_a = start_b = end_b = None
                    if layer_device.type == 'cuda':
                        start_a = torch.cuda.Event(enable_timing=True)
                        end_a = torch.cuda.Event(enable_timing=True)
                    if replica_device.type == 'cuda':
                        start_b = torch.cuda.Event(enable_timing=True)
                        end_b = torch.cuda.Event(enable_timing=True)
                    
                    # 并行执行A
                    if stream_a is not None:
                        with torch.cuda.stream(stream_a):
                            if start_a is not None:
                                start_a.record(stream_a)
                            set_context(
                                is_prefill=orig_ctx.is_prefill,
                                cu_seqlens_q=cu_seqlens_q_a,
                                cu_seqlens_k=cu_seqlens_k_a,
                                max_seqlen_q=orig_ctx.max_seqlen_q,
                                max_seqlen_k=orig_ctx.max_seqlen_k,
                                slot_mapping=slot_mapping_a,
                                context_lens=context_lens_a,
                                block_tables=block_tables_a
                            )
                            out_a, res_out_a = layer(pos_a, hs_a, res_a)
                            if end_a is not None:
                                end_a.record(stream_a)
                    else:
                        set_context(
                            is_prefill=orig_ctx.is_prefill,
                            cu_seqlens_q=cu_seqlens_q_a,
                            cu_seqlens_k=cu_seqlens_k_a,
                            max_seqlen_q=orig_ctx.max_seqlen_q,
                            max_seqlen_k=orig_ctx.max_seqlen_k,
                            slot_mapping=slot_mapping_a,
                            context_lens=context_lens_a,
                            block_tables=block_tables_a
                        )
                        out_a, res_out_a = layer(pos_a, hs_a, res_a)
                    
                    # 并行执行B
                    if stream_b is not None:
                        with torch.cuda.stream(stream_b):
                            if start_b is not None:
                                start_b.record(stream_b)
                            set_context(
                                is_prefill=orig_ctx.is_prefill,
                                cu_seqlens_q=cu_seqlens_q_b,
                                cu_seqlens_k=cu_seqlens_k_b,
                                max_seqlen_q=orig_ctx.max_seqlen_q,
                                max_seqlen_k=orig_ctx.max_seqlen_k,
                                slot_mapping=slot_mapping_b,
                                context_lens=context_lens_b,
                                block_tables=block_tables_b
                            )
                            out_b, res_out_b = replica(pos_b, hs_b, res_b)
                            if end_b is not None:
                                end_b.record(stream_b)
                    else:
                        set_context(
                            is_prefill=orig_ctx.is_prefill,
                            cu_seqlens_q=cu_seqlens_q_b,
                            cu_seqlens_k=cu_seqlens_k_b,
                            max_seqlen_q=orig_ctx.max_seqlen_q,
                            max_seqlen_k=orig_ctx.max_seqlen_k,
                            slot_mapping=slot_mapping_b,
                            context_lens=context_lens_b,
                            block_tables=block_tables_b
                        )
                        out_b, res_out_b = replica(pos_b, hs_b, res_b)
                    
                    # 同步
                    if stream_a is not None:
                        stream_a.synchronize()
                    if stream_b is not None:
                        stream_b.synchronize()
                    
                    # 恢复context
                    set_context(
                        is_prefill=orig_ctx.is_prefill,
                        cu_seqlens_q=orig_ctx.cu_seqlens_q,
                        cu_seqlens_k=orig_ctx.cu_seqlens_k,
                        max_seqlen_q=orig_ctx.max_seqlen_q,
                        max_seqlen_k=orig_ctx.max_seqlen_k,
                        slot_mapping=orig_ctx.slot_mapping,
                        context_lens=orig_ctx.context_lens,
                        block_tables=orig_ctx.block_tables
                    )
                    
                    # 移回layer_device
                    if out_b.device != layer_device:
                        out_b = out_b.to(layer_device)
                    if res_out_b is not None and res_out_b.device != layer_device:
                        res_out_b = res_out_b.to(layer_device)
                    
                    # 合并结果
                    hidden_states = torch.cat([out_a, out_b], dim=0)
                    if res_out_a is None and res_out_b is None:
                        residual = None
                    elif res_out_a is None:
                        residual = torch.cat([torch.zeros_like(out_a), res_out_b], dim=0)
                    elif res_out_b is None:
                        residual = torch.cat([res_out_a, torch.zeros_like(out_b)], dim=0)
                    else:
                        residual = torch.cat([res_out_a, res_out_b], dim=0)
                    
                    # Autotune
                    if layer_id in self.replica_autotune and start_a is not None and end_a is not None and start_b is not None and end_b is not None:
                        time_a = start_a.elapsed_time(end_a) if layer_device.type == 'cuda' else 0.0
                        time_b = start_b.elapsed_time(end_b) if replica_device.type == 'cuda' else 0.0
                        total = time_a + time_b
                        if total > 0:
                            # 目标：按耗时反比分配，使下一次原设备比例约等于 time_b/total
                            target_ratio = time_b / total
                            cfg = self.replica_autotune[layer_id]
                            beta = cfg["beta"]
                            new_ratio = (1.0 - beta) * ratio + beta * target_ratio
                            new_ratio = max(cfg["min"], min(new_ratio, cfg["max"]))
                            self.replica_split_ratio[layer_id] = new_ratio
                            # 可选日志
                            import os
                            if os.environ.get("HB_REPLICA_LOG", "0") != "0":
                                print(
                                    f"[Replica][layer {layer_id}] time_a={time_a:.3f}ms time_b={time_b:.3f}ms "
                                    f"ratio(old)={ratio:.3f} -> ratio(new)={new_ratio:.3f} (target={target_ratio:.3f})"
                                )
            else:
                # 2. layer() ==> 将hidden_states和positions传递给layer
                hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 3. norm() ==> 将hidden_states和residual传递给norm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _sync_kv_cache_for_decode(
        self, 
        src_layer: nn.Module, 
        dst_layer: nn.Module, 
        split_idx: int,
        block_tables: torch.Tensor | None
    ) -> None:
        """
        在 decode 阶段同步 KV cache 到副本层。
        
        策略：将完整的 KV cache 复制到副本设备（简单但有开销）。
        如果 cache 很大，可以优化为只复制需要的 blocks。
        """
        # 先定义变量
        src_attn = src_layer.self_attn.attn
        dst_attn = dst_layer.self_attn.attn
        
        # 如果源 cache 为空，跳过
        if src_attn.k_cache.numel() == 0:
            return
        
        dst_device = next(dst_layer.parameters()).device
        
        # 将完整的 KV cache 复制到副本设备
        # 注意：这里复制整个 cache，因为 block_tables 可能指向任意 blocks
        if dst_attn.k_cache.numel() == 0 or dst_attn.k_cache.shape != src_attn.k_cache.shape:
            # 第一次或形状不匹配，需要重新分配
            dst_attn.k_cache = src_attn.k_cache.to(dst_device, non_blocking=True)
            dst_attn.v_cache = src_attn.v_cache.to(dst_device, non_blocking=True)
        else:
            # 已有 cache，只需复制数据
            dst_attn.k_cache.copy_(src_attn.k_cache, non_blocking=True)
            dst_attn.v_cache.copy_(src_attn.v_cache, non_blocking=True)

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
        logits = self.lm_head(hidden_states)
        return logits
