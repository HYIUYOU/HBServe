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
        # 跟踪每层的设备位置
        self.layer_devices = {}
        # 复制执行：记录被复制的层及其副本和设备
        self.replicas: dict[int, nn.Module] = {}
        self.replica_devices: dict[int, torch.device] = {}
        self.replica_split_ratio: dict[int, float] = {}
        # 自动调参：按两侧耗时自适应比例
        self.replica_autotune: dict[int, dict] = {}

    def move_layer_to_device(self, layer_id: int, device: str | torch.device) -> None:
        """
        将指定层移动到目标设备
        
        Args:
            layer_id: 层的索引 (从0开始)
            device: 目标设备，可以是字符串如 'cuda:1' 或 torch.device 对象
        """
        if layer_id < 0 or layer_id >= len(self.layers):
            raise ValueError(f"层索引 {layer_id} 超出范围 [0, {len(self.layers)-1}]")
        
        # 转换为torch.device对象
        if isinstance(device, str):
            device = torch.device(device)
        
        # 移动层到指定设备
        self.layers[layer_id] = self.layers[layer_id].to(device)
        
        # 记录层的设备位置
        self.layer_devices[layer_id] = device
        
        print(f"层 {layer_id} 已移动到设备 {device}")

    def get_layer_device(self, layer_id: int) -> torch.device:
        """
        获取指定层的当前设备
        
        Args:
            layer_id: 层的索引
            
        Returns:
            层当前所在的设备
        """
        if layer_id in self.layer_devices:
            return self.layer_devices[layer_id]
        else:
            # 如果没有记录，返回层当前的实际设备
            return next(self.layers[layer_id].parameters()).device

    def set_layer_device_distribution(self, layer_device_map: dict[int, str | torch.device]) -> None:
        """
        批量设置层的设备分布
        
        Args:
            layer_device_map: 字典，键为层索引，值为目标设备
            例如: {9: 'cuda:1', 10: 'cuda:1', 15: 'cuda:2'}
        """
        for layer_id, device in layer_device_map.items():
            self.move_layer_to_device(layer_id, device)

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
        # 深拷贝一份并迁移到目标设备
        replica = copy.deepcopy(self.layers[layer_id]).to(device)
        self.replicas[layer_id] = replica
        self.replica_devices[layer_id] = device
        self.replica_split_ratio[layer_id] = float(split_ratio)

    def clear_layer_replication(self, layer_id: int | None = None) -> None:
        """
        清除指定层或全部层的复制副本。
        """
        if layer_id is None:
            self.replicas.clear()
            self.replica_devices.clear()
            self.replica_split_ratio.clear()
            return
        self.replicas.pop(layer_id, None)
        self.replica_devices.pop(layer_id, None)
        self.replica_split_ratio.pop(layer_id, None)

    def update_replication_split_ratio(self, layer_id: int, split_ratio: float) -> None:
        """更新已复制层的切分比例（原设备比例）。"""
        if layer_id not in self.replicas:
            raise ValueError(f"层 {layer_id} 未配置复制，无法更新split_ratio")
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio 需在 (0, 1) 区间内")
        self.replica_split_ratio[layer_id] = float(split_ratio)

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
        self.replica_autotune[layer_id] = {"beta": float(beta), "min": float(min_ratio), "max": float(max_ratio)}

    def disable_replication_autotune(self, layer_id: int) -> None:
        self.replica_autotune.pop(layer_id, None)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 1. embed_tokens() ==> 将input_ids转换为hidden_states
        # 注意这里的input_ids 不包含prefix caching命中的部分 ==> hidden_states 不包含prefix caching命中的部分
        hidden_states = self.embed_tokens(input_ids) 
        residual = None
        
        for layer_id, layer in enumerate(self.layers):
            # 检查层是否在不同的设备上
            layer_device = self.get_layer_device(layer_id)
            current_device = hidden_states.device
            
            # 如果层在不同的设备上，需要移动tensor到层的设备
            if layer_device != current_device:
                hidden_states = hidden_states.to(layer_device)
                positions = positions.to(layer_device)
                if residual is not None:
                    residual = residual.to(layer_device)
            
            # 如果该层配置了复制到另一设备，则对batch维度进行切分并行计算
            if layer_id in self.replicas:
                replica = self.replicas[layer_id]
                replica_device = self.replica_devices[layer_id]
                # 按配置比例切分batch：前半在原设备，后半在副本设备
                batch_size = hidden_states.size(0)
                ratio = self.replica_split_ratio.get(layer_id, 0.5)
                split_idx = int(round(batch_size * ratio))
                # 边界保护，至少1且保留另一侧至少1
                split_idx = max(1, min(split_idx, batch_size - 1))
                if split_idx == 0:  # 小批次不切分
                    hidden_states, residual = layer(positions, hidden_states, residual)
                else:
                    # 准备两份输入
                    hs_a = hidden_states[:split_idx]
                    hs_b = hidden_states[split_idx:]
                    pos_a = positions[:split_idx]
                    pos_b = positions[split_idx:]
                    res_a = None if residual is None else residual[:split_idx]
                    res_b = None if residual is None else residual[split_idx:]

                    # A 在原层设备计算（layer_device）
                    if hs_a.device != layer_device:
                        hs_a = hs_a.to(layer_device)
                        pos_a = pos_a.to(layer_device)
                        if res_a is not None:
                            res_a = res_a.to(layer_device)
                    # B 在副本设备计算
                    if hs_b.device != replica_device:
                        hs_b = hs_b.to(replica_device)
                        pos_b = pos_b.to(replica_device)
                        if res_b is not None:
                            res_b = res_b.to(replica_device)

                    # 并行执行（两个CUDA stream）
                    stream_a = torch.cuda.Stream(device=layer_device.type + ":" + str(layer_device.index)) if layer_device.type == 'cuda' else None
                    stream_b = torch.cuda.Stream(device=replica_device.type + ":" + str(replica_device.index)) if replica_device.type == 'cuda' else None

                    out_a = out_b = res_out_a = res_out_b = None
                    # cuda计时事件
                    start_a = end_a = start_b = end_b = None
                    if layer_device.type == 'cuda':
                        start_a = torch.cuda.Event(enable_timing=True)
                        end_a = torch.cuda.Event(enable_timing=True)
                    if replica_device.type == 'cuda':
                        start_b = torch.cuda.Event(enable_timing=True)
                        end_b = torch.cuda.Event(enable_timing=True)

                    if stream_a is not None:
                        with torch.cuda.stream(stream_a):
                            if start_a is not None:
                                start_a.record(stream_a)
                            out_a, res_out_a = layer(pos_a, hs_a, res_a)
                            if end_a is not None:
                                end_a.record(stream_a)
                    else:
                        out_a, res_out_a = layer(pos_a, hs_a, res_a)

                    if stream_b is not None:
                        with torch.cuda.stream(stream_b):
                            if start_b is not None:
                                start_b.record(stream_b)
                            out_b, res_out_b = replica(pos_b, hs_b, res_b)
                            if end_b is not None:
                                end_b.record(stream_b)
                    else:
                        out_b, res_out_b = replica(pos_b, hs_b, res_b)

                    # 同步与回传到layer_device以便后续继续
                    if stream_a is not None:
                        stream_a.synchronize()
                    if stream_b is not None:
                        stream_b.synchronize()

                    if out_b.device != layer_device:
                        out_b = out_b.to(layer_device)
                    if res_out_b is not None and res_out_b.device != layer_device:
                        res_out_b = res_out_b.to(layer_device)

                    hidden_states = torch.cat([out_a, out_b], dim=0)
                    if res_out_a is None and res_out_b is None:
                        residual = None
                    elif res_out_a is None:
                        residual = torch.cat([torch.zeros_like(out_a), res_out_b], dim=0)
                    elif res_out_b is None:
                        residual = torch.cat([res_out_a, torch.zeros_like(out_b)], dim=0)
                    else:
                        residual = torch.cat([res_out_a, res_out_b], dim=0)

                    # 自适应更新比例：依据两侧耗时估计目标比例
                    if layer_id in self.replica_autotune and start_a is not None and end_a is not None and start_b is not None and end_b is not None:
                        # 注意：必须在相应device上同步后再读取时间
                        if stream_a is not None:
                            stream_a.synchronize()
                        if stream_b is not None:
                            stream_b.synchronize()
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
