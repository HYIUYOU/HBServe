import torch
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
