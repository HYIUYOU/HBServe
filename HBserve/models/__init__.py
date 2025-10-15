# HBserve/models/__init__.py

from typing import Dict, Type
import torch.nn as nn

# 模型注册表
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(model_type: str):
    """注册模型的装饰器"""
    def decorator(cls):
        _MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator

def get_model_class(model_type: str):
    """根据 model_type 获取模型类"""
    model_type = model_type.lower()
    
    # 处理别名
    aliases = {
        'qwen2': 'qwen3',
    }
    model_type = aliases.get(model_type, model_type)
    
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"不支持的模型类型: {model_type}\n"
            f"支持的类型: {list(_MODEL_REGISTRY.keys())}"
        )
    
    return _MODEL_REGISTRY[model_type]

def create_model_from_config(hf_config):
    """根据 HuggingFace config 自动创建模型"""
    model_type = hf_config.model_type
    print(f"✓ 检测到模型类型: {model_type}")
    
    model_class = get_model_class(model_type)
    model = model_class(hf_config)
    
    return model

# ========== 重要：在这里导入所有模型，触发注册 ==========
from HBserve.models.qwen3 import Qwen3ForCausalLM
from HBserve.models.llama import LlamaForCausalLM
from HBserve.models.opt import OPTForCausalLM     

__all__ = [
    'register_model',
    'get_model_class',
    'create_model_from_config',
    'Qwen3ForCausalLM',
    'OPTForCausalLM',
    'LlamaForCausalLM',
]