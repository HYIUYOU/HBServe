import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

import json
from transformers import AutoTokenizer

def truncate_tokens(tokens, max_length, strategy='context'):
    """
    截断token列表
    
    Args:
        tokens: token ID列表
        max_length: 最大长度
        strategy: 截断策略
    
    Returns:
        截断后的token列表
    """
    if len(tokens) <= max_length:
        return tokens
    
    if strategy == 'context' or strategy == 'end':
        # 从末尾截断
        return tokens[:max_length]
    
    elif strategy == 'middle':
        # 保留开头和结尾，截断中间
        keep_head = max_length // 2
        keep_tail = max_length - keep_head
        return tokens[:keep_head] + tokens[-keep_tail:]
    
    elif strategy == 'start':
        # 从开头截断（保留末尾）
        return tokens[-max_length:]
    
    else:
        raise ValueError(f"Unknown truncate strategy: {strategy}")


def truncate_text(text, max_chars, strategy='context'):
    """
    截断文本字符串
    
    Args:
        text: 文本字符串
        max_chars: 最大字符数
        strategy: 截断策略
    
    Returns:
        截断后的文本
    """
    if len(text) <= max_chars:
        return text
    
    if strategy == 'context' or strategy == 'end':
        # 从末尾截断
        return text[:max_chars]
    
    elif strategy == 'middle':
        # 保留开头和结尾，截断中间
        keep_head = max_chars // 2
        keep_tail = max_chars - keep_head
        return text[:keep_head] + " [...] " + text[-keep_tail:]
    
    elif strategy == 'start':
        # 从开头截断（保留末尾）
        return text[-max_chars:]
    
    else:
        raise ValueError(f"Unknown truncate strategy: {strategy}")

def load_longbench_prompts(
    jsonl_file, 
    max_samples=None, 
    max_length=10000,
    tokenizer=None,
    truncate_strategy='context'  # 'context', 'end', 'middle'
):
    """
    从LongBench JSONL文件加载prompts，并限制长度
    
    Args:
        jsonl_file: JSONL文件路径
        max_samples: 最多加载多少个样本
        max_length: 最大token长度（默认10000）
        tokenizer: tokenizer实例（如果为None，使用字符估算）
        truncate_strategy: 截断策略
            - 'context': 只截断context部分（推荐）
            - 'end': 从末尾截断
            - 'middle': 保留开头和结尾，截断中间
    
    Returns:
        prompts: prompt列表
    """
    prompts = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            item = json.loads(line.strip())
            context = item['context']
            question = item['input']
            
            # 构建基础prompt模板
            prefix = "Based on the following passages, answer the question.\n\n"
            suffix = f"\n\nQuestion: {question}\nAnswer:"
            
            # 如果提供了tokenizer，使用精确的token计数
            if tokenizer:
                prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
                context_tokens = tokenizer.encode(context, add_special_tokens=False)
                
                fixed_length = len(prefix_tokens) + len(suffix_tokens)
                available_length = max_length - fixed_length
                
                if len(context_tokens) > available_length:
                    # 根据策略截断context
                    context_tokens = truncate_tokens(
                        context_tokens, 
                        available_length, 
                        strategy=truncate_strategy
                    )
                    context = tokenizer.decode(context_tokens, skip_special_tokens=True)
                    #print(f"Sample {i}: Context truncated from {len(tokenizer.encode(item['context'], add_special_tokens=False))} to {len(context_tokens)} tokens")
            else:
                # 使用字符估算（粗略：1 token ≈ 4 characters）
                char_per_token = 4
                max_chars = max_length * char_per_token
                fixed_chars = len(prefix) + len(suffix)
                available_chars = max_chars - fixed_chars
                
                if len(context) > available_chars:
                    context = truncate_text(
                        context, 
                        available_chars, 
                        strategy=truncate_strategy
                    )
                    #print(f"Sample {i}: Context truncated to ~{len(context)} chars")
            
            # 构建最终prompt
            prompt = f"{prefix}{context}{suffix}"
            prompts.append(prompt)
    
    return prompts

# import json

# def load_longbench_prompts(jsonl_file, max_samples=None):
#     """从LongBench JSONL文件加载prompts"""
#     prompts = []
#     with open(jsonl_file, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             if max_samples and i >= max_samples:
#                 break
#             item = json.loads(line.strip())
            
#             # 构建prompt - 根据任务类型调整格式
#             prompt = f"""Based on the following passages, answer the question.

# {item['context']}

# Question: {item['input']}
# Answer:"""
            
#             prompts.append(prompt)
#     return prompts
