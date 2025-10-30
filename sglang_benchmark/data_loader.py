#!/usr/bin/env python3
"""
Data loader for Alpaca, ShareGPT, and LongBench datasets
"""
import json
import random
import os
from typing import List, Dict

class DatasetLoader:
    def __init__(self, dataset_name: str, dataset_path: str = None, 
                 max_length: int = None, tokenizer_name: str = None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.data = []
        
        # 如果是 LongBench，尝试加载 tokenizer
        self.tokenizer = None
        if dataset_name == "longbench" and tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, 
                    trust_remote_code=True
                )
                print(f"Loaded tokenizer: {tokenizer_name}")
            except Exception as e:
                print(f"Warning: Failed to load tokenizer: {e}")
                print("Will use character-based truncation as fallback")
        
        self.load_data()
    
    def load_data(self):
        """Load dataset"""
        if self.dataset_name == "alpaca":
            self.load_alpaca()
        elif self.dataset_name == "sharegpt":
            self.load_sharegpt()
        elif self.dataset_name == "longbench":
            self.load_longbench()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # 过滤空 prompt
        original_count = len(self.data)
        self.data = [d for d in self.data if d.get("prompt", "").strip()]
        filtered_count = original_count - len(self.data)
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} empty prompts")
        
        if not self.data:
            raise ValueError("No valid data loaded! Please check your dataset.")
            
        print(f"Loaded {len(self.data)} valid samples from {self.dataset_name}")
        
        # 显示长度统计
        if self.dataset_name == "longbench":
            self._print_length_stats()
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max_length tokens"""
        if not max_length:
            return text
        
        if self.tokenizer:
            # 使用 tokenizer 精确截断
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    return truncated_text
                return text
            except Exception as e:
                print(f"Tokenizer truncation failed: {e}, using char-based fallback")
                # 回退到字符截断
                estimated_chars = max_length * 4
                if len(text) > estimated_chars:
                    return text[:estimated_chars]
                return text
        else:
            # 简单的字符截断（粗略估计：1 token ≈ 4 chars）
            estimated_chars = max_length * 4
            if len(text) > estimated_chars:
                return text[:estimated_chars]
            return text
    
    def _print_length_stats(self):
        """Print length statistics for prompts"""
        sample_size = min(100, len(self.data))
        
        if self.tokenizer:
            lengths = []
            for d in self.data[:sample_size]:
                try:
                    tokens = self.tokenizer.encode(d["prompt"], add_special_tokens=False)
                    lengths.append(len(tokens))
                except:
                    lengths.append(len(d["prompt"]) // 4)
        else:
            lengths = [len(d["prompt"]) // 4 for d in self.data[:sample_size]]
        
        if lengths:
            print(f"\nPrompt length statistics (tokens, sample size={len(lengths)}):")
            print(f"  Mean: {sum(lengths) / len(lengths):.0f}")
            print(f"  Median: {sorted(lengths)[len(lengths)//2]:.0f}")
            print(f"  Min: {min(lengths):.0f}")
            print(f"  Max: {max(lengths):.0f}")
            if len(lengths) >= 20:
                print(f"  P95: {sorted(lengths)[int(len(lengths)*0.95)]:.0f}")
                print(f"  P99: {sorted(lengths)[int(len(lengths)*0.99)]:.0f}")
            print()
    
    def load_alpaca(self):
        """Load Alpaca dataset"""
        if self.dataset_path and os.path.exists(self.dataset_path):
            print(f"Loading from local file: {self.dataset_path}")
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            print("Please provide dataset path with --dataset-path")
            raise FileNotFoundError("Dataset file not found")
        
        for item in data:
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            
            if not instruction:
                continue
            
            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}"
            else:
                prompt = instruction
            
            self.data.append({
                "prompt": prompt,
                "type": "alpaca"
            })
    
    def load_sharegpt(self):
        """Load ShareGPT dataset"""
        if self.dataset_path and os.path.exists(self.dataset_path):
            print(f"Loading from local file: {self.dataset_path}")
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            print("Please provide dataset path with --dataset-path")
            raise FileNotFoundError("Dataset file not found")
        
        skipped = 0
        for idx, item in enumerate(data):
            conversations = item.get("conversations", [])
            if not conversations:
                skipped += 1
                continue
            
            messages = []
            for conv in conversations:
                from_field = conv.get("from", "")
                value = conv.get("value", "")
                
                if not value or not value.strip():
                    continue
                
                if from_field in ["human", "user"]:
                    role = "user"
                elif from_field in ["gpt", "assistant"]:
                    role = "assistant"
                else:
                    continue
                
                messages.append({
                    "role": role,
                    "content": value.strip()
                })
            
            if not messages or messages[0]["role"] != "user" or not messages[0]["content"]:
                skipped += 1
                continue
            
            self.data.append({
                "messages": messages,
                "prompt": messages[0]["content"],
                "type": "sharegpt",
                "id": item.get("id", f"item_{idx}")
            })
        
        if skipped > 0:
            print(f"Skipped {skipped} invalid conversations")
    
    def load_longbench(self):
        """Load LongBench dataset"""
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            print("Please provide dataset path with --dataset-path")
            raise FileNotFoundError("Dataset file not found")
        
        print(f"Loading LongBench from: {self.dataset_path}")
        
        skipped = 0
        truncated = 0
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                    
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Line {idx}: JSON decode error: {e}")
                    skipped += 1
                    continue
                
                # LongBench 数据格式
                context = item.get("context", "").strip()
                input_text = item.get("input", "").strip()
                
                if not context and not input_text:
                    skipped += 1
                    continue
                
                # 构建 prompt
                if context and input_text:
                    prompt = f"{context}\n\nQuestion: {input_text}"
                elif context:
                    prompt = context
                else:
                    prompt = input_text
                
                # 记录原始长度
                original_length = len(prompt)
                
                # 截断处理
                if self.max_length:
                    prompt = self._truncate_text(prompt, self.max_length)
                    if len(prompt) < original_length:
                        truncated += 1
                
                self.data.append({
                    "prompt": prompt,
                    "type": "longbench",
                    "task": os.path.basename(self.dataset_path).replace(".jsonl", ""),
                    "original_length": item.get("length", len(context)),
                    "answers": item.get("answers", []),
                    "id": idx
                })
        
        print(f"Loaded {len(self.data)} samples")
        if skipped > 0:
            print(f"Skipped {skipped} invalid items")
        if truncated > 0:
            print(f"Truncated {truncated} items to max_length={self.max_length}")
    
    def get_request(self, index: int = None) -> Dict:
        """Get a request"""
        if not self.data:
            raise ValueError("No valid data loaded")
        
        if index is None:
            index = random.randint(0, len(self.data) - 1)
        return self.data[index]
    
    def get_random_requests(self, num: int) -> List[Dict]:
        """Get multiple random requests"""
        if not self.data:
            raise ValueError("No valid data loaded")
        return random.choices(self.data, k=num)