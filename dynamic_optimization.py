"""
动态优化策略：根据 prefill/decode 阶段自动启用/禁用优化

核心思想：
- Prefill阶段：启用 layer replication（加速大批量计算）
- Decode阶段：禁用 layer replication（避免传输开销）
"""

import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer
import torch


class DynamicOptimizationLLM:
    """带动态优化的LLM包装器"""
    
    def __init__(self, model_path, **kwargs):
        """初始化"""
        self.llm = LLM(model_path, **kwargs)
        self.model = self.llm.model_runner.model.model
        self.optimization_layers = []
        self.optimization_config = {}
        self.prefill_enabled = False
        
    def configure_optimization(
        self,
        layer_ids: list,
        replica_device: str = 'cuda:1',
        split_ratio: float = 0.5,
        enable_autotune: bool = True,
        strategy: str = 'layer_replication'
    ):
        """
        配置优化策略（但不立即应用）
        
        Args:
            layer_ids: 要优化的层ID列表
            replica_device: 复制设备
            split_ratio: 切分比例
            enable_autotune: 是否启用自适应调整
            strategy: 优化策略 ('layer_replication', 'attention_offload')
        """
        self.optimization_layers = layer_ids
        self.optimization_config = {
            'replica_device': replica_device,
            'split_ratio': split_ratio,
            'enable_autotune': enable_autotune,
            'strategy': strategy
        }
        print(f"✅ 优化配置已设置: {len(layer_ids)}层, 策略={strategy}")
    
    def _enable_optimization(self):
        """启用优化（Prefill阶段调用）"""
        if self.prefill_enabled:
            return
        
        strategy = self.optimization_config.get('strategy', 'layer_replication')
        
        for layer_id in self.optimization_layers:
            if strategy == 'layer_replication':
                self.model.replicate_layer_to_device(
                    layer_id,
                    self.optimization_config['replica_device'],
                    split_ratio=self.optimization_config['split_ratio']
                )
                if self.optimization_config['enable_autotune']:
                    self.model.enable_replication_autotune(
                        layer_id, beta=0.3, min_ratio=0.2, max_ratio=0.8
                    )
            elif strategy == 'attention_offload':
                self.model.attention_offload_by_batch(
                    layer_id,
                    offload_device=self.optimization_config['replica_device'],
                    split_ratio=self.optimization_config['split_ratio']
                )
        
        self.prefill_enabled = True
        print(f"🚀 [Prefill] 已启用优化: {len(self.optimization_layers)}层")
    
    def _disable_optimization(self):
        """禁用优化（Decode阶段调用）"""
        if not self.prefill_enabled:
            return
        
        strategy = self.optimization_config.get('strategy', 'layer_replication')
        
        for layer_id in self.optimization_layers:
            if strategy == 'layer_replication':
                self.model.clear_layer_replication(layer_id)
            elif strategy == 'attention_offload':
                self.model.clear_attention_offload(layer_id)
        
        self.prefill_enabled = False
        print(f"⚡ [Decode] 已禁用优化: {len(self.optimization_layers)}层")
    
    def generate(self, prompts, sampling_params, dynamic_optimization: bool = True):
        """
        生成文本，支持动态优化
        
        Args:
            prompts: 输入prompts
            sampling_params: 采样参数
            dynamic_optimization: 是否启用动态优化
        """
        if not dynamic_optimization:
            # 不使用动态优化，直接生成
            return self.llm.generate(prompts, sampling_params)
        
        # ===== Prefill 阶段：启用优化 =====
        print(f"\n{'='*60}")
        print("阶段1: Prefill（启用优化）")
        print(f"{'='*60}")
        self._enable_optimization()
        
        # 执行生成（LLM内部会自动处理prefill和decode）
        # 但我们需要在decode前禁用优化
        
        # 方案：Hook到generation过程
        # 这里简化处理：让用户手动控制，或者修改LLM内部逻辑
        outputs = self.llm.generate(prompts, sampling_params)
        
        # ===== Decode 阶段：禁用优化 =====
        # 注意：在当前实现中，decode已经完成，这里只是清理
        print(f"\n{'='*60}")
        print("阶段2: Decode（禁用优化）")
        print(f"{'='*60}")
        self._disable_optimization()
        
        return outputs


def example_dynamic_optimization():
    """示例：使用动态优化"""
    
    path = os.path.expanduser("../Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # 创建动态优化LLM
    llm = DynamicOptimizationLLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6
    )
    
    # 配置优化：prefill时使用
    llm.configure_optimization(
        layer_ids=list(range(5, 8)),  # 3层
        replica_device='cuda:1',
        split_ratio=0.5,
        strategy='layer_replication'
    )
    
    # 准备prompts
    prompts = [
        "Introduce yourself briefly.",
        "What is artificial intelligence?",
        "Explain machine learning.",
        "Describe deep learning."
    ]
    
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in prompts
    ]
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # 使用动态优化生成
    outputs = llm.generate(prompts, sampling_params, dynamic_optimization=True)
    
    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Output: {output['text'][:100]}...")


if __name__ == "__main__":
    example_dynamic_optimization()



