#!/usr/bin/env python3
"""
HBServe 动态层设备管理 - 快速开始

这是一个简化的示例，展示如何在HBServe中快速使用动态层设备管理功能。
"""

import os
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer


def quick_start_example():
    """快速开始示例"""
    print("HBServe 动态层设备管理 - 快速开始")
    print("=" * 50)
    
    # 1. 设置模型路径（请根据实际情况修改）
    model_path = os.path.expanduser("../Qwen3-0.6B")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("请修改model_path变量为正确的模型路径")
        return
    
    try:
        # 2. 加载模型和tokenizer
        print("📥 加载模型...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        print("✅ 模型加载成功")
        
        # 3. 动态调整层设备分布
        print("\n🔧 调整层设备分布...")
        # 通过 llm.model_runner.model.model 访问 Qwen3Model
        try:
            from torch.cuda import device_count
            num_gpus = device_count()
        except Exception:
            num_gpus = 1

        if hasattr(llm, 'model_runner') and hasattr(llm.model_runner, 'model') and hasattr(llm.model_runner.model, 'model'):
            model = llm.model_runner.model.model

            if num_gpus < 2:
                print("⚠️ 仅检测到1个GPU，跳过跨设备层迁移示例")
            else:
                # 将第10层移动到GPU 1
                print("将第10层移动到GPU 1...")
                model.move_layer_to_device(9, 'cuda:1')

                # 批量设置更多层（示例）
                layer_device_map = {
                    0: 'cuda:0',   # 第1层在GPU 0
                    1: 'cuda:0',   # 第2层在GPU 0
                    10: 'cuda:1',  # 第11层在GPU 1
                    11: 'cuda:1',  # 第12层在GPU 1
                }
                model.set_layer_device_distribution(layer_device_map)

                print("✅ 层设备分布调整完成")

            # 显示当前分布（不论是否迁移成功，都打印前若干层所在设备）
            print("\n📊 当前层设备分布:")
            for layer_id in range(min(15, len(model.layers))):
                device = model.get_layer_device(layer_id)
                print(f"  层 {layer_id+1:2d}: {device}")
        else:
            print("⚠️ 无法通过 llm.model_runner 访问底层模型，跳过层设备管理")
        
        # 4. 进行推理测试
        print("\n🚀 开始推理测试...")
        sampling_params = SamplingParams(temperature=0.6, max_tokens=50)
        
        prompts = [
            "Hello, how are you?",
            "What is machine learning?",
        ]
        
        # 应用聊天模板
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            for prompt in prompts
        ]
        
        # 生成回复
        outputs = llm.generate(formatted_prompts, sampling_params)
        
        # 显示结果
        print("\n📝 推理结果:")
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"\n问题 {i+1}: {prompt}")
            print(f"回答: {output['text']}")
        
        print("\n✅ 快速开始示例完成！")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查模型路径和依赖项")


if __name__ == "__main__":
    quick_start_example()
