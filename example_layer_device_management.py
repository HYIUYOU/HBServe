#!/usr/bin/env python3
"""
HBServe 动态层设备管理示例

这个示例展示了如何在HBServe的实际使用场景中应用动态层设备管理功能，
将不同的层分配到不同的GPU设备上执行。
"""

import os
import torch
from HBserve import LLM, SamplingParams
from transformers import AutoTokenizer


def example_basic_usage():
    """示例：基本使用 - 加载模型并进行推理"""
    print("=== 基本使用示例 ===")
    
    # 设置模型路径（请根据实际情况修改）
    model_path = os.path.expanduser("../Qwen3-0.6B")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请修改model_path变量为正确的模型路径")
        return None
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 创建LLM实例
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        
        # 设置采样参数
        sampling_params = SamplingParams(temperature=0.6, max_tokens=100)
        
        # 准备提示词
        prompts = [
            "introduce yourself",
            "what is artificial intelligence?",
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
        for prompt, output in zip(formatted_prompts, outputs):
            print("\n" + "="*50)
            print(f"Prompt: {prompt[:100]}...")
            print(f"Completion: {output['text']}")
        
        return llm
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None


def example_layer_device_management():
    """示例：动态层设备管理"""
    print("\n=== 动态层设备管理示例 ===")
    
    # 设置模型路径
    model_path = os.path.expanduser("../Qwen3-0.6B")
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return None
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 创建LLM实例
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        
        # 获取模型实例（假设可以访问底层的Qwen3Model）
        # 注意：这需要根据HBServe的实际API进行调整
        if hasattr(llm, 'model') and hasattr(llm.model, 'model'):
            model = llm.model.model  # 获取Qwen3Model实例
            
            print("开始动态调整层设备分布...")
            
            # 示例1：将第10层移动到GPU 1
            print("将第10层移动到GPU 1...")
            try:
                model.move_layer_to_device(9, 'cuda:1')
                print(f"第10层当前设备: {model.get_layer_device(9)}")
            except Exception as e:
                print(f"移动层时出错: {e}")
            
            # 示例2：批量设置层设备分布
            print("\n批量设置层设备分布...")
            layer_device_map = {
                0: 'cuda:0',   # 第1层在GPU 0
                1: 'cuda:0',   # 第2层在GPU 0
                9: 'cuda:1',   # 第10层在GPU 1
                10: 'cuda:1',  # 第11层在GPU 1
            }
            
            try:
                model.set_layer_device_distribution(layer_device_map)
                
                # 显示层设备分布
                print("\n当前层设备分布:")
                for layer_id in range(min(15, len(model.layers))):
                    device = model.get_layer_device(layer_id)
                    print(f"层 {layer_id+1:2d}: {device}")
                    
            except Exception as e:
                print(f"批量设置设备分布时出错: {e}")
        
        else:
            print("无法访问底层模型，跳过层设备管理示例")
        
        return llm
        
    except Exception as e:
        print(f"层设备管理示例出错: {e}")
        return None


def example_runtime_adjustment():
    """示例：运行时动态调整"""
    print("\n=== 运行时动态调整示例 ===")
    
    model_path = os.path.expanduser("../Qwen3-0.6B")
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return None
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 创建LLM实例
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        
        # 设置采样参数
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        
        # 第一次推理
        print("第一次推理（默认设备分布）...")
        prompt = "Tell me a short story about a robot."
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        outputs1 = llm.generate([formatted_prompt], sampling_params)
        print(f"第一次推理结果: {outputs1[0]['text'][:100]}...")
        
        # 动态调整层设备分布
        if hasattr(llm, 'model') and hasattr(llm.model, 'model'):
            model = llm.model.model
            
            print("\n动态调整：将部分层移动到GPU 1...")
            try:
                # 将第5-8层移动到GPU 1
                for layer_id in range(4, 8):
                    model.move_layer_to_device(layer_id, 'cuda:1')
                
                print("层设备调整完成，进行第二次推理...")
                
                # 第二次推理
                outputs2 = llm.generate([formatted_prompt], sampling_params)
                print(f"第二次推理结果: {outputs2[0]['text'][:100]}...")
                
                # 比较结果
                if outputs1[0]['text'] == outputs2[0]['text']:
                    print("✅ 两次推理结果一致，层设备调整成功！")
                else:
                    print("⚠️ 两次推理结果不同，可能是由于随机性或其他因素")
                    
            except Exception as e:
                print(f"运行时调整出错: {e}")
        
        return llm
        
    except Exception as e:
        print(f"运行时调整示例出错: {e}")
        return None


def example_memory_optimization():
    """示例：内存优化策略"""
    print("\n=== 内存优化示例 ===")
    
    model_path = os.path.expanduser("../Qwen3-0.6B")
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return None
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 创建LLM实例
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        
        if hasattr(llm, 'model') and hasattr(llm.model, 'model'):
            model = llm.model.model
            
            print("应用内存优化策略...")
            
            # 策略：将后半部分层移动到GPU 1以节省GPU 0的内存
            num_layers = len(model.layers)
            split_point = num_layers // 2
            
            print(f"将前{split_point}层保持在GPU 0，后{num_layers - split_point}层移动到GPU 1...")
            
            try:
                for layer_id in range(split_point, num_layers):
                    model.move_layer_to_device(layer_id, 'cuda:1')
                
                # 显示优化后的分布
                print("\n内存优化后的层分布:")
                gpu_0_layers = []
                gpu_1_layers = []
                
                for layer_id in range(len(model.layers)):
                    device = model.get_layer_device(layer_id)
                    if device == torch.device('cuda:0'):
                        gpu_0_layers.append(layer_id + 1)
                    elif device == torch.device('cuda:1'):
                        gpu_1_layers.append(layer_id + 1)
                
                print(f"GPU 0: 层 {gpu_0_layers}")
                print(f"GPU 1: 层 {gpu_1_layers}")
                
                # 测试优化后的推理性能
                print("\n测试优化后的推理...")
                sampling_params = SamplingParams(temperature=0.6, max_tokens=30)
                prompt = "What is the meaning of life?"
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                
                outputs = llm.generate([formatted_prompt], sampling_params)
                print(f"优化后推理结果: {outputs[0]['text']}")
                
            except Exception as e:
                print(f"内存优化出错: {e}")
        
        return llm
        
    except Exception as e:
        print(f"内存优化示例出错: {e}")
        return None


def main():
    """主函数：运行所有示例"""
    print("HBServe 动态层设备管理示例")
    print("=" * 60)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，示例将无法正常运行")
        print("请确保在有CUDA的环境中运行此示例")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU设备")
    
    if num_gpus < 2:
        print("警告: 需要至少2个GPU来演示跨设备层分布")
        print("示例将仅展示基本功能")
    
    try:
        # 运行示例
        print("\n1. 基本使用示例")
        llm1 = example_basic_usage()
        
        if llm1 is not None:
            print("\n2. 动态层设备管理示例")
            llm2 = example_layer_device_management()
            
            if num_gpus >= 2:
                print("\n3. 运行时动态调整示例")
                llm3 = example_runtime_adjustment()
                
                print("\n4. 内存优化示例")
                llm4 = example_memory_optimization()
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("\n使用说明:")
        print("- 确保模型路径正确")
        print("- 确保有足够的GPU内存")
        print("- 根据实际需求调整层设备分布")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请检查模型路径和依赖项")


if __name__ == "__main__":
    main()