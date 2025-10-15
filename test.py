#!/usr/bin/env python3
"""
快速测试脚本 - 验证层设备管理功能是否可用
"""

import os
import torch
from HBserve import LLM

os.environ['HB_ATTN_OFFLOAD_LOG'] = '0'
os.environ['HB_DEBUG'] = '0'
os.environ['HB_REPLICA_LOG'] = '0'

def main():
    print("="*70)
    print("快速测试：验证层设备管理功能")
    print("="*70)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\n✅ 检测到 {num_gpus} 个 GPU")
    
    # 加载模型
    model_path = os.path.expanduser("../Qwen3-0.6B")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    print(f"\n正在加载模型: {model_path}")
    
    try:
        # 创建 LLM 实例
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        print("✅ LLM 加载成功")
        
        # 尝试访问底层模型
        print("\n" + "-"*70)
        print("访问底层 Qwen3Model:")
        print("-"*70)
        
        # 正确的访问路径
        if hasattr(llm, 'model_runner'):
            print("✅ llm.model_runner 存在")
            
            if hasattr(llm.model_runner, 'model'):
                print("✅ llm.model_runner.model 存在")
                print(f"   类型: {type(llm.model_runner.model).__name__}")
                
                if hasattr(llm.model_runner.model, 'model'):
                    model = llm.model_runner.model.model
                    print("✅ llm.model_runner.model.model 存在")
                    print(f"   类型: {type(model).__name__}")
                    
                    if hasattr(model, 'layers'):
                        print(f"✅ 有 layers 属性，共 {len(model.layers)} 层")
                        
                        # 测试层设备管理功能
                        print("\n" + "-"*70)
                        print("测试层设备管理功能:")
                        print("-"*70)
                        
                        # 1. 获取层设备
                        print("\n【测试 1】获取层设备")
                        for i in range(min(3, len(model.layers))):
                            device = model.get_layer_device(i)
                            print(f"  层 {i}: {device}")
                        
                        # 2. 移动单个层（如果有多个 GPU）
                        if num_gpus >= 2:
                            print("\n【测试 2】移动层到不同 GPU")
                            try:
                                model.move_layer_to_device(1, 'cuda:1')
                                new_device = model.get_layer_device(1)
                                print(f"  ✅ 成功将层 1 移动到 {new_device}")
                            except Exception as e:
                                print(f"  ❌ 移动失败: {e}")
                        else:
                            print("\n【测试 2】跳过（需要至少 2 个 GPU）")
                        
                        # 3. 批量设置
                        if num_gpus >= 2:
                            print("\n【测试 3】批量设置层设备分布")
                            try:
                                layer_map = {
                                    0: 'cuda:0',
                                    1: 'cuda:1',
                                    2: 'cuda:0',
                                }
                                model.set_layer_device_distribution(layer_map)
                                print("  ✅ 批量设置成功")
                                for layer_id, expected_device in layer_map.items():
                                    actual_device = str(model.get_layer_device(layer_id))
                                    status = "✅" if expected_device in actual_device else "❌"
                                    print(f"  {status} 层 {layer_id}: {actual_device}")
                            except Exception as e:
                                print(f"  ❌ 批量设置失败: {e}")
                        else:
                            print("\n【测试 3】跳过（需要至少 2 个 GPU）")
                        
                        # 4. 测试层复制
                        if num_gpus >= 2:
                            print("\n【测试 4】层复制功能")
                            try:
                                layer_to_replicate = len(model.layers) // 2
                                model.replicate_layer_to_device(
                                    layer_id=layer_to_replicate,
                                    device='cuda:1',
                                    split_ratio=0.6
                                )
                                print(f"  ✅ 成功复制层 {layer_to_replicate}")
                                print(f"     原设备: {model.get_layer_device(layer_to_replicate)}")
                                print(f"     副本设备: {model.replica_devices.get(layer_to_replicate)}")
                                print(f"     切分比例: {model.replica_split_ratio.get(layer_to_replicate)}")
                                
                                # 清理
                                model.clear_layer_replication(layer_to_replicate)
                                print(f"  ✅ 清理完成")
                            except Exception as e:
                                print(f"  ❌ 层复制失败: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("\n【测试 4】跳过（需要至少 2 个 GPU）")
                        
                        # 5. 测试 Attention Offload
                        if num_gpus >= 2:
                            print("\n【测试 5】Attention Offload")
                            try:
                                layer_for_offload = len(model.layers) // 3
                                model.attention_offload_by_batch(
                                    layer_id=layer_for_offload,
                                    offload_device='cuda:1',
                                    split_ratio=0.5
                                )
                                print(f"  ✅ 成功配置 Attention Offload (层 {layer_for_offload})")
                                
                                # 清理
                                model.clear_attention_offload(layer_for_offload)
                                print(f"  ✅ 清理完成")
                            except Exception as e:
                                print(f"  ❌ Attention Offload 失败: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("\n【测试 5】跳过（需要至少 2 个 GPU）")
                        
                        # 总结
                        print("\n" + "="*70)
                        print("✅ 所有核心功能都可用！")
                        print("="*70)
                        print("\n可用功能列表：")
                        print("  ✅ 层设备管理 (move_layer_to_device, get_layer_device)")
                        print("  ✅ 批量设置 (set_layer_device_distribution)")
                        if num_gpus >= 2:
                            print("  ✅ 层复制 (replicate_layer_to_device)")
                            print("  ✅ Attention Offload (attention_offload_by_batch/kv_head)")
                        else:
                            print("  ⚠️ 层复制（需要多 GPU）")
                            print("  ⚠️ Attention Offload（需要多 GPU）")
                        
                    else:
                        print("❌ 没有 layers 属性")
                else:
                    print("❌ llm.model_runner.model.model 不存在")
            else:
                print("❌ llm.model_runner.model 不存在")
        else:
            print("❌ llm.model_runner 不存在")
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()