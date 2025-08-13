"""
Chunked Prefill实现，支持分块处理长序列
"""
import asyncio
import torch
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import math

from hbserve.engine.sequence import Sequence
from hbserve.engine.model_runner import ModelRunner
from .config import PdDisaggConfig


@dataclass
class ChunkInfo:
    """分块信息"""
    chunk_id: int
    start_pos: int
    end_pos: int
    token_ids: List[int]
    processed: bool = False
    kv_cache: Optional[torch.Tensor] = None
    
    @property
    def chunk_size(self) -> int:
        return len(self.token_ids)


@dataclass
class ChunkedSequence:
    """分块序列"""
    sequence: Sequence
    chunks: List[ChunkInfo]
    current_chunk_idx: int = 0
    total_processed_tokens: int = 0
    kv_cache_accumulated: Optional[torch.Tensor] = None
    
    @property
    def is_completed(self) -> bool:
        return all(chunk.processed for chunk in self.chunks)
    
    @property
    def next_chunk(self) -> Optional[ChunkInfo]:
        if self.current_chunk_idx < len(self.chunks):
            return self.chunks[self.current_chunk_idx]
        return None


class CudaGraphManager:
    """CUDA Graph管理器"""
    
    def __init__(self, config: PdDisaggConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = config.cuda_graph_enabled
        
        # CUDA Graph缓存
        self.graph_cache: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.static_outputs: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        
    def can_use_cuda_graph(self, batch_size: int, seq_len: int) -> bool:
        """检查是否可以使用CUDA Graph"""
        if not self.enabled:
            return False
            
        # 只对小batch和适中长度的序列使用CUDA Graph
        if batch_size > 4:
            return False
            
        if seq_len > self.config.cuda_graph_max_seq_len:
            return False
            
        return True
        
    def get_or_create_graph(self, 
                           batch_size: int, 
                           seq_len: int,
                           model_runner: ModelRunner) -> Optional[Tuple[torch.cuda.CUDAGraph, Dict, Dict]]:
        """获取或创建CUDA Graph"""
        if not self.can_use_cuda_graph(batch_size, seq_len):
            return None
            
        key = (batch_size, seq_len)
        
        if key in self.graph_cache:
            return (
                self.graph_cache[key],
                self.static_inputs[key],
                self.static_outputs[key]
            )
            
        try:
            # 创建静态输入
            static_inputs = self._create_static_inputs(batch_size, seq_len)
            
            # 预热运行
            with torch.cuda.graph(capture_error="eager"):
                for _ in range(3):  # warmup runs
                    _ = self._run_model_with_static_inputs(
                        model_runner, static_inputs
                    )
                    
            # 创建CUDA Graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error="eager"):
                static_outputs = self._run_model_with_static_inputs(
                    model_runner, static_inputs
                )
                
            # 缓存
            self.graph_cache[key] = graph
            self.static_inputs[key] = static_inputs
            self.static_outputs[key] = static_outputs
            
            self.logger.info(f"创建CUDA Graph: batch_size={batch_size}, seq_len={seq_len}")
            
            return graph, static_inputs, static_outputs
            
        except Exception as e:
            self.logger.warning(f"创建CUDA Graph失败: {e}")
            return None
            
    def _create_static_inputs(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """创建静态输入张量"""
        device = torch.cuda.current_device()
        
        return {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
            "position_ids": torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        }
        
    def _run_model_with_static_inputs(self, 
                                     model_runner: ModelRunner,
                                     static_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """使用静态输入运行模型"""
        # 这里需要根据hbserve的实际接口调整
        # 简化版实现
        with torch.no_grad():
            # 模拟模型前向传播
            logits = torch.randn(
                static_inputs["input_ids"].shape[0],
                static_inputs["input_ids"].shape[1],
                50257,  # vocab_size
                device=static_inputs["input_ids"].device,
                dtype=torch.float16
            )
            
            return {"logits": logits}
            
    def run_with_graph(self,
                      graph: torch.cuda.CUDAGraph,
                      static_inputs: Dict[str, torch.Tensor],
                      static_outputs: Dict[str, torch.Tensor],
                      actual_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """使用CUDA Graph运行"""
        # 复制实际输入到静态张量
        for key, tensor in actual_inputs.items():
            if key in static_inputs:
                static_inputs[key].copy_(tensor)
                
        # 运行图
        graph.replay()
        
        # 返回输出副本
        outputs = {}
        for key, tensor in static_outputs.items():
            outputs[key] = tensor.clone()
            
        return outputs
        
    def clear_cache(self):
        """清理缓存"""
        self.graph_cache.clear()
        self.static_inputs.clear()
        self.static_outputs.clear()


class ChunkedPrefillManager:
    """分块Prefill管理器"""
    
    def __init__(self, config: PdDisaggConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 分块配置
        self.chunk_size = config.chunk_size
        self.max_chunk_prefill_tokens = config.max_chunk_prefill_tokens
        
        # CUDA Graph管理器
        self.cuda_graph_manager = CudaGraphManager(config)
        
        # 活跃的分块序列
        self.active_chunked_sequences: Dict[str, ChunkedSequence] = {}
        
        # 模型运行器引用
        self.model_runner: Optional[ModelRunner] = None
        
        # 统计信息
        self.total_chunks_processed = 0
        self.total_sequences_processed = 0
        self.cuda_graph_hits = 0
        self.cuda_graph_misses = 0
        
    async def initialize(self, model_runner: ModelRunner):
        """初始化分块prefill管理器"""
        self.model_runner = model_runner
        self.logger.info("分块Prefill管理器初始化完成")
        
    def should_use_chunked_prefill(self, sequence: Sequence) -> bool:
        """判断是否应该使用分块prefill"""
        return len(sequence.prompt_token_ids) > self.chunk_size
        
    async def chunked_prefill(self, sequence: Sequence) -> str:
        """执行分块prefill"""
        if not self.should_use_chunked_prefill(sequence):
            # 序列太短，使用标准prefill
            return await self._standard_prefill(sequence)
            
        try:
            # 创建分块序列
            chunked_seq = self._create_chunked_sequence(sequence)
            sequence_id = sequence.seq_id
            self.active_chunked_sequences[sequence_id] = chunked_seq
            
            self.logger.info(f"开始分块prefill: {sequence_id}, "
                           f"总长度: {len(sequence.prompt_token_ids)}, "
                           f"分块数: {len(chunked_seq.chunks)}")
            
            # 逐块处理
            while not chunked_seq.is_completed:
                await self._process_next_chunk(chunked_seq)
                
            # 生成KV缓存键
            kv_cache_key = f"chunked_kv_{sequence_id}_{int(time.time() * 1000)}"
            
            # 清理
            del self.active_chunked_sequences[sequence_id]
            self.total_sequences_processed += 1
            
            self.logger.info(f"分块prefill完成: {sequence_id}")
            
            return kv_cache_key
            
        except Exception as e:
            self.logger.error(f"分块prefill失败 {sequence.seq_id}: {e}")
            if sequence.seq_id in self.active_chunked_sequences:
                del self.active_chunked_sequences[sequence.seq_id]
            raise
            
    def _create_chunked_sequence(self, sequence: Sequence) -> ChunkedSequence:
        """创建分块序列"""
        token_ids = sequence.prompt_token_ids
        chunks = []
        
        # 计算分块
        num_chunks = math.ceil(len(token_ids) / self.chunk_size)
        
        for i in range(num_chunks):
            start_pos = i * self.chunk_size
            end_pos = min(start_pos + self.chunk_size, len(token_ids))
            
            chunk = ChunkInfo(
                chunk_id=i,
                start_pos=start_pos,
                end_pos=end_pos,
                token_ids=token_ids[start_pos:end_pos]
            )
            chunks.append(chunk)
            
        return ChunkedSequence(sequence=sequence, chunks=chunks)
        
    async def _process_next_chunk(self, chunked_seq: ChunkedSequence):
        """处理下一个分块"""
        chunk = chunked_seq.next_chunk
        if not chunk:
            return
            
        try:
            self.logger.debug(f"处理分块 {chunk.chunk_id}/{len(chunked_seq.chunks)}, "
                            f"长度: {chunk.chunk_size}")
            
            # 选择处理策略
            if self._should_use_cuda_graph(chunk):
                await self._process_chunk_with_cuda_graph(chunked_seq, chunk)
                self.cuda_graph_hits += 1
            else:
                await self._process_chunk_standard(chunked_seq, chunk)
                self.cuda_graph_misses += 1
                
            chunk.processed = True
            chunked_seq.current_chunk_idx += 1
            chunked_seq.total_processed_tokens += chunk.chunk_size
            self.total_chunks_processed += 1
            
        except Exception as e:
            self.logger.error(f"处理分块失败: {e}")
            raise
            
    def _should_use_cuda_graph(self, chunk: ChunkInfo) -> bool:
        """判断是否使用CUDA Graph"""
        # 对于固定大小的分块，更容易使用CUDA Graph
        return (self.cuda_graph_manager.can_use_cuda_graph(1, chunk.chunk_size) and
                chunk.chunk_size == self.chunk_size)
                
    async def _process_chunk_with_cuda_graph(self, 
                                           chunked_seq: ChunkedSequence,
                                           chunk: ChunkInfo):
        """使用CUDA Graph处理分块"""
        try:
            graph_info = self.cuda_graph_manager.get_or_create_graph(
                batch_size=1,
                seq_len=chunk.chunk_size,
                model_runner=self.model_runner
            )
            
            if not graph_info:
                # CUDA Graph创建失败，回退到标准处理
                await self._process_chunk_standard(chunked_seq, chunk)
                return
                
            graph, static_inputs, static_outputs = graph_info
            
            # 准备实际输入
            device = torch.cuda.current_device()
            actual_inputs = {
                "input_ids": torch.tensor([chunk.token_ids], dtype=torch.long, device=device),
                "attention_mask": torch.ones(1, chunk.chunk_size, dtype=torch.long, device=device),
                "position_ids": torch.arange(
                    chunked_seq.total_processed_tokens,
                    chunked_seq.total_processed_tokens + chunk.chunk_size,
                    dtype=torch.long, device=device
                ).unsqueeze(0)
            }
            
            # 使用CUDA Graph运行
            outputs = self.cuda_graph_manager.run_with_graph(
                graph, static_inputs, static_outputs, actual_inputs
            )
            
            # 处理输出（这里需要根据实际需求处理KV缓存）
            chunk.kv_cache = outputs.get("logits")  # 简化处理
            
            self.logger.debug(f"CUDA Graph处理分块完成: {chunk.chunk_id}")
            
        except Exception as e:
            self.logger.warning(f"CUDA Graph处理失败，回退到标准处理: {e}")
            await self._process_chunk_standard(chunked_seq, chunk)
            
    async def _process_chunk_standard(self, 
                                    chunked_seq: ChunkedSequence,
                                    chunk: ChunkInfo):
        """标准方式处理分块"""
        try:
            # 创建临时序列
            temp_sequence = Sequence(chunk.token_ids, chunked_seq.sequence.sampling_params)
            
            # 调用模型运行器（这里需要根据hbserve的实际接口调整）
            seqs = [temp_sequence]
            is_prefill = True
            
            # 设置正确的位置编码（考虑前面已处理的token）
            # 这里需要传递position offset信息
            
            # 执行模型推理
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            
            # 简化：只记录处理完成
            chunk.kv_cache = torch.tensor(token_ids)  # 简化处理
            
            self.logger.debug(f"标准处理分块完成: {chunk.chunk_id}")
            
        except Exception as e:
            self.logger.error(f"标准处理分块失败: {e}")
            raise
            
    async def _standard_prefill(self, sequence: Sequence) -> str:
        """标准prefill（非分块）"""
        try:
            # 检查是否可以使用CUDA Graph
            batch_size = 1
            seq_len = len(sequence.prompt_token_ids)
            
            if self.cuda_graph_manager.can_use_cuda_graph(batch_size, seq_len):
                graph_info = self.cuda_graph_manager.get_or_create_graph(
                    batch_size, seq_len, self.model_runner
                )
                
                if graph_info:
                    return await self._standard_prefill_with_cuda_graph(
                        sequence, graph_info
                    )
                    
            # 标准处理
            seqs = [sequence]
            is_prefill = True
            
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            
            kv_cache_key = f"standard_kv_{sequence.seq_id}_{int(time.time() * 1000)}"
            
            self.logger.info(f"标准prefill完成: {sequence.seq_id}")
            
            return kv_cache_key
            
        except Exception as e:
            self.logger.error(f"标准prefill失败: {e}")
            raise
            
    async def _standard_prefill_with_cuda_graph(self, 
                                              sequence: Sequence,
                                              graph_info: Tuple) -> str:
        """使用CUDA Graph的标准prefill"""
        try:
            graph, static_inputs, static_outputs = graph_info
            
            # 准备输入
            device = torch.cuda.current_device()
            actual_inputs = {
                "input_ids": torch.tensor([sequence.prompt_token_ids], dtype=torch.long, device=device),
                "attention_mask": torch.ones(1, len(sequence.prompt_token_ids), dtype=torch.long, device=device),
                "position_ids": torch.arange(len(sequence.prompt_token_ids), dtype=torch.long, device=device).unsqueeze(0)
            }
            
            # 运行CUDA Graph
            outputs = self.cuda_graph_manager.run_with_graph(
                graph, static_inputs, static_outputs, actual_inputs
            )
            
            kv_cache_key = f"graph_kv_{sequence.seq_id}_{int(time.time() * 1000)}"
            
            self.logger.info(f"CUDA Graph标准prefill完成: {sequence.seq_id}")
            self.cuda_graph_hits += 1
            
            return kv_cache_key
            
        except Exception as e:
            self.logger.error(f"CUDA Graph标准prefill失败: {e}")
            self.cuda_graph_misses += 1
            raise
            
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "total_sequences_processed": self.total_sequences_processed,
            "active_chunked_sequences": len(self.active_chunked_sequences),
            "cuda_graph_hits": self.cuda_graph_hits,
            "cuda_graph_misses": self.cuda_graph_misses,
            "cuda_graph_hit_rate": (
                self.cuda_graph_hits / max(1, self.cuda_graph_hits + self.cuda_graph_misses)
            ),
            "chunk_size": self.chunk_size,
            "max_chunk_prefill_tokens": self.max_chunk_prefill_tokens,
            "cuda_graphs_cached": len(self.cuda_graph_manager.graph_cache)
        }
        
    async def cleanup(self):
        """清理资源"""
        self.active_chunked_sequences.clear()
        self.cuda_graph_manager.clear_cache()
        self.logger.info("分块Prefill管理器已清理")


class PrefillOptimizer:
    """Prefill优化器，包含各种优化策略"""
    
    def __init__(self, config: PdDisaggConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def should_use_optimized_attention(self, seq_len: int) -> bool:
        """判断是否使用优化的attention"""
        # 对于长序列使用Flash Attention等优化
        return seq_len > 1024
        
    def get_optimal_chunk_size(self, seq_len: int, available_memory: int) -> int:
        """根据序列长度和可用内存计算最优分块大小"""
        base_chunk_size = self.config.chunk_size
        
        # 根据内存限制调整
        if available_memory < 1024 * 1024 * 1024:  # 1GB
            return min(base_chunk_size, 256)
        elif available_memory > 4 * 1024 * 1024 * 1024:  # 4GB
            return min(base_chunk_size * 2, 1024)
        else:
            return base_chunk_size
            
    def estimate_prefill_time(self, seq_len: int, use_chunked: bool = False) -> float:
        """估计prefill时间"""
        base_time_per_token = 0.001  # 1ms per token
        
        if use_chunked:
            # 分块prefill有额外开销
            overhead = 0.01  # 10ms overhead per chunk
            num_chunks = math.ceil(seq_len / self.config.chunk_size)
            return seq_len * base_time_per_token + num_chunks * overhead
        else:
            return seq_len * base_time_per_token
