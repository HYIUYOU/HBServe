#!/usr/bin/env python3
"""
Torch Profiler 工具：用于分析 HBserve 中前向优化路径的性能瓶颈。

核心能力：
- 支持对 Qwen3 模型的 baseline / Attention Offload / KV Head Split / Layer Replication 场景做性能采样。
- 自动初始化单进程分布式环境（tp=1），避免手动准备 torch.distributed。
- 同时输出 TensorBoard 兼容的 trace 以及控制台汇总表，便于定位 CPU/GPU 时间热点与显存使用情况。

使用示例：
    python -m HBserve.tools.profile_forward \
        --mode attn_offload \
        --layer-id 10 \
        --primary-device cuda:0 \
        --offload-device cuda:1 \
        --seq-len 2048 \
        --batch-size 2 \
        --profile-steps 40 \
        --trace-dir /tmp/hbserve_profile

更多命令行参数可通过 `-h` 查看。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from transformers import Qwen3Config

from HBserve.models.qwen3 import Qwen3ForCausalLM
from HBserve.utils.context import reset_context, set_context


@dataclass
class ForwardInputs:
    label: str
    input_ids: torch.Tensor
    positions: torch.Tensor
    context_kwargs: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HBserve Torch Profiler 工具")

    parser.add_argument("--mode", choices=["baseline", "attn_offload", "kv_split", "layer_replica"], default="baseline",
                        help="选择待分析的优化路径")
    parser.add_argument("--layer-id", type=int, default=0, help="应用优化的层索引（baseline 可忽略）")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="批次切分比例 (0,1)")
    parser.add_argument("--split-kv-index", type=int, default=None,
                        help="KV Head Split 的 KV 切分索引；默认取一半")
    parser.add_argument("--enable-autotune", action="store_true", help="启用可用的自适应调参逻辑")

    parser.add_argument("--primary-device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="原始层所在设备")
    parser.add_argument("--offload-device", type=str, default=None, help="Attention Offload / KV Split 的目标设备")
    parser.add_argument("--replica-device", type=str, default=None, help="Layer Replication 的副本设备")

    parser.add_argument("--batch-size", type=int, default=2, help="批大小（prefill 为序列个数，decode 为并发 token 数）")
    parser.add_argument("--seq-len", type=int, default=1024, help="prefill 阶段的序列长度（token 数）")
    parser.add_argument("--prefill-seq-len", type=int, default=None,
                        help="decode 之前用于构建 KV cache 的 prefill 长度，默认与 --seq-len 一致")
    parser.add_argument("--decode-context-len", type=int, default=None,
                        help="decode 阶段已有上下文长度，默认与 prefill 长度一致")
    parser.add_argument("--scenario", choices=["prefill", "decode"], default="prefill",
                        help="采样阶段：prefill 或 decode")

    parser.add_argument("--num-layers", type=int, default=4, help="模型层数（可调小以便实验）")
    parser.add_argument("--hidden-size", type=int, default=1024, help="隐藏层维度")
    parser.add_argument("--intermediate-size", type=int, default=None,
                        help="MLP 中间层维度，默认 4 * hidden-size")
    parser.add_argument("--num-heads", type=int, default=16, help="Attention 头数")
    parser.add_argument("--num-kv-heads", type=int, default=16, help="KV 头数")
    parser.add_argument("--max-position", type=int, default=4096, help="RoPE 最大位置")
    parser.add_argument("--vocab-size", type=int, default=32000, help="词表大小（随机权重实验即可）")

    parser.add_argument("--dist-backend", type=str, default="gloo", help="单进程初始化使用的分布式后端")

    parser.add_argument("--warmup-steps", type=int, default=10, help="Profiler 等待/预热步数")
    parser.add_argument("--profile-steps", type=int, default=40, help="Profiler 记录步数")
    parser.add_argument("--label", type=str, default="hbserve_forward", help="record_function 标签")
    parser.add_argument("--synchronize", action="store_true", help="每步结束后执行 torch.cuda.synchronize()")

    parser.add_argument("--record-shapes", action="store_true", help="记录算子输入形状")
    parser.add_argument("--profile-memory", action="store_true", help="采样显存占用")
    parser.add_argument("--with-stack", action="store_true", help="采样 Python 调用栈")
    parser.add_argument("--print-top", type=int, default=20, help="汇总表显示的算子条数")
    parser.add_argument("--sort-by", type=str, default="self_cuda_time_total",
                        help="汇总表排序字段 (self_cuda_time_total/self_cpu_time_total/etc)")

    parser.add_argument("--export-trace", action="store_true", help="导出 TensorBoard/Chrome trace")
    parser.add_argument("--trace-dir", type=str, default="./profiles",
                        help="trace 输出目录（--export-trace 时生效）")
    parser.add_argument("--run-name", type=str, default=None, help="trace 子目录名称，默认自动生成")

    return parser.parse_args()


def init_distributed(backend: str) -> None:
    if not dist.is_available():
        raise RuntimeError("torch.distributed 不可用，无法运行该脚本")
    if dist.is_initialized():
        return
    dist.init_process_group(backend=backend, rank=0, world_size=1)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def to_device(device_str: Optional[str]) -> Optional[torch.device]:
    if device_str is None:
        return None
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"当前环境无 CUDA，但请求使用 {device_str}")
    return device


def build_config(args: argparse.Namespace) -> Qwen3Config:
    intermediate = args.intermediate_size or args.hidden_size * 4
    config = Qwen3Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=intermediate,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        max_position_embeddings=args.max_position,
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
    )
    return config


def ensure_layer_devices(model: Qwen3ForCausalLM, device: torch.device) -> None:
    layer_map = {idx: device for idx in range(len(model.model.layers))}
    model.model.set_layer_device_distribution(layer_map)


def configure_optimizations(model: Qwen3ForCausalLM, args: argparse.Namespace,
                            primary_device: torch.device) -> None:
    mode = args.mode
    layer_id = args.layer_id

    if mode == "baseline":
        return

    if mode in {"attn_offload", "kv_split"}:
        offload_device = to_device(args.offload_device)
        if offload_device is None:
            raise ValueError("使用 Attention Offload/KV Split 时必须指定 --offload-device")

    if mode == "layer_replica":
        replica_device = to_device(args.replica_device)
        if replica_device is None:
            raise ValueError("使用 Layer Replication 时必须指定 --replica-device")
        model.model.replicate_layer_to_device(
            layer_id=layer_id,
            device=replica_device,
            split_ratio=args.split_ratio,
        )
        if args.enable_autotune:
            model.model.enable_replication_autotune(layer_id)
        return

    if mode == "attn_offload":
        model.model.attention_offload_by_batch(
            layer_id=layer_id,
            offload_device=offload_device,
            split_ratio=args.split_ratio,
            enable_autotune=args.enable_autotune,
        )
        return

    if mode == "kv_split":
        model.model.attention_offload_by_kv_head(
            layer_id=layer_id,
            offload_device=offload_device,
            split_kv_head_idx=args.split_kv_index,
            enable_autotune=args.enable_autotune,
        )


def make_prefill_inputs(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> ForwardInputs:
    tokens = batch_size * seq_len
    input_ids = torch.randint(0, vocab_size, (tokens,), device=device)
    positions = torch.cat([
        torch.arange(seq_len, device=device, dtype=torch.long)
        for _ in range(batch_size)
    ], dim=0)

    cu_seqlens = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * seq_len
    context_kwargs = dict(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        slot_mapping=torch.arange(tokens, device=device, dtype=torch.int32),
        context_lens=torch.full((batch_size,), seq_len, device=device, dtype=torch.int32),
        block_tables=None,
    )

    return ForwardInputs("prefill", input_ids, positions, context_kwargs)


def make_decode_inputs(batch_size: int, vocab_size: int, context_len: int,
                       device: torch.device, decode_position: Optional[int] = None) -> ForwardInputs:
    input_ids = torch.randint(0, vocab_size, (batch_size,), device=device)
    position_value = context_len if decode_position is None else decode_position
    positions = torch.full((batch_size,), position_value, device=device, dtype=torch.long)

    context_kwargs = dict(
        is_prefill=False,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=context_len + 1,
        max_seqlen_k=context_len + 1,
        slot_mapping=torch.arange(batch_size, device=device, dtype=torch.int32),
        context_lens=torch.full((batch_size,), context_len, device=device, dtype=torch.int32),
        block_tables=None,
    )

    return ForwardInputs("decode", input_ids, positions, context_kwargs)


def prepare_inputs(args: argparse.Namespace, device: torch.device) -> tuple[ForwardInputs, Optional[ForwardInputs]]:
    if args.scenario == "prefill":
        profile_inputs = make_prefill_inputs(args.batch_size, args.seq_len, args.vocab_size, device)
        return profile_inputs, None

    prefill_len = args.prefill_seq_len or args.seq_len
    context_len = args.decode_context_len or prefill_len

    warmup_inputs = make_prefill_inputs(args.batch_size, prefill_len, args.vocab_size, device)
    decode_inputs = make_decode_inputs(args.batch_size, args.vocab_size, context_len, device)
    return decode_inputs, warmup_inputs


def run_initial_prefill(model: Qwen3ForCausalLM, warmup_inputs: ForwardInputs, synchronize: bool) -> None:
    with torch.no_grad():
        model.model(warmup_inputs.input_ids, warmup_inputs.positions)
    if synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_profiler(args: argparse.Namespace, trace_dir: Optional[Path]):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    sched = schedule(wait=0, warmup=args.warmup_steps, active=args.profile_steps, repeat=1)

    if trace_dir is not None:
        trace_handler = tensorboard_trace_handler(str(trace_dir))

        def on_trace_ready(prof):  # noqa: ANN001
            trace_handler(prof)
            table = prof.key_averages().table(sort_by=args.sort_by, row_limit=args.print_top)
            print(table)

        return dict(activities=activities, schedule=sched, on_trace_ready=on_trace_ready,
                    record_shapes=args.record_shapes, profile_memory=args.profile_memory,
                    with_stack=args.with_stack)

    def on_trace_ready(prof):  # noqa: ANN001
        table = prof.key_averages().table(sort_by=args.sort_by, row_limit=args.print_top)
        print(table)

    return dict(activities=activities, schedule=sched, on_trace_ready=on_trace_ready,
                record_shapes=args.record_shapes, profile_memory=args.profile_memory,
                with_stack=args.with_stack)


def ensure_trace_dir(args: argparse.Namespace) -> Optional[Path]:
    if not args.export_trace:
        return None
    base = Path(args.trace_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.mode}_{args.scenario}_bs{args.batch_size}_sl{args.seq_len}"
    trace_dir = base / run_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def main() -> None:
    torch.manual_seed(0)
    args = parse_args()

    init_distributed(args.dist_backend)

    primary_device = to_device(args.primary_device)
    if primary_device is None:
        primary_device = torch.device("cpu")

    if primary_device.type == "cuda":
        torch.cuda.set_device(primary_device)

    config = build_config(args)
    model = Qwen3ForCausalLM(config)
    model.eval()

    ensure_layer_devices(model, primary_device)

    configure_optimizations(model, args, primary_device)

    if primary_device.type == "cuda":
        model = model.to(primary_device)
    else:
        model = model.to(primary_device)

    reset_context()
    profile_inputs, warmup_inputs = prepare_inputs(args, primary_device)

    if warmup_inputs is not None:
        set_context(**warmup_inputs.context_kwargs)
        run_initial_prefill(model, warmup_inputs, args.synchronize)

    trace_dir = ensure_trace_dir(args)
    profiler_kwargs = make_profiler(args, trace_dir)

    total_steps = args.warmup_steps + args.profile_steps

    with torch.no_grad():
        set_context(**profile_inputs.context_kwargs)
        with profile(**profiler_kwargs) as prof:
            for step in range(total_steps):
                with record_function(args.label):
                    model.model(profile_inputs.input_ids, profile_inputs.positions)
                if args.synchronize and torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof.step()

    reset_context()
    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup_distributed()
        raise

