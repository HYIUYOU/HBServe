from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, Tuple, Optional

import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


def _normalize_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            try:
                current = torch.cuda.current_device()
            except RuntimeError:
                current = None
            if current is not None:
                return torch.device("cuda", current)
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, int):
        return torch.device("cuda", device)
    raise TypeError(f"Unsupported device specifier: {device!r}")


def _normalize_stream(stream: Optional[torch.cuda.Stream], device: torch.device) -> Optional[torch.cuda.Stream]:
    if device.type != "cuda":
        return None
    if stream is not None:
        return stream
    return torch.cuda.current_stream(device)


def _stream_key(device: torch.device, stream: Optional[torch.cuda.Stream]) -> Tuple[str, Optional[int], int]:
    if device.type != "cuda":
        return (device.type, None, 0)
    stream_obj = _normalize_stream(stream, device)
    stream_id = stream_obj.cuda_stream if stream_obj is not None else 0
    return (device.type, device.index, stream_id)


class _ContextStore:
    def __init__(self) -> None:
        self._store: Dict[Tuple[str, Optional[int], int], Context] = {}
        self._lock = Lock()

    def set(self, context: Context, *, device: torch.device, stream: Optional[torch.cuda.Stream]) -> Context:
        key = _stream_key(device, stream)
        with self._lock:
            self._store[key] = context
        return context

    def get(self, *, device: torch.device, stream: Optional[torch.cuda.Stream]) -> Context:
        key = _stream_key(device, stream)
        with self._lock:
            ctx = self._store.get(key)
            if ctx is None:
                ctx = Context()
                self._store[key] = ctx
        return ctx

    def reset(self, *, device: Optional[torch.device], stream: Optional[torch.cuda.Stream]) -> None:
        if device is None and stream is None:
            with self._lock:
                self._store.clear()
            return

        dev = _normalize_device(device)
        key = _stream_key(dev, stream)
        with self._lock:
            self._store.pop(key, None)


_STORE = _ContextStore()


def get_context(*, device: torch.device | str | int | None = None, stream: Optional[torch.cuda.Stream] = None) -> Context:
    dev = _normalize_device(device)
    return _STORE.get(device=dev, stream=stream)


def set_context(
    is_prefill: bool | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    *,
    context: Context | None = None,
    device: torch.device | str | int | None = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> Context:
    dev = _normalize_device(device)
    if context is None:
        if is_prefill is None:
            raise ValueError("is_prefill must be provided when context is None")
        ctx = Context(
            is_prefill=is_prefill,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
    else:
        ctx = context
    return _STORE.set(ctx, device=dev, stream=stream)


def reset_context(*, device: torch.device | str | int | None = None, stream: Optional[torch.cuda.Stream] = None) -> None:
    _STORE.reset(device=device, stream=stream)
