"""Device selection wrapper.

Tries torch_directml (AMD/Intel GPU on Windows) first, falls back to CUDA,
then CPU. Provides a no-op `sync()` shim so callers can write portable
timing code that works regardless of backend.
"""
from __future__ import annotations

import torch


_DEVICE: torch.device | None = None


def get_device(force_cpu: bool = False) -> torch.device:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if force_cpu:
        _DEVICE = torch.device("cpu")
        return _DEVICE
    # Try DirectML first (AMD/Intel/NVIDIA on Windows)
    try:
        import torch_directml  # type: ignore
        _DEVICE = torch_directml.device()
        return _DEVICE
    except ImportError:
        pass
    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda:0")
        return _DEVICE
    _DEVICE = torch.device("cpu")
    return _DEVICE


def sync() -> None:
    """Block until queued device ops finish.

    DirectML has no public sync API; force a sync via a tiny CPU
    materialization. CUDA uses cuda.synchronize. CPU is no-op.
    """
    d = get_device()
    if d.type == "cuda":
        torch.cuda.synchronize()
    elif d.type == "privateuseone":
        # DirectML: round-trip a 1-element tensor through CPU to flush queue.
        torch.zeros(1, device=d).cpu()
