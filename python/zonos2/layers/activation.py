from __future__ import annotations

import torch

# Try to import flashinfer; fall back to PyTorch on Windows.
try:
    from flashinfer import silu_and_mul as _silu_and_mul_fn
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _silu_and_mul_fn = None
    _FLASHINFER_AVAILABLE = False


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    if _silu_and_mul_fn is not None:
        return _silu_and_mul_fn(x)
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


__all__ = ["silu_and_mul"]
