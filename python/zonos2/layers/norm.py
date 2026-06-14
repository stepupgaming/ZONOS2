from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BaseOP

# Try to import flashinfer; fall back to PyTorch on Windows.
try:
    from flashinfer import rmsnorm as _rmsnorm_fn
    from flashinfer import fused_add_rmsnorm as _fused_add_rmsnorm_fn
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _rmsnorm_fn = None
    _fused_add_rmsnorm_fn = None
    _FLASHINFER_AVAILABLE = False


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self._rmsnorm_fn = _rmsnorm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._rmsnorm_fn is not None:
            return self._rmsnorm_fn(x, self.weight, self.eps)
        return F.rms_norm(x, (self.weight.numel(),), self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if self._rmsnorm_fn is not None:
            self._rmsnorm_fn(x, self.weight, self.eps, out=x)
        else:
            x.copy_(F.rms_norm(x, (self.weight.numel(),), self.weight, self.eps))


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float, elementwise_affine: bool = True) -> None:
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self._size = size

        if elementwise_affine:
            self.weight = torch.empty(size)
        # When elementwise_affine=False, we use a ones buffer created lazily
        # to ensure correct device/dtype

        self._rmsnorm_fn = _rmsnorm_fn
        self._fused_add_rmsnorm_fn = _fused_add_rmsnorm_fn
        self._ones_buffer: torch.Tensor | None = None

    def _get_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self.elementwise_affine:
            return self.weight
        # Use cached ones buffer, recreate if needed for device/dtype match
        if self._ones_buffer is None or self._ones_buffer.device != x.device:
            self._ones_buffer = torch.ones(self._size, device=x.device, dtype=x.dtype)
        return self._ones_buffer

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self._get_weight(x)
        if residual is None:
            if self._rmsnorm_fn is not None:
                return self._rmsnorm_fn(x, weight, self.eps), x
            return F.rms_norm(x, (weight.numel(),), weight, self.eps), x
        if self._fused_add_rmsnorm_fn is not None:
            self._fused_add_rmsnorm_fn(x, residual, weight, self.eps)
            return x, residual
        residual.add_(x)
        return F.rms_norm(residual, (weight.numel(),), weight, self.eps), residual
