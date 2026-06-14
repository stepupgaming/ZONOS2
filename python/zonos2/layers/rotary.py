from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP

# Try to import flashinfer; fall back to PyTorch on Windows.
try:
    from flashinfer import apply_rope_with_cos_sin_cache_inplace as _apply_rope_fn
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _apply_rope_fn = None
    _FLASHINFER_AVAILABLE = False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (non-interleaved / Neox format)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (interleaved format)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids, is_neox=True):
    """Applies Rotary Position Embedding to the query and key tensors.

    Supports both 2D interleaved (flashinfer-style) and 3D/4D non-interleaved inputs.
    """
    # cos/sin are 2D: (max_seq_len, head_dim//2)
    cos = cos[position_ids].to(q.dtype)  # (bs, head_dim//2)
    sin = sin[position_ids].to(q.dtype)  # (bs, head_dim//2)

    # Expand cos/sin to full head_dim by interleaving each value twice
    cos = cos.repeat_interleave(2, dim=-1)  # (bs, head_dim)
    sin = sin.repeat_interleave(2, dim=-1)  # (bs, head_dim)

    if is_neox:
        # Non-interleaved: q/k may be 2D (bs, num_heads*head_dim) or 3D/4D
        # If 2D, reshape to 3D, apply RoPE, then flatten back
        q_was_2d = q.dim() == 2
        k_was_2d = k.dim() == 2

        if q_was_2d:
            head_dim = cos.shape[-1]
            num_heads_q = q.shape[-1] // head_dim
            q = q.view(-1, num_heads_q, head_dim)
        if k_was_2d:
            head_dim = cos.shape[-1]
            num_heads_k = k.shape[-1] // head_dim
            k = k.view(-1, num_heads_k, head_dim)

        while cos.dim() < q.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)

        if q_was_2d:
            q_embed = q_embed.view(-1, q_embed.shape[1] * q_embed.shape[2])
        if k_was_2d:
            k_embed = k_embed.view(-1, k_embed.shape[1] * k_embed.shape[2])
    else:
        # Interleaved: q/k are 2D (bs, num_heads * head_dim)
        # Repeat cos/sin for each head
        num_heads_q = q.shape[-1] // cos.shape[-1]
        if num_heads_q > 1:
            cos_q = cos.repeat(1, num_heads_q)
            sin_q = sin.repeat(1, num_heads_q)
        else:
            cos_q = cos
            sin_q = sin

        num_heads_k = k.shape[-1] // cos.shape[-1]
        if num_heads_k > 1:
            cos_k = cos.repeat(1, num_heads_k)
            sin_k = sin.repeat(1, num_heads_k)
        else:
            cos_k = cos
            sin_k = sin

        q_embed = (q * cos_q) + (_rotate_half_interleaved(q) * sin_q)
        k_embed = (k * cos_k) + (_rotate_half_interleaved(k) * sin_k)

    return q_embed, k_embed


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        self._apply_rope_fn = _apply_rope_fn

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        is_neox: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._apply_rope_fn is not None:
            self._apply_rope_fn(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self._cos_sin_cache,
                is_neox=is_neox,
            )
            return query, key
        # PyTorch fallback
        cos_sin = self._cos_sin_cache.to(query.device)
        cos = cos_sin[:, : cos_sin.shape[-1] // 2]
        sin = cos_sin[:, cos_sin.shape[-1] // 2 :]
        return _apply_rotary_pos_emb(query, key, cos, sin, positions, is_neox=is_neox)


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    raise ValueError(f"Unsupported rotary scaling in TTS release: {rope_scaling!r}")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@lru_cache()
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
