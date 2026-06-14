from __future__ import annotations

from typing import Tuple


def indexing(
    weights,
    indices,
    *,
    output=None,
    vocab_range: Tuple[int, int] | None = None,
):
    """PyTorch fallback for the CUDA indexing kernel.

    Equivalent to ``weights[indices]`` with optional output buffer.
    """
    if vocab_range is not None:
        start, length = vocab_range
        weights = weights[start : start + length]
    gathered = weights[indices]
    if output is not None:
        output.copy_(gathered)
        return output
    return gathered


def store_cache(k_cache, v_cache, indices, k, v):
    """PyTorch fallback for the CUDA store_cache kernel.

    Writes ``k`` and ``v`` into the paged KV cache at ``indices``.

    k_cache has shape (num_pages, local_kv_heads, head_dim)
    indices are page indices of shape (num_tokens,)
    k has shape (num_tokens, num_kv_heads * head_dim) or (num_tokens, num_kv_heads, head_dim)
    """
    # Flatten the last two dimensions: (num_pages, local_kv_heads * head_dim)
    k_cache_flat = k_cache.view(k_cache.shape[0], -1)
    v_cache_flat = v_cache.view(v_cache.shape[0], -1)
    k_flat = k.view(k.shape[0], -1).to(k_cache_flat.dtype)
    v_flat = v.view(v.shape[0], -1).to(v_cache_flat.dtype)
    k_cache_flat[indices] = k_flat
    v_cache_flat[indices] = v_flat


def fast_compare_key(x, y) -> int:
    """CPU fallback for the C++ fast_compare_key kernel.

    Returns the number of matching elements at the start of two 1-D int CPU tensors.
    """
    common_len = min(x.numel(), y.numel())
    if common_len == 0:
        return 0
    diff = (x[:common_len] != y[:common_len]).nonzero(as_tuple=True)[0]
    if diff.numel() == 0:
        return common_len
    return int(diff[0].item())


def test_tensor(x, y) -> int:
    """Fallback for the C++ test_tensor kernel.

    This is a no-op for testing; it always returns 0.
    """
    return 0
