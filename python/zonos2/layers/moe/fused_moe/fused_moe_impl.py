import functools
from typing import Dict, Optional, Tuple

import torch

# Try to import triton and sgl_kernel; fall back to CPU-only path if unavailable (e.g., Windows).
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

try:
    from sgl_kernel import gelu_and_mul, silu_and_mul
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
    _SGL_KERNEL_AVAILABLE = True
except ImportError:
    gelu_and_mul = None
    silu_and_mul = None
    sgl_moe_align_block_size = None
    _SGL_KERNEL_AVAILABLE = False

try:
    from zonos2.kernel.moe_impl import fused_moe_kernel_triton
    from zonos2.kernel.triton.fused_moe import moe_sum_reduce_triton
    _TRITON_KERNEL_AVAILABLE = True
except ImportError:
    fused_moe_kernel_triton = None
    moe_sum_reduce_triton = None
    _TRITON_KERNEL_AVAILABLE = False

from zonos2.layers.moe.fused_moe.topk import select_experts


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda


def _ensure_cuda_libs():
    if not _TRITON_AVAILABLE or not _TRITON_KERNEL_AVAILABLE:
        raise RuntimeError(
            "Triton is required for the fused MoE expert path on this platform."
        )


def _fused_experts_impl_torch(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):
    """Pure PyTorch fallback for fused_experts_impl (no Triton/sgl_kernel)."""
    import torch.nn.functional as F

    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    E, N, _ = w1.shape
    intermediate_size = N // 2

    out_dtype = hidden_states.dtype
    compute_dtype = torch.float32

    if no_combine:
        out = torch.zeros(
            (num_tokens, top_k, w2.shape[1]),
            device=hidden_states.device,
            dtype=compute_dtype,
        )
    else:
        out = torch.zeros_like(hidden_states, dtype=compute_dtype)

    # Flatten to (num_tokens * top_k, hidden_size)
    hidden_states_expanded = hidden_states.to(compute_dtype).unsqueeze(1).expand(-1, top_k, -1)
    hidden_states_flat = hidden_states_expanded.reshape(-1, hidden_size)
    topk_ids_flat = topk_ids.reshape(-1)
    topk_weights_flat = topk_weights.to(compute_dtype).reshape(-1)

    if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
        out_flat = torch.zeros(
            (num_tokens * top_k, w2.shape[1]),
            device=hidden_states.device,
            dtype=compute_dtype,
        )
        for expert_id in range(E):
            gate_up = torch.matmul(hidden_states_flat, w1[expert_id].to(compute_dtype).t())
            gate = gate_up[:, :intermediate_size]
            up = gate_up[:, intermediate_size:]

            if activation == "silu":
                activated = up * F.silu(gate)
            elif activation == "gelu":
                activated = up * F.gelu(gate)
            else:
                raise ValueError(f"Unsupported activation: {activation=}")

            expert_out = torch.matmul(activated, w2[expert_id].to(compute_dtype).t())
            mask = (topk_ids_flat == expert_id).to(compute_dtype).unsqueeze(-1)
            out_flat = out_flat + expert_out * mask

        weights = topk_weights_flat.unsqueeze(-1)
        if apply_router_weight_on_input:
            out_flat = out_flat * weights

        if no_combine:
            return out_flat.view(num_tokens, top_k, w2.shape[1]).to(out_dtype)

        out = (out_flat.view(num_tokens, top_k, w2.shape[1]) * weights.view(num_tokens, top_k, 1)).sum(dim=1)
        if routed_scaling_factor is not None:
            out.mul_(routed_scaling_factor)
        return out.to(out_dtype)

    # For each expert, gather tokens and batch-process
    for expert_id in range(E):
        mask = topk_ids_flat == expert_id
        if not mask.any():
            continue
        expert_hidden = hidden_states_flat[mask]
        expert_weights = topk_weights_flat[mask]

        # gate_up_proj: (2*intermediate, H)
        gate_up = torch.matmul(expert_hidden, w1[expert_id].to(compute_dtype).t())
        gate = gate_up[:, :intermediate_size]
        up = gate_up[:, intermediate_size:]

        if activation == "silu":
            activated = up * F.silu(gate)
        elif activation == "gelu":
            activated = up * F.gelu(gate)
        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        expert_out = torch.matmul(activated, w2[expert_id].to(compute_dtype).t())

        if apply_router_weight_on_input:
            expert_out = expert_out * expert_weights.unsqueeze(-1)

        # Scatter back
        token_indices = torch.arange(num_tokens * top_k, device=hidden_states.device)[mask]
        token_idx = token_indices // top_k
        k_idx = token_indices % top_k

        if no_combine:
            out[token_idx, k_idx] = expert_out
        else:
            out[token_idx] += expert_out * expert_weights.unsqueeze(-1)

    if not no_combine and routed_scaling_factor is not None:
        out.mul_(routed_scaling_factor)

    return out.to(out_dtype)


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.
    """
    _ensure_cuda_libs()
    if not _SGL_KERNEL_AVAILABLE:
        flat = topk_ids.reshape(-1).to(torch.int32)
        total = flat.numel()
        per_expert = []
        for expert_id in range(num_experts):
            ids = torch.nonzero(flat == expert_id, as_tuple=False).flatten().to(torch.int32)
            pad = (-ids.numel()) % block_size
            if pad:
                ids = torch.cat(
                    [
                        ids,
                        torch.full((pad,), total, dtype=torch.int32, device=topk_ids.device),
                    ]
                )
            per_expert.append(ids)

        sorted_ids = torch.cat(per_expert) if per_expert else torch.empty(0, dtype=torch.int32, device=topk_ids.device)
        if sorted_ids.numel() == 0:
            sorted_ids = torch.full((block_size,), total, dtype=torch.int32, device=topk_ids.device)

        block_expert_ids = []
        for expert_id, ids in enumerate(per_expert):
            blocks = ids.numel() // block_size
            if blocks:
                block_expert_ids.append(
                    torch.full((blocks,), expert_id, dtype=torch.int32, device=topk_ids.device)
                )
        expert_ids = (
            torch.cat(block_expert_ids)
            if block_expert_ids
            else torch.empty(0, dtype=torch.int32, device=topk_ids.device)
        )
        num_tokens_post_padded = torch.tensor(
            [sorted_ids.numel()], dtype=torch.int32, device=topk_ids.device
        )
        return sorted_ids, expert_ids, num_tokens_post_padded

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)

    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    is_marlin: bool,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    # A heuristic: fused marlin works faster with this config for small M
    if M <= E or (is_marlin and M <= 32):
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
    is_marlin: bool = False,
):
    E, _, N = w2_shape

    config = get_default_config(M, E, N, w1_shape[2], top_k, is_marlin)
    return config


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):
    if (
        not _TRITON_AVAILABLE
        or not _TRITON_KERNEL_AVAILABLE
        or not _SGL_KERNEL_AVAILABLE
        or (not _SGL_KERNEL_AVAILABLE and topk_ids.shape[1] != 1)
    ):
        return _fused_experts_impl_torch(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace,
            activation,
            apply_router_weight_on_input,
            no_combine,
            routed_scaling_factor,
        )

    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape

    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk_ids.shape[1], w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[: tokens_in_chunk * topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )

        fused_moe_kernel_triton(
            curr_hidden_states,
            w1,
            intermediate_cache1,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
        )

        if activation == "silu":
            if silu_and_mul is not None:
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                gate = intermediate_cache1.view(-1, N)[:, : N // 2]
                up = intermediate_cache1.view(-1, N)[:, N // 2 :]
                torch.mul(torch.nn.functional.silu(gate), up, out=intermediate_cache2)
        elif activation == "gelu":
            if gelu_and_mul is not None:
                gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                gate = intermediate_cache1.view(-1, N)[:, : N // 2]
                up = intermediate_cache1.view(-1, N)[:, N // 2 :]
                torch.mul(torch.nn.functional.gelu(gate), up, out=intermediate_cache2)
        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        fused_moe_kernel_triton(
            intermediate_cache2,
            w2,
            (
                intermediate_cache3
                if not no_combine and topk_ids.shape[1] != 1
                else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
            ),
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass

        if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
            pass  # we write directly into out_hidden_states
        elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
            torch.add(
                intermediate_cache3[:, 0],
                intermediate_cache3[:, 1],
                out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
            ).squeeze(dim=1)
        else:
            if tokens_in_chunk <= 32:
                moe_sum_reduce_torch_compile(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    routed_scaling_factor,
                )
            else:
                if moe_sum_reduce_triton is not None:
                    moe_sum_reduce_triton(
                        intermediate_cache3,
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    torch.sum(
                        intermediate_cache3,
                        dim=1,
                        out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    )
                    out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor)
    return out_hidden_states


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> None:

    result = fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        True,
        activation,
        apply_router_weight_on_input,
        False,
        routed_scaling_factor,
    )
    if result is not hidden_states:
        hidden_states.copy_(result)


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,
        activation,
        apply_router_weight_on_input,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
    )


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):

    if inplace:
        assert not no_combine, "no combine + inplace makes no sense"
        inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            routed_scaling_factor,
        )
        return hidden_states
    else:
        return outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    no_combine: bool = False,
) -> torch.Tensor:

    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=gating_output,
        top_k=topk,
        renormalize=renormalize,
    )
    return fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        no_combine=no_combine,
    )
