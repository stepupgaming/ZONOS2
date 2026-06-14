from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData, make_positions

if TYPE_CHECKING:
    from zonos2.core import TTSBatch
    from zonos2.kvcache import BaseKVCache
    from zonos2.models import ModelConfig


@dataclass
class TorchCaptureData(BaseCaptureData):
    pass


@dataclass
class TorchMetadata(BaseAttnMetadata):
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TorchAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
        self.config = config
        self.kvcache = kvcache
        self.page_table = page_table
        self.capture: TorchCaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim ** -0.5

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: TTSBatch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchMetadata)

        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        k_cache_flat = k_cache.view(k_cache.shape[0], -1)
        v_cache_flat = v_cache.view(v_cache.shape[0], -1)

        n_rep = self.config.num_qo_heads // self.config.num_kv_heads

        if batch.is_decode and torch.cuda.is_current_stream_capturing():
            outputs = []
            max_seq_len = metadata.page_table.shape[1]
            key_mask = torch.arange(max_seq_len, device=q.device).view(1, 1, 1, max_seq_len)

            for i in range(batch.size):
                store_pages = batch.out_loc[i : i + 1]
                k_cache_flat[store_pages] = k[i : i + 1].view(1, -1).to(k_cache_flat.dtype)
                v_cache_flat[store_pages] = v[i : i + 1].view(1, -1).to(v_cache_flat.dtype)

                pages = metadata.page_table[i]
                k_i = k_cache[pages].squeeze(1)
                v_i = v_cache[pages].squeeze(1)

                if n_rep > 1:
                    k_i = k_i.unsqueeze(2).repeat(1, 1, n_rep, 1).flatten(1, 2)
                    v_i = v_i.unsqueeze(2).repeat(1, 1, n_rep, 1).flatten(1, 2)

                q_i = q[i : i + 1].unsqueeze(0).transpose(1, 2)
                k_i = k_i.unsqueeze(0).transpose(1, 2)
                v_i = v_i.unsqueeze(0).transpose(1, 2)
                attn_mask = key_mask < metadata.seq_lens[i].view(1, 1, 1, 1)

                out_i = F.scaled_dot_product_attention(
                    q_i.to(k_i.dtype),
                    k_i,
                    v_i,
                    attn_mask=attn_mask,
                    is_causal=False,
                    scale=self.scale,
                )
                outputs.append(out_i.transpose(1, 2).squeeze(0))

            return torch.cat(outputs, dim=0)

        outputs = []
        q_offset = 0
        kv_offset = 0

        for i, req in enumerate(batch.reqs):
            seq_len = req.device_len
            extend_len = req.extend_len
            store_pages = metadata.page_table[i, req.cached_len : req.device_len]
            k_cache_flat[store_pages] = k[kv_offset : kv_offset + extend_len].view(
                extend_len, -1
            ).to(k_cache_flat.dtype)
            v_cache_flat[store_pages] = v[kv_offset : kv_offset + extend_len].view(
                extend_len, -1
            ).to(v_cache_flat.dtype)
            kv_offset += extend_len

            pages = metadata.page_table[i, :seq_len]

            k_i = k_cache[pages].squeeze(1)
            v_i = v_cache[pages].squeeze(1)

            if n_rep > 1:
                k_i = k_i.unsqueeze(2).repeat(1, 1, n_rep, 1).flatten(1, 2)
                v_i = v_i.unsqueeze(2).repeat(1, 1, n_rep, 1).flatten(1, 2)

            if batch.is_decode:
                q_i = q[i : i + 1].unsqueeze(0).transpose(1, 2)
            else:
                q_i = q[q_offset : q_offset + extend_len].unsqueeze(0).transpose(1, 2)
                q_offset += extend_len

            k_i = k_i.unsqueeze(0).transpose(1, 2)
            v_i = v_i.unsqueeze(0).transpose(1, 2)

            # During decode the KV cache only contains the visible prefix for this
            # request, so an extra causal mask would hide valid prefix tokens.
            out_i = F.scaled_dot_product_attention(
                q_i.to(k_i.dtype),
                k_i,
                v_i,
                attn_mask=None,
                is_causal=not batch.is_decode,
                scale=self.scale,
            )

            outputs.append(out_i.transpose(1, 2).squeeze(0))

        return torch.cat(outputs, dim=0)

    def prepare_metadata(self, batch: TTSBatch) -> None:
        reqs = batch.padded_reqs
        padded_size = len(reqs)
        max_seqlen_k = max(req.device_len for req in reqs)
        device = self.kvcache.device

        page_table = torch.stack([
            self.page_table[req.table_idx, :max_seqlen_k] for req in reqs
        ])

        positions = make_positions(device, reqs)
        seq_lens = torch.tensor(
            [req.device_len for req in reqs], device=device, dtype=torch.int32
        )

        seqlens_q = [req.extend_len for req in reqs]
        cu_seqlens_q = torch.tensor(
            [0] + seqlens_q, device=device, dtype=torch.int32
        ).cumsum_(dim=0)

        batch.attn_metadata = TorchMetadata(
            positions=positions,
            page_table=page_table,
            seq_lens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
        )

    def init_capture_graph(
        self, max_seq_len: int, bs_list: List[int], frame_width: int = 1
    ) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = TorchCaptureData.create(max_bs, max_seq_len, self.kvcache.device, frame_width)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: TTSBatch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = TorchMetadata(
            positions=capture.positions[:bs],
            page_table=capture.page_table[:bs, :],
            seq_lens=capture.seq_lens[:bs],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
        )
        batch.attn_metadata = metadata
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]

    def prepare_for_replay(self, batch: TTSBatch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, TorchMetadata)
        assert self.capture is not None and bs in self.capture_bs
        capture = self.capture

        capture.input_ids[:bs].copy_(batch.input_ids)
        capture.out_loc[:bs].copy_(batch.out_loc)
        capture.positions[:bs].copy_(metadata.positions)
        capture.seq_lens[:bs].copy_(metadata.seq_lens)
        page_table_width = metadata.page_table.size(1)
        capture.page_table[:bs, :page_table_width].copy_(metadata.page_table)
        capture.cu_seqlens_q[: bs + 1].copy_(metadata.cu_seqlens_q)

        metadata.positions = capture.positions[:bs]
        metadata.seq_lens = capture.seq_lens[:bs]
        metadata.page_table = capture.page_table[:bs]
        metadata.cu_seqlens_q = capture.cu_seqlens_q[: bs + 1]
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]
