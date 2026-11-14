"""Unexecuted PyTorch 2.9.1 Flex list attention, exclusively compiled/no fallback.

Only block directories and O(T) ownership are constructed. Every admitted block
is partial so the exact lane predicate always runs, even at field/candidate
boundaries. There is no token-square mask and no repeated KV heads.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from list_layout import CandidateLayout, MODEL_CONTEXT_LIMIT


BLOCK_SIZE = 128


@dataclass(frozen=True)
class BlockMetadataReport:
    mode: str
    total_real_tokens: int
    fields: int
    candidates: int
    query_blocks: int
    kv_blocks: int
    directory_width: int
    partial_blocks: int
    full_blocks: int
    token_square_materialized: bool = False
    repeated_kv_heads: bool = False


def _partial_block_rows(layout: CandidateLayout, mode: str) -> list[list[int]]:
    """Union of permitted *interval* tiles, followed by exact mask_mod checks.

    A query tile can span several segments or fields. Its directory includes
    the union, not their intersection; forbidden lanes are removed by mask_mod.
    Directory generation does not enumerate token pairs.
    """
    total = layout.total_tokens
    blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    segment_ends = tuple(segment.end for segment in layout.segments)
    rows = []
    for tile in range(blocks):
        qa = tile * BLOCK_SIZE
        qb = min(total, qa + BLOCK_SIZE)
        index = bisect.bisect_right(segment_ends, qa)
        keys: set[int] = set()

        def add_interval(start: int, end: int) -> None:
            if end > start:
                keys.update(range(start // BLOCK_SIZE, (end + BLOCK_SIZE - 1) // BLOCK_SIZE))

        while index < len(layout.segments):
            segment = layout.segments[index]
            if segment.start >= qb:
                break
            field = segment.field_index
            field_start = layout.field_token_offsets[field]
            if mode == "interaction":
                add_interval(field_start, min(qb, layout.field_token_offsets[field + 1]))
            elif segment.kind == "prefix":
                add_interval(field_start, min(qb, segment.end))
            else:
                prefix_start, prefix_end = layout.prefix_offsets[field]
                add_interval(prefix_start, prefix_end)
                add_interval(segment.start, min(qb, segment.end))
            index += 1
        rows.append(sorted(keys))
    return rows


def build_list_block_mask(
    layout: CandidateLayout, mode: str
) -> tuple[BlockMask, BlockMetadataReport]:
    """Bind an immutable real-token layout; no eager dense-mask alternative."""
    if mode not in ("isolation", "interaction"):
        raise ValueError("mode must be isolation (J0/J2) or interaction (J1/J3)")
    total = layout.total_tokens
    if total < 1 or layout.batch.layout != "packed" or layout.batch.mask is not None:
        raise ValueError("Only nonempty, unpadded real-token layouts are admitted")
    if max(layout.batch.lengths) > MODEL_CONTEXT_LIMIT:
        raise ValueError(
            "A complete field exceeds common physical context 40960; retain it "
            "in static coverage records, do not truncate or silently delete candidates"
        )
    device = layout.batch.ids.device
    if device.type != "cuda":
        raise ValueError("Production Flex metadata is admitted only on CUDA")
    if any(
        tensor.shape != (total,) or tensor.device != device
        for tensor in (layout.token_owner, layout.token_candidate, layout.token_segment)
    ):
        raise ValueError("Ownership metadata must match exactly T real device tokens")
    owners = layout.token_owner
    candidates = layout.token_candidate
    if mode == "interaction":
        def mask_mod(b, h, q_idx, kv_idx):
            q_safe = q_idx.clamp(0, total - 1)
            k_safe = kv_idx.clamp(0, total - 1)
            valid = (q_idx >= 0) & (kv_idx >= 0) & (q_idx < total) & (kv_idx < total)
            return valid & (owners[q_safe] == owners[k_safe]) & (q_idx >= kv_idx)
    else:
        def mask_mod(b, h, q_idx, kv_idx):
            q_safe = q_idx.clamp(0, total - 1)
            k_safe = kv_idx.clamp(0, total - 1)
            valid = (q_idx >= 0) & (kv_idx >= 0) & (q_idx < total) & (kv_idx < total)
            visible = (candidates[k_safe] == -1) | (candidates[q_safe] == candidates[k_safe])
            return valid & (owners[q_safe] == owners[k_safe]) & (q_idx >= kv_idx) & visible
    rows = _partial_block_rows(layout, mode)
    blocks = len(rows)
    counts = torch.tensor(
        [len(row) for row in rows], dtype=torch.int32, device=device
    )[None, None, :]
    # 2.9.1's generated KV->Q transpose uses the LAST axis as the total KV
    # block universe. max(active_count) is NOT a safe capacity here.
    indices = torch.tensor(
        [row + [0] * (blocks - len(row)) for row in rows],
        dtype=torch.int32,
        device=device,
    )[None, None, :, :]
    block_mask = BlockMask.from_kv_blocks(
        counts,
        indices,
        None,
        None,
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(total, total),
    )
    report = BlockMetadataReport(
        mode=mode,
        total_real_tokens=total,
        fields=len(layout.field_keys),
        candidates=layout.candidate_count,
        query_blocks=blocks,
        kv_blocks=blocks,
        directory_width=blocks,
        partial_blocks=sum(len(row) for row in rows),
        full_blocks=0,
    )
    return block_mask, report


class BoundCandidateAttention:
    """One batch binding reused by every layer, never re-bound inside a layer.

    Tensor *identity* guards the cu contract without per-layer torch.equal or
    device synchronization. A caller that copies cu must bind/use the original
    object instead. Do not mutate the layout or its tensors after binding.
    """

    def __init__(
        self,
        backend: CandidateFlexBackend,
        layout: CandidateLayout,
        block_mask: BlockMask,
        metadata: BlockMetadataReport,
    ):
        self.backend = backend
        self.layout = layout
        self.block_mask = block_mask
        self.metadata = metadata
        self.total_tokens = layout.total_tokens
        self.max_field_length = max(layout.batch.lengths)
        self.forward_submissions = 0
        self.output_gradient_callbacks = 0

    def _validate_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        total = self.total_tokens
        if q.shape != (total, 16, 128) or k.shape != (total, 8, 128) or v.shape != k.shape:
            raise ValueError("Only exact real-token Q16/KV8/D128 self-attention is admitted")
        if any(t.device != self.layout.batch.ids.device for t in (q, k, v)):
            raise ValueError("Q/K/V must use this bound layout's CUDA device")
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            raise ValueError("Only matched BF16 Q/K/V are admitted in this component")
        if not torch.are_deterministic_algorithms_enabled():
            raise ValueError("The frozen deterministic-algorithms contract is required")
        if torch.is_deterministic_algorithms_warn_only_enabled():
            raise ValueError("Deterministic warn_only=True is not an admitted substitute")

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        self._validate_qkv(q, k, v)
        if softmax_scale is not None and (
            not isinstance(softmax_scale, (int, float))
            or isinstance(softmax_scale, bool)
            or not math.isfinite(softmax_scale)
            or softmax_scale <= 0
        ):
            raise ValueError("softmax_scale must be a finite positive Python number or None")
        # Singleton batch and real T. No unsqueeze into 5D, no repeat_interleave
        # of KV heads, no hardware-alignment hidden/QKV rows.
        out = self.backend.compiled(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            block_mask=self.block_mask,
            scale=softmax_scale,
            enable_gqa=True,
            kernel_options={"PRESCALE_QK": False},
        )
        self.forward_submissions += 1
        self.backend.forward_submissions += 1
        result = out.squeeze(0).transpose(0, 1).contiguous()
        if result.requires_grad:
            def observe_gradient(grad):
                self.output_gradient_callbacks += 1
                self.backend.output_gradient_callbacks += 1
                return grad
            result.register_hook(observe_gradient)
        return result

    def flash_attn_varlen_func(
        self,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        softmax_scale=None,
        causal=False,
        deterministic=False,
    ):
        """Narrow adapter, not a general flash-attention implementation.

        causal=True means physical causality plus the selected list edges.
        Dropout, padding, arbitrary masks, cache/decode, cross-attention,
        alternate heads/dtypes and unadvertised FA options are unsupported.
        Unknown keyword arguments naturally raise TypeError; no fallback.
        """
        if causal is not True or deterministic is not True:
            raise ValueError("Exact causal=True/deterministic=True contract required")
        if cu_q is not self.layout.batch.cu or cu_k is not self.layout.batch.cu:
            raise ValueError("Use the very same field-cu tensor object bound once for this batch")
        expected_max = self.max_field_length
        if type(max_q) is not int or type(max_k) is not int:
            raise TypeError("max_q/max_k must be host integers, not device scalars")
        if max_q != expected_max or max_k != expected_max:
            raise ValueError("max_q/max_k must match complete physical field lengths")
        return self.attention(q, k, v, softmax_scale=softmax_scale)


class CandidateFlexBackend:
    """Fixed visibility mode, one compiled entry, current-layout-only cache.

    Creating this object asks torch.compile for the production callable; the
    first forward is the actual lazy compile attempt. No warmup, fallback,
    graph-count assertion or performance measurement is hidden here.
    """

    def __init__(self, mode: str):
        if mode not in ("isolation", "interaction"):
            raise ValueError("Expected isolation or interaction")
        if torch.__version__.split("+", 1)[0] != "2.9.1" or torch.version.cuda != "12.8":
            raise RuntimeError("Only the specified PyTorch 2.9.1 + CUDA 12.8 stack is admitted")
        self._mode = mode
        self.compiled = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        self.current: BoundCandidateAttention | None = None
        self.batch_bindings = 0
        self.forward_submissions = 0
        self.output_gradient_callbacks = 0

    @property
    def mode(self) -> str:
        return self._mode

    def bind_batch(self, layout: CandidateLayout) -> BoundCandidateAttention:
        if self.current is not None and self.current.layout is layout:
            return self.current
        mask, report = build_list_block_mask(layout, self.mode)
        bound = BoundCandidateAttention(self, layout, mask, report)
        # Replacing current does not mutate old masks referenced by autograd
        # or an explicitly retained bound handle. There is no growing dict/LRU.
        self.current = bound
        self.batch_bindings += 1
        return bound

    def clear_current(self) -> None:
        self.current = None
