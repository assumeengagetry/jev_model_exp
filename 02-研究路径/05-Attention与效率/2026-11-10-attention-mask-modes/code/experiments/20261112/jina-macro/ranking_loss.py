"""Unexecuted EOS-margin objectives; supervision is separate from model input.

No candidate is selected from predictions. Hard positives/negatives are known
q==1 / q==0 only. Soft targets remain in BCE, not in ranking's denominator.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real

import torch
import torch.nn.functional as F

from list_layout import CandidateLayout, FieldKey


@dataclass(frozen=True)
class FieldSupervision:
    key: FieldKey
    case_index: int
    start: int
    end: int
    known_rows: tuple[int, ...]
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    hard_rows: tuple[int, ...]
    rank_active: bool


@dataclass(frozen=True)
class CandidateSupervision:
    """Prepared once outside attention; field/case addresses mirror the layout."""

    candidate_count: int
    case_ids: tuple
    candidate_keys: tuple
    fields: tuple[FieldSupervision, ...]
    case_field_rows: tuple[tuple[int, ...], ...]
    case_known_pairs: tuple[int, ...]
    case_known_fields: tuple[int, ...]
    # Required explicit choice; do not guess which historical denominator the
    # integrating trainer has frozen (see README's ambiguity ledger).
    bce_case_reduction: str
    targets: torch.Tensor
    known_mask: torch.Tensor
    known_indices: tuple[torch.Tensor, ...]
    positive_indices: tuple[torch.Tensor, ...]
    hard_indices: tuple[torch.Tensor, ...]
    statistics: Mapping[str, int]


@dataclass(frozen=True)
class LossResult:
    loss: torch.Tensor
    bce: torch.Tensor
    rank: torch.Tensor
    # Graph-connected values, so callers decide what to detach/log. Counts are
    # explicit; division by clamped/eligible-only rank counts is never hidden.
    raw: Mapping[str, torch.Tensor]
    statistics: Mapping[str, int]
    bce_case_reduction: str
    rank_case_reduction: str


@dataclass(frozen=True)
class MarginGradientCoefficients:
    coefficients: torch.Tensor
    loss: torch.Tensor
    bce: torch.Tensor
    rank: torch.Tensor
    raw: Mapping[str, torch.Tensor]
    statistics: Mapping[str, int]


def prepare_supervision(
    layout: CandidateLayout,
    targets: Sequence[float | None],
    known_mask: Sequence[bool],
    *,
    bce_case_reduction: str,
) -> CandidateSupervision:
    """Validate host supervision once, never sort or change the input stream.

    Supply entries in layout.candidate_keys order. Masked targets may be None
    or nonfinite; they are not read and are stored as harmless zero placeholders
    behind a false mask. Known targets must be finite probabilities in [0,1].
    Every supplied training case must have at least one known candidate. Pure
    inference uses only list_layout and does not call this function.
    """
    if bce_case_reduction not in ("known_pairs", "known_fields"):
        raise ValueError("Explicitly freeze bce_case_reduction=known_pairs or known_fields")
    if isinstance(targets, (torch.Tensor, str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a host sequence; no implicit device synchronization")
    if isinstance(known_mask, (torch.Tensor, str, bytes)) or not isinstance(known_mask, Sequence):
        raise TypeError("known_mask must be a host bool sequence")
    if len(targets) != layout.candidate_count or len(known_mask) != layout.candidate_count:
        raise ValueError("Supervision must contain exactly one entry per stable candidate key")
    q = []
    known = []
    for target, is_known in zip(targets, known_mask):
        if type(is_known) is not bool:
            raise TypeError("known_mask entries must be actual bools")
        known.append(is_known)
        if not is_known:
            q.append(0.0)
            continue
        if (
            not isinstance(target, Real)
            or isinstance(target, bool)
            or not math.isfinite(float(target))
            or not 0 <= float(target) <= 1
        ):
            raise ValueError("Every known target must be a finite probability in [0,1]")
        q.append(float(target))
    field_info = []
    case_fields: list[list[int]] = [[] for _ in layout.case_ids]
    case_known_pairs = [0] * len(layout.case_ids)
    case_known_fields = [0] * len(layout.case_ids)
    stats = {
        "cases": len(layout.case_ids),
        "fields": len(layout.field_keys),
        "candidates": layout.candidate_count,
        "known_candidates": sum(known),
        "unknown_candidates": len(known) - sum(known),
        "known_soft_candidates": 0,
        "known_positive_candidates": 0,
        "known_negative_candidates": 0,
        "fields_no_known": 0,
        "rank_fields_active": 0,
        "rank_fields_zero": 0,
        # Reasons may overlap; active+zero, unlike these reasons, partitions F.
        "rank_fields_k1": 0,
        "rank_fields_no_positive": 0,
        "rank_fields_no_negative": 0,
        "rank_fields_hard_k_lt_2": 0,
    }
    for f, key in enumerate(layout.field_keys):
        start, end = layout.field_candidate_offsets[f:f + 2]
        rows = tuple(i for i in range(start, end) if known[i])
        positives = tuple(i for i in rows if q[i] == 1.0)
        negatives = tuple(i for i in rows if q[i] == 0.0)
        hard = tuple(i for i in rows if q[i] == 1.0 or q[i] == 0.0)
        active = (end - start > 1) and bool(positives) and bool(negatives)
        case = layout.field_case_indices[f]
        case_fields[case].append(f)
        case_known_pairs[case] += len(rows)
        case_known_fields[case] += int(bool(rows))
        stats["known_soft_candidates"] += len(rows) - len(hard)
        stats["known_positive_candidates"] += len(positives)
        stats["known_negative_candidates"] += len(negatives)
        stats["fields_no_known"] += int(not rows)
        stats["rank_fields_active"] += int(active)
        stats["rank_fields_zero"] += int(not active)
        stats["rank_fields_k1"] += int(end - start == 1)
        stats["rank_fields_no_positive"] += int(not positives)
        stats["rank_fields_no_negative"] += int(not negatives)
        stats["rank_fields_hard_k_lt_2"] += int(len(hard) < 2)
        field_info.append(FieldSupervision(
            key, case, start, end, rows, positives, negatives, hard, active
        ))
    if any(count == 0 for count in case_known_pairs):
        raise ValueError(
            "Training loss rejects an entirely unknown case; keep inference "
            "layout separate rather than silently changing the case denominator"
        )
    stats["cases_without_rank_signal"] = sum(
        not any(field_info[f].rank_active for f in fields) for fields in case_fields
    )
    device = layout.batch.ids.device

    def indices(rows: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.long, device=device)

    return CandidateSupervision(
        candidate_count=layout.candidate_count,
        case_ids=layout.case_ids,
        candidate_keys=layout.candidate_keys,
        fields=tuple(field_info),
        case_field_rows=tuple(tuple(fields) for fields in case_fields),
        case_known_pairs=tuple(case_known_pairs),
        case_known_fields=tuple(case_known_fields),
        bce_case_reduction=bce_case_reduction,
        targets=torch.tensor(q, dtype=torch.float32, device=device),
        known_mask=torch.tensor(known, dtype=torch.bool, device=device),
        known_indices=tuple(indices(field.known_rows) for field in field_info),
        positive_indices=tuple(indices(field.positive_rows) for field in field_info),
        hard_indices=tuple(indices(field.hard_rows) for field in field_info),
        statistics=stats,
    )


def _validate_loss_options(lambda_rank: float, tau: float) -> None:
    if (
        not isinstance(lambda_rank, Real)
        or isinstance(lambda_rank, bool)
        or not math.isfinite(float(lambda_rank))
        or lambda_rank < 0
    ):
        raise ValueError("lambda_rank must be finite and nonnegative")
    if (
        not isinstance(tau, Real)
        or isinstance(tau, bool)
        or not math.isfinite(float(tau))
        or tau <= 0
    ):
        raise ValueError("tau must be finite and strictly positive")


def loss_from_margins(
    margins: torch.Tensor,
    supervision: CandidateSupervision,
    *,
    lambda_rank: float = 0.0,
    tau: float = 1.0,
) -> LossResult:
    """Case-balanced BCE + optional within-field multi-positive rank.

    margins[i] must be EOS logit_yes - logit_no, with the caller's frozen class
    order. The loss does not infer that order. No softmax across unrelated
    fields/cases and no positive/negative choice based on model scores.

    Rank denominator: mean over known fields within each case, then mean over
    ALL supplied cases, matching the newly frozen case-balanced field policy.
    Known zero-rank-signal fields remain in that denominator; unknown-only
    fields stay in raw records but supply no supervision denominator. Never
    re-normalize only active rank fields/cases.
    """
    _validate_loss_options(lambda_rank, tau)
    if margins.ndim != 1 or margins.shape[0] != supervision.candidate_count:
        raise ValueError("margins must be a flat vector in layout candidate-key order")
    if not margins.is_floating_point() or margins.device != supervision.targets.device:
        raise ValueError("Floating margins and prepared supervision must share a device")
    if margins.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Unsupported margin dtype")
    # Preserve float64 callers; use FP32 objective arithmetic for low precision.
    work = margins if margins.dtype == torch.float64 else margins.float()
    targets = supervision.targets.to(dtype=work.dtype)
    # Empty-slice sum is a graph-connected, exactly zero scalar that does not
    # read unknown margins. Multiplying margins.sum() by zero can propagate NaN.
    zero = work[:0].sum()
    field_bce_sums = []
    field_bce_means = []
    field_rank = []
    for f, field in enumerate(supervision.fields):
        known_idx = supervision.known_indices[f]
        if field.known_rows:
            bce_sum = F.binary_cross_entropy_with_logits(
                work.index_select(0, known_idx),
                targets.index_select(0, known_idx),
                reduction="sum",
            )
            bce_mean = bce_sum / len(field.known_rows)
        else:
            bce_sum = zero
            bce_mean = zero
        field_bce_sums.append(bce_sum)
        field_bce_means.append(bce_mean)
        if field.rank_active:
            hard_scores = work.index_select(0, supervision.hard_indices[f]) / tau
            positive_scores = work.index_select(0, supervision.positive_indices[f]) / tau
            rank = torch.logsumexp(hard_scores, dim=0) - torch.logsumexp(positive_scores, dim=0)
        else:
            rank = zero
        field_rank.append(rank)
    bce_sums = torch.stack(field_bce_sums)
    bce_means = torch.stack(field_bce_means)
    ranks = torch.stack(field_rank)
    case_bce_sums = []
    case_bce_values = []
    case_rank_sums = []
    case_rank_values = []
    for case, fields in enumerate(supervision.case_field_rows):
        bce_sum = torch.stack([field_bce_sums[f] for f in fields]).sum()
        case_bce_sums.append(bce_sum)
        if supervision.bce_case_reduction == "known_pairs":
            bce_value = bce_sum / supervision.case_known_pairs[case]
        else:
            # No-known fields do not supply a BCE denominator; every case was
            # explicitly required to have at least one known field above.
            bce_value = torch.stack([field_bce_means[f] for f in fields]).sum()
            bce_value = bce_value / supervision.case_known_fields[case]
        rank_sum = torch.stack([field_rank[f] for f in fields]).sum()
        case_bce_values.append(bce_value)
        case_rank_sums.append(rank_sum)
        case_rank_values.append(rank_sum / supervision.case_known_fields[case])
    case_bce = torch.stack(case_bce_values)
    case_rank = torch.stack(case_rank_values)
    bce = case_bce.mean()
    rank = case_rank.mean()
    # lambda=0 does not send rank gradients into the model. Raw rank is still
    # reported as an objective diagnostic; it never changes data admission.
    loss = bce if lambda_rank == 0 else bce + lambda_rank * rank
    device = work.device

    def counts(values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    raw = {
        "field_bce_sums": bce_sums,
        "field_known_candidate_counts": counts([len(field.known_rows) for field in supervision.fields]),
        "field_bce_means": bce_means,
        "field_rank_values": ranks,
        "field_rank_active": torch.tensor(
            [field.rank_active for field in supervision.fields], dtype=torch.bool, device=device
        ),
        "field_positive_counts": counts([len(field.positive_rows) for field in supervision.fields]),
        "field_negative_counts": counts([len(field.negative_rows) for field in supervision.fields]),
        "field_hard_counts": counts([len(field.hard_rows) for field in supervision.fields]),
        "case_bce_pair_sums": torch.stack(case_bce_sums),
        "case_known_pair_counts": counts(supervision.case_known_pairs),
        "case_known_field_counts": counts(supervision.case_known_fields),
        "case_bce_values": case_bce,
        "case_rank_sums": torch.stack(case_rank_sums),
        "case_rank_field_denominators": counts(supervision.case_known_fields),
        "case_rank_values": case_rank,
        "case_denominator": counts(len(supervision.case_ids)),
    }
    return LossResult(
        loss, bce, rank, raw, supervision.statistics, supervision.bce_case_reduction,
        "known_fields_including_zero_rank_signal_then_all_cases",
    )


def loss_from_logits(
    logits: torch.Tensor,
    supervision: CandidateSupervision,
    *,
    yes_index: int,
    lambda_rank: float = 0.0,
    tau: float = 1.0,
) -> LossResult:
    """Explicit EOS two-class head ordering; no new head/projector/readout."""
    if logits.ndim != 2 or logits.shape != (supervision.candidate_count, 2):
        raise ValueError("Expected exactly one [no/yes] two-class row per candidate")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Only floating EOS logits in an admitted dtype are supported")
    if type(yes_index) is not int or yes_index not in (0, 1):
        raise ValueError("Freeze the existing head's yes_index explicitly (0 or 1)")
    work = logits if logits.dtype == torch.float64 else logits.float()
    margins = work[:, yes_index] - work[:, 1 - yes_index]
    return loss_from_margins(margins, supervision, lambda_rank=lambda_rank, tau=tau)


def gradcoeff(
    margins: torch.Tensor,
    supervision: CandidateSupervision,
    *,
    lambda_rank: float = 0.0,
    tau: float = 1.0,
) -> MarginGradientCoefficients:
    """Exact dL/dmargin via torch.autograd for a future two-pass VJP caller.

    No hand-derived ranking formula or model replay. Caller must gather the
    WHOLE frozen macro's first-pass margins, preserve key order, recompute the
    same margins in pass two, and VJP with these coefficients exactly once.
    Chunking/averaging this function's already normalized coefficients again
    would change the objective. RNG/dropout/replay/DDP are not implemented.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("gradcoeff needs autograd, not torch.inference_mode")
    if margins.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("gradcoeff requires floating margins, not cast integer inputs")
    with torch.enable_grad():
        dtype = torch.float64 if margins.dtype == torch.float64 else torch.float32
        leaf = margins.detach().to(dtype=dtype).clone().requires_grad_(True)
        result = loss_from_margins(leaf, supervision, lambda_rank=lambda_rank, tau=tau)
        coefficients, = torch.autograd.grad(result.loss, leaf, create_graph=False)
    return MarginGradientCoefficients(
        coefficients.detach(),
        result.loss.detach(),
        result.bce.detach(),
        result.rank.detach(),
        {name: value.detach() for name, value in result.raw.items()},
        result.statistics,
    )
