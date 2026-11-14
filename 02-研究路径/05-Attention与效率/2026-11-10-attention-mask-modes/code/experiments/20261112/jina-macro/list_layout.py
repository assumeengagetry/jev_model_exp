"""Unexecuted candidate-list components: real tokens, no supervision in layout.

The integrating process must make the *frozen* padding-v3 ``packed_layout``
module importable. This file deliberately does not edit sys.path, duplicate its
TokenBatch class, tokenize data, choose candidates, or manufacture EOS tokens.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from packed_layout import TokenBatch


TRAIN_PAIR_LIMIT = 8192
TRAIN_FIELD_LIMIT = 16384
MODEL_CONTEXT_LIMIT = 40960

FieldKey = tuple[Hashable, Hashable]
CandidateKey = tuple[Hashable, Hashable, Hashable]


@dataclass(frozen=True)
class CandidateField:
    """One complete field in its original candidate order; no target fields."""

    prefix_ids: Sequence[int]
    suffix_ids: Sequence[Sequence[int]]
    case_id: Hashable
    field_id: Hashable
    keys: Sequence[Hashable]


@dataclass(frozen=True)
class FieldLengthRecord:
    field_key: FieldKey
    prefix_length: int
    suffix_lengths: tuple[int, ...]
    pair_lengths: tuple[int, ...]
    physical_length: int
    over_train_pair_indices: tuple[int, ...]
    over_train_field_limit: bool
    over_model_context_limit: bool


@dataclass(frozen=True)
class CaseLengthRejection:
    case_id: Hashable
    # Each reason identifies an offending *field*. All fields of this case are
    # rejected together; these are not instructions to remove candidates.
    reasons: tuple[tuple[FieldKey, str], ...]


@dataclass(frozen=True)
class LayoutAudit:
    split: str
    case_ids: tuple[Hashable, ...]
    fields: tuple[FieldLengthRecord, ...]
    rejected_cases: tuple[CaseLengthRejection, ...]
    unsupported_field_keys: tuple[FieldKey, ...]
    unsupported_case_ids: tuple[Hashable, ...]

    @property
    def rejected_case_ids(self) -> tuple[Hashable, ...]:
        return tuple(item.case_id for item in self.rejected_cases)


class CaseLengthError(ValueError):
    """Training admission failed, without filtering or allocating a batch."""

    def __init__(self, audit: LayoutAudit):
        self.audit = audit
        super().__init__(
            "Whole training case(s) exceed the frozen length limits; inspect "
            "exception.audit.rejected_cases. No field/candidate was truncated "
            "or silently dropped."
        )


@dataclass(frozen=True)
class TokenSegment:
    """Disjoint physical half-open interval; candidate_index=-1 for prefix."""

    field_index: int
    candidate_index: int
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class CandidateRestoration:
    key: CandidateKey
    row: int
    field_index: int
    candidate_in_field: int
    suffix_start: int
    suffix_end: int
    eos_offset: int


@dataclass(frozen=True)
class CandidateLayout:
    batch: TokenBatch
    position_mode: str
    pad_id: int
    eos_token_id: int
    audit: LayoutAudit
    case_ids: tuple[Hashable, ...]
    field_keys: tuple[FieldKey, ...]
    candidate_keys: tuple[CandidateKey, ...]
    field_case_indices: tuple[int, ...]
    candidate_field_indices: tuple[int, ...]
    # cu-like arrays on the host: fields -> physical tokens / candidate rows.
    field_token_offsets: tuple[int, ...]
    field_candidate_offsets: tuple[int, ...]
    prefix_offsets: tuple[tuple[int, int], ...]
    candidate_token_offsets: tuple[tuple[int, int], ...]
    eos_offsets: tuple[int, ...]
    restoration: tuple[CandidateRestoration, ...]
    segments: tuple[TokenSegment, ...]
    # All three tensors contain exactly T real entries, never tile padding.
    token_owner: torch.Tensor
    token_candidate: torch.Tensor
    token_segment: torch.Tensor

    @property
    def total_tokens(self) -> int:
        return self.field_token_offsets[-1]

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_keys)

    def gather_eos(self, hidden: torch.Tensor) -> torch.Tensor:
        """Gather *every* candidate EOS from real hidden [T, ...]."""
        if hidden.ndim < 2 or hidden.shape[0] != self.total_tokens:
            raise ValueError("EOS gather requires real hidden [T, ...]")
        if hidden.device != self.batch.ids.device:
            raise ValueError("Hidden and layout must be on the same device")
        return hidden.index_select(0, self.batch.eos_indices)

    def unflatten_candidates(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return per-field rows in precisely the supplied field/candidate order."""
        if values.ndim < 1 or values.shape[0] != self.candidate_count:
            raise ValueError("Expected one row per candidate")
        return tuple(
            values[a:b]
            for a, b in zip(
                self.field_candidate_offsets[:-1], self.field_candidate_offsets[1:]
            )
        )

    def restore_candidate_order(
        self, values: torch.Tensor, reference_keys: Sequence[CandidateKey]
    ) -> torch.Tensor:
        """Restore a complete frozen key order after an explicit permutation.

        This restores identities, not attention outputs: interaction and
        sequential positions can legitimately change under permutation.
        Partial selections and missing/duplicate keys are rejected.
        """
        if values.ndim < 1 or values.shape[0] != self.candidate_count:
            raise ValueError("Expected one row per candidate")
        reference = tuple(reference_keys)
        if len(reference) != self.candidate_count or len(set(reference)) != len(reference):
            raise ValueError("Reference keys must be a complete permutation")
        rows = {key: row for row, key in enumerate(self.candidate_keys)}
        if any(key not in rows for key in reference):
            raise ValueError("Reference keys differ from this complete candidate set")
        order = torch.tensor(
            [rows[key] for key in reference], dtype=torch.long, device=values.device
        )
        return values.index_select(0, order)


def _token_ids(value: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(value, (torch.Tensor, str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a host sequence of token integers")
    result = tuple(value)
    if not result or any(
        not isinstance(token, Integral) or isinstance(token, bool) or token < 0
        for token in result
    ):
        raise ValueError(f"{name} must contain nonempty, nonnegative token IDs")
    return tuple(int(token) for token in result)


def _identity(value: Any, name: str) -> Hashable:
    # Restrict identities to stable, serializable primitives, not tensor values
    # or objects whose equality depends on mutable state / device execution.
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} cannot be empty")
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, tuple) and value:
        return tuple(_identity(item, name) for item in value)
    raise TypeError(f"{name} must be a nonempty string, integer, or tuple thereof")


def _normalize_fields(
    fields: Sequence[CandidateField | Mapping[str, Any]], eos_token_id: int
) -> tuple[CandidateField, ...]:
    if (
        not isinstance(eos_token_id, Integral)
        or isinstance(eos_token_id, bool)
        or eos_token_id < 0
    ):
        raise ValueError("Pass the actual base tokenizer's EOS token ID explicitly")
    if isinstance(fields, (torch.Tensor, str, bytes)) or not isinstance(fields, Sequence):
        raise TypeError("fields must be a host sequence of complete field records")
    if not fields:
        raise ValueError("At least one complete field is required")
    required = {"prefix_ids", "suffix_ids", "case_id", "field_id", "keys"}
    normalized = []
    seen_fields = set()
    for item in fields:
        if isinstance(item, Mapping):
            if set(item) != required:
                raise ValueError(
                    "Field mappings require exactly prefix_ids/suffix_ids/"
                    "case_id/field_id/keys; supervision belongs outside layout"
                )
            item = CandidateField(**item)
        if not isinstance(item, CandidateField):
            raise TypeError("Expected CandidateField or its exact five-key mapping")
        case_id = _identity(item.case_id, "case_id")
        field_id = _identity(item.field_id, "field_id")
        field_key = (case_id, field_id)
        if field_key in seen_fields:
            raise ValueError("Duplicate (case_id, field_id); do not split a field")
        seen_fields.add(field_key)
        prefix = _token_ids(item.prefix_ids, "prefix_ids")
        if isinstance(item.suffix_ids, (torch.Tensor, str, bytes)) or not isinstance(
            item.suffix_ids, Sequence
        ):
            raise TypeError("suffix_ids must be a host sequence of complete suffixes")
        suffixes = tuple(_token_ids(ids, "suffix_ids entry") for ids in item.suffix_ids)
        if not suffixes or any(ids[-1] != eos_token_id for ids in suffixes):
            raise ValueError("Each original suffix must already end in the real EOS")
        if isinstance(item.keys, (torch.Tensor, str, bytes)) or not isinstance(
            item.keys, Sequence
        ):
            raise TypeError("keys must be a host sequence of stable candidate IDs")
        keys = tuple(_identity(key, "candidate key") for key in item.keys)
        if len(keys) != len(suffixes) or len(set(keys)) != len(keys):
            raise ValueError("Each suffix needs one unique, stable key within its field")
        normalized.append(CandidateField(prefix, suffixes, case_id, field_id, keys))
    return tuple(normalized)


def _audit(fields: tuple[CandidateField, ...], split: str) -> LayoutAudit:
    if split not in ("train", "validation", "inference"):
        raise ValueError("split must be train, validation, or inference")
    cases = tuple(dict.fromkeys(field.case_id for field in fields))
    records = []
    reasons: dict[Hashable, list[tuple[FieldKey, str]]] = {}
    unsupported_fields = []
    unsupported_cases = set()
    for field in fields:
        key = (field.case_id, field.field_id)
        p = len(field.prefix_ids)
        suffix_lengths = tuple(len(ids) for ids in field.suffix_ids)
        pair_lengths = tuple(p + length for length in suffix_lengths)
        physical = p + sum(suffix_lengths)
        long_pairs = tuple(i for i, length in enumerate(pair_lengths) if length > TRAIN_PAIR_LIMIT)
        long_field = physical > TRAIN_FIELD_LIMIT
        unsupported = physical > MODEL_CONTEXT_LIMIT
        records.append(
            FieldLengthRecord(key, p, suffix_lengths, pair_lengths, physical,
                              long_pairs, long_field, unsupported)
        )
        if split == "train":
            if long_pairs:
                reasons.setdefault(field.case_id, []).append((key, "pair_length_gt_8192"))
            if long_field:
                reasons.setdefault(field.case_id, []).append((key, "physical_field_gt_16384"))
        if unsupported:
            unsupported_fields.append(key)
            unsupported_cases.add(field.case_id)
    return LayoutAudit(
        split, cases, tuple(records),
        tuple(CaseLengthRejection(case_id, tuple(reasons[case_id]))
              for case_id in cases if case_id in reasons),
        tuple(unsupported_fields),
        tuple(case_id for case_id in cases if case_id in unsupported_cases),
    )


def audit_candidate_fields(
    fields: Sequence[CandidateField | Mapping[str, Any]],
    *,
    eos_token_id: int,
    split: str = "train",
) -> LayoutAudit:
    """Host-only static report; no tensor allocation, filtering, or tokenization.

    The caller must supply all fields of every complete case. This component
    cannot detect siblings omitted upstream. Validation/inference never apply
    training rejection and always report the shared physical 40960 coverage.
    """
    return _audit(_normalize_fields(fields, eos_token_id), split)


def build_candidate_layout(
    fields: Sequence[CandidateField | Mapping[str, Any]],
    device: torch.device | str,
    pad: int,
    position_mode: str = "sequential",
    *,
    eos_token_id: int,
    split: str = "train",
) -> CandidateLayout:
    """Build prefix-once + original suffix/EOS streams; never trim or select.

    ``pad`` is retained as an explicit compatibility argument, but no pad token
    is inserted. ``batch.cu`` and ``lengths`` describe fields, while
    ``batch.eos_indices`` describes candidates (usually more entries).
    Treat the resulting tensors as immutable for the lifetime of all graphs.
    """
    if position_mode not in ("sequential", "candidate_local"):
        raise ValueError("position_mode must be sequential or candidate_local")
    if not isinstance(pad, Integral) or isinstance(pad, bool) or pad < 0:
        raise ValueError("pad must be a nonnegative token ID; no padding is performed")
    normalized = _normalize_fields(fields, eos_token_id)
    audit = _audit(normalized, split)
    if audit.rejected_cases:
        raise CaseLengthError(audit)
    case_ids = audit.case_ids
    case_rows = {case_id: i for i, case_id in enumerate(case_ids)}
    ids: list[int] = []
    positions: list[int] = []
    owners: list[int] = []
    token_candidates: list[int] = []
    token_segments: list[int] = []
    field_offsets = [0]
    field_candidate_offsets = [0]
    prefix_offsets = []
    candidate_offsets = []
    field_keys = []
    candidate_keys = []
    field_case_indices = []
    candidate_field_indices = []
    restoration = []
    segments = []
    eos = []
    for field_index, field in enumerate(normalized):
        field_start = len(ids)
        p = len(field.prefix_ids)
        prefix_end = field_start + p
        field_keys.append((field.case_id, field.field_id))
        field_case_indices.append(case_rows[field.case_id])
        prefix_offsets.append((field_start, prefix_end))
        segments.append(TokenSegment(field_index, -1, field_start, prefix_end, "prefix"))
        ids.extend(field.prefix_ids)
        positions.extend(range(p))
        owners.extend([field_index] * p)
        token_candidates.extend([-1] * p)
        token_segments.extend([0] * p)
        for local_index, (key, suffix) in enumerate(zip(field.keys, field.suffix_ids)):
            candidate_index = len(candidate_keys)
            start = len(ids)
            end = start + len(suffix)
            full_key = (field.case_id, field.field_id, key)
            candidate_keys.append(full_key)
            candidate_field_indices.append(field_index)
            candidate_offsets.append((start, end))
            eos.append(end - 1)
            restoration.append(CandidateRestoration(
                full_key, candidate_index, field_index, local_index, start, end, end - 1
            ))
            segments.append(TokenSegment(field_index, candidate_index, start, end, "suffix"))
            ids.extend(suffix)
            pos_start = start - field_start if position_mode == "sequential" else p
            positions.extend(range(pos_start, pos_start + len(suffix)))
            owners.extend([field_index] * len(suffix))
            token_candidates.extend([candidate_index] * len(suffix))
            token_segments.extend([1] * len(suffix))
        field_offsets.append(len(ids))
        field_candidate_offsets.append(len(candidate_keys))
    lengths = tuple(b - a for a, b in zip(field_offsets[:-1], field_offsets[1:]))
    total = len(ids)
    if total > torch.iinfo(torch.int32).max:
        raise ValueError("Packed offsets exceed the admitted int32 metadata range")
    device = torch.device(device)
    batch = TokenBatch(
        ids=torch.tensor(ids, dtype=torch.long, device=device),
        positions=torch.tensor(positions, dtype=torch.long, device=device),
        cu=torch.tensor(field_offsets, dtype=torch.int32, device=device),
        real_indices=torch.arange(total, dtype=torch.long, device=device),
        eos_indices=torch.tensor(eos, dtype=torch.long, device=device),
        lengths=lengths,
        layout="packed",
        mask=None,
    )
    return CandidateLayout(
        batch=batch,
        position_mode=position_mode,
        pad_id=int(pad),
        eos_token_id=int(eos_token_id),
        audit=audit,
        case_ids=case_ids,
        field_keys=tuple(field_keys),
        candidate_keys=tuple(candidate_keys),
        field_case_indices=tuple(field_case_indices),
        candidate_field_indices=tuple(candidate_field_indices),
        field_token_offsets=tuple(field_offsets),
        field_candidate_offsets=tuple(field_candidate_offsets),
        prefix_offsets=tuple(prefix_offsets),
        candidate_token_offsets=tuple(candidate_offsets),
        eos_offsets=tuple(eos),
        restoration=tuple(restoration),
        segments=tuple(segments),
        token_owner=torch.tensor(owners, dtype=torch.int32, device=device),
        token_candidate=torch.tensor(token_candidates, dtype=torch.int32, device=device),
        token_segment=torch.tensor(token_segments, dtype=torch.int8, device=device),
    )
