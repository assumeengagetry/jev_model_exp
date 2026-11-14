"""Real-token layout and once-per-batch, read-only attention metadata.

The first eight TokenBatch fields retain the archived PFv3 layout contract.
No renderer, truncation, relationship grouping, or EOS insertion lives here.
"""

from dataclasses import dataclass
import itertools
import operator

import torch


MODES = ('causal', 'bidirectional', 'prefix_block')
_INT32_MAX = 2**31 - 1


def _integers(values, label):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
            raise ValueError(label + ' must be a one-dimensional integer tensor')
        # This optional device-to-host transfer happens at binding, never in a layer.
        values = values.detach().cpu().tolist()
    result = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(label + ' must contain integers, not booleans')
        result.append(operator.index(value))
    return tuple(result)


def _offsets(lengths):
    result = tuple(itertools.accumulate((0, *lengths)))
    if result[-1] > _INT32_MAX:
        raise ValueError('Native FA3 cu_seqlens would overflow int32')
    return result


@dataclass(frozen=True)
class TokenBatch:
    ids: torch.Tensor
    positions: torch.Tensor
    cu: torch.Tensor
    real_indices: torch.Tensor
    eos_indices: torch.Tensor
    lengths: tuple
    layout: str
    mask: torch.Tensor | None

    @property
    def total(self):
        return sum(self.lengths)

    @property
    def max_length(self):
        return max(self.lengths)

    def real(self, tensor):
        flat = tensor.reshape(-1, *tensor.shape[self.ids.ndim:])
        return flat if self.layout == 'packed' else flat.index_select(0, self.real_indices)

    def restore(self, tensor):
        if self.layout == 'packed':
            return tensor
        result = tensor.new_zeros((self.ids.numel(), *tensor.shape[1:]))
        return result.index_copy(0, self.real_indices, tensor).view(
            *self.ids.shape, *tensor.shape[1:])


def make_batch(rows, device, layout='packed', pad=151643, prefix_length=0):
    """PFv3 positions and EOS indices, with no padded matrix in packed mode.

    prefix_length is a shared positional offset for suffix-only inference, not
    the per-row prefix/candidate boundary of a complete training pair.
    """
    if layout not in ('packed', 'dense'):
        raise ValueError('Explicit packed or dense layout required')
    prefix_length = operator.index(prefix_length)
    if prefix_length < 0:
        raise ValueError('Negative positional offset')
    rows = tuple(_integers(row, 'token IDs') for row in rows)
    if not rows or any(not row for row in rows):
        raise ValueError('Nonempty actual token rows required')
    lengths = tuple(map(len, rows))
    offsets = _offsets(lengths)
    cu = torch.tensor(offsets, dtype=torch.int32, device=device)
    positions = tuple(tuple(range(prefix_length, prefix_length + n)) for n in lengths)
    if layout == 'packed':
        ids = torch.tensor(tuple(itertools.chain.from_iterable(rows)), dtype=torch.long, device=device)
        pos = torch.tensor(tuple(itertools.chain.from_iterable(positions)), dtype=torch.long, device=device)
        real = torch.arange(offsets[-1], dtype=torch.long, device=device)
        eos = torch.tensor(tuple(end - 1 for end in offsets[1:]), dtype=torch.long, device=device)
        mask = None
    else:
        width = (max(lengths) + 7) // 8 * 8
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        pos = torch.zeros_like(ids)
        mask = torch.zeros_like(ids)
        indices = []
        for i, (row, row_pos) in enumerate(zip(rows, positions)):
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            pos[i, :len(row)] = torch.tensor(row_pos, dtype=torch.long)
            mask[i, :len(row)] = 1
            indices.extend(range(i * width, i * width + len(row)))
        ids, pos, mask = ids.to(device), pos.to(device), mask.to(device)
        real = torch.tensor(indices, dtype=torch.long, device=device)
        eos = torch.tensor(tuple(i * width + n - 1 for i, n in enumerate(lengths)),
                           dtype=torch.long, device=device)
    return TokenBatch(ids, pos, cu, real, eos, lengths, layout, mask)


@dataclass(frozen=True)
class BoundMask:
    """Owned by one TokenBatch; all tensors must stay read-only until backward.

    role: full = complete pairs, prefix = one prefill owner, suffix = branches
    reading one captured owner. prefix/suffix indices address *real* flattened T,
    never the dense physical offsets. cu_prefix/cu_suffix retain row order.
    """
    batch: TokenBatch
    mode: str
    role: str
    prefix_lengths: tuple
    prefix_indices: torch.Tensor | None
    suffix_indices: torch.Tensor | None
    cu_prefix: torch.Tensor | None
    cu_suffix: torch.Tensor | None
    max_prefix: int
    max_suffix: int
    cached_prefix_length: int | None


def _check_batch(batch, position_offset):
    """Complete value checks once at entry, not 28 times at attention dispatch."""
    lengths = _integers(batch.lengths, 'row lengths')
    if not lengths or any(n < 1 for n in lengths):
        raise ValueError('Nonempty real sequences required')
    offsets = _offsets(lengths)
    if batch.layout not in ('packed', 'dense'):
        raise ValueError('Unknown layout')
    device = batch.ids.device
    named = (('ids', batch.ids, torch.long), ('positions', batch.positions, torch.long),
             ('cu', batch.cu, torch.int32), ('real_indices', batch.real_indices, torch.long),
             ('eos_indices', batch.eos_indices, torch.long))
    for name, tensor, dtype in named:
        if tensor.device != device or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(name + ' has an invalid device/dtype/storage layout')
    if batch.positions.shape != batch.ids.shape:
        raise ValueError('Position and token shapes differ')
    if batch.cu.ndim != 1 or batch.real_indices.ndim != 1 or batch.eos_indices.ndim != 1:
        raise ValueError('cu, real_indices and eos_indices must be one-dimensional')
    if batch.layout == 'packed':
        if batch.ids.shape != (offsets[-1],) or batch.mask is not None:
            raise ValueError('Packed IDs must contain exactly real T tokens')
        expected_real = tuple(range(offsets[-1]))
        expected_eos = tuple(n - 1 for n in offsets[1:])
    else:
        if batch.ids.ndim != 2 or batch.ids.shape[0] != len(lengths):
            raise ValueError('Dense row shape differs from lengths')
        width = batch.ids.shape[1]
        if width < max(lengths) or batch.mask is None:
            raise ValueError('Dense width/mask invalid')
        if batch.mask.shape != batch.ids.shape or batch.mask.device != device:
            raise ValueError('Dense mask shape/device invalid')
        expected_real = tuple(i * width + j for i, n in enumerate(lengths) for j in range(n))
        expected_eos = tuple(i * width + n - 1 for i, n in enumerate(lengths))
        expected_mask = [[int(j < n) for j in range(width)] for n in lengths]
        if batch.mask.detach().cpu().tolist() != expected_mask:
            raise ValueError('Dense mask must select exactly the real right-padded rows')
    if tuple(batch.cu.detach().cpu().tolist()) != offsets:
        raise ValueError('cu must be int32 cumulative actual row lengths')
    if tuple(batch.real_indices.detach().cpu().tolist()) != expected_real:
        raise ValueError('Real-token mapping differs from row-major PF layout')
    if tuple(batch.eos_indices.detach().cpu().tolist()) != expected_eos:
        raise ValueError('EOS indices must point at the last actual token of every row')
    # Validated CPU indexing avoids a device-side gather on unchecked indices.
    flat_pos = batch.positions.detach().cpu().reshape(-1).tolist()
    actual_pos = tuple(flat_pos[i] for i in expected_real)
    expected_pos = tuple(position_offset + j for n in lengths for j in range(n))
    if actual_pos != expected_pos:
        raise ValueError('RoPE positions must be row-local, plus the actual cached P offset')
    return lengths, offsets


def bind_mask(batch, prefix_lengths, mode, *, prefix_only=False, cached_prefix_length=None):
    """Bind/validate on the host once, then reuse this immutable object in all layers.

    Tensor prefix_lengths may synchronize once here. Prefer collator-owned CPU
    lengths. A caller may prebind and pass BoundMask as forward's second argument;
    modifying batch/metadata tensors afterwards is outside the contract.
    """
    if mode not in MODES:
        raise ValueError('Unknown attention mode')
    if prefix_only and cached_prefix_length is not None:
        raise ValueError('Cannot capture a prefix from a suffix fork')
    if mode == 'bidirectional' and (prefix_only or cached_prefix_length is not None):
        raise ValueError('Whole-row bidirectional attention cannot cache a prefix')
    if cached_prefix_length is not None:
        cached_prefix_length = operator.index(cached_prefix_length)
        if cached_prefix_length < 1:
            raise ValueError('A real cached prefix is required')
    role = 'prefix' if prefix_only else 'suffix' if cached_prefix_length is not None else 'full'
    lengths, offsets = _check_batch(batch, cached_prefix_length or 0)
    if prefix_lengths is None:
        if mode == 'prefix_block' or role != 'full':
            raise ValueError('Explicit per-row actual prefix lengths required')
        prefixes = (0,) * len(lengths)
    else:
        prefixes = _integers(prefix_lengths, 'prefix lengths')
    if len(prefixes) != len(lengths):
        raise ValueError('One prefix boundary per row required')
    if role == 'prefix':
        if len(lengths) != 1 or prefixes != lengths:
            raise ValueError('Prefill must contain exactly one complete real prefix block')
    elif role == 'suffix':
        if any(p != cached_prefix_length for p in prefixes):
            raise ValueError('Every branch must refer to the complete shared prefix')
    elif any(p < 0 or p > n for p, n in zip(prefixes, lengths)):
        raise ValueError('Prefix boundary outside its own row')
    if mode == 'prefix_block' and role == 'full':
        if any(not 0 < p < n for p, n in zip(prefixes, lengths)):
            raise ValueError('Training/full pairs require P > 0 and S >= 1 including actual EOS')

    p_indices = s_indices = cu_p = cu_s = None
    max_p = max_s = 0
    if mode == 'prefix_block' and role != 'suffix':
        suffixes = tuple(n - p for p, n in zip(prefixes, lengths))
        p_indices = torch.tensor(tuple(i for start, p in zip(offsets, prefixes)
                                       for i in range(start, start + p)),
                                 dtype=torch.long, device=batch.ids.device)
        s_indices = torch.tensor(tuple(i for start, p, n in zip(offsets, prefixes, lengths)
                                       for i in range(start + p, start + n)),
                                 dtype=torch.long, device=batch.ids.device)
        cu_p = torch.tensor(_offsets(prefixes), dtype=torch.int32, device=batch.ids.device)
        cu_s = torch.tensor(_offsets(suffixes), dtype=torch.int32, device=batch.ids.device)
        max_p, max_s = max(prefixes), max(suffixes)
    return BoundMask(batch, mode, role, prefixes, p_indices, s_indices,
                     cu_p, cu_s, max_p, max_s, cached_prefix_length)
