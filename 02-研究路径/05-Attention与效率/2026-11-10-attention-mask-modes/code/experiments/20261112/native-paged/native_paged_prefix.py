"""Unadmitted, forward-only native FA3 paged-prefix candidate.

No model/runtime adapter, tokenwise computation, fallback, or native import is
provided here. The caller supplies the installed FA3 module and actual prefix
KV, and owns eval state, RoPE, exact-token grouping, and parameter validity.
Packed 3-D Q + paged block-size 1 still requires actual GPU/API validation.
"""

from collections import Counter
from dataclasses import dataclass
import math
import numbers
import operator

import torch


_LAYERS = 28
_Q_HEADS = 16
_KV_HEADS = 8
_HEAD_DIM = 128
_INT32_MAX = 2**31 - 1
_MODES = ('causal', 'prefix_block')


def _integer(value, label):
    # Tensor scalar conversion could synchronize; metadata must be host-owned.
    if isinstance(value, (bool, torch.Tensor)):
        raise TypeError(label + ' must be a host integer, not bool/tensor')
    return operator.index(value)


def _host_integers(values, label):
    if isinstance(values, torch.Tensor):
        raise TypeError(label + ' must be a host sequence, not a tensor')
    return tuple(_integer(value, label) for value in values)


def _versions(values):
    """Freeze the existing PrefixKV (parameter identity, _version) contract."""
    if isinstance(values, torch.Tensor):
        raise TypeError('parameter_versions must be host integer pairs')
    result = []
    for pair in values:
        pair = _host_integers(pair, 'parameter version pair')
        if len(pair) != 2:
            raise ValueError('Each parameter version entry must contain two integers')
        result.append(pair)
    if not result:
        raise ValueError('Actual nonempty parameter_versions are required')
    return tuple(result)


def _identity(mode, token_ids, rendered_prefix_length):
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError('Only causal/prefix_block can reuse a prefix; bidirectional is rejected')
    tokens = _host_integers(token_ids, 'prefix token IDs')
    if not tokens or any(token < 0 for token in tokens):
        raise ValueError('Actual nonempty, nonnegative prefix token IDs are required')
    boundary = _integer(rendered_prefix_length, 'original rendered prefix boundary')
    if boundary < len(tokens):
        raise ValueError('Prefix extends beyond the original rendered prefix boundary')
    if mode == 'prefix_block' and boundary != len(tokens):
        raise ValueError('prefix_block requires the COMPLETE original rendered prefix')
    return tokens, boundary


def _context(device, stream):
    if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
        raise RuntimeError('Paged owner construction/bind/fork require inference_mode for the whole lifetime')
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError('Strict Torch deterministic algorithms must already be enabled')
    if stream.device != device or torch.cuda.current_stream(device).cuda_stream != stream.cuda_stream:
        raise RuntimeError('Use the exact owner CUDA stream; no implicit stream handoff is supported')


def _kv(tensor, rows, device, label):
    if (not isinstance(tensor, torch.Tensor)
            or tensor.shape != (rows, _KV_HEADS, _HEAD_DIM)
            or tensor.device != device or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous() or tensor.requires_grad):
        raise ValueError(label + ' must be contiguous, no-grad BF16 CUDA [real_rows,8,128]')


@dataclass(frozen=True, eq=False)
class PageTable:
    """One wave's read-only metadata; identity-bound to its creating owner.

    Frozen Python fields do not make tensors immutable. The caller must never
    modify these tensors. No tensor value checks or GPU sync occur per layer.
    """

    lengths: tuple
    offsets: tuple
    total: int
    max_seqlen_q: int
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    page_table: torch.Tensor


class PagedPrefixOwner:
    """One request/group, one stream, one exact prefix, all 28 actual layers.

    Construction copies each real prefix K/V once into new [P+C,8,128]
    backing. The source tensors are NOT retained by this object. C must be the
    caller's known maximum actual sum(S) across this group's planned waves,
    not a context or CUDA Graph bucket ceiling. Only sum(S) <= C is locally
    checkable; the exact-capacity provenance remains the caller's contract.
    """

    def __init__(self, prefix_layers, *, mode, prefix_token_ids,
                 rendered_prefix_length, core_owner, parameter_versions,
                 stream, suffix_capacity, native_fa3):
        tokens, boundary = _identity(mode, prefix_token_ids, rendered_prefix_length)
        versions = _versions(parameter_versions)
        capacity = _integer(suffix_capacity, 'actual suffix-wave capacity')
        if capacity < 1 or len(tokens) + capacity > _INT32_MAX:
            raise ValueError('P+C must fit positive int32 page indices, with C >= 1')
        if core_owner is None:
            raise ValueError('An actual captured core prefix-owner token is required')
        native = getattr(native_fa3, 'flash_attn_with_kvcache', None)
        if not callable(native):
            raise TypeError('Supply the actual installed native FA3 cache API; no fallback')
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError('Supply the current CUDA stream explicitly')
        layers = tuple(prefix_layers)
        if len(layers) != _LAYERS or any(len(pair) != 2 for pair in layers):
            raise ValueError('Exactly 28 actual prefix (K,V) layer pairs are required')
        first_k = layers[0][0]
        if not isinstance(first_k, torch.Tensor) or first_k.device.type != 'cuda':
            raise ValueError('Actual CUDA prefix tensors are required; no CPU fallback')
        device = first_k.device
        _context(device, stream)
        for k, v in layers:
            _kv(k, len(tokens), device, 'Prefix K')
            _kv(v, len(tokens), device, 'Prefix V')

        self._mode = mode
        self._token_ids = tokens
        self._boundary = boundary
        self._core_owner = core_owner
        self._parameter_versions = versions
        self._stream = stream
        self._device = device
        self._length = len(tokens)
        self._capacity = capacity
        self._native = native
        self._valid = True
        self._busy = False
        self._active_wave = None
        self._next_layer = 0
        self._counts = Counter()

        backings, prefix_views, paged_views = [], [], []
        for k, v in layers:
            kb = torch.empty((self._length + capacity, _KV_HEADS, _HEAD_DIM),
                             device=device, dtype=torch.bfloat16)
            vb = torch.empty_like(kb)
            kb[:self._length].copy_(k)
            self._counts['prefix_k_copy_submissions'] += 1
            vb[:self._length].copy_(v)
            self._counts['prefix_v_copy_submissions'] += 1
            backings.append((kb, vb))
            prefix_views.append((kb[:self._length], vb[:self._length]))
            paged_views.append((kb.view(-1, 1, _KV_HEADS, _HEAD_DIM),
                                vb.view(-1, 1, _KV_HEADS, _HEAD_DIM)))
        self._backings = tuple(backings)
        self._prefix_views = tuple(prefix_views)
        self._paged_views = tuple(paged_views)

    @property
    def prefix_views(self):
        """Actual P-row views of NEW backing; read-only by caller contract."""
        return self._prefix_views

    @property
    def length(self):
        return self._length

    @property
    def suffix_capacity(self):
        return self._capacity

    @property
    def mode(self):
        return self._mode

    def invalidate(self):
        """Permanently discard this owner; reconstruct from fresh prefix KV.

        Required at request/group end and after out-of-band edits/loads/adapter
        switching. It is not a CUDA synchronization or a tensor-value check.
        External prefix_views/PageTable references remain caller-owned.
        """
        if self._busy:
            raise RuntimeError('Do not invalidate while an owner operation is in progress')
        self._discard()

    def _discard(self):
        self._valid = False
        self._active_wave = None
        self._backings = ()
        self._prefix_views = ()
        self._paged_views = ()

    def _require_context(self):
        if not self._valid:
            raise RuntimeError('Owner invalidated; capture and construct a new owner')
        if self._busy:
            raise RuntimeError('Owner operations must be serial; reentry/concurrency is unsupported')
        _context(self._device, self._stream)

    def bind_wave(self, lengths, *, mode, prefix_token_ids, rendered_prefix_length,
                  core_owner, parameter_versions):
        """Build canonical CUDA int32 metadata ONCE, before all 28 layers.

        lengths is a host sequence in the EXACT order of packed suffix Q/K/V.
        Caller supplies CURRENT core ownership/version state, not a recycled
        capture snapshot. No external/cu_k overlap trick is accepted here.
        """
        self._require_context()
        if self._active_wave is not None:
            raise RuntimeError('Finish all 28 layers before binding another wave, or invalidate')
        tokens, boundary = _identity(mode, prefix_token_ids, rendered_prefix_length)
        versions = _versions(parameter_versions)
        if (mode != self._mode or core_owner is not self._core_owner
                or versions != self._parameter_versions or tokens != self._token_ids
                or boundary != self._boundary):
            self._discard()
            raise ValueError('Prefix/core/mode/boundary/version changed; owner invalidated, recapture required')
        lengths = _host_integers(lengths, 'actual suffix lengths')
        if not lengths or any(length < 1 for length in lengths):
            raise ValueError('Every branch needs its actual nonempty suffix, including caller-owned EOS')
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        total, max_s = offsets[-1], max(lengths)
        if total > self._capacity:
            raise ValueError('Actual suffix sum exceeds C; no truncation, growth, or fallback')
        # Metadata repeats shared prefix PAGE INDICES, never K/V or KV heads.
        # Valid suffix page IDs for branch b are P + cu[b] + arange(S_b).
        prefix_pages = tuple(range(self._length))
        rows = tuple(prefix_pages
                     + tuple(range(self._length + start, self._length + end))
                     + (0,) * (max_s - length)
                     for length, start, end in zip(lengths, offsets[:-1], offsets[1:]))
        self._busy = True
        try:
            table = torch.tensor(rows, dtype=torch.int32, device=self._device)
            cu_q = torch.tensor(offsets, dtype=torch.int32, device=self._device)
            cache_lengths = torch.tensor(tuple(self._length + length for length in lengths),
                                         dtype=torch.int32, device=self._device)
            bound = PageTable(lengths, tuple(offsets), total, max_s,
                              cu_q, cache_lengths, table)
            self._active_wave = bound
            self._next_layer = 0
            self._counts['bound_page_table_waves'] += 1
            return bound
        except Exception:
            self._discard()
            raise
        finally:
            self._busy = False

    def fork(self, q, k, v, *, layer, bound, softmax_scale):
        """Write only this layer's real suffix, then submit ONE cache API call.

        All 28 calls must use the same bound object and layer order 0..27.
        The returned tensor is packed [sum(S),16,128], if the unvalidated
        packed-Q native API is compatible. No alternate route is attempted.
        """
        self._require_context()
        if not isinstance(bound, PageTable) or bound is not self._active_wave:
            raise ValueError('Pass this owner\'s active, unmodified PageTable object')
        layer = _integer(layer, 'layer index')
        if layer != self._next_layer or not 0 <= layer < _LAYERS:
            raise ValueError('Layers must execute once each in order 0..27')
        if (not isinstance(q, torch.Tensor)
                or q.shape != (bound.total, _Q_HEADS, _HEAD_DIM)
                or q.device != self._device or q.dtype != torch.bfloat16
                or not q.is_contiguous() or q.requires_grad):
            raise ValueError('Q must be contiguous, no-grad BF16 CUDA [actual_sumS,16,128]')
        _kv(k, bound.total, self._device, 'Suffix K')
        _kv(v, bound.total, self._device, 'Suffix V')
        if (isinstance(softmax_scale, bool) or not isinstance(softmax_scale, numbers.Real)
                or not math.isfinite(softmax_scale) or softmax_scale <= 0):
            raise ValueError('Pass the model\'s actual positive, finite host softmax scale')
        kb, vb = self._backings[layer]
        # Do not allow suffix writes to clobber Q or overlapping source slots.
        backing_ptrs = {kb.untyped_storage().data_ptr(), vb.untyped_storage().data_ptr()}
        if any(t.untyped_storage().data_ptr() in backing_ptrs for t in (q, k, v)):
            raise ValueError('Q/K/V must not alias this layer\'s owned cache backing')

        self._busy = True
        try:
            start, end = self._length, self._length + bound.total
            kb[start:end].copy_(k)
            self._counts['suffix_k_copy_submissions'] += 1
            vb[start:end].copy_(v)
            self._counts['suffix_v_copy_submissions'] += 1
            k_cache, v_cache = self._paged_views[layer]
            self._counts['native_cache_call_attempts'] += 1
            # k/v MUST stay None: cache API appending would write shared storage.
            # No rotary arguments: caller has ALREADY applied model RoPE to Q/K.
            # No deterministic/dropout arguments: this API has neither parameter.
            result = self._native(
                q=q, k_cache=k_cache, v_cache=v_cache, k=None, v=None,
                cache_seqlens=bound.cache_seqlens, page_table=bound.page_table,
                cu_seqlens_q=bound.cu_seqlens_q, max_seqlen_q=bound.max_seqlen_q,
                softmax_scale=float(softmax_scale), causal=self._mode == 'causal',
                num_splits=1, return_softmax_lse=False)
            self._counts['native_cache_call_returns'] += 1
            if (not isinstance(result, torch.Tensor) or result.shape != q.shape
                    or result.device != self._device or result.dtype != q.dtype
                    or result.requires_grad):
                raise RuntimeError('Unexpected native cache output; packed-Q/API validation required')
            self._next_layer += 1
            if self._next_layer == _LAYERS:
                self._active_wave = None
                self._counts['waves_with_all_28_native_calls_returned'] += 1
            return result
        except Exception:
            # Partial copies/native submissions must not leave a reusable owner.
            self._discard()
            raise
        finally:
            self._busy = False

    def structural_report(self):
        """Host-only structure/counters, NOT kernel instrumentation/admission."""
        kv_bytes_per_row = 2 * _KV_HEADS * _HEAD_DIM * 2  # K+V, BF16
        bound = self._active_wave
        return {
            'owner_valid': self._valid,
            'mode': self._mode,
            'layers': _LAYERS,
            'actual_prefix_rows': self._length,
            'actual_suffix_capacity_rows': self._capacity,
            'backing_shape_per_k_or_v': (self._length + self._capacity, _KV_HEADS, _HEAD_DIM),
            'paged_shape_per_k_or_v': (self._length + self._capacity, 1, _KV_HEADS, _HEAD_DIM),
            'prefix_view_shape_per_k_or_v': (self._length, _KV_HEADS, _HEAD_DIM),
            'configured_kv_bytes_all_layers': _LAYERS * (self._length + self._capacity) * kv_bytes_per_row,
            'configured_prefix_bytes_within_backing': _LAYERS * self._length * kv_bytes_per_row,
            'configured_suffix_capacity_bytes_within_backing': _LAYERS * self._capacity * kv_bytes_per_row,
            'owner_retained_backing_bytes': (
                _LAYERS * (self._length + self._capacity) * kv_bytes_per_row if self._valid else 0),
            'active_wave_actual_sumS': None if bound is None else bound.total,
            'active_wave_metadata_bytes': None if bound is None else 4 * (
                len(bound.lengths) * (self._length + bound.max_seqlen_q)
                + len(bound.lengths) + len(bound.offsets)),
            'num_splits': 1,
            'counts': dict(self._counts),
            'counts_semantics': 'host API submissions/returns; no GPU completion or kernel-launch proof',
            'native_backward': 'unsupported and not instrumented',
            'tokenwise_rows': 'caller-owned; no tokenwise operators run or measured in this module',
            'gpu_api_and_numerical_admission': 'not established by this module; external evidence required',
        }
