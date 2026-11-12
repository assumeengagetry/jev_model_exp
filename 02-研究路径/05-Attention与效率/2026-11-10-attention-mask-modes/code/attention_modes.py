"""Attention-mask modes for the Qwen3 pair classifier: causal / bidirectional / prefix_block.

Row layout (unchanged): prefix tokens, candidate tokens, EOS, right padding.
  causal         stock behaviour, nothing here is used.
  bidirectional  every real token sees every real token (padding-only mask).
  prefix_block   prefix tokens see the prefix; candidate+EOS tokens see the whole row;
                 the prefix never sees the candidate (the Mammoth / d3-turbo layout).
Written against transformers 4.57.6 (Qwen3Attention / Qwen2Attention, AttentionInterface, flash wrapper).
"""
import contextlib
import contextvars

import torch

MODES = ('causal', 'bidirectional', 'prefix_block')
ATTENTION_LAYERS = ('Qwen3Attention', 'Qwen2Attention')    # Qwen3 and Qwen2.5: every layer is softmax attention
_LAYOUT = contextvars.ContextVar('pair_attention_layout', default=None)


def attention_layers(model):
    layers = [m for m in model.modules() if type(m).__name__ in ATTENTION_LAYERS]
    if len(layers) != model.config.num_hidden_layers:
        raise RuntimeError('Expected one attention layer per hidden layer')
    return layers


# ---- SDPA: an explicit 4D mask, passed as `attention_mask` -------------------------------------

def dense_mask(mode, row_len, width, prefix_len=None):
    """Bool mask, True = may attend. [B,1,1,L] for bidirectional, [B,1,L,L] for prefix_block.

    Padded query rows keep the real keys, so no row is fully masked (that would give NaN).
    """
    pos = torch.arange(width, device=row_len.device)
    key_ok = pos[None] < row_len[:, None]                          # real (non-padding) keys
    if mode == 'bidirectional':
        return key_ok[:, None, None, :]
    if mode == 'prefix_block':
        candidate = pos[None] >= prefix_len[:, None]               # candidate, EOS and padding
        allow = candidate[:, :, None] | ~candidate[:, None, :]     # query after the prefix, or key inside it
        return (allow & key_ok[:, None, :])[:, None]
    raise ValueError(f'No dense mask for attention mode {mode!r}')


# ---- Flash backends: no dense mask exists, the kernel takes a causal flag and lengths ----------

def enable_bidirectional(model):
    """Whole-row bidirectional on a flash backend: its wrapper reads `module.is_causal`."""
    for layer in attention_layers(model):
        layer.is_causal = False
    model.config.is_causal = False     # read by transformers v5 mask creation and by vLLM, not by 4.57.6


@contextlib.contextmanager
def attention_layout(prefix_len, row_len):
    """Per-row prefix length and total length for prefix_block.

    With gradient checkpointing the layers run again inside backward(), so keep this open over it.
    """
    token = _LAYOUT.set((prefix_len.to(torch.int32), row_len.to(torch.int32)))
    try:
        yield
    finally:
        _LAYOUT.reset(token)


def _cumulative(lengths):
    return torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))


def prefix_block_flash_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                               sliding_window=None, **kwargs):
    """prefix_block as two unmasked flash calls per layer: prefix -> prefix, candidate -> whole row."""
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
    layout = _LAYOUT.get()
    if layout is None:
        raise RuntimeError('prefix_block attention needs attention_layout(...) around forward and backward')
    if sliding_window is not None:
        raise RuntimeError('prefix_block attention does not cover sliding-window layers')
    prefix_len, row_len = (lengths.to(query.device) for lengths in layout)
    candidate_len = row_len - prefix_len
    if len(row_len) != query.shape[0] or int(prefix_len.min()) < 1 or int(candidate_len.min()) < 1:
        raise RuntimeError('Every row needs a non-empty prefix and a non-empty candidate part')
    if query.dtype == torch.float32 and torch.is_autocast_enabled():   # same cast as the stock flash wrapper
        dtype = torch.get_autocast_gpu_dtype()
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    (_, varlen, _, _), flash_kwargs_for = lazy_import_flash_attention(module.config._attn_implementation)
    flash_kwargs = flash_kwargs_for(query_length=query.shape[2], key_length=key.shape[2], is_causal=False,
                                    dropout=dropout, softmax_scale=scaling)
    q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)      # [B, L, heads, D]
    pos = torch.arange(q.shape[1], device=q.device)[None]
    in_prefix = pos < prefix_len[:, None]
    in_row = pos < row_len[:, None]
    in_candidate = in_row & ~in_prefix

    def attend(queries, keys, query_len, key_len):
        out = varlen(q[queries], k[keys], v[keys],
                     cu_seqlens_q=_cumulative(query_len), cu_seqlens_k=_cumulative(key_len),
                     max_seqlen_q=int(query_len.max()), max_seqlen_k=int(key_len.max()), **flash_kwargs)
        return out[0] if isinstance(out, tuple) else out

    out = torch.zeros_like(q)
    out[in_prefix] = attend(in_prefix, in_prefix, prefix_len, prefix_len)
    out[in_candidate] = attend(in_candidate, in_row, candidate_len, row_len)
    return out, None


def install_prefix_block_flash(backend):
    """Point an already registered flash backend (e.g. 'flash_attn3_hopper') at the prefix_block forward."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(backend, prefix_block_flash_forward)
