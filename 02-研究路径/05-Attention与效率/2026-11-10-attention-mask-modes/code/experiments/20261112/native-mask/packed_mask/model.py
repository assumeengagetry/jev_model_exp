"""Differentiable PFv3 tokenwise Qwen3 with native-FA3 mask-only changes.

No PEFT replacement, custom norm/MLP, model loading, or parameter conversion.
Keep saving the ORIGINAL PeftModel, not this execution wrapper.
"""

from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from .layout import BoundMask, MODES, bind_mask


@dataclass(frozen=True)
class PrefixKV:
    """Request-local inference value; tensor storage is read-only by contract."""
    mode: str
    token_ids: tuple
    layers: tuple
    owner: object
    parameter_versions: tuple

    @property
    def length(self):
        return len(self.token_ids)


class MaskedBackbone(nn.Module):
    def __init__(self, backbone, head, native_fa3, mode, *, strict_deterministic=True):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown attention mode')
        if strict_deterministic is not True:
            raise ValueError('This execution contract requires strict_deterministic=True')
        config = backbone.config
        if (config.model_type != 'qwen3' or config.num_hidden_layers != 28
                or config.hidden_size != 1024 or config.num_attention_heads != 16
                or config.num_key_value_heads != 8 or config.head_dim != 128
                or config.rope_scaling is not None or config.attention_dropout != 0
                or config.attention_bias
                or any(kind != 'full_attention' for kind in config.layer_types)
                or len(backbone.layers) != 28):
            raise ValueError('Only the archived admitted Qwen3/GQA configuration is supported')
        if not callable(getattr(native_fa3, 'flash_attn_varlen_func', None)):
            raise TypeError('Pass the actual native FA3 module, not an SDPA/HF mask adapter')
        # These are the existing modules, not new/cached/merged copies. Their
        # original PeftModel registration and save_pretrained keys are untouched.
        self.backbone = backbone
        self.head = head
        self.native_fa3 = native_fa3
        self._mode = mode
        self.strict_deterministic = True
        self._prefix_owner = object()

    @property
    def mode(self):
        return self._mode

    def invalidate_prefixes(self):
        """Required after out-of-band .data edits/load/adapter switching."""
        self._prefix_owner = object()

    def _versions(self):
        # Host version counters, no tensor values or GPU synchronization.
        return tuple((id(p), p._version) for p in self.parameters())

    def _inference_only(self):
        if torch.is_grad_enabled() or self.training or self.backbone.training or self.head.training:
            raise RuntimeError('Immutable prefix/fork is eval + no-grad only; train complete pairs')

    def _native(self, q, k, v, cu_q, cu_k, max_q, max_k, scaling, *, causal):
        # Actual pinned FA3 signature has no dropout_p keyword. Native backward
        # sees both calls directly; no training prefix is ever detached.
        result = self.native_fa3.flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max_q, max_k,
            softmax_scale=scaling, causal=causal, deterministic=True)
        return result[0] if isinstance(result, tuple) else result

    def _attention(self, q, k, v, bound, scaling):
        batch = bound.batch
        if self.mode != 'prefix_block':
            return self._native(q, k, v, batch.cu, batch.cu,
                                batch.max_length, batch.max_length, scaling,
                                causal=self.mode == 'causal')
        pi, si = bound.prefix_indices, bound.suffix_indices
        prefix_out = self._native(
            q.index_select(0, pi).contiguous(),
            k.index_select(0, pi).contiguous(),
            v.index_select(0, pi).contiguous(),
            bound.cu_prefix, bound.cu_prefix, bound.max_prefix, bound.max_prefix,
            scaling, causal=False)
        # Functional index_copy is differentiable for the complete partition.
        out = torch.zeros_like(q).index_copy(0, pi, prefix_out)
        if bound.max_suffix:
            suffix_out = self._native(
                q.index_select(0, si).contiguous(), k, v,
                bound.cu_suffix, batch.cu, bound.max_suffix, batch.max_length,
                scaling, causal=False)
            out = out.index_copy(0, si, suffix_out)
        return out

    def forward(self, batch, prefix_lengths, prefix=None, capture_prefix=False):
        """Return (EOS logits, PrefixKV | None), never train through PrefixKV.

        prefix_lengths: original Candidate:\\n boundary per COMPLETE row, or
        BoundMask from bind_mask. For suffix-only inference each value is the
        complete captured P length; batch positions already start at that P.
        """
        if (not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()):
            raise RuntimeError('Enable strict torch deterministic algorithms before execution')
        if capture_prefix or prefix is not None:
            self._inference_only()
            if self.mode == 'bidirectional':
                raise ValueError('Bidirectional mode must recompute the complete P+S row')
        if prefix is not None:
            if capture_prefix or not isinstance(prefix, PrefixKV):
                raise ValueError('A captured PrefixKV owner is required, without recapture')
            if (prefix.owner is not self._prefix_owner or prefix.mode != self.mode
                    or prefix.parameter_versions != self._versions()):
                raise ValueError('Prefix belongs to another core/mode/parameter version')
            if len(prefix.layers) != len(self.backbone.layers):
                raise ValueError('Incomplete prefix layers')
        cached_length = None if prefix is None else prefix.length
        expected_role = 'prefix' if capture_prefix else 'suffix' if prefix is not None else 'full'
        if isinstance(prefix_lengths, BoundMask):
            bound = prefix_lengths
            if (bound.batch is not batch or bound.mode != self.mode or bound.role != expected_role
                    or bound.cached_prefix_length != cached_length):
                raise ValueError('Bound metadata does not belong to this exact forward')
        else:
            bound = bind_mask(batch, prefix_lengths, self.mode,
                              prefix_only=capture_prefix, cached_prefix_length=cached_length)
        if batch.ids.device.type != 'cuda':
            raise ValueError('Actual native FA3 CUDA execution required; no CPU fallback')
        actual_max_position = batch.max_length + (cached_length or 0)
        if actual_max_position > self.backbone.config.max_position_embeddings:
            raise ValueError('Context exceeded; no truncation')

        captured_ids = None
        if capture_prefix:
            # One entry-time identity snapshot, not a per-layer cu/value sync.
            captured_ids = tuple(batch.real(batch.ids).detach().cpu().tolist())
        x = self.backbone.embed_tokens(batch.ids)
        cos, sin = self.backbone.rotary_emb(x, batch.positions.reshape(1, -1))
        cos = cos.reshape(*batch.ids.shape, 128).unsqueeze(-2)
        sin = sin.reshape(*batch.ids.shape, 128).unsqueeze(-2)
        cache = []
        for index, layer in enumerate(self.backbone.layers):
            residual = x
            normalized = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_norm(attn.q_proj(normalized).view(*batch.ids.shape, 16, 128))
            k = attn.k_norm(attn.k_proj(normalized).view(*batch.ids.shape, 8, 128))
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            v = attn.v_proj(normalized).view(*batch.ids.shape, 8, 128)
            q, k, v = (batch.real(t).contiguous() for t in (q, k, v))
            if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
                raise ValueError('Frozen BF16 QKV contract required; use the admitted autocast context')
            if capture_prefix:
                cache.append((k.detach(), v.detach()))
            if prefix is None:
                out = self._attention(q, k, v, bound, attn.scaling)
            else:
                # Lazy import keeps Triton out of the differentiable full-row core.
                from .fork import fork_attention
                pk, pv = prefix.layers[index]
                if pk.shape[0] != prefix.length:
                    raise ValueError('Prefix KV must contain real P rows, not capacity padding')
                out = fork_attention(q, k, v, pk, pv, batch.cu, batch.max_length,
                                     suffix_causal=self.mode == 'causal',
                                     softmax_scale=attn.scaling)
            out = batch.restore(out.reshape(batch.total, -1))
            x = residual + attn.o_proj(out)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.backbone.norm(x)
        # Do NOT gather hidden states before the head: that is the PFv1 BF16
        # GEMM-shape error. PF packed head consumes all real T token rows.
        scores = self.head(x)
        logits = scores.reshape(-1, scores.shape[-1]).index_select(0, batch.eos_indices)
        captured = None
        if capture_prefix:
            captured = PrefixKV(self.mode, captured_ids, tuple(cache), self._prefix_owner, self._versions())
        return logits, captured
