"""Forward-only PFv3 packed suffix fork, with a mode-specific suffix mask.

Only SUFFIX_CAUSAL changes the archived attention semantics: causal=True uses
the old suffix triangle; False reads the complete *own* suffix plus shared P.
This file is deliberately not imported by complete-row training.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _online(q, k, v, allowed, acc, denom, maximum):
    scores = tl.dot(q, k) * (0.08838834764831845 * 1.4426950408889634)
    scores = tl.where(allowed, scores, -float('inf'))
    new_max = tl.maximum(maximum, tl.max(scores, 1))
    probabilities = tl.exp2(scores - new_max[:, None])
    alpha = tl.exp2(maximum - new_max)
    acc = acc * alpha[:, None]
    acc = tl.dot(probabilities.to(tl.bfloat16), v, acc)
    denom = denom * alpha + tl.sum(probabilities, 1)
    return acc, denom, new_max


@triton.jit(do_not_specialize=['P'])
def _fork(Q, K, V, PK, PV, CU, OUT, P,
          SUFFIX_CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    head_batch = tl.program_id(1)
    batch = head_batch // 16
    head = head_batch % 16
    kv = head // 2
    begin = tl.load(CU + batch)
    end = tl.load(CU + batch + 1)
    length = end - begin
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    if tl.program_id(0) * BM >= length:
        return
    q = tl.load(Q + ((begin + m[:, None]) * 16 + head) * D + d[None, :],
                m[:, None] < length, 0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(0, tl.cdiv(P, BN)):
        p = start * BN + n
        k = tl.load(PK + (p[None, :] * 8 + kv) * D + d[:, None], p[None, :] < P, 0)
        v = tl.load(PV + (p[:, None] * 8 + kv) * D + d[None, :], p[:, None] < P, 0)
        acc, denom, maximum = _online(q, k, v, p[None, :] < P, acc, denom, maximum)
    for start in range(0, tl.cdiv(length, BN)):
        p = start * BN + n
        k = tl.load(K + ((begin + p[None, :]) * 8 + kv) * D + d[:, None],
                    p[None, :] < length, 0)
        v = tl.load(V + ((begin + p[:, None]) * 8 + kv) * D + d[None, :],
                    p[:, None] < length, 0)
        allowed = p[None, :] < length
        if SUFFIX_CAUSAL:
            allowed = allowed & (p[None, :] <= m[:, None])
        acc, denom, maximum = _online(q, k, v, allowed, acc, denom, maximum)
    tl.store(OUT + ((begin + m[:, None]) * 16 + head) * D + d[None, :],
             acc / denom[:, None], m[:, None] < length)


def fork_attention(q, k, v, prefix_k, prefix_v, cu, max_length, *,
                   suffix_causal, softmax_scale=0.08838834764831845):
    """Requires cu/max_length validated by bind_mask once before the layer loop.

    Prefix tensors have exactly [P,8,128] rows; current q/k/v have exactly
    [real suffix T,16|8,128]. P is shared by one owner; CU selects only each
    branch's own suffix. No prefix writes, backward, or dense-mask fallback.
    """
    if torch.is_grad_enabled():
        raise RuntimeError('Packed mask fork is forward-only; training must use native FA3')
    if not isinstance(suffix_causal, bool):
        raise TypeError('suffix_causal must be the explicit mode-derived boolean')
    if softmax_scale != 0.08838834764831845:
        raise ValueError('Only the archived D128 attention scale is admitted')
    if (q.ndim != 3 or q.shape[1:] != (16, 128) or q.shape[0] < 1
            or k.shape != (q.shape[0], 8, 128) or v.shape != k.shape
            or prefix_k.ndim != 3 or prefix_k.shape != prefix_v.shape
            or prefix_k.shape[1:] != (8, 128) or prefix_k.shape[0] < 1):
        raise ValueError('Only admitted real-T/real-P Qwen3 GQA layout is supported')
    tensors = (q, k, v, prefix_k, prefix_v)
    if q.device.type != 'cuda' or any(t.device != q.device or t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError('All QKV must share the admitted CUDA device and BF16 dtype')
    if (cu.device != q.device or cu.dtype != torch.int32 or cu.ndim != 1
            or cu.numel() < 2 or max_length < 1):
        raise ValueError('Once-bound native cu/max_length required')
    if any(not t.is_contiguous() for t in (*tensors, cu)):
        raise ValueError('Contiguous packed storage required')
    out = torch.empty_like(q)
    _fork[(triton.cdiv(max_length, 16), (cu.numel() - 1) * 16)](
        q, k, v, prefix_k, prefix_v, cu, out, prefix_k.shape[0], suffix_causal,
        16, 64, 128, num_warps=4, num_stages=2, enable_fp_fusion=False)
    return out
