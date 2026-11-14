"""Inference-only packed suffixes reading immutable actual-length prefix KV."""
import torch
import triton
import triton.language as tl


@triton.jit
def _online(q,k,v,allowed,acc,denom,maximum):
    scores=tl.dot(q,k)*(0.08838834764831845*1.4426950408889634)
    scores=tl.where(allowed,scores,-float('inf'))
    new_max=tl.maximum(maximum,tl.max(scores,1))
    probabilities=tl.exp2(scores-new_max[:,None])
    alpha=tl.exp2(maximum-new_max)
    acc=acc*alpha[:,None]
    acc=tl.dot(probabilities.to(tl.bfloat16),v,acc)
    denom=denom*alpha+tl.sum(probabilities,1)
    return acc,denom,new_max


@triton.jit(do_not_specialize=['P'])
def _fork(Q,K,V,PK,PV,CU,OUT,P,BM:tl.constexpr,BN:tl.constexpr,D:tl.constexpr):
    head_batch=tl.program_id(1);batch=head_batch//16;head=head_batch%16;kv=head//2
    begin=tl.load(CU+batch);end=tl.load(CU+batch+1);length=end-begin
    m=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.arange(0,BN);d=tl.arange(0,D)
    # Tile padding is masked; OUT and all tokenwise tensors have exactly real T rows.
    if tl.program_id(0)*BM>=length:return
    q=tl.load(Q+((begin+m[:,None])*16+head)*D+d[None,:],m[:,None]<length,0)
    maximum=tl.full((BM,),-float('inf'),tl.float32)
    denom=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,D),tl.float32)
    for start in range(0,tl.cdiv(P,BN)):
        p=start*BN+n
        k=tl.load(PK+(p[None,:]*8+kv)*D+d[:,None],p[None,:]<P,0)
        v=tl.load(PV+(p[:,None]*8+kv)*D+d[None,:],p[:,None]<P,0)
        acc,denom,maximum=_online(q,k,v,p[None,:]<P,acc,denom,maximum)
    for start in range(0,tl.cdiv(length,BN)):
        p=start*BN+n
        k=tl.load(K+((begin+p[None,:])*8+kv)*D+d[:,None],p[None,:]<length,0)
        v=tl.load(V+((begin+p[:,None])*8+kv)*D+d[None,:],p[:,None]<length,0)
        allowed=(p[None,:]<length)&(p[None,:]<=m[:,None])
        acc,denom,maximum=_online(q,k,v,allowed,acc,denom,maximum)
    tl.store(OUT+((begin+m[:,None])*16+head)*D+d[None,:],acc/denom[:,None],m[:,None]<length)


def fork_attention(q,k,v,prefix_k,prefix_v,cu,max_length):
    if torch.is_grad_enabled():raise RuntimeError('Packed fork is forward-only')
    if (q.ndim!=3 or q.shape[1:]!=(16,128) or k.shape!=(q.shape[0],8,128)
        or v.shape!=k.shape or prefix_k.shape!=prefix_v.shape or prefix_k.shape[1:]!=(8,128)
        or prefix_k.shape[0]<1 or q.dtype!=torch.bfloat16):
        raise ValueError('Only admitted real-token Qwen3 GQA layout supported')
    if any(not t.is_contiguous() for t in (q,k,v,prefix_k,prefix_v,cu)):
        raise ValueError('Contiguous packed storage required')
    out=torch.empty_like(q)
    _fork[(triton.cdiv(max_length,16),(cu.numel()-1)*16)](q,k,v,prefix_k,prefix_v,cu,out,
        prefix_k.shape[0],16,64,128,num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out
