"""Exact packed causal mask with block metadata, never a token-square matrix."""
import bisect
import itertools
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention


class FlexBackend:
    def __init__(self):
        self.compiled=torch.compile(flex_attention,fullgraph=True,dynamic=True)
        self.block=None;self.cu_expected=None;self.lengths=None
        self.forward_calls=0;self.backward_calls=0

    def bind(self,lengths,device):
        lengths=tuple(map(int,lengths))
        if not lengths or min(lengths)<1:raise ValueError('Nonempty real sequences required')
        offsets=list(itertools.accumulate((0,*lengths)));total=offsets[-1];bs=128
        blocks=(total+bs-1)//bs
        owners=torch.repeat_interleave(torch.arange(len(lengths),device=device,dtype=torch.int32),
            torch.tensor(lengths,device=device),output_size=total)
        def mask(b,h,q_idx,kv_idx):
            # Flex may evaluate lanes in the final hardware tile. Clamp only
            # the metadata access, then explicitly reject out-of-range lanes.
            q_safe=q_idx.clamp(0,total-1);k_safe=kv_idx.clamp(0,total-1)
            return (q_idx<total)&(kv_idx<total)&(owners[q_safe]==owners[k_safe])&(q_idx>=kv_idx)
        partial=[];full=[]
        for tile in range(blocks):
            qa=tile*bs;qb=min(total,qa+bs)
            first=bisect.bisect_right(offsets,qa)-1
            last=bisect.bisect_left(offsets,qb)-1
            keys=set()
            for sequence in range(first,last+1):
                sa=offsets[sequence];sb=offsets[sequence+1]
                # Any query in this tile may attend through the last real
                # query of that sequence; the fine mask decides each lane.
                end=min(qb,sb)
                keys.update(range(sa//bs,(end+bs-1)//bs))
            p=[];f=[]
            for key in sorted(keys):
                ka=key*bs;kb=min(total,ka+bs)
                if (first==last and qa>=offsets[first] and qb<=offsets[first+1]
                    and ka>=offsets[first] and kb<=offsets[first+1] and kb-1<=qa):
                    f.append(key)
                else:p.append(key)
            partial.append(p);full.append(f)
        def directory(rows):
            counts=torch.tensor([len(r) for r in rows],dtype=torch.int32,device=device)[None,None,:]
            # Block-index capacity is metadata only; no hidden/QKV dummy rows.
            # 2.9.1's generated transpose uses this axis as the KV block
            # universe, not just max active entries per row.
            width=blocks
            indices=torch.tensor([r+[0]*(width-len(r)) for r in rows],dtype=torch.int32,device=device)[None,None,:,:]
            return counts,indices
        pn,pi=directory(partial);fn,fi=directory(full)
        self.block=BlockMask.from_kv_blocks(pn,pi,fn,fi,BLOCK_SIZE=bs,mask_mod=mask,seq_lengths=(total,total))
        self.cu_expected=torch.tensor(offsets,dtype=torch.int32,device=device)
        self.lengths=lengths
        self.metadata={'total_real_tokens':total,'sequences':len(lengths),'block_size':bs,
            'partial_blocks':sum(map(len,partial)),'full_blocks':sum(map(len,full)),
            'materialized_token_square_mask':False,'gqa_repeat_kv':False}

    def flash_attn_varlen_func(self,q,k,v,cu_q,cu_k,max_q,max_k,softmax_scale=None,causal=False,deterministic=False):
        if (self.block is None or not causal or not deterministic or not torch.are_deterministic_algorithms_enabled()
            or q.shape!=(sum(self.lengths),16,128) or k.shape!=(q.shape[0],8,128) or v.shape!=k.shape
            or max_q!=max(self.lengths) or max_k!=max_q):
            raise ValueError('Exact admitted real-token causal/GQA/deterministic contract required')
        if not torch.equal(cu_q,self.cu_expected) or not torch.equal(cu_k,self.cu_expected):
            raise ValueError('Bound mask belongs to different sequence offsets')
        self.forward_calls+=1
        if q.requires_grad:
            def observed_backward(grad):self.backward_calls+=1;return grad
            q.register_hook(observed_backward)
        out=self.compiled(q.transpose(0,1).unsqueeze(0),k.transpose(0,1).unsqueeze(0),
            v.transpose(0,1).unsqueeze(0),block_mask=self.block,scale=softmax_scale,
            enable_gqa=True,kernel_options={'PRESCALE_QK':False})
        return out.squeeze(0).transpose(0,1).contiguous()
