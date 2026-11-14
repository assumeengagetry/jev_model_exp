"""Real-token metadata; packed construction never materializes a padded matrix."""
from dataclasses import dataclass
import itertools
import torch


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
    def total(self):return sum(self.lengths)

    @property
    def max_length(self):return max(self.lengths)

    def real(self,tensor):
        flat=tensor.reshape(-1,*tensor.shape[self.ids.ndim:])
        return flat if self.layout=='packed' else flat.index_select(0,self.real_indices)

    def restore(self,tensor):
        if self.layout=='packed':return tensor
        result=tensor.new_zeros((self.ids.numel(),*tensor.shape[1:]))
        return result.index_copy(0,self.real_indices,tensor).view(*self.ids.shape,*tensor.shape[1:])


def make_batch(rows,device,layout,pad,prefix_length=0):
    if layout not in ('packed','dense') or not rows or any(not len(r) for r in rows):
        raise ValueError('Nonempty real sequences and packed/dense layout required')
    lengths=tuple(map(len,rows)); offsets=list(itertools.accumulate((0,*lengths)))
    positions=[list(range(prefix_length,prefix_length+n)) for n in lengths]
    cu=torch.tensor(offsets,dtype=torch.int32,device=device)
    if layout=='packed':
        ids=torch.tensor(list(itertools.chain.from_iterable(rows)),dtype=torch.long,device=device)
        pos=torch.tensor(list(itertools.chain.from_iterable(positions)),dtype=torch.long,device=device)
        real=torch.arange(sum(lengths),dtype=torch.long,device=device)
        eos=cu[1:].long()-1;mask=None
    else:
        width=(max(lengths)+7)//8*8
        ids=torch.full((len(rows),width),pad,dtype=torch.long)
        pos=torch.zeros_like(ids);mask=torch.zeros_like(ids)
        index=[]
        for i,(r,p) in enumerate(zip(rows,positions)):
            ids[i,:len(r)]=torch.as_tensor(r);pos[i,:len(r)]=torch.as_tensor(p);mask[i,:len(r)]=1
            index.extend(range(i*width,i*width+len(r)))
        ids=ids.to(device);pos=pos.to(device);mask=mask.to(device)
        real=torch.tensor(index,dtype=torch.long,device=device)
        eos=torch.tensor([i*width+n-1 for i,n in enumerate(lengths)],dtype=torch.long,device=device)
    return TokenBatch(ids,pos,cu,real,eos,lengths,layout,mask)


def real_atomic_microbatches(batch,max_pairs,max_tokens):
    tags=batch['relation_tag'];start=0
    while start<len(tags):
        end=start;tokens=0
        while end<len(tags):
            tag=int(tags[end]);width=1 if tag==-1 else 2
            if tag!=-1 and (tag<0 or tag%2 or end+1>=len(tags) or int(tags[end+1])!=tag+1):
                raise ValueError('Incomplete relationship unit')
            if width>max_pairs:raise ValueError('Whole unit exceeds row limit')
            proposed=tokens+int(batch['length'][end:end+width].sum())
            if end>start and (end-start+width>max_pairs or proposed>max_tokens):break
            tokens=proposed;end+=width
        yield {k:v[start:end] for k,v in batch.items()}
        start=end
