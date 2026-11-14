"""Dynamic real-shape eager runtime; no Graph bucket or dummy business tokens."""
from collections import Counter
import torch
from packed_layout import make_batch


class PackedRuntime:
    def __init__(self,core,pad,eos,context,device,layout):
        self.core=core;self.pad=pad;self.eos=eos;self.context=context;self.device=device;self.layout=layout
        self.prefix=None;self.prefix_length=None;self.counts=Counter()
    def begin_request(self):
        self.prefix=None;self.prefix_length=None
    def finish_request(self):
        self.prefix=None;self.prefix_length=None
    def batch(self,rows,prefix_length=0):
        if any(len(r)+prefix_length>self.context for r in rows):raise ValueError('Context exceeded; no truncation')
        batch=make_batch(rows,self.device,self.layout,self.pad,prefix_length)
        self.counts['actual_tokenwise_rows']+=batch.ids.numel()
        self.counts['real_execution_tokens']+=batch.total
        self.counts['waves']+=1
        return batch
    @torch.inference_mode()
    def prefill(self,prefix):
        if not prefix:raise ValueError('Nonempty prefix required')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,self.prefix=self.core(self.batch([prefix]),capture_prefix=True)
        self.prefix_length=len(prefix)
        self.counts['prefix_owners']+=1
        self.counts['prefix_tokens_once']+=len(prefix)
    @torch.inference_mode()
    def branched(self,prefix_length,rows):
        if self.prefix is None or prefix_length!=self.prefix_length:raise ValueError('Wrong prefix owner')
        if any(r[-1]!=self.eos for r in rows):raise ValueError('Missing actual EOS')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_=self.core(self.batch(rows,prefix_length),prefix=self.prefix)
        self.counts['fork_waves']+=1
        return logits[:,1].float()-logits[:,0].float()
    @torch.inference_mode()
    def full(self,rows):
        if any(r[-1]!=self.eos for r in rows):raise ValueError('Missing actual EOS')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_=self.core(self.batch(rows))
        return logits[:,1].float()-logits[:,0].float()
    def runtime(self):
        return {'layout':self.layout,'execution':'dynamic eager real-shape','cuda_graph_dummy_rows':0,
            'counts':dict(self.counts),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(self.device)}
