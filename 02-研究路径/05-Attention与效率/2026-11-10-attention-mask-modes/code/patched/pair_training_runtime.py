"""Shared compact token cache and full-exposure candidate-classifier data runtime."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from pair_data_common import DOMAINS

HOME = Path('/home/dev/assumeengage')
POOL = HOME / 'cache/pair-training/global-pool'


class TokenStore:
    def __init__(self, root=POOL):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'MANIFEST.json').read_text())
        if not self.manifest.get('complete'): raise RuntimeError('Global pool is incomplete')
        typed = Path(self.manifest['typed_tokens']); new = Path(self.manifest['new_tokens'])
        for folder in [typed, new]:
            if not json.loads((folder / 'MANIFEST.json').read_text()).get('complete'):
                raise RuntimeError('Shared token cache is not complete: ' + str(folder))
        self.data = {}
        self.boundaries = {'prefix': self.manifest['typed_prefix_boundary'],
                           'suffix': self.manifest['typed_suffix_boundary']}
        self.lengths = {}
        for role in ['prefix','suffix']:
            streams=[]; lengths=[]
            for folder in [typed,new]:
                ids=np.memmap(folder/(role+'_ids.bin'),mode='r',dtype=np.int32)
                offsets=np.memmap(folder/(role+'_offsets.bin'),mode='r',dtype=np.uint64)
                streams.append((ids,offsets));lengths.append(np.diff(offsets).astype(np.uint32))
            self.data[role]=streams;self.lengths[role]=np.concatenate(lengths)
    def get(self, role, index):
        boundary=self.boundaries[role];stream=0 if index<boundary else 1
        local=index if stream==0 else index-boundary;ids,offsets=self.data[role][stream]
        return ids[int(offsets[local]):int(offsets[local+1])]


class PairPool:
    def __init__(self, name, root=POOL, domains=None):
        self.root=Path(root);self.name=name
        active=DOMAINS if domains is None else list(domains)
        if not active or len(set(active))!=len(active) or any(domain not in DOMAINS for domain in active):
            raise ValueError('Active training domains must be unique, known and nonempty')
        self.domains=active if name=='MIXED_ALL' else [name.replace('SPECIALIST_','').lower()]
        self.tables={};self.ends=[];total=0
        for domain in self.domains:
            arrays={key:np.load(self.root/'pairs'/f'{domain}_{key}.npy',mmap_mode='r')
                    for key in ['state','candidate','target','source','group']}
            arrays['length']=np.load(self.root/'pairs'/f'{domain}_length.npy',mmap_mode='r')
            self.tables[domain]=arrays;total+=len(arrays['state']);self.ends.append(total)
        self.ends=np.array(self.ends,dtype=np.int64);self.size=total
        self.ready=json.loads((self.root/'TRAINING_READY.json').read_text())
    def __len__(self):return self.size
    def fetch(self, indices):
        indices=np.asarray(indices,dtype=np.int64);which=np.searchsorted(self.ends,indices,side='right')
        result={key:np.empty(len(indices),dtype=np.float32 if key=='target' else np.uint32)
                for key in ['state','candidate','target','source','group','length']}
        result['domain']=np.empty(len(indices),dtype=np.uint8)
        result['max_length']=np.empty(len(indices),dtype=np.uint32)
        for d,domain in enumerate(self.domains):
            mask=which==d
            if not mask.any():continue
            local=indices[mask]-(int(self.ends[d-1]) if d else 0)
            for key in result:
                if key not in ['domain','max_length']:result[key][mask]=self.tables[domain][key][local]
            result['domain'][mask]=DOMAINS.index(domain)
            result['max_length'][mask]=self.ready['domains'][domain]['max_length']
        return result
    def order(self,seed,bucket_size=8192,batch_size=128):
        rng=np.random.default_rng(seed);order=rng.permutation(self.size).astype(np.uint32)
        for start in range(0,self.size,bucket_size):
            block=order[start:start+bucket_size].copy()
            lengths=self.fetch(block)['length'];block=block[np.argsort(lengths,kind='stable')]
            full=len(block)//batch_size
            if full:
                groups=block[:full*batch_size].reshape(full,batch_size)
                groups=groups[rng.permutation(full)]
                block[:full*batch_size]=groups.reshape(-1)
            order[start:start+len(block)]=block
        return order


def clip_tokens(ids,budget):
    budget=max(0,int(budget))
    if len(ids)<=budget:return ids
    if not budget:return np.empty(0,dtype=np.int32)
    head=(budget+1)//2;tail=budget-head
    return np.concatenate((ids[:head],ids[-tail:] if tail else np.empty(0,dtype=np.int32)))


def assemble(store,state,candidate,max_length,eos,boundary=False):
    prefix=store.get('prefix',int(state));suffix=store.get('suffix',int(candidate))
    prefix_reserve=min(len(prefix),min(1024,max_length//2))
    suffix=clip_tokens(suffix,max_length-1-prefix_reserve)
    prefix=clip_tokens(prefix,max_length-1-len(suffix))
    ids=np.concatenate((prefix,suffix,np.array([eos],dtype=np.int32)))
    return (ids,len(prefix)) if boundary else ids


def collate(store,batch,pad,eos,device,layout=False):
    rows=[assemble(store,state,candidate,int(limit),eos,boundary=True)
          for state,candidate,limit in zip(batch['state'],batch['candidate'],batch['max_length'])]
    ids=[row for row,_ in rows]
    width=((max(len(row) for row in ids)+7)//8)*8
    input_ids=torch.full((len(ids),width),pad,dtype=torch.long,pin_memory=True)
    mask=torch.zeros((len(ids),width),dtype=torch.bool,pin_memory=True)
    for i,row in enumerate(ids):
        input_ids[i,:len(row)]=torch.from_numpy(row.astype(np.int64,copy=False));mask[i,:len(row)]=True
    targets=torch.from_numpy(np.asarray(batch['target'],dtype=np.float32).copy()).pin_memory()
    result=({'input_ids':input_ids.to(device,non_blocking=True),'attention_mask':mask.to(device,non_blocking=True)},targets.to(device,non_blocking=True),sum(map(len,ids)))
    if not layout:return result
    lengths=lambda values:torch.tensor(values,dtype=torch.int32,device=device)
    return (*result,(lengths([prefix for _,prefix in rows]),lengths([len(row) for row in ids])))


def microbatches(batch,max_pairs,max_tokens):
    start=0
    while start<len(batch['state']):
        end=start;longest=0
        while end<len(batch['state']) and end-start<max_pairs:
            proposed=max(longest,int(batch['length'][end]))
            if end>start and proposed*(end-start+1)>max_tokens:break
            longest=proposed;end+=1
        yield {key:value[start:end] for key,value in batch.items()}
        start=end
