#!/usr/bin/env python3
"""Finalize real pool lengths/exposure records after complete shared tokenization."""
import json
from pathlib import Path

import numpy as np
from pair_training_runtime import TokenStore, POOL
from pair_data_common import DOMAINS

store=TokenStore();manifest=store.manifest
domains={}
for domain in DOMAINS:
    state=np.load(POOL/'pairs'/f'{domain}_state.npy',mmap_mode='r')
    candidate=np.load(POOL/'pairs'/f'{domain}_candidate.npy',mmap_mode='r')
    total_raw=0;over4096=0;over8192=0;maximum=0
    for start in range(0,len(state),1000000):
        raw=store.lengths['prefix'][state[start:start+1000000]].astype(np.uint64)+store.lengths['suffix'][candidate[start:start+1000000]]+1
        total_raw+=int(raw.sum());over4096+=int((raw>4096).sum());over8192+=int((raw>8192).sum())
        if len(raw):maximum=max(maximum,int(raw.max()))
    # The approved plan permits increasing context for long-document-heavy specialists.
    limit=8192 if len(state) and over4096/len(state)>0.10 else 4096
    path=POOL/'pairs'/f'{domain}_length.npy'
    lengths=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint32,shape=(len(state),))
    actual_tokens=0;clipped=0;long_candidates=0
    for start in range(0,len(state),1000000):
        p=store.lengths['prefix'][state[start:start+1000000]].astype(np.uint64)
        c=store.lengths['suffix'][candidate[start:start+1000000]].astype(np.uint64)
        raw=p+c+1;usable=np.minimum(raw,limit).astype(np.uint32)
        lengths[start:start+len(usable)]=usable;actual_tokens+=int(usable.sum());clipped+=int((raw>limit).sum())
        long_candidates+=int((c>limit-1-np.minimum(p,min(1024,limit//2))).sum())
    lengths.flush()
    domains[domain]=dict(pairs=len(state),max_length=limit,effective_training_tokens=actual_tokens,
                         original_unclipped_tokens=total_raw,over4096=over4096,over8192=over8192,
                         long_pairs_head_tail_clipped=clipped,long_candidates_head_tail_clipped=long_candidates,
                         maximum_original_pair_length=maximum,dropped_for_length=0)
    print('Ready for actual full epoch:',domain,domains[domain],flush=True)
ready=dict(complete=True,domains=domains,mixed_all_pairs=sum(row['pairs'] for row in domains.values()),
           mixed_all_effective_tokens=sum(row['effective_training_tokens'] for row in domains.values()),
           mixed_exposure='one effective update exposure for every pair in the exact five-domain union',
           token_cap=None,pair_cap=None,extra_tests=False,extra_file_hash_checks=False)
(POOL/'TRAINING_READY.json').write_text(json.dumps(ready,indent=2))
print('Complete shared training pool and fair MIXED_ALL exposure finalized.',flush=True)
