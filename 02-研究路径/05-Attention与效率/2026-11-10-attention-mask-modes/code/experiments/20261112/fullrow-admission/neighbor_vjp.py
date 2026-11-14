"""Whole real macro: same-geometry neighbor content sensitivity, no updates.

Extends the actual numeric probe from one observed target to all 128 original
rows and their parameter VJP, keeping the original four micro geometries. This
does NOT regrade old solo/mixed failures or authorize prefix-cache serving.
"""
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ['FULLROW_ROOT'])
MASK=Path('/tmp/opencode/jev-packed-mask-20261112-v2')
PF=Path('/tmp/opencode/jev-padding-free-20261112-v3')
COMMIT=Path('/tmp/opencode/jev-attention-2e334fb-20261112-v1')
PRECISION=Path('/tmp/opencode/jev-padding-precision-20261112-v2')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('causal','bidirectional','prefix_block'),required=True)
    args=parser.parse_args();started=time.monotonic()
    sys.path[:0]=[str(MASK/'code'),str(PF/'code'),'/tmp/opencode/jev-joint-generic-continuation-20261109/code',
        '/home/dev/assumeengage/scripts','/tmp/opencode/jev-06b-research-20261112-r1-bridge/code']
    from pf_common import read,atomic,metadata,MODEL,BASE
    dependencies=read(ROOT/'DEPENDENCIES.json')
    for value in dependencies['files']:assert metadata(value['path'])==value,'Bound external source changed'
    assert read(ROOT/'CODE_BINDING.json')=={p.name:base64.b64encode(p.read_bytes()).decode() for p in (ROOT/'code').glob('*.py')}
    output=ROOT/args.mode;output.mkdir(exist_ok=False)
    import numpy as np
    import torch
    from packed_mask import MaskedBackbone,make_batch
    from packed_model import load_model,RowTrace
    from joint_data import PairPool,TokenStore
    from group_training_runtime import load_plan,atomic_microbatches
    from student_flash_attention3_repro import attention_record
    cfg=read('/tmp/opencode/jev-06b-research-20261112-r1-bridge/train/config.json')
    source=read(COMMIT/'runs'/args.mode/'INPUT.json');fixture=read(PRECISION/'FIXTURE_READY.json')
    plan=load_plan(cfg['group_plan_root'],cfg['group_plan_identity']);macro=source['macro']
    a,b=map(int,plan['macro_offsets'][macro:macro+2]);indices=np.asarray(source['indices'],dtype=np.int64)
    assert b-a==len(indices)==128 and np.array_equal(indices,plan['order'][a:b])
    dataset=PairPool(root=cfg['training_pool'],domains=cfg['active_training_domains']);store=TokenStore()
    batch=dataset.fetch(indices);batch['relation_tag']=np.asarray(source['tags'],dtype=np.int32)
    assert np.array_equal(batch['relation_tag'],plan['relation_tag'][a:b])
    spec=importlib.util.spec_from_file_location('exact_fullrow_collate',COMMIT/'commit-code/patched/pair_training_runtime.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    def assemble(m):
        raw=[original.assemble(store,s,c,int(l),151645,boundary=True) for s,c,l in zip(m['state'],m['candidate'],m['max_length'])]
        return [r.tolist() for r,_ in raw],[int(p) for _,p in raw]
    all_rows,all_prefixes=assemble(batch)
    neighbors=fixture['neighbors_global_positions'];assert all(x<128 for x in neighbors),'Frozen fixture neighbors not in macro0'
    nb,nc=neighbors;assert all_rows[nb]!=all_rows[nc] and len(all_rows[nb])==len(all_rows[nc]) and all_prefixes[nb]==all_prefixes[nc]
    micros=[assemble(m) for m in atomic_microbatches(batch,cfg['max_microbatch_pairs'],cfg['max_microbatch_tokens'])]
    assert len(micros)==4 and sum(len(r) for rows,_ in micros for r in rows)==45216
    torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    # Preserve original actual same-native training flags. A precision change
    # would need its own gradient/update comparison, not an inference-only vote.
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    model,backbone,head,native=load_model(MODEL,BASE,torch.device('cuda:0'),trainable=True)
    core=MaskedBackbone(backbone,head,native,args.mode)
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad];assert len(params)==393
    runs={};reference=None
    for label,neighbor in (('B',nb),('C',nc)):
        for _,p in params:p.grad=None
        outputs=[];traces=[];loss_total=0.;before=attention_record()
        torch.cuda.reset_peak_memory_stats()
        for rows,prefixes in micros:
            assert time.monotonic()-started<900 and not (ROOT/'STOP.json').exists(),'Bounded mechanism stop/deadline'
            count=len(rows);batch_layout=make_batch(rows+[all_rows[neighbor]],'cuda:0')
            with RowTrace(backbone,batch_layout) as trace,torch.autocast('cuda',dtype=torch.bfloat16):
                logits,_=core(batch_layout,prefixes+[all_prefixes[neighbor]])
                # This derivative is a real-data content-sensitivity probe,
                # not a learned objective or pseudo label. The neighbor has
                # zero upstream coefficient; original targets are not opened.
                objective=logits[:count,1].float().sum()/128
                objective.backward()
            assert torch.isfinite(logits).all() and torch.isfinite(objective)
            outputs.append(logits[:count].detach().float().cpu());traces.append(trace.result());loss_total+=float(objective.detach())
        gradients={n:p.grad.detach().cpu().clone() for n,p in params}
        assert all(torch.isfinite(g).all() for g in gradients.values())
        data={'logits':torch.cat(outputs),'gradients':gradients,'objective':loss_total}
        torch.save(data,output/(label+'.pt'))
        after=attention_record();expected=4*28*(2 if args.mode=='prefix_block' else 1)
        assert after['fa3_forward_calls']-before['fa3_forward_calls']==after['fa3_backward_calls']-before['fa3_backward_calls']==expected
        runs[label]={'neighbor_position':neighbor,'micro_count':4,'original_target_rows':128,
            'original_target_tokens':45216,'actual_neighbor_rows_forwarded':4,
            'neighbor_geometry':fixture['neighbor_geometry'],'trace':traces,'objective':loss_total,
            'native_forward_backward_calls_each':expected,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
        atomic(output/'METRICS.json',runs)
        if reference is None:reference=data
        else:
            comparisons={n:{'bitwise_equal':torch.equal(reference['gradients'][n],g),
                'max_abs':float((reference['gradients'][n]-g).abs().max())} for n,g in gradients.items()}
            accepted=torch.equal(reference['logits'],data['logits']) and all(v['bitwise_equal'] for v in comparisons.values())
            atomic(output/'VJP_COMPARISON.json',{'same_geometry_logits_bitwise_equal':torch.equal(reference['logits'],data['logits']),
                'gradient_groups':comparisons,'all_393_gradients_bitwise_equal':all(v['bitwise_equal'] for v in comparisons.values())})
    for value in dependencies['files']:assert metadata(value['path'])==value,'Bound external source changed during run'
    atomic(output/'COMPLETE.json',{'execution_complete':True,'same_geometry_content_isolation_pass':accepted,
        'original_target_rows':128,'original_target_tokens':45216,'mode':args.mode,'all_393_gradients_bitwise_equal':all(v['bitwise_equal'] for v in comparisons.values()),
        'weight_updates':0,'elapsed_seconds':time.monotonic()-started,'fullrow_only':True,
        'old_solo_mixed_failure_regraded':False,'prefix_cache_serving_admitted':False})
    print(json.dumps(read(output/'COMPLETE.json'),indent=2),flush=True)
    assert accepted,'Whole real macro same-geometry content isolation/VJP failed; preserve artifacts'


if __name__=='__main__':main()
