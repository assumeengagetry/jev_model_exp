"""Same-native-kernel packed migration against completed exact-commit artifacts.

Only real frozen original training data are executed. This is not clean quality
training and never creates or combines official public predictions.
"""
import argparse
import base64
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ.get('PACKED_MASK_ROOT','/tmp/opencode/jev-packed-mask-20261112-v1'))
PF=Path('/tmp/opencode/jev-padding-free-20261112-v3')
COMMIT=Path('/tmp/opencode/jev-attention-2e334fb-20261112-v1')
OLD=Path('/tmp/opencode/jev-joint-generic-continuation-20261109')
R1=Path('/tmp/opencode/jev-06b-research-20261112-r1-bridge')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('causal','bidirectional','prefix_block'),required=True)
    args=parser.parse_args()
    sys.path[:0]=[str(ROOT/'code'),str(PF/'code'),str(OLD/'code'),'/home/dev/assumeengage/scripts',str(R1/'code')]
    from pf_common import atomic,read,metadata,MODEL,BASE
    actual={str(p.relative_to(ROOT/'code')):base64.b64encode(p.read_bytes()).decode() for p in sorted((ROOT/'code').rglob('*.py'))}
    assert actual==read(ROOT/'CODE_BINDING.json'),'Packed mask source changed'
    dependencies=read(ROOT/'DEPENDENCIES.json')
    for value in dependencies['files']:assert metadata(value['path'])==value,'External dependency changed'
    assert not (ROOT/'NO_AUTOMATIC_RESUME').exists(),'Packed mask stopped'
    output=ROOT/'runs'/args.mode;output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from packed_mask import MaskedBackbone,MaskedRuntime,make_batch,bind_mask
    from packed_model import load_model,RowTrace
    from joint_data import PairPool,TokenStore
    from group_training_runtime import load_plan,atomic_microbatches
    from train_bridge import group_micro_loss
    from student_flash_attention3_repro import attention_record
    spec=importlib.util.spec_from_file_location('exact_commit_pair_runtime',COMMIT/'commit-code/patched/pair_training_runtime.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    cfg=read(R1/'train/config.json');source=read(COMMIT/'runs'/args.mode/'INPUT.json')
    old_done=read(COMMIT/'runs'/args.mode/'COMPLETE.json')
    assert old_done['execution_complete'] and old_done['pairs']==128 and old_done['mode']==args.mode
    # Cross-backend acceptance was false; it is not rewritten here. Reuse only
    # the source-bound actual FA3 tensors as this same-kernel reference.
    reference=torch.load(COMMIT/'runs'/args.mode/'fa3.pt',map_location='cpu',weights_only=True)
    ref_metrics=read(COMMIT/'runs'/args.mode/'METRICS.json')['fa3']
    plan=load_plan(cfg['group_plan_root'],cfg['group_plan_identity']);macro=int(source['macro'])
    a,b=map(int,plan['macro_offsets'][macro:macro+2]);indices=np.asarray(source['indices'],dtype=np.int64)
    tags=np.asarray(source['tags'],dtype=np.int32)
    assert b-a==len(indices)==128 and np.array_equal(indices,plan['order'][a:b]) and np.array_equal(tags,plan['relation_tag'][a:b])
    assert np.any((tags>=0)&(tags%2==0))
    dataset=PairPool(root=cfg['training_pool'],domains=cfg['active_training_domains']);store=TokenStore()
    batch=dataset.fetch(indices);batch['relation_tag']=tags
    effective=np.load(cfg['effective_target_path'],mmap_mode='r');batch['distillation_target']=np.asarray(effective[indices]).copy()
    micros=list(atomic_microbatches(batch,cfg['max_microbatch_pairs'],cfg['max_microbatch_tokens']))
    assembled=[[original.assemble(store,s,c,int(l),151645,boundary=True)
        for s,c,l in zip(m['state'],m['candidate'],m['max_length'])] for m in micros]
    rows_by_micro=[[ids.tolist() for ids,_ in rows] for rows in assembled]
    prefixes_by_micro=[[int(p) for _,p in rows] for rows in assembled]
    frozen=read('/tmp/opencode/jev-padding-free-20261112-v2/mechanism/INPUT.json')
    assert [list(map(len,rows)) for rows in rows_by_micro]==frozen['lengths']
    torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda:0');model,backbone,head,native=load_model(MODEL,BASE,device,trainable=True)
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    initial={n:p.detach().cpu().clone() for n,p in params}
    assert len(params)==393 and set(initial)==set(reference['gradients'])==set(reference['updates'])
    core=MaskedBackbone(backbone,head,native,args.mode)
    optimizer=torch.optim.AdamW([p for _,p in params],lr=2e-5,weight_decay=.01,fused=True)
    logits_all=[];losses=[];traces=[];head_shapes=[]
    handle=head.register_forward_pre_hook(lambda module,inputs:head_shapes.append(list(inputs[0].shape)))
    before=attention_record();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    for micro,rows,prefixes in zip(micros,rows_by_micro,prefixes_by_micro):
        assert not (ROOT/'NO_AUTOMATIC_RESUME').exists(),'Packed mask stop requested'
        layout=make_batch(rows,device,'packed',151643);bound=bind_mask(layout,prefixes,args.mode)
        with RowTrace(backbone,layout) as trace,torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_=core(layout,bound);margin=logits[:,1].float()-logits[:,0].float()
            loss,_,_=group_micro_loss(torch,F,np,margin,micro,plan['teacher_relation_target'],cfg,device)
            (loss/len(indices)).backward()
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        logits_all.append(logits.detach().float().cpu());losses.append(float(loss.detach())/len(indices));traces.append(trace.result())
    handle.remove()
    assert all(shape==[sum(map(len,rows)),1024] for shape,rows in zip(head_shapes,rows_by_micro))
    gradients={n:p.grad.detach().cpu().clone() for n,p in params}
    norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,foreach=True));optimizer.step();torch.cuda.synchronize()
    updates={n:p.detach().cpu()-initial[n] for n,p in params};logits=torch.cat(logits_all)
    elapsed=time.monotonic()-started;after=attention_record()
    torch.save({'logits':logits,'gradients':gradients,'updates':updates,'optimizer':optimizer.state_dict()},output/'packed.pt')
    def metric(ref,value,kind):
        ref=ref.double();value=value.double();delta=value-ref;norm=float(ref.norm());rms=float(ref.square().mean().sqrt())
        error=float(delta.norm());cos=None
        if rms>1e-7:cos=float(F.cosine_similarity(ref.reshape(1,-1),value.reshape(1,-1),dim=1))
        if kind=='logits':ok=float(delta.square().mean().sqrt())<=.01*max(rms,1e-3) and float(delta.abs().max())<=.02+.02*float(ref.abs().max())
        else:ok=error<=.02*norm+1e-7*ref.numel()**.5 and (cos is None or cos>=.999)
        return {'pass':ok,'relative_l2':error/max(norm,1e-30),'max_abs':float(delta.abs().max()),'cosine':cos,'reference_rms':rms}
    result={'mode':args.mode,'reference':'actual exact-commit HF + native FA3, not SDPA',
        'old_cross_backend_acceptance':old_done['numerical_acceptance'],'loss':sum(losses),
        'reference_loss':ref_metrics['loss'],'loss_pass':abs(sum(losses)-ref_metrics['loss'])<=1e-3*(1+abs(ref_metrics['loss'])),
        'logits':metric(reference['logits'],logits,'logits'),'seconds':elapsed,
        'peak_memory_bytes':torch.cuda.max_memory_allocated(),'clip_input_norm':norm,
        'trace':traces,'head_input_shapes':head_shapes,'attention_before':before,'attention_after':after}
    for kind,values in [('gradients',gradients),('updates',updates)]:
        result[kind]={n:metric(reference[kind][n],v,kind) for n,v in values.items()}
    expected=28*len(micros)*(2 if args.mode=='prefix_block' else 1)
    result['native_call_count_pass']=after['fa3_forward_calls']-before['fa3_forward_calls']==after['fa3_backward_calls']-before['fa3_backward_calls']==expected
    result['training_layout_pass']=result['loss_pass'] and result['logits']['pass'] and result['native_call_count_pass'] and all(v['pass'] for kind in ('gradients','updates') for v in result[kind].values())
    atomic(output/'TRAINING.json',result)
    del optimizer,reference,gradients,updates
    for n,p in params:p.data.copy_(initial[n].to(device));p.grad=None
    core.invalidate_prefixes();model.eval();model.requires_grad_(False);core.eval()
    inference_traces=[]
    def inferred(layout,prefixes,**options):
        shapes=[]
        h=head.register_forward_pre_hook(lambda module,inputs:shapes.append(list(inputs[0].shape)))
        try:
            with RowTrace(backbone,layout) as trace:
                value=core(layout,prefixes,**options)
            assert shapes==[[layout.ids.numel(),1024]],'Inference head processed non-real rows'
            inference_traces.append({'trace':trace.result(),'head_input_shapes':shapes,'capture_prefix':bool(options.get('capture_prefix')),'uses_prefix':options.get('prefix') is not None})
            return value
        finally:h.remove()
    all_rows=[row for rows in rows_by_micro for row in rows]
    all_prefixes=[p for values in prefixes_by_micro for p in values]
    groups={}
    for i,(row,p) in enumerate(zip(all_rows,all_prefixes)):groups.setdefault(tuple(row[:p]),[]).append(i)
    candidates=[(p,ids) for p,ids in groups.items() if len(ids)>=2]
    assert candidates,'Actual macro has no multi-candidate prefix coverage; do not invent synthetic rows'
    prefix,chosen=max(candidates,key=lambda x:len(x[1]));suffixes=[all_rows[i][all_prefixes[i]:] for i in chosen]
    selected_rows=[all_rows[i] for i in chosen];p_lengths=[len(prefix)]*len(chosen)
    neighbor=next((i for i in range(len(all_rows)) if i not in chosen),None)
    assert neighbor is not None,'Actual macro has no independent replacement neighbor; coverage incomplete'
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        one,_=inferred(make_batch(selected_rows[:1],device),p_lengths[:1])
        mixed,_=inferred(make_batch(selected_rows,device),p_lengths)
        rev,_=inferred(make_batch(list(reversed(selected_rows)),device),p_lengths)
        replaced,_=inferred(make_batch([selected_rows[0],all_rows[neighbor]],device),[len(prefix),all_prefixes[neighbor]])
    isolation={'mixed':metric(one.cpu(),mixed[:1].cpu(),'logits'),
        'reversed':metric(one.cpu(),rev[-1:].cpu(),'logits'),
        'replaced_neighbor':metric(one.cpu(),replaced[:1].cpu(),'logits')}
    atomic(output/'ISOLATION.json',isolation)
    runtime=MaskedRuntime(core,151643,151645,40960,device)
    prefix_result={'supported':args.mode!='bidirectional','prefix_length':len(prefix),'candidate_count':len(suffixes),
        'suffix_lengths':list(map(len,suffixes)),'numerical_contract':'raw two-class logits, same frozen contract'}
    if args.mode!='bidirectional':
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            _,cache=inferred(make_batch([prefix],device),[len(prefix)],capture_prefix=True)
            snapshot=[(k.clone(),v.clone()) for k,v in cache.layers]
            full,_=inferred(make_batch(selected_rows,device),p_lengths)
            branch,_=inferred(make_batch(suffixes,device,prefix_length=len(prefix)),p_lengths,prefix=cache)
            reversed_logits,_=inferred(make_batch(list(reversed(suffixes)),device,prefix_length=len(prefix)),p_lengths,prefix=cache)
            unchanged=all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(cache.layers,snapshot))
            full_margin=full[:,1].float()-full[:,0].float()
            branch_margin=branch[:,1].float()-branch[:,0].float()
        prefix_result.update(raw_full_vs_fork=metric(full.cpu(),branch.cpu(),'logits'),
            raw_reverse=metric(branch.cpu(),reversed_logits.flip(0).cpu(),'logits'),prefix_immutable=unchanged,
            margin_full_vs_fork=metric(full_margin.cpu(),branch_margin.cpu(),'logits'))
        # Exercise the actual request-owner adapter separately from core logits.
        with runtime.request():
            runtime.prefill(prefix,rendered_prefix_length=len(prefix))
            first=runtime.branched(prefix,suffixes,rendered_prefix_length=len(prefix)).cpu()
            other=next((p for p in groups if p!=prefix),None)
            assert other is not None,'Actual macro has no distinct B owner; A-B-A coverage incomplete'
            runtime.prefill(other,rendered_prefix_length=len(other))
            oi=groups[other][0]
            runtime.branched(other,[all_rows[oi][all_prefixes[oi]:]],rendered_prefix_length=len(other))
            runtime.prefill(prefix,rendered_prefix_length=len(prefix))
            repeated=runtime.branched(prefix,suffixes,rendered_prefix_length=len(prefix)).cpu()
        prefix_result['A_B_A_repeat_exact']=torch.equal(first,repeated)
        prefix_result['B_owner_executed']=True
        prefix_result['runtime_margin_vs_full']=metric(full_margin.cpu(),first,'logits')
        prefix_ok=all(prefix_result[k]['pass'] for k in ('raw_full_vs_fork','raw_reverse','margin_full_vs_fork','runtime_margin_vs_full')) and unchanged and prefix_result['A_B_A_repeat_exact'] and prefix_result['B_owner_executed']
    else:
        with runtime.request():
            rejected=False
            try:runtime.prefill(prefix,rendered_prefix_length=len(prefix))
            except ValueError:rejected=True
        prefix_result['cache_rejected']=rejected;prefix_ok=rejected
    atomic(output/'PREFIX.json',prefix_result)
    atomic(output/'INFERENCE_TRACE.json',inference_traces)
    for value in dependencies['files']:assert metadata(value['path'])==value,'External dependency changed during run'
    accepted=result['training_layout_pass'] and all(v['pass'] for v in isolation.values()) and prefix_ok
    atomic(output/'COMPLETE.json',{'complete':accepted,'mode':args.mode,'actual_macro':macro,'pairs':128,
        'real_tokens':sum(map(len,all_rows)),'training_layout_pass':result['training_layout_pass'],
        'isolation_pass':all(v['pass'] for v in isolation.values()),'prefix_pass':prefix_ok,
        'four_rank_training_verified':False,'new_public_score':None,'clean_training':False})
    print(__import__('json').dumps(read(output/'COMPLETE.json'),indent=2),flush=True)
    assert accepted,'Packed mask mechanism not admitted; preserved raw tensors and frozen comparison'


if __name__=='__main__':main()
