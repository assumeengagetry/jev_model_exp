"""Exact commit attention modes, original real macro, actual native FA3 versus SDPA."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('causal','bidirectional','prefix_block'),required=True)
    args=parser.parse_args();root=Path(os.environ['ATTN_ROOT'])
    assert root.name.startswith('jev-attention-2e334fb-20261112-')
    output=root/'runs'/args.mode;output.mkdir(parents=True,exist_ok=False)
    actual={str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for folder in ('code','commit-code') for p in sorted((root/folder).rglob('*.py'))}
    assert actual==json.loads((root/'CODE_BINDING.json').read_text()),'Frozen commit/harness changed'
    sys.path[:0]=[str(root/'commit-code'),str(root/'commit-code/patched'),
        '/tmp/opencode/jev-padding-free-20261112-v1/code',
        '/tmp/opencode/jev-joint-generic-continuation-20261109/code',
        '/home/dev/assumeengage/scripts','/tmp/opencode/jev-06b-research-20261112-r1-bridge/code']
    # Import exact collate before joint_data's read-only path insertion.
    import importlib.util
    spec=importlib.util.spec_from_file_location('commit_pair_runtime',root/'commit-code/patched/pair_training_runtime.py')
    patched=importlib.util.module_from_spec(spec);spec.loader.exec_module(patched)
    from attention_modes import attention_layout,enable_bidirectional,install_prefix_block_flash,dense_mask,attention_layers
    from pf_common import atomic,read,metadata,MODEL,BASE
    dependencies=read(root/'DEPENDENCIES.json')
    for record in dependencies['files']:assert metadata(record['path'])==record,'Bound external dependency changed'
    from packed_model import load_model
    import numpy as np
    import torch
    from torch.nn import functional as F
    from joint_data import PairPool,TokenStore
    from group_training_runtime import load_plan,atomic_microbatches
    from train_bridge import group_micro_loss
    from student_flash_attention3_repro import BACKEND,attention_record,assert_student_uses_fa3
    cfg=read('/tmp/opencode/jev-06b-research-20261112-r1-bridge/train/config.json')
    source=read('/tmp/opencode/jev-padding-free-20261112-v1/mechanism/INPUT.json')
    dataset=PairPool(root=cfg['training_pool'],domains=cfg['active_training_domains']);store=TokenStore()
    plan=load_plan(cfg['group_plan_root'],cfg['group_plan_identity']);indices=np.asarray(source['indices'],dtype=np.int64)
    macro=int(source['macro']);a,b=map(int,plan['macro_offsets'][macro:macro+2])
    assert b-a==128 and len(indices)==128 and np.array_equal(indices,plan['order'][a:b]),'Not the frozen full original macro'
    tags=np.asarray(source['tags'],dtype=np.int32)
    assert np.array_equal(tags,plan['relation_tag'][a:b]) and np.any((tags>=0)&(tags%2==0)),'Relation membership differs'
    batch=dataset.fetch(indices);batch['relation_tag']=tags
    targets=np.load(cfg['effective_target_path'],mmap_mode='r');batch['distillation_target']=np.asarray(targets[indices]).copy()
    micros=list(atomic_microbatches(batch,cfg['max_microbatch_pairs'],cfg['max_microbatch_tokens']))
    torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True;device=torch.device('cuda:0')
    model,backbone,head,native=load_model(MODEL,BASE,device,trainable=True)
    assert_student_uses_fa3(model)
    if args.mode=='bidirectional':enable_bidirectional(model)
    elif args.mode=='prefix_block':install_prefix_block_flash(BACKEND)
    model.config.pair_attention_mode=args.mode
    assert all(m.config is model.config for m in attention_layers(model)), 'Attention config alias mismatch'
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    initial={n:p.detach().cpu().clone() for n,p in parameters};assert len(initial)==393
    atomic(output/'INPUT.json',{'commit':'2e334fb76c4cd4826df37424450b2734101db9f8','mode':args.mode,
        'macro':source['macro'],'indices':source['indices'],'tags':source['tags'],
        'diagnostic_model':str(MODEL),'model_files':[metadata(p) for p in sorted(MODEL.iterdir()) if p.is_file()],
        'scope':'real historical checkpoint mechanism only; no clean training quality claim',
        'padding_free':False,'boundary':'exact commit, after Candidate newline'})
    results={};values={}
    for backend in ('fa3','sdpa'):
        for n,p in parameters:p.data.copy_(initial[n].to(device));p.grad=None
        model.config._attn_implementation=BACKEND if backend=='fa3' else 'sdpa'
        if backend=='fa3':assert_student_uses_fa3(model)
        optimizer=torch.optim.AdamW([p for _,p in parameters],lr=2e-5,weight_decay=.01,fused=True)
        logits_all=[];losses=[];layouts=[];before=attention_record()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        for micro_index,micro in enumerate(micros):
            assert not (root/'NO_AUTOMATIC_RESUME').exists(),'User stop requested'
            inputs,target,tokens,layout=patched.collate(store,micro,151643,151645,device,layout=True)
            prefix_len,row_len=layout
            assert tokens==int(row_len.sum())
            assert row_len.cpu().tolist()==source['lengths'][micro_index],'Patched collate changed actual token lengths'
            if backend=='sdpa' and args.mode!='causal':
                inputs['attention_mask']=dense_mask(args.mode,row_len,inputs['input_ids'].shape[1],prefix_len)
            with attention_layout(*layout):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(**inputs).logits; margin=logits[:,1].float()-logits[:,0].float()
                    loss,_,_=group_micro_loss(torch,F,np,margin,micro,plan['teacher_relation_target'],cfg,device)
                (loss/len(indices)).backward()
            assert torch.isfinite(logits).all() and torch.isfinite(loss),'Nonfinite actual result'
            logits_all.append(logits.detach().float().cpu());losses.append(float(loss.detach())/len(indices))
            layouts.append({'prefix_lengths':prefix_len.cpu().tolist(),'row_lengths':row_len.cpu().tolist(),
                'input_shape':list(inputs['input_ids'].shape),'mask_shape':list(inputs['attention_mask'].shape)})
        gradients={n:p.grad.detach().cpu().clone() for n,p in parameters}
        norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,foreach=True))
        optimizer.step();torch.cuda.synchronize()
        updates={n:p.detach().cpu()-initial[n] for n,p in parameters}
        artifact={'logits':torch.cat(logits_all),'gradients':gradients,'updates':updates,'optimizer':optimizer.state_dict()}
        torch.save(artifact,output/(backend+'.pt'));values[backend]=artifact
        results[backend]={'loss':sum(losses),'seconds':time.monotonic()-started,
            'peak_memory_bytes':torch.cuda.max_memory_allocated(),'clip_input_norm':norm,
            'layouts':layouts,'attention_before':before,'attention_after':attention_record()}
        atomic(output/'METRICS.json',results)
        del optimizer
    def metric(ref,current,kind):
        ref=ref.double();current=current.double();diff=current-ref
        rms=float(ref.square().mean().sqrt());norm=float(ref.norm());error=float(diff.norm());cos=None
        if rms>1e-7:cos=float(F.cosine_similarity(ref.reshape(1,-1),current.reshape(1,-1),dim=1))
        if kind=='logits':ok=float(diff.square().mean().sqrt())<=.01*max(rms,1e-3) and float(diff.abs().max())<=.02+.02*float(ref.abs().max())
        else:ok=error<=.02*norm+1e-7*ref.numel()**.5 and (cos is None or cos>=.999)
        return {'pass':ok,'relative_l2':error/max(norm,1e-30),'max_abs':float(diff.abs().max()),'cosine':cos,'reference_rms':rms}
    comparison={'logits':metric(values['sdpa']['logits'],values['fa3']['logits'],'logits'),
        'loss_pass':abs(results['fa3']['loss']-results['sdpa']['loss'])<=1e-3*(1+abs(results['sdpa']['loss']))}
    for kind in ('gradients','updates'):
        comparison[kind]={n:metric(values['sdpa'][kind][n],v,kind) for n,v in values['fa3'][kind].items()}
    expected=28*len(micros)*(2 if args.mode=='prefix_block' else 1)
    actual_fwd=results['fa3']['attention_after']['fa3_forward_calls']-results['fa3']['attention_before']['fa3_forward_calls']
    actual_bwd=results['fa3']['attention_after']['fa3_backward_calls']-results['fa3']['attention_before']['fa3_backward_calls']
    comparison['expected_native_calls']=expected;comparison['native_forward_calls']=actual_fwd;comparison['native_backward_calls']=actual_bwd
    comparison['native_call_count_pass']=actual_fwd==actual_bwd==expected
    accepted=comparison['logits']['pass'] and comparison['loss_pass'] and comparison['native_call_count_pass'] and all(v['pass'] for kind in ('gradients','updates') for v in comparison[kind].values())
    atomic(output/'COMPARISON.json',comparison)
    for record in dependencies['files']:assert metadata(record['path'])==record,'External dependency changed during run'
    atomic(output/'COMPLETE.json',{'execution_complete':True,'numerical_acceptance':accepted,'mode':args.mode,
        'commit':'2e334fb76c4cd4826df37424450b2734101db9f8','actual_macro':source['macro'],'pairs':len(indices),
        'strict_deterministic_fa3_backward':True,'public_score':None,'training_quality_gain':None,
        'native_forward_calls':actual_fwd,'native_backward_calls':actual_bwd})
    print(json.dumps(read(output/'COMPLETE.json'),indent=2),flush=True)
    assert accepted,'Actual kernel comparison exceeded frozen contract; preserve artifacts, no implicit pass'


if __name__=='__main__':main()
