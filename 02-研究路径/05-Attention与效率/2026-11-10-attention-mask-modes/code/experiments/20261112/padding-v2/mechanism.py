"""Actual original-data mechanism contract; no synthetic/unit/smoke corpus."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    missing = [name for name in ('packed_layout', 'packed_model', 'packed_runtime')
               if importlib.util.find_spec(name) is None]
    assert not missing, 'Real padding-free mechanism unavailable: '+', '.join(missing)
    from pf_common import ROOT, OLD, MODEL, BASE, FA3, atomic, read, metadata
    import numpy as np
    import torch
    from torch.nn import functional as F
    from packed_layout import make_batch
    from packed_model import PackedBackbone, load_model, RowTrace, install_inference
    from packed_runtime import PackedRuntime

    output = Path(os.environ.get('PF_MECHANISM_OUTPUT',str(ROOT/'mechanism')))
    output.mkdir(parents=True, exist_ok=False)
    cfg = read('/tmp/opencode/jev-06b-research-20261112-r1-bridge/train/config.json')
    sys.path[:0] = [str(OLD/'code'), '/home/dev/assumeengage/scripts',
                   '/tmp/opencode/jev-06b-research-20261112-r1-bridge/code']
    from joint_data import PairPool, TokenStore
    from pair_training_runtime import assemble
    from group_training_runtime import load_plan, atomic_microbatches
    from train_bridge import group_micro_loss
    from student_flash_attention3_repro import attention_record
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    plan = load_plan(cfg['group_plan_root'], cfg['group_plan_identity'])
    dataset = PairPool(root=cfg['training_pool'], domains=cfg['active_training_domains'])
    store = TokenStore()
    targets = np.load(cfg['effective_target_path'], mmap_mode='r')
    relation_pos = int(np.flatnonzero(plan['relation_tag'] >= 0)[0])
    macro = int(np.searchsorted(plan['macro_offsets'], relation_pos, side='right')-1)
    a, b = map(int, plan['macro_offsets'][macro:macro+2])
    indices = plan['order'][a:b]
    batch = dataset.fetch(indices)
    batch['distillation_target'] = np.asarray(targets[indices]).copy()
    batch['relation_tag'] = np.asarray(plan['relation_tag'][a:b]).copy()
    micros = list(atomic_microbatches(batch, cfg['max_microbatch_pairs'], cfg['max_microbatch_tokens']))
    rows_by_micro = [[assemble(store,s,c,int(l),151645).tolist()
        for s,c,l in zip(m['state'],m['candidate'],m['max_length'])] for m in micros]
    atomic(output/'INPUT.json', {'macro':macro,'indices':indices.tolist(),
        'tags':batch['relation_tag'].tolist(),'lengths':[list(map(len,r)) for r in rows_by_micro],
        'model_files':[metadata(p) for p in sorted(MODEL.iterdir()) if p.is_file()],
        'scope':'historical exposed weights/data; mechanism only, not clean ranking training'})
    model, backbone, head, fa3 = load_model(MODEL, BASE, device, trainable=True)
    core = PackedBackbone(backbone, head, fa3)
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    initial = {n:p.detach().cpu().clone() for n,p in params}
    assert len(initial) == 393
    results = {}
    saved = {}

    def metrics(ref, actual, kind):
        ref=ref.double(); actual=actual.double(); diff=actual-ref
        rms=float(ref.square().mean().sqrt()); norm=float(ref.norm())
        err=float(diff.norm()); cosine=None
        if rms>1e-7:
            cosine=float(F.cosine_similarity(ref.reshape(1,-1),actual.reshape(1,-1),dim=1))
        if kind == 'gradient':
            ok=err <= .02*norm+1e-7*ref.numel()**.5 and (cosine is None or cosine>=.999)
        else:
            ok=float(diff.square().mean().sqrt()) <= .01*max(rms,1e-3) and float(diff.abs().max()) <= .02+.02*float(ref.abs().max())
        return {'pass':ok,'reference_rms':rms,'relative_l2':err/max(norm,1e-30),
                'max_abs':float(diff.abs().max()),'cosine':cosine}

    for mode in ('hf','dense','packed'):
        for n,p in params: p.data.copy_(initial[n].to(device)); p.grad=None
        optimizer=torch.optim.AdamW([p for _,p in params], lr=2e-5, weight_decay=.01, fused=True)
        model.train(); losses=[]; logits_all=[]; traces=[]
        torch.cuda.synchronize(); started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        for micro,rows in zip(micros,rows_by_micro):
            layout=make_batch(rows,device,'dense' if mode=='hf' else mode,151643)
            with RowTrace(backbone,layout) as trace, torch.autocast('cuda',dtype=torch.bfloat16):
                if mode=='hf':
                    logits=model(input_ids=layout.ids,attention_mask=layout.mask,
                        position_ids=layout.positions,use_cache=False).logits
                else:
                    logits,_=core(layout)
                margin=logits[:,1].float()-logits[:,0].float()
                loss,_,_=group_micro_loss(torch,F,np,margin,micro,plan['teacher_relation_target'],cfg,device)
                (loss/len(indices)).backward()
            logits_all.append(logits.detach().cpu()); losses.append(float(loss.detach())/len(indices))
            traces.append(trace.result())
        torch.cuda.synchronize()
        grads={n:p.grad.detach().cpu().clone() for n,p in params}
        grad_norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,foreach=True))
        optimizer.step(); torch.cuda.synchronize()
        delta={n:p.detach().cpu()-initial[n] for n,p in params}
        result={'loss':sum(losses),'seconds':time.monotonic()-started,
            'peak_memory_bytes':torch.cuda.max_memory_allocated(),'clip_input_norm':grad_norm,
            'trace':traces,'attention':attention_record()}
        logits=torch.cat(logits_all).float()
        if mode=='hf': saved={'logits':logits,'gradients':grads,'delta':delta,'loss':sum(losses)}
        else:
            result['logits']=metrics(saved['logits'],logits,'logits')
            result['loss_pass']=abs(sum(losses)-saved['loss'])<=1e-3*(1+abs(saved['loss']))
            result['gradients']={n:metrics(saved['gradients'][n],g,'gradient') for n,g in grads.items()}
            result['updates']={n:metrics(saved['delta'][n],d,'gradient') for n,d in delta.items()}
            result['pass']=result['logits']['pass'] and result['loss_pass'] and all(v['pass'] for v in result['gradients'].values()) and all(v['pass'] for v in result['updates'].values())
        results[mode]=result
        torch.save({'logits':logits,'gradients':grads,'updates':delta,'optimizer':optimizer.state_dict()},output/(mode+'.pt'))
        atomic(output/'TRAINING.json',results)
        del optimizer,grads,delta
    for n,p in params:p.data.copy_(initial[n].to(device));p.grad=None
    model.eval(); model.requires_grad_(False)
    # Real sequence isolation uses complete original assembled sequences, never
    # mutates relationship membership in the actual macro update above.
    rows=rows_by_micro[0]
    assert len(rows)>=3
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        one,_=core(make_batch(rows[:1],device,'packed',151643))
        mixed,_=core(make_batch(rows[:3],device,'packed',151643))
        swapped,_=core(make_batch([rows[2],rows[0],rows[1]],device,'packed',151643))
        replaced,_=core(make_batch([rows[0],rows[-1]],device,'packed',151643))
    isolation={name:metrics(one.cpu().float(),v.cpu().float(),'logits')
        for name,v in [('mixed',mixed[:1]),('swapped',swapped[1:2]),('replaced',replaced[:1])]}
    atomic(output/'ISOLATION.json',isolation)
    # Actual prefix/suffix IDs from the original token store, no re-tokenization.
    state=int(micros[0]['state'][0]); prefix=store.get('prefix',state).tolist()
    candidates=[int(c) for s,c in zip(batch['state'],batch['candidate']) if int(s)==state]
    suffixes=[store.get('suffix',c).tolist()+[151645] for c in candidates]
    assert prefix and suffixes
    suffixes=suffixes[:min(8,len(suffixes))]
    runtime=PackedRuntime(core,151643,151645,40960,device,layout='packed')
    with torch.inference_mode():
        runtime.prefill(prefix)
        snapshot=[(k.clone(),v.clone()) for k,v in runtime.prefix]
        branch=runtime.branched(len(prefix),suffixes)
        full=runtime.full([prefix+s for s in suffixes])
        immutable=all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(runtime.prefix,snapshot))
        runtime.branched(len(prefix),list(reversed(suffixes)))
        again=runtime.branched(len(prefix),suffixes)
        immutable=immutable and all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(runtime.prefix,snapshot))
    prefix_result={'fork_vs_full':metrics(full.cpu(),branch.cpu(),'logits'),
        'repeat':metrics(branch.cpu(),again.cpu(),'logits'),'prefix_immutable':immutable,
        'prefix_length':len(prefix),'suffix_lengths':list(map(len,suffixes))}
    atomic(output/'PREFIX.json',prefix_result)
    # The production inference fusions must be validated separately; never infer
    # their correctness from the differentiable path.
    reference_full=full.cpu().clone()
    reference_branch=branch.cpu().clone()
    sys.path.insert(0,'/tmp/opencode/jev-06b-research-20261112-r1-eval-recovery/code/inference')
    fused_head=install_inference(backbone,head)
    fused_core=PackedBackbone(backbone,fused_head,fa3,fused_inference=False)
    fused={};fused_values={}
    for layout_name in ('dense','packed'):
        execution=PackedRuntime(fused_core,151643,151645,40960,device,layout=layout_name)
        with torch.inference_mode():
            trace_batch=execution.batch(rows[:3])
            with RowTrace(backbone,trace_batch) as trace,torch.autocast('cuda',dtype=torch.bfloat16):
                fused_core(trace_batch)
            execution.prefill(prefix)
            snap=[(k.clone(),v.clone()) for k,v in execution.prefix]
            value=execution.branched(len(prefix),suffixes).cpu()
            value_full=execution.full([prefix+s for s in suffixes]).cpu()
            execution.branched(len(prefix),list(reversed(suffixes)))
            repeat=execution.branched(len(prefix),suffixes).cpu()
            unchanged=all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(execution.prefix,snap))
        fused_values[layout_name]=(value,value_full)
        fused[layout_name]={'fork_vs_differentiable':metrics(reference_branch,value,'logits'),
            'full_vs_differentiable':metrics(reference_full,value_full,'logits'),
            'fork_vs_full':metrics(value_full,value,'logits'),'repeat':metrics(value,repeat,'logits'),
            'prefix_immutable':unchanged,'tokenwise_trace':trace.result(),'runtime':execution.runtime()}
    fused['dense_packed']={'fork':metrics(fused_values['dense'][0],fused_values['packed'][0],'logits'),
        'full':metrics(fused_values['dense'][1],fused_values['packed'][1],'logits')}
    atomic(output/'FUSED.json',fused)
    fused_ok=all(v['prefix_immutable'] and all(v[k]['pass'] for k in ('fork_vs_differentiable','full_vs_differentiable','fork_vs_full','repeat')) for v in [fused['dense'],fused['packed']]) and all(v['pass'] for v in fused['dense_packed'].values())
    passed=all(results[m]['pass'] for m in ('dense','packed')) and all(v['pass'] for v in isolation.values()) and immutable and prefix_result['fork_vs_full']['pass'] and prefix_result['repeat']['pass'] and fused_ok
    atomic(output/'COMPLETE.json',{'complete':passed,'single_gpu_actual_macro':True,
        'four_rank_update_verified':False,'macro':macro,'pairs':len(indices),'native_attention':attention_record()})
    assert passed, 'Real mechanism exceeded frozen tolerance; inspect preserved results'


if __name__=='__main__':main()
