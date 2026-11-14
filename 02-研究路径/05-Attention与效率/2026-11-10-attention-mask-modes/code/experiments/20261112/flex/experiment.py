"""Real frozen training macro backend comparison; no synthetic acceptance input."""
import importlib.util
import os
from pathlib import Path
import sys
import time

STATE={'stage':'startup','started_monotonic':time.monotonic()}


def main():
    assert importlib.util.find_spec('flex_backend') is not None, 'Real packed Flex backend is not implemented'
    pf=Path(os.environ['PF_ROOT'])
    root=Path(os.environ['FLEX_ROOT']);STATE['root']=str(root)
    assert Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])==root/'inductor-cache'
    assert Path(os.environ['TRITON_CACHE_DIR'])==root/'triton-cache'
    assert os.environ.get('FLEX_EXCLUSIVE_GPU_ADMITTED')=='1','Queue must admit an idle exclusive GPU before import'
    sys.path.insert(0,str(pf/'code'))
    from pf_common import verify,read,atomic,MODEL,BASE,OLD
    verify()
    mechanism=Path(os.environ.get('PF_MECHANISM_ROOT',str(pf/'mechanism')))
    require_gate=read(mechanism/'COMPLETE.json')
    assert require_gate['complete'], 'PF real macro must be admitted before Flex reuse'
    import numpy as np
    import torch
    from torch.nn import functional as F
    from flex_backend import FlexBackend
    from packed_model import load_model,PackedBackbone,RowTrace
    from packed_layout import make_batch
    sys.path[:0]=[str(OLD/'code'),'/home/dev/assumeengage/scripts',
                  '/tmp/opencode/jev-06b-research-20261112-r1-bridge/code']
    from joint_data import PairPool,TokenStore
    from pair_training_runtime import assemble
    from group_training_runtime import load_plan,atomic_microbatches
    from train_bridge import group_micro_loss
    from student_flash_attention3_repro import attention_record
    output=root/'results';output.mkdir(exist_ok=False)
    cfg=read('/tmp/opencode/jev-06b-research-20261112-r1-bridge/train/config.json')
    source=read(mechanism/'INPUT.json')
    dataset=PairPool(root=cfg['training_pool'],domains=cfg['active_training_domains']);store=TokenStore()
    plan=load_plan(cfg['group_plan_root'],cfg['group_plan_identity'])
    indices=np.asarray(source['indices'],dtype=np.int64);batch=dataset.fetch(indices)
    batch['relation_tag']=np.asarray(source['tags'],dtype=np.int32)
    targets=np.load(cfg['effective_target_path'],mmap_mode='r')
    batch['distillation_target']=np.asarray(targets[indices]).copy()
    micros=list(atomic_microbatches(batch,cfg['max_microbatch_pairs'],cfg['max_microbatch_tokens']))
    rows_by_micro=[[assemble(store,s,c,int(l),151645).tolist()
        for s,c,l in zip(m['state'],m['candidate'],m['max_length'])] for m in micros]
    assert [list(map(len,r)) for r in rows_by_micro]==source['lengths']
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda:0')
    model,backbone,head,native=load_model(MODEL,BASE,device,trainable=True)
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    initial={n:p.detach().cpu().clone() for n,p in parameters}
    assert len(initial)==393
    reference=None;results={};flex=FlexBackend()
    def metric(ref,current,kind):
        ref=ref.double();current=current.double();delta=current-ref
        norm=float(ref.norm());rms=float(ref.square().mean().sqrt())
        error=float(delta.norm());cosine=None
        if rms>1e-7:cosine=float(F.cosine_similarity(ref.reshape(1,-1),current.reshape(1,-1),dim=1))
        if kind=='logits':ok=float(delta.square().mean().sqrt())<=.01*max(rms,1e-3) and float(delta.abs().max())<=.02+.02*float(ref.abs().max())
        else:ok=error<=.02*norm+1e-7*ref.numel()**.5 and (cosine is None or cosine>=.999)
        return {'pass':ok,'relative_l2':error/max(norm,1e-30),'max_abs':float(delta.abs().max()),'cosine':cosine,'reference_rms':rms}
    for name,backend in [('fa3',native),('flex',flex)]:
        STATE.update(backend=name,stage='initialization')
        for n,p in parameters:p.data.copy_(initial[n].to(device));p.grad=None
        optimizer=torch.optim.AdamW([p for _,p in parameters],lr=2e-5,weight_decay=.01,fused=True)
        core=PackedBackbone(backbone,head,backend);traces=[];logits_all=[];losses=[];mask_metadata=[]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        for ordinal,(micro,rows) in enumerate(zip(micros,rows_by_micro)):
            assert not (root/'NO_AUTOMATIC_RESUME').exists(),'Flex stop requested'
            STATE.update(micro=ordinal,lengths=list(map(len,rows)),real_tokens=sum(map(len,rows)),
                stage='layout',forward_calls=flex.forward_calls,backward_calls=flex.backward_calls)
            layout=make_batch(rows,device,'packed',151643)
            if name=='flex':
                STATE['stage']='mask_metadata'
                flex.bind(layout.lengths,device);mask_metadata.append(dict(flex.metadata))
            with RowTrace(backbone,layout) as trace,torch.autocast('cuda',dtype=torch.bfloat16):
                STATE['stage']='forward'
                logits,_=core(layout);margin=logits[:,1].float()-logits[:,0].float()
                loss,_,_=group_micro_loss(torch,F,np,margin,micro,plan['teacher_relation_target'],cfg,device)
                STATE.update(stage='backward',forward_calls=flex.forward_calls)
                (loss/len(indices)).backward()
                STATE['backward_calls']=flex.backward_calls
            traces.append(trace.result());logits_all.append(logits.detach().float().cpu());losses.append(float(loss.detach())/len(indices))
        gradients={n:p.grad.detach().cpu().clone() for n,p in parameters}
        norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,foreach=True))
        STATE['stage']='optimizer_step';optimizer.step();torch.cuda.synchronize()
        updates={n:p.detach().cpu()-initial[n] for n,p in parameters}
        values={'logits':torch.cat(logits_all),'gradients':gradients,'updates':updates,'loss':sum(losses)}
        result={'seconds_including_first_production_compile':time.monotonic()-start,
            'peak_memory_bytes':torch.cuda.max_memory_allocated(),'loss':sum(losses),
            'clip_input_norm':norm,'tokenwise_trace':traces,'mask_metadata':mask_metadata,
            'native_attention':attention_record()}
        if name=='fa3':reference=values
        else:
            result['logits']=metric(reference['logits'],values['logits'],'logits')
            result['loss_pass']=abs(values['loss']-reference['loss'])<=1e-3*(1+abs(reference['loss']))
            for kind in ('gradients','updates'):
                result[kind]={n:metric(reference[kind][n],v,kind) for n,v in values[kind].items()}
            result['pass']=result['logits']['pass'] and result['loss_pass'] and all(v['pass'] for kind in ('gradients','updates') for v in result[kind].values())
            result['actual_flex_forward_calls']=flex.forward_calls;result['actual_q_backward_hooks']=flex.backward_calls
            result['compiler_counters']={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}
        torch.save(values,output/(name+'.pt'));results[name]=result;atomic(output/'METRICS.json',results)
    assert results['flex']['actual_q_backward_hooks']==28*len(micros)
    atomic(output/'COMPLETE.json',{'complete':results['flex']['pass'],'macro':source['macro'],'pairs':len(indices),
        'scope':'same-weight causal backend mechanism, not mask retraining or quality gain',
        'first_production_call_timed':True,'cold_cache_claim':False,
        'inductor_cache':os.environ['TORCHINDUCTOR_CACHE_DIR'],'triton_cache':os.environ['TRITON_CACHE_DIR'],
        'public_score':None,'clean_ranking':False})
    assert results['flex']['pass'], 'Flex exceeded pre-frozen numerical contract; retain evidence'


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import json,traceback
        STATE.update(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-STATE['started_monotonic'],complete=False)
        if 'root' in STATE:
            with (Path(STATE['root'])/'FAILURE.json').open('x') as stream:json.dump(STATE,stream,indent=2)
        raise
