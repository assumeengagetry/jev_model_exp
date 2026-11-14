"""Actual complete macro0 multi-candidate cache group, forward-only research.

The harness temporarily routes the frozen core's lazy fork callable to this
candidate; source files and original Triton path are never edited. It is not a
production adapter or a full public admission.
"""
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ['PAGED_ROOT'])
MASK=Path('/tmp/opencode/jev-packed-mask-20261112-v2')
PF=Path('/tmp/opencode/jev-padding-free-20261112-v3')
COMMIT=Path('/tmp/opencode/jev-attention-2e334fb-20261112-v1')


def main():
    sys.path[:0]=[str(ROOT),str(MASK/'code'),str(PF/'code'),'/tmp/opencode/jev-joint-generic-continuation-20261109/code',
        '/home/dev/assumeengage/scripts']
    from pf_common import atomic,read,metadata,MODEL,BASE,process_ids
    assert not process_ids(3),'Assigned GPU3 must be idle; never preempt'
    assert read(ROOT/'CODE_BINDING.json')=={p.name:base64.b64encode(p.read_bytes()).decode() for p in ROOT.glob('*.py')}
    dependencies=read(ROOT/'DEPENDENCIES.json')
    for v in dependencies['files']:assert metadata(v['path'])==v,'Bound source changed'
    import numpy as np
    import torch
    from packed_model import load_model,RowTrace
    from packed_mask import MaskedBackbone,make_batch
    from packed_mask.model import PrefixKV
    import packed_mask.fork as fork_module
    from native_paged_prefix import PagedPrefixOwner
    from joint_data import PairPool,TokenStore
    cfg=read('/tmp/opencode/jev-06b-research-20261112-r1-bridge/train/config.json')
    source=read(COMMIT/'runs/causal/INPUT.json');dataset=PairPool(root=cfg['training_pool'],domains=cfg['active_training_domains'])
    store=TokenStore();batch=dataset.fetch(np.asarray(source['indices'],dtype=np.int64))
    spec=importlib.util.spec_from_file_location('paged_case_original',COMMIT/'commit-code/patched/pair_training_runtime.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    assembled=[original.assemble(store,s,c,int(l),151645,boundary=True) for s,c,l in zip(batch['state'],batch['candidate'],batch['max_length'])]
    rows=[r.tolist() for r,_ in assembled];prefixes=[int(p) for _,p in assembled];groups={}
    for i,(r,p) in enumerate(zip(rows,prefixes)):groups.setdefault(tuple(r[:p]),[]).append(i)
    prefix,chosen=max(((p,ix) for p,ix in groups.items() if len(ix)>=2),key=lambda x:len(x[1]))
    fullrows=[rows[i] for i in chosen];suffixes=[rows[i][prefixes[i]:] for i in chosen];capacity=sum(map(len,suffixes))
    torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    model,backbone,head,native=load_model(MODEL,BASE,torch.device('cuda:0'));model.requires_grad_(False);model.eval()
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();summaries={};tensors={};traces=[]
    def metric(ref,value):
        ref=ref.double();value=value.double();d=value-ref;rms=float(ref.square().mean().sqrt())
        return {'pass':float(d.square().mean().sqrt())<=.01*max(rms,1e-3) and float(d.abs().max())<=.02+.02*float(ref.abs().max()),
            'relative_rms':float(d.square().mean().sqrt())/max(rms,1e-3),'max_abs':float(d.abs().max()),'bitwise_equal':torch.equal(ref,value)}
    def run_core(core,layout,boundary,**options):
        with RowTrace(backbone,layout) as trace:value=core(layout,boundary,**options)
        traces.append(trace.result());return value
    old_callable=fork_module.fork_attention
    try:
        for mode in ('causal','prefix_block'):
            assert time.monotonic()-started<900 and not (ROOT/'STOP.json').exists(),'Paged case stop/deadline'
            core=MaskedBackbone(backbone,head,native,mode).eval();boundary=[len(prefix)]*len(chosen)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                full,_=run_core(core,make_batch(fullrows,'cuda:0'),boundary)
                _,captured=run_core(core,make_batch([prefix],'cuda:0'),[len(prefix)],capture_prefix=True)
                original_prefix_snapshot=[(k.clone(),v.clone()) for k,v in captured.layers]
                triton,_=run_core(core,make_batch(suffixes,'cuda:0',prefix_length=len(prefix)),boundary,prefix=captured)
                owner=PagedPrefixOwner(captured.layers,mode=mode,prefix_token_ids=prefix,rendered_prefix_length=len(prefix),
                    core_owner=captured.owner,parameter_versions=core._versions(),stream=torch.cuda.current_stream(),
                    suffix_capacity=capacity,native_fa3=native)
                owned=PrefixKV(mode,prefix,owner.prefix_views,captured.owner,captured.parameter_versions)
                del captured  # No old prefix tensors retained by owner/harness.
                backing_snapshot=[(k.clone(),v.clone()) for k,v in owner.prefix_views]
            # Verify the narrowed API rejects context switching before suffix writes.
            with torch.no_grad():
                rejected=False
                try:owner.bind_wave(list(map(len,suffixes)),mode=mode,prefix_token_ids=prefix,rendered_prefix_length=len(prefix),
                    core_owner=owned.owner,parameter_versions=core._versions())
                except RuntimeError as exc:rejected='inference_mode' in str(exc)
                assert rejected,'no_grad-only must be rejected before writable inference tensor use'
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                def paged_wave(real_suffixes):
                    layout=make_batch(real_suffixes,'cuda:0',prefix_length=len(prefix))
                    bound=owner.bind_wave(layout.lengths,mode=mode,prefix_token_ids=prefix,rendered_prefix_length=len(prefix),
                        core_owner=core._prefix_owner,parameter_versions=core._versions())
                    layer_index=0
                    def actual_paged(q,k,v,pk,pv,cu,max_length,*,suffix_causal,softmax_scale):
                        nonlocal layer_index
                        assert cu is layout.cu and max_length==bound.max_seqlen_q and suffix_causal==(mode=='causal')
                        expect_k,expect_v=owner.prefix_views[layer_index]
                        assert pk.data_ptr()==expect_k.data_ptr() and pv.data_ptr()==expect_v.data_ptr()
                        result=owner.fork(q,k,v,layer=layer_index,bound=bound,softmax_scale=softmax_scale)
                        layer_index+=1;return result
                    fork_module.fork_attention=actual_paged
                    try:result,_=run_core(core,layout,[len(prefix)]*len(real_suffixes),prefix=owned)
                    finally:fork_module.fork_attention=old_callable
                    assert layer_index==28
                    return result
                paged=paged_wave(suffixes)
                reverse=paged_wave(list(reversed(suffixes))).flip(0)
                repeat=paged_wave(suffixes)
                torch.cuda.synchronize()  # Surface asynchronous API/kernel errors before accepting a receipt.
                immutable=all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(owner.prefix_views,backing_snapshot))
                copied_exact=all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(owner.prefix_views,original_prefix_snapshot))
                fm=full[:,1].float()-full[:,0].float();pm=paged[:,1].float()-paged[:,0].float()
                summary={'full_vs_paged_raw':metric(full.cpu(),paged.cpu()),'full_vs_paged_margin':metric(fm.cpu(),pm.cpu()),
                    'full_vs_triton_raw':metric(full.cpu(),triton.cpu()),
                    'full_vs_triton_margin':metric(fm.cpu(),(triton[:,1].float()-triton[:,0].float()).cpu()),
                    'paged_reverse':metric(paged.cpu(),reverse.cpu()),'paged_repeat_exact':torch.equal(paged,repeat),
                    'prefix_immutable':immutable,'prefix_copied_exactly_once_per_layer':copied_exact,
                    'no_grad_context_rejected_before_write':rejected,'structure':owner.structural_report(),
                    'actual_prefix_rows':len(prefix),'actual_suffix_lengths':list(map(len,suffixes))}
                tensors[mode]={'full':full.float().cpu(),'triton':triton.float().cpu(),'paged':paged.float().cpu(),
                    'reverse':reverse.float().cpu(),'repeat':repeat.float().cpu()}
                owner.invalidate()
            summaries[mode]=summary;atomic(ROOT/'SUMMARY.json',summaries)
        torch.save(tensors,ROOT/'ACTUAL_LOGITS.pt');atomic(ROOT/'TRACE.json',traces)
        for v in dependencies['files']:assert metadata(v['path'])==v,'Bound source changed during run'
        accepted=all(s['full_vs_paged_raw']['pass'] and s['full_vs_paged_margin']['pass'] and s['paged_reverse']['pass'] and
            s['paged_repeat_exact'] and s['prefix_immutable'] and s['prefix_copied_exactly_once_per_layer'] for s in summaries.values())
        atomic(ROOT/'COMPLETE.json',{'execution_complete':True,'numeric_acceptance':accepted,'modes':list(summaries),
            'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
            'new_public_score':None,'whole_serving_admitted':False,'wholegroup_candidates_preserved':len(chosen)})
        print(json.dumps(read(ROOT/'COMPLETE.json'),indent=2),flush=True)
    finally:fork_module.fork_attention=old_callable


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        with (ROOT/'FAILURE.json').open('x') as f:json.dump({'execution_complete':False,'error':str(exc),'traceback':traceback.format_exc(),'no_fallback':True},f,indent=2)
        raise
