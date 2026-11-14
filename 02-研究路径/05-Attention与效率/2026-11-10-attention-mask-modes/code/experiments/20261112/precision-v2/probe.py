"""Real input-only numeric probe, no labels/teacher cache/optimizer/public scoring."""
import argparse
import base64
from collections import defaultdict
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ.get('PRECISION_ROOT','/tmp/opencode/jev-padding-precision-20261112-v1'))
MASK=Path('/tmp/opencode/jev-packed-mask-20261112-v2')
PF=Path('/tmp/opencode/jev-padding-free-20261112-v3')
COMMIT=Path('/tmp/opencode/jev-attention-2e334fb-20261112-v1')
OLD=Path('/tmp/opencode/jev-joint-generic-continuation-20261109')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tf32',type=int,choices=(0,1),required=True)
    parser.add_argument('--bf16-reduction',type=int,choices=(0,1),required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    sys.path[:0]=[str(MASK/'code'),str(PF/'code'),str(COMMIT/'commit-code'),str(OLD/'code'),'/home/dev/assumeengage/scripts']
    from pf_common import read,atomic,metadata,MODEL,BASE
    assert {p.name:base64.b64encode(p.read_bytes()).decode() for p in (ROOT/'code').glob('*.py')}==read(ROOT/'CODE_BINDING.json'),'Precision source changed'
    dependencies=read(ROOT/'DEPENDENCIES.json')
    for value in dependencies['files']:assert metadata(value['path'])==value,'Bound precision input changed'
    output=ROOT/f'tf32-{args.tf32}-bf16red-{args.bf16_reduction}'
    import torch
    import numpy as np
    from packed_mask import MaskedBackbone,make_batch
    from packed_model import load_model
    from joint_data import PairPool,TokenStore
    from group_training_runtime import load_plan
    from attention_modes import attention_layout,enable_bidirectional,install_prefix_block_flash
    from student_flash_attention3_repro import register_student_fa3,BACKEND,attention_record
    cfg=read('/tmp/opencode/jev-06b-research-20261112-r1-bridge/train/config.json')
    source=read(COMMIT/'runs/causal/INPUT.json');plan=load_plan(cfg['group_plan_root'],cfg['group_plan_identity'])
    macro=int(source['macro']);a,b=map(int,plan['macro_offsets'][macro:macro+2]);indices=np.asarray(source['indices'],dtype=np.int64)
    assert b-a==len(indices)==128 and np.array_equal(indices,plan['order'][a:b])
    dataset=PairPool(root=cfg['training_pool'],domains=cfg['active_training_domains']);store=TokenStore();batch=dataset.fetch(indices)
    spec=importlib.util.spec_from_file_location('precision_original_collate',COMMIT/'commit-code/patched/pair_training_runtime.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    assembled=[original.assemble(store,s,c,int(l),151645,boundary=True) for s,c,l in zip(batch['state'],batch['candidate'],batch['max_length'])]
    rows=[r.tolist() for r,_ in assembled];prefixes=[int(p) for _,p in assembled]
    groups=defaultdict(list)
    for i,(row,p) in enumerate(zip(rows,prefixes)):groups[tuple(row[:p])].append(i)
    group=max((v for v in groups.values() if len(v)>=2),key=len)
    ia,ib=group[:2]  # Keep the previous actual failing full/fork fixture fixed.
    buckets={};neighbor_pair=None;neighbor_sources=[]
    for fixture_macro in range(min(64,len(plan['macro_offsets'])-1)):
        if fixture_macro==macro:
            candidates=[(rows[i],prefixes[i],fixture_macro,i,int(indices[i])) for i in range(len(rows)) if i!=ia]
        else:
            lo,hi=map(int,plan['macro_offsets'][fixture_macro:fixture_macro+2]);ix=plan['order'][lo:hi]
            mb=dataset.fetch(ix);candidates=[]
            for j,(s,c,l) in enumerate(zip(mb['state'],mb['candidate'],mb['max_length'])):
                ids,p=original.assemble(store,s,c,int(l),151645,boundary=True)
                candidates.append((ids.tolist(),int(p),fixture_macro,j,int(ix[j])))
        for record in candidates:
            row,p,fm,j,di=record;key=(len(row),p)
            if key in buckets and buckets[key][0]!=row:
                neighbor_pair=(buckets[key],record)
                break
            buckets.setdefault(key,record)
        if neighbor_pair is not None:break
    assert neighbor_pair is not None,'No actual equal-row/P/S neighbor pair in bounded64macro metadata cohort'
    selected_neighbors=[]
    for row,p,fm,j,di in neighbor_pair:
        if fm==macro:index=j
        else:
            index=len(rows);rows.append(row);prefixes.append(p)
        selected_neighbors.append(index);neighbor_sources.append({'macro':fm,'position':j,'dataset_index':di})
    nb,nc=selected_neighbors
    fixture={'ready':True,'target_macro':macro,'target_and_cache_positions':[ia,ib],
        'neighbor_sources':neighbor_sources,'neighbors_global_positions':[nb,nc],
        'neighbor_geometry':{'row_length':len(rows[nb]),'prefix_length':prefixes[nb],
            'suffix_length':len(rows[nb])-prefixes[nb]},'selected_by_input_metadata_only':True,
        'synthetic_tokens':False,'model_run':False}
    assert len(rows[nb])==len(rows[nc]) and prefixes[nb]==prefixes[nc] and rows[nb]!=rows[nc]
    if args.prepare_only:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Fixture preparation must hide CUDA'
        assert not (ROOT/'FIXTURE_READY.json').exists(),'Existing fixture preserved'
        atomic(ROOT/'FIXTURE_READY.json',fixture)
        return
    assert read(ROOT/'FIXTURE_READY.json')==fixture,'Actual numeric fixture differs from frozen metadata admission'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=bool(args.tf32)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=bool(args.bf16_reduction)
    device=torch.device('cuda:0');model,backbone,head,native=load_model(MODEL,BASE,device)
    model.requires_grad_(False);model.eval()
    atomic(output/'INPUT.json',{'macro':macro,'original_macro_indices':source['indices'],
        'selected_positions':[ia,ib,nb,nc],'lengths':[len(rows[i]) for i in (ia,ib,nb,nc)],
        'prefix_lengths':[prefixes[i] for i in (ia,ib,nb,nc)],'fixture':fixture,'labels_or_teacher_targets_used':False,
        'tf32':torch.backends.cuda.matmul.allow_tf32,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'model_files':[metadata(p) for p in sorted(MODEL.iterdir()) if p.is_file()]})
    started=time.monotonic();summaries={};saved={}
    def metric(ref,value):
        ref=ref.double();value=value.double();d=value-ref;rms=float(ref.square().mean().sqrt())
        return {'pass':float(d.square().mean().sqrt())<=.01*max(rms,1e-3) and float(d.abs().max())<=.02+.02*float(ref.abs().max()),
            'relative_rms':float(d.square().mean().sqrt())/max(rms,1e-3),'max_abs':float(d.abs().max()),'bitwise_equal':torch.equal(ref,value)}
    for mode in ('causal','bidirectional','prefix_block'):
        assert time.monotonic()-started<900 and not (ROOT/'STOP.json').exists(),'Probe deadline/stop'
        register_student_fa3(cfg['student_attention_kernel_directory'])
        for layer in backbone.layers:layer.self_attn.is_causal=True
        model.config.is_causal=True
        if mode=='bidirectional':enable_bidirectional(model)
        elif mode=='prefix_block':install_prefix_block_flash(BACKEND)
        core=MaskedBackbone(backbone,head,native,mode).eval();frames={}
        specs=[('solo_packed',[ia],'packed',False),('mixed_packed',[ia,ib],'packed',False),
               ('geometry_packed',[ia,nb],'packed',False),('replacement_packed',[ia,nc],'packed',False),('mixed_dense',[ia,nb],'dense',False),
               ('hf_solo',[ia],'dense',True),('hf_mixed',[ia,ib],'dense',True),
               ('hf_geometry',[ia,nb],'dense',True),('hf_replacement',[ia,nc],'dense',True)]
        for name,chosen,layout_name,hf in specs:
            assert time.monotonic()-started<900 and not (ROOT/'STOP.json').exists(),'Probe deadline/stop'
            layout=make_batch([rows[i] for i in chosen],device,layout_name)
            p=[prefixes[i] for i in chosen];captured={};n=len(rows[ia])
            def rope_hook(module,inputs,out):
                captured['cos']=out[0].reshape(-1,128)[:n].detach().float().cpu()
                captured['sin']=out[1].reshape(-1,128)[:n].detach().float().cpu()
                captured['rope_position_shape']=list(inputs[1].shape)
            def hidden_hook(module,inputs):
                x=inputs[0];captured['layer0_hidden']=(x[0,:n] if x.ndim==3 else x[:n]).detach().float().cpu()
            def head_hook(module,inputs):
                x=inputs[0];captured['eos_hidden']=(x[0,n-1] if x.ndim==3 else x[n-1]).detach().float().cpu()
            def flat_hook(key):
                def observe(module,inputs,out):captured[key]=out.reshape(-1,out.shape[-1])[:n].detach().float().cpu()
                return observe
            def context_hook(module,inputs):
                x=inputs[0];captured['attention_context']=x.reshape(-1,x.shape[-1])[:n].detach().float().cpu()
            handles=[backbone.rotary_emb.register_forward_hook(rope_hook),
                     backbone.layers[0].input_layernorm.register_forward_hook(flat_hook('input_norm0')),
                     backbone.layers[0].self_attn.q_proj.register_forward_hook(flat_hook('q_projection0')),
                     backbone.layers[0].self_attn.o_proj.register_forward_pre_hook(context_hook),
                     backbone.layers[1].input_layernorm.register_forward_pre_hook(hidden_hook),
                     head.register_forward_pre_hook(head_hook)]
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    if hf:
                        with attention_layout(torch.tensor(p,dtype=torch.int32,device=device),
                                torch.tensor(layout.lengths,dtype=torch.int32,device=device)):
                            logits=model(input_ids=layout.ids,attention_mask=layout.mask,use_cache=False).logits
                    else:logits,_=core(layout,p)
                assert torch.isfinite(logits).all(),'Nonfinite actual inference'
                captured['logits']=logits.detach().float().cpu();captured['input_shape']=list(layout.ids.shape)
                frames[name]=captured
            finally:
                for h in handles:h.remove()
        comparison={}
        for label,left,right in [('packed_solo_mixed','solo_packed','mixed_packed'),
                ('packed_same_geometry_neighbor','geometry_packed','replacement_packed'),
                ('hf_solo_mixed','hf_solo','hf_mixed'),('hf_same_geometry_neighbor','hf_geometry','hf_replacement'),
                ('dense_packed_mixed','mixed_dense','geometry_packed'),('hf_packed_mixed','hf_geometry','geometry_packed')]:
            comparison[label]={k:metric(frames[left][k][:1] if k=='logits' else frames[left][k],
                frames[right][k][:1] if k=='logits' else frames[right][k]) for k in ('cos','sin','input_norm0','q_projection0','attention_context','layer0_hidden','eos_hidden','logits')}
        if mode!='bidirectional':
            prefix=tuple(rows[ia][:prefixes[ia]]);suffix=[rows[i][prefixes[i]:] for i in (ia,ib)]
            assert prefixes[ia]==prefixes[ib] and tuple(rows[ib][:prefixes[ib]])==prefix
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                _,cache=core(make_batch([prefix],device),[len(prefix)],capture_prefix=True)
                branch,_=core(make_batch(suffix,device,prefix_length=len(prefix)),[len(prefix)]*2,prefix=cache)
            full=frames['mixed_packed']['logits'];branch=branch.float().cpu()
            comparison['prefix_full_fork']={'raw':metric(full,branch),
                'margin':metric(full[:,1]-full[:,0],branch[:,1]-branch[:,0])}
        saved[mode]=frames;summaries[mode]=comparison
        atomic(output/'SUMMARY.json',{'tf32':bool(args.tf32),'bf16_reduction':bool(args.bf16_reduction),
            'modes':summaries,'elapsed_seconds':time.monotonic()-started,'native_attention':attention_record(),
            'learning_updates':0,'new_public_score':None})
    torch.save(saved,output/'ACTUAL_TENSORS.pt')
    for value in dependencies['files']:assert metadata(value['path'])==value,'Precision dependency changed during run'
    atomic(output/'COMPLETE.json',{'execution_complete':True,'tf32':bool(args.tf32),'bf16_reduction':bool(args.bf16_reduction),
        'modes':list(summaries),'learning_updates':0,'public_scored':False,'elapsed_seconds':time.monotonic()-started})
    print(__import__('json').dumps(read(output/'COMPLETE.json'),indent=2),flush=True)


if __name__=='__main__':main()
