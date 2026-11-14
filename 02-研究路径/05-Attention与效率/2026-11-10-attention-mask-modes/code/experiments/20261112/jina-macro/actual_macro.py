"""Real complete first 16-case clean macro wiring, NOT clean model training.

Uses the declared historical diagnostic adapter only to validate prefix-once,
multi-EOS, compiled backward and rank coefficients before fresh method training.
No checkpoint choice/public/confirm data or teacher labels are opened.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ['JINA_ROOT'])
DATA=Path('/tmp/opencode/jev-clean-origin-20261112-v1/prepared-01')
PF=Path('/tmp/opencode/jev-padding-free-20261112-v3')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('J0','J1','J2'),required=True);args=parser.parse_args()
    sys.path[:0]=[str(ROOT/'code'),str(PF/'code'),'/tmp/opencode/jev-joint-generic-continuation-20261109/code','/home/dev/assumeengage/scripts']
    from pf_common import atomic,read,metadata,MODEL,BASE
    assert read(ROOT/'CODE_BINDING.json')=={p.name:base64.b64encode(p.read_bytes()).decode() for p in (ROOT/'code').glob('*.py')}
    for v in read(ROOT/'DEPENDENCIES.json')['files']:assert metadata(v['path'])==v,'Bound source changed'
    complete=read(DATA/'COMPLETE.json');assert complete['DATA_READY'] and complete['data_ready']==metadata(DATA/'DATA_READY.json') and not (DATA/'BLOCKED.json').exists()
    output=ROOT/args.arm;output.mkdir(exist_ok=False)
    import torch
    from packed_model import load_model,PackedBackbone,RowTrace
    from list_layout import CandidateField,build_candidate_layout
    from list_attention import CandidateFlexBackend
    from ranking_loss import prepare_supervision,gradcoeff
    chosen=[]
    with (DATA/'data/exposure-seed42.jsonl').open() as f:
        for _ in range(16):
            r=json.loads(next(f));assert r['macro_index']==0;chosen.append(r['case_id'])
    required=set(chosen);found={}
    with (DATA/'data/train.jsonl').open() as f:
        for line in f:
            case=json.loads(line)
            if case['case_id'] in required:found[case['case_id']]=case
            if len(found)==len(required):break
    assert set(found)==required and len(chosen)==16
    fields=[];targets=[];known=[];raw_unknown=0
    for cid in chosen:
        c=found[cid]
        for ordinal,field in enumerate(c['fields']):
            if not field['known']:
                assert field['forward_policy']=='masked_unknown_no_forward';raw_unknown+=1;continue
            field_id=ordinal;key=(cid,field_id);k=len(field['options'])
            fields.append(CandidateField(field['prefix_ids'],field['suffix_ids'],cid,field_id,tuple(range(k))))
            targets.extend(field['target']);known.extend([True]*k)
    torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    layout=build_candidate_layout(fields,'cuda:0',151643,'sequential',eos_token_id=151645,split='train')
    supervision=prepare_supervision(layout,targets,known,bce_case_reduction='known_fields')
    model,backbone,head,native=load_model(MODEL,BASE,torch.device('cuda:0'),trainable=True)
    mode='interaction' if args.arm=='J1' else 'isolation';rank=1. if args.arm=='J2' else 0.
    backend=CandidateFlexBackend(mode);bound=backend.bind_batch(layout);core=PackedBackbone(backbone,head,bound)
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad];assert len(params)==393
    initial={n:p.detach().cpu().clone() for n,p in params}
    optimizer=torch.optim.AdamW([p for _,p in params],lr=2e-5,weight_decay=.01,fused=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    # Separate autocast scopes are essential to avoid no-grad's cached FP32
    # adapter casts contaminating the formal grad-enabled replay.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):first,_=core(layout.batch)
    margins=first[:,1].float()-first[:,0].float();coeff=gradcoeff(margins,supervision,lambda_rank=rank,tau=1.)
    with RowTrace(backbone,layout.batch) as trace,torch.autocast('cuda',dtype=torch.bfloat16):
        second,_=core(layout.batch);replayed=second[:,1].float()-second[:,0].float()
        replay_equal=torch.equal(margins,replayed.detach())
        atomic(output/'REPLAY.json',{'equal':replay_equal,'max_abs':float((margins-replayed.detach()).abs().max())})
        assert replay_equal,'Real fullcase no-grad/grad replay differs; no coefficient-VJP claim'
        (replayed*coeff.coefficients.to(replayed.device)).sum().backward()
    grads={n:p.grad.detach().cpu().clone() for n,p in params};assert all(torch.isfinite(g).all() for g in grads.values())
    clip=float(torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,foreach=True));optimizer.step();torch.cuda.synchronize()
    torch.save({'first_logits':first.detach().cpu(),'second_logits':second.detach().cpu(),'coefficients':coeff.coefficients.cpu(),
        'gradients':grads,'updates':{n:p.detach().cpu()-initial[n] for n,p in params},'optimizer':optimizer.state_dict()},output/'ACTUAL_MACRO.pt')
    atomic(output/'TRACE.json',trace.result())
    atomic(output/'COMPLETE.json',{'execution_complete':True,'arm':args.arm,'mask':mode,'lambda_rank':rank,
        'complete_cases':len(chosen),'known_fields_forwarded':len(fields),'unknown_raw_fields_retained_not_forwarded':raw_unknown,
        'candidates':layout.candidate_count,'physical_real_tokens':layout.total_tokens,
        'logical_pair_tokens':sum(found[c]['logical_supervised_pair_tokens'] for c in chosen),
        'field_cu_segments':len(layout.batch.lengths),'candidate_eos_count':layout.batch.eos_indices.numel(),
        'exact_two_pass_replay':replay_equal,'finite_gradient_groups':len(grads),'clip_input_norm':clip,
        'flex_forward_submissions':bound.forward_submissions,'flex_backward_output_callbacks':bound.output_gradient_callbacks,
        'loss':float(coeff.loss),'statistics':coeff.statistics,'mask_metadata':bound.metadata.__dict__,
        'seconds_including_first_production_compile':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
        'historical_diagnostic_weights_only':True,'fresh_clean_training':False,'new_public_score':None,
        'first_macro_counts_as_clean_exposure':False,'all_full_public_coverage_admitted':False})
    print(json.dumps(read(output/'COMPLETE.json'),indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        with (ROOT/(os.environ.get('JINA_ARM','unknown')+'-FAILURE.json')).open('x') as f:
            json.dump({'execution_complete':False,'error':str(exc),'traceback':traceback.format_exc(),'no_fallback':True},f,indent=2)
        raise
