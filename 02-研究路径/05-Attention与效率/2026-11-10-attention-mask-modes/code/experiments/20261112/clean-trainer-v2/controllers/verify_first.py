"""Primary direct readback of actual formal exposure artifacts; no model forward."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(os.environ['CLEAN_TRAIN_ROOT'])
ARMS=('C_D_BCE','C_P_BCE','C_D_CE','C_P_CE')


def main():
    os.setpriority(os.PRIO_PROCESS,0,19)
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/'source'))
    from bindings import read_json,stat,verify,write_json
    wave=read_json(ROOT/'FIRST_WAVE.json');assert wave['all_paused_at_cursor1']
    init=load_file(str(ROOT/'init-seed42/adapter_model.safetensors'),device='cpu')
    reports={};pins={};reference_slots=None;reference_before=None
    for arm in ARMS:
        folder=ROOT/(arm+'-seed42');artifact=folder/'first-actual-macro'
        assert not list(folder.glob('BLOCKED*')) and not (folder/'COMPLETE.json').exists()
        commit=read_json(artifact/'COMPLETE.json');assert commit['complete'] and commit['is_formal_exposure_macro0']
        verify(commit['files'])
        slots=read_json(artifact/'SLOTS.json');assert len(slots['exposures'])==16
        mapping={k:slots[k] for k in ('exposures','slots','fields','micros')}
        if reference_slots is None:reference_slots=mapping
        else:assert mapping==reference_slots,'Four-arm input/slot/micro composition differs'
        cache=torch.load(artifact/'actual-cache-and-coefficients.pt',map_location='cpu',weights_only=True)
        replay=torch.load(artifact/'actual-replay-logits.pt',map_location='cpu',weights_only=True)
        for key in ('logits','margins'):
            assert torch.equal(cache[key].contiguous().view(torch.uint8),replay[key].contiguous().view(torch.uint8))
        assert len(cache['margins'])==len(slots['slots']) and replay['bitwise_equal_to_first']
        m=cache['margins'].clone().requires_grad_(True);losses=[]
        for field in slots['fields']:
            v=m[field['start']:field['end']];q=torch.tensor(field['target'],dtype=torch.float32)
            value=F.binary_cross_entropy_with_logits(v,q,reduction='mean') if arm.endswith('BCE') else -(q*F.log_softmax(v,0)).sum()
            losses.append(value/field['known_fields_in_case']/16)
        loss=torch.stack(losses).sum();coefficient,=torch.autograd.grad(loss,m)
        assert torch.equal(coefficient,cache['coefficients']) and float(loss.detach())==cache['objective']
        raw=torch.load(artifact/'all393-gradients-unclipped.pt',map_location='cpu',weights_only=True)
        clipped=torch.load(artifact/'all393-gradients-clipped.pt',map_location='cpu',weights_only=True)
        actual=torch.load(artifact/'all393-actual-update-and-optimizer.pt',map_location='cpu',weights_only=True)
        assert len(raw)==len(clipped)==393 and set(raw)==set(clipped)
        assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for values in (raw,clipped) for v in values.values())
        assert set(actual['before'])==set(actual['after'])==set(actual['delta'])==set(init) and len(init)==393
        assert all(torch.equal(v.contiguous().view(torch.uint8),init[k].contiguous().view(torch.uint8)) for k,v in actual['before'].items())
        assert all(torch.equal(actual['after'][k]-actual['before'][k],actual['delta'][k]) and torch.isfinite(actual['after'][k]).all() for k in init)
        assert len(actual['optimizer']['state'])==393 and actual['scheduler']['completed']==1
        assert all(float(v['step'])==1 and all(torch.isfinite(v[k]).all() for k in ('exp_avg','exp_avg_sq')) for v in actual['optimizer']['state'].values())
        meta=read_json(artifact/'ACTUAL_MACRO.json');assert meta['optimizer_steps']==1 and meta['logical_pair_tokens']==20947
        assert meta['actual_completed_forward_real_tokens_this_macro']==41894 and meta['actual_completed_backward_real_tokens_this_macro']==20947
        assert meta['first_replay_logits_and_margins_bitwise_equal'] and meta['all393_gradients_finite']
        assert meta['gradient_keys']==393 and meta['used_lr']==2e-5/107
        with (folder/'ledger.jsonl').open() as f:events=[json.loads(line) for line in f]
        updates=[v for v in events if v['event']=='macro_updated'];starts=[v for v in events if v['event']=='macro_started']
        assert len(updates)==len(starts)==1 and starts[0]['macro_index']==0 and updates[0]['cursor']=={'macro_updates':1,'cases':16,'logical_pair_tokens':20947}
        assert events[-1]['event']=='paused' and events[-1]['exit_code']==75
        tips=[v for v in events if v['event']=='checkpoint_committed'];tip=tips[-1]['complete'];verify([tip])
        checkpoint=Path(tip['path']).parent;c=read_json(tip['path']);assert c['macro_updates']==1;verify(c['files'])
        saved=load_file(str(checkpoint/'adapter_model.safetensors'),device='cpu')
        assert set(saved)==set(init) and all(torch.equal(saved[k],actual['after'][k]) for k in saved)
        state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
        assert state['cursor']=={'macro_updates':1,'cases':16,'logical_pair_tokens':20947} and state['scheduler']['completed']==1
        assert len(state['optimizer']['state'])==393 and state['dev_done']==[]
        pin_file=ROOT/(arm+'-resume-pin.json');write_json(pin_file,tip);pin_file.chmod(0o444)
        pins[arm]={'checkpoint':str(checkpoint),'complete':tip,'pin_file':str(pin_file)}
        reports[arm]={'actual_393_fp32_before_equal_shared_fresh_init':True,'cached_logits_margins_replay_bitwise_equal':True,
            'actual_cached_macro_loss_and_coefficients_recomputed_exactly':True,'393_raw_and_clipped_gradients_finite':True,
            '393_actual_deltas_and_Adam_states_verified':True,'latest_cursor1_checkpoint_weights_match_actual_after':True,
            'logical_pair_tokens_consumed':20947,'actual_completed_forward_real_tokens':41894,'actual_completed_backward_real_tokens':20947,
            'macro0_consumed_exactly_once':True,'used_lr':meta['used_lr'],'loss':meta['objective'],'preclip_l2':meta['preclip_l2'],
            'runtime':meta['runtime_after'],'resume_pin':tip}
    report={'complete':True,'all_four_formal_macro0_verified':True,'arms':reports,'resume':pins,
        'same_fresh_init_and_same_complete_case_micro_exposure':True,'next_macro_index':1,'repeat_macro0_allowed':False,
        'D_select_opened':False,'D_confirm_opened':False,'public_run':False}
    write_json(ROOT/'FIRST_PRIMARY_VERIFICATION.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('arms','resume')},indent=2),flush=True)


if __name__=='__main__':main()
