#!/usr/bin/env python3
"""One approved five-domain E24 extension; preserve the strict numerical loop."""
import argparse
import collections
from datetime import timedelta
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, set_seed
from peft import PeftModel
EXP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, '/home/dev/assumeengage/scripts')
from student_flash_attention3_repro import (
    BACKEND, register_student_fa3, assert_student_uses_fa3, attention_record,
    restore_attention_record,
)

from pair_data_common import DOMAINS
from pair_training_runtime import HOME, POOL, PairPool, TokenStore, collate, microbatches
from group_training_runtime import load_plan, atomic_microbatches


def atomic_json(path, value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.writing')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def process_identity(pid):
    try:
        folder=Path('/proc')/str(pid)
        fields=(folder/'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z':return None
        return {'starttime':fields[19],'command':(folder/'cmdline').read_bytes().split(b'\0')}
    except (OSError,IndexError):return None


def claim_arm(cfg,arm,out,world):
    lock=(out/'TRAINING.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise RuntimeError('Another trainer owns this run/arm; no duplicate GPU work started')
    path=out/'OWNER.json'
    try:
        if path.exists():
            previous=json.loads(path.read_text())
            if previous.get('arm')!=arm or previous.get('student_parent')!=cfg['student_parent']:
                raise RuntimeError('Existing arm ownership belongs to another continuation; preserved')
            live=process_identity(previous.get('worker_pid'))
            if live and live['starttime']==previous.get('worker_starttime') and previous.get('worker_pid')!=os.getpid():
                raise RuntimeError('An already running trainer is recorded for this arm; no duplicate started')
        own=process_identity(os.getpid());coordinator=os.getpid();coordinator_info=own
        parent=process_identity(os.getppid())
        if world>1 and parent and str(Path(__file__).absolute()).encode() in parent['command']:
            coordinator=os.getppid();coordinator_info=parent
        record={'arm':arm,'student_parent':cfg['student_parent'],
                'active_training_domains':cfg['active_training_domains'],'pool_pairs':cfg['pairs'],
                'worker_pid':os.getpid(),'worker_starttime':own['starttime'],
                'coordinator_pid':coordinator,'coordinator_starttime':coordinator_info['starttime'],
                'world_size':world,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'lock_holder_pid':os.getpid(),'lock_mode':'worker_owned',
                'command':[piece.decode() for piece in own['command'] if piece]}
        temporary=path.with_suffix('.writing');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)
    except BaseException:
        lock.close();raise
    return lock


def connect_checkpoint_storage(cfg,out):
    target=Path(cfg['checkpoint_storage_root'])
    expected=EXP_ROOT/'checkpoints'
    if target!=expected or not target.is_absolute():raise RuntimeError('Unexpected scratch checkpoint namespace')
    owner=target.parent/'STORAGE_OWNER.json'
    if not owner.is_file() or json.loads(owner.read_text())!=cfg['checkpoint_storage_identity']:
        raise RuntimeError('Scratch checkpoint storage is not admitted for this trial')
    if not target.is_dir() or target.is_symlink():raise RuntimeError('Scratch checkpoint directory differs')
    link=out/'checkpoints'
    if link.is_symlink():
        if link.resolve()!=target.resolve():raise RuntimeError('Existing checkpoint symlink belongs to another location')
    elif link.exists():raise RuntimeError('Unfamiliar local checkpoint directory preserved')
    else:link.symlink_to(target,target_is_directory=True)


def group_micro_loss(margin,micro,teacher_targets,cfg,device):
    point_target=torch.from_numpy(np.asarray(micro['distillation_target'],dtype=np.float32).copy()).to(device)
    point=F.binary_cross_entropy_with_logits(margin,point_target,reduction='sum')
    tags=micro['relation_tag'];positive=np.flatnonzero((tags>=0)&(tags%2==0))
    if cfg['relation_target_kind']=='point' or not len(positive):return point
    if np.any(positive+1>=len(tags)) or not np.array_equal(tags[positive+1],tags[positive]+1):
        raise RuntimeError('A relation was split across actual microbatches')
    relation_ids=tags[positive]//2
    kind=cfg['relation_target_kind']
    if kind=='source':target=np.ones(len(positive),dtype=np.float32)
    else:
        target=np.asarray(teacher_targets[relation_ids],dtype=np.float32).copy()
        if kind=='mixed':
            gamma=cfg.get('relative_teacher_fraction',0.5)
            # Retain exactly the old arithmetic for the replay/default .5 path.
            target=0.5+0.5*target if gamma==0.5 else (1-gamma)+gamma*target
    locations=torch.from_numpy(positive.astype(np.int64)).to(device)
    relative_margin=margin[locations]-margin[locations+1]
    relative_target=torch.from_numpy(target).to(device)
    relation_loss=F.binary_cross_entropy_with_logits(relative_margin,relative_target,reduction='sum')
    # Caller divides the whole sum by the ACTUAL global macro size, not by
    # micro rows, local relation count, or configured full batch 128.
    return point+cfg['relation_lambda']*cfg['pairs']/cfg['relation_count']*relation_loss


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True)
    args=parser.parse_args()
    args.model=args.arm
    if (EXP_ROOT/'NO_AUTOMATIC_RESUME').exists():
        raise RuntimeError('This five-domain experiment was paused; no automatic resume authorized')
    cfg=json.loads((HOME/os.environ.get('JEV_FRESH_DISTILL_CONFIG',str(EXP_ROOT/'config.json'))).read_text())
    if cfg.get('attention')!='flash_attention_3' or cfg.get('student_attention_implementation')!=BACKEND:
        raise RuntimeError('Student training requires explicit FlashAttention-3; SDPA fallback is forbidden')
    if cfg['student_parent']!=cfg['common_initialization'] or not 1250000<=cfg['pairs']<1260000:
        raise RuntimeError('Five-domain extension requires untrained initialization and admitted native-language exposure')
    if args.arm not in cfg['arms']:raise RuntimeError('Unknown distillation arm')
    kind=cfg.get('relation_target_kind')
    recipe=(cfg.get('relative_teacher_fraction',0.5),cfg.get('relation_lambda'))
    data_ready=json.loads((EXP_ROOT/'DATA_READY.json').read_text())
    if (kind!='mixed' or recipe!=(0.5,0.05) or cfg['run_root']!=str(EXP_ROOT)
        or data_ready.get('complete') is not True or cfg['pairs']!=data_ready.get('selected_pairs')
        or args.arm!='KD_HYBRID' or cfg.get('optimizer_steps')!=math.ceil(cfg['pairs']/128)
        or cfg.get('relation_count')!=3647
        or cfg.get('learning_rate')!=1e-4 or cfg.get('global_effective_batch_pairs')!=128):
        raise RuntimeError('Only the approved five-domain E24 extension may use this entry')
    controls=cfg.get('numerical_controls',{})
    if (controls.get('torch_deterministic_algorithms') is not True or controls.get('warn_only') is not False
        or controls.get('fa3_backward_deterministic') is not True
        or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'
        or os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1'):
        raise RuntimeError('Strict numerical controls must be admitted and set before CUDA startup')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if cfg['active_training_domains']!=['knowledge','retrieval','tools','arts','language']:
        raise RuntimeError('Five-domain identity differs; old four-domain cache indices must remain unchanged')
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=1 or rank!=0 or local!=0:raise RuntimeError('Atomic group trials require independent single-GPU processes')
    out=HOME/cfg['run_root']/args.model;out.mkdir(parents=True,exist_ok=True)
    # Publish ownership before CUDA/model initialization. The reference keeps the
    # rank-zero arm lock alive until this worker has fully left main().
    arm_lock=claim_arm(cfg,args.arm,out,world) if rank==0 else None
    connect_checkpoint_storage(cfg,out)
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('gloo',timeout=timedelta(minutes=60))
    set_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=True
    dataset=PairPool('MIXED_ALL',root=HOME/cfg['training_pool'],domains=cfg['active_training_domains']);store=TokenStore()
    if len(dataset)!=cfg['pairs']:raise RuntimeError('Five-domain exposure differs from the approved scope')
    ready=json.loads((HOME/cfg['effective_target_ready_path']).read_text())
    if ready.get('complete') is not True:raise RuntimeError('Prepared exploration targets are required')
    if (ready.get('student_parent')!=cfg['student_parent'] or ready.get('training_pool')!=cfg['training_pool']
        or ready.get('exploration_policy')!=cfg['exploration_policy'] or ready.get('pairs')!=len(dataset)):
        raise RuntimeError('Exploration target identity differs from this trial')
    effective_targets=np.load(HOME/cfg['effective_target_path'],mmap_mode='r')
    if effective_targets.shape!=(len(dataset),) or effective_targets.dtype!=np.dtype('float32'):
        raise RuntimeError('Effective target / selected-pool index or dtype mismatch')
    if ready.get('sampler_identity')!=cfg['group_plan_identity'] or ready.get('effective_target_path')!=cfg['effective_target_path']:
        raise RuntimeError('Point targets are not admitted for this shared group plan')
    plan=load_plan(HOME/cfg['group_plan_root'],cfg['group_plan_identity'])
    if not len(dataset):raise RuntimeError('Empty formal training pool is not allowed')
    completion=out/'complete.json'
    previous=None
    if rank==0 and completion.exists():
        try:previous=json.loads(completion.read_text())
        except (ValueError,UnicodeError):previous=None
        if not isinstance(previous,dict) or previous.get('complete') is not True:
            recovery=out/'recovery';recovery.mkdir(exist_ok=True)
            completion.rename(recovery/f'complete-unfinished-{time.time_ns()}.json')
            print(json.dumps(dict(event='incomplete_completion_preserved',model=args.model,
                                  action='resume owned checkpoint and final export')),flush=True)
            previous=None
    if world>1:
        shared=[previous];dist.broadcast_object_list(shared,src=0);previous=shared[0]
    if isinstance(previous,dict) and previous.get('complete') is True:
        if previous.get('pairs_seen')!=len(dataset):raise RuntimeError('Existing result belongs to a different pool; preserved')
        if previous.get('student_parent')!=cfg['student_parent']:raise RuntimeError('Existing result has a different Student parent; preserved')
        if previous.get('exploration_policy')!=cfg['exploration_policy']:raise RuntimeError('Existing result has a different distillation policy; preserved')
        if rank==0:print('Reusing completed full epoch:',args.model,flush=True)
        if world>1:dist.destroy_process_group()
        return
    latest=out/'latest.json';checkpoint=None;state=None
    if latest.exists():
        checkpoint=out/json.loads(latest.read_text())['path']
        state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
        if state['model']!=args.model or state['pool_pairs']!=len(dataset) or state.get('student_parent')!=cfg['student_parent']:
            raise RuntimeError('Checkpoint / pool / common Student parent mismatch; preserved')
        if state.get('exploration_policy')!=cfg['exploration_policy']:
            raise RuntimeError('Checkpoint belongs to another exploration policy; preserved')
    common=HOME/cfg['common_initialization'];base=HOME/cfg['base_model'];parent=HOME/cfg['student_parent']
    register_student_fa3(HOME/cfg['student_attention_kernel_directory'])
    model_config=AutoConfig.from_pretrained(common,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(base,config=model_config,torch_dtype=torch.bfloat16,
                                                            attn_implementation="flash_attn3_hopper",local_files_only=True)
    model=PeftModel.from_pretrained(model,checkpoint or parent,is_trainable=True)
    assert_student_uses_fa3(model)
    model.config.use_cache=False
    for parameter in model.parameters():
        if parameter.requires_grad:parameter.data=parameter.data.float()
    model.to(device);model.train()
    resume_checkpointing=cfg['gradient_checkpointing'] or bool(state and state.get('gradient_checkpointing'))
    if resume_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    tokenizer=AutoTokenizer.from_pretrained(common,local_files_only=True)
    pad,eos=tokenizer.pad_token_id,tokenizer.eos_token_id
    parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=True)
    macro=cfg['global_effective_batch_pairs'];total_steps=math.ceil(len(dataset)/macro)
    if total_steps!=cfg['optimizer_steps']:raise RuntimeError('Group plan optimizer exposure differs')
    warmup=max(1,int(total_steps*cfg['warmup_ratio']))
    def schedule(step):
        if step<warmup:return max(1e-8,step/warmup)
        progress=min(1.0,(step-warmup)/max(1,total_steps-warmup))
        return 0.5*(1.0+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    completed=0;pairs_seen=0;tokens_done=0;prior_seconds=0.0;seen_domains=collections.Counter();micro_tokens=cfg['max_microbatch_tokens']
    seen_relations=0
    if state:
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        completed=state['step'];pairs_seen=state['pairs_seen'];tokens_done=state['tokens_done'];prior_seconds=state['elapsed_seconds']
        seen_domains.update(state.get('seen_domains',{}));micro_tokens=state.get('micro_tokens',micro_tokens)
        if state.get('sampler_identity')!=cfg['group_plan_identity']:raise RuntimeError('Checkpoint sampler plan differs')
        seen_relations=state.get('seen_relations',0)
        restore_attention_record(state['attention_evidence'])
        if state.get('torch_rng') is not None:torch.set_rng_state(state['torch_rng'])
        if state.get('cuda_rng') is not None:torch.cuda.set_rng_state(state['cuda_rng'],device)
        del state
    if completed>total_steps or (completed==total_steps and pairs_seen!=len(dataset)):
        raise RuntimeError('Checkpoint final-step counters differ from the selected milestone')
    # One reusable flat FP32 buffer makes the four-rank SUM exact for a global
    # macro-batch, including variable microbatch counts and uneven final tails.
    sizes=[parameter.numel() for parameter in parameters];offsets=np.cumsum([0]+sizes)
    gradients=torch.zeros(int(offsets[-1]),device=device,dtype=torch.float32)
    host_gradients=torch.empty(int(offsets[-1]),device='cpu',dtype=torch.float32,pin_memory=True) if world>1 else None
    for index,parameter in enumerate(parameters):parameter.grad=gradients[offsets[index]:offsets[index+1]].view_as(parameter)
    failed=torch.zeros(1,device='cpu',dtype=torch.int32)
    order=plan['order'];relation_tags=plan['relation_tag'];offsets=plan['macro_offsets']
    expected_relations=int(np.count_nonzero((relation_tags[:int(offsets[completed])]>=0)&(relation_tags[:int(offsets[completed])]%2==0)))
    if seen_relations!=expected_relations:raise RuntimeError('Restored relation exposure differs from committed plan cursor')
    if rank==0:
        print(json.dumps(dict(event='formal_training_process_ready',model=args.model,pairs=len(dataset),
                              optimizer_steps=total_steps,resume_step=completed,world_size=world,
                                global_effective_batch=macro,epochs=1,pair_cap=cfg['pairs'],token_cap=None,
                               student_parent=cfg['student_parent'],active_training_domains=dataset.domains,
                                supervision=args.arm,exploration_policy=cfg['exploration_policy'],
                                 loss_implementation='E06 point BCE plus optional globally normalized atomic relation BCE',
                                 sampler_identity=cfg['group_plan_identity'],
                                 relation_target_kind=kind,relation_lambda=cfg['relation_lambda'],
                                 relative_teacher_fraction=cfg.get('relative_teacher_fraction',0.5),
                                 total_relations=cfg['relation_count'],
                                 checkpoint_storage_root=cfg['checkpoint_storage_root'],
                                communication='gloo_cpu_float32_adapter_gradients',
                               **attention_record())),flush=True)
    started=time.monotonic();last_log=started;last_tokens=tokens_done;loss_window=torch.zeros(1,device=device,dtype=torch.float64)
    log_pairs=0;oom_retries=0;checkpointing=resume_checkpointing
    committed_checkpoint=checkpoint
    committed_step=completed if checkpoint is not None else None
    committed_elapsed=prior_seconds

    def save(step,final=False):
        nonlocal committed_checkpoint,committed_step,committed_elapsed
        if world>1:dist.barrier()
        if rank==0:
            folder=out/'checkpoints'/f'step-{step:09d}'
            reusing_committed=committed_checkpoint==folder and committed_step==step
            elapsed=committed_elapsed if reusing_committed else prior_seconds+time.monotonic()-started
            if not reusing_committed:
                folder.mkdir(parents=True,exist_ok=True)
                model.save_pretrained(folder,safe_serialization=True);model.config.save_pretrained(folder)
                payload=dict(model=args.model,student_parent=cfg['student_parent'],pool_pairs=len(dataset),step=step,pairs_seen=pairs_seen,tokens_done=tokens_done,
                              exploration_policy=cfg['exploration_policy'],
                              sampler_identity=cfg['group_plan_identity'],seen_relations=seen_relations,
                             elapsed_seconds=elapsed,seen_domains=dict(seen_domains),
                              micro_tokens=micro_tokens,gradient_checkpointing=checkpointing,
                               optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                               attention_evidence=attention_record(),
                             torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device))
                temporary=folder/'training_state.pt.writing'
                torch.save(payload,temporary);temporary.replace(folder/'training_state.pt')
                metadata=dict(owner='five-domain-E24-20261109',model=args.model,step=step,pool_pairs=len(dataset))
                atomic_json(folder/'checkpoint.json',metadata)
                atomic_json(latest,{'path':str(folder.relative_to(out)),'step':step})
                committed_checkpoint=folder;committed_step=step;committed_elapsed=elapsed
                # Preserve all checkpoints; this experiment is not authorized to clean old artifacts.
            else:
                print(json.dumps(dict(event='committed_checkpoint_reused_for_final_export',
                                      model=args.model,step=step,checkpoint=str(folder))),flush=True)
            if final:
                if (pairs_seen!=len(dataset) or tokens_done!=cfg['effective_student_tokens']
                    or seen_relations!=cfg['relation_count']):
                    raise RuntimeError('Final full-pool or relation exposure differs; no completion published')
                final_dir=out/'final';model.save_pretrained(final_dir,safe_serialization=True);model.config.save_pretrained(final_dir)
                tokenizer.save_pretrained(final_dir)
                summary=dict(model=args.model,complete=True,pool_pairs=len(dataset),pairs_seen=pairs_seen,epochs=1,
                             effective_tokens=tokens_done,elapsed_seconds=elapsed,
                             effective_tokens_per_second=tokens_done/max(1,elapsed),
                             seen_domains=dict(seen_domains),optimizer_steps=step,global_effective_batch=macro,
                             student_parent=cfg['student_parent'],active_training_domains=dataset.domains,
                              supervision=args.arm,learning_rate=cfg['learning_rate'],seed=cfg['seed'],
                               exploration_policy=cfg['exploration_policy'],
                                loss_implementation='E06 point BCE plus optional globally normalized atomic relation BCE',
                                sampler_identity=cfg['group_plan_identity'],relations_seen=seen_relations,
                                relation_target_kind=kind,relation_lambda=cfg['relation_lambda'],
                                relative_teacher_fraction=cfg.get('relative_teacher_fraction',0.5),
                                checkpoint_storage_root=cfg['checkpoint_storage_root'],
                               mixing_statistics=ready['mixing_statistics'],
                             common_initialization=str(common),precision='bf16',**attention_record(),
                              micro_token_budget_final=micro_tokens,oom_retries=oom_retries,
                              gradient_checkpointing=checkpointing,communication='gloo_cpu_float32_adapter_gradients',
                             peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1024**3,
                               training_pair_cap=cfg['pairs'],training_token_cap=None,
                               milestone_pairs=cfg['pairs'],student_previously_finetuned=False,
                               selection_metadata=str(HOME/cfg['training_pool']/'SELECTION.json'))
                atomic_json(completion,summary);print(json.dumps(dict(event='formal_epoch_complete',**summary)),flush=True)
        if world>1:dist.barrier()

    for step in range(completed,total_steps):
        begin,end=int(offsets[step]),int(offsets[step+1]);indices=order[begin:end]
        global_batch=dataset.fetch(indices);local_batch={key:value[rank::world] for key,value in global_batch.items()}
        probabilities=np.asarray(effective_targets[indices[rank::world]],dtype=np.float32).copy()
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities<0)|(probabilities>1)):
            raise RuntimeError('Effective targets must be complete and finite; no pairs dropped')
        local_batch['distillation_target']=probabilities
        local_batch['relation_tag']=np.asarray(relation_tags[begin:end],dtype=np.int32)
        macro_relations=int(np.count_nonzero((local_batch['relation_tag']>=0)&(local_batch['relation_tag']%2==0)))
        while True:
            gradients.zero_();failed.zero_();local_loss=torch.zeros(1,device=device,dtype=torch.float64)
            inputs=targets=output=loss=margin=q=kl=entropy=None
            try:
                for micro in atomic_microbatches(local_batch,cfg['max_microbatch_pairs'],micro_tokens):
                    padded_width=((int(micro['length'].max())+7)//8)*8
                    if padded_width*len(micro['state'])>micro_tokens:
                        print(json.dumps(dict(event='atomic_unit_over_micro_budget',step=step,
                            rows=len(micro['state']),padded_tokens=padded_width*len(micro['state']),
                            budget=micro_tokens)),flush=True)
                    inputs,targets,_=collate(store,micro,pad,eos,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        output=model(**inputs).logits
                        margin=output[:,1].float()-output[:,0].float()
                        loss=group_micro_loss(margin,micro,plan['teacher_relation_target'],cfg,device)
                    (loss/len(indices)).backward();local_loss+=loss.detach().double()
                    inputs=targets=output=loss=margin=q=kl=entropy=None
            except torch.cuda.OutOfMemoryError:
                failed.fill_(1);inputs=targets=output=loss=margin=q=kl=entropy=None;gc.collect();torch.cuda.empty_cache()
            if world>1:dist.all_reduce(failed,op=dist.ReduceOp.MAX)
            if failed.item():
                gradients.zero_();oom_retries+=1
                if micro_tokens>8192:micro_tokens=max(8192,micro_tokens//2)
                elif not checkpointing:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});checkpointing=True
                else:raise RuntimeError('An intact relation cannot fit actual training; no groups split or samples dropped')
                if rank==0:print(json.dumps(dict(event='actual_training_oom_retry',step=step,micro_tokens=micro_tokens,checkpointing=checkpointing)),flush=True)
                continue
            break
        if world>1:
            host_gradients.copy_(gradients,non_blocking=True)
            torch.cuda.synchronize(device)
            dist.all_reduce(host_gradients,op=dist.ReduceOp.SUM)
            gradients.copy_(host_gradients,non_blocking=True)
        grad_norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,foreach=True)
        optimizer.step();scheduler.step()
        pairs_seen+=len(indices);tokens_done+=int(global_batch['length'].sum());loss_window+=local_loss;log_pairs+=len(indices)
        seen_relations+=macro_relations
        if step<5 or (step+1)%1000==0:
            print(json.dumps(dict(event='repro_optimizer_trace',step=step+1,pairs_seen=pairs_seen,
                effective_tokens=tokens_done,relations_seen=seen_relations,
                actual_macro_loss=local_loss.item()/len(indices),clip_before_gradient_norm=grad_norm.item(),
                learning_rate=scheduler.get_last_lr()[0],relation_target_kind=kind,
                micro_token_budget=micro_tokens,gradient_checkpointing=checkpointing,
                **attention_record())),flush=True)
        for value,count in zip(*np.unique(global_batch['domain'],return_counts=True)):seen_domains[DOMAINS[int(value)]]+=int(count)
        if step==0 or (step+1)%20==0 or step+1==total_steps:
            if world>1:
                host_loss=loss_window.cpu()
                dist.all_reduce(host_loss,op=dist.ReduceOp.SUM)
                loss_window.copy_(host_loss)
            torch.cuda.synchronize(device);now=time.monotonic()
            if rank==0:print(json.dumps(dict(event='formal_optimizer_progress',model=args.model,step=step+1,total_steps=total_steps,
                                            pairs_seen=pairs_seen,effective_tokens=tokens_done,
                                             loss=loss_window.item()/max(1,log_pairs),learning_rate=scheduler.get_last_lr()[0],
                                             relations_seen=seen_relations,total_relations=cfg['relation_count'],
                                             macro_relations=macro_relations,relation_target_kind=kind,
                                             relative_teacher_fraction=cfg.get('relative_teacher_fraction',0.5),
                                             relation_lambda=cfg['relation_lambda'],
                                            effective_tokens_per_second=(tokens_done-last_tokens)/max(0.001,now-last_log),
                                            elapsed_seconds=prior_seconds+now-started,**attention_record())),flush=True)
            loss_window.zero_();log_pairs=0;last_log=now;last_tokens=tokens_done
        if (step+1)%cfg['checkpoint_steps']==0:save(step+1)
    save(total_steps,final=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
