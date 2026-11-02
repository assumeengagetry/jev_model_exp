#!/usr/bin/env python3
"""Full four-domain HARD / SOFT / HYBRID continuation from the identical MIXED_ALL parent."""
import argparse
import collections
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, set_seed
from peft import PeftModel

from pair_data_common import DOMAINS
from pair_training_runtime import HOME, POOL, PairPool, TokenStore, collate, microbatches


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True)
    args=parser.parse_args()
    args.model=args.arm
    cfg=json.loads((HOME/'config/experiment-four-domain-distillation.json').read_text())
    if args.arm not in cfg['arms']:raise RuntimeError('Unknown distillation arm')
    if cfg['active_training_domains']!=['knowledge','retrieval','tools','arts']:
        raise RuntimeError('This approved continuation must use the exact four active domains')
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('gloo',timeout=timedelta(minutes=60))
    set_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=True
    out=HOME/cfg['run_root']/args.model;out.mkdir(parents=True,exist_ok=True)
    dataset=PairPool('MIXED_ALL',domains=cfg['active_training_domains']);store=TokenStore()
    if len(dataset)!=cfg['pairs']:raise RuntimeError('Four-domain exposure differs from the approved scope')
    teacher=None
    if args.arm!='KD_HARD':
        ready=json.loads((HOME/cfg['run_root']/'TEACHER_LABELS_READY.json').read_text())
        if ready.get('complete') is not True:raise RuntimeError('Complete Teacher probabilities are required')
        teacher=np.load(HOME/cfg['run_root']/'teacher-labels/prob-match.npy',mmap_mode='r')
        if teacher.shape!=(len(dataset),):raise RuntimeError('Teacher probability / pool index mismatch')
    if not len(dataset):raise RuntimeError('Empty formal training pool is not allowed')
    completion=out/'complete.json'
    if completion.exists():
        previous=json.loads(completion.read_text())
        if previous.get('pairs_seen')!=len(dataset):raise RuntimeError('Existing result belongs to a different pool; preserved')
        if previous.get('student_parent')!=cfg['student_parent']:raise RuntimeError('Existing result has a different Student parent; preserved')
        if rank==0:print('Reusing completed full epoch:',args.model,flush=True)
        if world>1:dist.destroy_process_group()
        return
    latest=out/'latest.json';checkpoint=None;state=None
    if latest.exists():
        checkpoint=out/json.loads(latest.read_text())['path']
        state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
        if state['model']!=args.model or state['pool_pairs']!=len(dataset) or state.get('student_parent')!=cfg['student_parent']:
            raise RuntimeError('Checkpoint / pool / common Student parent mismatch; preserved')
    common=HOME/cfg['common_initialization'];base=HOME/cfg['base_model'];parent=HOME/cfg['student_parent']
    model_config=AutoConfig.from_pretrained(common,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(base,config=model_config,torch_dtype=torch.bfloat16,
                                                            attn_implementation='sdpa',local_files_only=True)
    model=PeftModel.from_pretrained(model,checkpoint or parent,is_trainable=True)
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
    warmup=max(1,int(total_steps*cfg['warmup_ratio']))
    def schedule(step):
        if step<warmup:return max(1e-8,step/warmup)
        progress=min(1.0,(step-warmup)/max(1,total_steps-warmup))
        return 0.5*(1.0+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    completed=0;pairs_seen=0;tokens_done=0;prior_seconds=0.0;seen_domains=collections.Counter();micro_tokens=cfg['max_microbatch_tokens']
    if state:
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        completed=state['step'];pairs_seen=state['pairs_seen'];tokens_done=state['tokens_done'];prior_seconds=state['elapsed_seconds']
        seen_domains.update(state.get('seen_domains',{}));micro_tokens=state.get('micro_tokens',micro_tokens)
        if state.get('torch_rng') is not None:torch.set_rng_state(state['torch_rng'])
        if state.get('cuda_rng') is not None:torch.cuda.set_rng_state(state['cuda_rng'],device)
        del state
    # One reusable flat FP32 buffer makes the four-rank SUM exact for a global
    # macro-batch, including variable microbatch counts and uneven final tails.
    sizes=[parameter.numel() for parameter in parameters];offsets=np.cumsum([0]+sizes)
    gradients=torch.zeros(int(offsets[-1]),device=device,dtype=torch.float32)
    host_gradients=torch.empty(int(offsets[-1]),device='cpu',dtype=torch.float32,pin_memory=True) if world>1 else None
    for index,parameter in enumerate(parameters):parameter.grad=gradients[offsets[index]:offsets[index+1]].view_as(parameter)
    failed=torch.zeros(1,device='cpu',dtype=torch.int32)
    order=dataset.order(cfg['seed'],batch_size=macro)
    if rank==0:
        print(json.dumps(dict(event='formal_training_process_ready',model=args.model,pairs=len(dataset),
                              optimizer_steps=total_steps,resume_step=completed,world_size=world,
                               global_effective_batch=macro,epochs=1,pair_cap=None,token_cap=None,
                               student_parent=cfg['student_parent'],active_training_domains=dataset.domains,
                               supervision=args.arm,communication='gloo_cpu_float32_adapter_gradients')),flush=True)
    started=time.monotonic();last_log=started;last_tokens=tokens_done;loss_window=torch.zeros(1,device=device,dtype=torch.float64)
    log_pairs=0;oom_retries=0;checkpointing=resume_checkpointing

    def save(step,final=False):
        if world>1:dist.barrier()
        if rank==0:
            folder=out/'checkpoints'/f'step-{step:09d}';folder.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(folder,safe_serialization=True);model.config.save_pretrained(folder)
            payload=dict(model=args.model,student_parent=cfg['student_parent'],pool_pairs=len(dataset),step=step,pairs_seen=pairs_seen,tokens_done=tokens_done,
                         elapsed_seconds=prior_seconds+time.monotonic()-started,seen_domains=dict(seen_domains),
                          micro_tokens=micro_tokens,gradient_checkpointing=checkpointing,
                          optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device))
            torch.save(payload,folder/'training_state.pt')
            metadata=dict(owner='four-domain-kd-v1',model=args.model,step=step,pool_pairs=len(dataset))
            (folder/'checkpoint.json').write_text(json.dumps(metadata))
            temporary=out/'latest.json.tmp';temporary.write_text(json.dumps({'path':str(folder.relative_to(out)),'step':step}));temporary.replace(latest)
            old=sorted((out/'checkpoints').glob('step-*'))[:-2]
            for path in old:
                marker=path/'checkpoint.json'
                if marker.exists() and json.loads(marker.read_text()).get('owner')=='four-domain-kd-v1':shutil.rmtree(path)
            if final:
                final_dir=out/'final';model.save_pretrained(final_dir,safe_serialization=True);model.config.save_pretrained(final_dir)
                summary=dict(model=args.model,complete=True,pool_pairs=len(dataset),pairs_seen=pairs_seen,epochs=1,
                             effective_tokens=tokens_done,elapsed_seconds=payload['elapsed_seconds'],
                             effective_tokens_per_second=tokens_done/max(1,payload['elapsed_seconds']),
                             seen_domains=dict(seen_domains),optimizer_steps=step,global_effective_batch=macro,
                             student_parent=cfg['student_parent'],active_training_domains=dataset.domains,
                             supervision=args.arm,learning_rate=cfg['learning_rate'],seed=cfg['seed'],
                             original_supervision_weight=1.0 if args.arm=='KD_HARD' else 0.0 if args.arm=='KD_SOFT' else 0.5,
                             teacher_supervision_weight=0.0 if args.arm=='KD_HARD' else 1.0 if args.arm=='KD_SOFT' else 0.5,
                             common_initialization=str(common),precision='bf16',attention='causal_sdpa',
                              micro_token_budget_final=micro_tokens,oom_retries=oom_retries,
                              gradient_checkpointing=checkpointing,communication='gloo_cpu_float32_adapter_gradients',
                             peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1024**3,
                             training_pair_cap=None,training_token_cap=None)
                completion.write_text(json.dumps(summary,indent=2));print(json.dumps(dict(event='formal_epoch_complete',**summary)),flush=True)
        if world>1:dist.barrier()

    for step in range(completed,total_steps):
        indices=order[step*macro:min((step+1)*macro,len(dataset))]
        global_batch=dataset.fetch(indices);local_batch={key:value[rank::world] for key,value in global_batch.items()}
        if teacher is not None:
            probabilities=np.asarray(teacher[indices[rank::world]],dtype=np.float32).copy()
            if not np.all(np.isfinite(probabilities)) or np.any((probabilities<0)|(probabilities>1)):
                raise RuntimeError('Teacher probabilities must be complete and finite; no pairs dropped')
            local_batch['teacher']=probabilities
        while True:
            gradients.zero_();failed.zero_();local_loss=torch.zeros(1,device=device,dtype=torch.float64)
            inputs=targets=output=loss=margin=q=kl=entropy=None
            try:
                for micro in microbatches(local_batch,cfg['max_microbatch_pairs'],micro_tokens):
                    inputs,targets,_=collate(store,micro,pad,eos,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        output=model(**inputs).logits
                        margin=output[:,1].float()-output[:,0].float()
                        if args.arm=='KD_HARD':
                            loss=F.binary_cross_entropy_with_logits(margin,targets,reduction='sum')
                        else:
                            q=torch.from_numpy(np.asarray(micro['teacher'],dtype=np.float32).copy()).to(device)
                            entropy=-torch.special.xlogy(q,q)-torch.special.xlogy(1-q,1-q)
                            kl=(F.binary_cross_entropy_with_logits(margin,q,reduction='none')-entropy).sum()
                            loss=kl if args.arm=='KD_SOFT' else (1-cfg['hybrid_teacher_weight'])*F.binary_cross_entropy_with_logits(margin,targets,reduction='sum')+cfg['hybrid_teacher_weight']*kl
                    (loss/len(indices)).backward();local_loss+=loss.detach().double()
                    inputs=targets=output=loss=margin=q=kl=entropy=None
            except torch.cuda.OutOfMemoryError:
                failed.fill_(1);inputs=targets=output=loss=margin=q=kl=entropy=None;gc.collect();torch.cuda.empty_cache()
            if world>1:dist.all_reduce(failed,op=dist.ReduceOp.MAX)
            if failed.item():
                gradients.zero_();oom_retries+=1
                if micro_tokens>4096:micro_tokens=max(4096,micro_tokens//2)
                elif not checkpointing:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});checkpointing=True
                else:raise RuntimeError('Actual full training is blocked by memory even at one long sample; no pairs dropped')
                if rank==0:print(json.dumps(dict(event='actual_training_oom_retry',step=step,micro_tokens=micro_tokens,checkpointing=checkpointing)),flush=True)
                continue
            break
        if world>1:
            host_gradients.copy_(gradients,non_blocking=True)
            torch.cuda.synchronize(device)
            dist.all_reduce(host_gradients,op=dist.ReduceOp.SUM)
            gradients.copy_(host_gradients,non_blocking=True)
        torch.nn.utils.clip_grad_norm_(parameters,1.0,foreach=True)
        optimizer.step();scheduler.step()
        pairs_seen+=len(indices);tokens_done+=int(global_batch['length'].sum());loss_window+=local_loss;log_pairs+=len(indices)
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
                                            effective_tokens_per_second=(tokens_done-last_tokens)/max(0.001,now-last_log),
                                            elapsed_seconds=prior_seconds+now-started)),flush=True)
            loss_window.zero_();log_pairs=0;last_log=now;last_tokens=tokens_done
        if (step+1)%cfg['checkpoint_steps']==0:save(step+1)
    save(total_steps,final=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
