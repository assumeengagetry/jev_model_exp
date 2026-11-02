#!/usr/bin/env python3
"""Full-epoch takeover trainer; CPU Gloo sums avoid this Pod's 64MiB NCCL IPC limit."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True)
    args=parser.parse_args()
    cfg=json.loads((HOME/'config/experiment-five-specialists.json').read_text())
    if args.model not in cfg['models']:raise RuntimeError('Unknown formal model name')
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('gloo',timeout=timedelta(minutes=60))
    set_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=True
    out=HOME/cfg['run_root']/args.model;out.mkdir(parents=True,exist_ok=True)
    dataset=PairPool(args.model,domains=cfg.get('active_training_domains'));store=TokenStore()
    if not len(dataset):raise RuntimeError('Empty formal training pool is not allowed')
    completion=out/'complete.json'
    if completion.exists():
        previous=json.loads(completion.read_text())
        if previous.get('pairs_seen')!=len(dataset):raise RuntimeError('Existing result belongs to a different pool; preserved')
        if rank==0:print('Reusing completed full epoch:',args.model,flush=True)
        if world>1:dist.destroy_process_group()
        return
    latest=out/'latest.json';checkpoint=None;state=None
    if latest.exists():
        checkpoint=out/json.loads(latest.read_text())['path']
        state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
        if state['model']!=args.model or state['pool_pairs']!=len(dataset):raise RuntimeError('Checkpoint/pool mismatch; existing checkpoint preserved')
    common=HOME/cfg['common_initialization'];base=HOME/cfg['base_model']
    model_config=AutoConfig.from_pretrained(common,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(base,config=model_config,torch_dtype=torch.bfloat16,
                                                            attn_implementation='sdpa',local_files_only=True)
    model=PeftModel.from_pretrained(model,checkpoint or common,is_trainable=True)
    model.config.use_cache=False
    for parameter in model.parameters():
        if parameter.requires_grad:parameter.data=parameter.data.float()
    model.to(device);model.train()
    if cfg['gradient_checkpointing']:
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
                               communication='gloo_cpu_float32_adapter_gradients')),flush=True)
    started=time.monotonic();last_log=started;last_tokens=tokens_done;loss_window=torch.zeros(1,device=device,dtype=torch.float64)
    log_pairs=0;oom_retries=0;checkpointing=cfg['gradient_checkpointing']

    def save(step,final=False):
        if world>1:dist.barrier()
        if rank==0:
            folder=out/'checkpoints'/f'step-{step:09d}';folder.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(folder,safe_serialization=True);model.config.save_pretrained(folder)
            payload=dict(model=args.model,pool_pairs=len(dataset),step=step,pairs_seen=pairs_seen,tokens_done=tokens_done,
                         elapsed_seconds=prior_seconds+time.monotonic()-started,seen_domains=dict(seen_domains),
                          micro_tokens=micro_tokens,gradient_checkpointing=checkpointing,
                          optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device))
            torch.save(payload,folder/'training_state.pt')
            metadata=dict(owner='formal-five-specialists-v1',model=args.model,step=step,pool_pairs=len(dataset))
            (folder/'checkpoint.json').write_text(json.dumps(metadata))
            temporary=out/'latest.json.tmp';temporary.write_text(json.dumps({'path':str(folder.relative_to(out)),'step':step}));temporary.replace(latest)
            old=sorted((out/'checkpoints').glob('step-*'))[:-2]
            for path in old:
                marker=path/'checkpoint.json'
                if marker.exists() and json.loads(marker.read_text()).get('owner')=='formal-five-specialists-v1':shutil.rmtree(path)
            if final:
                final_dir=out/'final';model.save_pretrained(final_dir,safe_serialization=True);model.config.save_pretrained(final_dir)
                summary=dict(model=args.model,complete=True,pool_pairs=len(dataset),pairs_seen=pairs_seen,epochs=1,
                             effective_tokens=tokens_done,elapsed_seconds=payload['elapsed_seconds'],
                             effective_tokens_per_second=tokens_done/max(1,payload['elapsed_seconds']),
                             seen_domains=dict(seen_domains),optimizer_steps=step,global_effective_batch=macro,
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
        while True:
            gradients.zero_();failed.zero_();local_loss=torch.zeros(1,device=device,dtype=torch.float64)
            inputs=targets=output=loss=None
            try:
                for micro in microbatches(local_batch,cfg['max_microbatch_pairs'],micro_tokens):
                    inputs,targets,_=collate(store,micro,pad,eos,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        output=model(**inputs).logits
                        loss=F.binary_cross_entropy_with_logits(output[:,1].float()-output[:,0].float(),targets,reduction='sum')
                    (loss/len(indices)).backward();local_loss+=loss.detach().double()
                    inputs=targets=output=loss=None
            except torch.cuda.OutOfMemoryError:
                failed.fill_(1);inputs=targets=output=loss=None;gc.collect();torch.cuda.empty_cache()
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
        if (step+1)%5000==0:save(step+1)
    save(total_steps,final=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
