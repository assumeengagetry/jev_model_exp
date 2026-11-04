#!/usr/bin/env python3
"""Four-slot raw-init experiment: three genuine training arms and a fresh baseline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from kd_runtime import ROOT,PARENT,WORKSPACE,atomic_json,pool,shared_order


def check_initialization():
    from safetensors import safe_open
    import torch
    torch.set_num_threads(4)
    metadata=json.loads((PARENT/'initialization.json').read_text())
    if metadata.get('purpose')!='actual common initialization for five specialists and MIXED_ALL':
        raise RuntimeError('Requested raw common initialization provenance is missing')
    keys=[];nonzero=0;heads={}
    with safe_open(str(PARENT/'adapter_model.safetensors'),framework='pt',device='cpu') as tensors:
        for key in tensors.keys():
            if 'lora_B' in key:
                keys.append(key)
                nonzero+=int(torch.count_nonzero(tensors.get_tensor(key)))
            if 'score' in key and key.endswith('weight'):
                value=tensors.get_tensor(key)
                if not torch.isfinite(value).all():raise RuntimeError('Initial common head is nonfinite')
                heads[key]=list(value.shape)
    if not keys or nonzero or not heads:
        raise RuntimeError('Raw initialization must have zero LoRA updates and its saved common random head')
    value=dict(complete=True,initialization='raw_pretrained_backbone_untrained_adapter_and_head',
               backbone=str(WORKSPACE/'models/Qwen3-0.6B'),initial_adapter_and_head=str(PARENT),
               lora_B_matrices=len(keys),lora_B_nonzero_elements=nonzero,saved_random_heads=heads,
               trained_MIXED_ALL_adapter_used=False,trained_MIXED_ALL_head_used=False,metadata=metadata)
    atomic_json(ROOT/'initialization-check.json',value)
    print(json.dumps(dict(event='raw_initialization_confirmed',**value)),flush=True)


def run():
    # Same saved initialization already established by the completed raw1M run.
    subprocess.run([sys.executable,'-u',str(ROOT/'code/teacher_diagnostic.py')],check=True)
    shared_order(pool())
    devices=[v for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v]
    if len(devices)!=4:raise RuntimeError('This matched raw-init stage requires exactly four allocated GPU slots')
    names=['KD_MIX25','KD_AGREE50','KD_TEMP2_50','KD_LS10']
    train=[];evaluate=[];done=set();active={}
    for name in names:
        path=ROOT/'outputs/evaluation'/name/'complete.json'
        if path.exists() and json.loads(path.read_text()).get('scope')=='pilot_subset':done.add(name)
        elif (ROOT/'outputs'/name/'complete.json').exists():evaluate.append(name)
        else:train.append(name)

    def launch(name,phase,device):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='4' if phase=='train' else '2',
                   MKL_NUM_THREADS='4' if phase=='train' else '2',HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        script='train_student.py' if phase=='train' else 'evaluate_student.py'
        args=['--arm',name.removeprefix('KD_'),'--attention','sdpa'] if phase=='train' else ['--name',name]
        handle=(ROOT/'logs'/f'{phase}-{name}.log').open('a',buffering=1)
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'code'/script),*args],env=env,
                               stdout=handle,stderr=subprocess.STDOUT)
        active[device]=dict(name=name,phase=phase,process=child,handle=handle)
        print(json.dumps(dict(event='raw_component_started',name=name,phase=phase,device=device,pid=child.pid)),flush=True)
    try:
        while len(done)<4:
            for device,job in list(active.items()):
                code=job['process'].poll()
                if code is None:continue
                job['handle'].close()
                if code:raise RuntimeError(f"Raw-init {job['name']} {job['phase']} exited {code}; recovery artifacts retained")
                if job['phase']=='train':evaluate.append(job['name'])
                else:done.add(job['name'])
                del active[device]
            for device in devices:
                if device in active:continue
                if train:launch(train.pop(0),'train',device)
                elif evaluate:launch(evaluate.pop(0),'evaluate',device)
            atomic_json(ROOT/'progress.json',dict(completed=sorted(done),pending_training=train,pending_evaluation=evaluate,
              active={device:dict(name=j['name'],phase=j['phase'],pid=j['process'].pid) for device,j in active.items()},unix_time=time.time()))
            if len(done)<4:time.sleep(5)
        subprocess.run([sys.executable,'-u',str(ROOT/'code/report_results.py')],check=True)
        marker=json.loads((ROOT/'outputs/pilot-complete.json').read_text())
        atomic_json(ROOT/'COMPLETE.json',dict(marker,initialization='raw_pretrained_qwen',unix_time=time.time()))
    finally:
        for job in active.values():
            if job['process'].poll() is None:job['process'].terminate()
        for job in active.values():
            job['process'].wait(timeout=90)
            job['handle'].close()


def main():
    allocation=dict(pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),
                    devices=os.environ.get('CUDA_VISIBLE_DEVICES'),started_unix=time.time())
    atomic_json(ROOT/'allocation.json',allocation)
    code=1
    try:
        run();code=0
    except BaseException as error:
        atomic_json(ROOT/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),unix_time=time.time()))
        traceback.print_exc()
        raise
    finally:
        atomic_json(ROOT/'STEP_EXIT.json',dict(allocation,exit_code=code,ended_unix=time.time()))


if __name__=='__main__':main()
