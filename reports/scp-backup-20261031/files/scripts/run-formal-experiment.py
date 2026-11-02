#!/usr/bin/env python3
"""Resumable owner of six real training jobs, five merges, and formal evaluation."""
import ctypes
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HOME=Path('/home/dev/assumeengage')
PYTHON=HOME/'envs/jev/bin/python'
CONFIG=json.loads((HOME/'config/experiment-five-specialists.json').read_text())
ROOT=HOME/CONFIG['run_root'];ROOT.mkdir(parents=True,exist_ok=True)
STATUS=ROOT/'PIPELINE.json'


def status(stage,**extra):
    data=dict(stage=stage,updated_unix=time.time(),plan=CONFIG['plan'],**extra)
    temporary=STATUS.with_suffix('.json.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2));temporary.replace(STATUS)
    print(json.dumps(data,ensure_ascii=False),flush=True)


def complete(path):
    try:return json.loads(path.read_text()).get('complete',False)
    except (OSError,ValueError):return False


def wait_for_cache(path):
    if complete(path):return
    path.parent.mkdir(parents=True,exist_ok=True)
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0:raise OSError(ctypes.get_errno(),'Cannot watch cache completion events')
    # Local NFS writes from the tokenizer generate close/move/create events.
    watch=libc.inotify_add_watch(fd,os.fsencode(path.parent),0x00000008|0x00000080|0x00000100)
    if watch<0:
        os.close(fd);raise OSError(ctypes.get_errno(),'Cannot watch token-cache directory')
    try:
        while not complete(path):os.read(fd,65536)
    finally:os.close(fd)


def run(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',buffering=1) as output:
        process=subprocess.Popen([str(value) for value in command],cwd=HOME,env=os.environ.copy(),stdout=output,stderr=subprocess.STDOUT)
        status('running_process',pid=process.pid,command=[str(v) for v in command],log=str(log.relative_to(HOME)))
        code=process.wait()
    if code:raise RuntimeError(f'Actual experiment process exited {code}; see {log.relative_to(HOME)}')


def main():
    lock=(ROOT/'EXPERIMENT.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('An existing owner is already running this experiment; not starting duplicate GPU jobs.',flush=True);return
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    status('waiting_for_complete_new_text_token_cache',formal_training_started=False)
    wait_for_cache(HOME/'cache/pair-training/tokenized/global-new/MANIFEST.json')
    run([PYTHON,HOME/'scripts/finalize-training-cache.py'],HOME/'logs/finalize-training-cache.log')
    # Four ranks keep the global effective batch exactly 128; no short pilot runs.
    os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
    os.environ.setdefault('NCCL_SHM_DISABLE','1');os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING','1')
    order=['SPECIALIST_KNOWLEDGE','SPECIALIST_TOOLS','SPECIALIST_ARTS','SPECIALIST_RETRIEVAL','SPECIALIST_LANGUAGE','MIXED_ALL']
    for name in order:
        if complete(ROOT/name/'complete.json'):
            status('reusing_completed_formal_model',model=name);continue
        status('starting_formal_training_process',model=name,epochs=1,world_size=4,pair_cap=None,token_cap=None)
        run([PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
             HOME/'scripts/train-pair-model.py','--model',name],HOME/'logs'/f'formal-{name}.log')
    status('training_complete_starting_effective_weight_merges')
    run([PYTHON,HOME/'scripts/merge-formal-models.py'],HOME/'logs/formal-merges.log')
    status('preparing_unified_formal_evaluation')
    run([PYTHON,HOME/'scripts/run-formal-evaluation.py'],HOME/'logs/formal-evaluation.log')
    status('complete',models_trained=6,merge_methods=5,major_evaluation_conditions=11)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('blocked',error=str(error));raise
