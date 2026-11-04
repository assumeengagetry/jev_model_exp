#!/usr/bin/env python3
"""Preserve SAMerging-first priority, then launch one fresh raw-Qwen comparison."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

SOURCE=Path('/data/run01/scwb721/jev-kd-20261101')
ROOT=SOURCE/'experiments/kd-raw-qwen-1m-20261103'
SAM=SOURCE/'experiments/samerging-four-experts-20261103'


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def write(value):
    temp=ROOT/'launcher-state.writing'
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(ROOT/'launcher-state.json')


def main():
    owner=dict(pid=os.getpid(),host=socket.gethostname(),started_unix=time.time())
    if (ROOT/'COMPLETE.json').exists():
        write(dict(owner,phase='already_complete'));return
    deadline=time.monotonic()+7*86400
    while time.monotonic()<deadline:
        if read(SAM/'COMPLETE.json').get('complete') and read(SAM/'STEP_EXIT.json').get('exit_code')==0:
            break
        write(dict(owner,phase='waiting_for_samerging_success',sam_status=read(SAM/'status.json'),
                   sam_failure=read(SAM/'FAILURE.json'),checked_unix=time.time()))
        time.sleep(20)
    else:raise RuntimeError('SAMerging dependency wait window ended')
    command=['srun','--jobid=275711','--exact','-N1','-n1','-c32','--gpus=4','--mem=384G','--time=12:00:00',
             f'--chdir={ROOT}','env',f'JEV_KD_ROOT={ROOT}','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
             'TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4',
             str(SOURCE/'envs/student/bin/python'),'-u',str(ROOT/'code/run_job.py')]
    with (ROOT/'logs/job.log').open('a',buffering=1) as handle:
        child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        write(dict(owner,phase='raw_step_requested',srun_pid=child.pid,command=command,unix_time=time.time()))
        code=child.wait()
        write(dict(owner,phase='complete' if code==0 else 'failed',exit_code=code,ended_unix=time.time()))
        if code:raise RuntimeError(f'Raw-Qwen stage exited {code}; saved artifacts retained')


if __name__=='__main__':
    with (ROOT/'launcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
