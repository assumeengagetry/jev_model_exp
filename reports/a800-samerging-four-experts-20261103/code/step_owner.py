#!/usr/bin/env python3
"""Record the owned Slurm step's exit even when the experiment child fails."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path('/data/run01/scwb721/jev-kd-20261101/experiments/samerging-four-experts-20261103')


def write(name,value):
    path=ROOT/name
    temp=path.with_suffix('.writing')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def main():
    value=dict(pid=os.getpid(),host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
               step_id=os.environ.get('SLURM_STEP_ID'),devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               started_unix=time.time())
    write('STEP_STARTED.json',value)
    code=1
    try:
        code=subprocess.call([sys.executable,'-u',str(ROOT/'code/run_samerging.py')])
    finally:
        write('STEP_EXIT.json',dict(value,exit_code=code,ended_unix=time.time()))
    return code


if __name__=='__main__':raise SystemExit(main())
