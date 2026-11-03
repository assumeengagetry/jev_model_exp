#!/usr/bin/env python3
"""Own one Slurm step and publish its exit status for the login-node controller."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

SOURCE = Path('/data/run01/scwb721/jev-kd-20261101')
SERIES = SOURCE/'staged/scale-series-20261102'
CODE = SERIES/'code'


def write(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('role', choices=['student', 'teacher'])
    p.add_argument('size', type=int)
    p.add_argument('attempt', type=int)
    args = p.parse_args()
    root = SERIES/('stages' if args.role == 'student' else 'producer')/f'{args.size//1000000}m'
    status = SERIES/'logs'/f'{args.role}-{args.size}-{args.attempt}-step.json'
    value = dict(role=args.role, size=args.size, attempt=args.attempt, pid=os.getpid(),
                 host=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'),
                 step_id=os.environ.get('SLURM_STEP_ID'), devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 started_unix=time.time(), state='running')
    write(status, value)
    env = os.environ.copy()
    env.update(JEV_KD_ROOT=str(root), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    code = 1
    try:
        code = subprocess.call([str(SOURCE/'envs'/args.role/'bin/python'), '-u', str(CODE/f'{args.role}_stage.py')], env=env)
        if code == 0 and args.role == 'teacher':
            env['JEV_KD_ROOT'] = str(SERIES)
            code = subprocess.call([str(SOURCE/'envs/student/bin/python'), '-u', str(CODE/'stage_data.py'),
                                    '--materialize', str(args.size)], env=env)
    finally:
        write(status, dict(value, state='complete' if code == 0 else 'failed', exit_code=code, ended_unix=time.time()))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
