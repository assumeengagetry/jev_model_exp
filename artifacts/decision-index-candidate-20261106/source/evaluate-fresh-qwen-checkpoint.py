#!/usr/bin/env python3
"""Use the existing candidate-classifier scorer for an isolated fresh-Qwen model."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HOME = Path('/home/dev/assumeengage')
CFG = json.loads((HOME / os.environ.get('JEV_FRESH_DISTILL_CONFIG',
                    'config/experiment-fresh-qwen-1m-20261103.json')).read_text())
ROOT = HOME / CFG['run_root']
spec = importlib.util.spec_from_file_location('fresh_formal_evaluation',
                                             HOME / 'scripts/run-formal-evaluation.py')
EVAL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EVAL)
EVAL.ROOT = ROOT
EVAL.CFG = {**EVAL.CFG, 'run_root': CFG['run_root'], 'base_model': CFG['base_model'],
            'common_initialization': CFG['common_initialization']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    folder = ROOT / 'evaluation' / args.model
    folder.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        with (folder / f'.worker-{args.worker}.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if EVAL.done(folder / f'worker-{args.worker}-complete.json'):
                return
            EVAL.worker(args)
        return
    with (folder / '.model.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if EVAL.done(folder / 'complete.json'):
            return
        gpus = [x for x in args.gpus.split(',') if x]
        if not gpus:
            raise RuntimeError('At least one GPU is required')
        checkpoint = args.checkpoint or str(HOME / CFG['student_parent'] if args.model == 'BASE_QWEN'
                                             else ROOT / args.model / 'final')
        processes, logs = [], []
        (ROOT / 'logs').mkdir(exist_ok=True)
        for rank, gpu in enumerate(gpus):
            if EVAL.done(folder / f'worker-{rank}-complete.json'):
                continue
            stream = (ROOT / 'logs' / f'evaluate-{args.model}-gpu{gpu}.log').open('a', buffering=1)
            logs.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            processes.append(subprocess.Popen([
                sys.executable, '-B', '-u', __file__, '--worker', str(rank), '--workers', str(len(gpus)),
                '--model', args.model, '--checkpoint', checkpoint,
            ], cwd=HOME, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        for stream in logs:
            stream.close()
        if any(codes):
            raise RuntimeError(f'{args.model} evaluation worker exit codes: {codes}')
        EVAL.score_model(args.model, folder)


if __name__ == '__main__':
    main()
