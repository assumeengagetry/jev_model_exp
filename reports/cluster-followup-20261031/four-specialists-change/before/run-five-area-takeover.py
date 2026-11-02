#!/usr/bin/env python3
"""Own the approved full six-model experiment after the explicit user handoff."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

HOME = Path('/home/dev/assumeengage')
PYTHON = HOME / 'envs/jev/bin/python'
CONFIG = json.loads((HOME / 'config/experiment-five-specialists.json').read_text())
ROOT = HOME / CONFIG['run_root']


def status(stage, **extra):
    data = {'stage': stage, 'updated_unix': time.time(), 'owner': 'ssh-five-area-takeover', **extra}
    path = ROOT / 'PIPELINE.json'
    temporary = path.with_suffix('.takeover.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def complete(path):
    try:
        return json.loads(path.read_text()).get('complete') is True
    except (OSError, ValueError):
        return False


def run(command, name):
    log = HOME / 'logs' / ('takeover-' + name + '.log')
    with log.open('a', buffering=1) as output:
        process = subprocess.Popen([str(x) for x in command], cwd=HOME, env=os.environ.copy(),
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        status('running', model=name, child_pid=process.pid, log=str(log),
               pairs_cap=None, token_cap=None, communication='gloo_cpu_adapter_gradient_sum')
        code = process.wait()
    if code:
        raise RuntimeError(f'{name} exited {code}; see {log}')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'EXPERIMENT.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Existing formal owner is alive; no duplicate launched.', flush=True)
        return
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    if not complete(HOME / 'cache/pair-training/global-pool/TRAINING_READY.json'):
        ready = json.loads((HOME / 'cache/pair-training/global-pool/TRAINING_READY.json').read_text())
        if not ready.get('domains'):
            raise RuntimeError('Full training cache is not ready')
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['PYTHONUNBUFFERED'] = '1'
    order = ['SPECIALIST_KNOWLEDGE', 'SPECIALIST_TOOLS', 'SPECIALIST_ARTS',
             'SPECIALIST_RETRIEVAL', 'SPECIALIST_LANGUAGE', 'MIXED_ALL']
    for name in order:
        if complete(ROOT / name / 'complete.json'):
            status('reusing_completed_training', model=name)
            continue
        run([PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
             '--nproc_per_node=4', HOME / 'scripts/train-pair-model-takeover.py', '--model', name], name)
    status('all_six_training_complete')
    run([PYTHON, HOME / 'scripts/merge-formal-models.py'], 'merges')
    run([PYTHON, HOME / 'scripts/run-formal-evaluation.py'], 'evaluation')
    status('complete', trained_models=6, weight_merge_methods=5, primary_conditions=11)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('blocked', error=str(error))
        traceback.print_exc()
        raise
