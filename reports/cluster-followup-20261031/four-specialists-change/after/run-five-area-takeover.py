#!/usr/bin/env python3
"""Own the currently approved experiment, preserving any adopted training process."""
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


def process_identity(pid):
    try:
        folder = Path('/proc') / str(pid)
        fields = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'starttime': fields[19],
                'command': (folder / 'cmdline').read_bytes().split(b'\0')}
    except (OSError, IndexError):
        return None


def adopt_training(previous, name):
    if previous.get('model') != name or not previous.get('child_pid'):
        return False
    pid = previous['child_pid']
    identity = process_identity(pid)
    if identity is None:
        return False
    command = identity['command']
    trainer = str(HOME / 'scripts/train-pair-model-takeover.py').encode()
    if trainer not in command or b'--model' not in command or name.encode() not in command:
        raise RuntimeError('Existing PID is not the expected owned trainer; no duplicate launched')
    if previous.get('child_starttime', identity['starttime']) != identity['starttime']:
        raise RuntimeError('Existing trainer PID identity changed; no duplicate launched')
    status('running', model=name, child_pid=pid, child_starttime=identity['starttime'],
           adopted_existing_process=True, log=previous['log'], pairs_cap=None, token_cap=None,
           communication='gloo_cpu_adapter_gradient_sum')
    while True:
        current = process_identity(pid)
        if current is None or current['starttime'] != identity['starttime']:
            break
        time.sleep(15)
    if not complete(ROOT / name / 'complete.json'):
        raise RuntimeError(f'Adopted {name} exited without its full-epoch completion marker')
    return True


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
    try:
        previous = json.loads((ROOT / 'PIPELINE.json').read_text())
    except (OSError, ValueError):
        previous = {}
    if not complete(HOME / 'cache/pair-training/global-pool/TRAINING_READY.json'):
        ready = json.loads((HOME / 'cache/pair-training/global-pool/TRAINING_READY.json').read_text())
        if not ready.get('domains'):
            raise RuntimeError('Full training cache is not ready')
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['PYTHONUNBUFFERED'] = '1'
    order = [name for name in ['SPECIALIST_KNOWLEDGE', 'SPECIALIST_TOOLS', 'SPECIALIST_ARTS',
                              'SPECIALIST_RETRIEVAL', 'SPECIALIST_LANGUAGE', 'MIXED_ALL']
             if name in CONFIG['models']]
    for name in order:
        if adopt_training(previous, name):
            continue
        if complete(ROOT / name / 'complete.json'):
            status('reusing_completed_training', model=name)
            continue
        run([PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
             '--nproc_per_node=4', HOME / 'scripts/train-pair-model-takeover.py', '--model', name], name)
    status('all_active_training_complete', trained_models=len(CONFIG['models']))
    run([PYTHON, HOME / 'scripts/merge-formal-models.py'], 'merges')
    run([PYTHON, HOME / 'scripts/run-formal-evaluation.py'], 'evaluation')
    status('complete', trained_models=len(CONFIG['models']), weight_merge_methods=len(CONFIG['merges']),
           primary_conditions=len(CONFIG['models']) + len(CONFIG['merges']))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('blocked', error=str(error))
        traceback.print_exc()
        raise
