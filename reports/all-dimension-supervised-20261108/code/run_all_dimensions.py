#!/usr/bin/env python3
"""Prepare immediately, then train when the selected GPU resources are free."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def atomic(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    path = Path(parser.parse_args().config).resolve()
    cfg = json.loads(path.read_text())
    root = Path(cfg['run_root'])
    code = Path(__file__).resolve().parent
    lock = (root / 'PIPELINE.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None

    def status(stage, **extra):
        atomic(root / 'PIPELINE.json', {'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(),
               'coordinator_pid': os.getpid(), 'all_training_domains': cfg['active_training_domains'],
               'extra_tests': False, 'hash_checks': False, **extra})

    try:
        status('preparing_training_schedules')
        with (root / 'prepare.log').open('a', buffering=1) as log:
            subprocess.run([sys.executable, '-B', '-u', str(code / 'prepare_all_dimensions.py'),
                            '--config', str(path)], check=True, stdout=log, stderr=subprocess.STDOUT)
        markers = [Path(p) for p in cfg['wait_for_completion_any']]
        while True:
            marked = not markers
            for marker in markers:
                if marker.is_file():
                    try:
                        marked = marked or json.loads(marker.read_text()).get('complete') is True
                    except ValueError:
                        pass
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                    text=True, capture_output=True, check=True, timeout=20)
            active = [x for x in result.stdout.splitlines() if x.strip().isdigit()]
            if marked and not active:
                break
            status('queued_after_existing_evaluations', data_ready=True,
                   waiting_for_marker=not marked, active_gpu_processes=len(active))
            time.sleep(30)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, cfg['gpus'])),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [sys.executable, '-B', '-u', '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', f"--nproc_per_node={len(cfg['gpus'])}",
                   str(code / 'train_all_dimensions.py'), '--config', str(path)]
        with (root / 'train.log').open('a', buffering=1) as log:
            child = subprocess.Popen(command, env=env, cwd=root, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            status('training', launcher_pid=child.pid, gpus=cfg['gpus'])
            exit_code = child.wait()
        if exit_code:
            raise RuntimeError(f'Formal training exited {exit_code}; see train.log')
        if not (root / 'COMPLETE.json').is_file():
            raise RuntimeError('Trainer exited without its final model completion record')
        status('complete', final_model=str(root / 'stage2/final'))
    except BaseException as error:
        status('blocked', error=str(error), launcher_pid=child.pid if child else None)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
