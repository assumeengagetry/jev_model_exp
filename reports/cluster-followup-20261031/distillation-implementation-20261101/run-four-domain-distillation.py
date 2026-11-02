#!/usr/bin/env python3
"""Resume the approved four-domain distillation without duplicate jobs or budget changes."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

HOME = Path('/home/dev/assumeengage')
CFG = json.loads((HOME / 'config/experiment-four-domain-distillation.json').read_text())
ROOT = HOME / CFG['run_root']
JOBS = ROOT / 'RUNTIME.json'
CHILDREN = {}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def done(path):
    return read(path).get('complete') is True


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def status(stage, **extra):
    data = {'stage': stage, 'updated_unix': time.time(), 'owner': 'approved-four-domain-distillation',
            'active_training_domains': CFG['active_training_domains'], 'pair_cap': None,
            'token_cap': None, **extra}
    atomic(ROOT / 'PIPELINE.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def identity(pid):
    try:
        folder = Path('/proc') / str(pid)
        fields = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'starttime': fields[19], 'command': (folder / 'cmdline').read_bytes().split(b'\0')}
    except (OSError, IndexError):
        return None


def alive(job):
    current = identity(job['pid'])
    return bool(current and current['starttime'] == job['starttime'])


def adopt_arm(jobs, name, marker):
    if name not in CFG['arms']:
        return False
    owner = read(ROOT / name / 'OWNER.json')
    if owner.get('arm') != name or owner.get('student_parent') != CFG['student_parent']:
        return False
    if owner.get('active_training_domains') != CFG['active_training_domains']:
        return False
    worker = identity(owner.get('worker_pid'))
    expected_script = str(HOME / 'scripts/train-four-domain-kd.py').encode()
    def matches(info, starttime):
        if not info or info['starttime'] != starttime or expected_script not in info['command']:
            return False
        try:
            position = info['command'].index(b'--arm')
            return info['command'][position + 1] == name.encode()
        except (ValueError, IndexError):
            return False
    if not matches(worker, owner.get('worker_starttime')):
        return False
    coordinator = identity(owner.get('coordinator_pid'))
    use_coordinator = matches(coordinator, owner.get('coordinator_starttime'))
    pid = owner['coordinator_pid'] if use_coordinator else owner['worker_pid']
    current = coordinator if use_coordinator else worker
    jobs[name] = {'pid': pid, 'starttime': current['starttime'], 'marker': str(marker),
                  'adopted': True, 'ownership_file': str(ROOT / name / 'OWNER.json'),
                  'gpus': owner.get('cuda_visible_devices'),
                  'log': str(HOME / 'logs' / ('distillation-' + name + '.log'))}
    atomic(JOBS, jobs)
    return True


def launch(jobs, name, command, marker, gpus=None, adopt_script=None):
    if done(marker):
        return
    if adopt_arm(jobs, name, marker):
        return
    if name in jobs and alive(jobs[name]):
        return
    if adopt_script:
        # The same script also has per-file download helpers. Adopt the actual
        # separate asset-lock owner, never a short-lived --download-one child.
        try:
            owner = int((ROOT / 'TEACHER_ASSETS.lock').read_text().strip())
        except (OSError, ValueError):
            owner = None
        info = identity(owner) if owner else None
        if info and str(adopt_script).encode() in info['command'] and b'--download-one' not in info['command']:
            jobs[name] = {'pid': owner, 'starttime': info['starttime'], 'marker': str(marker),
                          'adopted': True, 'gpus': gpus,
                          'log': str(ROOT / 'logs/prepare-knowline-teacher.log')}
            atomic(JOBS, jobs)
            return
    log = HOME / 'logs' / ('distillation-' + name + '.log')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               GLOO_SOCKET_IFNAME='lo')
    if gpus is not None:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    with log.open('a', buffering=1) as output:
        child = subprocess.Popen([str(value) for value in command], cwd=HOME, env=env,
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
    CHILDREN[name] = child
    info = identity(child.pid)
    if info is None:
        raise RuntimeError(f'{name} failed during startup; see {log}')
    jobs[name] = {'pid': child.pid, 'starttime': info['starttime'], 'marker': str(marker),
                  'log': str(log), 'gpus': gpus, 'command': list(map(str, command))}
    atomic(JOBS, jobs)


def check(jobs):
    for name, job in list(jobs.items()):
        if name in CHILDREN:
            CHILDREN[name].poll()  # reap exited children before consulting /proc
        if not done(Path(job['marker'])) and not alive(job):
            if adopt_arm(jobs, name, Path(job['marker'])):
                continue
            raise RuntimeError(f'{name} stopped before completion; see {job.get("log", "Teacher preparation log")}')


def trainer(arm, gpus):
    python = HOME / CFG['student_python']
    command = [python]
    if len(gpus) > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                    '--nproc_per_node=' + str(len(gpus))]
    return command + [HOME / 'scripts/train-four-domain-kd.py', '--arm', arm]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'EXPERIMENT.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing distillation owner is active; no duplicate launched.', flush=True)
            return
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        previous = HOME / CFG['previous_run_root']
        if read(previous / 'PIPELINE.json').get('stage') != 'complete' or not done(previous / 'EVALUATION_COMPLETE.json'):
            raise RuntimeError('Previous complete-round gate is not satisfied')
        if read(previous / 'EVALUATION_COMPLETE.json').get('conditions') != 10:
            raise RuntimeError('Ten previous public conditions are required')
        with (previous / 'EXPERIMENT.lock').open('a+') as prior:
            fcntl.flock(prior.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(prior.fileno(), fcntl.LOCK_UN)
        scope = read(previous / 'ACTIVE_TRAINING_SCOPE.json')
        parent = read(previous / 'MIXED_ALL/complete.json')
        if CFG['active_training_domains'] != scope.get('active_training_domains') or CFG['pairs'] != parent.get('pairs_seen'):
            raise RuntimeError('Approved four-domain scope differs from Student parent')
        if set(parent.get('seen_domains', {})) != set(CFG['active_training_domains']):
            raise RuntimeError('Parent contains an unexpected training domain')
        atomic(ROOT / 'ACTIVE_TRAINING_SCOPE.json', {**scope, 'student_parent': CFG['student_parent'],
               'arms': CFG['arms'], 'epochs_each': 1, 'learning_rate': CFG['learning_rate']})
        jobs = read(JOBS)
        hard = ROOT / 'KD_HARD/complete.json'
        launch(jobs, 'KD_HARD', trainer('KD_HARD', [0]), hard, [0])
        teacher_ready = ROOT / 'TEACHER_READY.json'
        prep = HOME / 'scripts/prepare-knowline-teacher.py'
        # Reconcile a previous asset failure exactly once on controller resume;
        # later death is still surfaced rather than retried forever in this loop.
        if not done(teacher_ready) and prep.is_file():
            launch(jobs, 'teacher-assets', [HOME / CFG['student_python'], '-u', prep], teacher_ready,
                   adopt_script=prep)
        while not done(teacher_ready):
            if 'teacher-assets' not in jobs and prep.is_file():
                launch(jobs, 'teacher-assets', [HOME / CFG['student_python'], '-u', prep], teacher_ready,
                       adopt_script=prep)
            check(jobs)
            status('running', phase='hard_training_and_teacher_preparation', jobs=jobs)
            time.sleep(15)
        if not done(ROOT / 'TEACHER_LABELS_READY.json'):
            for rank, gpu in enumerate(CFG['teacher_gpus']):
                marker = ROOT / 'teacher-labels' / f'worker-{rank}-complete.json'
                command = [HOME / CFG['teacher_python'], '-u', HOME / 'scripts/label-knowline-pairs.py',
                           '--rank', rank, '--workers', CFG['teacher_workers']]
                launch(jobs, f'teacher-{rank}', command, marker, [gpu])
            while not all(done(ROOT / 'teacher-labels' / f'worker-{rank}-complete.json')
                          for rank in range(CFG['teacher_workers'])):
                check(jobs)
                status('running', phase='hard_training_and_full_teacher_labeling', jobs=jobs)
                time.sleep(15)
            command = [HOME / CFG['teacher_python'], '-u', HOME / 'scripts/label-knowline-pairs.py',
                       '--merge', '--workers', CFG['teacher_workers']]
            launch(jobs, 'teacher-label-merge', command, ROOT / 'TEACHER_LABELS_READY.json')
        while not done(hard) or not done(ROOT / 'TEACHER_LABELS_READY.json'):
            check(jobs)
            status('running', phase='completing_hard_and_teacher_labels', jobs=jobs)
            time.sleep(15)
        # Reap all completed Teacher processes before assigning their GPUs to training.
        while any((name.startswith('teacher-') or name == 'KD_HARD') and alive(job)
                  for name, job in jobs.items()):
            check(jobs); time.sleep(2)
        for arm, gpus in [('KD_SOFT', [0, 1]), ('KD_HYBRID', [2, 3])]:
            launch(jobs, arm, trainer(arm, gpus), ROOT / arm / 'complete.json', gpus)
        while not all(done(ROOT / arm / 'complete.json') for arm in CFG['arms']):
            check(jobs)
            status('running', phase='soft_and_hybrid_full_continuation', jobs=jobs)
            time.sleep(15)
        while any(name in CFG['arms'] and alive(job) for name, job in jobs.items()):
            check(jobs); time.sleep(2)
        command = [HOME / CFG['student_python'], '-u', HOME / 'scripts/run-distillation-evaluation.py']
        launch(jobs, 'evaluation', command, ROOT / 'DISTILLATION_COMPLETE.json', [0, 1, 2, 3])
        while not done(ROOT / 'DISTILLATION_COMPLETE.json'):
            check(jobs)
            status('running', phase='three_arm_public_evaluation', jobs=jobs)
            time.sleep(15)
        status('complete', trained_models=3, public_conditions=3, baseline_reused=True,
               finished_utc=datetime.now(timezone.utc).isoformat())


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('blocked', error=str(error), jobs=read(JOBS))
        traceback.print_exc()
        raise
