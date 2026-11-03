#!/usr/bin/env python3
"""Durable two-role scheduler inside the user's existing four-GPU allocation."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

SOURCE = Path('/data/run01/scwb721/jev-kd-20261101')
SERIES = SOURCE/'staged/scale-series-20261102'
CODE = SERIES/'code'
JOB = '275711'
MILESTONES = [1000000, 2000000, 3000000]
children = {}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def emit(event, **fields):
    value = dict(event=event, unix_time=time.time(), **fields)
    print(json.dumps(value), flush=True)
    with (SERIES/'logs/events.jsonl').open('a') as handle:
        handle.write(json.dumps(value)+'\n')


def alive(record):
    proc = Path('/proc')/str(record['pid'])
    try:
        command = (proc/'cmdline').read_bytes().decode().split('\0')
        return str(CODE/'role_runner.py') in command and (proc/'stat').read_text().split()[2] != 'Z'
    except (OSError, UnicodeError):
        return False


def launch(role, size, state):
    key = f'{role}-{size}'
    attempt = 1 + state['attempts'].get(key, 0)
    if attempt > 3:
        state['blocked'][role] = dict(size=size, reason='Three attempts failed; inspect saved logs')
        emit('series_role_blocked', role=role, size=size)
        return
    root = SERIES/('stages' if role == 'student' else 'producer')/f'{size//1000000}m'
    command = ['srun', f'--jobid={JOB}', '--exact', '-N1', '-n1', '-c16', '--gpus=2', '--mem=128G',
               '--time=21-00:00:00', f'--chdir={root}', '/usr/bin/python3', '-u', str(CODE/'role_runner.py'),
               role, str(size), str(attempt)]
    with (SERIES/'logs'/f'{key}-attempt-{attempt}.log').open('a', buffering=1) as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    children[role] = process
    state['attempts'][key] = attempt
    state['active'][role] = dict(role=role, size=size, attempt=attempt, pid=process.pid,
                                  host=socket.gethostname(), started_unix=time.time(), command=command)
    emit('series_role_launched', **state['active'][role])


def main():
    path = SERIES/'controller-state.json'
    state = read(path, dict(active={}, attempts={}, blocked={}, reported=[]))
    emit('series_controller_started', pid=os.getpid(), host=socket.gethostname(), allocation=JOB)
    while True:
        for role, record in list(state['active'].items()):
            process = children.get(role)
            if process is not None:
                running = process.poll() is None
                result = process.returncode
            else:
                if record['host'] != socket.gethostname():
                    raise RuntimeError('Controller must resume on the recorded login host; refusing duplicate steps')
                running = alive(record)
                result = None
            if running:
                continue
            status = read(SERIES/'logs'/f"{role}-{record['size']}-{record['attempt']}-step.json", {})
            emit('series_role_exited', role=role, size=record['size'], result=result, step_status=status)
            del state['active'][role]
            children.pop(role, None)
        for size in MILESTONES:
            marker = SERIES/'stages'/f'{size//1000000}m'/'outputs/pilot-complete.json'
            if marker.exists() and size not in state['reported']:
                completion = read(marker)
                emit('scale_stage_report_ready', size=size, **completion)
                state['reported'].append(size)
        if len(state['reported']) == len(MILESTONES):
            state.update(complete=True, ended_unix=time.time())
            write(path, state)
            emit('scale_series_complete', milestones=MILESTONES)
            return
        if 'student' not in state['active'] and 'student' not in state['blocked']:
            for size in MILESTONES:
                root = SERIES/'stages'/f'{size//1000000}m'
                if (root/'outputs/pilot-complete.json').exists():
                    continue
                if (root/'outputs/teacher/complete.json').exists():
                    launch('student', size, state)
                break
        if 'teacher' not in state['active'] and 'teacher' not in state['blocked']:
            for size in MILESTONES[1:]:
                root = SERIES/'producer'/f'{size//1000000}m'
                ready = SERIES/'stages'/f'{size//1000000}m'/'outputs/teacher/complete.json'
                if ready.exists() and (root/'outputs/teacher/complete.json').exists():
                    continue
                launch('teacher', size, state)
                break
        state['updated_unix'] = time.time()
        write(path, state)
        time.sleep(20)


if __name__ == '__main__':
    with (SERIES/'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
