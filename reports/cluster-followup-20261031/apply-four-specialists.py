#!/usr/bin/env python3
"""Apply the approved scope change, preserving the live Retrieval launcher."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import sys
import time


REMOTE_CODE = r'''
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import json
import os
import signal
import subprocess
import time

HOME = Path('/home/dev/assumeengage')
ROOT = HOME / 'runs/five-specialists-20261030'


def process(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'state': fields[0], 'starttime': fields[19],
                'command': (path / 'cmdline').read_bytes().split(b'\0')}
    except (OSError, IndexError):
        return None


def atomic(path, text):
    temporary = path.with_name(path.name + '.scope-change.tmp')
    temporary.write_text(text)
    if path.exists():
        temporary.chmod(path.stat().st_mode & 0o777)
    temporary.replace(path)


def apply(payload):
    previous = json.loads((ROOT / 'PIPELINE.json').read_text())
    assert previous['stage'] == 'running' and previous['model'] == 'SPECIALIST_RETRIEVAL', previous
    old_pid = int((ROOT / 'EXPERIMENT.lock').read_text())
    old = process(old_pid)
    child_pid = previous['child_pid']
    child = process(child_pid)
    controller = str(HOME / 'scripts/run-five-area-takeover.py')
    assert old and controller.encode() in old['command'], 'Original controller identity changed'
    assert child and b'SPECIALIST_RETRIEVAL' in child['command'], 'Retrieval launcher identity changed'
    assert not (ROOT / 'MIXED_ALL/latest.json').exists(), 'Mixed pool already has a checkpoint; preserved'
    assert not (ROOT / 'MIXED_ALL/complete.json').exists(), 'Mixed pool already completed; preserved'
    assert not (ROOT / 'MERGES_COMPLETE.json').exists(), 'Merges already exist; preserved'
    for item in payload['files']:
        path = HOME / item['path']
        assert path.read_text() == item['before'], 'Remote file changed since capture: ' + str(path)
        if path.suffix == '.json':
            json.loads(item['after'])
        elif path.suffix == '.py':
            compile(item['after'], str(path), 'exec')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = ROOT / ('scope-change-backup-' + stamp)
    backup.mkdir()
    old_scope = ROOT / 'ACTIVE_TRAINING_SCOPE.json'
    if old_scope.exists():
        (backup / old_scope.name).write_bytes(old_scope.read_bytes())
    (backup / 'PIPELINE.json').write_text(json.dumps(previous, indent=2) + '\n')
    for item in payload['files']:
        (backup / Path(item['path']).name).write_text(item['before'])
    paused = False
    terminated = False
    written = []
    try:
        os.kill(old_pid, signal.SIGSTOP)
        paused = True
        for _ in range(50):
            current = process(old_pid)
            if current and current['state'] in ('T', 't'):
                break
            time.sleep(0.02)
        else:
            raise RuntimeError('Controller did not stop; nothing changed')
        current = process(child_pid)
        assert current and current['starttime'] == child['starttime'], 'Retrieval changed before deployment'
        for item in payload['files']:
            path = HOME / item['path']
            atomic(path, item['after'])
            written.append(item)
        config = json.loads((HOME / 'config/experiment-five-specialists.json').read_text())
        ready = json.loads((HOME / 'cache/pair-training/global-pool/TRAINING_READY.json').read_text())
        domains = config['active_training_domains']
        scope = {'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                 'reason': 'User explicitly cancelled Language specialist and its mixed/KD training exposure',
                 'active_training_domains': domains, 'models': config['models'],
                 'cancelled_models': ['SPECIALIST_LANGUAGE'],
                 'domains': {name: ready['domains'][name] for name in domains},
                 'mixed_all_pairs': sum(ready['domains'][name]['pairs'] for name in domains),
                 'mixed_all_effective_tokens': sum(ready['domains'][name]['effective_training_tokens'] for name in domains),
                 'trained_models': len(config['models']), 'merge_methods': len(config['merges']),
                 'evaluation_conditions': len(config['models']) + len(config['merges']),
                 'public_evaluation': 'Original five-area scope including Language; no metric redefinition',
                 'cache_policy': 'Full original cache unchanged; pass active_training_domains to PairPool for MIXED_ALL and KD',
                 'backup_directory': str(backup)}
        atomic(old_scope, json.dumps(scope, indent=2) + '\n')
        previous['child_starttime'] = child['starttime']
        atomic(ROOT / 'PIPELINE.json', json.dumps(previous, indent=2) + '\n')
        os.kill(old_pid, signal.SIGTERM)
        terminated = True
        os.kill(old_pid, signal.SIGCONT)
        paused = False
        for _ in range(100):
            current = process(old_pid)
            if current is None or current['starttime'] != old['starttime']:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError('Old controller has not exited; no second controller launched')
        with (ROOT / 'EXPERIMENT.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        with (HOME / 'logs/five-area-takeover-controller.log').open('a', buffering=1) as output:
            new = subprocess.Popen([str(HOME / 'envs/jev/bin/python'), '-u', controller],
                                   cwd=HOME, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        handoff = {'time_utc': datetime.now(timezone.utc).isoformat(), 'old_controller_pid': old_pid,
                   'new_controller_pid': new.pid, 'retrieval_child_pid': child_pid,
                   'retrieval_starttime_before': child['starttime'],
                   'retrieval_starttime_after': (process(child_pid) or {}).get('starttime'),
                   'retrieval_was_not_signalled': True, 'scope': scope}
        atomic(ROOT / 'FOUR_SPECIALISTS_HANDOFF.json', json.dumps(handoff, indent=2) + '\n')
        print(json.dumps(handoff, ensure_ascii=False), flush=True)
    except BaseException:
        if not terminated:
            for item in reversed(written):
                atomic(HOME / item['path'], item['before'])
            atomic(ROOT / 'PIPELINE.json', json.dumps(previous, indent=2) + '\n')
            saved_scope = backup / old_scope.name
            if saved_scope.exists():
                atomic(old_scope, saved_scope.read_text())
            elif old_scope.exists():
                old_scope.unlink()
        if paused:
            os.kill(old_pid, signal.SIGCONT)
        raise
'''


def restart_watcher(root):
    metadata_path = root / 'watcher-process.json'
    metadata = json.loads(metadata_path.read_text())
    old_pid = metadata['pid']
    folder = Path('/proc') / str(old_pid)
    script = root / 'watch-stages.py'
    if folder.exists():
        command = (folder / 'cmdline').read_bytes().split(b'\0')
        if str(script).encode() not in command:
            raise RuntimeError('Watcher PID belongs to another command; preserved')
        for _ in range(1200):
            channel = (folder / 'wchan').read_text().strip()
            children = (folder / 'task' / str(old_pid) / 'children').read_text().strip()
            state = json.loads((root / 'state.json').read_text())
            observed = datetime.fromisoformat(state['last_observed_utc'])
            age = (datetime.now(timezone.utc) - observed).total_seconds()
            if not children and 'nanosleep' in channel and 0 <= age < 100:
                print(json.dumps({'watcher_restart_precondition': 'completed_cycle_and_idle',
                                  'saved_milestones': list(state['sent']), 'cycle_age_seconds': round(age, 2)}))
                break
            time.sleep(0.1)
        else:
            raise RuntimeError('Watcher not idle after a completed cycle; preserved without forced restart')
        os.kill(old_pid, signal.SIGTERM)
        for _ in range(100):
            try:
                state = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state == 'Z':
                break
            time.sleep(0.05)
        else:
            raise RuntimeError('Old watcher has not exited; no duplicate watcher launched')
    with (root / 'watcher.log').open('a', buffering=1) as output:
        watcher = subprocess.Popen([sys.executable, '-u', str(script)], cwd=root.parents[1],
                                   stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    metadata.update(pid=watcher.pid, previous_pid=old_pid, scope='four-specialists-five-trained-ten-evaluated')
    temporary = metadata_path.with_suffix('.scope-change.tmp')
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(metadata_path)
    print(json.dumps({'watcher_restarted': True, 'old_pid': old_pid, 'new_pid': watcher.pid,
                      'notification_state_preserved': True}))


def main():
    root = Path(__file__).resolve().parent / 'four-specialists-change'
    names = ['scripts/run-five-area-takeover.py', 'scripts/pair_training_runtime.py',
             'scripts/train-pair-model-takeover.py', 'scripts/merge-formal-models.py',
             'scripts/run-formal-evaluation.py', 'scripts/five-area-stage-snapshot.py',
             'config/experiment-five-specialists.json', 'config/queued-distillation.json',
             'config/next-distillation-REPORT.md',
             'config/training-plan-five-specialists-2026-10-30.md']
    payload = {'files': [{'path': rel,
                          'before': (root / 'before' / Path(rel).name).read_text(),
                          'after': (root / 'after' / Path(rel).name).read_text()}
                         for rel in names]}
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'ClearAllForwardings=yes',
               'sh-09-gpu', 'python3 -B -']
    result = subprocess.run(command, input=REMOTE_CODE + '\napply(' + repr(payload) + ')\n',
                            text=True, capture_output=True, timeout=90)
    (root / 'deployment-stdout.json').write_text(result.stdout)
    (root / 'deployment-stderr.log').write_text(result.stderr)
    print(result.stdout, end='')
    if result.returncode:
        print(result.stderr, end='')
        raise SystemExit(result.returncode)
    restart_watcher(root.parent)


if __name__ == '__main__':
    main()
