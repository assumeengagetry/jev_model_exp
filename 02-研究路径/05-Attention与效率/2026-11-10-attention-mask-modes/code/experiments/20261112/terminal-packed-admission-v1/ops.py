"""Install-only local deployer / one-shot SYSTEM-Python GPU0/1 admission manager.

No model imports, progress polling, retry, preemption, GPU2/3 control or training.
Run is explicit and separate from install. All outputs, including failures, stay.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import fcntl
import os
from pathlib import Path
import select
import shutil
import signal
import stat as modes
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
from consumer import (ARMS, CONTROL, FOLLOWUP, LOCAL, MODEL_PYTHON, NEW_CAP, OLD_CAP,
                      OLD_EVIDENCE, ROOT, SHARED, commit, dump, pin,
                      producer_record, read_json, readonly, require, sealed_config,
                      sync_dir, verify)

LEASES = Path('/tmp/opencode/jev-owned-h800-leases-20261112')
ASSIGNMENTS = {'C_D_BCE': 0, 'C_D_CE': 1}
ACTIVE_ARMS = ('STATE_NL_PB_P_CE', 'JPAIR_P_BCE_RANK')
GUARD_SECONDS, GRACE_SECONDS = 15, 900
SIGNALS = {signal.SIGTERM, signal.SIGINT}


def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': int(pid), 'starttime': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:
        return None


def pidfd(record):
    require(identity(record['pid']) == record, 'PID identity changed before registration')
    fd = os.pidfd_open(record['pid'])
    try:
        require(identity(record['pid']) == record, 'PID identity changed during registration')
    except BaseException:
        os.close(fd)
        raise
    return fd


def publication_temporary(path, folder):
    """Only these original-helper COMPLETE publication links may disappear."""
    if folder == ROOT:
        return path in {ROOT / (arm + '-packed') / 'D_select/COMPLETE.json.writing' for arm in ARMS}
    require(folder in {FOLLOWUP / (arm + '-seed42') for arm in ACTIVE_ARMS}, 'Unexpected usage namespace')
    relative = path.relative_to(folder).parts
    return (relative == ('COMPLETE.json.writing',) or
            (len(relative) == 2 and relative[1] == 'COMPLETE.json.writing' and
             any(relative[0].startswith(prefix) and len(relative[0]) > len(prefix)
                 for prefix in ('checkpoint-', 'D_select-'))))


def usage(folder):
    """Separate all-inode cap usage from conservative stable-inode credit.

    Transient-only bytes count toward the cap, never toward future deduction.
    A final stable alias credits its inode once, independent of temporary aliases.
    """
    require(folder.is_dir() and folder.resolve() == folder, 'Actual nonsymlink retained directory required: ' + str(folder))
    require(folder == ROOT or folder in {FOLLOWUP / (arm + '-seed42') for arm in ACTIVE_ARMS},
            'Only fixed new/active-old usage namespaces')
    cap_by_inode, stable_by_inode, stable_paths = {}, {}, []
    def walk_error(error):
        raise error  # Do not silently omit an unreadable/disappearing stable directory.
    for root, dirs, names in os.walk(folder, followlinks=False, onerror=walk_error):
        for name in dirs:
            require(modes.S_ISDIR((Path(root) / name).lstat().st_mode), 'Non-directory/symlink output directory')
        for name in names:
            p = Path(root) / name
            temporary = publication_temporary(p, folder)
            try:
                s = p.lstat()
            except FileNotFoundError:
                require(temporary, 'Retained file disappeared: ' + str(p))
                continue
            require(modes.S_ISREG(s.st_mode), 'Nonregular retained output: ' + str(p))
            inode = (s.st_dev, s.st_ino)
            cap_by_inode[inode] = max(cap_by_inode.get(inode, 0), s.st_size)
            if not temporary:
                stable_paths.append((p, inode, s.st_size))
    for p, inode, before_size in stable_paths:
        # Stable artifacts never receive the missing-temp exception, including
        # if they disappear/change inode after their first successful lstat.
        s = p.lstat()
        require(modes.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) == inode,
                'Stable retained file replaced: ' + str(p))
        cap_by_inode[inode] = max(cap_by_inode[inode], s.st_size)
        credit = min(before_size, s.st_size)
        stable_by_inode[inode] = min(stable_by_inode.get(inode, credit), credit)
    return {'cap_used_bytes': sum(cap_by_inode.values()),
            'future_deduction_bytes': sum(stable_by_inode.values())}


def disk_budget():
    """Conservative old caps NEVER released, even if an old completion appears.

    Do not pin/poll old mutable ledgers/controllers. All old files stay. Caps use
    every observed unique inode, including transient-only publication files;
    remaining allowances subtract only conservative stable unique-inode credit.
    """
    old_usage, outstanding = {}, {}
    for arm in ACTIVE_ARMS:
        folder = FOLLOWUP / (arm + '-seed42')
        require(folder.stat().st_dev == ROOT.stat().st_dev, 'Old/new budget must refer to the same filesystem')
        measured = usage(folder)
        require(measured['cap_used_bytes'] <= OLD_CAP, 'Active old arm exceeds its retained 25GiB cap: ' + str(folder))
        old_usage[str(folder)] = measured
        outstanding[str(folder)] = OLD_CAP - measured['future_deduction_bytes']
    own = usage(ROOT)
    require(own['cap_used_bytes'] <= NEW_CAP, 'New ROOT aggregate5GiB cap exceeded; preserve all outputs')
    new_remaining = NEW_CAP - own['future_deduction_bytes']
    free = shutil.disk_usage(ROOT).free
    required = sum(outstanding.values()) + OLD_EVIDENCE + new_remaining + SHARED
    require(free >= required, 'Old future + OLD evidence5 + NEW remaining5 + shared20GiB cannot fit: ' +
            dump({'free_bytes': free, 'required_free_bytes': required, 'old_remaining_bytes': outstanding,
                   'old_usage': old_usage, 'new_usage': own, 'new_remaining_bytes': new_remaining}))
    return {'free_bytes': free, 'required_free_bytes': required, 'old_usage': old_usage,
            'old_remaining_bytes': outstanding, 'old_evidence_bytes': OLD_EVIDENCE,
            'old_arm_cap_bytes': OLD_CAP, 'new_total_cap_bytes': NEW_CAP, 'new_usage': own,
            'new_remaining_bytes': new_remaining,
            'shared_floor_bytes': SHARED, 'old_completion_budget_release': False,
            'all_old_files_retained': True, 'system_wide_reservation': False,
            'active_old_controller_untouched': str(CONTROL)}


def nvidia():
    result = subprocess.run(['nvidia-smi', '-i', '0,1', '--query-gpu=index,uuid',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True,
                            text=True, timeout=20)
    pairs = [tuple(v.strip() for v in line.split(',')) for line in result.stdout.splitlines() if line.strip()]
    require(len(pairs) == 2 and all(len(p) == 2 for p in pairs), 'Invalid GPU0/1 inventory')
    devices = {int(i): uuid for i, uuid in pairs}
    require(set(devices) == {0, 1} and len(set(devices.values())) == 2 and
            all(u.startswith('GPU-') for u in devices.values()), 'GPU0/1 UUID scope differs')
    xml = subprocess.run(['nvidia-smi', '-i', '0,1', '-q', '-x'], check=True, capture_output=True,
                         text=True, timeout=20)
    gpus = ET.fromstring(xml.stdout).findall('gpu')
    require(len(gpus) == 2 and {g.findtext('uuid') for g in gpus} == set(devices.values()), 'XML UUID scope differs')
    occupants = {u: [] for u in devices.values()}
    for gpu in gpus:
        require('H800' in gpu.findtext('product_name', ''), 'Actual H800 required')
        processes = gpu.find('processes')
        require(processes is not None, 'Unavailable GPU process inventory')
        for process in processes.findall('process_info'):
            value = process.findtext('pid', '').strip()
            require(value.isdecimal(), 'Unidentified graphics/MPS/compute process; no assignment')
            occupants[gpu.findtext('uuid')].append(int(value))
    return devices, occupants


class Events:
    """Only own-root STOP/child-exit events and a 15s RESOURCE guard timer."""
    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1')
        self.wd = self.libc.inotify_add_watch(self.fd, os.fsencode(ROOT), 0x8 | 0x80 | 0x100 | 0x400 | 0x800)
        if self.wd < 0:
            os.close(self.fd)
            raise OSError(ctypes.get_errno(), 'inotify_add_watch')

    def wait(self, fds, timeout):
        ready, _, _ = select.select([self.fd, *fds], [], [], timeout)
        if self.fd in ready:
            while True:
                try:
                    data = os.read(self.fd, 65536)
                except BlockingIOError:
                    break
                require(data, 'inotify closed')
                offset = 0
                while offset < len(data):
                    wd, mask, cookie, size = struct.unpack_from('iIII', data, offset)
                    require(wd == self.wd and not mask & (0x4000 | 0x400 | 0x800 | 0x8000),
                            'Own-root event overflow/watch loss/movement')
                    offset += 16 + size
        return set(ready) - {self.fd}

    def close(self):
        os.close(self.fd)


class Manager:
    def __init__(self):
        self.owner = identity(os.getpid())
        self.children, self.leases = [], []
        self.lease_records = {}
        self.devices, self.events = None, None
        self.stop_requested, self.stop_signal = False, None

    def stop_check(self):
        require(not self.stop_requested and not any((ROOT / n).exists() or (ROOT / n).is_symlink()
                for n in ('STOP.json', 'STOP_QUEUE.json')),
                'External STOP; preserve/no automatic restart')

    def source_check(self):
        require('torch' not in sys.modules and 'transformers' not in sys.modules and 'peft' not in sys.modules,
                'SYSTEM manager must never import model libraries')
        require(sealed_config(ROOT / 'config.json') == self.cfg, 'Frozen config changed')
        verify([self.cfg_pin, self.install_pin])
        require((ROOT / 'config.json').read_bytes() == self.cfg_bytes, 'Exact config bytes changed')
        for arm in ARMS:
            require(producer_record(arm)[1] == self.cfg['producers'][arm], 'Original terminal producer changed')

    def guard(self):
        self.stop_check()
        self.source_check()
        for fd, (path, inode, data) in self.lease_records.items():
            fs, ls = os.fstat(fd), path.lstat()
            require(modes.S_ISREG(fs.st_mode) and modes.S_ISREG(ls.st_mode) and
                    fs.st_uid == ls.st_uid == os.getuid() and fs.st_nlink == ls.st_nlink == 1 and
                    modes.S_IMODE(fs.st_mode) == modes.S_IMODE(ls.st_mode) == 0o600 and
                    (fs.st_dev, fs.st_ino) == (ls.st_dev, ls.st_ino) == inode and
                    fs.st_size == len(data) and os.pread(fd, len(data) + 1, 0) == data,
                    'Held lease inode/owner/UUID/scope changed')
        devices, occupants = nvidia()
        require(devices == self.devices, 'Selected GPU physical UUID changed')
        for gpu, uuid in devices.items():
            allowed = {c['id']['pid']: c['id'] for c in self.children if c['gpu'] == gpu and
                       c['id'] is not None and c['p'] is not None and c['p'].returncode is None}
            for pid in occupants[uuid]:
                require(pid in allowed and identity(pid) == allowed[pid],
                        'Foreign process on selected GPU; stop only exact own0/1 children: ' +
                        dump({'gpu': gpu, 'uuid': uuid, 'pid': pid}))
        return disk_budget()

    def lease(self):
        try:
            LEASES.mkdir(mode=0o700)
            sync_dir(LEASES.parent)
        except FileExistsError:
            pass
        s = LEASES.lstat()
        require(modes.S_ISDIR(s.st_mode) and LEASES.resolve() == LEASES and
                s.st_uid == os.getuid() and modes.S_IMODE(s.st_mode) == 0o700, 'Unsafe global lease directory')
        for gpu in (0, 1):
            path = LEASES / ('gpu' + str(gpu) + '.lock')
            try:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            except FileExistsError:
                fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
            self.leases.append(fd)
            fs, ls = os.fstat(fd), path.lstat()
            require(modes.S_ISREG(fs.st_mode) and modes.S_ISREG(ls.st_mode) and
                    fs.st_uid == ls.st_uid == os.getuid() and fs.st_nlink == ls.st_nlink == 1 and
                    modes.S_IMODE(fs.st_mode) == modes.S_IMODE(ls.st_mode) == 0o600 and
                    (fs.st_dev, fs.st_ino) == (ls.st_dev, ls.st_ino), 'Unsafe cooperative lease')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Pre-flock observations do not authorize metadata writes.
            fs, ls = os.fstat(fd), path.lstat()
            require(modes.S_ISREG(fs.st_mode) and modes.S_ISREG(ls.st_mode) and
                    fs.st_uid == ls.st_uid == os.getuid() and fs.st_nlink == ls.st_nlink == 1 and
                    modes.S_IMODE(fs.st_mode) == modes.S_IMODE(ls.st_mode) == 0o600 and
                    (fs.st_dev, fs.st_ino) == (ls.st_dev, ls.st_ino), 'Lease changed before locked validation')
            require(fs.st_size == 0 and os.pread(fd, 1, 0) == b'', 'Nonempty unknown/stale lease retained; no takeover')
            data = (dump({'root': str(ROOT), 'gpu': gpu, 'uuid': self.devices[gpu], 'owner': self.owner,
                          'config': self.cfg_pin, 'source_binding': self.cfg['source_binding'],
                          'advisory_only': True, 'only_fixed375_select': True}) + '\n').encode()
            require(os.write(fd, data) == len(data), 'Partial lease metadata write')
            os.fsync(fd)
            self.lease_records[fd] = (path, (fs.st_dev, fs.st_ino), data)
        sync_dir(LEASES)

    def child_receipt(self, child):
        p = child['p']
        return {'arm': child['arm'], 'gpu': child['gpu'], 'uuid': self.devices[child['gpu']],
                'process': child['id'], 'created': p is not None, 'actual_wait_complete': child['waited'],
                'exit_code': None if p is None else p.returncode, 'output': str(ROOT / (child['arm'] + '-packed')),
                'wait_error': child.get('wait_error'), 'wait_note': child.get('wait_note')}

    def spawn(self, arm):
        self.guard()  # UUID/process/budget immediately before EACH spawn
        output = ROOT / (arm + '-packed')
        require(not output.exists() and not output.is_symlink(), 'Existing consumer output; no retry')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.devices[ASSIGNMENTS[arm]],
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', FLASH_ATTENTION_DETERMINISTIC='1',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   TOKENIZERS_PARALLELISM='false', USE_TORCH='1', USE_TF='0', USE_FLAX='0',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   WORLD_SIZE='1', RANK='0', LOCAL_RANK='0', WANDB_DISABLED='true')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        args = [MODEL_PYTHON, '-B', '-u', str(ROOT / 'consumer.py'), '--config', str(ROOT / 'config.json'),
                '--arm', arm, '--output', str(output)]
        child = {'p': None, 'id': None, 'fd': None, 'gpu': ASSIGNMENTS[arm], 'arm': arm, 'waited': False}
        parent = self.owner
        prior = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        try:
            self.children.append(child)  # slot exists before any constructor/close/registration failure
            def parent_guard():
                # Do not let an inherited Python manager handler consume a
                # parent-death TERM in the preexec window before exec resets it.
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                libc = ctypes.CDLL(None, use_errno=True)
                if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or identity(os.getppid()) != parent:
                    os._exit(125)
                signal.pthread_sigmask(signal.SIG_SETMASK, prior)
            with (ROOT / (arm + '.log')).open('x') as log:
                child['p'] = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True,
                                             preexec_fn=parent_guard)
            child['id'] = identity(child['p'].pid)
            require(child['id'] is not None, 'Child ended before exact registration; preserve/no retry')
            child['fd'] = pidfd(child['id'])
            commit(ROOT / (arm + '-OWNER.json'), self.child_receipt(child))
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, prior)
        self.guard()  # advisory lease is not GPU allocation; XML again after EACH spawn
        return child

    def wait_children(self):
        pending = {c['fd']: c for c in self.children}
        deadline = time.monotonic() + GUARD_SECONDS
        while pending:
            ready = self.events.wait(pending, max(0, deadline - time.monotonic()))
            self.stop_check()
            for fd in ready:
                child = pending.pop(fd)
                require(identity(child['id']['pid']) in (None, child['id']), 'Owned PID identity changed before reap')
                child['p'].wait()  # readable pidfd, no progress polling
                child['waited'] = True
                commit(ROOT / (child['arm'] + '-EXIT.json'), self.child_receipt(child))
                require(child['p'].returncode == 0, 'Programming/coverage/identity failure; no alternative/retry')
                self.result(child['arm'])  # actual metadata, not exit0 alone
            if time.monotonic() >= deadline:
                self.guard()
                deadline = time.monotonic() + GUARD_SECONDS
        self.guard()

    def result(self, arm):
        folder = ROOT / (arm + '-packed')
        require(folder.resolve() == folder and not list(folder.glob('BLOCKED*')), 'Consumer blocked/redirected')
        done = read_json(folder / 'COMPLETE.json')
        admission = read_json(folder / 'ADMISSION.json')
        numeric = read_json(folder / 'NUMERICAL.json')
        require(done['complete'] is True and all(done.get(k) is True for k in
                ('execution_complete', 'coverage_complete', 'weights_unchanged')) and
                all(done[k] == admission[k] for k in admission) and
                type(done['numerical_acceptance']) is bool and type(done['eligible_packed_candidate']) is bool and
                done['numerical_acceptance'] == done['eligible_packed_candidate'] == numeric['numerical_acceptance'] ==
                numeric['eligible_packed_candidate'] and numeric['case_count'] == 375 and
                len(numeric['per_case']) == 375 and done['scope']['producer_scope'] == self.cfg['producers'][arm]['scope'] and
                done['scope']['producer_final_checkpoint'] == self.cfg['producers'][arm]['terminal_complete'] and
                done['scope']['config_pin'] == self.cfg_pin and done['scope']['consumer_arm'] ==
                {'mode': 'causal', 'layout': 'packed', 'loss': 'BCE' if arm == 'C_D_BCE' else 'CE'} and
                all(done[k] is False for k in ('D_confirm_opened', 'public_run', 'winner_selected')),
                'Consumer execution/weights/eligibility/scope receipt differs')
        verify(done['files'])
        actual = list(folder.rglob('*'))
        require(not any(p.is_symlink() for p in actual) and
                {str(p) for p in actual if p.is_file() and p != folder / 'COMPLETE.json'} ==
                {r['path'] for r in done['files']}, 'Missing/unbound consumer files')
        return {'complete': pin(folder / 'COMPLETE.json'), 'admission': pin(folder / 'ADMISSION.json'),
                'numerical': pin(folder / 'NUMERICAL.json'), 'quality': pin(folder / 'QUALITY.json'),
                'execution_complete': True, 'coverage_complete': True, 'weights_unchanged': True,
                'numerical_acceptance': done['numerical_acceptance'],
                'eligible_packed_candidate': done['eligible_packed_candidate']}

    def await_owned_exits(self):
        """Unconditional actual-child barrier, including partial registration failures."""
        for child in self.children:
            p = child['p']
            if p is None or child['waited']:
                continue
            try:
                p.wait()
            except BaseException as error:
                child['wait_error'] = type(error).__name__ + ': ' + str(error)
                while True:
                    try:
                        returned, status = os.waitpid(p.pid, 0)  # actual child relation, never a guessed signal
                        require(returned == p.pid, 'Own waitpid returned another child')
                        p.returncode = os.waitstatus_to_exitcode(status)
                        break
                    except InterruptedError:
                        continue  # only interrupted blocking wait, not a work retry/poll
                    except ChildProcessError:
                        child['wait_note'] = 'own child already reaped; no fabricated exit code'
                        break
                    except BaseException as wait_error:
                        child['wait_error'] += '; waitpid: ' + type(wait_error).__name__ + ': ' + str(wait_error)
                        # No proof of exit => never escape to lease-close; no unknown signals.
                        signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
                        while True:
                            signal.sigwait(SIGNALS)
            child['waited'] = True

    def stop_owned(self):
        pending, errors = {}, []
        for child in self.children:
            if child['p'] is None or child['waited']:
                continue
            if child['id'] is None or child['fd'] is None:
                errors.append({'arm': child['arm'], 'signal': False, 'reason': 'unregistered Popen: wait only'})
                continue
            if identity(child['id']['pid']) == child['id']:
                try:
                    signal.pidfd_send_signal(child['fd'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    errors.append({'arm': child['arm'], 'TERM_error': str(error)})
                pending[child['fd']] = child
        deadline = time.monotonic() + GRACE_SECONDS
        while pending:
            ready, _, _ = select.select(list(pending), [], [], max(0, deadline - time.monotonic()))
            if not ready:
                for fd, child in pending.items():
                    if identity(child['id']['pid']) == child['id']:
                        try:
                            signal.pidfd_send_signal(fd, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except OSError as error:
                            errors.append({'arm': child['arm'], 'KILL_error': str(error)})
                # No lease release until the exact child actually exits, even kernel D-state.
                while pending:
                    exited, _, _ = select.select(list(pending), [], [])
                    for fd in exited:
                        child = pending.pop(fd)
                        child['p'].wait()
                        child['waited'] = True
                break
            for fd in ready:
                child = pending.pop(fd)
                child['p'].wait()
                child['waited'] = True
        self.await_owned_exits()  # unknown/unregistered own Popen also waits, never receives guessed signal
        return errors

    def run(self):
        require(Path(__file__).resolve() == ROOT / 'ops.py' and Path(__file__).absolute() == ROOT / 'ops.py',
                'Run only deployed manager, not LOCAL or another copy')
        # The model venv may legitimately symlink the SAME system binary; prefix,
        # not binary-inode inequality, distinguishes SYSTEM from model Python.
        require(Path(sys.executable).absolute().parent == Path('/usr/bin') and sys.prefix == sys.base_prefix,
                'Use /usr/bin/python3 SYSTEM manager')
        require(ROOT.is_dir() and ROOT.resolve() == ROOT and 'torch' not in sys.modules, 'Fixed Torch-free ROOT')
        require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'SYSTEM exact pidfd APIs required')
        require(not signal.pthread_sigmask(signal.SIG_BLOCK, set()) & SIGNALS,
                'Manager TERM/INT must not be inherited blocked')
        # This file permanently survives even a failed/aborted run. No restart path.
        lock = (ROOT / 'RUN.lock').open('x+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            def interrupted(signum, frame):
                # Never raise asynchronously out of the unconditional child wait
                # barrier. Guards/STOP events observe this within resource15s.
                self.stop_requested, self.stop_signal = True, signum
            for sig in SIGNALS:
                signal.signal(sig, interrupted)
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.cfg = sealed_config(ROOT / 'config.json')
            self.cfg_pin, self.cfg_bytes = pin(ROOT / 'config.json'), (ROOT / 'config.json').read_bytes()
            self.install_pin = pin(ROOT / 'INSTALL_COMPLETE.json')
            readonly(self.install_pin['path'])
            install_receipt = read_json(self.install_pin['path'])
            require(install_receipt['complete'] is True and install_receipt['config'] == self.cfg_pin and
                    install_receipt['source_binding'] == self.cfg['source_binding'] and
                    install_receipt['main_reviewed_deployment_ready'] is True and
                    install_receipt['workers_launched'] is False, 'Completed install-only source receipt required')
            self.events = Events()  # watch before initial STOP read; only own-root events
            self.stop_check()
            self.source_check()
            require(not list(ROOT.glob('*BLOCKED*')), 'Installation/run blocker preserved')
            for arm in ARMS:
                p = ROOT / (arm + '-packed')
                require(not p.exists() and not p.is_symlink(), 'No previous consumer output/restart')
            commit(ROOT / 'OWNER.json', {'owner': self.owner, 'config': self.cfg_pin,
                                        'source_binding': self.cfg['source_binding'], 'automatic_restart': False})
            self.devices, occupants = nvidia()
            require(not any(occupants.values()), 'GPU0/1 busy; BLOCKED without preemption/retry')
            budget = disk_budget()
            self.lease()
            self.guard()
            commit(ROOT / 'LAUNCH.json', {'owner': self.owner, 'devices': self.devices, 'assignments': ASSIGNMENTS,
                    'config': self.cfg_pin, 'budget': budget, 'leases_fsynced_before_spawn': True,
                    'single_full375_attempt_per_arm': True, 'gpus2_3_touched': False})
            for arm in ARMS:
                self.spawn(arm)
            self.wait_children()
            results = {arm: self.result(arm) for arm in ARMS}
            self.source_check()
            final_budget = self.guard()
            require(all(c['waited'] and c['p'].returncode == 0 for c in self.children) and len(self.children) == 2,
                    'Actual both-child exit0/wait barrier required')
            commit(ROOT / 'MANAGER_COMPLETE.json', {'execution_complete': True, 'owner': self.owner,
                    'arms': results, 'workers': [self.child_receipt(c) for c in self.children],
                    'all_actual_exits_zero': True, 'resource_budget': final_budget,
                    'numeric_negative_is_complete_scientific_result': True, 'automatic_retry': False,
                    'D_confirm_opened': False, 'public_run': False, 'winner_selected': False})
            return 0
        except BaseException as error:
            for sig in SIGNALS:
                signal.signal(sig, signal.SIG_IGN)
            try:
                shutdown = self.stop_owned()
            except BaseException as cleanup_error:
                shutdown = [{'error_type': type(cleanup_error).__name__, 'error': str(cleanup_error),
                             'unconditional_actual_wait_still_required': True}]
            self.await_owned_exits()
            commit(ROOT / 'MANAGER_BLOCKED.json', {'execution_complete': False, 'owner': self.owner,
                    'error_type': type(error).__name__, 'error': str(error), 'shutdown': shutdown,
                    'workers': [self.child_receipt(c) for c in self.children], 'partial_preserved': True,
                    'automatic_restart': False, 'gpu2_3_signals': 0, 'public_run': False})
            raise
        finally:
            for sig in SIGNALS:
                signal.signal(sig, signal.SIG_IGN)
            self.await_owned_exits()  # independent of every cleanup/introspection/receipt failure
            for child in self.children:
                if child['fd'] is not None:
                    os.close(child['fd'])
            if self.events is not None:
                self.events.close()
            for fd in self.leases:
                os.close(fd)  # metadata intentionally retained, no stale-record takeover
            lock.close()


def public_source(name):
    """Read only LOCAL/{consumer,ops}.py using one held nofollow component chain."""
    require(name in ('consumer.py', 'ops.py'), 'Only two public sources are uploadable')
    p = LOCAL / name
    fds, chain = [], []
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    def state(s):
        return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid, s.st_nlink,
                s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    try:
        parent = os.open('/', flags)
        fds.append(parent)
        for component in p.parts[1:-1]:
            held = os.open(component, flags, dir_fd=parent)
            fds.append(held)
            chain.append((parent, component, held))
            parent = held
        published = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(modes.S_ISREG(published.st_mode) and published.st_nlink == 1, 'Regular single-link public source only')
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
        fds.append(fd)
        before = os.fstat(fd)
        require(modes.S_ISREG(before.st_mode) and before.st_nlink == 1 and state(before) == state(published),
                'Local source changed during nofollow open')
        def check():
            for directory, component, held in chain:
                current, opened = os.stat(component, dir_fd=directory, follow_symlinks=False), os.fstat(held)
                require(modes.S_ISDIR(current.st_mode) and modes.S_ISDIR(opened.st_mode) and
                        (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino), 'Source ancestor replaced')
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            require(state(current) == state(before) == state(os.fstat(fd)), 'Local source FD/published name changed')
        check()
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b''.join(chunks)
        check()
        require(len(data) == before.st_size, 'Public source byte count changed')
        return {'stat': {'path': str(p), 'resolved': str(p), 'bytes': len(data), 'mtime_ns': before.st_mtime_ns},
                'base64': base64.b64encode(data).decode('ascii')}
    finally:
        for fd in reversed(fds):
            os.close(fd)


REMOTE_INSTALL = r'''
import base64,json,os,stat,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('/tmp/opencode/jev-terminal-packed-admission-20261112-v1')
os.umask(0o077)
if ROOT.exists() or ROOT.is_symlink():raise RuntimeError('Existing ROOT preserved; exclusive install only')
if ROOT.parent.resolve()!=ROOT.parent:raise RuntimeError('Redirected parent')
ROOT.mkdir(mode=0o700)
try:
 for name,item in SOURCES.items():
  if name not in ('consumer.py','ops.py'):raise RuntimeError('Unexpected public source')
  data=base64.b64decode(item['base64'],validate=True)
  with (ROOT/name).open('xb') as f:
   f.write(data);f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
  if (ROOT/name).read_bytes()!=data:raise RuntimeError('Source byte readback differs')
 sys.path.insert(0,str(ROOT))
 from consumer import build_config,commit,pin,require,sealed_config,sync_dir
 from ops import disk_budget
 require(set(SOURCES)=={'consumer.py','ops.py'} and DEPLOYMENT_READY is True,'MAIN deployment-ready required')
 commit(ROOT/'INSTALL_OWNER.json',{'source_local_FD_records':{n:v['stat'] for n,v in SOURCES.items()},
  'main_reviewed_deployment_ready':True,'workers_launched':False})
 budget=disk_budget()
 cfg=build_config(DEPLOYMENT_READY)
 require('torch' not in sys.modules and 'transformers' not in sys.modules and 'peft' not in sys.modules,'Install must be stdlib only')
 for item in cfg['source_files']:
  name=Path(item['path']).name
  require((ROOT/name).read_bytes()==base64.b64decode(SOURCES[name]['base64'],validate=True),'Final upload exact bytes differ')
 require(sealed_config(ROOT/'config.json')==cfg,'Final config/source readback failed')
 sync_dir(ROOT)
 commit(ROOT/'INSTALL_COMPLETE.json',{'complete':True,'root':str(ROOT),'config':pin(ROOT/'config.json'),
  'source_binding':cfg['source_binding'],'source_files':cfg['source_files'],'exact_byte_readback':True,
  'main_reviewed_deployment_ready':True,'workers_launched':False,'model_forward_calls':0,'Torch_imported':False,
  'budget':budget,'D_confirm_opened':False,'public_run':False})
 print(json.dumps({'complete':True,'install':pin(ROOT/'INSTALL_COMPLETE.json'),'config':pin(ROOT/'config.json'),
  'exact_byte_readback':True,'workers_launched':False},sort_keys=True))
except BaseException as error:
 with (ROOT/'INSTALL_BLOCKED.json').open('x') as f:
  json.dump({'complete':False,'error_type':type(error).__name__,'error':str(error),'partial_preserved':True,
   'workers_launched':False,'automatic_retry':False},f);f.write('\n');f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
 raise
'''


def install(host, log_path, deployment_ready):
    require(Path(__file__).absolute() == LOCAL / 'ops.py' and Path(__file__).resolve() == LOCAL / 'ops.py',
            'Install from fixed reviewed LOCAL source only')
    require(host == 'sh-10-gpu' and deployment_ready is True, 'Explicit main-reviewed deployment-ready install only')
    require(log_path.is_absolute() and log_path.parent == LOCAL and log_path.parent.resolve() == LOCAL and
            log_path.name not in ('consumer.py', 'ops.py', 'implementation-report.md') and not log_path.is_symlink(),
            'New direct-child LOCAL execution log required')
    sources = {name: public_source(name) for name in ('consumer.py', 'ops.py')}
    script = 'SOURCES=' + repr(sources) + '\nDEPLOYMENT_READY=True\n' + REMOTE_INSTALL
    # Exclusive durable log BEFORE the first SSH/network side effect.
    with log_path.open('x') as log:
        log.write(dump({'operation': 'install-only', 'root': str(ROOT), 'host': host,
                        'main_reviewed_deployment_ready': True,
                        'public_sources': {n: v['stat'] for n, v in sources.items()}, 'workers_launched': False}) + '\n')
        log.flush()
        os.fsync(log.fileno())
        sync_dir(LOCAL)
        result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectionAttempts=1',
                '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ControlMaster=no',
                '-o', 'ControlPath=none', '-o', 'ForwardAgent=no', '-o', 'ForwardX11=no',
                '-o', 'PermitLocalCommand=no', host, '/usr/bin/python3 -B -'], input=script,
                stdout=log, stderr=subprocess.STDOUT, text=True, timeout=300)
        log.flush()
        os.fsync(log.fileno())
    require(result.returncode == 0, 'Install failed; preserve exclusive local log/remote partial; no retry')
    print(dump({'installed': True, 'workers_launched': False, 'execution_log': str(log_path)}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=('install', 'run'))
    parser.add_argument('--root', required=True, choices=(str(ROOT),))
    parser.add_argument('--host', choices=('sh-10-gpu',))
    parser.add_argument('--log', type=Path)
    parser.add_argument('--deployment-ready', action='store_true', help='MAIN reviewed these exact two sources before install')
    args = parser.parse_args()
    os.umask(0o077)
    require('torch' not in sys.modules, 'Ops must stay Torch-free')
    if args.action == 'install':
        require(args.host is not None and args.log is not None and args.deployment_ready,
                'Install needs --host --log --deployment-ready; never starts run')
        return install(args.host, args.log, args.deployment_ready)
    require(args.host is None and args.log is None and not args.deployment_ready, 'Run accepts no install/network flags')
    return Manager().run()


if __name__ == '__main__':
    raise SystemExit(main())
