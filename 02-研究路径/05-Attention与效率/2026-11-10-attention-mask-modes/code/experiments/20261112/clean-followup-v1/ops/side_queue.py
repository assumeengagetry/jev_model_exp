"""One-shot GPU2/3 scientific queue. SYSTEM python3; no Torch/model imports.

Invariants: original-four exit -> both leases/budget -> formal macro0/exit75 ->
saved-only primary -> external approval -> explicit cursor1 resume. Never retry.
"""
import argparse
import base64
import ctypes
import fcntl
import json
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
import traceback
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
OLD = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
LEASES = Path('/tmp/opencode/jev-owned-h800-leases-20261112')
PYTHON = '/home/dev/assumeengage/envs/jev/bin/python'
ARMS = {'STATE_NL_PB_P_CE': 2, 'JPAIR_P_BCE_RANK': 3}
FIRST = ('C_D_BCE', 'C_P_BCE', 'C_D_CE', 'C_P_CE')
CURSOR = {'macro_updates': 3557, 'cases': 56912, 'logical_pair_tokens': 99824365}
ONE = {'macro_updates': 1, 'cases': 16, 'logical_pair_tokens': 20947}
CAP, GUARD_SECONDS, GRACE_SECONDS = 25 * 2**30, 15, 900
PIN_KEYS = {'path', 'resolved', 'bytes', 'mtime_ns'}
SOURCE_NAMES = ('bindings.py', 'fresh_init.py', 'case_runtime.py', 'objectives.py', 'model_runtime.py',
                'checkpoints.py', 'train_case.py', 'evaluate_native_dev.py', 'state_boundaries.py')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def pin(path):
    p = Path(path)
    s = p.stat()
    require(modes.S_ISREG(s.st_mode), 'Not regular: ' + str(p))
    return {'path': str(p), 'resolved': str(p.resolve()), 'bytes': s.st_size,
            'mtime_ns': s.st_mtime_ns}


def verify(value):
    """Verify every nested stat pin, without opening raw confirm/model data."""
    if isinstance(value, dict):
        if set(value) == PIN_KEYS:
            require(pin(value['path']) == value, 'Changed pin: ' + value['path'])
        else:
            for item in value.values():
                verify(item)
    elif isinstance(value, list):
        for item in value:
            verify(item)


def readonly(path):
    p = Path(path)
    require(not p.is_symlink() and p.resolve() == p and
            modes.S_ISREG(p.lstat().st_mode) and p.stat().st_mode & 0o222 == 0,
            'Immutable nonsymlink file required: ' + str(p))


def exact(entry):
    verify(entry['stat'])
    require(Path(entry['stat']['path']).read_bytes() ==
            base64.b64decode(entry['base64'], validate=True), 'Exact source/input bytes changed')


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def commit(path, value):
    """Durable exclusive .writing + link publication; never replace old evidence."""
    p = Path(path)
    pending = p.with_name(p.name + '.writing')
    with pending.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fchmod(f.fileno(), 0o444)
        os.fsync(f.fileno())
    os.link(pending, p, follow_symlinks=False)
    pending.unlink()  # Only this invocation's committed staging link.
    sync_dir(p.parent)


def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': int(pid), 'starttime': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:
        return None


def process_record(value):
    return {k: value[k] for k in ('pid', 'starttime', 'boot_id')}


def exact_pidfd(record):
    current = identity(record['pid'])
    require(current is None or current == record, 'PID reuse/identity mismatch')
    if current is None:
        return None
    try:
        fd = os.pidfd_open(record['pid'])
    except ProcessLookupError:
        require(identity(record['pid']) is None, 'PID reuse during pidfd open')
        return None
    if identity(record['pid']) != record:
        os.close(fd)
        raise RuntimeError('Identity changed during pidfd open')
    return fd


def require_dead(record):
    fd = exact_pidfd(record)
    if fd is not None:
        try:
            require(bool(select.select([fd], [], [], 0)[0]), 'Original exact worker still live')
        finally:
            os.close(fd)


class Events:
    """Watch directories BEFORE initial reads. Event-driven; no progress polling."""
    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1')
        self.watches = {}
        try:
            for folder in (ROOT, OLD):
                wd = self.libc.inotify_add_watch(self.fd, os.fsencode(folder),
                                               0x8 | 0x80 | 0x100 | 0x400 | 0x800)
                if wd < 0:
                    raise OSError(ctypes.get_errno(), 'inotify_add_watch')
                self.watches[wd] = folder
        except BaseException:
            self.close()
            raise

    def wait(self, pidfds=(), timeout=None):
        ready, _, _ = select.select([self.fd, *pidfds], [], [], timeout)
        paths = set()
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
                    name = os.fsdecode(data[offset + 16:offset + 16 + size].split(b'\0', 1)[0])
                    offset += 16 + size
                    require(not mask & (0x4000 | 0x400 | 0x800 | 0x8000),
                            'inotify overflow/watch loss/root movement; fail closed')
                    require(wd in self.watches, 'Unknown inotify watch')
                    paths.add(self.watches[wd] / name)
        return paths, set(ready) - {self.fd}

    def close(self):
        os.close(self.fd)


def used(folder):
    if not folder.exists():
        return 0
    require(not folder.is_symlink() and folder.is_dir(), 'Unsafe output directory')
    total = 0
    for root, dirs, files in os.walk(folder, followlinks=False):
        require(not any((Path(root) / d).is_symlink() for d in dirs), 'Symlink directory')
        for name in files:
            p = Path(root) / name
            require(not p.is_symlink() and modes.S_ISREG(p.lstat().st_mode), 'Nonregular output')
            total += p.stat().st_size
    return total


def disk_budget(completed):
    outstanding, actual = {}, {}
    for parent, arms in ((OLD, ('BI_P_CE', 'PB_P_CE')), (ROOT, tuple(ARMS))):
        for arm in arms:
            folder = parent / (arm + '-seed42')
            n = used(folder)
            require(n <= CAP, 'Retained arm exceeds 25GiB: ' + str(folder))
            actual[str(folder)] = n
            remaining = 0 if str(folder) in completed else CAP - n
            outstanding[str(folder)] = remaining
    free = shutil.disk_usage(ROOT).free
    required = sum(outstanding.values()) + 5 * 2**30 + 20 * 2**30
    require(free >= required, 'Known old BI/PB + BOTH new arms + evidence/shared floor cannot fit: ' +
            json.dumps({'free_bytes': free, 'required_bytes': required, 'outstanding': outstanding}, sort_keys=True))
    return {'free_bytes': free, 'required_free_bytes': required, 'retained_bytes': actual,
            'outstanding_bytes': outstanding, 'evidence_bytes': 5 * 2**30,
            'shared_floor_bytes': 20 * 2**30, 'system_wide_reservation': False}


def nvidia():
    def query(fields):
        p = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=' + fields,
                            '--format=csv,noheader,nounits'], check=True, capture_output=True,
                           text=True, timeout=20)
        return [tuple(x.strip() for x in line.split(',')) for line in p.stdout.splitlines() if line.strip()]
    devices = {int(i): uuid for i, uuid in query('index,uuid')}
    require(set(devices) == {2, 3} and len(set(devices.values())) == 2 and
            all(u.startswith('GPU-') for u in devices.values()), 'GPU2/3 UUID inventory differs')
    occupants = {u: [] for u in devices.values()}
    # XML process_info includes graphics/MPS as well as compute processes.
    process_query = subprocess.run(['nvidia-smi', '-i', '2,3', '-q', '-x'], check=True,
                                   capture_output=True, text=True, timeout=20)
    gpus = ET.fromstring(process_query.stdout).findall('gpu')
    require(len(gpus) == 2 and {g.findtext('uuid') for g in gpus} == set(occupants), 'GPU process UUID scope differs')
    for gpu in gpus:
        processes = gpu.find('processes')
        require(processes is not None, 'GPU process inventory unavailable')
        for process in processes.findall('process_info'):
            pid = process.findtext('pid', '').strip()
            require(pid.isdecimal(), 'Unidentified GPU process; fail closed')
            occupants[gpu.findtext('uuid')].append(int(pid))
    return devices, occupants


def scope_check(scope, cfg, arm):
    require(scope['config'] == cfg and scope['arm_name'] == arm and scope['arm'] == cfg['arms'][arm],
            'Actual scope/config/arm differs')
    require(scope['config_pin'] == pin(Path(cfg['output_parent']) / 'seed42.json'), 'Wrong config pin')
    for key, files in (('trainer_code', 'trainer_files'), ('core_code', 'core_files')):
        require(scope[key] == {p['path']: base64.b64encode(Path(p['path']).read_bytes()).decode()
                               for p in cfg[files]}, 'Actual source bytes differ: ' + key)
    verify(scope)


def committed(marker, scope):
    require(marker.resolve() == marker and not marker.is_symlink(), 'Commit redirected')
    verify(pin(marker))
    value = read(marker)
    require(value['complete'] is True and value['scope'] == scope, 'Committed scope differs')
    verify(value)
    if 'files' in value:
        paths = list(marker.parent.rglob('*'))
        require(not any(p.is_symlink() for p in paths), 'Symlink committed output')
        files = {str(p) for p in paths if p.is_file() and p != marker}
        require(files == {p['path'] for p in value['files']}, 'Unbound/missing committed files')
    return value


def actual_nodes(done, folder, scope):
    schedule = Path(scope['config']['data_root']) / 'data/schedule-seed42.json'
    require(pin(schedule) in scope['data']['files'] and done['schedule'] == read(schedule),
            'Actual completion schedule is not the pinned prepared seed42 schedule')
    nodes = done['schedule']['observations']
    expected = {n['cap']: n for n in nodes}
    require(len(nodes) == len(done['D_select']) == 3 and
            set(expected) == {10000000, 30000000, 100000000}, 'Actual node inventory differs')
    seen = set()
    # Trainer stores lexically sorted filenames, NOT numerically sorted caps.
    for node_pin in done['D_select']:
        path = Path(node_pin['path'])
        require(path.parent.parent == folder and path.name == 'COMPLETE.json', 'Node redirected')
        observed = committed(path, scope)
        cap = observed['node']['cap']
        require(cap in expected and cap not in seen and observed['node'] == expected[cap] and
                observed['checkpoint_identity']['macro_updates'] == expected[cap]['macro_updates'],
                'Actual node cursor/scope differs')
        summary = read(path.parent / 'SUMMARY.json')
        require(summary['scope'] == scope and summary['node'] == expected[cap] and
                summary['D_confirm_opened'] is False and summary['public_run'] is False and
                summary['winner_selected'] is False, 'Node scientific scope differs')
        seen.add(cap)


class Queue:
    def __init__(self):
        self.owner = identity(os.getpid())
        self.children, self.leases = [], []
        self.lease_records = {}
        self.events, self.controller_fd = None, None
        self.devices, self.scopes = None, {}
        self.eligible = None
        self.disk_done = set()

    def stop_check(self):
        require(not any((ROOT / name).exists() or (ROOT / name).is_symlink()
                        for name in ('STOP.json', 'STOP_QUEUE.json')), 'External STOP; no automatic resume')

    def pins_check(self):
        require(pin(ROOT / 'OPS_BUNDLE.json') == self.bundle_pin and
                (ROOT / 'OPS_BUNDLE.json').read_bytes() == self.bundle_bytes, 'OPS bundle stat/bytes changed')
        readonly(ROOT / 'OPS_BUNDLE.json')
        require(self.bundle['installation']['path'] == str(ROOT / 'OPS_INSTALLING.json'), 'Ops installation redirected')
        verify(self.bundle['installation'])
        for item in self.bundle['files']:
            exact(item)
            readonly(item['stat']['path'])
        for item in self.bundle['evidence']:
            exact(item)
        for item in self.bundle['original']['files']:
            exact(item)
        verify(self.bundle['fresh'])
        verify(self.bundle['fresh_files'])
        if self.eligible is not None:
            verify(self.eligible)
        require(read(ROOT / 'FRESH_PRIMARY_VERIFICATION.json') == self.bundle['fresh'] and
                self.bundle['fresh']['complete'] is True and
                self.bundle['fresh']['393_initial_tensors_bitwise_equal_old_v2'] is True and
                self.bundle['fresh']['actual_saved_fp32_tensor_keys'] == 393 and
                self.bundle['fresh']['final_config'] == pin(ROOT / 'seed42.json') and
                self.bundle['fresh']['new_init_complete'] == pin(ROOT / 'init-seed42/COMPLETE.json'),
                'Actual fresh393/config evidence changed')
        require(not list(ROOT.glob('BOOTSTRAP_BLOCKED*')) and not list(ROOT.glob('OPS_BLOCKED*')),
                'Bootstrap/ops blocked')

    def guard(self):
        self.stop_check()
        for fd, (path, inode, data) in self.lease_records.items():
            fs, ls = os.fstat(fd), path.lstat()
            require((ls.st_dev, ls.st_ino) == inode == (fs.st_dev, fs.st_ino) and
                    modes.S_ISREG(ls.st_mode) and ls.st_uid == os.getuid() and ls.st_nlink == 1 and
                    modes.S_IMODE(ls.st_mode) == 0o600 and os.pread(fd, len(data) + 1, 0) == data,
                    'Held cooperative lease path/owner/UUID record changed')
        devices, occupants = nvidia()
        require(devices == self.devices, 'Physical GPU UUID changed')
        for gpu, uuid in devices.items():
            allowed = {c['id']['pid']: c['id'] for c in self.children
                       if c['gpu'] == gpu and c['p'] is not None and c['id'] is not None and
                       c['p'].returncode is None}
            for pid in occupants[uuid]:
                current = identity(pid)
                require(pid in allowed and current == allowed[pid],
                        'Foreign GPU process; stop only exact owned children: ' +
                        json.dumps({'gpu': gpu, 'uuid': uuid, 'pid': pid, 'identity': current}, sort_keys=True))
        return disk_budget(self.disk_done)

    def disk_completion(self, parent, arm):
        """Initial admission / completion event / exact child exit ONLY, never timer."""
        folder = parent / (arm + '-seed42')
        marker = folder / 'COMPLETE.json'
        if marker.exists():
            done = read(marker)
            require(done['complete'] is True and done['cursor'] == CURSOR and
                    done['observations'] == [10000000, 30000000, 100000000] and
                    done['D_confirm_opened'] is False and done['public_run'] is False,
                    'Invalid old/new completion in disk accounting')
            self.disk_done.add(str(folder))

    def resource_events(self, paths):
        if OLD / 'fresh-mask-COMPLETE.json' in paths:
            for arm in ('BI_P_CE', 'PB_P_CE'):
                self.disk_completion(OLD, arm)

    def lease(self):
        try:
            LEASES.mkdir(mode=0o700)
            sync_dir(LEASES.parent)
        except FileExistsError:
            pass
        s = LEASES.lstat()
        require(modes.S_ISDIR(s.st_mode) and not LEASES.is_symlink() and
                LEASES.resolve() == LEASES and s.st_uid == os.getuid() and
                modes.S_IMODE(s.st_mode) == 0o700, 'Unsafe cooperative lease directory')
        for gpu in (2, 3):
            path = LEASES / ('gpu' + str(gpu) + '.lock')
            try:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            except FileExistsError:
                fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
            self.leases.append(fd)
            fs, ls = os.fstat(fd), path.lstat()
            require(modes.S_ISREG(fs.st_mode) and fs.st_uid == os.getuid() and fs.st_nlink == 1 and
                    modes.S_IMODE(fs.st_mode) == 0o600 and (fs.st_dev, fs.st_ino) == (ls.st_dev, ls.st_ino),
                    'Unsafe lease lock')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Pre-lock metadata cannot authorize a write after a competing holder.
            fs, ls = os.fstat(fd), path.lstat()
            require(modes.S_ISREG(fs.st_mode) and modes.S_ISREG(ls.st_mode) and
                    fs.st_uid == ls.st_uid == os.getuid() and fs.st_nlink == ls.st_nlink == 1 and
                    modes.S_IMODE(fs.st_mode) == modes.S_IMODE(ls.st_mode) == 0o600 and
                    (fs.st_dev, fs.st_ino) == (ls.st_dev, ls.st_ino), 'Lease changed before locked validation')
            # Fail closed even for a stale foreign record: explicit new scope needed.
            require(fs.st_size == 0 and os.pread(fd, 1, 0) == b'',
                    'Existing foreign/stale lease metadata preserved; no takeover')
            data = json.dumps({'root': str(ROOT), 'gpu': gpu, 'uuid': self.devices[gpu],
                               'owner': self.owner, 'bundle': self.bundle_pin,
                               'advisory_only': True}, sort_keys=True).encode() + b'\n'
            require(os.write(fd, data) == len(data), 'Partial lease metadata write')
            os.fsync(fd)
            self.lease_records[fd] = (path, (fs.st_dev, fs.st_ino), data)
        sync_dir(LEASES)

    def controller_check(self):
        original = self.bundle['original']
        owner = process_record(read(OLD / 'CONTINUATION_OWNER.json'))
        require(owner == original['owner'], 'Original controller owner differs')
        current = identity(owner['pid'])
        require(current is None or current == owner, 'Original controller PID reused')
        if current is not None and (self.controller_fd is None or
                                   not select.select([self.controller_fd], [], [], 0)[0]):
            cmdline = (Path('/proc') / str(owner['pid']) / 'cmdline').read_bytes()
            environ = (Path('/proc') / str(owner['pid']) / 'environ').read_bytes().split(b'\0')
            require(cmdline == base64.b64decode(original['cmdline_base64'], validate=True) and
                    ('CLEAN_TRAIN_ROOT=' + str(OLD)).encode() in environ,
                    'Original controller executable/root changed')
        # Pinned exact archived controller's only post-four tasks are BI/PB on0/1.
        return owner

    def original_complete(self):
        self.pins_check()
        self.controller_check()
        marker = OLD / 'continuation-COMPLETE.json'
        done = read(marker)
        active = read(OLD / 'continuation-ACTIVE.json')
        cfg = read(OLD / 'seed42.json')
        require(done['execution_complete'] is True and done['all_completed'] is True and
                set(done['workers']) == set(FIRST) and active['owner'] == self.bundle['original']['owner'],
                'Original four producer receipt/owner incomplete')
        review = read(OLD / 'FIRST_PRIMARY_VERIFICATION.json')
        require(review['complete'] is True and review['all_four_formal_macro0_verified'] is True and
                review['repeat_macro0_allowed'] is False, 'Original formal-first review absent')
        receipts = {}
        for gpu, arm in enumerate(FIRST):
            worker = done['workers'][arm]
            folder = OLD / (arm + '-seed42')
            require(worker['state'] == 'completed' and worker['exit_code'] == 0 and worker['gpu'] == gpu and
                    worker['resumed_cursor'] == 1 and worker['output'] == str(folder) and
                    worker['process'] == active['workers'][arm]['process'],
                    'Original actual worker/exit binding differs')
            require_dead(worker['process'])
            final = read(folder / 'COMPLETE.json')
            scope = final['scope']
            scope_check(scope, cfg, arm)
            require(read(folder / 'OWN_SCOPE.json') == scope and final['complete'] is True and
                    final['cursor'] == CURSOR and final['observations'] == [10000000, 30000000, 100000000] and
                    final['D_confirm_opened'] is False and final['public_run'] is False,
                    'Original actual completion/cursor/scope differs')
            verify(final)
            require(worker['actual_completion'] == {k: final[k] for k in
                    ('cursor', 'observations', 'actual_runtime', 'first_actual_macro')},
                    'Original receipt not bound to actual completion')
            require(final['first_actual_macro'] == pin(folder / 'first-actual-macro/COMPLETE.json') and
                    final['final_checkpoint'] in final['checkpoints'], 'Original first/final pointer differs')
            first = committed(folder / 'first-actual-macro/COMPLETE.json', scope)
            require(first['is_formal_exposure_macro0'] is True, 'Original first was not formal macro0')
            for checkpoint in final['checkpoints']:
                require(Path(checkpoint['path']).parent.parent == folder, 'Original checkpoint redirected')
                committed(Path(checkpoint['path']), scope)
            require(read(Path(final['final_checkpoint']['path']).parent / 'CHECKPOINT.json')['cursor'] == CURSOR,
                    'Original final checkpoint cursor differs')
            actual_nodes(final, folder, scope)
            receipts[arm] = {'complete': pin(folder / 'COMPLETE.json'), 'worker': worker,
                             'first': final['first_actual_macro'], 'nodes': final['D_select'],
                             'final_checkpoint': final['final_checkpoint']}
        self.eligible = {'complete': True, 'receipt': pin(marker),
               'active': pin(OLD / 'continuation-ACTIVE.json'), 'arms': receipts,
               'controller': self.bundle['original']['owner'], 'later_controller_gpus': [0, 1]}
        commit(ROOT / 'ORIGINAL_FOUR_ELIGIBLE.json', self.eligible)

    def wait_original(self):
        owner = self.bundle['original']['owner']
        self.controller_fd = exact_pidfd(owner)
        self.controller_check()
        commit(ROOT / 'WAITING_ORIGINAL.json', {'owner': self.owner, 'original_controller': owner,
                                               'progress_polling': False})
        # Watch was armed before this single initial read; only receipt events reread.
        marker = OLD / 'continuation-COMPLETE.json'
        if marker.exists():
            self.original_complete()
            return
        while True:
            self.stop_check()
            require(self.controller_fd is not None,
                    'Original controller died without four-arm phase completion')
            paths, dead = self.events.wait([self.controller_fd])
            self.stop_check()
            if marker not in paths and not dead:
                # Unrelated inotify events never cause a progress/artifact read.
                continue
            if marker.exists():
                self.original_complete()
                return
            require(self.controller_fd not in dead, 'Original controller exited without phase receipt')

    def spawn(self, arm, phase, resume=None, verifier=False):
        self.pins_check()
        self.guard()
        output = ROOT / (arm + '-seed42')
        if verifier:
            args = [PYTHON, '-B', '-u', str(ROOT / 'ops/verify_first.py'), '--root', str(ROOT), '--arm', arm]
        else:
            args = [PYTHON, '-B', '-u', str(ROOT / 'source/train_case.py'), '--config', str(ROOT / 'seed42.json'),
                    '--arm', arm, '--output', str(output)]
            if resume is None:
                require(not output.exists() and not output.is_symlink(), 'Existing new arm; no retry')
                args += ['--pause-after-macros', '1']
            else:
                args += ['--resume-checkpoint', resume['checkpoint'], '--resume-complete-pin', resume['pin_file']]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if verifier else self.devices[ARMS[arm]],
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', FLASH_ATTENTION_DETERMINISTIC='1',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   USE_TORCH='1', USE_TF='0', USE_FLAX='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0', WANDB_DISABLED='true')
        parent = self.owner
        child = {'p': None, 'id': None, 'fd': None, 'gpu': None if verifier else ARMS[arm], 'arm': arm,
                 'phase': phase, 'output': str(output), 'waited': False}
        prior = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            # Allocate/track the slot BEFORE construction. Store the returned Popen
            # inside the log context, before close/identity/pidfd/commit can fail.
            self.children.append(child)
            def parent_guard():
                libc = ctypes.CDLL(None, use_errno=True)
                if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or identity(os.getppid()) != parent:
                    os._exit(125)
                signal.pthread_sigmask(signal.SIG_SETMASK, prior)
            with (ROOT / (arm + '-' + phase + '.log')).open('x') as log:
                child['p'] = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True,
                                             preexec_fn=parent_guard)
            child['id'] = identity(child['p'].pid)
            require(child['id'] is not None, 'Worker exited before exact registration; preserve/no retry')
            child['fd'] = exact_pidfd(child['id'])
            require(child['fd'] is not None, 'Worker exited before pidfd registration')
            commit(ROOT / (arm + '-' + phase + '-OWNER.json'), self.child_receipt(child))
        finally:
            # A pending handler can raise here only after every created Popen is
            # reachable by cleanup/final exit barrier, even if registration failed.
            signal.pthread_sigmask(signal.SIG_SETMASK, prior)
        self.guard()  # Query again after EACH spawn; advisory locks are not allocation.
        return child

    def child_receipt(self, child):
        p = child['p']
        return {'arm': child['arm'], 'phase': child['phase'], 'process': child['id'],
                'gpu': child['gpu'], 'uuid': self.devices.get(child['gpu']),
                'output': child['output'], 'created': p is not None,
                'exit_code': p.returncode if p is not None else None,
                'actual_wait_complete': child['waited'] or (p is not None and p.returncode is not None),
                'wait_note': child.get('wait_note'), 'wait_error': child.get('wait_error')}

    def wait_children(self, children):
        pending = {c['fd']: c for c in children}
        next_guard = time.monotonic() + GUARD_SECONDS
        while pending:
            paths, dead = self.events.wait(pending, max(0, next_guard - time.monotonic()))
            self.stop_check()
            self.resource_events(paths)
            for fd in dead:
                child = pending.pop(fd)
                require(identity(child['id']['pid']) in (None, child['id']), 'Owned PID changed before reap')
                child['p'].wait()  # pidfd is readable: no progress polling.
                commit(ROOT / (child['arm'] + '-' + child['phase'] + '-EXIT.json'), self.child_receipt(child))
                expected = 0 if child['phase'] in ('primary', 'resume') else 75
                require(child['p'].returncode == expected, 'Unexpected child exit; no retry')
                if child['phase'] == 'resume':
                    self.disk_completion(ROOT, child['arm'])
            if time.monotonic() >= next_guard:
                self.guard()
                next_guard = time.monotonic() + GUARD_SECONDS
        self.guard()

    def paused(self, arm):
        folder = ROOT / (arm + '-seed42')
        require(not list(folder.glob('BLOCKED*')) and not (folder / 'COMPLETE.json').exists(), 'Paused arm blocked/complete')
        require(read(folder / 'OWN_SCOPE.json') == self.scopes[arm], 'Own paused scope changed')
        entries = [json.loads(line) for line in (folder / 'ledger.jsonl').read_text().splitlines()]
        starts = [e for e in entries if e['event'] == 'macro_started']
        updates = [e for e in entries if e['event'] == 'macro_updated']
        require(len(starts) == len(updates) == 1 and starts[0]['macro_index'] == 0 and
                updates[0]['cursor'] == ONE and entries[-1]['event'] == 'paused' and
                entries[-1]['cursor'] == ONE and entries[-1]['exit_code'] == 75 and
                not any(e['event'] in ('dev_started', 'dev_completed', 'failed', 'explicit_own_resume',
                                      'resumed_environment', 'arm_completed') for e in entries),
                'Formal first cursor/exclusive exposure/paused ledger differs')
        commits = [e for e in entries if e['event'] == 'checkpoint_committed']
        require(commits and commits[-1]['cursor'] == ONE, 'Latest checkpoint not cursor1')
        last = commits[-1]
        require(entries.index(last) == len(entries) - 2, 'Extra work/events after latest checkpoint')
        checkpoint = Path(last['complete']['path']).parent
        require(checkpoint.parent == folder, 'Own latest checkpoint redirected')
        verify(last['complete'])
        ckpt = committed(checkpoint / 'COMPLETE.json', self.scopes[arm])
        require(ckpt['macro_updates'] == 1 and ckpt['resumable_full_macro_boundary'] is True and
                read(checkpoint / 'CHECKPOINT.json')['cursor'] == ONE, 'Nonresumable first checkpoint')
        first = committed(folder / 'first-actual-macro/COMPLETE.json', self.scopes[arm])
        require(first['is_formal_exposure_macro0'] is True, 'Not formal first actual macro')
        return {'checkpoint': str(checkpoint), 'complete': last['complete'],
                'ledger': pin(folder / 'ledger.jsonl'), 'first': pin(folder / 'first-actual-macro/COMPLETE.json')}

    def primary_binding(self, arm, first):
        path = ROOT / 'primary' / arm / 'FIRST_PRIMARY_VERIFICATION.json'
        require(not list(path.parent.glob('BLOCKED*')), 'Primary blocked')
        require(path.resolve() == path and not path.is_symlink(), 'Primary report redirected')
        report_pin = pin(path)
        report = read(path)
        require(report['version'] == 'followup-first-primary-saved-only-v1' and
                report['complete'] is True and report['arm'] == arm and report['arm_name'] == arm and
                report['scope'] == self.scopes[arm] and report['cursor'] == ONE and
                report['allow_limited_scientific_resume'] is True and report['next_macro_index'] == 1 and
                report['main_external_authorization_required'] is True and report['automatic_resume'] is False and
                report['paused_exit_code'] == 75 and report['durable_ledger'] == first['ledger'] and
                report['formal_first_complete'] == first['first'] and
                report['repeat_macro0_allowed'] is False and report['D_confirm_opened'] is False and
                report['public_run'] is False, 'Primary scoped resume admission absent')
        for key in ('allow_production', 'allow_serve', 'allow_prefix_cache', 'allow_confirm', 'allow_public',
                    'D_select_opened', 'winner_selected', 'AUTHORIZE_RESUME_created', 'training_launched', 'CUDA_initialized'):
            require(report[key] is False, 'Primary claimed forbidden scope/action')
        require(report['macro0_consumed_exactly_once'] is True and report['no_later_model_or_dev_event'] is True and
                report['admission_scope'] == 'scientific-full-row-train-dev-only' and report['model_forward_calls'] == 0,
                'Primary formal-first/limited-saved-only scope differs')
        verify(report)
        resume = report['resume']
        require(resume['checkpoint'] == first['checkpoint'] and resume['complete'] == first['complete'],
                'Primary not bound to actual latest first checkpoint')
        pinfile = Path(resume['pin_file'])
        require(pinfile.parent == path.parent and pinfile.resolve() == pinfile, 'Primary resume pin outside own primary')
        readonly(pinfile)
        require(report['resume_pin_file'] == pin(pinfile) and
                read(pinfile) == first['complete'] and self.paused(arm) == first,
                'Resume pin/ledger changed since formal first')
        require(pin(path) == report_pin, 'Primary changed during readback')
        return {'report': report_pin, 'scope': self.scopes[arm], 'checkpoint': first['checkpoint'],
                'complete': first['complete'], 'pin_file': pin(pinfile), 'ledger': first['ledger'],
                'next_macro_index': 1, 'repeat_macro0_allowed': False}

    def approval(self, bindings, ready_pin):
        path = ROOT / 'AUTHORIZE_RESUME.json'
        next_guard = time.monotonic() + GUARD_SECONDS
        available = path.exists()  # One initial lookup; later only authorization events.
        while not available:
            self.stop_check()
            paths, _ = self.events.wait(timeout=max(0, next_guard - time.monotonic()))
            self.stop_check()
            self.resource_events(paths)
            if time.monotonic() >= next_guard:
                self.guard()
                next_guard = time.monotonic() + GUARD_SECONDS
            if path in paths:
                available = path.exists()
        readonly(path)
        auth = read(path)
        require(pin(path)['mtime_ns'] >= ready_pin['mtime_ns'], 'Authorization predates actual PRIMARY_READY')
        expected = {'version': 'external-main-resume-v1', 'root': str(ROOT), 'authorize_resume': True,
                    'primary_ready': ready_pin, 'arms': bindings, 'D_confirm_opened': False, 'public_run': False}
        # Canonical JSON equality also rejects Python's misleading True == 1.
        require(json.dumps(auth, sort_keys=True, allow_nan=False) ==
                json.dumps(expected, sort_keys=True, allow_nan=False),
                'External main authorization does not bind exact primary/scope/latest checkpoint/index1')
        verify(auth)
        self.pins_check()
        for arm in ARMS:
            require(self.primary_binding(arm, self.first[arm]) == bindings[arm], 'Primary/ledger changed before resume')
        self.guard()
        commit(ROOT / 'RESUME_AUTHORIZATION_ACCEPTED.json', {'authorization': pin(path),
               'primary_ready': ready_pin, 'arms': bindings, 'owner': self.owner})
        return pin(path)

    def cleanup_owned(self):
        result, pending = [], {}
        for child in self.children:
            if child['p'] is None:
                continue  # Constructor did not return a created Popen.
            if child['p'].returncode is not None:
                continue
            if child['id'] is None or child['fd'] is None:
                result.append({'pid': child['p'].pid, 'signalled': False,
                               'reason': 'no exact pidfd registration; own Popen wait required'})
                continue
            current = identity(child['id']['pid'])
            if current == child['id']:
                try:
                    signal.pidfd_send_signal(child['fd'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    result.append({'process': child['id'], 'TERM_error': str(error), 'leases_retained_until_exit': True})
                pending[child['fd']] = child
            elif current is not None:
                result.append({'process': child['id'], 'signalled': False,
                               'reason': 'identity mismatch; own Popen wait required'})
            else:
                child['p'].wait()
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
                            result.append({'process': child['id'], 'KILL_error': str(error), 'leases_retained_until_exit': True})
                # Never release leases over a still-owned child, even an uninterruptible
                # kernel wait. After bounded grace+exact KILL, wait actual pidfd exits.
                try:
                    commit(ROOT / 'OWNED_SHUTDOWN_PENDING.json', {'owner': self.owner,
                           'children': [self.child_receipt(c) for c in pending.values()],
                           'leases_held': bool(self.leases), 'bounded_grace_exhausted': True})
                except OSError as error:
                    result.append({'shutdown_status_write_error': str(error), 'leases_retained_until_exit': True})
                while pending:
                    exited, _, _ = select.select(list(pending), [], [])
                    for fd in exited:
                        child = pending.pop(fd)
                        child['p'].wait()
                        result.append({**self.child_receipt(child), 'bounded_grace_exhausted': True})
                break
            for fd in ready:
                child = pending.pop(fd)
                child['p'].wait()
                result.append({**self.child_receipt(child), 'safe_pause_requested': True})
        # Missing registration/mismatched identity forbids signalling, NOT waiting.
        # Wait these known Popen children too, even if natural completion takes long.
        self.await_owned_exits()
        return result

    def await_owned_exits(self):
        """Unconditional lease-release barrier, independent of cleanup success.

        Waiting on our own Popen/waitpid does not signal a guessed/reused PID.
        An unregistered live child may finish naturally; leases stay held.
        """
        for child in self.children:
            p = child['p']
            if p is None or child['waited']:
                continue
            try:
                p.wait()
            except BaseException as error:
                child['wait_error'] = type(error).__name__ + ': ' + str(error)
                # Cleanup/introspection/Popen API errors must not bypass the wait.
                # waitpid is restricted by the kernel to our actual child relation.
                while True:
                    try:
                        pid, status = os.waitpid(p.pid, 0)
                        if pid != p.pid:
                            raise RuntimeError('Own blocking waitpid returned another PID')
                        p.returncode = os.waitstatus_to_exitcode(status)
                        break
                    except InterruptedError:
                        continue  # Retry only an interrupted blocking wait, not progress.
                    except ChildProcessError:
                        # Already reaped; do not invent an unavailable exit code.
                        child['wait_note'] = 'own child already reaped; exit code may be unavailable'
                        break
                    except BaseException as wait_error:
                        child['wait_error'] += '; waitpid: ' + type(wait_error).__name__ + ': ' + str(wait_error)
                        # Exit cannot be confirmed. Fail closed instead of raising
                        # out of the lease barrier. No timed retry or unknown signal.
                        blocked = {signal.SIGTERM, signal.SIGINT}
                        signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
                        while True:
                            signal.sigwait(blocked)
            child['waited'] = True

    def run(self):
        require(Path(__file__).absolute() == ROOT / 'ops/side_queue.py' and
                Path(__file__).resolve() == ROOT / 'ops/side_queue.py', 'Run the deployed pinned queue, not another copy')
        require(ROOT.is_dir() and ROOT.resolve() == ROOT and not ROOT.is_symlink(), 'Existing fixed ROOT required')
        # Exclusive run gate deliberately survives crashes. No automatic restart path.
        lock = (ROOT / 'QUEUE.lock').open('x+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            self.stop_check()
            require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'SYSTEM pidfd APIs required')
            require(not (ROOT / 'AUTHORIZE_RESUME.json').exists() and
                    not (ROOT / 'AUTHORIZE_RESUME.json').is_symlink(), 'Premature external resume approval')
            for arm in ARMS:
                for path in (ROOT / (arm + '-seed42'), ROOT / 'primary' / arm):
                    require(not path.exists() and not path.is_symlink(), 'Fresh one-shot queue scope required')
            self.bundle_bytes = (ROOT / 'OPS_BUNDLE.json').read_bytes()
            self.bundle = read(ROOT / 'OPS_BUNDLE.json')
            self.bundle_pin = pin(ROOT / 'OPS_BUNDLE.json')
            require(self.bundle['complete'] is True and self.bundle['root'] == str(ROOT) and
                    self.bundle['version'] == 'followup-ops-bundle-v1' and
                    self.bundle['training_launched'] is False and self.bundle['queue_launched'] is False,
                    'Final ops installation bundle required')
            expected = {str(ROOT / 'source' / name) for name in SOURCE_NAMES}
            expected |= {str(ROOT / 'ops' / name) for name in ('freeze_fresh.py', 'side_queue.py', 'verify_first.py')}
            expected.add(str(ROOT / 'seed42.json'))
            require(len(self.bundle['files']) == len(expected) and
                    {x['stat']['path'] for x in self.bundle['files']} == expected and
                    self.bundle['original']['post_four_gpus'] == [0, 1], 'Exact ops/source/config inventory required')
            self.pins_check()
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            # Pinned bindings/state-boundaries are CPU JSON code; no trainer/Torch imports.
            sys.path.insert(0, str(ROOT / 'source'))
            from bindings import Context
            for arm in ARMS:
                self.scopes[arm] = Context(ROOT / 'seed42.json', arm).snapshot
            require('torch' not in sys.modules, 'Queue must not import Torch')
            self.events = Events()
            commit(ROOT / 'QUEUE_OWNER.json', {'owner': self.owner, 'bundle': self.bundle_pin,
                   'root': str(ROOT), 'arms': ARMS, 'scopes': self.scopes, 'automatic_restart': False})
            self.wait_original()
            self.devices, occupants = nvidia()
            require(not any(occupants.values()), 'GPU2/3 occupied; no preemption/retry')
            for arm in ('BI_P_CE', 'PB_P_CE'):
                self.disk_completion(OLD, arm)
            budget = disk_budget(self.disk_done)  # BOTH fit before either cooperative lease.
            self.lease()
            self.guard()
            commit(ROOT / 'RESOURCES_READY.json', {'devices': self.devices, 'budget': budget,
                   'lease_directory': str(LEASES), 'leases_held': True, 'advisory_only': True})
            first_children = [self.spawn(arm, 'first') for arm in ARMS]
            self.wait_children(first_children)
            self.first = {arm: self.paused(arm) for arm in ARMS}
            commit(ROOT / 'FORMAL_FIRST_WAVE.json', {'execution_complete': True,
                   'all_paused_at_cursor1': True, 'automatic_continuation': False,
                   'arms': {c['arm']: {**self.child_receipt(c), 'state': 'paused_at_cursor1',
                            'scope': self.scopes[c['arm']], **self.first[c['arm']]} for c in first_children}})
            for arm in ARMS:
                self.wait_children([self.spawn(arm, 'primary', verifier=True)])
            bindings = {arm: self.primary_binding(arm, self.first[arm]) for arm in ARMS}
            commit(ROOT / 'PRIMARY_READY.json', {'complete': True, 'root': str(ROOT), 'arms': bindings,
                   'main_direct_actual_review_required': True, 'automatic_continuation': False,
                   'D_confirm_opened': False, 'public_run': False})
            authorization = self.approval(bindings, pin(ROOT / 'PRIMARY_READY.json'))
            resumed = []
            for arm in ARMS:
                verify(authorization)
                require(self.primary_binding(arm, self.first[arm]) == bindings[arm], 'Pre-spawn primary changed')
                resumed.append(self.spawn(arm, 'resume', resume={
                    'checkpoint': bindings[arm]['checkpoint'], 'pin_file': bindings[arm]['pin_file']['path']}))
            self.wait_children(resumed)
            finals = {}
            for arm in ARMS:
                folder = ROOT / (arm + '-seed42')
                require(not list(folder.glob('BLOCKED*')), 'Final arm blocked')
                done = committed(folder / 'COMPLETE.json', self.scopes[arm])
                require(done['cursor'] == CURSOR and done['observations'] == [10000000, 30000000, 100000000] and
                        done['D_confirm_opened'] is False and done['public_run'] is False, 'Final exposure/scope differs')
                entries = [json.loads(line) for line in (folder / 'ledger.jsonl').read_text().splitlines()]
                starts = [e['macro_index'] for e in entries if e['event'] == 'macro_started']
                updates = [e['cursor']['macro_updates'] for e in entries if e['event'] == 'macro_updated']
                resumes = [e for e in entries if e['event'] == 'explicit_own_resume']
                require(starts == list(range(3557)) and updates == list(range(1, 3558)) and
                        len(resumes) == 1 and resumes[0]['complete'] == self.first[arm]['complete'] and
                        entries[-1]['event'] == 'arm_completed' and entries[-1]['cursor'] == CURSOR and
                        not any(e['event'] == 'failed' for e in entries), 'Final contiguous exposure/resume ledger differs')
                for item in done['checkpoints'] + done['D_select'] + [done['first_actual_macro']]:
                    require(Path(item['path']).parent.parent == folder, 'Final artifact redirected')
                    committed(Path(item['path']), self.scopes[arm])
                require(done['first_actual_macro'] == self.first[arm]['first'] and
                        done['final_checkpoint'] in done['checkpoints'] and
                        read(Path(done['final_checkpoint']['path']).parent / 'CHECKPOINT.json')['cursor'] == CURSOR,
                        'Final first/latest checkpoint differs')
                actual_nodes(done, folder, self.scopes[arm])
                finals[arm] = {'complete': pin(folder / 'COMPLETE.json'), 'cursor': done['cursor'],
                               'nodes': done['D_select'], 'first': done['first_actual_macro'],
                               'final_checkpoint': done['final_checkpoint'], 'scope': self.scopes[arm]}
            self.pins_check()
            commit(ROOT / 'QUEUE_COMPLETE.json', {'execution_complete': True, 'all_completed': True,
                   'owner': self.owner, 'bundle': self.bundle_pin, 'authorization': authorization,
                   'workers': [self.child_receipt(c) for c in resumed], 'arms': finals,
                   'published': False, 'winner_selected': False, 'production': False, 'serve': False,
                   'D_confirm_opened': False, 'public_run': False, 'scope': 'limited-full-row-scientific-only'})
        except BaseException as error:
            # Ignore repeat interrupts only while preserving/stopping registered owned children.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                cleanup = self.cleanup_owned()
            except BaseException as cleanup_error:
                cleanup = [{'cleanup_error_type': type(cleanup_error).__name__, 'cleanup_error': str(cleanup_error),
                            'own_exit_barrier_required': True}]
            self.await_owned_exits()
            commit(ROOT / 'QUEUE_BLOCKED.json', {'complete': False, 'owner': self.owner,
                   'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                   'cleanup': cleanup, 'partial_preserved': True, 'automatic_resume': False,
                   'workers': [self.child_receipt(c) for c in self.children], 'public_run': False})
            raise
        finally:
            # This runs even if cleanup or a failure-marker write raises. No child
            # pidfd, resource lease, or queue lock closes before the actual waits.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.await_owned_exits()
            for child in self.children:
                if child['fd'] is not None:
                    os.close(child['fd'])
            if self.controller_fd is not None:
                os.close(self.controller_fd)
            if self.events is not None:
                self.events.close()
            for fd in self.leases:
                os.close(fd)
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, choices=[str(ROOT)])
    parser.parse_args()
    os.umask(0o077)
    def interrupted(signum, frame):
        raise InterruptedError('Queue interrupted; preserve and pause owned work')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Queue().run()


if __name__ == '__main__':
    main()
