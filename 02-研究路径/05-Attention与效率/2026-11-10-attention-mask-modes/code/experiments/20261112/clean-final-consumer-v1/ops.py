"""Install-only local deployment and one-shot remote SYSTEM manager.

No Torch, model, SDK, producer runtime, or private dataset import at top level.
Installation is not execution authorization without MAIN's explicit review flag.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
from common import (ROOT, ROLES, MODEL_PYTHON, SIGNALS, Guard, canonical, commit,
                    entry, exact, identity, need, pidfd, pin, read, require_dead,
                    same, verify)

LOCAL = Path('/tmp/opencode/jev-clean-final-eval-20261112-local')
SYSTEM_PYTHON = '/usr/bin/python3'
SOURCES = ('common.py', 'backend.py', 'evaluate.py', 'ops.py')
LEASE_ROOT = Path('/tmp/opencode/jev-owned-h800-leases-20261112')
FIXED_RESOURCES = {'max_output_bytes': 20 * 2**30, 'free_reserve_bytes': 20 * 2**30,
                   'fixed_future_bytes': 20 * 2**30, 'guard_seconds': 15,
                   'grace_seconds': 900}
_STOP_SIGNAL = None


def record_stop(signum, frame):
    """Manager-wide handler: record only; never raise across an actual-wait barrier."""
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum


def phase_plan():
    """The exact fifteen legal children of the frozen one-shot plan, in launch order."""
    plan = [('codec', None, None)]
    plan.extend(('confirm', role, None) for role in ROLES)
    plan.extend((('confirm-analysis', None, None), ('public-prepare', None, None)))
    for role in ROLES:
        plan.extend(('public-worker', role, rank) for rank in range(4))
        plan.append(('public-score', role, None))
    need(len(plan) == 15, 'Frozen plan must contain exactly fifteen children')
    return tuple(plan)


def directory(path):
    path = Path(path)
    s = path.lstat()
    need(stat.S_ISDIR(s.st_mode) and not path.is_symlink() and path.resolve() == path,
         'Fixed nonsymlink directory required: ' + str(path))


def absent(path):
    path = Path(path)
    need(not path.exists() and not path.is_symlink(), 'Existing partial preserved: ' + str(path))


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def local_install(host, log_path, deployment_ready):
    """One SSH connection; source-root allowlist is read through retained nofollow FDs."""
    need(deployment_ready is True, 'MAIN must finish broad source review and pass --deployment-ready')
    need(host == 'sh-10-gpu' and Path(__file__).absolute() == LOCAL / 'source' / 'ops.py' and
         Path(__file__).resolve() == LOCAL / 'source' / 'ops.py', 'Install only from fixed LOCAL to sh-10-gpu')
    need('torch' not in sys.modules, 'Install must remain Torch-free')
    for path in (LOCAL, LOCAL / 'source'):
        directory(path)
    log_path = Path(log_path).absolute()
    directory(log_path.parent)
    absent(log_path)
    need(log_path not in {LOCAL / 'source' / n for n in SOURCES} and
         not log_path.is_relative_to(ROOT), 'Log must not overwrite source/remote scope')
    fds, source_dir = [], None
    try:
        source_dir = os.open(LOCAL / 'source', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        records = {}
        for name in SOURCES:
            path = LOCAL / 'source' / name
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=source_dir)
            fds.append(fd)
            before = os.fstat(fd)
            ls = path.lstat()
            need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1 and
                 (before.st_dev, before.st_ino) == (ls.st_dev, ls.st_ino) and
                 path.resolve() == path and 0 < before.st_size <= 2**20,
                 'Unsafe/nonpublic source upload: ' + name)
            data = os.pread(fd, before.st_size + 1, 0)
            need(len(data) == before.st_size, 'Public source changed during held-FD read')
            after = os.fstat(fd)
            need((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
                 (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns), 'Source changed during snapshot')
            records[name] = {'base64': base64.b64encode(data).decode('ascii'),
                             'local': pin(path), 'inode': [before.st_dev, before.st_ino]}
        need(set(records) == set(SOURCES), 'Only the four public modules may be uploaded')
        payload = {'version': 'clean-final-four-source-upload-v1', 'root': str(ROOT),
                   'main_reviewed': True, 'sources': records}
        # A tiny inline bootstrap is transport code, not an extra deployed module.
        bootstrap = r'''
import base64,json,os,stat,sys
from pathlib import Path
sys.dont_write_bytecode=True
os.umask(0o077)
root=Path('/tmp/opencode/jev-clean-final-eval-20261112-v1')
names=('common.py','backend.py','evaluate.py','ops.py')
def need(ok,message):
    if not ok: raise RuntimeError(message)
def sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)
payload=json.load(sys.stdin)
need(set(payload)=={'version','root','main_reviewed','sources'} and
     payload['version']=='clean-final-four-source-upload-v1' and payload['root']==str(root) and
     payload['main_reviewed'] is True and set(payload['sources'])==set(names),'Bad reviewed upload')
need(root.is_dir() and root.resolve()==root and not root.is_symlink() and
     root.stat().st_uid==os.getuid(),'Existing frozen ROOT required')
need((root/'FROZEN_CANDIDATES.json').is_file(),'Existing candidates must precede source install')
for name in ('config.json','SOURCE_SNAPSHOT.json','INSTALL_COMPLETE.json','RUN.lock','BLOCKED.json'):
    p=root/name
    need(not p.exists() and not p.is_symlink(),'Prior partial/control preserved: '+name)
source=root/'source'
source.mkdir(mode=0o700)  # Exclusive install claim; ROOT is never created/overwritten.
sync(root)
for name in names:
    data=base64.b64decode(payload['sources'][name]['base64'],validate=True)
    need(0<len(data)<=2**20,'Source size out of allowlist')
    fd=os.open(source/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as handle:
        handle.write(data);handle.flush();os.fchmod(handle.fileno(),0o444);os.fsync(handle.fileno())
    need((source/name).read_bytes()==data and stat.S_IMODE((source/name).stat().st_mode)==0o444,
         'Actual source byte readback failed')
sync(source)
sys.path.insert(0,str(source))
from common import install_config,commit,pin,Guard
cfg=install_config(True)
need('torch' not in sys.modules,'Installation imported Torch')
for name in names:
    need((source/name).read_bytes()==base64.b64decode(payload['sources'][name]['base64'],validate=True),
         'Source changed during config freeze')
Guard(root/'config.json').check()
receipt={'complete':True,'config':pin(root/'config.json'),'source_binding':cfg['source_binding'],
         'source_files':cfg['source_files'],'exact_byte_readback':True,'main_reviewed':True,
         'no_workers_launched':True,'model_forward_calls':0,'confirm_opened':False,
         'public_payload_opened':False,'frozen_candidates_unchanged':True}
commit(root/'INSTALL_COMPLETE.json',receipt)
print(json.dumps(receipt,ensure_ascii=False,sort_keys=True),flush=True)
'''
        args = ['ssh', '-T', '-a', '-x', '-S', 'none',
                '-oBatchMode=yes', '-oStrictHostKeyChecking=yes', '-oForwardAgent=no',
                '-oForwardX11=no', '-oForwardX11Trusted=no', '-oClearAllForwardings=yes',
                '-oPermitLocalCommand=no', '-oControlMaster=no', '-oControlPath=none',
                '-oControlPersist=no', '-oRequestTTY=no', '-oConnectionAttempts=1',
                '-oConnectTimeout=30', host,
                shlex.join([SYSTEM_PYTHON, '-B', '-s', '-c', bootstrap])]
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(log_fd, 'wb') as log:
            header = {'action': 'install-only', 'root': str(ROOT), 'host': host, 'main_reviewed': True,
                      'public_source_files': {n: records[n]['local'] for n in SOURCES}, 'workers_launched': False}
            log.write((canonical(header) + '\n').encode())
            log.flush()
            os.fsync(log.fileno())
            sync_dir(log_path.parent)  # Durable new log BEFORE any SSH side effect.
            for fd, name in zip(fds, SOURCES):
                s = os.fstat(fd)
                need([s.st_dev, s.st_ino] == records[name]['inode'] and
                     os.pread(fd, s.st_size + 1, 0) == base64.b64decode(records[name]['base64'], validate=True),
                     'Held public source changed before transfer')
                same(pin(LOCAL / 'source' / name), records[name]['local'], 'Local source changed before transfer')
            result = subprocess.run(args, input=(canonical(payload) + '\n').encode(), stdout=log,
                                    stderr=subprocess.STDOUT, check=False, close_fds=True)
            log.flush()
            os.fsync(log.fileno())
            need(result.returncode == 0, 'Install failed; retain remote partial source claim and local log, no retry')
            for fd, name in zip(fds, SOURCES):
                s = os.fstat(fd)
                need([s.st_dev, s.st_ino] == records[name]['inode'] and
                     os.pread(fd, s.st_size + 1, 0) == base64.b64decode(records[name]['base64'], validate=True),
                     'Held public source changed during transfer')
                same(pin(LOCAL / 'source' / name), records[name]['local'], 'Local source changed during transfer')
    finally:
        for fd in fds:
            os.close(fd)
        if source_dir is not None:
            os.close(source_dir)


def nvidia():
    """Physical UUID and XML process inventory, including graphics/MPS, for 0..3."""
    result = subprocess.run(['nvidia-smi', '-i', '0,1,2,3', '--query-gpu=index,uuid',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True,
                            text=True, timeout=20)
    devices = {}
    for line in result.stdout.splitlines():
        if line.strip():
            index, uuid = (s.strip() for s in line.split(','))
            need(index.isdecimal() and int(index) not in devices and uuid.startswith('GPU-'),
                 'Invalid physical GPU inventory')
            devices[int(index)] = uuid
    need(set(devices) == set(range(4)) and len(set(devices.values())) == 4, 'Exact GPU0..3 inventory required')
    xml = subprocess.run(['nvidia-smi', '-i', '0,1,2,3', '-q', '-x'], check=True,
                         capture_output=True, text=True, timeout=20)
    gpus = ET.fromstring(xml.stdout).findall('gpu')
    occupants = {uuid: [] for uuid in devices.values()}
    need(len(gpus) == 4 and {g.findtext('uuid') for g in gpus} == set(occupants),
         'Physical process UUID inventory differs')
    for gpu in gpus:
        processes = gpu.find('processes')
        need(processes is not None, 'NVIDIA process inventory unavailable')
        for process in processes.findall('process_info'):
            pid = process.findtext('pid', '').strip()
            need(pid.isdecimal(), 'Unknown GPU process; never take over')
            occupants[gpu.findtext('uuid')].append(int(pid))
    return devices, occupants


def known_temporary(path):
    """Only SDK's exact atomic-json temporaries and our commit staging may vanish."""
    from evaluate import key
    relative = Path(path).relative_to(ROOT)
    parts = relative.parts
    plan = phase_plan()
    child_keys = {key(phase, role, rank) for phase, role, rank in plan}
    if relative.name.endswith('.json.writing'):
        final_name = relative.name[:-len('.writing')]
        if len(parts) == 1:
            fixed = {'OWNER.json', 'LAUNCH.json', 'COMPLETE.json', 'BLOCKED.json',
                     'OWNED_SHUTDOWN_PENDING.json', 'CODEC_COMPLETE.json', 'CODEC_BLOCKED.json',
                     'CONFIRM_ANALYSIS.json', 'PUBLIC_READY.json'}
            phase_names = {name + '-' + suffix + '.json' for name in child_keys for suffix in ('OWNER', 'EXIT')}
            # Codec's actual failure receipt is CODEC_BLOCKED.json, not codec-BLOCKED.json.
            phase_names.update(key(phase, role, rank) + '-BLOCKED.json'
                               for phase, role, rank in plan if phase != 'codec')
            journals = {f'{i:06d}-PHASE-COMPLETE.json' for i in range(1, len(plan) + 1)}
            return (final_name in fixed or final_name in phase_names or
                    final_name in {'gpu' + str(i) + '-SUCCESSOR.json' for i in range(4)} or
                    final_name in journals)
        if len(parts) == 2 and parts[0] == 'grants':
            return final_name in {name + '-GRANT.json' for name in child_keys}
        if len(parts) == 3 and parts[0] == 'confirm' and parts[1] in ROLES:
            return final_name in ('SUMMARY.json', 'WEIGHTS.json', 'COMPLETE.json')
        if len(parts) == 5 and parts[0] == 'public' and parts[1] in ROLES and parts[2] == 'workers' and parts[3] in ('0', '1', '2', '3'):
            return final_name in ('WEIGHTS.json', 'COMPLETE.json')
        return len(parts) == 3 and parts[0] == 'public' and parts[1] in ROLES and final_name == 'COMPLETE.json'
    if (len(parts) == 5 and parts[0] == 'public' and parts[1] in ROLES and
            parts[2] == 'workers' and parts[3] in ('0', '1', '2', '3')):
        return parts[4] in ('status.json.tmp', 'environment.json.tmp')
    if len(parts) == 3 and parts[0] == 'public' and parts[1] in ROLES:
        return parts[2] in ('benchmark-summary.json.tmp', 'index.json.tmp', 'scores.json.tmp')
    return False


def used_unique():
    """Observed unique inode bytes, separate from the fixed future-free reserve."""
    directory(ROOT)
    sizes = {}
    def scan_error(error):
        raise error  # Subtree EACCES/EIO/EMFILE/ENOENT is never a temporary-file exemption.
    for parent, dirs, files in os.walk(ROOT, followlinks=False, onerror=scan_error):
        for name in dirs:
            path = Path(parent) / name
            s = path.lstat()
            need(stat.S_ISDIR(s.st_mode) and not path.is_symlink(), 'Unsafe output directory')
        for name in files:
            path = Path(parent) / name
            try:
                s = path.lstat()
            except FileNotFoundError:
                need(known_temporary(path), 'Unexpected output disappearance: ' + str(path))
                continue
            need(stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode), 'Nonregular output file')
            inode = (s.st_dev, s.st_ino)
            sizes[inode] = max(sizes.get(inode, 0), s.st_size)
    return sum(sizes.values())


def budget(cfg):
    same(cfg['resources'], FIXED_RESOURCES, 'Frozen resource contract changed')
    actual = used_unique()
    need(actual <= cfg['resources']['max_output_bytes'], 'New ROOT exceeds separate 20GiB unique-inode cap')
    free = shutil.disk_usage(ROOT).free
    required = cfg['resources']['fixed_future_bytes'] + cfg['resources']['free_reserve_bytes']
    need(required == 40 * 2**30 and free >= required,
         'Free space below fixed future20GiB + shared20GiB; own-used never gives credit')
    return {'root_unique_inode_bytes': actual, 'root_cap_bytes': 20 * 2**30,
            'free_bytes': free, 'required_free_bytes': required,
            'fixed_future_bytes': 20 * 2**30, 'shared_floor_bytes': 20 * 2**30,
            'own_used_credit_bytes': 0, 'system_wide_reservation': False}


class Events:
    """Only root STOP/control events and exact child pidfds, never progress polling."""
    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1')
        self.wd = self.libc.inotify_add_watch(self.fd, os.fsencode(ROOT),
                                            0x8 | 0x80 | 0x100 | 0x400 | 0x800)
        if self.wd < 0:
            os.close(self.fd)
            raise OSError(ctypes.get_errno(), 'inotify_add_watch')

    def wait(self, pidfds, timeout):
        ready, _, _ = select.select([self.fd, *pidfds], [], [], timeout)
        paths = set()
        if self.fd in ready:
            while True:
                try:
                    data = os.read(self.fd, 65536)
                except BlockingIOError:
                    break
                need(data, 'inotify closed')
                offset = 0
                while offset < len(data):
                    wd, mask, cookie, size = struct.unpack_from('iIII', data, offset)
                    name = os.fsdecode(data[offset + 16:offset + 16 + size].split(b'\0', 1)[0])
                    offset += 16 + size
                    need(not mask & (0x4000 | 0x400 | 0x800 | 0x8000) and wd == self.wd,
                         'inotify overflow/watch loss/root moved; fail closed')
                    paths.add(ROOT / name)
        return paths, set(ready) - {self.fd}

    def close(self):
        os.close(self.fd)


class Manager:
    def __init__(self):
        need('torch' not in sys.modules, 'SYSTEM manager must not import Torch')
        self.guard_inputs = Guard(ROOT / 'config.json')
        self.cfg = self.guard_inputs.cfg
        self.owner = identity(os.getpid())
        need(self.owner is not None, 'Cannot identify actual manager')
        self.children, self.leases = [], []
        self.lease_records, self.receipts = {}, []
        self.events, self.run_fd = None, None
        self.devices = {i: self.cfg['predecessors'][str(i)]['uuid'] for i in range(4)}
        self.journal_index = 0
        from evaluate import installed
        installed(self.guard_inputs)
        same(self.cfg['resources'], FIXED_RESOURCES, 'Fixed resource guard required')
        need(set(self.cfg['predecessors']) == {'0', '1', '2', '3'}, 'Exactly four predecessor contracts required')

    def stop_check(self):
        need(_STOP_SIGNAL is None, 'Manager TERM/INT stop request recorded; stop only owned work')
        for name in ('STOP.json', 'STOP_QUEUE.json', 'STOP_FINAL.json'):
            absent(ROOT / name)

    def journal(self, label, value):
        need(label == 'PHASE-COMPLETE' and self.journal_index < len(phase_plan()),
             'Only the fifteen actual phase-completion journal positions are legal')
        self.journal_index += 1
        commit(ROOT / (f'{self.journal_index:06d}-' + label + '.json'), {'owner': self.owner, **value})

    def predecessors(self):
        for gpu in range(4):
            record = self.cfg['predecessors'][str(gpu)]
            need(record['owners_to_require_dead'] and record['original_owner'] in record['owners_to_require_dead'],
                 'Missing original predecessor owner death contract')
            for owner in record['owners_to_require_dead']:
                require_dead(owner)
            verify(record['completion_pins'])

    def lease_check(self, gpu, fd):
        record = self.cfg['predecessors'][str(gpu)]
        path = Path(record['lease_path'])
        need(path == LEASE_ROOT / ('gpu' + str(gpu) + '.lock'), 'Lease root/index redirected')
        data = base64.b64decode(record['lease_bytes_base64'], validate=True)
        fs, ls = os.fstat(fd), path.lstat()
        need(data and [fs.st_dev, fs.st_ino] == [ls.st_dev, ls.st_ino] == record['lease_inode'] and
             stat.S_ISREG(fs.st_mode) and stat.S_ISREG(ls.st_mode) and
             fs.st_uid == ls.st_uid == os.getuid() and fs.st_nlink == ls.st_nlink == 1 and
             stat.S_IMODE(fs.st_mode) == stat.S_IMODE(ls.st_mode) == 0o600 and
             os.pread(fd, len(data) + 1, 0) == data, 'Changed/unknown held predecessor lease')
        metadata = json.loads(data)
        need(metadata['gpu'] == gpu and metadata['uuid'] == record['uuid'] and
             metadata['owner'] == record['original_owner'], 'Accepted old lease owner/device differs')
        return data

    def acquire(self):
        directory(LEASE_ROOT)
        s = LEASE_ROOT.stat()
        need(s.st_uid == os.getuid() and stat.S_IMODE(s.st_mode) == 0o700, 'Unsafe cooperative lease directory')
        self.predecessors()
        devices, occupants = nvidia()
        same(devices, self.devices, 'GPU UUID changed before locks')
        need(not any(occupants.values()), 'GPU0..3 must all be empty; no waiting/retry/takeover')
        for gpu in range(4):
            self.predecessors()
            path = Path(self.cfg['predecessors'][str(gpu)]['lease_path'])
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
            self.leases.append(fd)  # Track actual FD before any check/flock/journal can fail.
            self.lease_check(gpu, fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            data = self.lease_check(gpu, fd)
            self.lease_records[fd] = gpu
            self.predecessors()
            devices, occupants = nvidia()
            same(devices, self.devices, 'GPU UUID changed after lock')
            need(not any(occupants.values()), 'GPU occupancy changed during successor admission')
            commit(ROOT / ('gpu' + str(gpu) + '-SUCCESSOR.json'),
                   {'owner': self.owner, 'gpu': gpu, 'uuid': self.devices[gpu], 'root': str(ROOT),
                    'predecessor': self.cfg['predecessors'][str(gpu)],
                    'config': pin(ROOT / 'config.json'), 'source_files': self.cfg['source_files'],
                    'same_inode_fd_held': True, 'old_metadata_written': False,
                    'advisory_only': True, 'lease_bytes_length': len(data)})
        self.guard()

    def guard(self):
        self.stop_check()
        self.guard_inputs.check()
        self.predecessors()
        if self.receipts:
            verify(self.receipts)
        need('torch' not in sys.modules, 'Manager unexpectedly imported Torch')
        for fd, gpu in self.lease_records.items():
            self.lease_check(gpu, fd)
        devices, occupants = nvidia()
        same(devices, self.devices, 'Physical UUID inventory changed')
        for gpu in range(4):
            allowed = {c['id']['pid']: c['id'] for c in self.children if c['gpu'] == gpu and
                       c['p'] is not None and c['id'] is not None and not c['waited']}
            for pid in occupants[self.devices[gpu]]:
                need(pid in allowed and identity(pid) == allowed[pid],
                     'Foreign process on owned GPU; stop only exact own children')
        return budget(self.cfg)

    def child_receipt(self, child):
        p = child['p']
        return {'owner': self.owner, 'phase': child['phase'], 'role': child['role'], 'rank': child['rank'],
                'process': child['id'], 'gpu': child['gpu'], 'uuid': self.devices.get(child['gpu']),
                'output': child['output'], 'grant': child['grant'], 'created': p is not None,
                'exit_code': p.returncode if p is not None else None, 'actual_wait_complete': child['waited'],
                'spawn_monotonic': child.get('spawn_monotonic'), 'wait_monotonic': child.get('wait_monotonic'),
                'wait_note': child.get('wait_note'), 'wait_error': child.get('wait_error')}

    def spawn(self, phase, role=None, rank=None):
        from evaluate import key, output_for
        self.guard()
        need((rank is None or type(rank) is int) and (phase, role, rank) in phase_plan(),
             'Illegal phase/role/rank outside frozen fifteen-child plan')
        name = key(phase, role, rank)
        gpu = ROLES.index(role) if phase == 'confirm' else (rank if phase == 'public-worker' else None)
        if gpu is not None:
            need(not any(c['gpu'] == gpu and c['p'] is not None and not c['waited'] for c in self.children),
                 'GPU already assigned to own worker')
        output = output_for(phase, role, rank)
        grant_path = ROOT / 'grants' / (name + '-GRANT.json')
        commit(grant_path, {'version': 'clean-final-phase-grant-v1', 'owner': self.owner,
                           'root': str(ROOT), 'phase': phase, 'role': role, 'rank': rank,
                           'output': str(output), 'gpu': gpu, 'device_uuid': self.devices.get(gpu),
                           'config': pin(ROOT / 'config.json'), 'source_files': self.cfg['source_files']})
        # CPU analysis is deliberately independent SYSTEM stdlib. Tokenizer/SDK CPU phases use model env.
        python = SYSTEM_PYTHON if phase == 'confirm-analysis' else MODEL_PYTHON
        args = [python, '-B', '-s', '-u', str(ROOT / 'source' / 'evaluate.py'),
                '--phase', phase, '--config', str(ROOT / 'config.json')]
        if role is not None:
            args += ['--role', role]
        if rank is not None:
            args += ['--rank', str(rank)]
        env = dict(os.environ)
        for unsafe in ('PYTHONPATH', 'PYTHONHOME', 'JEV_FINAL_GRANT', 'CUDA_VISIBLE_DEVICES'):
            env.pop(unsafe, None)
        env.update(CUDA_VISIBLE_DEVICES=self.devices.get(gpu, ''),
                   JEV_FINAL_GRANT=str(grant_path), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', USE_TORCH='1' if gpu is not None else '0',
                   USE_TF='0', USE_FLAX='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                   FLASH_ATTENTION_DETERMINISTIC='1', WANDB_DISABLED='true')
        if gpu is not None:
            env['CUBLAS_WORKSPACE_CONFIG'] = self.guard_inputs.producer(role)['cfg']['precision']['cublas_workspace_config']
        child = {'p': None, 'id': None, 'fd': None, 'gpu': gpu, 'phase': phase, 'role': role,
                 'rank': rank, 'output': str(output), 'waited': False, 'grant': pin(grant_path)}
        parent = self.owner
        self.stop_check()
        prior = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        try:
            self.children.append(child)  # Reachable before actual Popen constructor.
            def parent_guard():
                # Do not inherit the manager's record-only handler in the pre-exec
                # child: a pending PDEATHSIG must terminate it, not be swallowed.
                for sig in SIGNALS:
                    signal.signal(sig, signal.SIG_DFL)
                libc = ctypes.CDLL(None, use_errno=True)
                if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or identity(os.getppid()) != parent:
                    os._exit(125)
                signal.pthread_sigmask(signal.SIG_SETMASK, prior)  # Child must not inherit critical mask.
            with (ROOT / (name + '.log')).open('x', encoding='utf-8') as log:
                # Assignment occurs before log-context exit, identity, pidfd, or commit can fail.
                child['spawn_monotonic'] = time.monotonic()
                child['p'] = subprocess.Popen(args, env=env, cwd=ROOT / 'source', stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                                             start_new_session=True, preexec_fn=parent_guard)
            child['id'] = identity(child['p'].pid)
            need(child['id'] is not None, 'Own child died before registration; preserve, never retry')
            child['fd'] = pidfd(child['id'])
            need(child['fd'] is not None, 'Own child exited before exact pidfd registration')
            commit(ROOT / (name + '-OWNER.json'), self.child_receipt(child))
        finally:
            # Pending TERM/INT records a stop request without throwing; the next
            # guard converts it to orderly failure after actual Popen registration.
            signal.pthread_sigmask(signal.SIG_SETMASK, prior)
        self.guard()  # UUID/process/disk/static identities rechecked after EACH spawn.
        return child

    def wait_children(self, children):
        from evaluate import key
        pending = {c['fd']: c for c in children}
        need(None not in pending and len(pending) == len(children), 'Missing/duplicate exact child pidfd')
        next_guard = time.monotonic() + self.cfg['resources']['guard_seconds']
        while pending:
            self.stop_check()
            paths, dead = self.events.wait(pending, max(0, next_guard - time.monotonic()))
            # A record-only signal can restart select; the existing resource
            # timer bounds its next phase-wait check to fifteen seconds, not an
            # asynchronous first-signal interruption or a progress-polling loop.
            self.stop_check()
            for fd in dead:
                child = pending.pop(fd)
                need(identity(child['id']['pid']) in (None, child['id']), 'Own PID changed before actual wait')
                child['p'].wait()
                child['waited'] = True
                child['wait_monotonic'] = time.monotonic()
                exit_path = ROOT / (key(child['phase'], child['role'], child['rank']) + '-EXIT.json')
                commit(exit_path, self.child_receipt(child))
                need(child['p'].returncode == 0, 'Own phase failed; preserve partial and no automatic retry')
                self.receipts.append(pin(exit_path))
                self.accept(child)
            if time.monotonic() >= next_guard:
                self.guard()  # Resource/STOP/static guard only, never progress or completion polls.
                next_guard = time.monotonic() + self.cfg['resources']['guard_seconds']
        self.guard()

    def accept(self, child):
        """Read each completion only after that exact Popen has actually exited zero."""
        from evaluate import analysis_receipt, codec_receipt, ready_receipt, weights_ok
        phase, role = child['phase'], child['role']
        if phase == 'codec':
            path = ROOT / 'CODEC_COMPLETE.json'
            codec_receipt(self.guard_inputs)
        elif phase == 'confirm-analysis':
            path = ROOT / 'CONFIRM_ANALYSIS.json'
            analysis_receipt(self.guard_inputs)
        elif phase == 'public-prepare':
            path = ROOT / 'PUBLIC_READY.json'
            ready_receipt(self.guard_inputs)
        else:
            folder = Path(child['output'])
            path = folder / 'COMPLETE.json'
            done = read(path)
            need(done['complete'] is True and done['role'] == role, 'Missing exact own phase completion')
            same(done['source_files'], self.cfg['source_files'], 'Phase source differs')
            same(done['frozen_candidates'], self.guard_inputs.frozen_pin, 'Phase selected predictor differs')
            verify(done['files'])
            self.receipts.extend(done['files'])
            if phase == 'confirm':
                need(done['cases'] == 250 and done['domain_quota'] == {d: 50 for d in
                     ('knowledge', 'language', 'retrieval', 'tools', 'arts')}, 'Incomplete confirm roster')
                weights_ok(read(folder / 'WEIGHTS.json'))
            elif phase == 'public-worker':
                need(done['rank'] == child['rank'] and done['executed_requests'] == 35155 and
                     done['counts'] == {'ok': 35155}, 'Incomplete own public shard')
                weights_ok(read(folder / 'WEIGHTS.json'))
            else:
                need(phase == 'public-score' and done['executed_requests'] == 140620 and
                     done['scored_requests'] == 140178 and done['counts'] == {'ok': 140620} and
                     done['public_list_changed'] is False, 'Incomplete full official role score')
        self.receipts.append(pin(path))
        self.journal('PHASE-COMPLETE', {'child': self.child_receipt(child), 'receipt': pin(path)})

    def await_owned_exits(self):
        """Unconditional actual-child wait barrier, even if registration/cleanup failed."""
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
                        pid, status = os.waitpid(p.pid, 0)
                        need(pid == p.pid, 'Actual waitpid returned a different child')
                        p.returncode = os.waitstatus_to_exitcode(status)
                        break
                    except InterruptedError:
                        continue  # Only interrupted blocking wait, never timed progress polling.
                    except ChildProcessError:
                        child['wait_note'] = 'own child already reaped; unavailable exit code not invented'
                        break
                    except BaseException as wait_error:
                        child['wait_error'] += '; waitpid: ' + type(wait_error).__name__ + ': ' + str(wait_error)
                        # Cannot establish exit: keep leases forever rather than raise past barrier.
                        signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
                        while True:
                            signal.sigwait(SIGNALS)
            child['waited'] = True
            child['wait_monotonic'] = time.monotonic()

    def begin_cleanup(self):
        """Block BOTH signals before disposition changes; keep blocked through actual waits."""
        signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        for sig in SIGNALS:
            signal.signal(sig, signal.SIG_IGN)

    def stop_owned(self):
        pending, notes = {}, []
        for child in self.children:
            if child['p'] is None or child['waited']:
                continue
            if child['id'] is None or child['fd'] is None:
                notes.append({'pid': child['p'].pid, 'signalled': False, 'actual_Popen_wait_required': True})
                continue
            if identity(child['id']['pid']) == child['id']:
                try:
                    signal.pidfd_send_signal(child['fd'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    notes.append({'process': child['id'], 'TERM_error': str(error)})
                pending[child['fd']] = child
            else:
                notes.append({'process': child['id'], 'signalled': False, 'actual_Popen_wait_required': True})
        deadline = time.monotonic() + self.cfg['resources']['grace_seconds']
        exhausted = False
        while pending:
            ready, _, _ = select.select(list(pending), [], [], None if exhausted else max(0, deadline - time.monotonic()))
            if not ready:
                for fd, child in pending.items():
                    if identity(child['id']['pid']) == child['id']:
                        try:
                            signal.pidfd_send_signal(fd, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except OSError as error:
                            notes.append({'process': child['id'], 'KILL_error': str(error)})
                exhausted = True
                try:
                    commit(ROOT / 'OWNED_SHUTDOWN_PENDING.json', {'owner': self.owner,
                           'children': [self.child_receipt(c) for c in pending.values()],
                           'bounded_grace_exhausted': True, 'leases_retained_until_actual_exit': True})
                except OSError as error:
                    notes.append({'shutdown_receipt_error': str(error)})
                continue
            for fd in ready:
                child = pending.pop(fd)
                child['p'].wait()
                child['waited'] = True
                child['wait_monotonic'] = time.monotonic()
                notes.append(self.child_receipt(child))
        # Unknown/missing registration forbids signals, NOT actual wait on own Popen.
        self.await_owned_exits()
        return notes

    def run(self):
        from evaluate import key
        need(Path(__file__).absolute() == ROOT / 'source' / 'ops.py' and
             Path(__file__).resolve() == ROOT / 'source' / 'ops.py', 'Run only installed frozen manager')
        need(Path(sys.executable).resolve() == Path(SYSTEM_PYTHON).resolve() and
             sys.prefix == sys.base_prefix, 'Use actual non-venv SYSTEM Python for manager')
        need(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'Exact pidfd APIs required')
        need(all(signal.getsignal(sig) is record_stop for sig in SIGNALS),
             'Install the manager-wide non-raising record-only signal handler before run')
        need(not (signal.pthread_sigmask(signal.SIG_BLOCK, set()) & SIGNALS),
             'Manager must start with TERM/INT unblocked for child prior-mask restoration')
        directory(ROOT)
        self.run_fd = os.open(ROOT / 'RUN.lock', os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(self.run_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.fsync(self.run_fd)
            sync_dir(ROOT)
            self.stop_check()
            self.guard_inputs.check()
            for name in ('OWNER.json', 'LAUNCH.json', 'COMPLETE.json', 'BLOCKED.json', 'grants', 'confirm', 'public',
                         'CODEC_COMPLETE.json', 'CODEC_BLOCKED.json', 'CONFIRM_ANALYSIS.json', 'PUBLIC_READY.json'):
                absent(ROOT / name)
            self.events = Events()  # Watches exist before any initial control/completion reads.
            for name in ('grants', 'confirm', 'public'):
                (ROOT / name).mkdir(mode=0o700)
            for role in ROLES:
                (ROOT / 'public' / role).mkdir(mode=0o700)
                (ROOT / 'public' / role / 'workers').mkdir(mode=0o700)
            sync_dir(ROOT)
            commit(ROOT / 'OWNER.json', {'owner': self.owner, 'root': str(ROOT),
                   'config': pin(ROOT / 'config.json'), 'source_files': self.cfg['source_files'],
                   'automatic_restart': False, 'manager_python': str(Path(sys.executable).resolve())})
            self.acquire()
            commit(ROOT / 'LAUNCH.json', {'owner': self.owner, 'root': str(ROOT),
                   'config': pin(ROOT / 'config.json'), 'source_files': self.cfg['source_files'],
                   'roles': list(ROLES), 'devices': self.devices, 'leases_held': [0, 1, 2, 3],
                   'initial_budget': self.guard(), 'old_metadata_written': False,
                   'confirm_attempts_per_role': 1, 'public_attempts_per_role': 1,
                   'signal_stop_detection': {'handler': 'record-only, never raises',
                                             'boundary': 'guard or phase-wait',
                                             'phase_wait_timer_seconds': 15,
                                             'asynchronous_immediate_interrupt': False},
                   'automatic_restart': False, 'same_uid_advisory_not_strong_authentication': True})
            self.wait_children([self.spawn('codec')])
            self.wait_children([self.spawn('confirm', role) for role in ROLES])
            self.wait_children([self.spawn('confirm-analysis')])
            self.wait_children([self.spawn('public-prepare')])
            for role in ROLES:  # Fixed baseline then challenger, never chosen by confirm/public quality.
                self.wait_children([self.spawn('public-worker', role, rank) for rank in range(4)])
                self.wait_children([self.spawn('public-score', role)])
            self.await_owned_exits()
            self.guard()
            commit(ROOT / 'COMPLETE.json', {'complete': True, 'owner': self.owner,
                   'roles': {role: pin(ROOT / 'public' / role / 'COMPLETE.json') for role in ROLES},
                   'codec': pin(ROOT / 'CODEC_COMPLETE.json'), 'confirm_analysis': pin(ROOT / 'CONFIRM_ANALYSIS.json'),
                   'public_ready': pin(ROOT / 'PUBLIC_READY.json'), 'config': pin(ROOT / 'config.json'),
                   'source_binding': self.cfg['source_binding'], 'source_files': self.cfg['source_files'],
                   'frozen_candidates': self.guard_inputs.frozen_pin,
                   'children': [self.child_receipt(c) for c in self.children],
                   'all_own_children_actual_waited': True, 'four_leases_held_until_actual_exit': True,
                   'public_list_changed': False, 'winner_selected': False, 'published': False,
                   'full_geometry_or_cached_serving_admission': False, 'single_card_latency_admission': False})
        except BaseException as error:
            self.begin_cleanup()
            try:
                cleanup = self.stop_owned()
            except BaseException as cleanup_error:
                cleanup = [{'error_type': type(cleanup_error).__name__, 'error': str(cleanup_error),
                            'unconditional_actual_wait_barrier_required': True}]
            self.await_owned_exits()
            if not (ROOT / 'BLOCKED.json').exists() and not (ROOT / 'BLOCKED.json').is_symlink():
                commit(ROOT / 'BLOCKED.json', {'complete': False, 'owner': self.owner,
                       'error_type': type(error).__name__, 'error': str(error), 'cleanup': cleanup,
                       'recorded_stop_signal': _STOP_SIGNAL,
                       'children': [self.child_receipt(c) for c in self.children],
                       'partial_preserved': True, 'automatic_retry': False, 'automatic_restart': False,
                       'public_list_changed': False, 'all_private_artifacts_remain_remote': True})
            raise
        finally:
            self.begin_cleanup()
            self.await_owned_exits()  # Even failure in cleanup or BLOCKED publication cannot bypass waits.
            for child in self.children:
                if child['fd'] is not None:
                    os.close(child['fd'])
            if self.events is not None:
                self.events.close()
            for fd in self.leases:
                os.close(fd)  # No unlock/close before all actual Popen exits, registered or not.
            if self.run_fd is not None:
                os.close(self.run_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=('install', 'run'))
    parser.add_argument('--root', required=True, choices=(str(ROOT),))
    parser.add_argument('--host', choices=('sh-10-gpu',))
    parser.add_argument('--log', type=Path)
    parser.add_argument('--deployment-ready', action='store_true')
    args = parser.parse_args(argv)
    os.umask(0o077)
    need('torch' not in sys.modules, 'No Torch in ops')
    if args.action == 'install':
        need(args.host is not None and args.log is not None, 'Install needs fixed host and new durable log')
        local_install(args.host, args.log, args.deployment_ready)
    else:
        need(args.host is None and args.log is None and args.deployment_ready is False,
             'Run is remote SYSTEM-only; install flags do not authorize or redirect it')
        for sig in SIGNALS:
            signal.signal(sig, record_stop)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        Manager().run()


if __name__ == '__main__':
    main()
