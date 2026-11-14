"""Local install-only deployer. Never starts readback/queue or creates AUTH.

Only three new public Python sources plus an explicit read-only old public
helper template cross SSH stdin. No tensor/data/credential is read locally.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import stat
import subprocess

LOCAL = Path('/tmp/opencode/jev-clean-followup-staged-20261112-v2-local')
OLD_LOCAL = Path('/tmp/opencode/jev-clean-followup-20261112-local')
CTRL = '/tmp/opencode/jev-clean-followup-staged-20261112-v2'
PUBLIC = ('source/main_readback.py', 'source/staged_queue.py', 'deploy_staged.py')

REMOTE = r'''
import base64,importlib.util,json,os,stat,sys,traceback
from pathlib import Path
sys.dont_write_bytecode=True
CTRL=Path('/tmp/opencode/jev-clean-followup-staged-20261112-v2')
def need(ok,message):
 if not ok:raise RuntimeError(message)
def sync(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
def commit(path,value):
 pending=path.with_name(path.name+'.writing')
 with pending.open('x') as f:
  json.dump(value,f,ensure_ascii=False,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
 os.link(pending,path,follow_symlinks=False)
 pending.unlink();sync(path.parent)
need(set(PAYLOAD)=={'sources','old_helper_template'},'Public payload schema differs')
need(set(PAYLOAD['sources'])=={'source/main_readback.py','source/staged_queue.py','deploy_staged.py'},'Only three new public sources allowed')
need(CTRL.parent.is_dir() and CTRL.parent.resolve()==CTRL.parent and not CTRL.parent.is_symlink(),'Unsafe new control parent')
# mkdir is the exclusive ownership claim. A losing installer cannot write any
# BLOCKED receipt in the already existing root. No reuse even of empty roots.
claimed=False
try:
 CTRL.mkdir(mode=0o700)
 claimed=True
 sync(CTRL.parent)
 need(CTRL.resolve()==CTRL and CTRL.stat().st_uid==os.getuid() and stat.S_IMODE(CTRL.stat().st_mode)==0o700,'Unsafe owned control root')
 commit(CTRL/'INSTALLING.json',{'control_root':str(CTRL),'operation':'install-only',
  'local_sources':{k:v['stat'] for k,v in PAYLOAD['sources'].items()},
  'old_helper_public_template':PAYLOAD['old_helper_template']['stat'],
  'readback_launched':False,'queue_launched':False,'authorization_created':False})
 (CTRL/'source').mkdir(mode=0o700)
 for name,item in PAYLOAD['sources'].items():
  path=CTRL/name;data=base64.b64decode(item['base64'],validate=True)
  with path.open('xb') as f:
   f.write(data);f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
  need(not path.is_symlink() and path.resolve()==path and stat.S_ISREG(path.lstat().st_mode) and
   stat.S_IMODE(path.stat().st_mode)==0o444 and path.read_bytes()==data,'Installed public source readback differs')
 sync(CTRL/'source');sync(CTRL)
 spec=importlib.util.spec_from_file_location('staged_install_validation',CTRL/'source/staged_queue.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.build_installation(PAYLOAD)
 need('torch' not in sys.modules,'Installer must not import Torch')
 commit(CTRL/'INSTALL_COMPLETE.json',result)
 print(json.dumps(result,sort_keys=True))
except BaseException as error:
 if claimed:
  commit(CTRL/'INSTALL_BLOCKED.json',{'complete':False,'error_type':type(error).__name__,'error':str(error),
   'traceback':traceback.format_exc(),'partial_preserved':True,'automatic_retry':False,
   'readback_launched':False,'queue_launched':False,'authorization_created':False})
 raise
'''


def source_entry(path, anchor=LOCAL):
    """One verified nofollow FD and ancestor chain; never reopen source bytes."""
    path = Path(path)
    if anchor not in (LOCAL, OLD_LOCAL) or not path.is_absolute() or '..' in path.parts:
        raise ValueError('Fixed trusted public source anchor required')
    relative = path.relative_to(anchor)
    if anchor == LOCAL:
        if relative.as_posix() not in PUBLIC:
            raise ValueError('Only the three fixed new public files are uploadable')
    elif relative.as_posix() != 'ops/side_queue.py':
        raise ValueError('Only explicit read-only old public helper template is allowed')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fds, chain = [], []
    def snapshot(s):
        return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid, s.st_nlink,
                s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    try:
        parent = os.open('/', flags)
        fds.append(parent)
        for component in path.parts[1:-1]:
            opened = os.open(component, flags, dir_fd=parent)
            fds.append(opened)
            chain.append((parent, component, opened))
            parent = opened
        published = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(published.st_mode) or published.st_nlink != 1:
            raise ValueError('Nonsymlink single-link public source required: ' + str(path))
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
        fds.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or snapshot(published) != snapshot(before):
            raise RuntimeError('Source changed between nofollow stat/open')
        def check():
            for directory, name, held in chain:
                current, opened = os.stat(name, dir_fd=directory, follow_symlinks=False), os.fstat(held)
                if not stat.S_ISDIR(current.st_mode) or not stat.S_ISDIR(opened.st_mode) or \
                        (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                    raise RuntimeError('Public source ancestor changed/redirected')
            current, opened = os.stat(path.name, dir_fd=parent, follow_symlinks=False), os.fstat(fd)
            if not stat.S_ISREG(current.st_mode) or snapshot(current) != snapshot(before) or snapshot(opened) != snapshot(before):
                raise RuntimeError('Public source FD/path changed during read')
        check()
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b''.join(chunks)
        check()
        if len(data) != before.st_size:
            raise RuntimeError('Public source byte count changed')
        return {'stat': {'path': str(path), 'resolved': str(path), 'bytes': before.st_size,
                         'mtime_ns': before.st_mtime_ns}, 'base64': base64.b64encode(data).decode('ascii')}
    finally:
        for fd in reversed(fds):
            os.close(fd)


def main():
    if Path(__file__).absolute() != LOCAL / 'deploy_staged.py' or Path(__file__).resolve() != LOCAL / 'deploy_staged.py':
        raise ValueError('Run fixed local reviewed install-only deployer')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, choices=['sh-10-gpu'])
    parser.add_argument('--control-root', required=True, choices=[CTRL])
    parser.add_argument('--log', required=True, type=Path)
    args = parser.parse_args()
    if args.log.parent != LOCAL or args.log.parent.resolve() != LOCAL or args.log.is_symlink():
        raise ValueError('New exclusive log must be a direct fixed LOCAL child')
    payload = {'sources': {name: source_entry(LOCAL / name) for name in PUBLIC},
               'old_helper_template': source_entry(OLD_LOCAL / 'ops/side_queue.py', OLD_LOCAL)}
    script = 'PAYLOAD=' + repr(payload) + '\n' + REMOTE
    # Hold verified LOCAL directory FD, and publish durable exclusive log before
    # any SSH side effect. No inherited agent/X11/forward/control-master channel.
    directory_fd = os.open(LOCAL, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        s = os.fstat(directory_fd)
        if (s.st_dev, s.st_ino) != (LOCAL.lstat().st_dev, LOCAL.lstat().st_ino):
            raise RuntimeError('Local log directory changed')
        fd = os.open(args.log.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=directory_fd)
        with os.fdopen(fd, 'w') as log:
            log.write(json.dumps({'operation': 'install-only', 'host': args.host, 'control_root': CTRL,
                                 'public_sources': {k: v['stat'] for k, v in payload['sources'].items()},
                                 'readback_launched': False, 'queue_launched': False, 'authorization_created': False}) + '\n')
            log.flush()
            os.fsync(log.fileno())
            os.fsync(directory_fd)
            result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectionAttempts=1',
                '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ControlMaster=no',
                '-o', 'ControlPath=none', '-o', 'ForwardAgent=no', '-o', 'ForwardX11=no',
                '-o', 'PermitLocalCommand=no', args.host, 'python3 -B -'],
                input=script, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=240)
            log.flush()
            os.fsync(log.fileno())
    finally:
        os.close(directory_fd)
    if result.returncode:
        raise RuntimeError('Installation failed; preserve local log and partial new control root; no retry')
    print(json.dumps({'install_only': True, 'log': str(args.log), 'queue_launched': False, 'authorization_created': False}))


if __name__ == '__main__':
    main()
