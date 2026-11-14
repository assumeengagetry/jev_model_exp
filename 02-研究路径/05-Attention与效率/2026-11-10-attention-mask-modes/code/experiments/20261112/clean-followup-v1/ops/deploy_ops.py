"""Explicit local ops-only installation into the ALREADY prepared follow-up ROOT.

Transfers public Python via SSH stdin; never starts the queue or changes frozen
source9/config/fresh. A failed/partial deployment is preserved, never retried.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import stat
import subprocess

LOCAL = Path('/tmp/opencode/jev-clean-followup-20261112-local')
ROOT = '/tmp/opencode/jev-clean-trainer-20261112-followup-v1'
SOURCE9 = ('bindings.py', 'fresh_init.py', 'case_runtime.py', 'objectives.py', 'model_runtime.py',
           'checkpoints.py', 'train_case.py', 'evaluate_native_dev.py', 'state_boundaries.py')
OLD_OPS = Path('/tmp/opencode/jev-clean-trainer-ops-20261112-local')

REMOTE = r'''
import base64,json,os,stat,sys,traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
OLD=Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
KEYS={'path','resolved','bytes','mtime_ns'}
def require(ok,message):
 if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text())
def pin(path):
 p=Path(path);s=p.stat();require(stat.S_ISREG(s.st_mode),'Nonregular pinned file')
 return {'path':str(p),'resolved':str(p.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def verify(value):
 if isinstance(value,dict):
  if set(value)==KEYS:require(pin(value['path'])==value,'Changed pin: '+value['path'])
  else:
   for v in value.values():verify(v)
 elif isinstance(value,list):
  for v in value:verify(v)
def immutable(path):
 p=Path(path);require(not p.is_symlink() and p.resolve()==p and p.stat().st_mode&0o222==0,'Immutable file required: '+str(p))
def directory(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
def commit(path,value):
 global claimed
 pending=path.with_name(path.name+'.writing')
 with pending.open('x') as f:
  json.dump(value,f,ensure_ascii=False,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
 os.link(pending,path,follow_symlinks=False)
 # Only the exclusive FINAL publication wins ownership. Set before any later
 # unlink/fsync can fail; losing/pre-link/staging failures cannot poison a winner.
 if path.name=='OPS_INSTALLING.json':claimed=True
 pending.unlink();directory(path.parent)
def entry(path):return {'stat':pin(path),'base64':base64.b64encode(Path(path).read_bytes()).decode()}
def same(path,encoded):
 require(Path(path).read_bytes()==base64.b64decode(encoded,validate=True),'Exact local/remote bytes differ: '+str(path))
def identity(pid):
 try:
  f=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
  return {'pid':int(pid),'starttime':f[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
 except FileNotFoundError:return None
require(ROOT.is_dir() and ROOT.resolve()==ROOT and not ROOT.is_symlink(),'Existing actual bootstrap ROOT required')
require((ROOT/'ops').is_dir() and (ROOT/'ops').resolve()==ROOT/'ops','Existing nonsymlink ops directory required')
require(not any((ROOT/n).exists() or (ROOT/n).is_symlink() for n in
 ('OPS_BUNDLE.json','OPS_BLOCKED.json','OPS_INSTALLING.json','QUEUE.lock')),'Existing ops/queue scope preserved; no retry')
require(not any((ROOT/'ops'/n).exists() or (ROOT/'ops'/n).is_symlink() for n in OPS),'Existing ops source preserved')
claimed=False
try:
 commit(ROOT/'OPS_INSTALLING.json',{'root':str(ROOT),'local_deploy_source':DEPLOY_SOURCE,'training_launched':False})
 require(not list(ROOT.glob('BOOTSTRAP_BLOCKED*')),'Actual bootstrap is blocked')
 cfg=read(ROOT/'seed42.json');fresh=read(ROOT/'FRESH_PRIMARY_VERIFICATION.json');deployment=read(ROOT/'DEPLOYMENT.json')
 require(cfg['version']=='clean-case-followup-v1' and cfg['seed']==42 and cfg['approved_for_execution'] is True and
  cfg['output_parent']==str(ROOT) and cfg['checkpoint_every_macros']==100 and cfg['world_size']==1 and
  cfg['resources']=={'max_output_bytes':25*2**30,'free_reserve_bytes':20*2**30},'Final frozen training contract differs')
 require({p['path'] for p in cfg['trainer_files']}=={str(ROOT/'source'/n) for n in SOURCES},'Exact source9 inventory required')
 require(set(SOURCES)==set(SOURCE_NAMES) and set(OPS)=={'side_queue.py','verify_first.py'},'Installation inventory differs')
 verify(cfg);verify(fresh)
 for name,encoded in SOURCES.items():immutable(ROOT/'source'/name);same(ROOT/'source'/name,encoded)
 immutable(ROOT/'seed42.json');immutable(ROOT/'ops/freeze_fresh.py')
 same(ROOT/'ops/freeze_fresh.py',BOOTSTRAP)
 require(deployment['complete'] is True and deployment['training_launched'] is False and
  deployment['source_files']==cfg['trainer_files'] and deployment['source_bytes']==SOURCES and
  deployment['bootstrap_source']==pin(ROOT/'ops/freeze_fresh.py'),'Actual bootstrap deployment source binding differs')
 require(fresh['complete'] is True and fresh['independently_generated_before_old_oracle_load'] is True and
  fresh['393_initial_tensors_bitwise_equal_old_v2'] is True and fresh['actual_saved_fp32_tensor_keys']==393 and
  fresh['model_forward_calls']==0 and fresh['optimizer_steps']==0 and fresh['CUDA_initialized'] is False and
  fresh['training_launched'] is False and fresh['D_confirm_opened'] is False and fresh['public_run'] is False and
  fresh['final_config']==pin(ROOT/'seed42.json') and fresh['trainer_files']==cfg['trainer_files'] and
  fresh['trainer_code']=={str(ROOT/'source'/n):v for n,v in SOURCES.items()} and
  fresh['bootstrap_source']==pin(ROOT/'ops/freeze_fresh.py') and fresh['deployment']==pin(ROOT/'DEPLOYMENT.json') and
  fresh['new_init_complete']==cfg['fresh_init']['complete']==pin(ROOT/'init-seed42/COMPLETE.json'),
  'Actual fresh393/final-config/source chain differs')
 init=read(ROOT/'init-seed42/COMPLETE.json');verify(init)
 require(init['complete'] is True and init['seed']==42 and not list((ROOT/'init-seed42').glob('BLOCKED*')),
  'Actual completed fresh init required; regeneration forbidden')
 old_cfg=read(OLD/'seed42.json');verify(old_cfg)
 require({Path(p['path']).name for p in old_cfg['trainer_files']}==set(BASELINE),'Old source8 inventory differs')
 for p in old_cfg['trainer_files']:same(p['path'],BASELINE[Path(p['path']).name])
 owner_value=read(OLD/'CONTINUATION_OWNER.json');owner={k:owner_value[k] for k in ('pid','starttime','boot_id')}
 current=identity(owner['pid']);require(current is None or current==owner,'Original controller PID reused')
 cmdline=None
 if current is not None:
  raw=(Path('/proc')/str(owner['pid'])/'cmdline').read_bytes();argv=[os.fsdecode(x) for x in raw.split(b'\0') if x]
  environ=(Path('/proc')/str(owner['pid'])/'environ').read_bytes().split(b'\0')
  require(('CLEAN_TRAIN_ROOT='+str(OLD)).encode() in environ,'Wrong original controller root')
  candidates=[Path(x) for x in argv if Path(x).is_absolute() and Path(x).name=='run_continuation.py']
  cmdline=base64.b64encode(raw).decode()
 else:
  require((OLD/'continuation-COMPLETE.json').is_file(),'Dead original controller without phase receipt')
  candidates=[p for p in OLD.rglob('run_continuation.py') if p.is_file() and not p.is_symlink()]
 require(len(candidates)==1,'Cannot unambiguously bind archived original controller')
 controller=candidates[0];same(controller,CONTROLLER)
 old_review=read(OLD/'FIRST_PRIMARY_VERIFICATION.json')
 require(old_review['complete'] is True and old_review['all_four_formal_macro0_verified'] is True and
  old_review['repeat_macro0_allowed'] is False,'Original first-primary review absent')
 original={'owner':owner,'cmdline_base64':cmdline,'post_four_gpus':[0,1],
  'files':[entry(OLD/'seed42.json'),entry(OLD/'CONTINUATION_OWNER.json'),
           entry(OLD/'FIRST_PRIMARY_VERIFICATION.json'),entry(controller)]+
          [entry(p['path']) for p in old_cfg['trainer_files']]}
 evidence=[entry(ROOT/n) for n in ('FRESH_PRIMARY_VERIFICATION.json','DEPLOYMENT.json','init-seed42/COMPLETE.json',
                                  'init-seed42/FRESH_MANIFEST.json','fresh-compare-analysis.py')]
 # Only these TWO source files are installed. Nothing above writes source/config/fresh.
 for name,encoded in OPS.items():
  path=ROOT/'ops'/name;data=base64.b64decode(encoded,validate=True)
  with path.open('xb') as f:f.write(data);f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
  same(path,encoded);immutable(path)
 directory(ROOT/'ops')
 files=([entry(ROOT/'source'/n) for n in SOURCE_NAMES]+[entry(ROOT/'seed42.json')]+
        [entry(ROOT/'ops'/n) for n in ('freeze_fresh.py','side_queue.py','verify_first.py')])
 for item in files+evidence+original['files']:
  verify(item['stat']);same(item['stat']['path'],item['base64'])
 verify(fresh);verify(init);require(read(ROOT/'seed42.json')==cfg,'Final cfg changed during ops transfer')
 bundle={'version':'followup-ops-bundle-v1','complete':True,'root':str(ROOT),'files':files,'evidence':evidence,
  'fresh':fresh,'fresh_files':init['files'],'original':original,'local_deploy_source':DEPLOY_SOURCE,
  'local_ops_sources':LOCAL_OPS,'installation':pin(ROOT/'OPS_INSTALLING.json'),
  'training_launched':False,'queue_launched':False,'source_config_fresh_written':False,
  'D_confirm_opened':False,'public_run':False}
 commit(ROOT/'OPS_BUNDLE.json',bundle)
 require(read(ROOT/'OPS_BUNDLE.json')==bundle,'Bundle publication readback differs')
 same(ROOT/'ops/side_queue.py',OPS['side_queue.py']);same(ROOT/'ops/verify_first.py',OPS['verify_first.py'])
 print(json.dumps({'complete':True,'ops_bundle':pin(ROOT/'OPS_BUNDLE.json'),'installed':list(OPS),
  'exact_byte_readback':True,'queue_launched':False,'training_launched':False,'frozen_cfg':pin(ROOT/'seed42.json')},sort_keys=True))
except BaseException as error:
 if claimed:
  commit(ROOT/'OPS_BLOCKED.json',{'complete':False,'error_type':type(error).__name__,'error':str(error),
   'traceback':traceback.format_exc(),'partial_preserved':True,'automatic_retry':False,'queue_launched':False})
 raise
'''


def entry(path, anchor=LOCAL):
    """Collect public source through ONE verified file FD, never a path reopen.

    Walk even the fixed anchor's ancestors from / with O_NOFOLLOW directory FDs;
    validate the held chain against its published names before/after byte reads.
    """
    p = Path(path)
    if anchor not in (LOCAL, OLD_OPS) or not p.is_absolute() or '..' in p.parts:
        raise ValueError('Fixed trusted absolute source anchor required')
    try:
        relative = p.relative_to(anchor)
    except ValueError:
        raise ValueError('Local source escaped its trusted anchor: ' + str(p)) from None
    if not relative.parts:
        raise ValueError('A source file beneath the anchor is required')
    fds, chain = [], []
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    def snapshot(s):
        return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid, s.st_nlink,
                s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    try:
        parent = os.open('/', directory_flags)
        fds.append(parent)
        for component in p.parts[1:-1]:
            opened = os.open(component, directory_flags, dir_fd=parent)
            fds.append(opened)
            chain.append((parent, component, opened))
            parent = opened
        leaf = p.name
        published = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(published.st_mode) or published.st_nlink != 1:
            raise ValueError('Only a nonsymlink, single-link public regular source is uploadable: ' + str(p))
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
        fds.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or snapshot(published) != snapshot(before):
            raise RuntimeError('Source changed between nofollow stat/open: ' + str(p))
        def check_published():
            for directory, name, held in chain:
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                opened = os.fstat(held)
                if not stat.S_ISDIR(current.st_mode) or not stat.S_ISDIR(opened.st_mode) or \
                        (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                    raise RuntimeError('Source ancestor changed/redirected: ' + str(p))
            current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(fd)
            if not stat.S_ISREG(current.st_mode) or snapshot(current) != snapshot(before) or \
                    snapshot(opened) != snapshot(before):
                raise RuntimeError('Source FD no longer matches unchanged published file: ' + str(p))
        check_published()  # Reject any path/ancestor change before reading bytes.
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b''.join(chunks)
        check_published()
        if len(data) != before.st_size:
            raise RuntimeError('Source byte count changed during FD read: ' + str(p))
        # Every component is verified nonsymlink; resolved is the same absolute
        # published path. Do not resolve/reopen it to obtain bytes after checking.
        return {'stat': {'path': str(p), 'resolved': str(p), 'bytes': before.st_size,
                         'mtime_ns': before.st_mtime_ns},
                'base64': base64.b64encode(data).decode()}
    finally:
        for fd in reversed(fds):
            os.close(fd)


def main():
    if Path(__file__).absolute() != LOCAL / 'ops/deploy_ops.py':
        raise ValueError('Run the fixed local reviewed deployer, not a different copy')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, choices=[ROOT])
    parser.add_argument('--host', required=True, choices=['sh-10-gpu'])
    parser.add_argument('--log', required=True, type=Path, help='New exclusive local execution log under this workspace')
    args = parser.parse_args()
    if args.log.parent.resolve() != LOCAL or args.log.is_symlink():
        raise ValueError('Execution log must be a new direct child of the local follow-up workspace')
    sources = {name: entry(LOCAL / 'source' / name)['base64'] for name in SOURCE9}
    baseline = {name: entry(LOCAL / 'baseline' / name)['base64'] for name in SOURCE9 if name != 'state_boundaries.py'}
    local_ops = {name: entry(LOCAL / 'ops' / name) for name in ('side_queue.py', 'verify_first.py')}
    payload = {'SOURCES': sources, 'SOURCE_NAMES': SOURCE9, 'BASELINE': baseline,
               'OPS': {name: item['base64'] for name, item in local_ops.items()}, 'LOCAL_OPS': local_ops,
               'BOOTSTRAP': entry(LOCAL / 'ops/freeze_fresh.py')['base64'],
               'CONTROLLER': entry(OLD_OPS / 'run_continuation.py', anchor=OLD_OPS)['base64'],
               'DEPLOY_SOURCE': entry(LOCAL / 'ops/deploy_ops.py')}
    script = '\n'.join(key + '=' + repr(value) for key, value in payload.items()) + '\n' + REMOTE
    # The exclusive durable log is open BEFORE the first SSH side effect.
    with args.log.open('x') as log:
        log.write(json.dumps({'root': ROOT, 'host': args.host, 'operation': 'install-ops-only',
                              'local_deploy_source': payload['DEPLOY_SOURCE']['stat']}) + '\n')
        log.flush()
        os.fsync(log.fileno())
        result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectionAttempts=1',
            '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ControlMaster=no',
            '-o', 'ForwardAgent=no', '-o', 'ForwardX11=no', '-o', 'PermitLocalCommand=no',
            '-o', 'ControlPath=none', args.host, 'python3 -B -'], input=script, stdout=log,
            stderr=subprocess.STDOUT, text=True, timeout=180)
        log.flush()
        os.fsync(log.fileno())
    print(args.log.read_text(), end='')
    if result.returncode:
        raise RuntimeError('Ops deployment failed; preserve execution log/partial remote scope; no retry')


if __name__ == '__main__':
    main()
