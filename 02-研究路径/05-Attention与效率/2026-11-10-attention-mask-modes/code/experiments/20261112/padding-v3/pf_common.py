"""Independent PF namespace; old trainers and pause markers are never resumed."""
import base64
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(os.environ.get('PF_ROOT','/tmp/opencode/jev-padding-free-20261112-v1'))
OLD=Path('/tmp/opencode/jev-joint-generic-continuation-20261109')
MODEL=Path('/tmp/opencode/jev-06b-research-20261112-r1-bridge/train/JOINT_BRIDGE/final')
BASE=OLD/'official-frozen/base-model'
FA3='/home/dev/assumeengage/cache/attention-kernels/flash-attn3-torch29-cu128'
KIT=OLD/'decision-index'
SUITE=OLD/'official-suite-0.3-verified'
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'
ENGINE='packed_engine:PaddingFreeEngine'
MECHANISM_ROOT=Path(os.environ.get('PF_MECHANISM_ROOT',str(ROOT/'mechanism')))


def require(ok,message):
    if not ok:raise RuntimeError(message)


def utc():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.writing')
    with temp.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def metadata(path):
    path=Path(path).absolute();s=path.stat()
    require(path.is_file(),'Expected file: '+str(path))
    return {'path':str(path),'resolved':str(path.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def verify():
    require(not (ROOT/'NO_AUTOMATIC_RESUME').exists(),'PF stopped; preserve partials')
    actual={str(p.relative_to(ROOT/'code')):base64.b64encode(p.read_bytes()).decode()
            for p in sorted((ROOT/'code').rglob('*.py'))}
    require(actual==read(ROOT/'CODE_BINDING.json'),'Sealed PF source changed')
    value=read(ROOT/'SOURCE_BINDING.json')
    for record in value['files']:
        require(metadata(record['path'])==record,'PF bound source changed: '+record['path'])
    return value


def own_lock(path):
    handle=Path(path).open('a+')
    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    return handle


def process_ids(gpu):
    result=subprocess.run(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    return [int(s) for s in result.stdout.splitlines() if s.strip()]
