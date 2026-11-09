#!/usr/bin/env python3
"""Own one labeling/training queue; no old queue recovery and no publishing."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent.parent
HOME=Path('/home/dev/assumeengage')
STUDENT=HOME/'envs/jev/bin/python'
TEACHER=HOME/'envs/knowline/bin/python'
CHILDREN={}


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def atomic(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def identity(pid):
    try:
        stat=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
        if stat[0]=='Z':return None
        return {'pid':pid,'starttime':stat[19],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError,IndexError):return None


def stopped():
    if (ROOT/'NO_AUTOMATIC_RESUME').exists():
        raise RuntimeError('User paused this queue; no automatic recovery authorized')


def status(stage,**extra):
    atomic(ROOT/'PIPELINE.json',{'stage':stage,'complete':False,
        'updated_utc':datetime.now(timezone.utc).isoformat(),
        'coordinator':identity(os.getpid()),'children':{k:identity(p.pid) for k,p in CHILDREN.items()},**extra})


def admission(gpus):
    stopped()
    result=subprocess.run(['nvidia-smi','--id='+gpus,'--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
    if result.stdout.strip():raise RuntimeError('Requested GPU is occupied; refusing to preempt another task')
    if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('Insufficient isolated scratch storage')


def launch(name,python,script,args,gpus=None):
    stopped()
    if gpus is not None:admission(gpus)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONPATH=str(HOME/'scripts'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',
        JEV_FRESH_DISTILL_CONFIG=str(ROOT/'config.json'))
    if gpus is not None:env['CUDA_VISIBLE_DEVICES']=gpus
    with (ROOT/(name+'.log')).open('a',buffering=1) as stream:
        process=subprocess.Popen([str(python),'-B','-u',str(ROOT/'code'/script),*args],
            cwd=HOME,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    CHILDREN[name]=process
    atomic(ROOT/'RUNTIME.json',{k:identity(p.pid) for k,p in CHILDREN.items()})
    return process


def wait(processes,stage):
    while True:
        stopped()
        codes=[p.poll() for p in processes]
        if any(code not in (None,0) for code in codes):
            raise RuntimeError(stage+' child failed; preserve partials and inspect own logs')
        if all(code==0 for code in codes):return
        status(stage)
        time.sleep(20)


def execute():
    stopped();status('preparing_frozen_pool')
    wait([launch('prepare',STUDENT,'prepare_five_dimension.py',[])],'preparing_frozen_pool')
    cfg=read(ROOT/'config.json')
    language=ROOT/'language-labeling'
    if not read(language/'TEACHER_LABELS_READY.json').get('complete'):
        workers=[]
        for rank in range(3):
            marker=language/'teacher-labels'/f'worker-{rank}-complete.json'
            if not read(marker).get('complete'):
                workers.append(launch('teacher-'+str(rank),TEACHER,'label_language.py',
                    ['--rank',str(rank),'--workers','3'],str(rank)))
        wait(workers,'labeling_native_language')
        wait([launch('teacher-merge',TEACHER,'label_language.py',['--merge','--workers','3'])],
             'merging_native_language')
    wait([launch('targets',STUDENT,'prepare_five_dimension.py',['--targets'])],'freezing_distillation_targets')
    completion=ROOT/'KD_HYBRID/complete.json'
    if not read(completion).get('complete'):
        wait([launch('student',STUDENT,'train_five_dimension.py',['--arm','KD_HYBRID'],'0')],
            'training_five_dimension_distilled')
    result=read(completion)
    if result.get('complete') is not True or result.get('pairs_seen')!=cfg['pairs']:
        raise RuntimeError('Student exited without the complete own five-domain export')
    atomic(ROOT/'TRAINING_COMPLETE.json',{'complete':True,'training':result,
        'checkpoint':str(ROOT/'KD_HYBRID/final'),'official_public_complete':False,'uploaded':False})
    status('trained_waiting_for_official_engine',checkpoint=str(ROOT/'KD_HYBRID/final'),
        training_complete=True,official_public_complete=False,uploaded=False)
    print(json.dumps({'event':'five_dimension_distilled_training_complete',
        'checkpoint':str(ROOT/'KD_HYBRID/final'),'official_public_complete':False}),flush=True)


def interrupted(signum,frame):
    raise InterruptedError('Coordinator received signal '+str(signum))


def main():
    with (ROOT/'PIPELINE.lock').open('a+') as lock:
        # Failure to acquire ownership must not touch the live owner's state.
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read(ROOT/'RUNTIME.json')
        for name,record in previous.items():
            if record and identity(record.get('pid'))==record:
                raise RuntimeError('Live previous child '+name+' remains; original runtime preserved')
        signal.signal(signal.SIGTERM,interrupted)
        try:
            execute()
        except BaseException as exc:
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            # Cleanup and terminal state remain inside the ownership lock.
            for child in CHILDREN.values():
                if child.poll() is None:child.terminate()
            for child in CHILDREN.values():
                if child.poll() is None:
                    try:child.wait(timeout=120)
                    except subprocess.TimeoutExpired:pass
            atomic(ROOT/'RUNTIME.json',{k:identity(p.pid) for k,p in CHILDREN.items()})
            status('blocked',error=str(exc));traceback.print_exc();raise


if __name__=='__main__':main()
