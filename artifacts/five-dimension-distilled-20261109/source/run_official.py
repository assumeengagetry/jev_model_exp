#!/usr/bin/env python3
"""Wait for this final, then run exactly the upstream full public pipeline locally."""
from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
HOME=Path('/home/dev/assumeengage')
COMMIT='25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d'
CHILD=None


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def atomic(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def publish(stage,**extra):
    atomic(ROOT/'OFFICIAL_STATUS.json',{'stage':stage,'complete':False,
        'updated_utc':datetime.now(timezone.utc).isoformat(),'upstream_commit':COMMIT,**extra})


def live(record):
    try:
        stat=(Path('/proc')/str(record['pid'])/'stat').read_text().rsplit(') ',1)[1].split()
        return stat[0]!='Z' and stat[19]==record['starttime'] and record['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except (OSError,IndexError,KeyError):return False


def check_pause():
    if (ROOT/'NO_AUTOMATIC_RESUME').exists():raise RuntimeError('This queue is paused; official run not continued')


def file_list(folder):
    return [{'name':str(p.relative_to(folder)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
            for p in sorted(folder.rglob('*')) if p.is_file()]


def verify_snapshot(record):
    if record.get('complete') is not True or file_list(ROOT/'official-frozen')!=record.get('files'):
        raise RuntimeError('Frozen model/base/engine snapshot changed; no reuse or completion allowed')


def freeze(final,base):
    marker=ROOT/'OFFICIAL_SNAPSHOT.json';destination=ROOT/'official-frozen'
    if marker.exists():
        record=read(marker);verify_snapshot(record);return record
    if destination.exists():raise RuntimeError('Incomplete snapshot preserved; refusing to overwrite or adopt')
    sources={'model':sorted(p for p in final.iterdir() if p.is_file()),
        'base-model':sorted(p for p in base.iterdir() if p.is_file() and p.name!='DOWNLOAD.json'
            and (p.suffix in ('.json','.safetensors','.bin','.model','.txt','.tiktoken','.md')
                 or p.name in ('LICENSE','NOTICE'))),
        'engine':[ROOT/'code/decision_engine.py',ROOT/'code/requirements-inference.txt']}
    def source_metadata():
        return {role:[{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
            for p in paths] for role,paths in sources.items()}
    before=source_metadata()
    for role,paths in sources.items():
        folder=destination/role;folder.mkdir(parents=True)
        for path in paths:shutil.copy2(path,folder/path.name)
    if source_metadata()!=before:raise RuntimeError('Source changed while freezing; partial snapshot preserved')
    record={'complete':True,'sources':before,'files':file_list(destination),
        'base_origin':'actual training base, not an assumed replacement HF main revision'}
    for p in destination.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted(destination.rglob('*'),reverse=True):
        if p.is_dir():p.chmod(0o555)
    destination.chmod(0o555)
    atomic(marker,record);verify_snapshot(record);return record


def interrupted(signum,frame):
    raise InterruptedError('Official coordinator received signal '+str(signum))


@contextmanager
def ownership():
    with (ROOT/'OFFICIAL_RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if live(read(ROOT/'OFFICIAL_CHILD.json')):
            raise RuntimeError('Previous official child remains alive; runtime preserved')
        signal.signal(signal.SIGTERM,interrupted)
        try:
            yield lock
        except BaseException as exc:
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            if CHILD is not None and CHILD.poll() is None:
                CHILD.terminate();CHILD.wait()
            publish('blocked',error=str(exc));raise


def main():
    global CHILD
    with ownership() as run_lock:
        while not read(ROOT/'TRAINING_COMPLETE.json').get('complete'):
            check_pause()
            pipeline=read(ROOT/'PIPELINE.json')
            if pipeline.get('stage')=='blocked':raise RuntimeError('Own training pipeline is blocked')
            if pipeline and not live(pipeline.get('coordinator',{})):
                raise RuntimeError('Own training coordinator exited before its completed final')
            publish('waiting_for_completed_five_dimension_final')
            time.sleep(30)
        check_pause()
        training=read(ROOT/'TRAINING_COMPLETE.json')
        cfg=read(ROOT/'config.json');final=Path(training['checkpoint'])
        if final!=ROOT/'KD_HYBRID/final' or training['training'].get('pairs_seen')!=cfg['pairs']:
            raise RuntimeError('Final belongs to another checkpoint or exposure')
        snapshot=freeze(final,HOME/cfg['base_model'])
        frozen=ROOT/'official-frozen'
        binding={'original_checkpoint':str(final),'training':training['training'],
            'snapshot':snapshot,'upstream_commit':COMMIT,
            'checkpoint':str(frozen/'model'),'base_model':str(frozen/'base-model'),
            'inference':{'device':'cuda:0','max_microbatch_pairs':32,'max_microbatch_tokens':32768,'attention':'sdpa'}}
        run=ROOT/'runs/five-dimension-distilled';run.mkdir(parents=True,exist_ok=True)
        saved=run/'CHECKPOINT_BINDING.json'
        if saved.exists() and read(saved)!=binding:raise RuntimeError('Existing official run is bound to another model')
        if not saved.exists():atomic(saved,binding)
        completed=read(ROOT/'OFFICIAL_COMPLETE.json');previous_scores=read(run/'scores.json')
        if completed:
            if (completed.get('complete') is not True or previous_scores.get('complete') is not True
                    or completed.get('scores')!=str(run/'scores.json') or read(saved)!=binding):
                raise RuntimeError('Completed official materials are inconsistent; no automatic rerun')
            verify_snapshot(snapshot)
            publish('official_public_complete_waiting_for_publication_approval',official_public_complete=True,
                public_index=previous_scores.get('decision_index'),uploaded=False)
            print(json.dumps({'event':'completed_official_run_reused_without_model_load'}),flush=True)
            return
        if previous_scores.get('complete') is True:
            raise RuntimeError('Complete scores without own completion receipt; preserve for explicit adoption, never rerun automatically')
        while True:
            check_pause()
            query=subprocess.run(['nvidia-smi','--id=0','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
            if not query.stdout.strip():break
            publish('waiting_for_unoccupied_official_gpu');time.sleep(30)
        verify_snapshot(snapshot)
        env=dict(os.environ,PYTHONPATH=str(frozen/'engine')+os.pathsep+str(ROOT/'decision-index'),
            CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        command=[str(HOME/'envs/jev/bin/python'),'-B','-u','-m','decision_index','pipeline',
            '--engine','decision_engine:PairClassifierEngine','--model',str(frozen/'model'),
            '--option','base_model='+str(frozen/'base-model'),
            '--option','max_microbatch_pairs=32','--option','max_microbatch_tokens=32768',
            '--edition','0.3','--suite-dir',str(ROOT/'official-suite-0.3-verified'),'--out',str(run)]
        atomic(ROOT/'OFFICIAL_COMMAND.json',{'command':command,'upstream_commit':COMMIT,
            'suite':'official import strict verification passed','upload':False,'paid_jobs':False})
        publish('running_official_public_0.3',run=str(run),checkpoint=str(final))
        blocked=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT})
        try:
            with (ROOT/'official-public.log').open('a',buffering=1) as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,pass_fds=(run_lock.fileno(),),
                    # This coordinator is single-threaded. Restore the child's mask
                    # before exec; only the parent's registration window stays blocked.
                    preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,blocked))
            CHILD=child
            stat=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()
            atomic(ROOT/'OFFICIAL_CHILD.json',{'pid':child.pid,'starttime':stat[19],
                'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,blocked)
        try:
            while child.poll() is None:
                check_pause();time.sleep(20)
        except BaseException:
            child.terminate();child.wait();raise
        if child.returncode:raise RuntimeError('Official pipeline stopped; partials preserved for same-run resume')
        verify_snapshot(snapshot)
        scores=read(run/'scores.json')
        if scores.get('complete') is not True:raise RuntimeError('Official scores.json is not complete')
        atomic(ROOT/'OFFICIAL_COMPLETE.json',{'complete':True,'scores':str(run/'scores.json'),
            'public_index':scores.get('decision_index'),'upstream_commit':COMMIT,
            'private_full_score':None,'uploaded':False})
        publish('official_public_complete_waiting_for_publication_approval',official_public_complete=True,
            public_index=scores.get('decision_index'),uploaded=False)
        print(json.dumps({'event':'official_public_complete','public_index':scores.get('decision_index'),
            'scores':str(run/'scores.json'),'private_full_score':None,'uploaded':False}),flush=True)


if __name__=='__main__':main()
