"""Independent follow-on mechanism queue; waits on exact predecessor pidfd.

Never restarts the predecessor, retries a public row, or touches its outputs.
"""
import base64
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ.get('PACKED_MASK_ROOT','/tmp/opencode/jev-packed-mask-20261112-v1'))
PREDECESSOR=Path('/tmp/opencode/jev-attention-queue-20261112-v1')
PF=Path('/tmp/opencode/jev-padding-free-20261112-v3')
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'
OWNED=False
sys.path[:0]=[str(PF/'code'),'/tmp/opencode/jev-06b-research-20261112-r1-bridge/code']
from pf_common import atomic,read,metadata,require,own_lock,process_ids,utc
import experiment_common as controls


def stopped():require(not (ROOT/'NO_AUTOMATIC_RESUME').exists(),'Packed mask queue stopped')


def main():
    global OWNED
    with own_lock(ROOT/'QUEUE.lock'):
        OWNED=True
        require(not (ROOT/'QUEUE.json').exists(),'Existing native-mask queue preserved; no implicit resume')
        actual={str(p.relative_to(ROOT/'code')):base64.b64encode(p.read_bytes()).decode() for p in sorted((ROOT/'code').rglob('*.py'))}
        require(actual==read(ROOT/'CODE_BINDING.json'),'Native-mask source changed')
        for value in read(ROOT/'DEPENDENCIES.json')['files']:require(metadata(value['path'])==value,'Bound source changed')
        owner=read(PREDECESSOR/'OBSERVER_OWNER.json');expected={k:owner[k] for k in ('pid','starttime','boot_id')}
        atomic(ROOT/'QUEUE.json',{'stage':'waiting_predecessor_pidfd','complete':False,'owner':controls.identity(os.getpid()),'predecessor':expected,'utc':utc()})
        if controls.identity(expected['pid'])==expected:
            # The actual jev Python omits os.pidfd_open; reuse the already
            # admitted x86_64 syscall helper, without signalling the target.
            fd=controls.pidfd_open_owned(expected['pid'])
            try:
                if controls.identity(expected['pid'])==expected:
                    # Blocking readiness wait, not repeated reading of progress.
                    ready,_,_=select.select([fd],[],[],47000)
                    require(bool(ready),'Bounded predecessor wait expired; GPU work untouched')
            finally:os.close(fd)
        stopped()
        prior=read(PREDECESSOR/'QUEUE.json')
        atomic(ROOT/'PREDECESSOR_EXIT_REFERENCE.json',{'queue_state':prior,
            'note':'An independent predecessor failure does not certify or suppress this mechanism; never retried here',
            'file':metadata(PREDECESSOR/'QUEUE.json')})
        for gpu in range(4):require(not process_ids(gpu),'GPU occupied after predecessor; no preemption')
        children=[];results={};parent=controls.identity(os.getpid())
        try:
            for gpu,mode in enumerate(('causal','bidirectional','prefix_block')):
                stopped();require(not process_ids(gpu),'GPU changed during admission')
                env=dict(os.environ,PACKED_MASK_ROOT=str(ROOT),PF_ROOT=str(PF),CUDA_VISIBLE_DEVICES=str(gpu),
                    PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',TRITON_CACHE_DIR=str(ROOT/'triton-cache'/mode))
                def own_setup():controls.parent_death_guard(parent)
                with (ROOT/(mode+'.log')).open('x') as log:
                    p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'code/mechanism.py'),'--mode',mode],env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=own_setup)
                record=controls.identity(p.pid);require(record is not None,'Own mechanism exited before registration')
                children.append((p,record));results[mode]={'gpu':gpu,'process':record,'state':'running'}
            atomic(ROOT/'QUEUE.json',{'stage':'mechanisms_running','complete':False,'owner':parent,'tasks':results,'utc':utc()})
            started=time.monotonic()
            while any(p.poll() is None for p,_ in children):
                stopped();require(time.monotonic()-started<7200,'Two-hour mechanism limit exhausted')
                for (mode,entry),(p,record) in zip(results.items(),children):
                    code=p.poll()
                    if code is not None:
                        entry.update(state='complete' if code==0 else 'failed',exit_code=code);continue
                    foreign=[pid for pid in process_ids(entry['gpu']) if pid!=record['pid'] or
                        (controls.identity(pid)!=record and not (p.poll() is not None and controls.identity(pid) is None))]
                    require(not foreign,'External GPU process appeared; stop only own tasks')
                time.sleep(2)
            for (mode,entry),(p,_) in zip(results.items(),children):
                entry.update(state='complete' if p.returncode==0 else 'failed',exit_code=p.returncode)
                marker=ROOT/'runs'/mode/'COMPLETE.json'
                if marker.exists():entry['mechanism_receipt']=read(marker)
            stopped()
            atomic(ROOT/'QUEUE.json',{'stage':'executions_finished','execution_complete':True,
                'all_numerically_accepted':all(e['exit_code']==0 for e in results.values()),
                'tasks':results,'public_score':None,'clean_training_admitted':False,'utc':utc()})
            print(json.dumps(read(ROOT/'QUEUE.json'),ensure_ascii=False,indent=2),flush=True)
        except BaseException as exc:
            cleanup=controls.terminate_owned(children)
            atomic(ROOT/'FAILED.json',{'error':str(exc),'cleanup':cleanup,'utc':utc(),'partial_preserved':True})
            raise


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        if OWNED:
            atomic(ROOT/'ENTRY_FAILURE.json',{'complete':False,'error_type':type(exc).__name__,
                'error':str(exc),'traceback':traceback.format_exc(),'utc':utc(),'partial_preserved':True})
        raise
