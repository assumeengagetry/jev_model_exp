"""One bounded four-GPU numeric factorial, never an official prediction retry."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ.get('PRECISION_ROOT','/tmp/opencode/jev-padding-precision-20261112-v1'))
sys.path[:0]=['/tmp/opencode/jev-padding-free-20261112-v3/code','/tmp/opencode/jev-06b-research-20261112-r1-bridge/code']
from pf_common import atomic,read,own_lock,process_ids,require,utc
import experiment_common as controls
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'


def main():
    children=[];results={}
    def interrupted(signum,frame):raise InterruptedError('Precision probe interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with own_lock(ROOT/'RUN.lock'):
        require(not (ROOT/'RUN.json').exists(),'Existing probe preserved, no implicit retry')
        # Admit actual equal-geometry input coverage once before any GPU model
        # load. The target/cache case remains the previous failing macro0 case;
        # independent neighbors come only from bounded real metadata, no toys.
        fixture_env=dict(os.environ,PRECISION_ROOT=str(ROOT),PF_ROOT='/tmp/opencode/jev-padding-free-20261112-v3',
            CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        with (ROOT/'fixture-preparation.log').open('x') as output:
            fixture=subprocess.run([PYTHON,'-B','-u',str(ROOT/'code/probe.py'),'--tf32','1','--bf16-reduction','1','--prepare-only'],
                env=fixture_env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=300)
        require(fixture.returncode==0 and read(ROOT/'FIXTURE_READY.json')['ready'],'Actual input geometry coverage unavailable; no GPU launch')
        for gpu in range(4):require(not process_ids(gpu),'GPU occupied; no preemption')
        parent=controls.identity(os.getpid());started=time.monotonic()
        try:
            for gpu,(tf32,red) in enumerate(((1,1),(0,1),(1,0),(0,0))):
                require(not process_ids(gpu),'GPU changed during admission')
                env=dict(os.environ,PRECISION_ROOT=str(ROOT),PF_ROOT='/tmp/opencode/jev-padding-free-20261112-v3',
                    CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',
                    PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TRITON_CACHE_DIR=str(ROOT/'triton-cache'/str(gpu)))
                def own_setup():controls.parent_death_guard(parent)
                key=f'tf32-{tf32}-bf16red-{red}'
                with (ROOT/(key+'.log')).open('x') as log:
                    p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'code/probe.py'),'--tf32',str(tf32),'--bf16-reduction',str(red)],
                        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=own_setup)
                record=controls.identity(p.pid);require(record is not None,'Own probe exited before registration')
                children.append((p,record));results[key]={'gpu':gpu,'process':record,'state':'running'}
            atomic(ROOT/'RUN.json',{'state':'running','owner':parent,'workers':results,'utc':utc()})
            while any(p.poll() is None for p,_ in children):
                require(not (ROOT/'STOP.json').exists() and time.monotonic()-started<1200,'Probe stop/deadline')
                for (key,entry),(p,record) in zip(results.items(),children):
                    if p.poll() is not None:continue
                    foreign=[pid for pid in process_ids(entry['gpu']) if pid!=record['pid'] or
                        (controls.identity(pid)!=record and not (p.poll() is not None and controls.identity(pid) is None))]
                    require(not foreign,'External GPU process appeared; only own tasks can be stopped')
                time.sleep(2)
            for (key,entry),(p,_) in zip(results.items(),children):
                entry.update(state='executed' if p.returncode==0 else 'failed',exit_code=p.returncode)
                if (ROOT/key/'COMPLETE.json').exists():entry['receipt']=read(ROOT/key/'COMPLETE.json')
            atomic(ROOT/'RUN.json',{'execution_complete':True,'workers':results,'all_executed':all(p.returncode==0 for p,_ in children),
                'new_public_score':None,'learning_updates':0,'utc':utc()})
            print(json.dumps(read(ROOT/'RUN.json'),indent=2),flush=True)
        except BaseException as exc:
            cleanup=controls.terminate_owned(children)
            atomic(ROOT/'FAILURE.json',{'error':str(exc),'cleanup':cleanup,'partial_preserved':True,'utc':utc()})
            raise


if __name__=='__main__':main()
