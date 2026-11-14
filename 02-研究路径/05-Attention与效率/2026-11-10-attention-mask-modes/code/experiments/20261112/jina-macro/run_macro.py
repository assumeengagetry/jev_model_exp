import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ['JINA_ROOT'])
sys.path[:0]=['/tmp/opencode/jev-padding-free-20261112-v3/code','/tmp/opencode/jev-06b-research-20261112-r1-bridge/code']
from pf_common import atomic,read,own_lock,process_ids,require,utc
import experiment_common as controls
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'


def main():
    children=[];results={}
    def interrupted(signum,frame):raise InterruptedError('Jina component macro interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with own_lock(ROOT/'RUN.lock'):
        require(not (ROOT/'RUN.json').exists(),'Prior macro experiment preserved')
        parent=controls.identity(os.getpid());started=time.monotonic()
        try:
            for gpu,arm in enumerate(('J0','J1','J2')):
                require(not process_ids(gpu),'GPU occupied; no preemption')
                env=dict(os.environ,JINA_ROOT=str(ROOT),JINA_ARM=arm,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',
                    PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                    TORCHINDUCTOR_CACHE_DIR=str(ROOT/'inductor-cache'/arm),TRITON_CACHE_DIR=str(ROOT/'triton-cache'/arm))
                def own_setup():controls.parent_death_guard(parent)
                with (ROOT/(arm+'.log')).open('x') as log:
                    p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'code/actual_macro.py'),'--arm',arm],env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=own_setup)
                identity=controls.identity(p.pid);require(identity is not None,'Own worker exited before registration')
                children.append((p,identity));results[arm]={'gpu':gpu,'process':identity,'state':'running'}
            atomic(ROOT/'RUN.json',{'owner':parent,'workers':results,'execution_complete':False})
            while any(p.poll() is None for p,_ in children):
                require(time.monotonic()-started<1800 and not (ROOT/'STOP.json').exists(),'Macro experiment deadline/stop')
                for entry,(p,identity) in zip(results.values(),children):
                    if p.poll() is not None:continue
                    # Inductor compilation may have CPU-only subprocesses; CUDA
                    # belongs to the exact single-GPU worker, no external sharing.
                    foreign=[pid for pid in process_ids(entry['gpu']) if pid!=identity['pid'] or
                        (controls.identity(pid)!=identity and not (p.poll() is not None and controls.identity(pid) is None))]
                    require(not foreign,'External GPU process appeared; stop only own workers')
                time.sleep(2)
            for (arm,entry),(p,_) in zip(results.items(),children):
                entry.update(exit_code=p.returncode,state='executed' if p.returncode==0 else 'failed')
                if (ROOT/arm/'COMPLETE.json').exists():entry['receipt']=read(ROOT/arm/'COMPLETE.json')
            atomic(ROOT/'RUN.json',{'execution_complete':True,'workers':results,'all_executed':all(p.returncode==0 for p,_ in children),
                'fresh_clean_training':False,'public_score':None,'utc':utc()})
            print(json.dumps(read(ROOT/'RUN.json'),indent=2),flush=True)
        except BaseException as exc:
            cleanup=controls.terminate_owned(children)
            atomic(ROOT/'FAILURE.json',{'error':str(exc),'cleanup':cleanup,'partial_preserved':True});raise


if __name__=='__main__':main()
