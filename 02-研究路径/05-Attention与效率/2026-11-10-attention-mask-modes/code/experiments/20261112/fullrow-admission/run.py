"""Three independent full-row VJP mechanisms, never preempting other jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ['FULLROW_ROOT'])
sys.path[:0]=['/tmp/opencode/jev-padding-free-20261112-v3/code','/tmp/opencode/jev-06b-research-20261112-r1-bridge/code']
from pf_common import atomic,read,own_lock,process_ids,require,utc
import experiment_common as controls
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'


def main():
    children=[];results={}
    def interrupted(signum,frame):raise InterruptedError('Fullrow admission interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with own_lock(ROOT/'RUN.lock'):
        require(not (ROOT/'RUN.json').exists(),'Prior admission run preserved')
        parent=controls.identity(os.getpid());started=time.monotonic()
        try:
            for gpu,mode in enumerate(('causal','bidirectional','prefix_block')):
                require(not process_ids(gpu),'GPU occupied; no preemption')
                env=dict(os.environ,FULLROW_ROOT=str(ROOT),PF_ROOT='/tmp/opencode/jev-padding-free-20261112-v3',
                    CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',
                    PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                def own_setup():controls.parent_death_guard(parent)
                with (ROOT/(mode+'.log')).open('x') as log:
                    p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'code/neighbor_vjp.py'),'--mode',mode],env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=own_setup)
                identity=controls.identity(p.pid);require(identity is not None,'Own worker exited before registration')
                children.append((p,identity));results[mode]={'gpu':gpu,'process':identity,'state':'running'}
            atomic(ROOT/'RUN.json',{'owner':parent,'workers':results,'complete':False})
            while any(p.poll() is None for p,_ in children):
                require(time.monotonic()-started<1200 and not (ROOT/'STOP.json').exists(),'Admission deadline/stop')
                for entry,(p,identity) in zip(results.values(),children):
                    if p.poll() is not None:continue
                    foreign=[pid for pid in process_ids(entry['gpu']) if pid!=identity['pid'] or
                        (controls.identity(pid)!=identity and not (p.poll() is not None and controls.identity(pid) is None))]
                    require(not foreign,'External process appeared; terminate only our tasks')
                time.sleep(2)
            for (mode,entry),(p,_) in zip(results.items(),children):
                entry.update(exit_code=p.returncode,state='executed' if p.returncode==0 else 'failed')
                if (ROOT/mode/'COMPLETE.json').exists():entry['receipt']=read(ROOT/mode/'COMPLETE.json')
            atomic(ROOT/'RUN.json',{'execution_complete':True,'workers':results,'all_accepted':all(p.returncode==0 for p,_ in children),'utc':utc()})
            print(json.dumps(read(ROOT/'RUN.json'),indent=2),flush=True)
        except BaseException as exc:
            cleanup=controls.terminate_owned(children)
            atomic(ROOT/'FAILURE.json',{'error':str(exc),'cleanup':cleanup,'partial_preserved':True});raise


if __name__=='__main__':main()
