"""Bounded autonomous sequence; every failure preserves its namespace and stops."""
import os
import signal
import subprocess
import sys
import time
from pf_common import ROOT,PYTHON,verify,atomic,read,require,own_lock,process_ids,utc

sys.path.insert(0,'/tmp/opencode/jev-06b-research-20261112-r1-bridge/code')
import experiment_common as controls


def main():
    verify()
    with own_lock(ROOT/'WORKFLOW.lock'):
        require(not (ROOT/'WORKFLOW.json').exists(),'Existing workflow/partial preserved; no implicit resume')
        env=dict(os.environ,PF_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',
            TRITON_CACHE_DIR=str(ROOT/'triton-cache'))
        atomic(ROOT/'WORKFLOW.json',{'stage':'starting','complete':False,'owner':controls.identity(os.getpid()),'utc':utc()})
        for name,args,visible,limit in [('mechanism',['mechanism.py'],'0',7200),
                ('dense',['evaluate.py','--layout','dense'],'0,1,2,3',21600),
                ('packed',['evaluate.py','--layout','packed'],'0,1,2,3',21600),
                ('compare',['compare.py'],'',3600)]:
            verify()
            for gpu in visible.split(',') if visible else []:require(not process_ids(int(gpu)),'GPU occupied; no preemption')
            parent=controls.identity(os.getpid())
            def child_setup():controls.parent_death_guard(parent)
            children=[]
            try:
                with (ROOT/(name+'.log')).open('x') as log:
                    p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'code'/args[0]),*args[1:]],
                        env=dict(env,CUDA_VISIBLE_DEVICES=visible),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_setup)
                record=controls.identity(p.pid);require(record is not None,'Own workflow child exited before registration')
                children=[(p,record)]
                atomic(ROOT/'WORKFLOW.json',{'stage':name,'complete':False,'owner':parent,'child':record,'utc':utc()})
                deadline=time.monotonic()+limit+120
                while p.poll() is None:
                    require(not (ROOT/'NO_AUTOMATIC_RESUME').exists(),'PF stop requested during '+name)
                    require(time.monotonic()<deadline,'PF stage timeout: '+name)
                    try:p.wait(timeout=2)
                    except subprocess.TimeoutExpired:pass
                code=p.returncode
                verify()
                require(code==0,'PF stage failed: '+name+'; retain all partial outputs')
            except BaseException as exc:
                cleanup=controls.terminate_owned(children)
                atomic(ROOT/'WORKFLOW.json',{'stage':'blocked','failed_stage':name,'complete':False,
                    'error':str(exc),'cleanup':cleanup,'utc':utc()})
                raise
        verify()
        comparison=read(ROOT/'COMPARISON.json')
        require(comparison['complete'] and comparison['numerical_acceptance'],'Unresolved full-public numerical contract violation')
        atomic(ROOT/'WORKFLOW.json',{'stage':'complete','complete':True,'utc':utc(),
            'public_index':read(ROOT/'COMPARISON.json')['public_index'],'uploaded':False})


if __name__=='__main__':main()
