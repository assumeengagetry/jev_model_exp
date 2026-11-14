"""Safe four-GPU experiment queue: independent failures do not cancel other methods."""
import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/tmp/opencode/jev-attention-queue-20261112-v1')
MASK=Path('/tmp/opencode/jev-attention-2e334fb-20261112-v1')
FLEX=Path('/tmp/opencode/jev-flex-20261112-v1')
PF=Path('/tmp/opencode/jev-padding-free-20261112-v3')
MECHANISM=Path('/tmp/opencode/jev-padding-free-20261112-v2/mechanism')
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'
sys.path.insert(0,'/tmp/opencode/jev-06b-research-20261112-r1-bridge/code')
import experiment_common as controls
sys.path.insert(0,str(PF/'code'))
from pf_common import atomic,read,require,utc,own_lock,process_ids


def stopped():require(not (ROOT/'NO_AUTOMATIC_RESUME').exists(),'Queue stopped; preserve all partials')


def verify(root):
    expected=read(root/'CODE_BINDING.json')
    actual={str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for folder in ('code','commit-code') if (root/folder).exists() for p in sorted((root/folder).rglob('*.py'))}
    require(actual==expected,'Bound source changed: '+str(root))


def run_phase(tasks):
    stopped();children=[];results={};start=time.monotonic()
    try:
        for name,args,gpus,extra,limit in tasks:
            stopped()
            for gpu in gpus:require(not process_ids(gpu),'GPU occupied; never preempt: '+str(gpu))
            parent=controls.identity(os.getpid())
            env=dict(os.environ,PF_ROOT=str(PF),PF_MECHANISM_ROOT=str(MECHANISM),ATTN_ROOT=str(MASK),FLEX_ROOT=str(FLEX),
                CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',**extra)
            def child_setup():controls.parent_death_guard(parent)
            with (ROOT/(name+'.log')).open('x') as output:
                p=subprocess.Popen([PYTHON,'-B','-u',*args],env=env,stdin=subprocess.DEVNULL,
                    stdout=output,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_setup)
            record=controls.identity(p.pid);require(record is not None,'Own child exited before registration')
            children.append((p,record))
            results[name]={'state':'running','process':record,'gpus':gpus,'limit_seconds':limit,'started_monotonic':time.monotonic()}
        atomic(ROOT/'ACTIVE.json',{'tasks':results,'utc':utc()})
        while any(p.poll() is None for p,_ in children):
            stopped()
            for (name,args,gpus,extra,limit),(p,record) in zip(tasks,children):
                code=p.poll();entry=results[name]
                if code is not None:
                    if entry['state']=='running':entry.update(state='complete' if code==0 else 'failed',exit_code=code,completed_utc=utc())
                    continue
                require(time.monotonic()-entry['started_monotonic']<limit,'Queue task time limit: '+name)
                # Single-GPU mechanism workers own their CUDA context directly.
                # Full-public managers independently monitor their exact leaves.
                if len(gpus)==1:
                    foreign=[pid for pid in process_ids(gpus[0]) if pid!=record['pid'] or (controls.identity(pid)!=record and not (p.poll() is not None and controls.identity(pid) is None))]
                    require(not foreign,'External GPU processes appeared: '+repr(foreign))
            atomic(ROOT/'ACTIVE.json',{'tasks':results,'utc':utc()})
            time.sleep(2)
        for (name,*_),(p,_) in zip(tasks,children):
            results[name].update(state='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,completed_utc=utc())
        atomic(ROOT/'ACTIVE.json',{'tasks':results,'utc':utc(),'phase_seconds':time.monotonic()-start})
        return results
    except BaseException as exc:
        cleanup=controls.terminate_owned(children)
        atomic(ROOT/'BLOCKED.json',{'error':str(exc),'cleanup':cleanup,'tasks':results,'utc':utc()})
        raise


def main():
    def interrupted(signum,frame):raise InterruptedError('Queue interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with own_lock(ROOT/'QUEUE.lock'):
        require(not (ROOT/'QUEUE.json').exists(),'Existing queue preserved; no implicit resume')
        verify(ROOT);verify(MASK);verify(FLEX)
        from pf_common import verify as verify_pf
        verify_pf();require(read(MECHANISM/'COMPLETE.json')['complete'],'PF v2 mechanism not passed')
        binding={'pf':str(PF),'mask':str(MASK),'flex':str(FLEX),'commit':'2e334fb76c4cd4826df37424450b2734101db9f8'}
        atomic(ROOT/'QUEUE.json',{'stage':'mechanism_parallel','complete':False,'owner':controls.identity(os.getpid()),'binding':binding,'utc':utc()})
        tasks=[]
        for gpu,mode in enumerate(('causal','bidirectional','prefix_block')):
            tasks.append(('commit-'+mode,[str(MASK/'code/run_modes.py'),'--mode',mode],[gpu],
                {'TRITON_CACHE_DIR':str(MASK/'triton-cache'/mode)},7200))
        tasks.append(('flex-causal',[str(FLEX/'code/experiment.py')],[3],
            {'FLEX_EXCLUSIVE_GPU_ADMITTED':'1','TRITON_CACHE_DIR':str(FLEX/'triton-cache'),
             'TORCHINDUCTOR_CACHE_DIR':str(FLEX/'inductor-cache')},7200))
        mechanisms=run_phase(tasks)
        atomic(ROOT/'MECHANISMS.json',mechanisms)
        # A failed independent mask/Flex method must not suppress the already
        # admitted PF causal full-public control.
        for layout in ('dense','packed'):
            stopped();verify_pf()
            atomic(ROOT/'QUEUE.json',{'stage':'public-'+layout,'complete':False,'binding':binding,'mechanisms':mechanisms,'utc':utc()})
            result=run_phase([('public-'+layout,[str(PF/'code/evaluate.py'),'--layout',layout],[0,1,2,3],
                {'TRITON_CACHE_DIR':str(PF/'triton-cache')},21600)])
            require(result['public-'+layout]['exit_code']==0,'Public stage failed; no retry or prediction substitution')
        result=run_phase([('pf-compare',[str(PF/'code/compare.py')],[],{},3600)])
        comparison=read(PF/'COMPARISON.json')
        stopped();verify_pf()
        atomic(ROOT/'QUEUE.json',{'stage':'available_experiments_finished','complete':True,'binding':binding,
            'mechanisms':mechanisms,'pf_comparison_exit':result['pf-compare']['exit_code'],
            'pf_coverage_complete':comparison['coverage_complete'],'pf_numerical_acceptance':comparison['numerical_acceptance'],
            'public_index':comparison['public_index'],'pending':[
                'clean-origin data admission and mask-matched training',
                'Base two-block vs full-bidirectional clean paired training',
                'Jina-inspired candidate-interaction and loss controls',
                'Zhihu403 awaiting readable article'],
            'uploaded':False,'utc':utc()})
        print(json.dumps(read(ROOT/'QUEUE.json'),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
