"""Continue reviewed cursor1 arms, then matched fresh bidirectional/prefix CE.

No implicit retries or completed-arm reuse; only original exact checkpoint pins
are resumed. Independent worker failure is retained, never repaired by rewinding.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ['CLEAN_TRAIN_ROOT'])
sys.path[:0]=['/tmp/opencode/jev-padding-free-20261112-v3/code','/tmp/opencode/jev-06b-research-20261112-r1-bridge/code']
from pf_common import atomic,read,own_lock,process_ids,require,utc
import experiment_common as controls
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'
FIRST=('C_D_BCE','C_P_BCE','C_D_CE','C_P_CE')


def phase(tasks,label):
    children=[];result={};parent=controls.identity(os.getpid());started=time.monotonic()
    try:
        for gpu,arm,resume in tasks:
            require(not (ROOT/'STOP_CONTINUATION.json').exists(),'Continuation stopped; no automatic resume')
            require(not process_ids(gpu),'GPU occupied; no preemption')
            output=ROOT/(arm+'-seed42')
            args=[PYTHON,'-B','-u',str(ROOT/'source/train_case.py'),'--config',str(ROOT/'seed42.json'),'--arm',arm,'--output',str(output)]
            if resume:
                args+=['--resume-checkpoint',resume['checkpoint'],'--resume-complete-pin',resume['pin_file']]
            else:require(not output.exists(),'Existing later-method arm preserved')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',
                PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                USE_TORCH='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            def own_setup():controls.parent_death_guard(parent)
            with (ROOT/(arm+'-'+label+'.log')).open('x') as log:
                p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=own_setup)
            record=controls.identity(p.pid);require(record is not None,'Own worker exited before registration')
            children.append((p,record));result[arm]={'gpu':gpu,'process':record,'state':'running','resumed_cursor':1 if resume else 0,'output':str(output)}
        atomic(ROOT/(label+'-ACTIVE.json'),{'phase':label,'owner':parent,'workers':result,'utc':utc()})
        while any(p.poll() is None for p,_ in children):
            require(time.monotonic()-started<86400 and not (ROOT/'STOP_CONTINUATION.json').exists(),'Continuation day-limit/stop')
            for entry,(p,record) in zip(result.values(),children):
                if p.poll() is not None:continue
                foreign=[pid for pid in process_ids(entry['gpu']) if pid!=record['pid'] or
                    (controls.identity(pid)!=record and not (p.poll() is not None and controls.identity(pid) is None))]
                require(not foreign,'External GPU process appeared; stop only owned work')
            time.sleep(2)
        for (arm,entry),(p,_) in zip(result.items(),children):
            marker=Path(entry['output'])/'COMPLETE.json'
            entry.update(exit_code=p.returncode,state='completed' if p.returncode==0 and marker.exists() else 'paused' if p.returncode==75 else 'failed')
            if marker.exists():
                done=read(marker);require(done['complete'] and done['cursor']=={'macro_updates':3557,'cases':56912,'logical_pair_tokens':99824365},'Final exposure receipt differs')
                require(done['D_confirm_opened'] is False and done['public_run'] is False and done['observations']==[10000000,30000000,100000000],'Selection scope drifted')
                entry['actual_completion']={'cursor':done['cursor'],'observations':done['observations'],'actual_runtime':done['actual_runtime'],'first_actual_macro':done['first_actual_macro']}
        atomic(ROOT/(label+'-COMPLETE.json'),{'execution_complete':True,'workers':result,'all_completed':all(v['state']=='completed' for v in result.values()),'utc':utc()})
        return result
    except BaseException as exc:
        cleanup=controls.terminate_owned(children)
        atomic(ROOT/(label+'-FAILURE.json'),{'error':str(exc),'cleanup':cleanup,'workers':result,'partial_preserved':True,'utc':utc()});raise


def main():
    def interrupted(signum,frame):raise InterruptedError('Continuation controller interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with own_lock(ROOT/'CONTINUATION.lock'):
        require(not (ROOT/'CONTINUATION.json').exists(),'Existing continuation preserved; no duplicate')
        approved=read(ROOT/'FIRST_PRIMARY_VERIFICATION.json');require(approved['complete'] and approved['all_four_formal_macro0_verified'] and approved['repeat_macro0_allowed'] is False,'Direct formal first-macro review missing')
        atomic(ROOT/'CONTINUATION.json',{'stage':'resume_four_matched_arms','complete':False,'owner':controls.identity(os.getpid()),'first_macro_reviewed':True,'utc':utc()})
        first=phase([(gpu,arm,approved['resume'][arm]) for gpu,arm in enumerate(FIRST)],'continuation')
        require(all(v['state']=='completed' for v in first.values()),'First matched wave incomplete; no silent rerun or unpaired later methods')
        atomic(ROOT/'CONTINUATION.json',{'stage':'fresh_matched_mask_CE','complete':False,'first_wave':first,'utc':utc()})
        masks=phase([(0,'BI_P_CE',None),(1,'PB_P_CE',None)],'fresh-mask')
        atomic(ROOT/'CONTINUATION.json',{'stage':'available_fullrow_quality_experiments_exited','execution_complete':True,
            'first_wave':first,'fresh_masks':masks,'all_completed':all(v['state']=='completed' for v in masks.values()),
            'pending':['D_select interpretation and frozen candidate confirmation','Jina matched fresh training and full coverage','Base exact state boundary matched training','full public for frozen admitted candidates'],
            'published':False,'utc':utc()})
        print(json.dumps(read(ROOT/'CONTINUATION.json'),indent=2),flush=True)


if __name__=='__main__':main()
