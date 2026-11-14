"""Four independent formal macro0 executions; pause at cursor1 for direct review."""
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
ARMS=('C_D_BCE','C_P_BCE','C_D_CE','C_P_CE')
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'


def main():
    children=[];results={}
    def interrupted(signum,frame):raise InterruptedError('Formal first-wave controller interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with own_lock(ROOT/'FIRST_WAVE.lock'):
        require(not (ROOT/'FIRST_WAVE.json').exists(),'Prior formal wave preserved; no implicit retry')
        require(read(ROOT/'FRESH_PRIMARY_VERIFICATION.json')['complete'],'Actual fresh init not verified')
        parent=controls.identity(os.getpid());started=time.monotonic()
        try:
            for gpu,arm in enumerate(ARMS):
                require(not process_ids(gpu),'GPU occupied; never preempt')
                out=ROOT/(arm+'-seed42');require(not out.exists(),'Prior arm assets preserved')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',
                    PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    USE_TORCH='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                def own_setup():controls.parent_death_guard(parent)
                with (ROOT/(arm+'-first.log')).open('x') as log:
                    p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'source/train_case.py'),'--config',str(ROOT/'seed42.json'),
                        '--arm',arm,'--output',str(out),'--pause-after-macros','1'],env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=own_setup)
                record=controls.identity(p.pid);require(record is not None,'Own first worker exited before registration')
                children.append((p,record));results[arm]={'gpu':gpu,'process':record,'state':'running','output':str(out)}
            atomic(ROOT/'FIRST_WAVE.json',{'stage':'formal_macro0_running','owner':parent,'arms':results,'execution_complete':False,'utc':utc()})
            while any(p.poll() is None for p,_ in children):
                require(time.monotonic()-started<3600 and not (ROOT/'STOP_FIRST_WAVE.json').exists(),'Formal wave deadline/stop')
                for entry,(p,record) in zip(results.values(),children):
                    if p.poll() is not None:continue
                    foreign=[pid for pid in process_ids(entry['gpu']) if pid!=record['pid'] or
                        (controls.identity(pid)!=record and not (p.poll() is not None and controls.identity(pid) is None))]
                    require(not foreign,'Foreign GPU process appeared; only owned workers may stop')
                time.sleep(2)
            for (arm,entry),(p,_) in zip(results.items(),children):
                out=Path(entry['output']);marker=out/'first-actual-macro/COMPLETE.json'
                entry.update(exit_code=p.returncode,state='paused_at_cursor1' if p.returncode==75 and marker.exists() else 'failed')
                if marker.exists():entry['first_macro_committed']=str(marker)
                entry['checkpoint_markers']=[str(p) for p in sorted(out.glob('checkpoint-*/COMPLETE.json'))]
            atomic(ROOT/'FIRST_WAVE.json',{'stage':'first_wave_exited','arms':results,'execution_complete':True,
                'all_paused_at_cursor1':all(e['state']=='paused_at_cursor1' for e in results.values()),
                'automatic_continuation':False,'primary_actual_macro_review_required':True,'utc':utc()})
            print(json.dumps(read(ROOT/'FIRST_WAVE.json'),indent=2),flush=True)
        except BaseException as exc:
            cleanup=controls.terminate_owned(children)
            atomic(ROOT/'FIRST_WAVE_FAILURE.json',{'error':str(exc),'cleanup':cleanup,'partial_preserved':True,'utc':utc()});raise


if __name__=='__main__':main()
