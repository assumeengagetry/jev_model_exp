"""No-retry complete official public, one independent output namespace per layout."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from pf_common import ROOT,MODEL,BASE,KIT,SUITE,PYTHON,ENGINE,verify,read,atomic,metadata,require,own_lock,utc,process_ids

sys.path.insert(0,'/tmp/opencode/jev-06b-research-20261112-r1-bridge/code')
import experiment_common as controls


def worker(layout,rank):
    verify();ready=read(ROOT/'DATA_READY.json');folder=ROOT/'eval'/layout/'workers'/str(rank)
    folder.mkdir(parents=True,exist_ok=False)
    require(os.environ['CUDA_VISIBLE_DEVICES']==str(rank),'Wrong GPU')
    manager=json.loads(os.environ['PF_MANAGER'])
    controls.parent_death_guard(manager)
    sys.path[:0]=[str(ROOT/'code'),str(KIT)]
    from decision_index.runner import run
    from decision_index.suite.io import read_jsonl
    def log(message):
        verify_stop_only()
        require(controls.live(manager),'Own PF manager exited')
        print(message,flush=True)
        try:v=json.loads(message)
        except (ValueError,TypeError):return
        if v.get('event')=='progress':
            require(not any(n for k,n in v.get('counts',{}).items() if k!='ok'),'Non-ok actual request; no retry')
    options={'model':str(MODEL),'base_model':str(BASE),'layout':layout,'device':'cuda:0',
             'max_microbatch_pairs':32,'max_microbatch_tokens':32768}
    atomic(folder/'OWNER.json',{'process':controls.identity(os.getpid()),'manager':manager,'rank':rank,'layout':layout})
    os.environ['PF_MARGINS_PATH']=str(folder/'margins.jsonl')
    started=time.monotonic()
    result=run(ENGINE,options,Path(ready['shards'][rank]['rows']),folder,
        corpus_sha256=ready['official_verification']['sha256'],log=log)
    actual=Counter();seen=set()
    for row in read_jsonl(folder/'results.jsonl'):
        require(row['run_id'] not in seen,'Duplicate output');seen.add(row['run_id'])
        require(row['engine']==ENGINE and row['response']['model']=='qwen3-0.6b-pf-mechanism-'+layout,'Wrong model output')
        actual[row['status']]+=1
    require(result['completed']==35155 and actual=={'ok':35155},'Incomplete official shard')
    verify()
    atomic(folder/'COMPLETE.json',{'complete':True,'layout':layout,'rank':rank,'requests':35155,
        'counts':dict(actual),'seconds':time.monotonic()-started,'files':[metadata(folder/n) for n in ('results.jsonl','environment.json','status.json','margins.jsonl')]})


def verify_stop_only():
    require(not (ROOT/'NO_AUTOMATIC_RESUME').exists(),'PF stopped; preserve partials')


def merge_score(layout):
    sys.path.insert(0,str(KIT))
    from decision_index.suite.io import Suite,dumps,read_jsonl
    from decision_index.pipeline import score_run
    suite=Suite(SUITE,'0.3');folder=ROOT/'eval'/layout;streams=[]
    for rank in range(4):
        part=folder/'workers'/str(rank);done=read(part/'COMPLETE.json')
        require(done['complete'] and done['rank']==rank and done['layout']==layout,'Wrong completion')
        for v in done['files']:require(metadata(v['path'])==v,'Shard file changed')
        streams.append(iter(read_jsonl(part/'results.jsonl')))
    count=0;seen=set()
    with (folder/'results.jsonl').open('x') as output:
        for i,source in enumerate(suite.rows(apply_exclusions=False)):
            row=next(streams[i%4],None);expected=source['_evaluation']
            require(row is not None and all(row.get(k)==v for k,v in expected.items()),'Official metadata/shard alignment failed')
            require(row.get('status')=='ok' and row.get('engine')==ENGINE and row.get('response',{}).get('model')=='qwen3-0.6b-pf-mechanism-'+layout
                and row.get('payload')=={'state':source['state'],'questions':source['questions']},'Payload/model/status changed')
            require(row['run_id'] not in seen,'Duplicate actual output');seen.add(row['run_id'])
            output.write(dumps(row)+'\n');count+=1
        require(count==140620 and all(next(s,None) is None for s in streams),'Missing/extra official output')
        output.flush();os.fsync(output.fileno())
    scores=score_run(suite,folder/'results.jsonl',ENGINE,folder)
    require(scores['complete'] and scores['completed']==140178 and scores['counts']=={'ok':140178},'Incomplete official scores')
    atomic(folder/'COMPLETE.json',{'complete':True,'layout':layout,'executed_requests':140620,
        'scored_requests':140178,'counts':{'ok':140620},'public_index':scores['decision_index'],
        'official_single_gpu_latency_admission':False,'clean_ranking_candidate':False,
        'files':[metadata(folder/n) for n in ('results.jsonl','scores.json','benchmark-summary.json','index.json')],
        'completed_utc':utc()})
    print(json.dumps({'event':'pf_full_public_complete','layout':layout,'public_index':scores['decision_index']}),flush=True)


def manager(layout):
    verify();require(read(ROOT/'mechanism/COMPLETE.json')['complete'],'Real mechanism gate not passed')
    folder=ROOT/'eval'/layout;folder.mkdir(parents=True,exist_ok=False)
    children=[];started=time.monotonic()
    def interrupted(signum,frame):raise InterruptedError('PF coordinator interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with own_lock(folder/'RUN.lock') as lock:
        try:
            parent=controls.identity(os.getpid())
            for rank in range(4):
                require(not process_ids(rank),'Occupied GPU; no preemption')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),PF_MANAGER=json.dumps(parent))
                # Exact own parent cutoff; never touches an old controller.
                def child_setup():controls.parent_death_guard(parent)
                with (folder/f'worker-{rank}.log').open('x') as log:
                    child=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'code/evaluate.py'),'--layout',layout,'--worker',str(rank)],
                        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        pass_fds=(lock.fileno(),),preexec_fn=child_setup)
                record=controls.identity(child.pid);require(record is not None,'Own child exited before registration')
                children.append((child,record))
            atomic(folder/'WORKERS.json',{'manager':parent,'workers':[r for _,r in children]})
            while any(p.poll() is None for p,_ in children):
                verify_stop_only();require(time.monotonic()-started<21600,'Six-hour declared per-layout limit')
                for rank,(p,record) in enumerate(children):
                    code=p.poll()
                    if code is not None:require(code==0,'Official worker failed; retain all partials')
                    foreign=[]
                    for pid in process_ids(rank):
                        if pid==record['pid'] and (controls.identity(pid)==record or (p.poll() is not None and controls.identity(pid) is None)):continue
                        foreign.append(pid)
                    require(not foreign,'External GPU processes appeared: '+repr(foreign))
                time.sleep(10)
            require(all(p.returncode==0 for p,_ in children),'A worker failed')
            verify();merge_score(layout)
            atomic(folder/'TIMING.json',{'parallel_pipeline_seconds':time.monotonic()-started,'single_gpu_admission':False})
        except BaseException:
            signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
            cleanup=controls.terminate_owned(children)
            atomic(folder/'FAILED.json',{'complete':False,'cleanup':cleanup,'utc':utc()})
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--layout',choices=('dense','packed'),required=True)
    parser.add_argument('--worker',type=int,choices=range(4));args=parser.parse_args()
    worker(args.layout,args.worker) if args.worker is not None else manager(args.layout)
