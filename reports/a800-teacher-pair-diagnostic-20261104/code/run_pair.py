#!/usr/bin/env python3
"""Same Teacher, previous generic pair prompt, complete frozen evaluation inputs."""
import asyncio
import collections
import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from kd_runtime import ROOT,SOURCE_ROOT,WORKSPACE,atomic_json
from pair_data_common import render_pair,text
sys.path.append(str(SOURCE_ROOT/'staged/scale-series-20261102/code'))
import teacher_stage as serving

NAME='TEACHER_PAIR';MODEL=serving.MODEL;STUDENT=SOURCE_ROOT/'envs/student/bin/python'
PROMPT='Is the candidate a correct match for the input, question, and supplied criteria? Judge this pair independently. Answer Yes for match and No for not_match.'
used_ports=set()


def port():
    while True:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));value=sock.getsockname()[1]
        if value not in used_ports:used_ports.add(value);return value


def data():
    selection=json.loads((ROOT/'selection/evaluation.json').read_text());chosen=set(selection['run_ids'])
    cache=WORKSPACE/'datasets/evaluation-only/decision-index-suite-0.3/candidate-cache/normalized'
    table=pq.read_table(cache/'field-options.parquet')
    table=table.filter(pc.is_in(table['run_id'],value_set=pa.array(sorted(chosen))))
    fields=table.to_pydict();states=np.asarray(fields['state_id'],dtype=np.int64);candidates=np.asarray(fields['candidate_id'],dtype=np.int64)
    pairs,inverse=np.unique(np.stack((states,candidates),axis=1),axis=0,return_inverse=True)
    request_indices=collections.defaultdict(list)
    for i,rid in enumerate(fields['run_id']):request_indices[rid].append(i)
    catalog={}
    with gzip.open(cache/'rows-for-evaluation.jsonl.gz','rt') as source:
        for line in source:
            row=json.loads(line);meta=row['_evaluation']
            if meta['run_id'] in chosen:catalog[meta['run_id']]=meta['catalog_id']
    if set(request_indices)!=chosen or set(catalog)!=chosen:raise RuntimeError('Complete frozen requests/candidates are required')
    inputs=pq.read_table(cache/'inputs.parquet',columns=['state','question','criteria']).combine_chunks()
    answers=pq.read_table(cache/'candidates.parquet',columns=['candidate'])['candidate'].combine_chunks()
    atomic_json(ROOT/'pair-manifest.json',dict(selected_requests=len(chosen),selected_field_options=len(states),
      unique_pair_calls=len(pairs),deduplicated_calls=len(states)-len(pairs),all_candidates_retained=True,
      input_policy='full original rendered pair text; no4096 training cap',target_dtype='float32 two columns',
      readout='log(Pmatch)-log(Pnotmatch), floor1e-30; same candidate normalization as Student',
      prompt=PROMPT,replica_assignment='input_state_id modulo2, all candidates of a state on same replica'))
    return fields,pairs,inverse,request_indices,catalog,inputs,answers


def recover(path):
    seen=set()
    if not path.exists():return seen
    with path.open('rb+') as stream:
        end=0
        while True:
            line=stream.readline()
            if not line:break
            try:value=json.loads(line)
            except ValueError:
                if stream.read().strip():raise RuntimeError('Non-tail damage in owned result file')
                stream.truncate(end);break
            if value.get('status')=='ok':seen.add(value['run_id'])
            end=stream.tell()
        stream.seek(0,2)
        if stream.tell():
            stream.seek(-1,2)
            if stream.read(1)!=b'\n':stream.seek(0,2);stream.write(b'\n')
    return seen


def aggregate(fields,positions,margins):
    groups=collections.OrderedDict()
    for position,margin in zip(positions,margins):
        key=fields['field'][position]
        entry=groups.setdefault(key,dict(kind=fields['kind'][position],keys=[],scores=[]))
        entry['keys'].append(fields['option_key'][position]);entry['scores'].append(margin)
    answers={}
    for field,entry in groups.items():
        values=np.asarray(entry['scores'],dtype=np.float64);prob=np.exp(values-values.max());prob/=prob.sum()
        options={key:float(p) for key,p in zip(entry['keys'],prob)}
        if entry['kind']=='choice':answers[field]=dict(choice=max(options,key=options.get),probabilities=options)
        else:
            positive=[key for key in options if str(key).lower() in ['true','yes','1']]
            if len(positive)!=1:raise RuntimeError('Unrecognized noul keys in original candidate cache')
            yes=options[positive[0]];answers[field]=dict(noul=yes,probabilities=dict(yes=yes,no=1-yes))
    return answers


async def infer(endpoints,loaded):
    fields,pairs,inverse,request_indices,catalog,inputs,candidates=loaded
    folder=ROOT/'outputs/evaluation'/NAME;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'worker-0-complete.json').exists():return
    path=ROOT/'pair-probabilities.npy'
    if not path.exists():
        probabilities=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(pairs),2))
        probabilities[:]=np.nan;probabilities.flush()
    probabilities=np.load(path,mmap_mode='r+')
    if probabilities.shape!=(len(pairs),2):raise RuntimeError('Existing private pair cache does not match frozen inputs')
    recovered=int(np.isfinite(probabilities).all(axis=1).sum());new=0;prefix={};futures={};started=time.time()
    seen=recover(folder/'part-0.jsonl')
    api_sem=asyncio.Semaphore(64);row_sem=asyncio.Semaphore(32)
    timeout=aiohttp.ClientTimeout(total=1200);connector=aiohttp.TCPConnector(limit=64)
    def publish():
        value=dict(processed_requests=len(seen),selected_requests=len(request_indices),completed_pairs=recovered+new,
                   unique_pairs=len(pairs),recovered_pairs=recovered,new_pairs=new,elapsed_seconds=time.time()-started,updated_unix=time.time())
        atomic_json(ROOT/'pair-progress.json',value);print(json.dumps(dict(event='teacher_pair_progress',**value)),flush=True)
    async with aiohttp.ClientSession(timeout=timeout,connector=connector) as session:
        async def score(index):
            nonlocal new
            async with api_sem:
                if np.isfinite(probabilities[index]).all():return
                state,candidate=map(int,pairs[index])
                if state not in prefix:
                    prefix[state]=render_pair(inputs['state'][state].as_py(),inputs['question'][state].as_py(),'',inputs['criteria'][state].as_py())[0]
                payload=dict(model='m',state=prefix[state]+text(candidates[candidate].as_py()),questions={'pair':dict(type='noul',
                    instructions=PROMPT,criteria={'true':'The candidate is a correct match.','false':'The candidate is not a correct match.'})})
                for attempt in range(6):
                    try:
                        async with session.post(endpoints[state%2],json=payload) as response:
                            answer=await response.json()
                            if response.status!=200:raise RuntimeError(f'Pair frontend HTTP{response.status}')
                        yes=float(answer['answers']['pair']['noul'])
                        if not np.isfinite(yes) or not 0<=yes<=1:raise RuntimeError('Nonfinite/out-of-range actual Teacher probability')
                        probabilities[index]=(1.0-yes,yes);new+=1
                        if new%512==0:probabilities.flush();publish()
                        return
                    except Exception as error:
                        print(json.dumps(dict(event='pair_request_retry',pair_index=index,attempt=attempt+1,error_type=type(error).__name__)),flush=True)
                        if attempt==5:raise RuntimeError(f'Actual Teacher pair {index} failed; private cache preserved') from error
                        await asyncio.sleep(min(30,2**attempt))
        def ensure(index):
            index=int(index)
            if np.isfinite(probabilities[index]).all():return None
            if index not in futures:futures[index]=asyncio.create_task(score(index))
            return futures[index]
        with (folder/'part-0.jsonl').open('a',buffering=1) as output:
            async def request(rid):
                if rid in seen:return
                async with row_sem:
                    stamp=time.monotonic();positions=request_indices[rid];indices=inverse[positions]
                    pending=[future for future in (ensure(i) for i in indices) if future is not None]
                    if pending:await asyncio.gather(*pending)
                    p=np.asarray(probabilities[indices],dtype=np.float64)
                    # Keep both original target columns: Pmatch may round to1 while Pnotmatch remains nonzero.
                    margins=np.log(p[:,1].clip(1e-30))-np.log(p[:,0].clip(1e-30))
                    value=dict(run_id=rid,catalog_id=catalog[rid],status='ok',response=dict(answers=aggregate(fields,positions,margins)),
                               total_wall_ms=(time.monotonic()-stamp)*1000,latency_scope='actual paired-adapter request wall time; concurrent, not isolated latency-v1')
                    value['model_request_wall_ms']=value['total_wall_ms']
                    output.write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n');seen.add(rid)
                    if len(seen)%50==0:publish()
            tasks=[asyncio.create_task(request(rid)) for rid in request_indices if rid not in seen]
            try:await asyncio.gather(*tasks)
            except BaseException:
                for task in [*tasks,*futures.values()]:
                    if not task.done():task.cancel()
                await asyncio.gather(*tasks,*futures.values(),return_exceptions=True);probabilities.flush();raise
    probabilities.flush();publish()
    atomic_json(folder/'worker-0-complete.json',dict(complete=True,terminal_requests=len(seen),new_pair_calls=new,
                recovered_pair_calls=recovered,unique_pairs=len(pairs),seconds=time.time()-started))


def main():
    allocation=dict(job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),pid=os.getpid(),
                    devices=os.environ.get('CUDA_VISIBLE_DEVICES'),started_unix=time.time())
    atomic_json(ROOT/'allocation.json',allocation);code=1
    try:
        loaded=data();folder=ROOT/'outputs/evaluation'/NAME
        if not (folder/'worker-0-complete.json').exists():
            devices=[v for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v]
            if len(devices)!=4:raise RuntimeError('Pair diagnosis needs four allocated GPUs for twoTP2replicas')
            if not (MODEL/'DOWNLOAD_COMPLETE.json').exists():raise RuntimeError('Original node-local Teacher weights missing')
            atomic_json(ROOT/'status.json',dict(phase='starting_two_teacher_replicas',unix_time=time.time()))
            backends=[];frontends=[];endpoints=[]
            for replica in range(2):
                bp,fp,bootstrap=port(),port(),port()
                backend=serving.launch([sys.executable,'-m','sglang.launch_server','--model-path',str(MODEL),
                  '--served-model-name','m','--tp-size','2','--dtype','bfloat16','--attention-backend','triton',
                  '--enable-fp32-lm-head','--mem-fraction-static','0.85','--host','127.0.0.1','--port',str(bp),
                  '--engine-info-bootstrap-port',str(bootstrap)],f'pair-backend-{replica}',','.join(devices[2*replica:2*replica+2]))
                backends.append((backend,bp,fp))
            for backend,bp,fp in backends:serving.wait_ready(bp,backend)
            for replica,(_,bp,fp) in enumerate(backends):
                frontend=serving.launch([sys.executable,str(MODEL/'knowline_server.py'),'--model',str(MODEL),
                  '--backend','sglang','--url',f'http://127.0.0.1:{bp}','--served-model-name','m','--temperature','1',
                  '--workers','8','--read-twice','--twice-max-tokens','3000','--temperatures',str(MODEL/'temperatures.json'),
                  '--host','127.0.0.1','--port',str(fp)],f'pair-frontend-{replica}')
                frontends.append((frontend,fp));endpoints.append(f'http://127.0.0.1:{fp}/v1/systemone')
            for frontend,fp in frontends:serving.wait_ready(fp,frontend,timeout=180)
            atomic_json(ROOT/'status.json',dict(phase='teacher_pair_inference',replicas=2,tensor_parallel=2,unix_time=time.time()))
            asyncio.run(infer(endpoints,loaded))
        atomic_json(ROOT/'status.json',dict(phase='pair_scoring',unix_time=time.time()))
        subprocess.run([str(STUDENT),'-u',str(ROOT/'code/score_teacher.py'),'--name',NAME,'--score-only'],check=True)
        subprocess.run([str(STUDENT),'-u',str(ROOT/'code/report_pair.py')],check=True);code=0
    except BaseException as error:
        atomic_json(ROOT/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),unix_time=time.time()))
        traceback.print_exc();raise
    finally:
        for process in reversed(serving.processes):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for process in reversed(serving.processes):
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        for handle in serving.handles:handle.close()
        atomic_json(ROOT/'STEP_EXIT.json',dict(allocation,exit_code=code,ended_unix=time.time()))


if __name__=='__main__':
    with (ROOT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
