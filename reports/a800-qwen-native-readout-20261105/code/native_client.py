#!/usr/bin/env python3
"""Actual KnowLine evaluation of original task requests; never produces training labels."""
import asyncio
import gzip
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

import aiohttp
from kd_runtime import ROOT,SOURCE_ROOT,WORKSPACE,atomic_json
sys.path.append(str(SOURCE_ROOT/'staged/scale-series-20261102/code'))
import teacher_stage as serving

MODEL=serving.MODEL
NAME='TEACHER_NATIVE'
STUDENT=SOURCE_ROOT/'envs/student/bin/python'


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


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
                if stream.read().strip():raise RuntimeError('Non-tail damage in owned prediction file')
                stream.truncate(end);break
            if value.get('status') in ['ok','unsupported']:seen.add(value['run_id'])
            end=stream.tell()
        stream.seek(0,2)
        if stream.tell():
            stream.seek(-1,2)
            if stream.read(1)!=b'\n':stream.seek(0,2);stream.write(b'\n')
    return seen


def rows():
    selection=json.loads((ROOT/'selection/evaluation.json').read_text())
    selected=set(selection['run_ids']);result=[]
    path=WORKSPACE/'datasets/evaluation-only/decision-index-suite-0.3/candidate-cache/normalized/rows-for-evaluation.jsonl.gz'
    with gzip.open(path,'rt') as source:
        for line in source:
            row=json.loads(line)
            if row['_evaluation']['run_id'] in selected:result.append(row)
    if {r['_evaluation']['run_id'] for r in result}!=selected:raise RuntimeError('Frozen evaluation rows are incomplete')
    return result


def finite_answers(value):
    if isinstance(value,dict):return all(finite_answers(v) for v in value.values())
    if isinstance(value,list):return all(finite_answers(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


async def infer(endpoint,name):
    NAME=name
    requested=rows();folder=ROOT/'outputs/evaluation'/NAME;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'worker-0-complete.json').exists():return
    path=folder/'part-0.jsonl';seen=recover(path);started=time.time();new=0
    semaphore=asyncio.Semaphore(16)
    timeout=aiohttp.ClientTimeout(total=1200)
    connector=aiohttp.TCPConnector(limit=16)
    with path.open('a',buffering=1) as output:
        async with aiohttp.ClientSession(timeout=timeout,connector=connector) as session:
            async def one(row):
                nonlocal new
                meta=row['_evaluation'];rid=meta['run_id']
                if rid in seen:return
                async with semaphore:
                    stamp=time.monotonic()
                    # Deliberately whitelist original inputs; gold/expected/scoring never enter payload.
                    payload=dict(model='m',state=row.get('state',''),questions=row['questions'])
                    reason=None
                    if len(row['questions'])>64:reason='Official frontend maximum64 questions'
                    for question in row['questions'].values():
                        if question.get('type') in ['choice','score'] and len(question.get('criteria',[]))>255:
                            reason='Official frontend maximum255 criteria/levels'
                    reply=None
                    if reason is None:
                        for attempt in range(6):
                            try:
                                async with session.post(endpoint,json=payload) as response:
                                    reply=await response.json()
                                    if response.status!=200:raise RuntimeError(f"Frontend HTTP{response.status}: {str(reply)[:400]}")
                                if set(reply.get('answers',{}))!=set(row['questions']):raise RuntimeError('Native response question IDs disagree')
                                if not finite_answers(reply['answers']):raise RuntimeError('Nonfinite model answer')
                                break
                            except Exception as error:
                                print(json.dumps(dict(event='native_request_retry',run_id=rid,attempt=attempt+1,error_type=type(error).__name__,error=str(error)[:500])),flush=True)
                                if attempt==5:raise RuntimeError(f'Native Teacher failed actual request {rid}; partial predictions retained') from error
                                await asyncio.sleep(min(30,2**attempt))
                    value=dict(run_id=rid,catalog_id=meta['catalog_id'],status='unsupported' if reason else 'ok')
                    if reason:value['reason']=reason
                    else:value['response']=reply
                    value['total_wall_ms']=(time.monotonic()-stamp)*1000
                    value['model_request_wall_ms']=value['total_wall_ms']
                    output.write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
                    seen.add(rid);new+=1
                    if new%50==0:
                        progress=dict(model=NAME,processed_requests=len(seen),selected_requests=len(requested),
                                      new_requests=new,elapsed_seconds=time.time()-started,updated_unix=time.time())
                        atomic_json(ROOT/f'progress-{NAME}.json',progress)
                        print(json.dumps(dict(event='native_teacher_progress',**progress)),flush=True)
            tasks=[asyncio.create_task(one(row)) for row in requested if row['_evaluation']['run_id'] not in seen]
            try:await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True);output.flush();raise
    atomic_json(folder/'worker-0-complete.json',dict(complete=True,terminal_requests=len(seen),new_requests=new,
                seconds=time.time()-started,mode='original native state/questions, no golds sent, no pair conversion'))


def main():
    allocation=dict(job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),pid=os.getpid(),
                    devices=os.environ.get('CUDA_VISIBLE_DEVICES'),started_unix=time.time())
    atomic_json(ROOT/'allocation.json',allocation);code=1
    try:
        if not (MODEL/'DOWNLOAD_COMPLETE.json').exists():raise RuntimeError('Existing complete node-local Teacher weights missing')
        devices=[d for d in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if d]
        if len(devices)!=2:raise RuntimeError('Native Teacher evaluation needs exactly two allocated GPUs')
        folder=ROOT/'outputs/evaluation'/NAME
        if not (folder/'worker-0-complete.json').exists():
            backend_port,frontend_port,bootstrap=port(),port(),port()
            atomic_json(ROOT/'status.json',dict(phase='starting_native_teacher',unix_time=time.time()))
            backend=serving.launch([sys.executable,'-m','sglang.launch_server','--model-path',str(MODEL),
                '--served-model-name','m','--tp-size','2','--dtype','bfloat16','--attention-backend','triton',
                '--enable-fp32-lm-head','--mem-fraction-static','0.85','--host','127.0.0.1',
                '--port',str(backend_port),'--engine-info-bootstrap-port',str(bootstrap)],'native-backend',','.join(devices))
            serving.wait_ready(backend_port,backend)
            frontend=serving.launch([sys.executable,str(MODEL/'knowline_server.py'),'--model',str(MODEL),
                '--backend','sglang','--url',f'http://127.0.0.1:{backend_port}','--served-model-name','m',
                '--temperature','1','--workers','8','--read-twice','--twice-max-tokens','3000',
                '--temperatures',str(MODEL/'temperatures.json'),'--box-prompt','gemma','--box-coords','yxyx',
                '--host','127.0.0.1','--port',str(frontend_port)],'native-frontend')
            serving.wait_ready(frontend_port,frontend,timeout=180)
            atomic_json(ROOT/'status.json',dict(phase='native_teacher_inference',frontend_port=frontend_port,unix_time=time.time()))
            asyncio.run(infer(f'http://127.0.0.1:{frontend_port}/v1/systemone'))
        atomic_json(ROOT/'status.json',dict(phase='native_scoring',unix_time=time.time()))
        subprocess.run([str(STUDENT),'-u',str(ROOT/'code/score_teacher.py'),'--name',NAME,'--score-only'],check=True)
        subprocess.run([str(STUDENT),'-u',str(ROOT/'code/report_native.py')],check=True)
        code=0
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


if __name__=='__main__':raise SystemExit('Import this client from run_readout.py; no standalone Teacher launch.')
