#!/usr/bin/env python3
"""Three existing Qwen backbones, original LMhead, matched native evaluation; no training."""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from kd_runtime import ROOT,SOURCE_ROOT,atomic_json
from native_client import port,infer,serving,MODEL

NAMES=['QWEN_RAW_LM','QWEN_HARD_LM','QWEN_LS_LM']
STUDENT=SOURCE_ROOT/'envs/student/bin/python'
ports=set()


def unique_port():
    while True:
        value=port()
        if value not in ports:ports.add(value);return value


async def evaluate_all(endpoints):
    tasks=[asyncio.create_task(infer(endpoints[name],name)) for name in NAMES]
    try:await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True);raise


def main():
    allocation=dict(pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),
                    devices=os.environ.get('CUDA_VISIBLE_DEVICES'),started_unix=time.time())
    atomic_json(ROOT/'allocation.json',allocation);code=1
    try:
        atomic_json(ROOT/'status.json',dict(phase='exporting_existing_backbones',new_training_steps=0,unix_time=time.time()))
        subprocess.run([str(STUDENT),'-u',str(ROOT/'code/export_backbones.py')],check=True)
        paths=json.loads((ROOT/'native-model-paths.json').read_text())
        devices=[v for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v]
        if len(devices)!=3:raise RuntimeError('Native-readout diagnostic requires exactly three allocatedGPUslots')
        backends=[];frontends=[];endpoints={}
        atomic_json(ROOT/'status.json',dict(phase='starting_qwen_native_services',unix_time=time.time()))
        for name,device in zip(NAMES,devices):
            bp,fp,bootstrap=unique_port(),unique_port(),unique_port()
            backend=serving.launch([sys.executable,'-m','sglang.launch_server','--model-path',paths[name],
               '--served-model-name','m','--tp-size','1','--dtype','bfloat16','--attention-backend','triton',
               '--mem-fraction-static','0.35','--host','127.0.0.1','--port',str(bp),
               '--engine-info-bootstrap-port',str(bootstrap)],f'backend-{name}',device)
            backends.append((name,backend,bp,fp))
        for name,backend,bp,fp in backends:serving.wait_ready(bp,backend)
        for name,_,bp,fp in backends:
            # Same generic native frontend, but own Qwen tokenizer/chat template and no Gemma calibrationfile.
            frontend=serving.launch([sys.executable,str(MODEL/'knowline_server.py'),'--model',paths[name],
               '--backend','sglang','--url',f'http://127.0.0.1:{bp}','--served-model-name','m','--temperature','1',
               '--workers','4','--read-twice','--twice-max-tokens','3000','--host','127.0.0.1','--port',str(fp)],f'frontend-{name}')
            frontends.append((frontend,fp));endpoints[name]=f'http://127.0.0.1:{fp}/v1/systemone'
        for frontend,fp in frontends:serving.wait_ready(fp,frontend,timeout=180)
        atomic_json(ROOT/'status.json',dict(phase='qwen_native_readout_inference',conditions=NAMES,new_training_steps=0,unix_time=time.time()))
        asyncio.run(evaluate_all(endpoints))
        for name in NAMES:
            subprocess.run([str(STUDENT),'-u',str(ROOT/'code/score_readout.py'),'--name',name,'--score-only'],check=True)
        subprocess.run([str(STUDENT),'-u',str(ROOT/'code/report_readout.py')],check=True);code=0
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


if __name__=='__main__':main()
