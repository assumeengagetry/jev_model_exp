#!/usr/bin/env python3
"""Run staged three-arm comparisons using only the GPUs granted to this step."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from kd_runtime import ROOT, pool, shared_order, atomic_json


def main():
    atomic_json(ROOT/"logs/allocation.json",dict(job_id=os.environ.get("SLURM_JOB_ID"),step_id=os.environ.get("SLURM_STEP_ID"),
                pid=os.getpid(),role="student",root=str(ROOT),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    if (ROOT/"outputs/pilot-complete.json").exists():
        print(json.dumps({"event":"stage_already_complete","root":str(ROOT)}),flush=True)
        return
    shared_order(pool())
    devices=[v for v in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if v]
    if not devices:
        raise RuntimeError("No GPUs granted to this Student step")
    names=["KD_HARD","KD_SOFT","KD_HYBRID"]
    training=[];evaluation=[];complete=set();active={};handles=[]
    for name in names:
        evaluated=ROOT/"outputs/evaluation"/name/"complete.json"
        if evaluated.exists() and json.loads(evaluated.read_text()).get("complete") and json.loads(evaluated.read_text()).get("scope")=="pilot_subset":complete.add(name)
        elif (ROOT/"outputs"/name/"complete.json").exists():evaluation.append(name)
        else:training.append(name)
    def launch(name,phase,device):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS="4" if phase=="train" else "2",MKL_NUM_THREADS="4" if phase=="train" else "2",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
        script="train_student.py" if phase=="train" else "evaluate_student.py"
        arguments=["--arm",name.removeprefix("KD_")] if phase=="train" else ["--name",name]
        handle=(ROOT/"logs"/f"{phase}-{name}.log").open("a",buffering=1);handles.append(handle)
        child=subprocess.Popen([sys.executable,"-u",str(ROOT/"code"/script),*arguments],env=env,stdout=handle,stderr=subprocess.STDOUT)
        active[device]=dict(name=name,phase=phase,process=child)
        print(json.dumps({"event":"stage_component_started","model":name,"phase":phase,"device":device,"pid":child.pid}),flush=True)
    while len(complete)<len(names):
        for device,job in list(active.items()):
            code=job["process"].poll()
            if code is None:continue
            if code:raise RuntimeError(f"Stage {job['name']} {job['phase']} exited {code}; saved artifacts retained")
            if job["phase"]=="train":evaluation.append(job["name"])
            else:complete.add(job["name"])
            del active[device]
        for device in devices:
            if device in active:continue
            if training:launch(training.pop(0),"train",device)
            elif evaluation:launch(evaluation.pop(0),"evaluate",device)
        atomic_json(ROOT/"outputs/stage-progress.json",dict(completed_models=sorted(complete),pending_training=training,
                    pending_evaluation=evaluation,active={device:dict(model=j["name"],phase=j["phase"],pid=j["process"].pid) for device,j in active.items()},updated_unix=time.time()))
        if len(complete)<len(names):time.sleep(5)
    for handle in handles:handle.close()
    subprocess.run([sys.executable,str(ROOT/"code/report_results.py")],check=True)
    print(json.dumps({"event":"scale_stage_complete","root":str(ROOT),"pairs":len(pool())}),flush=True)


if __name__=="__main__":main()
