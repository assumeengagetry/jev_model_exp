"""Actual CPU fresh creation/readback, then exec formal first-wave controller."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(os.environ['CLEAN_TRAIN_ROOT'])
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'


def meta(path):
    p=Path(path);s=p.stat();return {'path':str(p),'resolved':str(p.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def main():
    init=ROOT/'init-seed42'
    if init.exists() or (ROOT/'seed42.json').exists():raise RuntimeError('Prior init/final config preserved; no implicit retry')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',USE_TORCH='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
    with (ROOT/'fresh-init.log').open('x') as log:
        result=subprocess.run([PYTHON,'-B','-u',str(ROOT/'source/fresh_init.py'),'--config',str(ROOT/'init-config.json'),
            '--arm','C_P_CE','--output',str(init)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:raise RuntimeError('Actual CPU fresh init failed; preserve log/partial, no first-wave launch')
    import torch
    from safetensors.torch import load_file
    read=lambda p:json.loads(Path(p).read_text())
    done=read(init/'COMPLETE.json');manifest=read(init/'FRESH_MANIFEST.json')
    assert done['complete'] and done['seed']==42 and not (init/'BLOCKED.json').exists()
    for item in done['files']:assert meta(item['path'])==item
    values=load_file(str(init/'adapter_model.safetensors'),device='cpu')
    a=[k for k in values if '.lora_A.' in k];b=[k for k in values if '.lora_B.' in k];h=[k for k in values if k.endswith('score.weight')]
    assert len(values)==393 and len(a)==len(b)==196 and len(h)==1 and values[h[0]].shape==(2,1024)
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in values.values())
    assert all(torch.count_nonzero(values[k])==0 for k in b) and torch.count_nonzero(values[h[0]])>0
    assert manifest['model_forward_calls']==0 and manifest['audit']['active_served_parameters']==616237056
    cfg=read(ROOT/'init-config.json');cfg['fresh_init']['complete']=meta(init/'COMPLETE.json')
    with (ROOT/'seed42.json').open('x') as f:json.dump(cfg,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    (ROOT/'seed42.json').chmod(0o444)
    report={'complete':True,'actual_saved_fp32_tensor_keys':393,'actual_lora_pairs':196,'all_B_exactly_zero':True,
        'head_shape':[2,1024],'active_served_parameters':616237056,'trainable_parameters':20187136,
        'base_origin':cfg['stock_directory'],'model_forward_calls':0,'historical_project_adapter_or_teacher_loaded':False,
        'init_complete':meta(init/'COMPLETE.json'),'final_config':meta(ROOT/'seed42.json')}
    with (ROOT/'FRESH_PRIMARY_VERIFICATION.json').open('x') as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(report,indent=2),flush=True)
    # exec replaces the CPU verifier completely: no Torch state is inherited
    # by the first-wave controller or by any training child.
    os.execve(PYTHON,[PYTHON,'-B','-u',str(ROOT/'run_first.py')],dict(os.environ))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        with (ROOT/'PREPARE_FIRST_FAILURE.json').open('x') as f:json.dump({'complete':False,'error':str(error),'traceback':traceback.format_exc(),'partial_preserved':True},f,indent=2)
        raise
