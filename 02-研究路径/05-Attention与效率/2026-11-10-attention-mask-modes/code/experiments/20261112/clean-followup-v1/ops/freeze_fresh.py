"""Exclusive nine-source deployment and independently generated zero-forward fresh.

This CPU-only bootstrap never starts a training worker or allocates a GPU.
"""
import base64
import json
from pathlib import Path
import subprocess


LOCAL = Path(__file__).parent.parent
NAMES = ('bindings.py', 'fresh_init.py', 'case_runtime.py', 'objectives.py', 'model_runtime.py',
         'checkpoints.py', 'train_case.py', 'evaluate_native_dev.py', 'state_boundaries.py')
REMOTE = r'''
import base64,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
OLD=Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
PRODUCT=Path('/tmp/opencode/jev-base-state-boundary-20261112-v2')
PYTHON='/home/dev/assumeengage/envs/jev/bin/python'
def pin(path):
 p=Path(path);s=p.stat();return {'path':str(p),'resolved':str(p.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def read(path):return json.loads(Path(path).read_text())
def verify(pins):
 for p in pins:
  if pin(p['path'])!=p:raise RuntimeError('Input changed: '+p['path'])
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def exact_write(path,data):
 with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 Path(path).chmod(0o444)
 if Path(path).read_bytes()!=data:raise RuntimeError('Transfer/readback differs')
def used(path):
 return sum((Path(folder)/name).stat().st_size for folder,dirs,names in os.walk(path) for name in names)
if ROOT.exists() or ROOT.is_symlink():raise RuntimeError('Existing followup namespace preserved; no implicit retry')
old_cfg_pin=pin(OLD/'seed42.json');old=read(OLD/'seed42.json')
verify(old['trainer_files']+old['core_files']+old['native_files']+old['stock_files']+[old['data_complete']])
if {Path(p['path']).name for p in old['trainer_files']}!=set(BASELINE):raise RuntimeError('Old source inventory differs')
for p in old['trainer_files']:
 if base64.b64encode(Path(p['path']).read_bytes()).decode()!=BASELINE[Path(p['path']).name]:raise RuntimeError('Actual old frozen source differs')
future_old=0
for arm in ('C_D_BCE','C_P_BCE','C_D_CE','C_P_CE','BI_P_CE','PB_P_CE'):
 folder=OLD/(arm+'-seed42')
 if not (folder/'COMPLETE.json').exists():future_old+=max(0,25*2**30-(used(folder) if folder.exists() else 0))
free=shutil.disk_usage(OLD).free
if free<future_old+5*2**30+20*2**30:raise RuntimeError('Insufficient preserved old-queue/fresh/shared reserve; no cleanup')
os.umask(0o077);ROOT.mkdir(mode=0o700)
try:
 (ROOT/'source').mkdir(mode=0o700);(ROOT/'ops').mkdir(mode=0o700);(ROOT/'primary').mkdir(mode=0o700)
 for name,encoded in SOURCES.items():exact_write(ROOT/'source'/name,base64.b64decode(encoded,validate=True))
 exact_write(ROOT/'ops/freeze_fresh.py',base64.b64decode(BOOTSTRAP,validate=True))
 cfg=json.loads(json.dumps(old));cfg.update(version='clean-case-followup-v1',output_parent=str(ROOT))
 cfg['arms'].update(STATE_NL_PB_P_CE={'mode':'prefix_block','layout':'packed','loss':'CE'},JPAIR_P_BCE_RANK={'mode':'causal','layout':'packed','loss':'BCE_RANK_HARD'})
 cfg['followup_objective']={'version':'hard-label-bce-rank-v1','lambda':1.0,'tau':1.0,'hard_label_rule':'original-host-q-exact-zero-or-one','normalization':'known-field-mean-then-case-mean'}
 cfg['state_boundary']={'root':str(PRODUCT/'prepared-01'),'complete':pin(PRODUCT/'prepared-01/COMPLETE.json'),'primary':pin(PRODUCT/'FINAL_PRIMARY_VERIFICATION.json')}
 cfg['trainer_files']=[pin(ROOT/'source'/name) for name in SOURCES]
 cfg['fresh_init']={'directory':str(ROOT/'init-seed42'),'complete':{'path':str(ROOT/'init-seed42/COMPLETE.json'),'resolved':str(ROOT/'init-seed42/COMPLETE.json'),'bytes':0,'mtime_ns':0}}
 if cfg['seed']!=42 or cfg['checkpoint_every_macros']!=100:raise RuntimeError('Frozen seed/checkpoint cadence differs')
 write(ROOT/'init-config.json',cfg);(ROOT/'init-config.json').chmod(0o444)
 write(ROOT/'DEPLOYMENT.json',{'complete':True,'training_launched':False,'source_files':cfg['trainer_files'],
  'source_bytes':SOURCES,'original_config':old_cfg_pin,'baseline_source_bytes_verified':True,
  'bootstrap_source':pin(ROOT/'ops/freeze_fresh.py'),'old_future_budget_bytes':future_old,'initial_free_bytes':free})
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',USE_TORCH='1',USE_TF='0',USE_FLAX='0',
  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
 started=time.monotonic()
 with (ROOT/'fresh-init.log').open('x') as log:
  result=subprocess.run([PYTHON,'-B','-u',str(ROOT/'source/fresh_init.py'),'--config',str(ROOT/'init-config.json'),
   '--arm','JPAIR_P_BCE_RANK','--output',str(ROOT/'init-seed42')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 if result.returncode:raise RuntimeError('Independent stock fresh generation failed; preserve partial and log')
 # A separate CPU analysis process reads the old oracle only AFTER generation completed.
 analysis=r"""
import base64,json,os,sys,time
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
r=Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1');old=Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
sys.path.insert(0,str(r/'source'))
from bindings import Context,read_json,stat,verify,write_json
ctx=Context(r/'init-config.json','JPAIR_P_BCE_RANK')
done=read_json(r/'init-seed42/COMPLETE.json');manifest=read_json(r/'init-seed42/FRESH_MANIFEST.json')
assert done['complete'] and done['seed']==42 and not list((r/'init-seed42').glob('BLOCKED*'))
verify(done['files']);assert manifest['model_forward_calls']==0 and manifest['trainer_code']==ctx.code
assert manifest['no_project_adapter_head_optimizer_teacher_loaded'] and manifest['audit']['active_served_parameters']==616237056
import torch
from safetensors.torch import load_file
torch.set_num_threads(2);assert not torch.cuda.is_initialized()
new=load_file(str(r/'init-seed42/adapter_model.safetensors'),device='cpu')
assert len(new)==393 and all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in new.values())
a=[k for k in new if '.lora_A.' in k];b=[k for k in new if '.lora_B.' in k];h=[k for k in new if k.endswith('score.weight')]
assert len(a)==len(b)==196 and len(h)==1 and list(new[h[0]].shape)==[2,1024]
assert all(torch.count_nonzero(new[k])==0 for k in b)
original_cfg=read_json(old/'seed42.json');old_done_pin=original_cfg['fresh_init']['complete'];verify([old_done_pin])
old_done=read_json(old_done_pin['path']);verify(old_done['files'])
old_manifest=read_json(old/'init-seed42/FRESH_MANIFEST.json')
assert old_manifest['model_forward_calls']==0 and old_manifest['construction']==manifest['construction'] and old_manifest['stock_files']==manifest['stock_files']
assert old_manifest['trainer_code']=={p['path']:base64.b64encode(Path(p['path']).read_bytes()).decode() for p in original_cfg['trainer_files']}
oracle=load_file(str(old/'init-seed42/adapter_model.safetensors'),device='cpu')
assert set(new)==set(oracle) and all(v.shape==oracle[k].shape and v.dtype==oracle[k].dtype and torch.equal(v.contiguous().view(torch.uint8),oracle[k].contiguous().view(torch.uint8)) for k,v in new.items())
verify(done['files']+old_done['files']+[old_done_pin]);ctx.check();assert not torch.cuda.is_initialized()
cfg=ctx.cfg;cfg['fresh_init']['complete']=stat(r/'init-seed42/COMPLETE.json');write_json(r/'seed42.json',cfg);(r/'seed42.json').chmod(0o444)
final_ctx=Context(r/'seed42.json','JPAIR_P_BCE_RANK')
from fresh_init import validate_init
validate_init(final_ctx);final_ctx.check()
report={'complete':True,'independently_generated_before_old_oracle_load':True,'393_initial_tensors_bitwise_equal_old_v2':True,
 'actual_saved_fp32_tensor_keys':393,'lora_pairs':196,'B_exactly_zero':True,'head_shape':[2,1024],
 'active_served_parameters':616237056,'trainable_parameters':20187136,'model_forward_calls':0,'optimizer_steps':0,
 'old_oracle_complete':old_done_pin,'old_oracle_manifest':stat(old/'init-seed42/FRESH_MANIFEST.json'),
 'new_init_complete':stat(r/'init-seed42/COMPLETE.json'),'new_manifest':stat(r/'init-seed42/FRESH_MANIFEST.json'),
 'analysis_source':stat(r/'fresh-compare-analysis.py'),'bootstrap_source':stat(r/'ops/freeze_fresh.py'),'deployment':stat(r/'DEPLOYMENT.json'),
 'final_config':stat(r/'seed42.json'),'trainer_files':final_ctx.cfg['trainer_files'],'trainer_code':final_ctx.code,
 'construction':manifest['construction'],'old_project_trained_adapter_head_optimizer_teacher_loaded':False,
 'D_confirm_opened':False,'public_run':False,'training_launched':False,'CUDA_initialized':False}
write_json(r/'FRESH_PRIMARY_VERIFICATION.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('trainer_code','trainer_files')},indent=2))
"""
 with (ROOT/'fresh-compare-analysis.py').open('x') as f:f.write(analysis);f.flush();os.fsync(f.fileno())
 (ROOT/'fresh-compare-analysis.py').chmod(0o444)
 with (ROOT/'fresh-compare.log').open('x') as log:
  comparison=subprocess.run([PYTHON,'-B',str(ROOT/'fresh-compare-analysis.py')],env=env,stdout=log,stderr=subprocess.STDOUT,text=True,timeout=180)
 comparison_output=(ROOT/'fresh-compare.log').read_text()
 if comparison.returncode:raise RuntimeError('Independent fresh oracle comparison failed: '+comparison_output)
 verify([old_cfg_pin]+cfg['trainer_files'])
 print(comparison_output,end='')
except BaseException as error:
 write(ROOT/'BOOTSTRAP_BLOCKED.json',{'complete':False,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'partial_preserved':True,'training_launched':False})
 raise
'''


def main():
    sources = {name: base64.b64encode((LOCAL / 'source' / name).read_bytes()).decode() for name in NAMES}
    baseline = {p.name: base64.b64encode(p.read_bytes()).decode() for p in sorted((LOCAL / 'baseline').glob('*.py'))}
    script = ('SOURCES=' + repr(sources) + '\nBASELINE=' + repr(baseline) + '\nBOOTSTRAP=' +
              repr(base64.b64encode(Path(__file__).read_bytes()).decode()) + '\n' + REMOTE)
    with (LOCAL / 'BOOTSTRAP_EXECUTION.log').open('x') as handle:
        result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectionAttempts=1', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ClearAllForwardings=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', 'sh-10-gpu', 'python3 -B -'],
            input=script, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=850)
    output = (LOCAL / 'BOOTSTRAP_EXECUTION.log').read_text()
    print(output, end='')
    if result.returncode:
        raise RuntimeError(output)


if __name__ == '__main__':
    main()
