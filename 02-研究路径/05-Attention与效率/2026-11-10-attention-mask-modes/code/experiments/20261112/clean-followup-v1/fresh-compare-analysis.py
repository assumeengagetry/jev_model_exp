
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
