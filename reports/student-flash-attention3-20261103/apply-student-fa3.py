#!/usr/bin/env python3
"""Publish only the queued Student's explicit FA3 backend, with intact backups."""
from pathlib import Path
import json
import subprocess


def main():
    folder = Path(__file__).resolve().parent
    payload = {
        'trainer_before': (folder/'before/train-fresh-qwen-milestone.py').read_text(),
        'trainer_after': (folder/'after/train-fresh-qwen-milestone.py').read_text(),
        'config_before': (folder/'before/experiment-fresh-qwen-1m-20261103.json').read_text(),
        'config_after': (folder/'after/experiment-fresh-qwen-1m-20261103.json').read_text(),
        'helper': (folder/'student_flash_attention3.py').read_text(),
    }
    code = r'''
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
root=Path('/home/dev/assumeengage');run=root/'runs/distillation-fresh-qwen-1m-20261103'
trainer=root/'scripts/train-fresh-qwen-milestone.py';config=root/'config/experiment-fresh-qwen-1m-20261103.json';helper=root/'scripts/student_flash_attention3.py'
kernel=root/'cache/attention-kernels/flash-attn3-torch29-cu128'
ready=json.loads((kernel/'READY.json').read_text());assert ready.get('complete') is True and ready.get('native_backward_export') is True
assert trainer.read_text()==payload['trainer_before'],'Student source changed since capture; preserved'
assert config.read_text()==payload['config_before'],'Student configuration changed since capture; preserved'
before=json.loads(payload['config_before']);after=json.loads(payload['config_after'])
allowed={'attention','student_attention_implementation','student_attention_provider','student_attention_kernel_directory','student_attention_required'}
assert {k:v for k,v in before.items() if k not in allowed}=={k:v for k,v in after.items() if k not in allowed},'Non-attention experiment fields must not change'
for arm in before['arms']:
    assert not (run/arm/'OWNER.json').exists() and not (run/arm/'latest.json').exists(),'Student training has started; do not silently mix backends'
if helper.exists() and helper.read_text()!=payload['helper']:raise RuntimeError('Unfamiliar helper source preserved')
compile(payload['trainer_after'],str(trainer),'exec');compile(payload['helper'],str(helper),'exec')
def atomic(path,text):
    temporary=path.with_name(path.name+'.fa3.tmp');temporary.write_text(text);temporary.replace(path)
atomic(helper,payload['helper'])
# Required installation/integration imports only: no model construction, CUDA
# tensor, forward, backward, training update, benchmark or software test run.
installation_code=r"""import sys,json
sys.path.insert(0,'/home/dev/assumeengage/scripts')
from student_flash_attention3 import register_student_fa3,attention_record,BACKEND
register_student_fa3('/home/dev/assumeengage/cache/attention-kernels/flash-attn3-torch29-cu128',require_device=False)
from transformers.modeling_flash_attention_utils import _flash_fn,_flash_varlen_fn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
assert BACKEND in ALL_ATTENTION_FUNCTIONS and BACKEND in ALL_MASK_ATTENTION_FUNCTIONS
assert _flash_fn.__module__.startswith('_jev_student_hopper_fa3') and _flash_varlen_fn.__module__.startswith('_jev_student_hopper_fa3')
print(json.dumps({'registration':attention_record(),'dense_kernel_module':_flash_fn.__module__,'varlen_kernel_module':_flash_varlen_fn.__module__,'model_constructed':False,'gpu_calls_run':False}))
"""
check=subprocess.run([str(root/'envs/jev/bin/python'),'-B','-c',installation_code],cwd=root,text=True,capture_output=True,timeout=90)
if check.returncode:
    print(json.dumps({'student_fa3_import_failed':True,'trainer_and_config_not_modified':True}))
    print(check.stderr[-2500:]);raise SystemExit(check.returncode)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup=run/('student-attention-before-fa3-'+stamp);backup.mkdir()
(backup/trainer.name).write_text(payload['trainer_before']);(backup/config.name).write_text(payload['config_before'])
try:
    # Constructor first: an unlikely concurrent launch during the tiny publish
    # window fails closed on the old config, never trains through SDPA.
    atomic(trainer,payload['trainer_after']);atomic(config,payload['config_after'])
except BaseException:
    atomic(trainer,payload['trainer_before']);atomic(config,payload['config_before']);raise
receipt={'changed_utc':datetime.now(timezone.utc).isoformat(),'scope':'Student training only','attention':'flash_attention_3','attention_implementation':'flash_attn3_hopper','provider':'kernels-community/flash-attn3','sdpa_fallback':False,'teacher_modified':False,'baseline_evaluation_modified':False,'processes_signalled_or_restarted':False,'core_python_packages_modified':False,'data_and_optimizer_fields_unchanged':True,'backup':str(backup),'native_extension_import_verified':True,'actual_student_training_started':False,'actual_training_forward_backward_verified':False}
atomic(run/'STUDENT_FLASH_ATTENTION3_REQUIRED.json',json.dumps(receipt,indent=2)+'\n')
print(check.stdout,end='');print(json.dumps(receipt))
'''
    command = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=yes',
               '-o','ClearAllForwardings=yes','sh-10-gpu','python3 -B -']
    result = subprocess.run(command, input='payload = '+repr(payload)+'\n'+code,
                            capture_output=True, text=True, timeout=120)
    (folder/'deployment.log').write_text(result.stdout)
    print(result.stdout,end='')
    if result.returncode:
        print('FA3 deployment not completed; exit',result.returncode)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
