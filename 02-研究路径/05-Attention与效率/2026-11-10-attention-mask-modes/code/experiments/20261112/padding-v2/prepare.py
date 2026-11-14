"""CPU-only source admission and original Suite's mandatory verification."""
import sys
import os
from pathlib import Path
from pf_common import ROOT,OLD,MODEL,KIT,SUITE,read,atomic,metadata,require,utc


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Prepare hides CUDA')
    require(not (ROOT/'SOURCE_BINDING.json').exists(),'Existing prepare artifacts preserved')
    reference=Path('/tmp/opencode/jev-06b-research-20261112-r1-eval-recovery')
    sys.path[:0]=[str(reference/'code'),str(KIT)]
    import eval_common
    old_ready=eval_common.verify_ready()
    done=eval_common.verify_training_complete()
    from decision_index.suite.io import Suite
    check=Suite(SUITE,'0.3').verify(strict=True)
    require(check['match'] and check.get('exclusions_match',True),'Mandatory Suite verification failed')
    files=[metadata(p) for p in sorted(MODEL.iterdir()) if p.is_file()]
    files += old_ready['files']+eval_common.verify_source()['files']
    files += [metadata(reference/name) for name in ('DATA_READY.json','SOURCE_BINDING.json','CODE_BINDING.json')]
    files += [metadata(p) for p in sorted((reference/'code').rglob('*.py'))]
    files += [metadata(p) for p in sorted((OLD/'code').glob('*.py'))]
    files += [metadata(Path('/home/dev/assumeengage/scripts')/name) for name in
              ('student_flash_attention3.py','student_flash_attention3_repro.py','pair_training_runtime.py','group_training_runtime.py')]
    files += [metadata('/tmp/opencode/jev-06b-research-20261112-r1-bridge/code/train_bridge.py'),
              metadata('/tmp/opencode/jev-06b-research-20261112-r1-bridge/train/config.json')]
    unique={v['path']:v for v in files}
    atomic(ROOT/'SOURCE_BINDING.json',{'files':[unique[k] for k in sorted(unique)],'created_utc':utc(),
        'scope':'old exposed model, numerical and efficiency mechanism only','training_receipt':done['checkpoint']})
    atomic(ROOT/'DATA_READY.json',{'complete':True,'requests':140620,'scored_requests':140178,
        'shards':old_ready['shards'],'official_verification':check,'model':str(MODEL)})
    print('PF CPU admission complete; mandatory Suite verification passed',flush=True)


if __name__=='__main__':main()
