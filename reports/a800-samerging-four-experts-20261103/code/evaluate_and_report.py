#!/usr/bin/env python3
"""Reuse the native evaluator and compare like-for-like public and frozen-subset scores."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

from run_samerging import ROOT, WORK, SOURCE, NAME, atomic_json, status, emit

AXES = ['knowledge','language','retrieval','tools','arts']


def evaluator():
    path = WORK/'scripts/run-formal-evaluation.py'
    source = path.read_text().replace("HOME = Path('/home/dev/assumeengage')",f'HOME = Path({str(WORK)!r})')
    module = types.ModuleType('samerging_native_evaluation')
    module.__file__ = str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    module.ROOT = ROOT/'outputs'
    native = module.benchmark_summary

    def scored_fields(suite,results,*args,**kwargs):
        rows = kwargs.get('rows')
        if rows is not None:
            scored = []
            for row in rows:
                excluded = set(row.get('scoring',{}).get('unscored_fields',[]))
                if excluded:
                    row = dict(row)
                    row['questions'] = {k:v for k,v in row['questions'].items() if k not in excluded}
                    row['expected'] = {k:v for k,v in row['expected'].items() if k not in excluded}
                scored.append(row)
            kwargs['rows'] = scored
        return native(suite,results,*args,**kwargs)

    module.benchmark_summary = scored_fields
    margins = module.margins_for_request

    def complete_candidates(*args,**kwargs):
        value = margins(*args,**kwargs)
        if value[0] is None:
            raise RuntimeError('Public evaluation cannot silently lose candidate coverage')
        return value

    module.margins_for_request = complete_candidates
    return module


def subset(module, full_folder):
    path = SOURCE/'pilots/idea-100k-20261102/selection/evaluation.json'
    selection = json.loads(path.read_text())
    run_ids = set(selection['run_ids'])
    module.ROOT = ROOT/'outputs/subset'
    folder = module.ROOT/'evaluation'/NAME
    folder.mkdir(parents=True,exist_ok=True)
    if module.done(folder/'complete.json'):
        return json.loads((folder/'scores.json').read_text())
    with (full_folder/'results.jsonl').open() as source,(folder/'part-0.jsonl').open('w') as target:
        for line in source:
            if json.loads(line)['run_id'] in run_ids:target.write(line)
    original = module.Suite

    class FrozenSuite:
        def __init__(self,*args,**kwargs):self.base=original(*args,**kwargs)
        def __getattr__(self,key):return getattr(self.base,key)
        def rows(self,*args,**kwargs):
            for row in self.base.rows(*args,**kwargs):
                if row['_evaluation']['run_id'] in run_ids:yield row

    module.Suite = FrozenSuite
    scores = module.score_model(NAME,folder)
    scores.update(scope='pilot_subset',official_full_public_run=False,private_full_score=False)
    atomic_json(folder/'scores.json',scores)
    completed = json.loads((folder/'complete.json').read_text())
    completed.pop('public_score',None)
    completed.update(scope='pilot_subset',pilot_score=scores['index'],selected_cases=selection['selected_cases'])
    atomic_json(folder/'complete.json',completed)
    return scores


def report(full,selected):
    folder = ROOT/'report'
    folder.mkdir(exist_ok=True)
    records = {}
    for name in ['MIXED_ALL','MERGE_AVERAGE','MERGE_TIES','MERGE_SLERP_HIERARCHICAL']:
        records[name] = json.loads((ROOT/'baselines'/name/'scores.json').read_text())
    records[NAME] = full
    header = ['model',*AXES,'public_index','coverage']
    with (folder/'public-comparison.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(header)
        for name,score in records.items():
            values={a['id']:100*a['skill'] for a in score['areas']}
            writer.writerow([name,*(values[a] for a in AXES),score['index'],score['coverage']])
    deltas = {name:dict(index=full['index']-score['index'],
                      areas={a:100*(next(v['skill'] for v in full['areas'] if v['id']==a)-
                                    next(v['skill'] for v in score['areas'] if v['id']==a)) for a in AXES})
              for name,score in records.items() if name != NAME}
    atomic_json(folder/'differences.json',deltas)
    merged = json.loads((ROOT/'MERGE_COMPLETE.json').read_text())
    lines = ['# 四专家 SAMerging 实测', '',
       f"**SAMerging 公共总分：{full['index']:.2f}；相对 MIXED_ALL：{deltas['MIXED_ALL']['index']:+.2f}；相对 average：{deltas['MERGE_AVERAGE']['index']:+.2f}；相对 SLERP：{deltas['MERGE_SLERP_HIERARCHICAL']['index']:+.2f}。**", '',
       '## 完整公共37项对比', '',
       '| 模型 | Knowledge | Language | Retrieval | Tools | Arts | 公共总分 | coverage |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,score in records.items():
        values={a['id']:100*a['skill'] for a in score['areas']}
        lines.append('| '+name+' | '+' | '.join(f'{values[a]:.2f}' for a in AXES)+f" | {score['index']:.2f} | {100*score['coverage']:.2f}% |")
    lines += ['', '使用同一重建公共0.3评测的37项、110,087个请求和全部候选；不是私测 Full。原基线直接复用已完成预测/评分，不重跑推理。', '',
       '## 与当前 KD 的冻结子集对照', '',
       '以下只比较同一冻结子集分数，不能把它当成相同训练量/监督来源的因果对照；SAMerging学习系数，KD训练LoRA和head。', '',
       '| 模型 | 冻结子集 Index |', '|---|---:|']
    for name in ['BASELINE','KD_HARD','KD_SOFT','KD_HYBRID']:
        score=json.loads((ROOT/'baselines/kd-1m'/name/'scores.json').read_text())
        lines.append(f"| {name} | {score['index']:.2f} |")
    lines.append(f"| {NAME} | {selected['index']:.2f} |")
    lines += ['', '## 方法与实际配置', '',
       '- Knowledge/Retrieval/Tools/Arts四专家；以它们实际共同的pair-classifier初始化为W0。每个有效权重tensor学习每专家一个系数，包含共享二分类head。不是直接平均LoRA的A/B矩阵。',
       '- 每域1,600个无标签输入；不读原始target、不用评测题校准。KL(expert||merged)，T1，任务等权。',
       f"- {merged['coefficient_count']}个可训练系数，600步，每步每域batch16；init0.5，不clamp；官方SAM实现，SGD base，lr1e-3，momentum0.99，weight decay5e-4，rho0.07，adaptive=True。", 
       f"- 校准总耗时{merged['elapsed_seconds']/60:.1f}分钟。FP32合并运算、BF16 forward及单模型导出；推理不保留多专家或额外adapter。", '',
       '## 与论文/示例的差异', '',
       '- 这是SAMerging在现有Qwen3二分类专家上的适配实测，不是复现论文的GPT2/GLUE或CLIP数值。论文附录写Adam，但官方NLP代码实际用SGD；本次跟随代码并明确披露。',
       '- 官方GPT2示例分别读取两轮shuffle batch，预计算logits与后续输入可能错位。本次按固定input IDs准确配对。',
       '- 官方GPT2示例在SAM first_step后没有刷新缓存的merged weights。本次每次forward都重建，包括perturbed pass，确保真正执行SAM。',
       '- 所有专家的label space相同，因此合并出一个共享head；不使用GPT2/GLUE示例的每task独立head。', '',
       '官方代码：https://github.com/arshandalili/SAMerging/tree/a194e8ed8ab790dcae9c02a8d1467e00e86b53e5',
       '论文：https://arxiv.org/abs/2512.21288', '']
    (folder/'REPORT.md').write_text('\n'.join(lines))
    completed=dict(complete=True,public_index=full['index'],subset_index=selected['index'],
                   coverage=full['coverage'],differences=deltas,report=str(folder/'REPORT.md'),unix_time=time.time())
    atomic_json(ROOT/'COMPLETE.json',completed)
    emit('samerging_experiment_complete',**completed)


def main():
    import pyarrow as pa
    pa.set_cpu_count(2);pa.set_io_thread_count(2)
    module=evaluator()
    folder=module.ROOT/'evaluation'/NAME
    if not module.done(folder/'complete.json'):
        workers=4
        pending=[rank for rank in range(workers) if not module.done(folder/f'worker-{rank}-complete.json')]
        devices=[v for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v]
        if not devices:raise RuntimeError('Evaluation step has no allocated GPUs')
        active={}
        status('full_public_evaluation',model=NAME,workers=workers,pending_shards=pending,
               devices=devices,scheduling='one top-level Slurm step, no nested srun')
        atomic_json(ROOT/'evaluation-parallel-allocation.json',dict(job_id=os.environ.get('SLURM_JOB_ID'),
                    step_id=os.environ.get('SLURM_STEP_ID'),devices=devices,pending_shards=pending,
                    existing_complete_shards=[rank for rank in range(workers) if rank not in pending]))
        try:
            while pending or active:
                for device,job in list(active.items()):
                    code=job['process'].poll()
                    if code is None:continue
                    job['handle'].close()
                    if code or not module.done(folder/f"worker-{job['rank']}-complete.json"):
                        raise RuntimeError(f"SAMerging scorer {job['rank']} exited {code}; shard outputs preserved")
                    del active[device]
                for device in devices:
                    if device in active or not pending:continue
                    rank=pending.pop(0)
                    env=os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                    handle=(ROOT/'logs'/f'evaluation-worker-{rank}.log').open('a',buffering=1)
                    process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(rank)],
                                             env=env,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    active[device]=dict(rank=rank,process=process,handle=handle)
                atomic_json(ROOT/'evaluation-workers.json',dict(pending_shards=pending,
                  active={device:dict(rank=j['rank'],pid=j['process'].pid) for device,j in active.items()},updated_unix=time.time()))
                if pending or active:time.sleep(2)
        finally:
            for job in active.values():
                if job['process'].poll() is None:job['process'].send_signal(signal.SIGINT)
            for job in active.values():
                job['process'].wait(timeout=90)
                job['handle'].close()
        module.score_model(NAME,folder)
    completion=json.loads((folder/'complete.json').read_text())
    full=json.loads((folder/'scores.json').read_text())
    if completion['requests'] != 110087 or completion['benchmarks'] != 37 or full['coverage'] != 1.0:
        raise RuntimeError('SAMerging public evaluation must match prior full public scope and coverage')
    status('scoring_frozen_subset_from_saved_predictions')
    selected=subset(evaluator(),folder)
    if selected['coverage'] != 1.0:raise RuntimeError('Frozen subset coverage is incomplete')
    report(full,selected)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    if args.worker is not None:
        import pyarrow as pa
        pa.set_cpu_count(2);pa.set_io_thread_count(2)
        module=evaluator()
        if not module.done(ROOT/'outputs/evaluation'/NAME/f'worker-{args.worker}-complete.json'):
            module.worker(argparse.Namespace(worker=args.worker,workers=4,model=NAME,
                          checkpoint=str(ROOT/'outputs'/NAME/'final')))
    else:main()
