#!/usr/bin/env python3
"""One full-coverage hierarchical SLERP supplement; never interrupts other jobs."""
import argparse
import ctypes
import csv
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

HOME = Path('/home/dev/assumeengage')
PRIOR = HOME / 'runs/five-specialists-20261030'
ROOT = HOME / 'runs/four-specialists-slerp-20261102'
NAME = 'MERGE_SLERP_HIERARCHICAL'
SPECIALISTS = ['SPECIALIST_KNOWLEDGE', 'SPECIALIST_RETRIEVAL', 'SPECIALIST_TOOLS', 'SPECIALIST_ARTS']
AXES = ['knowledge', 'language', 'retrieval', 'tools', 'arts']
WORKERS = 3


def read(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


def done(path):
    return read(path).get('complete') is True


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def status(stage, **extra):
    value = {'stage': stage, 'updated_unix': time.time(), 'owner': 'four-specialist-slerp-supplement',
             'model': NAME, 'new_training_tokens': 0, **extra}
    atomic(ROOT / 'PIPELINE.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation_module():
    module = load_module('slerp_formal_evaluation', HOME / 'scripts/run-formal-evaluation.py')
    module.ROOT = ROOT
    original = module.margins_for_request
    def full_coverage(*args, **kwargs):
        result = original(*args, **kwargs)
        if result[0] is None:
            raise RuntimeError('Original suite was fully supported; SLERP must not silently reduce coverage after memory pressure')
        return result
    module.margins_for_request = full_coverage
    return module


def slerp(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError('SLERP endpoint shape/dtype mismatch')
    if torch.equal(a, b):
        return a, {'mode': 'identical', 'cosine': 1.0}
    na, nb = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    if not math.isfinite(na) or not math.isfinite(nb):
        raise RuntimeError('Nonfinite source weights; no synthetic replacement permitted')
    if na <= 1e-8 or nb <= 1e-8:
        return .5 * a + .5 * b, {'mode': 'linear_zero_norm', 'cosine': None}
    dot = float(((a / na) * (b / nb)).sum().clamp(-1, 1))
    if abs(dot) > .9995:
        return .5 * a + .5 * b, {'mode': 'linear_collinear', 'cosine': dot}
    angle = math.acos(dot)
    coefficient = math.sin(.5 * angle) / math.sin(angle)
    return coefficient * a + coefficient * b, {'mode': 'spherical', 'cosine': dot}


@torch.no_grad()
def merge():
    marker = ROOT / 'MERGE_COMPLETE.json'
    if done(marker): return
    cfg = read(HOME / 'config/experiment-five-specialists.json')
    if [n for n in cfg['models'] if n.startswith('SPECIALIST_')] != SPECIALISTS:
        raise RuntimeError('Current specialist set/order differs from the approved four-expert tree')
    if not done(PRIOR / 'EVALUATION_COMPLETE.json'):
        raise RuntimeError('Previous baseline round is incomplete')
    for name in SPECIALISTS:
        if not done(PRIOR / name / 'complete.json'):
            raise RuntimeError('Missing completed specialist: ' + name)
    native = load_module('slerp_existing_weight_loader', HOME / 'scripts/merge-formal-models.py')
    torch.set_num_threads(8)
    started = time.monotonic()
    states = [native.effective_state(PRIOR / name / 'final') for name in SPECIALISTS]
    keys = states[0].keys()
    if 'score.weight' not in keys or any(state.keys() != keys for state in states):
        raise RuntimeError('Effective model tensors or classification head differ')
    output, statistics = {}, {}
    original_average = PRIOR / 'merges/MERGE_AVERAGE/final/model.safetensors'
    changed_elements, total_elements, largest_difference = 0, 0, 0.0
    with safe_open(str(original_average), framework='pt', device='cpu') as average:
        for index, key in enumerate(sorted(keys)):
            values = [state[key] for state in states]
            if any(v.shape != values[0].shape for v in values):
                raise RuntimeError('Tensor shape mismatch: ' + key)
            if not values[0].is_floating_point():
                if not all(torch.equal(values[0], value) for value in values):
                    raise RuntimeError('Nonfloating tensors differ: ' + key)
                final, details = values[0], {'mode': 'shared_nonfloating'}
            else:
                a, info_a = slerp(values[0], values[1])
                b, info_b = slerp(values[2], values[3])
                final, info_final = slerp(a, b)
                if not bool(torch.isfinite(final).all()):
                    raise RuntimeError('Nonfinite merged tensor: ' + key)
                details = {'knowledge_retrieval': info_a, 'tools_arts': info_b, 'final': info_final}
                final = final.to(torch.bfloat16).contiguous()
            output[key] = final
            old = average.get_tensor(key)
            if old.shape != final.shape:
                raise RuntimeError('Stored average model has incompatible tensor: ' + key)
            difference = final.float() - old.float()
            changed_elements += int(torch.count_nonzero(difference))
            total_elements += final.numel()
            largest_difference = max(largest_difference, float(difference.abs().max()))
            statistics[key] = details
            if index % 30 == 0:
                print(json.dumps({'event': 'slerp_merge_progress', 'tensor': key,
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
    del states
    final_dir = ROOT / NAME / 'final'; final_dir.mkdir(parents=True, exist_ok=True)
    temporary = final_dir / 'model.safetensors.writing'
    save_file(output, str(temporary), metadata={'format': 'pt'})
    temporary.replace(final_dir / 'model.safetensors')
    common = HOME / cfg['common_initialization']
    model_config = AutoConfig.from_pretrained(common, local_files_only=True)
    model_config.architectures = ['Qwen3ForSequenceClassification']; model_config.torch_dtype = torch.bfloat16
    model_config.use_cache = False; model_config.save_pretrained(final_dir)
    AutoTokenizer.from_pretrained(common, local_files_only=True).save_pretrained(final_dir)
    atomic(ROOT / 'tensor-interpolation-statistics.json', statistics)
    counters = {}
    for value in statistics.values():
        for stage in ('knowledge_retrieval', 'tools_arts', 'final'):
            if stage in value:
                mode = value[stage]['mode']; counters[mode] = counters.get(mode, 0) + 1
    record = {'complete': True, 'model': NAME, 'method': 'hierarchical_slerp_per_tensor',
              'specialists': SPECIALISTS, 'tree': [['knowledge', 'retrieval'], ['tools', 'arts']],
              't_each_stage': .5, 'dot_threshold': .9995, 'classification_head_included': True,
              'compute_dtype': 'float32', 'storage_dtype': 'bfloat16', 'new_training_tokens': 0,
              'order_independent_multislerp': False, 'interpolation_modes': counters,
              'bf16_comparison_to_average': {'different_elements': changed_elements, 'total_elements': total_elements,
                                            'maximum_absolute_difference': largest_difference},
              'elapsed_seconds': time.monotonic() - started, 'checkpoint': str(final_dir),
              'artifacts': [str(final_dir), str(ROOT / 'tensor-interpolation-statistics.json')]}
    atomic(ROOT / NAME / 'complete.json', record); atomic(marker, record)


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=15, check=True)
    ids = {line.split(',')[1].strip(): int(line.split(',')[0]) for line in result.stdout.splitlines() if line.strip()}
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=15, check=True)
    out = []
    for line in result.stdout.splitlines():
        if not line.strip(): continue
        pid, uuid = line.split(',')[:2]; out.append((int(pid.strip()), ids[uuid.strip()]))
    return out


def process_alive(child):
    if child['process'].poll() is not None: return False
    return True


def yield_worker(child):
    # Only this Popen child is signalled. Its native output context closes/flushed
    # valid JSONL lines; original tail recovery handles an interrupted last line.
    if child['process'].poll() is None:
        child['process'].send_signal(signal.SIGINT)
        try: child['process'].wait(timeout=60)
        except subprocess.TimeoutExpired:
            child['process'].terminate(); child['process'].wait(timeout=30)
    child['handle'].close()


def evaluate(gpu_pool):
    folder = ROOT / 'evaluation' / NAME; folder.mkdir(parents=True, exist_ok=True)
    if done(folder / 'complete.json'): return
    # A controller owns its children. If it dies, each child receives SIGINT;
    # wait for any previous generation to finish flushing before reopening files.
    old = read(ROOT / 'WORKERS.json')
    for record in old.values():
        path = Path('/proc') / str(record['pid'])
        if not path.exists(): continue
        fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] != 'Z' and fields[19] == record['starttime']:
            raise RuntimeError('Previous SLERP worker is still alive; wait before resuming, no duplicate launched')
    running = {}
    try:
        while not all(done(folder / f'worker-{rank}-complete.json') for rank in range(WORKERS)):
            apps = gpu_processes()
            ours = {child['process'].pid for child in running.values()}
            busy = {gpu for pid, gpu in apps if pid not in ours}
            for rank, child in list(running.items()):
                if child['gpu'] in busy:
                    yield_worker(child); del running[rank]
                elif not process_alive(child):
                    code = child['process'].returncode; child['handle'].close(); del running[rank]
                    if code or not done(folder / f'worker-{rank}-complete.json'):
                        raise RuntimeError(f'SLERP worker {rank} exited {code}; see its preserved log')
            used = {child['gpu'] for child in running.values()}
            free = [gpu for gpu in gpu_pool if gpu not in busy and gpu not in used]
            for rank in range(WORKERS):
                if rank in running or done(folder / f'worker-{rank}-complete.json') or not free: continue
                gpu = free.pop(0)
                handle = (ROOT / f'evaluate-worker-{rank}.log').open('a', buffering=1)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           SLERP_PARENT_PID=str(os.getpid()))
                child = subprocess.Popen([sys.executable, '-B', '-u', __file__, '--worker', str(rank),
                                          '--workers', str(WORKERS), '--model', NAME,
                                          '--checkpoint', str(ROOT / NAME / 'final')],
                                          cwd=HOME, env=env, stdin=subprocess.DEVNULL,
                                          stdout=handle, stderr=subprocess.STDOUT)
                fields = Path('/proc', str(child.pid), 'stat').read_text().rsplit(') ', 1)[1].split()
                running[rank] = {'process': child, 'handle': handle, 'gpu': gpu, 'starttime': fields[19]}
            record = {str(rank): {'pid': child['process'].pid, 'gpu': child['gpu'], 'starttime': child['starttime']}
                      for rank, child in running.items()}
            atomic(ROOT / 'WORKERS.json', record)
            status('running' if running else 'waiting_gpu', phase='public_evaluation', workers=record,
                   yielding_to_other_jobs=True, total_shards=WORKERS)
            time.sleep(3 if running else 20)
    finally:
        for child in running.values(): yield_worker(child)
    module = evaluation_module()
    scores = module.score_model(NAME, folder)
    completion = read(folder / 'complete.json')
    if completion['coverage'] != 1.0 or completion.get('counts') != {'ok': completion['requests']}:
        raise RuntimeError('SLERP requires the same complete supported public suite; incomplete coverage preserved for diagnosis')
    completion['latency_scope'] = 'Up to three independent workers on otherwise idle GPUs; resumable priority yielding; not isolated latency-v1'
    atomic(folder / 'complete.json', completion)


def report():
    module = evaluation_module()
    cfg = read(HOME / 'config/experiment-five-specialists.json')
    names = cfg['models'] + [record['model'] for record in read(PRIOR / 'MERGES_COMPLETE.json')['models']]
    records = [(name, read(PRIOR / 'evaluation' / name / 'scores.json')) for name in names]
    records.append((NAME, read(ROOT / 'evaluation' / NAME / 'scores.json')))
    if len(records) != 11 or any(not scores.get('areas') for _, scores in records):
        raise RuntimeError('Original ten-condition score records are incomplete; not overwritten')
    folder = ROOT / 'report'; folder.mkdir(exist_ok=True)
    selected = [(name, scores) for name, scores in records if name in ('MIXED_ALL', 'MERGE_AVERAGE', 'MERGE_TIES', NAME)]
    module.radar(selected, folder / 'slerp-vs-baselines.svg', 'Hierarchical SLERP vs fixed baselines')
    mixed = dict(records)['MIXED_ALL']; average = dict(records)['MERGE_AVERAGE']; current = dict(records)[NAME]
    changes = {'minus_mixed': current['index'] - mixed['index'], 'minus_average': current['index'] - average['index']}
    atomic(folder / 'differences.json', changes)
    header = ['model', *AXES, 'public_total', 'coverage']
    with (folder / 'eleven-condition-results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(header)
        for name, scores in records:
            values = {a['id']: 100*a['skill'] for a in scores['areas']}
            writer.writerow([name, *(f'{values[a]:.2f}' for a in AXES), f'{scores["index"]:.2f}', f'{scores["coverage"]:.2f}'])
    lines = ['# 四专家分层 SLERP：补充正式评测', '',
             f'**SLERP 公共总分：{current["index"]:.2f}；相对 MIXED_ALL：{changes["minus_mixed"]:+.2f}；相对等权平均：{changes["minus_average"]:+.2f}。**', '',
             '固定树：SLERP(SLERP(Knowledge, Retrieval, 0.5), SLERP(Tools, Arts, 0.5), 0.5)。',
             '这是逐 tensor 的分层 SLERP，结果依赖该固定分组；不是顺序无关的 Multi-SLERP。包含分类头，FP32 计算、最终 BF16 保存，无新增训练。', '',
             '| 模型 | Knowledge | Language | Retrieval | Tools | Arts | 公共总分 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, scores in records:
        values = {a['id']: 100*a['skill'] for a in scores['areas']}
        lines.append('| '+name+' | '+' | '.join(f'{values[a]:.2f}' for a in AXES)+f' | {scores["index"]:.2f} |')
    details = read(ROOT / 'MERGE_COMPLETE.json')
    lines += ['', '## 球面插值与线性回退', '',
              '近共线或零范数采用标准线性回退；以下统计避免把接近等权平均的模型误报为完全不同的方法效果。', '',
              '```json', json.dumps({'modes': details['interpolation_modes'], 'comparison_to_average': details['bf16_comparison_to_average']}, indent=2), '```', '',
              '所有模型使用相同37项公共五维评分，原十个条件直接复用已完成结果；只对新 SLERP 模型重新推理评分，不是私测 Full。', '',
              '![与基线对照](slerp-vs-baselines.svg)']
    (folder / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic(ROOT / 'COMPLETE.json', {'complete': True, 'model': NAME, 'public_score': current['index'],
           'benchmarks': 37, 'requests': read(ROOT/'evaluation'/NAME/'complete.json')['requests'],
           'artifacts': [str(folder), str(ROOT/'evaluation'/NAME), str(ROOT/'MERGE_COMPLETE.json')]})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', type=int)
    parser.add_argument('--workers', type=int, default=WORKERS); parser.add_argument('--model'); parser.add_argument('--checkpoint')
    parser.add_argument('--gpu-pool', nargs='+', type=int, default=[1, 2, 3, 0])
    parser.add_argument('--merge-only', action='store_true'); args = parser.parse_args()
    if args.worker is not None:
        parent = int(os.environ['SLERP_PARENT_PID'])
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGINT, 0, 0, 0) != 0:
            raise RuntimeError('Could not install owned-worker parent-death cleanup')
        if os.getppid() != parent: raise RuntimeError('SLERP owner exited before worker initialization')
        with (ROOT / f'worker-{args.worker}.lock').open('a+') as worker_lock:
            try: fcntl.flock(worker_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('An earlier SLERP shard still owns its output; no duplicate GPU work started')
            evaluation_module().worker(args)
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'EXPERIMENT.lock').open('a+') as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing SLERP owner active; no duplicate launched.', flush=True); return
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        fields = Path('/proc', str(os.getpid()), 'stat').read_text().rsplit(') ', 1)[1].split()
        atomic(ROOT/'OWNER.json', {'pid': os.getpid(), 'starttime': fields[19], 'script': str(Path(__file__).resolve())})
        if done(ROOT/'COMPLETE.json'): return
        status('running', phase='cpu_effective_weight_slerp'); merge()
        if args.merge_only:
            status('merge_complete', phase='ready_for_public_evaluation'); return
        evaluate(args.gpu_pool); report(); status('complete', phase='report_ready')


if __name__ == '__main__':
    try: main()
    except Exception as error:
        if ROOT.exists() and '--worker' not in sys.argv: status('blocked', error=str(error))
        traceback.print_exc(); raise
