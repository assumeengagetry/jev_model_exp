#!/usr/bin/env python3
"""Resume four independent GPU scorers for the eleven approved public conditions."""
import argparse
import collections
import csv
import gc
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

HOME = Path('/home/dev/assumeengage')
CFG = json.loads((HOME / 'config/experiment-five-specialists.json').read_text())
ROOT = HOME / CFG['run_root']
SUITE_PATH = HOME / 'datasets/evaluation-only/decision-index-suite-0.3'
CACHE = SUITE_PATH / 'candidate-cache'
sys.path.insert(0, str(HOME / 'src/decision-index-main'))
from decision_index.scoring import added, index02
from decision_index.scoring.report import benchmark_summary, load_results
from decision_index.suite.io import Suite


def done(path):
    try:
        return json.loads(path.read_text()).get('complete') is True
    except (OSError, ValueError):
        return False


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class EvaluationTokens:
    def __init__(self):
        folder = CACHE / 'tokenized'
        self.ids = {role: np.memmap(folder / (role + '_ids.bin'), dtype=np.int32, mode='r')
                    for role in ('prefix', 'suffix')}
        self.offsets = {role: np.memmap(folder / (role + '_offsets.bin'), dtype=np.uint64, mode='r')
                        for role in ('prefix', 'suffix')}
        self.lengths = {role: np.diff(value).astype(np.int64) for role, value in self.offsets.items()}

    def get(self, role, index):
        offset = self.offsets[role]
        return self.ids[role][int(offset[index]):int(offset[index + 1])]


def prior_results(path):
    seen = set()
    if not path.exists():
        return seen
    with path.open('rb+') as stream:
        valid_end = 0
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                value = json.loads(line)
            except ValueError:
                if stream.read().strip():
                    raise RuntimeError('Non-tail damage in owned evaluation output: ' + str(path))
                stream.truncate(valid_end)
                break
            if value['status'] in ('ok', 'unsupported'):
                seen.add(value['run_id'])
            valid_end = stream.tell()
    return seen


def load_classifier(checkpoint):
    common = HOME / CFG['common_initialization']
    tokenizer = AutoTokenizer.from_pretrained(common, local_files_only=True)
    if (checkpoint / 'adapter_config.json').exists():
        config = AutoConfig.from_pretrained(common, local_files_only=True)
        base = AutoModelForSequenceClassification.from_pretrained(
            HOME / CFG['base_model'], config=config, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True)
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    model.config.use_cache = False
    model.to('cuda:0').eval()
    return model, tokenizer


def padded_inputs(tokens, states, candidates, chosen, pad, eos):
    ids = [np.concatenate((tokens.get('prefix', int(states[i])), tokens.get('suffix', int(candidates[i])), [eos]))
           for i in chosen]
    width = (max(map(len, ids)) + 7) // 8 * 8
    x = torch.full((len(ids), width), pad, dtype=torch.long, pin_memory=True)
    mask = torch.zeros((len(ids), width), dtype=torch.bool, pin_memory=True)
    for row, values in enumerate(ids):
        x[row, :len(values)] = torch.from_numpy(values.astype(np.int64, copy=False))
        mask[row, :len(values)] = True
    return {'input_ids': x.to('cuda:0', non_blocking=True),
            'attention_mask': mask.to('cuda:0', non_blocking=True)}


@torch.inference_mode()
def margins_for_request(model, tokenizer, tokens, states, candidates, token_budget):
    lengths = tokens.lengths['prefix'][states] + tokens.lengths['suffix'][candidates] + 1
    limit = int(model.config.max_position_embeddings)
    if np.any(lengths > limit):
        return None, token_budget, int(lengths.sum())
    order = np.argsort(lengths, kind='stable')
    scores = np.empty(len(states), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and end - start < 32:
            width = (int(lengths[order[end]]) + 7) // 8 * 8
            if width * (end - start + 1) > token_budget:
                break
            end += 1
        chosen = order[start:end]
        inputs = output = None
        try:
            inputs = padded_inputs(tokens, states, candidates, chosen,
                                   tokenizer.pad_token_id, tokenizer.eos_token_id)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(**inputs).logits
            values = (output[:, 1].float() - output[:, 0].float()).cpu().numpy()
            if not np.all(np.isfinite(values)):
                raise RuntimeError('Nonfinite classifier margins during formal inference')
            scores[chosen] = values
            start = end
        except torch.cuda.OutOfMemoryError:
            inputs = output = None
            gc.collect()
            torch.cuda.empty_cache()
            if len(chosen) == 1:
                return None, token_budget, int(lengths.sum())
            token_budget = max(1024, token_budget // 2)
        finally:
            inputs = output = None
    return scores, token_budget, int(lengths.sum())


def answers_for_request(table, start, end, scores):
    fields = collections.OrderedDict()
    for offset, index in enumerate(range(start, end)):
        field = table['field'][index]
        if field not in fields:
            fields[field] = {'kind': table['kind'][index], 'keys': [], 'scores': []}
        fields[field]['keys'].append(table['option_key'][index])
        fields[field]['scores'].append(scores[offset])
    answers = {}
    for field, values in fields.items():
        array = np.asarray(values['scores'], dtype=np.float64)
        array = np.exp(array - array.max())
        array /= array.sum()
        probabilities = {key: float(p) for key, p in zip(values['keys'], array)}
        if values['kind'] == 'choice':
            answers[field] = {'choice': max(probabilities, key=probabilities.get),
                              'probabilities': probabilities}
        else:
            positive = [key for key in probabilities if str(key).lower() in ('true', 'yes', '1')]
            if len(positive) != 1:
                raise RuntimeError('Unrecognized noul semantic keys: ' + repr(list(probabilities)))
            yes = probabilities[positive[0]]
            answers[field] = {'noul': yes, 'probabilities': {'yes': yes, 'no': 1 - yes}}
    return answers


def worker(args):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, tokenizer = load_classifier(Path(args.checkpoint))
    tokens = EvaluationTokens()
    table = pq.read_table(CACHE / 'normalized/field-options.parquet').to_pydict()
    bounds = {}
    ids = table['run_id']
    begin = 0
    for index in range(1, len(ids) + 1):
        if index == len(ids) or ids[index] != ids[begin]:
            bounds[ids[begin]] = (begin, index)
            begin = index
    allowed = {n for area in index02.spec('0.3')['areas'] for n in area['benchmarks']}
    folder = ROOT / 'evaluation' / args.model
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / f'part-{args.worker}.jsonl'
    seen = prior_results(output_path)
    count = 0
    total_tokens = 0
    started = time.monotonic()
    budget = 32768
    progress = folder / f'progress-{args.worker}.json'
    with gzip.open(CACHE / 'normalized/rows-for-evaluation.jsonl.gz', 'rt', encoding='utf-8') as source, \
            output_path.open('a', buffering=1024 * 1024) as output:
        ordinal = 0
        for line in source:
            row = json.loads(line)
            evaluation = row['_evaluation']
            if evaluation['catalog_id'] not in allowed:
                continue
            shard = ordinal % args.workers
            ordinal += 1
            if shard != args.worker:
                continue
            run_id = evaluation['run_id']
            if run_id in seen:
                continue
            wall = time.monotonic()
            first, last = bounds[run_id]
            states = np.asarray(table['state_id'][first:last], dtype=np.int64)
            candidates = np.asarray(table['candidate_id'][first:last], dtype=np.int64)
            scores, budget, token_count = margins_for_request(model, tokenizer, tokens, states, candidates, budget)
            record = {'run_id': run_id, 'catalog_id': evaluation['catalog_id'],
                      'status': 'unsupported' if scores is None else 'ok'}
            if scores is None:
                record['reason'] = 'Whole candidate request exceeds model context or single-pair GPU memory; not truncated'
            else:
                record['response'] = {'answers': answers_for_request(table, first, last, scores),
                                      'usage': {'input_tokens': token_count}}
            record['total_wall_ms'] = (time.monotonic() - wall) * 1000
            record['model_request_wall_ms'] = record['total_wall_ms']
            output.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            seen.add(run_id)
            count += 1
            total_tokens += token_count
            if count % 200 == 0:
                output.flush()
                data = {'model': args.model, 'worker': args.worker, 'new_requests': count,
                        'terminal_requests_in_shard': len(seen), 'seconds': time.monotonic() - started,
                        'nonpadding_tokens': total_tokens, 'micro_token_budget': budget}
                atomic_json(progress, data)
                print(json.dumps(data), flush=True)
    atomic_json(folder / f'worker-{args.worker}-complete.json',
                {'complete': True, 'terminal_requests': len(seen), 'new_requests': count,
                 'nonpadding_tokens': total_tokens, 'seconds': time.monotonic() - started})


def score_model(name, folder):
    results_path = folder / 'results.jsonl'
    temporary = folder / 'results.jsonl.writing'
    with temporary.open('wb') as out:
        for part in sorted(folder.glob('part-*.jsonl')):
            with part.open('rb') as src:
                for line in src:
                    if line.strip():
                        out.write(line)
    temporary.replace(results_path)
    suite = Suite(SUITE_PATH, edition='0.3')
    spec = index02.spec('0.3')
    allowed = {n for area in spec['areas'] for n in area['benchmarks']}
    rows = [row for row in suite.rows(apply_exclusions=True) if row['_evaluation']['catalog_id'] in allowed]
    results = load_results(results_path)
    missing = [row['_evaluation']['run_id'] for row in rows if row['_evaluation']['run_id'] not in results]
    if missing:
        raise RuntimeError(f'{name} has {len(missing)} unprocessed public requests')
    summary = benchmark_summary(suite, results, engine=name, rows=rows)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['_evaluation']['catalog_id']].append(row)
    added_reports = {int(n): added.report(int(n), groups[int(n)], results)
                     for n in spec['added'] if int(n) in groups}
    scores = index02.index_entry(suite, results, summary, added_reports, s=spec)
    atomic_json(folder / 'benchmark-summary.json', summary)
    atomic_json(folder / 'added-reports.json', added_reports)
    atomic_json(folder / 'scores.json', scores)
    atomic_json(folder / 'complete.json',
                {'complete': True, 'model': name, 'requests': len(rows), 'benchmarks': len(groups),
                 'counts': dict(collections.Counter(results[row['_evaluation']['run_id']]['status'] for row in rows)),
                 'public_score': scores['index'], 'coverage': scores['coverage'],
                 'latency_scope': 'Four-GPU score run; per-row synchronized candidate inference, not isolated latency-v1',
                 'evaluation_inputs': 'Locally reconstructed public 0.3 inputs; not a private Full score'})
    print(json.dumps({'event': 'formal_public_score_complete', 'model': name,
                      'public_score': scores['index'], 'coverage': scores['coverage']}), flush=True)
    return scores


def radar(records, path, title):
    axes = ['knowledge', 'language', 'retrieval', 'tools', 'arts']
    cx, cy, radius = 250, 230, 170
    def point(index, value):
        angle = -math.pi / 2 + index * 2 * math.pi / 5
        return cx + radius * value * math.cos(angle), cy + radius * value * math.sin(angle)
    def poly(values):
        return ' '.join(f'{x:.2f},{y:.2f}' for x, y in (point(i, v) for i, v in enumerate(values)))
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="470" viewBox="0 0 1050 470">',
             '<rect width="1050" height="470" fill="white"/>',
             f'<text x="25" y="28" font-size="19">{title}</text>']
    for value in (.2, .4, .6, .8, 1):
        lines.append(f'<polygon points="{poly([value] * 5)}" fill="none" stroke="#ccc"/>')
    for i, axis in enumerate(axes):
        x, y = point(i, 1.15)
        lines.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="14">{axis}</text>')
    colors = ['#222222', '#2673d9', '#ee8d22', '#9d49c6', '#238a70', '#c64e62']
    for index, (name, scores) in enumerate(records):
        values = {a['id']: a['skill'] for a in scores['areas']}
        color = colors[index % len(colors)]
        lines.append(f'<polygon points="{poly([values[a] for a in axes])}" fill="none" stroke="{color}" stroke-width="2"/>')
        y = 70 + index * 36
        lines += [f'<line x1="525" x2="555" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                  f'<text x="565" y="{y+5}" font-size="13">{name} : {scores["index"]:.2f}</text>']
    lines.append('</svg>')
    path.write_text('\n'.join(lines))


def final_report(records):
    folder = ROOT / 'report'
    folder.mkdir(exist_ok=True)
    axes = ['knowledge', 'language', 'retrieval', 'tools', 'arts']
    with (folder / 'five-area-results.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['model', *axes, 'public_total', 'coverage'])
        for name, scores in records:
            values = {a['id']: 100 * a['skill'] for a in scores['areas']}
            writer.writerow([name, *(values[a] for a in axes), scores['index'], scores['coverage']])
    by_name = dict(records)
    mixed = by_name['MIXED_ALL']
    mixed_axes = {a['id']: 100 * a['skill'] for a in mixed['areas']}
    differences = {}
    for name, scores in records:
        values = {a['id']: 100 * a['skill'] for a in scores['areas']}
        differences[name] = {axis: values[axis] - mixed_axes[axis] for axis in axes}
        differences[name]['public_total'] = scores['index'] - mixed['index']
    atomic_json(folder / 'differences-vs-mixed.json', differences)
    radar([('MIXED_ALL', mixed)] + [(n, s) for n, s in records if n.startswith('SPECIALIST_')],
          folder / 'specialists-vs-mixed.svg', 'Five specialists vs mixed: public skill')
    radar([('MIXED_ALL', mixed)] + [(n, s) for n, s in records if n.startswith('MERGE_')],
          folder / 'merges-vs-mixed.svg', 'Effective-weight merges vs mixed: public skill')
    lines = ['# 五专科、混合训练与模型融合结果', '',
             '| 模型 | Knowledge | Language | Retrieval | Tools | Arts | 公共总分 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, scores in records:
        values = {a['id']: 100 * a['skill'] for a in scores['areas']}
        lines.append('| ' + name + ' | ' + ' | '.join(f'{values[a]:.2f}' for a in axes) + f' | {scores["index"]:.2f} |')
    lines += ['', '分数为统一公共五维评分，不是私测Full。', '',
              '![专科与混合](specialists-vs-mixed.svg)', '', '![融合与混合](merges-vs-mixed.svg)']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if args.worker is not None:
        worker(args)
        return
    records = []
    conditions = [(name, ROOT / name / 'final') for name in CFG['models']]
    merges = json.loads((ROOT / 'MERGES_COMPLETE.json').read_text())
    conditions += [(entry['model'], Path(entry['checkpoint'])) for entry in merges['models']]
    if not json.loads((CACHE / 'tokenized/MANIFEST.json').read_text()).get('complete'):
        raise RuntimeError('Formal evaluation token cache is not ready')
    for name, checkpoint in conditions:
        folder = ROOT / 'evaluation' / name
        folder.mkdir(parents=True, exist_ok=True)
        if done(folder / 'complete.json'):
            records.append((name, json.loads((folder / 'scores.json').read_text())))
            continue
        processes = []
        handles = []
        for rank in range(4):
            if done(folder / f'worker-{rank}-complete.json'):
                continue
            handle = (HOME / 'logs' / f'evaluate-{name}-gpu{rank}.log').open('a', buffering=1)
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            processes.append(subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(rank),
                                               '--workers', '4', '--model', name, '--checkpoint', str(checkpoint)],
                                              cwd=HOME, env=env, stdin=subprocess.DEVNULL,
                                              stdout=handle, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        for handle in handles:
            handle.close()
        if any(codes):
            raise RuntimeError(f'Formal evaluation worker failed for {name}; exit codes {codes}')
        records.append((name, score_model(name, folder)))
    final_report(records)
    atomic_json(ROOT / 'EVALUATION_COMPLETE.json', {'complete': True, 'conditions': len(records),
                                                 'report': str(ROOT / 'report/REPORT.md')})


if __name__ == '__main__':
    main()
