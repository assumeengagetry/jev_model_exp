#!/usr/bin/env python3
"""Evaluate the three KD arms on the unchanged public suite and reuse the fixed parent baseline."""
import argparse
import csv
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HOME = Path('/home/dev/assumeengage')
CFG = json.loads((HOME / 'config/experiment-four-domain-distillation.json').read_text())
ROOT = HOME / CFG['run_root']
PREVIOUS = HOME / CFG['previous_run_root']
AXES = ['knowledge', 'language', 'retrieval', 'tools', 'arts']
spec = importlib.util.spec_from_file_location('formal_public_evaluation', HOME / 'scripts/run-formal-evaluation.py')
EVAL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EVAL)
EVAL.ROOT = ROOT
EVAL.CFG = {**EVAL.CFG, 'run_root': CFG['run_root'], 'models': CFG['arms']}


def done(path):
    return EVAL.done(path)


def axes(scores):
    return {area['id']: 100 * area['skill'] for area in scores['areas']}


def heatmap(records, baseline, path):
    ids = [n for area in EVAL.index02.spec('0.3')['areas'] for n in area['benchmarks']]
    cell, left, top = 44, 150, 165
    width, height = left + len(ids) * cell + 30, top + len(records) * 48 + 65
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="28" font-size="19">Benchmark skill delta vs parent (percentage points)</text>']
    names = {b['catalog_id']: b['dataset'] for b in
             json.loads((PREVIOUS / 'evaluation/MIXED_ALL/benchmark-summary.json').read_text())['benchmarks']}
    for column, number in enumerate(ids):
        x = left + column * cell + cell / 2
        label = html.escape(f'{number}: {names.get(number, number)}')
        lines.append(f'<text transform="translate({x},{top-12}) rotate(-60)" font-size="10">{label}</text>')
    for row, (name, scores) in enumerate(records):
        y = top + row * 48
        lines.append(f'<text x="10" y="{y+27}" font-size="13">{html.escape(name)}</text>')
        for column, number in enumerate(ids):
            difference = 100 * (scores['benchmarks'][str(number)]['skill'] - baseline['benchmarks'][str(number)]['skill'])
            intensity = min(1, abs(difference) / 25)
            pale = round(255 * (1 - intensity))
            color = f'rgb({pale},255,{pale})' if difference >= 0 else f'rgb(255,{pale},{pale})'
            x = left + column * cell
            lines += [f'<rect x="{x}" y="{y}" width="{cell}" height="44" fill="{color}" stroke="white"/>',
                      f'<text x="{x+cell/2}" y="{y+27}" text-anchor="middle" font-size="10">{difference:+.2f}</text>']
    lines.append('</svg>'); path.write_text('\n'.join(lines))


def bars(records, path):
    width, height, bottom, scale_height = 760, 440, 345, 280
    ceiling = max(25, math.ceil(max(scores['index'] for _, scores in records) / 5) * 5)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="28" font-size="19">Decision Index 0.3 public total</text>']
    for tick in range(6):
        value = ceiling * tick / 5; y = bottom - scale_height * tick / 5
        lines += [f'<line x1="65" x2="730" y1="{y}" y2="{y}" stroke="#ddd"/>',
                  f'<text x="55" y="{y+4}" text-anchor="end" font-size="12">{value:.0f}</text>']
    colors = ['#777', '#2673d9', '#ee8d22', '#238a70']
    for index, (name, scores) in enumerate(records):
        x = 95 + index * 165; bar = scale_height * scores['index'] / ceiling
        lines += [f'<rect x="{x}" y="{bottom-bar}" width="100" height="{bar}" fill="{colors[index]}"/>',
                  f'<text x="{x+50}" y="{bottom-bar-10}" text-anchor="middle" font-size="16">{scores["index"]:.2f}</text>',
                  f'<text x="{x+50}" y="{bottom+25}" text-anchor="middle" font-size="12">{html.escape(name)}</text>']
    lines.append('</svg>'); path.write_text('\n'.join(lines))


def report(records):
    folder = ROOT / 'report'; folder.mkdir(exist_ok=True)
    parent = json.loads((PREVIOUS / 'evaluation/MIXED_ALL/scores.json').read_text())
    all_records = [('PARENT_MIXED_ALL', parent)] + records
    EVAL.radar(all_records, folder / 'four-model-radar.svg', 'Same parent: original / HARD / SOFT / HYBRID')
    heatmap(records, parent, folder / 'benchmark-delta-heatmap.svg')
    bars(all_records, folder / 'public-total-bars.svg')
    with (folder / 'results.csv').open('w', newline='') as out:
        writer = csv.writer(out); writer.writerow(['model', *AXES, 'public_total', 'coverage'])
        for name, scores in all_records:
            values = axes(scores)
            writer.writerow([name, *(f'{values[a]:.2f}' for a in AXES), f'{scores["index"]:.2f}', f'{100*scores["coverage"]:.2f}'])
    by_name = dict(records); comparisons = {}
    for name, scores in records:
        comparisons[name + '_minus_parent'] = {a: axes(scores)[a] - axes(parent)[a] for a in AXES}
        comparisons[name + '_minus_parent']['public_total'] = scores['index'] - parent['index']
    for name in ('KD_SOFT', 'KD_HYBRID'):
        comparisons[name + '_minus_KD_HARD'] = {a: axes(by_name[name])[a] - axes(by_name['KD_HARD'])[a] for a in AXES}
        comparisons[name + '_minus_KD_HARD']['public_total'] = by_name[name]['index'] - by_name['KD_HARD']['index']
    EVAL.atomic_json(folder / 'differences.json', comparisons)
    lines = ['# 四科完整池：Teacher 蒸馏与等量原始监督续训', '',
             '三组均从同一个已训练 MIXED_ALL 出发，使用相同四科完整池、seed、全局 batch 与一轮曝光；Language 不参与训练。', '',
             '| 模型 | Knowledge | Language | Retrieval | Tools | Arts | 公共总分 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, scores in all_records:
        values = axes(scores)
        lines.append('| ' + name + ' | ' + ' | '.join(f'{values[a]:.2f}' for a in AXES) + f' | {scores["index"]:.2f} |')
    lines += ['', '## 主要差值', '', '| 对照 | 公共总分差值 |', '|---|---:|']
    for name, values in comparisons.items():
        lines.append(f'| {name} | {values["public_total"]:+.2f} |')
    lines += ['', '## 正式训练量与耗时', '', '| 模型 | pair 数 | 输入 tokens | 训练小时 |', '|---|---:|---:|---:|']
    for name, _ in records:
        data = json.loads((ROOT / name / 'complete.json').read_text())
        lines.append(f'| {name} | {data["pairs_seen"]/10000:.2f} 万 | {data["effective_tokens"]/1e8:.2f} 亿 | {data["elapsed_seconds"]/3600:.2f} |')
    lines += ['', '以上为同一37项公共五维评分，不是私测 Full。起点成绩直接复用上一轮正式评测，HARD 是本轮新增完整续训，不以起点冒充。',
              'Teacher 使用原生独立每 pair 的匹配/不匹配概率，温度已折入权重，不再次缩放。', '',
              '![四条五维曲线](four-model-radar.svg)', '', '![逐项差值](benchmark-delta-heatmap.svg)', '',
              '![总体分数](public-total-bars.svg)']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    EVAL.atomic_json(ROOT / 'EVALUATION_COMPLETE.json', {'complete': True, 'conditions': 3,
                     'baseline_reused': str(PREVIOUS / 'evaluation/MIXED_ALL'), 'report': str(folder / 'REPORT.md')})
    EVAL.atomic_json(ROOT / 'DISTILLATION_COMPLETE.json', {'complete': True, 'arms': CFG['arms'],
                     'public_conditions': 3, 'benchmarks_each': 37, 'student_parent': CFG['student_parent'],
                     'active_training_domains': CFG['active_training_domains'], 'pairs_each': CFG['pairs'],
                     'artifacts': [str(folder), str(ROOT / 'EVALUATION_COMPLETE.json'),
                                   str(HOME / 'config/experiment-four-domain-distillation.json')]})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', type=int)
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--model'); parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if args.worker is not None:
        EVAL.worker(args); return
    records = []
    for name in CFG['arms']:
        folder = ROOT / 'evaluation' / name; folder.mkdir(parents=True, exist_ok=True)
        if done(folder / 'complete.json'):
            records.append((name, json.loads((folder / 'scores.json').read_text()))); continue
        children, handles = [], []
        for rank in range(4):
            if done(folder / f'worker-{rank}-complete.json'):
                continue
            output = (HOME / 'logs' / f'distillation-evaluate-{name}-gpu{rank}.log').open('a', buffering=1)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            children.append(subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(rank),
                            '--workers', '4', '--model', name, '--checkpoint', str(ROOT / name / 'final')],
                            cwd=HOME, env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT))
            handles.append(output)
        codes = [child.wait() for child in children]
        for handle in handles: handle.close()
        if any(codes): raise RuntimeError(f'{name} evaluation workers failed: {codes}')
        records.append((name, EVAL.score_model(name, folder)))
    report(records)


if __name__ == '__main__':
    main()
