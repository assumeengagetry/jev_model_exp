#!/usr/bin/env python3
"""Read completed experiment artifacts for notifications; never changes the runs."""
import fcntl
import json
from pathlib import Path

HOME = Path('/home/dev/assumeengage')
ROOT = HOME / 'runs/five-specialists-20261030'
NEXT = HOME / 'runs/distillation-after-five-specialists-20261031'
CONFIG = json.loads((HOME / 'config/experiment-five-specialists.json').read_text())
MODELS = CONFIG['models']
EXPECTED_CONDITIONS = len(MODELS) + len(CONFIG['merges'])


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def completed(path):
    return read(path).get('complete') is True


def main():
    milestones = []
    for name in MODELS:
        folder = ROOT / name
        data = read(folder / 'complete.json')
        if data.get('complete') is True:
            milestones.append({'key': 'five/train/' + name, 'stage': '正式模型训练完成',
                               'name': name, 'artifacts': [str(folder / 'complete.json'), str(folder / 'final')],
                               'log': str(HOME / 'logs' / ('takeover-' + name + '.log')),
                               'summary': {k: data.get(k) for k in ('pairs_seen', 'effective_tokens', 'elapsed_seconds')}})
    for folder in sorted((ROOT / 'merges').glob('*')):
        data = read(folder / 'complete.json')
        if data.get('complete') is True:
            milestones.append({'key': 'five/merge/' + folder.name, 'stage': '权重融合完成',
                               'name': folder.name, 'artifacts': [str(folder / 'complete.json'), str(folder / 'final')]})
    for folder in sorted((ROOT / 'evaluation').glob('*')):
        data = read(folder / 'complete.json')
        if data.get('complete') is True:
            milestones.append({'key': 'five/evaluation/' + folder.name, 'stage': '正式五维评分完成',
                               'name': folder.name, 'artifacts': [str(folder / p) for p in
                                   ('complete.json', 'scores.json', 'benchmark-summary.json', 'results.jsonl')],
                               'summary': {k: data.get(k) for k in ('public_score', 'coverage', 'requests')}})
    final = read(ROOT / 'EVALUATION_COMPLETE.json')
    if final.get('complete') is True:
        milestones.append({'key': 'five/final-report', 'stage': '本轮总报告与图表完成', 'name': '四专科与融合总报告',
                           'artifacts': [str(ROOT / 'report'), str(ROOT / 'EVALUATION_COMPLETE.json'),
                                         str(ROOT / 'MERGES_COMPLETE.json')]})
    for name in ('TEACHER_READY', 'TEACHER_LABELS_READY', 'DISTILLATION_COMPLETE'):
        path = NEXT / (name + '.json')
        data = read(path)
        if data.get('complete') is True:
            artifacts = data.get('artifacts', [])
            milestones.append({'key': 'distillation/' + name, 'stage': '蒸馏阶段成果', 'name': name,
                               'artifacts': [str(path)] + artifacts})
    for name in ('KD_HARD', 'KD_SOFT', 'KD_HYBRID'):
        folder = NEXT / name
        if completed(folder / 'complete.json'):
            milestones.append({'key': 'distillation/train/' + name, 'stage': '蒸馏模型训练完成', 'name': name,
                               'artifacts': [str(folder / 'complete.json'), str(folder / 'final')]})
        eval_folder = NEXT / 'evaluation' / name
        if completed(eval_folder / 'complete.json'):
            milestones.append({'key': 'distillation/evaluation/' + name, 'stage': '蒸馏五维评分完成', 'name': name,
                               'artifacts': [str(eval_folder)]})
    pipeline = read(ROOT / 'PIPELINE.json')
    all_done = (all(completed(ROOT / name / 'complete.json') for name in MODELS)
                and completed(ROOT / 'MERGES_COMPLETE.json')
                and final.get('complete') is True and final.get('conditions') == EXPECTED_CONDITIONS
                and pipeline.get('stage') == 'complete')
    lock_free = False
    if all_done:
        try:
            with (ROOT / 'EXPERIMENT.lock').open('rb') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                lock_free = True
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except (OSError, BlockingIOError):
            pass
    print(json.dumps({'milestones': milestones, 'five_area_complete': bool(all_done and lock_free),
                      'active_models': MODELS, 'active_training_domains': CONFIG.get('active_training_domains'),
                      'expected_conditions': EXPECTED_CONDITIONS,
                      'pipeline': pipeline, 'distillation_status': read(NEXT / 'PIPELINE.json'),
                      'distillation_complete': completed(NEXT / 'DISTILLATION_COMPLETE.json')},
                     ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
