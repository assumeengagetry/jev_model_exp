#!/usr/bin/env python3
"""Extend the read-only milestone snapshot; original experiment gates stay untouched."""
import json
from pathlib import Path
import subprocess

HOME = Path('/home/dev/assumeengage')
ROOT = HOME / 'runs/four-specialists-slerp-20261102'
NAME = 'MERGE_SLERP_HIERARCHICAL'


def read(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


result = subprocess.run(['python3', '-B', str(HOME/'scripts/five-area-stage-snapshot.py')],
                        capture_output=True, text=True, timeout=30, check=True)
current = json.loads(result.stdout)
merge = read(ROOT/'MERGE_COMPLETE.json')
if merge.get('complete') is True:
    current['milestones'].append({'key': 'slerp/merge/'+NAME, 'name': NAME, 'stage': '四专家分层SLERP融合完成',
        'artifacts': [str(ROOT/'MERGE_COMPLETE.json')]+merge.get('artifacts', [])})
completion = read(ROOT/'COMPLETE.json')
if completion.get('complete') is True:
    current['milestones'].append({'key': 'slerp/final-report', 'name': 'SLERP补充正式评测与11条件对照',
        'stage': 'SLERP公共37项评分完成', 'summary': {'public_score': completion.get('public_score')},
        'artifacts': [str(ROOT/'COMPLETE.json')]+completion.get('artifacts', [])})
current['slerp_status'] = read(ROOT/'PIPELINE.json')
if not completion.get('complete') and current['slerp_status'].get('stage') in ('running', 'waiting_gpu'):
    owner = read(ROOT/'OWNER.json')
    try:
        fields = Path('/proc', str(owner['pid']), 'stat').read_text().rsplit(') ', 1)[1].split()
        alive = fields[0] != 'Z' and fields[19] == owner['starttime']
    except (OSError, KeyError, IndexError):
        alive = False
    if not alive:
        current['slerp_status'] = {**current['slerp_status'], 'stage': 'blocked',
                                  'error': 'SLERP owner exited before completion; resume preserved shards'}
current['slerp_complete'] = completion.get('complete') is True
print(json.dumps(current, ensure_ascii=False, separators=(',', ':')))
