#!/usr/bin/env python3
"""Acquire independent train-only supplements while the training session prepares.

Uses existing requests and public domestic mirrors. No model runs, smoke tests,
hashing, changes to existing datasets, or changes to the shared download helpers.
"""

import concurrent.futures
import fnmatch
import json
import shutil
import threading
import time
from pathlib import Path

import requests

HOME = Path('/home/dev/assumeengage')
ROOT = HOME / 'datasets/overnight-supplement'
SOURCES = [
    ('knowledge', 'openbookqa', 'allenai/openbookqa', ['main/train-*.parquet']),
    ('knowledge', 'sciq', 'allenai/sciq', ['data/train-*.parquet']),
    ('knowledge', 'mbpp', 'google-research-datasets/mbpp', ['full/train-*.parquet']),
    ('knowledge', 'math', 'AI-ModelScope/MATH-lighteval', ['data/train-*.parquet']),
    ('language', 'snli', 'stanfordnlp/snli', ['plain_text/train-*.parquet']),
    ('language', 'multinli', 'nyu-mll/multi_nli', ['data/train-*.parquet']),
    ('language', 'sst2', 'stanfordnlp/sst2', ['data/train-*.parquet']),
    ('language', 'tweet_sentiment', 'cardiffnlp/tweet_eval', ['sentiment/train-*.parquet']),
    ('retrieval', 'massive', 'MTEB/amazon_massive_intent', ['train/en.json.gz']),
    ('retrieval', 'ag_news', 'torushy/flm-ag-news', ['dataset/train/data-*.arrow']),
    ('retrieval', 'dbpedia', 'MTEB/llm-eval-dbpedia_14', ['data/train-*.parquet']),
    ('retrieval', 'esci', 'thebajajra/amazon-esci-english-small', ['data/train-*.parquet']),
    ('tools', 'glaive', 'glaiveai/glaive-function-calling-v2', ['glaive-function-calling-v2.json']),
    ('tools', 'xlam', 'Salesforce/xlam-function-calling-60k', ['xlam_function_calling_60k.json']),
    ('arts', 'helpsteer2', 'AI-ModelScope/HelpSteer2', ['train.jsonl.gz']),
    ('arts', 'helpsteer', 'nv-community/HelpSteer', ['data/train-*.parquet', 'train.jsonl.gz']),
    ('arts', 'hh_rlhf', 'AI-ModelScope/hh-rlhf', [
        'helpful-online/train.jsonl', 'helpful-base/train.jsonl',
        'helpful-rejection-sampled/train.jsonl', 'harmless-base/train.jsonl',
    ]),
    ('arts', 'ultrafeedback', 'HuggingFaceH4/ultrafeedback_binarized', ['data/train_prefs-*.parquet']),
]
LOCAL = threading.local()
RESULTS = []
ROOT.mkdir(parents=True, exist_ok=True)


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session


def write_manifest():
    manifest = {
        'purpose': 'Train-only supplements for the user-authorized five-specialist experiment',
        'owner_session': 'ses_e7444cec5ffeRHet9AGZDT1Ckz',
        'canonical_base_pool': 'tasksource/zero-shot-label-nli default is still required separately',
        'extra_tests': False,
        'hash_checks': False,
        'sources_requested': len(SOURCES),
        'results': RESULTS,
        'updated_unix': time.time(),
    }
    temporary = ROOT / 'DOWNLOAD.json.tmp'
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(ROOT / 'DOWNLOAD.json')


def discover(source):
    area, name, repo, patterns = source
    base = 'https://modelscope.cn/api/v1/datasets/' + repo
    entries = {}
    for page in range(1, 101):
        response = session().get(base + '/repo/tree', params={
            'Revision': 'master', 'Recursive': 'True', 'PageSize': 100, 'PageNumber': page,
        }, timeout=(8, 35))
        response.raise_for_status()
        payload = response.json()
        if payload.get('Code') not in (None, 200):
            raise RuntimeError(str(payload.get('Message', payload.get('Code'))))
        rows = payload.get('Data', {}).get('Files', [])
        for row in rows:
            entries[row['Path']] = row
        total = payload.get('TotalCount') or payload.get('Data', {}).get('TotalCount')
        if not rows or (total and len(entries) >= total) or (not total and len(rows) < 100):
            break
    selected = [r for p, r in entries.items() if r['Type'] == 'blob'
                and any(fnmatch.fnmatchcase(p, pattern) for pattern in patterns)]
    if not selected:
        return [], {'area': area, 'name': name, 'repo': repo, 'status': 'needs_source_mapping',
                    'available_files': [p for p, r in entries.items() if r['Type'] == 'blob'][:120]}
    return [(area, name, repo, row) for row in selected], None


def download(job):
    area, name, repo, row = job
    target = ROOT / 'train' / area / name / row['Path']
    target.parent.mkdir(parents=True, exist_ok=True)
    result = {'area': area, 'name': name, 'repo': repo,
              'file': row['Path'], 'path': str(target.relative_to(HOME)),
              'bytes': row['Size'], 'mirror_revision': row.get('Revision'), 'split': 'train'}
    if target.exists():
        if target.stat().st_size == row['Size']:
            return dict(result, status='present')
        return dict(result, status='blocked', error='Existing different-sized file preserved')
    if shutil.disk_usage(ROOT).free - row['Size'] < 20 * 1024**3:
        return dict(result, status='blocked', error='Insufficient remaining space alongside model outputs')
    partial = target.with_name(target.name + '.download.part')
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with session().get('https://modelscope.cn/api/v1/datasets/' + repo + '/repo',
                               params={'Revision': row.get('Revision') or 'master', 'FilePath': row['Path']},
                               headers=headers, stream=True, timeout=(12, 90)) as response:
                response.raise_for_status()
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                with partial.open(mode) as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            if partial.stat().st_size != row['Size']:
                raise OSError('Incomplete HTTP transfer; resumable partial retained')
            partial.replace(target)
            return dict(result, status='downloaded')
        except Exception as error:
            if attempt == 2:
                return dict(result, status='failed', error=str(error)[:400])


def main():
    jobs = []
    print('Preparing train-only domestic supplement downloads; no tests or hashes.', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        future_sources = {pool.submit(discover, source): source for source in SOURCES}
        for future in concurrent.futures.as_completed(future_sources):
            source = future_sources[future]
            try:
                found, note = future.result()
                jobs.extend(found)
                if note:
                    RESULTS.append(note)
                print('SOURCE', source[1], 'train_files', len(found), flush=True)
            except Exception as error:
                RESULTS.append({'area': source[0], 'name': source[1], 'repo': source[2],
                                'status': 'source_unavailable', 'error': str(error)[:400]})
                print('SOURCE_UNAVAILABLE', source[1], type(error).__name__, flush=True)
            write_manifest()
    print('Downloading', len(jobs), 'training files,', sum(j[3]['Size'] for j in jobs), 'bytes.', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            RESULTS.append(result)
            print(result['status'], result['name'], result['file'], result.get('bytes'), flush=True)
            write_manifest()
    print('Finished supplement pass:', json.dumps(dict(__import__('collections').Counter(
        r['status'] for r in RESULTS))), flush=True)


if __name__ == '__main__':
    main()
