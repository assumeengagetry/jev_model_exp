#!/usr/bin/env python3
"""Download official benchmark train mirrors from domestic ModelScope storage."""
import concurrent.futures
import json
from pathlib import Path
from urllib.parse import urlencode

import requests

HOME = Path('/home/dev/assumeengage')
ROOT = HOME / 'datasets/benchmarks-train'
ITEMS = [
    ('gsm8k', 'knowledge', 'AI-ModelScope/gsm8k', 'main/train-00000-of-00001.parquet'),
    ('winogrande', 'knowledge', 'allenai/winogrande', 'winogrande_xl/train-00000-of-00001.parquet'),
    ('anli', 'language', 'facebook/anli', 'plain_text/train_r1-00000-of-00001.parquet'),
    ('anli', 'language', 'facebook/anli', 'plain_text/train_r2-00000-of-00001.parquet'),
    ('anli', 'language', 'facebook/anli', 'plain_text/train_r3-00000-of-00001.parquet'),
    ('when2call', 'tools', 'nv-community/When2Call', 'train/when2call_train_pref.jsonl'),
    ('when2call', 'tools', 'nv-community/When2Call', 'train/when2call_train_sft.jsonl'),
]
EXTRAS = [
    ('mmlu-source', 'https://modelscope.cn/api/v1/datasets/modelscope/mmlu/repo?' + urlencode({'Revision': 'master', 'FilePath': 'data.tar'}),
     HOME / 'datasets/benchmark-sources/knowledge/mmlu-data.tar'),
    ('hellaswag-source', 'https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/hellaswag/hellaswag_train.jsonl',
     HOME / 'datasets/benchmark-sources/language/hellaswag-train.jsonl'),
]


def fetch(name, url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {'name': name, 'path': str(destination.relative_to(HOME)),
                'bytes': destination.stat().st_size, 'status': 'present', 'source': url}
    partial = destination.with_name(destination.name + '.modelscope.part')
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                expected = response.headers.get('content-length')
                received = 0
                with partial.open('wb') as output:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
            if expected and received != int(expected):
                raise RuntimeError('Incomplete HTTP response')
            partial.replace(destination)
            print('Downloaded', destination.relative_to(HOME), received, flush=True)
            return {'name': name, 'path': str(destination.relative_to(HOME)), 'bytes': received,
                    'status': 'downloaded', 'source': url}
        except Exception as error:
            if attempt == 2:
                return {'name': name, 'path': str(destination.relative_to(HOME)),
                        'status': 'failed', 'error': str(error), 'source': url}


jobs = []
for name, area, repo, filename in ITEMS:
    url = f'https://modelscope.cn/api/v1/datasets/{repo}/repo?' + urlencode({'Revision': 'master', 'FilePath': filename})
    jobs.append((name, url, ROOT / area / name / filename))
jobs.extend(EXTRAS)
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    for result in pool.map(lambda job: fetch(*job), jobs):
        results.append(result)
        print(result['name'], result['status'], flush=True)
        (ROOT / 'MODELSCOPE_DOWNLOADS.json').write_text(json.dumps({
            'hash_verification': False, 'train_only': True,
            'files': results, 'complete': len(results) == len(jobs) and all(r['status'] != 'failed' for r in results),
        }, ensure_ascii=False, indent=2))
raise SystemExit(any(r['status'] == 'failed' for r in results))
