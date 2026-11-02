#!/usr/bin/env python3
"""Download every text shard of the public domestic Tasksource mirror snapshot."""
import concurrent.futures
import json
from pathlib import Path

import requests

HOME = Path('/home/dev/assumeengage')
REPO = 'saiting/tasksource-jev-typed-decisions'
DEST = HOME / 'datasets/tasksource-jev-typed-decisions-ms'
BASE = 'https://modelscope.cn/api/v1/datasets/' + REPO
DEST.mkdir(parents=True, exist_ok=True)
entries = {}
page = 1
while True:
    response = requests.get(BASE + '/repo/tree', params={
        'Revision': 'master', 'Recursive': 'True', 'PageSize': 100, 'PageNumber': page,
    }, timeout=(10, 40))
    response.raise_for_status()
    data = response.json()
    current = data['Data']['Files']
    for row in current:
        entries[row['Path']] = row
    total = data.get('TotalCount') or data['Data'].get('TotalCount') or len(entries)
    if len(entries) >= total or not current: break
    page += 1

selected = [row for name, row in entries.items() if row['Type'] == 'blob' and (
    (name.startswith(('data/', 'full/', 'filtered-full/')) and name.endswith(('.parquet', '.json', '.md')))
    or name in {'README.md', 'sources.yaml', 'build-summary.json'}
)]
print('Downloading the full domestic text snapshot:', len(selected), 'files,',
      sum(row['Size'] for row in selected), 'bytes. Vision excluded.', flush=True)


def fetch(row):
    name = row['Path'];target = DEST / name;target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != row['Size']:
            return {'file': name, 'status': 'failed', 'error': 'Existing different-size file preserved'}
        return {'file': name, 'status': 'present', 'bytes': row['Size']}
    partial = target.with_name(target.name + '.ms.part')
    for attempt in range(3):
        try:
            with requests.get(BASE + '/repo', params={'Revision': row.get('Revision') or 'master', 'FilePath': name},
                              stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with partial.open('wb') as output:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        output.write(chunk)
            if partial.stat().st_size != row['Size']:
                raise RuntimeError('Incomplete HTTP byte count')
            partial.replace(target)
            print('Downloaded', name, row['Size'], flush=True)
            return {'file': name, 'status': 'downloaded', 'bytes': row['Size'],
                    'mirror_file_revision': row.get('Revision')}
        except Exception as error:
            if attempt == 2:
                return {'file': name, 'status': 'failed', 'error': str(error)}


results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    pending = {pool.submit(fetch, row): row['Path'] for row in selected}
    for future in concurrent.futures.as_completed(pending):
        results.append(future.result())
        (DEST / 'DOWNLOAD.json').write_text(json.dumps({
            'upstream_repo': 'tasksource/tasksource-jev-typed-decisions',
            'mirror_repo': REPO, 'source': 'ModelScope domestic public snapshot',
            'configs': ['default', 'full', 'filtered-full'],
            'note': 'Separate mirror snapshot, not mixed with the newer incomplete HF snapshot.',
            'extra_file_hash_checks': False, 'files': results,
            'complete': len(results) == len(selected) and all(r['status'] != 'failed' for r in results),
        }, ensure_ascii=False, indent=2))
failures = [row for row in results if row['status'] == 'failed']
print('Domestic Tasksource snapshot:', len(results), 'files,', len(failures), 'failures', flush=True)
raise SystemExit(bool(failures))
