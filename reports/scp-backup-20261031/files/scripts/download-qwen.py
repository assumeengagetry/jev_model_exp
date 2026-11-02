#!/usr/bin/env python3
"""Download the original Qwen3-0.6B files from ModelScope, without hashing."""
import concurrent.futures
import json
import time
from pathlib import Path

import requests

ROOT = Path('/home/dev/assumeengage')
DEST = ROOT / 'models/Qwen3-0.6B'
API = 'https://modelscope.cn/api/v1/models/Qwen/Qwen3-0.6B/repo'
DEST.mkdir(parents=True, exist_ok=True)
metadata = requests.get(API + '/files', params={'Revision': 'master', 'Recursive': 'true'}, timeout=30).json()
files = [f for f in metadata['Data']['Files'] if f['Type'] == 'blob']


def download(info):
    path = DEST / info['Path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == info['Size']:
        return {'file': info['Path'], 'bytes': path.stat().st_size, 'status': 'present'}
    partial = path.with_name(path.name + '.part')
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(API, params={'Revision': 'master', 'FilePath': info['Path']},
                              headers=headers, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                with partial.open(mode) as output:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        output.write(chunk)
            if partial.stat().st_size != info['Size']:
                raise RuntimeError(f"Incomplete download: {info['Path']}")
            partial.replace(path)
            print(f"Downloaded {info['Path']}: {path.stat().st_size:,} bytes", flush=True)
            return {'file': info['Path'], 'bytes': path.stat().st_size, 'status': 'downloaded'}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(download, files))
(DEST / 'DOWNLOAD.json').write_text(json.dumps({
    'repo': 'Qwen/Qwen3-0.6B', 'source': 'ModelScope',
    'revision': 'master', 'hash_verification': False, 'files': results,
}, ensure_ascii=False, indent=2))
print(f'Complete: {DEST}', flush=True)
