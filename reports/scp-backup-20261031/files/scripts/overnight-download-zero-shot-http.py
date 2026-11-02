#!/usr/bin/env python3
"""Download the exact canonical NLI train shards via HTTP, without XET rebuilds."""

import concurrent.futures
import json
import socket
import threading
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection

from download_utils import HF_ENDPOINT, direct_hf_download

HOME = Path('/home/dev/assumeengage')
DEST = HOME / 'datasets/zero-shot-label-nli-http'
META = json.loads((HOME / 'config/hub-metadata/tasksource__zero-shot-label-nli.json').read_text())
REVISION = META['sha']
NAMES = [r['rfilename'] for r in META['siblings']
         if r['rfilename'].startswith('data/train-') and r['rfilename'].endswith('.parquet')]
LOCAL = threading.local()
RESULTS = []
DEST.mkdir(parents=True, exist_ok=True)


class PodHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        kwargs['socket_options'] = HTTPConnection.default_socket_options + [
            (socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1024),
        ]
        return super().init_poolmanager(connections, maxsize, block=block, **kwargs)


def client():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.mount('https://', PodHTTPAdapter())
    return LOCAL.session


def fetch(name):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {'file': name, 'status': 'present', 'bytes': target.stat().st_size}
    partial = target.with_name(target.name + '.http.part')
    failures = []
    for endpoint in ['https://hf-mirror.com', 'https://hf-cdn.sufy.com']:
        try:
            url = f'{endpoint}/datasets/tasksource/zero-shot-label-nli/resolve/{REVISION}/{name}?download=true'
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            print('START', endpoint, name, 'resume_bytes', offset, flush=True)
            with client().get(url, headers=headers, stream=True, timeout=(12, 45)) as response:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '')
                if content_type.startswith('text/') or 'application/json' in content_type:
                    raise OSError('Mirror returned a message rather than a training shard')
                expected = response.headers.get('Content-Length')
                total = response.headers.get('Content-Range', '').rsplit('/', 1)[-1]
                received = 0
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                with partial.open(mode) as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                if expected is not None and received != int(expected):
                    raise OSError('Incomplete HTTP body; resumable partial retained')
                if not partial.stat().st_size or (total.isdigit() and partial.stat().st_size != int(total)):
                    raise OSError('Incomplete training shard; resumable partial retained')
            partial.replace(target)
            print('READY', name, target.stat().st_size, endpoint, flush=True)
            return {'file': name, 'status': 'downloaded', 'bytes': target.stat().st_size,
                    'transport': endpoint, 'revision': REVISION}
        except Exception as error:
            failures.append({'endpoint': endpoint, 'error_type': type(error).__name__})
            print('RETRY', name, endpoint, type(error).__name__, flush=True)
    try:
        print('ORIGIN_FALLBACK', name, flush=True)
        direct_hf_download(f'{HF_ENDPOINT}/datasets/tasksource/zero-shot-label-nli/resolve/{REVISION}/{name}', partial)
        partial.replace(target)
        return {'file': name, 'status': 'downloaded', 'bytes': target.stat().st_size,
                'transport': 'existing direct HTTP fallback', 'revision': REVISION}
    except Exception as error:
        failures.append({'endpoint': 'origin', 'error_type': type(error).__name__})
        return {'file': name, 'status': 'failed', 'failures': failures,
                'partial_bytes': partial.stat().st_size if partial.exists() else 0}


def main():
    print('Downloading all', len(NAMES), 'canonical default train shards at revision', REVISION, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, name) for name in NAMES]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            RESULTS.append(row)
            data = {'repo': 'tasksource/zero-shot-label-nli', 'config': 'default', 'split': 'train',
                    'revision': REVISION, 'extra_hash_checks': False, 'files': RESULTS,
                    'complete': len(RESULTS) == len(NAMES) and all(r['status'] != 'failed' for r in RESULTS)}
            temporary = DEST / 'DOWNLOAD.json.tmp'
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(DEST / 'DOWNLOAD.json')
            print('RESULT', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
