#!/usr/bin/env python3
"""Acquire frozen official evaluation inputs; never evaluate a model or hash files."""

import concurrent.futures
import json
import socket
import sys
from urllib.parse import urlsplit
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection

from download_utils import HF_ENDPOINT, direct_hf_download

HOME = Path('/home/dev/assumeengage')
sys.path.insert(0, str(HOME / 'src/decision-index'))
from decision_index import editions

EDITION = '0.3'
REPO = editions.get(EDITION)['dataset']
ROOT = HOME / 'datasets/evaluation-only/decision-index-suite-0.3'
MIRROR = 'https://hf-mirror.com'
ROOT.mkdir(parents=True, exist_ok=True)
HEADERS = {'Accept-Encoding': 'identity'}


class PodAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        kwargs['socket_options'] = HTTPConnection.default_socket_options + [
            (socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1024),
        ]
        return super().init_poolmanager(connections, maxsize, block=block, **kwargs)


def client():
    s = requests.Session()
    s.mount('https://', PodAdapter())
    return s


def main():
    session = client()
    source_repo = REPO
    source_edition = EDITION
    response = session.get(f'{MIRROR}/api/datasets/{source_repo}', timeout=(12, 45))
    if response.status_code in (401, 403):
        token_file = HOME / '.secrets/hf_token'
        if token_file.exists():
            HEADERS['Authorization'] = 'Bearer ' + token_file.read_text().strip()
            response = session.get(f'{MIRROR}/api/datasets/{source_repo}', headers=HEADERS, timeout=(12, 45))
    if response.status_code == 404:
        # The kit documents the official dataset name as a private staging
        # example. Reuse publicly available frozen 0.2 base rows, but do not
        # pretend the 0.3 GSM8K replacement has already been prepared.
        source_repo = 'adi060/decision-index-suite-0.2'
        source_edition = '0.2'
        response = session.get(f'{MIRROR}/api/datasets/{source_repo}', headers=HEADERS, timeout=(12, 45))
    response.raise_for_status()
    revision = response.json()['sha']
    results = []

    def fetch(name):
        target_name = 'mirror-0.2-manifest.json' if source_edition == '0.2' and name == 'manifest.json' else name
        target = ROOT / target_name
        if target.exists():
            return {'file': target_name, 'status': 'present', 'bytes': target.stat().st_size}
        part = target.with_name(target.name + '.download.part')
        if name.endswith('.gz'):
            # These uncached gzip objects redirect to us.aws.cdn.hf.co, where
            # whole-body requests made no progress. Reuse the existing IPv4,
            # FQDN/SNI and bounded-range transport instead of retrying that path.
            try:
                print('RANGED_ORIGIN_TRANSFER', target_name, flush=True)
                direct_hf_download(f'{HF_ENDPOINT}/datasets/{source_repo}/resolve/{revision}/{name}', part)
                part.replace(target)
                return {'file': target_name, 'status': 'downloaded', 'bytes': target.stat().st_size,
                        'transport': 'existing IPv4 ranged HTTP transport'}
            except Exception as error:
                return {'file': target_name, 'status': 'failed', 'error_type': type(error).__name__,
                        'partial_bytes': part.stat().st_size if part.exists() else 0,
                        'transport': 'existing IPv4 ranged HTTP transport'}
        failures = []
        for attempt, endpoint in enumerate([MIRROR, MIRROR, 'https://hf-cdn.sufy.com', 'https://hf-cdn.sufy.com']):
            try:
                offset = part.stat().st_size if part.exists() else 0
                headers = dict(HEADERS)
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                url = f'{endpoint}/datasets/{source_repo}/resolve/{revision}/{name}?download=true'
                print('TRANSFER', target_name, 'attempt', attempt + 1, 'resume_bytes', offset, flush=True)
                with client().get(url, headers=headers, stream=True, timeout=(12, 90)) as r:
                    r.raise_for_status()
                    print('HTTP', target_name, 'host', urlsplit(r.url).hostname,
                          'encoding', r.headers.get('Content-Encoding'),
                          'length', r.headers.get('Content-Length'), flush=True)
                    expected = r.headers.get('Content-Length')
                    received = 0
                    with part.open('ab' if offset and r.status_code == 206 else 'wb') as out:
                        # Preserve stored .gz bytes even if a mirror sets an
                        # HTTP Content-Encoding header for an already-gzipped file.
                        for chunk in r.raw.stream(1024 * 1024, decode_content=False):
                            out.write(chunk)
                            received += len(chunk)
                    if expected is not None and received != int(expected):
                        raise OSError('Incomplete HTTP transfer; partial retained')
                    total = r.headers.get('Content-Range', '').rsplit('/', 1)[-1]
                    if not part.stat().st_size or (total.isdigit() and part.stat().st_size != int(total)):
                        raise OSError('Incomplete evaluation file; partial retained')
                part.replace(target)
                return {'file': target_name, 'status': 'downloaded', 'bytes': target.stat().st_size}
            except Exception as error:
                request = getattr(error, 'request', None)
                failures.append({'error_type': type(error).__name__,
                                 'request_host': urlsplit(request.url).hostname if request is not None else endpoint,
                                 'partial_bytes': part.stat().st_size if part.exists() else 0})
                print('RETRY', target_name, json.dumps(failures[-1]), flush=True)
        return {'file': target_name, 'status': 'failed', 'failures': failures}

    names = editions.files(source_edition)
    print('Acquiring frozen evaluation inputs:', source_repo, revision, list(names), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch, name) for name in names]):
            row = future.result()
            results.append(row)
            print(json.dumps(row), flush=True)
            source_complete = len(results) == len(names) and all(r['status'] != 'failed' for r in results)
            manifest = {'role': 'EVALUATION_ONLY_NEVER_TRAIN', 'repo': source_repo, 'revision': revision,
                        'edition': EDITION, 'source_edition': source_edition, 'files': results,
                        'hash_checks': False, 'model_tests_run': False,
                        'source_files_complete': source_complete,
                        'pending': ['0.3 corrected GSM8K rows and local edition manifest'] if source_edition != EDITION else [],
                        'complete': source_complete and source_edition == EDITION}
            temporary = ROOT / 'DOWNLOAD.json.tmp'
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(ROOT / 'DOWNLOAD.json')


if __name__ == '__main__':
    main()
