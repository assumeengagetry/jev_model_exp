#!/usr/bin/env python3
"""Download real training shards using HF's official chunk transfer protocol."""
import argparse
import json
import os
from pathlib import Path

HOME = Path('/home/dev/assumeengage')
os.environ.pop('HF_HUB_DISABLE_XET', None)
os.environ['HF_HOME'] = str(HOME / 'cache/huggingface')
os.environ['HF_XET_CACHE'] = str(HOME / 'cache/huggingface/xet')
os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
os.environ['HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY'] = '1'
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'

import requests
from requests.structures import CaseInsensitiveDict
from hf_xet import PyXetDownloadInfo, download_files
from huggingface_hub.utils._xet import parse_xet_file_data_from_response, parse_xet_connection_info_from_headers
from download_utils import direct_hf_request, hf_metadata

parser = argparse.ArgumentParser()
parser.add_argument('--repo', default='tasksource/zero-shot-label-nli')
parser.add_argument('--prefix', nargs='+', default=['data/train-'])
parser.add_argument('--destination', default=str(HOME / 'datasets/zero-shot-label-nli'))
args = parser.parse_args()
destination = Path(args.destination)
destination.mkdir(parents=True, exist_ok=True)
metadata = hf_metadata(args.repo)
revision = metadata['sha']
selected = [entry['rfilename'] for entry in metadata['siblings']
            if any(entry['rfilename'].startswith(prefix) for prefix in args.prefix)
            and entry['rfilename'].endswith('.parquet')]
if not selected:
    raise RuntimeError('No requested training shards found; refusing an empty download manifest')
infos = []
files = []
refresh_route = None
print(f'Downloading {len(selected)} full training shards from {args.repo}; no sampling or model tests.', flush=True)
for name in selected:
    url = f'https://huggingface.co/datasets/{args.repo}/resolve/{revision}/{name}'
    response, connection = direct_hf_request(url, method='HEAD', follow_redirects=False)
    try:
        wrapped = requests.Response()
        wrapped.status_code = response.status
        wrapped.url = url
        wrapped.headers = CaseInsensitiveDict(response.getheaders())
        data = parse_xet_file_data_from_response(wrapped, endpoint='https://huggingface.co')
        size = int(wrapped.headers['X-Linked-Size'])
        if data is None: raise RuntimeError('Official chunk metadata missing for ' + name)
        refresh_route = data.refresh_route
    finally:
        response.close(); connection.close()
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {'file': name, 'bytes': size, 'present': target.exists() and target.stat().st_size == size}
    files.append(row)
    if not row['present']:
        part = target.with_name(target.name + '.xet.part')
        infos.append(PyXetDownloadInfo(destination_path=str(part), hash=data.file_hash, file_size=size))


def connection_info():
    response, connection = direct_hf_request(refresh_route)
    try:
        if response.status != 200: raise RuntimeError('Chunk transfer authorization HTTP ' + str(response.status))
        info = parse_xet_connection_info_from_headers(CaseInsensitiveDict(response.getheaders()))
        if info is None: raise RuntimeError('Chunk transfer authorization metadata missing')
        return info
    finally:
        response.close(); connection.close()


if infos:
    initial = connection_info()
    def refresh():
        info = connection_info()
        return info.access_token, info.expiration_unix_epoch
    print('Starting official parallel chunk transfer.', flush=True)
    download_files(infos, endpoint=initial.endpoint,
                   token_info=(initial.access_token, initial.expiration_unix_epoch),
                   token_refresher=refresh, progress_updater=[lambda n: None for _ in infos])
for row in files:
    target = destination / row['file']
    if not row['present']:
        part = target.with_name(target.name + '.xet.part')
        if part.stat().st_size != row['bytes']:
            raise RuntimeError('Incomplete training shard: ' + row['file'])
        part.replace(target)
    row['status'] = 'ready'
(destination / 'DOWNLOAD.json').write_text(json.dumps({
    'repo': args.repo, 'revision': revision, 'complete': True,
    'transport': 'official hf_xet', 'extra_file_hash_checks': False, 'files': files,
}, ensure_ascii=False, indent=2))
print('Training shards downloaded:', destination, flush=True)
