#!/usr/bin/env python3
"""Resume frozen evaluation gzip files with small independent HTTP ranges."""

import concurrent.futures as futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from download_utils import direct_hf_request

HOME = Path('/home/dev/assumeengage')
ROOT = HOME / 'datasets/evaluation-only/decision-index-suite-0.3'
REPO = 'adi060/decision-index-suite-0.2'
REVISION = 'ec174a4e7aa03e786a7e7e7e384db0f6c69a23e2'
# Published HTTP Content-Length values for this fixed revision, not checksums.
FILES = {'selected-rows.jsonl.gz': 71281743, 'added-rows.jsonl.gz': 10892541}
CHUNK = 32768


def download_file(name, size):
    target = ROOT / name
    if target.exists():
        return {'file': name, 'status': 'present', 'bytes': target.stat().st_size}
    part = ROOT / (name + '.range.part')
    progress = ROOT / (name + '.range-progress.json')
    completed = set()
    if progress.exists():
        state = json.loads(progress.read_text())
        if state.get('revision') != REVISION or state.get('chunk_bytes') != CHUNK:
            raise RuntimeError('Preserving a range cache belonging to another revision/format')
        completed = set(state['completed_chunks'])
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    prefix = ROOT / (name + '.download.part')
    if not completed and prefix.exists():
        with prefix.open('rb') as source:
            offset = 0
            while True:
                data = source.read(CHUNK)
                if not data:
                    break
                os.pwrite(fd, data, offset)
                if len(data) == CHUNK:
                    completed.add(offset // CHUNK)
                offset += len(data)
    root_url = f'https://hf-mirror.com/datasets/{REPO}/resolve/{REVISION}/{name}?download=true'
    try:
        # This mirror rejects dotted SNI; resolve it with its normal TLS name,
        # then use the existing ranged transport only for the CDN object.
        with requests.get(root_url, stream=True, headers={'Accept-Encoding': 'identity'},
                          timeout=(10, 30)) as response:
            response.raise_for_status()
            signed_url = response.url
    except Exception as error:
        os.close(fd)
        return {'file': name, 'status': 'partial', 'error_type': type(error).__name__,
                'stage': 'resolve_signed_object'}
    total_chunks = (size + CHUNK - 1) // CHUNK
    initial = len(completed)
    deadline = time.monotonic() + 1200
    errors = 0

    def piece(index):
        start = index * CHUNK
        end = min(size, start + CHUNK) - 1
        for attempt in range(2):
            response = connection = None
            try:
                response, connection = direct_hf_request(signed_url, {'Range': f'bytes={start}-{end}'})
                if response.status != 206:
                    raise OSError('Object did not honor the requested byte range')
                if response.getheader('content-range', '') != f'bytes {start}-{end}/{size}':
                    raise OSError('HTTP response describes a different byte range')
                block = bytearray()
                while len(block) < end - start + 1:
                    chunk = response.read1(end - start + 1 - len(block))
                    if not chunk:
                        raise OSError('Incomplete ranged HTTP body')
                    block.extend(chunk)
                written = 0
                while written < len(block):
                    written += os.pwrite(fd, block[written:], start + written)
                return index, True
            except Exception:
                if attempt == 1:
                    return index, False
            finally:
                if response is not None:
                    response.close()
                if connection is not None:
                    connection.close()

    def checkpoint():
        state = {'repo': REPO, 'revision': REVISION, 'file': name,
                 'chunk_bytes': CHUNK, 'completed_chunks': sorted(completed), 'total_chunks': total_chunks}
        temporary = progress.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state))
        temporary.replace(progress)

    todo = iter(i for i in range(total_chunks) if i not in completed)
    print('SMALL_RANGES', name, 'ready', initial, '/', total_chunks, flush=True)
    pool = futures.ThreadPoolExecutor(max_workers=16)
    pending = {}
    try:
        for _ in range(16):
            index = next(todo, None)
            if index is not None:
                pending[pool.submit(piece, index)] = index
        while pending and time.monotonic() < deadline:
            ready, _ = futures.wait(pending, timeout=10, return_when=futures.FIRST_COMPLETED)
            for future in ready:
                pending.pop(future)
                index, success = future.result()
                if success:
                    completed.add(index)
                else:
                    errors += 1
                if len(completed) % 32 == 0 or success and len(completed) == total_chunks:
                    checkpoint()
                    print('RANGE_PROGRESS', name, len(completed), '/', total_chunks, 'failed', errors, flush=True)
                if errors >= 32 and len(completed) == initial:
                    deadline = 0
                    break
                index = next(todo, None)
                if index is not None:
                    pending[pool.submit(piece, index)] = index
        for future in pending:
            future.cancel()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        checkpoint()
        os.close(fd)
    if len(completed) == total_chunks:
        part.replace(target)
        return {'file': name, 'status': 'downloaded', 'bytes': size,
                'transport': 'independent 32KiB byte ranges', 'hash_checks': False}
    return {'file': name, 'status': 'partial', 'ready_chunks': len(completed),
            'total_chunks': total_chunks, 'failed_chunks': errors}


def main():
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: download_file(*item), FILES.items()))
    manifest = json.loads((ROOT / 'DOWNLOAD.json').read_text())
    previous = {row['file']: row for row in manifest['files']}
    previous.update({row['file']: row for row in results})
    manifest['files'] = list(previous.values())
    manifest['source_files_complete'] = all((ROOT / name).exists() for name in [
        'selected-rows.jsonl.gz', 'added-rows.jsonl.gz', 'excluded-questions.json', 'mirror-0.2-manifest.json'])
    manifest['complete'] = False
    (ROOT / 'DOWNLOAD.json').write_text(json.dumps(manifest, indent=2))
    print('RANGE_RESULTS', json.dumps(results), flush=True)
    if manifest['source_files_complete']:
        raise SystemExit(subprocess.run([sys.executable, str(HOME / 'scripts/finalize-frozen-evaluation-data.py')]).returncode)


if __name__ == '__main__':
    main()
