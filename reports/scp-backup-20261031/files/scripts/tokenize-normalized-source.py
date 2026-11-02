#!/usr/bin/env python3
"""Cache full prefix/candidate tokens once, sharing repeated text across pairs."""
import argparse
import fcntl
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HOME = Path('/home/dev/assumeengage')
TOKENIZER = HOME / 'models/Qwen3-0.6B-common-pair-init-r32'
FORMAT = 'candidate-pair-v1-separated-prefix-suffix-eos'
_tokenizer = None


def worker_init():
    global _tokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def encode(job):
    role, rows = job
    if role == 'prefix':
        from pair_data_common import render_pair
        texts = [render_pair(row['state'], row['question'], '', row['criteria'])[0] for row in rows]
    else:
        texts = [row['candidate'] for row in rows]
    return _tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']


def available_workers():
    cores = len(os.sched_getaffinity(0))
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max': cores = min(cores, max(1, int(quota) // int(period)))
    except (OSError, ValueError): pass
    return max(1, min(32, cores - 2))


def cache_role(source, output, role, pool):
    filename = source / ('inputs.parquet' if role == 'prefix' else 'candidates.parquet')
    parquet = pq.ParquetFile(filename)
    signature = {'format': FORMAT, 'source_rows': parquet.metadata.num_rows,
                 'source_bytes': filename.stat().st_size, 'source_mtime_ns': filename.stat().st_mtime_ns}
    done = output / (role + '.DONE.json')
    if done.exists():
        metadata = json.loads(done.read_text())
        if any(metadata.get(key) != value for key, value in signature.items()):
            raise RuntimeError('Existing token cache metadata differs; preserving it: ' + str(done))
        print('Reusing', role, 'token cache:', metadata['rows'], 'texts', flush=True)
        return metadata
    progress = output / (role + '.PROGRESS.json')
    state = json.loads(progress.read_text()) if progress.exists() else dict(signature, rows=0, tokens=0)
    if any(state.get(key) != value for key, value in signature.items()):
        raise RuntimeError('Existing resumable token cache belongs to different source metadata')
    resumed_rows, tokens = state['rows'], state['tokens']
    ids_path = output / (role + '_ids.bin')
    offsets_path = output / (role + '_offsets.bin')
    ids = ids_path.open('r+b' if ids_path.exists() else 'w+b', buffering=8 * 1024 * 1024)
    offsets = offsets_path.open('r+b' if offsets_path.exists() else 'w+b', buffering=1024 * 1024)
    ids.truncate(tokens * 4); ids.seek(0, 2)
    offsets.truncate((resumed_rows + 1) * 8)
    if resumed_rows == 0:
        offsets.seek(0); offsets.write(np.array([0], dtype=np.uint64).tobytes())
    offsets.seek(0, 2)
    started = time.monotonic()
    processed = resumed_rows
    max_length = state.get('max_length', 0)

    def jobs():
        skipped = 0
        for batch in parquet.iter_batches(batch_size=512):
            if skipped + len(batch) <= resumed_rows:
                skipped += len(batch); continue
            rows = batch.to_pylist()
            if skipped < resumed_rows: rows = rows[resumed_rows - skipped:]
            skipped += len(batch)
            if rows: yield role, rows

    try:
        for batch_number, encoded in enumerate(pool.imap(encode, jobs(), chunksize=1), 1):
            lengths = np.fromiter((len(row) for row in encoded), dtype=np.uint64)
            flat = np.fromiter((value for row in encoded for value in row), dtype=np.int32,
                               count=int(lengths.sum()))
            ids.write(flat.tobytes())
            cumulative = np.cumsum(lengths) + tokens
            offsets.write(cumulative.astype(np.uint64).tobytes())
            tokens += int(lengths.sum()); processed += len(encoded)
            if len(lengths): max_length = max(max_length, int(lengths.max()))
            if batch_number % 32 == 0 or processed == parquet.metadata.num_rows:
                ids.flush(); offsets.flush()
                state = dict(signature, rows=processed, tokens=tokens, max_length=max_length)
                temporary = progress.with_suffix('.json.tmp'); temporary.write_text(json.dumps(state)); temporary.replace(progress)
                elapsed = max(0.001, time.monotonic() - started)
                print(role, 'texts', processed, '/', parquet.metadata.num_rows, 'cached_tokens', tokens,
                      'new_texts_per_s', round((processed - resumed_rows) / elapsed), flush=True)
    finally:
        ids.close(); offsets.close()
    if processed != parquet.metadata.num_rows: raise RuntimeError('Tokenization did not finish every source text')
    state = dict(signature, rows=processed, tokens=tokens, max_length=max_length)
    done.write_text(json.dumps(state, indent=2))
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(HOME / 'cache/pair-training/normalized/typed-full'))
    parser.add_argument('--output', default=str(HOME / 'cache/pair-training/tokenized/typed-full'))
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    if not manifest.get('complete'): raise RuntimeError('Source conversion is not complete')
    output.mkdir(parents=True, exist_ok=True)
    # Coordination and execution sessions may launch the same resumable cache.
    # Only one process may truncate/append these shared token files at a time.
    cache_lock = (output / 'TOKENIZATION.lock').open('a+')
    try:
        fcntl.flock(cache_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Tokenization cache is already being written; leaving the active job untouched:', output, flush=True)
        return
    cache_lock.seek(0); cache_lock.truncate(); cache_lock.write(str(os.getpid())); cache_lock.flush()
    workers = available_workers()
    print('Actual full-source tokenization, workers:', workers, 'without length truncation or pair caps.', flush=True)
    context = mp.get_context('spawn')
    with context.Pool(workers, initializer=worker_init) as pool:
        prefix = cache_role(source, output, 'prefix', pool)
        suffix = cache_role(source, output, 'suffix', pool)
    (output / 'MANIFEST.json').write_text(json.dumps({
        'complete': True, 'source': str(source), 'format': FORMAT, 'tokenizer': str(TOKENIZER),
        'prefix': prefix, 'suffix': suffix, 'source_domain_pairs': manifest.get('domain_pairs'),
        'pair_cap': None, 'tokenization_truncation': False, 'extra_file_hash_checks': False,
    }, ensure_ascii=False, indent=2))
    print('Reusable full prefix/candidate token cache ready:', output, flush=True)


if __name__ == '__main__': main()
