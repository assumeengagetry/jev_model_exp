#!/usr/bin/env python3
"""Convert the entire typed full pool, retaining all candidates and soft labels."""
import collections
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pair_data_common import DOMAINS, NO_TRAIN, criteria_context, domain_for, text

HOME = Path('/home/dev/assumeengage')
SOURCE = HOME / 'datasets/tasksource-jev-typed-decisions-ms'
OUTPUT = HOME / 'cache/pair-training/normalized/typed-full'
OUTPUT.mkdir(parents=True, exist_ok=True)
if (OUTPUT / 'MANIFEST.json').exists():
    previous = json.loads((OUTPUT / 'MANIFEST.json').read_text())
    if previous.get('complete'):
        print('Reusing complete full-pool normalization:', OUTPUT, flush=True)
        raise SystemExit(0)
if not json.loads((SOURCE / 'DOWNLOAD.json').read_text()).get('complete'):
    raise RuntimeError('Full typed mirror download is not complete')

start = time.monotonic()
inputs = {}
candidates = {}
input_values = []
candidate_values = []
pairs = {}
counts = collections.Counter()
tasks = {}
sources = sorted((SOURCE / 'full').glob('train-*.parquet'))
if not sources: raise RuntimeError('Full training shards are missing')

for file in sources:
    for batch in pq.ParquetFile(file).iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            counts['input_decisions'] += 1
            source = text(row.get('source', 'unknown'))
            split = text(row.get('split', 'train')).lower()
            if split in {'test', 'validation', 'valid', 'dev', 'iid_test', 'ood_test'}:
                counts['heldout_rows_excluded'] += 1; continue
            if NO_TRAIN.search(source):
                counts['no_official_train_source_rows_excluded'] += 1; continue
            domain = domain_for(source)
            state = text(row.get('state'))
            question = text(row.get('question'))
            kind = text(row.get('kind'))
            target = row.get('target') or []
            options = [text(value) for value in (row.get('options') or [])]
            if not state or not target:
                counts['unusable_rows'] += 1; continue
            if kind == 'noul':
                options = ['Yes.', 'No.']
                targets = [float(target[0]), 1.0 - float(target[0])]
            elif kind in {'choice', 'score'} and len(options) == len(target) and options:
                targets = [float(value) for value in target]
            else:
                counts['unsupported_rows'] += 1; continue
            context = criteria_context(question, options)
            input_key = (state, question, context)
            state_id = inputs.get(input_key)
            if state_id is None:
                state_id = len(input_values); inputs[input_key] = state_id; input_values.append(input_key)
            tasks.setdefault(source, {'domain': domain, 'decisions': 0, 'candidate_pairs_before_dedup': 0})
            tasks[source]['decisions'] += 1
            for option, probability in zip(options, targets):
                if not option or not 0.0 <= probability <= 1.0:
                    counts['invalid_candidate_labels'] += 1; continue
                candidate_id = candidates.get(option)
                if candidate_id is None:
                    candidate_id = len(candidate_values); candidates[option] = candidate_id; candidate_values.append(option)
                key = (state_id, candidate_id)
                counts['candidate_pairs_before_dedup'] += 1
                tasks[source]['candidate_pairs_before_dedup'] += 1
                if key in pairs:
                    item = pairs[key]
                    item[2] += probability; item[3] += 1
                    counts['duplicate_pairs_merged'] += 1
                    if item[0] != domain: counts['cross_domain_duplicate_pairs'] += 1
                else:
                    pairs[key] = [domain, source, probability, 1, text(row.get('id'))]
        if counts['input_decisions'] % 100000 < 8192:
            print('Converted decisions:', counts['input_decisions'], 'unique inputs:', len(input_values),
                  'unique candidate pairs:', len(pairs), 'elapsed_s:', round(time.monotonic() - start), flush=True)

# Exact input/criterion deduplication is part of the requested data preparation;
# no file-content checksums, model tests, pair caps, or token-budget short runs.
schema = pa.schema([('state_id', pa.uint32()), ('candidate_id', pa.uint32()), ('target', pa.float32()),
                    ('source', pa.string()), ('row_id', pa.string())])
writers = {domain: pq.ParquetWriter(OUTPUT / (domain + '.parquet'), schema, compression='zstd') for domain in DOMAINS}
buffers = {domain: [] for domain in DOMAINS}
domain_counts = collections.Counter()
try:
    for (state_id, candidate_id), item in pairs.items():
        domain, source, total, repeats, row_id = item
        buffers[domain].append({'state_id': state_id, 'candidate_id': candidate_id,
                                'target': total / repeats, 'source': source, 'row_id': row_id})
        domain_counts[domain] += 1
        if len(buffers[domain]) >= 32768:
            writers[domain].write_table(pa.Table.from_pylist(buffers[domain], schema=schema)); buffers[domain].clear()
    for domain in DOMAINS:
        if buffers[domain]: writers[domain].write_table(pa.Table.from_pylist(buffers[domain], schema=schema))
finally:
    for writer in writers.values(): writer.close()
pq.write_table(pa.table({'state': [r[0] for r in input_values], 'question': [r[1] for r in input_values],
                          'criteria': [r[2] for r in input_values]}), OUTPUT / 'inputs.parquet', compression='zstd')
pq.write_table(pa.table({'candidate': candidate_values}), OUTPUT / 'candidates.parquet', compression='zstd')
manifest = {'complete': True, 'source': str(SOURCE), 'source_config': 'full',
            'role': 'supplement; does not replace zero-shot-label-nli default',
            'all_native_candidates_retained': True, 'training_pair_cap': None,
            'counts': dict(counts), 'domain_pairs': dict(domain_counts), 'source_tasks': tasks,
            'unique_inputs': len(input_values), 'unique_candidates': len(candidate_values),
            'deduplication': 'exact (state, question, required criteria context, candidate); targets averaged on repeats',
            'format_version': 'candidate-pair-v1', 'elapsed_seconds': time.monotonic() - start,
            'extra_file_hash_checks': False, 'model_tests': False}
(OUTPUT / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print('Full-pool conversion complete:', dict(domain_counts), 'elapsed_s:', round(manifest['elapsed_seconds']), flush=True)
