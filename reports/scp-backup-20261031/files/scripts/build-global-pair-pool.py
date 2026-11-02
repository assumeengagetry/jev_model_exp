#!/usr/bin/env python3
"""Merge the complete actual pools, exact-deduplicate, and preserve cached typed IDs."""
import array
import collections
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pair_data_common import DOMAINS, NO_TRAIN, domain_for, text

HOME = Path('/home/dev/assumeengage')
TYPED = HOME / 'cache/pair-training/normalized/typed-full'
TYPED_TOKENS = HOME / 'cache/pair-training/tokenized/typed-full'
BASE = HOME / 'datasets/zero-shot-label-nli-http'
OVERNIGHT = HOME / 'datasets/overnight-supplement'
EXTRA = HOME / 'cache/pair-training/extra-pairs'
GENERATED = HOME / 'cache/pair-training/generated-pairs'
OUT = HOME / 'cache/pair-training/global-pool'
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'MANIFEST.json').exists() and json.loads((OUT / 'MANIFEST.json').read_text()).get('complete'):
    print('Reusing complete global pool:', OUT, flush=True); raise SystemExit(0)
for path in [BASE / 'DOWNLOAD.json', TYPED / 'MANIFEST.json', TYPED_TOKENS / 'MANIFEST.json', EXTRA / 'MANIFEST.json', GENERATED / 'MANIFEST.json']:
    if not path.exists() or not json.loads(path.read_text()).get('complete'):
        raise RuntimeError('Required prepared source is not complete: ' + str(path))

start = time.monotonic()
overrides = json.loads((HOME / 'config/source-domain-overrides.json').read_text())
input_map = {}; input_values = []; candidate_map = {}; candidate_values = []
source_map = {}; source_values = []; group_map = {}; group_values = []
seen = {}
state_ids = array.array('I'); candidate_ids = array.array('I'); sums = array.array('d')
repeats = array.array('I'); domain_ids = array.array('B'); source_ids = array.array('I'); group_ids = array.array('I')
counts = collections.Counter(); by_source = collections.defaultdict(collections.Counter); inputs_used = []


def batches(path, size=8192):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=size): yield batch.to_pylist()


for batch in batches(TYPED / 'inputs.parquet'):
    for row in batch:
        key = (row['state'], row['question'], row['criteria']); input_map[key] = len(input_values); input_values.append(key)
for batch in batches(TYPED / 'candidates.parquet'):
    for row in batch:
        value = row['candidate']; candidate_map[value] = len(candidate_values); candidate_values.append(value)
prefix_boundary, suffix_boundary = len(input_values), len(candidate_values)
print('Preserved cached typed IDs:', prefix_boundary, 'prefixes,', suffix_boundary, 'candidates', flush=True)


def intern_input(state, question='', criteria=''):
    key = (text(state), text(question), text(criteria)); value = input_map.get(key)
    if value is None: value = len(input_values); input_map[key] = value; input_values.append(key)
    return value


def intern_candidate(candidate):
    candidate = text(candidate); value = candidate_map.get(candidate)
    if value is None: value = len(candidate_values); candidate_map[candidate] = value; candidate_values.append(candidate)
    return value


def add(state, candidate, label, domain, source, group, source_repo=''):
    if not 0 <= label <= 1: counts['invalid_labels_excluded'] += 1; return
    sid = source_map.get(source)
    if sid is None:
        sid = len(source_values); source_map[source] = sid
        source_values.append({'name': source, 'domain': domain, 'source_repo': source_repo})
    by_source[source]['pairs_before_global_dedup'] += 1
    key = (int(state) << 32) | int(candidate)
    previous = seen.get(key)
    if previous is not None:
        if abs(sums[previous] / repeats[previous] - label) > 1e-6: counts['repeated_inputs_with_different_targets'] += 1
        if domain_ids[previous] != DOMAINS.index(domain): counts['cross_domain_duplicates_kept_in_first_domain'] += 1
        sums[previous] += label; repeats[previous] += 1
        counts['global_duplicates_merged'] += 1; by_source[source]['duplicates_merged'] += 1
        return
    group_key = (sid, str(group)); gid = group_map.get(group_key)
    if gid is None: gid = len(group_values); group_map[group_key] = gid; group_values.append(group_key)
    seen[key] = len(state_ids)
    state_ids.append(state); candidate_ids.append(candidate); sums.append(label); repeats.append(1)
    domain_ids.append(DOMAINS.index(domain)); source_ids.append(sid); group_ids.append(gid)
    by_source[source]['retained_unique_pairs'] += 1
    if len(state_ids) % 1000000 == 0:
        print('Global unique pairs:', len(state_ids), 'new prefixes:', len(input_values)-prefix_boundary,
              'elapsed_s:', round(time.monotonic()-start), flush=True)


# Canonical base is ingested first, so duplicate/domain precedence remains explicit.
base_files = sorted((BASE / 'data').glob('train-*.parquet'))
if len(base_files) != 5: raise RuntimeError('All five canonical default train shards are required')
for path in base_files:
    inputs_used.append(str(path.relative_to(HOME)))
    for batch in batches(path):
        for row in batch:
            index = counts['canonical_base_raw_rows']; counts['canonical_base_raw_rows'] += 1
            source_task = text(row['task']); raw_label = int(row['labels'])
            if raw_label not in (0,1,2): counts['canonical_invalid_labels'] += 1; continue
            if NO_TRAIN.search(source_task): counts['canonical_no_official_train_source_rows_excluded'] += 1; continue
            state, candidate = intern_input(row['premise']), intern_candidate(row['hypothesis'])
            domain = overrides.get(source_task, domain_for(source_task))
            add(state, candidate, float(raw_label == 0), domain, 'canonical_nli/' + source_task,
                index, 'tasksource/zero-shot-label-nli default@9f0763b798363a6d4ca8f65de2dc80e0fcb48ae2')
if counts['canonical_base_raw_rows'] != 1899993:
    raise RuntimeError('The canonical full base row count differs from the prepared source manifest')
print('Canonical base fully incorporated:', counts['canonical_base_raw_rows'], flush=True)

for old_domain in DOMAINS:
    path = TYPED / (old_domain + '.parquet'); inputs_used.append(str(path.relative_to(HOME)))
    for batch in batches(path, 32768):
        for row in batch:
            task = row['source']; domain = overrides.get(task, domain_for(task))
            add(row['state_id'], row['candidate_id'], float(row['target']), domain,
                'typed_full/' + task, row['row_id'], 'saiting/tasksource-jev-typed-decisions full snapshot')
print('Typed full pairs incorporated without re-tokenizing cached text.', flush=True)

flat_sources = []
for root in [OVERNIGHT / 'pairs', EXTRA, GENERATED]:
    flat_sources.extend(sorted(root.rglob('*.parquet')))
for path in flat_sources:
    if 'evaluation-only' in path.parts: raise RuntimeError('Evaluation input is not permitted in training')
    inputs_used.append(str(path.relative_to(HOME)))
    for batch in batches(path):
        for row in batch:
            label = float(row['label']); domain = row['domain'].lower()
            if domain not in DOMAINS: raise RuntimeError('Unknown explicit training domain')
            source = ('overnight/' if OVERNIGHT in path.parents else 'generated/' if GENERATED in path.parents else 'native/') + row['source']
            state = intern_input(row['state'], row.get('question',''), row.get('criteria',''))
            candidate = intern_candidate(row['candidate'])
            add(state, candidate, label, domain, source, row.get('group_id',''), row.get('source_repo',''))
    print('Incorporated actual training source:', path.relative_to(HOME), flush=True)

new_inputs_schema = pa.schema([('state',pa.string()),('question',pa.string()),('criteria',pa.string())])
with pq.ParquetWriter(OUT / 'inputs.parquet.preparing',new_inputs_schema,compression='zstd') as writer:
    for offset in range(prefix_boundary,len(input_values),32768):
        chunk=input_values[offset:offset+32768]
        writer.write_table(pa.table({'state':[r[0] for r in chunk],'question':[r[1] for r in chunk],'criteria':[r[2] for r in chunk]},schema=new_inputs_schema))
(OUT / 'inputs.parquet.preparing').replace(OUT / 'inputs.parquet')
with pq.ParquetWriter(OUT / 'candidates.parquet.preparing',pa.schema([('candidate',pa.string())]),compression='zstd') as writer:
    for offset in range(suffix_boundary,len(candidate_values),65536):
        writer.write_table(pa.table({'candidate':candidate_values[offset:offset+65536]}))
(OUT / 'candidates.parquet.preparing').replace(OUT / 'candidates.parquet')

arrays = {'state': np.asarray(state_ids,dtype=np.uint32),'candidate':np.asarray(candidate_ids,dtype=np.uint32),
          'target':(np.asarray(sums)/np.asarray(repeats)).astype(np.float32),
          'source':np.asarray(source_ids,dtype=np.uint32),'group':np.asarray(group_ids,dtype=np.uint32)}
domain_array=np.asarray(domain_ids,dtype=np.uint8);domain_counts={}
pair_dir=OUT/'pairs';pair_dir.mkdir(exist_ok=True)
for index,domain in enumerate(DOMAINS):
    selected=np.flatnonzero(domain_array==index);domain_counts[domain]=len(selected)
    for name,values in arrays.items():np.save(pair_dir/f'{domain}_{name}.npy',values[selected])
    print('Final unique domain pool:',domain,len(selected),flush=True)
pq.write_table(pa.table({'source_id':[row[0] for row in group_values],'original_group_id':[row[1] for row in group_values]}),OUT/'groups.parquet',compression='zstd')
(OUT/'SOURCES.json').write_text(json.dumps({'sources':source_values,'counts':{key:dict(value) for key,value in by_source.items()}},ensure_ascii=False,indent=2))
manifest=dict(complete=True,format_version='candidate-pair-v1',primary_base='tasksource/zero-shot-label-nli default',
              primary_label_mapping={'0_entailment':1.0,'1_neutral':0.0,'2_contradiction':0.0},
              canonical_base_raw_rows=counts['canonical_base_raw_rows'],counts=dict(counts),domain_pairs=domain_counts,
              total_pairs=len(state_ids),new_unique_inputs=len(input_values)-prefix_boundary,
              new_unique_candidates=len(candidate_values)-suffix_boundary,typed_prefix_boundary=prefix_boundary,
              typed_suffix_boundary=suffix_boundary,typed_normalized=str(TYPED),typed_tokens=str(TYPED_TOKENS),
              new_tokens=str(HOME/'cache/pair-training/tokenized/global-new'),source_files=inputs_used,
              full_epoch_required=True,pair_cap=None,training_token_cap=None,
              mixed_all_exposure='exact concatenation of all five final unique pools once without replacement',
              extra_file_hash_checks=False,model_tests=False,elapsed_seconds=time.monotonic()-start)
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Global complete pool ready:',domain_counts,'elapsed_s:',round(manifest['elapsed_seconds']),flush=True)
