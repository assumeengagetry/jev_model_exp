#!/usr/bin/env python3
"""Prepare one shared CPU-only candidate cache for all formal evaluations."""

import collections
import gzip
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HOME = Path('/home/dev/assumeengage')
sys.path.insert(0, str(HOME / 'src/decision-index'))
from decision_index.suite.io import Suite
from pair_data_common import criteria_context, render_pair, text

SUITE = HOME / 'datasets/evaluation-only/decision-index-suite-0.3'
OUT = SUITE / 'candidate-cache/normalized'


def main():
    source_ready = json.loads((SUITE / 'LOCAL_PREPARATION.json').read_text())
    if not source_ready.get('complete'):
        raise RuntimeError('Formal evaluation inputs are not yet prepared')
    if (OUT / 'MANIFEST.json').exists() and json.loads((OUT / 'MANIFEST.json').read_text()).get('complete'):
        print('Reusing the completed formal evaluation candidate cache:', OUT, flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    input_map, inputs = {}, []
    candidate_map, candidates = {}, []
    counts = collections.Counter()
    benchmarks = collections.Counter()
    schema = pa.schema([('run_id', pa.string()), ('field', pa.string()), ('kind', pa.string()),
                        ('option_key', pa.string()), ('state_id', pa.uint32()), ('candidate_id', pa.uint32())])
    pair_temp = OUT / 'field-options.parquet.preparing'
    row_temp = OUT / 'rows-for-evaluation.jsonl.gz.preparing'
    batch = []
    with pq.ParquetWriter(pair_temp, schema, compression='zstd') as writer, \
            gzip.open(row_temp, 'wt', encoding='utf-8', compresslevel=1) as raw_rows:
        # Suite.rows applies official edition retirement/replacement/subsets;
        # no .verify() call or integrity hashes are performed.
        for row in Suite(SUITE, edition='0.3').rows(apply_exclusions=True):
            evaluation = row['_evaluation']
            run_id = evaluation['run_id']
            state = text(row['state'])
            raw_rows.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
            counts['requests'] += 1
            benchmarks[str(evaluation['catalog_id'])] += 1
            for field, question in row['questions'].items():
                kind = question['type']
                if kind not in ('choice', 'noul'):
                    raise RuntimeError('Unmapped evaluation question type: ' + kind)
                options = question.get('criteria')
                if kind == 'noul' and not options:
                    options = {'true': 'Yes.', 'false': 'No.'}
                if not isinstance(options, dict) or not options:
                    raise RuntimeError('Missing named evaluation criteria for ' + run_id + ':' + field)
                instructions = text(question.get('instructions', ''))
                values = [text(value) for value in options.values()]
                context = criteria_context(instructions, values)
                key = (state, instructions, context)
                state_id = input_map.get(key)
                if state_id is None:
                    state_id = len(inputs); input_map[key] = state_id; inputs.append(key)
                for option_key, value in options.items():
                    value = text(value)
                    candidate_id = candidate_map.get(value)
                    if candidate_id is None:
                        candidate_id = len(candidates); candidate_map[value] = candidate_id; candidates.append(value)
                    batch.append({'run_id': run_id, 'field': field, 'kind': kind,
                                  'option_key': option_key, 'state_id': state_id, 'candidate_id': candidate_id})
                    counts['candidate_pairs'] += 1
                    if len(batch) >= 16384:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema)); batch.clear()
                counts['fields'] += 1
                counts[kind + '_fields'] += 1
            if counts['requests'] % 10000 == 0:
                print('FORMAL_EVAL_CACHE', dict(counts), flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    pair_temp.replace(OUT / 'field-options.parquet')
    row_temp.replace(OUT / 'rows-for-evaluation.jsonl.gz')
    pq.write_table(pa.table({'state': [r[0] for r in inputs], 'question': [r[1] for r in inputs],
                            'criteria': [r[2] for r in inputs]}), OUT / 'inputs.parquet', compression='zstd')
    pq.write_table(pa.table({'candidate': candidates}), OUT / 'candidates.parquet', compression='zstd')
    manifest = {'complete': True, 'role': 'EVALUATION_ONLY_NEVER_TRAIN',
                'source': str(SUITE), 'semantic_input_edition': '0.3',
                'format': 'candidate-pair-v1-separated-prefix-suffix-eos',
                'counts': dict(counts), 'benchmark_requests': dict(benchmarks),
                'unique_inputs': len(inputs), 'unique_candidates': len(candidates),
                'options': 'Original per-field criterion keys are preserved in field-options.parquet.',
                'decoder': 'Use classifier logit(match)-logit(not_match); softmax within each field. '
                           'Choice returns named option; noul returns probability of true. '
                           'Original expected/scoring/group metadata is retained in rows-for-evaluation.jsonl.gz.',
                'model_tests_run': False, 'file_hash_validation': False,
                'scope': 'Same official exclusion/retirement/replacement/subset inputs for all 11 primary conditions.'}
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
