#!/usr/bin/env python3
"""Stage 0.3 evaluation semantics without model tests or file/payload checksums.

Reuses published 0.2 frozen inputs and the official GSM8K option construction.
The official deterministic group/tiebreak algorithm is unchanged; optional
whole-file and payload integrity hash calculations are deliberately not called.
"""

import gzip
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import requests

HOME = Path('/home/dev/assumeengage')
sys.path.insert(0, str(HOME / 'src/decision-index'))
from decision_index.suite.build import gsm8k_v3 as gsm
from importlib import util

spec = util.spec_from_file_location('evaluation_download_transport', HOME / 'scripts/prepare-official-evaluation-data.py')
transport = util.module_from_spec(spec)
spec.loader.exec_module(transport)

ROOT = HOME / 'datasets/evaluation-only/decision-index-suite-0.3'
GSM_REVISION = '740312add88f781978c0658806c59bc2815b9866'


def main():
    download = json.loads((ROOT / 'DOWNLOAD.json').read_text())
    if not download.get('source_files_complete'):
        raise RuntimeError('Published frozen base inputs have not finished downloading')
    if (ROOT / 'LOCAL_PREPARATION.json').exists():
        previous = json.loads((ROOT / 'LOCAL_PREPARATION.json').read_text())
        if previous.get('complete'):
            print('Reusing already prepared evaluation inputs:', ROOT, flush=True)
            return
    test_file = ROOT / 'gsm8k-official-test.parquet'
    if not test_file.exists():
        url = f'https://hf-mirror.com/datasets/openai/gsm8k/resolve/{GSM_REVISION}/main/test-00000-of-00001.parquet'
        with transport.client().get(url, headers={'Accept-Encoding': 'identity'},
                                    stream=True, timeout=(12, 90)) as response:
            response.raise_for_status()
            partial = test_file.with_suffix('.parquet.part')
            received = 0
            with partial.open('wb') as target:
                for chunk in response.iter_content(1024 * 1024):
                    target.write(chunk)
                    received += len(chunk)
            expected = response.headers.get('Content-Length')
            if not received or (expected is not None and received != int(expected)):
                raise OSError('Incomplete official GSM8K test transfer')
            partial.replace(test_file)

    # Do not call gsm.build()/write(): they run optional file integrity hashes.
    test_rows = pq.read_table(test_file).to_pylist()
    golds = {i: gsm.gold_of(row['answer']) for i, row in enumerate(test_rows)}
    old_rows = gsm.base_rows(ROOT / 'selected-rows.jsonl.gz')
    if len(old_rows) != 2 * len(golds):
        raise RuntimeError('Frozen base rows do not contain both official GSM8K tracks')
    option_sets = gsm.option_sets(golds)
    output = ROOT / 'gsm8k-rows.jsonl.gz'
    temporary = output.with_suffix('.gz.preparing')
    prepared = 0
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as target:
        for row in old_rows:
            i = row['metadata']['provenance']['source_id']
            count = len(row['questions']['answer']['criteria'])
            gold = golds[i]
            if gsm.num(gold) != row['metadata']['gold_numeric']:
                raise RuntimeError('GSM8K source row identity differs from the published reference')
            ordered, correct = gsm.options(gold, option_sets[i][count], count, i)
            source_run_id = row['_evaluation']['run_id']
            new = dict(row)
            new['questions'] = {'answer': dict(row['questions']['answer'],
                                              criteria={f'option_{j}': value for j, value in enumerate(ordered)})}
            new['expected'] = {'answer': f'option_{correct}'}
            new['metadata'] = dict(row['metadata'], distractor_method=gsm.METHOD,
                                   answer_group={'id': option_sets[i]['group'],
                                                 'band': option_sets[i]['band'],
                                                 'source_ids': option_sets[i]['members']},
                                   replaces_run_id=source_run_id)
            evaluation = dict(row['_evaluation'])
            evaluation.pop('payload_sha256', None)
            evaluation.update(run_id=gsm.RUN_PREFIX + source_run_id, source_run_id=source_run_id,
                              source_path=gsm.SOURCE_PATH, edition='0.3')
            evaluation['proxy_tokens'] = max(1, len(json.dumps({'model': gsm.MODEL,
                                                              'state': new['state'],
                                                              'questions': new['questions']},
                                                             ensure_ascii=False)) // 4)
            new['_evaluation'] = evaluation
            target.write(json.dumps(new, ensure_ascii=False, separators=(',', ':')) + '\n')
            prepared += 1
    temporary.replace(output)

    manifest = {'name': 'Locally staged Decision Index public evaluation inputs',
                'edition': 'release-v3', 'base_edition': 'release-v2.1',
                'base_source_repo': download['repo'], 'base_source_revision': download['revision'],
                'gsm8k_source_repo': 'openai/gsm8k', 'gsm8k_source_revision': GSM_REVISION,
                'gsm8k_method': gsm.METHOD, 'gsm8k_requests': prepared,
                'file_hash_validation': False, 'payload_hash_fields_generated': False,
                'note': 'Published frozen 0.2 base/added rows plus the official 0.3 GSM8K semantics. '
                        'Deterministic option construction unchanged; byte-level parity is not claimed. '
                        'No model tests or optional integrity hashes were run.'}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    report = dict(manifest, role='EVALUATION_ONLY_NEVER_TRAIN', complete=True)
    (ROOT / 'LOCAL_PREPARATION.json').write_text(json.dumps(report, indent=2))
    download.update(complete=True, pending=[], local_preparation='LOCAL_PREPARATION.json',
                    gsm8k_requests=prepared, file_hash_validation=False)
    (ROOT / 'DOWNLOAD.json').write_text(json.dumps(download, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
