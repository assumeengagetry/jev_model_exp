"""Full real sidecar reader admission; CPU/JSON only, no retokenization."""
import base64
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/tmp/opencode/jev-state-reader-admission-20261112-v1')
OLD = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
PRODUCT = Path('/tmp/opencode/jev-base-state-boundary-20261112-v2')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU reader admission only')
    started = time.monotonic()
    sys.path.insert(0, str(ROOT / 'source'))
    from bindings import read_json, stat, verify, write_json
    from case_runtime import CaseStore, Exposure
    from state_boundaries import BoundaryStore, bind_prepared
    original_cfg = OLD / 'seed42.json'
    original_cfg_pin = stat(original_cfg)
    cfg = read_json(original_cfg)
    cfg['state_boundary'] = {'root': str(PRODUCT / 'prepared-01'),
                             'complete': stat(PRODUCT / 'prepared-01/COMPLETE.json'),
                             'primary': stat(PRODUCT / 'FINAL_PRIMARY_VERIFICATION.json')}
    binding = bind_prepared(cfg)
    train = select = exposure = reader = None
    try:
        train = CaseStore(Path(cfg['data_root']) / 'data/train.jsonl', 'train')
        select = CaseStore(Path(cfg['data_root']) / 'data/select.jsonl', 'validation')
        reader = BoundaryStore(binding, train, select)
        exposure = Exposure(cfg['data_root'], 42, train)
        rows = exposure.macro(0)
        seen = 0
        for row in rows:
            case = train.get(row['case_id'])
            first, repeated = reader.for_case(case), reader.for_case(case)
            assert first == repeated and first is not repeated
            assert len(first) == len(case['fields'])
            seen += 1
        for case in select:
            cuts = reader.for_case(case, dev=True)
            assert len(cuts) == len(case['fields'])
        reader.check()
        verify([original_cfg_pin])
        assert 'torch' not in sys.modules and 'transformers' not in sys.modules
        sources = [stat(p) for p in sorted((ROOT / 'source').iterdir())]
        result = {'complete': True, 'scope': 'full-original-train-select-state-v2-reader-admission',
            'actual_full_roster_counts': binding['report']['counts'],
            'actual_indexed_cases': {k: len(v) for k, v in reader.index.items()},
            'actual_formal_exposure_train_lookups_with_repeat': seen, 'actual_select_lookups': len(select.offsets),
            'lossless_rows_candidates_EOS_unknown_and_coordinates_verified': True,
            'actual_stored_offsets_and_fresh_cut_lists_verified': True, 'sustained_guards_executed': True,
            'source_files': sources,
            'source_bytes': {p.name: base64.b64encode(p.read_bytes()).decode() for p in (ROOT / 'source').iterdir()},
            'executed_analysis_source': stat(Path(__file__)), 'original_config': original_cfg_pin,
            'prepared_complete': cfg['state_boundary']['complete'], 'prepared_primary': cfg['state_boundary']['primary'],
            'boundary_policy': binding['boundary_policy'], 'actual_seconds': time.monotonic() - started,
            'model_run': False, 'torch_imported': False, 'tokenizer_run': False,
            'D_confirm_opened': False, 'public_run': False, 'model_forward_calls': 0,
            'new_arm_training_admitted': False,
            'unexecuted_branches': ['manufactured malformed/missing/extra/mutated-pin failures not injected',
                                   'new integrated trainer/fresh/formal macro0 not yet executed']}
        write_json(ROOT / 'COMPLETE.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('source_bytes', 'source_files')}, indent=2))
    finally:
        for handle in (reader, exposure, select, train):
            if handle is not None:
                handle.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        with (ROOT / 'BLOCKED.json').open('x') as handle:
            json.dump({'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                       'partial_preserved': True, 'model_run': False}, handle)
        raise
