"""Primary actual-artifact readback, independent offset split calculation."""
import base64
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/tmp/opencode/jev-base-state-boundary-20261112-v2')
DATA = Path('/tmp/opencode/jev-clean-origin-20261112-v1/prepared-01')
HEADER = 'Decide whether the candidate is a correct match.\nInput:\n'


def read(path):
    return json.loads(Path(path).read_text())


def meta(path):
    path = Path(path)
    s = path.stat()
    return {'path': str(path), 'resolved': str(path.resolve()), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def verify(pins):
    for pin in pins:
        assert meta(pin['path']) == pin, 'Actual artifact/input changed: ' + pin['path']


def independent_cut(text, state_end, offsets):
    # A different calculation from producer's left/right counter: validate the
    # full character-span union, then enumerate adjacent exact-endpoint joins.
    if not offsets or not 0 < state_end < len(text):
        return None
    frontier, previous = 0, -1
    for start, end in offsets:
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            return None
        if start < previous or start > frontier:
            return None
        previous, frontier = start, max(frontier, end)
    if frontier != len(text) or any(start < state_end < end for start, end in offsets):
        return None
    joins = [i for i in range(1, len(offsets)) if offsets[i - 1][1] == state_end == offsets[i][0]]
    if len(joins) != 1:
        return None
    k = joins[0]
    if any(end > state_end for start, end in offsets[:k]) or any(start < state_end for start, end in offsets[k:]):
        return None
    return k


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    started = time.monotonic()
    output = ROOT / 'prepared-01'
    assert not (output / 'BLOCKED.json').exists() and not (ROOT / 'FINAL_PRIMARY_VERIFICATION.json').exists()
    config_pin = meta(ROOT / 'projection-config.json')
    cfg = read(config_pin['path'])
    verify([config_pin, cfg['source']] + cfg['input_pins'])
    assert base64.b64encode(Path(cfg['source']['path']).read_bytes()).decode() == cfg['source_bytes']
    completion_pin = meta(output / 'COMPLETE.json')
    done, report = read(completion_pin['path']), read(output / 'PREPARATION.json')
    assert done['complete'] and report['complete_scan_and_readback'] and done['ready'] == report['ready']
    assert done['boundary_policy'] == report['boundary_policy'] == cfg['boundary_policy'] == 'state-plus-one-existing-newline-v2'
    assert done['config'] == report['config'] == config_pin and done['source'] == report['source'] == cfg['source']
    verify(done['files'])
    assert {p['path'] for p in done['files']} == {str(output / name) for name in ('train-boundaries.jsonl', 'select-boundaries.jsonl', 'unavailable-known-fields.jsonl', 'PREPARATION.json')}
    import transformers
    from transformers import AutoTokenizer
    assert transformers.__version__ == '4.57.6'
    tokenizer = AutoTokenizer.from_pretrained(cfg['stock_directory'], local_files_only=True, trust_remote_code=False)
    assert tokenizer.is_fast
    spec = importlib.util.spec_from_file_location('primary_original_pair_helper', cfg['helper'])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    counts, reasons, domains = {}, Counter(), {}
    with (output / 'unavailable-known-fields.jsonl').open() as failed:
        for roster, native_split in (('train', 'train'), ('select', 'validation')):
            count, domain, seen = Counter(), Counter(), set()
            with (DATA / 'data' / (roster + '.jsonl')).open() as original, (output / (roster + '-boundaries.jsonl')).open() as sidecar:
                for ordinal, line in enumerate(original):
                    if ordinal % 128 == 0:
                        verify([config_pin, cfg['source'], completion_pin] + cfg['input_pins'] + done['files'])
                        assert time.monotonic() - started < 1800
                    case, saved = json.loads(line), json.loads(next(sidecar))
                    assert saved['version'] == 'base-state-boundary-fixed-separator-v2' and saved['roster'] == roster
                    assert saved['boundary_policy'] == 'state-plus-one-existing-newline-v2'
                    assert saved['case_ordinal'] == ordinal and saved['case_id'] == case['case_id'] and saved['component'] == case['component']
                    assert saved['native_split'] == case['split'] == native_split and saved['primary_domain'] == case['primary_domain']
                    assert case['case_id'] not in seen and saved['known_mask'] == case['known_mask']
                    seen.add(case['case_id'])
                    assert len(saved['fields']) == len(case['fields'])
                    count['cases'] += 1
                    domain[case['primary_domain']] += 1
                    for index, (field, value) in enumerate(zip(case['fields'], saved['fields'])):
                        assert value['field_index'] == index and value['coordinate'] == field['coordinate'] and value['known'] == field['known']
                        count['fields'] += 1
                        count[value['status']] += 1
                        if not field['known']:
                            assert value['status'] == 'masked_unknown_no_forward'
                            assert all(value[k] is None for k in ('state_prefix_tokens', 'original_candidate_prefix_tokens', 'candidate_count', 'pair_lengths'))
                            continue
                        count['known_fields'] += 1
                        assert value['candidate_count'] == len(field['suffix_ids']) == len(field['options'])
                        assert value['pair_lengths'] == field['pair_lengths'] and value['original_candidate_prefix_tokens'] == len(field['prefix_ids'])
                        count['complete_candidates'] += len(field['suffix_ids'])
                        count['logical_pair_tokens_once'] += sum(field['pair_lengths'])
                        criteria = helper.criteria_context(helper.text(field['question']), field['options'])
                        prefix, _ = helper.render_pair(field['state'], helper.text(field['question']), field['options'][0], criteria)
                        state = HEADER + helper.text(field['state']) + '\n'
                        assert prefix.startswith(state) and len(prefix) > len(state)
                        actual = tokenizer(prefix, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
                        assert actual['input_ids'] == field['prefix_ids']
                        k = independent_cut(prefix, len(state), actual['offset_mapping'])
                        assert value['state_prefix_tokens'] == k, 'Sidecar is not the independently recovered exact state cut'
                        if k is None:
                            assert value['status'] == 'boundary_unavailable'
                            reasons[roster + ':' + value['reason']] += 1
                            assert json.loads(next(failed)) == {'roster': roster, 'case_id': case['case_id'], **value}
                        else:
                            assert value['status'] == 'exact_state_boundary' and value['reason'] is None and 0 < k < len(field['prefix_ids'])
                        for length, suffix in zip(field['pair_lengths'], field['suffix_ids']):
                            assert suffix[-1] == 151645 and length == len(field['prefix_ids']) + len(suffix)
                            if k is not None:
                                assert field['prefix_ids'][:k] + (field['prefix_ids'][k:] + suffix) == field['prefix_ids'] + suffix
                assert sidecar.read() == ''
            assert dict(count) == report['counts'][roster] and dict(domain) == report['primary_domain_cases'][roster]
            counts[roster], domains[roster] = dict(count), dict(domain)
        assert failed.read() == ''
    assert dict(reasons) == report['unavailable_known_field_reasons'] and done['ready'] == (not reasons)
    verify([config_pin, cfg['source'], completion_pin] + cfg['input_pins'] + done['files'])
    assert 'torch' not in sys.modules and not (output / 'BLOCKED.json').exists()
    result = {'complete': True, 'boundary_policy': 'state-plus-one-existing-newline-v2', 'prepared_scan_complete': True, 'ready': done['ready'], 'allow_new_state_boundary_training': done['ready'],
        'actual_original_and_sidecar_counts_verified': counts, 'primary_domain_cases': domains, 'unavailable_known_field_reasons': dict(reasons),
        'independent_full_prefix_ids_and_exact_character_offset_cut_verified': True,
        'full_known_candidate_lengths_EOS_and_available_rows_reconstruction_verified': True,
        'every_unknown_preserved_without_tokenization_or_candidate_reinterpretation': True,
        'source_config_data_outputs_stat_and_source_bytes_verified': True, 'prepared_completion': completion_pin,
        'producer_source': cfg['source'], 'primary_analysis_source': meta(__file__), 'config': config_pin,
        'D_confirm_opened': False, 'model_run': False, 'torch_imported': False, 'public_run': False,
        'actual_seconds': time.monotonic() - started, 'verified_utc': datetime.now(timezone.utc).isoformat()}
    with (ROOT / 'FINAL_PRIMARY_VERIFICATION.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
