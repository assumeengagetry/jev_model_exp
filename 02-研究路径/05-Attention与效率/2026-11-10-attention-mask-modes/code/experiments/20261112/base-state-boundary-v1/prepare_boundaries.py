"""CPU-only full actual train/select state-boundary projection, no model run."""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

sys.dont_write_bytecode = True
VERSION = 'base-state-boundary-v1'
HEADER = 'Decide whether the candidate is a correct match.\nInput:\n'
SPLITS = {'train': 'train', 'select': 'validation'}
STOCK = Path('/home/dev/assumeengage/models/Qwen3-0.6B')
DATA = Path('/tmp/opencode/jev-clean-origin-20261112-v1/prepared-01')


def read(path):
    return json.loads(Path(path).read_text())


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def meta(path):
    path = Path(path)
    s = path.stat()
    if not path.is_file():
        raise RuntimeError('Not a regular file: ' + str(path))
    return {'path': str(path), 'resolved': str(path.resolve()), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def verify(pins):
    for p in pins:
        if meta(p['path']) != p:
            raise RuntimeError('Pinned source/input changed: ' + p['path'])


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(dump(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def commit(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.writing')
    write(pending, value)
    os.link(pending, path)
    pending.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class BoundaryUnavailable(ValueError):
    pass


def verify_record(original, record, roster, ordinal):
    """Actual-output acceptance, including failed rows; no toy/model input."""
    assert record['version'] == VERSION and record['roster'] == roster
    assert record['native_split'] == original['split'] == SPLITS[roster]
    assert record['case_id'] == original['case_id'] and record['component'] == original['component']
    assert record['case_ordinal'] == ordinal and record['primary_domain'] == original['primary_domain']
    assert len(record['fields']) == len(original['fields'])
    assert record['known_mask'] == original['known_mask']
    for index, (field, value) in enumerate(zip(original['fields'], record['fields'])):
        assert value['field_index'] == index and value['coordinate'] == field['coordinate']
        assert value['known'] == field['known']
        if not field['known']:
            assert value['status'] == 'masked_unknown_no_forward'
            assert all(value[k] is None for k in ('state_prefix_tokens', 'original_candidate_prefix_tokens', 'candidate_count', 'pair_lengths'))
            assert field['prefix_ids'] == field['suffix_ids'] == field['pair_lengths'] == []
            continue
        assert value['candidate_count'] == len(field['options']) == len(field['suffix_ids'])
        assert value['original_candidate_prefix_tokens'] == len(field['prefix_ids'])
        assert value['pair_lengths'] == field['pair_lengths']
        if value['status'] == 'exact_state_boundary':
            k = value['state_prefix_tokens']
            assert type(k) is int and 0 < k < len(field['prefix_ids'])
            assert value['reason'] is None
            for length, suffix in zip(value['pair_lengths'], field['suffix_ids']):
                assert suffix[-1] == 151645 and len(field['prefix_ids']) + len(suffix) == length
                new_p, new_s = field['prefix_ids'][:k], field['prefix_ids'][k:] + suffix
                assert new_p + new_s == field['prefix_ids'] + suffix and len(new_p) + len(new_s) == length
        else:
            assert value['status'] == 'boundary_unavailable' and value['state_prefix_tokens'] is None and value['reason']


def state_boundary(prefix_ids, prefix_text, state_end, offsets):
    """Exact unique split in Python character coordinates, never rounded."""
    if not 0 < state_end < len(prefix_text) or len(offsets) != len(prefix_ids):
        raise BoundaryUnavailable('offset_unrecoverable')
    left, right, frontier, previous_start, saw_right = 0, 0, 0, -1, False
    for pair in offsets:
        if len(pair) != 2 or any(type(v) is not int for v in pair):
            raise BoundaryUnavailable('offset_unrecoverable')
        start, end = pair
        # Overlapping/repeated Unicode spans are allowed, gaps and zero-length
        # unassignable special offsets are not. This is not a UTF-8 byte index.
        if not 0 <= start < end <= len(prefix_text) or start < previous_start or start > frontier:
            raise BoundaryUnavailable('offset_unrecoverable')
        previous_start, frontier = start, max(frontier, end)
        if start < state_end < end:
            raise BoundaryUnavailable('crossing_bpe')
        if end <= state_end:
            if saw_right:
                raise BoundaryUnavailable('offset_unrecoverable')
            left += 1
        elif start >= state_end:
            right += 1
            saw_right = True
        else:
            raise BoundaryUnavailable('offset_unrecoverable')
    if frontier != len(prefix_text) or left == 0 or right == 0:
        raise BoundaryUnavailable('offset_unrecoverable')
    if offsets[left - 1][1] != state_end or offsets[left][0] != state_end:
        raise BoundaryUnavailable('offset_unrecoverable')
    return left


def project(case, roster, ordinal, tokenizer, helper):
    assert case['split'] == SPLITS[roster] and case['case_id'] == f"clean-semantic-{case['component']}"
    assert not case['rejection_reasons'] and not case['truncated']
    result = {'version': VERSION, 'roster': roster, 'native_split': case['split'], 'case_ordinal': ordinal,
        'case_id': case['case_id'], 'component': case['component'], 'primary_domain': case['primary_domain'],
        'known_mask': case['known_mask'], 'fields': []}
    for index, field in enumerate(case['fields']):
        value = {'field_index': index, 'coordinate': field['coordinate'], 'known': field['known'],
            'state_prefix_tokens': None, 'original_candidate_prefix_tokens': None, 'candidate_count': None, 'pair_lengths': None}
        if not field['known']:
            value.update(status='masked_unknown_no_forward', reason=None)
        else:
            assert field['forward_policy'] == 'known_complete_candidates'
            assert field['prefix_ids'] and field['suffix_ids'] and field['options']
            value.update(original_candidate_prefix_tokens=len(field['prefix_ids']), candidate_count=len(field['suffix_ids']), pair_lengths=field['pair_lengths'])
            # Only input fields reach renderer and boundary selection, never q.
            criteria = helper.criteria_context(helper.text(field['question']), field['options'])
            prefix, _ = helper.render_pair(field['state'], helper.text(field['question']), field['options'][0], criteria)
            state_prefix = HEADER + helper.text(field['state'])
            assert prefix.startswith(state_prefix) and prefix.endswith('\nCandidate:\n')
            encoded = tokenizer(prefix, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
            assert encoded['input_ids'] == field['prefix_ids'], 'Original independent prefix tokenization changed'
            try:
                cut = state_boundary(field['prefix_ids'], prefix, len(state_prefix), encoded['offset_mapping'])
            except BoundaryUnavailable as exc:
                value.update(status='boundary_unavailable', reason=str(exc))
            else:
                value.update(state_prefix_tokens=cut, status='exact_state_boundary', reason=None)
        result['fields'].append(value)
    verify_record(case, result, roster, ordinal)
    return result


def run(args, root, output):
    started = time.monotonic()
    config_pin = meta(args.config)
    cfg = read(args.config)
    verify([config_pin])
    assert cfg['version'] == VERSION and cfg['approved_for_cpu_projection'] is True
    assert cfg['helper'] == '/home/dev/assumeengage/scripts/pair_data_common.py'
    assert cfg['stock_directory'] == str(STOCK) and cfg['data_root'] == str(DATA)
    assert cfg['source']['path'] == str(Path(__file__).resolve())
    verify(cfg['input_pins'] + [cfg['source']])
    assert base64.b64encode(Path(__file__).read_bytes()).decode() == cfg['source_bytes']
    assert meta(DATA / 'COMPLETE.json') == cfg['data_complete']
    done, ready = read(DATA / 'COMPLETE.json'), read(DATA / 'DATA_READY.json')
    assert done['complete'] and done['DATA_READY'] and ready['DATA_READY'] and not (DATA / 'BLOCKED.json').exists()
    assert done['data_ready'] == meta(DATA / 'DATA_READY.json')
    verify(ready['files'])
    for roster in SPLITS:
        assert meta(DATA / 'data' / (roster + '.jsonl')) in ready['files']
    limits = cfg['limits']
    assert limits == {'wall_seconds': 1800, 'rss_bytes': 2 * 2**30, 'output_bytes': 256 * 2**20, 'free_reserve_bytes': 20 * 2**30, 'cpus': 2}
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity[-limits['cpus']:]))
    os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    generated_pins = []

    def check():
        assert time.monotonic() - started < limits['wall_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < limits['rss_bytes']
        assert not (root / 'STOP.json').exists() and shutil.disk_usage(output).free >= limits['free_reserve_bytes']
        assert sum(p.stat().st_size for p in output.iterdir() if p.is_file()) < limits['output_bytes']
        verify(cfg['input_pins'] + [cfg['source'], config_pin])
        verify(generated_pins)

    try:
        import transformers
        from transformers import AutoTokenizer
        assert transformers.__version__ == '4.57.6'
        tokenizer = AutoTokenizer.from_pretrained(str(STOCK), local_files_only=True, trust_remote_code=False)
        assert tokenizer.is_fast and tokenizer.pad_token_id == 151643 and tokenizer.eos_token_id == 151645
        spec = importlib.util.spec_from_file_location('original_state_boundary_helper', cfg['helper'])
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        counts, failures, domains = {}, Counter(), {}
        with (output / 'unavailable-known-fields.jsonl').open('x') as failed:
            for roster in SPLITS:
                count, domain, seen = Counter(), Counter(), set()
                with (DATA / 'data' / (roster + '.jsonl')).open() as source, (output / (roster + '-boundaries.jsonl')).open('x') as target:
                    for ordinal, line in enumerate(source):
                        if ordinal % 128 == 0:
                            check()
                        case = json.loads(line)
                        assert case['case_id'] not in seen
                        seen.add(case['case_id'])
                        record = project(case, roster, ordinal, tokenizer, helper)
                        target.write(dump(record) + '\n')
                        count['cases'] += 1
                        domain[case['primary_domain']] += 1
                        for field, value in zip(case['fields'], record['fields']):
                            count['fields'] += 1
                            count[value['status']] += 1
                            if value['known']:
                                count['known_fields'] += 1
                                count['complete_candidates'] += len(field['suffix_ids'])
                                count['logical_pair_tokens_once'] += sum(field['pair_lengths'])
                                if value['status'] == 'boundary_unavailable':
                                    failures[roster + ':' + value['reason']] += 1
                                    failed.write(dump({'roster': roster, 'case_id': case['case_id'], **value}) + '\n')
                    target.flush()
                    os.fsync(target.fileno())
                    generated_pins.append(meta(output / (roster + '-boundaries.jsonl')))
                counts[roster], domains[roster] = dict(count), dict(domain)
            failed.flush()
            os.fsync(failed.fileno())
            generated_pins.append(meta(output / 'unavailable-known-fields.jsonl'))
        assert counts['train']['cases'] == 24600 and counts['select']['cases'] == 375
        assert domains['select'] == {d: 75 for d in ('knowledge', 'language', 'retrieval', 'tools', 'arts')}
        aggregates = read(DATA / 'AGGREGATES.json')
        for roster in SPLITS:
            expected = aggregates[roster]
            assert all(counts[roster][key] == expected['counts'][key] for key in ('cases', 'fields', 'known_fields'))
            assert counts[roster].get('masked_unknown_no_forward', 0) == expected['counts']['unknown_fields']
            assert domains[roster] == expected['primary_domain_cases']
        # Actual full artifact readback: identity join, unknown/null policy,
        # candidate/EOS/length and every exact cut's row reconstruction.
        reread, reread_failures = Counter(), Counter()
        with (output / 'unavailable-known-fields.jsonl').open() as failed:
            for roster in SPLITS:
                reread_counts, reread_domains = Counter(), Counter()
                with (DATA / 'data' / (roster + '.jsonl')).open() as source, (output / (roster + '-boundaries.jsonl')).open() as target:
                    for ordinal, line in enumerate(source):
                        if ordinal % 128 == 0:
                            check()
                        case, value = json.loads(line), json.loads(next(target))
                        verify_record(case, value, roster, ordinal)
                        # Re-derive the actual character/offset split, not only
                        # the tautological reconstruction for an arbitrary k.
                        assert value == project(case, roster, ordinal, tokenizer, helper), 'Readback differs from actual precise cut'
                        reread[roster + '_cases'] += 1
                        reread[roster + '_fields'] += len(value['fields'])
                        reread_counts['cases'] += 1
                        reread_domains[case['primary_domain']] += 1
                        for field, item in zip(case['fields'], value['fields']):
                            reread_counts['fields'] += 1
                            reread_counts[item['status']] += 1
                            if item['known']:
                                reread_counts['known_fields'] += 1
                                reread_counts['complete_candidates'] += len(field['suffix_ids'])
                                reread_counts['logical_pair_tokens_once'] += sum(field['pair_lengths'])
                                if item['status'] == 'boundary_unavailable':
                                    reread_failures[roster + ':' + item['reason']] += 1
                                    assert json.loads(next(failed)) == {'roster': roster, 'case_id': case['case_id'], **item}
                    assert target.read() == '', 'Extra sidecar rows'
                assert dict(reread_counts) == counts[roster] and dict(reread_domains) == domains[roster]
            assert failed.read() == '', 'Extra unavailable-field records'
        assert reread_failures == failures, 'Readback ready/failure status differs'
        check()
        report = {'version': VERSION, 'complete_scan_and_readback': True, 'ready': not failures,
            'counts': counts, 'primary_domain_cases': domains, 'unavailable_known_field_reasons': dict(failures),
            'readback': dict(reread), 'boundary_contract': 'unique-exact-token-split-at-original-state-character-end; no rounding/filtering/retokenization',
            'same_original_pair_tokens_and_lengths_verified': True, 'unknown_raw_not_tokenized_or_forwarded': True,
            'target_used_for_cut': False, 'D_confirm_opened': False, 'model_run': False, 'public_run': False,
            'source': cfg['source'], 'source_bytes': cfg['source_bytes'], 'input_pins': cfg['input_pins'], 'config': config_pin,
            'elapsed_seconds': time.monotonic() - started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'utc': datetime.now(timezone.utc).isoformat()}
        write(output / 'PREPARATION.json', report)
        generated_pins.append(meta(output / 'PREPARATION.json'))
        files = sorted(generated_pins, key=lambda p: p['path'])
        assert {str(p) for p in output.iterdir() if p.is_file()} == {p['path'] for p in files}
        check()
        commit(output / 'COMPLETE.json', {'version': VERSION, 'complete': True, 'ready': not failures, 'files': files,
            'source': cfg['source'], 'config': config_pin, 'model_run': False, 'D_confirm_opened': False})
        check()
        print(json.dumps({k: v for k, v in report.items() if k not in ('source_bytes', 'input_pins')}, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        write(output / 'BLOCKED.json', {'complete': False, 'ready': False, 'error_type': type(error).__name__, 'error': str(error),
            'partial_preserved': True, 'model_run': False, 'D_confirm_opened': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only execution required'
    os.umask(0o077)
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    root, output = args.config.parent, args.output
    # Only these filesystem ownership checks precede structured failure capture.
    # Never place a BLOCKED file in an existing/unfamiliar or redirected output.
    assert root.parent == Path('/tmp/opencode') and root.name.startswith('jev-base-state-boundary-20261112-')
    assert root.is_dir() and not root.is_symlink() and root.resolve() == root
    assert output == root / 'prepared-01' and not output.exists() and not output.is_symlink()
    output.mkdir(mode=0o700)
    try:
        run(args, root, output)
    except BaseException as error:
        if not (output / 'BLOCKED.json').exists():
            write(output / 'BLOCKED.json', {'complete': False, 'ready': False, 'phase': 'preflight_or_execution',
                'error_type': type(error).__name__, 'error': str(error), 'partial_preserved': True,
                'model_run': False, 'D_confirm_opened': False})
        raise


if __name__ == '__main__':
    main()
