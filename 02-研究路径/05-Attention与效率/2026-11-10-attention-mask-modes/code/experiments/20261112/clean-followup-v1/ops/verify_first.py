"""Independent CPU readback of one saved formal macro0; never launches work.

Usage: CUDA_VISIBLE_DEVICES='' python -B ops/verify_first.py --root ROOT --arm ARM
This is an external admission gate, not an objectives/model implementation.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import stat as stat_mode
import sys

sys.dont_write_bytecode = True

ROOT = Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
OLD = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
ARMS = {
    'STATE_NL_PB_P_CE': ('prefix_block', 'CE', 'C_P_CE'),
    'JPAIR_P_BCE_RANK': ('causal', 'BCE_RANK_HARD', 'C_P_BCE'),
}
CURSOR = {'macro_updates': 1, 'cases': 16, 'logical_pair_tokens': 20947}
# Fixed before any readback. These bounds apply ONLY to cross-device norm
# diagnostics, never to losses, margin coefficients, tensor bytes or book scalars.
PRECLIP_REL_TOL = 2e-5
PRECLIP_ABS_TOL = 1e-7
GRAD_STAT_REL_TOL = 1e-12
GRAD_STAT_ABS_TOL = 1e-12
PIN_KEYS = {'path', 'resolved', 'bytes', 'mtime_ns'}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def same(left, right, message):
    need(dump(left) == dump(right), message)


def parse(text):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON member: ' + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def pin(path):
    path = Path(path)
    s = path.stat()
    need(stat_mode.S_ISREG(s.st_mode), 'Not a regular file: ' + str(path))
    return {'path': str(path), 'resolved': str(path.resolve()),
            'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(dump(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    sync_dir(Path(path).parent)


class Evidence:
    """Accumulate actual file pins; recheck every input before publication."""
    def __init__(self):
        self.files = {}
        self.bytes = {}

    def add(self, path, expected=None, exact=False):
        current = pin(path)
        if expected is not None:
            need(isinstance(expected, dict) and set(expected) == PIN_KEYS,
                 'Bad expected stat pin')
            same(current, expected, 'Stat pin changed: ' + str(path))
        prior = self.files.get(current['path'])
        need(prior is None or prior == current, 'Input changed while reading: ' + str(path))
        self.files[current['path']] = current
        if exact:
            encoded = base64.b64encode(Path(path).read_bytes()).decode('ascii')
            need(current['path'] not in self.bytes or self.bytes[current['path']] == encoded,
                 'Exact source/config bytes changed: ' + str(path))
            self.bytes[current['path']] = encoded
            same(pin(path), current, 'Input changed during byte read')
        return current

    def collect(self, value):
        if isinstance(value, dict):
            if set(value) == PIN_KEYS:
                self.add(value['path'], value)
            else:
                for item in value.values():
                    self.collect(item)
        elif isinstance(value, list):
            for item in value:
                self.collect(item)

    def read(self, path, exact=False):
        self.add(path, exact=exact)
        result = parse(Path(path).read_text(encoding='utf-8'))
        self.add(path)
        return result

    def source_map(self, records):
        need(isinstance(records, list) and records, 'Missing source pin list')
        result = {}
        for record in records:
            self.add(record['path'], record, exact=True)
            need(record['path'] not in result, 'Duplicate source pin')
            result[record['path']] = self.bytes[record['path']]
        return result

    def check(self):
        for name, record in self.files.items():
            same(pin(name), record, 'Evidence changed: ' + name)
        for name, encoded in self.bytes.items():
            need(base64.b64encode(Path(name).read_bytes()).decode('ascii') == encoded,
                 'Exact evidence bytes changed: ' + name)
        for name, record in self.files.items():
            same(pin(name), record, 'Evidence changed during final byte check: ' + name)


def inventory(directory):
    need(directory.is_dir() and not directory.is_symlink() and directory.resolve() == directory,
         'Saved directory redirected: ' + str(directory))
    files = set()
    for path in directory.rglob('*'):
        need(not path.is_symlink() and not path.name.startswith('BLOCKED'),
             'Blocked/symlink saved output: ' + str(path))
        if path.is_dir():
            continue
        need(stat_mode.S_ISREG(path.stat().st_mode), 'Nonregular saved output')
        if path != directory / 'COMPLETE.json':
            files.add(str(path))
    return files


def complete(evidence, directory, expected=None):
    marker = directory / 'COMPLETE.json'
    evidence.add(marker, expected)
    value = evidence.read(marker)
    need(value.get('complete') is True and isinstance(value['files'], list) and value['files'],
         'Incomplete saved artifact: ' + str(directory))
    names = []
    for record in value['files']:
        need(Path(record['path']).is_relative_to(directory) and
             record['path'] != str(marker), 'COMPLETE redirected a saved member')
        evidence.add(record['path'], record)
        names.append(record['path'])
    need(len(set(names)) == len(names) and set(names) == inventory(directory),
         'Unbound/missing/duplicate saved file: ' + str(directory))
    evidence.collect(value)
    return value


def tensor_equal(torch, left, right):
    return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and
            left.device.type == right.device.type == 'cpu' and
            left.shape == right.shape and left.dtype == right.dtype and
            torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                        right.detach().contiguous().reshape(-1).view(torch.uint8)))


def tree_equal(torch, left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return tensor_equal(torch, left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys() and
                all(tree_equal(torch, left[k], right[k]) for k in left))
    if isinstance(left, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(tree_equal(torch, a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def finite_tensor(torch, value, shape=None, dtype=None):
    need(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
         (shape is None or tuple(value.shape) == tuple(shape)) and
         (dtype is None or value.dtype == dtype) and bool(torch.isfinite(value).all()),
         'Nonfinite/wrong CPU tensor shape or dtype')


def load_pt(torch, evidence, path):
    evidence.add(path)
    value = torch.load(path, map_location='cpu', weights_only=True)
    evidence.add(path)
    return value


def saved_tensors(torch, evidence, directory):
    from safetensors.torch import load_file
    path = directory / 'adapter_model.safetensors'
    evidence.add(path)
    result = load_file(str(path), device='cpu')
    evidence.add(path)
    need(isinstance(result, dict) and len(result) == 393, 'Expected393 saved trainables')
    for value in result.values():
        finite_tensor(torch, value, dtype=torch.float32)
    return result


def canonical(name):
    """Only the frozen PEFT0.18.1 active default-adapter translations."""
    if '.lora_A.default.' in name:
        return name.replace('.lora_A.default.', '.lora_A.')
    if '.lora_B.default.' in name:
        return name.replace('.lora_B.default.', '.lora_B.')
    if '.score.modules_to_save.default.' in name:
        return name.replace('.score.modules_to_save.default.', '.score.')
    raise RuntimeError('Unrecognized full trainable name: ' + name)


def compare_parameters(torch, left, right, message):
    need(isinstance(left, dict) and isinstance(right, dict) and len(left) == len(right) == 393 and
         left.keys() == right.keys(), message + ': keys')
    for name in left:
        need(tensor_equal(torch, left[name], right[name]), message + ': ' + name)


def old_scope(evidence, ctx, cfg, value, arm_name):
    need(cfg['version'] == 'clean-case-trainer-v1' and cfg['approved_for_execution'] is True and
         cfg['output_parent'] == str(OLD) and cfg['seed'] == 42 and value['arm_name'] == arm_name and
         value['version'] == cfg['version'], 'Old control version/seed/arm differs')
    same(value['config'], cfg, 'Old scope config differs from actual old config')
    same(value['config_pin'], evidence.add(OLD / 'seed42.json'), 'Old config pin differs')
    same(value['arm'], cfg['arms'][arm_name], 'Old scope arm differs')
    same(value['arm'], {'mode': 'causal', 'layout': 'packed',
                        'loss': 'BCE' if arm_name == 'C_P_BCE' else 'CE'},
         'Wrong matched old control')
    for key in ('core_files', 'native_files', 'stock_files', 'data_root', 'data_complete',
                'primary_admission', 'precision', 'macro_cases', 'micro_plan', 'optimizer'):
        same(cfg[key], ctx.cfg[key], 'Old/new controlled input differs: ' + key)
    same(value['trainer_code'], evidence.source_map(cfg['trainer_files']), 'Old source bytes differ')
    same(value['core_code'], evidence.source_map(cfg['core_files']), 'Old core bytes differ')
    same(value['data'], ctx.data, 'Old/new prepared data identity differs')
    same(value['primary']['pin'], cfg['primary_admission'], 'Old model admission pin differs')
    same(value['primary']['admission'], ctx.admission['admission'], 'Old admission bytes differ')
    for name in ('train_layout', 'full_row_geometry_isolation'):
        pointer = value['primary']['admission']['modes']['causal']['checks'][name]
        need(pointer['file'] in value['primary']['admission']['evidence_files'],
             'Old causal model evidence is not bound by actual admission')
        actual = evidence.read(pointer['file']['path'])
        for key in pointer['keys']:
            actual = actual[key]
        same(value['primary']['actual_results'][name], actual, 'Old actual admission evidence differs')
        need(actual['complete'] is True and actual['passed'] is True and
             actual['actual_model_run'] is True and actual['mode'] == 'causal' and
             actual['backend'] == 'native-fa3' and actual['SDPA_fallback'] is False,
             'Old matched control lacks actual causal model admission')
        for key in ('precision', 'core_files', 'native_files'):
            same(actual[key], cfg[key], 'Old actual model evidence input differs: ' + key)
    evidence.collect(value)


def verify_fresh(torch, evidence, ctx, identity, old_cfg):
    report = evidence.read(ROOT / 'FRESH_PRIMARY_VERIFICATION.json', exact=True)
    need(report['complete'] is True and
         report['independently_generated_before_old_oracle_load'] is True and
         report['393_initial_tensors_bitwise_equal_old_v2'] is True and
         report['B_exactly_zero'] is True and report['model_forward_calls'] == 0 and
         report['optimizer_steps'] == 0 and report['CUDA_initialized'] is False and
         report['old_project_trained_adapter_head_optimizer_teacher_loaded'] is False and
         report['training_launched'] is False and report['D_confirm_opened'] is False and
         report['public_run'] is False,
         'Independent zero-forward generation-before-oracle primary report required')
    same(report['new_init_complete'], ctx.cfg['fresh_init']['complete'], 'Fresh primary COMPLETE differs')
    same(report['final_config'], ctx.snapshot['config_pin'], 'Fresh primary final config differs')
    same(report['trainer_files'], ctx.cfg['trainer_files'], 'Fresh primary trainer pins differ')
    same(report['trainer_code'], ctx.code, 'Fresh primary source bytes differ')
    fresh_dir = Path(ctx.cfg['fresh_init']['directory'])
    need(fresh_dir == ROOT / 'init-seed42', 'New fresh namespace differs')
    fresh_commit = complete(evidence, fresh_dir, ctx.cfg['fresh_init']['complete'])
    same(fresh_commit, identity['commit'], 'Fresh validation/actual COMPLETE differ')
    same(report['new_manifest'], evidence.add(fresh_dir / 'FRESH_MANIFEST.json'), 'New manifest pin differs')
    need(report['actual_saved_fp32_tensor_keys'] == 393 and report['lora_pairs'] == 196 and
         report['head_shape'] == [2, 1024] and report['active_served_parameters'] == 616237056 and
         report['trainable_parameters'] == 20187136,
         'Fresh primary393/head/served scope differs')
    same(report['construction'], identity['manifest']['construction'], 'Fresh construction differs')
    manifest = identity['manifest']
    need(manifest['audit']['active_served_parameters'] == 616237056 and
         manifest['audit']['trainable_parameters'] == 20187136, 'Fresh served/trainable scope differs')
    old_dir = Path(old_cfg['fresh_init']['directory'])
    need(old_dir == OLD / 'init-seed42', 'Old same-seed fresh oracle redirected')
    same(report['old_oracle_complete'], old_cfg['fresh_init']['complete'], 'Old oracle pin differs')
    old_commit = complete(evidence, old_dir, old_cfg['fresh_init']['complete'])
    old_manifest = evidence.read(old_dir / 'FRESH_MANIFEST.json', exact=True)
    same(report['old_oracle_manifest'], evidence.add(old_dir / 'FRESH_MANIFEST.json'),
         'Old oracle manifest pin differs')
    need(old_commit['kind'] == 'fresh-stock-init-v1' and old_commit['seed'] == 42 and
         old_manifest['seed'] == 42 and old_manifest['model_forward_calls'] == 0 and
         old_manifest['no_project_adapter_head_optimizer_teacher_loaded'] is True,
         'Old oracle is not the original zero-forward fresh init')
    same(old_commit['manifest'], evidence.add(old_dir / 'FRESH_MANIFEST.json'),
         'Old fresh COMPLETE manifest pointer differs')
    same(old_manifest['construction'], manifest['construction'], 'Old/new fresh construction differs')
    same(old_manifest['stock_files'], manifest['stock_files'], 'Old/new fresh stock differs')
    same(old_manifest['trainer_code'], evidence.source_map(old_cfg['trainer_files']),
         'Old fresh source bytes differ')
    fresh = saved_tensors(torch, evidence, fresh_dir)
    oracle = saved_tensors(torch, evidence, old_dir)
    compare_parameters(torch, fresh, oracle, 'New independent fresh != old same-seed oracle')
    a = {k for k in fresh if '.lora_A.' in k}
    b = {k for k in fresh if '.lora_B.' in k}
    head = [k for k in fresh if k.endswith('score.weight')]
    need(len(a) == len(b) == 196 and len(head) == 1 and
         {k.replace('.lora_A.', '.lora_B.') for k in a} == b and
         list(fresh[head[0]].shape) == [2, 1024] and
         all(fresh[k].shape[0] == 32 for k in a) and
         all(fresh[k].shape[1] == 32 and int(torch.count_nonzero(fresh[k])) == 0 for k in b),
         'Fresh actual tensor architecture/zero-B differs')
    need(sum(v.numel() for v in fresh.values()) == 20187136, 'Actual trainable parameter count differs')
    same(sorted(fresh), manifest['audit']['saved_keys'], 'Fresh manifest actual saved keys differ')
    evidence.collect(report)
    return fresh


def inspect_case(case):
    """Read actual original fields; never interpret unknown raw options/targets."""
    need(case['split'] == 'train' and isinstance(case['fields'], list) and case['fields'] and
         case['truncated'] is False and not case['rejection_reasons'], 'Incomplete original train case')
    same(case['known_mask'], [f['known'] for f in case['fields']], 'Original known mask differs')
    known, unknown, candidates, tokens = 0, 0, 0, 0
    for field in case['fields']:
        need(type(field['known']) is bool, 'Original known flag is not bool')
        if not field['known']:
            need(field['target'] is None and field['prefix_ids'] == field['suffix_ids'] ==
                 field['pair_lengths'] == [] and 'raw_target' in field and 'raw_options' in field and
                 field['forward_policy'] == 'masked_unknown_no_forward',
                 'Unknown field lost original raw/no-forward contract')
            unknown += 1
            continue
        known += 1
        prefix, suffixes, q = field['prefix_ids'], field['suffix_ids'], field['target']
        need(prefix and all(type(t) is int and t >= 0 for t in prefix) and
             isinstance(q, list) and len(q) == len(suffixes) == len(field['options']) >= 1 and
             all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in q) and
             math.isclose(math.fsum(q), 1.0, rel_tol=1e-6, abs_tol=1e-6) and
             field['forward_policy'] == 'known_complete_candidates' and
             field['static_unsupported_reason'] is None, 'Original complete known field differs')
        lengths = []
        for suffix in suffixes:
            need(suffix and suffix[-1] == 151645 and
                 all(type(t) is int and t >= 0 for t in suffix), 'Original row IDs/EOS invalid')
            lengths.append(len(prefix) + len(suffix))
        same(lengths, field['pair_lengths'], 'Original candidate lengths differ')
        need(max(lengths) <= 8192, 'Original full train case overlength')
        candidates += len(q)
        tokens += sum(lengths)
    need(known > 0 and case['logical_supervised_pair_tokens'] ==
         case['all_field_pair_tokens_once'] == case['scheduled_forward_pair_tokens_once'] == tokens and
         case['unknown_fields_no_forward'] == unknown and case['unknown_field_forward_tokens_once'] == 0,
         'Original case accounting differs')
    return {'cases': 1, 'raw_fields': known + unknown, 'known_fields': known,
            'unknown_fields_retained_no_forward': unknown, 'cases_with_known_fields': int(known > 0),
            'cases_with_unknown_fields': int(unknown > 0), 'forwarded_complete_candidates': candidates,
            'logical_pair_tokens': tokens}


def state_cuts(case, record, ordinal):
    need(set(record) == {'version', 'boundary_policy', 'roster', 'native_split', 'case_ordinal',
                         'case_id', 'component', 'primary_domain', 'known_mask', 'fields'},
         'Actual sidecar record schema differs')
    need(record['version'] == 'base-state-boundary-fixed-separator-v2' and
         record['boundary_policy'] == 'state-plus-one-existing-newline-v2' and
         record['roster'] == 'train' and record['native_split'] == 'train' and
         type(record['case_ordinal']) is int and record['case_ordinal'] == ordinal,
         'Actual sidecar version/roster/ordinal differs')
    for key in ('case_id', 'component', 'primary_domain', 'known_mask'):
        same(record[key], case[key], 'Actual sidecar original identity differs: ' + key)
    need(len(record['fields']) == len(case['fields']), 'Sidecar lost original fields')
    cuts = []
    for index, (field, value) in enumerate(zip(case['fields'], record['fields'])):
        need(set(value) == {'field_index', 'coordinate', 'known', 'state_prefix_tokens',
                            'original_candidate_prefix_tokens', 'candidate_count', 'pair_lengths',
                            'status', 'reason'}, 'Actual sidecar field schema differs')
        need(type(value['field_index']) is int and value['field_index'] == index and
             type(value['known']) is bool and value['known'] is field['known'] and
             value['reason'] is None, 'Sidecar field index/known/reason differs')
        same(value['coordinate'], field['coordinate'], 'Sidecar original field coordinate differs')
        if not field['known']:
            need(value['status'] == 'masked_unknown_no_forward' and all(value[k] is None for k in
                 ('state_prefix_tokens', 'original_candidate_prefix_tokens', 'candidate_count', 'pair_lengths')),
                 'Unknown sidecar cut/geometry must be null')
            cuts.append(None)
            continue
        cut, prefix = value['state_prefix_tokens'], field['prefix_ids']
        need(value['status'] == 'exact_state_boundary' and type(cut) is int and 0 < cut < len(prefix) and
             type(value['original_candidate_prefix_tokens']) is int and
             value['original_candidate_prefix_tokens'] == len(prefix) and
             type(value['candidate_count']) is int and value['candidate_count'] == len(field['suffix_ids']),
             'Known sidecar exact cut/geometry differs')
        same(value['pair_lengths'], field['pair_lengths'], 'Sidecar full candidate lengths differ')
        for length, suffix in zip(field['pair_lengths'], field['suffix_ids']):
            p, s = prefix[:cut], prefix[cut:] + suffix
            need(p + s == prefix + suffix and len(p) + len(s) == length and s[-1] == 151645,
                 'State cut changed full row IDs/length/EOS')
        cuts.append(cut)
    return cuts


def original_mapping(evidence, ctx):
    # Only the original CPU offset/exposure APIs are reused. No slots/micro_plan,
    # BoundaryStore, objectives, trainer, model, tokenizer or native import.
    from case_runtime import CaseStore, Exposure
    train = exposure = None
    try:
        train = CaseStore(Path(ctx.cfg['data_root']) / 'data/train.jsonl', 'train')
        exposure = Exposure(ctx.cfg['data_root'], 42, train)
        evidence.add(train.path, train.pin)
        evidence.add(exposure.path, exposure.pin)
        evidence.add(exposure.schedule_path, exposure.schedule_pin, exact=True)
        rows = exposure.macro(0)
        cases = [train.get(row['case_id']) for row in rows]
        need(len(cases) == 16 and sum(r['logical_pair_tokens'] for r in rows) == 20947,
             'Actual macro0 exposure differs')
        wanted = {case['case_id'] for case in cases}
        cuts_by_id, raw_cases, totals, domain_cases = {}, {}, Counter(), Counter()
        side_path = Path(ctx.state_boundary['sidecars']['train']['path'])
        use_state = ctx.arm_name == 'STATE_NL_PB_P_CE'
        side = side_path.open('rb') if use_state else None
        try:
            if use_state:
                evidence.add(side_path, ctx.state_boundary['sidecars']['train'])
            with train.path.open('rb') as handle:
                seen = set()
                for ordinal, line in enumerate(handle):
                    case = parse(line)
                    need(case['case_id'] not in seen, 'Duplicate actual train case')
                    seen.add(case['case_id'])
                    totals.update(inspect_case(case))
                    domain_cases[case['primary_domain']] += 1
                    if case['case_id'] in wanted:
                        raw_cases[case['case_id']] = case
                    if use_state:
                        record = parse(side.readline())
                        cuts = state_cuts(case, record, ordinal)
                        if case['case_id'] in wanted:
                            cuts_by_id[case['case_id']] = cuts
                need(seen == set(train.meta) and len(seen) == 24600, 'Actual full train roster differs')
            for case in cases:
                same(case, raw_cases[case['case_id']], 'Original CaseStore macro record != independent raw JSON')
            if use_state:
                need(side.read(1) == b'' and set(cuts_by_id) == wanted, 'Extra/missing actual sidecar rows')
                expected = {'cases': totals['cases'], 'fields': totals['raw_fields'],
                            'known_fields': totals['known_fields'],
                            'exact_state_boundary': totals['known_fields'],
                            'complete_candidates': totals['forwarded_complete_candidates'],
                            'logical_pair_tokens_once': totals['logical_pair_tokens'],
                            'masked_unknown_no_forward': totals['unknown_fields_retained_no_forward']}
                same(expected, ctx.state_boundary['report']['counts']['train'], 'Actual sidecar counts differ')
                same(dict(domain_cases), ctx.state_boundary['report']['primary_domain_cases']['train'],
                     'Actual sidecar domains differ')
        finally:
            if side is not None:
                side.close()
        expected, old_expected, fields = [], [], []
        coverage = Counter()
        for ci, (case, row) in enumerate(zip(cases, rows)):
            counts = inspect_case(case)
            coverage.update(counts)
            known = counts['known_fields']
            for fi, field in enumerate(case['fields']):
                if not field['known']:
                    continue
                start = len(expected)
                prefix = field['prefix_ids']
                cut = cuts_by_id[case['case_id']][fi] if use_state else len(prefix)
                for ki, suffix in enumerate(field['suffix_ids']):
                    need(prefix[:cut] + (prefix[cut:] + suffix) == prefix + suffix,
                         'Independent manual resplit changed original complete row')
                    item = {'slot': len(expected), 'case_index': ci, 'case_id': case['case_id'],
                            'field_index': fi, 'candidate_index': ki, 'coordinate': field['coordinate'],
                            'exposure_index': row['exposure_index'],
                            'pair_tokens': len(prefix) + len(suffix), 'prefix_tokens': cut}
                    expected.append(item)
                    old_expected.append({**item, 'prefix_tokens': len(prefix)})
                fields.append({'case_index': ci, 'field_index': fi, 'start': start, 'end': len(expected),
                               'known_fields_in_case': known, 'target': field['target'],
                               'zero_K1': bool(field.get('zero_field_ce_gradient_K1'))})
        ordered = sorted(range(len(expected)), key=lambda i: expected[i]['pair_tokens'])
        micros, current = [], []
        for slot in ordered:
            width = (expected[slot]['pair_tokens'] + 7) // 8 * 8
            if current and (len(current) == 32 or width * (len(current) + 1) > 16384):
                micros.append(current)
                current = []
            current.append(slot)
        if current:
            micros.append(current)
        need(sorted(s for micro in micros for s in micro) == list(range(len(expected))) and
             sum(s['pair_tokens'] for s in expected) == 20947, 'Independent slot/micro coverage differs')
        macro_coverage = {key: coverage[key] for key in
                          ('raw_fields', 'known_fields', 'unknown_fields_retained_no_forward',
                           'cases_with_known_fields', 'cases_with_unknown_fields', 'forwarded_complete_candidates')}
        macro_coverage['case_occurrences'] = coverage['cases']
        return rows, expected, old_expected, fields, micros, macro_coverage, dict(totals), exposure.schedule
    finally:
        if exposure is not None:
            exposure.close()
        if train is not None:
            train.close()


def recompute_objective(torch, margins, fields, kind):
    import torch.nn.functional as F
    need(kind in ('CE', 'BCE_RANK_HARD'), 'Explicit arm objective required')
    m = margins.detach().float().clone().requires_grad_(True)
    total_terms, bce_terms, rank_terms = [], [], []
    book = {'cases': 16, 'known_fields': len(fields), 'known_candidates': 0, 'K1_fields': 0}
    if kind == 'BCE_RANK_HARD':
        book.update({key: 0 for key in
                     ('hard_positive_candidates', 'hard_negative_candidates', 'soft_candidates',
                      'rank_active_fields', 'rank_zero_fields', 'rank_zero_no_positive_fields',
                      'rank_zero_no_negative_fields', 'rank_zero_K1_fields', 'rank_zero_soft_only_fields')})
    active_cases = set()
    for field in fields:
        v = m[field['start']:field['end']]
        host_q = field['target']
        # Hard membership MUST precede target tensor conversion. Native q stays
        # unrenormalized, including host-soft values that round to FP32 endpoints.
        positive = [i for i, q in enumerate(host_q) if q == 1]
        negative = [i for i, q in enumerate(host_q) if q == 0]
        q = torch.tensor(host_q, dtype=v.dtype, device=v.device)
        denominator = field['known_fields_in_case']
        need(v.numel() == len(host_q) > 0 and denominator > 0, 'Incomplete independent objective field')
        book['known_candidates'] += len(host_q)
        book['K1_fields'] += int(len(host_q) == 1)
        if kind == 'CE':
            total_terms.append((-(q * torch.log_softmax(v, dim=0)).sum()) / denominator / 16)
            continue
        bce = F.binary_cross_entropy_with_logits(v, q, reduction='mean')
        active = bool(positive) and bool(negative)
        rank = (torch.logsumexp(v[positive + negative], dim=0) -
                torch.logsumexp(v[positive], dim=0)) if active else (v * 0.0).sum()
        total_terms.append((bce + rank) / denominator / 16)
        bce_terms.append(bce / denominator / 16)
        rank_terms.append(rank / denominator / 16)
        if active:
            active_cases.add(field['case_index'])
        book['hard_positive_candidates'] += len(positive)
        book['hard_negative_candidates'] += len(negative)
        book['soft_candidates'] += len(host_q) - len(positive) - len(negative)
        book['rank_active_fields'] += int(active)
        book['rank_zero_fields'] += int(not active)
        book['rank_zero_no_positive_fields'] += int(not positive)
        book['rank_zero_no_negative_fields'] += int(not negative)
        book['rank_zero_K1_fields'] += int(len(host_q) == 1)
        book['rank_zero_soft_only_fields'] += int(not positive and not negative)
    total = torch.stack(total_terms).sum()
    if kind == 'BCE_RANK_HARD':
        bce_total = torch.stack(bce_terms).sum()
        rank_total = torch.stack(rank_terms).sum()
        bce_coef, = torch.autograd.grad(bce_total, m, retain_graph=True)
        rank_coef, = torch.autograd.grad(rank_total, m, retain_graph=True)
        coef, = torch.autograd.grad(total, m)
        book['no_rank_cases'] = 16 - len(active_cases)
        book['macro_bce_loss'] = float(bce_total.detach())
        book['macro_rank_loss'] = float(rank_total.detach())
        book['macro_total_loss'] = float(total.detach())
        for name, coefficient in (('bce', bce_coef), ('rank', rank_coef), ('total', coef)):
            finite_tensor(torch, coefficient)
            values = coefficient.detach().double()
            book['margin_' + name + '_l2'] = float(torch.linalg.vector_norm(values))
            book['margin_' + name + '_maxabs'] = float(values.abs().max())
        for name in ('rank', 'total'):
            for norm in ('l2', 'maxabs'):
                denominator = book['margin_bce_' + norm]
                book['margin_' + name + '_over_bce_' + norm] = (
                    book['margin_' + name + '_' + norm] / denominator if denominator else None)
        need(len(book) == 27, 'Independent hard-rank book schema is not27 scalars')
    else:
        coef, = torch.autograd.grad(total, m)
    finite_tensor(torch, total)
    finite_tensor(torch, coef, shape=m.shape, dtype=torch.float32)
    need(all(type(v) in (int, float) or v is None for v in book.values()) and
         all(v is None or math.isfinite(v) for v in book.values()), 'Nonfinite/non-scalar independent book')
    return coef.detach(), float(total.detach()), book


def cumulative(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def verify_dispatch(document, arm_name, slots, micros):
    mode = ARMS[arm_name][0]
    per_layer = 2 if mode == 'prefix_block' else 1
    header = {'version': 'actual-native-dispatch-v1', 'macro_index': 0,
              'is_formal_exposure_macro0': True, 'complete': True, 'arm_name': arm_name,
              'mode': mode, 'layout': 'packed', 'layers': 28, 'calls_per_layer': per_layer,
              'call_range_convention': 'zero-based call_start inclusive; call_end exclusive',
              'layer_index_convention': 'zero-based, derived from actual call order in the pinned core loop',
              'qkv_tensor_values_saved': False, 'full_model_input_ids_saved': False,
              'eos_token_ids_only': True, 'independent_partition_primary_required': True,
              'new_prefix_cross_backend_or_neighbor_isolation_claimed': False}
    same({k: v for k, v in document.items() if k not in ('waves', 'calls')}, header,
         'Actual dispatch scope/header differs')
    need(len(document['waves']) == 2 * len(micros), 'Actual dispatch pass/wave coverage differs')
    at = 0
    expected_native = Counter()
    for pass_index, phase in enumerate(('train_first', 'train_replay')):
        for wave_index, micro in enumerate(micros):
            lengths = [slots[i]['pair_tokens'] for i in micro]
            prefixes = [slots[i]['prefix_tokens'] for i in micro]
            suffixes = [length - prefix for length, prefix in zip(lengths, prefixes)]
            need(all(p > 0 and s > 0 for p, s in zip(prefixes, suffixes)), 'Empty actual partition')
            cu_l, cu_p, cu_s = map(cumulative, (lengths, prefixes, suffixes))
            wave = {'phase': phase, 'wave_index': wave_index, 'slots': micro,
                    'row_lengths': lengths, 'prefix_lengths': prefixes,
                    'total_real_tokens': sum(lengths), 'physical_token_rows': sum(lengths),
                    'batch_cu': cu_l, 'positions': [p for length in lengths for p in range(length)],
                    'eos_indices': [end - 1 for end in cu_l[1:]],
                    'eos_positions': [length - 1 for length in lengths],
                    'eos_token_ids': [151645] * len(micro), 'actual_positions_and_eos_checked': True,
                    'call_start': at, 'call_end': at + 28 * per_layer, 'completed': True}
            same(document['waves'][pass_index * len(micros) + wave_index], wave,
                 'Actual batch positions/EOS/rows/wave range differ')
            if mode == 'causal':
                partitions = [(sum(lengths), sum(lengths), cu_l, cu_l, max(lengths), max(lengths), True)]
            else:
                partitions = [(sum(prefixes), sum(prefixes), cu_p, cu_p, max(prefixes), max(prefixes), False),
                              (sum(suffixes), sum(lengths), cu_s, cu_l, max(suffixes), max(lengths), False)]
            for layer in range(28):
                for ordinal, (tq, tk, cq, ck, mq, mk, causal) in enumerate(partitions):
                    expected = {'call_index': at, 'phase': phase, 'wave_index': wave_index,
                                'call_index_in_wave': layer * per_layer + ordinal,
                                'layer_index_from_call_order': layer, 'call_index_in_layer': ordinal,
                                'q_shape': [tq, 16, 128], 'k_shape': [tk, 8, 128],
                                'v_shape': [tk, 8, 128], 'cu_q': cq, 'cu_k': ck,
                                'max_q': mq, 'max_k': mk,
                                'kwargs': {'softmax_scale': 128 ** -0.5,
                                           'causal': causal, 'deterministic': True}, 'completed': True}
                    need(at < len(document['calls']), 'Actual native call missing')
                    same(document['calls'][at], expected, 'Actual QKV partition/flags/layer call differs')
                    at += 1
                    expected_native['attempted_calls'] += 1
                    expected_native['completed_calls'] += 1
                    expected_native['attempted_query_tokens'] += tq
                    expected_native['completed_query_tokens'] += tq
                    expected_native['completed_key_tokens'] += tk
                    expected_native['completed_causal_calls' if causal else 'completed_noncausal_calls'] += 1
    need(at == len(document['calls']), 'Extra actual native calls')
    forward = {}
    for phase in ('train_first', 'train_replay'):
        for name in ('attempted_waves', 'completed_waves'):
            forward[phase + '.' + name] = len(micros)
        for name in ('attempted_real_tokens', 'completed_real_tokens',
                     'attempted_tokenwise_rows_per_operator', 'completed_tokenwise_rows_per_operator'):
            forward[phase + '.' + name] = 20947
        forward[phase + '.completed_native_FA3_calls'] = len(micros) * 28 * per_layer
    forward['train_replay.completed_backward_waves'] = len(micros)
    forward['train_replay.backward_real_tokens'] = 20947
    runtime = {'execution': 'native-FA3-full-row-no-SDPA-no-fork', 'mode': mode, 'layout': 'packed',
               'prefix_cache': False, 'forward': forward, 'native_forward_calls': dict(expected_native),
               'backward_counter_scope': 'autograd native backward invoked by each replay wave; not separately instrumented'}
    checks = {'actual_calls': at, 'actual_waves_both_passes': 2 * len(micros), 'layers': 28,
              'Q_heads': 16, 'KV_heads': 8, 'head_dim': 128,
              'all_actual_partitions_positions_EOS_flags_and_call_ranges_match': True,
              'independent_numeric_backend_verified': False, 'new_prefix_neighbor_isolation_verified': False}
    return runtime, checks


def read_paused(evidence, folder, scope):
    need(not list(folder.glob('BLOCKED*')) and not (folder / 'COMPLETE.json').exists(),
         'Blocked/completed arm cannot be admitted')
    same(evidence.read(folder / 'OWN_SCOPE.json', exact=True), scope, 'Actual OWN_SCOPE differs')
    path = folder / 'ledger.jsonl'
    evidence.add(path)
    text = path.read_bytes()
    need(text.endswith(b'\n'), 'Durable ledger must end at a complete JSON line')
    events = [parse(line) for line in text.splitlines()]
    evidence.add(path)
    allowed = {'initialized_no_forward', 'checkpoint_started', 'checkpoint_committed',
               'macro_started', 'macro_updated', 'paused'}
    need(events and all(e['event'] in allowed for e in events),
         'Resume/dev/failed/extra model work is not a first paused arm')
    same([e['event'] for e in events],
         ['initialized_no_forward', 'checkpoint_started', 'checkpoint_committed',
          'macro_started', 'macro_updated', 'checkpoint_started', 'checkpoint_committed', 'paused'],
         'Actual first-invocation durable event order differs')
    starts = [e for e in events if e['event'] == 'macro_started']
    updates = [e for e in events if e['event'] == 'macro_updated']
    tips = [e for e in events if e['event'] == 'checkpoint_committed']
    begun = [e for e in events if e['event'] == 'checkpoint_started']
    need(len(starts) == len(updates) == 1 and starts[0]['macro_index'] == 0 and
         len(tips) == len(begun) == 2 and
         sum(e['event'] == 'initialized_no_forward' for e in events) == 1 and
         sum(e['event'] == 'paused' for e in events) == 1,
         'Formal macro0 must be started/updated exactly once, without retry')
    same(updates[0]['cursor'], CURSOR, 'Actual updated cursor differs')
    need(events[-1]['event'] == 'paused' and events[-1]['exit_code'] == 75,
         'Latest durable paused exit75 is required')
    same(events[-1]['cursor'], CURSOR, 'Actual paused cursor differs')
    need(events.index(starts[0]) < events.index(updates[0]) < events.index(tips[-1]) == len(events) - 2,
         'Later/uncheckpointed work after macro0')
    need([e['sequence'] for e in tips] == [1, 2], 'Actual checkpoint sequence differs')
    checkpoint_paths = set()
    for index, (start, tip) in enumerate(zip(begun, tips)):
        expected_cursor = {'macro_updates': 0, 'cases': 0, 'logical_pair_tokens': 0} if index == 0 else CURSOR
        same(start['cursor'], expected_cursor, 'Checkpoint-start cursor differs')
        same(tip['cursor'], expected_cursor, 'Checkpoint-commit cursor differs')
        same(start['reasons'], tip['reasons'], 'Checkpoint-start/commit reasons differ')
        directory = Path(tip['complete']['path']).parent
        need(directory.parent == folder and start['path'] == str(directory) and
             Path(tip['complete']['path']) == directory / 'COMPLETE.json', 'Checkpoint tip redirected')
        need(events.index(start) < events.index(tip), 'Checkpoint commit precedes start')
        committed = complete(evidence, directory, tip['complete'])
        same(committed['scope'], scope, 'Actual checkpoint scope differs')
        need(committed['resumable_full_macro_boundary'] is True and
             committed['macro_updates'] == index, 'Not the actual cursor0/cursor1 checkpoint')
        checkpoint_paths.add(directory)
    need(checkpoint_paths == set(folder.glob('checkpoint-*')), 'Extra partial/later checkpoint retained: blocked')
    return events, starts[0], updates[0], tips[-1]


def verify_gradients(torch, raw, clipped, actual, fresh, manifest, meta, schedule):
    names = sorted(raw)
    need(len(names) == 393 and set(raw) == set(clipped) and
         names == actual['optimizer_parameter_names'] == manifest['audit']['trainable_keys'],
         '393 full trainable/optimizer/manifest name order differs')
    mapping = {name: canonical(name) for name in names}
    need(len(set(mapping.values())) == 393 and set(mapping.values()) == set(fresh),
         'Full-to-saved parameter name map is not a393-key bijection')
    compare_parameters(torch, actual['before'], fresh, 'Actual before != independent new fresh')
    need(set(actual['before']) == set(actual['after']) == set(actual['delta']), 'Actual update key set differs')
    norms, nonzero_deltas = [], 0
    preclip = meta['preclip_l2']
    need(type(preclip) is float and math.isfinite(preclip) and preclip >= 0,
         'Actual preclip norm must be finite/nonnegative')
    same(meta['clip_norm'], 1.0, 'Clip norm changed')
    factor = min(1.0, 1.0 / (preclip + 1e-6))
    same(meta['clip_factor'], factor, 'Saved clip-factor scalar differs')
    need(0 < factor <= 1.0, 'Clip factor out of bounds')
    clip_tensor = (1.0 / (torch.tensor(preclip, dtype=torch.float32) + 1e-6)).clamp(max=1.0)
    need(set(meta['gradients']) == set(meta['clipped_gradients']) == set(names),
         'Saved per-key gradient statistics are incomplete')
    for name in names:
        saved_name = mapping[name]
        shape = fresh[saved_name].shape
        for values, statistics in ((raw, meta['gradients']), (clipped, meta['clipped_gradients'])):
            finite_tensor(torch, values[name], shape=shape, dtype=torch.float32)
            info = statistics[name]
            same(info['shape'], list(shape), 'Saved gradient shape differs')
            need(info['dtype'] == 'torch.float32' and
                 type(info['nonzero']) is int and info['nonzero'] == int(torch.count_nonzero(values[name])),
                 'Saved gradient dtype/nonzero statistic differs')
            measured = float(torch.linalg.vector_norm(values[name].double()))
            need(type(info['l2_float64']) is float and math.isfinite(info['l2_float64']) and
                 math.isclose(info['l2_float64'], measured,
                              rel_tol=GRAD_STAT_REL_TOL, abs_tol=GRAD_STAT_ABS_TOL),
                 'Saved double gradient norm inconsistent with actual tensor')
        need(tensor_equal(torch, clipped[name], raw[name] * clip_tensor),
             'Actual clipped bytes != raw times actual FP32 clip multiplier: ' + name)
        norms.append(torch.linalg.vector_norm(raw[name].double()))
        for part in ('after', 'delta'):
            finite_tensor(torch, actual[part][saved_name], shape=shape, dtype=torch.float32)
        need(tensor_equal(torch, actual['after'][saved_name] - actual['before'][saved_name],
                          actual['delta'][saved_name]), 'Actual delta bytes != after-before')
        nonzero_deltas += int(torch.count_nonzero(actual['delta'][saved_name]))
    independent_norm = float(torch.linalg.vector_norm(torch.stack(norms)))
    clipped_norm = float(torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(clipped[name].double()) for name in names])))
    need(math.isclose(preclip, independent_norm, rel_tol=PRECLIP_REL_TOL, abs_tol=PRECLIP_ABS_TOL),
         'Saved GPU preclip FP32 norm inconsistent with actual393 raw FP64 norms')
    need(math.isfinite(clipped_norm) and clipped_norm <= 1.0 + PRECLIP_REL_TOL + PRECLIP_ABS_TOL and
         nonzero_deltas > 0, 'Clipped norm bound or actual nonzero update failed')
    need(schedule['macro_updates'] == 3557 and math.ceil(0.03 * schedule['macro_updates']) == 107,
         'Frozen complete-macro schedule/warmup changed')
    used_lr, next_lr = 2e-5 / 107, 2e-5 * 2 / 107
    same(meta['used_lr'], used_lr, 'Actual used LR is not2e-5/107')
    expected_scheduler = {'kind': 'actual-macro-warmup-cosine-v1', 'total_steps': 3557,
                          'warmup_steps': 107, 'completed': 1, 'next_lr': next_lr}
    same(actual['scheduler'], expected_scheduler, 'Actual scheduler state differs')
    opt = actual['optimizer']
    need(set(opt) == {'state', 'param_groups'} and len(opt['param_groups']) == 1,
         'Actual optimizer structure differs')
    group = opt['param_groups'][0]
    same(group['param_names'], names, 'Actual optimizer full-name order differs')
    same(group['params'], list(range(393)), 'Actual optimizer parameter IDs/order differs')
    need(all(type(key) is int for key in opt['state']) and set(opt['state']) == set(range(393)),
         'Actual optimizer lacks all393 integer-keyed states')
    need(group['lr'] == next_lr and tuple(group['betas']) == (0.9, 0.999) and
         group['eps'] == 1e-8 and group['weight_decay'] == 0.01 and group['fused'] is True and
         group['amsgrad'] is False and group['maximize'] is False,
         'Actual AdamW frozen hyperparameters differ')
    for index, name in enumerate(names):
        state = opt['state'][index]
        need(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Actual AdamW state members differ')
        finite_tensor(torch, state['step'], dtype=torch.float32)
        need(state['step'].numel() == 1 and float(state['step']) == 1.0, 'Actual AdamW step !=1')
        for key in ('exp_avg', 'exp_avg_sq'):
            finite_tensor(torch, state[key], shape=raw[name].shape, dtype=torch.float32)
        need(bool((state['exp_avg_sq'] >= 0).all()), 'Actual AdamW second moment is negative')
    return {'gradient_keys': 393, 'actual_raw_l2_float64': independent_norm,
            'actual_clipped_l2_float64': clipped_norm, 'saved_preclip_l2': preclip,
            'saved_clip_factor': factor, 'actual_fp32_clip_multiplier': float(clip_tensor),
            'all_clipped_tensor_bytes_match_raw_times_actual_multiplier': True,
            'preclip_norm_fixed_relative_tolerance': PRECLIP_REL_TOL,
            'preclip_norm_fixed_absolute_tolerance': PRECLIP_ABS_TOL,
            'per_key_double_norm_fixed_relative_tolerance': GRAD_STAT_REL_TOL,
            'per_key_double_norm_fixed_absolute_tolerance': GRAD_STAT_ABS_TOL,
            'nonzero_delta_elements': nonzero_deltas, 'all393_actual_deltas_and_Adam_step1_verified': True,
            'used_lr': used_lr, 'scheduler': expected_scheduler,
            'independent_model_parameter_gradient_mathematics_verified': False}


def verify_saved(root, arm_name, stage, evidence):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CUDA_VISIBLE_DEVICES=\'\' is required')
    need('torch' not in sys.modules, 'Start a fresh CPU-only verifier process')
    need(not list(root.glob('*BLOCKED*')), 'Follow-up root is blocked')
    config_path = root / 'seed42.json'
    cfg = evidence.read(config_path, exact=True)
    need(cfg['output_parent'] == str(root) and cfg['seed'] == 42, 'Actual final root/seed differs')
    for record in cfg['trainer_files'] + cfg['core_files'] + cfg['native_files'] + cfg['stock_files']:
        evidence.add(record['path'], record)
    evidence.source_map(cfg['trainer_files'])
    evidence.source_map(cfg['core_files'])
    evidence.source_map([p for p in cfg['native_files'] if Path(p['path']).suffix == '.py'])
    sys.path.insert(0, str(root / 'source'))
    from bindings import Context
    from fresh_init import validate_init
    ctx = Context(config_path, arm_name)
    identity = validate_init(ctx)
    need(ctx.arm['mode'] == ARMS[arm_name][0] and ctx.arm['loss'] == ARMS[arm_name][1] and
         ctx.arm['layout'] == 'packed', 'Explicit actual arm mode/loss/layout differs')
    same(ctx.snapshot['trainer_code'], evidence.source_map(cfg['trainer_files']), 'Actual new source bytes differ')
    same(ctx.snapshot['core_code'], evidence.source_map(cfg['core_files']), 'Actual new core bytes differ')
    same(ctx.snapshot['config_pin'], evidence.add(config_path), 'Actual final config pin differs')
    evidence.collect(ctx.snapshot)
    evidence.collect(identity)
    # Explicitly retain the entire boundary producer/analysis/config byte chain.
    same(ctx.state_boundary['source_bytes'], evidence.source_map(ctx.state_boundary['exact_byte_pins']),
         'Actual boundary source/config byte identity differs')
    folder = root / (arm_name + '-seed42')
    need(folder.is_dir() and folder.resolve() == folder and not folder.is_symlink(), 'Actual arm redirected')
    # Hold the existing trainer lock, without changing its contents. A live
    # trainer or another own-resume prevents saved-only admission immediately.
    with (folder / 'RUN.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence.add(folder / 'RUN.lock')
        events, start, update, tip = read_paused(evidence, folder, ctx.snapshot)
        artifact = folder / 'first-actual-macro'
        first = complete(evidence, artifact)
        need(first['is_formal_exposure_macro0'] is True, 'Not the formal first actual macro')
        same(first['scope'], ctx.snapshot, 'Formal actual COMPLETE scope differs')
        old_cfg = evidence.read(OLD / 'seed42.json', exact=True)
        evidence.collect(old_cfg)
        control_name = ARMS[arm_name][2]
        control = OLD / (control_name + '-seed42') / 'first-actual-macro'
        control_complete = complete(evidence, control)
        need(control_complete['is_formal_exposure_macro0'] is True, 'Old control not actual formal macro0')
        old_scope(evidence, ctx, old_cfg, control_complete['scope'], control_name)
        mapping = original_mapping(evidence, ctx)
        rows, expected_slots, old_expected, fields, micros, coverage, full_train, schedule = mapping
        same(start['exposures'], rows, 'Ledger actual exposure differs from original stream')
        slots = evidence.read(artifact / 'SLOTS.json')
        old_slots = evidence.read(control / 'SLOTS.json')
        for value, expected in ((slots, expected_slots), (old_slots, old_expected)):
            need(value['macro_index'] == 0 and value['not_an_extra_smoke'] is True,
                 'SLOTS is not formal macro0')
            same(value['exposures'], rows, 'Actual/control exposure differs from original')
            same(value['fields'], fields, 'Actual/control complete fields/host-q differ from original')
            same(value['slots'], expected, 'Actual/control slot mapping differs from independent original')
            same(value['micros'], micros, 'Actual/control micros differ from independent original')
        without_prefix = lambda values: [{k: v for k, v in item.items() if k != 'prefix_tokens'} for item in values]
        same(without_prefix(slots['slots']), without_prefix(old_slots['slots']), 'Old/new full-row slot identities differ')
        if arm_name == 'JPAIR_P_BCE_RANK':
            same(slots, old_slots, 'Rank full slot manifest including original prefix differs')
        dispatch = evidence.read(artifact / 'ACTUAL_NATIVE_DISPATCH.json')
        runtime, dispatch_checks = verify_dispatch(dispatch, arm_name, expected_slots, micros)
        import torch
        need(torch.__version__.split('+')[0] == '2.9.1', 'Frozen Torch2.9.1 required for exact saved math')
        torch.set_num_threads(2)
        need(not torch.cuda.is_initialized(), 'Saved-only verifier must not initialize CUDA')
        fresh = verify_fresh(torch, evidence, ctx, identity, old_cfg)
        cache = load_pt(torch, evidence, artifact / 'actual-cache-and-coefficients.pt')
        replay = load_pt(torch, evidence, artifact / 'actual-replay-logits.pt')
        for value in (cache, replay):
            finite_tensor(torch, value['logits'], (len(expected_slots), 2), torch.float32)
            finite_tensor(torch, value['margins'], (len(expected_slots),), torch.float32)
            need(tensor_equal(torch, value['margins'], value['logits'][:, 1] - value['logits'][:, 0]),
                 'Actual cached margins do not equal actual float32 logit difference')
        need(replay['bitwise_equal_to_first'] is True and
             tensor_equal(torch, cache['logits'], replay['logits']) and
             tensor_equal(torch, cache['margins'], replay['margins']),
             'Actual first/replay logits or margins differ in shape/dtype/bytes')
        same(replay['completed_slots'], [s for micro in micros for s in micro], 'Actual replay slot order differs')
        old_actual = load_pt(torch, evidence, control / 'all393-actual-update-and-optimizer.pt')
        compare_parameters(torch, old_actual['before'], fresh, 'Old matched control before != same-seed fresh')
        if arm_name == 'JPAIR_P_BCE_RANK':
            control_cache = load_pt(torch, evidence, control / 'actual-cache-and-coefficients.pt')
            for key in ('logits', 'margins'):
                need(tensor_equal(torch, cache[key], control_cache[key]), 'Rank actual old-control cache bytes differ: ' + key)
        coefficients, loss, book = recompute_objective(torch, cache['margins'], fields, ARMS[arm_name][1])
        finite_tensor(torch, cache['coefficients'], (len(expected_slots),), torch.float32)
        need(tensor_equal(torch, coefficients, cache['coefficients']), 'Actual margin coefficient bytes differ from independent total-last autograd')
        same(cache['objective'], loss, 'Actual cached objective differs from independent operator order')
        same(cache['bookkeeping'], book, 'Actual cached scalar book differs from independent original-host-q recomputation')
        raw = load_pt(torch, evidence, artifact / 'all393-gradients-unclipped.pt')
        clipped = load_pt(torch, evidence, artifact / 'all393-gradients-clipped.pt')
        actual = load_pt(torch, evidence, artifact / 'all393-actual-update-and-optimizer.pt')
        meta = evidence.read(artifact / 'ACTUAL_MACRO.json')
        same(meta['objective'], loss, 'Actual macro objective differs')
        same(meta['bookkeeping'], book, 'Actual macro scalar book differs')
        same(meta['coverage'], coverage, 'Actual raw known/unknown/fullcase coverage differs')
        same(meta['runtime_after'], runtime, 'Actual per-wave/native token counters differ from actual dispatch')
        expected_meta = {'macro_index': 0, 'cases': 16, 'logical_pair_tokens': 20947,
                         'micro_waves_per_pass': len(micros), 'optimizer_steps': 1, 'gradient_keys': 393,
                         'actual_completed_forward_real_tokens_this_macro': 41894,
                         'actual_completed_backward_real_tokens_this_macro': 20947}
        same({key: meta[key] for key in expected_meta}, expected_meta, 'Actual macro counters differ')
        need(all(meta[k] is True for k in
                 ('first_replay_logits_and_margins_bitwise_equal', 'RNG_unchanged_zero_dropout',
                  'all393_gradients_finite', 'trainable_parameters_after_update_finite')),
             'Actual macro producer contract disagrees with saved evidence')
        saved_result = {k: v for k, v in meta.items() if k not in ('gradients', 'clipped_gradients', 'runtime_after')}
        same(update['result'], saved_result, 'Durable actual macro result differs from saved ACTUAL_MACRO')
        same(update['runtime'], runtime, 'Durable actual runtime differs from dispatch')
        gradient_checks = verify_gradients(torch, raw, clipped, actual, fresh,
                                           identity['manifest'], meta, schedule)
        checkpoint = Path(tip['complete']['path']).parent
        saved = saved_tensors(torch, evidence, checkpoint)
        compare_parameters(torch, saved, actual['after'], 'Latest cursor1 checkpoint != actual.after')
        state = load_pt(torch, evidence, checkpoint / 'training_state.pt')
        checkpoint_meta = evidence.read(checkpoint / 'CHECKPOINT.json')
        for value in (state, checkpoint_meta):
            same(value['scope'], ctx.snapshot, 'Checkpoint actual scope differs')
            same(value['cursor'], CURSOR, 'Checkpoint actual cursor differs')
            same(value['runtime'], runtime, 'Checkpoint runtime differs from actual dispatch')
            same(value['dev_done'], [], 'No dev may have run before first primary')
        same(state['trainable_names'], sorted(raw), 'Checkpoint full trainable name order differs')
        need(tree_equal(torch, state['optimizer'], actual['optimizer']) and
             tree_equal(torch, state['scheduler'], actual['scheduler']),
             'Latest checkpoint optimizer/scheduler bytes or scalar types != saved actual update')
        need(tree_equal(torch, state['rng'], cache['rng_before']), 'Zero-dropout checkpoint RNG differs from cached before')
        need(tree_equal(torch, state['initialization'], checkpoint_meta['initialization']),
             'Checkpoint initialization differs from CHECKPOINT metadata')
        same(state['initialization']['fresh_init'], identity, 'Checkpoint actual fresh identity differs')
        same(state['reasons'], checkpoint_meta['reasons'], 'Checkpoint reasons differ')
        same(state['reasons'], tip['reasons'], 'Checkpoint state/durable commit reasons differ')
        same(checkpoint_meta['audit']['saved_keys'], sorted(fresh), 'Checkpoint audit saved keys differ')
        same(checkpoint_meta['audit']['trainable_keys'], sorted(raw), 'Checkpoint audit full names differ')
        # Re-open/rebind all critical identity and latest ledger state while the
        # trainer lock is still held. Report is not a permission to skip this on
        # the main session's subsequent AUTHORIZE_RESUME readback.
        ctx.check()
        same(validate_init(ctx), identity, 'Fresh identity changed before primary publication')
        _, _, _, latest = read_paused(evidence, folder, ctx.snapshot)
        same(latest, tip, 'Latest durable checkpoint changed before publication')
        same(complete(evidence, artifact), first, 'Formal artifact inventory/COMPLETE changed')
        same(complete(evidence, control), control_complete, 'Old control inventory/COMPLETE changed')
        complete(evidence, Path(old_cfg['fresh_init']['directory']), old_cfg['fresh_init']['complete'])
        need(not list(root.glob('*BLOCKED*')), 'Follow-up root became blocked during primary readback')
        need(not torch.cuda.is_initialized(), 'CPU-only gate initialized CUDA')
        evidence.check()
        resume_pin = stage / 'RESUME_COMPLETE_PIN.json'
        write(resume_pin, tip['complete'])
        resume_pin.chmod(0o444)
        with resume_pin.open('rb') as handle:
            os.fsync(handle.fileno())
        sync_dir(stage)
        same(parse(resume_pin.read_text(encoding='utf-8')), tip['complete'], 'Resume pin readback differs')
        need(stat_mode.S_IMODE(resume_pin.stat().st_mode) == 0o444, 'Resume pin must be chmod444')
        report = {
            'version': 'followup-first-primary-saved-only-v1', 'complete': True,
            'arm': arm_name, 'arm_name': arm_name, 'scope': ctx.snapshot,
            'next_macro_index': 1, 'repeat_macro0_allowed': False,
            'resume': {'checkpoint': str(checkpoint), 'complete': tip['complete'], 'pin_file': str(resume_pin)},
            'resume_pin_file': pin(resume_pin), 'formal_first_complete': pin(artifact / 'COMPLETE.json'),
            'old_matched_control_complete': pin(control / 'COMPLETE.json'),
            'fresh_primary': pin(root / 'FRESH_PRIMARY_VERIFICATION.json'),
            'durable_ledger': pin(folder / 'ledger.jsonl'), 'paused_exit_code': 75, 'cursor': CURSOR,
            'admission_authority': 'actual-durable-ledger-paused-cursor1-and-saved-artifacts',
            'queue_file_required': False,
            'macro0_consumed_exactly_once': True, 'no_later_model_or_dev_event': True,
            'saved_cache_replay_shape_dtype_bytes_equal': True,
            'rank_old_control_logits_margins_shape_dtype_bytes_equal': arm_name == 'JPAIR_P_BCE_RANK',
            'state_old_control_logit_equality_required': False,
            'independent_original_case_exposure_slots_fields_micros_verified': True,
            'state_sidecar_original_train_rows_independently_scanned': arm_name == 'STATE_NL_PB_P_CE',
            'independent_full_train_coverage': full_train, 'independent_macro_coverage': coverage,
            'scalar_book': book, 'independent_objective': loss,
            'independent_margin_coefficient_bytes_equal': True,
            'all393_new_fresh_equal_old_oracle_and_actual_before': True,
            'latest_cursor1_checkpoint_weights_optimizer_scheduler_match_saved_actual': True,
            'preclip_and_update_checks': gradient_checks, 'actual_dispatch_checks': dispatch_checks,
            'evidence_pins': [evidence.files[name] for name in sorted(evidence.files)],
            'exact_source_and_config_bytes': dict(sorted(evidence.bytes.items())),
            'allow_limited_scientific_resume': True,
            'admission_scope': 'scientific-full-row-train-dev-only',
            'main_external_authorization_required': True, 'automatic_resume': False,
            'AUTHORIZE_RESUME_created': False, 'training_launched': False,
            'model_forward_calls': 0, 'CUDA_initialized': False,
            'allow_production': False, 'allow_serve': False, 'allow_prefix_cache': False,
            'allow_confirm': False, 'D_select_opened': False, 'D_confirm_opened': False,
            'allow_public': False, 'public_run': False, 'winner_selected': False,
            'gaps': [
                'Actual native geometry/dispatch is checked, not an independent numerical backend comparison.',
                'New-prefix neighbor isolation and prefix-cache/serving/production equivalence are not established.',
                'Saved finite393 gradients, clipping, actual deltas and Adam states do not prove independent model-gradient mathematics.',
                'Full input IDs and QKV values are not saved; original row resplits and actual saved positions/EOS/geometry are checked under exact source bindings.',
                'Generation-before-oracle ordering is bound to the external independent fresh primary report, not reconstructed by building stock here.',
            ],
        }
        # Last inputs check precedes the only success publication. Any exception
        # (even after publication) leaves a BLOCKED marker and cannot admit.
        evidence.check()
        write(stage / 'FIRST_PRIMARY_VERIFICATION.json', report)
        return stage / 'FIRST_PRIMARY_VERIFICATION.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=tuple(ARMS))
    args = parser.parse_args(argv)
    # No writing into an arbitrary root, a redirected directory, or a previous
    # primary attempt. A preexisting primary is preserved, never made writable.
    need(args.root == ROOT and ROOT.is_dir() and ROOT.resolve() == ROOT and not ROOT.is_symlink(),
         'Exact dedicated existing follow-up root required')
    parent = ROOT / 'primary'
    if not parent.exists():
        parent.mkdir(mode=0o700)
        sync_dir(ROOT)
    need(parent.is_dir() and parent.resolve() == parent and not parent.is_symlink(), 'Primary parent redirected')
    stage = parent / args.arm
    stage.mkdir(mode=0o700)
    evidence = Evidence()
    try:
        os.umask(0o077)
        sync_dir(parent)
        evidence.add(Path(__file__).absolute(), exact=True)
        need(Path(__file__).absolute() == ROOT / 'ops/verify_first.py', 'Run the deployed verifier, not an unbound copy')
        path = verify_saved(ROOT, args.arm, stage, evidence)
        print(dump({'complete': True, 'arm': args.arm, 'reportpath': str(path),
                    'automatic_resume': False}), flush=True)
        return 0
    except BaseException as error:
        blocked = stage / 'BLOCKED.json'
        write(blocked, {'complete': False, 'arm': args.arm, 'error_type': type(error).__name__,
                        'error': str(error), 'partial_preserved': True, 'automatic_resume': False,
                        'allow_limited_scientific_resume': False,
                        'next_macro_index': 1, 'repeat_macro0_allowed': False})
        print(dump({'complete': False, 'arm': args.arm, 'reportpath': str(blocked)}),
              file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
