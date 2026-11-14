"""Fixed terminal weights, one full frozen select, packed admission (not training).

Module imports are stdlib-only so ops may share preflight/receipt helpers without
importing Torch. All model imports occur in worker(), after source/data preflight.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import importlib
import json
import math
import os
from pathlib import Path
import stat as modes
import struct
import sys
import time
from types import MappingProxyType

sys.dont_write_bytecode = True
ROOT = Path('/tmp/opencode/jev-terminal-packed-admission-20261112-v1')
LOCAL = Path('/tmp/opencode/jev-terminal-packed-admission-20261112-local')
ORIGINAL = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
FOLLOWUP = Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
CONTROL = Path('/tmp/opencode/jev-clean-followup-staged-20261112-v2')
MODEL_PYTHON = '/home/dev/assumeengage/envs/jev/bin/python'
ARMS = ('C_D_BCE', 'C_D_CE')
CODE = ('bindings.py', 'fresh_init.py', 'case_runtime.py', 'objectives.py',
        'model_runtime.py', 'checkpoints.py', 'train_case.py', 'evaluate_native_dev.py')
DOMAINS = ('knowledge', 'language', 'retrieval', 'tools', 'arts')
CURSOR = {'macro_updates': 3557, 'cases': 56912, 'logical_pair_tokens': 99824365}
CAPS = [10_000_000, 30_000_000, 100_000_000]
NEW_CAP, OLD_CAP, SHARED, OLD_EVIDENCE = 5 * 2**30, 25 * 2**30, 20 * 2**30, 5 * 2**30
METRICS = ('coverage_adjusted_argmax_target_mass', 'supported_known_field_argmax_target_mass',
           'supported_known_field_NLL', 'supported_known_field_normalized_NLL',
           'supported_known_field_Brier_sum', 'known_field_coverage')
PIN_KEYS = {'path', 'resolved', 'bytes', 'mtime_ns'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def decode(text):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'Duplicate JSON key: ' + k)
            result[k] = v
        return result
    return json.loads(text, parse_constant=invalid, object_pairs_hook=unique)


def read_json(path):
    return decode(Path(path).read_text(encoding='utf-8'))


def pin(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and not p.is_symlink(), 'Redirected file: ' + str(p))
    s = p.lstat()
    require(modes.S_ISREG(s.st_mode), 'Not regular: ' + str(p))
    return {'path': str(p), 'resolved': str(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def verify(records):
    for record in records:
        require(isinstance(record, dict) and set(record) == PIN_KEYS and pin(record['path']) == record,
                'Missing/changed stat pin: ' + str(record.get('path')))


def readonly(path):
    pin(path)
    require(Path(path).stat().st_mode & 0o222 == 0, 'Frozen readonly input required: ' + str(path))


def entry(path):
    before = pin(path)
    data = Path(path).read_bytes()
    verify([before])
    return {'stat': before, 'base64': base64.b64encode(data).decode('ascii')}


def exact(item):
    verify([item['stat']])
    require(Path(item['stat']['path']).read_bytes() == base64.b64decode(item['base64'], validate=True),
            'Small-source exact bytes changed: ' + item['stat']['path'])


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes(path, data):
    with Path(path).open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())
    require(Path(path).read_bytes() == data, 'Exact write readback failed')
    sync_dir(Path(path).parent)


def commit(path, value):
    # Exclusive final names; no replace, unlink, cleanup, digest or mutable receipt.
    write_bytes(path, (dump(value) + '\n').encode('utf-8'))


def blocked(folder):
    require(not list(Path(folder).glob('BLOCKED*')), 'Existing producer/evaluation blocker: ' + str(folder))


def committed(marker, scope):
    marker = Path(marker)
    blocked(marker.parent)
    value = read_json(marker)
    require(value['complete'] is True and value['scope'] == scope, 'Committed scope mismatch')
    verify(value['files'])
    paths = list(marker.parent.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Symlink in committed artifact')
    require({str(p) for p in paths if p.is_file() and p != marker} ==
            {r['path'] for r in value['files']}, 'Unbound/missing committed artifact')
    return value


def original_bindings():
    source = ORIGINAL / 'source'
    require(source.resolve() == source, 'Original source directory redirected')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module('bindings')
    require(Path(module.__file__).resolve() == source / 'bindings.py', 'Wrong original bindings import')
    return module


def producer_record(arm):
    """Actual immutable metadata only, before any Torch import; never training_state."""
    require(arm in ARMS, 'Only the two fixed dense producers')
    b = original_bindings()
    ctx = b.Context(ORIGINAL / 'seed42.json', arm)
    cfg, scope = ctx.cfg, ctx.snapshot
    require(cfg['seed'] == 42 and cfg['output_parent'] == str(ORIGINAL) and
            cfg['precision']['cublas_workspace_config'] == ':4096:8' and
            ctx.arm == {'mode': 'causal', 'layout': 'dense', 'loss': 'BCE' if arm == 'C_D_BCE' else 'CE'},
            'Original producer seed/root/precision/dense arm changed')
    require({r['path'] for r in cfg['trainer_files']} == {str(ORIGINAL / 'source' / n) for n in CODE},
            'Frozen original eight sources required')
    for r in cfg['trainer_files']:
        readonly(r['path'])
    folder = ORIGINAL / (arm + '-seed42')
    blocked(folder)
    done_pin = pin(folder / 'COMPLETE.json')
    done = read_json(done_pin['path'])
    require(done['complete'] is True and done['scope'] == scope and read_json(folder / 'OWN_SCOPE.json') == scope and
            done['cursor'] == CURSOR and done['observations'] == CAPS and done['all_checkpoints_retained'] is True and
            done['public_run'] is False and done['D_confirm_opened'] is False, 'Actual producer terminal incomplete')
    schedule_path = Path(cfg['data_root']) / 'data/schedule-seed42.json'
    require(pin(schedule_path) in ctx.data['files'] and read_json(schedule_path) == done['schedule'],
            'Producer schedule not frozen original seed42')
    nodes = done['schedule']['observations']
    require(len(nodes) == 3 and {n['cap'] for n in nodes} == set(CAPS), 'Original three observations required')
    node = next(n for n in nodes if n['cap'] == 100_000_000)
    require({k: node[k] for k in CURSOR} == CURSOR and len(done['D_select']) == 3,
            'Original 100M observation cursor differs')
    verify(done['checkpoints'] + done['D_select'] + [done['final_checkpoint'], done['first_actual_macro']])
    require(done['final_checkpoint'] in done['checkpoints'] and
            {r['path'] for r in done['checkpoints']} ==
            {str(p) for p in folder.glob('checkpoint-*/COMPLETE.json')}, 'Retained original checkpoints differ')
    terminal = Path(done['final_checkpoint']['path']).parent
    require(terminal.parent == folder and done['final_checkpoint']['path'] == str(terminal / 'COMPLETE.json'),
            'Terminal checkpoint redirected')
    ckpt = committed(terminal / 'COMPLETE.json', scope)
    meta = read_json(terminal / 'CHECKPOINT.json')
    require(ckpt['macro_updates'] == 3557 and ckpt['resumable_full_macro_boundary'] is True and
            meta['scope'] == scope and meta['cursor'] == CURSOR and sorted(meta['dev_done']) == CAPS and
            meta['audit']['saved_key_count'] == 393 and meta['audit']['lora_pairs'] == 196 and
            meta['audit']['trainable_tensors'] == 393 and meta['audit']['trainable_parameters'] == 20187136 and
            meta['audit']['active_served_parameters'] == 616237056 and
            meta['audit']['unused_registered_parameters'] == 2048, 'Terminal metadata/audit differs')
    require({str(terminal / n) for n in ('CHECKPOINT.json', 'adapter_model.safetensors', 'adapter_config.json')} <=
            {r['path'] for r in ckpt['files']}, 'Terminal loader inputs unpinned')
    references, seen = {}, set()
    for r in done['D_select']:
        marker = Path(r['path'])
        require(marker.name == 'COMPLETE.json' and marker.parent.parent == folder, 'Dense node redirected')
        observation = committed(marker, scope)
        n = observation['node']
        require(n in nodes and n['cap'] not in seen and
                observation['checkpoint_identity']['macro_updates'] == n['macro_updates'], 'Dense node identity differs')
        summary = read_json(marker.parent / 'SUMMARY.json')
        require(summary['scope'] == scope and summary['node'] == n and
                summary['checkpoint_identity'] == observation['checkpoint_identity'] and
                summary['roster'] == 'D_select_only' and
                all(summary[k] is False for k in ('D_confirm_opened', 'public_run', 'winner_selected', 'prefix_cache')),
                'Dense reference scope differs')
        seen.add(n['cap'])
        references[n['cap']] = {'complete': r, 'summary': pin(marker.parent / 'SUMMARY.json'),
                                'raw': pin(marker.parent / 'case-metrics.jsonl')}
    require(seen == set(CAPS), 'Missing original dense nodes')
    ctx.check()
    verify([done_pin])
    return ctx, {'scope': scope, 'complete': done_pin, 'own_scope': pin(folder / 'OWN_SCOPE.json'),
                 'terminal_complete': done['final_checkpoint'], 'terminal_metadata': pin(terminal / 'CHECKPOINT.json'),
                 'terminal_files': ckpt['files'], 'node': node, 'references': {str(k): v for k, v in references.items()}}


def build_config(deployment_ready):
    """Install-only, stdlib old metadata/source checks; no model/FWD/GPU or queue."""
    require(deployment_ready is True, 'Main reviewed --deployment-ready acknowledgement required')
    producers = {arm: producer_record(arm)[1] for arm in ARMS}
    snapshot = {'version': 'terminal-packed-source-snapshot-v1', 'root': str(ROOT),
                'sources': [entry(ROOT / n) for n in ('consumer.py', 'ops.py')],
                'original': [entry(ORIGINAL / 'seed42.json')] +
                            [entry(ORIGINAL / 'source' / n) for n in CODE]}
    commit(ROOT / 'SOURCE_SNAPSHOT.json', snapshot)
    cfg = {'version': 'terminal-packed-admission-v1', 'root': str(ROOT), 'approved_for_execution': True,
           'main_reviewed_deployment_ready': True, 'seed': 42, 'arms': list(ARMS), 'producers': producers,
           'source_binding': pin(ROOT / 'SOURCE_SNAPSHOT.json'), 'source_files': [x['stat'] for x in snapshot['sources']],
           'resources': {'new_total_cap_bytes': NEW_CAP, 'active_arm_cap_bytes': OLD_CAP,
                         'old_evidence_bytes': OLD_EVIDENCE, 'shared_floor_bytes': SHARED},
           'numeric_contract': {'per_case': True, 'relative_RMS': 0.01, 'RMS_floor': 0.001,
                                'maxabs_absolute': 0.02, 'maxabs_relative': 0.02,
                                'flip_absolute': 0.04, 'flip_relative': 0.04, 'quality_abs_tolerance': 1e-12},
           'D_confirm_opened': False, 'public_run': False, 'winner_selected': False,
           'training': False, 'dense_reference_rerun': False}
    commit(ROOT / 'config.json', cfg)
    require(read_json(ROOT / 'config.json') == cfg, 'Sealed config readback differs')
    sealed_config(ROOT / 'config.json')
    return cfg


def sealed_config(path):
    require(Path(path) == ROOT / 'config.json', 'Fixed deployed config only')
    readonly(path)
    cfg = read_json(path)
    require(cfg['version'] == 'terminal-packed-admission-v1' and cfg['root'] == str(ROOT) and
            cfg['approved_for_execution'] is True and cfg['main_reviewed_deployment_ready'] is True and
            cfg['seed'] == 42 and cfg['arms'] == list(ARMS) and set(cfg['producers']) == set(ARMS),
            'Unreviewed or wrong consumer config')
    require(cfg['resources'] == {'new_total_cap_bytes': NEW_CAP, 'active_arm_cap_bytes': OLD_CAP,
                                'old_evidence_bytes': OLD_EVIDENCE, 'shared_floor_bytes': SHARED} and
            cfg['numeric_contract'] == {'per_case': True, 'relative_RMS': 0.01, 'RMS_floor': 0.001,
                                       'maxabs_absolute': 0.02, 'maxabs_relative': 0.02,
                                       'flip_absolute': 0.04, 'flip_relative': 0.04, 'quality_abs_tolerance': 1e-12} and
            all(cfg[k] is False for k in ('D_confirm_opened', 'public_run', 'winner_selected', 'training',
                                         'dense_reference_rerun')), 'Frozen admission contract changed')
    require(cfg['source_binding']['path'] == str(ROOT / 'SOURCE_SNAPSHOT.json'), 'Source binding redirected')
    verify([cfg['source_binding']])
    readonly(cfg['source_binding']['path'])
    snapshot = read_json(cfg['source_binding']['path'])
    require(snapshot['version'] == 'terminal-packed-source-snapshot-v1' and snapshot['root'] == str(ROOT) and
            len(snapshot['sources']) == 2 and {x['stat']['path'] for x in snapshot['sources']} ==
            {str(ROOT / n) for n in ('consumer.py', 'ops.py')} and
            cfg['source_files'] == [x['stat'] for x in snapshot['sources']] and
            len(snapshot['original']) == 9 and {x['stat']['path'] for x in snapshot['original']} ==
            {str(ORIGINAL / 'seed42.json')} | {str(ORIGINAL / 'source' / n) for n in CODE}, 'Source inventory differs')
    for item in snapshot['sources'] + snapshot['original']:
        exact(item)
        readonly(item['stat']['path'])
    require('torch' not in sys.modules or os.environ.get('JEV_PACKED_WORKER') == '1',
            'Manager must remain Torch-free')
    return cfg


class PackedView:
    """Read-only view; original Context remains a true dense identity guard."""
    def __init__(self, original):
        self.cfg = MappingProxyType(original.cfg)
        self.arm = MappingProxyType({'mode': 'causal', 'layout': 'packed', 'loss': original.arm['loss']})
        self.arm_name = original.arm_name + '-fixed-terminal-packed-consumer'


def near(actual, expected, label):
    if expected is None or isinstance(expected, (bool, str)):
        require(type(actual) is type(expected) and actual == expected, 'Metric mismatch: ' + label)
    elif isinstance(expected, int):
        require(type(actual) is int and actual == expected, 'Count mismatch: ' + label)
    else:
        require(type(actual) in (int, float) and math.isfinite(actual) and
                abs(actual - expected) <= 1e-12, 'Quality recomputation mismatch: ' + label)


def mean(values):
    return sum(values) / len(values) if values else None


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def field_quality(margins, q):
    best = max(range(len(margins)), key=lambda i: margins[i])  # first native tie
    maximum = max(margins)
    normalizer = math.log(math.fsum(math.exp(x - maximum) for x in margins))
    log_p = [(x - maximum) - normalizer for x in margins]
    p = [math.exp(x) for x in log_p]
    nll = -math.fsum(a * b for a, b in zip(q, log_p))
    return {'argmax_index': best, 'argmax_target_mass': float(q[best]), 'NLL': nll,
            'Brier_sum': math.fsum((a - b)**2 for a, b in zip(p, q)),
            'normalized_NLL': nll / math.log(len(q)) if len(q) > 1 else 0.0}, p


def recompute(raw_path, summary_path, store):
    """Full independent raw/roster/quality check. Does not import tensor libraries."""
    before = [pin(raw_path), pin(summary_path)]
    summary = read_json(summary_path)
    rows, coverage = {}, Counter()
    roster = list(store.offsets)
    primary, secondary = {d: [] for d in DOMAINS}, {d: [] for d in DOMAINS}
    totals = Counter()
    with Path(raw_path).open(encoding='utf-8') as handle:
        for index, line in enumerate(handle):
            result = decode(line)
            require(index < len(store.offsets), 'Extra raw case')
            cid = roster[index]
            case = store.get(cid)
            require(result['case_id'] == cid and cid not in rows and result['component'] == case['component'] and
                    result['primary_domain'] == case['primary_domain'] and
                    len(result['fields']) == len(case['fields']), 'Raw case/order/component mismatch')
            fields, known = [], sum(f['known'] for f in case['fields'])
            for fi, (item, raw) in enumerate(zip(result['fields'], case['fields'])):
                require(item['field_index'] == fi and item['coordinate'] == raw['coordinate'] and
                        item['domain'] == raw['domain'] and type(item['known']) is bool and
                        item['known'] == raw['known'] and item['K'] == (len(raw['options']) if raw['known'] else None),
                        'Raw native field identity/order differs')
                if not raw['known']:
                    require(item['status'] == 'masked_unknown_no_forward' and
                            item['unknown_reason'] == raw['unknown_reason'] and
                            not ({'logits', 'margins', 'native_target_not_renormalized', 'argmax_index',
                                  'argmax_target_mass', 'NLL', 'Brier_sum', 'normalized_NLL'} & set(item)),
                            'Unknown field fabricated prediction/target')
                    coverage['unknown_fields_retained_no_forward'] += 1
                    continue
                require(raw['static_unsupported_reason'] is None and item['status'] == 'supported_complete_field' and
                        item['native_target_not_renormalized'] == raw['target'], 'Missing/changed complete native field')
                margins, logits, q = item['margins'], item['logits'], raw['target']
                require(len(margins) == len(logits) == len(q) and
                        all(type(x) in (int, float) and math.isfinite(x) and fp32(x) == x for x in margins),
                        'Invalid FP32 margin array')
                for pair, margin in zip(logits, margins):
                    require(len(pair) == 2 and all(type(x) in (int, float) and math.isfinite(x) and fp32(x) == x and
                            struct.unpack('<I', struct.pack('<f', x))[0] & 0xffff == 0 for x in pair),
                            'Logits must be actual FP32 representations of BF16 outputs')
                    require(struct.pack('<f', fp32(pair[1] - pair[0])) == struct.pack('<f', margin),
                            'Saved margin is not the actual FP32 subtraction')
                quality, _ = field_quality(margins, q)
                for key, value in quality.items():
                    near(item[key], value, cid + '.' + str(fi) + '.' + key)
                fields.append({**quality, 'domain': raw['domain']})
                coverage['supported_known_fields'] += 1
                coverage['forwarded_complete_candidates'] += len(q)
                totals['real_tokens'] += sum(raw['pair_lengths'])
            exact_supported = case['complete_case_exact_static_supported']
            require(type(exact_supported) is bool and (not exact_supported or
                    (known == len(case['fields']) == len(fields) and all(f['onehot'] for f in case['fields']))),
                    'Original exact-eligible case is not complete known onehot')
            exact_value = all(f['argmax_target_mass'] == 1.0 for f in fields) if exact_supported else None
            values = {'known_fields': known, 'supported_known_fields': len(fields),
                      'unknown_fields': len(case['fields']) - known, 'known_field_coverage': len(fields) / known,
                      'coverage_adjusted_argmax_target_mass': sum(f['argmax_target_mass'] for f in fields) / known,
                      'supported_known_field_argmax_target_mass': mean([f['argmax_target_mass'] for f in fields]),
                      'supported_known_field_NLL': mean([f['NLL'] for f in fields]),
                      'supported_known_field_normalized_NLL': mean([f['normalized_NLL'] for f in fields]),
                      'supported_known_field_Brier_sum': mean([f['Brier_sum'] for f in fields]),
                      'complete_onehot_case_exact_supported': exact_supported,
                      'complete_onehot_case_exact': exact_value}
            for key, value in values.items():
                near(result[key], value, cid + '.' + key)
            primary[case['primary_domain']].append(values)
            for domain in DOMAINS:
                observed = [f for f in fields if f['domain'] == domain]
                domain_known = [f for f in case['fields'] if f['known'] and f['domain'] == domain]
                if domain_known:
                    secondary[domain].append({'argmax_target_mass': mean([f['argmax_target_mass'] for f in observed]),
                                              'NLL': mean([f['NLL'] for f in observed]),
                                              'Brier_sum': mean([f['Brier_sum'] for f in observed]),
                                              'known_field_coverage': len(observed) / len(domain_known)})
            coverage['cases'] += 1
            coverage['known_fields'] += known
            coverage['cases_with_any_supported_known_field'] += len(fields) > 0
            coverage['cases_with_all_known_fields_supported'] += len(fields) == known
            coverage['complete_onehot_case_exact_supported'] += exact_supported
            rows[cid] = result
    require(len(rows) == len(store.offsets) == 375 and store.domains == {d: 75 for d in DOMAINS} and
            coverage == Counter({'cases': 375, 'known_fields': 615, 'supported_known_fields': 615,
                                 'forwarded_complete_candidates': 5190, 'unknown_fields_retained_no_forward': 90,
                                 'cases_with_any_supported_known_field': 375, 'cases_with_all_known_fields_supported': 375,
                                 'complete_onehot_case_exact_supported': 259}) and totals['real_tokens'] == 1554484,
            'Full original select coverage/token/exact inventory differs')
    require(summary['coverage'] == dict(coverage) and summary['complete_onehot_case_exact_denominator'] == 259,
            'SUMMARY coverage differs from raw')
    per_domain = {}
    for d, values in primary.items():
        require(len(values) == 75 and summary['per_domain'][d]['cases'] == 75, 'Primary domain quota differs')
        per_domain[d] = {'cases': 75}
        for key in METRICS:
            values_k = [v[key] for v in values if v[key] is not None]
            per_domain[d][key] = mean(values_k)
            near(summary['per_domain'][d][key], mean(values_k), d + '.' + key)
            near(summary['per_domain'][d][key + '_case_denominator'], len(values_k), d + '.denominator')
    five = {k: mean([per_domain[d][k] for d in DOMAINS]) for k in METRICS}
    for key, value in five.items():
        near(summary['five_domain_case_means'][key], value, 'five.' + key)
    field_domain = {}
    for d, values in secondary.items():
        actual = summary['field_domain_case_means_secondary_not_selection_metric'][d]
        near(actual['cases_with_known_field_in_domain'], len(values), 'field-domain.' + d)
        field_domain[d] = {'cases_with_known_field_in_domain': len(values)}
        for key in ('argmax_target_mass', 'NLL', 'Brier_sum', 'known_field_coverage'):
            observed = [v[key] for v in values if v[key] is not None]
            field_domain[d][key] = mean(observed)
            near(actual[key], mean(observed), 'field-domain.' + d + '.' + key)
            near(actual[key + '_case_denominator'], len(observed), 'field-domain.denominator')
    exacts = [v['complete_onehot_case_exact'] for values in primary.values() for v in values
              if v['complete_onehot_case_exact_supported']]
    near(summary['complete_onehot_case_exact'], mean(exacts), 'complete exact')
    verify(before)
    return rows, {'coverage': dict(coverage), 'real_tokens': totals['real_tokens'],
                  'five_domain_case_means': five, 'per_domain': per_domain,
                  'field_domain_case_means_secondary_not_selection_metric': field_domain,
                  'complete_onehot_case_exact': mean(exacts), 'complete_onehot_case_exact_denominator': len(exacts),
                  'all_raw_field_case_domain_summary_metrics_recomputed': True,
                  'absolute_tolerance': 1e-12, 'native_q_renormalized': False}


def error_gate(actual, reference):
    require(len(actual) == len(reference) > 0, 'Numeric shape differs')
    delta = [x - y for x, y in zip(actual, reference)]
    reference_rms = math.sqrt(math.fsum(x*x for x in reference) / len(reference))
    error_rms = math.sqrt(math.fsum(x*x for x in delta) / len(delta))
    reference_max = max(abs(x) for x in reference)
    error_max = max(abs(x) for x in delta)
    rms_limit = 0.01 * max(reference_rms, 0.001)
    max_limit = 0.02 + 0.02 * reference_max
    return {'reference_RMS': reference_rms, 'error_RMS': error_rms, 'RMS_limit': rms_limit,
            'reference_maxabs': reference_max, 'error_maxabs': error_max, 'maxabs_limit': max_limit,
            'RMS_pass': error_rms <= rms_limit, 'maxabs_pass': error_max <= max_limit}


def numerical_compare(new, old):
    require(list(new) == list(old) and len(new) == 375, 'Numeric full roster mismatch')
    cases, violations = [], Counter()
    choice_flips = outside = 0
    max_probability_delta = 0.0
    for cid, current in new.items():
        reference = old[cid]
        for key in ('case_id', 'component', 'primary_domain', 'known_fields', 'supported_known_fields', 'unknown_fields'):
            require(current[key] == reference[key], 'Numeric case identity differs')
        require(len(current['fields']) == len(reference['fields']), 'Field array differs')
        nl, ol, nm, om, flips = [], [], [], [], []
        for item, ref in zip(current['fields'], reference['fields']):
            for key in ('field_index', 'coordinate', 'domain', 'known', 'K', 'status'):
                require(item[key] == ref[key], 'Numeric native field identity differs')
            if not item['known']:
                continue
            require(item['native_target_not_renormalized'] == ref['native_target_not_renormalized'], 'Native q changed')
            nl.extend(x for pair in item['logits'] for x in pair)
            ol.extend(x for pair in ref['logits'] for x in pair)
            nm.extend(item['margins'])
            om.extend(ref['margins'])
            _, np = field_quality(item['margins'], item['native_target_not_renormalized'])
            _, op = field_quality(ref['margins'], ref['native_target_not_renormalized'])
            max_probability_delta = max(max_probability_delta, max(abs(x-y) for x, y in zip(np, op)))
            if item['argmax_index'] != ref['argmax_index']:
                require(ref['K'] > 1, 'K1 cannot flip')
                ranked = sorted(ref['margins'], reverse=True)
                gap = ranked[0] - ranked[1]
                band = 0.04 + 0.04 * max(abs(x) for x in ref['margins'])
                in_band = gap <= band
                choice_flips += 1
                outside += not in_band
                flips.append({'field_index': item['field_index'], 'reference_gap': gap,
                              'reference_flip_band': band, 'in_band': in_band})
        logits, margins = error_gate(nl, ol), error_gate(nm, om)
        for name, gate in (('logits', logits), ('margins', margins)):
            for key in ('RMS_pass', 'maxabs_pass'):
                if not gate[key]:
                    violations[name + '.' + key] += 1
        cases.append({'case_id': cid, 'logits': logits, 'margins': margins, 'choice_flips': flips,
                      'four_numeric_predicates_pass': all(g[k] for g in (logits, margins)
                                                         for k in ('RMS_pass', 'maxabs_pass'))})
    accepted = not violations and outside == 0
    return {'scope': 'fixed-weights-original375-select-only-not-public-or-global-geometry',
            'per_case': cases, 'case_count': len(cases), 'violation_counts': dict(violations),
            'numeric_bad_cases': sum(not c['four_numeric_predicates_pass'] for c in cases),
            'choice_flips': choice_flips, 'outside_reference_band_flips': outside,
            'max_field_probability_absolute_difference': max_probability_delta,
            'max_case_error_RMS': {k: max(c[k]['error_RMS'] for c in cases) for k in ('logits', 'margins')},
            'max_case_error_maxabs': {k: max(c[k]['error_maxabs'] for c in cases) for k in ('logits', 'margins')},
            'numerical_acceptance': accepted, 'eligible_packed_candidate': accepted,
            'tolerance_relaxed': False, 'retry': False, 'aggregate_RMS_used_as_gate': False}


def worker(config_path, arm, output):
    require(Path(__file__).resolve() == ROOT / 'consumer.py' and Path(__file__).absolute() == ROOT / 'consumer.py',
            'Only deployed consumer may execute')
    require(Path(output) == ROOT / (arm + '-packed') and arm in ARMS and ROOT.resolve() == ROOT,
            'Exclusive fixed consumer output only')
    require(not Path(output).exists() and not Path(output).is_symlink(), 'Existing output preserved; no retry')
    Path(output).mkdir(mode=0o700)
    stage = Path(output)
    store, runtime, model, oracle = None, None, None, None
    started = time.monotonic()
    try:
        cfg = sealed_config(config_path)
        cfg_bytes, cfg_pin = Path(config_path).read_bytes(), pin(config_path)
        # A direct CLI invocation cannot select an arbitrary visible GPU: the
        # deployed manager must already have sealed the0/1 UUID launch grant.
        launch = read_json(ROOT / 'LAUNCH.json')
        owner = read_json(ROOT / 'OWNER.json')
        gpu = 0 if arm == 'C_D_BCE' else 1
        parent = os.getppid()
        parent_fields = (Path('/proc') / str(parent) / 'stat').read_text().rsplit(') ', 1)[1].split()
        parent_id = {'pid': parent, 'starttime': parent_fields[19],
                     'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        require(launch['owner'] == owner['owner'] == parent_id and launch['config'] == owner['config'] == cfg_pin and
                launch['assignments'] == {'C_D_BCE': 0, 'C_D_CE': 1} and launch['leases_fsynced_before_spawn'] is True and
                os.environ.get('CUDA_VISIBLE_DEVICES') == launch['devices'][str(gpu)],
                'Worker requires its exact manager parent and assigned GPU0/1 UUID')
        producer, record = producer_record(arm)
        require(record == cfg['producers'][arm], 'Pinned producer changed since install')
        b = original_bindings()
        b.offline_environment()
        require('torch' not in sys.modules, 'Preflight must precede Torch')
        from case_runtime import CaseStore, micro_plan, slots
        from model_runtime import FullRow, configure_cuda
        from fresh_init import adapter_state, audit_model, build_stock, load_saved_adapter
        from evaluate_native_dev import evaluate_select
        for name in ('case_runtime', 'model_runtime', 'fresh_init', 'evaluate_native_dev'):
            require(Path(sys.modules[name].__file__).resolve() == ORIGINAL / 'source' / (name + '.py'),
                    'Imported wrong old helper: ' + name)
        store = CaseStore(Path(producer.cfg['data_root']) / 'data/select.jsonl', 'validation')
        require(store.domains == {d: 75 for d in DOMAINS} and len(store.offsets) == 375, 'Frozen select only')
        ref = record['references']['100000000']
        old_rows, old_quality = recompute(ref['raw']['path'], ref['summary']['path'], store)
        waves = sum(len(micro_plan(slots([case], dev=True)[0], dev=True)) for case in store)
        original_snapshot = dump(producer.snapshot)
        original_arm = producer.arm
        view = PackedView(producer)
        require(view.arm is not producer.arm and view.arm['layout'] == 'packed' and producer.arm['layout'] == 'dense',
                'Consumer must not share/mutate producer arm')
        packed_admission = b.bind_admission(producer.cfg, dict(view.arm))
        scope = {'kind': 'fixed-terminal-packed-consumer-v1', 'producer_scope': producer.snapshot,
                 'producer_complete': record['complete'], 'producer_final_checkpoint': record['terminal_complete'],
                 'consumer_arm_name': view.arm_name, 'consumer_arm': dict(view.arm), 'config_pin': cfg_pin,
                 'source_binding': cfg['source_binding'], 'scientific_admission': packed_admission,
                 'fixed_weights': True, 'new_learning_tokens': 0, 'D_confirm_opened': False, 'public_run': False}
        commit(stage / 'CONSUMER_SCOPE.json', scope)

        def guard():
            require(Path(config_path).read_bytes() == cfg_bytes, 'Consumer config changed')
            verify([cfg_pin])
            require(sealed_config(config_path) == cfg, 'Consumer sealed source/config changed')
            producer.check()
            require(producer.arm is original_arm and dump(producer.snapshot) == original_snapshot and
                    dump(producer.cfg) == dump(record['scope']['config']), 'Shared producer scope mutated')
            require(producer_record(arm)[1] == record, 'Terminal/reference identity changed')
            require(b.bind_admission(producer.cfg, dict(view.arm)) == packed_admission, 'Packed admission changed')

        guard()
        environment = configure_cuda(producer)  # first Torch import is inside frozen helper
        import torch
        from safetensors.torch import load_file
        torch.set_num_threads(4)
        terminal = Path(record['terminal_complete']['path']).parent
        # Independent immutable CPU byte oracle, read once; never assigned into model.
        saved = load_file(str(terminal / 'adapter_model.safetensors'), device='cpu')
        oracle = MappingProxyType({k: (tuple(v.shape), str(v.dtype),
                                           v.contiguous().view(torch.uint8).numpy().tobytes())
                                  for k, v in saved.items()})
        require(len(oracle) == 393 and all(v.dtype == torch.float32 for v in saved.values()), '393 FP32 saved oracle')
        del saved
        model = build_stock(42)  # only stock loader, CPU construction, zero forward
        loaded = load_saved_adapter(model, terminal, fresh=False)  # exactly ONE model assignment
        audit = audit_model(model, require_fresh=False)
        require(audit == read_json(record['terminal_metadata']['path'])['audit'] and
                audit['active_served_parameters'] == 616237056 and audit['unused_registered_parameters'] == 2048,
                'Actual stock+trained393 architecture differs')

        def compare_weights():
            current = adapter_state(model)
            require(set(current) == set(oracle), 'Adapter key mutation')
            for key, value in current.items():
                shape, dtype, data = oracle[key]
                require(tuple(value.shape) == shape and str(value.dtype) == dtype and
                        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() == data,
                        'Immutable trained-weight byte mismatch: ' + key)
            params = list(model.parameters())
            require(sum(p.requires_grad for p in params) == 393 and all(p.grad is None for p in params),
                    'requires_grad contract changed or gradient allocated')
            return {'keys_compared': 393, 'shape_dtype_bytes_equal': True, 'all_grad_None': True,
                    'active_requires_grad_true': 393, 'method': 'pure dtype-preserving CPU bytes; no reload/digest'}

        runtime = FullRow(view, model)
        runtime.eval()
        before = compare_weights()  # after GPU move, before any actual forward
        guard()
        checkpoint_identity = {'kind': 'fixed-weight-consumer-not-fresh-training',
                               'producer_final_checkpoint': record['terminal_complete'],
                               'producer_complete': record['complete'], 'macro_updates': 3557,
                               'consumer_scope': 'causal-packed-nativeFA3-fullrow', 'weights_fixed': True}
        with torch.no_grad():
            summary = evaluate_select(runtime, store, stage / 'D_select', scope=scope, node=record['node'],
                                      checkpoint_identity=checkpoint_identity)
        runtime.eval()  # old evaluator finally calls train; eval restores flags, not weights
        after = compare_weights()  # PURE compare only; no reload and no subsequent forward
        receipt = runtime.receipt()
        forward, native = receipt['forward'], receipt['native_forward_calls']
        require(receipt['layout'] == 'packed' and receipt['mode'] == 'causal' and receipt['prefix_cache'] is False and
                set(forward) == {'dev_select.' + k for k in
                                 ('attempted_waves', 'completed_waves', 'attempted_real_tokens', 'completed_real_tokens',
                                  'attempted_tokenwise_rows_per_operator', 'completed_tokenwise_rows_per_operator',
                                  'completed_native_FA3_calls')} and
                all(forward['dev_select.' + k] == waves for k in ('attempted_waves', 'completed_waves')) and
                all(forward['dev_select.' + k] == 1554484 for k in
                    ('attempted_real_tokens', 'completed_real_tokens', 'attempted_tokenwise_rows_per_operator',
                     'completed_tokenwise_rows_per_operator')) and forward['dev_select.completed_native_FA3_calls'] == 28 * waves and
                native == {'attempted_calls': 28 * waves, 'completed_calls': 28 * waves,
                           'attempted_query_tokens': 28 * 1554484, 'completed_query_tokens': 28 * 1554484,
                           'completed_key_tokens': 28 * 1554484, 'completed_causal_calls': 28 * waves},
                'Actual realT/native28/full375 execution counters differ')
        raw_done = committed(stage / 'D_select/COMPLETE.json', scope)
        require(raw_done['node'] == record['node'] and raw_done['checkpoint_identity'] == checkpoint_identity and
                summary == read_json(stage / 'D_select/SUMMARY.json') and summary['runtime_after'] == receipt,
                'Actual packed raw completion identity differs')
        new_rows, new_quality = recompute(stage / 'D_select/case-metrics.jsonl', stage / 'D_select/SUMMARY.json', store)
        numeric = numerical_compare(new_rows, old_rows)
        guard()
        commit(stage / 'WEIGHTS.json', {'loaded_once': loaded, 'audit': audit, 'before': before, 'after': after,
                                       'weights_unchanged': True, 'optimizer_created': False, 'backward_calls': 0,
                                       'post_eval_reload': False})
        commit(stage / 'NUMERICAL.json', numeric)
        commit(stage / 'QUALITY.json', {'kind': 'fixed-weight-layout-delta-not-learning-gain',
                'dense_saved_reference': old_quality, 'actual_packed': new_quality,
                'five_domain_delta_packed_minus_dense': {k: new_quality['five_domain_case_means'][k] -
                                                         old_quality['five_domain_case_means'][k] for k in METRICS},
                'dense_reference_rerun': False, 'winner_selected': False})
        admission = {'execution_complete': True, 'coverage_complete': True, 'weights_unchanged': True,
                     'numerical_acceptance': numeric['numerical_acceptance'],
                     'eligible_packed_candidate': numeric['eligible_packed_candidate'],
                     'eligible_scope': 'this-fixed-checkpoint-full375-select-only',
                     'full_public_or_global_geometry_equivalence': False, 'prefix_serving_admitted': False,
                     'official_latency_claim': False, 'source_review_is_not_runtime_proof': True,
                     'D_confirm_opened': False, 'public_run': False, 'winner_selected': False,
                     'no_training': True, 'new_learning_tokens': 0, 'scope': scope, 'runtime': receipt,
                     'real_tokens': 1554484, 'expected_micro_waves': waves, 'native_calls_per_wave': 28,
                     'head_all_tokens_before_eos': True, 'EOS': 151645, 'elapsed_seconds': time.monotonic() - started,
                     'environment': environment, 'all_raw_retained': True, 'automatic_retry': False}
        commit(stage / 'ADMISSION.json', admission)
        files = [pin(p) for p in sorted(stage.rglob('*')) if p.is_file()]
        commit(stage / 'COMPLETE.json', {'complete': True, **admission, 'files': files})
        return 0  # complete numerical negative is a scientific result, not programming failure
    except BaseException as error:
        if not (stage / 'BLOCKED.json').exists():
            commit(stage / 'BLOCKED.json', {'execution_complete': False, 'eligible_packed_candidate': False,
                    'error_type': type(error).__name__, 'error': str(error), 'partial_preserved': True,
                    'automatic_retry': False, 'post_eval_reload': False, 'public_run': False})
        raise
    finally:
        if store is not None:
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ['JEV_PACKED_WORKER'] = '1'
    return worker(args.config, args.arm, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
