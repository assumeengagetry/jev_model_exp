"""Saved-only five-arm D_select analysis; stdlib CPU, never imports a trainer.

Private cases/targets/logits/token rows remain in memory only. Initialization
equality is attested by the bound actual fresh/MAIN bridges, not re-audited here.
Historical paused ledgers are NOT immutable inputs after cursor1 continuation.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct

ORIGINAL = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
FOLLOWUP = Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
CONTROL = Path('/tmp/opencode/jev-clean-followup-staged-20261112-v2')
DATA = Path('/tmp/opencode/jev-clean-origin-20261112-v1/prepared-01')
ANALYSIS = Path('/tmp/opencode/jev-followup-select-analysis-20261112-v1')
ARMS = ('C_P_CE', 'C_P_BCE', 'PB_P_CE', 'STATE_NL_PB_P_CE', 'JPAIR_P_BCE_RANK')
ROOTS = {arm: ORIGINAL if i < 3 else FOLLOWUP for i, arm in enumerate(ARMS)}
ARM_CONFIG = {
    'C_P_CE': {'mode': 'causal', 'layout': 'packed', 'loss': 'CE'},
    'C_P_BCE': {'mode': 'causal', 'layout': 'packed', 'loss': 'BCE'},
    'PB_P_CE': {'mode': 'prefix_block', 'layout': 'packed', 'loss': 'CE'},
    'STATE_NL_PB_P_CE': {'mode': 'prefix_block', 'layout': 'packed', 'loss': 'CE'},
    'JPAIR_P_BCE_RANK': {'mode': 'causal', 'layout': 'packed', 'loss': 'BCE_RANK_HARD'},
}
DOMAINS = ('knowledge', 'language', 'retrieval', 'tools', 'arts')
CODE = ('bindings.py', 'fresh_init.py', 'case_runtime.py', 'objectives.py',
        'model_runtime.py', 'checkpoints.py', 'train_case.py', 'evaluate_native_dev.py')
PIN_KEYS = {'path', 'resolved', 'bytes', 'mtime_ns'}
ONE = {'macro_updates': 1, 'cases': 16, 'logical_pair_tokens': 20947}
EXPECTED_NODES = {10000000: (365, 9988659), 30000000: (1122, 29985465),
                  100000000: (3557, 99824365)}
OBJECTIVE = {'version': 'hard-label-bce-rank-v1', 'lambda': 1.0, 'tau': 1.0,
             'hard_label_rule': 'original-host-q-exact-zero-or-one',
             'normalization': 'known-field-mean-then-case-mean'}
METRICS = {
    'mass': 'supported_known_field_argmax_target_mass',
    'NLL': 'supported_known_field_NLL',
    'normalized_NLL': 'supported_known_field_normalized_NLL',
    'Brier': 'supported_known_field_Brier_sum',
    'coverage': 'known_field_coverage',
    'coverage_adjusted_mass': 'coverage_adjusted_argmax_target_mass',
}
CONTRASTS = {
    'STATEminusCandidatePB': {'STATE_NL_PB_P_CE': 1, 'PB_P_CE': -1},
    'STATEminusCausalCE': {'STATE_NL_PB_P_CE': 1, 'C_P_CE': -1},
    'JPAIRminusBCE': {'JPAIR_P_BCE_RANK': 1, 'C_P_BCE': -1},
    'JPAIRminusCE': {'JPAIR_P_BCE_RANK': 1, 'C_P_CE': -1},
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def parse(text):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'Duplicate JSON key: ' + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def same(left, right, message):
    need(canonical(left) == canonical(right), message)


def meta(path):
    path = Path(path)
    s = path.stat()
    need(path.is_file(), 'Regular pinned file required: ' + str(path))
    return {'path': str(path), 'resolved': str(path.resolve()),
            'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def pins(value):
    if isinstance(value, dict):
        if set(value) == PIN_KEYS:
            yield value
        else:
            for item in value.values():
                yield from pins(item)
    elif isinstance(value, list):
        for item in value:
            yield from pins(item)


class Evidence:
    def __init__(self):
        self.files, self.exact_bytes, self.historical_ledgers = {}, {}, {}

    def add(self, path, expected=None):
        need(Path(path).name != 'ledger.jsonl', 'Live ledgers must never enter the static receipt set')
        actual = meta(path)
        if expected is not None:
            same(actual, expected, 'Pinned artifact changed: ' + str(path))
        prior = self.files.get(actual['path'])
        need(prior is None or prior == actual, 'Artifact changed during analysis: ' + str(path))
        self.files[actual['path']] = actual
        return actual

    def verify(self, value):
        for item in pins(value):
            need(Path(item['path']).name != 'ledger.jsonl',
                 'Mutable/historical ledger passed to static verification: ' + item['path'])
            self.add(item['path'], item)

    def historical_evidence(self, value):
        """Skip ONLY the two explicitly bound paused pins, never arbitrary ledgers."""
        for item in pins(value):
            if item['path'] in self.historical_ledgers:
                same(item, self.historical_ledgers[item['path']], 'Historical paused ledger binding differs')
            else:
                self.verify(item)

    def read(self, path):
        before = self.add(path)
        text = Path(path).read_bytes()
        self.add(path, before)
        return parse(text)

    def source_map(self, records):
        self.verify(records)
        result = {}
        for item in records:
            path = item['path']
            raw = Path(path).read_bytes()
            result[path] = base64.b64encode(raw).decode('ascii')
            prior = self.exact_bytes.get(path)
            need(prior is None or prior == raw, 'Source bytes changed: ' + path)
            self.exact_bytes[path] = raw
        self.verify(records)
        return result

    def exact(self, item):
        need(set(item) == {'stat', 'base64'}, 'Exact source/config entry schema differs')
        same(self.source_map([item['stat']])[item['stat']['path']], item['base64'],
             'Exact source/config bytes differ: ' + item['stat']['path'])

    def check(self):
        self.verify(list(self.files.values()))
        for path, raw in self.exact_bytes.items():
            need(Path(path).read_bytes() == raw, 'Exact source/config changed: ' + path)
        self.verify(list(self.files.values()))


def close(actual, expected):
    need(math.isfinite(actual) and math.isfinite(expected), 'Nonfinite metric')
    need(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
         'Independent metric differs at unchanged 1e-12 tolerance')


def mean(values):
    return math.fsum(values) / len(values)


def distribution(margins):
    peak = max(margins)
    normalizer = peak + math.log(math.fsum(math.exp(v - peak) for v in margins))
    log_p = [v - normalizer for v in margins]
    return log_p, [math.exp(v) for v in log_p]


def committed_files(e, directory, done):
    """Stat/inventory only, including opaque tensor files; never open weights."""
    directory = Path(directory)
    listed = done['files']
    names = [p['path'] for p in listed]
    need(names and len(names) == len(set(names)) and all(
        Path(name).is_relative_to(directory) and Path(name) != directory / 'COMPLETE.json'
        for name in names), 'Committed file inventory redirected/duplicated')
    actual = set()
    for path in directory.rglob('*'):
        need(not path.is_symlink() and not path.name.startswith('BLOCKED'), 'Blocked/symlink committed artifact')
        if path.is_file() and path != directory / 'COMPLETE.json':
            actual.add(str(path))
    need(set(names) == actual, 'Unbound/missing committed artifact file')
    e.verify(listed)


def unblocked(node=None):
    need(not list(CONTROL.glob('*BLOCKED*')), 'New controller BLOCKED; no automatic restart')
    # This exact old resource-controller failure is bound below, NOT an arm/eval failure.
    need(set(FOLLOWUP.glob('*BLOCKED*')) == {FOLLOWUP / 'QUEUE_BLOCKED.json'},
         'Unaccepted followup root blocker')
    need(not list(DATA.glob('*BLOCKED*')), 'Prepared data blocked')
    for arm in ARMS:
        need(not list((ROOTS[arm] / (arm + '-seed42')).glob('BLOCKED*')), 'Arm BLOCKED: ' + arm)
        if node is not None:
            folder = ROOTS[arm] / (arm + '-seed42') / f"D_select-cap{node['cap']}-m{node['macro_updates']:07d}"
            need(not list(folder.glob('BLOCKED*')), 'Node evaluation BLOCKED: ' + arm)
    for arm in ARMS[3:]:
        need(not list((FOLLOWUP / 'primary' / arm).glob('BLOCKED*')), 'Primary BLOCKED: ' + arm)


def configurations(e):
    cfgs, sources = {}, {}
    for root, version in ((ORIGINAL, 'clean-case-trainer-v1'), (FOLLOWUP, 'clean-case-followup-v1')):
        cfg = e.read(root / 'seed42.json')
        e.source_map([e.add(root / 'seed42.json')])
        need(cfg['version'] == version and cfg['approved_for_execution'] is True and
             cfg['output_parent'] == str(root), 'Wrong actual root configuration')
        need(cfg['seed'] == 42 and cfg['world_size'] == 1 and cfg['prefix_cache'] is False and
             cfg['checkpoint_every_macros'] == 100 and cfg['macro_cases'] == 16 and
             cfg['data_root'] == str(DATA), 'Unmatched seed/world/data/checkpoint contract')
        names = set(CODE) | ({'state_boundaries.py'} if root == FOLLOWUP else set())
        need(len(cfg['trainer_files']) == len(names) and
             {p['path'] for p in cfg['trainer_files']} == {str(root / 'source' / n) for n in names},
             'Exact own trainer inventory differs')
        for key in ('trainer_files', 'core_files', 'native_files', 'stock_files'):
            e.verify(cfg[key])
        e.verify([cfg['data_complete'], cfg['fresh_init']['complete'], cfg['primary_admission']])
        sources[root] = {'trainer_code': e.source_map(cfg['trainer_files']),
                         'core_code': e.source_map(cfg['core_files'])}
        need(cfg['fresh_init']['directory'] == str(root / 'init-seed42') and
             cfg['fresh_init']['complete']['path'] == str(root / 'init-seed42/COMPLETE.json'),
             'Own fresh namespace differs')
        cfgs[root] = cfg
    old, new = cfgs[ORIGINAL], cfgs[FOLLOWUP]
    allowed = {'version', 'output_parent', 'arms', 'trainer_files', 'fresh_init',
               'followup_objective', 'state_boundary'}
    same({k: v for k, v in old.items() if k not in allowed},
         {k: v for k, v in new.items() if k not in allowed},
         'Common STOCK/core/native/precision/data/micro/optimizer/schedule contract differs')
    same(new['arms'], {**old['arms'], **{a: ARM_CONFIG[a] for a in ARMS[3:]}}, 'Followup arm delta differs')
    same(new['followup_objective'], OBJECTIVE, 'Rank scales/objective contract differs')
    need(old['stock_directory'] == '/home/dev/assumeengage/models/Qwen3-0.6B', 'Wrong STOCK')
    for arm in ARMS:
        same(cfgs[ROOTS[arm]]['arms'][arm], ARM_CONFIG[arm], 'Arm mode/layout/loss differs: ' + arm)
    return cfgs, sources


def prepared_data(e, cfg):
    need(cfg['data_complete']['path'] == str(DATA / 'COMPLETE.json'), 'Data COMPLETE redirected')
    complete = e.read(DATA / 'COMPLETE.json')
    need(all(complete[k] is True for k in ('complete', 'DATA_READY', 'train_ready', 'dev_ready')), 'Data not ready')
    need(complete['data_ready']['path'] == str(DATA / 'DATA_READY.json'), 'DATA_READY redirected')
    e.verify(complete['data_ready'])
    ready = e.read(DATA / 'DATA_READY.json')
    need(all(ready[k] is True for k in ('complete', 'DATA_READY', 'train_ready', 'dev_ready')) and
         ready['requires_complete_commit_marker'] is True and ready['macro_cases'] == 16 and
         ready['unknown_forward_rule'] == 'skip-all-unknown-fields-v1', 'Prepared contract differs')
    e.verify(ready['files'])  # Includes confirm STAT only; its contents are never read.
    e.verify([ready['source_binding'], ready['code_binding']])
    need(ready['source_binding']['path'] == str(DATA / 'SOURCE_BINDING.json') and
         ready['code_binding']['path'] == str(DATA / 'CODE_BINDING.json'), 'Prepared bindings redirected')
    source = e.read(ready['source_binding']['path'])
    e.verify(source['files'])
    code = e.read(ready['code_binding']['path'])
    for name, encoded in code.items():
        matches = [p for p in source['files'] if Path(p['path']).name == name]
        need(Path(name).name == name and len(matches) == 1, 'Prepared source binding differs')
        same(e.source_map(matches)[matches[0]['path']], encoded, 'Prepared source bytes differ')
    indexed = {p['path']: p for p in ready['files']}
    need(len(indexed) == len(ready['files']), 'Duplicate prepared artifact pin')
    select_pin = indexed[str(DATA / 'data/select.jsonl')]
    schedule_pin = indexed[str(DATA / 'data/schedule-seed42.json')]
    exposure_pin = indexed[str(DATA / 'data/exposure-seed42.jsonl')]
    schedule = e.read(schedule_pin['path'])
    need(schedule['seed'] == 42 and schedule['data_seed'] == 20261112 and
         schedule['logical_pair_token_ceiling'] == 100000000 and not schedule['incomplete_reasons'] and
         schedule['unknown_forward_rule'] == 'skip-all-unknown-fields-v1' and
         schedule['same_physical_order_for_BCE_and_Field_CE'] is True, 'Pinned schedule differs')
    need(len(schedule['observations']) == 3, 'Exactly three observation nodes required')
    for node in schedule['observations']:
        macro, tokens = EXPECTED_NODES[node['cap']]
        same(node, {'cap': node['cap'], 'macro_updates': macro, 'cases': 16 * macro,
                    'logical_pair_tokens': tokens}, 'Frozen cap/cursor differs')
    need({n['cap'] for n in schedule['observations']} == set(EXPECTED_NODES), 'Schedule caps differ')
    with Path(select_pin['path']).open() as handle:
        roster = [parse(line) for line in handle]
    need(len(roster) == len({c['case_id'] for c in roster}) == 375, 'Select375 identity differs')
    same(dict(Counter(c['primary_domain'] for c in roster)), {d: 75 for d in DOMAINS}, 'Five-domain quota differs')
    e.verify([select_pin, schedule_pin, exposure_pin])
    return ready, roster, schedule, select_pin, schedule_pin, exposure_pin


def exposure_prefix(e, exposure_pin, node):
    macros = []
    with Path(exposure_pin['path']).open() as handle:
        for index in range(node['macro_updates']):
            rows = [parse(handle.readline()) for _ in range(16)]
            need(all(r['macro_index'] == index and r['exposure_index'] == 16 * index + j
                     for j, r in enumerate(rows)), 'Pinned exposure native order differs')
            macros.append(rows)
    need(sum(r['logical_pair_tokens'] for rows in macros for r in rows) == node['logical_pair_tokens'],
         'Pinned exposure token prefix differs')
    e.verify(exposure_pin)
    return macros


def committed_node_events(arm, node, exposures):
    """No stat pin on live ledger; stop immediately at the requested post commit."""
    folder = ROOTS[arm] / (arm + '-seed42')
    pre = post = started = completed = None
    starts = updates = tokens = 0
    with (folder / 'ledger.jsonl').open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n'):
                # readline reached EOF for this snapshot. Do not race a newly
                # appended suffix into the uncommitted record by reading again.
                try:
                    parse(line)
                except json.JSONDecodeError:
                    break  # Only an incomplete trailing JSON record may be ignored.
                raise RuntimeError('UNCOMMITTED_LEDGER_LINE_WITHOUT_NEWLINE: ' + arm)
            event = parse(line)  # Malformed newline-terminated records are never ignored.
            kind = event['event']
            need(kind != 'failed', 'Arm failure in node ledger; preserve evidence: ' + arm)
            if kind == 'macro_started':
                need(starts == updates and starts < len(exposures) and event['macro_index'] == starts,
                     'Macro repetition/gap or progress without node post checkpoint: ' + arm)
                same(event['exposures'], exposures[starts], 'Actual exposure differs from pinned seed42: ' + arm)
                starts += 1
            elif kind == 'macro_updated':
                need(starts == updates + 1, 'Macro update without sole exposure: ' + arm)
                tokens += sum(r['logical_pair_tokens'] for r in exposures[updates])
                updates += 1
                same(event['cursor'], {'macro_updates': updates, 'cases': 16 * updates,
                                      'logical_pair_tokens': tokens}, 'Actual training cursor differs: ' + arm)
            if kind == 'checkpoint_committed' and event['cursor']['macro_updates'] == node['macro_updates']:
                if started is None:
                    pre = event
                elif completed is not None:
                    post = event
                    break  # Never read later live progress or an unrelated partial tail.
            if kind in ('dev_started', 'dev_completed') and event['node']['cap'] == node['cap']:
                same(event['node'], node, 'Actual dev node differs')
                if kind == 'dev_started':
                    need(pre is not None and started is None and updates == node['macro_updates'], 'Dev start ordering differs')
                    started = event
                else:
                    need(started is not None and completed is None, 'Dev completion ordering differs')
                    completed = event
    need(post is not None, 'POST_DEV_CHECKPOINT_NOT_READY: ' + arm + ' cap=' + str(node['cap']))
    need(pre is not None and started is not None and completed is not None and starts == updates == node['macro_updates'],
         'Node event/exposure inventory differs')
    same(pre['cursor'], post['cursor'], 'Dev changed training cursor')
    same(post['cursor'], {k: node[k] for k in ('macro_updates', 'cases', 'logical_pair_tokens')}, 'Node checkpoint cursor differs')
    return pre, started, completed, post


def fresh_bridge(e, cfgs, sources):
    manifests = {}
    for root, cfg in cfgs.items():
        done = e.read(cfg['fresh_init']['complete']['path'])
        need(done['complete'] is True and done['kind'] == 'fresh-stock-init-v1' and done['seed'] == 42,
             'Fresh COMPLETE differs')
        committed_files(e, root / 'init-seed42', done)
        manifest_pin = e.add(root / 'init-seed42/FRESH_MANIFEST.json')
        need(done['manifest'] == manifest_pin and manifest_pin in done['files'], 'Fresh manifest unbound')
        manifest = e.read(manifest_pin['path'])
        same(manifest['trainer_code'], sources[root]['trainer_code'], 'Fresh own trainer bytes differ')
        same(manifest['stock_files'], cfg['stock_files'], 'Fresh STOCK identity differs')
        audit = manifest['audit']
        need(manifest['seed'] == 42 and manifest['model_forward_calls'] == 0 and
             manifest['no_project_adapter_head_optimizer_teacher_loaded'] is True and
             audit['saved_key_count'] == audit['trainable_tensors'] == 393 and audit['lora_pairs'] == 196 and
             audit['fresh_B_exactly_zero'] is True and audit['active_served_parameters'] == 616237056 and
             audit['trainable_parameters'] == 20187136, 'Fresh saved audit scope differs')
        tensors = audit['tensors']
        head = [v for k, v in tensors.items() if k.endswith('score.weight')]
        need(len(tensors) == 393 and all(v['dtype'] == 'torch.float32' for v in tensors.values()) and
             len(head) == 1 and head[0]['shape'] == [2, 1024], 'Fresh FP32/head manifest differs')
        need(not list((root / 'init-seed42').glob('BLOCKED*')), 'Fresh init BLOCKED')
        manifests[root] = manifest
    same(manifests[ORIGINAL]['construction'], manifests[FOLLOWUP]['construction'], 'Fresh construction differs')
    bridge = e.read(FOLLOWUP / 'FRESH_PRIMARY_VERIFICATION.json')
    need(bridge['complete'] is True and bridge['independently_generated_before_old_oracle_load'] is True and
         bridge['393_initial_tensors_bitwise_equal_old_v2'] is True and bridge['actual_saved_fp32_tensor_keys'] == 393 and
         bridge['lora_pairs'] == 196 and bridge['B_exactly_zero'] is True and bridge['head_shape'] == [2, 1024] and
         bridge['active_served_parameters'] == 616237056 and bridge['trainable_parameters'] == 20187136,
         'Actual independent equal-initial-values bridge missing')
    need(all(bridge[k] is False for k in ('old_project_trained_adapter_head_optimizer_teacher_loaded',
             'D_confirm_opened', 'public_run', 'training_launched', 'CUDA_initialized')) and
         bridge['model_forward_calls'] == bridge['optimizer_steps'] == 0, 'Fresh bridge scope differs')
    same(bridge['old_oracle_complete'], cfgs[ORIGINAL]['fresh_init']['complete'], 'Old oracle init differs')
    same(bridge['new_init_complete'], cfgs[FOLLOWUP]['fresh_init']['complete'], 'New own init differs')
    need(bridge['old_oracle_complete']['path'] != bridge['new_init_complete']['path'], 'Distinct fresh artifacts required')
    for key, path in (('old_oracle_manifest', ORIGINAL / 'init-seed42/FRESH_MANIFEST.json'),
                      ('new_manifest', FOLLOWUP / 'init-seed42/FRESH_MANIFEST.json'),
                      ('analysis_source', FOLLOWUP / 'fresh-compare-analysis.py'),
                      ('bootstrap_source', FOLLOWUP / 'ops/freeze_fresh.py'),
                      ('deployment', FOLLOWUP / 'DEPLOYMENT.json'), ('final_config', FOLLOWUP / 'seed42.json')):
        same(bridge[key], e.add(path), 'Fresh bridge input binding differs: ' + key)
    same(bridge['trainer_files'], cfgs[FOLLOWUP]['trainer_files'], 'Fresh final trainer pins differ')
    same(bridge['trainer_code'], sources[FOLLOWUP]['trainer_code'], 'Fresh bridge trainer bytes differ')
    same(bridge['construction'], manifests[FOLLOWUP]['construction'], 'Fresh bridge construction differs')
    e.verify(bridge)
    e.source_map([bridge['analysis_source'], bridge['bootstrap_source']])
    deployment = e.read(bridge['deployment']['path'])
    same(deployment['original_config'], e.add(ORIGINAL / 'seed42.json'), 'Fresh old config binding differs')
    same(deployment['source_files'], cfgs[FOLLOWUP]['trainer_files'], 'Fresh deployment source pins differ')
    same(deployment['source_bytes'], {Path(k).name: v for k, v in sources[FOLLOWUP]['trainer_code'].items()},
         'Fresh deployment source bytes differ')
    same(deployment['bootstrap_source'], bridge['bootstrap_source'], 'Fresh bootstrap binding differs')
    need(deployment['complete'] is True and deployment['training_launched'] is False and
         deployment['baseline_source_bytes_verified'] is True, 'Fresh deployment incomplete')
    return {'report': e.add(FOLLOWUP / 'FRESH_PRIMARY_VERIFICATION.json'),
            'old_init_complete': bridge['old_oracle_complete'], 'new_init_complete': bridge['new_init_complete'],
            'equal_initial_values_via_bound_actual_bridge': True, 'same_artifact_path_claimed': False,
            'actual_saved_fp32_tensor_keys': 393, 'lora_pairs': 196, 'head_shape': [2, 1024],
            'active_served_parameters': 616237056, 'tensor_audit_rerun': False}


def control_bridge(e, cfgs):
    manifest = e.read(CONTROL / 'MANIFEST.json')
    need(manifest['complete'] is True and manifest['version'] == 'staged-external-manifest-v1' and
         manifest['control_root'] == str(CONTROL), 'Wrong CONTROL manifest')
    need(manifest['config']['stat']['path'] == str(CONTROL / 'analysis-config.json'), 'CONTROL config redirected')
    e.exact(manifest['config'])
    cfg = e.read(CONTROL / 'analysis-config.json')
    need(cfg['version'] == 'staged-external-config-v1' and cfg['control_root'] == str(CONTROL) and
         cfg['data_root'] == str(FOLLOWUP) and cfg['original_root'] == str(ORIGINAL), 'CONTROL roots differ')
    need(set(cfg['arms']) == set(ARMS[3:]) and set(cfg['historical_ledger_pins']) == set(ARMS[3:]), 'CONTROL arm bindings differ')
    for arm in ARMS[3:]:
        item = cfg['historical_ledger_pins'][arm]
        need(set(item) == PIN_KEYS and item['path'] == str(FOLLOWUP / (arm + '-seed42') / 'ledger.jsonl'),
             'Historical ledger redirected')
        same(cfg['arms'][arm]['ledger'], item, 'CONTROL paused ledger binding differs')
        e.historical_ledgers[item['path']] = item  # Do NOT stat/reopen the historical paused pin.
    same(manifest['input_pins'], cfg['static_pins'], 'CONTROL static inventory differs')
    e.verify(cfg['static_pins'])
    for item in cfg['static_exact']:
        e.exact(item)
    required_sources = {'source/main_readback.py', 'source/staged_queue.py', 'deploy_staged.py'}
    need(set(manifest['sources']) == required_sources, 'CONTROL source inventory differs')
    for name, item in manifest['sources'].items():
        need(item['stat']['path'] == str(CONTROL / name), 'CONTROL source redirected')
        e.exact(item)
    installed = e.read(CONTROL / 'INSTALL_COMPLETE.json')
    need(installed['complete'] is True and installed['exact_byte_readback'] is True and
         all(installed[k] is False for k in ('readback_launched', 'queue_launched', 'authorization_created', 'old_data_written')),
         'CONTROL install receipt differs')
    same(installed['manifest'], e.add(CONTROL / 'MANIFEST.json'), 'Installed manifest differs')
    same(installed['config'], manifest['config']['stat'], 'Installed config differs')
    for key, name in (('old_failure', 'QUEUE_BLOCKED.json'), ('old_ready', 'PRIMARY_READY.json'),
                      ('old_launch', 'QUEUE_LAUNCH.json')):
        need(cfg[key]['stat']['path'] == str(FOLLOWUP / name), 'CONTROL old input redirected')
        e.exact(cfg[key])  # Exact files ONLY: do not recurse their old paused pins.
    failure = e.read(FOLLOWUP / 'QUEUE_BLOCKED.json')
    prefix = 'Known old BI/PB + BOTH new arms + evidence/shared floor cannot fit: '
    need(failure['complete'] is False and failure['error_type'] == 'RuntimeError' and
         failure['partial_preserved'] is True and failure['automatic_resume'] is False and
         failure['public_run'] is False and failure['error'].startswith(prefix), 'Not the accepted old resource failure')
    same(failure['owner'], e.read(FOLLOWUP / 'QUEUE_LAUNCH.json')['owner'], 'Old resource failure owner differs')
    diagnostics = parse(failure['error'][len(prefix):])
    need(diagnostics['free_bytes'] == 115269525504 and diagnostics['required_bytes'] == 116379385959,
         'Not the recorded resource-controller failure')
    ready = e.read(FOLLOWUP / 'PRIMARY_READY.json')
    need(ready['complete'] is True and ready['root'] == str(FOLLOWUP) and
         ready['main_direct_actual_review_required'] is True and ready['automatic_continuation'] is False and
         ready['D_confirm_opened'] is False and ready['public_run'] is False, 'PRIMARY_READY scope differs')
    same(ready['arms'], cfg['arms'], 'PRIMARY_READY arm bindings differ')
    review = e.read(CONTROL / 'MAIN_DIRECT_ACTUAL_REVIEW.json')
    need(review['complete'] is True and review['version'] == 'staged-main-direct-formal-macro-readback-v1' and
         review['control_root'] == str(CONTROL) and review['data_root'] == str(FOLLOWUP) and
         review['next_macro_index'] == 1 and review['repeat_macro0_allowed'] is False and
         review['authorization_created'] is False and review['model_forward_calls'] == review['optimizer_steps'] == 0 and
         review['D_confirm_opened'] is False and review['public_run'] is False, 'Actual MAIN bridge missing')
    for key, expected in (('analysis_config', manifest['config']['stat']), ('manifest', e.add(CONTROL / 'MANIFEST.json')),
                          ('accepted_resource_blocker', cfg['old_failure']['stat']), ('primary_ready', cfg['old_ready']['stat']),
                          ('analysis_source', manifest['sources']['source/main_readback.py']['stat'])):
        same(review[key], expected, 'MAIN immutable binding differs: ' + key)
    same(review['arm_bindings'], cfg['arms'], 'MAIN full arm bindings differ')
    need(set(review['arms']) == set(ARMS[3:]), 'Both actual MAIN arms required')
    e.historical_evidence(review['evidence_pins'])
    scopes = {}
    for arm in ARMS[3:]:
        binding, checks = cfg['arms'][arm], review['arms'][arm]
        scope = e.read(FOLLOWUP / (arm + '-seed42') / 'OWN_SCOPE.json')
        same(binding['scope'], scope, 'MAIN arm own scope differs')
        same(scope['config'], cfgs[FOLLOWUP], 'MAIN arm current followup cfg differs')
        same(scope['config_pin'], e.add(FOLLOWUP / 'seed42.json'), 'MAIN current config pin differs')
        need(binding['next_macro_index'] == 1 and binding['repeat_macro0_allowed'] is False, 'MAIN macro0 repetition admitted')
        for key in ('actual_replay_bytes_equal', 'MAIN_independent_actual_coef_bytes_equal',
                    '393_raw_clipped_before_after_delta_and_Adam_read_directly',
                    'actual_fresh_and_latest_checkpoint_bytes_equal', 'macro0_consumed_once'):
            need(checks[key] is True, 'Actual MAIN tensor check missing: ' + key)
        same(checks['cursor'], ONE, 'MAIN actual cursor differs')
        e.historical_evidence(binding)
        need(binding['report']['path'] == str(FOLLOWUP / 'primary' / arm / 'FIRST_PRIMARY_VERIFICATION.json'), 'Primary report redirected')
        report = e.read(binding['report']['path'])
        need(report['complete'] is True and report['version'] == 'followup-first-primary-saved-only-v1' and
             report['arm_name'] == report['arm'] == arm and report['allow_limited_scientific_resume'] is True and
             report['automatic_resume'] is False and report['next_macro_index'] == 1 and report['repeat_macro0_allowed'] is False and
             report['macro0_consumed_exactly_once'] is True and report['model_forward_calls'] == 0 and
             report['D_confirm_opened'] is False and report['public_run'] is False, 'Primary saved-only scope differs')
        for key in ('saved_cache_replay_shape_dtype_bytes_equal', 'independent_margin_coefficient_bytes_equal',
                    'all393_new_fresh_equal_old_oracle_and_actual_before',
                    'latest_cursor1_checkpoint_weights_optimizer_scheduler_match_saved_actual',
                    'independent_original_case_exposure_slots_fields_micros_verified'):
            need(report[key] is True, 'Primary actual saved-artifact proof missing: ' + key)
        same(report['scope'], scope, 'Primary own scope differs')
        same(report['cursor'], ONE, 'Primary cursor1 differs')
        same(report['durable_ledger'], binding['ledger'], 'Primary historical paused pin differs')
        same(report['fresh_primary'], e.add(FOLLOWUP / 'FRESH_PRIMARY_VERIFICATION.json'), 'Primary fresh bridge differs')
        same(report['resume_pin_file'], binding['pin_file'], 'Primary resume pin differs')
        same(e.read(binding['pin_file']['path']), binding['complete'], 'Immutable cursor1 pin file differs')
        same(report['resume'], {'checkpoint': binding['checkpoint'], 'complete': binding['complete'],
                                'pin_file': binding['pin_file']['path']}, 'Primary immutable resume binding differs')
        e.historical_evidence(report['evidence_pins'])
        for path, encoded in report['exact_source_and_config_bytes'].items():
            same(e.source_map([e.add(path)])[path], encoded, 'Primary exact source/config bytes differ')
        checkpoint_path = Path(binding['checkpoint'])
        need(checkpoint_path.parent == FOLLOWUP / (arm + '-seed42') and
             binding['complete']['path'] == str(checkpoint_path / 'COMPLETE.json'), 'Cursor1 checkpoint redirected')
        checkpoint = e.read(binding['complete']['path'])
        need(checkpoint['complete'] is True and checkpoint['macro_updates'] == 1 and
             checkpoint['resumable_full_macro_boundary'] is True, 'Cursor1 checkpoint incomplete')
        same(checkpoint['scope'], scope, 'Cursor1 checkpoint scope differs')
        committed_files(e, checkpoint_path, checkpoint)
        checkpoint_meta = e.read(checkpoint_path / 'CHECKPOINT.json')
        same(checkpoint_meta['cursor'], ONE, 'Cursor1 metadata differs')
        same(checkpoint_meta['scope'], scope, 'Cursor1 metadata scope differs')
        first_path = FOLLOWUP / (arm + '-seed42') / 'first-actual-macro/COMPLETE.json'
        same(report['formal_first_complete'], e.add(first_path), 'Immutable actual macro0 binding differs')
        first = e.read(first_path)
        need(first['complete'] is True and first['is_formal_exposure_macro0'] is True, 'Actual macro0 missing')
        same(first['scope'], scope, 'Actual macro0 scope differs')
        committed_files(e, first_path.parent, first)  # No tensor loading or repeated loss/gradient audit.
        scopes[arm] = scope
    return scopes, {'report': e.add(CONTROL / 'MAIN_DIRECT_ACTUAL_REVIEW.json'),
                    'control_root': str(CONTROL), 'data_root': str(FOLLOWUP),
                    'actual_two_arm_replay_coef_393_update_cursor1_bridge_verified': True,
                    'accepted_old_resource_controller_failure': cfg['old_failure']['stat'],
                    'historical_paused_ledger_pins_not_revalidated': list(e.historical_ledgers.values()),
                    'historical_ledger_policy': 'report bytes/stat and immutable artifacts verified; paused ledger pins are phase-limited history, not current static identity',
                    'tensor_audit_rerun': False}


def verify_scope(e, scope, root, arm, cfg, sources, ready):
    need(scope['version'] == cfg['version'] and scope['arm_name'] == arm, 'Wrong node own scope')
    same(scope['config'], cfg, 'Node actual own configuration differs')
    same(scope['config_pin'], e.add(root / 'seed42.json'), 'Node config pin differs')
    same(scope['arm'], ARM_CONFIG[arm], 'Node arm differs')
    for key in ('trainer_code', 'core_code'):
        same(scope[key], sources[root][key], 'Node own source bytes differ: ' + key)
    same(scope['data']['complete'], cfg['data_complete'], 'Node data COMPLETE differs')
    same(scope['data']['files'], ready['files'], 'Node data artifact pins differ')
    e.verify(scope['data'])
    same(scope['primary']['pin'], cfg['primary_admission'], 'Node primary admission pin differs')
    same(scope['primary']['admission'], e.read(cfg['primary_admission']['path']), 'Node actual admission differs')
    admission = scope['primary']['admission']
    need(admission['complete'] is True and admission['allow_training'] is True and
         admission['allow_prefix_cache'] is False, 'Node full-row admission differs')
    same(admission['core_code'], sources[root]['core_code'], 'Admission exact core bytes differ')
    e.verify(scope['primary'])
    if root == FOLLOWUP:
        boundary = scope['state_boundary']
        same(boundary['complete'], cfg['state_boundary']['complete'], 'Node state product differs')
        same(boundary['primary_pin'], cfg['state_boundary']['primary'], 'Node state primary differs')
        same(boundary['prepared'], e.read(boundary['complete']['path']), 'State prepared COMPLETE differs')
        same(boundary['primary'], e.read(boundary['primary_pin']['path']), 'State primary report differs')
        need(boundary['boundary_policy'] == 'state-plus-one-existing-newline-v2' and
             boundary['primary']['allow_new_state_boundary_training'] is True, 'State boundary not admitted')
        e.verify(boundary['guarded_pins'])
        same(e.source_map(boundary['exact_byte_pins']), boundary['source_bytes'], 'State product exact bytes differ')


def analyze_arm(e, arm, node, roster, cfg, sources, ready, events, control_scopes):
    root = ROOTS[arm]
    arm_root = root / (arm + '-seed42')
    folder = arm_root / f"D_select-cap{node['cap']}-m{node['macro_updates']:07d}"
    need(not list(folder.glob('BLOCKED*')), 'Node evaluation BLOCKED: ' + arm)
    done, s = e.read(folder / 'COMPLETE.json'), e.read(folder / 'SUMMARY.json')
    need(done['complete'] is True, 'Node COMPLETE missing')
    same(done['node'], node, 'Node COMPLETE cursor differs')
    same(s['node'], node, 'Node SUMMARY cursor differs')
    committed_files(e, folder, done)
    need(len(done['files']) == 2 and {p['path'] for p in done['files']} ==
         {str(folder / 'SUMMARY.json'), str(folder / 'case-metrics.jsonl')}, 'Node file inventory differs')
    scope = e.read(arm_root / 'OWN_SCOPE.json')
    same(done['scope'], scope, 'Node COMPLETE scope differs')
    same(s['scope'], scope, 'Node SUMMARY scope differs')
    verify_scope(e, scope, root, arm, cfg, sources, ready)
    if arm in control_scopes:
        same(scope, control_scopes[arm], 'Node scope is not MAIN actual tensor bridge scope')
    same(s['checkpoint_identity'], done['checkpoint_identity'], 'Node checkpoint identity differs')
    need(s['checkpoint_identity']['macro_updates'] == node['macro_updates'], 'Node checkpoint macro differs')
    initialization = s['checkpoint_identity']['initialization']
    same(initialization['fresh_init']['complete'], cfg['fresh_init']['complete'],
         'Node initialization not own actual fresh COMPLETE')
    same(initialization['fresh_init']['commit'], e.read(cfg['fresh_init']['complete']['path']),
         'Node saved fresh COMPLETE content differs')
    same(initialization['fresh_init']['manifest'], e.read(root / 'init-seed42/FRESH_MANIFEST.json'),
         'Node saved own fresh manifest differs')
    need(s['roster'] == 'D_select_only' and s['primary_metric'] == METRICS['mass'], 'Not fixed D_select metric')
    need(all(s[k] is False for k in ('D_confirm_opened', 'public_run', 'winner_selected', 'candidate_pruning',
                                   'truncation', 'short_case_replacement', 'prefix_cache')), 'Forbidden evaluation action')
    with (folder / 'case-metrics.jsonl').open() as handle:
        raw = [parse(line) for line in handle]
    need(len(raw) == 375 and [x['case_id'] for x in raw] == [c['case_id'] for c in roster], 'Native case order/identity differs')
    coverage = Counter()
    calculated, signatures, exact = [], [], []
    maximum_metric_error = 0.0
    for original, saved in zip(roster, raw):
        need(saved['component'] == original['component'] and saved['primary_domain'] == original['primary_domain'] and
             len(saved['fields']) == len(original['fields']), 'Case field/component/domain identity differs')
        field_values, chosen, supported = [], [], []
        known = sum(f['known'] for f in original['fields'])
        need(saved['known_fields'] == known and saved['unknown_fields'] == len(original['fields']) - known and known > 0,
             'Case known/unknown coverage differs')
        for i, (field, value) in enumerate(zip(original['fields'], saved['fields'])):
            need(value['field_index'] == i and value['coordinate'] == field['coordinate'] and
                 value['domain'] == field['domain'] and value['known'] == field['known'], 'Native field order/identity differs')
            if not field['known']:
                need(value['status'] == 'masked_unknown_no_forward' and value['K'] is None and
                     not any(k in value for k in ('margins', 'logits', 'native_target_not_renormalized')),
                     'Unknown field forwarded or reinterpreted')
                coverage['unknown_fields_retained_no_forward'] += 1
                chosen.append(None)
                continue
            coverage['known_fields'] += 1
            need(field['static_unsupported_reason'] is None and value['status'] == 'supported_complete_field',
                 'Expected full known coverage; no unsupported fallback')
            q, margins, logits = field['target'], value['margins'], value['logits']
            need(value['native_target_not_renormalized'] == q and
                 value['K'] == len(q) == len(field['options']) == len(margins) == len(logits) and len(q) > 0,
                 'Complete native candidates/unaltered q differ')
            need(all(math.isfinite(v) for v in margins) and
                 all(len(v) == 2 and all(math.isfinite(x) for x in v) for v in logits), 'Nonfinite saved margins/logits')
            # Reproduce actual FP32 subtraction of saved logits, NOT raw FP64 subtraction.
            need(all(v == struct.unpack('<f', struct.pack('<f', pair[1] - pair[0]))[0]
                     for v, pair in zip(margins, logits)), 'Saved FP32 logit margin differs')
            best = max(range(len(margins)), key=margins.__getitem__)
            log_p, p = distribution(margins)
            nll = -math.fsum(a * b for a, b in zip(q, log_p))
            brier = math.fsum((a - b) ** 2 for a, b in zip(p, q))
            result = {'mass': q[best], 'NLL': nll,
                      'normalized_NLL': nll / math.log(len(q)) if len(q) > 1 else 0.0, 'Brier': brier}
            need(value['argmax_index'] == best, 'Native tie-breaking/argmax differs')
            for key, target in (('mass', 'argmax_target_mass'), ('NLL', 'NLL'),
                                ('normalized_NLL', 'normalized_NLL'), ('Brier', 'Brier_sum')):
                close(result[key], value[target])
                maximum_metric_error = max(maximum_metric_error, abs(result[key] - value[target]))
            field_values.append(result)
            chosen.append(best)
            supported.append(i)
            coverage['supported_known_fields'] += 1
            coverage['forwarded_complete_candidates'] += len(q)
        need(len(supported) == known == saved['supported_known_fields'], 'Case known field coverage incomplete')
        values = {key: mean([f[key] for f in field_values]) for key in ('mass', 'NLL', 'normalized_NLL', 'Brier')}
        values.update(coverage=len(supported) / known,
                      coverage_adjusted_mass=math.fsum(f['mass'] for f in field_values) / known)
        for key, target in METRICS.items():
            close(values[key], saved[target])
        exact_supported = original['complete_case_exact_static_supported']
        need(saved['complete_onehot_case_exact_supported'] == exact_supported, 'Case exact support differs')
        if exact_supported:
            result = all(v['mass'] == 1.0 for v in field_values)
            need(saved['complete_onehot_case_exact'] == result, 'Case exact result differs')
            exact.append(result)
            coverage['complete_onehot_case_exact_supported'] += 1
        else:
            need(saved['complete_onehot_case_exact'] is None, 'Unsupported case exact must be null')
        calculated.append({'domain': original['primary_domain'], **values})
        signatures.append(chosen)
        coverage['cases'] += 1
        coverage['cases_with_all_known_fields_supported'] += len(supported) == known
        coverage['cases_with_any_supported_known_field'] += bool(supported)
    same(dict(coverage), s['coverage'], 'Actual coverage report differs')
    need(coverage['known_fields'] == coverage['supported_known_fields'] == 615 and
         coverage['forwarded_complete_candidates'] == 5190 and coverage['unknown_fields_retained_no_forward'] == 90,
         'Expected375/615/5190/90 coverage differs')
    per_domain = {}
    for domain in DOMAINS:
        entries = [v for v in calculated if v['domain'] == domain]
        need(len(entries) == s['per_domain'][domain]['cases'] == 75, 'Domain case denominator differs')
        per_domain[domain] = {key: mean([v[key] for v in entries]) for key in METRICS}
        for key, target in METRICS.items():
            close(per_domain[domain][key], s['per_domain'][domain][target])
            need(s['per_domain'][domain][target + '_case_denominator'] == 75, 'Domain metric denominator differs')
    five = {key: mean([per_domain[d][key] for d in DOMAINS]) for key in METRICS}
    for key, target in METRICS.items():
        close(five[key], s['five_domain_case_means'][target])
    need(len(exact) == s['complete_onehot_case_exact_denominator'] == 259, 'Exact259 denominator differs')
    close(mean(exact), s['complete_onehot_case_exact'])
    before, after = s['runtime_before']['forward'], s['runtime_after']['forward']
    train_tokens = node['logical_pair_tokens']
    need(before['train_first.completed_real_tokens'] == before['train_replay.completed_real_tokens'] ==
         before['train_replay.backward_real_tokens'] == train_tokens, 'Training token exposure differs')
    need(all(after[k] == v for k, v in before.items() if k.startswith(('train_first.', 'train_replay.'))),
         'Evaluation changed historical training counters')
    for phase in ('train_first', 'train_replay'):
        need(before[phase + '.attempted_waves'] == before[phase + '.completed_waves'], 'Incomplete train waves')
        need(before[phase + '.completed_native_FA3_calls'] ==
             28 * (2 if ARM_CONFIG[arm]['mode'] == 'prefix_block' else 1) * before[phase + '.completed_waves'],
             'Actual train native calls differ')
        need(before[phase + '.completed_tokenwise_rows_per_operator'] == train_tokens, 'Packed train not realT only')
    dev_tokens = after['dev_select.completed_real_tokens'] - before.get('dev_select.completed_real_tokens', 0)
    need(dev_tokens == sum(c['logical_supervised_pair_tokens'] for c in roster) == 1554484, 'D_select token delta differs')
    dev_rows = after['dev_select.completed_tokenwise_rows_per_operator'] - before.get('dev_select.completed_tokenwise_rows_per_operator', 0)
    dev_waves = after['dev_select.completed_waves'] - before.get('dev_select.completed_waves', 0)
    need(dev_rows == dev_tokens, 'Packed dev not realT only')
    pre, started, completed, post = events
    same(completed['complete'], e.add(folder / 'COMPLETE.json'), 'Ledger dev COMPLETE pin differs')
    checkpoint_metadata = []
    for item in (pre, post):
        path = Path(item['complete']['path'])
        need(path.name == 'COMPLETE.json' and path.parent.parent == arm_root, 'Node checkpoint redirected')
        e.verify(item['complete'])
        checkpoint = e.read(path)
        need(checkpoint['complete'] is True and checkpoint['macro_updates'] == node['macro_updates'] and
             checkpoint['resumable_full_macro_boundary'] is True, 'Node checkpoint incomplete')
        same(checkpoint['scope'], scope, 'Node checkpoint own scope differs')
        committed_files(e, path.parent, checkpoint)
        metadata = e.read(path.parent / 'CHECKPOINT.json')
        same(metadata['scope'], scope, 'Node checkpoint metadata scope differs')
        same(metadata['cursor'], item['cursor'], 'Node checkpoint actual cursor differs')
        same(metadata['initialization'], s['checkpoint_identity']['initialization'], 'Node initialization metadata differs')
        same(metadata['runtime'], s['runtime_before'] if item is pre else s['runtime_after'], 'Node checkpoint runtime differs')
        checkpoint_metadata.append(metadata)
    need(node['cap'] not in checkpoint_metadata[0]['dev_done'] and node['cap'] in checkpoint_metadata[1]['dev_done'],
         'Pre/post dev completion cursor differs')
    need(pre['complete']['path'] != post['complete']['path'] and post['sequence'] > pre['sequence'], 'Distinct pre/post commits required')
    return calculated, signatures, {
        'root': str(root), 'configuration': e.add(root / 'seed42.json'), 'arm_configuration': ARM_CONFIG[arm],
        'initial_fresh_complete': cfg['fresh_init']['complete'], 'node': node,
        'case_balanced_five_domain_means': five, 'per_domain': per_domain, 'coverage': dict(coverage),
        'complete_onehot_case_exact': mean(exact), 'complete_onehot_case_exact_denominator': len(exact),
        'complete_onehot_case_exact_count': sum(exact), 'maximum_independent_field_metric_absolute_error': maximum_metric_error,
        'actual_training_real_forward_tokens': 2 * train_tokens, 'actual_training_backward_real_tokens': train_tokens,
        'actual_training_tokenwise_rows_per_operator_two_passes': 2 * train_tokens,
        'D_select_real_tokens': dev_tokens, 'D_select_tokenwise_rows_per_operator': dev_rows, 'D_select_waves': dev_waves,
        'D_select_wall_seconds_in_training_ledger': completed['elapsed_this_invocation'] - started['elapsed_this_invocation'],
        'pre_dev_checkpoint': pre['complete'], 'post_dev_checkpoint': post['complete'],
        'actual_exposure_order_matches_original_pinned_seed42': True,
        'loss_layout_mode_seed_and_own_fresh_scope_verified': True,
        'complete_candidate_order_and_unknown_skip_verified': True,
        'independent_raw_logits_margin_field_case_domain_metrics_verified': True}


def paired_contrasts(records, signatures):
    paired, samples, case_mass_deltas = {}, {name: [] for name in CONTRASTS}, {}
    by_domain = {d: [i for i, row in enumerate(records[ARMS[0]]) if row['domain'] == d] for d in DOMAINS}
    for name, weights in CONTRASTS.items():
        deltas = [{k: math.fsum(weight * records[arm][i][k] for arm, weight in weights.items())
                   for k in METRICS} for i in range(375)]
        means = {k: mean([row[k] for row in deltas]) for k in METRICS}
        domain_means = {d: {k: mean([deltas[i][k] for i in indices]) for k in METRICS}
                        for d, indices in by_domain.items()}
        values = [row['mass'] for row in deltas]
        case_mass_deltas[name] = {d: [values[i] for i in indices] for d, indices in by_domain.items()}
        paired[name] = {'weights': weights, 'paired_cases': 375, 'metric_deltas': means, 'per_domain': domain_means,
            'mass_case_positive_negative_equal': {'positive': sum(v > 1e-12 for v in values),
                'negative': sum(v < -1e-12 for v in values), 'equal_within_1e-12': sum(abs(v) <= 1e-12 for v in values)},
            'mass_stratified_case_standard_error': math.sqrt(math.fsum(
                statistics.variance([values[i] for i in indices]) / len(indices) for indices in by_domain.values())) / 5}
        a, b = weights.keys()
        paired[name]['field_argmax_disagreements'] = sum(x != y for ra, rb in zip(signatures[a], signatures[b])
                                                        for x, y in zip(ra, rb) if x is not None)
        paired[name]['cases_with_field_argmax_disagreement'] = sum(any(x != y for x, y in zip(ra, rb))
                                                                 for ra, rb in zip(signatures[a], signatures[b]))
    rng = random.Random(20261112)
    for _ in range(10000):
        draw = {d: [rng.randrange(75) for _ in range(75)] for d in DOMAINS}
        for name in CONTRASTS:
            samples[name].append(math.fsum(case_mass_deltas[name][d][i] for d in DOMAINS for i in draw[d]) / 375)
    for name in CONTRASTS:
        values = sorted(samples[name])
        paired[name]['mass_paired_case_stratified_bootstrap_95pct_percentile_interval'] = [values[249], values[9749]]
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cap', required=True, type=int, choices=tuple(EXPECTED_NODES))
    args = parser.parse_args()
    out = ANALYSIS / f'FOLLOWUP_SELECT_PRIMARY-cap{args.cap}.json'
    need(not out.exists() and not out.is_symlink(), 'Previous primary evidence preserved; do not silently replace')
    need(ANALYSIS.is_dir() and ANALYSIS.resolve() == ANALYSIS and not ANALYSIS.is_symlink(),
         'MAIN must preprovision the exclusive analysis parent')
    os.umask(0o077)
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    e = Evidence()
    analysis_source = e.add(Path(__file__).absolute())
    e.source_map([analysis_source])
    unblocked()
    cfgs, sources = configurations(e)
    ready, roster, schedule, select_pin, schedule_pin, exposure_pin = prepared_data(e, cfgs[ORIGINAL])
    node = next(n for n in schedule['observations'] if n['cap'] == args.cap)
    exposures = exposure_prefix(e, exposure_pin, node)
    # All five post-dev ledger commits are required before any success output is opened.
    events = {arm: committed_node_events(arm, node, exposures) for arm in ARMS}
    fresh = fresh_bridge(e, cfgs, sources)
    control_scopes, main_bridge = control_bridge(e, cfgs)
    reports, records, signatures = {}, {}, {}
    for arm in ARMS:
        records[arm], signatures[arm], reports[arm] = analyze_arm(
            e, arm, node, roster, cfgs[ROOTS[arm]], sources, ready, events[arm], control_scopes)
    paired = paired_contrasts(records, signatures)
    unblocked(node)
    e.check()
    result = {'complete': True, 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'version': 'followup-matched-select-analysis-v1', 'frozen_node': node, 'arm_order': list(ARMS),
        'arms': reports, 'paired_case_contrasts': paired, 'common_fresh_bridge': fresh, 'actual_MAIN_bridge': main_bridge,
        'bootstrap': {'replicates': 10000, 'seed': 20261112,
            'unit': 'whole case, paired across all five arms, stratified by five primary domains',
            'interval_scope': 'descriptive single-seed scheduled D_select uncertainty only; not multiplicity-adjusted and not training-seed uncertainty'},
        'source_receipt_files': [e.files[p] for p in sorted(e.files)], 'original_select_roster': select_pin,
        'original_seed42_schedule': schedule_pin, 'original_seed42_exposure': exposure_pin,
        'analysis_source': analysis_source, 'model_forward_calls': 0, 'optimizer_steps': 0, 'evaluator_started': False,
        'D_confirm_opened': False, 'public_run': False, 'winner_selected': False,
        'training_source_or_config_modified': False, 'per_case_arrays_persisted': False,
        'contrast_scope': {'STATE': 'state+existing-newline P boundary versus candidate P; intact original rows',
                           'JPAIR': 'matched causal packed BCE+hard-rank objective/scales only; not full Jina'},
        'limitations': ['Post-10M/30M exploratory methods under existing Task4; no new selection protocol or training',
            '10M/30M intermediate and100M final; no public checkpoint selection, intermediate LR or exposure adaptation',
            'D_select target mass is not Public Index and cannot be compared numerically to51.99',
            'one training seed; paired-case intervals do not cover training-seed variability or prove numerical equivalence',
            'four descriptive contrasts, not multiplicity-adjusted; no component training loss is a quality metric',
            'tools dev mainly procedural, not BFCL/real-tool generalization',
            'fullrow scientific training/dev only; no cached-serving or official latency admission',
            'actual tensor equality/update audits are bound historical evidence, not rerun by this stdlib analyzer']}
    with out.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({'complete': True, 'frozen_node': node,
        'metrics': {a: v['case_balanced_five_domain_means'] for a, v in reports.items()},
        'paired_mass': {n: {'delta': v['metric_deltas']['mass'],
            '95pct_case_interval': v['mass_paired_case_stratified_bootstrap_95pct_percentile_interval']} for n, v in paired.items()},
        'saved_primary_verification': str(out), 'model_or_evaluator_started': False, 'D_confirm_opened': False},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
