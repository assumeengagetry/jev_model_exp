"""Fail-closed reviewed inputs. Stat + exact small source bytes; never hashes."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
STOCK = '/home/dev/assumeengage/models/Qwen3-0.6B'
DOMAINS = ('knowledge', 'language', 'retrieval', 'tools', 'arts')
CODE = ('bindings.py', 'fresh_init.py', 'case_runtime.py', 'objectives.py',
        'model_runtime.py', 'checkpoints.py', 'train_case.py', 'evaluate_native_dev.py',
        'state_boundaries.py')
PRECISION_KEYS = {'matmul_allow_tf32', 'cudnn_allow_tf32',
                  'allow_bf16_reduced_precision_reduction', 'float32_matmul_precision',
                  'cublas_workspace_config', 'autocast_cache_enabled'}
FIRST_ARMS = {
    'C_D_BCE': {'mode': 'causal', 'layout': 'dense', 'loss': 'BCE'},
    'C_P_BCE': {'mode': 'causal', 'layout': 'packed', 'loss': 'BCE'},
    'C_D_CE': {'mode': 'causal', 'layout': 'dense', 'loss': 'CE'},
    'C_P_CE': {'mode': 'causal', 'layout': 'packed', 'loss': 'CE'},
}
FOLLOWUP_ARMS = {
    'STATE_NL_PB_P_CE': {'mode': 'prefix_block', 'layout': 'packed', 'loss': 'CE'},
    'JPAIR_P_BCE_RANK': {'mode': 'causal', 'layout': 'packed', 'loss': 'BCE_RANK_HARD'},
}
ALL_ARMS = {
    **FIRST_ARMS,
    'BI_P_CE': {'mode': 'bidirectional', 'layout': 'packed', 'loss': 'CE'},
    'PB_P_CE': {'mode': 'prefix_block', 'layout': 'packed', 'loss': 'CE'},
    **FOLLOWUP_ARMS,
}
FOLLOWUP_OBJECTIVE = {
    'version': 'hard-label-bce-rank-v1', 'lambda': 1.0, 'tau': 1.0,
    'hard_label_rule': 'original-host-q-exact-zero-or-one',
    'normalization': 'known-field-mean-then-case-mean',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def stat(path):
    path = Path(path)
    s = path.stat()
    require(path.is_file(), 'Not a regular file: ' + str(path))
    return {'path': str(path), 'resolved': str(path.resolve()),
            'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def verify(records):
    require(isinstance(records, list) and records, 'Nonempty stat pins required')
    for item in records:
        require(set(item) == {'path', 'resolved', 'bytes', 'mtime_ns'}, 'Bad stat schema')
        require(stat(item['path']) == item, 'File changed: ' + item['path'])


def write_json(path, value):
    """Exclusive write, durable before a later COMPLETE marker is published."""
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(dump(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def fsync_tree(root):
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(folder) / name
            require(not path.is_symlink(), 'Symlink output rejected')
            with path.open('rb') as handle:
                os.fsync(handle.fileno())
        require(not any((Path(folder) / d).is_symlink() for d in dirs), 'Symlink directory')
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def commit_json(path, value):
    """Publish a durable JSON marker atomically, without replacing any old marker."""
    path = Path(path)
    pending = path.with_name(path.name + '.writing')
    write_json(pending, value)
    os.link(pending, path)
    pending.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def new_directory(path, parent):
    path, parent = Path(path), Path(parent)
    require(parent.is_absolute() and parent.parent == Path('/tmp/opencode') and
            parent.name.startswith('jev-clean-trainer-20261112-'), 'Bad private output namespace')
    require(parent.exists() and not parent.is_symlink(), 'Provision private output parent first')
    require(path.is_absolute() and path.resolve().is_relative_to(parent.resolve()) and
            path != parent and not path.exists() and not path.is_symlink(), 'New exclusive output required')
    require(path.parent.exists() and path.parent.resolve().is_relative_to(parent.resolve()),
            'Output parent must already exist in own namespace')
    path.mkdir(mode=0o700)
    return path


def source_bytes(records):
    verify(records)
    return {r['path']: base64.b64encode(Path(r['path']).read_bytes()).decode('ascii')
            for r in records}


def bind_data(cfg):
    root = Path(cfg['data_root'])
    require(root.is_absolute() and not (root / 'BLOCKED.json').exists() and
            not list(root.glob('BLOCKED*')), 'Prepared data blocked or root invalid')
    verify([cfg['data_complete']])
    require(cfg['data_complete']['path'] == str(root / 'COMPLETE.json'), 'Wrong COMPLETE pin')
    complete = read_json(root / 'COMPLETE.json')
    require(all(complete.get(k) is True for k in ('complete', 'DATA_READY', 'train_ready', 'dev_ready')),
            'Actual prepared COMPLETE/train/dev readiness required')
    require(complete['data_ready']['path'] == str(root / 'DATA_READY.json'), 'Wrong ready pointer')
    verify([complete['data_ready']])
    ready = read_json(root / 'DATA_READY.json')
    require(all(ready.get(k) is True for k in ('complete', 'DATA_READY', 'train_ready', 'dev_ready')),
            'DATA_READY must be an actual committed ready artifact')
    require(ready['requires_complete_commit_marker'] is True and ready['macro_cases'] == 16 and
            ready['logical_pair_token_ceiling'] == 100_000_000 and
            ready['unknown_forward_rule'] == 'skip-all-unknown-fields-v1', 'Prepared contract changed')
    verify(ready['files'])
    verify([ready['source_binding'], ready['code_binding']])
    require(ready['source_binding']['path'] == str(root / 'SOURCE_BINDING.json') and
            ready['code_binding']['path'] == str(root / 'CODE_BINDING.json'), 'Data bindings redirected')
    source = read_json(root / 'SOURCE_BINDING.json')
    verify(source['files'])
    code = read_json(root / 'CODE_BINDING.json')
    require(isinstance(code, dict) and code, 'Missing preparation source bytes')
    for name, encoded in code.items():
        require(Path(name).name == name, 'Unexpected preparation code key')
        matches = [p for p in source['files'] if Path(p['path']).name == name]
        require(len(matches) == 1 and
                base64.b64encode(Path(matches[0]['path']).read_bytes()).decode('ascii') == encoded,
                'Preparation code differs: ' + name)
    pins = {p['path']: p for p in ready['files']}
    required = ['data/train.jsonl', 'data/select.jsonl', 'data/confirm.jsonl',
                'FINAL_VERIFICATION.json', 'TRAIN_READY.json', 'DEV_READY.json', 'INVENTORY_READINESS.json']
    required += [f'data/{kind}-seed{seed}.{ext}' for seed in (42, 43)
                 for kind, ext in (('schedule', 'json'), ('exposure', 'jsonl'))]
    require(all(str(root / name) in pins for name in required), 'Missing pinned preparation artifact')
    final = read_json(root / 'FINAL_VERIFICATION.json')
    require(final == ready['verification'] and all(final.get(k) is True for k in
            ('complete', 'train_ready', 'dev_ready', 'exact_full_pair_cross_split_isolation',
             'old_roster_identity_gate_applied_to_every_dev_field', 'all_old_root_siblings_retained',
             'seed42_43_same_multiset_verified')), 'Actual full readback evidence missing')
    require(final['retained_original_fields_directly_reread'] > 0 and
            final['known_fields_retokenized'] > 0 and ready['runtime'], 'Empty verification/runtime')
    for domain in DOMAINS:
        counts = final['output_component_counts']
        require(counts['train:' + domain] >= 2500 and counts['select:' + domain] == 75 and
                counts['confirm:' + domain] == 50, 'Actual inventory quota mismatch')
    for name, key in (('TRAIN_READY.json', 'train_ready'), ('DEV_READY.json', 'dev_ready')):
        value = read_json(root / name)
        require(value['complete'] is True and value[key] is True and not value['incomplete_reasons'] and
                value['verification'] == pins[str(root / 'FINAL_VERIFICATION.json')], 'Sub-readiness incomplete')
    inventory = read_json(root / 'INVENTORY_READINESS.json')
    require(all(inventory[k]['ready'] is True and not inventory[k]['incomplete_reasons']
                for k in ('train', 'dev')), 'Inventory blockers remain')
    return {'complete': cfg['data_complete'], 'ready': complete['data_ready'],
            'files': ready['files'], 'source_files': source['files'], 'code': code}


def bind_admission(cfg, arm):
    verify([cfg['primary_admission']])
    admission = read_json(cfg['primary_admission']['path'])
    require(admission.get('version') == 'primary-model-admission-v1' and
            admission.get('allow_training') is True and admission.get('complete') is True,
            'Primary model admission not granted; old incomplete R1 is not admission')
    require(admission['admission_scope'] == 'scientific-full-row-train-dev-only' and
            admission['allow_full_row_dev'] is True and admission['allow_prefix_cache'] is False and
            admission['historical_solo_mixed_pass'] is False,
            'Full-row scientific admission must not rewrite failed solo/mixed or admit prefix serving')
    require(admission['precision'] == cfg['precision'] and
            admission['core_files'] == cfg['core_files'] and
            admission['native_files'] == cfg['native_files'] and
            admission['core_code'] == source_bytes(cfg['core_files']), 'Admission source/precision binding differs')
    verify(admission['evidence_files'])
    mode = admission['modes'][arm['mode']]
    require(mode['admitted'] is True and arm['layout'] in mode['layouts'], 'Mode/layout not admitted')
    results = {}
    for name in ('train_layout', 'full_row_geometry_isolation'):
        pointer = mode['checks'][name]
        verify([pointer['file']])
        require(pointer['file'] in admission['evidence_files'], 'Unbound admission evidence')
        value = read_json(pointer['file']['path'])
        for key in pointer['keys']:
            value = value[key]
        require(value['complete'] is True and value['passed'] is True and value['actual_model_run'] is True and
                value['mode'] == arm['mode'] and value['layouts'] and
                set(value['layouts']) <= {'dense', 'packed'} and
                value['precision'] == cfg['precision'] and value['core_files'] == cfg['core_files'] and
                value['native_files'] == cfg['native_files'] and value['frozen_tolerance_contract'],
                'Actual scoped primary evidence absent: ' + name)
        require(value['backend'] == 'native-fa3' and value['SDPA_fallback'] is False,
                'Primary evidence must use the actual no-fallback native FA3 backend')
        if name == 'train_layout':
            require({'dense', 'packed'} <= set(value['layouts']) and
                    value['gradient_keys'] == 393 and value['native_gqa'] is True and
                    value['head_all_tokens_before_eos'] is True and
                    all(value[k] is True for k in ('logits_pass', 'gradients_pass', 'optimizer_step_pass')),
                    'Full train-layout/logits/393grad/actual-step evidence missing')
        else:
            require(value['control_policy'] == 'same-target-micro-geometry-neighbor-append-B-or-C' and
                    value['target_pairs'] == 128 and value['target_real_tokens'] == 45216 and
                    value['micro_waves'] == 4 and value['gradient_keys'] == 393 and
                    value['layer_realT_verified'] == 28 and
                    value['optimizer_step_performed'] is False and
                    value['teacher_used'] is False and value['public_gold_used'] is False and
                    all(value[k] is True for k in ('neighbor_B_logits_bitwise_equal',
                                                  'neighbor_C_logits_bitwise_equal',
                                                  'neighbor_B_VJP_bitwise_equal',
                                                  'neighbor_C_VJP_bitwise_equal')),
                    'Actual same-geometry full-row neighbor B/C logits+393VJP evidence missing')
        results[name] = value
    return {'pin': cfg['primary_admission'], 'admission': admission, 'actual_results': results}


class Context:
    """All checks here run before importing torch or loading any model."""
    def __init__(self, config_path, arm_name):
        self.config_path = Path(config_path)
        self.config_text = self.config_path.read_text(encoding='utf-8')
        self.cfg = cfg = read_json(config_path)
        require(cfg.get('version') == 'clean-case-followup-v1' and cfg.get('approved_for_execution') is True,
                'Reviewed execution config required (no permissive defaults)')
        require(cfg['output_parent'] == '/tmp/opencode/jev-clean-trainer-20261112-followup-v1',
                'Dedicated isolated follow-up output namespace required')
        require(cfg['seed'] in (42, 43) and cfg['stock_directory'] == STOCK, 'Stock/seed contract')
        require(isinstance(cfg['arms'], dict) and dump(cfg['arms']) == dump(ALL_ARMS),
                'Exact unchanged controls and two follow-up arm definitions required')
        require(isinstance(cfg['followup_objective'], dict) and
                dump(cfg['followup_objective']) == dump(FOLLOWUP_OBJECTIVE),
                'Exact host-hard-label BCE/rank objective contract required')
        require(arm_name in FOLLOWUP_ARMS, 'Only the two isolated follow-up arms may execute')
        self.arm_name, self.arm = arm_name, cfg['arms'][arm_name]
        arm = self.arm
        require(set(arm) == {'mode', 'loss', 'layout'} and
                arm['mode'] in ('causal', 'bidirectional', 'prefix_block') and
                arm['layout'] == 'packed' and arm['loss'] in ('CE', 'BCE_RANK_HARD'), 'Bad follow-up arm')
        require(arm['mode'] == 'causal' or arm['loss'] == 'CE', 'Later mask studies are fresh CE arms')
        require(cfg['macro_cases'] == 16 and cfg['micro_plan'] ==
                {'policy': 'stable-length-common-padded-admission-v1', 'padded_tokens': 16384,
                 'max_pairs': 32, 'pad_multiple': 8}, 'Matched micro contract changed')
        require(cfg['optimizer'] == {'lr': 2e-5, 'weight_decay': 0.01, 'clip_norm': 1.0,
                                    'warmup_fraction': 0.03, 'fused': True}, 'Optimizer contract changed')
        require(type(cfg['checkpoint_every_macros']) is int and cfg['checkpoint_every_macros'] > 0,
                'Positive controlled checkpoint interval required')
        precision = cfg['precision']
        require(set(precision) == PRECISION_KEYS and
                all(type(precision[k]) is bool for k in PRECISION_KEYS -
                    {'float32_matmul_precision', 'cublas_workspace_config'}) and
                precision['float32_matmul_precision'] in ('highest', 'high', 'medium') and
                precision['cublas_workspace_config'] in (':4096:8', ':16:8'), 'Explicit precision flags required')
        require(cfg['prefix_cache'] is False and cfg['world_size'] == 1 and cfg['device'] == 'cuda:0',
                'Only single-GPU world1 full-row execution is implemented')
        require(set(Path(p['path']).name for p in cfg['trainer_files']) == set(CODE) and
                {p['path'] for p in cfg['trainer_files']} == {str(Path(__file__).parent / n) for n in CODE},
                'Every deployed trainer source must be reviewed/pinned')
        verify(cfg['trainer_files'])
        verify(cfg['core_files'])
        core = Path(cfg['core_directory']) / 'packed_mask'
        require({str(core / n) for n in ('__init__.py', 'layout.py', 'model.py', 'runtime.py')} <=
                {p['path'] for p in cfg['core_files']}, 'Core module source pins missing')
        require(all(Path(p['path']).resolve().is_relative_to(core.resolve()) for p in cfg['core_files']),
                'Core files outside declared package')
        verify(cfg['native_files'])
        require(any(Path(p['path']).suffix == '.so' for p in cfg['native_files']) and
                any(Path(p['path']).suffix == '.py' for p in cfg['native_files']),
                'Both native FA3 Python interface and actual binary must be pinned')
        verify(cfg['stock_files'])
        actual_stock = {str(p) for p in Path(STOCK).rglob('*') if p.is_file()}
        require(actual_stock == {p['path'] for p in cfg['stock_files']}, 'Pin all stock files, no unbound loader input')
        require(not any(Path(p).name.startswith('adapter_') for p in actual_stock), 'Stock cannot be an adapter')
        self.data = bind_data(cfg)
        self.admission = bind_admission(cfg, arm)
        # state_boundaries imports these helpers; delay until bindings is fully loaded.
        from state_boundaries import bind_prepared
        self.state_boundary = bind_prepared(cfg)
        self.code = source_bytes(cfg['trainer_files'])
        self.snapshot = {'version': cfg['version'], 'config': cfg, 'config_pin': stat(config_path),
                         'trainer_code': self.code, 'core_code': source_bytes(cfg['core_files']),
                         'data': self.data, 'primary': self.admission,
                         'state_boundary': self.state_boundary,
                         'arm_name': arm_name, 'arm': arm}

    def check(self):
        from state_boundaries import bind_prepared
        require(self.config_path.read_text(encoding='utf-8') == self.config_text, 'Execution config changed')
        require(source_bytes(self.cfg['trainer_files']) == self.code, 'Trainer source changed')
        require(bind_data(self.cfg) == self.data and
                bind_admission(self.cfg, self.arm) == self.admission, 'Data/admission changed')
        require(bind_prepared(self.cfg) == self.state_boundary, 'State-boundary identity chain changed')
        verify(self.cfg['native_files'])
        verify(self.cfg['stock_files'])


def offline_environment():
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                      WANDB_DISABLED='true')
