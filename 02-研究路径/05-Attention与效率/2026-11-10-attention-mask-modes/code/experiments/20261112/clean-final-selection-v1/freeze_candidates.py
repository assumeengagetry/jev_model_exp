"""Pure-CPU finite terminal selection. Never opens confirm/public payloads."""
import base64
import json
import math
import os
from pathlib import Path
import stat

ROOT = Path('/tmp/opencode/jev-clean-final-eval-20261112-v1')
ORIGINAL = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
FOLLOWUP = Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
ANALYSIS = Path('/tmp/opencode/jev-followup-select-analysis-20261112-v1')
MASK_ANALYSIS = Path('/tmp/opencode/jev-mask-select-analysis-20261112-v1')
CONVERSION = Path('/tmp/opencode/jev-terminal-packed-admission-20261112-v1')
POOL = ('C_P_BCE', 'C_P_CE', 'BI_P_CE', 'PB_P_CE', 'STATE_NL_PB_P_CE',
        'JPAIR_P_BCE_RANK', 'C_D_BCE-packed', 'C_D_CE-packed')
PIN_KEYS = {'path', 'resolved', 'bytes', 'mtime_ns'}
CURSOR = {'macro_updates': 3557, 'cases': 56912, 'logical_pair_tokens': 99824365}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    s = path.stat()
    need(stat.S_ISREG(s.st_mode), 'Regular frozen input required')
    return {'path': str(path), 'resolved': str(path.resolve()), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def verify(value):
    if isinstance(value, dict):
        if set(value) == PIN_KEYS:
            need(pin(value['path']) == value, 'Changed immutable input: ' + value['path'])
        else:
            for child in value.values():
                verify(child)
    elif isinstance(value, list):
        for child in value:
            verify(child)


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def choose(candidates):
    # Apply each tie level only within the globally tied set; do not introduce
    # a nontransitive epsilon comparator across a chain of almost-equal scores.
    tied = list(candidates)
    for key, maximum in (('mass', True), ('NLL', False), ('Brier', False)):
        best = (max if maximum else min)(x['metrics'][key] for x in tied)
        tied = [x for x in tied if abs(x['metrics'][key] - best) <= 1e-12]
    return min(tied, key=lambda x: POOL.index(x['id']))


def main():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU selection only')
    need(ROOT.is_dir() and ROOT.resolve() == ROOT and not ROOT.is_symlink(), 'Exclusive preprovisioned final root required')
    need(not (ROOT / 'FROZEN_CANDIDATES.json').exists() and not list(ROOT.glob('*BLOCKED*')), 'Prior frozen scope preserved')
    policy = ROOT / 'policy.md'
    need(policy.is_file() and policy.stat().st_mode & 0o222 == 0, 'Reviewed frozen policy required')
    quality = read(ANALYSIS / 'FOLLOWUP_SELECT_PRIMARY-cap100000000.json')
    masks = read(MASK_ANALYSIS / 'MASK_SELECT_PRIMARY-cap100000000.json')
    need(quality['complete'] and masks['complete'] and quality['frozen_node']['macro_updates'] == masks['frozen_node']['macro_updates'] == 3557,
         'All full terminal independent analyses required')
    verify(quality['source_receipt_files']); verify(masks['source_receipt_files'])
    records = []
    for arm in POOL[:6]:
        parent = FOLLOWUP if arm in POOL[4:6] else ORIGINAL
        folder = parent / (arm + '-seed42')
        done = read(folder / 'COMPLETE.json')
        need(done['complete'] and done['cursor'] == CURSOR and done['observations'] == [10000000, 30000000, 100000000] and
             done['all_checkpoints_retained'] and not list(folder.glob('BLOCKED*')) and not done['D_confirm_opened'] and not done['public_run'], 'Full model completion required: ' + arm)
        cfg = read(parent / 'seed42.json')
        need(done['scope'] == read(folder / 'OWN_SCOPE.json') and done['scope']['config'] == cfg and
             done['scope']['arm_name'] == arm and done['scope']['arm'] == cfg['arms'][arm] and cfg['arms'][arm]['layout'] == 'packed', 'Producer actual packed scope mismatch')
        verify(done['scope']); verify(done['checkpoints'] + done['D_select'])
        terminal = done['final_checkpoint']; verify(terminal)
        need(terminal in done['checkpoints'] and Path(terminal['path']).parent.parent == folder, 'Final own checkpoint mismatch')
        saved = read(terminal['path']); verify(saved['files'])
        metadata = read(Path(terminal['path']).parent / 'CHECKPOINT.json')
        need(saved['complete'] and saved['scope'] == done['scope'] and saved['macro_updates'] == 3557 and
             metadata['cursor'] == CURSOR and metadata['dev_done'] == [10000000, 30000000, 100000000], 'Terminal post-dev metadata mismatch')
        report = (masks if arm == 'BI_P_CE' else quality)['arms'][arm]
        need(report['coverage']['cases'] == 375 and report['coverage']['supported_known_fields'] == report['coverage']['known_fields'] == 615 and
             report['coverage']['forwarded_complete_candidates'] == 5190 and report['coverage']['unknown_fields_retained_no_forward'] == 90, 'Terminal full known coverage mismatch')
        metrics = {key: report['case_balanced_five_domain_means'][key] for key in ('mass', 'NLL', 'Brier')}
        need(all(math.isfinite(x) for x in metrics.values()), 'Finite terminal metrics required')
        records.append({'id': arm, 'eligible': True, 'producer_root': str(parent), 'arm_directory': str(folder),
            'producer_complete': pin(folder / 'COMPLETE.json'), 'producer_config': pin(parent / 'seed42.json'),
            'own_scope': pin(folder / 'OWN_SCOPE.json'), 'terminal_checkpoint': str(Path(terminal['path']).parent),
            'terminal_complete': terminal, 'metrics': metrics, 'mode': cfg['arms'][arm]['mode'], 'layout': 'packed',
            'loss_identity_not_inference_routing': cfg['arms'][arm]['loss'],
            'independent_analysis': pin((MASK_ANALYSIS / 'MASK_SELECT_PRIMARY-cap100000000.json') if arm == 'BI_P_CE' else (ANALYSIS / 'FOLLOWUP_SELECT_PRIMARY-cap100000000.json'))})
    for label in POOL[6:]:
        arm = label.removesuffix('-packed')
        folder = CONVERSION / label
        admission, numeric, q = read(folder / 'ADMISSION.json'), read(folder / 'NUMERICAL.json'), read(folder / 'QUALITY.json')
        consumer_done = read(folder / 'COMPLETE.json')
        need(not list(folder.glob('BLOCKED*')) and consumer_done['complete'] and
             all(consumer_done[key] is True for key in ('execution_complete', 'coverage_complete', 'weights_unchanged')),
             'Dense-consumer complete unblocked execution required')
        verify(consumer_done['files']); verify(admission['scope'])
        need(consumer_done['scope'] == admission['scope'] and admission['scope']['consumer_arm'] ==
             {'mode': 'causal', 'layout': 'packed', 'loss': 'BCE' if arm == 'C_D_BCE' else 'CE'},
             'Dense-consumer frozen scope differs')
        need(admission['execution_complete'] and admission['coverage_complete'] and admission['weights_unchanged'] and numeric['case_count'] == 375,
             'Complete dense-consumer negative or positive evidence required')
        eligible = admission['eligible_packed_candidate']
        need(type(eligible) is bool and eligible == admission['numerical_acceptance'] == numeric['numerical_acceptance'], 'Dense-consumer eligibility mismatch')
        metrics = q['actual_packed']['five_domain_case_means']
        coverage = q['actual_packed']['coverage']
        need(coverage['cases'] == 375 and coverage['supported_known_fields'] == coverage['known_fields'] == 615 and
             coverage['forwarded_complete_candidates'] == 5190 and coverage['unknown_fields_retained_no_forward'] == 90 and
             all(math.isfinite(metrics[key]) for key in
                 ('supported_known_field_argmax_target_mass', 'supported_known_field_NLL', 'supported_known_field_Brier_sum')),
             'Dense-consumer full coverage/finite quality required')
        records.append({'id': label, 'eligible': eligible, 'producer_arm': arm, 'mode': 'causal', 'layout': 'packed',
            'producer_root': str(ORIGINAL), 'arm_directory': str(ORIGINAL / (arm + '-seed42')),
            'producer_complete': admission['scope']['producer_complete'],
            'producer_config': pin(ORIGINAL / 'seed42.json'),
            'own_scope': pin(ORIGINAL / (arm + '-seed42') / 'OWN_SCOPE.json'),
            'terminal_checkpoint': str(Path(admission['scope']['producer_final_checkpoint']['path']).parent),
            'terminal_complete': admission['scope']['producer_final_checkpoint'],
            'admission': pin(folder / 'ADMISSION.json'), 'numerical': pin(folder / 'NUMERICAL.json'),
            'quality': pin(folder / 'QUALITY.json'), 'metrics': {'mass': metrics['supported_known_field_argmax_target_mass'],
                'NLL': metrics['supported_known_field_NLL'], 'Brier': metrics['supported_known_field_Brier_sum']},
            'rejection_reason': None if eligible else 'frozen-per-case-logits-margin-numeric-contract-failed'})
    baseline = next(x for x in records if x['id'] == 'C_P_CE')
    alternatives = [x for x in records if x['eligible'] and x['id'] != baseline['id']]
    challenger = choose(alternatives) if alternatives else None
    selected = {'baseline': baseline}
    if challenger is not None:
        selected['challenger'] = challenger
    need(all(x['mode'] == 'causal' for x in selected.values()), 'Selected noncausal model needs separately scoped consumer design; do not silently change model')
    ref_cfg = read(ORIGINAL / 'seed42.json')
    helper = Path('/home/dev/assumeengage/scripts/pair_data_common.py')
    data_sources = read(Path(ref_cfg['data_root']) / 'SOURCE_BINDING.json')['files']
    helper_pins = [x for x in data_sources if x['path'] == str(helper)]
    need(len(helper_pins) == 1 and pin(helper) == helper_pins[0], 'Original renderer source not uniquely bound')
    result = {'version': 'clean-terminal-frozen-candidates-v1', 'complete': True,
        'stage': 'selected-terminal-models-not-production-or-confirm-execution-admission',
        'policy': pin(policy), 'selection_source': pin(Path(__file__)), 'pool': records, 'selected': selected,
        'public_arm_roles_fixed_before_confirm': list(selected), 'D_confirm_may_not_change_public_list': True,
        'selection_metric': 'terminal packed375-case five-primary-domain-known-field-argmax-native-target-mass',
        'tie_policy': '1e-12 tied set then lowerNLL/lowerBrier/fixedpoolorder', 'eligible_checkpoint_macro': 3557,
        'only_seed': 42, 'logical_training_tokens_per_model': 99824365,
        'runtime_contract': {'backend': 'native-FA3-full-row', 'layout': 'packed', 'mode': 'causal',
            'prefix_cache': False, 'extra_fusions': False, 'graph_dummy_tokens': False, 'position_reset': True,
            'EOS': 151645, 'stock_directory': ref_cfg['stock_directory'], 'stock_files': ref_cfg['stock_files'],
            'core_files': ref_cfg['core_files'], 'native_files': ref_cfg['native_files'], 'precision': ref_cfg['precision'],
            'micro_plan': ref_cfg['micro_plan'], 'renderer_source': helper_pins[0],
            'renderer_source_bytes': base64.b64encode(helper.read_bytes()).decode(),
            'tokenizer_origin': 'same-pinned-stock-only', 'readout': 'FP64 stable field-softmax over FP32 match-minus-nonmatch; native first argmax',
            'served_parameters': 616237056, 'new_consumer_source_binding_required_before_forward': True},
        'confirm_protocol': {'cases': 250, 'primary_domain_quota': 50, 'native_split': 'validation', 'attempts_per_fixed_candidate': 1},
        'public_protocol': {'edition': '0.3', 'executed_requests_per_arm': 140620, 'scored_requests_per_arm': 140178,
            'official_exclusions': 442, 'mandatory_Suite_verify_strict': True, 'no_prediction_reuse_or_correctness_retry': True},
        'D_confirm_opened': False, 'public_run': False, 'model_forward_calls': 0, 'optimizer_steps': 0,
        'consumer_ready': False, 'no_public_checkpoint_or_runtime_selection': True, 'published': False}
    verify([x['producer_complete'] for x in records[:6]])
    write(ROOT / 'FROZEN_CANDIDATES.json', result)
    print(json.dumps({'complete': True, 'frozen_candidates': pin(ROOT / 'FROZEN_CANDIDATES.json'),
        'selected': {role: {'id': x['id'], 'metrics': x['metrics'], 'terminal_complete': x['terminal_complete']} for role, x in selected.items()},
        'ineligible_dense_conversions': [x['id'] for x in records[6:] if not x['eligible']],
        'consumer_ready': False, 'D_confirm_opened': False, 'public_run': False, 'model_forward_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
