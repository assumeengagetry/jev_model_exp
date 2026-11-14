"""Independent CPU analysis of a committed frozen D_select node, no model run.

Reads only saved logits/margins and the original pinned select roster. Detailed
case records stay remote; the output contains aggregates/pins, never case text,
case IDs, targets, logits or token arrays. No trainer/evaluator module is imported.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct

ROOT = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
DATA = Path('/tmp/opencode/jev-clean-origin-20261112-v1/prepared-01')
ARMS = ('C_P_CE', 'BI_P_CE', 'PB_P_CE')
DOMAINS = ('knowledge', 'language', 'retrieval', 'tools', 'arts')
METRICS = {
    'mass': 'supported_known_field_argmax_target_mass',
    'NLL': 'supported_known_field_NLL',
    'normalized_NLL': 'supported_known_field_normalized_NLL',
    'Brier': 'supported_known_field_Brier_sum',
    'coverage': 'known_field_coverage',
    'coverage_adjusted_mass': 'coverage_adjusted_argmax_target_mass',
}


def read(path):
    return json.loads(Path(path).read_text())


def meta(path):
    path = Path(path)
    s = path.stat()
    return {'path': str(path), 'resolved': str(path.resolve()), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def verify(pins):
    for pin in pins:
        assert meta(pin['path']) == pin, 'Pinned artifact changed: ' + pin['path']


def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def mean(values):
    return math.fsum(values) / len(values)


def distribution(margins):
    peak = max(margins)
    normalizer = peak + math.log(math.fsum(math.exp(v - peak) for v in margins))
    log_p = [v - normalizer for v in margins]
    return log_p, [math.exp(v) for v in log_p]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap', required=True, type=int, choices=(10_000_000, 30_000_000, 100_000_000))
    args = parser.parse_args()
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    cfg = read(ROOT / 'seed42.json')
    verify(cfg['trainer_files'] + cfg['core_files'] + cfg['native_files'] + [cfg['data_complete'], cfg['fresh_init']['complete']])
    ready = read(DATA / 'DATA_READY.json')
    verify(ready['files'])
    select_pin = next(p for p in ready['files'] if p['path'] == str(DATA / 'data/select.jsonl'))
    roster = [json.loads(line) for line in Path(select_pin['path']).open()]
    assert len(roster) == 375 and len({c['case_id'] for c in roster}) == 375
    assert Counter(c['primary_domain'] for c in roster) == {d: 75 for d in DOMAINS}
    node = next(n for n in read(DATA / 'data/schedule-seed42.json')['observations'] if n['cap'] == args.cap)
    name = f"D_select-cap{args.cap}-m{node['macro_updates']:07d}"
    out = Path('/tmp/opencode/jev-mask-select-analysis-20261112-v1') / f"MASK_SELECT_PRIMARY-cap{args.cap}.json"
    assert not out.exists(), 'Previous primary evidence preserved; do not silently replace'
    reports, records, signatures, pins = {}, {}, {}, []
    for arm in ARMS:
        arm_root = ROOT / (arm + '-seed42')
        folder = arm_root / name
        done, s = read(folder / 'COMPLETE.json'), read(folder / 'SUMMARY.json')
        assert done['complete'] and not (folder / 'BLOCKED.json').exists() and done['node'] == s['node'] == node
        verify(done['files'])
        assert {Path(p['path']).name for p in done['files']} == {'SUMMARY.json', 'case-metrics.jsonl'}
        assert done['scope'] == s['scope'] == read(arm_root / 'OWN_SCOPE.json')
        assert s['scope']['config'] == cfg and s['scope']['config_pin'] == meta(ROOT / 'seed42.json')
        assert s['scope']['arm_name'] == arm and s['scope']['arm'] == cfg['arms'][arm]
        assert s['checkpoint_identity'] == done['checkpoint_identity'] and s['checkpoint_identity']['macro_updates'] == node['macro_updates']
        assert s['checkpoint_identity']['initialization']['fresh_init']['complete'] == cfg['fresh_init']['complete']
        assert s['roster'] == 'D_select_only' and s['primary_metric'] == METRICS['mass']
        assert all(s[k] is False for k in ('D_confirm_opened', 'public_run', 'winner_selected', 'candidate_pruning', 'truncation', 'short_case_replacement', 'prefix_cache'))
        raw = [json.loads(line) for line in (folder / 'case-metrics.jsonl').open()]
        assert len(raw) == 375 and [x['case_id'] for x in raw] == [c['case_id'] for c in roster]
        coverage = Counter()
        calculated, arm_signatures, exact = [], [], []
        maximum_metric_error = 0.0
        for original, saved in zip(roster, raw):
            assert saved['component'] == original['component'] and saved['primary_domain'] == original['primary_domain']
            assert len(saved['fields']) == len(original['fields'])
            field_values, chosen, supported = [], [], []
            known = sum(f['known'] for f in original['fields'])
            unknown = len(original['fields']) - known
            assert saved['known_fields'] == known and saved['unknown_fields'] == unknown and known > 0
            for i, (field, value) in enumerate(zip(original['fields'], saved['fields'])):
                assert value['field_index'] == i and value['coordinate'] == field['coordinate'] and value['domain'] == field['domain']
                assert value['known'] == field['known']
                if not field['known']:
                    assert value['status'] == 'masked_unknown_no_forward' and value['K'] is None
                    assert not any(k in value for k in ('margins', 'logits', 'native_target_not_renormalized'))
                    coverage['unknown_fields_retained_no_forward'] += 1
                    chosen.append(None)
                    continue
                coverage['known_fields'] += 1
                assert field['static_unsupported_reason'] is None, 'This node is expected fully known-covered'
                assert value['status'] == 'supported_complete_field'
                q, m, logits = field['target'], value['margins'], value['logits']
                assert value['native_target_not_renormalized'] == q and value['K'] == len(q) == len(field['options']) == len(m) == len(logits)
                assert all(math.isfinite(v) for v in m) and all(len(v) == 2 and all(math.isfinite(x) for x in v) for v in logits)
                # Saved BF16 logits are represented exactly as FP32. Reproduce
                # actual FP32 subtraction rather than assume FP64 subtraction exactness.
                assert all(v == struct.unpack('<f', struct.pack('<f', pair[1] - pair[0]))[0] for v, pair in zip(m, logits))
                best = max(range(len(m)), key=m.__getitem__)
                log_p, p = distribution(m)
                nll = -math.fsum(a * b for a, b in zip(q, log_p))
                brier = math.fsum((a - b) ** 2 for a, b in zip(p, q))
                field_result = {'mass': q[best], 'NLL': nll, 'normalized_NLL': nll / math.log(len(q)) if len(q) > 1 else 0.0, 'Brier': brier}
                assert value['argmax_index'] == best
                for key, target in (('mass', 'argmax_target_mass'), ('NLL', 'NLL'), ('normalized_NLL', 'normalized_NLL'), ('Brier', 'Brier_sum')):
                    close(field_result[key], value[target])
                    maximum_metric_error = max(maximum_metric_error, abs(field_result[key] - value[target]))
                field_values.append(field_result)
                chosen.append(best)
                supported.append(i)
                coverage['supported_known_fields'] += 1
                coverage['forwarded_complete_candidates'] += len(q)
            assert len(supported) == known == saved['supported_known_fields']
            values = {key: mean([f[key] for f in field_values]) for key in ('mass', 'NLL', 'normalized_NLL', 'Brier')}
            values.update(coverage=len(supported) / known, coverage_adjusted_mass=math.fsum(f['mass'] for f in field_values) / known)
            for key, target in METRICS.items():
                close(values[key], saved[target])
            exact_supported = original['complete_case_exact_static_supported']
            assert saved['complete_onehot_case_exact_supported'] == exact_supported
            if exact_supported:
                result = all(v['mass'] == 1.0 for v in field_values)
                assert saved['complete_onehot_case_exact'] == result
                exact.append(result)
                coverage['complete_onehot_case_exact_supported'] += 1
            else:
                assert saved['complete_onehot_case_exact'] is None
            calculated.append({'domain': original['primary_domain'], **values})
            arm_signatures.append(chosen)
            coverage['cases'] += 1
            coverage['cases_with_all_known_fields_supported'] += len(supported) == known
            coverage['cases_with_any_supported_known_field'] += bool(supported)
        assert dict(coverage) == s['coverage'] and coverage['known_fields'] == coverage['supported_known_fields'] == 615
        assert coverage['forwarded_complete_candidates'] == 5190 and coverage['unknown_fields_retained_no_forward'] == 90
        per_domain = {}
        for domain in DOMAINS:
            entries = [v for v in calculated if v['domain'] == domain]
            assert len(entries) == s['per_domain'][domain]['cases'] == 75
            per_domain[domain] = {key: mean([v[key] for v in entries]) for key in METRICS}
            for key, target in METRICS.items():
                close(per_domain[domain][key], s['per_domain'][domain][target])
                assert s['per_domain'][domain][target + '_case_denominator'] == 75
        five = {key: mean([per_domain[d][key] for d in DOMAINS]) for key in METRICS}
        for key, target in METRICS.items():
            close(five[key], s['five_domain_case_means'][target])
        assert len(exact) == s['complete_onehot_case_exact_denominator'] == 259
        close(mean(exact), s['complete_onehot_case_exact'])
        before, after = s['runtime_before']['forward'], s['runtime_after']['forward']
        train_tokens = node['logical_pair_tokens']
        assert before['train_first.completed_real_tokens'] == before['train_replay.completed_real_tokens'] == before['train_replay.backward_real_tokens'] == train_tokens
        assert all(after[k] == v for k, v in before.items() if k.startswith(('train_first.', 'train_replay.'))), 'Evaluation changed historical training counters'
        for phase in ('train_first', 'train_replay'):
            assert before[phase + '.attempted_waves'] == before[phase + '.completed_waves']
            assert before[phase + '.completed_native_FA3_calls'] == 28 * (2 if cfg['arms'][arm]['mode'] == 'prefix_block' else 1) * before[phase + '.completed_waves']
            rows = before[phase + '.completed_tokenwise_rows_per_operator']
            assert rows == train_tokens if cfg['arms'][arm]['layout'] == 'packed' else rows >= train_tokens
        dev_tokens = after['dev_select.completed_real_tokens'] - before.get('dev_select.completed_real_tokens', 0)
        assert dev_tokens == sum(c['logical_supervised_pair_tokens'] for c in roster) == 1554484
        dev_rows = after['dev_select.completed_tokenwise_rows_per_operator'] - before.get('dev_select.completed_tokenwise_rows_per_operator', 0)
        dev_waves = after['dev_select.completed_waves'] - before.get('dev_select.completed_waves', 0)
        assert dev_rows == dev_tokens if cfg['arms'][arm]['layout'] == 'packed' else dev_rows >= dev_tokens
        # Stop at this node's post-dev checkpoint: do not repeatedly inspect live
        # progress, or infer current/final state from the observation's success.
        pre = post = started = completed = None
        with (arm_root / 'ledger.jsonl').open() as handle:
            for line in handle:
                event = json.loads(line)
                if event['event'] == 'checkpoint_committed' and event['cursor']['macro_updates'] == node['macro_updates']:
                    if started is None:
                        pre = event
                    elif completed is not None:
                        post = event
                        break
                if event['event'] in ('dev_started', 'dev_completed') and event['node']['cap'] == args.cap:
                    assert event['node'] == node
                    if event['event'] == 'dev_started':
                        assert started is None
                        started = event
                    else:
                        assert completed is None
                        completed = event
        assert pre and started and completed and post and pre['cursor'] == post['cursor']
        assert completed['complete'] == meta(folder / 'COMPLETE.json')
        for item in (pre, post):
            verify([item['complete']])
            checkpoint = read(item['complete']['path'])
            assert checkpoint['complete'] and checkpoint['scope'] == done['scope'] and checkpoint['macro_updates'] == node['macro_updates']
            verify(checkpoint['files'])
        records[arm], signatures[arm] = calculated, arm_signatures
        pins.extend([meta(folder / 'COMPLETE.json')] + done['files'] + [pre['complete'], post['complete']])
        reports[arm] = {'node': node, 'case_balanced_five_domain_means': five, 'per_domain': per_domain, 'coverage': dict(coverage),
            'complete_onehot_case_exact': mean(exact), 'complete_onehot_case_exact_denominator': len(exact),
            'complete_onehot_case_exact_count': sum(exact), 'maximum_independent_field_metric_absolute_error': maximum_metric_error,
            'actual_training_real_forward_tokens': 2 * train_tokens, 'actual_training_backward_real_tokens': train_tokens,
            'actual_training_tokenwise_rows_per_operator_two_passes': before['train_first.completed_tokenwise_rows_per_operator'] + before['train_replay.completed_tokenwise_rows_per_operator'],
            'D_select_real_tokens': dev_tokens, 'D_select_tokenwise_rows_per_operator': dev_rows, 'D_select_waves': dev_waves,
            'D_select_wall_seconds_in_training_ledger': completed['elapsed_this_invocation'] - started['elapsed_this_invocation'],
            'pre_dev_checkpoint': pre['complete'], 'post_dev_checkpoint': post['complete'],
            'loss_layout_mode_seed_and_common_fresh_scope_verified': True, 'complete_candidate_order_and_unknown_skip_verified': True,
            'independent_raw_logits_margin_field_case_domain_metrics_verified': True}
    contrasts = {
        'bidirectional_minus_causal_CE': {'BI_P_CE': 1, 'C_P_CE': -1},
        'candidate_prefix_block_minus_causal_CE': {'PB_P_CE': 1, 'C_P_CE': -1},
        'candidate_prefix_block_minus_bidirectional_CE': {'PB_P_CE': 1, 'BI_P_CE': -1},
    }
    paired = {}
    # Descriptive paired-case uncertainty only; no hypothesis-based stopping,
    # checkpoint choice or optimizer decisions use these intervals.
    bootstrap_replicates, bootstrap_seed = 10000, 20261112
    samples = {name: [] for name in contrasts}
    by_domain = {d: [i for i, row in enumerate(records[ARMS[0]]) if row['domain'] == d] for d in DOMAINS}
    case_mass_deltas = {}
    for name, weights in contrasts.items():
        deltas = [{k: math.fsum(weight * records[arm][i][k] for arm, weight in weights.items()) for k in METRICS} for i in range(375)]
        means = {k: mean([row[k] for row in deltas]) for k in METRICS}
        domain_means = {d: {k: mean([deltas[i][k] for i in indices]) for k in METRICS} for d, indices in by_domain.items()}
        values = [row['mass'] for row in deltas]
        case_mass_deltas[name] = {d: [values[i] for i in indices] for d, indices in by_domain.items()}
        paired[name] = {'weights': weights, 'paired_cases': 375, 'metric_deltas': means, 'per_domain': domain_means,
            'mass_case_positive_negative_equal': {'positive': sum(v > 1e-12 for v in values), 'negative': sum(v < -1e-12 for v in values), 'equal_within_1e-12': sum(abs(v) <= 1e-12 for v in values)},
            'mass_stratified_case_standard_error': math.sqrt(math.fsum(statistics.variance([values[i] for i in indices]) / len(indices) for indices in by_domain.values())) / 5}
        if len(weights) == 2:
            a, b = weights.keys()
            paired[name]['field_argmax_disagreements'] = sum(x != y for ra, rb in zip(signatures[a], signatures[b]) for x, y in zip(ra, rb) if x is not None)
            paired[name]['cases_with_field_argmax_disagreement'] = sum(any(x != y for x, y in zip(ra, rb)) for ra, rb in zip(signatures[a], signatures[b]))
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_replicates):
        draw = {d: [rng.randrange(75) for _ in range(75)] for d in DOMAINS}
        for name in contrasts:
            samples[name].append(math.fsum(case_mass_deltas[name][d][i] for d in DOMAINS for i in draw[d]) / 375)
    for name in contrasts:
        values = sorted(samples[name])
        paired[name]['mass_paired_case_stratified_bootstrap_95pct_percentile_interval'] = [values[249], values[9749]]
    verify(pins + [select_pin])
    result = {'complete': True, 'verified_utc': datetime.now(timezone.utc).isoformat(), 'version': 'independent-committed-select-analysis-v1',
        'frozen_node': node, 'arms': reports, 'paired_case_contrasts': paired,
        'bootstrap': {'replicates': bootstrap_replicates, 'seed': bootstrap_seed, 'unit': 'whole case, paired across all three mask arms, stratified by five primary domains',
            'interval_scope': 'descriptive single-seed scheduled D_select uncertainty only; not multiplicity-adjusted and not training-seed uncertainty'},
        'source_receipt_files': pins, 'original_select_roster': select_pin,
        'analysis_source': meta(__file__), 'model_forward_calls': 0, 'optimizer_steps': 0, 'evaluator_started': False,
        'D_confirm_opened': False, 'public_run': False, 'winner_selected': False, 'training_source_or_config_modified': False,
        'limitations': ['10M/30M intermediate and100M final; no public checkpoint selection or protocol adaptation',
            'D_select target mass is not Public Index and cannot be compared numerically to51.99',
            'one training seed; paired-case intervals do not cover training-seed variability or prove numerical equivalence',
            'tools dev mainly procedural, not BFCL/real-tool generalization',
            'fullrow scientific training/dev only; no cached-serving or official latency admission']}
    with out.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({'complete': True, 'frozen_node': node,
        'metrics': {a: v['case_balanced_five_domain_means'] for a, v in reports.items()},
        'paired_mass': {n: {'delta': v['metric_deltas']['mass'], '95pct_case_interval': v['mass_paired_case_stratified_bootstrap_95pct_percentile_interval']} for n, v in paired.items()},
        'saved_primary_verification': str(out), 'model_or_evaluator_started': False, 'D_confirm_opened': False}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
