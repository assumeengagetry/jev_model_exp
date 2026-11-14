"""Saved training-prefix diagnostics, not a quality score or new model run."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import verify_followup_node as verified


def summary(values):
    need_values = [x for x in values if x is not None]
    if not need_values:
        return {'observed': 0, 'null': len(values), 'mean': None, 'median': None, 'min': None, 'max': None}
    assert all(math.isfinite(x) for x in need_values)
    return {'observed': len(need_values), 'null': len(values) - len(need_values),
            'mean': math.fsum(need_values) / len(need_values), 'median': statistics.median(need_values),
            'min': min(need_values), 'max': max(need_values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cap', required=True, type=int, choices=tuple(verified.EXPECTED_NODES))
    args = parser.parse_args()
    parent_path = verified.ANALYSIS / f'FOLLOWUP_SELECT_PRIMARY-cap{args.cap}.json'
    out = verified.ANALYSIS / f'TRAIN_SCALING-cap{args.cap}.json'
    assert not out.exists() and not out.is_symlink(), 'Prior scaling evidence preserved'
    parent_pin = verified.meta(parent_path)
    parent = verified.parse(parent_path.read_text())
    assert parent['complete'] and parent['model_forward_calls'] == parent['optimizer_steps'] == 0
    assert verified.meta(Path(verified.__file__)) == parent['analysis_source']
    node = parent['frozen_node']
    assert (node['macro_updates'], node['logical_pair_tokens']) == verified.EXPECTED_NODES[args.cap]
    reports, common_counts = {}, None
    for arm in verified.ARMS:
        records = []
        ledger = verified.ROOTS[arm] / (arm + '-seed42') / 'ledger.jsonl'
        with ledger.open('rb') as handle:
            for line in handle:
                assert line.endswith(b'\n'), 'Committed prefix required; partial tail is not an event'
                event = verified.parse(line)
                if event['event'] != 'macro_updated':
                    continue
                index = len(records)
                assert event['cursor']['macro_updates'] == index + 1 and event['cursor']['cases'] == 16 * (index + 1)
                record = event['result']
                assert record['macro_index'] == index and record['cases'] == 16
                assert record['all393_gradients_finite'] and record['gradient_keys'] == 393
                assert record['clip_norm'] == 1.0 and math.isfinite(record['preclip_l2']) and record['preclip_l2'] >= 0
                assert record['clip_factor'] == min(1.0, 1.0 / (record['preclip_l2'] + 1e-6))
                assert math.isfinite(record['used_lr']) and record['used_lr'] >= 0
                records.append(record)
                if len(records) == node['macro_updates']:
                    assert event['cursor'] == {k: node[k] for k in ('macro_updates', 'cases', 'logical_pair_tokens')}
                    break  # Do not inspect any later live progress/tail.
        assert len(records) == node['macro_updates']
        counts = [(r['bookkeeping']['known_fields'], r['bookkeeping']['known_candidates'], r['bookkeeping']['K1_fields']) for r in records]
        if common_counts is None:
            common_counts = counts
        else:
            assert counts == common_counts, 'Same-exposure known-field/candidate/K1 accounting differs'
        report = {'actual_macro_updates': len(records), 'case_occurrences': 16 * len(records),
                  'logical_pair_tokens': sum(r['logical_pair_tokens'] for r in records),
                  'known_field_occurrences': sum(x[0] for x in counts), 'candidate_occurrences': sum(x[1] for x in counts),
                  'known_K1_field_occurrences': sum(x[2] for x in counts),
                  'positive_LR_calls': sum(r['used_lr'] > 0 for r in records),
                  'clip_engaged_macros': sum(r['clip_factor'] < 1.0 for r in records),
                  'clip_engaged_macro_fraction': sum(r['clip_factor'] < 1.0 for r in records) / len(records),
                  'preclip_parameter_gradient_l2': summary([r['preclip_l2'] for r in records]),
                  'recorded_scalar_clip_factor': summary([r['clip_factor'] for r in records]),
                  'actual_used_lr': summary([r['used_lr'] for r in records]),
                  'training_objective_over_changing_weights_not_quality': summary([r['objective'] for r in records])}
        assert report['logical_pair_tokens'] == node['logical_pair_tokens']
        if arm in ('STATE_NL_PB_P_CE', 'JPAIR_P_BCE_RANK'):
            report['raw_field_occurrences'] = sum(r['coverage']['raw_fields'] for r in records)
            report['unknown_field_occurrences_retained_no_forward'] = sum(r['coverage']['unknown_fields_retained_no_forward'] for r in records)
            assert report['raw_field_occurrences'] == report['known_field_occurrences'] + report['unknown_field_occurrences_retained_no_forward']
        if arm == 'JPAIR_P_BCE_RANK':
            count_keys = ('hard_positive_candidates', 'hard_negative_candidates', 'soft_candidates',
                          'rank_active_fields', 'rank_zero_fields', 'rank_zero_no_positive_fields',
                          'rank_zero_no_negative_fields', 'rank_zero_K1_fields', 'rank_zero_soft_only_fields', 'no_rank_cases')
            total = Counter({k: 0 for k in count_keys})
            for record in records:
                b = record['bookkeeping']
                assert b['rank_active_fields'] + b['rank_zero_fields'] == b['known_fields']
                assert b['hard_positive_candidates'] + b['hard_negative_candidates'] + b['soft_candidates'] == b['known_candidates']
                assert b['macro_total_loss'] == record['objective']
                total.update({k: b[k] for k in count_keys})
            report['rank_actual_occurrence_counts'] = dict(total)
            report['rank_active_known_field_fraction'] = total['rank_active_fields'] / report['known_field_occurrences']
            report['rank_zero_reasons_overlap_not_a_denominator_partition'] = True
            component_keys = ('macro_bce_loss', 'macro_rank_loss', 'macro_total_loss',
                              'margin_bce_l2', 'margin_rank_l2', 'margin_total_l2',
                              'margin_rank_over_bce_l2', 'margin_total_over_bce_l2')
            report['actual_component_scalars_and_dL_dmargin_norms'] = {
                key: summary([r['bookkeeping'][key] for r in records]) for key in component_keys}
            report['component_norms_are_not_393_parameter_gradient_norms'] = True
        reports[arm] = report
    assert verified.meta(parent_path) == parent_pin
    result = {'version': 'saved-training-scaling-prefix-v1', 'complete': True, 'frozen_node': node,
              'quality_primary': parent_pin, 'quality_analysis_source': parent['analysis_source'],
              'scaling_analysis_source': verified.meta(__file__), 'arms': reports,
              'statistical_unit': 'actual completed macro / case occurrence, not unique case or dev quality',
              'diagnostic_only_no_optimizer_protocol_adaptation': True,
              'scope': 'loss/scale/clip diagnostics on already-saved actual complete exposure prefixes; not directional causality proof',
              'model_forward_calls': 0, 'optimizer_steps': 0, 'evaluator_started': False,
              'D_confirm_opened': False, 'public_run': False}
    with out.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        import os
        os.fsync(handle.fileno())
    print(json.dumps({'complete': True, 'frozen_node': node,
        'clips': {a: {'fraction': v['clip_engaged_macro_fraction'], 'preclip_mean': v['preclip_parameter_gradient_l2']['mean']} for a, v in reports.items()},
        'rank_counts': reports['JPAIR_P_BCE_RANK']['rank_actual_occurrence_counts'],
        'rank_active_fraction': reports['JPAIR_P_BCE_RANK']['rank_active_known_field_fraction'],
        'saved_result': str(out), 'model_forward_calls': 0, 'optimizer_steps': 0}, indent=2))


if __name__ == '__main__':
    main()
