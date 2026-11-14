"""Only the three predeclared D_select nodes. No confirm/public interface."""
from __future__ import annotations

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from bindings import DOMAINS, commit_json, dump, fsync_tree, require, stat, write_json
from case_runtime import micro_plan, slots


def average(values):
    return sum(values) / len(values) if values else None


def evaluate_select(runtime, store, output, *, scope, node, checkpoint_identity):
    import torch
    require(node['cap'] in (10_000_000, 30_000_000, 100_000_000) and
            node['cases'] == node['macro_updates'] * 16 and node['macro_updates'] > 0,
            'Only scheduled complete-macro observation evaluation is legal')
    require(store.path.name == 'select.jsonl' and store.split == 'validation' and
            store.domains == {d: 75 for d in DOMAINS}, 'Only full fixed D_select roster is legal')
    output = Path(output)
    require(not output.exists(), 'Existing dev attempt must be preserved, never overwritten/reused')
    output.mkdir(mode=0o700)
    runtime.eval()
    runtime.phase = 'dev_select'
    before = runtime.receipt()
    per_domain = {d: [] for d in DOMAINS}
    field_domain_cases = {d: [] for d in DOMAINS}
    all_cases, coverage = [], Counter()
    try:
        with (output / 'case-metrics.jsonl').open('x', encoding='utf-8') as handle:
            with torch.no_grad():
                for case in store:
                    pairs, fields = slots([case], dev=True, boundary_store=runtime.boundary_store)
                    margins = torch.empty(len(pairs), dtype=torch.float32, device='cpu')
                    all_logits = torch.empty((len(pairs), 2), dtype=torch.float32, device='cpu')
                    for micro in micro_plan(pairs, dev=True):
                        logits, m = runtime.forward(micro)
                        indices = [p.slot for p in micro]
                        margins[indices], all_logits[indices] = m.cpu(), logits.cpu()
                        del logits, m
                    known = sum(f['known'] for f in case['fields'])
                    supported = len(fields)
                    mass, nll, normalized_nll, brier, field_results = [], [], [], [], []
                    field_by_index = {f['field_index']: f for f in fields}
                    for fi, raw in enumerate(case['fields']):
                        item = {'field_index': fi, 'coordinate': raw['coordinate'], 'domain': raw['domain'],
                                'known': raw['known'], 'K': len(raw['options']) if raw['known'] else None}
                        if not raw['known']:
                            item.update(status='masked_unknown_no_forward', unknown_reason=raw['unknown_reason'])
                            coverage['unknown_fields_retained_no_forward'] += 1
                        elif fi not in field_by_index:
                            require(raw['static_unsupported_reason'] == 'pair_context>40960', 'No dynamic candidate dropping')
                            item.update(status='static_unsupported', reason=raw['static_unsupported_reason'],
                                        original_pair_lengths=raw['pair_lengths'])
                            coverage['static_unsupported_known_fields'] += 1
                        else:
                            field = field_by_index[fi]
                            m = margins[field['start']:field['end']].double()
                            q = torch.tensor(raw['target'], dtype=torch.float64)
                            log_p = torch.log_softmax(m, dim=0)
                            p = log_p.exp()
                            best = int(torch.argmax(m).item())  # Frozen native candidate order breaks ties.
                            score = float(q[best].item())
                            value_nll = float(-(q * log_p).sum().item())
                            value_brier = float(((p - q) ** 2).sum().item())
                            value_normalized_nll = value_nll / math.log(len(q)) if len(q) > 1 else 0.0
                            mass.append(score)
                            nll.append(value_nll)
                            brier.append(value_brier)
                            normalized_nll.append(value_normalized_nll)
                            item.update(status='supported_complete_field', argmax_index=best, argmax_target_mass=score,
                                        NLL=value_nll, Brier_sum=value_brier, normalized_NLL=value_normalized_nll,
                                        margins=m.tolist(), logits=all_logits[field['start']:field['end']].tolist(),
                                        native_target_not_renormalized=raw['target'])
                            coverage['supported_known_fields'] += 1
                            coverage['forwarded_complete_candidates'] += len(q)
                        field_results.append(item)
                    require(known > 0 and supported == len(mass), 'Known-field evaluation bookkeeping differs')
                    exact_supported = bool(case['complete_case_exact_static_supported'])
                    require(not exact_supported or (known == len(case['fields']) == supported and
                            all(f['onehot'] for f in case['fields'])), 'Unsupported complete-case exact claim')
                    # All375 cases stay in records and coverage; unsupported-only cases have null
                    # conditional metrics. Zero-credit sensitivity uses the complete known denominator.
                    result = {'case_id': case['case_id'], 'component': case['component'],
                              'primary_domain': case['primary_domain'], 'known_fields': known,
                              'supported_known_fields': supported, 'unknown_fields': len(case['fields']) - known,
                              'known_field_coverage': supported / known,
                              'coverage_adjusted_argmax_target_mass': sum(mass) / known,
                              'supported_known_field_argmax_target_mass': average(mass),
                              'supported_known_field_NLL': average(nll),
                              'supported_known_field_normalized_NLL': average(normalized_nll),
                              'supported_known_field_Brier_sum': average(brier),
                              'complete_onehot_case_exact_supported': exact_supported,
                              'complete_onehot_case_exact': all(x == 1.0 for x in mass) if exact_supported else None,
                              'fields': field_results}
                    handle.write(dump(result) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    per_domain[case['primary_domain']].append(result)
                    all_cases.append(result)
                    for domain in DOMAINS:
                        domain_known = [f for f in field_results if f['known'] and f['domain'] == domain]
                        domain_supported = [f for f in domain_known if f['status'] == 'supported_complete_field']
                        if domain_known:
                            field_domain_cases[domain].append({
                                'argmax_target_mass': average([f['argmax_target_mass'] for f in domain_supported]),
                                'NLL': average([f['NLL'] for f in domain_supported]),
                                'Brier_sum': average([f['Brier_sum'] for f in domain_supported]),
                                'known_field_coverage': len(domain_supported) / len(domain_known)})
                    coverage['cases'] += 1
                    coverage['known_fields'] += known
                    coverage['cases_with_any_supported_known_field'] += supported > 0
                    coverage['cases_with_all_known_fields_supported'] += supported == known
                    coverage['complete_onehot_case_exact_supported'] += exact_supported
        metrics = ('coverage_adjusted_argmax_target_mass', 'supported_known_field_argmax_target_mass',
                   'supported_known_field_NLL', 'supported_known_field_normalized_NLL',
                   'supported_known_field_Brier_sum', 'known_field_coverage')
        domain_summary = {}
        for domain, values in per_domain.items():
            require(len(values) == 75, 'D_select fixed quota not fully evaluated/recorded')
            domain_summary[domain] = {'cases': len(values)}
            for metric in metrics:
                observed = [v[metric] for v in values if v[metric] is not None]
                domain_summary[domain][metric] = average(observed)
                domain_summary[domain][metric + '_case_denominator'] = len(observed)
        five_domain = {metric: (average([domain_summary[d][metric] for d in DOMAINS])
                                 if all(domain_summary[d][metric] is not None for d in DOMAINS) else None)
                       for metric in metrics}
        exact = [v['complete_onehot_case_exact'] for v in all_cases
                 if v['complete_onehot_case_exact_supported']]
        field_domain_summary = {}
        for domain, values in field_domain_cases.items():
            field_domain_summary[domain] = {'cases_with_known_field_in_domain': len(values)}
            for metric in ('argmax_target_mass', 'NLL', 'Brier_sum', 'known_field_coverage'):
                observed = [v[metric] for v in values if v[metric] is not None]
                field_domain_summary[domain][metric] = average(observed)
                field_domain_summary[domain][metric + '_case_denominator'] = len(observed)
        summary = {'scope': scope, 'node': node, 'checkpoint_identity': checkpoint_identity,
                   'roster': 'D_select_only', 'statistical_unit': 'complete native case; equal mean of five primary domains',
                   'primary_metric': 'supported_known_field_argmax_target_mass',
                   'primary_definition': 'mean q[argmax margin] over supported known fields/case, mean cases/domain, equal mean of five domains; missing case/domain explicitly null',
                   'coverage_sensitivity_metric': 'coverage_adjusted_argmax_target_mass',
                   'static_missing_definition': 'zero credit only; no fabricated target, all cases retained; conditional score/coverage also reported',
                   'five_domain_case_means': five_domain, 'per_domain': domain_summary,
                   'field_domain_case_means_secondary_not_selection_metric': field_domain_summary,
                   'coverage': dict(coverage), 'complete_onehot_case_exact': average(exact),
                   'complete_onehot_case_exact_denominator': len(exact),
                   'Brier_definition': 'sum over complete native candidates of (softmax(m)-native q)^2',
                   'NLL_definition': '-sum(native q*log_softmax(m)); no target renormalization; K1 normalized NLL=0',
                   'runtime_before': before, 'runtime_after': runtime.receipt(),
                   'candidate_pruning': False, 'truncation': False, 'short_case_replacement': False,
                   'prefix_cache': False, 'D_confirm_opened': False, 'public_run': False,
                   'winner_selected': False}
        write_json(output / 'SUMMARY.json', summary)
        fsync_tree(output)
        commit_json(output / 'COMPLETE.json', {'complete': True, 'scope': scope, 'node': node,
                                            'checkpoint_identity': checkpoint_identity,
                                            'files': [stat(p) for p in sorted(output.iterdir()) if p.is_file()]})
        fsync_tree(output)
        return summary
    except BaseException as error:
        write_json(output / 'BLOCKED.json', {'complete': False, 'error_type': type(error).__name__,
                                           'error': str(error), 'partial_preserved': True,
                                           'completed_case_records': len(all_cases)})
        raise
    finally:
        runtime.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--arm-dir', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--checkpoint-complete-pin', required=True, type=Path)
    args = parser.parse_args()
    # Reuse exactly the trainer's model/data/admission/resume/checkpoint gates, never an alternate loader.
    from train_case import main as train_main
    return train_main(['--config', str(args.config), '--arm', args.arm,
                       '--output', str(args.arm_dir), '--resume-checkpoint', str(args.checkpoint),
                       '--resume-complete-pin', str(args.checkpoint_complete_pin), '--evaluate-only'])


if __name__ == '__main__':
    raise SystemExit(main())
