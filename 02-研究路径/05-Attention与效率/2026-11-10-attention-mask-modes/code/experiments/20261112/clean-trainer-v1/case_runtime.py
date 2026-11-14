"""Bounded JSONL offset stores, frozen exposure and layout-independent micros."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

from bindings import DOMAINS, dump, read_json, require, stat


def check_case(case, split):
    require(case['case_id'] == f"clean-semantic-{case['component']}" and
            case['split'] == split and case['primary_domain'] in DOMAINS and
            case['fields'] and not case['rejection_reasons'] and case['truncated'] is False,
            'Illegal complete case')
    require(case['known_mask'] == [f['known'] for f in case['fields']] and
            all(type(f['known']) is bool for f in case['fields']) and
            case['unknown_forward_rule'] == 'skip-all-unknown-fields-v1', 'Mask/forward policy differs')
    coords, tokens, maximum, known, k1 = set(), 0, 0, 0, 0
    for field in case['fields']:
        coord = dump(field['coordinate'])
        require(coord not in coords, 'Duplicate native field coordinate')
        coords.add(coord)
        require(field['domain'] in DOMAINS, 'Unknown field domain')
        if not field['known']:
            require(field['target'] is None and
                    field['prefix_ids'] == field['suffix_ids'] == field['pair_lengths'] == [] and
                    field['forward_policy'] == 'masked_unknown_no_forward' and
                    field['static_unsupported_reason'] == 'unknown_supervision_masked_no_forward',
                    'Unknown must retain raw but never manufacture input/negatives')
            require('raw_target' in field and 'raw_options' in field, 'Lost unknown raw criteria')
            continue
        known += 1
        q = field['target']
        require(isinstance(q, list) and len(q) == len(field['options']) == len(field['suffix_ids']) >= 1 and
                all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in q) and
                math.isclose(math.fsum(q), 1, rel_tol=1e-6, abs_tol=1e-6), 'Invalid native distribution')
        require(field['prefix_ids'] and field['forward_policy'] == 'known_complete_candidates', 'Missing prefix')
        require(all(type(t) is int and t >= 0 for t in field['prefix_ids']), 'Invalid prefix IDs')
        lengths = []
        for suffix in field['suffix_ids']:
            require(suffix and suffix[-1] == 151645 and all(type(t) is int and t >= 0 for t in suffix),
                    'Every actual suffix must end in native EOS')
            lengths.append(len(field['prefix_ids']) + len(suffix))
        require(lengths == field['pair_lengths'], 'Pair length record differs')
        unsupported = 'pair_context>40960' if max(lengths) > 40960 else None
        require(field['static_unsupported_reason'] == unsupported, 'Static coverage changed')
        if split == 'train':
            require(max(lengths) <= 8192 and unsupported is None, 'Whole train case overlength')
        tokens += sum(lengths)
        maximum = max(maximum, max(lengths))
        k1 += bool(field.get('zero_field_ce_gradient_K1'))
    require(known > 0 and case['logical_supervised_pair_tokens'] ==
            case['all_field_pair_tokens_once'] == case['scheduled_forward_pair_tokens_once'] == tokens and
            case['max_pair_tokens'] == maximum and case['unknown_field_forward_tokens_once'] == 0 and
            case['unknown_fields_no_forward'] == len(case['fields']) - known and
            case['zero_field_ce_gradient_K1_fields'] == k1, 'Complete case accounting differs')
    return {'tokens': tokens, 'domain': case['primary_domain'], 'family': case['family_stratum'],
            'component': case['component'], 'known_fields': known, 'max_pair': maximum}


class CaseStore:
    """One JSON record at a time; RAM holds offsets + small scalar metadata only."""
    def __init__(self, path, split):
        self.path, self.split = Path(path), split
        self.pin = stat(path)
        self.offsets, self.meta = {}, {}
        self.handle = self.path.open('rb')
        domains, components = Counter(), set()
        try:
            while True:
                offset = self.handle.tell()
                line = self.handle.readline()
                if not line:
                    break
                case = json.loads(line)
                cid = case['case_id']
                require(cid not in self.offsets and case['component'] not in components, 'Duplicate case/component')
                self.meta[cid] = check_case(case, split)
                self.offsets[cid] = offset
                components.add(case['component'])
                domains[case['primary_domain']] += 1
            require(stat(path) == self.pin, 'Case file changed during index scan')
            self.domains = dict(domains)
        except BaseException:
            self.handle.close()
            raise

    def get(self, case_id):
        require(stat(self.path) == self.pin, 'Case store changed')
        self.handle.seek(self.offsets[case_id])
        case = json.loads(self.handle.readline())
        require(case['case_id'] == case_id and check_case(case, self.split) == self.meta[case_id],
                'Offset case identity changed')
        return case

    def __iter__(self):
        for cid in self.offsets:
            yield self.get(cid)

    def close(self):
        self.handle.close()


class Exposure:
    """Validate exact stream + frozen observations before any model is loaded."""
    def __init__(self, root, seed, store):
        self.path = Path(root) / 'data' / f'exposure-seed{seed}.jsonl'
        self.schedule_path = Path(root) / 'data' / f'schedule-seed{seed}.json'
        self.pin, self.schedule_pin = stat(self.path), stat(self.schedule_path)
        self.schedule = schedule = read_json(self.schedule_path)
        require(schedule['seed'] == seed and schedule['data_seed'] == 20261112 and
                schedule['logical_pair_token_ceiling'] == 100_000_000 and not schedule['incomplete_reasons'] and
                schedule['unknown_forward_rule'] == 'skip-all-unknown-fields-v1' and
                schedule['same_physical_order_for_BCE_and_Field_CE'] is True,
                'Exposure schedule contract differs (schedule DATA_READY false is not dataset readiness)')
        self.macro_offsets = []
        repeated, domains, domain_cases = Counter(), Counter(), Counter()
        nodes = [{'cap': cap, 'logical_pair_tokens': 0, 'cases': 0, 'macro_updates': 0}
                 for cap in (10_000_000, 30_000_000, 100_000_000)]
        total = count = 0
        with self.path.open('rb') as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if count % 16 == 0:
                    self.macro_offsets.append(offset)
                row = json.loads(line)
                ref = store.meta[row['case_id']]
                require(row['exposure_index'] == count and row['macro_index'] == count // 16 and
                        row['occurrence'] == repeated[row['case_id']] and row['logical_pair_tokens'] == ref['tokens'],
                        'Exposure index/order/repetition/tokens differ')
                repeated[row['case_id']] += 1
                domains[ref['domain']] += ref['tokens']
                domain_cases[ref['domain']] += 1
                total += ref['tokens']
                count += 1
                if count % 16 == 0:
                    for node in nodes:
                        if total <= node['cap']:
                            node.update(logical_pair_tokens=total, cases=count, macro_updates=count // 16)
        require(count > 0 and count % 16 == 0 and total <= 100_000_000 and
                all(0 < domains[d] <= 20_000_000 for d in DOMAINS), 'Exposure ceiling/macro/domain violation')
        require(schedule['cases'] == count and schedule['macro_updates'] == count // 16 and
                schedule['logical_pair_tokens'] == total and schedule['observations'] == nodes and
                schedule['case_exposure_counts'] == dict(repeated) and
                schedule['domain_logical_pair_tokens'] == dict(domains) and
                schedule['domain_case_exposures'] == dict(domain_cases) and
                schedule['all_field_forward_tokens_if_one_pass'] == total and
                schedule['all_field_forward_tokens_if_two_pass'] == 2 * total and
                schedule['unused_domain_quota'] == {d: 20_000_000 - domains[d] for d in DOMAINS},
                'Frozen exposure readback differs')
        require(len({n['macro_updates'] for n in nodes}) == 3 and all(n['cases'] > 0 for n in nodes),
                'Three nonempty distinct observation prefixes required')
        require(stat(self.path) == self.pin and stat(self.schedule_path) == self.schedule_pin, 'Exposure changed')
        self.handle = self.path.open('rb')

    def macro(self, index):
        require(stat(self.path) == self.pin, 'Exposure stream changed')
        self.handle.seek(self.macro_offsets[index])
        rows = [json.loads(self.handle.readline()) for _ in range(16)]
        require(all(r['macro_index'] == index and r['exposure_index'] == 16 * index + j
                    for j, r in enumerate(rows)), 'Macro cursor mismatch')
        return rows

    def prefix_tokens(self, macros):
        # Used only for own-resume validation; no model/corpus digest.
        return sum(r['logical_pair_tokens'] for i in range(macros) for r in self.macro(i))

    def close(self):
        self.handle.close()


@dataclass(frozen=True)
class Pair:
    slot: int
    case_index: int
    field_index: int
    candidate_index: int
    prefix: list
    suffix: list

    @property
    def length(self):
        return len(self.prefix) + len(self.suffix)

    def row(self):
        return self.prefix + self.suffix


def slots(cases, *, dev=False):
    pairs, fields = [], []
    for ci, case in enumerate(cases):
        known = sum(f['known'] for f in case['fields'])
        for fi, field in enumerate(case['fields']):
            if not field['known']:
                continue
            if dev and field['static_unsupported_reason']:
                continue  # Entire overlength field unsupported, never candidate deletion.
            start = len(pairs)
            for ki, suffix in enumerate(field['suffix_ids']):
                pairs.append(Pair(len(pairs), ci, fi, ki, field['prefix_ids'], suffix))
            fields.append({'case_index': ci, 'field_index': fi, 'start': start, 'end': len(pairs),
                           'known_fields_in_case': known, 'target': field['target'],
                           'zero_K1': bool(field.get('zero_field_ce_gradient_K1'))})
    return pairs, fields


def micro_plan(pairs, *, dev=False):
    """Same stable sorted composition for dense/packed; no production realT claim."""
    ordered = sorted(pairs, key=lambda p: p.length)  # Python sort preserves slot tie order.
    result, current = [], []
    for pair in ordered:
        width = (pair.length + 7) // 8 * 8
        require(pair.length <= (40960 if dev else 8192), 'No truncation allowed')
        # Dev's complete single long pair can exceed the train micro budget.
        if current and (len(current) == 32 or width * (len(current) + 1) > 16384):
            result.append(current)
            current = []
        current.append(pair)
    if current:
        result.append(current)
    for micro in result:
        cost = ((max(p.length for p in micro) + 7) // 8 * 8) * len(micro)
        require(len(micro) <= 32 and (cost <= 16384 or (dev and len(micro) == 1)), 'Bad common micro')
    return result


def slot_manifest(pairs, cases, exposures=None):
    return [{'slot': p.slot, 'case_index': p.case_index, 'case_id': cases[p.case_index]['case_id'],
             'field_index': p.field_index, 'candidate_index': p.candidate_index,
             'coordinate': cases[p.case_index]['fields'][p.field_index]['coordinate'],
             'exposure_index': None if exposures is None else exposures[p.case_index]['exposure_index'],
             'pair_tokens': p.length, 'prefix_tokens': len(p.prefix)} for p in pairs]
