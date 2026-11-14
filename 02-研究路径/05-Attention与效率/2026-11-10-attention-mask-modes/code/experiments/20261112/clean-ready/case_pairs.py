"""Pure CPU draft case, pair and exposure functions; NO model/training entrypoint.

Call only after a reviewed full-coordinate projection and semantic receipt.
These functions deliberately do not issue DATA_READY or load checkpoints.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import numbers
from pathlib import Path
import random

from semantic_taint import (DOMAINS, HISTORY, PUBLIC, FORBIDDEN, NATIVE_DEV,
                            INCOMPLETE, UNKNOWN, dump, family, owned, require,
                            state_text)

TYPED_COLUMNS = ('state', 'kind', 'id', 'example_id', 'options', 'target', 'question', 'source',
                 'variant', 'split', 'group_id', 'question_id', 'license', 'license_use')


def typed_rows(paths):
    """Actual original typed columns, source-order coordinates, no dedup pool.

    Import is deferred so inspection/synthetic checks need only stdlib. The
    caller must bind all 17 paths to the existing audit before iterating.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    for name in paths:
        path = Path(name)
        parquet = pq.ParquetFile(path)
        require(set(TYPED_COLUMNS) - {'example_id'} <= set(parquet.schema_arrow.names), 'typed field schema mismatch')
        columns = [name for name in TYPED_COLUMNS if name in parquet.schema_arrow.names]
        index = 0
        for batch in parquet.iter_batches(batch_size=512, columns=columns, use_threads=False):
            for raw in batch.to_pylist():
                yield {'file': str(path), 'row_index': index}, raw
                index += 1
        require(index == parquet.metadata.num_rows, 'incomplete typed file projection')


def probability(value):
    return not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(value) and 0 <= value <= 1


def raw_storage(value):
    """Lossless nonfinite tags for invalid labels; never a training target.

    Strict JSON cannot represent NaN/Infinity. Original values remain available
    at the immutable native coordinate; tagged diagnostics do not invent gold.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {'__native_nonfinite_float__': 'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'}
    if isinstance(value, list):
        return [raw_storage(v) for v in value]
    if isinstance(value, dict):
        return {k: raw_storage(v) for k, v in value.items()}
    return value


def normalized_field(raw, coordinate, domain, noul_contract=None):
    """Unknown score/illegal targets stay in the case with full raw criteria.

    Only native choice distributions and explicitly native single-probability
    noul receive supervision. No normalization, class filtering or pseudo-gold.
    """
    require(domain in DOMAINS, 'domain is not from the frozen five-domain registry')
    source = raw['source']
    source_valid = isinstance(source, str) and bool(source.strip())
    value = {'coordinate': coordinate, 'source': source, 'family': family(source) if source_valid else '',
             'domain': domain, 'native_kind': raw['kind'], 'state': raw['state'],
             'question': raw['question'], 'raw_options': raw_storage(raw['options']),
             'raw_target': raw_storage(raw['target']), 'raw_storage_codec': 'native-nonfinite-tag-v1',
             'native_metadata': {key: raw_storage(raw.get(key)) for key in TYPED_COLUMNS
                                 if key not in ('state', 'question', 'options', 'target', 'source', 'kind')},
             'known': False, 'options': raw_storage(raw['options']), 'target': None}
    value['native_metadata']['example_id_column_present'] = 'example_id' in raw
    value['supervision_contract_id'] = None
    options, target = raw['options'], raw['target']
    reason = 'unsupported_or_unproven_native_score_semantics'
    if raw['kind'] == 'noul':
        direction_known = bool(noul_contract and noul_contract.get('reviewed') is True and
                               noul_contract.get('q_semantics') == 'native_yes_probability' and
                               noul_contract.get('evidence_files') and noul_contract.get('contract_id'))
        if not direction_known:
            reason = 'noul_direction_importer_contract_missing'
        elif options == [] and isinstance(target, list) and len(target) == 1 and probability(target[0]):
            q = float(target[0])
            value.update(known=True, options=['Yes.', 'No.'], target=[q, 1.0 - q],
                         option_keys=['true', 'false'], supervision_contract_id=noul_contract['contract_id'])
        else:
            reason = 'noul_not_native_single_probability'
    elif raw['kind'] == 'choice':
        valid_options = isinstance(options, list) and len(options) >= 1 and all(isinstance(x, str) for x in options)
        if valid_options and isinstance(target, list) and len(target) == len(options) and all(map(probability, target)) and math.isclose(math.fsum(target), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            value.update(known=True, target=list(map(float, target)), option_keys=list(range(len(options))))
        else:
            reason = 'invalid_choice_distribution_or_complete_options'
    # Invalid input shape is a case-level admission error, never field deletion.
    value['input_shape_valid'] = source_valid and isinstance(raw['question'], str) and isinstance(raw['state'], (str, dict, list))
    if not value['known']:
        value['unknown_reason'] = reason
    value['onehot'] = bool(value['known'] and sum(t == 1.0 for t in value['target']) == 1
                           and all(t in (0.0, 1.0) for t in value['target']))
    value['zero_field_ce_gradient_K1'] = bool(value['known'] and raw['kind'] == 'choice' and len(options) == 1)
    return value


def make_case(component, bits, fields, expected_coordinates, license_decisions, source_decisions, split,
              dev_eligibility=None):
    """Coordinates must come from the COMPLETE metadata census, not a sample.

    expected_coordinates is a new derived contract, not a field in old SQLite.
    All sibling rows must be present, including unknown fields and other views.
    """
    observed = [dump(field['coordinate']) for field in fields]
    expected = [dump(value) for value in expected_coordinates]
    require(len(set(expected)) == len(expected) and len(set(observed)) == len(observed), 'duplicate field coordinate')
    require(set(observed) == set(expected), 'case is missing sibling rows or contains unbound fields')
    require(split in ('train', 'validation') and fields, 'invalid or empty case')
    known = Counter(f['domain'] for f in fields if f['known'])
    reasons = []
    forbidden = PUBLIC | FORBIDDEN | INCOMPLETE | UNKNOWN | (NATIVE_DEV if split == 'train' else HISTORY)
    if bits & forbidden:
        reasons.append('excluded_membership_flags:' + str(bits & forbidden))
    if not known:
        reasons.append('no_known_supervision')
    if split == 'validation' and (dev_eligibility is None or
            not all(dev_eligibility.get(dump(f['coordinate']), False) for f in fields)):
        reasons.append('not_all_native_dev_siblings_original_roster_eligible')
    for field in fields:
        native = field['native_metadata']
        source = field['source']
        source_label = source if isinstance(source, str) and source else '<missing-source>'
        declared_split = str(native['split']).strip().casefold()
        if declared_split not in (('train', 'training') if split == 'train' else ('dev', 'valid', 'validation')):
            reasons.append('non_native_' + split + '_sibling')
        # Exact reviewed license/license_use pair. No guess from a substring
        # such as "research" and no automatic legal claim about mixed licenses.
        key = dump([source, native['license'], native['license_use']])
        decision = license_decisions.get(key, {})
        raw_license = native['license']
        unresolved = {'', 'unspecified', 'unknown', 'other', 'none', 'null', 'n/a',
                      'custom', 'request', 'custom-license', 'request-only',
                      'requires permission', 'permission required', '需另行申请'}
        identifiable = isinstance(raw_license, str) and raw_license.strip().casefold() not in unresolved
        if not identifiable or decision.get('allowed_private_research') is not True or not decision.get('evidence'):
            reasons.append('license_unresolved:' + source_label)
        review = source_decisions.get(source, {})
        if review.get('provenance_complete') is not True or review.get('case_complete') is not True or not review.get('evidence'):
            reasons.append('source_or_complete_case_unverified:' + source_label)
        if not state_text(field['state']) or (field['known'] and not field['input_shape_valid']):
            reasons.append('invalid_original_input_shape')
    primary = max(DOMAINS, key=lambda domain: (known[domain], -DOMAINS.index(domain)))
    families = sorted({f['family'] for f in fields})
    # Mixed genuine families need a reviewed sampling rule, not an arbitrary
    # selected representative. Keep their tuple for census but quarantine now.
    if len(families) != 1:
        reasons.append('multi_family_sampling_assignment_unreviewed')
    return {'case_id': f'clean-semantic-{component}', 'component': component,
            'component_flags': bits, 'primary_domain': primary, 'families': families,
            'family_stratum': dump(families), 'fields': fields,
            'known_mask': [f['known'] for f in fields],
            'complete_onehot_case_exact_supported': all(f['known'] and f['onehot'] for f in fields),
            'rejection_reasons': sorted(set(reasons)), 'split': split,
            'validation_strength': 'procedural_tools_not_BFCL' if primary == 'tools' else 'native_registry'}


def round_robin_families(cases, rng):
    buckets = defaultdict(list)
    for case in sorted(cases, key=lambda x: x['case_id']):
        buckets[case['family_stratum']].append(case)
    for values in buckets.values():
        rng.shuffle(values)
    order = sorted(buckets)
    rng.shuffle(order)
    result = []
    while order:
        remaining = []
        for name in order:
            result.append(buckets[name].pop())
            if buckets[name]:
                remaining.append(name)
        order = remaining
    return result


def freeze_rosters(cases, candidate_limit=5000):
    """Legacy v1 pure helper; NOT the current readiness/entrypoint contract.

    integration_driver.freeze_output_rosters implements independent train/dev
    readiness and the mandatory2500 compiled traincases/domain threshold.
    """
    require(2500 <= candidate_limit <= 5000, 'candidate cap must be predeclared in2500..5000')
    require(len({c['component'] for c in cases}) == len(cases), 'case components were not globally assembled')
    outputs = {'select': [], 'confirm': [], 'train_candidates': []}
    blockers = []
    for i, domain in enumerate(DOMAINS):
        legal = [c for c in cases if not c['rejection_reasons'] and c['primary_domain'] == domain]
        dev = round_robin_families([c for c in legal if c['split'] == 'validation'], random.Random(20261112 + i))
        if len(dev) < 125:
            blockers.append(f'{domain}: eligible dev components {len(dev)} <125; quotas not reduced')
            continue
        # Balance EACH roster independently. A fixed3/5 selector over an
        # interleaved5-family sequence would segregate whole families!
        select = dev[:75]
        selected = {case['component'] for case in select}
        remaining = [case for case in dev if case['component'] not in selected]
        confirm = round_robin_families(remaining, random.Random(20261112 + 50 + i))[:50]
        require(len(select) == 75 and len(confirm) == 50, 'internal fixed-quota error')
        outputs['select'].extend(select)
        outputs['confirm'].extend(confirm)
        train = round_robin_families([c for c in legal if c['split'] == 'train'], random.Random(20261112 + 100 + i))
        outputs['train_candidates'].extend(train[:candidate_limit])
        if not train:
            blockers.append(domain + ': no legal training components')
    sets = [set(c['component'] for c in outputs[key]) for key in outputs]
    require(all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:]), 'component crossed frozen rosters')
    return outputs, {'incomplete_reasons': blockers, 'DATA_READY': False,
                     'data_seed': 20261112, 'train_candidate_cap_per_domain': candidate_limit}


def compile_case(case, tokenizer, helper, check=lambda: None):
    """Original KD renderer/helper injected read-only, no teacher/model loaded.

    All fields survive. Unknown fields are explicitly never forwarded in BOTH
    arms; their empty/unknown criteria are not a training-case rejection.
    Any KNOWN oversize training field rejects the ENTIRE case.
    """
    require(callable(helper.render_pair) and callable(helper.criteria_context), 'original pair helper API missing')
    eos = tokenizer.eos_token_id
    require(type(eos) is int and eos >= 0, 'tokenizer has no actual EOS')
    compiled, max_pair, logical, forward_once = [], 0, 0, 0
    errors = []
    for field in case['fields']:
        check()
        value = dict(field)
        if not field['known']:
            value.update(prefix_ids=[], suffix_ids=[], pair_lengths=[],
                         forward_policy='masked_unknown_no_forward',
                         static_unsupported_reason='unknown_supervision_masked_no_forward')
            compiled.append(value)
            continue
        options = field['options']
        if not isinstance(options, list) or not options or not all(isinstance(x, str) for x in options):
            value.update(static_unsupported_reason='unrenderable_native_criteria', pair_lengths=[])
            errors.append('unrenderable_native_criteria')
            compiled.append(value)
            continue
        question = state_text(field['question'])
        context = helper.criteria_context(question, options)
        prefix, _ = helper.render_pair(field['state'], question, options[0], context)
        def encode(text):
            check()
            result = list(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])
            check()
            return result
        prefix_ids = encode(prefix)
        suffixes = [encode(state_text(option)) + [eos] for option in options]
        require(all(type(t) is int and t >= 0 for t in prefix_ids) and
                all(type(t) is int and t >= 0 for suffix in suffixes for t in suffix), 'invalid token IDs')
        lengths = [len(prefix_ids) + len(suffix) for suffix in suffixes]
        maximum = max(lengths)
        max_pair = max(max_pair, maximum)
        count = sum(lengths)
        forward_once += count
        if field['known']:
            logical += count
        value.update(prefix_ids=prefix_ids, suffix_ids=suffixes, pair_lengths=lengths,
                     forward_policy='known_complete_candidates',
                     static_unsupported_reason='pair_context>40960' if maximum > 40960 else None)
        compiled.append(value)
    reasons = list(case['rejection_reasons'])
    if case['split'] == 'train' and (errors or max_pair > 8192):
        reasons.append('reject_whole_case_unrenderable_or_pair_context>8192')
    return {**case, 'fields': compiled, 'rejection_reasons': sorted(set(reasons)),
            'max_pair_tokens': max_pair, 'logical_supervised_pair_tokens': logical,
            'all_field_pair_tokens_once': forward_once,
            'scheduled_forward_pair_tokens_once': forward_once,
            'unknown_forward_rule': 'skip-all-unknown-fields-v1',
            'unknown_fields_no_forward': sum(not f['known'] for f in compiled),
            'unknown_field_forward_tokens_once': 0,
            'zero_field_ce_gradient_K1_fields': sum(f.get('zero_field_ce_gradient_K1', False) for f in compiled),
            'validation_known_field_static_coverage': {
                'known': sum(f['known'] for f in compiled),
                'supported_known': sum(f['known'] and not f.get('static_unsupported_reason') for f in compiled)},
            'complete_case_exact_static_supported': bool(case.get('complete_onehot_case_exact_supported'))
                and all(not f.get('static_unsupported_reason') for f in compiled),
            'layout': 'candidate-pair-v1-prefix-and-candidate-separately-tokenized-eos-last',
            'truncated': False}


def write_exposure(cases, path, seed=42, maximum=100_000_000, check=lambda: None):
    """Freeze one case-balanced, sqrt-family-weighted logical exposure stream.

    Both losses consume this same stream. Tail handling never redraws a shorter
    case to fit. Observation nodes are largest complete16-case prefixes<=cap.
    This is not a training loop, and forward accounting is NOT a measurement.
    """
    require(seed in (42, 43) and maximum == 100_000_000, 'only frozen100M seed42/43 exposure contract')
    ids = [c['case_id'] for c in cases]
    require(len(set(ids)) == len(ids), 'duplicate exposure inventory cases')
    pools = {d: defaultdict(list) for d in DOMAINS}
    for case in sorted(cases, key=lambda c: c['case_id']):
        require(case['split'] == 'train' and not case['rejection_reasons'], 'illegal training case')
        require(case['max_pair_tokens'] <= 8192 and case['logical_supervised_pair_tokens'] > 0, 'whole training case length/token error')
        pools[case['primary_domain']][case['family_stratum']].append(case)
    require(all(pools.values()), 'five-domain training inventory incomplete')
    # The multiset is data-seed frozen ONCE. seed42/43 affect order only, never
    # resample a different corpus or choose different length-budget tails.
    rng = random.Random(20261112)
    quotas = {d: maximum // 5 for d in DOMAINS}
    tokens, repeats, emitted = Counter(), Counter(), Counter()
    nodes = [{'cap': cap, 'logical_pair_tokens': 0, 'cases': 0, 'macro_updates': 0}
             for cap in (10_000_000, 30_000_000, 100_000_000)]
    total = count = all_once = 0
    tail, multiset = [], []
    while True:
        check()
        pending, reserved = [], Counter(tokens)
        for _ in range(16):
            domain = min(DOMAINS, key=lambda d: (reserved[d] / quotas[d], DOMAINS.index(d)))
            names = sorted(pools[domain])
            name = rng.choices(names, weights=[math.sqrt(len(pools[domain][n])) for n in names], k=1)[0]
            case = rng.choice(pools[domain][name])
            length = case['logical_supervised_pair_tokens']
            if reserved[domain] + length > quotas[domain]:
                tail = [c['case_id'] for c in pending] + [case['case_id']]
                break
            pending.append(case)
            reserved[domain] += length
        if len(pending) != 16:
            break
        multiset.extend(pending)
        tokens = reserved
        require(len(multiset) <= 2_000_000, 'bounded exposure-reference memory allowance exceeded; no partial admission')
    random.Random(seed).shuffle(multiset)
    tokens = Counter()
    with owned(path).open('x', encoding='utf-8') as handle:
        for case in multiset:
            if count % 512 == 0:
                check()
            domain, cid = case['primary_domain'], case['case_id']
            seen = repeats[cid]
            handle.write(dump({'exposure_index': count, 'macro_index': count // 16,
                               'case_id': cid, 'occurrence': seen,
                               'logical_pair_tokens': case['logical_supervised_pair_tokens']}) + '\n')
            repeats[cid] += 1
            emitted[domain] += 1
            tokens[domain] += case['logical_supervised_pair_tokens']
            total += case['logical_supervised_pair_tokens']
            all_once += case['all_field_pair_tokens_once']
            count += 1
            if count % 16 == 0:
                for node in nodes:
                    if total <= node['cap']:
                        node.update(logical_pair_tokens=total, cases=count, macro_updates=count // 16)
        handle.flush()
        __import__('os').fsync(handle.fileno())
    incomplete = []
    if not count or any(emitted[d] == 0 for d in DOMAINS):
        incomplete.append('missing_complete_macro_or_domain_exposure')
    if not all(n['cases'] > 0 for n in nodes) or len({n['cases'] for n in nodes}) != 3:
        incomplete.append('10M/30M/100M_observation_prefixes_not_distinct')
    return {'DATA_READY': False, 'seed': seed, 'data_seed': 20261112,
            'incomplete_reasons': incomplete, 'logical_pair_token_ceiling': maximum,
            'logical_pair_tokens': total, 'cases': count, 'macro_updates': count // 16,
            'observations': nodes, 'domain_logical_pair_tokens': dict(tokens),
            'domain_case_exposures': dict(emitted), 'case_exposure_counts': dict(repeats),
            'unused_domain_quota': {d: quotas[d] - tokens[d] for d in DOMAINS},
            'discarded_incomplete_macro_case_ids': tail,
            'all_field_forward_tokens_if_one_pass': all_once,
            'all_field_forward_tokens_if_two_pass': 2 * all_once,
            'actual_forward_token_measurement': None,
            'unknown_forward_rule': 'skip-all-unknown-fields-v1',
            'forward_contract': 'trainer must separately measure actual repeated/padded/packed forward tokens',
            'loss_contract': 'case-balanced; all known fields; unknown fields never fabricated as negatives',
            'sampling': 'equal logical domain quotas; family weight sqrt(eligible_cases); uniform case within family',
            'same_physical_order_for_BCE_and_Field_CE': True}


def summarize(cases):
    """Private aggregate census; no raw text/targets/token arrays in this view."""
    result = Counter()
    domains, field_domains, sources, families, kinds = (Counter() for _ in range(5))
    k_hist, lengths, rejections, unknown, coverage = (Counter() for _ in range(5))
    for case in cases:
        result['cases'] += 1
        result['fields'] += len(case['fields'])
        result['known_fields'] += sum(case['known_mask'])
        result['unknown_fields'] += len(case['known_mask']) - sum(case['known_mask'])
        domains[case['primary_domain']] += 1
        families[case['family_stratum']] += 1
        rejections.update(case['rejection_reasons'])
        for field in case['fields']:
            field_domains[field['domain']] += 1
            sources[field['source']] += 1
            kinds[field['native_kind']] += 1
            options = field['options']
            k_hist[str(len(options)) if isinstance(options, list) else 'unrenderable'] += 1
            lengths.update(map(str, field.get('pair_lengths', [])))
            if field['known']:
                coverage['known'] += 1
                if 'static_unsupported_reason' in field and not field['static_unsupported_reason']:
                    coverage['supported_known'] += 1
            else:
                unknown[field['unknown_reason']] += 1
    return {'counts': dict(result), 'primary_domain_cases': dict(domains),
            'field_domains': dict(field_domains), 'source_fields': dict(sources),
            'family_cases': dict(families), 'native_kind_fields': dict(kinds),
            'complete_option_K_histogram': dict(k_hist), 'pair_length_histogram': dict(lengths),
            'nonexclusive_case_rejections': dict(rejections), 'unknown_supervision_reasons': dict(unknown),
            'static_known_field_coverage': dict(coverage)}
