"""Mandatory actual-data readback, not an extra unit/smoke test entrypoint.

Re-read each retained original field, reconstruct targets/anchors, retokenize
known complete pairs and check every output/exposure. No model/teacher/cache.
"""
from __future__ import annotations

from collections import Counter
import json
import sqlite3

from semantic_taint import DOMAINS, UNKNOWN, INCOMPLETE, dump, require, state_text, stat
from case_pairs import normalized_field, compile_case, TYPED_COLUMNS
from input_projector import native_projection,ExclusionSeeds
from prepare_runtime import read_json

FIELD_COMPILED_KEYS = {'prefix_ids', 'suffix_ids', 'pair_lengths', 'forward_policy', 'static_unsupported_reason'}


def verify_prepared(stage, census, reviews, db, membership, tokenizer, helper, cfg, semantic, readiness, check):
    # Imported at call time to avoid an entrypoint/module cycle on import.
    from integration_driver import assemble_component
    require(semantic['public']['requests'] == 140620 and semantic['public']['text_entries'] == 5854755 and
            semantic['native_dev']['native_dev_rows'] == 15550 and semantic['old_graph']['old_graph_rebuilt'] is False,
            'complete declared exclusion passes missing')
    matches = semantic['seed_match_counts']
    require(semantic['new_negative_input_passes_complete'] is True and
            matches.get('public:representations_received') == 5854755 and
            matches.get('native_dev_state:representations_received') == 15550 and
            matches.get('native_dev_question:representations_received') == 15550 and
            matches.get('old_graph_root_taint_superset:representations_received', 0) == semantic['old_graph']['root_flagged_state_keys'],
            'old inventory counts cannot substitute for actual new negative conversion traversal')
    for origin in ('public', 'native_dev_state', 'native_dev_question', 'old_graph_root_taint_superset'):
        conversions = semantic['new_conversion_counts']
        require(conversions.get(origin + ':full_provided_representation_kept', 0) +
                conversions.get(origin + ':empty_or_generic_representation_skipped', 0) ==
                matches.get(origin + ':representations_received', 0), 'new conversion pass skipped unclassified provided inputs')
        require(conversions.get(origin + ':known_tool_wrapper_seen', 0) ==
                conversions.get(origin + ':known_tool_wrapper_unwrapped', 0) +
                conversions.get(origin + ':known_tool_wrapper_invalid_or_generic_request', 0), 'wrapper conversion failure classification incomplete')
    require(semantic['old_graph']['complete_state_key_cursor_exhausted'] is True, 'new negative old-state conversion pass incomplete')
    extra_origin='old_graph_unflagged_native_closure'
    conversions=semantic['new_conversion_counts']
    extra=matches.get(extra_origin+':representations_received',0)
    require(extra==semantic['old_graph']['unflagged_state_keys_checked_for_native_closure'] and
            extra+semantic['old_graph']['root_flagged_state_keys']==semantic['old_graph']['state_keys_read'] and
            conversions.get(extra_origin+':full_provided_representation_kept',0)+
            conversions.get(extra_origin+':empty_or_generic_representation_skipped',0)==extra,
            'unflagged old native states omitted from recognized semantic closure')
    # Verify closure using the actual old-state key and full census, not just
    # re-reading the same sampled positive component graph.
    verifier=object.__new__(ExclusionSeeds)
    verifier.profiles=[p for review in reviews.sources.values() for p in review['input_profiles'] if p['mode']!='state-text-v1']
    verifier.conversions=Counter()
    closure_count=closure_veto=0
    for value,key,node,root,bits,member,first in membership.db.execute('SELECT raw_state,anchor,state_node,old_root,old_bits,projected_member,candidate_root FROM native_closure'):
        check()
        actual=census.audit.db.execute('SELECT node FROM keys WHERE kind=? AND value=?',('state',value)).fetchone()
        require(actual is not None and actual[0]==node and census.audit.root(node)==root and census.audit.bits(root)==bits,
                'native semantic closure old-state identity/taint changed')
        require(key in verifier.representations(value,'closure_readback') and
                census.db.execute('SELECT 1 FROM rows WHERE old_root=? LIMIT 1',(root,)).fetchone() is not None,
                'closure evidence is not a recognized provided-state alias of a native component')
        projected=membership.db.execute('SELECT 1 FROM members WHERE value=? AND root=?',(key,root)).fetchone() is not None
        require(bool(member)==projected,'closure positive projection membership differs')
        if projected:
            require(db.execute('SELECT count(*) FROM native_fields WHERE old_root=?',(root,)).fetchone()[0]==
                    census.db.execute('SELECT count(*) FROM rows WHERE old_root=?',(root,)).fetchone()[0],
                    'recognized projected native sibling coordinates are incomplete')
        else:
            flags=membership.db.execute('SELECT bits FROM anchors WHERE value=?',(key,)).fetchone()[0]
            require(flags & (UNKNOWN|INCOMPLETE)==(UNKNOWN|INCOMPLETE) and
                    membership.root(first)[1] & (UNKNOWN|INCOMPLETE)==(UNKNOWN|INCOMPLETE),
                    'unrestored native semantic sibling did not quarantine its complete candidate component')
            require(db.execute('SELECT 1 FROM outputs o JOIN components c ON o.component=c.component WHERE c.old_root=? LIMIT 1',(first,)).fetchone() is None,
                    'incomplete recognized semantic component was emitted')
            closure_veto+=1
        closure_count+=1
    require(closure_count==semantic['old_graph']['recognized_native_closure_pairs']==
            matches.get('native_closure:matched_native_state_anchor_pairs',0),
            'recognized native closure traversal/evidence count differs')
    path = stage / 'readback.sqlite3'
    require(not path.exists(), 'previous readback artifacts preserved')
    vdb = sqlite3.connect(path)
    vdb.row_factory = sqlite3.Row
    vdb.executescript('''
        PRAGMA journal_mode=DELETE; PRAGMA temp_store=MEMORY; PRAGMA cache_size=-32768;
        CREATE TABLE expected(file_id INTEGER,row_index INTEGER,component INTEGER,split TEXT,
                              normalized TEXT,compiled TEXT,seen INTEGER DEFAULT 0,
                              PRIMARY KEY(file_id,row_index)) WITHOUT ROWID;
        CREATE TABLE pairs(sequence TEXT PRIMARY KEY,split TEXT) WITHOUT ROWID;
    ''')
    cases, old_roots, composition = set(), set(), Counter()
    train_refs, output_fields, known_fields, masked_fields, k1_fields = {}, 0, 0, 0, 0
    try:
        for split in ('select', 'confirm', 'train'):
            with (stage / 'data' / (split + '.jsonl')).open(encoding='utf-8') as handle:
                for line in handle:
                    check()
                    case = json.loads(line)
                    component = case['component']
                    require(component not in cases, 'semantic component crossed/duplicated output rosters')
                    cases.add(component)
                    actual_roots = [r[0] for r in db.execute('SELECT old_root FROM components WHERE component=? ORDER BY old_root', (component,))]
                    require(actual_roots and actual_roots == case['old_roots'] and not old_roots.intersection(actual_roots),
                            'output old-root closure incomplete or shared')
                    old_roots.update(actual_roots)
                    for root in actual_roots:
                        require(membership.root(root) == (component, case['component_flags']), 'output semantic final root/taint differs')
                    expected = assemble_component(component, actual_roots, census, reviews, db, membership)
                    require(not expected['rejection_reasons'] and not case['rejection_reasons'], 'output has an inadmissible complete case')
                    for key, value in expected.items():
                        if key != 'fields':
                            require(case[key] == value, 'output raw case metadata changed: ' + key)
                    require(case['split'] == ('train' if split == 'train' else 'validation') and
                            case['case_id'] == f'clean-semantic-{component}' and case['truncated'] is False and
                            case['unknown_forward_rule'] == 'skip-all-unknown-fields-v1', 'output case/forward contract differs')
                    reference = {dump(f['coordinate']): f for f in expected['fields']}
                    require(len(case['fields']) == len(reference), 'output dropped a sibling field')
                    logical, maximum, unknown_count, k1_count, supported_known = 0, 0, 0, 0, 0
                    seen_coordinates = set()
                    for field in case['fields']:
                        coordinate = field['coordinate']
                        key = dump(coordinate)
                        require(key in reference and key not in seen_coordinates, 'output field coordinate missing/duplicated')
                        seen_coordinates.add(key)
                        raw = {k: v for k, v in field.items() if k not in FIELD_COMPILED_KEYS}
                        require(raw == reference[key], 'output raw field/options/target/mask changed')
                        fid = census.file_by_path[coordinate['file']]
                        meta = census.meta(fid, coordinate['row_index'])
                        require(meta['old_root'] in actual_roots, 'field belongs to another original case')
                        if split != 'train':
                            require(census.dev_eligible(coordinate) and bool(state_text(field['state'])), 'dev flags-only/empty-identity bypass')
                        if field['known']:
                            require(len(field['suffix_ids']) == len(field['options']) == len(field['target']) >= 1,
                                    'complete known criteria were removed')
                            require(field['forward_policy'] == 'known_complete_candidates' and field['prefix_ids'], 'known prefix/forward missing')
                            lengths = []
                            for suffix in field['suffix_ids']:
                                require(suffix and suffix[-1] == tokenizer.eos_token_id, 'actual EOS is not final')
                                ids = field['prefix_ids'] + suffix
                                require(all(type(t) is int and t >= 0 for t in ids), 'noninteger pair token ID')
                                lengths.append(len(ids))
                                sequence = dump(ids)  # Full exact sequence, not a hash/digest.
                                previous = vdb.execute('SELECT split FROM pairs WHERE sequence=?', (sequence,)).fetchone()
                                require(previous is None or previous[0] == split, 'exact complete pair input crossed train/select/confirm')
                                if previous is None:
                                    vdb.execute('INSERT INTO pairs VALUES(?,?)', (sequence, split))
                            require(field['pair_lengths'] == lengths, 'pair length accounting differs')
                            maximum = max(maximum, max(lengths))
                            logical += sum(lengths)
                            unsupported = 'pair_context>40960' if max(lengths) > 40960 else None
                            require(field['static_unsupported_reason'] == unsupported, 'static validation context coverage differs')
                            supported_known += unsupported is None
                            known_fields += 1
                            k1_count += field['zero_field_ce_gradient_K1']
                            if split == 'train':
                                require(max(lengths) <= 8192, 'known overlength field leaked into training case')
                        else:
                            require(field['prefix_ids'] == field['suffix_ids'] == field['pair_lengths'] == [] and
                                    field['target'] is None and field['forward_policy'] == 'masked_unknown_no_forward' and
                                    field['static_unsupported_reason'] == 'unknown_supervision_masked_no_forward',
                                    'unknown score was dropped, forwarded inconsistently or fabricated as a negative')
                            unknown_count += 1
                            masked_fields += 1
                        vdb.execute('INSERT INTO expected(file_id,row_index,component,split,normalized,compiled) VALUES(?,?,?,?,?,?)',
                                    (fid, coordinate['row_index'], component, split, dump(raw), dump(field)))
                        output_fields += 1
                        if output_fields % 128 == 0:
                            check()
                    require(len(seen_coordinates) == len(reference) and logical > 0, 'incomplete or unsupervised whole case')
                    require(case['logical_supervised_pair_tokens'] == case['scheduled_forward_pair_tokens_once'] ==
                            case['all_field_pair_tokens_once'] == logical and case['max_pair_tokens'] == maximum and
                            case['unknown_fields_no_forward'] == unknown_count and case['unknown_field_forward_tokens_once'] == 0 and
                            case['zero_field_ce_gradient_K1_fields'] == k1_count, 'whole-case token/mask accounting differs')
                    require(case['validation_known_field_static_coverage'] ==
                            {'known': sum(f['known'] for f in case['fields']), 'supported_known': supported_known}, 'known field coverage differs')
                    exact = case['complete_onehot_case_exact_supported'] and supported_known == len(case['fields'])
                    require(case['complete_case_exact_static_supported'] == exact, 'unsupported/unknown field counted as complete case-exact')
                    composition[split + ':' + case['primary_domain']] += 1
                    k1_fields += k1_count
                    if split == 'train':
                        train_refs[case['case_id']] = {'tokens': logical, 'forward_once': logical,
                                                      'domain': case['primary_domain'], 'family': case['family_stratum']}
                    vdb.commit()
        for domain in DOMAINS:
            dev_counts = (composition['select:' + domain], composition['confirm:' + domain])
            require(dev_counts == (75, 50) if readiness['dev']['ready'] else dev_counts in ((0, 0), (75, 50)),
                    'dev shortage must be explicit, never a reduced75/50 quota')
            require(0 <= composition['train:' + domain] <= cfg['limits']['candidate_cases_per_domain'], 'train component cap exceeded')
            if readiness['train']['ready']:
                require(composition['train:' + domain] >= max(2500, cfg['min_train_components_per_domain']), 'under2500/domain cannot be train-ready')
        require(readiness['train']['ready'] == all(composition['train:' + d] >= cfg['min_train_components_per_domain'] for d in DOMAINS),
                'train readiness was not independently derived from actual compiled complete inventory')
        require(readiness['dev']['ready'] == all((composition['select:' + d], composition['confirm:' + d]) == (75, 50) for d in DOMAINS),
                'dev readiness was not independently derived from actual fixed quotas')
        require(len(cases) == db.execute('SELECT count(*) FROM outputs').fetchone()[0], 'output selection/readback component mismatch')
        # ACTUAL ORIGINAL SOURCE READBACK. This is stronger than comparing a
        # JSONL count to another derived count or trusting the first projector.
        selector = lambda fid: [r[0] for r in vdb.execute('SELECT row_index FROM expected WHERE file_id=? ORDER BY row_index', (fid,))]
        raw_count = 0
        for coordinate, raw, meta in census.iter_payload(selector, TYPED_COLUMNS, cfg['limits']['arrow_batch_rows']):
            expected = vdb.execute('SELECT * FROM expected WHERE file_id=? AND row_index=?', (meta['file_id'], meta['row_index'])).fetchone()
            require(expected is not None and not expected['seen'], 'raw readback duplicated/unselected coordinate')
            source = census.sources[meta['source_id']]
            anchors, reason, profile = native_projection({'state': raw['state'], 'question': raw['question']},
                                                          reviews.sources[meta['source_id']], {'variant': raw.get('variant')})
            contract = reviews.noul_contracts.get(profile.get('noul_contract_id')) if profile else None
            normalized = normalized_field(raw, coordinate, source['domain'], contract)
            normalized['reviewed_input_shape_id'] = profile['shape_id'] if profile else None
            require(dump(normalized) == expected['normalized'], 'retained original field/options/native target/direction contract differ on readback')
            first = db.execute('SELECT anchors FROM native_fields WHERE file_id=? AND row_index=?', (meta['file_id'], meta['row_index'])).fetchone()
            require(not reason and bool(state_text(raw['state'])) and dump(anchors) == first[0], 'actual raw input semantic projection changed')
            require(membership.root(meta['old_root'])[0] == expected['component'], 'raw field semantic component differs')
            whole = {'fields': [normalized], 'split': 'train' if expected['split'] == 'train' else 'validation', 'rejection_reasons': []}
            retokenized = compile_case(whole, tokenizer, helper, check)
            require(dump(retokenized['fields'][0]) == expected['compiled'], 'actual original input retokenization differs; no truncation/cache reuse allowed')
            vdb.execute('UPDATE expected SET seen=1 WHERE file_id=? AND row_index=?', (meta['file_id'], meta['row_index']))
            raw_count += 1
            if raw_count % 128 == 0:
                vdb.commit()
                check()
        require(raw_count == output_fields and vdb.execute('SELECT count(*) FROM expected WHERE seen!=1').fetchone()[0] == 0,
                'not every retained original field was directly re-read and retokenized')
        schedules, multisets = {}, {}
        for seed in ((42, 43) if readiness['train']['ready'] else ()):
            schedule = read_json(stage / 'data' / f'schedule-seed{seed}.json')
            require(schedule['seed'] == seed and schedule['data_seed'] == 20261112 and
                    schedule['logical_pair_token_ceiling'] == 100000000 and not schedule['incomplete_reasons'] and
                    schedule['unknown_forward_rule'] == 'skip-all-unknown-fields-v1' and
                    schedule['actual_forward_token_measurement'] is None, 'schedule contract differs')
            repeats, domain_tokens, domain_cases, family_cases = Counter(), Counter(), Counter(), Counter()
            nodes = [{'cap': cap, 'logical_pair_tokens': 0, 'cases': 0, 'macro_updates': 0} for cap in (10000000, 30000000, 100000000)]
            total = count = once = 0
            with (stage / 'data' / f'exposure-seed{seed}.jsonl').open() as handle:
                for line in handle:
                    row = json.loads(line)
                    require(row['case_id'] in train_refs, 'exposure references nontrain/unverified case')
                    ref = train_refs[row['case_id']]
                    require(row['exposure_index'] == count and row['macro_index'] == count // 16 and
                            row['occurrence'] == repeats[row['case_id']] and row['logical_pair_tokens'] == ref['tokens'], 'exposure identity/order/repetition/token record differs')
                    repeats[row['case_id']] += 1
                    domain_tokens[ref['domain']] += ref['tokens']
                    domain_cases[ref['domain']] += 1
                    family_cases[ref['family']] += 1
                    count += 1
                    total += ref['tokens']
                    once += ref['forward_once']
                    if count % 16 == 0:
                        for node in nodes:
                            if total <= node['cap']:
                                node.update(logical_pair_tokens=total, cases=count, macro_updates=count // 16)
                    if count % 512 == 0:
                        check()
            require(count > 0 and count % 16 == 0 and total <= 100000000 and all(0 < domain_tokens[d] <= 20000000 for d in DOMAINS), 'exposure budget/macro/domain contract violated')
            require(schedule['cases'] == count and schedule['macro_updates'] == count // 16 and
                    schedule['logical_pair_tokens'] == total and schedule['case_exposure_counts'] == dict(repeats) and
                    schedule['domain_logical_pair_tokens'] == dict(domain_tokens) and schedule['domain_case_exposures'] == dict(domain_cases) and
                    schedule['observations'] == nodes and schedule['all_field_forward_tokens_if_one_pass'] == once and
                    schedule['all_field_forward_tokens_if_two_pass'] == 2 * once, 'actual exposure readback disagrees with frozen schedule')
            require(all(n['cases'] for n in nodes) and len({n['cases'] for n in nodes}) == 3, 'predeclared observation prefixes not distinct')
            multisets[seed] = repeats
            schedules[str(seed)] = {'cases': count, 'logical_pair_tokens': total, 'observations': nodes,
                                    'family_case_exposures': dict(family_cases), 'static_two_pass_forward_tokens': 2 * once,
                                    'actual_forward_tokens': None}
        if readiness['train']['ready']:
            require(multisets[42] == multisets[43], 'seed42/43 changed frozen exposure multiset')
        else:
            require(not any((stage / 'data' / f'exposure-seed{seed}.jsonl').exists() for seed in (42, 43)),
                    'incomplete train inventory must not be repeated to manufacture100M exposure')
        permissions = read_json(stage / 'data' / 'PERMISSION_REVIEWS.json')
        require(permissions['reviews'] == cfg['permission_reviews'], 'source/permission review decisions changed')
        expected_attribution = [(pid, item, str(stage / 'data' / f'permission-{pid}-attribution-{index}.txt'))
                                for pid, review in sorted(reviews.permissions.items())
                                for index, item in enumerate(review.get('attribution_files', []))]
        require([(v['permission_id'], v['source'], v['output']['path']) for v in permissions['attribution']] == expected_attribution,
                'a required upstream attribution file was omitted or redirected')
        for item in permissions['attribution']:
            check()
            require(stat(item['source']['path']) == item['source'] and stat(item['output']['path']) == item['output'],
                    'private upstream attribution material identity differs')
            with open(item['source']['path'], 'rb') as original, open(item['output']['path'], 'rb') as retained:
                require(original.read() == retained.read(), 'private upstream attribution material was not retained exactly')
        return {'complete': True, 'retained_original_fields_directly_reread': raw_count,
                'known_fields_retokenized': known_fields, 'unknown_fields_retained_no_forward': masked_fields,
                'known_K1_zero_field_CE_gradient_fields': k1_fields,
                'output_component_counts': dict(composition), 'exact_full_pair_cross_split_isolation': True,
                'old_roster_identity_gate_applied_to_every_dev_field': True, 'all_old_root_siblings_retained': True,
                'seed42_43_same_multiset_verified': True if readiness['train']['ready'] else None,
                 'train_ready': readiness['train']['ready'], 'dev_ready': readiness['dev']['ready'],
                 'recognized_native_closure_pairs_directly_verified':closure_count,
                 'unrestored_native_sibling_veto_pairs_verified':closure_veto,
                'schedules': schedules, 'model_run': False,
                'scope': 'actual original-coordinate/input/target/token/exposure readback, exact reviewed membership only'}
    finally:
        vdb.close()
