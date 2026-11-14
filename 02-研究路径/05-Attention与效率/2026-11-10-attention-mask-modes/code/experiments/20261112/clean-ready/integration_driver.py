"""Bounded clean-origin native CPU preparation. SOURCE DRAFT, NOT RUN.

Real entrypoint: --config reviewed-config.json --output NEW_PRIVATE_STAGE.
No model weights, old adapters, teacher labels, downloads, training or tests.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
import importlib.util
from itertools import groupby
import json
import os
from pathlib import Path
import random
import sqlite3

from semantic_taint import (ROOT, AuditGraph, Membership, DOMAINS, HISTORY, PUBLIC,
                            FORBIDDEN, NATIVE_DEV, INCOMPLETE, UNKNOWN, dump, owned,
                            require, stat, state_text, write_json)
from prepare_runtime import Budget, Identity, jsonl, read_json
from census_interface import Census, Reviews
from input_projector import native_projection, ExclusionSeeds
from case_pairs import (TYPED_COLUMNS, normalized_field, make_case, compile_case,
                        round_robin_families, write_exposure, summarize)

def validate_config(cfg):
    require(cfg.get('version') == 'clean-origin-native-candidate-v2' and cfg.get('approved_for_cpu_prepare') is True,
            'review template is not approved for actual CPU preparation')
    policy = cfg['isolation_policy']
    require(policy.get('reviewed') is True and policy.get('evidence_files') and
            policy['scope'] == 'exact-reviewed-representation-membership-v1' and
            policy['public_seed_policy'] == 'independent-provided-texts-no-union' and
            policy['history_seed_policy'] == 'sealed-old-root-taint-superset' and
            policy['native_dev_seed_policy'] == 'all15550-inputs-plus-original-roster-identity',
            'candidate isolation policy review missing or contract changed')
    require(cfg['unknown_forward_rule'] == 'skip-all-unknown-fields-v1', 'both arms must share the same explicit unknown-forward policy')
    require(cfg['family_weight_basis'] == 'eligible-compiled-candidate-components', 'sqrt-family cardinality basis must be explicit')
    require(cfg['stock_tokenizer_directory'] == '/home/dev/assumeengage/models/Qwen3-0.6B' and
            cfg['pair_helper'] == '/home/dev/assumeengage/scripts/pair_data_common.py', 'stock tokenizer/original helper pointer changed')
    require(cfg['declared_split_storage'] == 'raw' and cfg['native_file_split_values'] == {'train': 'train', 'validation': 'validation'},
            'actual completed census split schema changed')
    require(cfg['roster_eligible_key'] == 'eligible_under_exact_policy', 'old roster gate must not be replaced by flags-only')
    limits = cfg['limits']
    require(type(limits['candidate_cases_per_domain']) is int and 2500 <= limits['candidate_cases_per_domain'] <= 5000,
            'native metadata candidate limit outside2500..5000')
    require(1 <= limits['arrow_batch_rows'] <= 1024 and 1 <= limits['max_candidate_fields'] <= 1_000_000,
            'raw CPU field/batch bound missing')
    require(1 <= limits['max_anchor_roots'] <= 512 and 2500 <= cfg['min_train_components_per_domain'] <= 5000,
            'anchor fanout/minimum train inventory bound missing')
    require(cfg['logical_pair_token_ceiling'] == 100_000_000 and cfg['macro_cases'] == 16 and cfg['data_seed'] == 20261112,
            'frozen100M/macro/data-seed contract changed')


def stage_database(path):
    require(not path.exists(), 'existing stage database preserved')
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript('''
        PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
        PRAGMA temp_store=MEMORY; PRAGMA cache_size=-32768;
        CREATE TABLE metadata(old_root INTEGER PRIMARY KEY, split TEXT, domain TEXT,
                              family TEXT, fields INTEGER, rank INTEGER);
        CREATE INDEX metadata_order ON metadata(split,domain,family,rank,old_root);
        CREATE TABLE candidates(old_root INTEGER PRIMARY KEY, split TEXT, domain TEXT,
                                family TEXT, fields INTEGER, ordinal INTEGER UNIQUE);
        CREATE TABLE coordinates(file_id INTEGER,row_index INTEGER,old_root INTEGER,
                                 PRIMARY KEY(file_id,row_index)) WITHOUT ROWID;
        CREATE INDEX coordinates_root ON coordinates(old_root,file_id,row_index);
        CREATE TABLE native_fields(file_id INTEGER,row_index INTEGER,old_root INTEGER,record TEXT,anchors TEXT,
                                  PRIMARY KEY(file_id,row_index)) WITHOUT ROWID;
        CREATE INDEX fields_root ON native_fields(old_root,file_id,row_index);
        CREATE TABLE components(component INTEGER,old_root INTEGER PRIMARY KEY);
        CREATE INDEX component_members ON components(component,old_root);
        CREATE TABLE compiled(component INTEGER PRIMARY KEY,split TEXT,domain TEXT,family TEXT,
                              logical_tokens INTEGER,max_pair INTEGER,forward_once INTEGER,record TEXT);
        CREATE TABLE outputs(component INTEGER PRIMARY KEY,split TEXT);
    ''')
    return db


def metadata_candidates(census, reviews, db, cfg, check):
    counts, rejects, rejected_rows, source_quarantine = Counter(), Counter(), Counter(), Counter()
    dev_roots = defaultdict(list)
    for key, value in census.roster.items():
        dev_roots[value['component']].append(census.dev_eligible(json.loads(key)))
    rng = random.Random(cfg['data_seed'])
    sql = '''SELECT r.old_root,r.old_bits,r.source_id,r.permission_id,r.kind,r.declared_split,r.group_present,
                    f.native_file_split,s.domain,s.family,count(*) AS n
             FROM rows r JOIN files f USING(file_id) JOIN sources s USING(source_id)
             GROUP BY r.old_root,r.old_bits,r.source_id,r.permission_id,r.kind,r.declared_split,
                      r.group_present,f.native_file_split,s.domain,s.family ORDER BY r.old_root'''
    for root, group in groupby(census.db.execute(sql), key=lambda r: r['old_root']):
        groups = list(group)
        count = sum(g['n'] for g in groups)
        reasons = set()
        splits = {g['native_file_split'] for g in groups}
        physical = next(iter(splits)) if len(splits) == 1 else 'mixed'
        if root is None:
            reasons.add('no_original_root')
            bits = INCOMPLETE
        else:
            require(census.audit.root(root) == root, 'census metadata old_root is not final')
            bits = census.audit.bits(root)
            require(all(g['old_bits'] == bits for g in groups), 'census old flags changed')
        veto = PUBLIC | FORBIDDEN | INCOMPLETE | UNKNOWN | (NATIVE_DEV if physical == 'train' else HISTORY)
        if bits & veto:
            reasons.add('old_root_membership_veto')
        if physical not in ('train', 'validation'):
            reasons.add('mixed_physical_train_dev_component')
        if physical == 'validation' and (root not in dev_roots or len(dev_roots[root]) != count or not all(dev_roots[root])):
            reasons.add('original_roster_identity_or_eligibility_veto')
        for g in groups:
            reason = reviews.reasons(g['source_id'], g['permission_id'])
            reasons.update(reason)
            for name in reason:
                source_quarantine[dump([g['source_id'], name])] += g['n']
            valid_declared = ('train', 'training') if physical == 'train' else ('dev', 'valid', 'validation')
            if str(g['declared_split'] or '').strip().casefold() not in valid_declared:
                reasons.add('non_native_declared_split_sibling')
            if not g['group_present']:
                reasons.add('missing_original_group')
        families = {g['family'] for g in groups}
        if len(families) != 1:
            reasons.add('mixed_family_assignment_not_reviewed')
        domains = Counter()
        for g in groups:
            if g['kind'] in ('choice', 'noul'):
                domains[g['domain']] += g['n']
        if not domains:
            reasons.add('metadata_has_no_supported_native_supervision_kind')
        counts['native_rows_considered'] += count
        counts['old_components_considered'] += 1
        if reasons:
            rejects.update(reasons)
            for reason in reasons:
                rejected_rows[reason] += count
        else:
            domain = max(DOMAINS, key=lambda d: (domains[d], -DOMAINS.index(d)))
            db.execute('INSERT INTO metadata VALUES(?,?,?,?,?,?)',
                       (root, physical, domain, dump(sorted(families)), count, rng.getrandbits(63)))
        if counts['old_components_considered'] % 1024 == 0:
            check()
            db.commit()
    require(counts['native_rows_considered'] == 2544892, 'metadata grouped scan did not cover the complete census')
    db.commit()
    ordinal = 0
    for split in ('validation', 'train'):
        for i, domain in enumerate(DOMAINS):
            families = [r[0] for r in db.execute('SELECT DISTINCT family FROM metadata WHERE split=? AND domain=? ORDER BY family', (split, domain))]
            random.Random(cfg['data_seed'] + i + (100 if split == 'train' else 0)).shuffle(families)
            cursors = {family: iter(db.execute('SELECT * FROM metadata WHERE split=? AND domain=? AND family=? ORDER BY rank,old_root',
                                             (split, domain, family))) for family in families}
            limit = cfg['limits']['candidate_cases_per_domain'] if split == 'train' else None
            selected = 0
            while families and (limit is None or selected < limit):
                remaining = []
                for family in families:
                    row = next(cursors[family], None)
                    if row is None:
                        continue
                    db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?)',
                               (row['old_root'], split, domain, family, row['fields'], ordinal))
                    ordinal += 1
                    selected += 1
                    remaining.append(family)
                    if limit is not None and selected == limit:
                        break
                families = remaining
                check()
            counts[split + ':' + domain + ':candidate_components'] = selected
    db.commit()
    fields = 0
    for candidate in db.execute('SELECT old_root,fields FROM candidates ORDER BY ordinal'):
        seen = 0
        for row in census.db.execute('SELECT file_id,row_index FROM rows WHERE old_root=? ORDER BY file_id,row_index', (candidate['old_root'],)):
            db.execute('INSERT INTO coordinates VALUES(?,?,?)', (row['file_id'], row['row_index'], candidate['old_root']))
            seen += 1
        require(seen == candidate['fields'], 'complete candidate sibling coordinate census differs')
        fields += seen
        require(fields <= cfg['limits']['max_candidate_fields'], 'bounded candidate field allowance exhausted; no sampled sibling truncation')
        check()
    db.commit()
    return {'counts': dict(counts), 'nonexclusive_rejected_components': dict(rejects),
            'nonexclusive_rejected_rows': dict(rejected_rows), 'source_review_quarantine_rows': dict(source_quarantine),
            'selected_candidate_fields': fields, 'candidate_order_before_raw_targets': True,
            'original_roster_identity_gate': census.identity_gate_stats,
            'metadata_primary_domain_is_provisional': True}


def project_candidates(census, reviews, db, membership, cfg, check):
    seen = 0
    shapes, failures, example_columns = Counter(), Counter(), Counter()
    selector = lambda fid: [r[0] for r in db.execute('SELECT row_index FROM coordinates WHERE file_id=? ORDER BY row_index', (fid,))]
    for coordinate, raw, meta in census.iter_payload(selector, TYPED_COLUMNS, cfg['limits']['arrow_batch_rows']):
        root = meta['old_root']
        selected = db.execute('SELECT old_root FROM coordinates WHERE file_id=? AND row_index=?', (meta['file_id'], meta['row_index'])).fetchone()
        require(selected is not None and selected[0] == root, 'raw candidate coordinate/root mismatch')
        source = census.sources[meta['source_id']]
        anchors, reason, profile = native_projection({'state': raw['state'], 'question': raw['question']},
                                                      reviews.sources[meta['source_id']], {'variant': raw.get('variant')})
        contract = reviews.noul_contracts.get(profile.get('noul_contract_id')) if profile else None
        field = normalized_field(raw, coordinate, source['domain'], contract)
        field['reviewed_input_shape_id'] = profile['shape_id'] if profile else None
        if reason:
            failures[dump([meta['source_id'], reason])] += 1
        else:
            shapes[dump([meta['source_id'], profile['shape_id']])] += 1
        example_columns[dump([meta['file_id'], 'present' if 'example_id' in raw else 'absent'])] += 1
        require(field['family'] == source['family'], 'field family differs from frozen census registry')
        if not state_text(raw['state']):
            reason = 'empty_native_provided_input'
        split = db.execute('SELECT split FROM candidates WHERE old_root=?', (root,)).fetchone()[0]
        membership.observe('native_train' if split == 'train' else 'native_dev', dump(coordinate), root, meta['old_bits'], anchors, reason)
        db.execute('INSERT INTO native_fields VALUES(?,?,?,?,?)',
                   (meta['file_id'], meta['row_index'], root, dump(field), dump(anchors)))
        seen += 1
        if seen % 256 == 0:
            check()
            db.commit()
            membership.db.commit()
    require(seen == db.execute('SELECT count(*) FROM coordinates').fetchone()[0], 'candidate native field projection incomplete')
    for row in db.execute('SELECT old_root,fields FROM candidates'):
        require(db.execute('SELECT count(*) FROM native_fields WHERE old_root=?', (row['old_root'],)).fetchone()[0] == row['fields'],
                'projector lost a candidate sibling field')
    db.commit()
    membership.db.commit()
    return {'fields_projected': seen, 'targets_not_passed_to_anchor_projector': True,
            'positive_shape_field_counts': dict(shapes), 'positive_shape_quarantine_counts': dict(failures),
            'example_id_column_presence': dict(example_columns), 'example_id_used_as_group': False}


def assemble_component(component, roots, census, reviews, db, membership):
    fields, expected, splits = [], [], set()
    for root in roots:
        candidate = db.execute('SELECT split FROM candidates WHERE old_root=?', (root,)).fetchone()
        require(candidate is not None and membership.root(root)[0] == component, 'semantic component membership changed')
        splits.add(candidate['split'])
        for row in db.execute('SELECT record FROM native_fields WHERE old_root=? ORDER BY file_id,row_index', (root,)):
            fields.append(json.loads(row[0]))
        for row in census.db.execute('SELECT file_id,row_index FROM rows WHERE old_root=? ORDER BY file_id,row_index', (root,)):
            expected.append({'file': census.files[row['file_id']]['path'], 'row_index': row['row_index']})
    split = next(iter(splits)) if len(splits) == 1 else 'validation'
    flags = membership.root(roots[0])[1]
    case = make_case(component, flags, fields, expected, reviews.license_decisions,
                     reviews.source_decisions, split,
                     {dump(c): census.dev_eligible(c) for c in expected} if split == 'validation' else None)
    case['old_roots'] = sorted(roots)
    if len(splits) != 1:
        case['rejection_reasons'].append('mixed_semantic_train_dev_component')
    return case


def compile_inventory(census, reviews, db, membership, tokenizer, helper, check):
    for row in db.execute('SELECT old_root FROM candidates ORDER BY old_root'):
        db.execute('INSERT INTO components VALUES(?,?)', (membership.root(row[0])[0], row[0]))
    db.commit()
    rejected, accepted = Counter(), Counter()
    for (component,) in db.execute('SELECT DISTINCT component FROM components ORDER BY component'):
        check()
        roots = [r[0] for r in db.execute('SELECT old_root FROM components WHERE component=? ORDER BY old_root', (component,))]
        case = assemble_component(component, roots, census, reviews, db, membership)
        if not case['rejection_reasons']:
            case = compile_case(case, tokenizer, helper, check)
        if case['rejection_reasons']:
            rejected.update(case['rejection_reasons'])
            continue
        db.execute('INSERT INTO compiled VALUES(?,?,?,?,?,?,?,?)',
                   (component, case['split'], case['primary_domain'], case['family_stratum'],
                    case['logical_supervised_pair_tokens'], case['max_pair_tokens'],
                    case['scheduled_forward_pair_tokens_once'], dump(case)))
        accepted[case['split'] + ':' + case['primary_domain']] += 1
        db.commit()
    return {'accepted_components': dict(accepted), 'nonexclusive_rejected_components': dict(rejected)}


def freeze_output_rosters(db, cfg):
    train_blockers, dev_blockers = [], []
    for i, domain in enumerate(DOMAINS):
        references = [{'component': r[0], 'case_id': f'clean-semantic-{r[0]}', 'family_stratum': r[1]}
                      for r in db.execute('SELECT component,family FROM compiled WHERE split=? AND domain=? ORDER BY component', ('validation', domain))]
        order = round_robin_families(references, random.Random(cfg['data_seed'] + i))
        if len(order) < 125:
            dev_blockers.append(domain + ': fewer_than125_complete_reviewed_dev_components')
        else:
            select = order[:75]
            used = {c['component'] for c in select}
            confirm = round_robin_families([c for c in order if c['component'] not in used], random.Random(cfg['data_seed'] + 50 + i))[:50]
            for name, values in (('select', select), ('confirm', confirm)):
                for value in values:
                    db.execute('INSERT INTO outputs VALUES(?,?)', (value['component'], name))
        train = [{'component': r[0], 'case_id': f'clean-semantic-{r[0]}', 'family_stratum': r[1]}
                 for r in db.execute('SELECT component,family FROM compiled WHERE split=? AND domain=? ORDER BY component', ('train', domain))]
        train = round_robin_families(train, random.Random(cfg['data_seed'] + 100 + i))[:cfg['limits']['candidate_cases_per_domain']]
        if len(train) < cfg['min_train_components_per_domain']:
            train_blockers.append(domain + ': fewer_than' + str(cfg['min_train_components_per_domain']) + '_legal_compiled_train_components')
        for value in train:
            db.execute('INSERT INTO outputs VALUES(?,?)', (value['component'], 'train'))
    db.commit()
    return {'train': {'ready': not train_blockers, 'incomplete_reasons': train_blockers},
            'dev': {'ready': not dev_blockers, 'incomplete_reasons': dev_blockers}}


def output_cases(db, split):
    for row in db.execute('SELECT c.record FROM outputs o JOIN compiled c USING(component) WHERE o.split=? ORDER BY c.domain,c.family,c.component', (split,)):
        yield json.loads(row[0])


def load_tokenizer_helper(cfg, identity, stage):
    review = cfg['tokenizer_review']
    require(review.get('approved') is True and review.get('evidence_files'), 'stock tokenizer review missing')
    directory = Path(cfg['stock_tokenizer_directory'])
    allowed = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json',
               'vocab.json', 'merges.txt', 'added_tokens.json', 'tokenizer.model'}
    actual = {str(p) for p in directory.iterdir() if p.is_file() and p.name in allowed}
    require({v['path'] for v in review['files']} == actual and str(directory / 'tokenizer_config.json') in actual,
            'stock tokenizer file set not completely pinned')
    identity.add(review['files'] + review['evidence_files'])
    identity.require_pins([Path(cfg['pair_helper'])])
    spec = importlib.util.spec_from_file_location('clean_original_pair_helper', cfg['pair_helper'])
    require(spec is not None and spec.loader is not None, 'original helper cannot be imported')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False,
                                             cache_dir=str(stage / 'offline-cache'))
    require(type(tokenizer.eos_token_id) is int and tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id,
            'stock tokenizer actual EOS/PAD missing')
    return tokenizer, helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only launch required')
    os.umask(0o077)
    stage = owned(args.output)
    require(stage != ROOT and not stage.exists(), 'new exclusive child stage required; existing outputs never adopted')
    stage.mkdir(mode=0o700)
    budget = audit = census = membership = db = None
    phase = 'review_config'
    try:
        cfg = read_json(args.config)
        validate_config(cfg)
        budget = Budget(stage, cfg['limits'])
        identity = Identity(args.config, cfg)
        identity.add(cfg['isolation_policy']['evidence_files'])
        phase = 'bind_existing_audit_census'
        audit = AuditGraph(cfg['audit_directory'])
        identity.add(audit.bound)
        identity.add([stat(audit.directory / n) for n in ('COMPLETE.json', 'SOURCE_BINDING.json')])
        census = Census(cfg, audit, identity, budget.check)
        reviews = Reviews(cfg, census, identity)
        db = stage_database(stage / 'preparation.sqlite3')
        membership = Membership(stage / 'membership.sqlite3', cfg['limits']['max_anchor_roots'])
        phase = 'metadata_candidate_freeze'
        metadata = metadata_candidates(census, reviews, db, cfg, budget.check)
        write_json(stage / 'METADATA_SELECTION.json', metadata)
        jsonl(stage / 'CANDIDATES.jsonl', (dict(r) for r in db.execute('SELECT * FROM candidates ORDER BY ordinal')), budget.check)
        phase = 'bounded_native_field_projection'
        projection = project_candidates(census, reviews, db, membership, cfg, budget.check)
        phase = 'complete_exclusion_seed_membership'
        seeds = ExclusionSeeds(membership, [p for r in reviews.sources.values() for p in r['input_profiles']], budget.check)
        public_path = audit.directory / 'public-input-texts.jsonl'
        require(audit.source['public_input_view'] == stat(public_path), 'public view is not the originally bound input-only projection')
        public_receipt = seeds.public_view(public_path)
        graph_receipt = seeds.old_graph(audit,census)
        dev_receipt = seeds.native_dev(census, census.all_dev_inputs())
        membership.seal_candidates()
        semantic = {'public': public_receipt, 'old_graph': graph_receipt, 'native_dev': dev_receipt,
                    'candidate_projection': projection, 'seed_match_counts': dict(seeds.counts),
                    'new_conversion_counts': dict(seeds.conversions), 'new_negative_input_passes_complete': True,
                    'scope': cfg['isolation_policy'], 'near_duplicate_or_pretraining_absence_claimed': False}
        write_json(stage / 'SEMANTIC_MEMBERSHIP.json', semantic)
        phase = 'tokenizer_and_complete_case_compile'
        tokenizer, helper = load_tokenizer_helper(cfg, identity, stage)
        identity.publish(stage)
        compiled = compile_inventory(census, reviews, db, membership, tokenizer, helper, budget.check)
        write_json(stage / 'COMPILED_INVENTORY.json', compiled)
        readiness = freeze_output_rosters(db, cfg)
        write_json(stage / 'INVENTORY_READINESS.json', readiness)
        data = stage / 'data'
        data.mkdir(mode=0o700)
        summaries = {}
        for split in ('select', 'confirm', 'train'):
            jsonl(data / (split + '.jsonl'), output_cases(db, split), budget.check)
            summaries[split] = summarize(output_cases(db, split))
        # Preserve actual reviewed permission/license_use and upstream legal
        # texts for an exact mixed-license exception; no data raw/git copying.
        attribution = []
        for pid, review in sorted(reviews.permissions.items()):
            for index, item in enumerate(review.get('attribution_files', [])):
                target = data / f'permission-{pid}-attribution-{index}.txt'
                with target.open('xb') as stream:
                    stream.write(Path(item['path']).read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                attribution.append({'permission_id': pid, 'source': item, 'output': stat(target)})
        write_json(data / 'PERMISSION_REVIEWS.json', {'reviews': cfg['permission_reviews'], 'attribution': attribution})
        phase = 'freeze100M_exposures'
        refs = [{'case_id': f'clean-semantic-{r[0]}', 'primary_domain': r[1], 'family_stratum': r[2],
                 'logical_supervised_pair_tokens': r[3], 'max_pair_tokens': r[4], 'all_field_pair_tokens_once': r[5],
                 'split': 'train', 'rejection_reasons': []}
                for r in db.execute('SELECT c.component,c.domain,c.family,c.logical_tokens,c.max_pair,c.forward_once FROM outputs o JOIN compiled c USING(component) WHERE o.split=? ORDER BY c.component', ('train',))]
        schedules = {}
        for seed in ((42, 43) if readiness['train']['ready'] else ()):
            schedule = write_exposure(refs, data / f'exposure-seed{seed}.jsonl', seed=seed, check=budget.check)
            require(not schedule['incomplete_reasons'], 'exposure observations incomplete')
            write_json(data / f'schedule-seed{seed}.json', schedule)
            schedules[str(seed)] = schedule
        write_json(stage / 'AGGREGATES.json', summaries)
        before_readback = [stat(p) for p in sorted(data.iterdir()) if p.is_file()]
        phase = 'actual_raw_and_token_artifact_readback'
        from verify_prepared import verify_prepared
        verification = verify_prepared(stage, census, reviews, db, membership, tokenizer, helper, cfg, semantic, readiness, budget.check)
        identity.check()
        from semantic_taint import verify
        verify(before_readback)
        require(read_json(stage / 'CODE_BINDING.json') == identity.code and
                read_json(stage / 'SOURCE_BINDING.json')['files'] == identity.files,
                'published source/code binding changed during actual readback')
        audit.close()
        audit = None
        # Close/flush every writable internal DB before the final resource and
        # commit boundary; finalizer must not grow a DB after COMPLETE appears.
        membership.db.commit()
        membership.db.close()
        membership = None
        db.commit()
        db.close()
        db = None
        census.db.close()
        census = None
        budget.check(force=True)
        write_json(stage / 'FINAL_VERIFICATION.json', verification)
        artifacts = [stat(p) for p in sorted(data.iterdir()) if p.is_file()]
        artifacts += [stat(stage / n) for n in ('CODE_BINDING.json', 'SOURCE_BINDING.json', 'CANDIDATES.jsonl',
                                               'METADATA_SELECTION.json', 'SEMANTIC_MEMBERSHIP.json',
                                               'COMPILED_INVENTORY.json', 'AGGREGATES.json', 'INVENTORY_READINESS.json', 'FINAL_VERIFICATION.json')]
        experiment_ready = readiness['train']['ready'] and readiness['dev']['ready']
        write_json(stage / 'TRAIN_READY.json', {'complete': True, 'train_ready': readiness['train']['ready'],
                                               'minimum_legal_compiled_cases_per_domain': cfg['min_train_components_per_domain'],
                                               'incomplete_reasons': readiness['train']['incomplete_reasons'], 'verification': stat(stage / 'FINAL_VERIFICATION.json')})
        write_json(stage / 'DEV_READY.json', {'complete': True, 'dev_ready': readiness['dev']['ready'],
                                             'select_per_domain': 75, 'confirm_per_domain': 50,
                                             'incomplete_reasons': readiness['dev']['incomplete_reasons'], 'verification': stat(stage / 'FINAL_VERIFICATION.json')})
        artifacts += [stat(stage / n) for n in ('TRAIN_READY.json', 'DEV_READY.json')]
        runtime = budget.receipt()
        runtime['own_bytes_before_final_receipts'] = runtime.pop('own_bytes')
        runtime['own_bytes_measurement_scope'] = 'before_DATA_READY_and_COMPLETE; final_total_in_COMPLETE'
        ready = {'complete': True, 'DATA_READY': experiment_ready, 'train_ready': readiness['train']['ready'],
                 'dev_ready': readiness['dev']['ready'], 'stage': 'bounded_native_clean_origin_data',
                 'source_binding': stat(stage / 'SOURCE_BINDING.json'), 'code_binding': stat(stage / 'CODE_BINDING.json'),
                  'files': artifacts, 'runtime': runtime, 'verification': verification,
                 'logical_pair_token_ceiling': 100000000, 'macro_cases': 16,
                 'requires_complete_commit_marker': True,
                 'unknown_forward_rule': cfg['unknown_forward_rule'], 'model_run': False,
                 'scope': 'exact reviewed candidate representation membership; not near-dupe/pretraining proof',
                 'fresh_model_contract': 'stock ModelScope Qwen3-0.6B + fresh LoRA/head; no weights loaded by CPU preparation'}
        write_json(stage / 'DATA_READY.json', ready)  # Only after actual source+token+exposure readback.
        budget.check(force=True)
        base_bytes = budget.own_bytes
        commit = {'complete': True, 'DATA_READY': experiment_ready,
                  'train_ready': readiness['train']['ready'], 'dev_ready': readiness['dev']['ready'],
                  'data_ready': stat(stage / 'DATA_READY.json'), 'model_run': False,
                  'final_receipt_budget_verified':True,'own_bytes_including_final_receipts':0}
        for _ in range(16):
            measured = base_bytes + len((dump(commit)+'\n').encode('utf-8'))
            if measured == commit['own_bytes_including_final_receipts']:
                break
            commit['own_bytes_including_final_receipts'] = measured
        else:
            raise RuntimeError('Final receipt byte-accounting fixed point did not converge')
        pending = stage / 'COMPLETE.json.writing'
        write_json(pending,commit)
        budget.check(force=True,reserve_bytes=65536)
        require(budget.own_bytes == commit['own_bytes_including_final_receipts'],'Final post-write budget accounting differs')
        # Atomic no-overwrite commit; do not replace an unfamiliar COMPLETE.
        os.link(pending,stage / 'COMPLETE.json')
        pending.unlink()
    except BaseException as error:
        write_json(stage / 'BLOCKED.json', {'complete': False, 'DATA_READY': False, 'phase': phase,
                                          'error_type': type(error).__name__, 'error': str(error), 'partial_preserved': True,
                                          'model_run': False})
        raise
    finally:
        if membership is not None:
            membership.db.close()
        if db is not None:
            db.close()
        if census is not None:
            census.db.close()
        if audit is not None:
            audit.close()
        if budget is not None:
            budget.close()


if __name__ == '__main__':
    main()
