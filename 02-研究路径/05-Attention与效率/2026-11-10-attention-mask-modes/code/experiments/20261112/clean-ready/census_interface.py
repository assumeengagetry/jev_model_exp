"""Read-only adapter for the ACTUAL native-coordinate/license census schema."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import sqlite3

from semantic_taint import ROOT, DOMAINS, dump, require, state_text, stat, verify
from prepare_runtime import read_json
from input_projector import validate_profile

TABLES = {
    'files': ('file_id', 'path', 'native_file_split', 'expected_rows'),
    'sources': ('source_id', 'source', 'family', 'domain'),
    'permissions': ('permission_id', 'source_id', 'license', 'license_use'),
    'rows': ('file_id', 'row_index', 'old_root', 'old_bits', 'source_id', 'permission_id', 'kind', 'declared_split', 'group_present'),
}
INPUT_COLUMNS = ('state', 'question', 'source', 'id', 'group_id', 'split')


class Reviews:
    def __init__(self, cfg, census, identity):
        self.sources, self.permissions = {}, {}
        self.source_decisions, self.license_decisions = {}, {}
        self.negative_scope_quarantine = set()
        for gap in cfg['isolation_policy']['known_relevant_negative_gaps']:
            require(gap.get('resolution') == 'quarantine-native-source-scope' and gap.get('native_source_ids') and
                    gap.get('evidence_files') and set(gap['native_source_ids']) <= set(census.sources),
                    'known related negative-layout gap must have an explicit affected native scope quarantine')
            self.negative_scope_quarantine.update(gap['native_source_ids'])
            identity.add(gap['evidence_files'])
        self.noul_contracts = cfg['noul_contracts']
        for name, contract in self.noul_contracts.items():
            require(contract.get('contract_id') == name and contract.get('reviewed') is True and
                    contract.get('q_semantics') == 'native_yes_probability' and contract.get('evidence_files'),
                    'noul direction must cite an actual typed importer contract, not only a [q] shape')
            identity.add(contract['evidence_files'])
        aliases, allowed = cfg['license_aliases'], set(cfg['allowed_canonical_licenses'])
        denied = {'', 'custom', 'request', 'unspecified', 'other', 'unknown', 'none', 'null', 'n/a',
                  'custom-license', 'request-only', 'requires permission', 'permission required', '需另行申请',
                  'custom (dpi)', 'other (dpi)', 'request form', 'request form (dpi)', 'various', 'various (dpi)', 'openai (dpi)'}
        for review in cfg['source_reviews']:
            sid = review['source_id']
            require(sid not in self.sources and sid in census.sources, 'duplicate/unbound reviewed source')
            source = census.sources[sid]
            require(all(review[k] == source[k] for k in ('source', 'family', 'domain')), 'review source identity changed')
            require(review.get('reviewed') is True and review.get('provenance_complete') is True and
                    review.get('case_complete') is True and review.get('target_not_in_provided_input') is True and
                    review.get('evidence_files'), 'source review is incomplete')
            profiles = review['input_profiles']
            require(profiles and len({p['shape_id'] for p in profiles}) == len(profiles), 'source shape gates missing/duplicated')
            for profile in profiles:
                validate_profile(profile)
                identity.add(profile['evidence_files'])
                contract_id = profile.get('noul_contract_id')
                require(contract_id is None or contract_id in self.noul_contracts, 'shape references an unreviewed noul direction contract')
            identity.add(review['evidence_files'])
            if source['source'] == 'UltraFeedback-paired':
                require(isinstance(review.get('local_source_revision'), str) and review['local_source_revision'] and
                        review.get('revision_evidence_files') and review.get('converter_evidence_files'),
                        'public MIT/new-card evidence does not establish the actual local UltraFeedback revision/conversion')
                identity.add(review['revision_evidence_files'] + review['converter_evidence_files'])
            self.sources[sid] = review
            self.source_decisions[source['source']] = {'provenance_complete': True, 'case_complete': True,
                                                      'evidence': review['evidence_files']}
        for review in cfg['permission_reviews']:
            pid = review['permission_id']
            require(pid not in self.permissions and pid in census.permissions, 'duplicate/unbound reviewed permission')
            permission = census.permissions[pid]
            require(all(review[k] == permission[k] for k in ('source_id', 'license', 'license_use')), 'review permission identity changed')
            require(review['source'] == census.sources[permission['source_id']]['source'], 'permission source changed')
            raw = permission['license']
            require(isinstance(raw, str) and raw.strip().casefold() not in denied, 'Custom/Request/unknown license cannot be granted')
            method = review['method']
            mixed = 'Apache License 2.0 (DPI), CC BY 4.0 (DPI), OpenAI (DPI)'
            if method == 'exact-source-mixed-exception-v1':
                require((review['source_id'], review['source'], raw, review['license_use']) ==
                        (530, 'summarize_from_feedback/comparisons', mixed, 'non-commercial'),
                        'mixed license exception is restricted to the actual source530 permission tuple')
                require(review['evidence_commit'] == '56b6bb613a1b58a8aa7a5e29266f65c7b980ee48' and
                        review.get('retain_non_commercial') is True and review.get('attribution_files'),
                        'mixed license requires fixed-commit evidence, retained non-commercial flag and full private attribution materials')
                require(review.get('feedback_data_license') == 'Modified-MIT-Software-and-Datasets' and
                        review.get('reddit_text_license') == 'CC-BY-4.0', 'mixed exception must separately retain both upstream obligations')
                require(review.get('attribution_complete') is True and
                        set(review.get('attribution_roles', {})) == {'feedback_full_license', 'reddit_cc_by_4_full_license', 'upstream_notices'} and
                        set(review['attribution_roles'].values()) <= {v['path'] for v in review['attribution_files']} and
                        any(v['path'] == str(ROOT / 'primary-license-evidence.md')
                            for v in review['evidence_files']), 'source530 precise primary evidence/full private attribution bundle is missing')
                identity.add(review['attribution_files'])
            else:
                require(method == 'standard-exact-alias-v1' and raw != mixed and raw in aliases and
                        aliases[raw] in allowed and review['canonical_license'] == aliases[raw],
                        'license alias must be an exact approved map, never a substring or mixed-source inference')
            require(review.get('reviewed') is True and review.get('allowed_private_research') is True and
                    review.get('evidence_files'), 'permission review is incomplete')
            identity.add(review['evidence_files'])
            self.permissions[pid] = review
            self.license_decisions[dump([review['source'], raw, permission['license_use']])] = {
                'allowed_private_research': True, 'evidence': review['evidence_files']}
        require(self.sources and self.permissions, 'empty review template is NOT an approved preparation config')

    def reasons(self, source_id, permission_id):
        reasons = []
        if source_id in self.negative_scope_quarantine:
            reasons.append('related_uncovered_negative_layout_native_scope_quarantined')
        if source_id not in self.sources:
            reasons.append('source_profile_or_provenance_unreviewed')
        if permission_id not in self.permissions:
            reasons.append('source_permission_unreviewed')
        return reasons


class Census:
    def __init__(self, cfg, audit, identity, check):
        self.cfg, self.audit, self.check = cfg, audit, check
        directory = Path(cfg['census_directory'])
        core = [directory / n for n in ('COMPLETE.json', 'INVENTORY.json', 'native-coordinates.sqlite3')]
        identity.require_pins(core + [Path(cfg['census_source_binding'])])
        receipt = read_json(core[0])
        require(receipt['complete'] is True and receipt['DATA_READY'] is False, 'census receipt is not the completed non-ready coordinate census')
        identity.add(receipt['files'])
        inherited = read_json(cfg['census_source_binding'])
        identity.add(inherited[cfg['census_source_files_key']])
        self.db = sqlite3.connect(core[2].resolve().as_uri() + '?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA query_only=ON')
        self.db.execute('PRAGMA temp_store=MEMORY')
        def progress():
            self.check()
            return 0
        self.db.set_progress_handler(progress, 100000)
        for table, columns in TABLES.items():
            require(set(columns) <= {r['name'] for r in self.db.execute('PRAGMA table_info(' + table + ')')},
                    'census schema differs: ' + table)
        self.files = {r['file_id']: dict(r) for r in self.db.execute('SELECT * FROM files')}
        self.sources = {r['source_id']: dict(r) for r in self.db.execute('SELECT * FROM sources')}
        self.permissions = {r['permission_id']: dict(r) for r in self.db.execute('SELECT * FROM permissions')}
        require(len(self.files) == 17 and len(self.sources) == len(self.permissions) == 667, 'wrong completed census inventory')
        split_values = cfg['native_file_split_values']
        require(set(split_values) == {'train', 'validation'} and len(set(split_values.values())) == 2, 'native file split convention not explicit')
        self.file_by_path = {v['path']: k for k, v in self.files.items()}
        require(len(self.file_by_path) == 17, 'duplicate census file path')
        expected = set(audit.source['train_paths'] + audit.source['validation_paths'])
        require(set(self.file_by_path) == expected, 'census original17-file inventory differs from audit')
        inventory = read_json(core[1])
        for table, actual in (('sources', self.sources), ('permissions', self.permissions)):
            key = 'source_id' if table == 'sources' else 'permission_id'
            listed = {v[key]: v for v in inventory[table]}
            require(set(listed) == set(actual), 'INVENTORY/SQLite identities differ')
            for ident, row in actual.items():
                require(all(listed[ident][k] == row[k] for k in TABLES[table]), 'INVENTORY/SQLite source/permission content differs')
        total = 0
        for fid, file in self.files.items():
            self.check()
            physical = 'train' if file['path'] in audit.source['train_paths'] else 'validation'
            require(file['native_file_split'] == split_values[physical], 'unexpected native_file_split value')
            n, lo, hi = self.db.execute('SELECT count(*),min(row_index),max(row_index) FROM rows WHERE file_id=?', (fid,)).fetchone()
            require(n == file['expected_rows'] == audit.summary['native_scan_rows'][file['path']] and
                    lo == 0 and hi == n - 1, 'census coordinate range incomplete')
            total += n
        require(total == 2544892 and all(s['domain'] in DOMAINS for s in self.sources.values()), 'census total/domain registry mismatch')
        self.roster = {}
        self.expected_bad_identity = set()
        with (audit.directory / 'validation-roster.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                coordinate = {'file': row['file'], 'row_index': row['row_index']}
                key = dump(coordinate)
                require(key not in self.roster and row['file'] in audit.source['validation_paths'], 'duplicate/unbound dev roster coordinate')
                meta = self.meta(self.file_by_path[row['file']], row['row_index'])
                require(row['source'] == self.sources[meta['source_id']]['source'], 'original roster/census source mismatch')
                if meta['group_present']:
                    # A valid group missing/mismatching in the sealed graph is
                    # an integrity error, not a reason to silently quarantine.
                    require(row['component'] == meta['old_root'] and row['component_flags'] == meta['old_bits'],
                            'valid original roster/census root or flags mismatch')
                else:
                    # The census deliberately has root=None,bits16 for a bad
                    # group; the old audit can still have a state-only node.
                    # This expected bad-data shape is local quarantine.
                    self.expected_bad_identity.add(key)
                self.roster[key] = row
        require(len(self.roster) == 15550, 'old native-dev roster incomplete')
        require(sum(self.dev_eligible(json.loads(k)) for k in self.roster) == audit.summary['eligible_rows'], 'original eligible policy changed')
        potential = Counter(r['domain'] for r in self.roster.values() if not r['component_flags'] & (1 | 2 | 4 | 16 | 32))
        extra = [r for key, r in self.roster.items() if not r['component_flags'] & (1 | 2 | 4 | 16 | 32)
                 and not self.dev_eligible(json.loads(key))]
        self.identity_gate_stats = {'flags_only_potential_dev_rows': dict(potential),
                                    'flags_only_but_original_identity_rejected_rows': len(extra),
                                    'matches_actual149_missing_identity_reference': len(extra) == 149 and
                                        all('missing_original_identity' in r['rejection_reasons'] for r in extra),
                                    'original_eligible_rows': audit.summary['eligible_rows']}
        self.identity_gate_stats['expected_missing_group_coordinates_quarantined'] = len(self.expected_bad_identity)

    def meta(self, file_id, index):
        row = self.db.execute('SELECT * FROM rows WHERE file_id=? AND row_index=?', (file_id, index)).fetchone()
        require(row is not None, 'raw coordinate missing from complete census')
        return dict(row)

    def dev_eligible(self, coordinate):
        row = self.roster.get(dump(coordinate))
        return bool(dump(coordinate) not in self.expected_bad_identity and row and row.get(self.cfg['roster_eligible_key']) is True and row['state_present'] and
                    row['native_group_present'] and not row['rejection_reasons'])

    def validate_raw(self, coordinate, raw, meta):
        source = self.sources[meta['source_id']]
        require((raw['source'] or '') == source['source'], 'raw/census source mismatch')
        stored = raw['split'] if self.cfg['declared_split_storage'] == 'raw' else str(raw['split'] or '').strip().casefold()
        require(stored == meta['declared_split'], 'raw/census declared split mismatch')
        group = raw['group_id']
        present = bool(source['family']) and group is not None and not isinstance(group, bool) and bool(str(group).strip())
        require(present == bool(meta['group_present']), 'raw/census native group presence differs')
        root = meta['old_root']
        if root is not None:
            require(self.audit.root(root) == root and self.audit.bits(root) == meta['old_bits'], 'census root/flags not final sealed-root values')
            if present:
                require(self.audit.native_root(raw) == root, 'raw source/group/state root differs from census')
            elif state_text(raw['state']):
                require(self.audit.lookup('state', state_text(raw['state'])) == root, 'identity-poor raw state differs from census')
        if 'kind' in raw:
            permission = self.permissions[meta['permission_id']]
            require(raw['kind'] == meta['kind'] and permission['source_id'] == meta['source_id'] and
                    raw['license'] == permission['license'] and raw['license_use'] == permission['license_use'], 'raw label-kind/permission metadata differs')
        key = dump(coordinate)
        if key in self.roster:
            row = self.roster[key]
            require(row['id'] == raw['id'] and row['state_present'] == bool(state_text(raw['state'])) and
                    row['native_group_present'] == present, 'raw dev identity differs from original roster')

    def iter_payload(self, indices_for_file, columns, batch_size):
        """Read only selected row groups; convert only selected original rows.

        Arrow necessarily reads a selected row-group's projected column pages;
        this is NOT a claim of byte-level random row I/O.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        for fid in sorted(self.files):
            indices = indices_for_file(fid)
            if not indices:
                continue
            file = self.files[fid]
            p = pq.ParquetFile(file['path'])
            required = set(columns) - {'example_id'}
            require(required <= set(p.schema_arrow.names) and p.metadata.num_rows == file['expected_rows'], 'raw parquet schema/count changed')
            actual_columns = [name for name in columns if name in p.schema_arrow.names]
            group_start = 0
            for rg in range(p.num_row_groups):
                size = p.metadata.row_group(rg).num_rows
                lo, hi = bisect_left(indices, group_start), bisect_left(indices, group_start + size)
                if lo < hi:
                    offset = group_start
                    for batch in p.iter_batches(batch_size=batch_size, row_groups=[rg], columns=actual_columns, use_threads=False):
                        self.check()
                        a, b = bisect_left(indices, offset, lo, hi), bisect_left(indices, offset + batch.num_rows, lo, hi)
                        for index in indices[a:b]:
                            raw = {name: batch.column(j)[index - offset].as_py() for j, name in enumerate(actual_columns)}
                            coordinate = {'file': file['path'], 'row_index': index}
                            meta = self.meta(fid, index)
                            self.validate_raw(coordinate, raw, meta)
                            yield coordinate, raw, meta
                        offset += batch.num_rows
                    require(offset == group_start + size, 'selected row-group decode incomplete')
                group_start += size

    def all_dev_inputs(self):
        validation = self.cfg['native_file_split_values']['validation']
        return self.iter_payload(lambda fid: range(self.files[fid]['expected_rows'])
                                 if self.files[fid]['native_file_split'] == validation else [],
                                 INPUT_COLUMNS, self.cfg['limits']['arrow_batch_rows'])
