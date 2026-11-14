"""Reviewed native inputs and independent conservative exclusion seeds.

No targets/gold are accepted by this interface. Nonreviewed source layouts are
quarantined. Public/old-graph seed strings are NEVER unioned with one another.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

from semantic_taint import GENERIC, PUBLIC, NATIVE_DEV, UNKNOWN, dump, require, state_text, at_path


def task_text(value):
    if not isinstance(value, str) or not value.strip() or value.strip().casefold() in GENERIC:
        return None
    text = value.strip()
    try:
        decoded = json.loads(text)
    except ValueError:
        decoded = None
    return None if isinstance(decoded, (dict, list)) else text


def anchor(text):
    return dump(['task-request-text-v1', text])


def json_anchor(value):
    """Canonical COMPLETE reviewed provided-state object, never its leaf words."""
    return dump(['task-provided-json-object-v1', value])


def decoded_state(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def matching_object(value, profile):
    """An exact reviewed whole-object shape; unknown keys never get dropped."""
    if not isinstance(value, dict):
        return None
    required = set(profile['required_state_keys'])
    allowed = required | set(profile.get('optional_state_keys', []))
    if not required <= set(value) or not set(value) <= allowed:
        return None
    try:
        json_anchor(value)  # Reject nonfinite/non-JSON provided values.
    except (ValueError, TypeError):
        return None
    return value


def from_object(value, profile):
    """Reviewed textual request paths only, not a generic leaf extractor."""
    value = matching_object(value, profile)
    if value is None:
        return None
    values = set()
    for path in profile['request_paths']:
        try:
            text = task_text(at_path(value, path))
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError):
            continue
        if text is not None:
            values.add(text)
    return values.pop() if len(values) == 1 else None


def validate_profile(profile):
    require(profile.get('reviewed') is True and profile.get('input_scope_complete') is True,
            'source task-input scope is not reviewed complete')
    require(isinstance(profile.get('shape_id'), str) and profile['shape_id'] and profile.get('evidence_files'),
            'each positive native shape needs its own named gate and actual evidence')
    require(isinstance(profile.get('native_variants'), list) and profile['native_variants'] and
            '*' not in profile['native_variants'], 'positive shape must explicitly name reviewed native variants')
    mode = profile.get('mode')
    require(mode in ('state-text-v1', 'tool-request-v1', 'reviewed-json-request-path-v1',
                     'state-json-object-v1'), 'unsupported input profile mode')
    if mode == 'tool-request-v1':
        require(profile.get('required_state_keys') == ['request', 'available_tools'] and
                profile.get('request_paths') == [['request']], 'tool request profile changed known converter structure')
    if mode != 'state-text-v1':
        require(isinstance(profile.get('required_state_keys'), list) and profile['required_state_keys'] and
                all(isinstance(k, str) for k in profile['required_state_keys']), 'missing exact JSON shape')
        if mode == 'state-json-object-v1':
            require('request_paths' not in profile, 'whole-state JSON must not silently select a partial request path')
        else:
            require(profile.get('request_paths') and all(isinstance(p, list) and p for p in profile['request_paths']),
                    'missing reviewed request paths')
        # Gold cannot be reached: the caller passes state only. Candidate
        # profile evidence must still establish that this is task input.
    return profile


def native_anchors(raw_input, profile):
    require(set(raw_input) == {'state', 'question'}, 'projector must receive input-only state/question, not labels')
    if not profile:
        return [], 'unreviewed_source_profile'
    validate_profile(profile)
    if profile['mode'] == 'state-json-object-v1':
        value = matching_object(decoded_state(raw_input['state']), profile)
        return ([json_anchor(value)], None) if value is not None else ([], 'unreviewed_json_provided_state_shape')
    if profile['mode'] == 'state-text-v1':
        text = task_text(raw_input['state'])
    else:
        text = from_object(decoded_state(raw_input['state']), profile)
    if text is None:
        return [], 'input_shape_or_task_anchor_not_supported'
    return [anchor(text)], None


def native_projection(raw_input, source_review, provenance):
    """Exactly one reviewed shape must match, not a source-wide assumed fallback."""
    matches = []
    for profile in source_review['input_profiles']:
        if provenance.get('variant') not in profile['native_variants']:
            continue
        anchors, reason = native_anchors(raw_input, profile)
        if reason is None:
            matches.append((anchors, profile))
    if len(matches) != 1:
        return [], 'ambiguous_positive_shape_gates' if matches else 'unreviewed_or_unmatched_positive_shape', None
    return matches[0][0], None, matches[0][1]


class ExclusionSeeds:
    """Match the bounded candidate-anchor inventory while streaming ALL seeds.

    No5.85M-text copy is created. Ban membership can overexclude; it is not case
    equality between public strings and does not merge arbitrary provided words.
    """
    def __init__(self, membership, profiles, check):
        self.membership, self.check = membership, check
        self.profiles = []
        seen = set()
        for profile in profiles:
            validate_profile(profile)
            if profile['mode'] != 'state-text-v1':
                key = dump(profile)
                if key not in seen:
                    seen.add(key)
                    self.profiles.append(profile)
        self.counts = Counter()
        self.conversions = Counter()
        self.received = 0
        self.membership.db.execute('''CREATE TABLE native_closure(
            raw_state TEXT,anchor TEXT,state_node INTEGER,old_root INTEGER,old_bits INTEGER,
            projected_member INTEGER,candidate_root INTEGER,
            PRIMARY KEY(raw_state,anchor)) WITHOUT ROWID''')

    def representations(self, value, origin):
        text = state_text(value)
        values = set()
        if text and text.casefold() not in GENERIC:
            values.add(text)  # Full provided representation, ban-only, no union.
            self.conversions[origin + ':full_provided_representation_kept'] += 1
        else:
            self.conversions[origin + ':empty_or_generic_representation_skipped'] += 1
        obj = decoded_state(value)
        # Known old converter: symmetric request unwrap regardless of source.
        if isinstance(obj, dict) and {'request', 'available_tools'} <= set(obj):
            self.conversions[origin + ':known_tool_wrapper_seen'] += 1
            request = task_text(obj['request'])
            if request:
                values.add(request)
                self.conversions[origin + ':known_tool_wrapper_unwrapped'] += 1
            else:
                self.conversions[origin + ':known_tool_wrapper_invalid_or_generic_request'] += 1
        elif isinstance(obj, dict):
            self.conversions[origin + ':other_json_object_full_representation_only'] += 1
        else:
            self.conversions[origin + ':non_json_object_representation'] += 1
        keys = {anchor(v) for v in values}
        for profile in self.profiles:
            if profile['mode'] == 'state-json-object-v1':
                complete = matching_object(obj, profile)
                if complete is not None:
                    keys.add(json_anchor(complete))
                    self.conversions[origin + ':reviewed_complete_json_state_canonicalized'] += 1
            else:
                request = from_object(obj, profile)
                if request:
                    keys.add(anchor(request))
                    self.conversions[origin + ':reviewed_json_shape_alias_unwrapped'] += 1
        return sorted(keys)

    def observe(self, value, bits, origin):
        self.counts[origin + ':representations_received'] += 1
        self.received += 1
        if self.received % 512 == 0:
            self.check()
        matched = []
        for key in self.representations(value, origin):
            row = self.membership.db.execute('SELECT first_root,bits FROM anchors WHERE value=?', (key,)).fetchone()
            if row is not None:
                self.membership.db.execute('UPDATE anchors SET bits=? WHERE value=?', (row[1] | bits, key))
                self.counts[origin + ':candidate_membership_hits'] += 1
                matched.append((key,row[0]))
        return matched

    def public_view(self, path):
        requests = texts = 0
        with Path(path).open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                require(set(row) == {'texts'} and isinstance(row['texts'], list) and row['texts'], 'invalid bound public input-only view')
                for text in row['texts']:
                    require(isinstance(text, str) and bool(text), 'invalid public provided text')
                    self.observe(text, PUBLIC, 'public')
                    texts += 1
                requests += 1
                if requests % 256 == 0:
                    self.check()
                    self.membership.db.commit()
        require((requests, texts) == (140620, 5854755), 'public provided-input seed inventory incomplete')
        return {'requests': requests, 'text_entries': texts, 'gold_read': False,
                'semantics': 'independent conservative ban seeds, never a public leaf union'}

    def old_graph(self, audit, census):
        scanned = flagged = 0
        for value, node in audit.db.execute("SELECT value,node FROM keys WHERE kind='state'"):
            root = audit.root(node)
            bits = audit.bits(root)
            scanned += 1
            if bits:
                # Includes every old selected-history state, plus conservatively
                # connected siblings. HISTORY alone still does not veto train.
                matched = self.observe(value, bits, 'old_graph_root_taint_superset')
                flagged += 1
            else:
                # Unflagged native roots must participate in local recognized
                # semantic closure, even when outside metadata/candidate caps.
                matched = self.observe(value, 0, 'old_graph_unflagged_native_closure')
            if matched and census.db.execute('SELECT 1 FROM rows WHERE old_root=? LIMIT 1',(root,)).fetchone() is not None:
                for key,first in matched:
                    require(first is not None,'candidate anchor lacks a native owner')
                    member = self.membership.db.execute('SELECT 1 FROM members WHERE value=? AND root=?',(key,root)).fetchone() is not None
                    if not member:
                        # Do not silently restore/sample an outside sibling or
                        # assume its source/shape/permission is reviewed. Veto
                        # the entire affected candidate component instead.
                        current = self.membership.db.execute('SELECT bits FROM anchors WHERE value=?',(key,)).fetchone()[0]
                        self.membership.db.execute('UPDATE anchors SET bits=? WHERE value=?',(current|UNKNOWN|16,key))
                        if self.membership.db.execute('SELECT 1 FROM roots WHERE root=?',(root,)).fetchone() is not None:
                            self.membership.mark(root,UNKNOWN|16)
                        self.counts['native_closure:unrestored_sibling_veto_observations'] += 1
                    self.membership.db.execute('INSERT INTO native_closure VALUES(?,?,?,?,?,?,?)',
                        (value,key,node,root,bits,int(member),first))
                    self.counts['native_closure:matched_native_state_anchor_pairs'] += 1
            if scanned % 1024 == 0:
                self.check()
                self.membership.db.commit()
        return {'state_keys_read': scanned, 'root_flagged_state_keys': flagged,
                'complete_state_key_cursor_exhausted': True,
                'unflagged_state_keys_checked_for_native_closure':scanned-flagged,
                'recognized_native_closure_pairs':self.membership.db.execute('SELECT count(*) FROM native_closure').fetchone()[0],
                'outside_or_unprojected_native_siblings_veto_complete_component':True,
                'selected_history_scope': 'all sealed old HISTORY states plus root-connected conservative superset',
                'old_graph_rebuilt': False}

    def native_dev(self, census, loader):
        count = 0
        for coordinate, raw, meta in loader:
            eligible = census.dev_eligible(coordinate)
            bits = NATIVE_DEV | (0 if eligible else UNKNOWN)
            if meta['old_root'] is not None:
                bits |= census.audit.bits(meta['old_root'])
            self.observe(raw['state'], bits, 'native_dev_state')
            self.observe(raw['question'], bits, 'native_dev_question')
            count += 1
        require(count == 15550, 'native-dev input seed projection incomplete')
        return {'native_dev_rows': count, 'fields_read': ['state', 'question', 'source', 'id', 'group_id', 'split'],
                'ineligible_roster_siblings_seed_UNKNOWN': True, 'gold_read': False}
