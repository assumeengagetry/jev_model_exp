"""Draft CPU-only semantic membership overlay; existing audit is always read-only.

No automatic source rules are supplied: an unreviewed input layout is unknown,
not a string-leaf fallback. See SCHEMA.md before integrating or executing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import mmap
import os
from pathlib import Path
import re
import sqlite3
import struct
import sys

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get('CLEAN_ORIGIN_ROOT', '/tmp/opencode/jev-clean-origin-20261112-local'))
if ROOT.parent != Path('/tmp/opencode') or not ROOT.name.startswith('jev-clean-origin-20261112-'):
    raise RuntimeError('Unexpected private clean-origin namespace')
AUDIT = Path('/tmp/opencode/jev-06b-research-20261112-validation-audit-v2')
HISTORY, PUBLIC, FORBIDDEN, NATIVE_DEV, INCOMPLETE, UNKNOWN = 1, 2, 4, 8, 16, 32
DOMAINS = ('knowledge', 'language', 'retrieval', 'tools', 'arts')
EXPECTED = {'native_train': 2529342, 'native_dev': 15550,
            'history_task': 818069, 'history_generic': 416934, 'public': 140620}
ORIGIN_BITS = {'native_train': 0, 'native_dev': NATIVE_DEV,
               'history_task': HISTORY, 'history_generic': HISTORY, 'public': PUBLIC}
GENERIC = {'user', 'assistant', 'system', 'string', 'object', 'array', 'number',
           'integer', 'boolean', 'null', 'true', 'false', 'yes', 'no', 'yes.', 'no.'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def stat(path):
    path = Path(path)
    s = path.stat()
    require(path.is_file(), 'not a regular input file: ' + str(path))
    return {'path': str(path), 'resolved': str(path.resolve()), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def verify(records):
    for item in records:
        require(stat(item['path']) == item, 'source metadata changed: ' + item['path'])


def owned(path):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()), 'write outside authorized namespace')
    require(not path.is_symlink(), 'symlink output rejected')
    return path


def write_json(path, value):
    with owned(path).open('x', encoding='utf-8') as handle:
        handle.write(dump(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def family(source):
    # Exact old audit rule, NOT a new source alias registry.
    value = re.sub(r'(?i)(?:[/_:.-](?:train|training|test|dev|valid|validation))$', '', str(source).strip()).casefold()
    for name in ('weblinx', 'mind2web', 'websrc'):
        if value == name or value.startswith(name + '/'):
            return name
    return value


def state_text(value):
    if value is None:
        return ''
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


class AuditGraph:
    """Actual v2 schema. No path compression or writes to the sealed graph."""
    def __init__(self, directory=AUDIT):
        self.directory = Path(directory)
        require(sys.byteorder == 'little', 'audit array(q) byte order requires confirmation on non-little-endian hosts')
        self.receipt = json.loads((self.directory / 'COMPLETE.json').read_text())
        self.summary = json.loads((self.directory / 'SUMMARY.json').read_text())
        self.source = json.loads((self.directory / 'SOURCE_BINDING.json').read_text())
        require(self.receipt['complete'] is True and self.summary['complete_original_scan'] is True,
                'original audit is incomplete')
        require(self.summary['graph_nodes'] == 5561805 and self.summary['validation_rows'] == 15550,
                'wrong original audit inventory')
        self.bound = self.source['files'] + self.receipt['files'] + [self.receipt['source_binding'], self.receipt['code_binding']]
        verify(self.bound)
        self.handles = [(self.directory / name).open('rb') for name in ('parent-int64.bin', 'flags-uint8.bin')]
        self.parents, self.flags = [mmap.mmap(h.fileno(), 0, access=mmap.ACCESS_READ) for h in self.handles]
        self.n = self.summary['graph_nodes']
        require(len(self.parents) == 8 * self.n and len(self.flags) == self.n, 'parent/flag byte size mismatch')
        self.db = sqlite3.connect((self.directory / 'case-graph.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
        self.db.execute('PRAGMA query_only=ON')
        require([r[1] for r in self.db.execute('PRAGMA table_info(keys)')] == ['kind', 'value', 'node'],
                'unexpected audit SQLite schema')

    def root(self, node):
        require(type(node) is int and 0 <= node < self.n, 'invalid graph node')
        for _ in range(128):
            parent = struct.unpack_from('<q', self.parents, 8 * node)[0]
            require(0 <= parent < self.n, 'parent out of range')
            if parent == node:
                return node
            node = parent
        raise RuntimeError('parent cycle/depth exceeds audited bound')

    def bits(self, node):
        return self.flags[self.root(node)]  # NEVER flags[child].

    def lookup(self, kind, value):
        row = self.db.execute('SELECT node FROM keys WHERE kind=? AND value=?', (kind, value)).fetchone()
        require(row is not None, 'original identity missing from sealed graph')
        return self.root(row[0])

    def native_root(self, row):
        f, group = family(row['source']), row['group_id']
        require(f and group is not None and not isinstance(group, bool) and str(group).strip(), 'missing native identity')
        key = json.dumps([f, str(group)], ensure_ascii=False, separators=(',', ':'))
        result = self.lookup('native_group', key)
        text = state_text(row['state'])
        if text:
            require(self.lookup('state', text) == result, 'state/native-group roots disagree')
        return result

    def close(self):
        self.db.close()
        self.parents.close()
        self.flags.close()
        for handle in self.handles:
            handle.close()
        verify(self.bound)


def at_path(value, path):
    """Explicit source-reviewed paths only; no recursive scalar-leaf mining."""
    for part in path:
        if part == '$json':
            require(isinstance(value, str), 'JSON decode requested on nonstring')
            value = json.loads(value)
        elif isinstance(part, str) and isinstance(value, dict):
            value = value[part]
        elif type(part) is int and isinstance(value, list):
            value = value[part]
        else:
            raise KeyError('source path mismatch')
    return value


def request_anchors(provided, profile):
    """Same canonical task-request space on both sides, with no gold access.

    A profile supplies alternative equivalent extraction paths. They are not
    arbitrary pieces of one request. Distinct successful values are blocked.
    A composite source requires its own reviewed adapter, not concatenation here.
    """
    if not profile or profile.get('reviewed') is not True or not profile.get('evidence'):
        return [], 'unreviewed_source_input_layout'
    values = set()
    for path in profile.get('alternative_request_paths', []):
        try:
            value = at_path(provided, path)
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError):
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if value.casefold() in GENERIC:
            continue
        # A serialized container is not automatically a task request. Require
        # an explicit $json/request path, so schema descriptions cannot slip in.
        try:
            container = json.loads(value)
        except ValueError:
            container = None
        if isinstance(container, (dict, list)):
            continue
        values.add(value)
    if len(values) != 1:
        return [], 'ambiguous_request_paths' if values else 'no_verified_task_request_anchor'
    return [dump(['task-request-text-v1', values.pop()])], None


class Membership:
    """New private SQLite overlay. Only verified request equality joins roots.

    Public/history records seed each verified anchor; their input leaf values
    never get unioned. All native views of an old root share every taint.
    """
    def __init__(self, path, max_roots_per_anchor=512):
        path = owned(path)
        require(not path.exists(), 'existing overlay preserved; no adoption')
        self.db = sqlite3.connect(path)
        self.db.executescript('''
            PRAGMA journal_mode=DELETE;
            PRAGMA temp_store=MEMORY;
            PRAGMA cache_size=-16384;
            CREATE TABLE roots(root INTEGER PRIMARY KEY, parent INTEGER NOT NULL, bits INTEGER NOT NULL);
            CREATE TABLE anchors(value TEXT PRIMARY KEY, first_root INTEGER, bits INTEGER NOT NULL) WITHOUT ROWID;
            CREATE TABLE members(value TEXT, root INTEGER, PRIMARY KEY(value,root)) WITHOUT ROWID;
            CREATE TABLE coverage(origin TEXT, event_id TEXT, root INTEGER, reason TEXT,
                                  PRIMARY KEY(origin,event_id)) WITHOUT ROWID;
        ''')
        self.max_roots_per_anchor = max_roots_per_anchor

    def root(self, node):
        visited = []
        for _ in range(128):
            row = self.db.execute('SELECT parent,bits FROM roots WHERE root=?', (node,)).fetchone()
            require(row is not None, 'overlay root missing')
            if row[0] == node:
                for child in visited:
                    self.db.execute('UPDATE roots SET parent=? WHERE root=?', (node, child))
                return node, row[1]
            visited.append(node)
            node = row[0]
        raise RuntimeError('overlay parent depth/cycle requires review')

    def mark(self, root, bits):
        root, old = self.root(root)
        self.db.execute('UPDATE roots SET bits=? WHERE root=?', (old | bits, root))

    def join(self, a, b):
        a, fa = self.root(a)
        b, fb = self.root(b)
        # Min-root yields stable IDs independent of event order; no mutation of
        # the original parent mmap. Deep chains fail closed rather than guess.
        if a != b:
            lo, hi = sorted((a, b))
            self.db.execute('UPDATE roots SET parent=? WHERE root=?', (lo, hi))
            self.db.execute('UPDATE roots SET bits=? WHERE root=?', (fa | fb, lo))

    def observe(self, origin, event_id, root, old_bits, anchors, reason):
        bits = ORIGIN_BITS[origin]
        self.db.execute('INSERT INTO coverage VALUES(?,?,?,?)', (origin, event_id, root, reason))
        if root is not None:
            self.db.execute('INSERT OR IGNORE INTO roots VALUES(?,?,?)', (root, root, old_bits))
            self.mark(root, bits | (UNKNOWN if reason else 0))
        for value in anchors:
            self.db.execute('INSERT OR IGNORE INTO anchors VALUES(?,NULL,0)', (value,))
            first, seeds = self.db.execute('SELECT first_root,bits FROM anchors WHERE value=?', (value,)).fetchone()
            self.db.execute('UPDATE anchors SET bits=? WHERE value=?', (seeds | bits, value))
            if root is not None:
                if first is None:
                    self.db.execute('UPDATE anchors SET first_root=? WHERE value=?', (root, value))
                else:
                    self.join(first, root)
                self.db.execute('INSERT OR IGNORE INTO members VALUES(?,?)', (value, root))
                n = self.db.execute('SELECT count(*) FROM members WHERE value=?', (value,)).fetchone()[0]
                require(n <= self.max_roots_per_anchor, 'anchor fanout exceeds predeclared review bound; no silent truncation')

    def finish(self):
        for _, first, seeds in self.db.execute('SELECT value,first_root,bits FROM anchors'):
            if first is not None:
                self.mark(first, seeds)
        # Flatten overlay only; this cannot change sealed audit parents/flags.
        for (node,) in self.db.execute('SELECT root FROM roots ORDER BY root'):
            root, _ = self.root(node)
            self.db.execute('UPDATE roots SET parent=? WHERE root=?', (root, node))
        self.db.commit()
        counts = dict(self.db.execute('SELECT origin,count(*) FROM coverage GROUP BY origin'))
        unknown = dict(self.db.execute('SELECT origin,count(*) FROM coverage WHERE reason IS NOT NULL GROUP BY origin'))
        blockers = [f'{key}: expected {count}, got {counts.get(key, 0)}' for key, count in EXPECTED.items() if counts.get(key, 0) != count]
        # A failed public/history projection is an unbounded exclusion hole.
        # Native layout failures instead quarantine whole connected components.
        blockers += [f'{key}: {count} unmapped exclusion requests' for key, count in unknown.items()
                     if key in ('public', 'history_task', 'history_generic')]
        return {'counts': counts, 'unknown_layout_counts': unknown, 'blockers': blockers,
                'complete_event_inventory': not blockers, 'DATA_READY': False,
                'scope': 'exact reviewed task-request membership only; no near-duplicate or pretraining claim'}

    def seal_candidates(self):
        """Bounded-driver path, NOT the legacy all-origin event contract."""
        for first, seeds in self.db.execute('SELECT first_root,bits FROM anchors'):
            if first is not None:
                self.mark(first, seeds)
        for (node,) in self.db.execute('SELECT root FROM roots ORDER BY root'):
            root, _ = self.root(node)
            self.db.execute('UPDATE roots SET parent=? WHERE root=?', (root, node))
        self.db.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', required=True, type=Path)
    parser.add_argument('--profiles', required=True, type=Path)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation must hide CUDA')
    os.umask(0o077)
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    cpus = sorted(os.sched_getaffinity(0))
    require(cpus, 'no CPU affinity available')
    os.sched_setaffinity(0, set(cpus[-4:]))
    stage = owned(args.stage)
    stage.mkdir(mode=0o700, exist_ok=False)
    inputs = [stat(args.events), stat(args.profiles)]
    profiles = json.loads(args.profiles.read_text())
    audit = AuditGraph()
    overlay = Membership(stage / 'membership.sqlite3')
    try:
        with args.events.open(encoding='utf-8') as handle:
            for i, line in enumerate(handle):
                row = json.loads(line)
                origin = row['origin']
                require(origin in EXPECTED and isinstance(row['event_id'], str), 'invalid semantic event')
                require(isinstance(row['source'], str), 'event source missing')
                native = origin in ('native_train', 'native_dev')
                # Root is reconstructed from provided state + source/group, not
                # trusted from an external child-flags field or dedup ID.
                if native:
                    coordinate = row['coordinate']
                    name, index = coordinate['file'], coordinate['row_index']
                    inventory = audit.source['train_paths' if origin == 'native_train' else 'validation_paths']
                    require(name in inventory and type(index) is int and
                            0 <= index < audit.summary['native_scan_rows'][name], 'native coordinate outside original inventory')
                    require(row['event_id'] == dump([name, index]), 'native event ID must be its exact raw coordinate')
                    require(row['native_identity']['source'] == row['source'] and
                            row['provided']['state'] == row['native_identity']['state'], 'source/state identity mismatch')
                    root = audit.native_root(row['native_identity'])
                else:
                    root = None
                profile = profiles.get(origin + ':' + row['source'])
                anchors, reason = request_anchors(row['provided'], profile)
                overlay.observe(origin, row['event_id'], root, audit.bits(root) if native else 0, anchors, reason)
                if i % 4096 == 0:
                    overlay.db.commit()
        result = overlay.finish()
        verify(inputs)
        result['inputs'] = inputs
        result['projection_review_verified'] = False
        result['source_review_and_origin_projection_receipts_required'] = True
        write_json(stage / 'SEMANTIC_DRAFT.json', result)
        require(not result['blockers'], 'semantic coverage blocked; inspect private draft receipt')
    finally:
        overlay.db.close()
        audit.close()


if __name__ == '__main__':
    main()
