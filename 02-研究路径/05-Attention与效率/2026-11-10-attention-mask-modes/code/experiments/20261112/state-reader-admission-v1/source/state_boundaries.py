"""CPU/JSON-only, fail-closed reader for the lossless v2 state-cut sidecars."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import stat as stat_mode

from bindings import DOMAINS, read_json, require, source_bytes, stat, verify


VERSION = 'base-state-boundary-fixed-separator-v2'
POLICY = 'state-plus-one-existing-newline-v2'
ROOT = Path('/tmp/opencode/jev-base-state-boundary-20261112-v2/prepared-01')
PRIMARY = ROOT.parent / 'FINAL_PRIMARY_VERIFICATION.json'
SPLITS = {'train': 'train', 'select': 'validation'}
CASE_COUNTS = {'train': 24600, 'select': 375}
COUNT_KEYS = {'cases', 'fields', 'known_fields', 'exact_state_boundary',
              'complete_candidates', 'logical_pair_tokens_once', 'masked_unknown_no_forward'}
RECORD_KEYS = {'version', 'boundary_policy', 'roster', 'native_split', 'case_ordinal',
               'case_id', 'component', 'primary_domain', 'known_mask', 'fields'}
FIELD_KEYS = {'field_index', 'coordinate', 'known', 'state_prefix_tokens',
              'original_candidate_prefix_tokens', 'candidate_count', 'pair_lengths',
              'status', 'reason'}
PRIMARY_CLAIMS = (
    'independent_full_prefix_ids_and_exact_character_offset_cut_verified',
    'full_known_candidate_lengths_EOS_and_available_rows_reconstruction_verified',
    'every_unknown_preserved_without_tokenization_or_candidate_reinterpretation',
    'source_config_data_outputs_stat_and_source_bytes_verified',
)


def _pin(value, expected=None):
    require(isinstance(value, dict) and
            set(value) == {'path', 'resolved', 'bytes', 'mtime_ns'}, 'Bad boundary stat pin')
    require(all(isinstance(value[k], str) and Path(value[k]).is_absolute()
                for k in ('path', 'resolved')) and
            all(type(value[k]) is int and value[k] >= 0 for k in ('bytes', 'mtime_ns')),
            'Invalid boundary stat values')
    if expected is not None:
        require(value['path'] == str(expected), 'Boundary pin redirected: ' + str(expected))
    verify([value])
    return value


def _pins(values):
    require(isinstance(values, list) and values, 'Nonempty boundary pin list required')
    indexed = {}
    for value in values:
        _pin(value)
        require(value['path'] not in indexed, 'Duplicate boundary pin: ' + value['path'])
        indexed[value['path']] = value
    return indexed


def _merge_pins(*groups):
    indexed = {}
    for group in groups:
        for value in group:
            prior = indexed.get(value['path'])
            require(prior is None or prior == value, 'Conflicting boundary stat pins')
            indexed[value['path']] = value
    return [indexed[name] for name in sorted(indexed)]


def _inventory(root):
    require(root == ROOT and root.is_dir() and root.resolve() == root and
            not root.is_symlink() and not root.parent.is_symlink(), 'Boundary root redirected')
    require(not list(root.parent.glob('BLOCKED*')), 'Boundary parent is blocked')
    files = set()
    for path in root.rglob('*'):
        require(not path.is_symlink() and not path.name.startswith('BLOCKED'),
                'Blocked or symlink boundary output: ' + str(path))
        if path.is_dir():
            continue
        require(stat_mode.S_ISREG(path.stat().st_mode), 'Nonregular boundary output: ' + str(path))
        if path != root / 'COMPLETE.json':
            files.add(str(path))
    return files


def _same(left, right):
    """JSON identity without Python's bool/int or int/float equality aliases."""
    return json.dumps(left, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _versioned(value, name):
    require(isinstance(value, dict) and value.get('version') == VERSION and
            value.get('boundary_policy') == POLICY, name + ' version/policy differs')


def _report_counts(report):
    require(isinstance(report['counts'], dict) and set(report['counts']) == set(SPLITS) and
            isinstance(report['primary_domain_cases'], dict) and
            set(report['primary_domain_cases']) == set(SPLITS), 'Boundary report rosters differ')
    readback = {}
    for roster, expected in CASE_COUNTS.items():
        count = report['counts'][roster]
        domain = report['primary_domain_cases'][roster]
        require(isinstance(count, dict) and set(count) == COUNT_KEYS and
                all(type(value) is int and value >= 0 for value in count.values()) and
                count['cases'] == expected and count['known_fields'] > 0 and
                count['exact_state_boundary'] == count['known_fields'] and
                count['fields'] == count['known_fields'] + count['masked_unknown_no_forward'],
                'Boundary report accounting differs: ' + roster)
        require(isinstance(domain, dict) and set(domain) == set(DOMAINS) and
                all(type(value) is int and value > 0 for value in domain.values()) and
                sum(domain.values()) == expected and
                (roster != 'select' or all(value == 75 for value in domain.values())),
                'Boundary report domains differ: ' + roster)
        readback[roster + '_cases'] = count['cases']
        readback[roster + '_fields'] = count['fields']
    require(_same(report['readback'], readback), 'Boundary report readback accounting differs')


def _guard(binding):
    verify(binding['guarded_pins'])
    require(_inventory(Path(binding['root'])) ==
            {pin['path'] for pin in binding['files']}, 'Boundary prepared inventory changed')
    require(source_bytes(binding['exact_byte_pins']) == binding['source_bytes'],
            'Boundary producer/analysis/config exact bytes changed')
    verify(binding['guarded_pins'])


def bind_prepared(cfg):
    """Bind actual v2 completion, independent primary evidence and all input identity.

    The returned dictionary is JSON-only and suitable for Context.snapshot. No
    sidecar is admitted on a ready flag alone, and no tokenizer is imported.
    """
    boundary = cfg['state_boundary']
    require(isinstance(boundary, dict) and set(boundary) == {'root', 'complete', 'primary'} and
            isinstance(boundary['root'], str) and boundary['root'] == str(ROOT),
            'Exact v2 state_boundary root/complete/primary configuration required')
    complete_pin = _pin(boundary['complete'], ROOT / 'COMPLETE.json')
    primary_pin = _pin(boundary['primary'], PRIMARY)
    complete = read_json(complete_pin['path'])
    _versioned(complete, 'Boundary COMPLETE')
    require(complete.get('complete') is True and complete.get('ready') is True and
            complete.get('model_run') is False and complete.get('D_confirm_opened') is False,
            'Actual complete/ready CPU-only state-boundary product required')
    files = _pins(complete['files'])
    require(set(files) == _inventory(ROOT), 'COMPLETE must pin every prepared regular file')
    required = {str(ROOT / name) for name in ('PREPARATION.json', 'train-boundaries.jsonl',
                'select-boundaries.jsonl', 'unavailable-known-fields.jsonl')}
    require(required <= set(files) and all(Path(name).is_relative_to(ROOT) for name in files),
            'Missing or redirected prepared boundary artifact')
    report_pin = _pin(files[str(ROOT / 'PREPARATION.json')], ROOT / 'PREPARATION.json')
    producer = _pin(complete['source'], ROOT.parent / 'prepare_boundaries.py')
    config_pin = _pin(complete['config'], ROOT.parent / 'projection-config.json')
    report = read_json(report_pin['path'])
    projection = read_json(config_pin['path'])
    _versioned(report, 'Boundary report')
    _versioned(projection, 'Boundary projection config')
    require(all(report.get(key) is True for key in
                ('complete_scan_and_readback', 'ready',
                 'same_original_pair_tokens_and_lengths_verified',
                 'unknown_raw_not_tokenized_or_forwarded')) and
            all(report.get(key) is False for key in
                ('target_used_for_cut', 'D_confirm_opened', 'model_run', 'public_run')) and
            report.get('unavailable_known_field_reasons') == {},
            'Boundary report lacks complete lossless/unknown/no-target evidence')
    require(report['boundary_contract'] ==
            'unique-exact-token-split-after-state-plus-one-existing-newline; no rounding/filtering/retokenization',
            'Boundary character-offset contract changed')
    require(report['source'] == projection['source'] == producer and
            report['config'] == config_pin and projection.get('approved_for_cpu_projection') is True,
            'Boundary report/config/producer identity differs')
    inputs = _pins(projection['input_pins'])
    require(_same(projection['input_pins'], report['input_pins']),
            'Boundary report and projection input pins differ')
    require(isinstance(cfg['data_root'], str) and Path(cfg['data_root']).is_absolute() and
            projection['data_root'] == cfg['data_root'] and
            projection['data_complete'] == cfg['data_complete'],
            'Boundary projection does not bind the trainer original data')
    data_root = Path(cfg['data_root'])
    data_complete = _pin(cfg['data_complete'], data_root / 'COMPLETE.json')
    require(inputs.get(data_complete['path']) == data_complete,
            'Original data COMPLETE is absent from projection input pins')
    original_files = {}
    sidecars = {}
    for roster in SPLITS:
        original_path = str(data_root / 'data' / (roster + '.jsonl'))
        require(original_path in inputs, 'Original roster missing from projection input pins')
        original_files[roster] = inputs[original_path]
        sidecars[roster] = files[str(ROOT / (roster + '-boundaries.jsonl'))]
    unavailable = files[str(ROOT / 'unavailable-known-fields.jsonl')]
    require(unavailable['bytes'] == 0, 'Unavailable known-field sidecar is not empty')
    _report_counts(report)

    primary = read_json(primary_pin['path'])
    require(isinstance(primary, dict) and primary.get('boundary_policy') == POLICY and
            all(primary.get(key) is True for key in
                ('complete', 'prepared_scan_complete', 'ready', 'allow_new_state_boundary_training') +
                PRIMARY_CLAIMS) and
            all(primary.get(key) is False for key in
                ('D_confirm_opened', 'model_run', 'torch_imported', 'public_run')) and
            primary.get('unavailable_known_field_reasons') == {},
            'Actual independent full primary boundary admission required')
    require(primary['prepared_completion'] == complete_pin and
            primary['producer_source'] == producer and primary['config'] == config_pin and
            _same(primary['actual_original_and_sidecar_counts_verified'], report['counts']) and
            _same(primary['primary_domain_cases'], report['primary_domain_cases']),
            'Primary prepared/source/config/count/domain identity differs')
    analysis = _pin(primary['primary_analysis_source'], ROOT.parent / 'verify_prepared.py')
    exact_byte_pins = [producer, analysis, config_pin]
    encoded = source_bytes(exact_byte_pins)
    require(isinstance(projection['source_bytes'], str) and
            projection['source_bytes'] == report['source_bytes'] == encoded[producer['path']],
            'Projection/report producer exact base64 source bytes differ')
    guarded = _merge_pins([complete_pin, primary_pin, producer, config_pin, analysis, data_complete],
                          complete['files'], projection['input_pins'])
    binding = {
        'version': VERSION, 'boundary_policy': POLICY, 'root': str(ROOT),
        'complete': complete_pin, 'prepared': complete, 'files': complete['files'],
        'report_pin': report_pin, 'report': report, 'primary_pin': primary_pin, 'primary': primary,
        'projection_config_pin': config_pin, 'projection_config': projection,
        'producer_source': producer, 'primary_analysis_source': analysis,
        'data_root': cfg['data_root'], 'data_complete': data_complete,
        'original_files': original_files, 'sidecars': sidecars, 'unavailable': unavailable,
        'input_pins': projection['input_pins'], 'guarded_pins': guarded,
        'exact_byte_pins': exact_byte_pins, 'source_bytes': encoded,
    }
    _guard(binding)
    return deepcopy(binding)


def _record(line):
    require(bool(line), 'Missing boundary sidecar record')

    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate boundary JSON key: ' + key)
            value[key] = item
        return value

    def invalid(value):
        raise ValueError('Nonfinite boundary JSON: ' + value)

    value = json.loads(line, object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict) and set(value) == RECORD_KEYS, 'Bad boundary record schema')
    return value


def _validate_record(case, record, roster, ordinal):
    _versioned(record, 'Boundary record')
    require(record['roster'] == roster and record['native_split'] == case['split'] == SPLITS[roster] and
            type(record['case_ordinal']) is int and record['case_ordinal'] == ordinal and
            isinstance(case['case_id'], str) and record['case_id'] == case['case_id'] and
            _same(record['component'], case['component']) and
            case['case_id'] == f"clean-semantic-{case['component']}" and
            record['primary_domain'] == case['primary_domain'] and case['primary_domain'] in DOMAINS,
            'Boundary original case identity/order/split differs')
    fields, values = case['fields'], record['fields']
    require(isinstance(fields, list) and fields and isinstance(values, list) and
            len(values) == len(fields), 'Boundary lost or added original fields')
    for mask in (case['known_mask'], record['known_mask']):
        require(isinstance(mask, list) and len(mask) == len(fields) and
                all(type(item) is bool for item in mask), 'Invalid boundary known mask')
    require(record['known_mask'] == case['known_mask'] == [field['known'] for field in fields],
            'Boundary known mask differs')
    count = Counter({key: 0 for key in COUNT_KEYS})
    count['cases'], count['fields'] = 1, len(fields)
    cuts = []
    for index, (field, value) in enumerate(zip(fields, values)):
        require(isinstance(value, dict) and set(value) == FIELD_KEYS and
                type(value['field_index']) is int and value['field_index'] == index and
                _same(value['coordinate'], field['coordinate']) and
                type(field['known']) is bool and type(value['known']) is bool and
                value['known'] is field['known'] and value['reason'] is None,
                'Boundary original field coordinate/index/known/reason differs')
        if not field['known']:
            # Deliberately never inspect unknown raw options, targets or text.
            require(value['status'] == 'masked_unknown_no_forward' and
                    all(value[key] is None for key in
                        ('state_prefix_tokens', 'original_candidate_prefix_tokens',
                         'candidate_count', 'pair_lengths')) and
                    field['prefix_ids'] == field['suffix_ids'] == field['pair_lengths'] == [],
                    'Unknown boundary must be null and preserve empty original token rows')
            count['masked_unknown_no_forward'] += 1
            cuts.append(None)
            continue
        prefix, suffixes, lengths = field['prefix_ids'], field['suffix_ids'], field['pair_lengths']
        require(isinstance(prefix, list) and prefix and
                all(type(token) is int and token >= 0 for token in prefix) and
                isinstance(suffixes, list) and suffixes and isinstance(field['options'], list) and
                isinstance(lengths, list) and len(lengths) == len(suffixes) == len(field['options']) and
                all(type(length) is int and length > 0 for length in lengths),
                'Invalid original known candidate rows')
        cut = value['state_prefix_tokens']
        require(value['status'] == 'exact_state_boundary' and type(cut) is int and
                0 < cut < len(prefix) and type(value['original_candidate_prefix_tokens']) is int and
                value['original_candidate_prefix_tokens'] == len(prefix) and
                type(value['candidate_count']) is int and value['candidate_count'] == len(suffixes) and
                _same(value['pair_lengths'], lengths), 'Known boundary/candidate geometry differs')
        new_prefix = prefix[:cut]
        for length, suffix in zip(lengths, suffixes):
            require(isinstance(suffix, list) and suffix and suffix[-1] == 151645 and
                    all(type(token) is int and token >= 0 for token in suffix) and
                    len(prefix) + len(suffix) == length,
                    'Known boundary candidate IDs/EOS/pair length differs')
            new_suffix = prefix[cut:] + suffix
            require(new_prefix + new_suffix == prefix + suffix and
                    len(new_prefix) + len(new_suffix) == length,
                    'Boundary cut changed original complete row IDs or length')
        count['known_fields'] += 1
        count['exact_state_boundary'] += 1
        count['complete_candidates'] += len(suffixes)
        count['logical_pair_tokens_once'] += sum(lengths)
        cuts.append(cut)
    return cuts, count


class BoundaryStore:
    """All original cases checked once; RAM keeps sidecar offsets/ordinals only.

    Caller-owned original CaseStores are never closed here. Each for_case call
    returns a fresh full-field cut list, including independently repeated cases.
    """

    def __init__(self, binding, train, select):
        self.handles, self.index, self._closed = {}, {}, False
        try:
            self.binding = deepcopy(binding)
            _versioned(self.binding, 'Boundary binding')
            self.check()
            for roster, store in (('train', train), ('select', select)):
                original_pin = self.binding['original_files'][roster]
                require(store.split == SPLITS[roster] and str(store.path) == original_pin['path'] and
                        store.pin == original_pin and stat(store.path) == original_pin and
                        len(store.offsets) == CASE_COUNTS[roster] ==
                        self.binding['report']['counts'][roster]['cases'],
                        'Boundary original store path/split/actual roster count differs')
                handle = Path(self.binding['sidecars'][roster]['path']).open('rb')
                self.handles[roster], self.index[roster] = handle, {}
                self.check()
                count, domains = Counter({key: 0 for key in COUNT_KEYS}), Counter()
                for ordinal, cid in enumerate(store.offsets):
                    if ordinal % 128 == 0:
                        self.check()
                    offset = handle.tell()
                    case, record = store.get(cid), _record(handle.readline())
                    require(case['case_id'] == cid and cid not in self.index[roster] and
                            all(cid not in indexed for other, indexed in self.index.items()
                                if other != roster), 'Duplicate/mismatched original boundary case')
                    _, actual = _validate_record(case, record, roster, ordinal)
                    self.index[roster][cid] = (offset, ordinal)
                    count.update(actual)
                    domains[case['primary_domain']] += 1
                require(handle.read(1) == b'', 'Extra boundary records or trailing junk')
                require(dict(count) == self.binding['report']['counts'][roster] and
                        dict(domains) == self.binding['report']['primary_domain_cases'][roster] and
                        len(self.index[roster]) == CASE_COUNTS[roster],
                        'Actual full boundary roster accounting/domains differ from report')
                self.check()
            self.check()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass  # Preserve the constructor's original validation failure.
            raise

    def check(self):
        require(not self._closed, 'BoundaryStore is closed')
        _guard(self.binding)
        for roster, handle in self.handles.items():
            require(not handle.closed, 'Boundary sidecar handle is closed')
            pin = self.binding['sidecars'][roster]
            current, opened = Path(pin['path']).stat(), os.fstat(handle.fileno())
            require(stat_mode.S_ISREG(opened.st_mode) and
                    (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) ==
                    (current.st_dev, current.st_ino, pin['bytes'], pin['mtime_ns']),
                    'Opened boundary sidecar no longer matches guarded file')

    def for_case(self, case, *, dev=False):
        require(type(dev) is bool, 'Boundary dev selector must be boolean')
        self.check()
        roster = 'select' if dev else 'train'
        require(case['split'] == SPLITS[roster], 'Boundary caller native split differs')
        cid = case['case_id']
        require(cid in self.index[roster], 'Original case has no exact boundary: ' + str(cid))
        offset, ordinal = self.index[roster][cid]
        handle = self.handles[roster]
        handle.seek(offset)
        record = _record(handle.readline())
        require(record['case_id'] == cid, 'Boundary seek returned a different original case')
        cuts, _ = _validate_record(case, record, roster, ordinal)
        self.check()
        return cuts

    def close(self):
        if self._closed:
            return
        self._closed = True
        error = None
        for handle in self.handles.values():
            try:
                handle.close()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error
