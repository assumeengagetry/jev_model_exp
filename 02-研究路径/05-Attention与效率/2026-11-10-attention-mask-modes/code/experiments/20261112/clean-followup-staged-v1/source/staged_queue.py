"""Explicit external cursor1 continuation; SYSTEM Python, no Torch/model import.

Shared installation/readback validation is stdlib-only. Historical ledger pins
are deliberately NOT recursively checked by the running controller's guards.
"""
import argparse
import base64
import ctypes
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import select
import shutil
import signal
import stat
import sys
import time
import traceback

sys.dont_write_bytecode = True
DATA = Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
CTRL = Path('/tmp/opencode/jev-clean-followup-staged-20261112-v1')
ORIGINAL = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
LEASES = Path('/tmp/opencode/jev-owned-h800-leases-20261112')
ARMS = {'STATE_NL_PB_P_CE': 2, 'JPAIR_P_BCE_RANK': 3}
OLD_ARMS = ('BI_P_CE', 'PB_P_CE')
CAP = 25 * 2**30
EVIDENCE, FLOOR = 5 * 2**30, 20 * 2**30
ONE = {'macro_updates': 1, 'cases': 16, 'logical_pair_tokens': 20947}
FINAL = {'macro_updates': 3557, 'cases': 56912, 'logical_pair_tokens': 99824365}
PIN_KEYS = {'path', 'resolved', 'bytes', 'mtime_ns'}
PREFIX = 'Known old BI/PB + BOTH new arms + evidence/shared floor cannot fit: '
SOURCE_PATHS = ('source/main_readback.py', 'source/staged_queue.py', 'deploy_staged.py')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def same(left, right, message):
    need(canonical(left) == canonical(right), message)


def parse(text):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return parse(Path(path).read_text())


def directory(path):
    path = Path(path)
    s = path.lstat()
    need(stat.S_ISDIR(s.st_mode) and path.resolve() == path and not path.is_symlink(),
         'Nonsymlink fixed directory required: ' + str(path))


def pin(path):
    """Match the frozen helper's existing stat schema for external inputs.

    Some already-admitted stock/native inputs may have a distinct resolved
    path. Their exact historical resolved pin must match; do not silently
    replace that contract with a new path policy. Our own entries/readonly
    controls and successor lease FDs separately require nonsymlink paths.
    """
    path = Path(path)
    s = path.stat()
    need(stat.S_ISREG(s.st_mode), 'Regular pinned input required: ' + str(path))
    return {'path': str(path), 'resolved': str(path.resolve()), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def pins(value):
    if isinstance(value, dict):
        if set(value) == PIN_KEYS:
            yield value
        else:
            for item in value.values():
                yield from pins(item)
    elif isinstance(value, list):
        for item in value:
            yield from pins(item)


def verify(value):
    for item in pins(value):
        same(pin(item['path']), item, 'Changed input: ' + item['path'])


def entry(path):
    path = Path(path)
    need(not path.is_symlink() and path.resolve() == path and stat.S_ISREG(path.lstat().st_mode),
         'Nonsymlink fixed exact input required: ' + str(path))
    before = pin(path)
    encoded = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    same(pin(path), before, 'Input changed while reading bytes')
    return {'stat': before, 'base64': encoded}


def exact(item):
    need(set(item) == {'stat', 'base64'}, 'Exact input schema differs')
    path = Path(item['stat']['path'])
    need(not path.is_symlink() and path.resolve() == path and stat.S_ISREG(path.lstat().st_mode),
         'Exact input redirected: ' + str(path))
    verify(item['stat'])
    need(Path(item['stat']['path']).read_bytes() == base64.b64decode(item['base64'], validate=True),
         'Input bytes changed: ' + item['stat']['path'])
    verify(item['stat'])


def readonly(path):
    path = Path(path)
    need(not path.is_symlink() and path.resolve() == path, 'Readonly control redirected: ' + str(path))
    pin(path)
    need(stat.S_IMODE(Path(path).stat().st_mode) == 0o444, 'chmod444 control evidence required: ' + str(path))


def absent(path):
    path = Path(path)
    need(not path.exists() and not path.is_symlink(), 'Existing evidence preserved: ' + str(path))


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def commit(path, value):
    """Exclusive durable publication; never overwrite or remove failed staging."""
    path = Path(path)
    pending = path.with_name(path.name + '.writing')
    with pending.open('x') as handle:
        handle.write(canonical(value) + '\n')
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())
    os.link(pending, path, follow_symlinks=False)
    pending.unlink()  # Only our successfully published staging link.
    sync_dir(path.parent)


def imported_helper(item):
    need(item['stat']['path'] == str(DATA / 'ops/side_queue.py'), 'Helper path differs')
    exact(item)
    path = Path(item['stat']['path'])
    need(path.stat().st_mode & 0o222 == 0, 'Frozen old helper required')
    spec = importlib.util.spec_from_file_location('frozen_followup_queue', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exact(item)
    need(module.ROOT == DATA and module.OLD == ORIGINAL and module.ARMS == ARMS and module.CAP == CAP,
         'Frozen helper fixed roots/resources differ')
    need('torch' not in sys.modules, 'No Torch in system controller')
    return module


def resource_exception(cfg, helper):
    """ONLY the accepted historical resource failure; safe after arms resume."""
    for name, filename in (('old_failure', 'QUEUE_BLOCKED.json'), ('old_launch', 'QUEUE_LAUNCH.json'),
                           ('old_ready', 'PRIMARY_READY.json'), ('old_resources', 'RESOURCES_READY.json'),
                           ('old_bundle', 'OPS_BUNDLE.json')):
        need(cfg[name]['stat']['path'] == str(DATA / filename), 'Historical input redirected: ' + name)
        exact(cfg[name])  # File bytes, NOT recursive historical pin verification.
    need(set(DATA.glob('*BLOCKED*')) == {DATA / 'QUEUE_BLOCKED.json'}, 'Unaccepted old root blocker')
    absent(DATA / 'AUTHORIZE_RESUME.json')
    need(not list(DATA.glob('BOOTSTRAP_BLOCKED*')) and not list(DATA.glob('OPS_BLOCKED*')),
         'Bootstrap/ops blocked')
    failure = read(DATA / 'QUEUE_BLOCKED.json')
    launch = read(DATA / 'QUEUE_LAUNCH.json')
    same(failure['owner'], launch['owner'], 'Failure is not exact old launch owner')
    need(set(launch['owner']) == {'pid', 'starttime', 'boot_id'}, 'Exact old owner schema required')
    helper.require_dead(launch['owner'])
    need(failure['complete'] is False and failure['error_type'] == 'RuntimeError' and
         failure['partial_preserved'] is True and failure['automatic_resume'] is False and
         failure['public_run'] is False and failure['error'].startswith(PREFIX), 'Not the allowed resource failure')
    diagnostics = parse(failure['error'][len(PREFIX):])
    need(set(diagnostics) == {'free_bytes', 'required_bytes', 'outstanding'}, 'Resource diagnostics schema differs')
    folders = {str(parent / (arm + '-seed42')) for parent, arms in ((ORIGINAL, OLD_ARMS), (DATA, ARMS))
               for arm in arms}
    need(set(diagnostics['outstanding']) == folders and
         all(type(n) is int and 0 <= n <= CAP for n in diagnostics['outstanding'].values()),
         'Resource exception must bind all four original capped arms')
    need(type(diagnostics['free_bytes']) is int and type(diagnostics['required_bytes']) is int and
         diagnostics['required_bytes'] == sum(diagnostics['outstanding'].values()) + EVIDENCE + FLOOR and
         diagnostics['free_bytes'] < diagnostics['required_bytes'] and
         diagnostics['free_bytes'] == 115269525504 and diagnostics['required_bytes'] == 116379385959,
         'Not the exact recorded 1.11GB resource shortfall')
    workers = failure['workers']
    need(len(workers) == 4 and {(w['arm'], w['phase']) for w in workers} ==
         {(arm, phase) for arm in ARMS for phase in ('first', 'primary')}, 'Exactly four old first/primary workers required')
    identities = []
    for worker in workers:
        need(worker['created'] is True and worker['actual_wait_complete'] is True and
             type(worker['exit_code']) is int and worker['exit_code'] == (75 if worker['phase'] == 'first' else 0) and
             worker['gpu'] == (ARMS[worker['arm']] if worker['phase'] == 'first' else None) and
             worker['output'] == str(DATA / (worker['arm'] + '-seed42')),
             'Old child did not actually wait with expected phase/exit')
        need(set(worker['process']) == {'pid', 'starttime', 'boot_id'}, 'Missing exact old child identity')
        helper.require_dead(worker['process'])
        identities.append(canonical(worker['process']))
        receipt = DATA / (worker['arm'] + '-' + worker['phase'] + '-EXIT.json')
        actual = read(receipt)
        for key in ('arm', 'phase', 'process', 'gpu', 'uuid', 'output', 'created', 'exit_code', 'actual_wait_complete'):
            same(actual[key], worker[key], 'Old child exit receipt mismatch: ' + key)
    need(len(set(identities)) == 4, 'Duplicate old child identity')
    for arm in ARMS:
        # These names must never have existed in the old failed controller.
        for suffix in ('-resume.log', '-resume-OWNER.json', '-resume-EXIT.json'):
            # New resume evidence is permitted only after our durable admission.
            if not (CTRL / (arm + '-ADMITTED.json')).exists():
                absent(DATA / (arm + suffix))
    for name in ('RESUME_AUTHORIZATION_ACCEPTED.json', 'QUEUE_COMPLETE.json', 'MAIN_DIRECT_ACTUAL_REVIEW.json'):
        absent(DATA / name)
    return launch['owner']


def initial_old_inputs(helper, cfg):
    """Pure JSON Context/bindings checks, before Torch or actual model reads."""
    base = helper.Queue()
    base.bundle = read(DATA / 'OPS_BUNDLE.json')
    base.bundle_pin = pin(DATA / 'OPS_BUNDLE.json')
    base.bundle_bytes = (DATA / 'OPS_BUNDLE.json').read_bytes()
    need(base.bundle['complete'] is True and base.bundle['version'] == 'followup-ops-bundle-v1' and
         base.bundle['root'] == str(DATA) and base.bundle['queue_launched'] is False and
         base.bundle['training_launched'] is False, 'Frozen old OPS bundle required')
    expected = {str(DATA / 'source' / name) for name in helper.SOURCE_NAMES}
    expected |= {str(DATA / 'ops' / name) for name in ('freeze_fresh.py', 'side_queue.py', 'verify_first.py')}
    expected.add(str(DATA / 'seed42.json'))
    need(len(base.bundle['files']) == len(expected) and
         {item['stat']['path'] for item in base.bundle['files']} == expected and
         base.bundle['original']['post_four_gpus'] == [0, 1], 'Exact old frozen source/ops inventory required')
    same([item for item in base.bundle['files'] if item['stat']['path'] == str(DATA / 'ops/side_queue.py')],
         [cfg['helper']], 'Reviewed helper is not the old OPS-bound source')
    helper.Queue.pins_check(base)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(DATA / 'source'))
    from bindings import Context
    base.contexts = {arm: Context(DATA / 'seed42.json', arm) for arm in ARMS}
    base.scopes = {arm: ctx.snapshot for arm, ctx in base.contexts.items()}
    need('torch' not in sys.modules, 'Context checks imported Torch')
    ready = read(DATA / 'PRIMARY_READY.json')
    need(ready['complete'] is True and ready['root'] == str(DATA) and set(ready['arms']) == set(ARMS) and
         ready['main_direct_actual_review_required'] is True and ready['automatic_continuation'] is False and
         ready['D_confirm_opened'] is False and ready['public_run'] is False, 'Two actual limited primaries required')
    same(cfg['arms'], ready['arms'], 'Exact PRIMARY_READY arms differ')
    for arm in ARMS:
        same(base.scopes[arm], cfg['arms'][arm]['scope'], 'Full own scope changed')
    return base


def paused_bindings(base, cfg, arms):
    result = {}
    for arm in arms:
        for folder in (DATA / (arm + '-seed42'), DATA / 'primary' / arm):
            need(not list(folder.glob('BLOCKED*')), 'Arm/primary blocked')
        first = base.paused(arm)
        binding = base.primary_binding(arm, first)
        same(binding, cfg['arms'][arm], 'Paused primary/latest checkpoint changed: ' + arm)
        result[arm] = first
    return result


class Inputs:
    def __init__(self, require_install=True):
        directory(CTRL)
        need(CTRL.stat().st_uid == os.getuid() and stat.S_IMODE(CTRL.stat().st_mode) == 0o700,
             'Owned private new control directory required')
        directory(CTRL / 'source')
        self.manifest = entry(CTRL / 'MANIFEST.json')
        readonly(CTRL / 'MANIFEST.json')
        manifest = read(CTRL / 'MANIFEST.json')
        need(set(manifest) == {'version', 'complete', 'control_root', 'config', 'sources', 'input_pins'} and
             manifest['version'] == 'staged-external-manifest-v1' and manifest['complete'] is True and
             manifest['control_root'] == str(CTRL), 'Manifest schema/root differs')
        need(manifest['config']['stat']['path'] == str(CTRL / 'analysis-config.json'), 'Config redirected')
        exact(manifest['config'])
        readonly(CTRL / 'analysis-config.json')
        self.config_entry = manifest['config']
        self.cfg = cfg = read(CTRL / 'analysis-config.json')
        need(cfg['version'] == 'staged-external-config-v1' and cfg['control_root'] == str(CTRL) and
             cfg['data_root'] == str(DATA) and cfg['original_root'] == str(ORIGINAL) and
             cfg['resources'] == {'arm_cap_bytes': CAP, 'evidence_bytes': EVIDENCE, 'shared_floor_bytes': FLOOR} and
             cfg['automatic_retry'] is False and cfg['D_confirm_opened'] is False and cfg['public_run'] is False,
             'Fixed external continuation contract differs')
        need(set(manifest['sources']) == set(SOURCE_PATHS), 'Exactly three public new sources required')
        for name, item in manifest['sources'].items():
            need(item['stat']['path'] == str(CTRL / name), 'Source redirected')
            exact(item)
            readonly(CTRL / name)
        self.sources = manifest['sources']
        same(manifest['input_pins'], cfg['static_pins'], 'Manifest/static input inventory differs')
        dynamic_paths = {str(DATA / (arm + '-seed42') / 'ledger.jsonl') for arm in ARMS}
        need(not any(p['path'] in dynamic_paths for p in cfg['static_pins']),
             'Historical mutable ledger incorrectly classified as common static input')
        self.install_entry = None
        if require_install:
            readonly(CTRL / 'INSTALL_COMPLETE.json')
            installed = read(CTRL / 'INSTALL_COMPLETE.json')
            need(installed['complete'] is True and installed['exact_byte_readback'] is True and
                 installed['readback_launched'] is False and installed['queue_launched'] is False and
                 installed['authorization_created'] is False and installed['old_data_written'] is False,
                 'Completed install-only receipt required')
            same(installed['manifest'], self.manifest['stat'], 'Installed manifest differs')
            same(installed['config'], self.config_entry['stat'], 'Installed config differs')
            self.install_entry = entry(CTRL / 'INSTALL_COMPLETE.json')
        self.helper = imported_helper(cfg['helper'])
        self.common()
        self.base = initial_old_inputs(self.helper, cfg)

    def common(self):
        exact(self.manifest)
        exact(self.config_entry)
        readonly(CTRL / 'MANIFEST.json')
        readonly(CTRL / 'analysis-config.json')
        if self.install_entry is not None:
            exact(self.install_entry)
        for item in self.sources.values():
            exact(item)
            readonly(item['stat']['path'])
        verify(self.cfg['static_pins'])
        for item in self.cfg['static_exact']:
            exact(item)
        resource_exception(self.cfg, self.helper)
        need(not list(CTRL.glob('*BLOCKED*')), 'New control root has a blocker; no retry')
        for name in ('INSTALL_BLOCKED.json', 'MAIN_READBACK_BLOCKED.json', 'QUEUE_BLOCKED.json'):
            absent(CTRL / name)


def auth_body(cfg, review_pin):
    """Exact all-mandatory/no-extras schema; ONLY independent MAIN publishes it."""
    return {'version': 'staged-external-main-resume-v1', 'control_root': str(CTRL),
            'data_root': str(DATA), 'authorize_resume': True,
            'accepted_resource_blocker': cfg['old_failure']['stat'],
            'primary_ready': cfg['old_ready']['stat'], 'main_review': review_pin,
            'arms': cfg['arms'], 'next_macro_index': 1, 'repeat_macro0_allowed': False,
            'D_confirm_opened': False, 'public_run': False}


def authorization(inputs):
    cfg = inputs.cfg
    path = CTRL / 'AUTHORIZE_RESUME.json'
    readonly(path)
    review_path = CTRL / 'MAIN_DIRECT_ACTUAL_REVIEW.json'
    readonly(review_path)
    need(path.stat().st_nlink == review_path.stat().st_nlink == 1, 'Published MAIN evidence must have one final link')
    absent(path.with_name(path.name + '.writing'))
    review_entry = entry(review_path)
    review = read(review_path)
    need(review['complete'] is True and review['version'] == 'staged-main-direct-formal-macro-readback-v1' and
         review['control_root'] == str(CTRL) and review['data_root'] == str(DATA) and
         review['authorization_created'] is False and review['model_forward_calls'] == 0 and
         review['optimizer_steps'] == 0 and review['next_macro_index'] == 1 and
         review['repeat_macro0_allowed'] is False and review['D_confirm_opened'] is False and
         review['public_run'] is False, 'Successful independent actual MAIN readback required')
    same(review['analysis_config'], inputs.config_entry['stat'], 'MAIN config differs')
    same(review['manifest'], inputs.manifest['stat'], 'MAIN manifest differs')
    same(review['accepted_resource_blocker'], cfg['old_failure']['stat'], 'MAIN old failure differs')
    same(review['primary_ready'], cfg['old_ready']['stat'], 'MAIN PRIMARY_READY differs')
    same(review['arm_bindings'], cfg['arms'], 'MAIN full scope/checkpoint bindings differ')
    same(review['analysis_source'], inputs.sources['source/main_readback.py']['stat'], 'MAIN analysis source differs')
    need(set(review['arms']) == set(ARMS), 'MAIN both-arm actual checks required')
    for checks in review['arms'].values():
        for key in ('actual_replay_bytes_equal', 'MAIN_independent_actual_coef_bytes_equal',
                    '393_raw_clipped_before_after_delta_and_Adam_read_directly',
                    'actual_fresh_and_latest_checkpoint_bytes_equal', 'macro0_consumed_once'):
            need(checks[key] is True, 'MAIN actual tensor check missing: ' + key)
        same(checks['cursor'], ONE, 'MAIN readback cursor differs')
    # This is the ONE pre-resume recursive historical-evidence validation.
    verify(review['evidence_pins'])
    paused_bindings(inputs.base, cfg, ARMS)
    same(read(path), auth_body(cfg, review_entry['stat']), 'Exact external MAIN authorization differs')
    need(pin(path)['mtime_ns'] >= review_entry['stat']['mtime_ns'] >= cfg['old_ready']['stat']['mtime_ns'],
         'Authorization must follow actual MAIN review and primaries')
    inputs.common()
    return entry(path), review_entry


def build_installation(payload):
    """Called only by install-only deployer AFTER exclusive new CTRL ownership."""
    for parent in (DATA, ORIGINAL, DATA / 'ops', DATA / 'source'):
        directory(parent)
    helper_entry = entry(DATA / 'ops/side_queue.py')
    need(helper_entry['base64'] == payload['old_helper_template']['base64'], 'Remote helper differs from reviewed local template')
    helper = imported_helper(helper_entry)
    cfg = {'version': 'staged-external-config-v1', 'control_root': str(CTRL), 'data_root': str(DATA),
           'original_root': str(ORIGINAL), 'helper': helper_entry,
           'resources': {'arm_cap_bytes': CAP, 'evidence_bytes': EVIDENCE, 'shared_floor_bytes': FLOOR},
           'arms': read(DATA / 'PRIMARY_READY.json')['arms'], 'automatic_retry': False,
           'D_confirm_opened': False, 'public_run': False}
    for key, name in (('old_failure', 'QUEUE_BLOCKED.json'), ('old_launch', 'QUEUE_LAUNCH.json'),
                      ('old_ready', 'PRIMARY_READY.json'), ('old_resources', 'RESOURCES_READY.json'),
                      ('old_bundle', 'OPS_BUNDLE.json')):
        cfg[key] = entry(DATA / name)
    resource_exception(cfg, helper)
    base = initial_old_inputs(helper, cfg)
    paused_bindings(base, cfg, ARMS)
    resources = read(DATA / 'RESOURCES_READY.json')
    need(resources['leases_held'] is True and resources['advisory_only'] is True and
         set(resources['devices']) == {'2', '3'} and len(set(resources['devices'].values())) == 2,
         'Historical resource UUID evidence required, not a live lease assertion')
    cfg['devices'] = resources['devices']
    cfg['old_leases'] = {}
    directory(LEASES)
    need(LEASES.stat().st_uid == os.getuid() and stat.S_IMODE(LEASES.stat().st_mode) == 0o700,
         'Unsafe existing global lease directory')
    for arm, gpu in ARMS.items():
        path = LEASES / ('gpu' + str(gpu) + '.lock')
        item = entry(path)
        s = path.lstat()
        need(s.st_uid == os.getuid() and s.st_nlink == 1 and stat.S_IMODE(s.st_mode) == 0o600 and s.st_size > 0,
             'Unsafe/nonempty old lease evidence required')
        expected = {'root': str(DATA), 'gpu': gpu, 'uuid': resources['devices'][str(gpu)],
                    'owner': read(DATA / 'QUEUE_LAUNCH.json')['owner'], 'bundle': cfg['old_bundle']['stat'],
                    'advisory_only': True}
        same(parse(base64.b64decode(item['base64'], validate=True)), expected,
             'Unknown old lease metadata; never take over')
        exact(item)
        cfg['old_leases'][str(gpu)] = {'entry': item, 'device': s.st_dev, 'inode': s.st_ino,
                                      'uid': s.st_uid, 'mode': 0o600, 'nlink': 1}
    # Collect every existing primary/scope/checkpoint/artifact pin, but split out
    # ONLY the two mutable paused ledgers. Reports remain immutable file inputs.
    all_records = {}
    static_exact = [cfg[k] for k in ('old_failure', 'old_launch', 'old_ready', 'old_resources', 'old_bundle', 'helper')]
    static_exact += [entry(CTRL / 'INSTALLING.json'), entry(DATA / 'FORMAL_FIRST_WAVE.json'),
                     entry(DATA / 'ORIGINAL_FOUR_ELIGIBLE.json'), entry(DATA / 'QUEUE_OWNER.json')]
    # The bundle ALSO contains LOCAL installer provenance pins which do not
    # name remote files. Mirror the old helper's verified remote inventory,
    # rather than recursively treating local provenance as live remote inputs.
    values = [base.bundle['files'], base.bundle['evidence'], base.bundle['original']['files'],
              base.bundle['installation'], base.bundle['fresh'], base.bundle['fresh_files']]
    for arm, binding in cfg['arms'].items():
        folder = DATA / (arm + '-seed42')
        report = read(binding['report']['path'])
        values += [binding, report, read(folder / 'OWN_SCOPE.json'),
                   read(Path(binding['checkpoint']) / 'COMPLETE.json'),
                   read(folder / 'first-actual-macro/COMPLETE.json')]
        static_exact += [entry(binding['report']['path']), entry(binding['pin_file']['path']),
                         entry(folder / 'OWN_SCOPE.json')]
        all_records[str(folder / 'OWN_SCOPE.json')] = pin(folder / 'OWN_SCOPE.json')
        for phase in ('first', 'primary'):
            static_exact += [entry(DATA / (arm + '-' + phase + '-OWNER.json')),
                             entry(DATA / (arm + '-' + phase + '-EXIT.json'))]
    for value in values:
        for record in pins(value):
            previous = all_records.get(record['path'])
            need(previous is None or canonical(previous) == canonical(record), 'Conflicting old evidence pins')
            all_records[record['path']] = record
    dynamic_paths = {str(DATA / (arm + '-seed42') / 'ledger.jsonl') for arm in ARMS}
    cfg['historical_ledger_pins'] = {arm: cfg['arms'][arm]['ledger'] for arm in ARMS}
    cfg['static_pins'] = [r for path, r in sorted(all_records.items()) if path not in dynamic_paths]
    cfg['static_exact'] = static_exact
    verify(cfg['static_pins'])
    verify(cfg['historical_ledger_pins'])
    for item in cfg['static_exact']:
        exact(item)
    resource_exception(cfg, helper)
    paused_bindings(base, cfg, ARMS)
    commit(CTRL / 'analysis-config.json', cfg)
    sources = {name: entry(CTRL / name) for name in SOURCE_PATHS}
    for name, item in sources.items():
        need(item['base64'] == payload['sources'][name]['base64'], 'Installed source bytes differ')
        readonly(CTRL / name)
    manifest = {'version': 'staged-external-manifest-v1', 'complete': True, 'control_root': str(CTRL),
                'config': entry(CTRL / 'analysis-config.json'), 'sources': sources,
                'input_pins': cfg['static_pins']}
    commit(CTRL / 'MANIFEST.json', manifest)
    checked = Inputs(require_install=False)
    paused_bindings(checked.base, checked.cfg, ARMS)
    return {'complete': True, 'manifest': pin(CTRL / 'MANIFEST.json'), 'config': pin(CTRL / 'analysis-config.json'),
            'exact_byte_readback': True, 'readback_launched': False, 'queue_launched': False,
            'authorization_created': False, 'old_data_written': False}


def controller_class(helper):
    class Events(helper.Events):
        def __init__(self):
            super().__init__()  # DATA/ORIGINAL remain the helper's fixed globals.
            try:
                for folder in (CTRL, *(ORIGINAL / (arm + '-seed42') for arm in OLD_ARMS)):
                    directory(folder)
                    wd = self.libc.inotify_add_watch(self.fd, os.fsencode(folder), 0x8 | 0x80 | 0x100 | 0x400 | 0x800)
                    if wd < 0:
                        raise OSError(ctypes.get_errno(), 'inotify_add_watch')
                    self.watches[wd] = folder
            except BaseException:
                self.close()
                raise

    class StagedQueue(helper.Queue):
        def __init__(self, inputs):
            super().__init__()
            self.inputs, self.cfg = inputs, inputs.cfg
            for name in ('bundle', 'bundle_pin', 'bundle_bytes', 'scopes'):
                setattr(self, name, getattr(inputs.base, name))
            self.devices = {int(k): v for k, v in self.cfg['devices'].items()}
            self.admitted, self.completed, self.first = set(), {}, {}
            self.held_gpus, self.pending_states = set(), {}
            self.completion_pins = {}
            self.journal_index = 0
            self.auth_entry = self.review_entry = None

        def journal(self, kind, value):
            self.journal_index += 1
            commit(CTRL / (f'{self.journal_index:06d}-' + kind + '.json'),
                   {'owner': self.owner, 'control_root': str(CTRL), **value})

        def stop_check(self):
            helper.Queue.stop_check(self)
            for name in ('STOP.json', 'STOP_QUEUE.json'):
                absent(CTRL / name)

        def pins_check(self):
            helper.Queue.pins_check(self)  # eligible=None: no mutable historical ledger.
            self.inputs.common()
            if self.auth_entry is not None:
                exact(self.auth_entry)
                exact(self.review_entry)  # File bytes only; NEVER recurse its old ledgers.
            verify(list(self.completion_pins.values()))
            for arm in ARMS:
                need(not list((DATA / (arm + '-seed42')).glob('BLOCKED*')) and
                     not list((DATA / 'primary' / arm).glob('BLOCKED*')), 'Arm/primary blocker')

        def budget(self, candidates=()):
            actual, outstanding = {}, {}
            wanted = self.admitted | set(candidates)
            for parent, arms in ((ORIGINAL, OLD_ARMS), (DATA, ARMS)):
                for arm in arms:
                    folder = parent / (arm + '-seed42')
                    n = helper.used(folder)
                    need(n <= CAP, 'Retained arm exceeds unchanged 25GiB cap: ' + str(folder))
                    actual[str(folder)] = n
                    reserve = (parent == ORIGINAL or arm in wanted) and str(folder) not in self.disk_done
                    outstanding[str(folder)] = CAP - n if reserve else 0
            free = shutil.disk_usage(DATA).free
            required = sum(outstanding.values()) + EVIDENCE + FLOOR
            return {'free_bytes': free, 'required_free_bytes': required, 'fits': free >= required,
                    'retained_bytes': actual, 'outstanding_bytes': outstanding,
                    'admitted': [a for a in ARMS if a in self.admitted],
                    'candidate': list(candidates), 'projection_only': bool(candidates),
                    'pending_unreserved': [a for a in ARMS if a not in self.admitted],
                    'evidence_bytes': EVIDENCE, 'shared_floor_bytes': FLOOR,
                    'arm_cap_bytes': CAP, 'system_wide_reservation': False}

        def guard(self):
            self.stop_check()
            self.pins_check()
            for fd, (path, inode, data) in self.lease_records.items():
                fs, ls = os.fstat(fd), path.lstat()
                need((fs.st_dev, fs.st_ino) == inode == (ls.st_dev, ls.st_ino) and
                     stat.S_ISREG(fs.st_mode) and stat.S_ISREG(ls.st_mode) and
                     fs.st_uid == ls.st_uid == os.getuid() and fs.st_nlink == ls.st_nlink == 1 and
                     stat.S_IMODE(fs.st_mode) == stat.S_IMODE(ls.st_mode) == 0o600 and
                     os.pread(fd, len(data) + 1, 0) == data, 'Held successor lease inode/old bytes changed')
            devices, occupants = helper.nvidia()
            same(devices, self.devices, 'Physical GPU UUID inventory changed')
            # Pending/unheld foreign work is NOT an active-arm failure.
            for gpu in self.held_gpus:
                allowed = {c['id']['pid']: c['id'] for c in self.children if c['gpu'] == gpu and
                           c['p'] is not None and c['id'] is not None and c['p'].returncode is None}
                for pid in occupants[devices[gpu]]:
                    need(pid in allowed and helper.identity(pid) == allowed[pid],
                         'Foreign process on held GPU; pause only exact owned work')
            budget = self.budget()
            need(budget['fits'], 'Admitted staged resource guard cannot fit: ' + canonical(budget))
            return budget

        def successor(self, gpu):
            """No create/truncate/write of old lock. Return False for lock contention."""
            directory(LEASES)
            need(LEASES.stat().st_uid == os.getuid() and stat.S_IMODE(LEASES.stat().st_mode) == 0o700,
                 'Unsafe global lease directory')
            record = self.cfg['old_leases'][str(gpu)]
            path = LEASES / ('gpu' + str(gpu) + '.lock')
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
            retained = False
            try:
                def checked():
                    fs, ls = os.fstat(fd), path.lstat()
                    data = base64.b64decode(record['entry']['base64'], validate=True)
                    need(data and (fs.st_dev, fs.st_ino) == (ls.st_dev, ls.st_ino) ==
                         (record['device'], record['inode']) and stat.S_ISREG(fs.st_mode) and stat.S_ISREG(ls.st_mode) and
                         fs.st_uid == ls.st_uid == record['uid'] == os.getuid() and
                         fs.st_nlink == ls.st_nlink == 1 and
                         stat.S_IMODE(fs.st_mode) == stat.S_IMODE(ls.st_mode) == 0o600 and
                         os.pread(fd, len(data) + 1, 0) == data, 'Unknown/changed old lease; no takeover')
                    same(pin(path), record['entry']['stat'], 'Old lease stat changed')
                    return fs, data
                checked()
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                fs, data = checked()
                resource_exception(self.cfg, helper)
                devices, occupants = helper.nvidia()
                same(devices, self.devices, 'GPU UUID changed before successor admission')
                if occupants[devices[gpu]]:
                    return False
                # Track FD before any journal operation can fail/interruption land.
                self.leases.append(fd)
                retained = True
                self.lease_records[fd] = (path, (fs.st_dev, fs.st_ino), data)
                self.held_gpus.add(gpu)
                commit(CTRL / ('gpu' + str(gpu) + '-SUCCESSOR.json'),
                       {'old_lease': record, 'old_owner': read(DATA / 'QUEUE_LAUNCH.json')['owner'],
                        'owner': self.owner, 'gpu': gpu, 'uuid': devices[gpu], 'control_root': str(CTRL),
                        'data_root': str(DATA), 'manifest': self.inputs.manifest['stat'],
                        'source': self.inputs.sources['source/staged_queue.py']['stat'],
                        'authorization': self.auth_entry['stat'], 'advisory_only': True,
                        'same_inode_fd_held': True, 'old_metadata_written': False})
                return True
            finally:
                if not retained:
                    os.close(fd)

        def final_arm(self, parent, arm, own=False):
            folder = parent / (arm + '-seed42')
            need(not list(folder.glob('BLOCKED*')), 'Completed arm blocked')
            scope = self.scopes[arm] if own else read(folder / 'OWN_SCOPE.json')
            helper.scope_check(scope, read(parent / 'seed42.json'), arm)
            same(read(folder / 'OWN_SCOPE.json'), scope, 'Complete own scope changed')
            done = helper.committed(folder / 'COMPLETE.json', scope)
            validation_pins = [pin(folder / 'OWN_SCOPE.json'), pin(folder / 'COMPLETE.json'), *pins(done)]
            need(done['cursor'] == FINAL and done['observations'] == [10000000, 30000000, 100000000] and
                 done['D_confirm_opened'] is False and done['public_run'] is False and
                 done['all_checkpoints_retained'] is True, 'Actual full completion/cursor differs')
            for item in done['checkpoints'] + done['D_select'] + [done['first_actual_macro']]:
                need(Path(item['path']).parent.parent == folder, 'Completion artifact redirected')
                saved = helper.committed(Path(item['path']), scope)
                validation_pins.extend(pins(saved))
            need(done['first_actual_macro'] == pin(folder / 'first-actual-macro/COMPLETE.json') and
                 done['final_checkpoint'] in done['checkpoints'] and
                 read(Path(done['final_checkpoint']['path']).parent / 'CHECKPOINT.json')['cursor'] == FINAL,
                 'Complete final checkpoint/first binding differs')
            helper.actual_nodes(done, folder, scope)
            if own:
                entries = [parse(line) for line in (folder / 'ledger.jsonl').read_text().splitlines()]
                starts = [e['macro_index'] for e in entries if e['event'] == 'macro_started']
                updates = [e['cursor']['macro_updates'] for e in entries if e['event'] == 'macro_updated']
                resumes = [e for e in entries if e['event'] == 'explicit_own_resume']
                need(starts == list(range(3557)) and updates == list(range(1, 3558)) and len(resumes) == 1 and
                     resumes[0]['complete'] == self.first[arm]['complete'] and
                     resumes[0]['checkpoint'] == self.cfg['arms'][arm]['checkpoint'] and
                     resumes[0]['resume_pin_file'] == self.cfg['arms'][arm]['pin_file'] and
                     entries[-1]['event'] == 'arm_completed' and entries[-1]['cursor'] == FINAL and
                     not any(e['event'] == 'failed' for e in entries), 'Contiguous sole cursor1 resume exposure differs')
                # A final post-wait ledger is immutable now; this is NOT the
                # historical paused pin which was intentionally phase-limited.
                validation_pins.append(pin(folder / 'ledger.jsonl'))
            verify(validation_pins)
            self.completion_pins.update({p['path']: p for p in validation_pins})
            return {'complete': pin(folder / 'COMPLETE.json'), 'scope': scope, 'cursor': done['cursor'],
                    'nodes': done['D_select'], 'first': done['first_actual_macro'],
                    'final_checkpoint': done['final_checkpoint']}

        def disk_completion(self, parent, arm):
            # Only old BI/PB COMPLETE events/initial reads may release old future cap.
            need(parent == ORIGINAL and arm in OLD_ARMS, 'Unexpected disk completion path')
            folder = parent / (arm + '-seed42')
            if str(folder) in self.disk_done:
                return
            if (folder / 'COMPLETE.json').exists():
                receipt = self.final_arm(parent, arm)
                self.disk_done.add(str(folder))
                self.journal('OLD-DISK-COMPLETE', {'arm': arm, 'actual_completion': receipt,
                                                'all_actual_files_retained': True})

        def resource_events(self, paths):
            for arm in OLD_ARMS:
                if ORIGINAL / (arm + '-seed42') / 'COMPLETE.json' in paths:
                    self.disk_completion(ORIGINAL, arm)

        def pending(self, arm, reason, budget):
            state = {'reason': reason, 'gpu': ARMS[arm], 'capacity_reserved': False, 'lease_held': False}
            if self.pending_states.get(arm) != state:
                self.pending_states[arm] = state
                self.journal('PENDING', {'arm': arm, **state, 'actual_budget': self.budget(),
                                         'hypothetical_admission_budget': budget})

        def admit(self):
            for arm, gpu in ARMS.items():  # Fixed STATE -> JPAIR order.
                if arm in self.admitted:
                    continue
                if arm == 'JPAIR_P_BCE_RANK' and 'STATE_NL_PB_P_CE' not in self.admitted:
                    self.pending(arm, 'STATE-first-not-yet-admitted', self.budget((arm,)))
                    continue
                budget = self.budget((arm,))
                if not budget['fits']:
                    self.pending(arm, 'pending-budget', budget)
                    continue
                devices, occupants = helper.nvidia()
                same(devices, self.devices, 'GPU inventory changed at admission')
                if occupants[devices[gpu]]:
                    self.pending(arm, 'pending-foreign-GPU-no-preemption', budget)
                    continue
                self.pins_check()
                # Only this unstarted arm's mutable ledger/history is revalidated.
                self.first.update(paused_bindings(self, self.cfg, (arm,)))
                for suffix in ('-resume.log', '-resume-OWNER.json', '-resume-EXIT.json'):
                    absent(DATA / (arm + suffix))
                prior = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
                try:
                    if not self.successor(gpu):
                        self.pending(arm, 'pending-lease-contention-or-GPU-race', budget)
                        continue
                    budget = self.budget((arm,))
                    need(budget['fits'], 'Candidate budget changed after lease; no automatic retry')
                    # Final paused primary/latest-tip read immediately before admission/spawn.
                    self.first.update(paused_bindings(self, self.cfg, (arm,)))
                    commit(CTRL / (arm + '-ADMITTED.json'),
                           {'owner': self.owner, 'arm': arm, 'gpu': gpu, 'uuid': devices[gpu], 'budget': budget,
                            'lease': pin(CTRL / ('gpu' + str(gpu) + '-SUCCESSOR.json')),
                            'authorization': self.auth_entry['stat'], 'manifest': self.inputs.manifest['stat'],
                            'source': self.inputs.sources['source/staged_queue.py']['stat'],
                            'binding': self.cfg['arms'][arm], 'next_macro_index': 1,
                            'repeat_macro0_allowed': False, 'capacity_reserved': True})
                    self.admitted.add(arm)
                    self.guard()
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, prior)
                # Restore the caller's mask BEFORE the helper's own spawn
                # critical section: its parent_guard restores that prior mask
                # in the actual child. Nesting the whole spawn under our mask
                # would incorrectly leave SIGTERM/SIGINT blocked in training.
                helper.Queue.spawn(self, arm, 'resume', resume={
                    'checkpoint': self.cfg['arms'][arm]['checkpoint'],
                    'pin_file': self.cfg['arms'][arm]['pin_file']['path']})
                self.pending_states.pop(arm, None)
                self.journal('ADMISSION', {'arm': arm, 'budget': budget, 'uuid': devices[gpu],
                                          'authorization': self.auth_entry['stat']})

        def cleanup_owned(self):
            """Reviewed helper cleanup algorithm; ONLY shutdown journal root differs.

            The old helper's implementation journals its rare exhausted-grace
            branch in DATA. Copy this method narrowly so controller evidence is
            CTRL-only; reuse its actual Popen wait barrier unchanged.
            """
            result, pending = [], {}
            for child in self.children:
                if child['p'] is None or child['p'].returncode is not None:
                    continue
                if child['id'] is None or child['fd'] is None:
                    result.append({'pid': child['p'].pid, 'signalled': False, 'own_Popen_wait_required': True})
                    continue
                current = helper.identity(child['id']['pid'])
                if current == child['id']:
                    try:
                        signal.pidfd_send_signal(child['fd'], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    except OSError as error:
                        result.append({'process': child['id'], 'TERM_error': str(error)})
                    pending[child['fd']] = child
                elif current is None:
                    child['p'].wait()
                else:
                    result.append({'process': child['id'], 'signalled': False, 'own_Popen_wait_required': True})
            deadline = time.monotonic() + helper.GRACE_SECONDS
            exhausted = False
            while pending:
                ready, _, _ = select.select(list(pending), [], [], None if exhausted else max(0, deadline - time.monotonic()))
                if not ready:
                    for fd, child in pending.items():
                        if helper.identity(child['id']['pid']) == child['id']:
                            try:
                                signal.pidfd_send_signal(fd, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            except OSError as error:
                                result.append({'process': child['id'], 'KILL_error': str(error)})
                    exhausted = True
                    try:
                        commit(CTRL / 'OWNED_SHUTDOWN_PENDING.json', {'owner': self.owner,
                               'children': [self.child_receipt(c) for c in pending.values()],
                               'leases_held': bool(self.leases), 'bounded_grace_exhausted': True})
                    except OSError as error:
                        result.append({'shutdown_status_write_error': str(error)})
                    continue
                for fd in ready:
                    child = pending.pop(fd)
                    child['p'].wait()
                    result.append({**self.child_receipt(child), 'safe_pause_requested': True})
            self.await_owned_exits()
            return result

        def run_staged(self):
            need(Path(__file__).absolute() == CTRL / 'source/staged_queue.py' and
                 Path(__file__).resolve() == CTRL / 'source/staged_queue.py', 'Run pinned deployed controller')
            for name in ('QUEUE_OWNER.json', 'QUEUE_COMPLETE.json', 'QUEUE_BLOCKED.json', 'RESUME_AUTHORIZATION_ACCEPTED.json'):
                absent(CTRL / name)
            lock = (CTRL / 'QUEUE.lock').open('x+')  # Permanent no-restart claim.
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.stop_check()
                self.auth_entry, self.review_entry = authorization(self.inputs)
                self.first = paused_bindings(self, self.cfg, ARMS)
                # Directory watches are all installed BEFORE initial COMPLETE reads.
                self.events = Events()
                commit(CTRL / 'QUEUE_OWNER.json', {'owner': self.owner, 'control_root': str(CTRL),
                       'data_root': str(DATA), 'manifest': self.inputs.manifest['stat'],
                       'authorization': self.auth_entry['stat'], 'automatic_restart': False})
                commit(CTRL / 'RESUME_AUTHORIZATION_ACCEPTED.json', {'owner': self.owner,
                       'authorization': self.auth_entry['stat'], 'main_review': self.review_entry['stat'],
                       'arms': self.cfg['arms'], 'next_macro_index': 1, 'repeat_macro0_allowed': False})
                for arm in OLD_ARMS:
                    self.disk_completion(ORIGINAL, arm)
                self.guard()
                self.admit()
                self.journal('BUDGET', {'budget': self.guard(), 'trigger': 'initial-admission'})
                next_guard = time.monotonic() + helper.GUARD_SECONDS
                while len(self.completed) < len(ARMS):
                    running = {c['fd']: c for c in self.children if c['p'] is not None and c['p'].returncode is None}
                    paths, dead = self.events.wait(running, max(0, next_guard - time.monotonic()))
                    self.stop_check()
                    self.resource_events(paths)
                    for fd in dead:
                        child = running[fd]
                        need(helper.identity(child['id']['pid']) in (None, child['id']), 'Owned PID identity changed before wait')
                        child['p'].wait()
                        child['waited'] = True
                        helper.commit(DATA / (child['arm'] + '-resume-EXIT.json'), self.child_receipt(child))
                        need(child['p'].returncode == 0, 'Resume did not fully complete; preserve/no restart')
                        self.completed[child['arm']] = self.final_arm(DATA, child['arm'], own=True)
                        self.disk_done.add(str(DATA / (child['arm'] + '-seed42')))
                        self.journal('OWN-COMPLETE', {'arm': child['arm'], 'completion': self.completed[child['arm']],
                                                     'worker': self.child_receipt(child), 'all_actual_files_retained': True})
                    timer = time.monotonic() >= next_guard
                    completion_event = bool(dead) or any(ORIGINAL / (a + '-seed42') / 'COMPLETE.json' in paths for a in OLD_ARMS)
                    if timer or completion_event:
                        self.guard()
                        self.admit()
                        self.journal('BUDGET', {'budget': self.guard(), 'trigger': 'resource-timer' if timer else 'completion-event'})
                        next_guard = time.monotonic() + helper.GUARD_SECONDS
                self.await_owned_exits()
                self.pins_check()
                commit(CTRL / 'QUEUE_COMPLETE.json', {'execution_complete': True, 'all_completed': True,
                       'owner': self.owner, 'manifest': self.inputs.manifest['stat'],
                       'authorization': self.auth_entry['stat'], 'arms': self.completed,
                       'workers': [self.child_receipt(c) for c in self.children],
                       'scope': 'limited-full-row-scientific-only', 'production': False, 'serve': False,
                       'published': False, 'winner_selected': False, 'D_confirm_opened': False, 'public_run': False})
            except BaseException as error:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    cleanup = self.cleanup_owned()
                except BaseException as cleanup_error:
                    cleanup = [{'error': str(cleanup_error), 'own_exit_barrier_required': True}]
                self.await_owned_exits()
                commit(CTRL / 'QUEUE_BLOCKED.json', {'complete': False, 'owner': self.owner,
                       'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                       'cleanup': cleanup, 'workers': [self.child_receipt(c) for c in self.children],
                       'partial_preserved': True, 'automatic_resume': False, 'public_run': False})
                raise
            finally:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                self.await_owned_exits()  # Never close leases before actual own waits.
                for child in self.children:
                    if child['fd'] is not None:
                        os.close(child['fd'])
                if self.events is not None:
                    self.events.close()
                for fd in self.leases:
                    os.close(fd)
                lock.close()
    return StagedQueue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-root', required=True, choices=[str(CTRL)])
    parser.parse_args()
    def interrupted(signum, frame):
        raise RuntimeError('External controller signal: ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    need(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'SYSTEM pidfd APIs required')
    inputs = Inputs()
    controller_class(inputs.helper)(inputs).run_staged()


if __name__ == '__main__':
    main()
