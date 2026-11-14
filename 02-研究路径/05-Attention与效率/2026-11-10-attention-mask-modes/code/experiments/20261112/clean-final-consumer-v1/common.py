"""Fixed final-consumer identities and installation; stdlib, no payload readers.

Stat pins are deliberately flat. Historical completion/config documents are
bound as files, not recursively interpreted as live ledger/status identities.
This is a cooperative same-UID protocol, not a security boundary.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import select
import signal
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path('/tmp/opencode/jev-clean-final-eval-20261112-v1')
ROLES = ('baseline', 'challenger')
MODEL_PYTHON = '/home/dev/assumeengage/envs/jev/bin/python'
SIGNALS = {signal.SIGTERM, signal.SIGINT}
_ORIGINAL = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
_FOLLOWUP = Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
_PACKED = Path('/tmp/opencode/jev-terminal-packed-admission-20261112-v1')
_STAGED = Path('/tmp/opencode/jev-clean-followup-staged-20261112-v2')
_LEASES = Path('/tmp/opencode/jev-owned-h800-leases-20261112')
_OLD_PUBLIC = Path('/tmp/opencode/jev-joint-generic-continuation-20261109')
_SDK_ALIAS = _OLD_PUBLIC / 'decision-index'
_SDK = Path('/tmp/opencode/jev-five-dimension-distilled-20261109/decision-index')
_SUITE_ALIAS = _OLD_PUBLIC / 'official-suite-0.3-verified'
_STOCK = '/home/dev/assumeengage/models/Qwen3-0.6B'
_SOURCE_NAMES = ('common.py', 'backend.py', 'evaluate.py', 'ops.py')
_PRODUCER_NAMES = ('bindings.py', 'fresh_init.py', 'case_runtime.py', 'objectives.py',
                   'model_runtime.py', 'checkpoints.py', 'train_case.py', 'evaluate_native_dev.py')
_PIN_KEYS = {'path', 'resolved', 'bytes', 'mtime_ns'}
_CURSOR = {'macro_updates': 3557, 'cases': 56912, 'logical_pair_tokens': 99824365}
_RESOURCES = {'max_output_bytes': 20 * 2**30, 'free_reserve_bytes': 20 * 2**30,
              'fixed_future_bytes': 20 * 2**30, 'guard_seconds': 15, 'grace_seconds': 900}
_GUARD_POLICY = 'flat-history-bounded-hot-v1'
_MAX_CODE_FILE = 2**20
_MAX_CODE_TOTAL = 4 * 2**20
_MAX_COMPACT = 2**20
_MAX_SNAPSHOT = 8 * 2**20
_HISTORY_CONTROLS = {
    'MANAGER_COMPLETE.json', 'LAUNCH.json', 'OWNER.json', 'config.json',
    'SOURCE_SNAPSHOT.json', 'QUEUE_BLOCKED.json', 'QUEUE_LAUNCH.json',
    'QUEUE_OWNER.json', 'OPS_BUNDLE.json', 'QUEUE_COMPLETE.json',
    'analysis-config.json', 'MANIFEST.json', 'COMPLETE.json',
}
_FROZEN_PIN = {'path': str(ROOT / 'FROZEN_CANDIDATES.json'),
               'resolved': str(ROOT / 'FROZEN_CANDIDATES.json'),
               'bytes': 37044, 'mtime_ns': 1794517053881585813}
_TERMINALS = {
    'baseline': (_ORIGINAL, 'C_P_CE', 535123, 1794495637563708302),
    'challenger': (_FOLLOWUP, 'JPAIR_P_BCE_RANK', 772665, 1794513433414682605),
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _parse(text):
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
    return _parse(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def same(a, b, message):
    need(canonical(a) == canonical(b), message)


def pin(path):
    path = Path(path)
    need(path.is_absolute(), 'Absolute stat pin required')
    s = path.stat()
    need(stat.S_ISREG(s.st_mode), 'Regular pinned file required: ' + str(path))
    return {'path': str(path), 'resolved': str(path.resolve()),
            'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def verify(pins):
    need(isinstance(pins, list), 'Only a flat list of stat pins is accepted')
    for item in pins:
        need(isinstance(item, dict) and set(item) == _PIN_KEYS, 'Bad flat stat pin schema')
        same(pin(item['path']), item, 'Immutable input changed: ' + item['path'])


def _directory(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and
         stat.S_ISDIR(path.lstat().st_mode), 'Fixed nonsymlink directory required: ' + str(path))


def _readonly(path):
    path = Path(path)
    need(path.resolve() == path and not path.is_symlink(), 'Redirected control: ' + str(path))
    pin(path)
    need(stat.S_IMODE(path.stat().st_mode) == 0o444, 'Readonly444 control required: ' + str(path))


def entry(path):
    path = Path(path)
    need(path.resolve() == path and not path.is_symlink(), 'Exact source/control redirected')
    before = pin(path)
    data = path.read_bytes()
    verify([before])
    return {'stat': before, 'base64': base64.b64encode(data).decode('ascii')}


def exact(item):
    need(set(item) == {'stat', 'base64'}, 'Bad exact entry schema')
    path = Path(item['stat']['path'])
    need(path.resolve() == path and not path.is_symlink(), 'Exact input redirected')
    verify([item['stat']])
    need(path.read_bytes() == base64.b64decode(item['base64'], validate=True),
         'Exact source/control bytes changed: ' + str(path))
    verify([item['stat']])


def _sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def commit(path, value):
    """Durable exclusive publication. Partial .writing is never adopted/deleted."""
    path = Path(path)
    _directory(path.parent)
    pending = path.with_name(path.name + '.writing')
    data = (canonical(value) + '\n').encode('utf-8')
    with pending.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())
    need(pending.read_bytes() == data, 'Commit byte readback failed')
    os.link(pending, path, follow_symlinks=False)
    pending.unlink()  # Only this successfully published invocation's temporary link.
    _sync(path.parent)


def identity(pid):
    need(type(pid) is int and pid > 0, 'Positive exact PID required')
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': pid, 'starttime': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:
        return None


def pidfd(record):
    need(isinstance(record, dict) and set(record) == {'pid', 'starttime', 'boot_id'},
         'Exact process record required')
    current = identity(record['pid'])
    # A reused numeric PID is NOT the old owner and must never be signalled.
    if current is None or current != record:
        return None
    try:
        fd = os.pidfd_open(record['pid'])
    except ProcessLookupError:
        need(identity(record['pid']) != record, 'Owner identity race')
        return None
    if identity(record['pid']) != record:
        os.close(fd)
        return None
    return fd


def require_dead(record):
    fd = pidfd(record)
    if fd is not None:
        try:
            need(bool(select.select([fd], [], [], 0)[0]), 'Exact prior owner/child still live')
        finally:
            os.close(fd)


def _unique(groups):
    result = {}
    for group in groups:
        need(isinstance(group, list), 'Flat pin group required')
        for item in group:
            need(set(item) == _PIN_KEYS, 'Flat immutable pin required')
            path = item['path']
            need(Path(path).name not in ('ledger.jsonl', 'status.json'),
                 'Mutable/historical ledger or SDK status cannot be a static pin')
            if path in result:
                same(result[path], item, 'Conflicting static pin')
            result[path] = item
    return [result[path] for path in sorted(result)]


def _frozen():
    _directory(ROOT)
    _readonly(ROOT / 'FROZEN_CANDIDATES.json')
    verify([_FROZEN_PIN])
    frozen = read(ROOT / 'FROZEN_CANDIDATES.json')
    need(frozen['version'] == 'clean-terminal-frozen-candidates-v1' and frozen['complete'] is True and
         frozen['stage'] == 'selected-terminal-models-not-production-or-confirm-execution-admission' and
         set(frozen['selected']) == set(ROLES) and frozen['public_arm_roles_fixed_before_confirm'] == list(ROLES) and
         frozen['D_confirm_may_not_change_public_list'] is True and frozen['only_seed'] == 42 and
         frozen['eligible_checkpoint_macro'] == 3557 and frozen['logical_training_tokens_per_model'] == 99824365 and
         frozen['consumer_ready'] is False and frozen['D_confirm_opened'] is False and frozen['public_run'] is False,
         'Actual frozen candidates required; selection readiness is not execution admission')
    for key, name in (('policy', 'policy.md'), ('selection_source', 'freeze_candidates.py')):
        need(frozen[key]['path'] == str(ROOT / name), 'Frozen policy/freezer redirected')
        _readonly(ROOT / name)
        verify([frozen[key]])
    runtime = frozen['runtime_contract']
    need(runtime['backend'] == 'native-FA3-full-row' and runtime['mode'] == 'causal' and
         runtime['layout'] == 'packed' and runtime['EOS'] == 151645 and runtime['position_reset'] is True and
         runtime['stock_directory'] == _STOCK and runtime['served_parameters'] == 616237056 and
         all(runtime[k] is False for k in ('prefix_cache', 'extra_fusions', 'graph_dummy_tokens')) and
         runtime['micro_plan'] == {'policy': 'stable-length-common-padded-admission-v1',
                                  'padded_tokens': 16384, 'max_pairs': 32, 'pad_multiple': 8},
         'Frozen native full-row contract differs')
    return frozen


def _producer_record(frozen, role):
    need(role in ROLES, 'Only the two frozen roles may load')
    root, arm, terminal_bytes, terminal_mtime = _TERMINALS[role]
    selected = frozen['selected'][role]
    folder = root / (arm + '-seed42')
    terminal = folder / 'checkpoint-m0003557-s00043'
    expected = {'path': str(terminal / 'COMPLETE.json'), 'resolved': str(terminal / 'COMPLETE.json'),
                'bytes': terminal_bytes, 'mtime_ns': terminal_mtime}
    need(selected['id'] == arm and selected['eligible'] is True and selected['producer_root'] == str(root) and
         selected['arm_directory'] == str(folder) and selected['terminal_checkpoint'] == str(terminal) and
         selected['mode'] == 'causal' and selected['layout'] == 'packed', 'Frozen role redirected')
    same(selected['terminal_complete'], expected, 'Frozen terminal identity differs')
    for key, path in (('producer_complete', folder / 'COMPLETE.json'),
                      ('producer_config', root / 'seed42.json'), ('own_scope', folder / 'OWN_SCOPE.json')):
        need(selected[key]['path'] == str(path), 'Producer metadata pointer redirected')
        verify([selected[key]])
    need(not list(folder.glob('BLOCKED*')), 'Actual producer arm is blocked')
    cfg = read(root / 'seed42.json')
    scope = read(folder / 'OWN_SCOPE.json')
    done = read(folder / 'COMPLETE.json')
    same(scope, done['scope'], 'Producer completion own scope changed')
    same(scope['config'], cfg, 'Producer scope/config changed')
    same(scope['config_pin'], selected['producer_config'], 'Producer own config pin differs')
    need(scope['arm_name'] == arm and scope['arm'] == cfg['arms'][arm] and
         cfg['seed'] == 42 and cfg['output_parent'] == str(root) and cfg['stock_directory'] == _STOCK and
         cfg['arms'][arm] == {'mode': 'causal', 'layout': 'packed',
                             'loss': 'CE' if role == 'baseline' else 'BCE_RANK_HARD'} and
         cfg['prefix_cache'] is False and cfg['world_size'] == 1 and cfg['device'] == 'cuda:0',
         'Wrong actual producer scope')
    rc = frozen['runtime_contract']
    for key in ('stock_files', 'core_files', 'native_files', 'precision', 'micro_plan'):
        same(cfg[key], rc[key], 'Frozen producer runtime changed: ' + key)
    names = _PRODUCER_NAMES + (('state_boundaries.py',) if role == 'challenger' else ())
    need({p['path'] for p in cfg['trainer_files']} == {str(root / 'source' / n) for n in names} and
         len(cfg['trainer_files']) == len(names), 'Actual one-role eight/nine source inventory required')
    for item in cfg['trainer_files']:
        _readonly(item['path'])
    verify(cfg['trainer_files'])
    same(scope['trainer_code'], {p['path']: entry(p['path'])['base64'] for p in cfg['trainer_files']},
         'Original producer source bytes differ')
    same(scope['core_code'], {p['path']: entry(Path(p['resolved']))['base64'] for p in cfg['core_files']},
         'Original producer core source bytes differ')
    need(done['complete'] is True and done['cursor'] == _CURSOR and
         done['observations'] == [10000000, 30000000, 100000000] and done['all_checkpoints_retained'] is True and
         done['D_confirm_opened'] is False and done['public_run'] is False and
         done['final_checkpoint'] == expected and expected in done['checkpoints'], 'Terminal producer incomplete')
    verify([expected])
    checkpoint = read(terminal / 'COMPLETE.json')
    meta = read(terminal / 'CHECKPOINT.json')
    same(checkpoint['scope'], scope, 'Terminal checkpoint scope changed')
    same(meta['scope'], scope, 'Terminal metadata scope changed')
    need(checkpoint['complete'] is True and checkpoint['macro_updates'] == 3557 and
         checkpoint['resumable_full_macro_boundary'] is True and meta['cursor'] == _CURSOR and
         sorted(meta['dev_done']) == [10000000, 30000000, 100000000], 'Actual post-dev terminal required')
    audit = meta['audit']
    need(audit['saved_key_count'] == audit['trainable_tensors'] == 393 and audit['lora_pairs'] == 196 and
         audit['trainable_parameters'] == 20187136 and audit['active_served_parameters'] == 616237056 and
         audit['unused_registered_parameters'] == 2048 and len(audit['tensors']) == 393,
         'Terminal393 metadata differs')
    tensors = audit['tensors']
    aa, bb = [v for k, v in tensors.items() if '.lora_A.' in k], [v for k, v in tensors.items() if '.lora_B.' in k]
    heads = [v for k, v in tensors.items() if k.endswith('score.weight')]
    need(len(aa) == len(bb) == 196 and all(v['shape'][0] == 32 for v in aa) and
         all(v['shape'][1] == 32 for v in bb) and len(heads) == 1 and heads[0]['shape'] == [2, 1024] and
         all(v['dtype'] == 'torch.float32' for v in tensors.values()), 'Frozen rank/head/FP32 metadata differs')
    need({str(terminal / n) for n in ('CHECKPOINT.json', 'adapter_model.safetensors', 'adapter_config.json')} <=
         {p['path'] for p in checkpoint['files']}, 'Unbound terminal model loader inputs')
    verify(checkpoint['files'])  # Stat only: no tensor/optimizer payload opened.
    return {'root': str(root), 'arm': arm, 'config_path': str(root / 'seed42.json'), 'cfg': cfg,
            'scope': scope, 'checkpoint': str(terminal), 'terminal': expected,
            'response_name': 'qwen3-0.6b-clean-' + arm + '-seed42-m3557',
            'complete': selected['producer_complete'], 'own_scope': selected['own_scope'],
            'metadata': pin(terminal / 'CHECKPOINT.json'), 'terminal_files': checkpoint['files']}


def _producer_reference(record):
    """Config stores file references, never the historical cfg/scope contents."""
    result = {key: record[key] for key in (
        'root', 'arm', 'config_path', 'checkpoint', 'terminal', 'response_name',
        'complete', 'own_scope', 'metadata', 'terminal_files')}
    result['producer_config'] = pin(record['config_path'])
    return result


def _startup_exact(pins):
    """Cold readback only; discard each historical byte oracle after checking it.

    Between startups these completed documents are stat-bound, not byte-guarded.
    No historical Base64 is returned or embedded in config/SOURCE_SNAPSHOT.
    """
    verify(pins)
    for wanted in pins:
        item = entry(wanted['path'])
        same(item['stat'], wanted, 'Startup historical input changed')
        exact(item)
    verify(pins)


def _data_static(record):
    """Read readiness/source JSON only, never any roster or sidecar payload."""
    cfg, scope = record['cfg'], record['scope']
    root = Path(cfg['data_root'])
    need(root == Path('/tmp/opencode/jev-clean-origin-20261112-v1/prepared-01') and
         not list(root.glob('BLOCKED*')), 'Fixed prepared-data root required')
    verify([cfg['data_complete']])
    complete = read(root / 'COMPLETE.json')
    need(cfg['data_complete']['path'] == str(root / 'COMPLETE.json') and
         all(complete.get(k) is True for k in ('complete', 'DATA_READY', 'train_ready', 'dev_ready')),
         'Actual data readiness required')
    same(scope['data']['complete'], cfg['data_complete'], 'Producer original data completion differs')
    ready = read(root / 'DATA_READY.json')
    need(complete['data_ready'] == pin(root / 'DATA_READY.json') and
         all(ready.get(k) is True for k in ('complete', 'DATA_READY', 'train_ready', 'dev_ready')) and
         ready['unknown_forward_rule'] == 'skip-all-unknown-fields-v1', 'Actual prepared readback required')
    same(scope['data']['ready'], complete['data_ready'], 'Producer original readiness differs')
    need(ready['source_binding']['path'] == str(root / 'SOURCE_BINDING.json') and
         ready['code_binding']['path'] == str(root / 'CODE_BINDING.json'), 'Data source/code binding redirected')
    same(read(root / 'CODE_BINDING.json'), scope['data']['code'], 'Producer original preparation CODE_BINDING differs')
    same(scope['data']['files'], ready['files'], 'Producer data-file binding changed')
    source = read(root / 'SOURCE_BINDING.json')
    same(scope['data']['source_files'], source['files'], 'Producer data-source binding changed')
    for name, encoded in scope['data']['code'].items():
        matches = [p for p in source['files'] if Path(p['path']).name == name]
        need(Path(name).name == name and name.endswith('.py') and len(matches) == 1 and
             entry(Path(matches[0]['resolved']))['base64'] == encoded,
             'Original preparation small-code bytes differ: ' + name)
    final = read(root / 'FINAL_VERIFICATION.json')
    same(final, ready['verification'], 'Prepared verification changed')
    need(all(final.get(k) is True for k in ('complete', 'train_ready', 'dev_ready',
         'exact_full_pair_cross_split_isolation', 'old_roster_identity_gate_applied_to_every_dev_field')),
         'Prepared full identity/isolation proof absent')
    for domain in ('knowledge', 'language', 'retrieval', 'tools', 'arts'):
        counts = final['output_component_counts']
        need(counts['select:' + domain] == 75 and counts['confirm:' + domain] == 50,
             'Frozen select/confirm quotas differ')
    admission = read(cfg['primary_admission']['path'])
    need(admission['complete'] is True and admission['allow_training'] is True and
         admission['admission_scope'] == 'scientific-full-row-train-dev-only' and
         admission['allow_full_row_dev'] is True and admission['allow_prefix_cache'] is False,
         'Original scientific admission required, not public permission')
    same(admission, scope['primary']['admission'], 'Original scientific evidence changed')
    same(admission['precision'], cfg['precision'], 'Scientific precision differs')
    same(admission['core_code'], scope['core_code'], 'Scientific core exact bytes differ')
    groups = [[cfg['data_complete'], complete['data_ready'], ready['source_binding'], ready['code_binding'],
               cfg['primary_admission']], ready['files'], source['files'], admission['evidence_files']]
    boundary = scope.get('state_boundary')
    if boundary is not None:
        # The causal JPAIR Context binds these existing train/select products;
        # final consumer never reads their IDs or invents a confirm state cut.
        groups.append(boundary['guarded_pins'])
        for item in boundary['exact_byte_pins']:
            need(entry(Path(item['resolved']))['base64'] == boundary['source_bytes'][item['path']],
                 'Original boundary preparation source bytes differ')
    return _unique(groups)


def _sdk_paths():
    return sorted(p for p in (_SDK / 'decision_index').rglob('*') if p.is_file() and
                  '__pycache__' not in p.parts and p.suffix != '.pyc') + [_SDK / 'pyproject.toml']


def _sdk_binding():
    need(_SDK_ALIAS.resolve() == _SDK, 'Known SDK alias/canonical source changed')
    _directory(_SDK)
    declared = '25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d'
    old_input = read(_OLD_PUBLIC / 'OFFICIAL_INPUT_BINDING.json')
    need(old_input['upstream_commit'] == declared, 'Official declared revision differs')
    old_binding = read(Path('/tmp/opencode/jev-kd-public-joint-lr1e3-20261111/SOURCE_BINDING.json'))
    need(old_binding['upstream_commit'] == declared, 'Previous SDK snapshot identity differs')
    # Only SDK source pins are selected, not old model/readiness/row artifacts.
    historical = {p['resolved']: p for p in old_input['files'] + old_binding['files']
                  if Path(p['resolved']).is_relative_to(_SDK)}
    mandatory = {'decision_index/runner.py': 5363, 'decision_index/engines/base.py': 2992,
                 'decision_index/suite/io.py': 5214, 'decision_index/pipeline.py': 7532, 'pyproject.toml': 973}
    for relative, size in mandatory.items():
        alias = _SDK_ALIAS / relative
        expected = {'path': str(alias), 'resolved': str(_SDK / relative),
                    'bytes': size, 'mtime_ns': 1794210609000000000}
        verify([expected])
    paths = _sdk_paths()
    need(all(not p.is_symlink() and p.resolve().is_relative_to(_SDK) for p in paths), 'SDK static tree redirected')
    files = [pin(p) for p in paths]
    for item in files:
        if Path(item['path']).suffix == '.py':
            prior = historical.get(item['resolved'])
            need(prior is not None, 'SDK source lacks known prior pin: ' + item['path'])
            verify([prior])
    return {'sdk_root': str(_SDK), 'sdk_alias': str(_SDK_ALIAS), 'sdk_files': files,
            'suite_root': str(_SUITE_ALIAS.resolve()), 'declared_pinned_revision': declared,
            'source_snapshot_is_not_git_checkout': True,
            'known_sdk_controls': [pin(_OLD_PUBLIC / 'OFFICIAL_INPUT_BINDING.json'),
                pin('/tmp/opencode/jev-kd-public-joint-lr1e3-20261111/SOURCE_BINDING.json')]}


def _workers_dead(workers, *, zero):
    need(isinstance(workers, list) and workers, 'Actual prior worker wait receipts required')
    result = []
    for worker in workers:
        need(worker['created'] is True and worker['actual_wait_complete'] is True and
             type(worker['exit_code']) is int and (worker['exit_code'] == 0 if zero else
                                                   worker['exit_code'] in (0, 75)),
             'Prior worker not actually waited at the expected terminal')
        require_dead(worker['process'])
        result.append(worker['process'])
    return result


def _wait_pins(root, workers):
    result = []
    for worker in workers:
        label = worker['arm'] + ('-' + worker['phase'] if 'phase' in worker else '')
        path = root / (label + '-EXIT.json')
        actual = read(path)
        for key in ('arm', 'process', 'gpu', 'uuid', 'output', 'created', 'exit_code', 'actual_wait_complete'):
            same(actual[key], worker[key], 'Actual prior EXIT receipt differs: ' + key)
        if 'phase' in worker:
            same(actual['phase'], worker['phase'], 'Prior EXIT phase differs')
        result.append(pin(path))
    return result


def _lease_record(gpu, metadata, owners, completions, *, allow_held=False):
    path = _LEASES / ('gpu' + str(gpu) + '.lock')
    _directory(_LEASES)
    parent = _LEASES.stat()
    need(parent.st_uid == os.getuid() and stat.S_IMODE(parent.st_mode) == 0o700, 'Unsafe lease directory')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        fs, ls = os.fstat(fd), path.lstat()
        need(stat.S_ISREG(fs.st_mode) and stat.S_ISREG(ls.st_mode) and
             fs.st_uid == ls.st_uid == os.getuid() and fs.st_nlink == ls.st_nlink == 1 and
             stat.S_IMODE(fs.st_mode) == stat.S_IMODE(ls.st_mode) == 0o600 and
             (fs.st_dev, fs.st_ino) == (ls.st_dev, ls.st_ino) and 0 < fs.st_size < 65536,
             'Known regular600/single-link same-inode lease required')
        data = os.pread(fd, fs.st_size + 1, 0)
        same(_parse(data), metadata, 'Unknown old lease metadata; no takeover')
        need(len(data) == fs.st_size, 'Old lease bytes changed while reading')
        # Inspect lock class without taking a lock or writing metadata at install.
        key = (os.major(fs.st_dev), os.minor(fs.st_dev), fs.st_ino)
        for line in Path('/proc/locks').read_text().splitlines():
            fields = line.split()
            if '->' in fields:
                fields.remove('->')
            if len(fields) < 6:
                continue
            device = fields[5].split(':')
            if len(device) == 3 and (int(device[0], 16), int(device[1], 16), int(device[2])) == key:
                need(fields[1] == 'FLOCK' and fields[2] == 'ADVISORY', 'Unexpected old lease lock class')
                need(allow_held, 'Known prior lease still held; install will not acquire or preempt')
        for owner in owners:
            require_dead(owner)
        def stable(s):
            return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid, s.st_nlink,
                    s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(stable(os.fstat(fd)) == stable(fs) and stable(path.lstat()) == stable(ls) and
             os.pread(fd, len(data) + 1, 0) == data,
             'Old lease changed before config sealing')
        return {'lease_path': str(path), 'lease_bytes_base64': base64.b64encode(data).decode('ascii'),
                'lease_inode': [fs.st_dev, fs.st_ino], 'uuid': metadata['uuid'],
                'original_owner': metadata['owner'], 'owners_to_require_dead': owners,
                'completion_pins': completions}
    finally:
        os.close(fd)


def _predecessors(*, allow_held=False):
    packed = read(_PACKED / 'MANAGER_COMPLETE.json')
    launch = read(_PACKED / 'LAUNCH.json')
    owner = read(_PACKED / 'OWNER.json')
    need(packed['execution_complete'] is True and packed['all_actual_exits_zero'] is True and
         packed['owner'] == launch['owner'] == owner['owner'] and
         launch['leases_fsynced_before_spawn'] is True and launch['assignments'] == {'C_D_BCE': 0, 'C_D_CE': 1},
         'Known terminal packed manager completion required')
    require_dead(packed['owner'])
    packed_owners = [packed['owner']] + _workers_dead(packed['workers'], zero=True)
    need(len(packed['workers']) == 2 and {w['gpu'] for w in packed['workers']} == {0, 1}, 'Wrong terminal workers')
    ppins = [pin(_PACKED / n) for n in ('MANAGER_COMPLETE.json', 'LAUNCH.json', 'OWNER.json',
                                          'config.json', 'SOURCE_SNAPSHOT.json')]
    same(launch['config'], owner['config'], 'Prior terminal launch config differs')
    same(owner['config'], pin(_PACKED / 'config.json'), 'Prior terminal config changed')
    same(owner['source_binding'], pin(_PACKED / 'SOURCE_SNAPSHOT.json'), 'Prior terminal source changed')
    ppins += _wait_pins(_PACKED, packed['workers'])
    for arm in ('C_D_BCE', 'C_D_CE'):
        for key in ('complete', 'admission', 'numerical', 'quality'):
            ppins.append(packed['arms'][arm][key])
    failed = read(_FOLLOWUP / 'QUEUE_BLOCKED.json')
    old_launch = read(_FOLLOWUP / 'QUEUE_LAUNCH.json')
    staged = read(_STAGED / 'QUEUE_COMPLETE.json')
    staged_owner = read(_STAGED / 'QUEUE_OWNER.json')
    staged_cfg = read(_STAGED / 'analysis-config.json')
    need(failed['complete'] is False and failed['owner'] == old_launch['owner'] and
         failed['partial_preserved'] is True and failed['automatic_resume'] is False and
         failed['error'].startswith('Known old BI/PB + BOTH new arms + evidence/shared floor cannot fit: ') and
         staged['execution_complete'] is True and staged['all_completed'] is True and
         staged['owner'] == staged_owner['owner'] and staged_owner['control_root'] == str(_STAGED) and
         set(staged['arms']) == {'STATE_NL_PB_P_CE', 'JPAIR_P_BCE_RANK'}, 'Known completed staged successor required')
    all_owners = [failed['owner'], staged['owner']]
    need(len(failed['workers']) == 4 and
         {(w['arm'], w['phase']) for w in failed['workers']} ==
         {(arm, phase) for arm in ('STATE_NL_PB_P_CE', 'JPAIR_P_BCE_RANK') for phase in ('first', 'primary')} and
         all(w['exit_code'] == (75 if w['phase'] == 'first' else 0) for w in failed['workers']) and
         len(staged['workers']) == 2 and {w['arm'] for w in staged['workers']} == {'STATE_NL_PB_P_CE', 'JPAIR_P_BCE_RANK'} and
         all(w['phase'] == 'resume' for w in staged['workers']), 'Known exact old/staged worker roster differs')
    all_owners += _workers_dead(failed['workers'], zero=False) + _workers_dead(staged['workers'], zero=True)
    for old_owner in all_owners:
        require_dead(old_owner)
    sp = [pin(_FOLLOWUP / n) for n in ('QUEUE_BLOCKED.json', 'QUEUE_LAUNCH.json', 'QUEUE_OWNER.json', 'OPS_BUNDLE.json')]
    sp += [pin(_STAGED / n) for n in ('QUEUE_COMPLETE.json', 'QUEUE_OWNER.json', 'analysis-config.json', 'MANIFEST.json')]
    sp += _wait_pins(_FOLLOWUP, failed['workers']) + _wait_pins(_FOLLOWUP, staged['workers'])
    for arm in ('STATE_NL_PB_P_CE', 'JPAIR_P_BCE_RANK'):
        prior = staged['arms'][arm]
        actual = read(_FOLLOWUP / (arm + '-seed42') / 'COMPLETE.json')
        need(prior['cursor'] == actual['cursor'] == _CURSOR and actual['complete'] is True,
             'Latest staged arm completion differs')
        same(actual['scope'], prior['scope'], 'Staged complete own scope differs')
        sp += [prior['complete'], prior['final_checkpoint']]
    result = {}
    for gpu in range(4):
        if gpu < 2:
            metadata = {'root': str(_PACKED), 'gpu': gpu, 'uuid': launch['devices'][str(gpu)],
                        'owner': packed['owner'], 'config': owner['config'], 'source_binding': owner['source_binding'],
                        'advisory_only': True, 'only_fixed375_select': True}
            record = _lease_record(gpu, metadata, packed_owners, ppins, allow_held=allow_held)
        else:
            successor = read(_STAGED / ('gpu' + str(gpu) + '-SUCCESSOR.json'))
            need(successor['owner'] == staged['owner'] and successor['old_owner'] == failed['owner'] and
                 successor['gpu'] == gpu and successor['same_inode_fd_held'] is True and
                 successor['old_metadata_written'] is False and successor['control_root'] == str(_STAGED),
                 'Latest staged exact lease succession differs')
            old = staged_cfg['old_leases'][str(gpu)]
            same(successor['old_lease'], old, 'Staged successor old lease binding differs')
            metadata = {'root': str(_FOLLOWUP), 'gpu': gpu, 'uuid': staged_cfg['devices'][str(gpu)],
                        'owner': failed['owner'], 'bundle': pin(_FOLLOWUP / 'OPS_BUNDLE.json'), 'advisory_only': True}
            record = _lease_record(gpu, metadata, all_owners,
                                   sp + [pin(_STAGED / ('gpu' + str(gpu) + '-SUCCESSOR.json'))],
                                   allow_held=allow_held)
            need(record['lease_bytes_base64'] == old['entry']['base64'] and
                 record['lease_inode'] == [old['device'], old['inode']] and
                 successor['uuid'] == record['uuid'], 'Known old lease identity changed')
        result[str(gpu)] = record
    need(len({r['uuid'] for r in result.values()}) == 4, 'Four distinct prior H800 UUIDs required')
    return result


def install_config(main_reviewed):
    need(main_reviewed is True and Path(__file__).resolve() == ROOT / 'source/common.py',
         'MAIN reviewed deployed source required before install')
    need(not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'decision_index', 'bindings')),
         'Installation must remain stdlib-only; no producer Context or SDK import')
    _directory(ROOT / 'source')
    need({p.name for p in (ROOT / 'source').iterdir()} == set(_SOURCE_NAMES), 'Exactly four deployed sources required')
    for name in _SOURCE_NAMES:
        _readonly(ROOT / 'source' / name)
    frozen = _frozen()
    producers = {role: _producer_record(frozen, role) for role in ROLES}
    sdk = _sdk_binding()
    _directory(Path(sdk['suite_root']))
    # Suite payload is only stat-pinned here. Mandatory SDK verification is later.
    suite_files = [pin(p) for p in sorted(Path(sdk['suite_root']).rglob('*')) if p.is_file()]
    need(suite_files, 'Existing official Suite required')
    groups = [[_FROZEN_PIN, frozen['policy'], frozen['selection_source']],
              frozen['runtime_contract']['stock_files'], frozen['runtime_contract']['core_files'],
              frozen['runtime_contract']['native_files'], [frozen['runtime_contract']['renderer_source']],
              sdk['sdk_files'], sdk['known_sdk_controls'], suite_files]
    exact_paths = {str(ROOT / 'source' / n) for n in _SOURCE_NAMES}
    exact_paths |= {frozen['policy']['path'], frozen['selection_source']['path'],
                     frozen['runtime_contract']['renderer_source']['path']}
    startup_paths = set()
    for record in producers.values():
        cfg = record['cfg']
        groups += [[record['complete'], record['own_scope'], record['terminal'], record['metadata'],
                    pin(record['config_path'])], record['terminal_files'], cfg['trainer_files'], _data_static(record)]
        startup_paths |= {record['config_path'], record['complete']['path'], record['own_scope']['path'],
                          record['terminal']['path'], record['metadata']['path']}
        exact_paths |= {p['resolved'] for p in cfg['trainer_files'] + cfg['core_files'] if Path(p['path']).suffix == '.py'}
        exact_paths |= {p['resolved'] for p in cfg['native_files'] if Path(p['path']).suffix == '.py'}
        exact_paths |= {p['resolved'] for p in record['scope']['data']['source_files'] if Path(p['path']).suffix == '.py'}
        if 'state_boundary' in record['scope']:
            for item in record['scope']['state_boundary']['exact_byte_pins']:
                (exact_paths if Path(item['path']).suffix == '.py' else startup_paths).add(item['resolved'])
    exact_paths |= {p['path'] for p in sdk['sdk_files'] if Path(p['path']).suffix in ('.py', '.toml')}
    predecessors = _predecessors()
    for record in predecessors.values():
        groups.append(record['completion_pins'])
        startup_paths |= {p['path'] for p in record['completion_pins']
                          if Path(p['path']).name in _HISTORY_CONTROLS or
                          Path(p['path']).name.endswith(('-EXIT.json', '-SUCCESSOR.json'))}
    startup_pins = _unique([[pin(path) for path in sorted(startup_paths)]])
    groups.append(startup_pins)
    static = _unique(groups)
    verify(static)
    need({str(p) for p in Path(_STOCK).rglob('*') if p.is_file()} ==
         {p['path'] for p in frozen['runtime_contract']['stock_files']}, 'Unbound or missing stock loader file')
    need(all(pin(path)['bytes'] <= _MAX_CODE_FILE for path in exact_paths) and
         sum(pin(path)['bytes'] for path in exact_paths) <= _MAX_CODE_TOTAL,
         'Hot exact sources must fit the fixed small-code byte bounds')
    entries = [entry(path) for path in sorted(exact_paths)]
    _startup_exact(startup_pins)
    need(next(x for x in entries if x['stat'] == frozen['runtime_contract']['renderer_source'])['base64'] ==
         frozen['runtime_contract']['renderer_source_bytes'], 'Actual frozen renderer bytes differ')
    sources = [pin(ROOT / 'source' / n) for n in _SOURCE_NAMES]
    snapshot = {'version': 'clean-final-source-snapshot-v1', 'root': str(ROOT),
                'source_files': sources, 'static_pins': static, 'exact_entries': entries,
                'identity_guard_policy': _GUARD_POLICY, 'startup_exact_pins': startup_pins,
                'sdk': sdk, 'suite_files': suite_files, 'candidate_binding': _FROZEN_PIN,
                'historical_nested_pins_recursively_verified': False,
                'model_or_roster_payloads_opened': False, 'Torch_imported': False}
    need(len(canonical({k: v for k, v in snapshot.items() if k != 'exact_entries'}).encode('utf-8')) <=
         _MAX_COMPACT and len((canonical(snapshot) + '\n').encode('utf-8')) <= _MAX_SNAPSHOT,
         'Source snapshot metadata/code exceeded its compact bounds')
    commit(ROOT / 'SOURCE_SNAPSHOT.json', snapshot)
    cfg = {'version': 'clean-final-consumer-v1', 'root': str(ROOT), 'main_reviewed': True,
           'frozen_candidates': _FROZEN_PIN, 'source_binding': pin(ROOT / 'SOURCE_SNAPSHOT.json'),
           'source_files': sources, 'sdk_root': sdk['sdk_root'], 'sdk_files': sdk['sdk_files'],
           'suite_root': sdk['suite_root'], 'data_root': producers['baseline']['cfg']['data_root'],
           'stock_directory': _STOCK, 'renderer_source': frozen['runtime_contract']['renderer_source'],
           'resources': _RESOURCES, 'predecessors': predecessors,
           'producer_records': {role: _producer_reference(record) for role, record in producers.items()},
           'identity_guard_policy': _GUARD_POLICY,
           'public_roles': list(ROLES), 'candidate_ready_is_not_production_ready': True,
           'automatic_retry': False, 'training': False}
    need(len((canonical(cfg) + '\n').encode('utf-8')) <= _MAX_COMPACT, 'Final config must remain compact')
    commit(ROOT / 'config.json', cfg)
    same(Guard().cfg, cfg, 'Installed final config readback differs')
    need('torch' not in sys.modules, 'Install imported Torch')
    return cfg


class Guard:
    def __init__(self, config_path=ROOT / 'config.json'):
        self.path = Path(config_path)
        need(self.path == ROOT / 'config.json', 'Only fixed final config is accepted')
        _readonly(self.path)
        need(pin(self.path)['bytes'] <= _MAX_COMPACT, 'Final config is not compact')
        self._config_entry = entry(self.path)
        self.cfg = read(self.path)
        cfg = self.cfg
        need(cfg['version'] == 'clean-final-consumer-v1' and cfg['root'] == str(ROOT) and
             cfg['identity_guard_policy'] == _GUARD_POLICY and
             cfg['main_reviewed'] is True and cfg['public_roles'] == list(ROLES) and
             cfg['stock_directory'] == _STOCK and cfg['sdk_root'] == str(_SDK) and
             cfg['resources'] == _RESOURCES and set(cfg['predecessors']) == {'0', '1', '2', '3'} and
             set(cfg['producer_records']) == set(ROLES) and cfg['automatic_retry'] is False and cfg['training'] is False,
             'Unreviewed/wrong fixed final consumer config')
        same(cfg['frozen_candidates'], _FROZEN_PIN, 'Frozen candidate pointer changed')
        need(cfg['source_binding']['path'] == str(ROOT / 'SOURCE_SNAPSHOT.json'), 'Snapshot pointer redirected')
        verify([cfg['source_binding']])
        _readonly(ROOT / 'SOURCE_SNAPSHOT.json')
        self._snapshot_pin = pin(ROOT / 'SOURCE_SNAPSHOT.json')
        need(self._snapshot_pin['bytes'] <= _MAX_SNAPSHOT, 'Source snapshot is not compact')
        snapshot_entry = entry(ROOT / 'SOURCE_SNAPSHOT.json')
        exact(snapshot_entry)  # Once per Guard construction, never in check().
        self.snapshot = read(ROOT / 'SOURCE_SNAPSHOT.json')
        need(self.snapshot['version'] == 'clean-final-source-snapshot-v1' and
             self.snapshot['identity_guard_policy'] == _GUARD_POLICY and
             self.snapshot['root'] == str(ROOT) and self.snapshot['candidate_binding'] == _FROZEN_PIN and
             cfg['source_files'] == self.snapshot['source_files'] and len(cfg['source_files']) == 4 and
             {p['path'] for p in cfg['source_files']} == {str(ROOT / 'source' / n) for n in _SOURCE_NAMES},
             'Four-source snapshot inventory differs')
        same(cfg['sdk_files'], self.snapshot['sdk']['sdk_files'], 'Canonical SDK source binding differs')
        self.frozen_pin = dict(_FROZEN_PIN)
        self.frozen = _frozen()
        self._cfg_value = canonical(cfg)
        self._frozen_value = canonical(self.frozen)
        self._snapshot_value = canonical({k: v for k, v in self.snapshot.items() if k != 'exact_entries'})
        need(len(self._snapshot_value.encode('utf-8')) <= _MAX_COMPACT, 'Snapshot metadata must stay compact')
        self._hot_entries, oracles, total = [], [], 0
        for item in self.snapshot['exact_entries']:
            need(item['stat']['bytes'] <= _MAX_CODE_FILE, 'Hot entry is not small source')
            data = base64.b64decode(item['base64'], validate=True)
            need(len(data) == item['stat']['bytes'], 'Hot source byte oracle size differs')
            total += len(data)
            self._hot_entries.append({'stat': dict(item['stat']), 'base64': item['base64']})
            oracles.append((dict(item['stat']), data))
        need(total <= _MAX_CODE_TOTAL, 'Hot source oracles exceed fixed aggregate bound')
        self._source_oracles = tuple(oracles)  # Private immutable byte values; no per-request Base64 decode.
        self.check()
        for role in ROLES:
            current = _producer_record(self.frozen, role)
            same(_producer_reference(current), cfg['producer_records'][role], 'Startup producer reference differs')
            verify(_data_static(current))
        same(_sdk_binding(), self.snapshot['sdk'], 'Startup SDK ancestry differs')
        # Workers may start while their own manager holds the known old inode.
        # This permits no lock takeover/acquisition and still validates exact
        # old bytes, inode/class, complete ancestry, actual EXITs and old deaths.
        same(_predecessors(allow_held=True), cfg['predecessors'], 'Startup predecessor ancestry differs')
        _startup_exact(self.snapshot['startup_exact_pins'])
        self.check()

    def check(self):
        need(not list(ROOT.glob('*BLOCKED*')) and not any((ROOT / n).exists() or (ROOT / n).is_symlink()
             for n in ('STOP.json', 'STOP_QUEUE.json', 'STOP_FINAL.json')), 'Own final scope has blocker/STOP; no retry')
        _directory(ROOT)
        _directory(ROOT / 'source')
        need({p.name for p in (ROOT / 'source').iterdir()} == set(_SOURCE_NAMES), 'Unbound final source file')
        for path in (self.path, ROOT / 'SOURCE_SNAPSHOT.json', ROOT / 'FROZEN_CANDIDATES.json',
                     ROOT / 'policy.md', ROOT / 'freeze_candidates.py'):
            _readonly(path)
        exact(self._config_entry)
        verify([self._snapshot_pin])
        need(canonical(self.cfg) == self._cfg_value and canonical(self.frozen) == self._frozen_value and
             canonical({k: v for k, v in self.snapshot.items() if k != 'exact_entries'}) == self._snapshot_value and
             self.snapshot['exact_entries'] == self._hot_entries and self.frozen_pin == _FROZEN_PIN,
             'In-memory frozen config/candidate mutated')
        verify(self.snapshot['static_pins'])
        for wanted, data in self._source_oracles:
            verify([wanted])
            path = Path(wanted['path'])
            need(path.resolve() == path and not path.is_symlink() and path.read_bytes() == data,
                 'Exact hot source bytes changed: ' + str(path))
            verify([wanted])
        for item in self.cfg['source_files']:
            _readonly(item['path'])
        need(_SDK_ALIAS.resolve() == Path(self.cfg['sdk_root']) and
             _SUITE_ALIAS.resolve() == Path(self.cfg['suite_root']), 'Known SDK/Suite alias resolution changed')
        need({str(p) for p in _sdk_paths()} == {p['path'] for p in self.cfg['sdk_files']},
             'Unbound/missing SDK static source or package-data file')
        for record in self.cfg['predecessors'].values():
            verify(record['completion_pins'])
            for old_owner in record['owners_to_require_dead']:
                require_dead(old_owner)
        return None

    def producer(self, role):
        self.check()
        current = _producer_record(self.frozen, role)
        same(_producer_reference(current), self.cfg['producer_records'][role], 'Frozen producer changed since install')
        return {'root': Path(current['root']), 'arm': current['arm'],
                'config_path': Path(current['config_path']), 'cfg': current['cfg'], 'scope': current['scope'],
                'checkpoint': Path(current['checkpoint']), 'terminal': current['terminal'],
                'response_name': current['response_name']}
