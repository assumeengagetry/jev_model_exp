#!/usr/bin/env python3
"""One-shot, explicit-file incremental SCP. No credentials, uploads, Git or watcher."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path('/home/dev/assumeengage')
LIMIT = 10 * 1024 * 1024
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
           '-o', 'ClearAllForwardings=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none']
SUFFIXES = {'.py', '.sh', '.cjs', '.js', '.json', '.csv', '.md', '.txt', '.yaml',
            '.yml', '.toml', '.jinja', '.svg', '.png', '.html', '.css', '.log', '.ini', '.cfg'}


def validate_source(value, root=ROOT):
    path = PurePosixPath(value)
    if '..' in path.parts:
        raise ValueError('Refusing source path traversal')
    try:
        relative = path.relative_to(str(root)) if path.is_absolute() else path
    except ValueError:
        raise ValueError('Refusing source outside the approved remote root') from None
    if not relative.parts or any(p.startswith('.') or not re.fullmatch(r'[\w.+-]+', p) for p in relative.parts):
        raise ValueError('Refusing hidden, wildcard or unsafe source path')
    if any(p in {'envs', 'private-transfer', 'credentials', 'credential', 'secrets', 'auth'} for p in relative.parts):
        raise ValueError('Refusing credentials, private transfer or environment directories')
    if relative.name in {'env.sh', 'env.fish', 'pip.conf', 'condarc', 'existing-pod-endpoint.json'}:
        raise ValueError('Refusing private environment or endpoint configuration')
    if re.search(r'(^|[_-])(token|cookie|password|secret|credentials?)([._-]|$)', relative.stem, re.I):
        raise ValueError('Refusing credential-like filename')
    if relative.suffix not in SUFFIXES and relative.name not in {'LICENSE', 'NOTICE', 'PKG-INFO'}:
        raise ValueError('Refusing weights, binary caches, raw data or unsupported file type')
    if relative.parts[0] == 'cache' and not str(relative).startswith('cache/pair-training/'):
        raise ValueError('Refusing authentication, browser or public Hub caches')
    if relative.parts[0] in {'cache', 'datasets'} and relative.suffix not in {'.json', '.yaml', '.yml', '.md', '.txt'}:
        raise ValueError('Only metadata is allowed from dataset/cache directories')
    if re.search(r'(^|[-_])(train|test|validation|dev|examples|samples|records|predictions?)([-_.]|$)', relative.name, re.I) and relative.parts[0] in {'datasets', 'cache'}:
        raise ValueError('Refusing raw dataset or prediction content')
    return str(PurePosixPath(str(root)) / relative)


def check_payload(data, name):
    if Path(name).suffix == '.png':
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Invalid PNG payload')
        return
    text = data.decode('utf-8')
    patterns = [r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
                r'\b(?:hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
                r'https?://[^\s/:]+:[^\s/@]+@',
                r'[?&](?:X-Amz-Signature|access_token|refresh_token)=[A-Za-z0-9%_-]{16,}',
                r'(?i)Bearer\s+[A-Za-z0-9._~+/=-]{24,}',
                r'''(?i)["']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|client[_-]?secret|hf_token|modelscope_token|modelscope_api_token|modelscope_sdk_token|ms_token)["']?\s*[:=]\s*["'][A-Za-z0-9_+/=.-]{16,}["']''']
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError('Sensitive payload rejected; contents are not returned')
    if Path(name).suffix == '.json':
        obj = json.loads(text)
        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower() in {'token', 'api_key', 'secret', 'password', 'cookie', 'authorization', 'git_token', 'modelscope_api_token', 'modelscope_sdk_token', 'modelscope_token', 'ms_token'} and isinstance(item, str) and len(item) >= 12:
                        raise ValueError('Sensitive JSON field rejected')
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(obj)


def probe_file(value, root=ROOT, max_bytes=LIMIT):
    path = Path(validate_source(value, root))
    for parent in [path, *path.parents]:
        if parent == Path(root).parent:
            break
        if parent.is_symlink():
            raise ValueError('Symlink sources are not accepted; use the canonical file path')
    if not path.is_file():
        raise ValueError('Only explicit regular files are accepted, not directories')
    before = path.stat()
    if before.st_size > max_bytes:
        raise ValueError('Source exceeds the small-file limit')
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Source changed during read; try again after it stabilizes')
    check_payload(data, path.name)
    relative = path.relative_to(root)
    snapshot = path.suffix == '.log' or path.name.lower() in {'latest.json', 'pipeline.json', 'takeover.json', 'status.json', 'progress.json'}
    if path.suffix == '.json':
        obj = json.loads(data)
        if isinstance(obj, dict) and (obj.get('complete') is False or str(obj.get('state', obj.get('status', ''))).lower() in {'running', 'queued', 'pending', 'in_progress', 'training'}):
            snapshot = True
    return {'source': str(path), 'relative': str(relative), 'size': len(data),
            'sha256': hashlib.sha256(data).hexdigest(), 'mtime_ns': after.st_mtime_ns, 'snapshot': snapshot}


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def already_backed_up(backup, meta):
    manifest = read_json(Path(backup) / 'SCP-COMPLETED.json', {'files': []})
    for entry in reversed(manifest['files']):
        if entry['source'] != meta['source'] or entry['sha256'] != meta['sha256']:
            continue
        path = Path(entry['local_path'])
        if path.resolve().is_relative_to(Path(backup).resolve()) and path.is_file() and not path.is_symlink() and path.stat().st_size == meta['size'] and hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']:
            return True
    return False


def record_copy(backup, downloaded, meta):
    backup = Path(backup).resolve()
    if validate_source(meta['relative']) != meta['source']:
        raise ValueError('Source metadata does not match its approved relative path')
    data = Path(downloaded).read_bytes()
    if len(data) > LIMIT or len(data) != meta['size'] or hashlib.sha256(data).hexdigest() != meta['sha256']:
        raise ValueError('Downloaded size/hash mismatch; not registering the file')
    check_payload(data, meta['relative'])
    manifest_path = backup / 'SCP-COMPLETED.json'
    manifest = read_json(manifest_path, {'files': []})
    for old in manifest['files']:
        old_path = Path(old['local_path'])
        if not old_path.resolve().is_relative_to(backup) or not old_path.is_file() or old_path.is_symlink() or hashlib.sha256(old_path.read_bytes()).hexdigest() != old['sha256']:
            raise ValueError('An existing completion entry no longer verifies; refusing to hide that gap')
    now = datetime.now(timezone.utc)
    relative = Path(meta['relative'])
    if relative.name == 'AGENTS.md':
        relative = relative.with_name('REMOTE-AGENTS.md')
    destination = backup / 'incremental' / (now.strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]) / 'files' / relative
    if not destination.resolve().is_relative_to(backup):
        raise ValueError('Destination escapes the backup')
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(downloaded, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Installed file does not verify')
    record = {'source': meta['source'], 'local_path': str(destination),
              'backup_relative_path': str(destination.relative_to(backup)),
              'bytes': len(data), 'sha256': meta['sha256'],
              'backup_status': 'snapshot_verified' if meta['snapshot'] else 'scp_verified',
              'completion_status': 'snapshot_nonfinal' if meta['snapshot'] else 'stable_file_not_execution_claim',
              'nonfinal_snapshot': meta['snapshot'], 'source_observed_mtime_ns': meta['mtime_ns'],
              'local_verified_utc': now.isoformat(), 'present_on_disk': True, 'origin': 'incremental_scp'}
    manifest['files'].append(record)
    manifest.update({'generated_utc': now.isoformat(), 'scope': 'Actually present and hash-verified source file versions; excludes inventory-only entries and reused large assets',
                     'file_count': len(manifest['files']), 'unique_sources': len({e['source'] for e in manifest['files']}),
                     'total_bytes': sum(e['bytes'] for e in manifest['files']),
                     'stable_files': sum(not e['nonfinal_snapshot'] for e in manifest['files']),
                     'nonfinal_snapshots': sum(e['nonfinal_snapshot'] for e in manifest['files'])})
    write_json(manifest_path, manifest)
    with (backup / 'SCP-COMPLETED.csv').open('w', newline='') as stream:
        columns = ['source', 'local_path', 'bytes', 'sha256', 'backup_status', 'completion_status', 'nonfinal_snapshot', 'local_verified_utc', 'present_on_disk']
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(manifest['files'])
    ledger_path = backup / 'LEDGER.json'
    ledger = read_json(ledger_path, {'entries': []})
    entry = next((e for e in ledger['entries'] if e['source'] == meta['source']), None)
    if entry is None:
        entry = {'source': meta['source'], 'path': meta['relative'], 'category': Path(meta['relative']).parts[0]}
        ledger['entries'].append(entry)
    if entry.get('installed'):
        entry.setdefault('previous_backups', []).append({k: entry.get(k) for k in ['local', 'local_absolute', 'size', 'sha256', 'status', 'verified_at_utc']})
    entry.update({'local': record['backup_relative_path'], 'local_absolute': str(destination), 'size': len(data),
                  'sha256': meta['sha256'], 'mtime_ns': meta['mtime_ns'], 'installed': True,
                  'status': record['backup_status'], 'completion_status': record['completion_status'], 'verified_at_utc': now.isoformat()})
    ledger.update({'updated_utc': now.isoformat(), 'completion_manifest': 'SCP-COMPLETED.json', 'inventory_entries_are_not_backup_proof': True})
    write_json(ledger_path, ledger)
    csv_path = backup / 'LEDGER.csv'
    columns = ['source', 'local', 'local_absolute', 'size', 'category', 'completion_status', 'status', 'installed', 'sha256', 'verified_at_utc']
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(ledger['entries'])
    return record


def remote_probe(sources):
    approved = [validate_source(source) for source in sources]
    code = Path(__file__).read_text()
    command = 'python3 -B -c ' + shlex.quote(code) + ' --probe'
    result = subprocess.run(['ssh', *OPTIONS, 'sh-09-gpu', command], input=json.dumps(approved),
                            text=True, stdout=subprocess.PIPE, check=True, timeout=90)
    records = json.loads(result.stdout)
    if {e['source'] for e in records} != set(approved):
        raise ValueError('Remote metadata does not match the requested sources')
    return records


def main():
    if sys.argv[1:] == ['--probe']:
        print(json.dumps([probe_file(source) for source in json.load(sys.stdin)]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, help='Explicit small file, relative to the approved root; repeat for multiple files. No directories/globs.')
    parser.add_argument('--dry-run', action='store_true', help='Read-only metadata probe; no SCP or local ledger changes')
    args = parser.parse_args()
    backup = Path(__file__).resolve().parent.parent
    metadata = remote_probe(args.source)
    copied = 0
    for meta in metadata:
        if already_backed_up(backup, meta):
            print('UNCHANGED_ALREADY_VERIFIED:', meta['relative']); continue
        if args.dry_run:
            print('WOULD_SCP:', meta['relative'], 'bytes=' + str(meta['size'])); continue
        with tempfile.TemporaryDirectory(prefix='scp-incremental-', dir='/tmp/opencode') as temporary:
            downloaded = Path(temporary) / Path(meta['relative']).name
            subprocess.run(['scp', '-p', *OPTIONS, 'sh-09-gpu:' + meta['source'], str(downloaded)], check=True, timeout=120)
            after = remote_probe([meta['source']])[0]
            if after['sha256'] != meta['sha256'] and not meta['snapshot']:
                raise ValueError('Stable source changed during transfer; no completion entry was written')
            record = record_copy(backup, downloaded, meta)
            print('SCP_VERIFIED:', record['backup_relative_path']); copied += 1
    print('DONE: copied=' + str(copied) + '; no authentication probes, uploads, Git operations or watcher')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except subprocess.SubprocessError:
        print('Transport failed; this is a one-shot command with no automatic retry.', file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
