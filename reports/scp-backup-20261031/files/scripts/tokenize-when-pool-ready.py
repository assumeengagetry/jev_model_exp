#!/usr/bin/env python3
"""Start the existing shared tokenizer at the normalization completion event."""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

HOME = Path('/home/dev/assumeengage')
REGISTRATION = Path(os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state'))) / 'opencode/service.json'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def includes(value, expected):
    if isinstance(value, dict):
        return any(includes(v, expected) for v in value.values())
    if isinstance(value, list):
        return any(includes(v, expected) for v in value)
    return value == expected


def complete(source):
    path = source / 'MANIFEST.json'
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text()).get('complete'))
    except (OSError, ValueError):
        return False


def tokenize(source, output):
    print('SOURCE_READY_STARTING_REAL_TOKENIZATION', source, '->', output, flush=True)
    result = subprocess.run([sys.executable, str(HOME / 'scripts/tokenize-normalized-source.py'),
                             '--source', str(source), '--output', str(output)])
    done = complete(output)
    print(json.dumps({'tokenization_exit_code': result.returncode, 'output': str(output),
                      'cache_complete': done,
                      'note': 'If another writer holds the shared lock, its job remains responsible for completion.'}), flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate-shell', required=True)
    parser.add_argument('--source', type=Path, default=HOME / 'cache/pair-training/global-pool')
    parser.add_argument('--output', type=Path, default=HOME / 'cache/pair-training/tokenized/global-new')
    parser.add_argument('--dependency', type=Path, help='Optional additional completed manifest directory before caching')
    args = parser.parse_args()

    def ready():
        return complete(args.source) and (args.dependency is None or complete(args.dependency))
    watch = json.loads(Path('/tmp/opencode/monitor-ses_e745f8288ffe-24h-20261030.json').read_text())
    expires = datetime.fromisoformat(watch['expires']).timestamp()
    while time.time() < expires:
        try:
            endpoint = json.loads(REGISTRATION.read_text())
            headers = {'Accept': 'text/event-stream'}
            if endpoint.get('password') is not None:
                credentials = ('opencode:' + endpoint['password']).encode()
                headers['Authorization'] = 'Basic ' + base64.b64encode(credentials).decode()
            request = urllib.request.Request(endpoint['url'].rstrip('/') + '/api/event?types=shell.exited', headers=headers)
            with OPENER.open(request, timeout=min(120, max(1, expires - time.time()))) as response:
                # Subscribe first, then inspect once: completion cannot fall in
                # the gap between a readiness snapshot and event subscription.
                if ready():
                    return tokenize(args.source, args.output)
                path = '/api/shell/' + args.gate_shell + '?' + urllib.parse.urlencode({
                    'location[directory]': str(HOME)})
                try:
                    status_request = urllib.request.Request(endpoint['url'].rstrip('/') + path, headers=headers)
                    with OPENER.open(status_request, timeout=15) as status_response:
                        status = json.load(status_response).get('data', {})
                    if status.get('status') == 'exited' and args.dependency is None:
                        print('GATE_EXITED_WITHOUT_COMPLETE_SOURCE', args.gate_shell, flush=True)
                        return 1
                except urllib.error.HTTPError as error:
                    if error.code != 404:
                        raise
                print('WAITING_FOR_NORMALIZATION_EVENT', args.gate_shell, flush=True)
                for raw in response:
                    if time.time() >= expires:
                        break
                    line = raw.decode('utf-8', errors='replace').strip()
                    if not line.startswith('data:'):
                        continue
                    event = json.loads(line[5:].strip())
                    if event.get('type') == 'shell.exited':
                        if ready():
                            return tokenize(args.source, args.output)
                        if includes(event.get('data', {}), args.gate_shell) and args.dependency is None:
                            print('NORMALIZATION_EXITED_WITHOUT_COMPLETE_SOURCE', args.gate_shell, flush=True)
                            return 1
        except (OSError, ValueError, TimeoutError) as error:
            print('EVENT_CONNECTION_RETRY', type(error).__name__, flush=True)
        time.sleep(min(3, max(0, expires - time.time())))
    print('ORIGINAL_WATCH_WINDOW_ENDED_WITHOUT_READY_SOURCE', flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
