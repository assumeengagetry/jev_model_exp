#!/usr/bin/env python3
"""Read-only next-milestone notifier inside the original 24-hour watch window."""

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SESSION = 'ses_e745f8288ffeB39xHMTRz7qprn'
LEGACY_STATE = Path('/tmp/opencode/monitor-ses_e745f8288ffe-24h-20261030.json')
REGISTRATION = Path(os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state'))) / 'opencode/service.json'
TYPES = ['session.text.ended', 'session.execution.succeeded',
         'session.execution.failed', 'session.execution.interrupted', 'shell.created']
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def belongs(value):
    if isinstance(value, dict):
        return any(belongs(v) for v in value.values())
    if isinstance(value, list):
        return any(belongs(v) for v in value)
    return value == SESSION


def main():
    watch = json.loads(LEGACY_STATE.read_text())
    expires = datetime.fromisoformat(watch['expires']).timestamp()
    print(json.dumps({'observer': 'read-only native next-milestone notification',
                      'sessionID': SESSION, 'expires': watch['expires']}), flush=True)
    while time.time() < expires:
        try:
            endpoint = json.loads(REGISTRATION.read_text())
            headers = {'Accept': 'text/event-stream'}
            if endpoint.get('password') is not None:
                raw = ('opencode:' + endpoint['password']).encode()
                headers['Authorization'] = 'Basic ' + base64.b64encode(raw).decode()
            url = endpoint['url'].rstrip('/') + '/api/event?' + urllib.parse.urlencode([('types', t) for t in TYPES])
            request = urllib.request.Request(url, headers=headers)
            with OPENER.open(request, timeout=min(120, max(1, expires - time.time()))) as response:
                for raw in response:
                    if time.time() >= expires:
                        break
                    line = raw.decode('utf-8', errors='replace').strip()
                    if not line.startswith('data:'):
                        continue
                    event = json.loads(line[5:].strip())
                    if event.get('type') in TYPES and belongs(event.get('data', {})):
                        print(json.dumps({'target_milestone': event['type'], 'sessionID': SESSION,
                                          'time': datetime.now(timezone.utc).isoformat(),
                                          'note': 'This observer never sends prompts, interrupts or process changes to the working session.'}), flush=True)
                        return
        except (OSError, ValueError, TimeoutError) as error:
            print(json.dumps({'observer_retry': type(error).__name__}), flush=True)
        # Reconnect after transport failure/closure; no shell-job completion polling.
        time.sleep(min(3, max(0, expires - time.time())))
    print(json.dumps({'observer': 'original 24-hour watch window ended'}), flush=True)


if __name__ == '__main__':
    main()
