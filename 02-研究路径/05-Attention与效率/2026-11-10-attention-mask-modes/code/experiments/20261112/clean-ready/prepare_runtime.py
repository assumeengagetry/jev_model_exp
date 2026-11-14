"""Private bounded execution/identity primitives. Nothing runs on import."""
from __future__ import annotations

import base64
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

from semantic_taint import ROOT, owned, require, stat, verify, dump, write_json

CODE_FILES = ('semantic_taint.py', 'case_pairs.py', 'input_projector.py',
              'prepare_runtime.py', 'census_interface.py', 'integration_driver.py', 'verify_prepared.py')


def merge_bindings(*groups):
    records = {}
    for group in groups:
        for item in group:
            require(set(item) == {'path', 'resolved', 'bytes', 'mtime_ns'}, 'unsupported file identity schema')
            require(item['path'] not in records or records[item['path']] == item, 'conflicting inherited source binding')
            records[item['path']] = item
    result = [records[k] for k in sorted(records)]
    verify(result)
    return result


class Budget:
    def __init__(self, stage, limits):
        self.stage, self.started = stage, time.monotonic()
        self.limits, self.last_disk, self.own_bytes = limits, 0, 0
        self.peak_rss = 0
        require(0 < limits['wall_seconds'] <= 7200 and 0 < limits['cpus'] <= 4, 'CPU/time authorization exceeds configured ceiling')
        require(0 < limits['disk_bytes'] <= 20 * 2**30 and limits['free_reserve_bytes'] >= 25 * 2**30,
                'disk budget/reserve outside conservative policy')
        require(0 < limits['rss_bytes'] <= 8 * 2**30, 'RSS cap outside conservative policy')
        self.lock = (stage / 'RUN.lock').open('x')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.old_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            self.old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self.interrupt)
        signal.setitimer(signal.ITIMER_REAL, limits['wall_seconds'])
        os.setpriority(os.PRIO_PROCESS, 0, 19)
        cpus = sorted(os.sched_getaffinity(0))
        require(cpus, 'no allowed CPUs')
        os.sched_setaffinity(0, set(cpus[-limits['cpus']:]))
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS'):
            os.environ[key] = '1'
        os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false', USE_TORCH='0', USE_TF='0',
                          USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                          TMPDIR=str(stage), SQLITE_TMPDIR=str(stage), HF_HOME=str(stage / 'offline-cache'))
        sys.dont_write_bytecode = True
        self.check(force=True)

    def interrupt(self, signum, frame):
        raise RuntimeError('CPU preparation interrupted/deadline; preserve partial')

    def check(self, force=False, reserve_bytes=0):
        require(type(reserve_bytes) is int and 0 <= reserve_bytes <= 2**20,'invalid final failure-receipt reserve')
        elapsed = time.monotonic() - self.started
        require(elapsed < self.limits['wall_seconds'], 'wall-time budget exhausted')
        require(not (self.stage / 'STOP.json').exists() and not (ROOT / 'STOP.json').exists(), 'explicit STOP.json')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.peak_rss = max(self.peak_rss, rss)
        require(rss <= self.limits['rss_bytes'], 'RSS budget exhausted')
        if force or elapsed - self.last_disk >= 2:
            self.own_bytes = 0
            for folder, dirs, files in os.walk(self.stage, followlinks=False):
                require(not any((Path(folder) / d).is_symlink() for d in dirs), 'symlink inside private stage')
                for name in files:
                    path = Path(folder) / name
                    require(not path.is_symlink(), 'symlink inside private stage')
                    self.own_bytes += path.stat().st_size
            require(self.own_bytes + reserve_bytes <= self.limits['disk_bytes'], 'private disk budget exhausted')
            require(shutil.disk_usage(self.stage).free >= self.limits['free_reserve_bytes'] + reserve_bytes, 'scratch reserve exhausted')
            self.last_disk = elapsed

    def receipt(self):
        self.check(force=True)
        return {'elapsed_seconds': time.monotonic() - self.started, 'peak_rss_bytes': self.peak_rss,
                'own_bytes': self.own_bytes, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'limits': self.limits}

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, previous in self.old_handlers.items():
            signal.signal(sig, previous)
        self.lock.close()


class Identity:
    def __init__(self, config_path, cfg):
        self.code = {name: base64.b64encode((ROOT / name).read_bytes()).decode('ascii') for name in CODE_FILES}
        self.config_text = Path(config_path).read_text()
        self.config_path = config_path
        self.files = merge_bindings(cfg['input_pins'])
        review = cfg['code_review']
        require(review.get('approved') is True and review.get('evidence_files'), 'new CPU driver code review missing')
        require({v['path'] for v in review['reviewed_code']} == {str(ROOT / name) for name in CODE_FILES},
                'code review does not cover every execution module')
        verify(review['reviewed_code'])
        self.files = merge_bindings(self.files, review['evidence_files'], review['reviewed_code'], [stat(config_path)])

    def add(self, records):
        self.files = merge_bindings(self.files, records)

    def require_pins(self, paths):
        pins = {v['path']: v for v in self.files}
        for path in paths:
            require(str(path) in pins and stat(path) == pins[str(path)], 'required reviewed input pin missing: ' + str(path))

    def check(self):
        verify(self.files)
        require(Path(self.config_path).read_text() == self.config_text, 'review config changed during preparation')
        for name, encoded in self.code.items():
            require(base64.b64encode((ROOT / name).read_bytes()).decode('ascii') == encoded, 'CPU source code changed')

    def publish(self, stage):
        self.check()
        write_json(stage / 'CODE_BINDING.json', self.code)
        write_json(stage / 'SOURCE_BINDING.json', {'files': self.files, 'identity_method': 'exact code/config + stat; no hashes'})


def jsonl(path, rows, check):
    with owned(path).open('x', encoding='utf-8') as handle:
        for index, row in enumerate(rows):
            if index % 128 == 0:
                check()
            handle.write(dump(row) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path):
    return json.loads(Path(path).read_text())
