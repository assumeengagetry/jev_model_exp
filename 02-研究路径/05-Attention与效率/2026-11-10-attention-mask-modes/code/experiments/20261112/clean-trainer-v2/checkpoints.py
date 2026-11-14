"""Exclusive own-arm commits, full RNG/optimizer/cursor and retained snapshots."""
from __future__ import annotations

from collections import Counter
import fcntl
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

from bindings import (commit_json, dump, fsync_tree, new_directory, read_json, require, stat,
                      verify, write_json)
from fresh_init import adapter_state, audit_model, load_saved_adapter


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_tensor_file(path, value):
    import torch
    # Open exclusive ourselves: torch.save(path) would overwrite unfamiliar work.
    with Path(path).open('xb') as handle:
        torch.save(cpu_tree(value), handle)
        handle.flush()
        os.fsync(handle.fileno())


def load_tensor_file(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def rng_state():
    import torch
    import numpy as np
    kind, state, pos, has_gauss, cached = np.random.get_state()
    return {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all(),
            'numpy': {'kind': str(kind), 'state': state.tolist(), 'pos': int(pos),
                      'has_gauss': int(has_gauss), 'cached_gaussian': float(cached)}}


def restore_rng(value):
    import torch
    import numpy as np
    random.setstate(value['python'])
    torch.set_rng_state(value['torch_cpu'])
    require(len(value['torch_cuda']) == 1, 'World1 RNG required')
    torch.cuda.set_rng_state_all(value['torch_cuda'])
    n = value['numpy']
    np.random.set_state((n['kind'], np.asarray(n['state'], dtype=np.uint32), n['pos'],
                         n['has_gauss'], n['cached_gaussian']))


class MacroCosine:
    """One scheduler step per actual complete macro; upcoming update is 1-based."""
    def __init__(self, optimizer, total_steps):
        self.optimizer, self.total_steps = optimizer, total_steps
        self.warmup_steps = max(1, math.ceil(0.03 * total_steps))
        self.completed = 0
        self.apply()

    def next_lr(self):
        upcoming = min(self.completed + 1, self.total_steps)
        if self.completed >= self.total_steps:
            return 0.0
        if upcoming <= self.warmup_steps:
            return 2e-5 * upcoming / self.warmup_steps
        progress = (upcoming - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 2e-5 * 0.5 * (1.0 + math.cos(math.pi * progress))

    def apply(self):
        for group in self.optimizer.param_groups:
            group['lr'] = self.next_lr()

    def step(self):
        require(self.completed < self.total_steps, 'Scheduler overrun')
        self.completed += 1
        self.apply()

    def state_dict(self):
        return {'kind': 'actual-macro-warmup-cosine-v1', 'total_steps': self.total_steps,
                'warmup_steps': self.warmup_steps, 'completed': self.completed, 'next_lr': self.next_lr()}

    def load_state_dict(self, value):
        require(value['kind'] == 'actual-macro-warmup-cosine-v1' and
                value['total_steps'] == self.total_steps and value['warmup_steps'] == self.warmup_steps and
                0 <= value['completed'] <= self.total_steps, 'Scheduler scope differs')
        self.completed = value['completed']
        require(value['next_lr'] == self.next_lr(), 'Scheduler LR changed')
        self.apply()


class Run:
    def __init__(self, ctx, output, *, resume=None, resume_pin=None):
        self.ctx, self.started = ctx, time.monotonic()
        self.stop_requested, self.stop_signal = False, None
        self.previous_handlers = {}
        self.resume, self.restored = None, None
        self.resume_commit = None
        self.last_checkpoint = None
        self.checkpoint_counter = 0
        require((resume is None) == (resume_pin is None), 'Resume requires checkpoint AND separate reviewed COMPLETE pin')
        if resume is None:
            self.root = new_directory(output, ctx.cfg['output_parent'])
            self.lock = (self.root / 'RUN.lock').open('x+')
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(self.root / 'OWN_SCOPE.json', ctx.snapshot)
            self.events = (self.root / 'ledger.jsonl').open('x', encoding='utf-8')
        else:
            self.root = Path(output)
            parent = Path(ctx.cfg['output_parent']).resolve()
            require(self.root.resolve().is_relative_to(parent) and self.root.resolve() != parent and
                    not self.root.is_symlink(), 'Resume outside own namespace')
            require(not (self.root / 'COMPLETE.json').exists(), 'Completed arms never implicitly reused/resumed')
            require(read_json(self.root / 'OWN_SCOPE.json') == ctx.snapshot, 'Resume data/mode/loss/layout/seed/source scope differs')
            self.lock = (self.root / 'RUN.lock').open('r+')
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.resume = Path(resume)
            self.last_checkpoint = self.resume
            require(self.resume.parent.resolve() == self.root.resolve() and
                    not self.resume.is_symlink(), 'Only own checkpoint is resumable')
            pin = read_json(resume_pin)
            verify([pin])
            require(pin['path'] == str(self.resume / 'COMPLETE.json'), 'Resume commit pin redirected')
            commit = read_json(self.resume / 'COMPLETE.json')
            require(commit['complete'] is True and commit['scope'] == ctx.snapshot and
                    commit['resumable_full_macro_boundary'] is True, 'Checkpoint not own full-macro boundary')
            verify(commit['files'])
            require({str(p) for p in self.resume.rglob('*') if p.is_file() and p.name != 'COMPLETE.json'} ==
                    {r['path'] for r in commit['files']}, 'Checkpoint has unbound/missing files')
            metadata = read_json(self.resume / 'CHECKPOINT.json')
            require(metadata['scope'] == ctx.snapshot and
                    metadata['cursor']['macro_updates'] == commit['macro_updates'], 'Checkpoint metadata scope/cursor differs')
            # Do not import torch here: CUDA/CUBLAS precision setup must happen first.
            self.resume_commit = commit
            # Refuse rewind over a completed macro, partial macro, or later checkpoint.
            with (self.root / 'ledger.jsonl').open(encoding='utf-8') as handle:
                entries = [__import__('json').loads(line) for line in handle]
            committed = [e for e in entries if e['event'] == 'checkpoint_committed']
            require(committed and committed[-1]['complete'] == pin, 'Resume must bind own latest checkpoint tip')
            tail = entries[entries.index(committed[-1]) + 1:]
            require(not any(e['event'] in ('macro_started', 'macro_updated', 'dev_started', 'failed') for e in tail),
                    'Uncheckpointed/failed model work after this boundary: no silent exposure rewind')
            self.checkpoint_counter = max(e['sequence'] for e in committed)
            self.events = (self.root / 'ledger.jsonl').open('a', encoding='utf-8')
            self.event('explicit_own_resume', checkpoint=str(self.resume), complete=pin,
                       resume_pin_file=stat(resume_pin))
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self.signal_stop)
        self.budget()

    def signal_stop(self, signum, frame):
        # Never raise in the middle of a macro or kill any other process.
        self.stop_requested, self.stop_signal = True, signum

    def should_stop(self):
        return self.stop_requested or (self.root / 'STOP.json').exists()

    def event(self, event, **values):
        self.events.write(dump({'event': event, 'elapsed_this_invocation': time.monotonic() - self.started,
                               **values}) + '\n')
        self.events.flush()
        os.fsync(self.events.fileno())

    def budget(self):
        resources = self.ctx.cfg['resources']
        require(type(resources['max_output_bytes']) is int and resources['max_output_bytes'] > 0 and
                type(resources['free_reserve_bytes']) is int and resources['free_reserve_bytes'] > 0,
                'Explicit private disk/reserve budget required')
        used = 0
        for folder, dirs, files in os.walk(self.root, followlinks=False):
            require(not any((Path(folder) / d).is_symlink() for d in dirs), 'Symlink own output directory')
            for name in files:
                p = Path(folder) / name
                require(not p.is_symlink(), 'Symlink own output file')
                used += p.stat().st_size
        require(used <= resources['max_output_bytes'] and
                shutil.disk_usage(self.root).free >= resources['free_reserve_bytes'], 'Private disk budget/reserve exhausted')
        return {'own_bytes': used, 'free_bytes': shutil.disk_usage(self.root).free}

    def checkpoint(self, model, optimizer, scheduler, runtime, cursor, dev_done, identity, reasons):
        import torch
        self.ctx.check()
        self.budget()
        require(cursor['macro_updates'] == scheduler.completed and cursor['cases'] == cursor['macro_updates'] * 16,
                'Only complete macro checkpoint allowed')
        self.checkpoint_counter += 1
        name = f"checkpoint-m{cursor['macro_updates']:07d}-s{self.checkpoint_counter:05d}"
        path = self.root / name
        path.mkdir(mode=0o700)  # No exist_ok; retain every checkpoint.
        self.event('checkpoint_started', path=str(path), reasons=reasons, cursor=cursor)
        model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
        model.config.save_pretrained(path)
        state = {'scope': self.ctx.snapshot, 'cursor': dict(cursor), 'dev_done': list(dev_done),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'rng': rng_state(), 'runtime': runtime.receipt(), 'initialization': identity,
                 'trainable_names': sorted(n for n, p in model.named_parameters() if p.requires_grad),
                 'units': {'logical': 'complete supervised P+candidate+EOS pair tokens per exposure',
                           'actual_forward_real': 'all actually completed first/replay full-row tokens',
                           'actual_tokenwise': 'per-operator physical T rows, padded in dense only',
                           'optimizer': 'one fused AdamW update per16-case macro'},
                 'reasons': reasons}
        save_tensor_file(path / 'training_state.pt', state)
        write_json(path / 'CHECKPOINT.json', {'scope': self.ctx.snapshot, 'cursor': cursor,
                                             'runtime': runtime.receipt(), 'initialization': identity,
                                             'audit': audit_model(model), 'dev_done': dev_done,
                                             'reasons': reasons})
        fsync_tree(path)
        self.budget()
        files = [stat(p) for p in sorted(path.rglob('*')) if p.is_file()]
        commit_json(path / 'COMPLETE.json', {'complete': True, 'scope': self.ctx.snapshot,
                                           'macro_updates': cursor['macro_updates'],
                                           'resumable_full_macro_boundary': True, 'files': files})
        fsync_tree(path)
        self.event('checkpoint_committed', complete=stat(path / 'COMPLETE.json'),
                   sequence=self.checkpoint_counter, cursor=cursor, reasons=reasons)
        self.last_checkpoint = path
        return path

    def restore(self, model, optimizer, scheduler, runtime):
        require(self.resume is not None and self.resume_commit is not None, 'No explicit resume bound')
        state = load_tensor_file(self.resume / 'training_state.pt')
        self.restored = state
        metadata = read_json(self.resume / 'CHECKPOINT.json')
        require(state['scope'] == self.ctx.snapshot and state['cursor'] == metadata['cursor'] and
                state['cursor']['macro_updates'] == self.resume_commit['macro_updates'] and
                state['dev_done'] == metadata['dev_done'] and state['runtime'] == metadata['runtime'] and
                state['initialization'] == metadata['initialization'], 'Training state differs from pinned metadata')
        expected_names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        require(state['trainable_names'] == expected_names and len(expected_names) == 393,
                'Optimizer parameter identity/order differs')
        groups = state['optimizer']['param_groups']
        require(len(groups) == 1 and groups[0]['param_names'] == expected_names and len(groups[0]['params']) == 393,
                'Saved optimizer group order differs from all393 leaves')
        require(len(state['optimizer']['state']) == (393 if state['cursor']['macro_updates'] else 0),
                'Saved optimizer state count differs from actual macro cursor')
        load_saved_adapter(model, self.resume)
        scheduler.load_state_dict(state['scheduler'])
        optimizer.load_state_dict(state['optimizer'])
        require(all(g['lr'] == scheduler.next_lr() for g in optimizer.param_groups), 'Restored optimizer LR differs')
        runtime.counts = Counter(state['runtime']['forward'])
        runtime.native.counts = Counter(state['runtime']['native_forward_calls'])
        restore_rng(state['rng'])
        return dict(state['cursor']), list(state['dev_done']), state['initialization']

    def failure(self, error, cursor, runtime=None):
        # No partial checkpoint is advertised as resumable or COMPLETE.
        self.event('failed', error_type=type(error).__name__, error=str(error), cursor=cursor,
                   runtime=None if runtime is None else runtime.receipt())
        index = 1
        while (self.root / f'BLOCKED-{index:04d}.json').exists():
            index += 1
        write_json(self.root / f'BLOCKED-{index:04d}.json',
                   {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                    'cursor_last_full_macro': cursor, 'partial_preserved': True, 'automatic_resume': False})

    def close(self):
        for sig, handler in self.previous_handlers.items():
            signal.signal(sig, handler)
        self.events.close()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
