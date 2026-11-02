#!/usr/bin/env python3
"""Official, resumable four-domain KnowLine pair labeling; no training or tests.

Run with HOME/config/experiment-four-domain-distillation.json and the isolated
Teacher Python. Each worker must see exactly one controller-selected CUDA device:

    label-knowline-pairs.py --rank 0 --workers 3
    label-knowline-pairs.py --merge --workers 3

The native release helpers must be next to the local Teacher weights. No network
access, inference server, additional temperature, candidate-group softmax, or
Teacher-side truncation is used. Only a bounded pair block and one prefetched
block are rendered in memory. Existing arrays with incompatible metadata are
preserved and rejected, never silently restarted.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from pair_data_common import DOMAINS
from pair_training_runtime import HOME, POOL, PairPool, TokenStore, assemble


SCHEMA_VERSION = 1
DOMAIN_ORDER = ['knowledge', 'retrieval', 'tools', 'arts']
DOMAIN_PAIRS = dict(knowledge=3073453, retrieval=11047617, tools=1579715, arts=2472921)
TOTAL_PAIRS = 18173706
RENDER_BLOCK_PAIRS = 512
ARRAY_SCAN_PAIRS = 262144
PROGRESS_PAIRS = 1000
PROGRESS_SECONDS = 60
QUESTION = {
    'type': 'noul',
    'instructions': 'Does the candidate correctly match the input and question in the state?',
    'criteria': {
        'true': 'The candidate is a correct match',
        'false': 'The candidate is not a correct match',
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps(dict(event=event, time_utc=utc_now(), **fields), allow_nan=False), flush=True)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, payload):
    path = Path(path)
    # A worker/metadata/merge lock owns this destination and its temporary file.
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


@contextmanager
def locked(path, blocking=False):
    # Do not unlink lock files: that would allow two distinct inodes/writers.
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(f'Another process owns {path}; no probability files changed') from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def require_equal(actual, expected, description):
    if actual != expected:
        raise RuntimeError(f'{description} mismatch; existing artifacts preserved')


def durable_array(array, path):
    array.flush()
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())


def open_array(path, length, mode='r'):
    array = np.load(path, mmap_mode=mode, allow_pickle=False)
    if array.shape != (length,) or array.dtype != np.dtype('float32'):
        raise RuntimeError(f'Wrong probability shape/dtype in {path}; existing file preserved')
    return array


def create_nan_array(path, length):
    """Publish only a fully initialized NaN array, including after an init crash."""
    if path.exists():
        raise RuntimeError(f'Refusing to replace existing probability array {path}')
    temporary = path.with_name(path.stem + '-initializing.npy')
    array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(length,))
    for start in range(0, length, ARRAY_SCAN_PAIRS):
        array[start:start + ARRAY_SCAN_PAIRS] = np.nan
    durable_array(array, temporary)
    del array
    os.replace(temporary, path)
    fsync_directory(path.parent)


def rank_length(total, rank, workers):
    return max(0, (total - rank + workers - 1) // workers)


def local_domain_spans(pool, rank, workers):
    spans = {}
    start = 0
    size = rank_length(len(pool), rank, workers)
    for domain, end in zip(pool.domains, pool.ends):
        # r + i*W belongs to the domain's half-open global interval.
        lo = min(size, max(0, (start - rank + workers - 1) // workers))
        hi = min(size, max(0, (int(end) - rank + workers - 1) // workers))
        spans[domain] = (lo, hi)
        start = int(end)
    return spans


def scan_probabilities(array, spans, allow_pending):
    """Authoritative counts/range checks, bounded memory; NaN alone is pending."""
    counts = {}
    low, high = None, None
    for domain, (begin, end) in spans.items():
        count = 0
        for start in range(begin, end, ARRAY_SCAN_PAIRS):
            block = np.asarray(array[start:min(end, start + ARRAY_SCAN_PAIRS)])
            finite = np.isfinite(block)
            if np.isinf(block).any() or ((block[finite] < 0) | (block[finite] > 1)).any():
                raise RuntimeError('Invalid Teacher probability; no repair or hard-label fallback is allowed')
            if not allow_pending and not finite.all():
                raise RuntimeError('Teacher probability array is incomplete; completion refused')
            count += int(finite.sum())
            if finite.any():
                values = block[finite]
                minimum, maximum = float(values.min()), float(values.max())
                low = minimum if low is None else min(low, minimum)
                high = maximum if high is None else max(high, maximum)
        counts[domain] = count
    return dict(valid_pairs=sum(counts.values()), domain_counts=counts,
                probability_min=low, probability_max=high)


def load_configuration(workers):
    cfg = read_json(HOME / 'config/experiment-four-domain-distillation.json')
    require_equal(cfg['active_training_domains'], DOMAIN_ORDER, 'Active domain order (Language must stay excluded)')
    require_equal(cfg['pairs'], TOTAL_PAIRS, 'Configured pair count')
    require_equal(cfg['max_length'], 4096, 'Student clipping limit')
    require_equal(cfg['teacher'], 'PelaAI/KnowLine-31B-Gen1', 'Teacher release')
    require_equal(cfg['teacher_dtype'], 'bfloat16', 'Teacher dtype')
    require_equal(cfg['teacher_read_twice'], True, 'Native read-twice setting')
    require_equal(cfg['teacher_twice_max_tokens'], 3000, 'Native read-twice cutoff')
    require_equal(cfg['teacher_workers'], workers, 'Configured worker count')
    if not 1 <= cfg['teacher_max_pairs'] <= 16 or not 1 <= cfg['teacher_max_padded_tokens'] <= 8192:
        raise RuntimeError('Teacher batch limits must be positive, at most 16 pairs / 8192 padded tokens')
    require_equal(str(HOME / cfg['training_pool']), str(POOL), 'Shared token/pair pool location')
    # This explicit domain argument is mandatory: the default pool includes Language.
    pool = PairPool('MIXED_ALL', domains=cfg['active_training_domains'])
    require_equal(pool.domains, DOMAIN_ORDER, 'Canonical pool domain order')
    require_equal(len(pool), TOTAL_PAIRS, 'Actual four-domain pool length')
    for domain in DOMAIN_ORDER:
        table = pool.tables[domain]
        lengths = {len(array) for array in table.values()}
        require_equal(lengths, {DOMAIN_PAIRS[domain]}, f'{domain} pair-table sizes')
        require_equal(pool.ready['domains'][domain]['pairs'], DOMAIN_PAIRS[domain], f'{domain} ready count')
        require_equal(pool.ready['domains'][domain]['max_length'], 4096, f'{domain} Student clipping')
    require_equal(sum(pool.ready['domains'][d]['effective_training_tokens'] for d in DOMAIN_ORDER),
                  cfg['effective_student_tokens'], 'Actual four-domain Student token exposure')
    root = HOME / cfg['run_root']
    labels = root / 'teacher-labels'
    labels.mkdir(parents=True, exist_ok=True)
    return cfg, pool, root, labels


def common_identity(cfg, pool, workers):
    cache = read_json(POOL / 'MANIFEST.json')
    require_equal(cache.get('complete'), True, 'Shared token-cache completion')
    return {
        'schema_version': SCHEMA_VERSION,
        'producer': 'label-knowline-pairs.py',
        'run_root': str(HOME / cfg['run_root']),
        'training_pool': str(HOME / cfg['training_pool']),
        'pool_name': 'MIXED_ALL',
        'domain_order': list(pool.domains),
        'domain_counts': dict(DOMAIN_PAIRS),
        'pairs': len(pool),
        'effective_student_tokens': cfg['effective_student_tokens'],
        'workers': workers,
        'global_index_mapping': 'global_index = rank + local_index * workers',
        'dtype': 'float32',
        'student_parent': str(HOME / cfg['student_parent']),
        'common_initialization': str(HOME / cfg['common_initialization']),
        'token_cache': {key: cache[key] for key in ('typed_tokens', 'new_tokens',
                                                   'typed_prefix_boundary', 'typed_suffix_boundary')},
        'input_semantics': {
            'assembly': 'pair_training_runtime.assemble(store,state,candidate,max_length,eos)',
            'max_length': 4096,
            'clipping': 'native prefix/suffix head-tail clipping, identical to Student training',
            'prefix_reserve': 'min(prefix_length,min(1024,max_length//2))',
            'suffix_budget': 'max_length-1-prefix_reserve',
            'prefix_budget': 'max_length-1-clipped_suffix_length',
            'head_tail': 'head=(budget+1)//2; tail=budget-head',
            'decode': {'remove_terminal_eos_only': True, 'skip_special_tokens': False,
                       'clean_up_tokenization_spaces': False},
            'state': 'exact decoded assembled Student pair, not original untruncated text',
            'question': QUESTION,
        },
        'teacher': {
            'release': cfg['teacher'],
            'model_path': str(HOME / cfg['teacher_model_path']),
            'transformers': '5.12.1',
            'torch': '2.9.1+cu128',
            'model_class': 'AutoModelForImageTextToText/Gemma4ForConditionalGeneration',
            'dtype': cfg['teacher_dtype'],
            'attention': 'sdpa',
            'layout': {'read_twice': True, 'twice_max_tokens': 3000, 'header': False,
                       'rotations': [0], 'referenced_fields': False, 'enable_thinking': False},
            'labels': ['A', 'B'],
            'answer_keys': ['true', 'false'],
            'readout': 'float32 last-position A/B logits softmax; P(A)=P(match)',
            'independent_bernoulli': True,
            'additional_temperature': None,
            'folded_temperature_retained': True,
            'additional_truncation': False,
            'max_pairs': cfg['teacher_max_pairs'],
            'max_padded_tokens': cfg['teacher_max_padded_tokens'],
            'long_singleton_policy': 'preserve entire input even above padded-token batch budget',
            'configured_probability': cfg['teacher_probability'],
        },
    }


def ensure_common_metadata(labels, identity, merge=False):
    path = labels / 'METADATA.json'
    with locked(labels / '.metadata.lock', blocking=True):
        if path.exists():
            require_equal(read_json(path), identity, 'Label metadata / parent input semantics')
        else:
            existing = [p for p in labels.iterdir() if not p.name.startswith('.')]
            if merge or existing:
                raise RuntimeError('Missing label metadata alongside existing artifacts; preserved, not adopted')
            atomic_json(path, identity)


def worker_identity(identity, rank):
    return dict(identity=identity, rank=rank, workers=identity['workers'],
                local_pairs=rank_length(identity['pairs'], rank, identity['workers']))


def worker_paths(labels, rank):
    return {
        'array': labels / f'prob-match-worker-{rank}.npy',
        'metadata': labels / f'worker-{rank}-metadata.json',
        'tokenizer_metadata': labels / f'worker-{rank}-tokenizer.json',
        'progress': labels / f'worker-{rank}-progress.json',
        'complete': labels / f'worker-{rank}-complete.json',
    }


def pending_blocks(array):
    for start in range(0, len(array), RENDER_BLOCK_PAIRS):
        block = np.asarray(array[start:start + RENDER_BLOCK_PAIRS])
        pending = np.flatnonzero(np.isnan(block)).astype(np.int64) + start
        if len(pending):
            yield pending


class Renderer:
    """Single CPU producer; actual prompt label IDs are discovered on real data."""

    def __init__(self, cfg, pool, rank, workers):
        import transformers
        from transformers import AutoTokenizer

        require_equal(transformers.__version__, '5.12.1', 'Isolated Teacher Transformers version')
        require_equal(torch.__version__, '2.9.1+cu128', 'Inherited Teacher Torch/CUDA version')
        model_path = HOME / cfg['teacher_model_path']
        sys.path.insert(0, str(model_path))
        import knowline_layout
        import knowline_server

        for module in (knowline_layout, knowline_server):
            require_equal(Path(module.__file__).resolve().parent, model_path.resolve(),
                          'Native KnowLine helper import location')
        self.render_request = knowline_layout.render_request
        self.find_labels = knowline_server.find_labels
        require_equal(knowline_server.options_of(QUESTION)[0], ['true', 'false'], 'Native noul option order')
        self.layout = knowline_layout.Layout(read_twice=True, twice_max_tokens=3000)
        self.student = AutoTokenizer.from_pretrained(HOME / cfg['common_initialization'], local_files_only=True,
                                                     trust_remote_code=False)
        self.teacher = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.teacher.padding_side = 'left'
        if self.student.eos_token_id is None or self.teacher.pad_token_id is None:
            raise RuntimeError('Native Student EOS and Teacher pad token IDs are required')
        self.pool, self.store = pool, TokenStore()
        self.rank, self.workers = rank, workers
        self.label_ids = None

    def render_block(self, local_indices):
        global_indices = self.rank + local_indices * self.workers
        batch = self.pool.fetch(global_indices)
        student_rows = []
        for position, (state, candidate, limit) in enumerate(zip(batch['state'], batch['candidate'], batch['max_length'])):
            limit = int(limit)
            require_equal(limit, 4096, 'Actual pair clipping limit')
            ids = assemble(self.store, state, candidate, limit, self.student.eos_token_id)
            if not 1 <= len(ids) <= limit or int(ids[-1]) != self.student.eos_token_id:
                raise RuntimeError(f'Invalid native Student assembly at pair {int(global_indices[position])}')
            require_equal(len(ids), int(batch['length'][position]), 'Cached versus assembled Student length')
            student_rows.append(ids[:-1].tolist())
        states = self.student.batch_decode(student_rows, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
        prompts, twice = [], []
        for state in states:
            request = {'state': state, 'questions': {'match': QUESTION}}
            rendered = self.render_request(request, self.teacher, self.layout, labels=['A', 'B'])
            if rendered['media']:
                raise RuntimeError('Pair labeling must remain text-only')
            prompt, keys = rendered['prompts']['match'][0]
            require_equal(keys, ['true', 'false'], 'Rendered Teacher answer order')
            if self.label_ids is None:
                labels, ids = self.find_labels(self.teacher, prompt, n=2)
                require_equal(labels, ['A', 'B'], 'Actual single-token labels after the first real prompt')
                if len(set(ids)) != 2:
                    raise RuntimeError('Native Teacher labels must have two distinct token IDs')
                self.label_ids = list(map(int, ids))
            prompts.append(prompt)
            twice.append(bool(rendered['twice']))
        # Native templates already provide BOS/assistant headers. Never append
        # special tokens, truncate to tokenizer.model_max_length, or re-template.
        encoded = self.teacher(prompts, add_special_tokens=False, truncation=False,
                               padding=False, return_attention_mask=False)['input_ids']
        require_equal(len(encoded), len(local_indices), 'Rendered versus requested pair count')
        rows = []
        for i, token_ids in enumerate(encoded):
            if not token_ids:
                raise RuntimeError('Empty native Teacher prompt')
            domain = DOMAINS[int(batch['domain'][i])]
            if domain not in DOMAIN_ORDER:
                raise RuntimeError('Excluded domain reached Teacher labeling')
            expected = self.pool.domains[int(np.searchsorted(self.pool.ends, global_indices[i], side='right'))]
            require_equal(domain, expected, 'Fetched versus canonical domain')
            rows.append(dict(local_index=int(local_indices[i]), global_index=int(global_indices[i]),
                             ids=np.asarray(token_ids, dtype=np.int64), domain=domain,
                             student_tokens=int(batch['length'][i]), read_twice=twice[i]))
        return sorted(rows, key=lambda row: (len(row['ids']), row['local_index']))

    def tokenizer_metadata(self):
        return dict(student_eos_token_id=int(self.student.eos_token_id),
                    student_decode_skip_special_tokens=False, student_decode_clean_up_tokenization_spaces=False,
                    teacher_pad_token_id=int(self.teacher.pad_token_id), labels=['A', 'B'],
                    label_token_ids=self.label_ids, answer_keys=['true', 'false'],
                    label_discovery='native find_labels(tokenizer, first real prompt, n=2)')


def microbatches(rows, max_pairs, max_padded_tokens):
    start = 0
    while start < len(rows):
        end, longest = start, 0
        while end < len(rows) and end - start < max_pairs:
            proposed = max(longest, len(rows[end]['ids']))
            if end > start and proposed * (end - start + 1) > max_padded_tokens:
                break
            longest = proposed
            end += 1
        yield rows[start:end]
        start = end


@torch.inference_mode()
def infer_once(model, rows, pad_token_id, label_ids):
    width = max(len(row['ids']) for row in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        length = len(row['ids'])
        ids[i, width - length:] = torch.from_numpy(row['ids'])
        mask[i, width - length:] = 1
    ids, mask = ids.to('cuda:0'), mask.to('cuda:0')
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                   use_cache=False, logits_to_keep=1, return_dict=True)
    # Left padding puts every real last token at -1, not mask.sum()-1.
    logits = output.logits[:, -1, :].float()
    selected = logits[:, label_ids]
    if not torch.isfinite(selected).all().item():
        raise RuntimeError('Non-finite actual A/B logits; refusing synthetic labels')
    # Softmax on the two labels is equivalent to native full-vocabulary
    # log_softmax followed by native label-only softmax. Temperature is folded.
    probabilities = selected.softmax(-1)[:, 0].cpu().numpy().astype(np.float32, copy=True)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError('Non-finite/out-of-range actual P(match); no fallback allowed')
    return probabilities


def infer_with_splitting(model, rows, pad, label_ids, metrics, rank):
    """Release an OOM traceback before retrying; successful halves are yielded."""
    stack = [rows]
    while stack:
        batch = stack.pop()
        oom = False
        try:
            values = infer_once(model, batch, pad, label_ids)
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            # Outside the except block: the failed forward's traceback and its
            # GPU tensors must be released before empty_cache / smaller retries.
            gc.collect()
            torch.cuda.empty_cache()
            metrics['oom_retries'] += 1
            emit('teacher_actual_oom_split', rank=rank, pairs=len(batch),
                 max_prompt_tokens=max(len(row['ids']) for row in batch),
                 oom_retries=metrics['oom_retries'])
            if len(batch) == 1:
                raise RuntimeError(f'Teacher OOM on intact global pair {batch[0]["global_index"]}; '
                                   'pair remains pending, with no truncation or replacement')
            middle = len(batch) // 2
            stack.extend([batch[middle:], batch[:middle]])
            continue
        yield batch, values


def install_interrupt_handler():
    def interrupted(signum, frame):
        raise InterruptedError(f'Teacher labeling interrupted by signal {signum}; durable entries can resume')
    signal.signal(signal.SIGTERM, interrupted)


def run_worker(cfg, pool, root, labels, identity, rank, workers):
    paths = worker_paths(labels, rank)
    expected = worker_identity(identity, rank)
    size = expected['local_pairs']
    with locked(labels / f'.worker-{rank}.lock'):
        ensure_common_metadata(labels, identity)
        if paths['metadata'].exists():
            require_equal(read_json(paths['metadata']), expected, f'Worker {rank} resume metadata')
        else:
            if any(path.exists() for key, path in paths.items() if key != 'metadata'):
                raise RuntimeError(f'Worker {rank} artifacts lack ownership metadata; preserved')
            atomic_json(paths['metadata'], expected)
        if not paths['array'].exists():
            if paths['complete'].exists() or paths['progress'].exists() or paths['tokenizer_metadata'].exists():
                raise RuntimeError('Worker array is missing alongside previous progress; preserved')
            create_nan_array(paths['array'], size)
        array = open_array(paths['array'], size, mode='r+')
        spans = local_domain_spans(pool, rank, workers)
        summary = scan_probabilities(array, spans, allow_pending=True)
        if summary['valid_pairs'] and not paths['tokenizer_metadata'].exists():
            raise RuntimeError('Finite probabilities lack tokenizer/label provenance; preserved')
        if paths['complete'].exists():
            previous = read_json(paths['complete'])
            require_equal(previous['worker'], expected, 'Completed worker identity')
            require_equal(previous.get('complete'), True, 'Worker completion flag')
            require_equal(summary['valid_pairs'], size, 'Completed worker finite count')
            require_equal(previous['summary'], summary, 'Completed worker actual summary')
            require_equal(previous['tokenizer_metadata'], read_json(paths['tokenizer_metadata']),
                          'Completed worker real-prompt label provenance')
            previous_progress = read_json(paths['progress'])
            require_equal(previous_progress['worker'], expected, 'Completed worker progress identity')
            require_equal(previous_progress['status'], 'complete', 'Completed worker progress status')
            require_equal(previous_progress['summary'], summary, 'Completed worker progress summary')
            emit('teacher_worker_reused', rank=rank, workers=workers, **summary)
            return
        install_interrupt_handler()
        started = time.monotonic()
        initial_valid = summary['valid_pairs']
        last_flushed, last_flush_time = initial_valid, started
        metrics = dict(session_labeled_pairs=0, student_tokens=0, teacher_prompt_tokens=0,
                       teacher_padded_tokens=0, read_twice_pairs=0, read_once_pairs=0,
                       successful_batches=0, oom_retries=0, over_budget_singletons=0)
        status = 'labeling'
        failure = None
        executor = None

        def progress():
            nonlocal last_flushed, last_flush_time
            # Data durability precedes publishing any progress/completion claim.
            durable_array(array, paths['array'])
            payload = dict(worker=expected, status=status, updated_at_utc=utc_now(),
                           local_pairs=size, remaining_pairs=size - summary['valid_pairs'],
                           resumed_valid_pairs=initial_valid, summary=summary.copy(),
                           session_metrics=dict(metrics), session_seconds=time.monotonic() - started,
                           probability_array=str(paths['array']))
            if failure is not None:
                payload['failure'] = failure
            atomic_json(paths['progress'], payload)
            emit('teacher_label_progress', rank=rank, workers=workers, status=status,
                 local_pairs=size, valid_pairs=summary['valid_pairs'],
                 remaining_pairs=size - summary['valid_pairs'], domain_counts=summary['domain_counts'],
                 session_pairs_per_second=metrics['session_labeled_pairs'] / max(0.001, time.monotonic() - started),
                 oom_retries=metrics['oom_retries'])
            last_flushed, last_flush_time = summary['valid_pairs'], time.monotonic()

        try:
            progress()
            blocks = iter(pending_blocks(array))
            first = next(blocks, None)
            if first is not None:
                renderer = Renderer(cfg, pool, rank, workers)
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='teacher-render')
                future = executor.submit(renderer.render_block, first)
                model = None
                while future is not None:
                    rows = future.result()
                    upcoming = next(blocks, None)
                    future = executor.submit(renderer.render_block, upcoming) if upcoming is not None else None
                    if model is None:
                        provenance = renderer.tokenizer_metadata()
                        if paths['tokenizer_metadata'].exists():
                            require_equal(read_json(paths['tokenizer_metadata']), provenance, 'Resumed tokenizer/label IDs')
                        else:
                            atomic_json(paths['tokenizer_metadata'], provenance)
                        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                            raise RuntimeError('Each Teacher process must see exactly one controller-selected CUDA GPU')
                        torch.cuda.set_device(0)
                        if not torch.cuda.is_bf16_supported():
                            raise RuntimeError('Teacher requires native CUDA bfloat16 support')
                        from transformers import AutoModelForImageTextToText
                        model = AutoModelForImageTextToText.from_pretrained(
                            HOME / cfg['teacher_model_path'], dtype=torch.bfloat16,
                            attn_implementation='sdpa', device_map={'': 'cuda:0'}, local_files_only=True,
                            trust_remote_code=False,
                        ).eval()
                        require_equal(model.__class__.__name__, 'Gemma4ForConditionalGeneration', 'Teacher architecture')
                        cap = model.config.get_text_config().final_logit_softcapping
                        if cap is None or not math.isclose(float(cap), 30.0 / 1.44, rel_tol=1e-12):
                            raise RuntimeError('KnowLine folded-temperature soft-cap differs from the native release; '
                                               'weights/config are not modified automatically')
                        emit('teacher_label_model_ready', rank=rank, workers=workers,
                             label_token_ids=renderer.label_ids, resumed_valid_pairs=initial_valid)
                    for batch in microbatches(rows, cfg['teacher_max_pairs'], cfg['teacher_max_padded_tokens']):
                        for successful, values in infer_with_splitting(
                                model, batch, renderer.teacher.pad_token_id, renderer.label_ids, metrics, rank):
                            indices = np.asarray([row['local_index'] for row in successful], dtype=np.int64)
                            if not np.isnan(array[indices]).all():
                                raise RuntimeError('Duplicate local assignment detected; refusing overwrite')
                            require_equal(values.shape, (len(successful),), 'Actual inference batch size')
                            array[indices] = values
                            summary['valid_pairs'] += len(successful)
                            low, high = float(values.min()), float(values.max())
                            summary['probability_min'] = low if summary['probability_min'] is None else min(summary['probability_min'], low)
                            summary['probability_max'] = high if summary['probability_max'] is None else max(summary['probability_max'], high)
                            width = max(len(row['ids']) for row in successful)
                            metrics['session_labeled_pairs'] += len(successful)
                            metrics['successful_batches'] += 1
                            metrics['teacher_padded_tokens'] += len(successful) * width
                            if len(successful) == 1 and width > cfg['teacher_max_padded_tokens']:
                                metrics['over_budget_singletons'] += 1
                            for row in successful:
                                summary['domain_counts'][row['domain']] += 1
                                metrics['student_tokens'] += row['student_tokens']
                                metrics['teacher_prompt_tokens'] += len(row['ids'])
                                metrics['read_twice_pairs' if row['read_twice'] else 'read_once_pairs'] += 1
                            if summary['valid_pairs'] - last_flushed >= PROGRESS_PAIRS or time.monotonic() - last_flush_time >= PROGRESS_SECONDS:
                                progress()
            actual = scan_probabilities(array, spans, allow_pending=False)
            require_equal(actual, summary, 'Actual versus running worker probability counts')
            require_equal(actual['valid_pairs'], size, 'Worker full finite coverage')
            require_equal(actual['domain_counts'], {d: hi - lo for d, (lo, hi) in spans.items()}, 'Worker domain coverage')
            if not paths['tokenizer_metadata'].exists():
                raise RuntimeError('Completed worker is missing real-prompt label metadata')
            status = 'complete'
        except BaseException as exc:
            status = 'interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError, SystemExit)) else 'blocked'
            failure = dict(type=type(exc).__name__, message=str(exc))
            # A signal can arrive between the array assignment and in-memory
            # counters. Recover progress from the actual finite entries, not a
            # possibly interrupted update. The original failure still propagates.
            summary = scan_probabilities(array, spans, allow_pending=True)
            raise
        finally:
            try:
                progress()
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
        completion = dict(complete=True, worker=expected, summary=summary,
                          tokenizer_metadata=read_json(paths['tokenizer_metadata']),
                          completed_at_utc=utc_now(), artifacts=[str(path) for key, path in paths.items() if key != 'complete'])
        atomic_json(paths['complete'], completion)
        emit('teacher_worker_complete', rank=rank, workers=workers, **summary)


def run_merge(cfg, pool, root, labels, identity, workers):
    with ExitStack() as locks:
        locks.enter_context(locked(labels / '.merge.lock'))
        # A merge cannot race an active worker or read a half-published marker.
        for rank in range(workers):
            locks.enter_context(locked(labels / f'.worker-{rank}.lock', blocking=True))
        ensure_common_metadata(labels, identity, merge=True)
        expected_markers = {f'worker-{rank}-complete.json' for rank in range(workers)}
        require_equal({p.name for p in labels.glob('worker-*-complete.json')}, expected_markers,
                      'Exact all-worker completion marker set')
        sources, worker_summaries, artifacts = [], [], [str(labels / 'METADATA.json')]
        tokenizer_metadata = None
        totals = dict.fromkeys(DOMAIN_ORDER, 0)
        for rank in range(workers):
            paths = worker_paths(labels, rank)
            expected = worker_identity(identity, rank)
            require_equal(read_json(paths['metadata']), expected, f'Merge worker {rank} metadata')
            marker = read_json(paths['complete'])
            require_equal(marker.get('complete'), True, f'Worker {rank} completion')
            require_equal(marker['worker'], expected, f'Worker {rank} completion identity')
            provenance = read_json(paths['tokenizer_metadata'])
            require_equal(marker['tokenizer_metadata'], provenance, f'Worker {rank} native label provenance')
            if tokenizer_metadata is None:
                tokenizer_metadata = provenance
            require_equal(provenance, tokenizer_metadata, 'Cross-worker tokenizer and native label IDs')
            array = open_array(paths['array'], expected['local_pairs'])
            summary = scan_probabilities(array, local_domain_spans(pool, rank, workers), allow_pending=False)
            require_equal(summary, marker['summary'], f'Worker {rank} actual versus completed summary')
            require_equal(summary['valid_pairs'], expected['local_pairs'], f'Worker {rank} finite size')
            progress = read_json(paths['progress'])
            require_equal(progress['worker'], expected, f'Worker {rank} progress identity')
            require_equal(progress['status'], 'complete', f'Worker {rank} final progress status')
            require_equal(progress['summary'], summary, f'Worker {rank} final progress count')
            for domain in DOMAIN_ORDER:
                totals[domain] += summary['domain_counts'][domain]
            sources.append(array)
            worker_summaries.append(dict(rank=rank, **summary))
            artifacts.extend(str(path) for path in paths.values())
        require_equal(totals, DOMAIN_PAIRS, 'Merged worker domain counts')
        require_equal(sum(totals.values()), len(pool), 'Merged worker total count')
        merged = labels / 'prob-match.npy'
        staging = labels / 'prob-match-merging.npy'
        metadata = labels / 'merge-metadata.json'
        progress_path = labels / 'merge-progress.json'
        ready = root / 'TEACHER_LABELS_READY.json'
        if metadata.exists():
            require_equal(read_json(metadata), identity, 'Merge resume metadata')
        else:
            if merged.exists() or staging.exists() or ready.exists():
                raise RuntimeError('Merge artifacts lack matching merge metadata; preserved')
            atomic_json(metadata, identity)
        if ready.exists():
            require_equal(read_json(ready).get('complete'), True, 'Existing labels completion flag')
            require_equal(read_json(ready)['identity'], identity, 'Existing labels completion identity')
            if not merged.exists():
                raise RuntimeError('Labels READY marker has no merged array; preserved, not trusted')
        existing_final = merged.exists()
        if existing_final:
            destination = open_array(merged, len(pool))
        else:
            if not staging.exists():
                create_nan_array(staging, len(pool))
            destination = open_array(staging, len(pool), mode='r+')
        copied = 0
        for rank, source in enumerate(sources):
            for start in range(0, len(source), ARRAY_SCAN_PAIRS):
                stop = min(len(source), start + ARRAY_SCAN_PAIRS)
                target = slice(rank + start * workers, rank + stop * workers, workers)
                if existing_final:
                    if not np.array_equal(destination[target], source[start:stop]):
                        raise RuntimeError('Existing merged array disagrees with canonical worker strides; preserved')
                else:
                    destination[target] = source[start:stop]
                copied += stop - start
            if not existing_final:
                durable_array(destination, staging)
            atomic_json(progress_path, dict(identity=identity, status='checking' if existing_final else 'merging',
                                           pairs_copied_or_checked=copied, total_pairs=len(pool),
                                           workers_finished=rank + 1, updated_at_utc=utc_now()))
            emit('teacher_merge_progress', rank=rank, workers=workers, pairs_copied_or_checked=copied,
                 total_pairs=len(pool), existing_final=existing_final)
        spans, start = {}, 0
        for domain, end in zip(pool.domains, pool.ends):
            spans[domain] = (start, int(end))
            start = int(end)
        summary = scan_probabilities(destination, spans, allow_pending=False)
        require_equal(summary['valid_pairs'], TOTAL_PAIRS, 'Final merged finite count')
        require_equal(summary['domain_counts'], DOMAIN_PAIRS, 'Final canonical domain counts')
        if not existing_final:
            durable_array(destination, staging)
            del destination
            os.replace(staging, merged)
            fsync_directory(labels)
        atomic_json(progress_path, dict(identity=identity, status='complete', pairs_copied_or_checked=copied,
                                       total_pairs=len(pool), workers_finished=workers,
                                       summary=summary, updated_at_utc=utc_now()))
        artifacts.extend([str(merged), str(metadata), str(progress_path)])
        completion = dict(complete=True, identity=identity, pairs=TOTAL_PAIRS,
                          active_training_domains=DOMAIN_ORDER, domain_counts=DOMAIN_PAIRS,
                          probability_array=str(merged), dtype='float32', shape=[TOTAL_PAIRS],
                          summary=summary, workers=workers, worker_summaries=worker_summaries,
                          tokenizer_metadata=tokenizer_metadata, completed_at_utc=utc_now(),
                          artifacts=artifacts)
        if ready.exists():
            require_equal(read_json(ready)['summary'], summary, 'Existing READY actual probability summary')
        else:
            # This is the only root-level publication; all arrays and summaries
            # are durable before downstream Student arms may observe complete.
            atomic_json(ready, completion)
        emit('teacher_labels_ready', probability_array=str(merged), workers=workers, **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rank', type=int, help='Worker rank; required unless --merge')
    parser.add_argument('--workers', type=int, required=True, help='Total workers; must match experiment config')
    parser.add_argument('--merge', action='store_true', help='CPU-only merge after exactly all worker markers exist')
    args = parser.parse_args()
    if not 1 <= args.workers <= TOTAL_PAIRS:
        parser.error('--workers must be positive and no greater than the full pair count')
    if args.merge and args.rank is not None:
        parser.error('--rank must not be supplied with --merge')
    if not args.merge and (args.rank is None or not 0 <= args.rank < args.workers):
        parser.error('Worker mode requires 0 <= --rank < --workers')
    cfg, pool, root, labels = load_configuration(args.workers)
    identity = common_identity(cfg, pool, args.workers)
    if args.merge:
        run_merge(cfg, pool, root, labels, identity, args.workers)
    else:
        run_worker(cfg, pool, root, labels, identity, args.rank, args.workers)


if __name__ == '__main__':
    main()
