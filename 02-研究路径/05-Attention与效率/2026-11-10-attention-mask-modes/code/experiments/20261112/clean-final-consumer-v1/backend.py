"""Frozen clean393, causal packed native-FA3 full-row consumer.

Imports here are stdlib-only. configure_role is called outside SDK.run because
that runner imports Torch before constructing its duck-typed engine.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import MappingProxyType

from common import (Guard, ROOT, ROLES, canonical, commit, entry, exact, identity,
                    need, pin, read, same, verify)

sys.dont_write_bytecode = True
_CONFIGURED = None
_RENDERER = None
_MODEL_CONSTRUCTED = False
_MAX_CONTROL_FILE = 256 * 2**10
_MAX_CURRENT_CONTROLS = 2**20
_MAX_CONTEXT_CONTRACT = 16 * 2**10


def _renderer(guard=None):
    global _RENDERER
    if _RENDERER is None:
        guard = Guard() if guard is None else guard
        item = {'stat': guard.cfg['renderer_source'],
                'base64': guard.frozen['runtime_contract']['renderer_source_bytes']}
        exact(item)
        path = Path(item['stat']['path'])
        need(path == Path('/home/dev/assumeengage/scripts/pair_data_common.py'), 'Wrong original renderer')
        spec = importlib.util.spec_from_file_location('_jev_final_original_pair_renderer', path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        need(Path(helper.__file__).resolve() == path and
             all(callable(getattr(helper, n, None)) for n in ('text', 'criteria_context', 'render_pair')),
             'Original renderer ABI missing')
        _RENDERER = (helper, item)
    helper, item = _RENDERER
    exact(item)
    return helper


def _control_entry(path):
    need(pin(path)['bytes'] <= _MAX_CONTROL_FILE, 'Current worker control must remain compact')
    return entry(path)


def _context_contract(ctx):
    """Only fields actually used by this frozen full-row execution, no ancestry.

    Full original Context snapshot/config equivalence is a startup check. This
    projection never walks snapshot.data/primary/state_boundary/code blobs.
    The original Context/config/arm objects are retained without modification.
    """
    return canonical({
        'config_path': str(ctx.config_path), 'arm_name': ctx.arm_name, 'arm': ctx.arm,
        'cfg': {key: ctx.cfg[key] for key in (
            'version', 'seed', 'stock_directory', 'core_directory', 'native_directory',
            'native_module', 'precision', 'prefix_cache', 'world_size', 'device', 'micro_plan')},
        'snapshot': {key: ctx.snapshot[key] for key in ('version', 'config_pin', 'arm_name', 'arm')},
    })


def _json_input(value):
    """Validate JSON values without reordering dictionary keys or rewriting text."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        need(math.isfinite(value), 'Nonfinite request input')
        return
    if type(value) is list:
        for item in value:
            _json_input(item)
        return
    need(type(value) is dict and all(type(k) is str for k in value), 'Only JSON request values are accepted')
    for item in value.values():
        _json_input(item)


def field_tokens(tokenizer, state, question, options):
    """Input-only shared training renderer; no target, model or benchmark access."""
    need(type(options) is list and options and all(type(v) is str for v in options),
         'All original candidate strings, without filtering, are required')
    need(tokenizer.eos_token_id == 151645 and tokenizer.pad_token_id == 151643,
         'Original stock tokenizer/EOS/pad IDs required')
    _json_input(state)
    _json_input(question)
    helper = _renderer()
    instruction = helper.text(question)
    # Native compile_case passes the original candidate strings to the context
    # helper (including their whitespace), and strips each suffix separately.
    # The public parser already applies the old API text() conversion first.
    context = helper.criteria_context(instruction, options)
    prefix, _ = helper.render_pair(state, instruction, options[0], context)
    def encode(text):
        ids = list(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])
        need(all(type(t) is int and 0 <= t < len(tokenizer) for t in ids), 'Illegal original token ID')
        return ids
    prefix_ids = encode(prefix)
    suffixes = [encode(helper.text(value)) + [151645] for value in options]
    need(prefix_ids and all(s and s[-1] == 151645 for s in suffixes), 'Original full pair/EOS missing')
    return prefix_ids, suffixes


def _grant(guard, role, phase=None, output=None):
    """Independent backend check of MAIN manager's per-child capability journal."""
    guard.check()
    need(role in ROLES, 'Wrong frozen role')
    for name in ('INSTALL_COMPLETE.json', 'CODEC_COMPLETE.json'):
        path = ROOT / name
        need(path.resolve() == path and not path.is_symlink() and path.stat().st_mode & 0o222 == 0,
             'Actual immutable install/codec receipt required')
        receipt = read(path)
        need(receipt['complete'] is True, 'Install/codec incomplete')
        if name == 'INSTALL_COMPLETE.json':
            same(receipt['config'], pin(ROOT / 'config.json'), 'Install config differs')
            same(receipt['source_binding'], guard.cfg['source_binding'], 'Install source differs')
        else:
            need(receipt['model_forward_calls'] == 0 and receipt['cases'] == 375 and
                 receipt['known_fields'] == 615 and receipt['candidates'] == 5190 and
                 receipt['unknown_fields_retained_no_forward'] == 90 and receipt['all_ids_exact'] is True,
                 'Actual full input-only codec required')
            same(receipt['frozen_candidates'], guard.frozen_pin, 'Codec candidates differ')
            same(receipt['source_files'], guard.cfg['source_files'], 'Codec source differs')
            verify(receipt['input_pins'])
    path = Path(os.environ.get('JEV_FINAL_GRANT', ''))
    need(path.is_absolute() and path.parent == ROOT / 'grants' and path.resolve() == path and
         not path.is_symlink() and path.name.endswith('-GRANT.json') and
         path.stat().st_mode & 0o222 == 0, 'Exact own per-worker grant required')
    grant = read(path)
    need(grant['version'] == 'clean-final-phase-grant-v1' and grant['root'] == str(ROOT) and
         grant['phase'] in ('confirm', 'public-worker') and grant['role'] == role and
         (phase is None or grant['phase'] == phase), 'Grant phase/role differs')
    parent = identity(os.getppid())
    need(parent is not None and grant['owner'] == parent, 'Exact live manager parent required')
    same(grant['config'], pin(ROOT / 'config.json'), 'Worker grant config differs')
    same(grant['source_files'], guard.cfg['source_files'], 'Worker grant source differs')
    owner, launch = read(ROOT / 'OWNER.json'), read(ROOT / 'LAUNCH.json')
    need(owner['owner'] == launch['owner'] == parent, 'Worker not owned by current final manager')
    same(owner['config'], grant['config'], 'Manager config differs')
    same(launch['config'], grant['config'], 'Launch config differs')
    rank = grant['rank']
    if grant['phase'] == 'confirm':
        need(rank is None, 'Confirm grant does not have a public rank')
        expected = ROOT / 'confirm' / role
        gpu = ROLES.index(role)
    else:
        need(type(rank) is int and 0 <= rank < 4, 'Public rank0..3 required')
        expected = ROOT / 'public' / role / 'workers' / str(rank)
        gpu = rank
        analysis = read(ROOT / 'CONFIRM_ANALYSIS.json')
        ready = read(ROOT / 'PUBLIC_READY.json')
        need(analysis['complete'] is True and analysis['public_list_changed'] is False and
             ready['complete'] is True and ready['executed_requests'] == 140620 and
             ready['scored_requests'] == 140178, 'Actual confirm-analysis/full Suite gates required')
    need(grant['output'] == str(expected) and (output is None or Path(output) == expected) and
         expected.resolve() == expected and not list(expected.glob('BLOCKED*')),
         'Only own exact fixed output is authorized')
    need(grant['gpu'] == gpu and grant['device_uuid'] == guard.cfg['predecessors'][str(gpu)]['uuid'] and
         os.environ.get('CUDA_VISIBLE_DEVICES') == grant['device_uuid'], 'Worker GPU UUID grant differs')
    return grant, _control_entry(path)


def _module(name, source, pins):
    already = sys.modules.get(name)
    if already is not None:
        need(Path(already.__file__).resolve() == source / (name + '.py'),
             'Mixed producer source module rejected: ' + name)
    module = importlib.import_module(name)
    need(Path(module.__file__).resolve() == source / (name + '.py') and pin(module.__file__) in pins,
         'Unbound own-role producer import: ' + name)
    return module


def configure_role(guard, role):
    global _CONFIGURED
    need(_CONFIGURED is None and 'torch' not in sys.modules,
         'One role per process; configure original flags before first Torch/SDK.run')
    need(Path(__file__).resolve() == ROOT / 'source/backend.py', 'Only deployed final backend may configure')
    grant, grant_entry = _grant(guard, role)
    record = guard.producer(role)
    source = record['root'] / 'source'
    need(source.resolve() == source and source.is_dir(), 'Own producer source root redirected')
    pins = record['cfg']['trainer_files']
    # Never delete, replace or monkeypatch an already imported other-role module.
    for name in {Path(item['path']).stem for item in pins} | {'state_boundaries'}:
        if name in sys.modules:
            need(Path(sys.modules[name].__file__).resolve() == source / (name + '.py'),
                 'Process already contains another producer: ' + name)
    sys.path.insert(0, str(source))
    bindings = _module('bindings', source, pins)
    ctx = bindings.Context(record['config_path'], record['arm'])
    same(ctx.snapshot, record['scope'], 'Actual own producer Context is not the frozen scope')
    same(ctx.cfg, record['cfg'], 'Actual producer config differs')
    need(ctx.arm == record['scope']['arm'] and ctx.arm['mode'] == 'causal' and ctx.arm['layout'] == 'packed',
         'Producer arm may not be substituted')
    bindings.offline_environment()
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ctx.cfg['precision']['cublas_workspace_config'] and
         os.environ.get('FLASH_ATTENTION_DETERMINISTIC') == '1' and
         all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')),
         'Launcher must configure CUBLAS/offline/determinism before process start')
    modules = {name: _module(name, source, pins) for name in ('fresh_init', 'case_runtime', 'model_runtime')}
    # Their top levels are stdlib. The first Torch import is inside this exact helper.
    need('torch' not in sys.modules, 'Producer helper imported Torch before original configuration')
    _renderer(guard)  # Bind the shared CPU helper during startup, not on the first public request.
    environment = modules['model_runtime'].configure_cuda(ctx)
    controls = ['INSTALL_COMPLETE.json', 'CODEC_COMPLETE.json', 'OWNER.json', 'LAUNCH.json']
    if grant['phase'] == 'public-worker':
        controls += ['CONFIRM_ANALYSIS.json', 'PUBLIC_READY.json']
    control_entries = [_control_entry(ROOT / name) for name in controls]
    need(sum(item['stat']['bytes'] for item in control_entries) + grant_entry['stat']['bytes'] <=
         _MAX_CURRENT_CONTROLS, 'Aggregate current worker controls must remain compact')
    context_value = _context_contract(ctx)
    need(len(context_value.encode('utf-8')) <= _MAX_CONTEXT_CONTRACT, 'Execution Context contract is not compact')
    _CONFIGURED = {'pid': os.getpid(), 'role': role, 'ctx': ctx, 'record': record, 'modules': modules,
                   'environment': environment, 'grant': grant, 'grant_entry': grant_entry,
                   'control_entries': control_entries, 'context_value': context_value,
                   'config_pin': pin(ROOT / 'config.json'), 'config_value': canonical(guard.cfg),
                   'cfg_object': ctx.cfg, 'snapshot_object': ctx.snapshot,
                   'arm_object': ctx.arm, 'source': source}
    _effective(guard, role)
    return environment


def _effective(guard, role):
    configured = _CONFIGURED
    need(configured is not None and configured['pid'] == os.getpid() and configured['role'] == role,
         'Matching preconfigured own-role process required; Engine cannot configure after SDK Torch import')
    guard.check()
    verify([configured['config_pin']])
    need(canonical(guard.cfg) == configured['config_value'], 'Preconfigured final config changed')
    exact(configured['grant_entry'])
    for item in configured['control_entries']:
        exact(item)
    ctx = configured['ctx']
    need(ctx.arm is configured['arm_object'] and ctx.cfg is configured['cfg_object'] and
         ctx.snapshot is configured['snapshot_object'] and ctx.snapshot['config'] is ctx.cfg and
         ctx.snapshot['arm'] is ctx.arm and _context_contract(ctx) == configured['context_value'],
         'Original producer execution contract/object identity mutated')
    need(identity(os.getppid()) == configured['grant']['owner'], 'Exact manager parent no longer live')
    for name, module in configured['modules'].items():
        need(Path(module.__file__).resolve() == configured['source'] / (name + '.py'), 'Producer module source changed')
    import torch
    precision = ctx.cfg['precision']
    need(torch.get_float32_matmul_precision() == precision['float32_matmul_precision'] and
         torch.backends.cuda.matmul.allow_tf32 == precision['matmul_allow_tf32'] and
         torch.backends.cudnn.allow_tf32 == precision['cudnn_allow_tf32'] and
         torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == precision['allow_bf16_reduced_precision_reduction'] and
         torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled() and
         torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark and
         os.environ.get('CUBLAS_WORKSPACE_CONFIG') == precision['cublas_workspace_config'] and
         os.environ.get('CUDA_VISIBLE_DEVICES') == configured['grant']['device_uuid'] and
         torch.cuda.device_count() == 1, 'Effective frozen CUDA/precision flags changed')
    return configured


class TerminalModel:
    def __init__(self, guard, role):
        global _MODEL_CONSTRUCTED
        need(not _MODEL_CONSTRUCTED, 'One trained model construction per process; no reload/retry')
        configured = _effective(guard, role)
        _MODEL_CONSTRUCTED = True  # A failed loader is not automatically retried.
        self.guard, self.role = guard, role
        self.record = configured['record']
        self.environment = configured['environment']
        self.ctx = configured['ctx']
        self.case_runtime = configured['modules']['case_runtime']
        self._fresh = configured['modules']['fresh_init']
        self._closed = False
        self._score_calls = self._request_calls = self._pairs = self._real_tokens = 0
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(guard.cfg['stock_directory'],
                                                      local_files_only=True, trust_remote_code=False)
        need(self.tokenizer.eos_token_id == 151645 and self.tokenizer.pad_token_id == 151643,
             'Stock tokenizer IDs differ')
        checkpoint = self.record['checkpoint']
        # This independent read-only CPU byte oracle is never assigned into model.
        saved = load_file(str(checkpoint / 'adapter_model.safetensors'), device='cpu')
        need(len(saved) == 393 and all(v.dtype == torch.float32 for v in saved.values()), '393 trained FP32 oracle required')
        self._oracle = MappingProxyType({k: (tuple(v.shape), str(v.dtype),
            v.contiguous().view(torch.uint8).numpy().tobytes()) for k, v in saved.items()})
        del saved
        self.model = self._fresh.build_stock(42)  # CPU stock architecture only; zero forwards.
        self.loaded_once = self._fresh.load_saved_adapter(self.model, checkpoint, fresh=False)
        self.audit = self._fresh.audit_model(self.model, require_fresh=False)
        same(self.audit, read(checkpoint / 'CHECKPOINT.json')['audit'], 'Actual trained393 audit differs')
        need(self.audit['active_served_parameters'] == 616237056 and self.audit['unused_registered_parameters'] == 2048 and
             self.audit['saved_key_count'] == self.audit['trainable_tensors'] == 393 and self.audit['lora_pairs'] == 196,
             'Actual served/rank/head architecture differs')
        need(self.model.config.max_position_embeddings == 40960 and self.model.config.use_cache is False,
             'Original40960/no-KV-cache construction required')
        self.runtime = configured['modules']['model_runtime'].FullRow(self.ctx, self.model)
        self.runtime.eval()
        self.before = self._compare()  # GPU move completed; no forward yet.
        self._check()

    def _check(self):
        need(not self._closed, 'No forwards after pure final weight proof')
        _effective(self.guard, self.role)
        need(not self.model.training and not self.runtime.core.training and
             all(not module.training for module in self.model.modules()) and
             all(p.grad is None for p in self.model.parameters()), 'Read-only eval/no-grad model required')

    def _compare(self):
        import torch
        current = self._fresh.adapter_state(self.model)
        need(set(current) == set(self._oracle), 'Saved393 key set mutated')
        for key, value in current.items():
            shape, dtype, data = self._oracle[key]
            need(tuple(value.shape) == shape and str(value.dtype) == dtype and
                 value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() == data,
                 'Trained tensor byte mismatch: ' + key)
        parameters = list(self.model.parameters())
        need(sum(p.requires_grad for p in parameters) == 393 and all(p.grad is None for p in parameters),
             '393 requires_grad or all-grad-None invariant changed')
        return {'keys_compared': 393, 'shape_dtype_bytes_equal': True, 'all_grad_None': True,
                'active_requires_grad_true': 393, 'method': 'pure CPU dtype-preserving bytes; no reload/digest'}

    def score(self, pairs, phase):
        self._check()
        need(phase in ('dev_confirm', 'public') and
             phase == ('dev_confirm' if _CONFIGURED['grant']['phase'] == 'confirm' else 'public'),
             'Only current authorized real confirm/public forwards are allowed')
        need(type(pairs) is list and all(isinstance(p, self.case_runtime.Pair) for p in pairs), 'Own producer Pair list required')
        need([p.slot for p in pairs] == list(range(len(pairs))) and
             all(p.prefix and p.suffix and p.suffix[-1] == 151645 and 0 < p.length <= 40960 and
                 all(type(t) is int and 0 <= t < len(self.tokenizer) for t in p.row()) for p in pairs),
             'Entire original-slot full request must validate before its first forward')
        micros = self.case_runtime.micro_plan(pairs, dev=True)
        need(sorted(p.slot for micro in micros for p in micro) == list(range(len(pairs))), 'Micro plan lost/duplicated slots')
        import torch
        logits = torch.empty((len(pairs), 2), dtype=torch.float32, device='cpu')
        margins = torch.empty((len(pairs),), dtype=torch.float32, device='cpu')
        self.runtime.phase = phase
        with torch.no_grad():
            for micro in micros:
                # Request-level source/effective checks are outside this inner
                # wave. The manager independently performs its15s resource guard.
                need(identity(os.getppid()) == _CONFIGURED['grant']['owner'], 'Exact manager parent exited')
                need(not self.model.training and not self.runtime.core.training, 'Eval flags changed before actual wave')
                actual_logits, actual_margins = self.runtime.forward(micro)
                need(not actual_logits.requires_grad and not actual_margins.requires_grad,
                     'Unexpected autograd allocation in read-only evaluation')
                slots = torch.tensor([p.slot for p in micro], dtype=torch.long, device='cpu')
                logits.index_copy_(0, slots, actual_logits.detach().cpu())
                margins.index_copy_(0, slots, actual_margins.detach().cpu())
        need(torch.isfinite(logits).all().item() and torch.isfinite(margins).all().item() and
             all(p.grad is None for p in self.model.parameters()), 'Invalid output or allocated gradient')
        self._score_calls += 1
        self._pairs += len(pairs)
        self._real_tokens += sum(p.length for p in pairs)
        self._check()
        return logits, margins

    def predict(self, state, questions):
        self._check()
        need(_CONFIGURED['grant']['phase'] == 'public-worker', 'Request API is public-worker only')
        _json_input(state)
        need(type(questions) is dict and questions and all(type(k) is str for k in questions), 'Named complete question map required')
        helper = _renderer()
        fields, pairs = [], []
        # Validate every field, all noul keys, all actual IDs and all lengths
        # before score() can perform any wave. No failure after partial request.
        for name, question in questions.items():
            need(type(question) is dict, 'Question must be an object')
            _json_input(question)
            kind = question.get('type')
            need(kind in ('choice', 'noul'), 'Only choice/noul question types are supported')
            options = question.get('criteria')
            if kind == 'noul' and not options:
                options = {'true': 'Yes.', 'false': 'No.'}
            need(type(options) is dict and options and all(type(k) is str for k in options), 'Complete named criteria required')
            keys = list(options)
            positive = [key for key in keys if str(key).lower() in ('true', 'yes', '1')]
            if kind == 'noul':
                need(len(keys) == 2 and len(positive) == 1, 'noul needs exactly one true/yes/1 key among two criteria')
            values = [helper.text(value) for value in options.values()]
            instruction = helper.text(question.get('instructions', ''))
            prefix, suffixes = field_tokens(self.tokenizer, state, instruction, values)
            need(all(len(prefix) + len(s) <= 40960 for s in suffixes),
                 'Whole candidate exceeds40960; no truncation, candidate removal or retry')
            start = len(pairs)
            for candidate, suffix in enumerate(suffixes):
                pairs.append(self.case_runtime.Pair(len(pairs), 0, len(fields), candidate, prefix, suffix))
            fields.append({'name': name, 'kind': kind, 'keys': keys, 'positive': positive,
                           'start': start, 'end': len(pairs)})
        _, margins = self.score(pairs, phase='public')
        scores = margins.tolist()  # Original FP32 margins; math operations below are binary64.
        answers = {}
        for field in fields:
            values = scores[field['start']:field['end']]
            maximum = max(values)
            exp = [math.exp(value - maximum) for value in values]
            total = math.fsum(exp)
            probabilities = [value / total for value in exp]
            distribution = dict(zip(field['keys'], probabilities))
            if field['kind'] == 'choice':
                best = max(range(len(values)), key=lambda i: values[i])  # First native candidate on a tie.
                answers[field['name']] = {'type': 'choice', 'choice': field['keys'][best], 'probabilities': distribution}
            else:
                answers[field['name']] = {'type': 'noul', 'noul': distribution[field['positive'][0]],
                                          'probabilities': distribution}
        self._request_calls += 1
        return {'model': self.record['response_name'], 'answers': answers,
                'usage': {'input_tokens': sum(p.length for p in pairs)}}, None

    def finish(self):
        need(not self._closed, 'finish must be called exactly once')
        self._check()
        import torch
        torch.cuda.synchronize(0)
        after = self._compare()  # PURE comparison: never load or assign trained tensors here.
        receipt = self.runtime.receipt()
        need(receipt['mode'] == 'causal' and receipt['layout'] == 'packed' and receipt['prefix_cache'] is False,
             'Actual final runtime differs')
        phase = 'dev_confirm' if _CONFIGURED['grant']['phase'] == 'confirm' else 'public'
        forward, native = receipt['forward'], receipt['native_forward_calls']
        waves = forward.get(phase + '.completed_waves', 0)
        need(waves > 0 and forward[phase + '.attempted_waves'] == waves and
             all(forward[phase + '.' + key] == self._real_tokens for key in
                 ('attempted_real_tokens', 'completed_real_tokens', 'attempted_tokenwise_rows_per_operator',
                  'completed_tokenwise_rows_per_operator')) and
             forward[phase + '.completed_native_FA3_calls'] == 28 * waves and
             native == {'attempted_calls': 28 * waves, 'completed_calls': 28 * waves,
                        'attempted_query_tokens': 28 * self._real_tokens, 'completed_query_tokens': 28 * self._real_tokens,
                        'completed_key_tokens': 28 * self._real_tokens, 'completed_causal_calls': 28 * waves},
             'Actual complete realT/native28/no-retry counters differ')
        self._closed = True  # No later score/predict/synchronize via this model.
        proof = {'complete': True, 'weights_unchanged': True, 'keys_compared': 393,
                 'served_parameters': 616237056, 'post_eval_reload': False, 'loaded_once': self.loaded_once,
                 'audit': self.audit, 'before': self.before, 'after': after, 'runtime': receipt,
                 'environment': self.environment, 'terminal': self.record['terminal'],
                 'model': self.record['response_name'], 'role': self.role, 'optimizer_created': False,
                 'optimizer_steps': 0, 'backward_calls': 0, 'all_grad_None': True,
                 'score_calls': self._score_calls, 'requests': self._request_calls,
                 'candidate_pairs': self._pairs, 'real_tokens': self._real_tokens,
                 'CPU_token_cache_maxsize': 0, 'head_all_tokens_before_eos': True,
                 'EOS': 151645, 'no_subsequent_forward': True}
        # Drop model references only AFTER the pure comparison and proof exist.
        # No CUDA allocator reset, GC probe, reload, or further forward follows.
        self.model = self.runtime = self._oracle = None
        return proof


class CleanEngine:
    name = 'clean-final-frozen393-native-fa3-fullrow'
    latency = ('Full real in-process request wall time including text/tokenization/all candidates/native forward; '
               'model loading excluded, first-real-request first-use cost included; no warmup or single-card admission claim.')

    def __init__(self, config: str, role: str, output: str, **options):
        need(not options, 'Unknown engine options rejected; frozen runtime cannot be tuned')
        need(Path(__file__).resolve() == ROOT / 'source/backend.py', 'Only installed final SDK engine may execute')
        self.guard = Guard(config)
        grant, self._grant_entry = _grant(self.guard, role, phase='public-worker', output=output)
        _effective(self.guard, role)
        self.output = Path(output)
        need(self.output.is_dir() and self.output.resolve() == self.output and
             not any((self.output / n).exists() or (self.output / n).is_symlink()
                     for n in ('results.jsonl', 'WEIGHTS.json', 'COMPLETE.json')),
             'Fresh SDK worker output required; append mode never adopts old results')
        self.model = TerminalModel(self.guard, role)
        self._closed = False
        self.provenance = {'kind': 'fixed-clean-terminal393-consumer-v1', 'role': role,
            'model': self.model.record['response_name'], 'checkpoint': str(self.model.record['checkpoint']),
            'terminal': self.model.record['terminal'], 'stock_directory': self.guard.cfg['stock_directory'],
            'frozen_candidates': self.guard.frozen_pin, 'source_binding': self.guard.cfg['source_binding'],
            'source_files': self.guard.cfg['source_files'], 'renderer_source': self.guard.cfg['renderer_source'],
            'sdk_source_snapshot': self.guard.snapshot['sdk'], 'served_parameters': 616237056,
            'runtime': 'causal-packed-nativeFA3-fullrow', 'readout': self.guard.frozen['runtime_contract']['readout'],
            'prefix_cache': False, 'extra_fusions': False, 'Graph': False, 'prediction_cache': False,
            'option_filtering': False, 'truncation': False, 'correctness_retry': False,
            'teacher_used_for_inference': False, 'lora_merged': False,
            'scientific_source_admission_is_not_production_permission': True,
            'worker_grant': pin(os.environ['JEV_FINAL_GRANT']), 'rank': grant['rank']}

    def __call__(self, state, questions):
        need(not self._closed, 'Engine already closed')
        exact(self._grant_entry)
        return self.model.predict(state, questions)

    def synchronize(self):
        need(not self._closed, 'Engine already closed')
        import torch
        torch.cuda.synchronize(0)

    def runtime(self):
        need(not self._closed, 'Engine already closed')
        actual = self.model.runtime.receipt()
        attempted = sum(v for k, v in actual['forward'].items() if k.endswith('.attempted_waves'))
        return {'precision': self.model.ctx.cfg['precision'], 'environment': self.model.environment,
                'actual_runtime': actual, 'model_forward_calls_before_data': attempted,
                'CPU_token_cache_maxsize': 0, 'context': 40960, 'micro_plan': self.model.ctx.cfg['micro_plan'],
                'saved393_loaded_once': self.model.loaded_once, 'before_weights': self.model.before,
                'served_parameters': 616237056, 'synthetic_warmup_calls': 0, 'prefix_cache': False}

    def warmup(self):
        raise RuntimeError('Synthetic warmup is forbidden; SDK.run must use warm=False')

    def close(self):
        need(not self._closed, 'Engine close/proof must happen exactly once')
        exact(self._grant_entry)
        proof = self.model.finish()
        commit(self.output / 'WEIGHTS.json', proof)
        self._closed = True
