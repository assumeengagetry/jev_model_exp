"""Native full-row execution only; no fork/cache, SDPA, compile or fallback."""
from __future__ import annotations

from collections import Counter
import importlib
import os
from pathlib import Path
import sys

from bindings import require, stat, verify
from fresh_init import components, versions


def configure_cuda(ctx):
    # This function must be called before the first torch import in the process.
    require('torch' not in sys.modules, 'Precision/CUBLAS setup must precede torch import')
    precision = ctx.cfg['precision']
    existing = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    require(existing is None or existing == precision['cublas_workspace_config'], 'Conflicting CUBLAS workspace')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = precision['cublas_workspace_config']
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(visible and len(visible.split(',')) == 1 and visible.strip() != '-1', 'Exactly one externally assigned GPU required')
    require(os.environ.get('WORLD_SIZE', '1') == '1' and os.environ.get('RANK', '0') == '0' and
            os.environ.get('LOCAL_RANK', '0') == '0', 'No DDP/multi-rank execution')
    import torch
    versions(cuda=True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_capability(0) == (9, 0) and 'H800' in torch.cuda.get_device_name(0),
            'One externally allocated actual H800 required')
    torch.set_float32_matmul_precision(precision['float32_matmul_precision'])
    # In 2.9 high/medium implies allow_tf32; contradictory requests fail, never quietly overwrite.
    require(torch.backends.cuda.matmul.allow_tf32 == precision['matmul_allow_tf32'],
            'float32_matmul_precision and matmul_allow_tf32 conflict')
    torch.backends.cuda.matmul.allow_tf32 = precision['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = precision['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = precision['allow_bf16_reduced_precision_reduction']
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(ctx.cfg['seed'])
    torch.cuda.manual_seed_all(ctx.cfg['seed'])
    return {'versions': versions(cuda=True), 'precision': precision,
            'gpu_name': torch.cuda.get_device_name(0), 'visible_device': visible,
            'deterministic_algorithms': True, 'cudnn_benchmark': False,
            'cudnn_deterministic': True, 'world_size': 1, 'prefix_cache': False}


class CountedNative:
    def __init__(self, module):
        self.module = module
        self.counts = Counter()
        self.records = None
        self.wave = None

    def set_recording(self, enabled):
        require(type(enabled) is bool, 'Explicit native recording selector required')
        if enabled:
            require(self.records is None and self.wave is None, 'Native recording already active')
            self.records = []
        else:
            # Drop references, not list contents: a returned JSON document may own them.
            self.records, self.wave = None, None

    def begin_wave(self, phase, wave_index, mode):
        require(self.records is not None and self.wave is None and
                phase in ('train_first', 'train_replay') and mode in ('causal', 'prefix_block'),
                'Formal macro0 native wave scope required')
        self.wave = {'phase': phase, 'wave_index': wave_index, 'call_start': len(self.records),
                     'calls_per_layer': 2 if mode == 'prefix_block' else 1}

    def end_wave(self):
        self.wave = None

    def flash_attn_varlen_func(self, *args, **kwargs):
        require(kwargs.get('deterministic') is True, 'Strict deterministic native FA3 required')
        q, k, v = args[:3]
        require(q.shape[-2:] == (16, 128) and k.shape[-2:] == v.shape[-2:] == (8, 128),
                'Native unexpanded GQA Q16/KV8/D128 required')
        self.counts['attempted_calls'] += 1
        self.counts['attempted_query_tokens'] += q.shape[0]
        record = None
        if self.records is not None:
            require(self.wave is not None and len(args) == 7 and
                    set(kwargs) == {'softmax_scale', 'causal', 'deterministic'} and
                    type(kwargs['causal']) is bool and type(kwargs['softmax_scale']) in (int, float) and
                    type(args[5]) is int and type(args[6]) is int,
                    'Unexpected actual native FA3 recording ABI')
            order = len(self.records) - self.wave['call_start']
            per_layer = self.wave['calls_per_layer']
            record = {
                'call_index': len(self.records), 'phase': self.wave['phase'],
                'wave_index': self.wave['wave_index'], 'call_index_in_wave': order,
                # The pinned core loops over 28 layers, with one or two calls per layer.
                'layer_index_from_call_order': order // per_layer,
                'call_index_in_layer': order % per_layer,
                'q_shape': list(q.shape), 'k_shape': list(k.shape), 'v_shape': list(v.shape),
                'cu_q': args[3].detach().cpu().tolist(),
                'cu_k': args[4].detach().cpu().tolist(),
                'max_q': args[5], 'max_k': args[6], 'kwargs': dict(kwargs), 'completed': False,
            }
            self.records.append(record)
        result = self.module.flash_attn_varlen_func(*args, **kwargs)
        self.counts['completed_calls'] += 1
        self.counts['completed_query_tokens'] += q.shape[0]
        self.counts['completed_key_tokens'] += k.shape[0]
        self.counts['completed_causal_calls' if kwargs['causal'] else 'completed_noncausal_calls'] += 1
        if record is not None:
            record['completed'] = True
        return result


class FullRow:
    def __init__(self, ctx, model):
        import torch
        core_dir = str(Path(ctx.cfg['core_directory']).resolve())
        native_dir = str(Path(ctx.cfg['native_directory']).resolve())
        sys.path.insert(0, native_dir)
        sys.path.insert(0, core_dir)
        packed = importlib.import_module('packed_mask')
        native = importlib.import_module(ctx.cfg['native_module'])
        require(stat(native.__file__) in ctx.cfg['native_files'], 'Imported native FA3 interface is unbound')
        verify(ctx.cfg['native_files'])
        native_pins = {r['resolved'] for r in ctx.cfg['native_files']}
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, '__file__', None)
            if filename and (name.startswith('flash_attn_3') or
                             Path(filename).resolve().is_relative_to(Path(native_dir))):
                require(str(Path(filename).resolve()) in native_pins,
                        'Native FA3 imported unbound code/binary: ' + name)
        expected_core = {r['resolved'] for r in ctx.cfg['core_files']}
        for name, module in tuple(sys.modules.items()):
            if name == 'packed_mask' or name.startswith('packed_mask.'):
                require(str(Path(module.__file__).resolve()) in expected_core, 'Imported wrong/unbound packed-mask module')
        self.native = CountedNative(native)
        backbone, head = components(model)
        model.to(torch.device('cuda:0'))  # Preserves stock BF16 and trainable FP32 separately.
        self.core = packed.MaskedBackbone(backbone, head, self.native, ctx.arm['mode'], strict_deterministic=True)
        self.make_batch = packed.make_batch
        self.ctx, self.model, self.counts = ctx, model, Counter()
        self.phase = 'unassigned'
        self.boundary_store = None
        self.dispatch_waves = None

    def begin_macro0_dispatch(self):
        require(self.dispatch_waves is None and self.ctx.arm['layout'] == 'packed',
                'Formal macro0 dispatch recording must start once, on packed input')
        self.native.set_recording(True)
        self.dispatch_waves = []

    def stop_macro0_dispatch(self):
        self.native.set_recording(False)
        self.dispatch_waves = None

    def finish_macro0_dispatch(self, *, complete=True):
        require(type(complete) is bool and self.dispatch_waves is not None and self.native.records is not None,
                'No formal macro0 dispatch recording')
        waves, calls = self.dispatch_waves, self.native.records
        per_layer = 2 if self.ctx.arm['mode'] == 'prefix_block' else 1
        if complete:
            first = [w for w in waves if w['phase'] == 'train_first']
            replay = [w for w in waves if w['phase'] == 'train_replay']
            require(first and len(first) == len(replay) and waves == first + replay and
                    [w['slots'] for w in first] == [w['slots'] for w in replay] and
                    all(w['completed'] and w['call_end'] - w['call_start'] == 28 * per_layer for w in waves) and
                    all(call['completed'] for call in calls) and
                    len(calls) == len(waves) * 28 * per_layer,
                    'Incomplete actual formal macro0 dispatch coverage')
        document = {
            'version': 'actual-native-dispatch-v1', 'macro_index': 0,
            'is_formal_exposure_macro0': True, 'complete': complete,
            'arm_name': self.ctx.arm_name, 'mode': self.ctx.arm['mode'], 'layout': self.ctx.arm['layout'],
            'layers': 28, 'calls_per_layer': per_layer,
            'call_range_convention': 'zero-based call_start inclusive; call_end exclusive',
            'layer_index_convention': 'zero-based, derived from actual call order in the pinned core loop',
            'qkv_tensor_values_saved': False, 'full_model_input_ids_saved': False,
            'eos_token_ids_only': True, 'waves': waves, 'calls': calls,
            'independent_partition_primary_required': True,
            'new_prefix_cross_backend_or_neighbor_isolation_claimed': False,
        }
        self.stop_macro0_dispatch()
        return document

    def _record_batch(self, batch, micro, rows, prefixes):
        """Read actual packed metadata once per recorded wave, never once per layer."""
        require(self.ctx.arm['layout'] == batch.layout == 'packed' and
                self.phase in ('train_first', 'train_replay'), 'Only formal train macro0 is recorded')
        lengths = [len(row) for row in rows]
        require(list(batch.lengths) == lengths and batch.ids.ndim == batch.positions.ndim == 1 and
                batch.ids.numel() == batch.positions.numel() == batch.total == sum(lengths),
                'Actual packed batch geometry differs from complete original rows')
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        actual_cu = batch.cu.detach().cpu().tolist()
        actual_positions = batch.positions.detach().cpu().reshape(-1).tolist()
        actual_eos = batch.eos_indices.detach().cpu().tolist()
        require(actual_cu == offsets and actual_eos == [end - 1 for end in offsets[1:]] and
                actual_positions == [position for length in lengths for position in range(length)],
                'Actual row-local positions/cu/EOS indices differ; no positional offset or dummy T allowed')
        # Only selected actual EOS values are copied, never full input IDs or Q/K/V values.
        eos_tokens = batch.ids.index_select(0, batch.eos_indices).detach().cpu().tolist()
        eos_positions = [actual_positions[index] for index in actual_eos]
        require(eos_tokens == [151645] * len(rows) and eos_positions == [length - 1 for length in lengths] and
                all(0 < prefix < length for prefix, length in zip(prefixes, lengths)),
                'Actual EOS token/position or prefix boundary differs')
        wave_index = sum(w['phase'] == self.phase for w in self.dispatch_waves)
        return {
            'phase': self.phase, 'wave_index': wave_index, 'slots': [p.slot for p in micro],
            'row_lengths': lengths, 'prefix_lengths': list(prefixes),
            'total_real_tokens': batch.total, 'physical_token_rows': batch.ids.numel(),
            'batch_cu': actual_cu, 'positions': actual_positions,
            'eos_indices': actual_eos, 'eos_positions': eos_positions, 'eos_token_ids': eos_tokens,
            'actual_positions_and_eos_checked': True,
            'call_start': len(self.native.records), 'call_end': len(self.native.records), 'completed': False,
        }

    def forward(self, micro):
        import torch
        rows = [p.row() for p in micro]
        require(rows and all(r[-1] == 151645 for r in rows), 'Native EOS-last full rows required')
        batch = self.make_batch(rows, torch.device('cuda:0'), layout=self.ctx.arm['layout'], pad=151643)
        require(self.ctx.arm['layout'] != 'packed' or batch.ids.numel() == batch.total,
                'Packed must contain realT only, no padded/dummy rows')
        prefixes = tuple(len(p.prefix) for p in micro)
        before = self.native.counts.copy()
        self.counts[self.phase + '.attempted_waves'] += 1
        self.counts[self.phase + '.attempted_real_tokens'] += batch.total
        self.counts[self.phase + '.attempted_tokenwise_rows_per_operator'] += batch.ids.numel()
        wave_record = None
        if self.dispatch_waves is not None:
            wave_record = self._record_batch(batch, micro, rows, prefixes)
            self.dispatch_waves.append(wave_record)
            self.native.begin_wave(self.phase, wave_record['wave_index'], self.ctx.arm['mode'])
        # Fresh context per single forward; no outer no-grad autocast weight-cache sharing.
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                 cache_enabled=self.ctx.cfg['precision']['autocast_cache_enabled']):
                logits, prefix = self.core(batch, prefixes)
        finally:
            if wave_record is not None:
                wave_record['call_end'] = len(self.native.records)
                self.native.end_wave()
        require(prefix is None and logits.shape == (len(micro), 2) and torch.isfinite(logits).all().item(),
                'Invalid native full-row classifier output')
        expected = 28 * (2 if self.ctx.arm['mode'] == 'prefix_block' else 1)
        require(self.native.counts['completed_calls'] - before['completed_calls'] == expected,
                'Not every layer used required native FA3 calls')
        float_logits = logits.float()
        margins = float_logits[:, 1] - float_logits[:, 0]
        require(torch.isfinite(margins).all().item(), 'Nonfinite actual classifier margin')
        self.counts[self.phase + '.completed_waves'] += 1
        self.counts[self.phase + '.completed_real_tokens'] += batch.total
        self.counts[self.phase + '.completed_tokenwise_rows_per_operator'] += batch.ids.numel()
        self.counts[self.phase + '.completed_native_FA3_calls'] += expected
        if wave_record is not None:
            require(wave_record['call_end'] - wave_record['call_start'] == expected,
                    'Actual native dispatch list omitted a layer call')
            wave_record['completed'] = True
        return float_logits, margins

    def train(self):
        self.model.train()
        self.core.train()

    def eval(self):
        self.model.eval()
        self.core.eval()

    def receipt(self):
        return {'execution': 'native-FA3-full-row-no-SDPA-no-fork', 'mode': self.ctx.arm['mode'],
                'layout': self.ctx.arm['layout'], 'prefix_cache': False,
                'forward': dict(self.counts), 'native_forward_calls': dict(self.native.counts),
                'backward_counter_scope': 'autograd native backward invoked by each replay wave; not separately instrumented'}
