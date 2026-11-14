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

    def flash_attn_varlen_func(self, *args, **kwargs):
        require(kwargs.get('deterministic') is True, 'Strict deterministic native FA3 required')
        q, k, v = args[:3]
        require(q.shape[-2:] == (16, 128) and k.shape[-2:] == v.shape[-2:] == (8, 128),
                'Native unexpanded GQA Q16/KV8/D128 required')
        self.counts['attempted_calls'] += 1
        self.counts['attempted_query_tokens'] += q.shape[0]
        result = self.module.flash_attn_varlen_func(*args, **kwargs)
        self.counts['completed_calls'] += 1
        self.counts['completed_query_tokens'] += q.shape[0]
        self.counts['completed_key_tokens'] += k.shape[0]
        self.counts['completed_causal_calls' if kwargs['causal'] else 'completed_noncausal_calls'] += 1
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
        # Fresh context per single forward; no outer no-grad autocast weight-cache sharing.
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            cache_enabled=self.ctx.cfg['precision']['autocast_cache_enabled']):
            logits, prefix = self.core(batch, prefixes)
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
