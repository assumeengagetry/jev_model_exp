"""Student-only, explicit Hopper FA3 integration. No SDPA fallback exists here."""
import functools
import importlib.util
import json
from pathlib import Path
import sys

import torch

BACKEND = 'flash_attn3_hopper'
PROVIDER = 'kernels-community/flash-attn3'
BUILD = 'build/torch29-cxx11-cu128-x86_64-linux'
_MODULE = None
_DIRECTORY = None
_COUNTS = {'forward': 0, 'backward': 0}


def _event(event, **values):
    print(json.dumps({'event': event, 'attention': 'flash_attention_3',
                      'attention_implementation': BACKEND, 'provider': PROVIDER,
                      'sdpa_fallback': False, **values}), flush=True)


def load_student_fa3(directory):
    global _MODULE, _DIRECTORY
    directory = Path(directory).resolve()
    if _MODULE is not None:
        if directory != _DIRECTORY: raise RuntimeError('A different FA3 build is already loaded; no backend substitution')
        return _MODULE
    if torch.__version__.split('+')[0] != '2.9.1' or torch.version.cuda != '12.8' or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError('Student FA3 requires unchanged Torch2.9.1/CUDA12.8/CXX11 ABI; no SDPA fallback')
    assets = json.loads((directory / 'ASSETS.json').read_text())
    ready = json.loads((directory / 'READY.json').read_text())
    if assets.get('complete') is not True or ready.get('complete') is not True or assets.get('repo') != PROVIDER or assets.get('build') != BUILD:
        raise RuntimeError('Complete matching FA3 installation is required; no SDPA fallback')
    name = '_jev_student_hopper_fa3'
    spec = importlib.util.spec_from_file_location(name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    interface = sys.modules[name + '.flash_attn_interface']
    ops = sys.modules[name + '._ops'].ops
    if not hasattr(ops, 'bwd') or not hasattr(interface, 'FlashAttnVarlenFunc'):
        raise RuntimeError('FA3 native training/backward is unavailable; no SDPA fallback')

    def wrap_forward(function, kind):
        @functools.wraps(function)
        def forward(*args, **kwargs):
            result = function(*args, **kwargs)
            _COUNTS['forward'] += 1
            if _COUNTS['forward'] == 1:
                q = args[0]
                _event('student_fa3_actual_forward', kind=kind, dtype=str(q.dtype),
                       device=str(q.device), head_dim=q.shape[-1])
            return result
        return forward
    module.flash_attn_func = wrap_forward(module.flash_attn_func, 'dense')
    module.flash_attn_varlen_func = wrap_forward(module.flash_attn_varlen_func, 'varlen')
    native_backward = interface._flash_attn_backward
    @functools.wraps(native_backward)
    def backward(*args, **kwargs):
        result = native_backward(*args, **kwargs)
        _COUNTS['backward'] += 1
        if _COUNTS['backward'] == 1:
            _event('student_fa3_actual_backward')
        return result
    interface._flash_attn_backward = backward
    _MODULE, _DIRECTORY = module, directory
    return module


def register_student_fa3(directory, require_device=True):
    import transformers
    if transformers.__version__ != '4.57.6':
        raise RuntimeError('This Student FA3 adapter targets unchanged Transformers4.57.6')
    if require_device:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            raise RuntimeError('Student FA3 requires its assigned Hopper GPU; no SDPA fallback')
    kernel = load_student_fa3(directory)
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
    from transformers.integrations.flash_attention import flash_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    # This is the same Flash wrapper and 2D/unpadding-mask registration used by
    # Transformers' public kernel integration. The actual functions are FA3.
    lazy_import_flash_attention(kernel, force_import=True)
    ALL_ATTENTION_FUNCTIONS.register(BACKEND, flash_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(BACKEND, ALL_MASK_ATTENTION_FUNCTIONS['flash_attention_2'])
    _event('student_fa3_registered', build=BUILD, kernel_directory=str(_DIRECTORY))
    return attention_record()


def assert_student_uses_fa3(model):
    if model.config._attn_implementation != BACKEND:
        raise RuntimeError('Student did not select the required FA3 implementation; refuse to train')
    if getattr(model.config, 'attention_dropout', 0) != 0 or model.config.head_dim != 128:
        raise RuntimeError('Unexpected Qwen attention parameters for this FA3 training recipe')
    layers = [module for module in model.modules() if module.__class__.__name__ == 'Qwen3Attention']
    if len(layers) != model.config.num_hidden_layers or any(module.config._attn_implementation != BACKEND for module in layers):
        raise RuntimeError('Every Qwen Student attention layer must use explicit FA3')


def attention_record():
    return {'attention': 'flash_attention_3', 'attention_implementation': BACKEND,
            'attention_provider': PROVIDER, 'attention_build': BUILD,
            'sdpa_fallback': False, 'fa3_forward_calls': _COUNTS['forward'],
            'fa3_backward_calls': _COUNTS['backward'],
            'actual_training_kernel_executed': bool(_COUNTS['forward'] and _COUNTS['backward'])}
