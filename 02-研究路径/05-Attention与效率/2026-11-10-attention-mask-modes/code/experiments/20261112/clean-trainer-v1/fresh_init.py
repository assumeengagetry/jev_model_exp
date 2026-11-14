"""CPU-only fresh stock + LoRA/head initialization; this file never forwards."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
from bindings import (Context, STOCK, commit_json, fsync_tree, new_directory, offline_environment,
                      read_json, require, stat, verify, write_json)

TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
CONSTRUCTION = {'stock': STOCK, 'r': 32, 'alpha': 64, 'dropout': 0.0,
                'target_modules': list(TARGETS), 'modules_to_save': ['score'],
                'task_type': 'SEQ_CLS', 'bias': 'none', 'num_labels': 2,
                'id2label': {'0': 'not_match', '1': 'match'}, 'pad': 151643, 'eos': 151645,
                'active_head_shape': [2, 1024], 'trainable_dtype': 'torch.float32',
                'stock_dtype': 'torch.bfloat16', 'loader_attention': 'eager-construction-only'}


def versions(*, cuda=False):
    import torch
    import transformers
    import peft
    require(torch.__version__.split('+')[0] == '2.9.1' and transformers.__version__ == '4.57.6' and
            peft.__version__ == '0.18.1', 'Frozen Torch/Transformers/PEFT versions required')
    if cuda:
        require(torch.version.cuda == '12.8', 'Frozen CUDA12.8 build required')
    return {'torch': str(torch.__version__), 'transformers': str(transformers.__version__),
            'peft': str(peft.__version__), 'cuda_build': torch.version.cuda}


def build_stock(seed):
    """Only authorized stock is passed to any pretrained loader. No forward."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model
    versions()
    require(seed in (42, 43), 'Frozen init seeds only')
    config = AutoConfig.from_pretrained(STOCK, local_files_only=True, trust_remote_code=False)
    require(config.model_type == 'qwen3' and config.num_hidden_layers == 28 and
            config.hidden_size == 1024 and config.num_attention_heads == 16 and
            config.num_key_value_heads == 8 and config.head_dim == 128 and config.attention_dropout == 0,
            'Wrong original stock architecture')
    config.num_labels = 2
    config.id2label = {0: 'not_match', 1: 'match'}
    config.label2id = {'not_match': 0, 'match': 1}
    config.pad_token_id, config.eos_token_id = 151643, 151645
    config.use_cache = False
    model = AutoModelForSequenceClassification.from_pretrained(
        STOCK, config=config, dtype=torch.bfloat16, attn_implementation='eager',
        local_files_only=True, trust_remote_code=False)
    require(next(model.parameters()).device.type == 'cpu', 'Stock construction must be on CPU')
    # Loader-created score is discarded, never adopted as an existing head.
    random.seed(seed)
    torch.manual_seed(seed)
    model.score = torch.nn.Linear(1024, 2, bias=False, dtype=torch.float32, device='cpu')
    torch.nn.init.normal_(model.score.weight, mean=0.0, std=config.initializer_range)
    peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                             target_modules=list(TARGETS), modules_to_save=['score'],
                             task_type='SEQ_CLS', bias='none', init_lora_weights=True)
    model = get_peft_model(model, peft_config)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    audit_model(model, require_fresh=True)
    return model


def components(model):
    base = model.get_base_model()
    require(hasattr(base, 'model') and hasattr(base, 'score'), 'Original sequence-classifier structure changed')
    wrapper = base.score
    require(hasattr(wrapper, 'modules_to_save') and list(wrapper.modules_to_save) == ['default'],
            'One active saved classifier required')
    head = wrapper.modules_to_save['default']
    require(tuple(head.weight.shape) == (2, 1024) and head.bias is None and
            head.weight.dtype.__str__() == 'torch.float32', 'Active FP32 2x1024 head required')
    return base.model, head


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return get_peft_model_state_dict(model, adapter_name='default', save_embedding_layers=False)


def audit_model(model, *, require_fresh=False):
    import torch
    backbone, head = components(model)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(len(trainable) == 393 and all(p.dtype == torch.float32 for p in trainable.values()),
            'All and only393 active saved trainables must be FP32')
    state = adapter_state(model)
    a = {k for k in state if '.lora_A.' in k}
    b = {k for k in state if '.lora_B.' in k}
    score = [k for k in state if k.endswith('score.weight')]
    require(len(state) == 393 and len(a) == len(b) == 196 and len(score) == 1 and
            {k.replace('.lora_A.', '.lora_B.') for k in a} == b and
            tuple(state[score[0]].shape) == (2, 1024), '393 saved keys/196 pairs/one active score contract')
    require(all(state[k].shape[0] == 32 for k in a) and all(state[k].shape[1] == 32 for k in b), 'Wrong LoRA rank')
    require(all(torch.isfinite(p).all().item() for p in state.values()), 'Nonfinite saved parameter')
    if require_fresh:
        require(all(torch.count_nonzero(state[k]).item() == 0 for k in b), 'Fresh LoRA B must be exactly zero')
    require(all(n.endswith('.weight') and ('.lora_A.' in n or '.lora_B.' in n or
                '.score.modules_to_save.default.' in n) for n in trainable), 'Unexpected trainable parameter')
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            require(module.p == 0, 'Zero dropout replay contract violated')
    served = {id(p): p for m in (backbone, head) for p in m.parameters()}
    physical = {id(p): p for p in model.parameters()}
    active_ids = {id(p) for p in trainable.values()}
    require(active_ids <= set(served), 'A trainable is not actually served')
    return {'saved_keys': sorted(state), 'saved_key_count': len(state), 'lora_pairs': len(a),
            'fresh_B_exactly_zero': True if require_fresh else None,
            'trainable_keys': sorted(trainable), 'trainable_tensors': len(trainable),
            'trainable_parameters': sum(p.numel() for p in trainable.values()),
            'active_served_parameters': sum(p.numel() for p in served.values()),
            'physical_registered_parameters': sum(p.numel() for p in physical.values()),
            'unused_registered_parameters': sum(p.numel() for i, p in physical.items() if i not in served),
            'served_scope': 'stock backbone(base+active LoRA)+active score; no original score/unused LM head',
            'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'numel': v.numel()}
                        for k, v in sorted(state.items())}}


def load_saved_adapter(model, directory, *, fresh=False):
    import torch
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    directory = Path(directory)
    saved = load_file(str(directory / 'adapter_model.safetensors'), device='cpu')
    current = adapter_state(model)
    require(set(saved) == set(current) and len(saved) == 393, 'Saved adapter key set differs')
    require(all(saved[k].shape == current[k].shape and saved[k].dtype == current[k].dtype for k in saved),
            'Saved adapter shape/dtype differs')
    set_peft_model_state_dict(model, saved, adapter_name='default')
    actual = adapter_state(model)
    require(all(torch.equal(actual[k].detach().cpu().contiguous().view(torch.uint8),
                            saved[k].contiguous().view(torch.uint8)) for k in saved), 'Per-key bitwise reload failed')
    audit_model(model, require_fresh=fresh)
    return {'exact_keys_compared': len(saved), 'all_tensors_bitwise_equal': True,
            'comparison': 'torch.equal on dtype-preserving byte views per key; no digest'}


def validate_init(ctx):
    cfg = ctx.cfg
    init = cfg['fresh_init']
    root = Path(init['directory'])
    require(not list(root.glob('BLOCKED*')), 'Fresh init blocked')
    verify([init['complete']])
    require(init['complete']['path'] == str(root / 'COMPLETE.json'), 'Init commit pin redirected')
    commit = read_json(root / 'COMPLETE.json')
    require(commit['complete'] is True and commit['kind'] == 'fresh-stock-init-v1' and
            commit['seed'] == cfg['seed'], 'Wrong/nonfresh init commit')
    verify(commit['files'])
    require(commit['manifest']['path'] == str(root / 'FRESH_MANIFEST.json') and
            commit['manifest'] in commit['files'], 'Init manifest unbound')
    manifest = read_json(root / 'FRESH_MANIFEST.json')
    require(manifest['construction'] == CONSTRUCTION and manifest['seed'] == cfg['seed'] and
            manifest['stock_files'] == cfg['stock_files'] and manifest['model_forward_calls'] == 0 and
            manifest['trainer_code'] == ctx.code and
            manifest['no_project_adapter_head_optimizer_teacher_loaded'] is True and
            manifest['audit']['saved_key_count'] == 393 and manifest['audit']['lora_pairs'] == 196 and
            manifest['audit']['fresh_B_exactly_zero'] is True, 'Fresh init origin/construction differs')
    actual_files = {str(p) for p in root.rglob('*') if p.is_file() and p.name != 'COMPLETE.json'}
    require(actual_files == {p['path'] for p in commit['files']}, 'Unbound or missing init files')
    adapter_cfg = read_json(root / 'adapter_config.json')
    require(adapter_cfg['base_model_name_or_path'] == STOCK and adapter_cfg['r'] == 32 and
            adapter_cfg['lora_alpha'] == 64 and adapter_cfg['lora_dropout'] == 0 and
            set(adapter_cfg['target_modules']) == set(TARGETS) and
            adapter_cfg['modules_to_save'] is not None and 'score' in adapter_cfg['modules_to_save'] and
            adapter_cfg['task_type'] == 'SEQ_CLS' and adapter_cfg['bias'] == 'none', 'Wrong fresh adapter config')
    return {'complete': init['complete'], 'commit': commit, 'manifest': manifest}


def load_fresh(ctx, identity=None):
    identity = validate_init(ctx) if identity is None else identity
    model = build_stock(ctx.cfg['seed'])
    comparison = load_saved_adapter(model, ctx.cfg['fresh_init']['directory'], fresh=True)
    require(audit_model(model, require_fresh=True) == identity['manifest']['audit'], 'Fresh model manifest mismatch')
    return model, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--arm', default='C_P_CE', help='Admission scope only; init is loss/layout/mode-independent')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    offline_environment()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only fresh-init launch required')
    ctx = Context(args.config, args.arm)
    stage = new_directory(args.output, ctx.cfg['output_parent'])
    try:
        model = build_stock(ctx.cfg['seed'])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(STOCK, local_files_only=True, trust_remote_code=False)
        require(tokenizer.pad_token_id == 151643 and tokenizer.eos_token_id == 151645, 'Native tokenizer IDs differ')
        model.save_pretrained(stage, safe_serialization=True, save_embedding_layers=False)
        model.config.save_pretrained(stage)
        tokenizer.save_pretrained(stage)
        manifest = {'kind': 'fresh-stock-init-v1', 'seed': ctx.cfg['seed'], 'construction': CONSTRUCTION,
                    'stock_files': ctx.cfg['stock_files'], 'audit': audit_model(model, require_fresh=True),
                    'versions': versions(), 'model_forward_calls': 0, 'trainer_code': ctx.code,
                    'no_project_adapter_head_optimizer_teacher_loaded': True}
        write_json(stage / 'FRESH_MANIFEST.json', manifest)
        ctx.check()
        fsync_tree(stage)
        files = [stat(p) for p in sorted(stage.rglob('*')) if p.is_file()]
        commit_json(stage / 'COMPLETE.json', {'complete': True, 'kind': 'fresh-stock-init-v1',
                                           'seed': ctx.cfg['seed'], 'files': files,
                                           'manifest': stat(stage / 'FRESH_MANIFEST.json')})
        fsync_tree(stage)
    except BaseException as error:
        if not (stage / 'BLOCKED.json').exists():
            write_json(stage / 'BLOCKED.json', {'complete': False, 'error_type': type(error).__name__,
                                              'error': str(error), 'partial_preserved': True})
        raise


if __name__ == '__main__':
    main()
