#!/usr/bin/env python3
"""Five effective-weight merges of the completed, common-origin specialists."""
import gc
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

HOME = Path('/home/dev/assumeengage')
CFG = json.loads((HOME / 'config/experiment-five-specialists.json').read_text())
ROOT = HOME / CFG['run_root']
COMMON = HOME / CFG['common_initialization']
SPECIALISTS = [name for name in CFG['models'] if name.startswith('SPECIALIST_')]
METHODS = {
    'simple_average': 'MERGE_AVERAGE',
    'task_arithmetic': 'MERGE_TASK_ARITHMETIC',
    'ties': 'MERGE_TIES',
    'dare_task_arithmetic': 'MERGE_DARE_TASK_ARITHMETIC',
    'dare_ties': 'MERGE_DARE_TIES',
}


def done(path):
    try:
        return json.loads(path.read_text()).get('complete') is True
    except (OSError, ValueError):
        return False


@torch.no_grad()
def effective_state(adapter):
    config = AutoConfig.from_pretrained(COMMON, local_files_only=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        HOME / CFG['base_model'], config=config, torch_dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    dense = model.merge_and_unload()
    return {name: value.detach().cpu().float().contiguous()
            if value.is_floating_point() else value.detach().cpu().clone()
            for name, value in dense.state_dict().items()}


def trim(values, density):
    if density >= 1:
        return values
    out = torch.zeros_like(values)
    for index, tensor in enumerate(values):
        keep = max(1, int(tensor.numel() * density))
        chosen = tensor.abs().reshape(-1).topk(keep, sorted=False).indices
        out[index].reshape(-1)[chosen] = tensor.reshape(-1)[chosen]
    return out


def ties(values, density):
    sparse = trim(values, density)
    sign = sparse.sum(dim=0).sign()
    selected = (sparse.sign() == sign.unsqueeze(0)) & (sparse != 0)
    return (sparse * selected).sum(dim=0) / selected.sum(dim=0).clamp_min(1)


def main():
    if done(ROOT / 'MERGES_COMPLETE.json'):
        print('Reusing all five completed effective-weight merges.', flush=True)
        return
    for name in SPECIALISTS:
        if not done(ROOT / name / 'complete.json'):
            raise RuntimeError('Specialist full epoch is not complete: ' + name)
    torch.set_num_threads(8)
    started = time.monotonic()
    print(f'Loading common and {len(SPECIALISTS)} specialist effective FP32 weights on CPU.', flush=True)
    base = effective_state(COMMON)
    states = [effective_state(ROOT / name / 'final') for name in SPECIALISTS]
    if 'score.weight' not in base:
        raise RuntimeError('Common classification head is absent')
    for state in states:
        if state.keys() != base.keys():
            raise RuntimeError('Effective tensor keys differ; no classification tensors will be discarded')
    outputs = {method: {} for method in METHODS}
    generators = [torch.Generator().manual_seed(CFG['merges']['dare_task_arithmetic']['seed'] + index)
                  for index in range(len(states))]
    modified = 0
    with torch.no_grad():
        for position, key in enumerate(sorted(base)):
            initial = base[key]
            if not initial.is_floating_point() or all(torch.equal(state[key], initial) for state in states):
                shared = initial.to(torch.bfloat16).contiguous() if initial.is_floating_point() else initial
                for output in outputs.values():
                    output[key] = shared
                continue
            modified += 1
            delta = torch.stack([state[key] - initial for state in states])
            keep = CFG['merges']['dare_task_arithmetic']['keep_probability']
            dare = torch.stack([delta[index] * (torch.rand(delta[index].shape, generator=generators[index]) < keep) / keep
                                for index in range(len(states))])
            mixtures = {
                'simple_average': delta.mean(dim=0),
                'task_arithmetic': CFG['merges']['task_arithmetic']['lambda'] * delta.sum(dim=0),
                'ties': CFG['merges']['ties']['lambda'] * ties(delta, CFG['merges']['ties']['density']),
                'dare_task_arithmetic': CFG['merges']['dare_task_arithmetic']['lambda'] * dare.sum(dim=0),
                'dare_ties': CFG['merges']['dare_ties']['lambda'] * ties(dare, CFG['merges']['dare_ties']['density']),
            }
            for method, update in mixtures.items():
                outputs[method][key] = (initial + update).to(torch.bfloat16).contiguous()
            if modified % 20 == 0:
                print(json.dumps({'event': 'effective_merge_progress', 'modified_tensors': modified,
                                  'tensor': key, 'elapsed_seconds': time.monotonic() - started}), flush=True)
            del delta, dare, mixtures
    del states, base
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(COMMON, local_files_only=True)
    config = AutoConfig.from_pretrained(COMMON, local_files_only=True)
    config.architectures = ['Qwen3ForSequenceClassification']
    config.torch_dtype = torch.bfloat16
    config.use_cache = False
    records = []
    for method, name in METHODS.items():
        folder = ROOT / 'merges' / name
        final = folder / 'final'
        final.mkdir(parents=True, exist_ok=True)
        save_file(outputs.pop(method), str(final / 'model.safetensors'), metadata={'format': 'pt'})
        config.save_pretrained(final)
        tokenizer.save_pretrained(final)
        record = {'complete': True, 'model': name, 'method': method, 'checkpoint': str(final),
                  'checkpoint_kind': 'dense_effective_sequence_classifier', 'specialists': SPECIALISTS,
                  'classification_head_included': True, 'compute_dtype': 'float32', 'storage_dtype': 'bfloat16',
                  'modified_tensor_count': modified, 'new_training_tokens': 0,
                  'parameters': CFG['merges'][method],
                  'dare_scope': 'Bernoulli retention with exact 1/keep rescale before the selected merge',
                  'ties_scope': 'per-tensor magnitude trim, sum sign, nonzero sign-consistent mean'}
        (folder / 'complete.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print(json.dumps({'event': 'effective_merge_saved', **record}), flush=True)
    summary = {'complete': True, 'models': records, 'new_training_models': 0,
               'elapsed_seconds': time.monotonic() - started}
    (ROOT / 'MERGES_COMPLETE.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
