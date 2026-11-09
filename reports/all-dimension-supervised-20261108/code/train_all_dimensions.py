#!/usr/bin/env python3
"""Actual fresh two-stage all-domain supervised training; no Teacher or eval."""
import argparse
from collections import Counter
from datetime import timedelta
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

HOME = Path('/home/dev/assumeengage')
sys.path.insert(0, str(HOME / 'scripts'))
from pair_data_common import DOMAINS
from pair_training_runtime import PairPool, TokenStore, collate, microbatches
from student_flash_attention3 import register_student_fa3, assert_student_uses_fa3, attention_record


def atomic(path, data):
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    root = Path(cfg['run_root'])
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    lock = None
    if rank == 0:
        lock = (root / 'TRAINING.lock').open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('gloo', timeout=timedelta(minutes=60))
    set_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    register_student_fa3(cfg['attention_kernel_directory'])
    pool = PairPool('MIXED_ALL')
    store = TokenStore()
    tokenizer = AutoTokenizer.from_pretrained(cfg['common_initialization'], local_files_only=True)
    stages = json.loads((Path(cfg['schedule_root']) / 'READY.json').read_text())['stages']
    stages = [s for s in stages if s['name'] not in cfg.get('skip_stages', [])]
    checkpoints = Path(cfg['checkpoint_root'])
    checkpoints.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        atomic(root / 'TRAINING_OWNER.json', {'pid': os.getpid(), 'world_size': world,
                                            'gpus': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                            'all_training_domains': DOMAINS})
    for stage in stages:
        name = stage['name']
        out = root / name
        out.mkdir(exist_ok=True)
        if (out / 'complete.json').is_file():
            continue
        set_seed(cfg['seed'] + (0 if name == 'stage1' else 2))
        latest = out / 'latest.json'
        checkpoint = Path(json.loads(latest.read_text())['path']) if latest.is_file() else None
        model_config = AutoConfig.from_pretrained(cfg['common_initialization'], local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg['base_model'], config=model_config, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attn3_hopper', local_files_only=True)
        if checkpoint:
            model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
        elif cfg.get('initial_adapter') and name == 'stage2':
            model = PeftModel.from_pretrained(model, cfg['initial_adapter'], is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.SEQ_CLS, r=stage['rank'], lora_alpha=stage['alpha'],
                lora_dropout=0.0, bias='none', modules_to_save=['score'],
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
            if name == 'stage2':
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.data = parameter.data.float()
                previous = load_file(str(root / 'stage1/final/adapter_model.safetensors'))
                expanded = get_peft_model_state_dict(model)
                for key, target in expanded.items():
                    source = previous[key]
                    if 'lora_A' in key:
                        target[:source.shape[0], :].copy_(source)
                    elif 'lora_B' in key:
                        target.zero_()
                        target[:, :source.shape[1]].copy_(source)
                    else:
                        target.copy_(source)
                set_peft_model_state_dict(model, expanded)
                del previous, expanded
        assert_student_uses_fa3(model)
        model.config.use_cache = False
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        model.to(device).train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=stage['learning_rate'],
                                      weight_decay=0.01, fused=True)
        order = np.load(stage['order_path'], mmap_mode='r')
        macro = cfg['batch_pairs']
        steps = math.ceil(len(order) / macro)
        warmup = max(1, int(steps * 0.03))

        def multiplier(step):
            if step < warmup:
                return max(1e-8, step / warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        completed = pairs = tokens = 0
        prior_seconds = 0.0
        seen = Counter()
        if checkpoint:
            state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=True)
            if state['stage'] != name or state['schedule_pairs'] != len(order):
                raise RuntimeError('Existing checkpoint belongs to another training schedule')
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            completed, pairs, tokens = state['step'], state['pairs'], state['tokens']
            prior_seconds = state['elapsed_seconds']
            seen.update(state['seen_domains'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state(state['cuda_rng'], device)
            del state
        sizes = [p.numel() for p in parameters]
        offsets = np.cumsum([0] + sizes)
        gradients = torch.zeros(int(offsets[-1]), device=device, dtype=torch.float32)
        host = torch.empty(int(offsets[-1]), dtype=torch.float32, pin_memory=True) if world > 1 else None
        for i, p in enumerate(parameters):
            p.grad = gradients[offsets[i]:offsets[i + 1]].view_as(p)
        token_budget = cfg['microbatch_tokens']
        checkpointing = False
        started = time.monotonic()
        last_time, last_tokens = started, tokens
        if rank == 0:
            print(json.dumps({'event': 'formal_training_started', 'stage': name, 'resume_step': completed,
                              'pairs': len(order), 'tokens': stage['tokens'], 'steps': steps,
                              'rank': stage['rank'], 'learning_rate': stage['learning_rate'],
                              'world_size': world, **attention_record()}), flush=True)

        def save(step, final=False):
            if world > 1:
                dist.barrier()
            if rank == 0:
                folder = checkpoints / name / f'step-{step:09d}'
                folder.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(folder, safe_serialization=True)
                model.config.save_pretrained(folder)
                elapsed = prior_seconds + time.monotonic() - started
                temporary = folder / 'training_state.pt.writing'
                torch.save({'stage': name, 'schedule_pairs': len(order), 'step': step,
                            'pairs': pairs, 'tokens': tokens, 'elapsed_seconds': elapsed,
                            'seen_domains': dict(seen), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
                            'cuda_rng': torch.cuda.get_rng_state(device)}, temporary)
                temporary.replace(folder / 'training_state.pt')
                atomic(latest, {'path': str(folder), 'step': step})
                if final:
                    destination = out / 'final'
                    destination.mkdir(exist_ok=True)
                    model.save_pretrained(destination, safe_serialization=True)
                    model.config.save_pretrained(destination)
                    tokenizer.save_pretrained(destination)
                    atomic(out / 'complete.json', {'complete': True, 'stage': name,
                           'pairs_seen': pairs, 'effective_tokens': tokens, 'optimizer_steps': step,
                           'seen_domains': dict(seen), 'elapsed_seconds': elapsed,
                           'learning_rate': stage['learning_rate'], 'lora_rank': stage['rank'],
                           'source_labels_only': True, **attention_record()})
            if world > 1:
                dist.barrier()

        logits = loss = None
        for step in range(completed, steps):
            indices = order[step * macro:min((step + 1) * macro, len(order))]
            global_batch = pool.fetch(indices)
            local_batch = {k: v[rank::world] for k, v in global_batch.items()}
            while True:
                gradients.zero_()
                failed = torch.zeros(1, dtype=torch.int32)
                batch_loss = 0.0
                inputs = target = logits = loss = None
                try:
                    for micro in microbatches(local_batch, cfg['microbatch_pairs'], token_budget):
                        inputs, target, _ = collate(store, micro, tokenizer.pad_token_id,
                                                    tokenizer.eos_token_id, device)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits = model(**inputs).logits
                            loss = F.binary_cross_entropy_with_logits(
                                logits[:, 1].float() - logits[:, 0].float(), target, reduction='sum')
                        (loss / len(indices)).backward()
                        batch_loss += float(loss.detach())
                        inputs = target = logits = loss = None
                except torch.cuda.OutOfMemoryError:
                    failed.fill_(1)
                    inputs = target = logits = loss = None
                    gc.collect()
                    torch.cuda.empty_cache()
                if world > 1:
                    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                if not failed.item():
                    break
                if token_budget > 4096:
                    token_budget = max(4096, token_budget // 2)
                elif not checkpointing:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                    checkpointing = True
                else:
                    raise RuntimeError('A full training example cannot fit; no samples dropped')
            if world > 1:
                host.copy_(gradients, non_blocking=True)
                torch.cuda.synchronize(device)
                dist.all_reduce(host, op=dist.ReduceOp.SUM)
                gradients.copy_(host, non_blocking=True)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=True)
            optimizer.step()
            scheduler.step()
            pairs += len(indices)
            tokens += int(global_batch['length'].sum())
            seen.update({DOMAINS[int(d)]: int(n) for d, n in zip(*np.unique(global_batch['domain'], return_counts=True))})
            if step == completed or (step + 1) % 10 == 0 or step + 1 == steps:
                torch.cuda.synchronize(device)
                now = time.monotonic()
                record = {'stage': name, 'step': step + 1, 'total_steps': steps,
                          'pairs_seen': pairs, 'effective_tokens': tokens, 'seen_domains': dict(seen),
                          'loss_local': batch_loss / max(1, len(indices)),
                          'learning_rate': scheduler.get_last_lr()[0],
                          'effective_tokens_per_second': (tokens - last_tokens) / max(.001, now - last_time),
                          'elapsed_seconds': prior_seconds + now - started, **attention_record()}
                if rank == 0:
                    atomic(root / 'progress.json', record)
                    print(json.dumps({'event': 'formal_optimizer_progress', **record}), flush=True)
                last_time, last_tokens = now, tokens
            if (step + 1) % cfg['checkpoint_steps'] == 0:
                save(step + 1)
        save(steps, final=True)
        del model, optimizer, scheduler, gradients, host, parameters, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        atomic(root / 'COMPLETE.json', {'complete': True, 'model': 'Qwen3-0.6B-all-dimensions',
               'final_model': str(root / 'stage2/final'), 'all_training_domains': DOMAINS,
               'continued_from': cfg.get('initial_adapter'),
               'stages': [json.loads((root / stage['name'] / 'complete.json').read_text()) for stage in stages]})
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
