#!/usr/bin/env python3
"""Create the actual shared LoRA/classification initialization for all six models."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model

HOME = Path('/home/dev/assumeengage')
BASE = HOME / 'models/Qwen3-0.6B'
OUTPUT = HOME / 'models/Qwen3-0.6B-common-pair-init-r32'

if (OUTPUT / 'adapter_model.safetensors').exists():
    print('Reusing the existing shared classifier/LoRA initialization:', OUTPUT, flush=True)
    raise SystemExit(0)

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
tokenizer.padding_side = 'right'
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2, torch_dtype=torch.bfloat16,
    attn_implementation='sdpa', local_files_only=True,
)
model.config.problem_type = 'single_label_classification'
model.config.id2label = {0: 'not_match', 1: 'match'}
model.config.label2id = {'not_match': 0, 'match': 1}
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.architectures = ['Qwen3ForSequenceClassification']
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.SEQ_CLS, r=32, lora_alpha=64, lora_dropout=0.0, bias='none',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    modules_to_save=['score'],
))
OUTPUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT, safe_serialization=True)
model.config.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
(OUTPUT / 'initialization.json').write_text(json.dumps({
    'base_model': str(BASE), 'model_class': 'Qwen3ForSequenceClassification',
    'seed': 42, 'labels': ['not_match', 'match'], 'precision': 'bfloat16',
    'lora_r': 32, 'lora_alpha': 64, 'classification_head_saved': True,
    'attention': 'causal SDPA', 'base_weights_duplicated': False,
    'purpose': 'actual common initialization for five specialists and MIXED_ALL',
    'model_forward_or_smoke_test_run': False,
}, ensure_ascii=False, indent=2))
print('Shared classifier head and identical initial LoRA weights saved:', OUTPUT, flush=True)
