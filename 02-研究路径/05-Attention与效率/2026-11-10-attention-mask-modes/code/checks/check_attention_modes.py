"""CPU check of attention_modes.py on transformers 4.57.6 with an emulated flash kernel.

The emulated kernel has FlashAttention-3's call shape (no dense mask: a causal flag plus
cumulative lengths) and is registered the same way as the training adapter registers the real one.
"""
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForSequenceClassification, Qwen2Config, Qwen2ForSequenceClassification,
                          Qwen3Config, Qwen3ForSequenceClassification)
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # attention_modes.py sits one level up
import attention_modes as am

BACKEND = 'flash_attn3_hopper'
PAD, EOS, VOCAB, LAYERS = 0, 1, 64, 3
CALLS = {'dense': 0, 'varlen': 0}


def _attend(q, k, v, scale, causal):                      # q [Lq,H,D], k/v [Lk,Hkv,D]
    groups = q.shape[1] // k.shape[1]
    q, k, v = q.transpose(0, 1), k.transpose(0, 1).repeat_interleave(groups, 0), v.transpose(0, 1).repeat_interleave(groups, 0)
    mask = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).tril(k.shape[1] - q.shape[1]) if causal else None
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale).transpose(0, 1)


def flash_attn_func(q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, deterministic=False):
    CALLS['dense'] += 1
    return torch.stack([_attend(q[i], k[i], v[i], softmax_scale, causal) for i in range(q.shape[0])])


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=None,
                           causal=False, window_size=(-1, -1), softcap=0.0, deterministic=False):
    CALLS['varlen'] += 1
    assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32 and len(cu_seqlens_q) == len(cu_seqlens_k)
    assert cu_seqlens_q[-1] == q.shape[0] and cu_seqlens_k[-1] == k.shape[0] == v.shape[0]
    assert max_seqlen_q == int(cu_seqlens_q.diff().max()) and max_seqlen_k == int(cu_seqlens_k.diff().max())
    return torch.cat([_attend(q[a:b], k[c:d], v[c:d], softmax_scale, causal)
                      for a, b, c, d in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:], cu_seqlens_k[:-1], cu_seqlens_k[1:])])


def register_stock():
    kernel = types.ModuleType('emulated_fa3')
    kernel.flash_attn_func, kernel.flash_attn_varlen_func = flash_attn_func, flash_attn_varlen_func
    lazy_import_flash_attention(kernel, force_import=True)
    ALL_ATTENTION_FUNCTIONS.register(BACKEND, flash_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(BACKEND, ALL_MASK_ATTENTION_FUNCTIONS['flash_attention_2'])


def rows_to_batch(prefixes, candidates):
    rows = [p + c + [EOS] for p, c in zip(prefixes, candidates)]
    width = (max(map(len, rows)) + 7) // 8 * 8
    ids = torch.full((len(rows), width), PAD)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row); mask[i, :len(row)] = True
    return ids, mask, torch.tensor([len(p) for p in prefixes], dtype=torch.int32), torch.tensor([len(r) for r in rows], dtype=torch.int32)


def tokens(generator, n):
    return torch.randint(2, VOCAB, (n,), generator=generator).tolist()


def same(a, b):
    return torch.allclose(a, b, atol=1e-10, rtol=1e-8)


def differs(a, b):
    return (a - b).abs().max().item() > 1e-5


RESULTS = []


def check(name, ok):
    RESULTS.append(bool(ok)); print(('PASS  ' if ok else 'FAIL  ') + name, flush=True)


def raises(function):
    try:
        function()
    except RuntimeError as error:
        return str(error)
    return None


def main():
    assert transformers.__version__ == '4.57.6', transformers.__version__
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    family = sys.argv[1] if len(sys.argv) > 1 else 'qwen3'
    shape = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=LAYERS,
                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                 pad_token_id=PAD, eos_token_id=EOS, bos_token_id=None, num_labels=2, initializer_range=0.2)
    folder = tempfile.mkdtemp()
    if family == 'qwen2':
        Qwen2ForSequenceClassification(Qwen2Config(**shape)).save_pretrained(folder)
    else:
        Qwen3ForSequenceClassification(Qwen3Config(head_dim=16, **shape)).save_pretrained(folder)
    print('model family:', family, flush=True)
    register_stock()

    def load(implementation):
        return AutoModelForSequenceClassification.from_pretrained(folder, attn_implementation=implementation,
                                                                  torch_dtype=torch.float64).eval()

    sdpa = load('sdpa')
    prefixes = [tokens(generator, n) for n in (9, 5, 13, 7)]
    candidates = [tokens(generator, n) for n in (3, 6, 1, 4)]
    ids, mask, prefix_len, row_len = rows_to_batch(prefixes, candidates)
    width = ids.shape[1]
    even_ids, even_mask, even_prefix, even_row = rows_to_batch([tokens(generator, 6) for _ in range(3)], [tokens(generator, 1) for _ in range(3)])
    assert even_ids.shape[1] == 8 and bool(even_mask.all()) and not bool(mask.all())

    with torch.no_grad():
        # 1. the emulated kernel reproduces stock causal behaviour
        causal = sdpa(input_ids=ids, attention_mask=mask).logits
        flash = load(BACKEND)
        check('causal: emulated flash == sdpa (padded batch)', same(flash(input_ids=ids, attention_mask=mask).logits, causal))
        even_causal = sdpa(input_ids=even_ids, attention_mask=even_mask).logits
        check('causal: emulated flash == sdpa (unpadded batch)', same(flash(input_ids=even_ids, attention_mask=even_mask).logits, even_causal))

        # 2. bidirectional
        full = sdpa(input_ids=ids, attention_mask=am.dense_mask('bidirectional', row_len, width)).logits
        even_full = sdpa(input_ids=even_ids, attention_mask=am.dense_mask('bidirectional', even_row, 8)).logits
        check('bidirectional differs from causal', differs(full, causal) and differs(even_full, even_causal))
        flash_full = load(BACKEND); am.enable_bidirectional(flash_full)
        before = dict(CALLS)
        check('bidirectional: flash with is_causal=False == sdpa dense mask (padded)', same(flash_full(input_ids=ids, attention_mask=mask).logits, full))
        check('bidirectional: flash with is_causal=False == sdpa dense mask (unpadded)', same(flash_full(input_ids=even_ids, attention_mask=even_mask).logits, even_full))
        check('bidirectional: padded batch used varlen, unpadded batch used the dense kernel',
              CALLS['varlen'] - before['varlen'] == LAYERS and CALLS['dense'] - before['dense'] == LAYERS)
        flagged = load('sdpa')
        for layer in am.attention_layers(flagged):
            layer.is_causal = False
        check('trap: sdpa + is_causal=False + padded 2D mask is still causal on 4.57.6', same(flagged(input_ids=ids, attention_mask=mask).logits, causal))
        check('trap: the same flag on an unpadded batch is bidirectional', same(flagged(input_ids=even_ids, attention_mask=even_mask).logits, even_full))
        flagged.config.is_causal = False
        check('trap: config.is_causal=False changes nothing on 4.57.6', same(flagged(input_ids=ids, attention_mask=mask).logits, causal))

        # 3. prefix_block
        block_mask = am.dense_mask('prefix_block', row_len, width, prefix_len)
        block = sdpa(input_ids=ids, attention_mask=block_mask).logits
        check('prefix_block differs from causal and from bidirectional', differs(block, causal) and differs(block, full))
        check('dense masks never fully mask a query row', bool(block_mask.any(-1).all()) and bool(am.dense_mask('bidirectional', row_len, width).any(-1).all()))
        am.install_prefix_block_flash(BACKEND)
        flash_block = load(BACKEND)
        before = dict(CALLS)
        with am.attention_layout(prefix_len, row_len):
            got = flash_block(input_ids=ids, attention_mask=mask).logits
        check('prefix_block: two flash calls per layer == sdpa dense mask (padded)', same(got, block))
        check('prefix_block: exactly 2 varlen calls per layer', CALLS['varlen'] - before['varlen'] == 2 * LAYERS and CALLS['dense'] == before['dense'])
        with am.attention_layout(even_prefix, even_row):
            got = flash_block(input_ids=even_ids, attention_mask=even_mask).logits
        check('prefix_block: flash == sdpa dense mask (unpadded)', same(got, sdpa(input_ids=even_ids, attention_mask=am.dense_mask('prefix_block', even_row, 8, even_prefix)).logits))
        message = raises(lambda: flash_block(input_ids=ids, attention_mask=mask))
        check('prefix_block: forward without a layout raises', message is not None and 'attention_layout' in message)
        with am.attention_layout(prefix_len[:2], row_len[:2]):
            check('prefix_block: a layout of the wrong batch size raises', raises(lambda: flash_block(input_ids=ids, attention_mask=mask)) is not None)

        # 4. the prefix does not depend on the candidate under prefix_block
        shared = tokens(generator, 10)
        pair_ids, pair_mask, pair_prefix, pair_row = rows_to_batch([shared, shared], [tokens(generator, 2), tokens(generator, 5)])

        def prefix_states(model, attention_mask, layout=None):
            with am.attention_layout(*layout) if layout else torch.no_grad():
                states = model(input_ids=pair_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
            return states[0, :10], states[1, :10], states
        a, b, states = prefix_states(sdpa, am.dense_mask('prefix_block', pair_row, pair_ids.shape[1], pair_prefix))
        check('prefix_block (sdpa): prefix states identical across candidates', same(a, b))
        check('prefix_block (sdpa): all hidden states finite, padding included', bool(torch.isfinite(states).all()))
        a, b, _ = prefix_states(flash_block, pair_mask, (pair_prefix, pair_row))
        check('prefix_block (flash): prefix states identical across candidates', same(a, b))
        a, b, states = prefix_states(sdpa, am.dense_mask('bidirectional', pair_row, pair_ids.shape[1]))
        check('bidirectional: prefix states depend on the candidate', differs(a, b))
        check('bidirectional (sdpa): all hidden states finite, padding included', bool(torch.isfinite(states).all()))

    # 5. gradients, with and without non-reentrant gradient checkpointing
    def gradients(model, forward):
        model.zero_grad()
        forward(model)
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    def loss_of(model, attention_mask):
        logits = model(input_ids=ids, attention_mask=attention_mask).logits
        return (logits[:, 1] - logits[:, 0]).square().sum()

    reference = load('sdpa').train()
    expected = gradients(reference, lambda m: loss_of(m, block_mask).backward())
    student = load(BACKEND).train()

    def inside(model):
        with am.attention_layout(prefix_len, row_len):
            loss_of(model, mask).backward()

    def backward_outside(model):
        with am.attention_layout(prefix_len, row_len):
            loss = loss_of(model, mask)
        loss.backward()

    check('prefix_block gradients: flash == sdpa dense mask', same(gradients(student, inside), expected))
    check('no checkpointing: backward outside the layout is still correct', same(gradients(student, backward_outside), expected))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    before = CALLS['varlen']
    got = gradients(student, inside)
    check('checkpointing: layers really run again in backward (4 varlen calls per layer)', CALLS['varlen'] - before == 4 * LAYERS)
    check('checkpointing: layout open over backward gives the same gradients', same(got, expected))
    student.zero_grad()
    message = raises(lambda: backward_outside(student))
    check('checkpointing: backward outside the layout raises instead of training on a wrong mask', message is not None and 'attention_layout' in message)

    # 6. PEFT LoRA wrapper (zero LoRA delta at init, so logits must equal the plain model)
    def lora(model):
        return get_peft_model(model, LoraConfig(task_type='SEQ_CLS', r=4, lora_alpha=8, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])).eval()
    with torch.no_grad():
        wrapped = lora(load('sdpa'))
        check('PEFT: 4D mask reaches the base model (sdpa prefix_block)', same(wrapped(input_ids=ids, attention_mask=block_mask).logits, block))
        check('PEFT: 4D mask reaches the base model (sdpa bidirectional)', same(wrapped(input_ids=ids, attention_mask=am.dense_mask('bidirectional', row_len, width)).logits, full))
        wrapped_flash = lora(load(BACKEND))
        with am.attention_layout(prefix_len, row_len):
            check('PEFT: flash prefix_block through the wrapper', same(wrapped_flash(input_ids=ids, attention_mask=mask).logits, block))
        register_stock()
        wrapped_full = lora(load(BACKEND)); am.enable_bidirectional(wrapped_full)
        check('PEFT: enable_bidirectional on the wrapper flips every layer', same(wrapped_full(input_ids=ids, attention_mask=mask).logits, full)
              and wrapped_full.config.is_causal is False and all(not layer.is_causal for layer in am.attention_layers(wrapped_full)))

    # 7. the mode survives a config save / load; old configs read as causal
    wrapped_full.config.pair_attention_mode = 'bidirectional'
    saved = tempfile.mkdtemp(); wrapped_full.config.save_pretrained(saved)
    check('config: pair_attention_mode round-trips, missing key reads as causal',
          AutoConfig.from_pretrained(saved).pair_attention_mode == 'bidirectional'
          and getattr(AutoConfig.from_pretrained(folder), 'pair_attention_mode', 'causal') == 'causal')

    print(f'{sum(RESULTS)}/{len(RESULTS)} checks passed; torch {torch.__version__}, transformers {transformers.__version__}')
    raise SystemExit(0 if all(RESULTS) else 1)


if __name__ == '__main__':
    main()
