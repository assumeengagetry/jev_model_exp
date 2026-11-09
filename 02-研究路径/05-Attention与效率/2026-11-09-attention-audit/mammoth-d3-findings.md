# Mammoth / d3 attention audit — 2026-11-09

Read-only source inspection. No model, tokenizer, test, smoke run, evaluation, or hash computation was run. Small author source/config text snapshots are in `mammoth-d3-sources/`.

## Results

The current Jev Index has these entries. Model sizes/base names below are from its `index.json` metadata. Attention is from each author's actual configuration and runtime; the board's `meta.technique = autoregressive` is **not** a reliable attention-mask field.

| Entry | Board base | Attention in full softmax-attention layers | Linear/recurrent layers |
|---|---|---|---|
| baselabs mammoth-1 (27B) | Qwen/Qwen3.5-27B | block bidirectional, prefix then question suffix | causal |
| vLLM-SR d3-turbo v1.1 (27B) | Qwen/Qwen3.8-27B | `prefix_bidir`, state block then question block | causal |
| vLLM-SR d3-ultra (Decision 3.0, 27B) | Qwen/Qwen3.8-27B | `prefix_bidir`, state block then question block | causal |
| vLLM-SR d3 (27B), tag v3.1.0 | Qwen/Qwen3.8-27B | `noncausal_full_attention`, whole prompt bidirectional | causal |
| vLLM-SR d3-flash (9B), tag v3.1.0 | Qwen/Qwen3.5-9B | `noncausal_full_attention` | causal |
| vLLM-SR d3-mini (4B), tag v3.1.0 | Qwen/Qwen3.5-4B | `noncausal_full_attention` | causal |
| vLLM-SR d3-nano (2B), tag v3.1.0 | Qwen/Qwen3.5-2B | `noncausal_full_attention` | causal |
| vLLM-SR d3-lite (0.8B), tag v3.1.0 | Qwen/Qwen3.5-0.8B | `noncausal_full_attention` | causal |
| vLLM-SR d3-edge (0.6B), tag v3.1.0 | Qwen/Qwen3.5-0.8B | `noncausal_full_attention` | causal |

The 0.6B edge model is a smaller served model based on 0.8B; it is not evidence for a Qwen3-0.6B backbone.

## Exact author evidence

Mammoth index revision `9d1962f193e48a68cf0f3598dc7fb4462fd23ad9`:

- [config.json L146–149](https://huggingface.co/baselabs/mammoth-1/blob/9d1962f193e48a68cf0f3598dc7fb4462fd23ad9/config.json#L146): `modeling_overrides.attention.full_attention = block_bidirectional`, `linear_attention = causal`.
- [serve.patch L250–257](https://huggingface.co/baselabs/mammoth-1/blob/9d1962f193e48a68cf0f3598dc7fb4462fd23ad9/serve.patch#L250): full-attention tokens see both directions inside a block and earlier blocks. Shared prefix = system turn + state; suffix = question + options + answer line + generation prompt. Prefix cannot see suffix, so prefix caching remains possible. GatedDeltaNet stays causal.
- Same patch L320–333 loads the mask from `config.json`, so `readout.json` not containing an attention field does not imply the default causal mask.

d3-turbo actual v1.1 tag:

- [decision_config.json L520](https://huggingface.co/vllm-sr/d3-turbo/blob/v1.1/decision_config.json#L520): `attention_mode = prefix_bidir`.
- [d3_runtime.py L40–54](https://huggingface.co/vllm-sr/d3-turbo/blob/v1.1/d3_runtime.py#L40): explicit prose description. State block = system, request images/videos and state text; question block = remaining question/options/readout. State sees itself bidirectionally, question sees state and itself bidirectionally, state does not see question. GatedDeltaNet remains causal over the prompt.
- [d3_runtime.py L1166–1203](https://huggingface.co/vllm-sr/d3-turbo/blob/v1.1/d3_runtime.py#L1166): actual implementation uses two unmasked SDPA calls, one state-to-state and one question-to-whole-row; this equals the stated dense block mask.

d3-ultra index revision:

- [decision_config.json L533](https://huggingface.co/vllm-sr/d3-ultra/blob/37fee01926f3de396620e794201b074fb427ac2d/decision_config.json#L533): `attention_mode = prefix_bidir`.
- [d3_runtime.py L44–58](https://huggingface.co/vllm-sr/d3-ultra/blob/37fee01926f3de396620e794201b074fb427ac2d/d3_runtime.py#L44): same block rule and causal GatedDeltaNet.

d3 main family index tag v3.1.0:

- [d3 decision_config.json L522](https://huggingface.co/vllm-sr/d3/blob/v3.1.0/decision_config.json#L522): `attention_mode = noncausal_full_attention`.
- [d3 runtime L40–44](https://huggingface.co/vllm-sr/d3/blob/v3.1.0/d3_runtime.py#L40) and [mask hook L903–939](https://huggingface.co/vllm-sr/d3/blob/v3.1.0/d3_runtime.py#L903): full softmax-attention layers see future tokens; linear recurrence remains causal. The mask dictionary replaces `full_attention` with the padding mask while preserving `create_recurrent_attention_mask` for linear layers.
- Each of [flash](https://huggingface.co/vllm-sr/d3-flash/blob/v3.1.0/decision_config.json#L522), [mini](https://huggingface.co/vllm-sr/d3-mini/blob/v3.1.0/decision_config.json#L522), [nano](https://huggingface.co/vllm-sr/d3-nano/blob/v3.1.0/decision_config.json#L522), [lite](https://huggingface.co/vllm-sr/d3-lite/blob/v3.1.0/decision_config.json#L522), [edge](https://huggingface.co/vllm-sr/d3-edge/blob/v3.1.0/decision_config.json#L522) declares the same attention mode.

## Plain-language explanation

`block-bidirectional` here splits the input into a scene/state block and a question/options block. Words can look both ways inside their own block. The question can also look back at the state; the state never looks forward into the question. That permits computing shared state once for several questions. It does **not** mean fixed-size chunks such as every 128 tokens, and it does **not** mean every layer became a bidirectional Transformer.

Qwen hybrid architecture (linear/recurrent plus softmax-attention layers) and causal/bidirectional masking are separate axes. These author configurations alter softmax-attention visibility while preserving causal recurrence in the GatedDeltaNet layers.
