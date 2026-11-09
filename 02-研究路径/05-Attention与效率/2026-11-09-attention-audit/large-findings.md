# Qwen 27B attention audit — 2026-11-09

Read-only source/card inspection. No model/tokenizer/test/evaluation was run. Current fetched index generated_utc=2026-11-06T09:05:08Z. Mammoth and d3 excluded because another agent handles them.

## Strong causal examples

- **Kev 27B**: Qwen3.8-27B post-trained release, **causal**. Pointer head does not imply bidirectional backbone. Its generic module also supports block-causal for attention-only backbones, but the Qwen3.8 hybrid branch always uses ordinary causal rows.
- **Sieve 27B v2**: Qwen3.8-27B, **causal**. Pointer-family loader, cached state then separate causal question rows.
- **linne-fern-1.0**: documented inference is stock causal Qwen AutoModelForCausalLM, no mask changes. Card is an inference example rather than full runtime audit.

## Evidence table

| Model | Classification | Evidence |
|---|---|---|
| vLLM-SR Decision 2.0 Vega 27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| Kev 27B | causal | Qwen hybrid backbones always run each question as one causal row; source is_hybrid/rows_form/_rows_hidden. Card calls it causal LM backbone and identifies Qwen post-trained release. [1](https://huggingface.co/jaredpalmer/kev-27b); [2](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L259-L262); [3](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L329-L360) |
| simple-jev · Qwen3.8-27B (featherless) | unknown | No direct causal/full/block classification justified from scanned card.  |
| JEV-27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| Eikos-27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| JADE | unknown | No direct causal/full/block classification justified from scanned card.  |
| Decider chat · Qwen3.6-27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| Cloudflare clef | unknown | No direct causal/full/block classification justified from scanned card.  |
| Jevfire | unknown | No direct causal/full/block classification justified from scanned card.  |
| Solomon v1.1 | unknown | No direct causal/full/block classification justified from scanned card.  |
| linne-fern 1.0 (27B) | causal | Card says standard Qwen architecture, stock transformers/vLLM/SGLang; full example loads AutoModelForCausalLM and passes ordinary tokenizer output without attention rewrite. [1](https://huggingface.co/MISA-core/linne-fern-1-0) |
| Qwen Decision 27B noncausal LoRA | full-attention layers bidirectional; recurrent layers native | Noncausal mask in full SDPA attention layers; recurrent layers native; no KV cache. [1](https://huggingface.co/jsaurabh/qwen-decision-27b-noncausal-lora) |
| reflex Qwen3.8-27B-FP8 (pmscr) | unknown | No direct causal/full/block classification justified from scanned card.  |
| RSI-Jev v6.1 VL 27B (effort auto) | unknown | No direct causal/full/block classification justified from scanned card.  |
| StartLux-Decision v1.1 27B | full-attention layers bidirectional; recurrent layers causal | Each question prompt independently; full layers read later prompt tokens, linear layers causal; jev_prompt_attention=bidirectional. [1](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant) |
| Perplexity Decider v1.2 (27B) | full-attention layers bidirectional; recurrent layers causal | Full-attention layers noncausal over entire joint request; native linear-attention remains causal. [1](https://huggingface.co/perplexity-ai/pplx-decider-v1.2) |
| Amos 1 Large (27B) | unknown | No direct causal/full/block classification justified from scanned card.  |
| Lychee-27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| Nace.AI Drex Max v1 (27B) | unknown | No direct causal/full/block classification justified from scanned card.  |
| Matilda Decision v1 | full-attention layers bidirectional | Trained with bidirectional attention and bundled runtime; full layers see whole prompt. [1](https://huggingface.co/Maincode/matilda-decision-v1) |
| Sieve 27B v2 | causal | load_sieve dispatches Sieve-27B-v2 to pointer loader. model.logits uses each question as a batch of causal rows, native backbone and only 2D padding mask, with optional cached state. [1](https://huggingface.co/sthanika-ai/Sieve-27B-v2); [2](https://github.com/sthanika-ai/Sieve/blob/1e64657c930f91db7119b5638774c0f11a77e441/sieve/load.py#L215-L234); [3](https://github.com/sthanika-ai/Sieve/blob/1e64657c930f91db7119b5638774c0f11a77e441/sieve/model.py#L112-L142) |
| Cerno 27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| OneJev1.6-27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| frontier-infra Jebadiah 27B v2.1 (t1e) | unknown | No direct causal/full/block classification justified from scanned card.  |
| VisionJEV 1.0.0 (27B) | unknown | No direct causal/full/block classification justified from scanned card.  |
| Lumma-fev-27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| Ainvest-Decision-27B v1.0 | full-attention layers bidirectional; recurrent layers native | Noncausal full-attention layers; recurrent preserve evaluated semantics. [1](https://huggingface.co/Ainvest-World/Ainvest-Decision-27b-v1.0) |
| WeMM-JEV-27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| OneShot 27B | unknown | No direct causal/full/block classification justified from scanned card.  |
| vera-large | full-attention layers bidirectional; recurrent layers causal | is_causal:false, encoder-only full attention; Gated DeltaNet stays causal. [1](https://huggingface.co/scar-ai/vera-large) |
| NTele-Hybrid-27B-v1 | unknown | No direct causal/full/block classification justified from scanned card.  |
| Bespoke Nimble v4.1 Max (27B) | unknown | No direct causal/full/block classification justified from scanned card.  |
| Piccolo 27B | bidirectional (author stated; exact mask not audited) | Tags bidirectional-attention and requires included reference implementation for bidirectional attention. [1](https://huggingface.co/agentscope-ai/Piccolo-27B) |
| GLiDE 27B | full-attention layers bidirectional; recurrent layers native | attention_masks returns full_attention=padding[:,None,None,:].bool(), no causal triangle; linear_attention is padding only. [1](https://huggingface.co/fastino/GLiDE-27B/blob/main/glide/model.py#L35-L39) |
| Torchcast Decision 27B | unknown | No direct causal/full/block classification justified from scanned card.  |

## Additional confirmed block/prefix example

**Nimble v4.1 Max (27B)**: prefix/block-bidirectional. Released schema_config sets `prefix_attention=bidirectional_v1`. All request tokens up to and including last `Requested field:` are block id 0 and read each other both ways in full-attention layers. The field name and chat tail are block id -1, causal. Recurrent layers remain causal. See [runtime](https://huggingface.co/datasets/bespokelabs/nimble-v4-max-decision-index/blob/main/code/nimble/training/prefix_attention.py) and [release contract](https://huggingface.co/bespokelabs/Nimble-v4-max/blob/main/schema_config.json). This is a third block/prefix-style example beyond Mammoth/d3.
