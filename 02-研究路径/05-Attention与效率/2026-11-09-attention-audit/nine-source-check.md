# 9B source checks (read-only, 2026-11-09)

## Sieve-9B and Sieve-9B-v2: causal

Cards route to the same author repository. Current source explicitly describes causal rows:
https://github.com/sthanika-ai/Sieve/blob/main/sieve/model.py
- Lines 31–32: a long single question runs as one causal row; stays on fused causal kernel.
- Lines 105–113: logits() uses cached state followed by a batch of causal question rows.
- Ordinary 2D padding mask is passed to the original Qwen backbone; no full-bidirectional override.
Cards:
https://huggingface.co/sthanika-ai/Sieve-9B
https://huggingface.co/sthanika-ai/Sieve-9B-v2
Both use post-trained Qwen/Qwen3.5-9B, not -Base.

## JPT-9B: native causal, source implementation

Card: https://huggingface.co/kirp/jpt-9b
Base explicitly Qwen/Qwen3.5-9B-Base. The card routes to standard SGLang/vLLM servers and llm2jev.
https://github.com/tic-top/llm2jev/blob/main/llm2jev/backends.py
- Lines 35–41: SGLang native /generate request, max_new_tokens=1 and selected token logprobs, no custom attention mask.
- Lines 80–83: vLLM native /v1/completions with max_tokens=1, no mask override.
- Lines 147–158: HF reference loads standard architecture class via from_pretrained, no custom attention class/remote patch.
- Lines 182–187: passes tokenized input straight to self.model(**inputs), reads last-position logits.
Classification follows native Qwen causal behavior, not the API spelling or the metadata technique label alone.

## Bespoke Nimble 9B v2: native causal, source implementation

Card: https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2
Actual card base is Qwen/Qwen3.5-9B, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. Index metadata lists -Base, so report actual card evidence.
Published inference: https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2/blob/main/inference.py
- Lines 51–63: standard 2D left-padding mask, original model forward, last token logits.
- Lines 109–111: standard Qwen3_5ForConditionalGeneration with LoRA, no attention mask patch.
Additional author source: https://github.com/bespokelabsai/nimble/blob/main/nimble/scoring/parallel_scorer.py
- Lines 139–141 explicitly state causality prevents right padding affecting valid outputs.

## Bespoke Nimble 9B v3: native causal, formal evaluation source

Card: https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3
Base Qwen/Qwen3.5-9B.
The card links its formal inference code:
https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index/blob/main/code/nimble/evaluation/decision_index_engine.py
- Lines 93–104: standard 2D left-padding mask, original forward, last-position logits.
- Lines 133–143: Qwen3_5ForConditionalGeneration, trust_remote_code=False, no attention patch.
- Lines 222–248: common prefix caching, copies hybrid attention KV/recurrent state to causal question tails with standard padding mask.

Only source files/model cards were read. No model, tokenizer, GPU, hash, test, or inference was executed.
