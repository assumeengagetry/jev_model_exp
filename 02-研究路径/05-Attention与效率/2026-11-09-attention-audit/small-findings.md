# Small Qwen model attention audit
Read-only author-source review on 2026-11-09. No weights, tokenizer execution, tests, smoke tests or model inference.

30 current small-model entries classified from author cards/source; d3 family delegated separately.

Important: leaderboard Bosun metadata has -Base suffix, but actual shipped config loads Qwen/Qwen3-0.6B and Qwen/Qwen3-1.7B (posttrained model names).

## Adjudex-0.8B
full-bidirectional in full-attention layers

README L36: Qwen3_5Model with noncausal attention in full-attention layers, last-token pooling. decision_config.json attention_mode=noncausal_full_attention.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/newsbang/adjudex-0.8b/blob/main/README.md#L36
- https://huggingface.co/newsbang/adjudex-0.8b/blob/main/decision_config.json

## Bosun v3.1 0.6B
causal

modeling_bosun.py L119 loads stock AutoModelForCausalLM and L229-235 forwards input_ids and 2D padding attention mask without replacement. Actual shipped config base is Qwen/Qwen3-0.6B or Qwen/Qwen3-1.7B, contrary to leaderboard -Base suffix.

Leaderboard base: Qwen/Qwen3-0.6B-Base

- https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/modeling_bosun.py#L119
- https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/modeling_bosun.py#L216
- https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/config.json
- https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b/blob/main/config.json

## Bosun v3.1 1.7B
causal

modeling_bosun.py L119 loads stock AutoModelForCausalLM and L229-235 forwards input_ids and 2D padding attention mask without replacement. Actual shipped config base is Qwen/Qwen3-0.6B or Qwen/Qwen3-1.7B, contrary to leaderboard -Base suffix.

Leaderboard base: Qwen/Qwen3-1.7B-Base

- https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/modeling_bosun.py#L119
- https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/modeling_bosun.py#L216
- https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/config.json
- https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b/blob/main/config.json

## crystal-0.8b
full-prompt bidirectional in full-attention layers; causal DeltaNet

crystal_config.json attention=noncausal,deltanet=causal. modeling.py distinguishes attention mode from DeltaNet mode explicitly; enable_noncausal replaces triangle with padding key mask.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/Kyunghwan-kim/crystal-0.8b/blob/main/crystal_config.json
- https://huggingface.co/Kyunghwan-kim/crystal-0.8b/blob/main/crystal/modeling.py#L193

## Decider 2B
causal (source-based inference)

Mapika decider/model.py loads original CausalLM and calls lm.model with ordinary 2D padding attention mask; no custom noncausal mask in this path.

Leaderboard base: Qwen/Qwen3.5-2B-Base

- https://github.com/Mapika/decider/blob/main/decider/model.py#L13
- https://github.com/Mapika/decider/blob/main/decider/model.py#L26

## vLLM-SR Decision 2.0 Eos 0.8B
causal

Bundled decision2/fast.py Masks explicitly says SDPA runs causal; padded full-attention mask uses torch.ones(T,T).tril(). Vendor decision_model.py selects stock Qwen3ForCausalLM or Qwen3_5ForConditionalGeneration; no bidirectional replacement.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/fast.py#L542
- https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/_vendor/dev2model/decision_model.py#L232

## vLLM-SR Decision 2.0 Kai 0.6B
causal

Bundled decision2/fast.py Masks explicitly says SDPA runs causal; padded full-attention mask uses torch.ones(T,T).tril(). Vendor decision_model.py selects stock Qwen3ForCausalLM or Qwen3_5ForConditionalGeneration; no bidirectional replacement.

Leaderboard base: Qwen/Qwen3-0.6B-Base

- https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/fast.py#L542
- https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/_vendor/dev2model/decision_model.py#L232

## vLLM-SR Decision 2.0 Sol 2B
causal

Bundled decision2/fast.py Masks explicitly says SDPA runs causal; padded full-attention mask uses torch.ones(T,T).tril(). Vendor decision_model.py selects stock Qwen3ForCausalLM or Qwen3_5ForConditionalGeneration; no bidirectional replacement.

Leaderboard base: Qwen/Qwen3.5-2B

- https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/fast.py#L542
- https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/_vendor/dev2model/decision_model.py#L232

## GLiDE 0.8B
full-bidirectional in full-attention layers; native causal linear-attention layers

decision_config.json attention_mode=noncausal_full_attention. glide/model.py attention_masks returns full_attention padding-only mask; docstring explicitly no causal mask.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/fastino/GLiDE-0.8B/blob/main/decision_config.json
- https://huggingface.co/fastino/GLiDE-0.8B/blob/main/glide/model.py#L45

## GLiDE 2B
full-bidirectional in full-attention layers; native causal linear-attention layers

decision_config.json attention_mode=noncausal_full_attention. glide/model.py attention_masks returns full_attention padding-only mask; docstring explicitly no causal mask.

Leaderboard base: Qwen/Qwen3.5-2B

- https://huggingface.co/fastino/GLiDE-2B/blob/main/decision_config.json
- https://huggingface.co/fastino/GLiDE-0.8B/blob/main/glide/model.py#L45

## Intern-Decision-0.8B
causal

Both author HF cards explicitly say Run one causal Hugging Face forward pass. HF backend docstring no runtime patches; stock Qwen3_5ForConditionalGeneration and native forward.

Leaderboard base: Qwen/Qwen3.5-0.8B-Base

- https://huggingface.co/internlm/Intern-Decision-0.8B/blob/main/README.md#L28
- https://huggingface.co/internlm/Intern-Decision-2B/blob/main/README.md#L28
- https://github.com/internlm/Intern-Decision/blob/main/src/inference/hf_backend.py#L37

## Intern-Decision-2B
causal

Both author HF cards explicitly say Run one causal Hugging Face forward pass. HF backend docstring no runtime patches; stock Qwen3_5ForConditionalGeneration and native forward.

Leaderboard base: Qwen/Qwen3.5-2B-Base

- https://huggingface.co/internlm/Intern-Decision-0.8B/blob/main/README.md#L28
- https://huggingface.co/internlm/Intern-Decision-2B/blob/main/README.md#L28
- https://github.com/internlm/Intern-Decision/blob/main/src/inference/hf_backend.py#L37

## Jev-Style 0.8B v3
causal

jev_style_decision.py L541-542 explicitly model is causal. L652 calls stock model.model(input_ids=ids,use_cache=False) without mask override. Chunked SDPA helper preserves causal mask (L500-514).

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3/blob/main/jev_style_decision.py#L541
- https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3/blob/main/jev_style_decision.py#L652

## Jev-Style 2B v4
block-bidirectional in full-attention layers; causal Gated DeltaNet

README L375-377: blocks of <=2048 tokens; each block attends all prior blocks and itself with no internal causal mask. Gated DeltaNet remains recurrent. Different from 0.8B v3.

Leaderboard base: Qwen/Qwen3.5-2B

- https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4/blob/main/README.md#L375

## jiwo 0.8B v2
causal

src/jiwo/model.py L324 explicitly every layer is causal; backbone forward uses ordinary padding mask.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://github.com/jiwidi/jiwo/blob/main/src/jiwo/model.py#L320

## Kev 0.8B r15
causal

Author model card calls it causal LM prefill-only plus pointer. kev/model.py explicitly uses one causal row per question for Qwen3.5 hybrid backbones; the packed block-causal mask is for attention-only models and does not mean block-bidirectional.

Leaderboard base: Qwen/Qwen3.5-0.8B-Base

- https://huggingface.co/jaredpalmer/kev-0.8b/blob/main/README.md#L77
- https://github.com/jaredpalmer/kev/blob/main/kev/model.py#L257

## KnowLine 0.8B Gen1
causal (ordinary Qwen SGLang/HF inference; source-based inference)

Shipped KnowLine engine launches ordinary sglang.launch_server, no mask override/custom architecture; HF alternative loads standard Qwen class and scores self.model(input_ids=inputs).logits[0,-1]. Read twice repeats input; does not change attention mask.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/PelaAI/KnowLine-0.8B-Gen1/blob/main/knowline_engine.py#L76
- https://huggingface.co/PelaAI/KnowLine-0.8B-Gen1/blob/main/knowline_server.py#L896

## KnowLine 2B Gen1
causal (ordinary Qwen SGLang/HF inference; source-based inference)

Shipped KnowLine engine launches ordinary sglang.launch_server, no mask override/custom architecture; HF alternative loads standard Qwen class and scores self.model(input_ids=inputs).logits[0,-1]. Read twice repeats input; does not change attention mask.

Leaderboard base: Qwen/Qwen3.5-2B

- https://huggingface.co/PelaAI/KnowLine-2B-Gen1/blob/main/knowline_engine.py#L76
- https://huggingface.co/PelaAI/KnowLine-2B-Gen1/blob/main/knowline_server.py#L896

## PKU-KCL-Meteor-0.8B
full-bidirectional in full-attention layers; causal Gated DeltaNet

decision_config.json attention_mode=noncausal_full_attention. meteor/model.py _bidirectional_prompt explicitly says full-attention whole prompt and Gated DeltaNet stays causal.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B/blob/main/decision_config.json
- https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B/blob/main/meteor/model.py#L71

## PKU-KCL-Meteor-2B
full-bidirectional in full-attention layers; causal Gated DeltaNet

decision_config.json attention_mode=noncausal_full_attention. meteor/model.py _bidirectional_prompt explicitly says full-attention whole prompt and Gated DeltaNet stays causal.

Leaderboard base: Qwen/Qwen3.5-2B

- https://huggingface.co/chiranko/PKU-KCL-Meteor-2B/blob/main/decision_config.json
- https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B/blob/main/meteor/model.py#L71

## Saif 0.8 Lightening v0.1 (0.8B)
segmented bidirectional prefix + isolated question branches; not sequential block-bidirectional

saif_v2_config.json attention=bidir, single_pass=true; singlepass.py segmented uses is_causal=False, prefix attends itself; question branch attends shared prefix and itself, other question branches excluded. Native DeltaNet remains causal/branch replay.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/GyrTechnology/saif-0.8-lightening/blob/main/saif_v2_config.json
- https://huggingface.co/GyrTechnology/saif-0.8-lightening/blob/main/singlepass.py#L19

## Sieve 0.8B
causal

Option-fork scalar runtime loads stock AutoModelForCausalLM at L92, state prefill and option continuation with 2D padding mask at L151/L173. No bidirectional mask replacement.

Leaderboard base: Qwen/Qwen3.5-0.8B-Base

- https://github.com/sthanika-ai/Sieve/blob/main/sieve/fork/model.py#L92
- https://github.com/sthanika-ai/Sieve/blob/main/sieve/fork/model.py#L151

## Sieve 2B
causal

Option-fork scalar runtime loads stock AutoModelForCausalLM at L92, state prefill and option continuation with 2D padding mask at L151/L173. No bidirectional mask replacement.

Leaderboard base: Qwen/Qwen3.5-2B-Base

- https://github.com/sthanika-ai/Sieve/blob/main/sieve/fork/model.py#L92
- https://github.com/sthanika-ai/Sieve/blob/main/sieve/fork/model.py#L151

## Sieve 2B v3
causal

New pointer design (different from old 2B option-fork scalar); sieve/load.py identifies pointer; sieve/model.py runs stock causal backbone with prefix-cache or full-row paths and ordinary padding masks, no noncausal patch.

Leaderboard base: Qwen/Qwen3.5-2B

- https://github.com/sthanika-ai/Sieve/blob/main/sieve/load.py
- https://github.com/sthanika-ai/Sieve/blob/main/sieve/model.py

## StartLux-Decision v1.1 0.8B
full-prompt bidirectional in full-attention layers; causal DeltaNet

Both author cards explicitly: full-attention layers one in four attend later prompt tokens; linear layers stay causal; config jev_prompt_attention=bidirectional.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant/blob/main/README.md#L26
- https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant/blob/main/README.md#L26

## StartLux-Decision v1.1 2B
full-prompt bidirectional in full-attention layers; causal DeltaNet

Both author cards explicitly: full-attention layers one in four attend later prompt tokens; linear layers stay causal; config jev_prompt_attention=bidirectional.

Leaderboard base: Qwen/Qwen3.5-2B

- https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant/blob/main/README.md#L26
- https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant/blob/main/README.md#L26

## Strands Decider 2B
causal

modeling.py docstring explicitly explains keeping attention causal; uses stock Qwen3_5ForCausalLM text torso.

Leaderboard base: Qwen/Qwen3.5-2B-Base

- https://github.com/strands-labs/strands-decider/blob/main/src/strands_decider/modeling.py#L10

## Valen-0.8B
causal backbone; trained Mixer decision head

config.json qwen_execution=shared_state. compiler.py _compile_shared docstring: This is ordinary causal attention. Instantiates stock Qwen3_5Model and passes 2D padding mask, no mask replacement.

Leaderboard base: Qwen/Qwen3.5-0.8B

- https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/config.json
- https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/compiler.py#L180
- https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/modeling_valen.py#L31

## Valen-2B
causal backbone; trained Mixer decision head

config.json qwen_execution=shared_state. compiler.py _compile_shared docstring: This is ordinary causal attention. Instantiates stock Qwen3_5Model and passes 2D padding mask, no mask replacement.

Leaderboard base: Qwen/Qwen3.5-2B

- https://huggingface.co/Valen-Team/Valen-2B/blob/main/config.json
- https://huggingface.co/Valen-Team/Valen-2B/blob/main/compiler.py#L180
- https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/modeling_valen.py#L31

## vera-core-2
full-prompt bidirectional in full-attention layers; causal DeltaNet

Author card L30-31 whole prompt in full-attention layers, Gated DeltaNet remains causal; L176 is_causal:false for full-attention runtime.

Leaderboard base: Qwen/Qwen3.5-2B

- https://huggingface.co/scar-ai/vera-core-2/blob/main/README.md#L30
- https://huggingface.co/scar-ai/vera-core-2/blob/main/README.md#L176

## Not deeply verified in this subtask
jpt-0.8b, tev1-0.8b, sifr-0.8b-v3.1, harsha, onejev1.6-0.8b, latentmesh-blitz-2b-v1, saina-helm-2-v2, saina-helm-2-omni, wyat-1-800m, jdi-ptr14-0.8b, joey-1
