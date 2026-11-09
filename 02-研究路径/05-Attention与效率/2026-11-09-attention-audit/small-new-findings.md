# New small-model attention findings
Read-only source audit on 2026-11-09. Current published runtime; no weights/model/tokenizer/tests executed.

## glide-0.8b
full-bidirectional in full-attention layers; native causal linear-attention layers

decision_config.json attention_mode=noncausal_full_attention. glide/model.py attention_masks returns full_attention padding-only mask; docstring explicitly no causal mask.

- https://huggingface.co/fastino/GLiDE-0.8B/blob/main/decision_config.json
- https://huggingface.co/fastino/GLiDE-0.8B/blob/main/glide/model.py#L45

## valen-0.8b-r2
causal backbone; trained Mixer decision head

config.json qwen_execution=shared_state. compiler.py _compile_shared docstring: This is ordinary causal attention. Instantiates stock Qwen3_5Model and passes 2D padding mask, no mask replacement.

- https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/config.json
- https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/compiler.py#L180
- https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/modeling_valen.py#L31

## knowline-0.8b-gen1
causal (ordinary Qwen SGLang/HF inference; source-based inference)

Shipped KnowLine engine launches ordinary sglang.launch_server, no mask override/custom architecture; HF alternative loads standard Qwen class and scores self.model(input_ids=inputs).logits[0,-1]. Read twice repeats input; does not change attention mask.

- https://huggingface.co/PelaAI/KnowLine-0.8B-Gen1/blob/main/knowline_engine.py#L76
- https://huggingface.co/PelaAI/KnowLine-0.8B-Gen1/blob/main/knowline_server.py#L896

## pku-kcl-meteor-0.8b
full-bidirectional in full-attention layers; causal Gated DeltaNet

decision_config.json attention_mode=noncausal_full_attention. meteor/model.py _bidirectional_prompt explicitly says full-attention whole prompt and Gated DeltaNet stays causal.

- https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B/blob/main/decision_config.json
- https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B/blob/main/meteor/model.py#L71

## glide-2b
full-bidirectional in full-attention layers; native causal linear-attention layers

decision_config.json attention_mode=noncausal_full_attention. glide/model.py attention_masks returns full_attention padding-only mask; docstring explicitly no causal mask.

- https://huggingface.co/fastino/GLiDE-2B/blob/main/decision_config.json
- https://huggingface.co/fastino/GLiDE-0.8B/blob/main/glide/model.py#L45

## valen-2b-r2
causal backbone; trained Mixer decision head

config.json qwen_execution=shared_state. compiler.py _compile_shared docstring: This is ordinary causal attention. Instantiates stock Qwen3_5Model and passes 2D padding mask, no mask replacement.

- https://huggingface.co/Valen-Team/Valen-2B/blob/main/config.json
- https://huggingface.co/Valen-Team/Valen-2B/blob/main/compiler.py#L180
- https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/modeling_valen.py#L31

## knowline-2b-gen1
causal (ordinary Qwen SGLang/HF inference; source-based inference)

Shipped KnowLine engine launches ordinary sglang.launch_server, no mask override/custom architecture; HF alternative loads standard Qwen class and scores self.model(input_ids=inputs).logits[0,-1]. Read twice repeats input; does not change attention mask.

- https://huggingface.co/PelaAI/KnowLine-2B-Gen1/blob/main/knowline_engine.py#L76
- https://huggingface.co/PelaAI/KnowLine-2B-Gen1/blob/main/knowline_server.py#L896

## pku-kcl-meteor-2b
full-bidirectional in full-attention layers; causal Gated DeltaNet

decision_config.json attention_mode=noncausal_full_attention. meteor/model.py _bidirectional_prompt explicitly says full-attention whole prompt and Gated DeltaNet stays causal.

- https://huggingface.co/chiranko/PKU-KCL-Meteor-2B/blob/main/decision_config.json
- https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B/blob/main/meteor/model.py#L71

## jev-style-0.8b-v3
causal

jev_style_decision.py L541-542 explicitly model is causal. L652 calls stock model.model(input_ids=ids,use_cache=False) without mask override. Chunked SDPA helper preserves causal mask (L500-514).

- https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3/blob/main/jev_style_decision.py#L541
- https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3/blob/main/jev_style_decision.py#L652

## jev-style-2b-v4
block-bidirectional in full-attention layers; causal Gated DeltaNet

README L375-377: blocks of <=2048 tokens; each block attends all prior blocks and itself with no internal causal mask. Gated DeltaNet remains recurrent. Different from 0.8B v3.

- https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4/blob/main/README.md#L375

## adjudex-0.8b
full-bidirectional in full-attention layers

README L36: Qwen3_5Model with noncausal attention in full-attention layers, last-token pooling. decision_config.json attention_mode=noncausal_full_attention.

- https://huggingface.co/newsbang/adjudex-0.8b/blob/main/README.md#L36
- https://huggingface.co/newsbang/adjudex-0.8b/blob/main/decision_config.json

## saif-0.8-lightening
segmented bidirectional prefix + isolated question branches; not sequential block-bidirectional

saif_v2_config.json attention=bidir, single_pass=true; singlepass.py segmented uses is_causal=False, prefix attends itself; question branch attends shared prefix and itself, other question branches excluded. Native DeltaNet remains causal/branch replay.

- https://huggingface.co/GyrTechnology/saif-0.8-lightening/blob/main/saif_v2_config.json
- https://huggingface.co/GyrTechnology/saif-0.8-lightening/blob/main/singlepass.py#L19
