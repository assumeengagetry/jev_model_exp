# Supplemental 4B attention audit
Checked 2026-11-09; source reading only, no model/tokenizer/test execution.

## OneJev1.6-4B
Qwen/Qwen3.5-4B — causal
Flide OneJev format defaults to causal when neither readout.json nor modeling_overrides exists; this HF release has neither. Current config and HF file inventory checked.
https://github.com/OmniJev/Flide/blob/main/plugins/flide_onejev/format.py#L234

## StartLux-Decision v1.1 4B
Qwen/Qwen3.5-4B — full-bidi (full_attention layers only)
README: prompt tokens attend to later prompt tokens in full_attention layers; Gated DeltaNet stays causal. config declares jev_prompt_attention=bidirectional.
https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant/blob/main/README.md#L26

## CoCo-Decision-4B 2.4
Qwen/Qwen3.5-4B — full-bidi (full_attention layers only)
README and reference code open full_attention layers to whole prompt; padding masked; no causal mask.
https://huggingface.co/corners-ai/CoCo-Decision-2.4-4B/blob/main/README.md#L54

## Nari Decision 4B v3.2
Qwen/Qwen3.5-4B — causal backbone + noncausal decision head
modeling_sysone.py passes ordinary 2D attention_mask.long() to unmodified backbone. Decision head separately uses two layers of option self-attention and cross-attention over sequence.
https://huggingface.co/blackmallangcow/nari-decision-4b/blob/main/modeling_sysone.py#L318

## Mica v0.5 4B
Qwen/Qwen3.5-4B — unknown for scored revision; current fast runtime causal
Pinned source uses ordinary llama.cpp direct backend; current v0.5 prefill_engine.py has is_causal=n>1. No explicit pinned mask setting verified.
https://github.com/akivet/Mica/blob/main/v0.5/serving/lib/prefill_engine.py#L145

## Jeffan-4B-2010
Qwen/Qwen3.5-4B — full-bidi (full_attention layers only)
README explicitly says every prompt token sees every other in full_attention layers; Gated DeltaNet stays causal; example passes mask dict.
https://huggingface.co/tattrongvu/Jeffan-4B-2010/blob/main/README.md#L26

## KnowLine 4B Gen5.5
Qwen/Qwen3.5-4B — causal
README explains read-twice under causal attention; HF backend uses ordinary self.model(input_ids=inputs).logits.
https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5/blob/main/knowline_server.py#L242

## Valen-4B
Qwen/Qwen3.5-4B — causal
compiler._compile_shared docstring explicitly ordinary causal attention. runtime forwards unmodified 2D masks to Qwen3_5Model.
https://huggingface.co/Valen-Team/Valen-4B/blob/main/compiler.py#L180

## Verdin-4B v1.1
Qwen/Qwen3.5-4B — state-block-bidi, question branches causal
README: state bidirectional within 8192-token blocks in 8 full_attention layers; question branches and 24 Gated DeltaNet layers causal.
https://huggingface.co/victorbennett/Verdin-4B-v1.1/blob/main/README.md#L140

## Amber Hybrid 4B
Qwen/Qwen3.5-4B — mixed models: causal Qwen decoder + bidirectional ModernBERT encoder
hybrid_engine.py Decoder retains original Qwen causal mask and explicit causal padding comment; combines probabilities with separate ModernBERT encoder. Not a Qwen mask conversion.
https://huggingface.co/MLukauskas/amber-hybrid-4b/blob/main/hybrid_engine.py#L159

## Amber 4B
Qwen/Qwen3.5-4B — causal
decoder_engine.py explicitly says right padding without mask exact for causal model; uses ordinary self.model(input_ids=...).
https://huggingface.co/MLukauskas/amber-4b/blob/main/decoder_engine.py#L151

## Saif 4B
Qwen/Qwen3.5-4B — block-bidi with question isolation
config attention=bidir. saif_v2_core attention_mask opens prefix-prefix, question-prefix and each question-self; distinct question blocks do not see each other. singlepass implements same segmented is_causal=False.
https://huggingface.co/GyrTechnology/saif-4b/blob/main/saif_v2_core.py#L118

## Piccolo 4B
Qwen/Qwen3.5-4B — full-bidi (full_attention layers only)
piccolo_inference.py descriptor requires attention_mode=bidirectional_prompt; hooks full_attention with all-zero additive LxL mask; GDN native.
https://huggingface.co/agentscope-ai/Piccolo-4B/blob/main/piccolo_inference.py#L125

## PurpleMIST-Mini-2.0
Qwen/Qwen3.5-4B-Base — two-block mask: causal prefix + bidirectional question region
README: eight full_attention layers prefix is causal and sees no questions; each question token sees whole prefix and every question; GDN causal.
https://huggingface.co/olaverse/PurpleMIST-Mini-2.0/blob/main/README.md#L61

## Cerno 4B v1.2
Qwen/Qwen3.5-4B — block-bidi
README: state and each question+options form blocks; within-block bidirectional, across blocks causal, only full_attention layers.
https://huggingface.co/apertixai/cerno-4b-v1.2/blob/main/README.md#L57

## BaseDecision-Dyno (4B)
Qwen/Qwen3.5-4B — full-bidi (full_attention layers only)
README attention table says bidirectional full_attention layers; native recurrence in linear_attention layers.
https://huggingface.co/onlyaady/BaseDecision-Dyno/blob/main/README.md#L42

## Senova-4B v4
Qwen/Qwen3.5-4B — unknown
README specifies read-twice and next-token distribution but no explicit mask; source server not inspected.
https://huggingface.co/WangSenQWQ/Senova-4B-v4/blob/main/README.md

## RSI-Jev v5.0 VL 3B
Qwen/Qwen3.5-4B-Base — causal (author documentation)
README explains option pooling issue caused by previous-option information in causal model. Actual served model is early-exit 3.25B, first 20 of Qwen3.5-4B-Base 32 layers, not full4B.
https://huggingface.co/shgao/rsi-jev-v5.0-vl-3b/blob/main/README.md#L101

## ezjev 4B s2
Qwen/Qwen3.5-4B — causal (documented ordinary vLLM serving)
README uses stock vLLM serve + llm2jev; no custom model/mask/plugin, ordinary Qwen3.5 serving. This is documented serving inference rather than direct mask source inspection.
https://huggingface.co/everettjf/ezjev-4b-s2/blob/main/README.md#L42

## Kas 4B
Qwen/Qwen3-4B — causal
engine loads ordinary Qwen3 AutoModelForCausalLM and forwards input_ids, position_ids, past_key_values/use_cache without overriding mask.
https://github.com/kashavpiya/kas-4b/blob/main/kas_engine/engine.py#L240
