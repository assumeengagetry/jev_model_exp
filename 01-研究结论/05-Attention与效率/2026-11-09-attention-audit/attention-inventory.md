# 当前 Jev Index：Qwen 相关模型 attention 清单

读取日期：2026-11-09。按底座名称分组。165条直接 Qwen 底座，加1条MiMo蒸馏Qwen记录。
“未确认”表示本次没有足够源码或发布配置证据；不能按名字、ForCausalLM类名或榜单autoregressive字段推断。
表中分块双向不代表相同分块规则；多数Qwen3.5/3.8双向修改只作用于full-attention层。
8B CLM的作者公开pooling路径为causal，已核对vLLM v0.10.2、v0.11.0及当前源码；未声称还原其榜单精确服务版本。

## Qwen/Qwen2.5-1.5B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) | causal | [证据1](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/engine_torch.py#L93) |

## Qwen/Qwen3-0.6B-Base

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [vLLM-SR Decision 2.0 Kai 0.6B](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B) | causal | [证据1](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/fast.py#L542)、[证据2](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/_vendor/dev2model/decision_model.py#L232) |
| [Bosun v3.1 0.6B](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b) | causal | [证据1](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/modeling_bosun.py#L119)、[证据2](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/modeling_bosun.py#L216) |

## Qwen/Qwen3-1.7B-Base

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [Bosun v3.1 1.7B](https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b) | causal | [证据1](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/modeling_bosun.py#L119)、[证据2](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b/blob/main/modeling_bosun.py#L216) |

## Qwen/Qwen3-4B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [Intelif Qwen3-4B](https://huggingface.co/UserMoonlight/intelif-qwen3-4b) | 未确认（详见证据） | [证据1](https://huggingface.co/UserMoonlight/intelif-qwen3-4b) |
| [Kas 4B](https://huggingface.co/kpiya/kas-4b) | causal | [证据1](https://github.com/kashavpiya/kas-4b/blob/main/kas_engine/engine.py#L240) |

## Qwen/Qwen3-4B-Instruct-2507

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [openvons](https://github.com/genai-craft/openvons) | 未确认（详见证据） | [证据1](https://github.com/genai-craft/openvons) |
| [mini-jev](https://github.com/r-ms/mini-jev) | causal | [证据1](https://github.com/r-ms/mini-jev/blob/main/minijev/engine.py#L36) |

## Qwen/Qwen3-8B-Base

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) | causal（作者公开路径） | [证据1](https://github.com/Contrastive-LM/CLM/blob/main/serve_qwen3_8b.sh)、[证据2](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/qwen3.py#L180) |

## Qwen/Qwen3-Omni-30B-A3B-Instruct

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [Cloudflare clef-omni (30B-A3B)](https://huggingface.co/Cloudflare/clef-omni) | causal 骨干＋额外全局 attention head | [证据1](https://huggingface.co/Cloudflare/clef-omni/blob/main/joint_schema_model.py#L591)、[证据2](https://huggingface.co/Cloudflare/clef-omni/blob/main/joint_schema_model.py#L380) |

## Qwen/Qwen3.5-0.8B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [vLLM-SR Decision 2.0 Eos 0.8B](https://huggingface.co/vllm-sr/Decision-2.0-Eos-0.8B) | causal | [证据1](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/fast.py#L542)、[证据2](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/_vendor/dev2model/decision_model.py#L232) |
| [Jev-Style 0.8B v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) | causal | [证据1](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3/blob/main/jev_style_decision.py#L541)、[证据2](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3/blob/main/jev_style_decision.py#L652) |
| [Sifr 0.8B v3.1](https://huggingface.co/mohamedlotfy50/sifr-0.8b-v3) | 未确认（详见证据） | 本次未确认 |
| [vLLM-SR d3-lite (0.8B)](https://huggingface.co/vllm-sr/d3-lite) | full-attention 层整段双向 | [证据1](https://huggingface.co/vllm-sr/d3-lite/blob/v3.1.0/decision_config.json) |
| [vLLM-SR d3-edge (0.6B)](https://huggingface.co/vllm-sr/d3-edge) | full-attention 层整段双向 | [证据1](https://huggingface.co/vllm-sr/d3-edge/blob/v3.1.0/decision_config.json) |
| [jiwo 0.8B v2](https://huggingface.co/eljiwo/jiwo-0.8b) | causal | [证据1](https://github.com/jiwidi/jiwo/blob/main/src/jiwo/model.py#L320) |
| [OneJev1.6-0.8B](https://huggingface.co/OmniJev/OneJev1.6-0.8B) | 未确认（详见证据） | 本次未确认 |
| [StartLux-Decision v1.1 0.8B](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant) | full-attention 层整段双向 | [证据1](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant/blob/main/README.md#L26)、[证据2](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant/blob/main/README.md#L26) |
| [Saina Helm 2.0 v2](https://huggingface.co/run-saina/saina-helm-2-0.8b) | 未确认（详见证据） | 本次未确认 |
| [Saina Helm 2 Omni (0.8B)](https://huggingface.co/run-saina/saina-helm-2-omni-0.8b) | 未确认（详见证据） | 本次未确认 |
| [jdi-ptr14-0.8b](https://huggingface.co/AnamK/jdi-ptr14-0.8b) | 未确认（详见证据） | 本次未确认 |
| [crystal-0.8b](https://huggingface.co/Kyunghwan-kim/crystal-0.8b) | full-attention 层整段双向 | [证据1](https://huggingface.co/Kyunghwan-kim/crystal-0.8b/blob/main/crystal_config.json)、[证据2](https://huggingface.co/Kyunghwan-kim/crystal-0.8b/blob/main/crystal/modeling.py#L193) |
| [Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B) | causal 骨干＋独立 Mixer head | [证据1](https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/config.json)、[证据2](https://huggingface.co/Valen-Team/Valen-0.8B/blob/main/compiler.py#L180) |
| [KnowLine 0.8B Gen1](https://huggingface.co/PelaAI/KnowLine-0.8B-Gen1) | causal | [证据1](https://huggingface.co/PelaAI/KnowLine-0.8B-Gen1/blob/main/knowline_engine.py#L76)、[证据2](https://huggingface.co/PelaAI/KnowLine-0.8B-Gen1/blob/main/knowline_server.py#L896) |
| [PKU-KCL-Meteor-0.8B](https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B) | full-attention 层整段双向 | [证据1](https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B/blob/main/decision_config.json)、[证据2](https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B/blob/main/meteor/model.py#L71) |
| [Adjudex-0.8B](https://huggingface.co/newsbang/adjudex-0.8b) | full-attention 层整段双向 | [证据1](https://huggingface.co/newsbang/adjudex-0.8b/blob/main/README.md#L36)、[证据2](https://huggingface.co/newsbang/adjudex-0.8b/blob/main/decision_config.json) |
| [Saif 0.8 Lightening v0.1 (0.8B)](https://huggingface.co/GyrTechnology/saif-0.8-lightening) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/GyrTechnology/saif-0.8-lightening/blob/main/saif_v2_config.json)、[证据2](https://huggingface.co/GyrTechnology/saif-0.8-lightening/blob/main/singlepass.py#L19) |
| [GLiDE 0.8B](https://huggingface.co/fastino/GLiDE-0.8B) | full-attention 层整段双向 | [证据1](https://huggingface.co/fastino/GLiDE-0.8B/blob/main/decision_config.json)、[证据2](https://huggingface.co/fastino/GLiDE-0.8B/blob/main/glide/model.py#L45) |

## Qwen/Qwen3.5-0.8B-Base

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [JPT-0.8B](https://huggingface.co/kirp/jpt-0.8b) | 未确认（详见证据） | 本次未确认 |
| [Tev1-0.8B-experimental](https://huggingface.co/togethercomputer/Tev1-0.8B-experimental) | 未确认（详见证据） | 本次未确认 |
| [Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) | causal | [证据1](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/main/README.md#L28)、[证据2](https://huggingface.co/internlm/Intern-Decision-2B/blob/main/README.md#L28) |
| [Kev 0.8B r15](https://huggingface.co/jaredpalmer/kev-0.8b) | causal | [证据1](https://huggingface.co/jaredpalmer/kev-0.8b/blob/main/README.md#L77)、[证据2](https://github.com/jaredpalmer/kev/blob/main/kev/model.py#L257) |
| [Sieve 0.8B](https://huggingface.co/sthanika-ai/Sieve-0.8B) | causal | [证据1](https://github.com/sthanika-ai/Sieve/blob/main/sieve/fork/model.py#L92)、[证据2](https://github.com/sthanika-ai/Sieve/blob/main/sieve/fork/model.py#L151) |
| [wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m) | 未确认（详见证据） | 本次未确认 |

## Qwen/Qwen3.5-27B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [baselabs mammoth-1 (27B)](https://huggingface.co/baselabs/mammoth-1) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/baselabs/mammoth-1/blob/9d1962f193e48a68cf0f3598dc7fb4462fd23ad9/config.json#L146)、[证据2](https://huggingface.co/baselabs/mammoth-1/blob/9d1962f193e48a68cf0f3598dc7fb4462fd23ad9/serve.patch#L250) |
| [Piccolo 27B](https://huggingface.co/agentscope-ai/Piccolo-27B) | 作者称双向；具体 mask 未核 | [证据1](https://huggingface.co/agentscope-ai/Piccolo-27B) |

## Qwen/Qwen3.5-2B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [vLLM-SR Decision 2.0 Sol 2B](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B) | causal | [证据1](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/fast.py#L542)、[证据2](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/_vendor/dev2model/decision_model.py#L232) |
| [vLLM-SR d3-nano (2B)](https://huggingface.co/vllm-sr/d3-nano) | full-attention 层整段双向 | [证据1](https://huggingface.co/vllm-sr/d3-nano/blob/v3.1.0/decision_config.json) |
| [StartLux-Decision v1.1 2B](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant) | full-attention 层整段双向 | [证据1](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant/blob/main/README.md#L26)、[证据2](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant/blob/main/README.md#L26) |
| [LatentMesh Blitz 2B v1](https://huggingface.co/LatentMesh-Labs/latentmesh-blitz-2b-v1) | 未确认（详见证据） | 本次未确认 |
| [Sieve 2B v3](https://huggingface.co/sthanika-ai/Sieve-2B-v3) | causal | [证据1](https://github.com/sthanika-ai/Sieve/blob/main/sieve/load.py)、[证据2](https://github.com/sthanika-ai/Sieve/blob/main/sieve/model.py) |
| [vera-core-2](https://huggingface.co/scar-ai/vera-core-2) | full-attention 层整段双向 | [证据1](https://huggingface.co/scar-ai/vera-core-2/blob/main/README.md#L30)、[证据2](https://huggingface.co/scar-ai/vera-core-2/blob/main/README.md#L176) |
| [Valen-2B](https://huggingface.co/Valen-Team/Valen-2B) | causal 骨干＋独立 Mixer head | [证据1](https://huggingface.co/Valen-Team/Valen-2B/blob/main/config.json)、[证据2](https://huggingface.co/Valen-Team/Valen-2B/blob/main/compiler.py#L180) |
| [Jev-Style 2B v4](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4/blob/main/README.md#L375) |
| [baselabs joey-1 (2B)](https://huggingface.co/baselabs/joey-1) | 未确认（详见证据） | 本次未确认 |
| [KnowLine 2B Gen1](https://huggingface.co/PelaAI/KnowLine-2B-Gen1) | causal | [证据1](https://huggingface.co/PelaAI/KnowLine-2B-Gen1/blob/main/knowline_engine.py#L76)、[证据2](https://huggingface.co/PelaAI/KnowLine-2B-Gen1/blob/main/knowline_server.py#L896) |
| [PKU-KCL-Meteor-2B](https://huggingface.co/chiranko/PKU-KCL-Meteor-2B) | full-attention 层整段双向 | [证据1](https://huggingface.co/chiranko/PKU-KCL-Meteor-2B/blob/main/decision_config.json)、[证据2](https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B/blob/main/meteor/model.py#L71) |
| [GLiDE 2B](https://huggingface.co/fastino/GLiDE-2B) | full-attention 层整段双向 | [证据1](https://huggingface.co/fastino/GLiDE-2B/blob/main/decision_config.json)、[证据2](https://huggingface.co/fastino/GLiDE-0.8B/blob/main/glide/model.py#L45) |

## Qwen/Qwen3.5-2B-Base

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [Sieve 2B](https://huggingface.co/sthanika-ai/sieve-2b) | causal | [证据1](https://github.com/sthanika-ai/Sieve/blob/main/sieve/fork/model.py#L92)、[证据2](https://github.com/sthanika-ai/Sieve/blob/main/sieve/fork/model.py#L151) |
| [Decider 2B](https://huggingface.co/Mapika/decider-2b) | causal | [证据1](https://github.com/Mapika/decider/blob/main/decider/model.py#L13)、[证据2](https://github.com/Mapika/decider/blob/main/decider/model.py#L26) |
| [Intern-Decision-2B](https://huggingface.co/internlm/Intern-Decision-2B) | causal | [证据1](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/main/README.md#L28)、[证据2](https://huggingface.co/internlm/Intern-Decision-2B/blob/main/README.md#L28) |
| [Strands Decider 2B](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) | causal | [证据1](https://github.com/strands-labs/strands-decider/blob/main/src/strands_decider/modeling.py#L10) |

## Qwen/Qwen3.5-35B-A3B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [JEV-35B](https://huggingface.co/autotrust/JEV-35B) | 未确认（详见证据） | 本次未确认 |
| [KnowLine 35B-A3B Gen1.5](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1.5) | causal | [证据1](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1.5)、[证据2](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1.5/blob/main/knowline_engine.py) |
| [JPT-35B-A3B](https://huggingface.co/kirp/jpt-35b-a3b) | causal | [证据1](https://huggingface.co/kirp/jpt-35b-a3b) |

## Qwen/Qwen3.5-35B-A3B-Base

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [Decider 35B-A3B](https://huggingface.co/Mapika/decider-35b-a3b-nvfp4) | causal | [证据1](https://huggingface.co/Mapika/decider-35b-a3b-nvfp4)、[证据2](https://github.com/Mapika/decider/blob/main/decider/engine.py) |

## Qwen/Qwen3.5-4B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [JevK5 v0.3](https://huggingface.co/alibiserikbay/JevK5) | 未确认（详见证据） | [证据1](https://huggingface.co/alibiserikbay/JevK5) |
| [Ryotide Qwen v0.3.1](https://github.com/csabag/ryotide) | causal | [证据1](https://github.com/csabag/ryotide/blob/main/src/ryotide/torch_backend.py#L285) |
| [Aplomb 1](https://huggingface.co/empiriolabsai/aplomb-1) | 未确认（详见证据） | [证据1](https://huggingface.co/empiriolabsai/aplomb-1) |
| [Sieve 4B](https://huggingface.co/sthanika-ai/Sieve-4B) | causal | [证据1](https://github.com/sthanika-ai/Sieve/blob/main/sieve/codes/shared_prefix.py#L255) |
| [jiwo 4B v2](https://huggingface.co/eljiwo/jiwo-4b) | causal | [证据1](https://github.com/jiwidi/jiwo/blob/main/src/jiwo/model.py#L320) |
| [Cam1](https://huggingface.co/cbuk10001/Cam1) | 未确认（详见证据） | [证据1](https://huggingface.co/cbuk10001/Cam1) |
| [vLLM-SR d3-mini (4B)](https://huggingface.co/vllm-sr/d3-mini) | full-attention 层整段双向 | [证据1](https://huggingface.co/vllm-sr/d3-mini/blob/v3.1.0/decision_config.json) |
| [FlyMy.AI Decision 4B v1.6](https://huggingface.co/flymy-ai/decision-4b-v1.6) | causal | [证据1](https://huggingface.co/flymy-ai/decision-4b-v1.6/blob/main/model.py#L13) |
| [Perplexity Decider Flash v1.2 (4B)](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) | full-attention 层整段双向 | [证据1](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/main/source/src/pplx_decider/model.py#L65) |
| [OneJev1.6-4B](https://huggingface.co/OmniJev/OneJev1.6-4B) | causal | [证据1](https://github.com/OmniJev/Flide/blob/main/plugins/flide_onejev/format.py#L234) |
| [StartLux-Decision v1.1 4B](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant) | full-attention 层整段双向 | [证据1](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant/blob/main/README.md#L26) |
| [CoCo-Decision-4B 2.4](https://huggingface.co/corners-ai/CoCo-Decision-2.4-4B) | full-attention 层整段双向 | [证据1](https://huggingface.co/corners-ai/CoCo-Decision-2.4-4B/blob/main/README.md#L54) |
| [Nari Decision 4B v3.2](https://huggingface.co/blackmallangcow/nari-decision-4b) | causal 骨干＋额外全局 attention head | [证据1](https://huggingface.co/blackmallangcow/nari-decision-4b/blob/main/modeling_sysone.py#L318) |
| [Mica v0.5 4B](https://huggingface.co/sky7350/Mica-v0.5-4B) | 未确认（详见证据） | [证据1](https://github.com/akivet/Mica/blob/main/v0.5/serving/lib/prefill_engine.py#L145) |
| [Jeffan-4B-2010](https://huggingface.co/tattrongvu/Jeffan-4B-2010) | full-attention 层整段双向 | [证据1](https://huggingface.co/tattrongvu/Jeffan-4B-2010/blob/main/README.md#L26) |
| [KnowLine 4B Gen5.5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5) | causal | [证据1](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5/blob/main/knowline_server.py#L242) |
| [Valen-4B](https://huggingface.co/Valen-Team/Valen-4B) | causal | [证据1](https://huggingface.co/Valen-Team/Valen-4B/blob/main/compiler.py#L180) |
| [Verdin-4B v1.1](https://huggingface.co/victorbennett/Verdin-4B-v1.1) | 部分双向：场景双向，问题 causal | [证据1](https://huggingface.co/victorbennett/Verdin-4B-v1.1/blob/main/README.md#L140) |
| [Amber Hybrid 4B](https://huggingface.co/MLukauskas/amber-hybrid-4b) | 混合：causal Qwen＋双向 ModernBERT | [证据1](https://huggingface.co/MLukauskas/amber-hybrid-4b/blob/main/hybrid_engine.py#L159) |
| [Amber 4B](https://huggingface.co/MLukauskas/amber-4b) | causal | [证据1](https://huggingface.co/MLukauskas/amber-4b/blob/main/decoder_engine.py#L151) |
| [Saif 4B](https://huggingface.co/GyrTechnology/saif-4b) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/GyrTechnology/saif-4b/blob/main/saif_v2_core.py#L118) |
| [Piccolo 4B](https://huggingface.co/agentscope-ai/Piccolo-4B) | full-attention 层整段双向 | [证据1](https://huggingface.co/agentscope-ai/Piccolo-4B/blob/main/piccolo_inference.py#L125) |
| [Cerno 4B v1.2](https://huggingface.co/apertixai/cerno-4b-v1.2) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/apertixai/cerno-4b-v1.2/blob/main/README.md#L57) |
| [BaseDecision-Dyno (4B)](https://huggingface.co/onlyaady/BaseDecision-Dyno) | full-attention 层整段双向 | [证据1](https://huggingface.co/onlyaady/BaseDecision-Dyno/blob/main/README.md#L42) |
| [Senova-4B v4](https://huggingface.co/WangSenQWQ/Senova-4B-v4) | 未确认（详见证据） | [证据1](https://huggingface.co/WangSenQWQ/Senova-4B-v4/blob/main/README.md) |
| [ezjev 4B s2](https://huggingface.co/everettjf/ezjev-4b-s2) | causal | [证据1](https://huggingface.co/everettjf/ezjev-4b-s2/blob/main/README.md#L42) |

## Qwen/Qwen3.5-4B-Base

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [vLLM-SR Decision 2.0 Nox 4B](https://huggingface.co/vllm-sr/Decision-2.0-Nox-4B) | causal | [证据1](https://huggingface.co/vllm-sr/Decision-2.0-Nox-4B/blob/main/decision2/shared_ctx.py#L673) |
| [JPT-4B](https://huggingface.co/kirp/jpt-4b) | causal | [证据1](https://github.com/tic-top/llm2jev/blob/main/llm2jev/backends.py#L142) |
| [Hopper (G) 1.2](https://huggingface.co/HopitAI/hopper-g) | causal | [证据1](https://github.com/hopit-ai/hopper/blob/main/hopper_decisions/model.py#L61) |
| [Kev 4B r10](https://huggingface.co/jaredpalmer/kev-4b) | causal | [证据1](https://github.com/jaredpalmer/kev/blob/main/kev/model.py#L259) |
| [lev](https://huggingface.co/interfaze-ai/lev) | causal | [证据1](https://github.com/Abhinavexists/lev/blob/main/packages/lev/src/lev/model.py#L566) |
| [NeoHorse-Jev-4B](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B) | causal | [证据1](https://github.com/TokenRhythm/NeoHorse/blob/main/jev/package/src/neohorse_decision/_vendor/model.py#L142) |
| [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) | causal | [证据1](https://github.com/internlm/Intern-Decision/blob/main/src/inference/hf_backend.py#L105) |
| [Decider 4B](https://huggingface.co/Mapika/decider-4b) | causal | [证据1](https://github.com/Mapika/decider/blob/main/decider/engine.py#L3) |
| [Jobe](https://github.com/MantisShrimpdev/jobe) | 未确认（详见证据） | [证据1](https://github.com/MantisShrimpdev/jobe) |
| [open-jev (pngwn)](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer-v2b) | 未确认（详见证据） | [证据1](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer-v2b) |
| [Tev1-4B-experimental](https://huggingface.co/togethercomputer/Tev1-4B-experimental) | 未确认（详见证据） | [证据1](https://github.com/togethercomputer/tev1) |
| [Metask-Jev-4B](https://huggingface.co/wayfind/metask-jev-4b-policy-mix) | causal | [证据1](https://github.com/metask-ai/metask-jev/blob/main/inference/jev_scorer.py#L29) |
| [SemIf](https://github.com/TheoLeeCJ/openjev) | 未确认（详见证据） | [证据1](https://github.com/TheoLeeCJ/openjev) |
| [Saracura PT-BR 4B](https://huggingface.co/felhen-ai/saracura-ptbr-4b) | 未确认（详见证据） | [证据1](https://huggingface.co/felhen-ai/saracura-ptbr-4b) |
| [RSI-Jev v6.1 VL 4B](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b) | causal | [证据1](https://github.com/Shanghua-Gao/RSI-Jev/blob/main/rsijev/arch.py#L1035) |
| [PurpleMIST-Mini-2.0](https://huggingface.co/olaverse/PurpleMIST-Mini-2.0) | 部分双向：前缀 causal，问题区双向 | [证据1](https://huggingface.co/olaverse/PurpleMIST-Mini-2.0/blob/main/README.md#L61) |
| [RSI-Jev v5.0 VL 3B](https://huggingface.co/shgao/rsi-jev-v5.0-vl-3b) | causal | [证据1](https://huggingface.co/shgao/rsi-jev-v5.0-vl-3b/blob/main/README.md#L101) |

## Qwen/Qwen3.5-9B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [vLLM-SR Decision 2.0 Lux 9B](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B) | causal | [证据1](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B/blob/main/decision2/fast.py) |
| [Bespoke Nimble 9B v3](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3) | causal | [证据1](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index/blob/main/code/nimble/evaluation/decision_index_engine.py) |
| [JevK5 9B v0.3.3](https://huggingface.co/alibiserikbay/JevK5-9B) | causal | [证据1](https://github.com/allebee/jevk5/blob/main/jevk5/runtime.py) |
| [JEV-9B](https://huggingface.co/autotrust/JEV-9B) | causal | [证据1](https://huggingface.co/autotrust/JEV-9B#usage) |
| [Sieve 9B](https://huggingface.co/sthanika-ai/Sieve-9B) | causal | [证据1](https://github.com/sthanika-ai/Sieve/blob/main/sieve/model.py) |
| [Cloudflare clef-flash](https://huggingface.co/Cloudflare/clef-flash) | causal 骨干＋额外全局 attention head | [证据1](https://huggingface.co/Cloudflare/clef-flash/blob/main/joint_schema_model.py) |
| [Sieve 9B v2](https://huggingface.co/sthanika-ai/Sieve-9B-v2) | causal | [证据1](https://github.com/sthanika-ai/Sieve/blob/main/sieve/model.py) |
| [vLLM-SR d3-flash (9B)](https://huggingface.co/vllm-sr/d3-flash) | full-attention 层整段双向 | [证据1](https://huggingface.co/vllm-sr/d3-flash/blob/v3.1.0/decision_config.json) |
| [baselabs elephant-1 (9B)](https://huggingface.co/baselabs/elephant-1) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/baselabs/elephant-1/blob/main/config.json) |
| [OneJev1.6-9B](https://huggingface.co/OmniJev/OneJev1.6-9B) | causal | [证据1](https://github.com/OmniJev/Flide/blob/main/plugins/flide_onejev/format.py) |
| [frontier-infra Jebadiah 9B v2.1 (t1e)](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1) | causal | [证据1](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1/blob/main/scripts/jebadiah_model.py) |
| [StartLux-Decision v1.1 9B](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant) | full-attention 层整段双向 | [证据1](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant) |
| [Trio-Spark DI (9B)](https://huggingface.co/MachineFi/Trio-Spark-DI) | 未确认（详见证据） | [证据1](https://huggingface.co/MachineFi/Trio-Spark-DI) |
| [KnowLine 9B Gen1](https://huggingface.co/PelaAI/KnowLine-9B-Gen1) | causal | [证据1](https://huggingface.co/PelaAI/KnowLine-9B-Gen1) |
| [Seb-9B v2](https://huggingface.co/ironbcc/Seb-9B-v2) | causal | [证据1](https://huggingface.co/ironbcc/Seb-9B-v2) |
| [Piccolo 9B](https://huggingface.co/agentscope-ai/Piccolo-9B) | full-attention 层整段双向 | [证据1](https://huggingface.co/agentscope-ai/Piccolo-9B/blob/main/piccolo_inference.py) |
| [Jegue 0.2 (9B)](https://huggingface.co/fdemelo/jegue) | causal | [证据1](https://huggingface.co/fdemelo/jegue/blob/main/code/answer_slot.py) |
| [Cerno 9B v1.2](https://huggingface.co/apertixai/cerno-9b-v1.2) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/apertixai/cerno-9b-v1.2/blob/main/inference/third_party/decider/decider/model.py) |
| [Adjudex-9B v2](https://huggingface.co/newsbang/adjudex-9b-v2) | full-attention 层整段双向 | [证据1](https://huggingface.co/newsbang/adjudex-9b-v2/blob/main/source/src/autojev/model.py) |
| [GLiDE 9B](https://huggingface.co/fastino/GLiDE-9B) | full-attention 层整段双向 | [证据1](https://huggingface.co/fastino/GLiDE-9B/blob/main/glide/model.py) |

## Qwen/Qwen3.5-9B-Base

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [JPT-9B](https://huggingface.co/kirp/jpt-9b) | causal | [证据1](https://github.com/tic-top/llm2jev/blob/main/llm2jev/backends.py) |
| [Kev 9B v2](https://huggingface.co/jaredpalmer/kev-9b) | causal | [证据1](https://huggingface.co/jaredpalmer/kev-9b) |
| [Bespoke Nimble 9B v2](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) | causal | [证据1](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2/blob/main/inference.py) |
| [PurpleMIST-Flash-1.0](https://huggingface.co/olaverse/PurpleMIST-Flash-1.0) | causal | [证据1](https://huggingface.co/olaverse/PurpleMIST-Flash-1.0/blob/main/purplemist.py) |

## Qwen/Qwen3.6-27B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [Decider chat · Qwen3.6-27B](https://github.com/Mapika/decider) | 未确认（详见证据） | 本次未确认 |

## Qwen/Qwen3.6-35B-A3B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [Xor 1.2](https://huggingface.co/juspay/xor) | 未确认（详见证据） | 本次未确认 |
| [Nace.AI Drex v2.0 (35B-A3B)](https://huggingface.co/nace-ai/drex-v2.0) | causal | [证据1](https://huggingface.co/nace-ai/drex-v2.0/blob/main/code/kev/model.py#L259) |
| [Tern-35B-A3B](https://huggingface.co/matthewagi/Tern-35B-A3B) | 未确认（详见证据） | 本次未确认 |
| [Standard One 35B SH (v1)](https://huggingface.co/StandardThinking/StandardOne-35B-SH) | 未确认（详见证据） | 本次未确认 |
| [decisio 0.4.0 (Qwen3.6-35B-A3B)](https://github.com/aminry/decisio) | 未确认（详见证据） | 本次未确认 |

## Qwen/Qwen3.8-27B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [vLLM-SR Decision 2.0 Vega 27B](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B) | 未确认（详见证据） | 本次未确认 |
| [Kev 27B](https://huggingface.co/jaredpalmer/kev-27b) | causal | [证据1](https://huggingface.co/jaredpalmer/kev-27b)、[证据2](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L259-L262) |
| [simple-jev · Qwen3.8-27B (featherless)](https://github.com/featherless-ai/simple-jev) | 未确认（详见证据） | 本次未确认 |
| [JEV-27B](https://huggingface.co/autotrust/JEV-27B) | 未确认（详见证据） | 本次未确认 |
| [Eikos-27B](https://huggingface.co/caiovicentino1/Eikos-27B-FP8) | 未确认（详见证据） | 本次未确认 |
| [JADE](https://huggingface.co/theunnecessarythings/JADE) | 未确认（详见证据） | 本次未确认 |
| [Cloudflare clef](https://huggingface.co/Cloudflare/clef) | 未确认（详见证据） | 本次未确认 |
| [Jevfire](https://github.com/kikoncuo/jevfire) | 未确认（详见证据） | 本次未确认 |
| [Solomon v1.1](https://huggingface.co/DoccyHealth/Solomon) | 未确认（详见证据） | 本次未确认 |
| [linne-fern 1.0 (27B)](https://huggingface.co/MISA-core/linne-fern-1-0) | causal | [证据1](https://huggingface.co/MISA-core/linne-fern-1-0) |
| [Qwen Decision 27B noncausal LoRA](https://huggingface.co/jsaurabh/qwen-decision-27b-noncausal-lora) | full-attention 层整段双向 | [证据1](https://huggingface.co/jsaurabh/qwen-decision-27b-noncausal-lora) |
| [RSI-Jev v6.1 VL 27B (effort auto)](https://huggingface.co/shgao/rsi-jev-v6.1-vl-27b) | 未确认（详见证据） | 本次未确认 |
| [vLLM-SR d3 (27B)](https://huggingface.co/vllm-sr/d3/tree/v3.1.0) | full-attention 层整段双向 | [证据1](https://huggingface.co/vllm-sr/d3/blob/v3.1.0/decision_config.json) |
| [StartLux-Decision v1.1 27B](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant) | full-attention 层整段双向 | [证据1](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant) |
| [Perplexity Decider v1.2 (27B)](https://huggingface.co/perplexity-ai/pplx-decider-v1.2) | full-attention 层整段双向 | [证据1](https://huggingface.co/perplexity-ai/pplx-decider-v1.2) |
| [Amos 1 Large (27B)](https://huggingface.co/mukaddamzaid/amos-1-large) | 未确认（详见证据） | 本次未确认 |
| [Lychee-27B](https://huggingface.co/LycheeAI/Lychee-27B) | 未确认（详见证据） | 本次未确认 |
| Nace.AI Drex Max v1 (27B) | 未确认（详见证据） | 本次未确认 |
| [Matilda Decision v1](https://huggingface.co/Maincode/matilda-decision-v1) | full-attention 层整段双向 | [证据1](https://huggingface.co/Maincode/matilda-decision-v1) |
| [Sieve 27B v2](https://huggingface.co/sthanika-ai/Sieve-27B-v2) | causal | [证据1](https://huggingface.co/sthanika-ai/Sieve-27B-v2)、[证据2](https://github.com/sthanika-ai/Sieve/blob/1e64657c930f91db7119b5638774c0f11a77e441/sieve/load.py#L215-L234) |
| [Cerno 27B](https://huggingface.co/apertixai/cerno-27b) | 未确认（详见证据） | 本次未确认 |
| [OneJev1.6-27B](https://huggingface.co/OmniJev/OneJev1.6-27B) | 未确认（详见证据） | 本次未确认 |
| [frontier-infra Jebadiah 27B v2.1 (t1e)](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1) | 未确认（详见证据） | 本次未确认 |
| [VisionJEV 1.0.0 (27B)](https://huggingface.co/taoszhang/VisionJEV) | 未确认（详见证据） | 本次未确认 |
| [vLLM-SR d3-turbo v1.1 (27B)](https://huggingface.co/vllm-sr/d3-turbo) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/vllm-sr/d3-turbo/blob/v1.1/decision_config.json)、[证据2](https://huggingface.co/vllm-sr/d3-turbo/blob/v1.1/d3_runtime.py) |
| [Lumma-fev-27B](https://huggingface.co/FrontiersMind/Lumma-fev-27B) | 未确认（详见证据） | 本次未确认 |
| [vLLM-SR d3-ultra (Decision 3.0, 27B)](https://huggingface.co/vllm-sr/d3-ultra) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/vllm-sr/d3-ultra/blob/37fee01926f3de396620e794201b074fb427ac2d/decision_config.json)、[证据2](https://huggingface.co/vllm-sr/d3-ultra/blob/37fee01926f3de396620e794201b074fb427ac2d/d3_runtime.py) |
| [Ainvest-Decision-27B v1.0](https://huggingface.co/Ainvest-World/Ainvest-Decision-27b-v1.0) | full-attention 层整段双向 | [证据1](https://huggingface.co/Ainvest-World/Ainvest-Decision-27b-v1.0) |
| [WeMM-JEV-27B](https://huggingface.co/Rover00/WeMM-JEV) | 未确认（详见证据） | 本次未确认 |
| [OneShot 27B](https://huggingface.co/OneShotOrg/OneShot) | 未确认（详见证据） | 本次未确认 |
| [vera-large](https://huggingface.co/scar-ai/vera-large) | full-attention 层整段双向 | [证据1](https://huggingface.co/scar-ai/vera-large) |
| [NTele-Hybrid-27B-v1](https://huggingface.co/ZTE-AIM/NTele-Hybrid-27B-v1) | 未确认（详见证据） | 本次未确认 |
| [Bespoke Nimble v4.1 Max (27B)](https://huggingface.co/bespokelabs/Nimble-v4-max) | 分块／分区双向（各自规则不同） | [证据1](https://huggingface.co/bespokelabs/Nimble-v4-max/blob/main/schema_config.json)、[证据2](https://huggingface.co/datasets/bespokelabs/nimble-v4-max-decision-index/blob/main/code/nimble/training/prefix_attention.py) |
| [GLiDE 27B](https://huggingface.co/fastino/GLiDE-27B) | full-attention 层整段双向 | [证据1](https://huggingface.co/fastino/GLiDE-27B/blob/main/glide/model.py#L35-L39) |
| [Torchcast Decision 27B](https://huggingface.co/torchcast-ai/torchcast-decision-27b) | 未确认（详见证据） | 本次未确认 |

## Qwen/Qwen3.8-27B-FP8

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [reflex Qwen3.8-27B-FP8 (pmscr)](https://github.com/kshetrajna12/reflex) | 未确认（详见证据） | 本次未确认 |

## XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B

| 榜单模型 | attention 判断 | 作者证据 |
|---|---|---|
| [Nace.AI Drex v1.5](https://huggingface.co/nace-ai/drex-v1.5) | causal | [证据1](https://huggingface.co/nace-ai/drex-v1.5/blob/main/code/kev/model.py) |

