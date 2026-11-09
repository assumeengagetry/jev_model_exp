# Jev Index 中 Qwen 模型的 attention：按规模核查

资料读取日期：2026-11-09。当前榜单 `data/index.json` 的生成时间为 2026-11-06 09:05:08 UTC。本报告只检查公开文档、配置和源码，没有运行模型或实验。

**答案是：并非都使用双向 attention。0.6B、0.8B、1.7B、2B、4B、9B、27B、35B 等量级都有保留 causal 的模型。Mammoth 和 d3-turbo 的分块双向，是其中一种路线。**

这里把“Qwen 为底座”理解为 Qwen 系列衍生模型。名字带 `-Base` 的未指令微调底座与 Qwen 发布的指令版本会另行说明，不能把两者混称为同一 checkpoint。

## 三种 mask 实际改变什么

| 类型 | 易懂的含义 |
|---|---|
| Causal | 每个位置只能使用自己和前面的文字。最后的答案位置仍然能看到前面的完整题目和候选。 |
| 整段输入双向 | 输入中较早的位置也能看到后面的文字。场景中的表示可以结合后面的提问重新计算。 |
| 分块双向 | 先划分输入；块内可以前后互看，块与块之间按指定规则读取。不同模型的分块规则可能不同。 |

Qwen3.5 等模型混合使用 Gated DeltaNet 线性／递归层与常规 softmax-attention 层。**“混合架构”与“是否双向”是两件事。** 下文称某款模型“双向”时，如无另行说明，指它的常规 full-attention 层放开了向后看的限制；线性／递归层往往仍保持 causal。

`full_attention` 是层的类型，不能单看这个名字就判断双向；SDPA、FlashAttention 是计算实现，也不能单独用于判断方向。榜单的 `technique: autoregressive` 同样不能当成 mask 字段。

## 已确认的 causal 代表

| 量级 | 榜单代表 | 底座及证据 |
|---|---|---|
| 0.6B | Decision 2.0 Kai 0.6B | Qwen3-0.6B-Base；[fast.py](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/main/decision2/fast.py#L542)明确写 causal，并构造下三角 mask。 |
| 0.8B | Intern-Decision、Kev、Sieve | Intern/Kev 为 Qwen3.5-0.8B-Base；[Intern 卡](https://huggingface.co/internlm/Intern-Decision-0.8B)明确 one causal forward；[Kev 卡](https://huggingface.co/jaredpalmer/kev-0.8b)明确 causal backbone + pointer。 |
| 1.5B | Harsha / Qwen-2.5-1B-RLCD | 作者实际加载 Qwen2.5-1.5B-Instruct；[engine_torch.py](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/engine_torch.py#L93)是原生 prefill 与普通 padding mask 续算。 |
| 1.7B | Bosun v3.1 1.7B | 作者实际加载 Qwen3-1.7B；[modeling_bosun.py](https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b/blob/main/modeling_bosun.py)保留原生 causal 调用。 |
| 2B | Intern-Decision、Mapika Decider、Sieve | Qwen3.5-2B 系列；[Intern 卡](https://huggingface.co/internlm/Intern-Decision-2B)明确 causal；Mapika 与 Sieve 实际路径也保留 causal。 |
| 4B | Decision 2.0 Nox、Mapika Decider、Kev、Intern-Decision、Sieve、KnowLine | [Nox 源码](https://huggingface.co/vllm-sr/Decision-2.0-Nox-4B/blob/main/decision2/shared_ctx.py#L673)显式构造三角 mask；[KnowLine 卡](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5/blob/main/README.md#L100)直接说明 causal。 |
| 8B | CLM-v0.1-8B | 作者公开路径为原生 Qwen3-8B causal 编码＋末位 pooling。[启动脚本](https://github.com/Contrastive-LM/CLM/blob/main/serve_qwen3_8b.sh)未修改方向；[vLLM 原生实现](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/qwen3.py#L180)默认 DECODER。项目未锁定精确 vLLM 版本，本结论针对公开路径。 |
| 9B | Decision 2.0 Lux、Kev v2、Sieve、Nimble v2/v3、JPT | [Lux fast.py](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B/blob/main/decision2/fast.py)明确 SDPA runs causal；[Kev 卡](https://huggingface.co/jaredpalmer/kev-9b)明确 causal + pointer。 |
| 27B | Kev、Sieve 27B v2 | Qwen3.8-27B；[Kev 固定源码](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L259)和 [Sieve 固定源码](https://github.com/sthanika-ai/Sieve/blob/1e64657c930f91db7119b5638774c0f11a77e441/sieve/model.py#L112)明确每题沿 causal row 计算。 |
| 30B-A3B | Cloudflare clef-omni | Qwen3-Omni-30B-A3B-Instruct，语言骨干保持 causal，另有全局 schema head；[发布源码](https://huggingface.co/Cloudflare/clef-omni/blob/main/joint_schema_model.py#L591)。 |
| 35B-A3B | KnowLine Gen1.5、Mapika Decider、JPT | KnowLine 为 Qwen3.5-35B-A3B；[作者说明](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1.5)明确 causal + read twice，[实际引擎](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1.5/blob/main/knowline_engine.py)使用标准 SGLang 路径。Mapika 为 Qwen3.5-35B-A3B-Base，[发布卡](https://huggingface.co/Mapika/decider-35b-a3b-nvfp4)给出标准 vLLM 读出。 |

35B-A3B 是约 35B 总参数、每个 token 约 3B 激活参数的 MoE，不能与 3B 总参数模型合成一档。当前直接 Qwen 底座库存中没有 14B、32B、72B 条目。

如果严格只找未指令微调的 `-Base`：Kai 0.6B、Intern-Decision 0.8/2/4B、Kev 0.8/4/9B、Mapika 2/4/35B 等提供了明确的 causal 例子。Bosun 的榜单底座写作 `-Base`，但作者配置加载的是无 `-Base` 后缀的发布版本；本报告按实际配置说明。Nimble v2、CLM 等也存在榜单底座与作者卡的差异，见分组记录。

## 双向及分块双向代表

| 模型 | 规模 | 常规 full-attention 层的规则 |
|---|---|---|
| Mammoth-1 | 27B | 场景与问题两个区域分块双向。 |
| d3-turbo v1.1 / d3-ultra | 27B | `prefix_bidir`，场景块与问题块分开；块内双向。 |
| 普通 d3 / flash / mini / nano / lite | 27 / 9 / 4 / 2 / 0.8B | `noncausal_full_attention`，整段 prompt 双向。 |
| d3-edge | 0.6B 服务规模 | 同上；它由 Qwen3.5-0.8B 缩减而来，不能当作 Qwen3-0.6B 的例子。 |
| Perplexity v1.2 / Flash v1.2 | 27 / 4B | 整条请求的 full-attention 层双向，线性层保留 causal。 |
| GLiDE | 0.8 / 2 / 9 / 27B | 整段 prompt 的 full-attention 层双向。 |
| StartLux | 0.8 / 2 / 4 / 9 / 27B | full-attention 层 prompt 双向。 |
| Jev-Style 2B v4 | 2B | 每块最多 2048 token，块内双向，可以看之前的块。 |
| Cerno v1.2 | 4 / 9B | 场景、问题与选项按块组织，块内双向，块间 causal。 |
| Nimble v4.1 Max | 27B | 请求前缀双向，具体 field name 与聊天尾部仍 causal。 |

作者直接证据：

- [Mammoth 固定配置](https://huggingface.co/baselabs/mammoth-1/blob/9d1962f193e48a68cf0f3598dc7fb4462fd23ad9/config.json#L146)分别写明 `full_attention: block_bidirectional`、`linear_attention: causal`；[实际 patch](https://huggingface.co/baselabs/mammoth-1/blob/9d1962f193e48a68cf0f3598dc7fb4462fd23ad9/serve.patch#L250)说明如何分块。
- [d3-turbo v1.1 runtime](https://huggingface.co/vllm-sr/d3-turbo/blob/v1.1/d3_runtime.py#L40)说明 `prefix_bidir`；[普通 d3 v3.1.0 配置](https://huggingface.co/vllm-sr/d3/blob/v3.1.0/decision_config.json#L522)则是 `noncausal_full_attention`。不能把 d3 家族统一写成同一种 mask。
- [Perplexity v1.2](https://huggingface.co/perplexity-ai/pplx-decider-v1.2)、[GLiDE 源码](https://huggingface.co/fastino/GLiDE-9B/blob/main/glide/model.py)、[Jev-Style 2B v4 说明](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4/blob/main/README.md#L375)。
- [Nimble v4 Max 发布配置](https://huggingface.co/bespokelabs/Nimble-v4-max/blob/main/schema_config.json)启用 `prefix_attention: bidirectional_v1`；[对应源码](https://huggingface.co/datasets/bespokelabs/nimble-v4-max-decision-index/blob/main/code/nimble/training/prefix_attention.py)给出请求前缀与尾部的不同规则。

Mammoth 和 d3-turbo 的“块”，主要是“场景”和“问题＋选项”这样的组成部分。场景内部可以前后互看；问题与选项内部也可以前后互看；问题可以看场景，但场景不看后面的问题。这样，同一个场景仍可以先计算一次，再供多个问题使用。这与 Jev-Style 的 2048-token 分块不是同一个设计。

## 与 readout 的关系

**Pointer 并不要求双向 attention。** Kev 与 Sieve 27B 就是 causal 骨干加 pointer。选项字母读出、独立分类层、pointer，决定最后怎样产生选项分数；mask 决定中间每个位置可以使用哪些文字的信息。

causal 的答案位置在输入最后，因此已经能使用前面全部题目与候选。双向的变化主要在于：场景与候选内部较早的位置，也可以结合后面的信息计算。它是否有收益，不能只由 readout 的名字决定。

KnowLine 提供了另一种保留 causal 的方案：把完整输入重复一遍。第二遍读场景时，第一遍的问题已经出现在前面，于是第二遍场景表示能够利用问题。它增加输入计算量，但没有把 mask 改成双向。

还要把骨干和额外 head 分开：Cloudflare clef-flash 9B、clef-omni 30B-A3B 保留 causal 语言骨干，随后使用额外的全局 schema attention head。不能因为 head 有候选交互，就称 Qwen 骨干是双向。见 [clef-omni 发布源码](https://huggingface.co/Cloudflare/clef-omni/blob/main/joint_schema_model.py#L591)。

## 核查范围与逐项资料

当前库存有 165 条直接以 Qwen 为底座的记录，另有 1 条 MiMo 蒸馏 Qwen 的相关记录。上表列已确认的代表，不声称其余模型均已确认 mask，也不据此计算全榜单双向占比。

完整库存与逐项证据在 `attention-inventory.md` / `attention-inventory.json`；无法从已读资料确定的条目明确保留“未确认”。分组详细证据见 `small-findings.md`、`four-b-findings.md`、`nine-b-findings.md`、`large-findings.md`、`mammoth-d3-findings.md`、`misc-findings.md`。官方榜单来源为 [当前数据目录](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/main/data)。
