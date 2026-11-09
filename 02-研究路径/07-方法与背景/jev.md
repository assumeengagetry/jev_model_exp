## TL;DR

1. Jev 既不是 encoder-only，也不是 diffusion。它是 TypeSafe AI 的闭源 transformer 判别式决策模型：输入 `state` + 一组 typed `questions`，一次前向把每个问题的概率分布并行读出，没有 token 级 autoregressive 解码；训练方法官方称 RLCD，细节未公开（[TypeSafe 发布博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、[Wikipedia](https://en.wikipedia.org/wiki/Jev_\(AI_model\))）。

2. 榜单前 30 名清一色是「开源 decoder LLM（Qwen3.8-27B 或 Gemma-4-31B-it）+ 决策 readout + SFT（部分加 calibration / RL）」。encoder 路线（ModernBERT、reranker）最高只有 28.96 分，diffusion 路线（DiffusionGemma）最高 49.08，不要走这两条路（数据来自 [Decision Index Space](https://huggingface.co/spaces/multimodalart/jev-decision-index) 的 index.json）。

3. 拉开差距的四件事：数据（规模 + 覆盖榜单 skill 的公开 train split + 程序合成 + teacher soft label）、结构（255-way readout + prompt 内 noncausal attention）、calibration（per-type / per-option-count temperature + 多 seed souping）、推理工程（单卡 96 GB、median/mean/p80 ≤ 1000 ms）。

4. 8×H200 绰绰有余：Sieve-31B 用 2×B200 跑 12 h（[Sieve-31B card](https://huggingface.co/sthanika-ai/Sieve-31B)）；Mapika 2B 一个 epoch 455M tokens 在单张 GH200 上 5.3 h（[Mapika/decider README](https://github.com/Mapika/decider/blob/main/README.md)）；27B full FT 1.5B tokens 在 8×H200 上估算约 1 天/epoch。

---

## 1. Jev 是什么

### 1.1 官方口径

- TypeSafe AI 2026-09-15 发布的「System One model」，创始人 Diogo Almeida 来自 OpenAI RLHF/InstructGPT 团队（[Wikipedia](https://en.wikipedia.org/wiki/Jev_\(AI_model\))）。官方描述为「a new model architecture, parallel sampler for maximum efficiency, and training method we call Reinforcement Learning for Calibrated Decisions (RLCD)」（[发布博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。

- 输入输出契约：`state`（string / JSON / array）+ `questions`，问题三种类型：`noul`（true/false）、`choice`（无序选项，最多 255 个）、`score`（有序等级）；每题返回一个概率分布和 confidence（[docs/models](https://docs.typesafe.ai/models.md)、[docs/primitives](https://docs.typesafe.ai/primitives.md)）。confidence = (p_max − 1/n)/(1 − 1/n)（[docs/confidence](https://docs.typesafe.ai/confidence.md)）。

- 所有问题并行回答，问题之间不能有依赖；官方对比表里 LLM 是「Sequential, one token at a time」，Jev 是「outputs all probabilities in parallel instead of autoregressively generating by token」（[发布博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。

- 限制与成本：64k/request、32k/question，加 `TypeSafe-Danger-Zone-Long-Context: true` header 可放宽（更慢更不准）；只按输入计费，输出免费（[docs/models](https://docs.typesafe.ai/models.md)）。Wikipedia 记录其端到端延迟 70–500 ms（[Wikipedia](https://en.wikipedia.org/wiki/Jev_\(AI_model\))）。

- 架构、权重、论文均未公开，只说 transformer-based；外界猜测是开源 LLM 或 BERT-like（[Wikipedia](https://en.wikipedia.org/wiki/Jev_\(AI_model\))）。Jev 本身不能 fine-tune，所以「训 Jev」实际上是训一个 Jev-like 开源模型。

### 1.2 训练：RLCD

- 官方 AI primer 把 RLCD 和 RLHF / RLVR 并列为第三条后训练路径：「trains TypeSafe to return decisions and calibrated probabilities instead of generated text」，目标是概率 0.8 的事件约 80% 发生（[docs/machine-learning-primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)）。

- Wikipedia 转述：Jev 全部用合成数据训练，RLCD 细节未公开，社区怀疑本质是带校准惩罚项（可能含 Brier score）的标准监督学习（[Wikipedia](https://en.wikipedia.org/wiki/Jev_\(AI_model\))）。

### 1.3 社区逆向结论（Archer Hume）

Archer Hume 通过 API 探测给出的重建（[Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/)）：

|             |                                                                      |                                                                     |
| ----------- | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 组件          | 推断                                                                   | 证据                                                                  |
| Backbone    | 很可能是 causal decoder transformer（不是实验能证明的，bidirectional encoder 不能排除） | MMLU-Pro 84.6%；RLCD 被描述为对预训练 LM 的后训练；causal 才方便 shared-prefix cache |
| Diffusion   | 不需要，也没有迭代去噪的任何证据                                                     | 并行答案用 prefill + 多 readout 即可                                        |
| 共享 state    | state 编码一次，各问题分支复用                                                   | token 计数可加：1 题 268、2 题 276、3 题 318 tokens                           |
| 问题隔离        | 问题之间互不可见                                                             | 把秘密放在兄弟问题里概率 0.00，放到 state 里变 0.90–0.92                             |
| Readout     | 直接数值读出 (z = Wh + b, softmax)                                         | 官方说概率并行输出                                                           |
| Option 联合处理 | 选项之间会互相影响                                                            | 追加一个无关选项使 log-odds 平均变化 −0.28                                       |
| 并行度         | 到约 100 个问题延迟几乎不变                                                     | 1,500 问题请求 median 610 ms                                            |

### 1.4 回答你的三个问题

- 是 encoder-only 吗：官方没说，行为上更像「decoder LLM + 判别式读出」；榜上 encoder 路线最高 28.96 分（Crivo-1B，base ettin-reranker），ModernBERT-large 的 Amber 0.4B 只有 25.41（index.json）。

- 是 diffusion 吗：不是。榜上 4 个 DiffusionGemma-26B-A4B 条目 35.47–49.08 分，远低于 AR 基座。

- 训练/推理和你熟悉的 AR、diffusion 有什么不同：
  
  - 训练 = 分类/排序式 SFT（CE 或 proper scoring rule）+ calibration，不需要 RLHF 式采样，也没有去噪调度；
  
  - 推理 = 纯 prefill + 一次 readout，没有 decode 循环，没有迭代步数。瓶颈变成 prefill 吞吐、长上下文 attention、同一 state 下多问题的共享、以及 head 读出；speculative decoding、decode 侧的 KV paging 优化全部无用，而 prefix caching、chunked prefill、FP8 GEMM、CUDA graphs、varlen packing 都直接有用。

---

## 2. Decision Index 榜：规则与现状

### 2.1 计分与规则

- 维护者 apolinario（HF multimodalart），与 TypeSafe 无关。0.4 版主排名 Full score = 60% 私有同技能测试 + 40% 私有新领域任务；公开套件仍必须完整跑（用于验证和对账）但不计分；训练数据与某项私测重叠则该项不计（[decision-index kit README](https://github.com/apolinario/decision-index)）。

- 公开套件 37 个 benchmark、5 个领域：Knowledge & Reasoning 25.85%、Language Understanding 25.85%、Retrieval & Classification 20%、Tools & Automation 18.3%、Arts & Human Taste 10%；gold 基准权重 1.2，分数做 chance 校正。完整跑的门槛是 120,340 条请求减去 442 条排除项 = 119,898 行（methodology.json，来自 [Space](https://huggingface.co/spaces/multimodalart/jev-decision-index)）。

- 硬规则（methodology.json）：no truncation（放不下就记为 unsupported）、no label filtering（255 个选项必须全给）、no prompt tuning、no labels in adapter、no correctness-based retry、native abstentions stand、unanswered = wrong。

- 延迟门槛：维护者在 1× RTX PRO 6000（Blackwell，96 GB）上单进程、单请求、模型最快路径测 750 行 + 10 行 warmup；median、mean、p80 任一超过 1,000 ms 就不上榜（[kit README](https://github.com/apolinario/decision-index)）。

- 验证：维护者用私有 held-out 样本复跑并比对答案，决策一致率门槛 98.5%（index.json 的条目备注）。污染案例：reflex-4B 的 LoRA 训练了 800 条 MMLU-Pro test 题，在这些题上 65.8% vs held-out 52.1%，被移除（methodology.json）。

- 提交流程：用 kit 跑完整 public 0.3 suite（`python -m decision_index pipeline --engine http --option base_url=...`，或 `hf jobs run --flavor rtx-pro-6000 --timeout 72h`），上传结果数据集，向 `submissions/README.md` 发 PR，维护者负责私测（[kit README](https://github.com/apolinario/decision-index)）。

- 榜单自己跑一遍 Jev 1.14 花了 $12.26、2.92 亿输入 token（methodology.json）。

### 2.2 当前排名（0.4，2026-10-20 快照，index.json）

|     |                             |       |        |                |            |         |                                                                                                                                               |
| --- | --------------------------- | ----- | ------ | -------------- | ---------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| #   | 模型                          | Full  | Public | 基座             | 类型         | 中位延迟 ms | 一句话配方                                                                                                                                         |
| 1   | mammoth-1 27B (baselabs)    | 71.24 | 75.18  | Qwen3.5-27B    | full FT    | 219     | 「one shared-prefix prefill」，其余不公开，权重仅供评测（[card](https://huggingface.co/baselabs/mammoth-1)）                                                   |
| 2   | Perplexity Decider v1.2 27B | 69.42 | 67.74  | Qwen3.8-27B    | full FT    | 144     | noncausal full-attn + [255×5120] readout + 多题 joint + souping（[card](https://huggingface.co/perplexity-ai/pplx-decider-v1.2)）                 |
| 3   | StartLux-Decision v1.1 27B  | 69.29 | 72.88  | Qwen3.8-27B    | full FT    | 80      | 两阶段 0.8M + 1.7M 例，bidirectional prompt attention，4 seed 平均（[card](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant)） |
| 4   | vLLM-SR d3 27B              | 69.06 | 65.45  | Qwen3.8-27B    | full FT    | 97      | vLLM Semantic Router 团队，在 AMD MI325X 训练（[card](https://huggingface.co/vllm-sr/d3)）                                                            |
| 5   | AJev lm6q                   | 69.05 | 65.46  | Gemma-4-31B-it | LoRA       | 144     | LoRA r=64 α=89.6，T=1 不做校准（[card](https://huggingface.co/andyzhang232/ajev-gemma4-31b-lm6q)）                                                   |
| 6   | vLLM-SR d3-turbo            | 69.04 | 71.61  | Qwen3.8-27B    | full FT    | 195     | `attention_mode: prefix_bidir`（[card](https://huggingface.co/vllm-sr/d3-turbo)）                                                               |
| 7   | linne-oak 1.0 31B           | 68.55 | 63.94  | Gemma-4-31B-it | full FT    | 105     | full-attn 层 bidirectional，option-letter logits + soft-capping（[card](https://huggingface.co/MISA-core/linne-oak-1-0)）                         |
| 8   | Nace.AI Drex Max v1         | 68.51 | 67.52  | Qwen3.8-27B    | full FT    | 104     | 未公开细节                                                                                                                                         |
| 9   | Lychee-27B                  | 68.28 | 62.76  | Qwen3.8-27B    | full FT    | 97      | 复用 Perplexity v1.1 的 autojev 配方（[card](https://huggingface.co/LycheeAI/Lychee-27B)）                                                           |
| 10  | Amos 1 Large                | 67.48 | 64.94  | Qwen3.8-27B    | full FT    | 148     | option 旋转 2 次读取（[card](https://huggingface.co/mukaddamzaid/amos-1-large)）                                                                     |
| 11  | Decider 31B NVFP4 (Mapika)  | 67.04 | 64.27  | Gemma-4-31B-it | full FT    | 72      | 全开源训练代码，~95 个公开数据集 + 27B teacher，NVFP4（[repo](https://github.com/Mapika/decider)）                                                             |
| 12  | Matilda Decision v1         | 65.91 | 65.39  | Qwen3.8-27B    | full FT    | 92      | SFT 后 RL，bidirectional attention（[card](https://huggingface.co/Maincode/matilda-decision-v1)）                                                 |
| 13  | Bobcat 1.3 31B              | 65.61 | 59.31  | Gemma-4-31B-it | LoRA       | 64      | 3 条 LoRA 臂权重平均，teacher KL + gold + RPS（[card](https://huggingface.co/sanghwa-na/bobcat-1.3)）                                                  |
| 14  | Jebadiah 27B v2.1           | 65.37 | 60.46  | Qwen3.8-27B    | LoRA       | 303     | LoRA r=16，仅 16,622 条数据（[card](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1)）                                                       |
| 15  | jev-brain · Gemma-4-31B     | 65.29 | 61.69  | Gemma-4-31B-it | 零训练        | 101     | 纯推理技巧                                                                                                                                         |
| 16  | Sieve 31B                   | 65.17 | 62.52  | Gemma-4-31B-it | LoRA       | 253     | LoRA r=128，soft-label CE + KL-to-base + Brier，2×B200 12 h（[card](https://huggingface.co/sthanika-ai/Sieve-31B)）                               |
| 17  | OneJev1.6-27B               | 64.54 | 69.55  | Qwen3.8-27B    | full FT    | 112     | public 高、new-domain 仅 54.88（[card](https://huggingface.co/OmniJev/OneJev1.6-27B)）                                                             |
| 18  | MORJ 1.0.1                  | 64.49 | 60.24  | Gemma-4-31B-it | 零训练        | 202     | letter-logit readout + rotation averaging（[repo](https://github.com/ALERIA-ML/MORJ)）                                                          |
| 19  | AnyJev L0                   | 64.24 | 59.79  | Gemma-4-31B-it | 零训练        | 98      | label-free running prior（[repo](https://github.com/MorrisZJ/anyjev-l0-gemma)）                                                                 |
| 28  | Jev 1.14（TypeSafe API）      | 62.22 | —      | 闭源             | hosted API | —       | 榜单基准线                                                                                                                                         |

小模型参考：elephant-1 9B 62.65（public 72.24，new-domain 51.05）、StartLux 9B 59.45、StartLux 4B 56.35、Perplexity Decider Flash 4B 55.39（index.json）。

### 2.3 组成统计（208 个条目，index.json）

- 类型：108 full fine-tune、43 LoRA、29 LoRA + head、25 inference technique、2 head/adapter、1 hosted API。

- 技术路线：172 autoregressive、28 encoder（最高 28.96）、4 diffusion（最高 49.08）、3 GLiNER 系。

- 前 40 名基座：Qwen3.8-27B 19 个，Gemma-4-31B-it 13 个。

- 结论：27–31B dense 是当前性价比上限区间（96 GB 单卡下 bf16 27B 约 49–54 GB、31B 约 62 GB 都放得下，FP8/NVFP4 不掉分）。

---

## 3. 榜上模型是怎么训出来的

### 3.1 关键配方对照

|                                                                                                                                                                |                                                                                                                                                  |                                                                        |                                                                                  |                                                                                                         |                                                                                                                   |                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 模型                                                                                                                                                             | 数据                                                                                                                                               | Readout                                                                | Attention                                                                        | Loss                                                                                                    | Calibration                                                                                                       | 算力                                                                                                                                 |
| Perplexity Decider v1.2（[card](https://huggingface.co/perplexity-ai/pplx-decider-v1.2)、[v1.1](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)）     | 部分 DI 任务的官方 train split（去污染）+ 大量其他决策数据，重点感谢 tasksource；v1 用 Bespoke Nimble converters 转换人工标注公开集                                                  | 独立 [255×5120] readout head，在每题后的 `<                                    | fim_suffix                                                                       | >` 标记处读出；多题放一个序列                                                                                        | full-attn 层 noncausal over whole request，linear-attn 层保持 causal；v1.1 明确把增益归于「lifting the causal mask + more data」 | 未公开                                                                                                                                |
| StartLux v1.1 27B（[card](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant)、[repo](https://github.com/startluxlabs/StartLux-Decision)） | Stage 1 约 0.8M 例 full FT；Stage 2 约 1.7M 例 adapter 后合并；覆盖 38 个基准中 33 个的 train split                                                               | 单次前向读概率；`none_probability` 弃权选项                                        | Stage 2 bidirectional prompt attention（full-attn 层中每 4 个 prompt token 有 1 个也看后文） | 未公开；AutoThink 版用「RLCD²」                                                                                 | per-type 温度；4 seed uniform average                                                                                | vLLM 插件实现 bidirectional；H200 单请求 25–52 ms；AutoThink 27B public 73.37 vs Instant 72.83                                              |
| Sieve 31B（[card](https://huggingface.co/sthanika-ai/Sieve-31B)）                                                                                                | 190,314 题 / 155 tracks / 76M tokens，13-gram / 8-gram 去污染                                                                                         | 在 `Answer: (` 槽位用 lm_head 中选项字母行读出                                     | 原生 causal                                                                        | soft-label CE（T=8.5495）+ 0.5·KL(frozen base) + 0.3·Brier                                                | held-out 温度                                                                                                       | LoRA r=128 全投影 60 层，AdamW 5e-5，2,742 步，8,192 token 上限，2×B200 约 12 h                                                                |
| Bobcat 1.3（[card](https://huggingface.co/sanghwa-na/bobcat-1.3)、[repo](https://github.com/foxl-ai/bobcat)）                                                     | 54,732 行 + 17 个公开源（MASSIVE、MedMCQA、ScienceQA、AQuA-RAT、WiCE、VitaminC、LEDGAR…）；injected-answer / insufficient-evidence / long-state counterfactual | letter logits                                                          | causal                                                                           | KL(teacher) + gold CE + Score RPS + KL(frozen base)                                                     | 温度在 LocalLLaMA/typed-decisions 上拟合（choice 1.008 / noul 0.445 / score 1.023）                                       | 3 条 LoRA r=16 α=32，1 epoch，LR 1e-4，32,768 token/GPU × 8 GPU，权重空间平均；长 state 掉点 −2.5/−5.3/−7.0 @30k/61k/92k                          |
| Mapika decider（[repo](https://github.com/Mapika/decider)、[RL.md](https://github.com/Mapika/decider/blob/main/docs/RL.md)）                                      | ~95 个公开集 + 本地 Qwen3.5-27B teacher 标注；mixture v2 加 26 个决策集 + 10 个可验证程序族；1 epoch = 1.47M 例 / 455M tokens                                           | 答案槽 hidden state 投到 A–J / K–Z / 两字母码，最多 255                            | causal                                                                           | SFT；RL 阶段 PPO（clip 0.2）+ log-score belief（proper scoring rule）+ rendering-consistency KL + retention KL | T(n)=a+b·ln n；top p<0.7 时反转选项再读一次                                                                                 | 35B 用 Muon + FP32 master（后发现 master copy 伤知识任务）；加入 CLINC150/iSarcasm/VAST/ACOS/POP909/GSM8K/NLI4CT 的 train split 后这些基准 +17 skill 分 |
| Jebadiah 27B v2.1（[card](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1)）                                                                             | 16,622 条（~80 个公开集，tasksource/*、typed-decisions、kev-suites、HelpSteer、13,291 条规则合成）                                                                | letter logits                                                          | causal                                                                           | CE                                                                                                      | per-type 温度 choice 1.06 / noul 1.04 / score 0.65                                                                  | LoRA r=16 α=32，1 epoch，LR 1e-4，2,048 token 上限                                                                                      |
| linne-oak 31B（[card](https://huggingface.co/MISA-core/linne-oak-1-0)）                                                                                          | 未公开                                                                                                                                              | option-letter next-token logits + final logit soft-capping             | full-attn 层 bidirectional，sliding-window 层原生                                     | 未公开                                                                                                     | —                                                                                                                 | RTX PRO 6000 105 ms / H200 63 ms                                                                                                   |
| MORJ / AnyJev / decisio（零训练，Gemma-4-31B-it，63–64.5 分，[MORJ](https://github.com/ALERIA-ML/MORJ)、[AnyJev](https://github.com/MorrisZJ/anyjev-l0-gemma)）          | 无                                                                                                                                                | letter-logit；>26 选项用字母+数字码                                             | causal                                                                           | 无                                                                                                       | rotation averaging、option-count 温度、label-free running prior                                                       | vLLM FP8                                                                                                                           |
| Bespoke Nimble 9B（[repo](https://github.com/bespokelabsai/nimble)）                                                                                             | 2,676 条对照式合成（改 ≤8 个词翻转标签）+ 公开集 converters                                                                                                        | 每个选项一个单 token 码，softmax over candidate logits                          | causal                                                                           | 候选 logits 上的 CE                                                                                         | T=2.179                                                                                                           | Qwen3.5-9B LoRA r=16，LR 5e-5，batch 8，1 epoch，2,048 token，H100                                                                      |
| Together tev1 4B（[博客](https://www.together.ai/blog/how-to-train-your-own-jev)）                                                                                 | ~38k 题：MultiNLI、BoolQ、Banking77、AG News、SST-5、程序化 policy、routing、论文分类                                                                            | 生成字母（temperature 0，max_tokens 8）                                       | causal                                                                           | SFT                                                                                                     | 无                                                                                                                 | Together 微调服务，$17、25 分钟                                                                                                            |
| NanoJev 0.6B（[repo](https://github.com/TianyuCodings/NanoJev)）                                                                                                 | 18,760 题游戏决策                                                                                                                                     | 共享 scoring head + set attention 给 2–255 候选打分；noul sigmoid；score 2–10 级 | —                                                                                | complete-question CE                                                                                    | —                                                                                                                 | backbone LR 1e-5 / head 1e-4                                                                                                       |

### 3.2 共性规律（直接可用的结论）

1. 基座：Qwen3.8-27B 或 Gemma-4-31B-it 的 instruct 版，dense。没有人用 >35B 的模型上榜（96 GB + 1 s 的约束）。

2. Readout 三流派：(a) 选项字母 / 码的 lm_head 行 logits，读答案槽位（MORJ、linne-oak、Sieve-31B、Mapika）；(b) 独立 255-way head [255×hidden]，可从 lm_head 对应行初始化（Perplexity、Sieve-4B）；(c) pointer / scalar head 给每个选项末尾 hidden state 打分（Sieve pointer head、NanoJev）。超过 26 个选项用两字母码（AA–ZZ 都是单 token）保证一次读出（methodology.json 记录 14,493/151,024 行需要 >62 选项或 >16 问题的「lift」）。

3. Attention：prompt 内 noncausal / bidirectional 是前 12 名里反复出现的共同点（Perplexity、StartLux、d3-turbo、linne-oak、Matilda）。代价是标准 vLLM 不支持，需要插件或自带 runtime，而且 state 前缀的 KV cache 不能跨问题复用（除非像 Perplexity 把多题放进同一序列）。

4. 数据量级 10^5–10^6 题。来源四类：tasksource（600+ 任务）、榜单基准源数据集的公开 train split（去污染后合法，Mapika 实测 +17 skill 分）、程序化合成（规则 / policy / 状态跟踪 / 路由 / 对照对 / injected-answer）、teacher soft label（Qwen3.5-27B 级别）。选项顺序必须随机打乱。

5. Loss：hard CE 是底线；更强的是 soft-label CE + KL-to-frozen-base（保知识、防漂移）+ Brier / RPS（score 型有序）。RL 阶段（Mapika 的 PPO + log-score belief）是加分项而非必需。

6. Calibration：per-type / per-option-count 温度在 held-out 上拟合；rotation averaging（翻转选项顺序读两次取平均）；多 seed 权重平均（StartLux 4 seed、Perplexity souping、Bobcat 3 臂）。

7. LoRA r=16–128 可以到 65–69 分，full FT 才到 69–71；FP8 / NVFP4 推理不掉分（StartLux vLLM FP8 同分，Mapika NVFP4 67.04）。

---

## 4. 生态速查

### 4.1 常用数据集

|                                                                                                        |                                                                                                                                          |                                                                       |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 数据集                                                                                                    | 内容                                                                                                                                       | 用途                                                                    |
| [tasksource](https://huggingface.co/tasksource)                                                        | 600+ 任务统一成 MultipleChoice / Classification 格式；有 `tasksource-jev-typed-decisions` 直接是 typed 格式                                            | Perplexity 主力来源；广覆盖                                                   |
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)               | 4 个工作流 × (300 train + 100 test)，每条 5 个 typed 问题，`state`+`questions` 就是 `/v1/systemone` 请求体，gold 为完整分布                                    | 校准温度 / 开发集（Bobcat、Jebadiah 都用）                                        |
| [Bespoke Nimble](https://github.com/bespokelabsai/nimble)                                              | 2,676 train / 324 eval 对照式合成，附公开集 converters，benchmark 子集按 record ID + checksum 钉死                                                       | 合成方法模板 + 去污染方法                                                        |
| jaredpalmer/kev-suites、pngwn/typed-decisions-v2-system-one、ZefanCai/Open-Jev、Benchmark-Heaven/JevBench | 社区 typed 决策集（Space 的 News 页列出）                                                                                                           | 注意 pngwn 版本被榜单标记与 MMLU-Pro test 重叠，慎用                                 |
| 榜单基准的公开 train split                                                                                    | BANKING77、CLINC150+OOS、ANLI、GSM8K、NLI4CT、VAST、ACOS、iSarcasmEval、POP909、ContractNLI、BFCL、API-Bank、MMLU 系、WinoGrande、HellaSwag、HelpSteer 等 | StartLux 覆盖 33/38 个基准的 train split；必须对 test 行做 n-gram 去重              |
| 程序化合成                                                                                                  | 规则 / policy 应用、路由、状态跟踪、智能家居、Minesweeper 等可验证 gold                                                                                        | Mapika mixture v2 的 10 个程序族、Together 13.5k policy 题、Jebadiah 13,291 条 |

### 4.2 训练代码 / 框架

- 完整开源训练代码：[Mapika/decider](https://github.com/Mapika/decider)（`decider/train.py`、`decider/data/mixture.py` 列出 ~95 个数据集、Muon、RL 文档）、[sthanika-ai/sieve](https://github.com/sthanika-ai/sieve)（pointer head / option fork / 255-way head 三种读出实现）、[foxl-ai/bobcat](https://github.com/foxl-ai/bobcat)（`student_train.py` LoRA，CE + Brier + proper-score RL，对抗数据生成脚本）、[getainode/jebadiah](https://github.com/getainode/jebadiah)、[bespokelabsai/nimble](https://github.com/bespokelabsai/nimble)、[togethercomputer/tev1](https://github.com/togethercomputer/tev1)、[TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)、Perplexity card 内附带的 `source/src/autojev/model.py`（[v1.1 card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)）。

- 底层框架：榜上公开的都是 HF Transformers + PEFT 自写训练循环或 Trainer，没有人公开用 Megatron；27B full FT 用 FSDP 即可（Perplexity 没有公布训练框架）。榜单按文件判类型：有 `adapter_config.json` 算 LoRA，整权重算 full FT（methodology.json）。

### 4.3 推理 / serving

- vLLM 是主流（StartLux 的 bidirectional 插件、vLLM-SR Engine mode、MORJ FP8）；SGLang 有 openjev-sglang 利用 radix cache 复用 state 前缀；Transformers `AutoModel` + `trust_remote_code` 自带 runtime（Matilda、d3）；ModelOpt NVFP4 + FP8 KV（Mapika decider-31b B300 上 39 ms）；GGUF 需带 non-causal 标志（StartLux）。

- 榜单 kit 的 `--engine http` 直接对接 System One 兼容 API（`POST /v1/systemone`），自带 runtime 的模型用 in-process engine。

### 4.4 评测

- 公开：decision-index kit（public 0.3 suite，[repo](https://github.com/apolinario/decision-index)）、LocalLLaMA/typed-decisions（400 test case / 2,000 decisions）、DecisionPivot（StartLux 的对照对准确率）、Nimble 324 held-out、JevBench。

- 自建 dev set 必须与 public suite 做 n-gram 隔离，否则无法预测 Full（私测）表现。

---

## 5. 8×H200 冲榜方案

### 5.1 目标分层

- 第一里程碑（约 4 周）：Full ≥ 66，进前 15；第二里程碑（再 4 周）：Full ≥ 69，进前 5。

- 以 Full 为准而不是 public：OneJev（public 69.55 / Full 64.54）、elephant-1（72.24 / 62.65）、KnowLine（69.49 / 62.28）都是堆榜单 train split 把 public 拉高但 new-domain 塌了的例子（index.json）。

### 5.2 基座与结构

- 主线：Qwen3.8-27B（前 40 名里 19 个，Perplexity / StartLux / d3 / Lychee / Amos 全部用它；hybrid 线性注意力 + full attention 结构，只需改 full-attn 层的 mask 就能做 noncausal）。备选：Gemma-4-31B-it（零训练 letter readout 就有 63–65，LoRA 友好）。

- Day 0 baseline：Gemma-4-31B-it + MORJ 式 letter readout，零训练跑通 kit 全流程并测延迟（预期 Full 63–64）。

- 输入模板：chat template、thinking off；state 在前，question + 选项（单 token 码：A–Z、a–z、0–9、AA–ZZ）在后，末尾答案槽标记。

- Readout：[255 × hidden] 头，从 lm_head 对应码行初始化；noul 用 2 行、score 用 n 行；softmax 只在合法码上做。

- Attention：full-attn 层对整个 prompt noncausal，linear-attn 层保持原样（Perplexity / StartLux 同做法）。第一版一题一行（实现简单，训练用 FlashAttention varlen 非因果即可）；第二版再做 Perplexity 式多题 joint（需要 block mask，用 flex_attention）。

### 5.3 数据 pipeline（目标 1.0–1.5M 题，约 1.5B tokens）

1. 覆盖榜单 skill：37 个基准源数据集的公开 train split，只取 train，对 public suite 所有 state 文本做 8/13-gram 去重（Sieve 做法）+ 按 record ID 钉死排除（Nimble 做法）。Mapika 加 7 个 train split 在对应基准 +17 skill 分，这是最便宜的分。

2. 广覆盖：tasksource 600+ 任务，每任务上限 3–5k 条防长尾淹没；加 `tasksource-jev-typed-decisions`。

3. 程序化合成（决定 40% new-domain 的部分）：规则 / policy 应用、JSON state 字段判断、状态跟踪、路由、对照对（改 ≤8 个词翻转标签）、injected-answer 对抗（Bobcat 把攻击成功率从 46.4% 压到 6.4%）、insufficient-evidence / abstain、30k–60k 长 state。

4. Teacher soft label：用 Qwen3.5/3.6-27B（或自己上一轮模型）输出选项分布，过滤自相矛盾样本（Mapika 发现 teacher 对自己的通用选项标签只有 72% 一致）。

5. 格式增强：选项顺序随机、选项数 2–255 随机、noul 正反双向、score 等级数随机、state 三种形态（text / JSON / array）、多语言（Jev 1.14 已支持多语言，榜单可能跟进）。

### 5.4 训练阶段

|        |                                               |                                                                                                                                                                                                                        |            |
| ------ | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 阶段     | 内容                                            | 关键设置                                                                                                                                                                                                                   | 8×H200 时间  |
| S0     | Gemma-4-31B-it 零训练 baseline + kit 跑通 + 延迟协议复现 | —                                                                                                                                                                                                                      | 1 天        |
| S1     | Qwen3.8-27B full FT，1 epoch，~1.2M 题           | loss = soft-CE(teacher, T≈8) 或 CE(gold) + 0.5·KL(frozen base) + 0.3·Brier/RPS(score)；LR 1e-5（full）/ 1e-4（LoRA）；seq cap 8k；AdamW（先别上 FP32 master + Muon，Mapika 发现伤知识任务）；FSDP + activation checkpointing；4 个 seed 不同数据顺序 | 约 1 天/seed |
| S2     | 长上下文 + noncausal 适配（StartLux Stage 2 思路）      | seq 32k，10–20% 长 state 样本；adapter 训后合并                                                                                                                                                                                 | 1–2 天      |
| S3     | souping + calibration                         | 4 seed uniform average；held-out 上拟合 per-type 温度 + T(n)=a+b·ln n；评估 rotation averaging 是否值得 2× 延迟                                                                                                                       | 半天         |
| S4（可选） | 校准 RL                                         | Mapika 式 PPO + log-score belief + rendering-consistency KL + retention KL；或 StartLux AutoThink（gate 0.9，public 73.37 vs 72.83）                                                                                         | 1 周        |

算力估算：27B full FT 的 bf16 权重 + fp32 master + AdamW 约 16 bytes/param ≈ 432 GB，FSDP 分到 8 卡约 54 GB/卡再加激活，141 GB 够用；按 6·27e9 ≈ 1.6e11 FLOPs/token、8 卡 40% MFU 约 3.2 PFLOPS 算，1.5B tokens 约 21 h/epoch。LoRA 路线快得多（Sieve 76M tokens 在 2×B200 上 12 h）。

### 5.5 推理 / 部署（单卡 96 GB，≤ 1 s）

- 目标 median 80–150 ms（StartLux 80、d3 97、Mapika NVFP4 72）；关键是长尾：最长请求约 2 万多 token 也要 < 1 s，且 mean 和 p80 都受长 state 拖累，所以 prefill 吞吐要稳定在 25k tok/s 以上，27B 在 RTX PRO 6000 上必须 FP8 + 高 MFU 才行。

- 栈：vLLM（Qwen3.8 hybrid 已支持）+ 自定义 attention 插件（full-attn 层 noncausal，参考 StartLux 插件）+ pooling / 分类 head 路径读 readout；FP8 W8A8（Perplexity 报 2.7×）；CUDA graphs + 静态 bucket；同一 state 的多题合成一个 batch 一次前向。

- 如果 noncausal 接 vLLM 太重：Transformers + FlashAttention varlen + packing + torch.compile + CUDA graph，Perplexity 用这条路单请求 96 ms。

- 协议注意：维护者是单进程、单请求测，batch 内并发不加分；rotation averaging 对长行要关掉或只对短行开。

- 提交版 serving 代码要默认启用正确 mask（StartLux、Matilda 都警告 causal 引擎下「是另一个模型」）；量化版要自测与 bf16 的决策一致率 ≥ 98.5%。

### 5.6 提交流程

1. `python -m decision_index pipeline --engine http --option base_url=http://localhost:8000` 跑完 public 0.3 suite（119,898 行门槛），或 `hf jobs run --flavor rtx-pro-6000`。

2. 上传 `results.jsonl.gz`、`scores.json`、`environment.json` 等到自己的 HF dataset。

3. 发布权重（LoRA 带 `adapter_config.json`，full FT 整权重）+ 可复现 serving 代码 + 模型卡（声明数据来源、去污染方法、开发期跑 public suite 的次数）。

4. PR 到 [apolinario/decision-index](https://github.com/apolinario/decision-index) 的 `submissions/README.md`，等维护者私测。

### 5.7 风险与坑

- 污染：test split 绝不能进训练；数据源钉 revision；训练集若与某项私测重叠，该项按 0 计（kit README）。

- 选项数覆盖：14,500 行需要 >26 个选项，不支持就直接丢 12% 请求（methodology.json）。

- 长 state 掉点（Bobcat −7.0 @92k）；S2 阶段必须补长样本。

- public 与 Full 脱节：只堆榜单 train split 会被 new-domain 私测打回原形。

- noncausal 模型与现成推理栈不兼容，提前留 1 周做 vLLM 插件 / 自带 runtime。

### 5.8 推荐阅读顺序

1. [Decision Index Space](https://huggingface.co/spaces/multimodalart/jev-decision-index) 与 [kit README](https://github.com/apolinario/decision-index)（规则、提交）。

2. [Perplexity Decider v1.2 card](https://huggingface.co/perplexity-ai/pplx-decider-v1.2)、[v1.1 card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)（结构与 fast path）。

3. [StartLux-Decision 27B card](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant) 与 [repo](https://github.com/startluxlabs/StartLux-Decision)（两阶段训练、vLLM bidirectional 插件）。

4. [Mapika/decider](https://github.com/Mapika/decider)（唯一端到端开源的数据 + 训练 + RL）。

5. [Sieve-31B card](https://huggingface.co/sthanika-ai/Sieve-31B)、[Bobcat 1.3 card](https://huggingface.co/sanghwa-na/bobcat-1.3)（loss 配方与超参）。

6. [Archer Hume 的架构逆向](https://archerhume.com/posts/jevs-architecture-unmasked/)、[TypeSafe ML primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)（理解 Jev 本身）。






