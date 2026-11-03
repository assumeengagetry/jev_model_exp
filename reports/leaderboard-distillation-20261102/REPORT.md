# 榜单决策模型如何做蒸馏

调研日期：2026-11-02。以作者模型卡、技术报告和公开代码为依据；本次可取得的榜单数据生成于 10.29。本文研究代表性公开配方，不统计全榜蒸馏使用率，不把模型整体分数当作蒸馏净增益。

## 先说结论

**最直接、且与决策任务最契合的路线，是离线获取 Teacher 对完整候选的概率分布，再让 Student 学习该分布；可靠原标签可同时保留。** 但“模型用了教师”不一定等于这种蒸馏：教师生成场景、教师打硬标签、多模型评审和程序验证，是另一类数据生产方法。

对当前 0.6B 候选匹配模型而言，首先应辨别自己在学 **每个 pair 的独立匹配概率**，还是 **同一道题全部候选之间的竞争概率**。二者都可蒸馏，但不是同一个目标。

### 代表配方快速对照

| 模型／明确版本 | Teacher 或监督来源 | 实际做法 | 不能混淆的点 |
|---|---|---|---|
| Perplexity Flash v1.2，4B | 27B 决策模型 | 在教师训练请求上学习完整候选概率分布 | 具体 KD loss／训练 T 未披露；发布还做 soup |
| Vela3-fast，约 380M 决策骨干 | d3，后续 breadth 用 d3／turbo／flash 面板 | 完整候选 soft CE；有 gold 时 half gold＋half teacher | 面板 2/3 同意是筛选，不是公开的概率平均公式；发布包还含其他适配器 |
| Crivo-150M v2 | Crivo-400M | 明确给出 gold CE／Brier＋温度化 Teacher→Student KL | 部分源与长上下文阶段关闭 KD，不强行全池模仿教师 |
| Bobcat 1.2／1.3，31B | Bobcat 1.1，Qwen3.8-27B | 旧行候选分布 KL＋已有 gold；1.3 新技能行用来源 gold；另有 base KL | 27B→31B 也叫蒸馏，不一定压缩参数；base KL 是另一项 |
| Mapika 4B v2.1，历史配方 | Qwen3.6-27B 生成文档题；冻结 4B v1 提供 replay 分布 | 新题学硬标签，回放题学父模型选项分布 | 思考教师生成答案，不等于学生学习 CoT；不是所有 Mapika 版本都相同 |
| StartLux v1.1 | 公开标签、程序 gold、未具名教师与验证器 | 决策 soft CE＋教师标注合成数据；生成题双验证器过滤 | 未披露完整 Teacher logits KD；没有 AutoThink→Instant 蒸馏证据 |
| Mammoth／Elephant | 生成器、五模型评审、程序与公共标签 | 高质量合成任务；至少 4/5 评审认可标签，CE＋Brier 训练 | 没有公开 Mammoth→Elephant KD，也未说把五个评审概率直接平均当 target |
| KnowLine 31B Gen1 | LLM teachers 生成并标注任务 | 过滤后的合成任务＋公开／程序数据训练 | 明确未以 Jev／其他决策模型输出为标签 |
| Sieve-31B 原版与 v3 | 公开数据、程序／生成记录；未披露具名标注 Teacher | 原版 soft CE＋base KL＋Brier；v3 主要 CE＋Brier | soft target、base KL 都不足以证明大模型 Teacher KD；不可跨版本套配方 |

## 一、已经明确核实的代表案例

### Perplexity Flash v1.2：27B 决策模型到 4B 的完整分布蒸馏

[作者模型卡](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2)明确写道，Qwen3.5-4B Student 从 27B 决策模型蒸馏，在教师训练请求上学习：

> “full probability distributions, not just their answers”

也就是同一道题的所有候选概率，而不是只保留 argmax 标签。发布版本还对多个训练 run 做权重平均；这是蒸馏之后的模型组合，不是蒸馏损失本身。

**已披露：**教师尺寸、Student 基座、训练请求来源和完整概率目标。**未在该模型卡披露：**教师数量／融合权重、具体 CE／KL 实现、蒸馏温度与 gold 混合比例。

卡中的 **1.554 是部署校准温度，不是公开的蒸馏温度**；不能据此补出训练超参。卡中 4B 与 27B 的总体成绩也不是“蒸馏相对硬标签”的配平消融。

### Vela3-fast：d3 完整分布到约 380M 编码器

[作者技术文章](https://vllm-sr.ai/blog/vela-3-0-name-it-vela-finds-it)明确描述，初始约两遍 131 万条混合数据，目标是 d3 在同一组选项上的完整答案分布，再补充公开 train、程序任务和 breadth pool。训练是对该目标分布的 CE，不是仅拟合 top answer。

有 gold 时 target 为 **50% gold＋50% Teacher**。Breadth 使用 d3 27B、d3-turbo v1.0、d3-flash 面板，至少两位同意 gold 才保留；没有公开三位教师概率如何聚合，不能自行补成均值公式。对教师强烈反对 gold 的行会删除，反映其将部分生成器 gold 也视为待核查对象。

**这是 d3→Vela 的明确蒸馏，不是 d3 本体上游配方的公开证明。** 文中 131 万条不能拿来反推 d3 最初由哪位教师、按何种 loss 训练；Vela 后续的四学生 checkpoint soup 也是另一个参数组合步骤。

### Crivo-150M v2：小模型中披露很完整的 KD 配方

[作者卡片](https://huggingface.co/tardellirs/crivo-150m#distillation)明确给出 Crivo-400M Teacher 的离线 logits，以及：

`L = α(CE + 0.5 Brier) + (1−α) τ² KL(Teacher || Student)`，`α=0.5`。

- 超过两个选项用 τ=2，否则 τ=1。
- 教师在该来源训练行与 gold 一致率低于 60% 的源及指定例外任务，只用 gold。
- 长上下文第三阶段也只用 gold。

这是“有选择地蒸馏”，而不是拿到 Teacher 就重标整个池。60% 和温度是其实际配方，不是推荐给本项目的通用阈值；该发布许可为非商用，借鉴方法不等于默认可以任意复用其权重／数据。

### Bobcat 1.2／1.3：候选分布＋原标签，跨基座也能蒸馏

[Bobcat 1.2](https://huggingface.co/sanghwa-na/bobcat-1.2#training-and-distillation)明确用 Bobcat 1.1（Qwen3.8-27B）为 Gemma 4 31B Student 提供候选概率。46,694 个普通问题均有教师分布，其中有 gold 的再加 `0.5×gold loss`；另有专门 gold-only 的注入不变性样本。[Bobcat 1.3](https://huggingface.co/sanghwa-na/bobcat-1.3#how-it-was-made)保留这些 teacher targets，再补来源自带 gold 的技能数据，并组合多个训练臂。

[公开 `row_loss`](https://github.com/foxl-ai/bobcat/blob/538169365291d7c46d2c116f8ee09ad9f4b3ac5e/src/bobcat/flash_train.py)确认方向是 `KL(Teacher || Student)`：Teacher logits 按自己的温度转概率，Student 正常 softmax；这条代码路径不是“双方都除同一 T，再乘 T²”的固定教科书实现。

**gold 冲突规则也不统一。** 主线 1.2／1.3 的普通行同时加入 KD 和 gold，没有按 Teacher argmax 是否同意 gold 自动关掉 KD。相反，同作者的 [Bobcat Flash 1.2](https://huggingface.co/sanghwa-na/bobcat-flash-1.2#training-and-distillation)在新增 GLM 判断题中披露：一致时混合 50% gold＋50% Teacher，不一致用 gold。后者是另一发布分支，不能套到主线 1.3。

这提供两种可研究的策略：同时折中两类监督，或对可靠 gold 采用一致性门控；不能说其中一种已被证明普遍最优。

### Mapika：大教师硬标签生成，另用父模型概率回放

公开 [teacher 生成器](https://github.com/Mapika/decider/blob/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/decider/data/teacher_questions.py)默认 Qwen3.5-27B，生成任务后重新判断；但进入公共 trainer 的是整数 `gold`，[训练 loss](https://github.com/Mapika/decider/blob/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/decider/train.py)采用硬标签 CE。保存过教师置信度，不等于训练用了完整分布。

[4B v2.1 的历史模型卡](https://huggingface.co/Mapika/decider-4b/blob/v2.1/README.md#training)则把两种监督明确分开：

- Qwen3.6-27B 开 thinking 生成 11,356 道文档题；同模型在新上下文、打乱选项后复答两次，两次都同意才保留。Student 学的是标签，不是 CoT 文本。
- 6,676 条 replay 使用冻结 4B v1 的候选分布，做 `KL(p_v1 || p_model)`、T=1，保留旧能力。

它报告 held-out replay 的父模型 KL 从 0.112 降至 0.021，但同期还移除 31 条关联行并改变温度配置，不能说这是严格单变量的蒸馏净收益，更不能外推成全榜提升。

### StartLux：生成、复核、决策监督与推理分支分开

[论文 Stage 1–3 与附录 A.5](https://arxiv.org/html/2610.17309v1)披露使用候选 soft CE、公开训练数据、程序标签及教师标注任务；model-written 题目由两名验证器在看不到作者答案时独立恢复标签，二者都认可才保留。教师与验证器名字没有完整公开。

Instant 在 Stage 2 使用双向 full-attention；AutoThink 保留因果注意力，再加入推理监督和 RL。**没有披露 AutoThink 把推理链或推理后概率蒸馏给 Instant 的环节。** 同样，多个 seed 的权重平均也不是多教师标签投票。

### Sieve：soft labels 与基础模型保留，不要自动补出 Teacher

[原 Sieve-31B](https://huggingface.co/sthanika-ai/Sieve-31B#training)为 soft-label CE＋0.5 base KL＋0.3 Brier；其 **8.5495 是自身候选 logits 的训练／推理温度**，不是已公开确认的 Teacher 温度。

[Sieve-31B-v3](https://huggingface.co/sthanika-ai/Sieve-31B-v3#training)公开的是 CE＋0.3 Brier，以及 0.3 base＋0.7 finetuned 的发布权重插值。未披露具名外部 Teacher，不能把旧版本的 base KL 或温度延续成 v3 蒸馏配方。带分布目标的记录也可能来自标注者投票、程序概率或其他数据设计，不必来自大模型。

### KnowLine 31B Gen1：教师生成并标注任务，不等于复制决策模型概率

[作者披露](https://huggingface.co/PelaAI/KnowLine-31B-Gen1#disclosures)同时说明：

- 没有将 Jev 或其他决策模型的输出用作训练标签。
- LLM teachers 生成并标注合成任务，标签经过过滤，但并非全部由人检查。

因此，应把它归为**教师辅助的数据生成／标注**，不能直接写成“用 Jev 或榜首的完整候选分布做 KL 蒸馏”。这不妨碍我们把 KnowLine 选为自己的 Teacher；Teacher 自己采用哪种训练法，与它能否提供有用监督是两回事。

### Mammoth／Elephant：五模型评审造数，不是已披露的五教师概率 KD

[作者报告 §2.4–2.6](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)给出约 621 万条 Mix 8；Mammoth 27B 与 Elephant 9B 均在该混合上全参训练一遍，没有披露 Mammoth→Elephant 的师生压缩流程。

生成样本时可使用 DeepSeek V4-Pro／GLM-5.3 等模型，再用五模型评审或确定性程序验证；评审要求至少四位选择预期答案。**4/5 是保留条件，不是把该题 target 写成 0.8，也不是已公开的五位 Teacher 概率平均。**

训练目标是候选上的 CE＋Brier，Brier 权重 1；其中软目标也包括人类投票和程序计算的精确概率。因而这是高质量多源决策数据与质量控制的参考，不宜用来补出未公布的 KL 蒸馏公式。

### 不同版本与同名家族不能混用

- [`autotrust/JEV-27B`](https://huggingface.co/autotrust/JEV-27B#training-details)明确使用 Jev 1.13 的完整输出分布，损失为 `KL(target || model) + 0.5×RPS(score)`，并对不可靠的均匀占位标签降权。这是明确的概率蒸馏案例。
- 但榜单中的 [`AutoTrust-JEV-27B-v1.1`](https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1)是另一个发布物，其当前卡片没有给出同样的训练配方。不能把前者的 Teacher、数据量和损失直接移到后者名下。
- [`CoCo-Decision-2.4-4B`](https://huggingface.co/corners-ai/CoCo-Decision-2.4-4B)披露多阶段 LoRA、官方 train 与新领域数据、读出结构；这些本身不是 Teacher 蒸馏证据。
- [`GLiDE`](https://huggingface.co/fastino/GLiDE)当前模型卡主要说明推理与部署，没有足以复原 Teacher／蒸馏损失的训练细节。未披露不能解释成“肯定没有蒸馏”。

## 二、常用方法应怎样理解

### 1. 完整候选概率蒸馏：最直接的决策模型 KD

对同一输入、问题和完整候选，Teacher 给出 `q`，Student 给出 `p`，学习：

`L_KD = KL(q || p)`

或等价的 soft-label CE：`−Σ q_i log p_i`。Teacher 固定时，两者仅差 Teacher 熵这一与 Student 无关的项，因此对 Student 的梯度相同；不能把 soft CE 和 forward KL 当成完全不同的蒸馏路线。

其价值是保留“第二候选也有可能”“这道题本来就模糊”等信息，而不是把所有标签都压成 100%／0%。不同 tokenizer 和不同模型架构没有关系，前提是**语义候选空间对齐**。

### 2. 原始监督＋教师监督：避免只学会模仿错误

常用的混合目标可以写为：

`L = (1−λ) CE(y, p) + λ KL(q_T || p)`

其中 `y` 是可信的人工／程序监督，`q_T` 是 Teacher 分布。`λ` 是需要验证的混合权重，不存在“榜单统一采用 0.5”的证据。SOFT 与 HYBRID 必须和同预算 HARD 比较，不能仅凭更接近 Teacher 就宣布能力提高。

对 `score` 这类有序等级，可加 RPS 等保留顺序的项；Brier 用于概率质量。它们不是额外的 Teacher，也不等于蒸馏本身。

### 训练配方中的这些项不要混在一起

| 项目 | 在解决什么问题 | 本轮公开案例 |
|---|---|---|
| 强 Teacher 的候选 KL | 学习教师对当前任务的答案偏好和不确定性 | Bobcat；Perplexity 明确完整分布监督，但未公布具体 KL 实现 |
| 冻结 base／父版本 KL | 避免新训练破坏旧行为，属于能力保持／回放式蒸馏 | Sieve 原版、Bobcat、Mapika 历史版本 |
| Brier／RPS | 分别约束候选概率误差和有序等级误差 | Sieve、Bobcat；不是 Teacher 身份或标签来源 |
| 训练后温度校准 | 调整概率尖锐程度，不改变候选 argmax | 多个系列；不能直接当作 KD 的训练温度 |
| 权重平均／base blending | 组合多个 run 或缓和微调偏移 | Perplexity、StartLux、Bobcat、Sieve；不是标签蒸馏 |

### 3. 多教师与多次判断聚合：融合标签，不是融合权重

多个教师对相同语义候选分别给出概率后，可以先求平均或按独立验证选择权重，再蒸馏到一个 Student。选项顺序交换后也可以先映射回相同候选再合并，减轻位置偏差。

**不能直接平均没对齐的 A／B 概率，也不能把它与之前的模型权重融合混称同一方法。** 多教师未必优于单教师：如果主要增加相关错误或挤占独立样本覆盖，收益可能很小。

还须区分**分布聚合**与**共识筛选**：Vela 的 2/3、Mammoth 的 4/5 是后者，公开资料不足以据此确定最终软 target 的聚合权重。

补充的一手研究（**不是这些榜单发布物的训练报告**）也支持这种谨慎：[多教师 Jev 蒸馏论文](https://arxiv.org/html/2610.09188v1)在检索中发现，同标注预算下双教师覆盖较少 pair，较单 Jev 覆盖更多 pair 低 0.032 NDCG@10；在另一象棋任务中多教师又有收益。结论是先验证互补性，而非预设教师越多越强。

### 4. 教师生成与伪标签：更多是在改进训练数据

教师可生成任务、改写输入、构造困难候选，或为新场景打标签；之后再用程序验证、多模型评审或人工抽查筛选。这可以称广义数据蒸馏／伪标签学习，但不能自动称为“完整 logits 蒸馏”。

对于可验证规则题，程序 gold 通常比未经验证的 Teacher 猜测更值得保留；对于偏好或模糊任务，完整评分分布可能比单一硬标签更有信息。

### 5. 推理链、隐藏层和全词表蒸馏

这些是一般 LLM 蒸馏中的可选方法，但不能因为某模型有 System 2、AutoThink 或使用 reasoning teacher，就认定它把长 CoT／隐藏状态蒸馏进 Student。需要作者明确披露监督目标与训练路径。

对本文核实的完整候选概率方案，没有必要保存全词表 logits 或让 Teacher 输出长解释：决策空间的概率已经足以定义蒸馏损失。

## 三、当前 0.6B 方案与公开路线的关系

### 已经一致的部分

- 先离线缓存 Teacher 概率，Student 训练不要求 Teacher 常驻同一 GPU。
- 保留原始标签，比较 HARD／SOFT／HYBRID，而不是只做一个软标签模型。
- Teacher 与 Student 接收相同可见文本，保留来源、候选顺序和稳定 ID。
- 使用 Teacher 的原生概率协议，不重复施加已经折入权重的校准温度。

### 最重要的差别：pair KD 与整题 KD

| 目标 | Teacher 输出 | Student 如何学习 | 更接近的任务 |
|---|---|---|---|
| 独立匹配／pointwise KD | 每个 `(输入,候选)` 的 P(match) | Bernoulli KL／soft BCE | 独立相关性、多标签、规则满足判断 |
| 整题候选／listwise KD | 同一道 choice 全部候选的分布，总和为 1 | 候选内 softmax 后的 KL／soft CE | 意图分类、单选、多候选竞争决策 |

当前代码采用独立 yes／no Teacher 判断，属于合法的 pointwise 概率蒸馏；Perplexity 卡片披露的完整 choice 分布，则包含同题候选竞争信息，不能称二者完全相同。

不能直接把某个 K 选 1 分布的每一维塞进语义不同的独立 yes／no 监督。若之后比较整题目标，HARD 对照也应使用同样的整题输入与 loss，避免把输入组织／目标函数改变的收益全算给 Teacher。

### 温度与可靠性

经典 softened-logit KD 可写成 `T² KL(softmax(z_T/T) || softmax(z_S/T))`，但不能把别人的某个 T 当通用默认。API 若已返回校准后的概率，先学习原生分布是清楚的参照；额外升温是另一个明确干预。

蒸馏效果应同时看真实任务 dev／留出成绩、概率质量及旧任务保持，不只看 Teacher–Student 一致率。分布越像 Teacher，也可能越完整地继承它的错误。

### 对当前实验最值得借鉴的四点

1. **保留现有 HARD／SOFT／HYBRID 这组问题定义。** PPLX／Vela 支持完整概率监督的可行性，Bobcat／Crivo 则说明原始监督值得保留；目前不能据此替现有实验预判赢家。
2. **先核对监督空间，而不是先换复杂 loss。** 原生 choice 学候选竞争，独立 pair 学匹配；在研究整题 KD 时要另设相同整题目标的 HARD 控制，不改动正在执行的 run。
3. **把 Teacher–gold 分歧记录为质量信号。** 可靠人类／程序 gold 不应仅因 Teacher 自信反对就被删；对生成器标签则可以复核。门控、混合、丢弃分别是不同策略，没有统一最优规则。
4. **能力保持与能力提升分开判断。** 更强 Teacher KD 用于学习新判断，父模型／base KL 用于保旧行为；温度、RPS／Brier 和权重平均都应分别记录，不把所有收益统称蒸馏提升。

## 四、证据边界与来源

- 代表性配方并不能证明所有榜单高分模型都靠蒸馏。
- 选择同一家族模型时，须核对具体发布版本，尤其 Sieve、AutoTrust、d3 等多版本系列。
- 对未公开的 Teacher 名单、数据量、loss 权重、蒸馏温度和效果消融，本文保留未知，不补成社区共识。
- 版本与主要入口见 [来源记录](sources.json)。本轮未下载模型权重、未查询私有资料、未运行集群实验。
