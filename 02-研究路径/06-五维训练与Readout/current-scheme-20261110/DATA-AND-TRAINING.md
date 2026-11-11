# 数据、Teacher、损失与两阶段训练

[返回总说明](README.md)。数值以 [`CURRENT_STATE.json`](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/CURRENT_STATE.json) 为准；实际源码按新目录约定留在[库外辅助归档](SOURCE-ARCHIVE.md)，[公开文件清单](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/SOURCE_MANIFEST.json)不包含执行代码本体。

## 1. 最小训练单位和标签语义

训练样本的逻辑结构是：

```text
(state, question, criteria_context, candidate, y, source, group)
y ∈ [0,1]：该候选是否正确匹配
```

它不是 language-model next-token 标签，不是“这条数据属于哪个领域”的五类标签，也不是直接预测固定的 255 个任务语义类。

### 1.1 原生任务如何成为 pair

- 互斥选择题：每个候选展开一行，正确候选 y=1，其余 y=0；合法原生软目标保留原值。
- 真假判断：转成完整正反候选的匹配任务；source converter 的具体问题和候选正文必须保留，不能因为都叫“二分类”就擅自替换文字。
- 多工具／多标签：独立候选可以同时为正，不能强制转换为一个互斥目标。
- 来源为评分／偏好时，目标语义取决于原 converter；原标签可能是软标签，不能统称“人类单一正确答案”。

共享 pair 池已经做过全局去重；历史构池可能对冲突目标取均值并仅保留首次来源／group。**所以去重池中的一个 group 不保证还是原生完整题。** 当前新 Language 加入前专门核对了完整性；readout 则直接回到原始完整题，不从该池反拼。

### 1.2 source 与 domain 不是模型输入标签

代码中的 domain 是采样、统计和目标混合用的元数据。模型不会被告知“你正在做 BFCL”或根据 benchmark 名调用特定 prompt。

`pair_data_common.py`（库外源码） 按来源名正则归域，顺序为 Arts → Tools → Retrieval → 逻辑 Knowledge → Language → 一般 Knowledge → 默认 Language。归域本身是工程规则，不能当成数据语义逐行人工验证的证明。

`NO_TRAIN` 显式排除若干已知 benchmark/test-like 名称（含 GPQA、MMLU-Pro、BBH、HLE、PhishNChips、SATA 等）。**来源名过滤不是全库去污染证明，也不证明所有上游没有交叠。**

## 2. 第一段的实际冻结池

| 域 | pair 曝光 | 有效 token | 原硬正／硬负／软标签 |
|---|---:|---:|---|
| Knowledge | 250,000 | 42,371,145 | 62,537／183,666／3,797 |
| Retrieval | 250,000 | 35,539,055 | 18,459／231,528／13 |
| Tools | 250,000 | 109,292,378 | 85,191／159,022／5,787 |
| Arts | 250,000 | 126,891,937 | 82,455／163,160／4,385 |
| Language | 250,308 | 52,891,268 | 本段报告不以未经重算的正负总数填表；unique pair 244,710 |
| **合计** | **1,250,308** | **366,985,783** | 不是相同题数或相同 token 配额 |

前四域是 E24 所用原四域一百万 pair 及有效监督目标的原样复用，不是重新抽的另一百万。该池来源于项目已有 typed/native/生成数据转换和选池资产；不能把全库约千万级库存当成本段实际训练量。

### 2.1 Tools 到底是什么数据

已审计的 selected 250,000 pair：

| 来源 | pair |
|---|---:|
| generated/workflow_policy | 63,014 |
| typed procedural 五类合计 | 82,036 |
| native/glaive | 52,463 |
| overnight/xlam | 36,092 |
| native/when2call | 16,395 |

约 58.02% 是 workflow／程序化来源，约 42% 是列出的原生调用来源。这并非一套完整 BFCL 式“给目录，逐字段判断哪些工具必需”的训练集。

原工具 converter 常把整段工具文档当候选，问题是“下一次 assistant action 需要哪个工具”；正式 BFCL 把目标工具写进问题，候选是 `Invoke this tool.` 和 `Do not invoke this tool.`。时间范围、字段包装和候选语义都不同。详见 [Tools 审计](../../../01-研究结论/04-数据与评测/tools-repair-research-20261110/README.md)。

### 2.2 Language 九任务的真实采样

生产代码：`prepare_five_dimension.py`（库外源码），主要函数 `grouped_rows/native_anli_groups/allocate/select_language`。

| 任务视图 | 实际 pair | 入选独立原题／group | 说明 |
|---|---:|---:|---|
| ANLI，三轮合并 | 94,887 | 31,629 | 每题三个完整关系候选 |
| HellaSwag | 46,952 | 11,738 | 每题四候选，完整组截止略溢出 |
| ACOS laptop | 12,996 | **38** | 一条 review 展开全部 aspect/sentiment 字段，不能把 pair 当 review 数 |
| ACOS restaurant | 9,216 | 256 | 同上 |
| RAGTruth | 28,848 | 14,424 | 回答级是否包含任何幻觉，不是 span-level 全部任务 |
| ContractNLI | 16,521 | 5,507 | 可用完整库存封顶 |
| NLI4CT | 3,368 | 1,684 | 可用完整库存封顶 |
| VAST | 25,020 | 8,340 | 完整三候选 |
| iSarcasm 原生 non-sarcastic | 6,250 | 2,589 unique；3,125 次 group 曝光 | 显式允许重复 |
| iSarcasm 原生 sarcastic | 6,250 | 862 unique；3,125 次 group 曝光 | 显式允许重复 |
| **合计** | **250,308** | 不能跨视图简单当作独立 state 之和 | unique pair 244,710 |

分配算法：

1. 预留 12,500 pair 给英语 iSarcasm 核心 `sarcastic`，原生两类各 6,250 pair。
2. 其它八任务按可用完整原题数的平方根分配 237,500 pair 配额。
3. 库存不足的任务封顶；剩余额度再按相同平方根权重分给未封顶任务。
4. 不放回抽完整 group，达到配额即停止，最后一个完整 group 可造成小幅溢出；不拆组凑数。
5. 只有讽刺平衡部分允许重复完整 group；不混 Arabic，也不混 irony/satire 等其它属性。

**这次“讽刺 5%”是 pair 曝光口径，不是 token 口径。** 历史 35 分语言专科用的是另一种起点和 token 配方，不能互相代替。

#### ANLI 的身份修正

旧 group ID 在 train_r1/r2/r3 之间有碰撞风险；本段按原生文件＋行号、精确 premise/hypothesis、原生标签及完整三候选恢复身份。实际入选 r1/r2/r3 分别 3,286／8,749／19,594 题。

代码使用 `2**31 + state_id` 避免旧 group 命名空间冲突。它是内部索引技巧，不是输入给模型的特征。

#### 一个不能忽视的事实

ACOS laptop 的 12,996 pair 来自仅 38 个 review group。它说明“某任务训练上万行”可能仍只有很少独立场景。**这值得怀疑曝光设计，但本次未做对照，不能直接断言它就是该任务分数不高的根因。**

## 3. pair 的精确文本和训练 token 组装

### 3.1 原样模板

```text
Decide whether the candidate is a correct match.
Input:
{state}
Question:
{question}
Available criteria:
{criteria_context}
Candidate:
{candidate}
```

`Question:` 段只在 question 非空时出现；`Available criteria:` 段只在条件触发时出现，空时整段省略。不是固定插两个空标题。

字符串 `.strip()`；非字符串 state 用 `json.dumps(..., ensure_ascii=False, sort_keys=True)`；None 转空串。

`criteria_context` 触发条件是：任一候选包含 all/none/both/neither…above/other/option/answer/criteria 等引用，或候选数≤8 且问题含 better/preferred/funnier/more helpful 等相对偏好词。实际完整正则见 `pair_data_common.py`（库外源码），不按 benchmark 特例切换。

### 3.2 分词边界

```text
prefix_ids = tokenizer(prefix, add_special_tokens=False)
suffix_ids = tokenizer(candidate, add_special_tokens=False)
training_row = clipped(prefix_ids) + clipped(suffix_ids) + [EOS]
```

**prefix 与 candidate 独立分词再拼接，不是先拼字符串后整体 tokenize。** 不加 chat template，不生成答案 token。更换该边界可能改变 BPE，即使肉眼文本一样也不是同一个输入。

### 3.3 训练会裁剪，正式评测不会

来自 `pair_training_runtime.py`（库外源码） 的 `assemble/clip_tokens`：

```text
L = 4096
reserve = min(len(prefix), min(1024, L//2))
suffix = head_tail_clip(suffix, L-1-reserve)
prefix = head_tail_clip(prefix, L-1-len(suffix))
row = prefix + suffix + [EOS]
head = (budget+1)//2，tail = budget-head
```

这不是把所有训练题都保持完整到 40,960。Teacher 看见的也是该训练输入，而不是原始未裁剪全文。**不能用“正式不截断”掩盖训练时的 head/tail 裁剪。**

collate 右填充，宽度向上取 8 的倍数；pad=151643，EOS=151645，显式 attention mask。有效 token 统计是实际 row 长度之和，不计 padding。

## 4. Teacher 具体怎样打标签

Teacher 是 **PelaAI/KnowLine-31B-Gen1**。新增语言标签复用原生 labeler，不是让聊天模型生成“对／错”字符串后解析。

代码：label_language.py（库外源码）、label-knowline-pairs.py（库外源码）。

1. 用 Student tokenizer 组装上述≤4096 token 的实际训练 row。
2. 只移除末尾 EOS，decode 时 `skip_special_tokens=False`、`clean_up_tokenization_spaces=False`。
3. 该完整 decoded pair 成为 Teacher 的 state，问题为“candidate 是否正确匹配 input/question”。
4. 用 KnowLine 自带 `knowline_layout/knowline_server`，不是自己改写一个相似 chat prompt。
5. 原生 true/false 映射 A/B，发现并绑定实际 tokenizer 的两个 label token ID。
6. 在最后输入位置取 A/B logits，转 FP32，在两标签上 softmax；`P(A)` 即独立 `P(match)`。

**不在同一道题的候选之间对 Teacher q 做 softmax。** 每个 pair 的 q 是独立 Bernoulli 监督，多个候选的 q 不要求和为 1。

| Teacher 设置 | 实际值 |
|---|---|
| 模型类 | Gemma4ForConditionalGeneration，经 AutoModelForImageTextToText 加载，纯文本使用 |
| 隔离 Transformers | 5.12.1 |
| Torch | 2.9.1+cu128 |
| dtype／attention | BF16／SDPA |
| 原生 read-twice | 开；native cutoff=3000 |
| rotation／thinking／header | 仅 rotation 0；thinking off；header false |
| 批上限 | 16 pair；8,192 padded token |
| 独立 workers | 3 个，每进程单卡 |
| 额外温度 | 无；保留发布权重／config 已折叠温度 |
| 超长 singleton | 保留完整 Teacher 输入，不裁剪以塞进批预算 |
| OOM | 拆半重试；单样本仍 OOM 则阻塞，不伪造 hard label |

read-twice 是**原生布局机制**，不是把两次正确率取最大，也不是按结果决定重试。精确布局需 Teacher 自带 helpers；这两件依赖并未在本次发布为自有源码。

worker 按 `global_index = rank + local_index*workers` 分片，NaN 表示 pending；所有项必须有限且在[0,1]。持久化数据后再写进度；merge 逐分片覆盖核对，缺项不准 `complete=true`。

只新增 **244,710** 个 unique Language pair 的标签，再通过 inverse mapping 还原 250,308 次曝光；原四维 q 和有效目标不重算。

## 5. 真正优化的损失

### 5.1 二分类 margin

对候选 i，分类头输出 `l_i[not_match], l_i[match]`：

```text
m_i = float(l_i[1]) - float(l_i[0])
P_pair(match) = sigmoid(m_i)
```

两列先分别转 FP32 再减。不要先 sigmoid 后又把概率当 logit。

### 5.2 原标签与 Teacher 目标混合

设原标签 `y_i∈[0,1]`，Teacher 独立概率为 `q_i`：

```text
t_i = (1-alpha_domain)*y_i + alpha_domain*q_i_effective
point_loss_i = BCEWithLogits(m_i, t_i)
```

| 域 | alpha |
|---|---:|
| Knowledge | 0.5 |
| Retrieval | 0.25 |
| Tools | 0.5 |
| Arts | 0 |
| Language | 0.5 |
| 第二段 generic | 0，只有原标签 |

Tools 的 Teacher 有效概率在历史目标准备时已做：

```text
q_tools_effective = sigmoid(logit(q_tools_raw) - 1.2826099976040561)
```

该偏置已烘焙进有效目标，**第二段不重复施加，正式推理也不加这项 bias**。不要把历史 Teacher 校准常数当作当前 BFCL 字段阈值。

BCE 对 soft target 训练相当于同一 Bernoulli 空间内混合监督，**不是完整题 K-way KL 蒸馏**。不含 LM 辅助 loss，不含 RL/PPO、不含 Brier/RPS 辅助项、不做当前模型逐步 Teacher 追随。

### 5.3 3,647 条原子关系

关系仅来自已审计闭集 Knowledge/Retrieval：同一原题的一个正 pair 和一个负 pair；两端互不重叠、无截断。没有新 Language/Tools ranking 关系。

历史相对 Teacher 目标：

```text
r_teacher = sigmoid(logit(q_pos) - logit(q_neg))
r_mixed = 0.5 + 0.5*r_teacher
relation_loss = BCEWithLogits(m_pos-m_neg, r_mixed)
```

本段直接复用已固定的 `teacher_relation_target.npy`，不重算、不过滤 Teacher 方向。该式来源见[第七波报告](../../../01-研究结论/01-蒸馏/distillation-exploration-20261104/seventh-wave-results/report/REPORT.md)。

设本阶段 N 个 pair，M=3647，对一个实际 macro batch B：

```text
loss_B = [sum(point_loss_i)
          + 0.05*(N/M)*sum(relation_loss_j in B)] / actual_pairs_in_B
```

`N/M` 是全局数量归一化，不能改成 micro 大小或当前 micro 的关系数；最后尾批按实际大小除，不永远除128。关系的正负两行不能拆到两个 microbatch，避免丢掉相对梯度。

### 5.4 组批顺序

`group_training_runtime.py`（库外源码） 把每个关系当二行原子单元，其余当一行：seed42 打乱单元→每8192单元稳定按长度排序→完整装入128行 macro（必要时找 singleton 填缝）→只打乱完整 macro，尾批保留。

每个 selected 索引恰好曝光一次；所谓“一遍”包含采样清单自身允许的显式重复，如讽刺重复。不能把“一遍索引”说成“一遍全部独立原题”。

## 6. 第一段训练

代码：`train_five_dimension.py`（库外源码）。

共同初始化已有[历史初始化记录](../../../03-证据与图表/03-专科与融合/共同初始化/Qwen3-0.6B-common-pair-init-r32/initialization.json)：seed42、Qwen3ForSequenceClassification、rank32/alpha64、保存分类头、当时无模型forward。初始[adapter配置](../../../03-证据与图表/03-专科与融合/共同初始化/Qwen3-0.6B-common-pair-init-r32/adapter_config.json)为vanilla LoRA、dropout0、bias none，无DoRA/RS-LoRA。该元数据并不替代真实共同初始化张量。

| 项目 | 实际配置／完成值 |
|---|---|
| 起点 | 共同**未训练** rank32 LoRA＋分类头；不是 E24 final |
| optimizer | fused AdamW；weight_decay=0.01 |
| LR／schedule | 1e-4；3% warmup 后 cosine |
| warmup 计算 | `max(1,int(total_steps*0.03))`，本段293步 |
| batch | 有效128 pair；micro最多32 pair／16,384 padded token |
| seed／epoch | 42／一遍固定曝光清单 |
| attention | 显式 `flash_attn3_hopper`，FA3；不回退 SDPA/FA2 |
| 数值控制 | Torch strict deterministic、cuDNN deterministic、FA3 deterministic backward；`CUBLAS_WORKSPACE_CONFIG=:4096:8`，`FLASH_ATTENTION_DETERMINISTIC=1`；TF32允许 |
| 梯度 | FP32 累积，clip norm=1.0 |
| checkpoint | 每1000步；保存模型、optimizer、scheduler、CPU/CUDA RNG、采样游标和 attention 证据 |
| 实际终点 | 9,769步；1,250,308 pair；366,985,783 token |
| 实际尾批 | 4 pair（不是旧百万池的64） |
| 实际 OOM retry／重算 | 0／未启用 gradient checkpointing |
| 报告耗时／峰值 allocated | 14,701.19秒；32.842 GiB |

strict deterministic 约束固定环境中的数值路径，不承诺跨库／跨 GPU／跨执行形状的逐位一致。

两段trainer都没有覆盖AdamW的其它默认项，因此按[PyTorch2.9.1固定源码](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/adamw.py#L22-L36)：`betas=(0.9,0.999)`、`eps=1e-8`、`amsgrad=False`、`maximize=False`、`capturable=False`、`differentiable=False`；显式`fused=True`。所有可训参数在同一参数列表中应用0.01 decay，没有另设“head无decay”分组，也没有GradScaler。

确切LR倍率以scheduler索引s、总步T和warmup步W定义：`s<W`时`max(1e-8,s/W)`；否则为`0.5*(1+cos(pi*min(1,(s-W)/max(1,T-W))))`。optimizer更新后调用scheduler.step，不能拿readout研究的`(s+1)/W` warmup替换它；后者是另一条实际数值路径。

## 7. 第二段：真正保留父参数的联合续训

代码：prepare_joint.py（库外源码）、joint_data.py（库外源码）、train_joint.py（库外源码）。

### 7.1 通用 NLI 原标签

canonical 原始 train 为五个 Parquet 分片，读取 1,899,993 行 `premise/hypothesis/labels/task`。

```text
labels=0（entailment）→ match=1
labels=1或2（neutral/contradiction）→ match=0
输入 = render_pair(premise, empty_question, hypothesis)
```

不是新 Teacher 数据，不是普通 next-token 语言训练，也不是榜单第六个计分域。

准入与抽样：

1. 检查非空文本和合法原标签，沿 `NO_TRAIN`／已知来源保护排除。
2. 按规范化 premise/hypothesis 合并同文本；冲突目标排除，不取均值冒充原标签。
3. seed42 不放回排列合法原始记录。
4. 用相同分词／训练裁剪／EOS组装实际模型输入。
5. 排除与五维任务实际模型输入相同的项；在扫描前缀内排除 token-identical 重复／已观察冲突。
6. 按顺序取完整样本前缀，不跳过长样本来凑 token 上限；最后下一整条放不下就停止。

实际：1,851,711 个 unique 文本 pair，5,599 个文本冲突；合法 unique 1,846,112。抽样前缀 tokenize 421,888 行，排除4,679个 task-input overlap、发现27个 token-identical duplicates和5个 token 冲突；最终 **416,934 pair／91,746,130 token**。

上述是明确范围内的精确重复排除，**不是公开／私有评测全库去污染已完成的声明**。

### 7.2 20% 的精确口径

原五维完整回放 366,985,783 token；generic 预算是其四分之一 **91,746,445 token**，实际差315 token即截止。总量458,731,913，generic token 占 **19.9999449%**。

但 generic 占 pair 数 **约25.01%**，而 point BCE 按 pair 平均。**20% token 配额不等于20% loss 权重，更不等于五域各等权。** 原五维绝对曝光没有被减少，却仍不保证函数能力不会变化。

### 7.3 参数加载和优化

在 PEFT 加载前 `base.score.float()`，避免把已保存 FP32 分类头先复制进 BF16 目标而舍入。加载后对规范 state dict 的全部 key／dtype／tensor 逐项相等核验，真实准入 **393 张量**；不重置 LoRA/head。

重新建立 AdamW 和 LR schedule，不继承上一段 optimizer 动量；这叫“保参数续训”，不是“优化器无缝接着走”。

| 项目 | 第二段 |
|---|---|
| 起点 | 本次已完成五维 KD final 的独立快照 |
| 新增数据 | 原五维全部再一遍＋generic 原标签 |
| LR | 2e-5 |
| optimizer／decay／warmup | fused AdamW／0.01／3%，本段390步 |
| batch／micro／seed | 128／32或16,384 padded token／42 |
| attention／精度／loss | 与第一段同机制；N改为本段1,667,242，M仍3647 |
| 实际终点 | **13,026步；1,667,242 pair；458,731,913 token** |
| 尾批／关系曝光 | 42 pair／全部3647关系 |
| OOM／gradient checkpointing | 0／false |
| 报告耗时／峰值 allocated | 19,410.81秒／32.843 GiB |

最终单独导出 `JOINT_HYBRID/final`，写保护后用于官方冻结快照。只有该段完成后的版本被跑了旧版与加速版 public；**没有 KD-only 的正式分数，不能把两段收益拆开归因。**

## 8. 未执行、未证明与恢复边界

- 训练完成依据是真实计数、关系曝光、数值执行证据和完成回执，不是代码评审结论。
- 本次文档整理没有运行训练／测试／smoke／额外 forward 或模型／语料 hash。
- 旧扩大原标签训练仍暂停，checkpoint／optimizer／RNG保留；它不是当前主线的额外训练步数。
- 没有本轮跨 seed 方差、训练预算消融、generic 边际对照或独立机制归因。
- 本文公式、算法步骤和配置摘录说明实际方案；执行源码另存库外，完整数据池、冻结token streams、Teacher helpers、共同初始化和权重也未随本次Git发布。人工复现的精确缺口见[REPRODUCTION.md](REPRODUCTION.md)，不把这些缺件隐含交给读者或AI猜。
