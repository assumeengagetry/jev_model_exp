# 数据、Teacher、loss 与两段训练

[返回总览](README.md)。数值以 [`CURRENT_STATE.json`](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/CURRENT_STATE.json) 为准；文中提到的 `.py` 文件均在[库外辅助归档](SOURCE-ARCHIVE.md)，[公开清单](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/SOURCE_MANIFEST.json)只列文件名。

## 结论

- **训练单位是 pair，loss 是二元 BCE。** 标签是“该候选是否匹配”的 [0,1] 值，Teacher 概率按域以 0–0.5 的比例混入目标，另有 3,647 条 Knowledge/Retrieval 关系项。
- **第一段五域各约 25 万 pair，token 数却相差约 3.6 倍。** 共 125 万 pair、367.0M token，从共同未训练的 LoRA/head 起步。
- **第二段保留第一段参数，五维再训一遍并加入 41.7 万 generic NLI pair。** generic 占 token 的 20%，占 pair 的 25%。
- **训练输入裁剪到 4,096 token，正式推理不裁剪**（上限 40,960）。Teacher 看到的也是裁剪后的输入。
- **只有第二段终点跑过 public。** 第一段和 generic 各自贡献多少，现有结果拆不开。

## 1. 训练样本是 pair，标签是“候选是否匹配”

```text
(state, question, criteria_context, candidate, y, source, group)
y ∈ [0,1]：该候选是否正确匹配
```

标签既不是 next-token，也不是领域分类。原生任务按下表转成 pair：

| 原生任务 | 转换方式 |
|---|---|
| 互斥选择题 | 每个候选一行，正确候选 y=1，其余 y=0；原生软目标保留原值 |
| 真假判断 | 转成正、反两个完整候选的匹配任务，问题和候选正文沿用 source converter 的原文 |
| 多工具、多标签 | 各候选独立判断，可以同时为正 |
| 评分、偏好 | 目标含义由原 converter 决定，可能是软标签 |

共享 pair 池做过全局去重，历史构池时目标冲突的行可能取了均值，并只保留首次出现的来源和 group。因此池中的一个 group 不一定还是完整原题。Language 入池前专门核对了完整性；readout 研究则直接回到原始完整题。

domain 只是采样、统计和目标混合用的元数据，不输入模型。`pair_data_common.py` 按来源名正则归域，顺序为 Arts → Tools → Retrieval → 逻辑 Knowledge → Language → 一般 Knowledge → 默认 Language。`NO_TRAIN` 按来源名排除已知的评测类数据（GPQA、MMLU-Pro、BBH、HLE、PhishNChips、SATA 等）；这是名称过滤，没有做全库去污染核查。

## 2. 第一段：五域 pair 数接近，token 数相差约 3.6 倍

| 域 | Pair | Token | 硬正 / 硬负 / 软标签 |
|---|---:|---:|---|
| Knowledge | 250,000 | 42.4M | 62.5k / 183.7k / 3.8k |
| Retrieval | 250,000 | 35.5M | 18.5k / 231.5k / 13 |
| Tools | 250,000 | 109.3M | 85.2k / 159.0k / 5.8k |
| Arts | 250,000 | 126.9M | 82.5k / 163.2k / 4.4k |
| Language | 250,308 | 52.9M | 汇总未提供；unique pair 244,710 |
| **合计** | **1,250,308** | **367.0M** | — |

前四域原样复用 E24 的一百万 pair 及其有效监督目标，没有重新抽样。Token 最多的 Arts 是最少的 Retrieval 的约 3.6 倍；Retrieval 的硬正例只占约 7%。

### 2.1 Tools 的训练数据以 workflow 和程序化来源为主，问法与 BFCL 不同

| 来源 | Pair |
|---|---:|
| generated/workflow_policy | 63,014 |
| typed procedural（五类合计） | 82,036 |
| native/glaive | 52,463 |
| overnight/xlam | 36,092 |
| native/when2call | 16,395 |

约 58% 来自 workflow 和程序化来源，约 42% 来自原生调用数据。训练中的问法和正式 BFCL 不一致：

| | 训练中的 Tools pair | 正式 BFCL 字段 |
|---|---|---|
| 问题 | 下一次 assistant action 需要哪个工具 | 目标工具写在问题里，问是否应该调用 |
| 候选 | 常为整段工具文档 | `Invoke this tool.` / `Do not invoke this tool.` |

两者的时间范围和字段包装也不同，详见 [Tools 审计](../../../01-研究结论/04-数据与评测/tools-repair-research-20261110/README.md)。

### 2.2 Language 九任务按原题数的平方根分配配额

采样代码是 `prepare_five_dimension.py`，主要函数为 `grouped_rows/native_anli_groups/allocate/select_language`。

| 任务 | Pair | 入选 / 可用原题 | 平均每题 pair | 说明 |
|---|---:|---:|---:|---|
| ANLI（三轮合并） | 94,887 | 31,629 / 162,601 | 3 | 三个关系候选 |
| HellaSwag | 46,952 | 11,738 / 39,811 | 4 | 完整组截止，比配额多 1 pair |
| ACOS laptop | 12,996 | 38 / 2,919 | 342 | 一条 review 展开全部 aspect/sentiment 字段 |
| ACOS restaurant | 9,216 | 256 / 1,526 | 36 | 同上 |
| RAGTruth | 28,848 | 14,424 / 15,029 | 2 | 回答级：是否含幻觉 |
| ContractNLI | 16,521 | 5,507 / 5,507 | 3 | 库存用尽 |
| NLI4CT | 3,368 | 1,684 / 1,684 | 2 | 库存用尽 |
| VAST | 25,020 | 8,340 / 11,305 | 3 | 三个候选 |
| iSarcasm non-sarcastic | 6,250 | 2,589 / 2,589 | 2 | 允许重复，3,125 次 group 曝光 |
| iSarcasm sarcastic | 6,250 | 862 / 862 | 2 | 允许重复，3,125 次 group 曝光 |
| **合计** | **250,308** | — | — | unique pair 244,710 |

分配步骤：

1. 预留 12,500 pair 给英语 iSarcasm 核心 `sarcastic` 属性，原生两类各 6,250。
2. 其余八个任务按“可用完整原题数的平方根”分配 237,500 pair 的配额。
3. 库存不足的任务封顶，余额按相同的平方根权重分给未封顶任务。
4. 不放回地抽取完整 group，达到配额即停；最后一个完整 group 可以小幅超出配额，不拆组。
5. 只有 iSarcasm 允许重复抽取完整 group。不混入 Arabic，也不混入 irony、satire 等其他属性。

需要留意的三点：

- **ACOS laptop 的独立场景很少。** 12,996 个 pair 只来自 38 条 review，每条展开 342 个 pair，而可用 review 有 2,919 条。这是否拖累该任务的得分，没有做对照。
- **讽刺 5% 按 pair 计**（12,500 / 250,308）。历史 35 分的 Language 专科按 token 配比，起点也不同，两套配方不能互相代替。
- **sarcastic 类只有 862 个原题**，3,125 次曝光意味着平均每题重复约 3.6 次。

ANLI 的旧 group ID 在 train_r1/r2/r3 之间会碰撞。本段按原生文件名 + 行号、premise/hypothesis 原文、原生标签和完整三候选恢复身份，r1/r2/r3 分别入选 3,286、8,749、19,594 题。group ID 用 `2**31 + state_id` 避开旧命名空间，它只是内部索引，不输入模型。

## 3. 输入：固定模板，prefix 与 candidate 分开分词，训练时裁剪到 4,096

### 3.1 模板

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

- `Question:` 段只在 question 非空时出现；`Available criteria:` 段只在触发条件满足时出现，否则整段省略。
- 字符串做 `.strip()`；非字符串 state 用 `json.dumps(..., ensure_ascii=False, sort_keys=True)`；None 转为空串。
- `criteria_context` 在两种情况下加入：任一候选含 all/none/both/neither…above/other/option/answer/criteria 等引用词；或候选数 ≤ 8 且问题含 better/preferred/funnier/more helpful 等相对偏好词。完整正则在 `pair_data_common.py`，不按 benchmark 切换。

### 3.2 分词边界

```text
prefix_ids = tokenizer(prefix, add_special_tokens=False)
suffix_ids = tokenizer(candidate, add_special_tokens=False)
training_row = clipped(prefix_ids) + clipped(suffix_ids) + [EOS]
```

prefix 和 candidate 各自分词后再拼接，不加 chat template。如果先拼字符串再整体分词，边界处的 BPE 结果可能不同，得到的就不是同一个输入。

### 3.3 训练裁剪

`pair_training_runtime.py` 的 `assemble/clip_tokens`：

```text
L = 4096
reserve = min(len(prefix), min(1024, L//2))
suffix = head_tail_clip(suffix, L-1-reserve)
prefix = head_tail_clip(prefix, L-1-len(suffix))
row = prefix + suffix + [EOS]
head = (budget+1)//2，tail = budget-head
```

先给 prefix 预留最多 1,024 token，candidate 在剩余预算内保留首尾、去掉中段；再用 candidate 裁完后剩下的预算裁 prefix。正式评测不做这一步裁剪，完整输入最长 40,960 token。

collate 时右填充，宽度向上取 8 的倍数，pad = 151643，EOS = 151645，使用显式 attention mask。有效 token 按实际 row 长度求和，不含 padding。

## 4. Teacher 对每个 pair 独立给出 P(match)

Teacher 是 `PelaAI/KnowLine-31B-Gen1`，打标代码为 `label_language.py` 和 `label-knowline-pairs.py`。流程：

1. 用 Student tokenizer 组装上述不超过 4,096 token 的训练 row。
2. 去掉末尾 EOS 后 decode，`skip_special_tokens=False`、`clean_up_tokenization_spaces=False`。
3. 把 decode 出的完整 pair 作为 Teacher 的 state，问题是“candidate 是否正确匹配 input/question”。
4. 使用 KnowLine 自带的 `knowline_layout/knowline_server` 构造输入，不自行改写 chat prompt。
5. 原生 true/false 映射到 A/B，绑定 tokenizer 中这两个 label token 的实际 ID。
6. 在最后一个输入位置取 A/B 的 logits，转 FP32 后在两个标签上 softmax，`P(A)` 即 `P(match)`。

每个 pair 的 q 是独立的 Bernoulli 概率，同一题各候选的 q 不要求和为 1。

| Teacher 设置 | 实际值 |
|---|---|
| 模型类 | Gemma4ForConditionalGeneration，经 AutoModelForImageTextToText 加载，只用文本 |
| Transformers / Torch | 5.12.1（独立环境）/ 2.9.1+cu128 |
| Dtype / attention | BF16 / SDPA |
| Read-twice | 开，native cutoff = 3000 |
| Rotation / thinking / header | 仅 rotation 0 / off / false |
| 批上限 | 16 pair，8,192 padded token |
| Workers | 3 个，每进程单卡 |
| 额外温度 | 无，沿用发布权重和 config 中已折叠的温度 |
| 超长单样本 | 保留完整输入，不为塞进批预算而裁剪 |
| OOM | 拆半重试；单样本仍 OOM 则阻塞，不用 hard label 代替 |

read-twice 是 KnowLine 原生的输入布局机制，不涉及按结果重试或取两次中的较好者；精确布局依赖 Teacher 自带的 helpers。

worker 按 `global_index = rank + local_index*workers` 分片，NaN 表示尚未完成，所有结果必须是有限值且落在 [0,1]。merge 时逐分片核对覆盖，缺项时不会标记 `complete=true`。

本次只给 244,710 个 unique Language pair 新打了标签，再通过 inverse mapping 还原为 250,308 次曝光。原四维的 q 和有效目标没有重算。

## 5. Loss：按域混合原标签与 Teacher 概率的 BCE，加 3,647 条关系项

### 5.1 margin

分类头对候选 i 输出 `l_i[not_match], l_i[match]`：

```text
m_i = float(l_i[1]) - float(l_i[0])
P_pair(match) = sigmoid(m_i)
```

两列先分别转 FP32 再相减。

### 5.2 目标混合

原标签 `y_i ∈ [0,1]`，Teacher 概率 `q_i`：

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
| generic（第二段） | 0 |

alpha 为 0 表示只用原标签，Arts 和 generic 属于这种情况。

Tools 的 Teacher 概率在历史目标准备时做过一次截距校准：

```text
q_tools_effective = sigmoid(logit(q_tools_raw) - 1.2826099976040561)
```

这个偏置已经写进有效目标。第二段不再重复施加，正式推理也不加。

这套 loss 是 pair 级的 soft-target BCE，不是整题 K-way KL 蒸馏。没有 LM 辅助 loss，没有 RL/PPO，也没有 Brier/RPS 项。

### 5.3 关系项

关系来自已审计的 Knowledge/Retrieval 闭集：同一原题的一个正 pair 和一个负 pair，两端不重叠、无截断，共 3,647 条。Language 和 Tools 没有关系项。

```text
r_teacher = sigmoid(logit(q_pos) - logit(q_neg))
r_mixed = 0.5 + 0.5*r_teacher
relation_loss = BCEWithLogits(m_pos-m_neg, r_mixed)
```

本段直接复用已固定的 `teacher_relation_target.npy`，公式来源见[第七波报告](../../../01-研究结论/01-蒸馏/distillation-exploration-20261104/seventh-wave-results/report/REPORT.md)。

设本阶段共 N 个 pair、M = 3647，一个 macro batch B 的 loss 为：

```text
loss_B = [sum(point_loss_i)
          + 0.05*(N/M)*sum(relation_loss_j in B)] / actual_pairs_in_B
```

`N/M` 是全局数量归一化，与 microbatch 大小无关；尾批按实际 pair 数除。一条关系的正负两行必须在同一个 microbatch 内，否则相对梯度会丢失。

### 5.4 组批

`group_training_runtime.py` 把每条关系当作两行的原子单元，其余 pair 各为一行：

1. seed 42 打乱全部单元。
2. 每 8,192 个单元内按长度稳定排序。
3. 完整装入 128 行的 macro batch，必要时用单行单元填缝。
4. 只打乱完整的 macro batch，尾批保留在最后。

每个入选索引恰好曝光一次。采样清单自身包含的显式重复（如 iSarcasm）照常计入，所以“一遍”指索引清单的一遍。

## 6. 两段训练：第一段 LR 1e-4 从头训，第二段 LR 2e-5 保参数续训

两段共用以下配置：

- **Optimizer**：fused AdamW，weight_decay 0.01，其余为 [PyTorch 2.9.1 默认值](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/adamw.py#L22-L36)（`betas=(0.9,0.999)`、`eps=1e-8`）。全部可训参数在同一组内使用同一 decay，没有 GradScaler。
- **Schedule**：3% warmup 后 cosine，warmup 步数为 `max(1, int(total_steps*0.03))`。
- **Batch**：effective batch 128 pair；microbatch 最多 32 pair 或 16,384 padded token。
- **Seed / epoch**：42 / 固定曝光清单一遍。
- **Attention**：显式 `flash_attn3_hopper`（FA3），不回退 SDPA 或 FA2。
- **确定性**：Torch strict deterministic、cuDNN deterministic、FA3 deterministic backward；`CUBLAS_WORKSPACE_CONFIG=:4096:8`，`FLASH_ATTENTION_DETERMINISTIC=1`；允许 TF32。
- **梯度**：FP32 累积，clip norm 1.0；gradient checkpointing 关闭。
- **Checkpoint**：每 1,000 步保存模型、optimizer、scheduler、CPU/CUDA RNG、采样游标和 attention 证据。

| 项目 | 第一段（五维 KD） | 第二段（joint） |
|---|---:|---:|
| 起点 | 共同未训练 rank32 LoRA + 分类头 | 第一段 KD final 的 393 个张量 |
| 数据 | 五维一遍 | 五维再一遍 + generic 原标签 |
| Pair | 1,250,308 | 1,667,242 |
| Token | 367.0M | 458.7M |
| Peak LR | 1e-4 | 2e-5 |
| Warmup 步数 | 293 | 390 |
| Optimizer 步数 | 9,769 | 13,026 |
| 尾批 | 4 pair | 42 pair |
| 关系曝光 | 3,647 | 3,647 |
| 耗时 | 约 4.1 小时 | 约 5.4 小时 |
| Token/s | 25.0k | 23.6k |
| Peak allocated | 32.8 GiB | 32.8 GiB |
| OOM 重试 | 0 | 0 |

代码分别是 `train_five_dimension.py` 和 `train_joint.py`。

**第一段的起点。** 共同初始化有[历史记录](../../../03-证据与图表/03-专科与融合/共同初始化/Qwen3-0.6B-common-pair-init-r32/initialization.json)：seed 42、rank32/alpha64、保存分类头。[adapter 配置](../../../03-证据与图表/03-专科与融合/共同初始化/Qwen3-0.6B-common-pair-init-r32/adapter_config.json)为 vanilla LoRA、dropout 0、bias none，不含 DoRA 或 RS-LoRA。这些是元数据；精确重训需要原始的初始化张量。

**第二段的参数加载。** 加载 PEFT 前先执行 `base.score.float()`，避免 FP32 的分类头被复制进 BF16 目标时舍入。加载后对全部 393 个张量逐项核对 key、dtype 和数值。AdamW 和 LR schedule 重新建立，不继承第一段的 optimizer 动量。

**LR 倍率。** 设 scheduler 索引为 s、总步数 T、warmup 步数 W：

```text
s < W： max(1e-8, s/W)
s ≥ W： 0.5*(1+cos(pi*min(1,(s-W)/max(1,T-W))))
```

optimizer 更新后再调用 `scheduler.step`。readout 研究用的是 `(s+1)/W`，属于另一条数值路径。

strict deterministic 约束的是固定环境内的数值路径，换库、换 GPU 或换执行形状后不保证逐位一致。

## 7. 第二段的 generic：41.7 万 NLI pair，占 token 20%、占 pair 25%

代码为 `prepare_joint.py`、`joint_data.py`。数据来自 canonical NLI 原始 train 的五个 Parquet 分片，只用原标签：

```text
labels=0（entailment）           → match=1
labels=1 或 2（neutral/contradiction）→ match=0
输入 = render_pair(premise, empty_question, hypothesis)
```

generic 不经过 Teacher，也不是榜单的第六个计分域。

| 筛选步骤 | 数量 |
|---|---:|
| 原始 train 行 | 1,899,993 |
| 文本为空、标签非法或来源被排除 | 136 |
| 按规范化 premise/hypothesis 合并后的 unique pair | 1,851,711 |
| 目标冲突，整体排除 | 5,599 |
| 合法 unique pair | 1,846,112 |
| seed 42 排列后实际 tokenize 的前缀 | 421,888 |
| 排除：与五维任务的模型输入相同 | 4,679 |
| 排除：token 完全相同的重复 | 27 |
| 排除：token 层面目标冲突 | 5 |
| **最终入选** | **416,934 pair，91.7M token** |

抽样按 seed 42 的排列顺序取完整样本，不为凑预算跳过长样本，下一条放不下就停止。已 tokenize 的前缀里可用样本合计 91.8M token，略高于预算，所以最终入选数少于前缀内的合格数。去重范围是原始 train 全量的文本冲突，以及抽样前缀内与五维任务的输入重叠；没有做公开或私有评测的全库去污染。

generic 的 token 预算是五维回放量的四分之一：366,985,783 / 4 ≈ 91,746,445，实际用到 91,746,130。它在第二段中占 token 的 20.0%，占 pair 的 25.0%。point BCE 按 pair 平均，所以 generic 在 loss 里的权重按 pair 计，约为 25%。

第二段结束后单独导出 `JOINT_HYBRID/final`，写保护后用于官方冻结快照。

## 8. 尚未做的对照

- 没有第一段终点（KD-only）的 public 分数，两段的收益拆不开，generic 的边际贡献未知。
- 没有跨 seed 重复，也没有训练预算消融。
- 扩大原标签训练仍然暂停，checkpoint、optimizer 和 RNG 保留；它不计入当前主线的训练量。
- 完整数据池、token streams、Teacher helpers、共同初始化和权重没有随 Git 发布，复现缺口见 [REPRODUCTION.md](REPRODUCTION.md)。
