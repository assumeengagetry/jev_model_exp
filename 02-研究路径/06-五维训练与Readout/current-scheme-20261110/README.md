# 当前方案，从零讲清楚

**结果截止：2026-11-10；证据重新读取时间见 [`CURRENT_STATE.json`](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/CURRENT_STATE.json) 的 `observed_utc`。**

本文假定读者没有参与过项目讨论，也不会让 AI 补全缺失步骤。出现一个缩写就解释它；出现一个结果就说明它的来源和适用范围；尚无证据的机制不写成事实。

## 1. 先用一句话说明我们做了什么

我们训练了一个“小型文本裁判”：把“当前材料＋问题＋一个候选答案”交给 Qwen3-0.6B，让它输出该候选“匹配”和“不匹配”的两个分数；一题有几个候选，就得到几个匹配分数，最后在本题全部候选之间归一化成概率。

**它不是聊天生成模型的常规使用方式。** 正式推理不逐 token 写答案、不调用 `generate()`、不输出解释或思维链，也不是部署时请一个 31B Teacher 帮忙回答。Teacher 只在训练数据准备阶段提供监督概率。

它也**不是 TypeSafe 闭源 Jev 的架构复现**。我们实现的是满足 Decision Index 输入输出约定的一种开源候选匹配路线；不能因为接口相似就推断内部结构相同。

## 2. 输入、问题和输出分别是什么

### 2.1 三个层次不能混

- **请求（request）：** 一个 `state` 加一组问题；同一 state 可有很多字段。
- **完整题（question/field）：** 一个问题及它的全部候选和目标。
- **候选对（pair）：** 从一题中取一个候选，形成一次匹配判断的训练／推理输入。

例如，一题四选一会变成四个 pair。另一题判断五个工具是否适用，可以是五个相互独立的真假字段；**不能把五个可同时成立的字段硬变成“只能选一个工具”**。

因此，“训练一百万 pair”不等于“一百万独立问题”，也不等于“一百万独立 state”。报告中的 token 是实际组装后的输入 token 曝光量，不是字数、独立信息量或去重语料规模。

### 2.2 不使用真实评测题的示意

```json
{
  "state": "包裹已付款，但尚未发货。",
  "questions": {
    "status": {
      "type": "choice",
      "instructions": "包裹现在处于哪个状态？",
      "criteria": {
        "paid": "已付款，待发货",
        "delivered": "已送达"
      }
    }
  }
}
```

模型分别判断两个候选。假设两个匹配 margin 是 2 和 −1，输出是 `softmax([2,-1])≈[0.9526,0.0474]`，选择 `paid`。这些数只是解释计算的人工示意，不是模型实测。

### 2.3 支持边界

当前正式 Engine 直接接受 `choice` 与 `noul`（真假）字段；其它直接传入的 type 会明确 `Unsupported`。内部 readout 研究能保留原生 `score` 等级并做分类分布，这不意味着正式 Engine 突然多了一个原生 `score` 分支。

固定官方套件的实际请求已经全部成功处理，不能把这件事外推为“任意外部 JSON/type 都受支持”。

## 3. 当前有哪几套东西

| 名称 | 参数起点／变化 | 主要作用 | 已有结果 |
|---|---|---|---|
| 旧四维 E24／受控 E31 | 原始 Qwen＋共同未训练 LoRA/head，K/R/Tools/Arts | 历史蒸馏参照 | E24 单次 23.96，重跑 20.51；受控 mixed 22.49 |
| 五维 KD final | 同一个共同未训练初始化，补原生 Language 重新联合训练 | 当前权重的第一段 | 训练完成；没有单独正式 public 分数 |
| **joint final** | **保留上述 KD final，再训通用原标签＋原五维 KD** | **当前正式权重** | **旧 Engine 24.97；新 Engine 24.99** |
| 原标签 stage2 final | 另一条通用分类→五维原标签训练路线 | readout 研究起点 | 没有本轮完整 public 分数 |
| 暂停的扩大原标签训练 | 从原标签 stage2 再扩大训练 | 已暂停保留 | 不是当前正式权重，不自动恢复 |
| Readout frozen／LoRA 四臂 | 从原标签 stage2 独立适配 A/Bcopy/Brand/Cptr | 比较整题读出 | 内部留出集，非官方 Index |

**关键区别：** 当前主线第一段不是“在 E24 23.96 权重上补语言”，而是“复用其数据与配方，从共同未训练初始化重新训练”。第二段 joint 才是真正保留第一段参数继续训练。复用数据不能保证复用已经学到的函数。

## 4. 模型里究竟有什么

### 4.1 骨干

实际模型是 `Qwen3ForSequenceClassification`，不是因果语言模型出口取 A/B 字母的正式方案。参数配置来自已冻结正式模型，而非凭模型名猜测：

| 配置 | 实际值 |
|---|---|
| 模型族 | Qwen3-0.6B |
| Transformer 层数 | 28 |
| hidden 大小 | 1,024 |
| MLP intermediate | 3,072 |
| attention heads／KV heads | 16／8 |
| head dimension | 128 |
| vocab | 151,936 |
| 声明上下文 | 40,960 token／完整候选序列 |
| attention | 全层 full attention，**causal**；不是 bidirectional |
| RoPE theta | 1,000,000；无额外 rope scaling |
| pad／EOS | 151643／151645，二者不同 |
| 输出类别 | 0=`not_match`，1=`match` |

“full attention”在这里表示非滑窗 attention 层类型，**不表示取消 causal mask**。

基座来自当时实际下载的 ModelScope `Qwen/Qwen3-0.6B`，历史来源标记为 `master`，正式评测另存实际基座快照。**没有核实它等于某个 HF 不可变 commit**，所以不能用今天的 HF main 悄悄替换并声称完全相同。

### 4.2 LoRA 与分类头

LoRA 是在原线性投影旁加入可训练低秩分支：

```text
原投影 W(x) + (alpha/r) × B(A(x))
r=32，alpha=64，因此缩放为 2
```

每层的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` 都有该分支，共 **28×7=196 个投影**，每个两块低秩参数，即 392 个 LoRA 张量。另有训练后的 `score.weight`，形状 **[2,1024]**，不带 bias；规范导出共 **393 个张量**。

- 原骨干权重不做全参数更新；训练 LoRA 和二分类头。
- 低秩和头的可训练参数用 FP32 存储，骨干 BF16。
- 正式推理不 merge LoRA，不量化，不换成词表头。
- `modules_to_save` 中确实保存了分类头；导出配置出现重复的 `score/classifier` 名称，**不能据此推断有多个实际分类头**，实际张量清单才是准据。

**FP32 权重存储不等于 FP32 分类投影。** 当前正式路径的整个分类 forward 在 BF16 autocast 里。readout 研究则另行明确使用 FP32 投影；两者不能混称完全同数值路径。

## 5. 从原料到当前权重的完整流程

```text
公开 train 来源／原生任务转换
  ├─ 原四维固定池：每维 250,000 pair
  └─ 原生语言九任务：250,308 pair（244,710 unique）
             ↓
KnowLine Teacher：复用原四维标签，只新增语言标签
             ↓
原标签与 Teacher 概率按域混合＋3,647 个 K/R 原子关系
             ↓
第一段：共同未训练 LoRA/head → 五维 KD final
  1,250,308 pair，366,985,783 token，9,769 步
             ↓ 保留全部父 LoRA/head，重新建立优化器
第二段：完整回放原五维＋canonical NLI 原始标签
  1,667,242 pair，458,731,913 token，13,026 步
             ↓
joint final（当前唯一正式权重）
  ├─ 原正式 Engine：完整 public 24.97，9h20m
  └─ prefix＋CUDA Graph Engine：完整 public 24.99，3h22m
```

两段合计 **2,917,550 次 pair 曝光、825,717,696 个有效输入 token、22,795 个优化步**。其中五维数据被训练两遍；这些合计不是去重样本数，也不包含此前历史实验消耗。

[数据与训练全文](DATA-AND-TRAINING.md) 写清楚每一种标签、每一个系数和实际采样数量。[推理与评测全文](INFERENCE-AND-EVALUATION.md) 则从 pair 如何转回一题分布开始。

## 6. 为什么训练 loss 变好，不保证榜单多几分

我们实际优化的是“这个候选是否匹配”的二元 BCE，另有少量 K/R 相对关系；榜单却综合了整题选择、多字段 case-exact、F1、相关性和偏好等不同指标。

- 某个工具 case 只有一个字段判断错，整 case 可能就错；pair 平均 loss 不直接等价于该目标。
- pair 数量与独立题数不同，长候选／多字段题可能贡献大量训练行。
- 五域 pair 数接近相等，**token 不相等，独立问题数也不相等**。
- 全量模型能力不等于某一语言专科：新任务训练会改变共享参数，旧能力没有自动“冻结在原地”。

这不是证明 BCE 路线必然无效，而是解释为什么“补齐数据数量”与“补齐目标能力”不是同一个动作。

## 7. 现在真正知道的结果

当前最新 public 是 **24.99**，不是 28，也不是 private Full。两次正式 public 使用同一权重，0.02 差异不能解读成训练能力提升。当前缺分的详细账目在 [瓶颈分析](../../../01-研究结论/06-五维训练与Readout/current-scheme-20261110/BOTTLENECKS.md)。

Readout 研究已经完整结束，Cptr 在本轮五域归一 NLL 上最佳，但换序一致率约 79.33%，远低于原 pair 98.67%。它提供后续研究线索，**不是已经得到一个可替换正式模型的更好方案**。

## 8. 证据与阅读顺序

| 要回答的问题 | 文档／机器证据 |
|---|---|
| 数据、监督、采样和训练到底是什么 | [DATA-AND-TRAINING.md](DATA-AND-TRAINING.md) |
| 一次请求如何得到输出 | [INFERENCE-AND-EVALUATION.md](INFERENCE-AND-EVALUATION.md) |
| 为什么没到 28，哪些解释成立 | [BOTTLENECKS.md](../../../01-研究结论/06-五维训练与Readout/current-scheme-20261110/BOTTLENECKS.md) |
| 人工如何复现，有哪些缺件 | [REPRODUCTION.md](REPRODUCTION.md) |
| 当前真实配置、完成计数 | [CURRENT_STATE.json](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/CURRENT_STATE.json) |
| 本次归档的实际源文件 | [公开清单](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/SOURCE_MANIFEST.json)、[库外源码位置与依赖](SOURCE-ARCHIVE.md) |
| 加速版原生分数和比较证据 | [results/](../../../03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results) |
| Readout 所有臂的聚合结果 | [最终报告](../../../01-研究结论/06-五维训练与Readout/readout-study-20261108/FINAL-REPORT-20261110.md) |

实际源码来自只读读取的执行目录，UTF-8字节逐一对照后，按新研究库约定保留在**库外辅助归档，不随本次Git推送**；上面提供可公开的文件清单，文中“库外源码”均按此定位。配置JSON做了路径脱敏及字段摘录。没有为写文档重新训练、forward、跑public或做额外模型／语料hash，也没有把模型或原始题目加入Git。
