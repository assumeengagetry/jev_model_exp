# Qwen3-0.6B 蒸馏：哪些配方有效，当前保留什么配置

**当前完整公共评测最好为 23.2 分，配方是“来源监督＋Teacher 软标签混合、Tools 校准、分领域控制 Teacher 权重”，比普通混合监督提高 0.5 分，比仅用来源监督提高 1.6 分。** 但在配平的固定子集实验中，普通 label smoothing 仍优于已测 KD；Teacher 独有信息是否带来额外收益，证据还不够。

Student 为 Qwen3-0.6B，Teacher 为 KnowLine-31B。HARD 表示使用来源标签，SOFT 表示只学习 Teacher 概率，HYBRID 表示两者混合；来源中的合法软标签均保留。除 warm-start 预算对照外，主要实验从原始模型和共同未训练分类头开始，各训练 100 万 pair。完整公共评测这组配方的训练池为 Knowledge、Retrieval、Tools、Arts 各 25 万，Language 不入训；固定子集组使用其自身配平的训练池。

本文保留两种评测范围：**完整公共评测**为 37 项、约 11 万请求；**固定子集评测**为同 37 项中的 4,786 个完整 case、5,935 请求。各表内按相同范围比较，不将两种分数混排。结果来自 seed42 的探索，尚无多 seed 复现；Skill／Index 不是 accuracy。

## 1. 分领域混合与 Tools 校准有效，统一降权和全域校准没有进一步收益

完整公共评测中，纯 SOFT 低于 HARD，HYBRID 则有提升。在此基础上，分别保护易退化领域、只校准 Tools，得到当前最好的组合。

| 监督与校准配方 | 公共 Index |
|---|---:|
| HARD：来源监督 | 21.6 |
| SOFT：仅 Teacher | 20.6 |
| HYBRID：来源／Teacher 各 0.5 | 22.6 |
| Teacher 权重统一降至 0.25 | 22.0 |
| 按样本可靠性降低 Teacher 权重 | 21.9 |
| 全领域 Teacher 截距校准 | 22.0 |
| 分领域控制 Teacher 权重 | 22.9 |
| 仅校准 Tools，保留原混合权重 | 22.8 |
| **Tools 校准＋分领域权重** | **23.2** |
| 上述组合，Tools 校准强度减半 | 21.3 |
| 上述组合，Tools 校准强度提高至 1.5 倍 | 22.6 |

分领域配方将 Knowledge／Tools 的 Teacher 权重设为 **0.5**，Retrieval 为 **0.25**，Arts 为 **0**。Tools 的校准截距约 **−1.28**，由训练来源标签与 Teacher 概率拟合。

在分领域权重上加入 Tools 校准，总分再提高 **0.3**。相应地，Tools、Retrieval、Arts 分别提高 **2.1、1.6、1.0** 分，但 Language、Knowledge 下降 **1.6、0.5** 分。进一步减弱或增强校准均未涨分，保留原校准强度。

## 2. 混合监督超过 HARD，但目前没有超过普通 label smoothing

### 原始起点：无 Teacher 的 LS0.1 是当前最强子集对照

下表固定起点、100 万 pair、样本顺序和训练量，只改变监督目标。

| 监督配方 | 固定子集 Index |
|---|---:|
| HARD | 18.5 |
| SOFT | 17.5 |
| HYBRID，Teacher 权重 0.5 | 19.1 |
| HYBRID，Teacher 权重 0.25 | 18.8 |
| Teacher／来源一致性门控 | 18.3 |
| 温度 2 HYBRID | 19.2 |
| 校准后混合监督 | 19.1 |
| 校准后纯 SOFT | 18.7 |
| Teacher 置信度自适应平滑 | 18.7 |
| 打乱样本对应关系的平滑对照 | 19.6 |
| **普通 LS0.1，无 Teacher** | **19.8** |

HYBRID 比 HARD 高 **0.6**，但 LS0.1 比 HARD 高 **1.3**，也高于温度 2 HYBRID **0.6**。温度 2 与原 HYBRID 表现接近。校准改善了纯 SOFT，却仍未超过普通 LS；置信度自适应平滑也低于打乱对应关系的对照。

因此，**目前观察到的 KD 收益还没有超过普通标签平滑的收益**。完整公共评测中的最佳配方尚缺匹配 LS 对照，这是下一步判断 Teacher 信息是否有用的关键。校准纯 SOFT 使用了来源标签，不属于完全 label-free 训练。

### 已训练起点：普通来源标签续训更好，扩大到 100 万没有改善

warm-start 从同一 MIXED_ALL 模型继续训练，每档各训练一遍，使用固定子集评测。

| 条件 | 10 万 pair | 100 万 pair |
|---|---:|---:|
| 训练前 MIXED_ALL | 22.4 | 22.4 |
| **HARD** | **23.8** | **22.5** |
| SOFT | 21.8 | 21.4 |
| HYBRID | 23.4 | 22.4 |

两档均由 HARD 领先。样本从 10 万扩至 100 万后，三种监督的成绩都下降；这里样本量与训练步数同时增加，不是固定计算预算的规模比较。

### Teacher 能力强，但没有直接传递为 Student 收益

同一固定子集上，Teacher 的原生任务接口为 **68.4**，改用 generic candidate-pair 后为 **64.0**，仍明显强于 Student。pair 改写造成了一定下降，但没有毁掉 Teacher 的主要能力。

Student 改回原 LM head、使用 native 协议读出后，也没有超过训练后的 classifier：

| Student backbone | Classifier | Native LM |
|---|---:|---:|
| 原始模型 | 1.8 | 8.0 |
| HARD 训练后 | **18.5** | 10.7 |
| LS0.1 训练后 | **19.8** | 12.8 |

这次同时改变了输入协议和读出方式，且没有进行 native 协议训练。另一个仍需定位的差别是：Teacher 诊断保留完整输入，而训练标注使用 **4,096-token head-tail cap**。当前尚未区分训练目标拟合不足、输入截断和泛化／数据覆盖问题。

## 3. 先调整 LR，再判断容量；更大的 KD 增益不等于更强的模型

下表均使用完整公共评测。LoRA 的 alpha/rank 保持为 2；LoRA 行的 LR 同时用于适配器和分类头，全参行列出 backbone LR，分类头 LR 固定为 1e-4。

| 训练配置 | LR | HARD | HYBRID | HYBRID−HARD |
|---|---|---:|---:|---:|
| rank32 | 1e-4 | 21.6 | 22.6 | +1.1 |
| rank64 | 1e-4 | 19.8 | 22.2 | +2.5 |
| rank64 | 5e-5 | 21.4 | **22.9** | +1.5 |
| rank64，加入 Tools 校准＋分领域权重 | 5e-5 | 21.4 | 21.9 | +0.4 |
| 全参数 | 1e-5 | 17.1 | 21.9 | +4.8 |
| 全参数 | 2e-5 | 未测 | 18.9 | — |

rank64 的 LR 从 1e-4 降到 5e-5 后，HARD 提高 **1.7**，HYBRID 提高 **0.7**。匹配后的 KD 增益由 **2.5** 缩小为 **1.5**，说明此前较大的增益部分来自偏弱的 HARD。

低 LR 的 rank64 HYBRID 已达到 22.9，但加入 rank32 上胜出的校准与分领域组合后，反而下降 **1.1**。全参提高 LR 后也下降 **3.0**。当前没有理由用更大容量替换表现最好的 rank32 配方。

相关融合实验也没有提供更好的替代：MIXED_ALL **22.8**、SLERP **21.6**、等权平均 **21.5**、SAMerging **20.2**。它们的训练预算和目标不同，单独保留为融合结果，不用于计算 KD 净增益。

## 当前保留的配置与下一步判断

| 项目 | 当前选择 |
|---|---|
| Student | 原始 Qwen3-0.6B＋共同未训练分类头 |
| 微调 | LoRA rank32／alpha64，LR 1e-4，batch128 |
| 数据 | 100 万 pair，四个训练方向各 25 万，一遍 |
| 监督 | 来源标签＋Teacher 概率混合 |
| Teacher 权重 | Knowledge／Tools 0.5，Retrieval 0.25，Arts 0 |
| Tools 校准 | Teacher logit 截距约 −1.28 |

**优先为该配置补齐相同数据和训练预算下的 LS 对照，再判断提升来自 Teacher 信息还是平滑效应。** 温度、温度退火、Teacher 权重渐入及匹配静态权重的实验仍待正式评分。排序蒸馏暂缓：来源审查后，可信关系仅覆盖约 **0.7%** 的百万 pair 池。

## 数据来源

- [完整公共评测：监督、领域权重与容量对照](../distillation-exploration-20261104/REPORT.md) · [校准与 LR 改动](../distillation-exploration-20261104/third-wave-results/report/REPORT.md) · [校准强度与匹配 HARD](../distillation-exploration-20261104/fourth-wave-results/report/REPORT.md)
- [固定子集：原始起点对照](../a800-kd-raw-qwen-1m-20261103/report/REPORT.md) · [监督改进](../a800-kd-improve-round1-20261104/report/REPORT.md) · [校准与自适应平滑](../a800-kd-improve-round2-20261104/report/REPORT.md)
- [warm-start 10 万](../a800-kd-pilot-100k-20261102/REPORT.md) · [warm-start 100 万](../a800-kd-scale-series-20261102/1m/REPORT.md)
- [Teacher 接口诊断](../a800-teacher-pair-diagnostic-20261104/report/REPORT.md) · [Student native 读出](../a800-qwen-native-readout-20261105/report/REPORT.md) · [融合对照](../a800-samerging-four-experts-20261103/report/SAMERGING_DETAILED_REPORT.md)

主表按一位小数展示，差值由原值计算后保留一位小数；[逐项数值与来源](../distillation-collection/results.csv)保留在原结果表。
