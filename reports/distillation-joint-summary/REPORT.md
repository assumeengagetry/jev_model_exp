# A800 × H800 蒸馏研究：阶段结论总汇

> 汇总当前本地已产出的报告，最新入口标记至 2026-11-05；两条研究线仍在持续产出。本页只归纳已有结果，不等待全部实验结束，也不重新训练、评测或计算哈希。

## 一页结论

**目前最值得保留的路线是“来源监督＋Teacher 软标签混合，并按领域控制 Teacher 权重”，但尚未证明 Teacher 信息本身优于普通标签平滑，也没有找到稳定的最优蒸馏配方。**

1. **纯 SOFT 不如 HARD，混合监督更有希望。** 原始 Qwen 的配平实验中，A800 的 HYBRID 比 HARD 高 **0.64**，H800 高 **1.06**；纯 SOFT 分别低 **0.92、0.98**。A800 的 warm-start 两档也没有显示纯 SOFT 优势。
2. **A800 最关键的新发现：普通标签平滑比已测 KD 更强。** LS0.1 为 **19.79**，超过普通 HYBRID **19.10**、温度 2 HYBRID **19.17**，也超过第二轮校准、自适应平滑等条件。只比较 HYBRID 与 HARD，会高估目前能归因于 Teacher 的收益。
3. **H800 当前蒸馏最高结果是领域保护 22.90。** 比原 HYBRID **22.63** 高 **0.27**，比 HARD **21.57** 高 **1.33**。主要提升 Tools、Retrieval，同时 Language、Arts 下降，不是全方向提升。
4. **增加可训练容量没有带来更高绝对成绩。** H800 的 r32 HYBRID **22.63**，r64 **22.20**，全参 **21.89**；容量更大时“HYBRID−HARD”变大，主要伴随 HARD 变弱，不能当成模型更强。
5. **Teacher 不弱，直接恢复 Student 原 LM head 也不是捷径。** A800 同一子集上，Teacher native **68.37**、generic pair **63.95**；Student 改用未专门训练的 native LM 读出后，HARD、LS 仅 **10.72、12.84**，均低于各自 classifier。

**统筹判断：当前已有的是“混合监督有小幅收益”和若干负结果；尚缺的是“超过强无 Teacher 对照、跨 seed 稳定、跨领域不退化”的 Teacher 独有增益。**

## 两条线必须分开看分数

| 项目 | A800 探索线 | H800 探索线 |
|---|---|---|
| 主要作用 | 快速对照、监督机制与读出诊断 | 完整公共评测、领域权重与容量探索 |
| 当前核心训练量 | 原始起点 100 万 pair；另有 warm-start 10 万／100 万 | 原始起点 100 万 pair，四科各 25 万 |
| 核心评测范围 | 冻结 37 项探索子集：4,786 完整 cases、5,935 请求 | 完整公共 37 项；第一波报告标记 coverage 100% |
| 当前主要参照 | HARD 18.46；HYBRID 19.10；LS0.1 19.79 | HARD 21.57；HYBRID 22.63；领域保护 22.90 |
| 结论边界 | 子集、单 seed，不是完整公共榜单 | 单 seed；第二波容量分数按完成通知归档 |

**A800 的 19.79 与 H800 的 22.90 不能直接排名，更不能解释成 H800 硬件让模型效果更好。** 两条线不是只更换 GPU 的同一实验。原始起点两线均未训练 Language 数据，但评测保留 Language。全部分数均不是官方 Private／Full。

## A800：已有结果及其意义

### 1. 原始起点：先打过普通 label smoothing，才能讨论 Teacher 独有收益

| 条件 | 探索子集 Index | 相对 HARD |
|---|---:|---:|
| HARD：来源监督 | 18.46 | — |
| SOFT：仅 Teacher | 17.54 | −0.92 |
| HYBRID：混合监督 | 19.10 | +0.64 |
| 温度 2 HYBRID | 19.17 | +0.71 |
| **LS0.1：普通标签平滑，无 Teacher** | **19.79** | **+1.33** |
| 第二轮 CAL50 | 19.07 | +0.61 |
| 第二轮 CALSOFT | 18.72 | +0.26 |
| 第二轮 ADAPT_LS | 18.67 | +0.21 |
| 第二轮 SHUFFLE_LS | 19.56 | +1.10 |

- **已得结论：** 目前没有已测 Teacher 蒸馏条件超过匹配的 LS0.1。温度 2 只比普通 HYBRID 高 **0.07**，不足以称为确定改进。
- **机制线索：** aligned 自适应平滑不如 shuffled control，说明目前利用 Teacher 样本对应信息的方法没有奏效；不能外推为所有 Teacher 信息都无用。CALSOFT 的校准使用来源标签，不是完全 label-free。

### 2. 已训练起点：更多样本也不保证更好

| warm-start 条件 | 10 万 pair | 100 万 pair |
|---|---:|---:|
| 原 MIXED_ALL 起点 | 22.39 | 22.39 |
| HARD | **23.79** | **22.49** |
| SOFT | 21.80 | 21.36 |
| HYBRID | 23.42 | 22.38 |

两档 SOFT−HARD 分别为 **−1.99、−1.13**；HYBRID−HARD 分别为 **−0.37、−0.11**。原始起点与 warm-start 不能合并成同一组对照；这里也不能推出“10 万普遍优于 100 万”或规模扩展必然无效。

### 3. 机制诊断：已缩小解释范围，但尚未找到根因

| 诊断 | 同一探索子集 Index | 已能说明的事 |
|---|---:|---|
| Teacher native | 68.37 | Teacher 在原任务协议上很强 |
| Teacher generic candidate-pair | 63.95 | 比 native 低 4.42，但能力没有被 pair 改写整体摧毁 |
| 原始 Qwen native LM | 8.01 | 与随机 classifier 起点 1.76 不是同一种 baseline |
| HARD 后 backbone，native LM | 10.72 | 低于其 classifier 18.46 |
| LS 后 backbone，native LM | 12.84 | 低于其 classifier 19.79 |

尚未区分训练 target 拟合不足、泛化／数据覆盖、训练长度和容量限制。**Teacher pair 诊断保留了完整评测输入，而训练标注管线有 4,096-token head-tail cap；因此 pair 高分不能直接证明训练标签缓存及长输入处理没有问题。** Student native 诊断同时改变 protocol/readout，且未训练 native 协议，不能据此宣布分类头完全无关。

## H800：已有结果及其意义

### 1. 原始起点的完整公共对照

| 条件 | 公共 Index | 相对 HARD |
|---|---:|---:|
| HARD | 21.57 | — |
| SOFT | 20.59 | −0.98 |
| HYBRID | 22.63 | +1.06 |
| **领域保护** | **22.90** | **+1.33** |
| 降 Teacher 权重至 0.25 | 21.95 | +0.38 |
| 样本可靠性门控 | 21.93 | +0.36 |
| 领域截距校准 | 22.02 | +0.45 |

领域保护对 Knowledge／Tools 保持 Teacher 权重 **0.5**，Retrieval 降至 **0.25**，Arts 为 **0**。相对原 HYBRID：**Tools +3.18、Retrieval +1.06**，但 Language **−1.75**、Arts **−0.56**、Knowledge **−0.08**。

**当前可保留的是“按领域区别使用 Teacher”的方向，不是“所有领域更好”或“Teacher 权重越低越好”。** 来源一致率只是排查线索，不是真实 Teacher 准确率。H800 目前报告未提供匹配 LS 对照，不能把 +1.33 全部解释成 Teacher 独有收益。

### 2. 容量对照：看绝对成绩，不只看 KD 净增益

| 配方 | HARD | HYBRID | HYBRID−HARD |
|---|---:|---:|---:|
| r32／alpha64 | 21.57 | **22.63** | +1.06 |
| r64／alpha128 | 19.75 | 22.20 | +2.45 |
| 全参数 | 17.06 | 21.89 | +4.83 |

**本轮没有支持“加大容量就能解决蒸馏瓶颈”。** r64／全参的 KD 净增益虽更大，HYBRID 绝对成绩反而低于 r32；这些是当前学习率／单 seed 的结果，不是容量路线的最终否定。第二波分数来自完成通知，源报告注明尚未独立刷新分项和覆盖细节。

## 相关融合结果：不混作蒸馏收益

完整公共 37 项中，MIXED_ALL **22.82**、SLERP **21.55**、等权平均 **21.46**、SAMerging **20.21**。已测融合没有超过混合训练；SAMerging 学的是融合系数，不是 HARD／SOFT／HYBRID 的同预算 Student 训练。

H800 领域保护的 **22.90** 数值上略高于 MIXED_ALL，但两者训练数据／预算不同，不能把这点差异直接解释成蒸馏的因果收益。历史 Language **36.33** 属于单维度，完全不放入以上总分排名。

## 统筹时继续追踪的三个问题

- **收益归因：** 混合／领域保护是否能超过匹配的普通 LS，而不仅仅超过 HARD？
- **传递瓶颈：** Student 是没有拟合好 Teacher target，还是训练池上拟合了但没有泛化？
- **稳定性与代价：** 当前微小增益是否跨 seed 保留，且不会以其他领域下降或过高标注成本换取？

以上为问题台账，不代表本页新增启动实验。后续结果仍由 A800、H800 原执行 session 产出，本汇总统一吸收已完成结论；不覆盖它们的源报告，也不替代原实验调度。

## 原始报告入口

- [A800 最新汇报入口](../LATEST_EXPERIMENT_REPORTS_20261105.md) · [前一轮汇报入口](../LATEST_EXPERIMENT_REPORTS_20261104.md)
- [A800 原始起点 100 万 pair](../a800-kd-raw-qwen-1m-20261103/report/REPORT.md)
- [A800 warm-start 10 万](../a800-kd-pilot-100k-20261102/REPORT.md) · [warm-start 100 万](../a800-kd-scale-series-20261102/1m/REPORT.md)
- [Teacher pair 诊断](../a800-teacher-pair-diagnostic-20261104/report/REPORT.md) · [Qwen native 读出诊断](../a800-qwen-native-readout-20261105/report/REPORT.md)
- [H800 原始起点 100 万 pair](../fresh-qwen-1m-distillation-20261103/results-20261104/report/REPORT.md)
- [H800 两波探索主报告](../distillation-exploration-20261104/REPORT.md) · [第一波完成报告](../distillation-exploration-20261104/first-wave-results/report/REPORT.md)
- [SAMerging 方法与完整公共结果](../a800-samerging-four-experts-20261103/report/SAMERGING_DETAILED_REPORT.md)
