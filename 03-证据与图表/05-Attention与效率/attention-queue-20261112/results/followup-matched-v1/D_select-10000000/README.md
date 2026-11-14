# 后续探索10M：五臂同曝光独立回算

**仅中间节点，不是Public Index，也不是完整训练结论。** 全部臂在同seed42／同初始393张量值／同完整曝光前缀m365、5840case occurrences、9988659logicaltokens对齐。新旧fresh artifact路径不同，但独立生成后的393byte一致与MAIN实际before/update/checkpoint回读均由源绑定bridge核对；本分析没有重跑旧模型或tensor审计。

主代理2026-11-12T18:34:46UTC，从原pinned select和实际保存logits/margins独立回算五臂：375case、615knownfield、5190candidate、90unknown不forward、coverage1；原完整onehot exact分母259。最大指标误差1.77636e-15。节点前后同cursor checkpoint、source/config/fresh和两条原paused ledger历史身份分类核对，不把正在增长的ledger误当不可变文件。

| 臂（全部packed） | target mass↑ | NLL↓ | Brier sum↓ |
|---|---:|---:|---:|
| C_P_CE | 52.0806% | 1.0740614 | .5574684 |
| C_P_BCE | 49.6898% | 1.2002229 | .5840133 |
| Candidate-PB CE | 41.1644% | 1.3378017 | .6310666 |
| State＋已有LF PB CE | 40.7510% | 1.3610731 | .6380916 |
| JPAIR BCE＋hard-rank | 50.9312% | 1.0805442 | .5545982 |

## 配对case描述性差异

whole-case配对、五域各75、10000分层bootstrap；区间未经多重比较调整，不覆盖训练seed方差。

| 差值 | 百分点 | 95%描述性case区间（百分点） |
|---|---:|---:|
| STATE−Candidate-PB | −.4134 | [−5.0458,+4.2709] |
| STATE−causal CE | −11.3296 | [−16.6618,−5.8678] |
| JPAIR−BCE | +1.2414 | [−2.0663,+4.4769] |
| JPAIR−CE | −1.1494 | [−3.9424,+1.7228] |

目前没有稳定state边界收益证据；JPAIR相对BCE的点估计略好但case区间跨0，相对CE的mass略低、NLL略差而Brier略好。不能据此选winner、提前停训、改变λ/τ/LR/clip或回选public checkpoint。两方法均是看过原10M/30M后提出的探索性消融，不追记为原四臂预注册比较，也不冒称完整Base/Jina复现。

## 实际训练尺度与clip诊断（另表，非质量指标）

从已保存的前365个实际`macro_updated`事件读取，五臂逐macro knownfield/candidate/K1计数序列一致，不读节点之后的进度、不训练新模型。

| 臂 | 记录的clip_factor<1宏比例 | clip前参数梯度L2均值 |
|---|---:|---:|
| C_P_CE | 100% | 8.9089 |
| C_P_BCE | 100% | 15.4189 |
| Candidate-PB | 100% | 20.0269 |
| STATE | 100% | 20.6861 |
| JPAIR | 100% | 21.0398 |

此频率按保存的Python scalar clip_factor定义，不冒充每宏GPU实际FP32乘数的逐位复核。第一正式macro的实际raw×GPU multiplier字节另已核验。

JPAIR：8224known field occurrences中7485rank-active（91.0141%）、739rank0；594case occurrences没有rank信号。hardpositive7485／hardnegative31117／soft1784，K1本前缀0；zero原因可重叠（无正739、无负526、soft-only526），不能相加当监督分母。

其实际`dL/dmargin` L2比值平均：rank/BCE2.7598，total/BCE3.6355；**不是393参数梯度分量比**。该固定目标同时改变方向／尺度；本阶段所有臂clip频率都为100%，clip强度仍不同，不能把质量变化单独归因“排序方向”。原schema约束不表达多个q==1；严格onehot时rank等于field CE，但上述aggregate并未证明每个active field都严格onehot。训练component loss只是变化权重下的优化诊断，不作提分指标。

## 证据与后续

`PRIMARY_VERIFICATION_AND_PAIRED.json`、`TRAIN_SCALING.json`及两新臂SUMMARY/COMPLETE view共6份；view明确省略重复scope。两份实际analysis源码逐byte回读归档；private case/q/logits/token/tensor留远端。

0模型forward／optimizer／新evaluator；未开D_confirm/public或选择赢家。保持原100M及三观察节点，30M只读事件观察器及新controller exact-pidfd完成观察器在等待，不轮询进度或重启评测。
