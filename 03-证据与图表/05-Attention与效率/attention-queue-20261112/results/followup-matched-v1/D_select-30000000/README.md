# 后续探索30M：固定协议下的中间结果

同一封存quality/scaling两份分析源码实际执行，于2026-11-12T18:46:56UTC独立核对m1122／29985465logicaltoken／17952case occurrences。五臂各完整375case／615knownfield／5190candidate／90unknown不forward，coverage1；onehot exact分母259。source/config/fresh393实际bridge、完整候选/native q、pre/post同cursorcheckpoint及目标曝光前缀核对，最大metric误差1.88738e-15。没有重跑模型/evaluator/tensor审计或改变先前预测。

| packed臂 | target mass↑ | NLL↓ | Brier sum↓ |
|---|---:|---:|---:|
| Causal CE | 55.6754% | 1.0365784 | .5288095 |
| Causal BCE | 53.1384% | 1.1624517 | .5816324 |
| Candidate-PB CE | 41.0244% | 1.2628358 | .6333305 |
| State＋已有LF PB CE | 41.6300% | 1.2491103 | .6225877 |
| JPAIR BCE＋hard-rank | 55.7496% | 1.0606366 | .5333580 |

## 成对整case描述性差异

| 差值 | 百分点 | 95%描述性case区间（百分点） |
|---|---:|---:|
| STATE−Candidate-PB | +.6056 | [−4.4214,+5.5657] |
| STATE−causal CE | −14.0454 | [−19.3799,−8.7407] |
| JPAIR−BCE | +2.6112 | [−.7468,+5.8556] |
| JPAIR−CE | +.0743 | [−3.1424,+3.3691] |

10000次whole-case配对、五域各75分层，未经多重比较调整，不含训练seed方差。JPAIR的mass与CE几乎持平，NLL/Brier仍略差；相对BCE的正点估计区间仍跨0。STATE相对Candidate-PB略高但区间宽跨0，两种PB都远低于causal CE。不能把55.75%当Public Index、显著胜出或超过51.99，也不因这些中间值调整LR/曝光/λτ/clip或提前选节点。

## 实际前1122宏的尺度诊断

五臂逐macro knownfield/candidate/K1序列相同，记录的scalar clip_factor<1比例全部100%；clip前参数梯度L2均值依CE/BCE/PB/STATE/JPAIR为10.5969／9.9946／13.9501／12.0913／17.6287。这里是历史优化轨迹诊断，不是质量排序，也不是每次GPU clip乘数的逐位审计。

JPAIR共有25157known field occurrences，其中22953active、2204zero，active91.2390%；hardpositive22953／hardnegative93632／soft5290。无rank case occurrences1773；zero理由可重叠：无正2204、无负1551、soft-only1551；K1本前缀0。`dL/dmargin` total/BCE L2比均值3.5671，不是393参数梯度比，不能单独归因排序方向或尺度；严格onehot时rank等于field CE的数学关系与全active字段是否strict-onehot仍须区分。

六份primary/scaling/两新臂SUMMARY/COMPLETE聚合view及原完整private产物都保留；仅聚合下载，原case/q/logit/token/tensor留远端。相同源码继续用于100M；30M观察器已退出，新100M只读事件观察器与controller exactpidfd观察器在等待。未开D_confirm/public、未选winner或发布。
