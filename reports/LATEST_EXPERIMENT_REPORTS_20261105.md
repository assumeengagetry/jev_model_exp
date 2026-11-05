# 2026-11-05 蒸馏研究：最新结果

**Teacher 在 native 和 pair 协议上都很强；Student 直接改用原 LM head/native 读出没有超过现有 classifier。尚无已测 Teacher 蒸馏方案超过匹配的普通 label smoothing 对照。**

全部以下数字来自同一冻结 37 项、4,786 完整 cases、5,935 请求及全部候选；均 coverage 100%。这是探索性固定子集，不是完整公共榜单或私测 Full，也不是多 seed 显著性结论。

## 新完成的两项机制测量

| 条件 | Index | 本次可说明什么 |
|---|---:|---|
| KnowLine-31B native | 68.37 | Teacher 在原始任务协议上很强 |
| 同 Teacher，generic candidate-pair | 63.95 | 比 native 低 4.42，仍高于最佳 raw Student 44.16 |
| 原始 Qwen3-0.6B，native LM | 8.01 | 不同于此前随机 classifier 起点的 1.76 |
| HARD 后 backbone，native LM | 10.72 | 比原始 native +2.71，低于其 classifier 18.46 |
| LS0.1 后 backbone，native LM | 12.84 | 比原始 native +4.83，低于其 classifier 19.79 |

- [Teacher pair：最终报告](a800-teacher-pair-diagnostic-20261104/report/REPORT.md) · [范围与运行记录](a800-teacher-pair-diagnostic-20261104/README.md)
- [Qwen native-LM：最终报告](a800-qwen-native-readout-20261105/report/REPORT.md) · [结论与实际完成依据](a800-qwen-native-readout-20261105/README.md)
- [Qwen native-LM 对比图 PNG](a800-qwen-native-readout-20261105/report/native-readout-comparison.png) · [SVG](a800-qwen-native-readout-20261105/report/native-readout-comparison.svg) · [CSV](a800-qwen-native-readout-20261105/report/native-readout-comparison.csv)

Teacher pair 实际 step `275711.68`，11 月 5 日 02:17 退出 0；Qwen native 实际 step `275711.69`，02:50 退出 0。三组 Qwen 各 5,935 ok。报告、原生预测、日志和图表均已落盘。本轮无新 Student 梯度更新、无新环境或上传；旧规模系列保持暂停，allocation 275711 未取消。

## 蒸馏收益的当前实测边界

匹配 raw 1M 条件：HARD **18.46**，SOFT **17.54**，HYBRID **19.10**，T2 HYBRID **19.17**，普通 LS0.1 **19.79**。HYBRID 相对 HARD 的 +0.64 可以报告，但不足以说明 Teacher 超过强无 Teacher 对照。

第二轮 CAL50 **19.07**、CALSOFT **18.72**、ADAPT_LS **18.67**、SHUFFLE_LS **19.56** 也未超过 LS。CALSOFT 用原训练标签拟合校准，不是完全 label-free；aligned smoothing 不如 shuffled control 也不能泛化为所有 Teacher 信息无用。

- [两轮改进实验及原生 Teacher 报告入口](LATEST_EXPERIMENT_REPORTS_20261104.md)
- [Raw 与 warm 初始化对照](a800-kd-raw-qwen-1m-20261103/report/REPORT.md)
- [SAMerging 完整公共评测及详细报告](a800-samerging-four-experts-20261103/report/SAMERGING_DETAILED_REPORT.md)

## 现在能排除与还不能定位的部分

1. 不支持“Teacher 整体很弱”，或“完整 eval 输入的 pair 改写摧毁其大部分能力”作为主要解释。
2. 不支持“直接恢复原 LM head 就能恢复一个很强的 Student”这条捷径；但本次同时变了 protocol/readout，而且未训练 native 协议，不能宣布 head 完全无关或 native 训练无效。
3. 尚未区分 Student 对训练 target 的拟合不足、泛化/训练覆盖差、训练长度影响与容量限制。优先查这个分界，不再用无目的的 loss 扫参代替机制证据。

所有正负结果均保留；不把 Teacher 的高分、普通 LS 的收益或推理读出变化冒充 Teacher 独有蒸馏收益。
