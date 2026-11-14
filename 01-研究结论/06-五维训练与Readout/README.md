# 五维训练与Readout · 研究结论

[返回分类](../README.md) · [返回总入口](../../README.md)

## 2026-11-13 完整八臂结果

- **[完整学习报告](clean-eight-official-20261113/REPORT.md)**：从用户材料、模型改造、完整 case 数据与目标、训练机制，到推理优化和官方测分的 19 章说明。
- 八个 seed42 终端权重全部完整 public；最高 `C_P_BCE` **14.42**，四卡逐模型队列 **14h38m03s**。本轮与旧 joint 谱系不同，不能直接作单变量归因。
- [344 行 benchmark、40 行五域及官方原始计分聚合](../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/README.md)。dense→packed 原数值失败、单 seed、未完整确认及非生产准入边界保留。

## 文档

- **[当前模型为何仍未达到28](current-scheme-20261110/BOTTLENECKS.md)**：事实、条件算术和未知机制分开。
- **[Readout两层四臂最终报告](readout-study-20261108/FINAL-REPORT-20261110.md)**：已完成、20来源核验、负结果和换序风险。
- [joint final原正式public24.97](joint-generic-continuation-20261109/README.md) · [同权重加速版24.99](../05-Attention与效率/inference-prefix-graphs-20261110/README.md)。
- [从零理解当前实际配方](../../02-研究路径/06-五维训练与Readout/current-scheme-20261110/README.md)。

- [原始Qwen LM-head/native协议与现有classifier读出诊断](a800-qwen-native-readout-20261105/report/REPORT.md)
- [Teacher原生协议直接评测](a800-teacher-native-diagnostic-20261104/report/REPORT.md)
- [同一Teacher：原生任务协议与candidate-pair改写对照](a800-teacher-pair-diagnostic-20261104/report/REPORT.md)
