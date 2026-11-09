> 整理标注（2026-11-09）：本文件为历史计划或阶段汇总，正文保留当时状态；最新阅读入口见[研究资料库](../../README.md)。

# 2026-11-03 实验报告与图表入口

本次整理时间：2026-11-03T19:30:36+08:00。

## SAMerging：已完成

- [详细方法与实验report](../../01-研究结论/03-专科与融合/a800-samerging-four-experts-20261103/report/SAMERGING_DETAILED_REPORT.md)
- [简版结果report](../../01-研究结论/03-专科与融合/a800-samerging-four-experts-20261103/report/REPORT.md)
- [公共总分对比图](../../03-证据与图表/03-专科与融合/a800-samerging-four-experts-20261103/report/figures/samerging-public-index.svg)
- [五方向雷达图](../../03-证据与图表/03-专科与融合/a800-samerging-four-experts-20261103/report/figures/samerging-five-area-radar.svg)
- [对比CSV](../../03-证据与图表/03-专科与融合/a800-samerging-four-experts-20261103/report/public-comparison.csv)

SAMerging公共Index20.21、coverage100%；低于average1.25、低于SLERP1.34。

## 原始Qwen1M蒸馏：已完成

- [最终对照report](../../01-研究结论/01-蒸馏/a800-kd-raw-qwen-1m-20261103/report/REPORT.md)
- [Raw vs warm总分对比](../../03-证据与图表/01-蒸馏/a800-kd-raw-qwen-1m-20261103/report/figures/raw-vs-warm-1m-index.svg)
- [不同初始化下的蒸馏收益](../../03-证据与图表/01-蒸馏/a800-kd-raw-qwen-1m-20261103/report/figures/distillation-gain-by-initialization.svg)
- [五方向雷达图PNG](../../03-证据与图表/01-蒸馏/a800-kd-raw-qwen-1m-20261103/report/four-model-five-area-radar.png)
- [逐benchmark差值热力图PNG](../../03-证据与图表/01-蒸馏/a800-kd-raw-qwen-1m-20261103/report/benchmark-deltas-heatmap.png)
- [Raw vs warm结果CSV](../../03-证据与图表/01-蒸馏/a800-kd-raw-qwen-1m-20261103/report/raw-vs-warm-1m.csv)

RAW HARD18.46、SOFT17.54、HYBRID19.10；SOFT−HARD−0.92，HYBRID−HARD＋0.64。四组coverage均100%。

- [当前训练进度report](../01-蒸馏/a800-kd-raw-qwen-1m-20261103/report/TRAINING_PROGRESS_20261103.md)
- [当前训练loss曲线](../../03-证据与图表/01-蒸馏/a800-kd-raw-qwen-1m-20261103/report/figures/raw-qwen-training-loss-progress.svg)
- [精确快照数据JSON](../../03-证据与图表/01-蒸馏/a800-kd-raw-qwen-1m-20261103/report/training-progress-snapshot.json)

