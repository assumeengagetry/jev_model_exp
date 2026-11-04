# 2026-11-03 实验报告与图表入口

本次整理时间：2026-11-03T19:30:36+08:00。

## SAMerging：已完成

- [详细方法与实验report](a800-samerging-four-experts-20261103/report/SAMERGING_DETAILED_REPORT.md)
- [简版结果report](a800-samerging-four-experts-20261103/report/REPORT.md)
- [公共总分对比图](a800-samerging-four-experts-20261103/report/figures/samerging-public-index.svg)
- [五方向雷达图](a800-samerging-four-experts-20261103/report/figures/samerging-five-area-radar.svg)
- [对比CSV](a800-samerging-four-experts-20261103/report/public-comparison.csv)

SAMerging公共Index20.21、coverage100%；低于average1.25、低于SLERP1.34。

## 原始Qwen1M蒸馏：已完成

- [最终对照report](a800-kd-raw-qwen-1m-20261103/report/REPORT.md)
- [Raw vs warm总分对比](a800-kd-raw-qwen-1m-20261103/report/figures/raw-vs-warm-1m-index.svg)
- [不同初始化下的蒸馏收益](a800-kd-raw-qwen-1m-20261103/report/figures/distillation-gain-by-initialization.svg)
- [五方向雷达图PNG](a800-kd-raw-qwen-1m-20261103/report/four-model-five-area-radar.png)
- [逐benchmark差值热力图PNG](a800-kd-raw-qwen-1m-20261103/report/benchmark-deltas-heatmap.png)
- [Raw vs warm结果CSV](a800-kd-raw-qwen-1m-20261103/report/raw-vs-warm-1m.csv)

RAW HARD18.46、SOFT17.54、HYBRID19.10；SOFT−HARD−0.92，HYBRID−HARD＋0.64。四组coverage均100%。

- [当前训练进度report](a800-kd-raw-qwen-1m-20261103/report/TRAINING_PROGRESS_20261103.md)
- [当前训练loss曲线](a800-kd-raw-qwen-1m-20261103/report/figures/raw-qwen-training-loss-progress.svg)
- [精确快照数据JSON](a800-kd-raw-qwen-1m-20261103/report/training-progress-snapshot.json)

