# 2026-11-04 蒸馏改进实验入口

## 第一轮：已完成

- [最终report](a800-kd-improve-round1-20261104/report/REPORT.md)
- [全部条件对比图PNG](a800-kd-improve-round1-20261104/report/variant-index-comparison.png)
- [相对固定对照增益图PNG](a800-kd-improve-round1-20261104/report/gains-over-controls.png)
- [结果CSV](a800-kd-improve-round1-20261104/report/all-model-five-area.csv)

HARD18.46、原HYBRID19.10；MIX25 18.76、AGREE50 18.30、T2 19.17、普通LS0.1 19.79。当前最强对照是LS，不能隐藏这一结果来宣称Teacher胜出。

## 第二轮：已启动，等待最终评测

- [四组设计和实际启动记录](a800-kd-improve-round2-20261104/README.md)

四卡分别运行calibrated hybrid、calibrated pureKD、Teacher-aligned adaptiveLS及分布完全匹配的shuffledLS。目标是检验校准和样本级Teacher信息是否超出普通正则化。

## 前期结果

- [原始起点与已训练起点对照](a800-kd-raw-qwen-1m-20261103/report/REPORT.md)
- [SAMerging详细report](a800-samerging-four-experts-20261103/report/SAMERGING_DETAILED_REPORT.md)
- [11月3日报告和图表入口](LATEST_EXPERIMENT_REPORTS_20261103.md)
