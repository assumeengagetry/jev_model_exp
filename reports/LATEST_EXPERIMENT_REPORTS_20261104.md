# 2026-11-04 蒸馏改进实验入口

最新进度见 [2026-11-05 汇总](LATEST_EXPERIMENT_REPORTS_20261105.md)：Teacher pair 和 Qwen native-LM 诊断均已完成。

## 第一轮：已完成

- [最终report](a800-kd-improve-round1-20261104/report/REPORT.md)
- [全部条件对比图PNG](a800-kd-improve-round1-20261104/report/variant-index-comparison.png)
- [相对固定对照增益图PNG](a800-kd-improve-round1-20261104/report/gains-over-controls.png)
- [结果CSV](a800-kd-improve-round1-20261104/report/all-model-five-area.csv)

HARD18.46、原HYBRID19.10；MIX25 18.76、AGREE50 18.30、T2 19.17、普通LS0.1 19.79。当前最强对照是LS，不能隐藏这一结果来宣称Teacher胜出。

## 第二轮：已完成

- [四组设计和实际启动记录](a800-kd-improve-round2-20261104/README.md)
- [最终report](a800-kd-improve-round2-20261104/report/REPORT.md)
- [所有条件对比PNG](a800-kd-improve-round2-20261104/report/variant-index-comparison.png)
- [相对固定对照增益PNG](a800-kd-improve-round2-20261104/report/gains-over-controls.png)

CAL50 19.07、CALSOFT18.72、ADAPT_LS18.67、SHUFFLE_LS19.56；aligned−shuffled−0.89。本轮仍未超过uniformLS19.79，不能宣称Teacher独有收益。

## Teacher原生协议诊断：已完成

- [目标、设置及运行记录](a800-teacher-native-diagnostic-20261104/README.md)
- [最终report](a800-teacher-native-diagnostic-20261104/report/REPORT.md)
- [Teacher与Student对比PNG](a800-teacher-native-diagnostic-20261104/report/native-teacher-vs-students.png)

Teacher原生Index68.37、coverage100%，5,935请求全部ok；本轮rawStudent最佳LS19.79。没有把这个结果当作pairtarget质量或KD必胜的证明。

## 同 Teacher 的 pair 协议对照：已完成

- [设计和实际完成记录](a800-teacher-pair-diagnostic-20261104/README.md)
- [最终 report](a800-teacher-pair-diagnostic-20261104/report/REPORT.md)

同一 frozen cases / 全部 candidates，Teacher pair **63.95**，原生 **68.37**，pair−native **−4.42**；5,935 请求全部 ok，coverage 100%。这不支持 pair 改写摧毁大部分 Teacher 能力；保留训练长度截断差异，不据此排除训练数据或 Student 学习问题。

## 前期结果

- [原始起点与已训练起点对照](a800-kd-raw-qwen-1m-20261103/report/REPORT.md)
- [SAMerging详细report](a800-samerging-four-experts-20261103/report/SAMERGING_DETAILED_REPORT.md)
- [11月3日报告和图表入口](LATEST_EXPERIMENT_REPORTS_20261103.md)
