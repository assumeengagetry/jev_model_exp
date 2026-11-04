# 蒸馏改进第一轮：已完成

2026-11-04 15:40 CST完成，step退出0。四组均实际训练1M pairs、7,813steps、228,021,289tokens，固定37项/5,935请求全部ok，coverage100%。

| 条件 | Index | vs HARD18.46 | vs LS0.1 |
|---|---:|---:|---:|
| MIX25 |18.76|+0.30|−1.03|
| AGREE50 |18.30|−0.16|−1.49|
| TEMP2_50 |19.17|+0.71|−0.62|
| 不用Teacher的LS10 |19.79|+1.33|0.00|

**本轮最佳为普通label smoothing，尚不能说Teacher提供了额外优势。** 简单agreement gating没有改善，T2仅比原HYBRID19.10高0.07分；单次seed/探索性子集不作显著性结论。

- [最终report](report/REPORT.md)
- [所有条件对比图PNG](report/variant-index-comparison.png)
- [相对固定对照增益图PNG](report/gains-over-controls.png)
- [结果CSV](report/all-model-five-area.csv)
- [最初研究与计划](RESEARCH_AND_PLAN.md)

按用户授权，第二轮已运行，集中检验Teacher校准及Teacher-to-example对应信息，而不是盲扫或隐藏LS结果。
