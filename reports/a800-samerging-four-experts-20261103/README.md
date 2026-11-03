# 四专家 SAMerging 实验

## 报告入口

- **[完整方法与实验报告：SAMERGING_DETAILED_REPORT.md](report/SAMERGING_DETAILED_REPORT.md)**：方法原理、Qwen/LoRA适配、数据与超参数、官方实现差异、调度恢复、完整结果及解释边界。
- [自动生成的简版结果报告](report/REPORT.md)；[完整公共结果CSV](report/public-comparison.csv)。

## 最终结果 —2026-11-03 14:59 CST

**已完成：公共37项总分20.21，coverage100%（110,087/110,087请求均ok）；KD冻结子集分数20.09（5,935/5,935请求均ok）。** 相对MIXED_ALL−2.61、average−1.25、SLERP−1.34、TIES＋1.21。本配置未超过average/SLERP/MIXED_ALL。

完整报告见[`report/REPORT.md`](report/REPORT.md)，逐方向CSV见`report/public-comparison.csv`。模型、系数和完整预测已保留；未自动开展额外SAMerging调参。下述运行记录是历史过程，不表示当前仍在校准或等待评测。

## 当前状态

2026-11-03 11:09 CST 实测：用户要求先暂停其他任务验证SAMerging。已暂停原有Student2M/Teacher3M及其自动推进控制器，保留checkpoint、预测和标签；没有取消allocation275711。

**SAMerging已于11:08:33实际启动为step275711.61**，完成197个有效权重tensor/788个系数的构造，正在缓存对齐的expert logits。尚无最终效果分数。

11:12更新：四个expert的各1,600-pair logits缓存均已完成，已观察到实际第1/600步系数更新；clean KL1.1363、SAM perturbed KL1.4295、实际扰动norm0.035。当前squeue仅有SAMerging计算step及原allocation的extern，旧KD计算step已退出。

**13:32更新：校准已完成600步并导出模型，耗时约53.9分钟。** 之前的三卡sidecar因继承主step的CPU binding而未能启动；分片0已完成27,522个请求。现已改为顶层Slurm step275711.63，三张卡仅补跑分片1–3，各分片均已实际处理200个请求。模型和分片0没有重跑；完整质量分数仍待全部评测完成。

旧失败退出证据保留于远端`attempts/attempt-1/`，本地`LIVE_STATUS.json`为新的恢复运行状态。原始Qwen对照仍等待本实验成功结束。

校准仍在一张卡完成；完整公共评测改为四卡四个不重叠shard并行（原step一张卡＋单独sidecar三张卡），使用同一checkpoint和全部候选，不缩减评测。

`LIVE_STATUS.json` 由本地只读 watcher 刷新；实际启动以 `STEP_STARTED.json` 和 calibration progress 为准，最终效果以 `report/REPORT.md`、`COMPLETE.json` 及原生evaluation记录为准。

## 本次方法配置

- 合并现有Knowledge/Retrieval/Tools/Arts四个Qwen3-0.6B专家，不合并KnowLine31B，也不把KD三组当作领域专家。
- 使用实际共同初始化W0、有效权重task vectors和可学习逐tensor系数，包含一个共享二分类head。不是LoRA A/B因子的直接平均。
- 每域1,600个无标签校准pair，seed42；任务等权KL(expert||merged)，T1；不使用原标签或评测输入调系数。
- init0.5，600步，每域batch16，lr1e-3，不clamp；采用官方NLP代码的adaptive SAM＋SGD，momentum0.99、weight decay5e-4、rho0.07。论文附录写Adam，而官方NLP代码用SGD；本次明确跟随代码。
- 修正官方GPT2示例的batch/logit错位和SAM扰动后缓存权重未刷新问题；官方SAM optimizer实现原样保留。Qwen共享label space需要合并成单个head，不保留task-specific heads。
- FP32合并计算、BF16 forward与导出，部署一个dense模型，无运行时多专家开销。

## 对比口径

1. 完整公共37项、110,087请求：对比同口径MIXED_ALL22.82、average21.46、TIES19.00和SLERP21.55。
2. 从保存的完整预测中重算KD冻结子集分数，与BASELINE/HARD/SOFT/HYBRID作同评测口径对照；两种训练方式的训练量和监督来源不同，不作严格等预算归因。

官方代码：https://github.com/arshandalili/SAMerging/tree/a194e8ed8ab790dcae9c02a8d1467e00e86b53e5

论文：https://arxiv.org/abs/2512.21288

本地实现与执行记录：`/home/assumeengage/profiler/a800/devlocal/samerging-four-experts-20261103/`；计划：`/home/assumeengage/profiler/a800/docs/superpowers/plans/2026-11-03-samerging-four-experts.md`。
