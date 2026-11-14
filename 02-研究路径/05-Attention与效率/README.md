# Attention与效率 · 研究路径

## 2026-11-11新增

- [推理调度与计算内核v2执行计划](2026-11-11-inference-kernel-runtime-v2.md) · [进度与边界记录](inference-kernel-runtime-20261111/progress.md)。执行源码遵循当前库外归档约定。

## 2026-11-10新增

- [共享前缀＋Graph实际实施计划](2026-11-10-inference-prefix-graphs.md)，对应[已完成的完整public结果](../../01-研究结论/05-Attention与效率/inference-prefix-graphs-20261110/README.md)。
- [当前方案推理全过程与限制](../06-五维训练与Readout/current-scheme-20261110/INFERENCE-AND-EVALUATION.md)，源码按库外辅助归档规则保留。

[返回分类](../README.md) · [返回总入口](../../README.md)

## 文档

- [当前 batch、长度截断与 benchmark 安排](batch-benchmark-20261028/length-and-batch-notes.md)
- [Batch benchmark：真实联合数据窗口的初步结果](batch-benchmark-20261028/partial-results.md)
- [输入长度与gradient checkpointing](checkpoint-length-scaling-20261029/PLAN.md)
- [归档计划与真实训练曝光：后续worker的记录要求](length-label-benchmark-20261029/EXPOSURE-ACCOUNTING-NOTE.md)
- [GPU协作交付：资源测试与后续比例首轮均已完成并释放](length-label-benchmark-20261029/GPU-COORDINATION.md)
- [正负pair比例：CPU准备与待交接方案](length-label-benchmark-20261029/PAIR-RATIO-PREP.md)
- [输入长度与正负比例：用户新研究方向](length-label-benchmark-20261029/PLAN.md)
- [首轮pair比例快筛：冻结设置与启动条件](length-label-benchmark-20261029/RATIO-EXECUTION-PLAN.md)
- [正负pair比例首轮：已完成、已复核、已释放](length-label-benchmark-20261029/RATIO-RUNNING-NOTE.md)
- [Student 显式 FlashAttention-3](student-flash-attention3-20261103/CHANGE.md)
- [Qwen 4B attention source audit (2026-11-09)](2026-11-09-attention-audit/four-b-findings.md)
- [Qwen 27B attention audit — 2026-11-09](2026-11-09-attention-audit/large-findings.md)
- [Mammoth / d3 attention audit — 2026-11-09](2026-11-09-attention-audit/mammoth-d3-findings.md)
- [少见参数量 attention 补充（2026-11-09，只读静态核查）](2026-11-09-attention-audit/misc-findings.md)
- [Supplemental 4B attention audit](2026-11-09-attention-audit/new-four-findings.md)
- [Qwen 8B/9B attention audit](2026-11-09-attention-audit/nine-b-findings.md)
- [9B source checks (read-only, 2026-11-09)](2026-11-09-attention-audit/nine-source-check.md)
- [Small Qwen model attention audit](2026-11-09-attention-audit/small-findings.md)
- [New small-model attention findings](2026-11-09-attention-audit/small-new-findings.md)
