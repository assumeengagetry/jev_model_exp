# Attention与效率 · 研究结论

## 2026-11-10新增

- [同joint final共享前缀＋CUDA Graph完整public结果](inference-prefix-graphs-20261110/README.md)：24.99、整套2.765×、输出漂移及调度退步。
- [原正式9h20m耗时解释](joint-generic-continuation-20261109/OFFICIAL-TIMING-EXPLAINED-20261110.md)。
- [参赛者推理后端调研](inference-backend-survey-20261110/README.md)：源码事实与我方未验证适配边界。

[返回分类](../README.md) · [返回总入口](../../README.md)

## 文档

- [Batch 吞吐与显存实测](batch-benchmark-20261028/results.md)
- [为什么固定2816能跑，真实混长2807却OOM](checkpoint-length-scaling-20261029/MIXED-LENGTH-DIAGNOSIS.md)
- [旧checkpoint benchmark的实际输入](checkpoint-length-scaling-20261029/ORIGINAL-INPUTS.md)
- [Gradient checkpointing：具体输入长度、显存与速度](checkpoint-length-scaling-20261029/REPORT.md)
- [输入长度、checkpointing与显存：实测研究报告](checkpoint-length-scaling-20261029/RESEARCH-REPORT.md)
- [把长度benchmark用于配置选择](checkpoint-length-scaling-20261029/SCALING-ANALYSIS.md)
- [输入长度与正负样本比例：快速诊断报告](length-label-benchmark-20261029/BENCHMARK-REPORT.md)
- [当前训练池：长度与正负比例](length-label-benchmark-20261029/data-inventory.md)
- [训练计划中的正负曝光](length-label-benchmark-20261029/exposure-ratios.md)
- [固定checkpoint输入长度诊断](length-label-benchmark-20261029/length-results.md)
- [当前 Jev Index：Qwen 相关模型 attention 清单](2026-11-09-attention-audit/attention-inventory.md)
- [Jev Index 中 Qwen 模型的 attention：按规模核查](2026-11-09-attention-audit/attention-survey.md)
- [Qwen3-0.6B 双向注意力：计划核查](attention/audit.md)
