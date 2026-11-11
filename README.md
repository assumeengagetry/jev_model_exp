# Jev 研究资料库

这里集中保存研究结论、研究路径，以及支撑结论的表格、图和精简证据。整理日期：2026-11-09。

## 当前方案与近期结果（截至2026-11-10）

**从头读：[当前方案完整说明](02-研究路径/06-五维训练与Readout/current-scheme-20261110/README.md)。** 按没有参与过对话、没有AI帮助也能理解的标准，分别写清任务、模型、数据、Teacher、损失、两段训练、正式推理、计分和人工复现缺口。

- [数据与训练每一步](02-研究路径/06-五维训练与Readout/current-scheme-20261110/DATA-AND-TRAINING.md) · [推理与评测](02-研究路径/06-五维训练与Readout/current-scheme-20261110/INFERENCE-AND-EVALUATION.md) · [人工复现手册](02-研究路径/06-五维训练与Readout/current-scheme-20261110/REPRODUCTION.md)。
- [为何仍未到28：事实、假设与缺失证据](01-研究结论/06-五维训练与Readout/current-scheme-20261110/BOTTLENECKS.md)。
- 当前joint final原正式public **24.97**；同权重加速版 **24.99、整套3h22m37.2s、2.765×**，不是private Full，也不是逐位无损加速。
- [Readout两层四臂最终报告](01-研究结论/06-五维训练与Readout/readout-study-20261108/FINAL-REPORT-20261110.md)：已完成并核验20来源；Cptr内部质量有信号，但换序明显弱于原pair，未替换正式模型。
- [近期全部报告导航](01-研究结论/近期实验-20261110.md)。本次只发布文档与聚合证据；实际执行源码按研究库规则留在库外辅助归档，不发布模型权重、原始语料或逐请求预测。

| 分类 | 内容 |
|---|---|
| [01 研究结论](01-研究结论/README.md) | 实验结果、复核报告、数据与外部方法调研结论 |
| [02 研究路径](02-研究路径/README.md) | 研究设计、假设、实施计划、阶段记录、历史草案 |
| [03 证据与图表](03-证据与图表/README.md) | 评分、配置、数据来源索引、结果表、图表和静态看板 |

## 按主题开始阅读

| 主题 | 结论入口 | 路径入口 | 阅读说明 |
|---|---|---|---|
| 蒸馏 | [研究结论](01-研究结论/01-蒸馏/distillation-exploration-20261104/REPORT.md) | [研究路径](02-研究路径/01-蒸馏/a800-kd-improve-round1-20261104/RESEARCH_AND_PLAN.md) | 优先阅读后续探索与第九波复核；旧汇集是当时快照。 |
| Language | [研究结论](01-研究结论/02-Language/language-understanding-study-20261029.md) | [研究路径](02-研究路径/02-Language/language_roadmap.md) | 历史九项 Language 口径；样本比例优先看复核稿。 |
| 专科与融合 | [研究结论](01-研究结论/03-专科与融合/five-specialists-20261030/report/REPORT.md) | [研究路径](02-研究路径/03-专科与融合/five-area-training-plan.md) | 四专科、混合、固定融合；另有 SLERP 和 SAMerging 对照。 |
| 数据与评测 | [研究结论](01-研究结论/04-数据与评测/benchmark-data-mining-20261031/REPORT.md) | [研究路径](02-研究路径/04-数据与评测/benchmark-data-mining-20261031/PLAN.md) | 任务画像、训练来源核查、数据转换与评测隔离。 |
| Attention 与效率 | [研究结论](01-研究结论/05-Attention与效率/2026-11-09-attention-audit/attention-survey.md) | [研究路径](02-研究路径/05-Attention与效率/2026-11-09-attention-audit/nine-source-check.md) | 公开资料与源码静态核查；效率实测单独保留。 |
| 五维训练与 Readout | [最新结论](01-研究结论/06-五维训练与Readout/readout-study-20261108/FINAL-REPORT-20261110.md) | [当前完整方案](02-研究路径/06-五维训练与Readout/current-scheme-20261110/README.md) | joint两段训练及两次public已完成；readout内部研究已完成，不与正式Index混排。 |

## 当前阅读顺序与边界

- 蒸馏先看[探索总报告](01-研究结论/01-蒸馏/distillation-exploration-20261104/REPORT.md)及[第九波完整结果](01-研究结论/01-蒸馏/distillation-exploration-20261104/ninth-wave-results/report/REPORT.md)：受控重复中 point 为 20.51、mixed 为 22.49；历史最高单次 23.96 未稳定复现。旧汇总保留其原观察日期，不作为当前统一排名。
- Attention 先看[最终综述](01-研究结论/05-Attention与效率/2026-11-09-attention-audit/attention-survey.md)和[逐项证据清单](01-研究结论/05-Attention与效率/2026-11-09-attention-audit/attention-inventory.md)：依据公开资料与源码静态核查，未运行模型；证据不足的条目保持“未确认”。
- 样本比例先看[RATIO-REVIEW](01-研究结论/02-Language/lu-pair-ratio-screen-20261029T133342Z/RATIO-REVIEW.md)；它更正了早期阶段稿的预算口径。
- [五维原标签训练](02-研究路径/06-五维训练与Readout/all-dimension-supervised-20261108/STATUS.md)是另一条“训练完成、未完整public评测”路线；[扩大训练](02-研究路径/06-五维训练与Readout/all-dimension-expanded-20261108/STATUS.md)仍暂停。[五维蒸馏准备](02-研究路径/06-五维训练与Readout/five-dimension-distilled-submission-20261109/README.md)的后续joint正式成绩已归档，优先看[当前结果](01-研究结论/06-五维训练与Readout/joint-generic-continuation-20261109/README.md)，不以历史进行中快照覆盖终点。
- 完整公共 37 项、冻结 pilot、历史 Language 九项使用不同口径，不跨口径直接排名。计划和早期草案不等于验证结论。

## 表格、图和阅读版

- [模型结果汇总表](03-证据与图表/08-综合汇总与历史记录/consolidated/model-results.csv) · [逐 benchmark 结果](03-证据与图表/08-综合汇总与历史记录/consolidated/benchmark-results.csv)（历史汇总，以具体新报告为准）。
- [104 个历史 Language 终点交互看板](03-证据与图表/02-Language/experiment-dashboard/index.html) · [PNG 图库](03-证据与图表/08-综合汇总与历史记录/png-export-20261105/index.html)。
- [训练机制阅读版](03-证据与图表/07-方法与背景/training-notes-rewritten-20261102/阅读版.html) · [项目背景综述](02-研究路径/07-方法与背景/jev.md) · [工作记录](02-研究路径/08-综合汇总与历史记录/jev-work-log-20261102.md)。

脚本、模型部署资源、原始训练数据、运行日志与批量下载快照已移至[库外辅助归档](../jev-辅助归档/2026-11-09整理/README.md)。所有原始文件另有整理前完整快照。分类规则、文件去向与校验说明见[整理记录](整理记录.md)。
