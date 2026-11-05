# Jev 实验报告与大表总入口

**蒸馏完整汇集：[37 个完成条件的大表与研究总图](distillation-collection/REPORT.md)。** [全仓 PNG 图库：79 张逻辑图](png-export-20261105/index.html) · [PNG 路径清单](png-export-20261105/png-index.csv)。原图保留，PNG 独立导出；下方旧汇总作为此前整理快照保留，最新蒸馏结论请以上述入口及源报告为准。

> 无损整理：所有原文件、原目录和旧看板保留；这里仅合并同类记录并增加分类导航。下列状态来自本地归档，不是集群实时查询。

**最新蒸馏阶段结论：[A800 × H800 双线总汇](distillation-joint-summary/REPORT.md)。** A800 已测 KD 尚未超过普通 LS0.1；H800 领域保护当前为 22.90。两线评测范围不同，不能直接排名；以下旧表与新汇总冲突时，以新汇总及其链接的源报告为准。

## 先看结论

- **完整公共37项：MIXED_ALL 22.82 仍最高；SLERP 21.55、平均融合21.46、SAMerging 20.21。** SLERP相对平均仅 +0.09，仍低于混合训练1.27；不能据此认为融合已超过混合训练。
- **warm-start蒸馏：两档均未显示纯SOFT优于等量HARD。** 10万样本SOFT−HARD为 −1.99，100万样本为 −1.13；100万HYBRID−HARD仅 −0.11，不宜外推微小差异。
- **历史Language：最高单次36.33，四seed配方35.05±0.96。** 这是九项Language口径，不能与五维公共Index或pilot Index直接排名。
- **原始Qwen100万样本对照已完成：HARD 18.46、SOFT 17.54、HYBRID 19.10。** HYBRID相对HARD为 +0.64，SOFT为 −0.92；这是单次seed/固定子集观察。随机head起点1.76不是原始Qwen的zero-shot能力。

## 合并大表

| 大表 | 记录数 | 包含什么 |
|---|---|---|
| [模型结果总表](consolidated/model-results.csv) | 129 | 完整公共37项、冻结pilot、历史Language；必须按evaluation_scope与comparison_group筛选 |
| [逐benchmark结果](consolidated/benchmark-results.csv) | 1861 | 原生指标名、两位显示值、0–100 Skill及来源；不同native_metric不能混称accuracy |
| [训练与校准状态](consolidated/training-status.csv) | 23 | 四科全量、两档pilot、原始起点、8组dev短训及SAMerging校准；未完成评测不填假分数 |
| [资源测量](consolidated/resource-measurements.csv) | 64 | 49个显存探针、9个真实batch/长尾条件、6个长度诊断；保留OOM及测量范围 |
| [公开任务画像](consolidated/benchmark-profiles.csv) | 43 | 37活跃＋6归档任务，合并输入结构、歧义与来源覆盖；active标志不能忽略 |
| [数据接入登记](consolidated/data-register.csv) | 1631 | 983个来源/子任务与648个typed任务；register_kind分开，不能相加为独立数据集 |
| [图表索引](consolidated/figure-index.csv) | 50 | 同一图的PNG/SVG合并一行，直接链接原图，不复制图片 |
| [报告分类目录](consolidated/report-catalog.csv) | 43 | 所有实验线及方法/工程/备份原入口 |
| [原始表格目录](consolidated/source-tables.csv) | 68 | 原表位置、行列数与相同列结构；没有把重复基线、子集和全量机械叠加 |

**使用约定：** CSV可用Excel/WPS筛选；sources是项目根目录下的原件路径，`#`后为JSON记录定位。空值表示未归档或不适用，不是0分。新表中的小数按两位展示，精确原生值回查sources。所有这些分数都不是官方私测Full或线上排名。

## 一、完整公共37项：12个去重方案

同一批11.01万请求、全部候选，coverage均100%。原10方案表、SLERP11方案表与SAMerging对比表中的重复基线已合并，所有来源仍在模型结果总表中。

| 方案 | Knowledge | Language | Retrieval | Tools | Arts | 公共Index |
|---|---|---|---|---|---|---|
| MIXED_ALL | 9.69 | 17.44 | 46.26 | 26.08 | 17.78 | **22.82** |
| MERGE_SLERP_HIERARCHICAL | 8.56 | 16.50 | 35.29 | 38.25 | 10.13 | **21.55** |
| MERGE_AVERAGE | 8.47 | 16.60 | 35.24 | 37.85 | 10.10 | **21.46** |
| MERGE_SAMERGING | 8.21 | 15.57 | 33.43 | 35.33 | 9.08 | **20.21** |
| MERGE_TIES | 7.14 | 14.40 | 40.81 | 24.29 | 8.27 | **19.00** |
| SPECIALIST_KNOWLEDGE | 12.63 | 14.45 | 17.12 | 36.44 | 7.40 | **17.83** |
| SPECIALIST_RETRIEVAL | 2.71 | 5.17 | 42.58 | 22.91 | 4.21 | **15.17** |
| MERGE_DARE_TIES | 3.83 | 6.40 | 29.77 | 19.22 | 6.33 | **12.75** |
| MERGE_TASK_ARITHMETIC | 3.87 | 7.08 | 34.44 | 13.39 | 5.77 | **12.75** |
| SPECIALIST_ARTS | 4.21 | 7.82 | 12.13 | 21.78 | 18.10 | **11.33** |
| SPECIALIST_TOOLS | 4.04 | 9.14 | 5.85 | 34.70 | 1.09 | **11.03** |
| MERGE_DARE_TASK_ARITHMETIC | 1.56 | 2.88 | 16.55 | 8.67 | 5.38 | **6.58** |

[四专科原报告](../artifacts/five-specialists-20261030/report/REPORT.md) · [SLERP完整报告](../artifacts/four-specialists-slerp-20261102/report/REPORT.md) · [SAMerging方法与结果](a800-samerging-four-experts-20261103/report/SAMERGING_DETAILED_REPORT.md)

## 二、冻结pilot：预算与初始化分组

warm-start两档均从MIXED_ALL继续训练，使用同一冻结37项中的4,786个完整case、5,935次请求；不与上面的完整公共Index混排。BASELINE在每个预算组保留，避免把两组HARD/SOFT/HYBRID错当同一模型。

| 方案 | 已训练起点：10万 | 已训练起点：100万 | 原始起点：100万 |
|---|---|---|---|
| BASELINE | 22.39 | 22.39 | 1.76 |
| KD_HARD | 23.79 | 22.49 | 18.46 |
| KD_SOFT | 21.80 | 21.36 | 17.54 |
| KD_HYBRID | 23.42 | 22.38 | 19.10 |

SAMerging同一子集参照为20.09，但它学习融合系数，不能当作同训练量的KD组。原始Qwen的未训练head baseline为1.76，初始化不同，不能与warm-start对照合并成一组。

[10万样本报告与雷达/热力图](a800-kd-pilot-100k-20261102/REPORT.md) · [100万样本报告与雷达/热力图](a800-kd-scale-series-20261102/1m/REPORT.md) · [原始Qwen100万样本完整pilot报告](a800-kd-raw-qwen-1m-20261103/report/REPORT.md)

### 原始Qwen训练与评测：完成回执优先，本地快照 2026-11-03T11:27:07+00:00

| 组别 | 状态 | 已见训练对 | 步数 | 数据进度 |
|---|---|---|---|---|
| KD_HARD | evaluated | 100.00万 | 7813/7813 | 100.00% |
| KD_SOFT | evaluated | 100.00万 | 7813/7813 | 100.00% |
| KD_HYBRID | evaluated | 100.00万 | 7813/7813 | 100.00% |

训练loss定义不同，不据此跨组判质量。[完成报告与最新结果图](a800-kd-raw-qwen-1m-20261103/report/REPORT.md)优先于[较早训练进度图（原件保留）](a800-kd-raw-qwen-1m-20261103/report/TRAINING_PROGRESS_20261103.md)；状态来源为[LIVE_STATUS](a800-kd-raw-qwen-1m-20261103/LIVE_STATUS.json)及各组complete.json。

## 三、历史Language与资源研究

历史Language保留104个终点（含4个工程链路终点），不混入跨领域榜单。现有看板保持原样；49个前后向资源单元、9个带optimizer更新的吞吐条件与6个推理长度诊断也在资源大表中分scope，不做跨机制速度排名。
历史终点采用已整理看板的runs记录，不再把runs.csv、runs-through-pair-ratio.csv、seed均值groups及融合副本重复相加。

[历史Language看板](experiment-dashboard/index.html) · [历史阶段汇报](language-understanding-study-20261029.md) · [显存与重算机制](checkpoint-length-scaling-20261029/RESEARCH-REPORT.md) · [输入长度与标签](length-label-benchmark-20261029/BENCHMARK-REPORT.md)

## 四、按主题找原档

### 完整公共评测与专科融合

| 资料 | 状态/口径 |
|---|---|
| [四专家SAMerging](a800-samerging-four-experts-20261103/report/SAMERGING_DETAILED_REPORT.md) | 完整公共37项完成 |
| [分层SLERP方法与阶段记录](slerp-supplement-20261102/PLAN.md) | 最终报告另见artifacts中的完成结果 |
| [四专科、混合与五种固定融合](../artifacts/five-specialists-20261030/report/REPORT.md) | 完整公共37项完成 |
| [分层SLERP正式评测与11条件对照](../artifacts/four-specialists-slerp-20261102/report/REPORT.md) | 完整公共37项完成 |

### 蒸馏pilot与规模对照

| 资料 | 状态/口径 |
|---|---|
| [10万样本warm-start蒸馏](a800-kd-pilot-100k-20261102/REPORT.md) | 三组训练与冻结子集评测完成 |
| [原始Qwen100万样本蒸馏](a800-kd-raw-qwen-1m-20261103/report/REPORT.md) | 三组训练与冻结子集评测完成；与warm-start分组 |
| [蒸馏规模曲线：10万/100万](a800-kd-scale-series-20261102/README.md) | 100万档完成；后续档以各自状态为准 |

### 历史Language单维研究

| 资料 | 状态/口径 |
|---|---|
| [历史Language专科与融合分析](axis-specialization-and-merging-20261030/REPORT.md) | 历史完整九项结果 |
| [历史Language交互看板](experiment-dashboard/index.html) | 104个完整终点；不是当前跨域五维看板 |
| [Language阶段实验与附录](language-understanding-study-20261029.md) | 104个历史完整终点的主要研究入口 |
| [Qwen3-0.6B：Language Understanding 首轮基线实验报告](language-baseline-20261027.md) | 保留原文；以文内观察时间为准 |
| [lu-batch-lr-pilot-20261029](../research/lu-batch-lr-pilot-20261029/STUDY-REVIEW.md) | 历史研究原档 |
| [lu-pair-ratio-screen-20261029T133342Z](../research/lu-pair-ratio-screen-20261029T133342Z/RATIO-REVIEW.md) | 历史研究原档 |
| [lu-specialist-fusion-20261029T152144Z](../research/lu-specialist-fusion-20261029T152144Z/FUSION-REPORT.md) | 历史研究原档 |

### 数据与公开任务画像

| 资料 | 状态/口径 |
|---|---|
| [公开任务数据画像](benchmark-data-mining-20261031/REPORT.md) | 37活跃任务与6归档任务分开 |

### 训练效率与资源研究

| 资料 | 状态/口径 |
|---|---|
| [真实batch与长尾吞吐](batch-benchmark-20261028/results.md) | 9个条件；不等于模型质量分数 |
| [长度、重算与显存机制](checkpoint-length-scaling-20261029/RESEARCH-REPORT.md) | 49个资源单元 |
| [长度与正负比例研究](length-label-benchmark-20261029/BENCHMARK-REPORT.md) | 诊断与正式模型分数分开 |

### 方法笔记与实验记录

| 资料 | 状态/口径 |
|---|---|
| [训练机制与效率阅读笔记](training-notes-rewritten-20261102/阅读版.html) | 方法笔记 |
| [Jev 决策模型实验工作记录](jev-work-log-20261102.md) | 保留原文；以文内观察时间为准 |

### 方案与文献调研

| 资料 | 状态/口径 |
|---|---|
| [8卡A800蒸馏资源方案](a800-8gpu-distillation-20261031/REPORT.md) | 方案 |
| [原始Qwen对照的前置方案](fresh-qwen-1m-distillation-20261103/PROGRESS.md) | 以raw-qwen目录的后续快照为准 |
| [榜单蒸馏与社区方案调研](leaderboard-distillation-20261102/REPORT.md) | 调研，不是本项目复现成绩 |
| [后续实验设计](next-experiments-plan-20261030/REPORT.md) | 方案 |
| [早期Teacher证据核查](teacher-distillation-evidence-20261030/REPORT.md) | 保留原观察日期，不当作当前未完成结论 |
| [2026-11-03 实验报告与图表入口](LATEST_EXPERIMENT_REPORTS_20261103.md) | 保留原文；以文内观察时间为准 |
| [集群环境重建、实验状态还原与后续规划](cluster-recovery-and-experiment-plan-20261030.md) | 保留原文；以文内观察时间为准 |
| [实验总规划与当前位置](experiment-roadmap-status-20261029.md) | 保留原文；以文内观察时间为准 |
| [后续实验：两小时快筛执行约定](experiment-timebox-2h.md) | 保留原文；以文内观察时间为准 |
| [Qwen3-0.6B：四专科训练、四科混合与模型融合](five-area-training-plan.md) | 保留原文；以文内观察时间为准 |
| [Decision Index 榜单模型蒸馏方法调研报告](leaderboard-distillation-20261103.md) | 保留原文；以文内观察时间为准 |
| [Language快筛阶段结论与当前阻塞](lu-progress-20261029T0320Z.md) | 保留原文；以文内观察时间为准 |
| [旧版六方向方案（已替代）](multi-axis-specialist-vs-mixed-fusion-20261030.md) | 保留原文；以文内观察时间为准 |
| [attention](../research/attention/audit.md) | 历史研究原档 |
| [language-axis](../research/language-axis/README.md) | 历史研究原档 |
| [top-models](../research/top-models/procedural-typed-decisions.README.md) | 历史研究原档 |
| [zero-shot-label-nli](../research/zero-shot-label-nli/README.md) | 历史研究原档 |

### 工程、恢复与备份

| 资料 | 状态/口径 |
|---|---|
| [集群接管、恢复与实现记录](cluster-followup-20261031/four-specialists-change.md) | 历史/阶段记录，不能替代最终评分 |
| [工作量台账及来源登记](cluster-work-inventory-20261102/WORK-LEDGER.md) | 登记条目不等于独立实验次数 |
| [集群工作区历史快照](cluster-workspace-snapshot-20261030/README.md) | 历史快照 |
| [ModelScope备份与恢复资料](modelscope-backup-20261031/NEW-MACHINE-RESTORE.md) | 本地回执；本次未实时验证云端 |
| [SCP归档、迁移清单与工具](scp-backup-20261031/INDEX.md) | 历史资产保护与恢复材料 |
| [Student显式FlashAttention3部署记录](student-flash-attention3-20261103/CHANGE.md) | 部署证据；不覆盖历史已完成训练的真实后端 |

## 工作记录

[Jev工作记录（保留历史正文并增加当前总览）](jev-work-log-20261102.md)

原目录、原表、原图、历史文档均保留；本次不运行模型、不新增权重或逐条结果副本。
