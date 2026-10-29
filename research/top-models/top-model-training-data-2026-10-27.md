# 顶部决策模型训练数据核查（2026-10-27）

本文只依据作者模型卡、作者技术报告、公开数据清单和数据集自身说明。未执行任何仓库脚本。

## Perplexity Decider

- v1.1 可核查训练清单：共 **626,033** 条，其中原有数据 **95,930** 条，Tasksource **530,103** 条。`source_rows` 的 **349 个来源/子任务条目合计恰好是 530,103**，不能误当成全体 626,033 条的完整来源表。
- 这些条目包括 FOL-nli 13,587、doc-nli 12,663、WANLI 11,328、DynaSent r1_votes 11,019、r2_votes 8,032、SPARTQA multiple-choice 7,902、MathQA 6,435；还包括 HelpSteer 1–3 的多个评分属性、civil_comments 多个投票属性、医疗/法律 NLI、安全与注入检测等。
- 13 个 procedural-typed-decisions 子任务在此清单中合计 **119,363** 条，覆盖实体信念追踪、表查找、路由、事件状态重建、算术、检索、证据充分性、部分观测校准等。
- v1 模型卡明确：用 Bespoke Nimble converters 将人工标注公开数据转为 Choice/Noul；Nimble 的评测行按 record ID 和 checksum 排除。
- v1.2 模型卡说明增加部分 Decision Index 基准的官方训练分割，并加入大量其他决策数据；仍致谢 Tasksource。但是未在该卡中给出 v1.2 精确训练条数或完整来源清单。不能把 v1.1 的 626,033 条说成 v1.2 的训练量。
- 已保存原始清单：[pplx-v1.1-data-manifest.json](pplx-v1.1-data-manifest.json)。

出处：
- https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/3b45dead91dfa6d95aad6b95764a606fab2bf7a6/training/data-manifest.json
- https://huggingface.co/perplexity-ai/pplx-decider-v1-27b
- https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b
- https://huggingface.co/perplexity-ai/pplx-decider-v1.2

## StartLux

- 技术报告把第一阶段数据称作 Lux-Release：27B 约 **0.8M** 条，小模型 0.8B–9B 约 **2.7M** 条。
- 第二阶段 Lux-Rounds 约 **1.7M** 条，主要来自基准训练分割及相关公开数据，也含程序生成、模型生成、select-all、teacher 标注和第一阶段回放。不同阶段可能重复，不能直接相加当作独立题数。
- 附录 A.3 列出的较早 preview-mixture 官方训练来源包括 BANKING77、CLINC150、HoVer、ContractNLI、ANLI、VAST、NLI4CT、RAGTruth、iSarcasmEval、GSM8K、New Yorker、WinoGrande、HellaSwag、cfcolor、Amazon ESCI、ACOS、When2Call、Habermas Machine、ForecastBench、Humicroedit 等。
- API-Bank/BFCL 用相关公开数据；GPQA Diamond/MMLU-Pro 只采用相关数据或版式信息；合成任务覆盖 appliance、因果、MuSR、棋、音乐等。
- 附录明确说该表是 **较早 preview 配方，非最终 checkpoint 的穷尽清单**。原文档“覆盖 33/38 train split”不应直接当作最终完整清单。
- 论文明确：模型生成的训练示例不再分发；未找到可直接下载的完整 Lux-Release / Lux-Rounds 训练集。

出处：
- https://arxiv.org/html/2610.17309v1#A1
- https://arxiv.org/html/2610.17309v1 （A.2–A.5，网页行 529–548）
- https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant

## Mammoth-1 / Elephant-1

- 作者已公开训练技术报告，因此旧文档“其余不公开”已过时。
- 两者都使用 Mix 8，共 **6.21M** 条。
- 核心决策技能 **3.62M / 58.3%**；安全与弃权 **1.00M / 16.1%**；评判与专业领域 **1.00M / 16.1%**；公开决策数据 **0.59M / 9.5%**。
- 公开部分概述为：**61 个公开数据集的 93 个任务、25 个社区决策数据集**，并包含代码执行/数学/棋/逻辑等程序答案任务。公开数据保留原标签；其中一部分具有人工投票软标签。
- 合成内容先由模型生成，再由程序或五模型评审组确定标签；评审组至少四票同意才保留，再清洗、去污染、独立审计、短跑训练。
- 报告没有给出完整 61 个数据集名字列表，也未提供完整 Mix 8 的可下载数据包。

出处：
- https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models （§2.4）
- https://huggingface.co/baselabs/mammoth-1

## d3 / d3-turbo

- 检查的作者模型卡以架构、API、评测和速度为主，没有列出完整训练数据来源、完整训练行数或可下载训练混合包。
- 不应把其仓库内的 Decision Index 结果数据集、性能测试集或公开评测数据集当成训练数据。
- 作者最近的 Vela 3.0 文章提及 d3 的 **1.31M-row soft-label decision mixture**，但这不能据此补出全部来源，更不能认为数据已公开下载。

出处：
- https://huggingface.co/vllm-sr/d3
- https://huggingface.co/vllm-sr/d3-turbo
- https://vllm-sr.com/blog/vela-3-0-name-it-vela-finds-it/

## 最实用、已确认可以直接加载的数据：Tasksource

数据集：https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions

2026-10-27 直接读取 README 的当前 `dataset_info` 得到以下数量（配置之间大量重叠，不能相加）：

| 配置 | train | validation | test | 用途 |
|---|---:|---:|---:|---|
| default | 999,832 | 15,000 | 15,000 | 精选文本混合 |
| full | 2,550,012 | 15,000 | 15,000 | 更大的文本池 |
| filtered-full | 1,305,478 | 无 | 无 | 历史商业标记子集，经清洗与 GLM 辅助审查 |
| vision | 722,314 | 972 | 475 | 多模态，体积远大于文本，不要误下载 |

`filtered-full` 基于 frozen revision `d2ab1d12be4463fb7ac1f877ed2193b3bb59642a`，并非 Perplexity v1.1 manifest 中那份 530,103 条文件的简单同名复刻。它从 2.5M 条中选择 1,305,478 条，预先排除非 commercial 元数据来源、指定基准来源及检测到的重叠，之后用 GLM-5.3-Flash 辅助检查；原文本、原标签和分布保持，不是 teacher 重标注。其检查并不证明对任意新评测全面无污染。

```python
from datasets import load_dataset
train = load_dataset(
    "tasksource/tasksource-jev-typed-decisions",
    "filtered-full",
    split="train",
    streaming=True,
)
row = next(iter(train))
print(row["state"], row["question"], row["options"], row["target"])
```

常用字段 `state / question / kind / options / target / source / group_id / license / license_use`。`choice`、`score` 的 target 是对齐选项的分布，`noul` 的 target 为 `[p_true]`，options 通常为空，转换时需要处理。

混合数据本身为 `license: other`，按行记录上游条款。`commercial` 是作者的尽力分类，并非自动覆盖所有上游条款。

出处与构建入口：
- https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/blob/main/README.md
- https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/blob/main/filtered-full/README.md
- https://github.com/sileod/tasksource/blob/main/docs/jev/README.md
- https://huggingface.co/datasets/tasksource/procedural-typed-decisions

原始数据卡副本保存在本目录，便于核对当日数量。
