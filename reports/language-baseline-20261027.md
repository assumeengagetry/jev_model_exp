# Qwen3-0.6B：Language Understanding 首轮基线实验报告

**实验日期：**2026-10-27  
**状态：**正式训练、内部测试、Language 九项评测均已完成。  
**实验目录：**`/home/jovyan/assumeengage/runs/language-baseline-20261027T061200Z`

本报告仅整理已落盘的实验记录和测量结果，没有追加训练、重新评测或执行新的对照实验。

## 一、结果摘要

| 项目 | 实测结果 |
| --- | ---: |
| 正式训练 | 1 epoch，14,171 个优化器更新步，已完成 |
| 内部测试样本数 | 25,004 |
| 内部测试 Accuracy | **82.40%** |
| 内部测试 Macro-F1 | **82.39%** |
| Decision Index 0.3 Language Skill | **17.74 / 100** |
| Language Raw 聚合分 | **43.05 / 100** |
| Language 评测覆盖率 | **100%** |
| 已完成评测请求 | **31,715 / 31,715** |
| Unsupported / 错误 / 未完成请求 | **0 / 0 / 0** |

**口径：**内部测试是 tasksource 留出的同源测试集；Language 是按 Decision Index 公开 0.3 版本计分逻辑执行的本地单维度评测。二者不是同一种分数。Language Skill 17.74 不等于准确率 17.74%，Language Raw 也不是统一准确率。本轮没有全榜总分或线上榜单排名。

## 二、实际训练的模型与配置

| 配置 | 本轮实际值 |
| --- | --- |
| 基座 | `Qwen/Qwen3-0.6B`，复用已下载的真实权重 |
| 架构 | `Qwen3ForSequenceClassification` |
| 分类头 | `Linear(hidden_size, 2, bias=False)`；从随机初始化开始训练 |
| 任务 | premise + hypothesis 的序列级二分类，交叉熵损失；不是聊天生成式 SFT |
| 类别 | `0=entailment`；`1=contradiction` |
| 微调方式 | Unsloth + SEQ_CLS LoRA；分类头同时参与训练和保存 |
| 可训练参数 | 10,094,592 |
| LoRA | r=16，alpha=32，dropout=0；attention 与 MLP 投影层 |
| 精度 | BF16；未使用 4-bit 量化 |
| 硬件与并行 | 4 张 NVIDIA H800 NVL，DDP |
| Batch | 每卡 32，梯度累积 1，有效 batch size 128 |
| 优化 | 学习率 1e-4，cosine 调度，warmup 3%，weight decay 0.01 |
| 训练长度上限 | 4,096 tokens |
| 随机种子 | 3407 |
| Checkpoint 选择 | 按 validation loss 最低选择，不使用内部 test 选择 |

环境记录：Python 3.11.17、PyTorch 2.11.0+cu130、Transformers 5.5.0、PEFT 0.18.1、Unsloth/unsloth_zoo 2026.10.9、datasets 4.3.0、accelerate 1.15.0。

先前已完成 32 步四卡分类试跑，确认分类头和 LoRA 能更新、保存和重载；该试跑仅是链路检查，不作为正式能力分数。

## 三、本轮使用的数据

- 数据集：`tasksource/zero-shot-label-nli`，`default` 配置；未使用 `triplet`。
- 原始 revision：`9f0763b798363a6d4ca8f65de2dc80e0fcb48ae2`。
- 原始文件实际标签只有 0、2，已映射为 `{0: 0, 2: 1}`。虽然数据卡声明三个类别，本轮实际训练的是二分类模型。

| 划分 | 原始样本数 | 处理后实际使用 | 移除数量 |
| --- | ---: | ---: | ---: |
| Train | 1,899,993 | **1,813,810** | 86,183 |
| Validation | 24,971 | **24,609** | 362 |
| Test | 25,440 | **25,004** | 436 |

本轮记录了重复样本、相同文本对的标签冲突、跨划分正文关联、评测内容关联以及不确定来源任务的过滤。整任务排除的来源包括 `isarcasm/intended`、`ragtruth/grounding`、`semeval2014-absa/laptops`、`semeval2014-absa/restaurants`。具体计数见 `data-report.json`。

处理后的训练集含 373,382,083 个非 padding tokens，平均有效长度约 205.86 tokens。训练长度处理影响 21,761 条样本，约占 1.20%；保留 hypothesis，按预算处理 premise。**该训练截断设置没有用于正式 Language 评测。**

## 四、训练完成情况与内部测试

| 项目 | 结果 |
| --- | ---: |
| 完成 epoch | 1.0 |
| 完成更新步数 | 14,171 |
| Trainer 记录的运行时间 | 6,003.54 秒，约 100 分钟 |
| 平均训练速度 | 2.36 步/秒；302.12 样本/秒 |
| 训练过程报告的平均 loss | 0.3879 |
| 最优 validation loss | 0.3522 |
| 最优 validation checkpoint | `checkpoint-13000` |
| 最终内部测试 loss | 0.3651 |
| 最终内部测试 Accuracy | 82.4028% |
| 最终内部测试 Macro-F1 | 82.3905% |

训练完成时已恢复最优 validation checkpoint，并将其导出到 `checkpoints/final`。因此，该目录是验证集选出的最终 LoRA/分类头产物，而不只是最后一步的权重；加载时仍需原始 Qwen3-0.6B 基座。内部测试和后续 Language 评测都使用该最终产物。

## 五、Language 评测方法与完成范围

- 工具：Decision Index 0.3；代码 commit 为 `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`。
- 范围：仅 Language Understanding 的 catalog `11,12,29,38,39,40,41,42,59`，不是全维度 suite。
- 规模：31,715 个请求，131,360 个问题字段；四张 GPU 分片执行，关联请求组不拆散。
- 推理：Transformers + PEFT 加载训练后的序列分类模型；不使用 CausalLM 生成答案。
- 适配：将 state、问题 instructions 和候选 criteria 转为文本对；候选分数取 entailment logit 减 contradiction logit，再对全部候选做固定温度 1 的 softmax。不是为不同 benchmark 手写答案规则。
- 长度：评测不截断输入、不删除候选；模型配置支持上限为 40,960 tokens。本轮没有 Unsupported 请求。
- 计分：使用官方 0.3 的轨道选择、随机水平校正、benchmark 权重、关联组规则等计分函数；单项 Skill 不是原始指标的简单百分数。
- 时间：流水线记录于 09:54:46 UTC 启动评测，10:03:16 UTC 完成，约 **8 分 30 秒**。

### 九项实测结果

下表“原始/轨道聚合分”将 0–1 的分数乘以 100 展示；各行指标不同，不能都解释为准确率。九项覆盖率均为 100%，Unsupported、errors、pending 均为 0。

| ID | 数据集 | 请求数 | 原始指标口径 | 原始/轨道聚合分 | Skill / 100 |
| --- | --- | ---: | --- | ---: | ---: |
| 11 | ContractNLI | 123 | Macro-F1 | 46.73 | 22.97 |
| 12 | ANLI | 3,200 | Macro-F1 | 30.37 | 0.00 |
| 29 | HellaSwag | 10,042 | Accuracy | 44.10 | 25.47 |
| 38 | ACOS | 1,565 | Per-review F1 | 7.87 | 4.92 |
| 39 | FinEntity | 979 | Macro-F1 | 76.27 | 65.10 |
| 40 | iSarcasmEval | 4,600 | 官方子轨道聚合 | 31.23 | 11.53 |
| 41 | VAST | 3,006 | Macro-F1 | 47.23 | 20.85 |
| 42 | NLI4CT | 5,500 | Macro-F1 | 54.06 | 10.62 |
| 59 | RAGTruth | 2,700 | 幻觉类别 F1 | 51.87 | 0.21 |

iSarcasmEval 的原生 `score` 字段为 null，因其按不同子任务分别计分；表内 31.23 来自官方轨道聚合后的 `raw_coverage_adjusted`，不是单一 Macro-F1。Language 总分使用官方权重，不是上表九个 Skill 的简单算术平均。

## 六、从已测结果能得出的结论

1. **首轮真实小模型分类基线已经建立。**完整完成了四卡训练、内部留出测试以及 Language 九项本地评测；不是仅跑通脚本或微型随机模型测试。
2. **内部测试与外部 Language 分数存在明显差距。**内部 Accuracy 为 82.40%，但 Language Skill 为 17.74；不能把内部准确率当作榜单能力。
3. **分项能力不均衡。**本轮 FinEntity 相对最强，Skill 为 65.10；ANLI 校正后的 Skill 为 0，RAGTruth 为 0.21，ACOS 为 4.92。ANLI 的 Skill 为 0 不表示它答对了零条，而是随机水平校正后的结果。
4. **目前不能宣称与该维度最高模型相当。**本轮没有记录同版本、同协议的顶尖模型对照，也没有数据量或模型规模的控制实验。因此不能仅凭此次结果判断“数据已经足够”或确定“必须换模型”。

## 七、未测内容及结果边界

- 未评测其他能力维度，未形成全榜总分或线上榜单排名。
- 本轮为 Language-only 重建和局部计分：`language_complete=true`，全局 `complete=false`；没有宣称完成整套 suite 的全局 hash 验证。
- 未测未经分类训练的原始 Qwen3 同协议成绩；随机初始化 score 头的训练前输出不能作为原始 Qwen3 能力基线。
- 未做不同数据规模、模型规模、微调方案或随机种子的对照；未测结果的多次运行方差。
- 报告只描述本轮已实际执行的实验，不把未经测试的改进方向写成结论。

## 八、产物位置与证据

所有下列路径均相对本次实验目录：

```text
/home/jovyan/assumeengage/runs/language-baseline-20261027T061200Z/
├── config.json                     # 实际模型、训练配置
├── environment.json                # 依赖版本
├── model-spec.json                 # 分类模型与可训练参数
├── data-report.json                # 样本划分、过滤及长度统计
├── training-complete.json           # 训练完成及最优 checkpoint 记录
├── internal-test.json               # 内部测试实测指标
├── train.log                       # 训练日志
├── training-progress.jsonl          # 历史 step/loss/吞吐记录
├── checkpoints/final/              # 最终 LoRA 与分类头
├── language-suite/language-manifest.json
└── evaluation/
    ├── language-score.json          # 本报告的 Language 总分与九项分数来源
    ├── benchmark-summary.json       # 单项原生指标
    ├── model-provenance.json        # 实际参评产物记录
    ├── launch.json                  # 四卡分片和实际评测命令
    └── results.jsonl                # 完整逐请求评测输出
```

**一句话结论：Qwen3-0.6B + 二分类 LoRA 的首轮训练与 Language 九项评测已完成；内部测试 Accuracy 82.40%，Language Skill 17.74/100，覆盖率 100%，已形成可追踪的首轮基线，但尚不足以证明达到该维度顶尖水平。**
