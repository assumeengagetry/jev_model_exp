# 第一轮：线性分类头 + Language Understanding

核查日期：2026-10-27。目标是先固定一个小模型和常规分类头，只研究语言理解这一个榜单维度的数据。

## 分类头就用这个

采用标准的 **Linear + 有效选项 softmax**。与 Perplexity Decider 的独立 255 路 readout 形式一致；小模型使用自己的 hidden_size，不能直接套 27B 模型的 5120 维权重。

```python
head = torch.nn.Linear(hidden_size, 255, bias=False)
logits = head(h)  # h：每道题末尾、位于全部选项之后的 marker hidden state
logits = logits.masked_fill(~valid_options, float("-inf"))
loss = torch.nn.functional.cross_entropy(logits, gold_option_index)
probs = logits.softmax(dim=-1)  # 推理时输出概率
```

`valid_options` 是每道题的合法选项 mask。255 维是选项位置，不是 255 个永久语义类别。第一轮训练 backbone 与 head，使用普通交叉熵；分类头结构在各个数据量实验中固定。

来源：[PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)、[Perplexity Decider readout 说明](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b#checkpoint-format-and-serving)。这里只确定接法，尚未创建模型或运行训练。

## 为什么选这个维度

当前榜单有 5 个独立计分维度。元数据另定义 Games & Strategy，第六类标有 `folded_into: knowledge`，不作为独立第六个计分维度。

Language Understanding 包含以下 **9 个评测任务**，覆盖推断、常识续写、情感、讽刺、立场和证据支持判断。多数是二分类或三分类。选择它是为了把第一轮的数据范围限定下来，不表示它最容易；ContractNLI 和 NLI4CT 还涉及长文本。

## 已找到并下载的训练来源

原始文件就在本目录。固定 revision、下载链接、选择哪些 train 记录和本地校验和见 [manifest.json](./manifest.json)。以下是原始数据规模，尚未去重、去污染或统一转换；评论数、合同数和转换后的判断题数不能直接视为同一种单位。

| 榜单任务 | 用于训练的数据 | 已核验规模 | 输入和转换方式 |
|---|---|---:|---|
| ANLI | [facebook/anli](https://huggingface.co/datasets/facebook/anli) 的 train_r1/r2/r3 | 162,865 对 | 前提 + 假设 → 蕴含 / 中立 / 矛盾；三个 round 都用 |
| HellaSwag | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) 的 train | 39,905 题 | 上下文 + 4 个候选续写 → 正确候选；榜单使用 validation，不能混入训练 |
| ContractNLI | [官方 ZIP](https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip) 内 train.json | 423 份合同，7,191 个判断 | 完整合同 + 17 个固定命题 → 蕴含 / 矛盾 / 未提及；不使用金标证据截取输入 |
| ACOS | [NUSTM/ACOS](https://github.com/NUSTM/ACOS) 的 laptop / restaurant train | 2,934 + 1,530 条评论 | 评论 + 方面类别与情感组合 → 是否成立；需构造负例，不能仅训练金标正组合 |
| iSarcasmEval | [官方英文 train](https://github.com/iabufarha/iSarcasmEval/tree/dfc708b53bde1bb571abfb5692f63231c2232195/train) | 3,468 条 | 原始 tweet → 是否讽刺；榜单主指标是英文 Task A；rephrase 不放进判断输入 |
| VAST | [官方 vast_train.csv](https://github.com/emilyallaway/zero-shot-stance/blob/e7c4775182b184730f350995f8260579c9e066fe/data/VAST/vast_train.csv) | 13,477 条 | 文本 + 目标话题 → 支持 / 反对 / 中立 |
| NLI4CT | [官方 training_data.zip](https://github.com/ai-systems/Task-2-SemEval-2024/blob/7f32fa6c7db43e577f22e9fc6c28eef1cc6d223e/training_data.zip) | 1,700 个 train 命题 | 保留题目所需完整临床试验章节及对照试验；判断命题是否成立，不喂金标 evidence 索引 |
| RAGTruth | [官方原始文件](https://github.com/ParticleMedia/RAGTruth/tree/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset)，仅 response.split=train | 15,090 个回答 | 用 source_id 连接 source_info 的原始 prompt；输入 prompt + response，标签为是否存在任意标注幻觉片段 |
| FinEntity | 用独立来源 [SEntFiN](https://github.com/pyRis/SEntFiN) 补训练 | 10,753 个标题，14,404 个实体情感标签的候选池 | 新闻标题 + 目标实体 → 正面 / 负面 / 中立；按标题分组留验证集，再展开实体标签 |

前 8 项合计约 24.8 万个原始样本/判断单位，另加 SEntFiN 金融实体情感候选池。这个合计只是量级，不能当作处理后的 JSONL 行数。阿拉伯语 iSarcasm train（3,102 条）也已下载，属于可选补充，不计入上面的英文主指标数据量。

**FinEntity 特别处理：** [榜单源说明](https://github.com/apolinario/decision-index/blob/main/docs/suite.md)使用其完整的 979 份未拆分文档做评测，因此不存在可直接使用的官方独立 train。不能从这个文件随意切 train。SEntFiN 是替代训练来源，不代表已经拿到榜首模型的训练配方，也不保证迁移后足以达到榜首。

SEntFiN 的 CSV 有 13 行实体名称中的撇号未转义，直接对 Decisions 列使用 ast.literal_eval 会报错；转换时需明确解析标签边界或修复后统计。按标题分组划分，避免同标题的不同实体进入训练和验证两边。

ContractNLI ZIP 包含 train/dev/test，NLI4CT ZIP 包含 train/dev，RAGTruth response 文件包含 train/test；下载到本地不等于这些记录都可以训练。必须按 manifest 的 train_selector 选择。RAGTruth 的 17,790 个回答已核验都能连接到 source_info。

原始来源许可仍然适用，manifest 保留许可记录。特别是 RAGTruth 的上下文包含限制再分发的来源，原始文本只用于本地构建，不随报告或模型重新发布。

## 对照目标

榜单快照时间：2026-10-26 15:30:58 UTC，Decision Index 0.4.1。本地保留了 index JSON；线上榜单会继续变化。

| 参考模型 | 参数规模 | Language 公开 skill × 100 |
|---|---:|---:|
| baselabs mammoth-1 | 27B | **76.11** |
| Jeffan-4B-2010 | 约 4.21B | **69.78** |

来源：[榜单](https://huggingface.co/spaces/multimodalart/jev-decision-index)、[原始 index.json](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json)。这不是普通准确率，也不是私测 combined 分数。应使用相同版本的官方 kit，按相同 9 个任务和 chance correction 聚合比较；只跑 ANLI 不代表跑完这个维度。

## 第一轮实验判据

1. 固定小模型、head、输入格式和任务混合比例，按源任务分层构造 25% / 50% / 100% 的嵌套训练子集。记录实际样本数、训练 token、步数和计算量。
2. 每档先在各源独立 dev 或按文档/标题/来源分组留出的验证集比较；榜单评测记录不参与训练。FinEntity 只用来评测。最终用官方 kit 对照完整 Language 维度。
3. 数据量增加仍持续涨分，继续投入数据有依据；固定数据后更大或改动后的模型显著提升，说明模型容量/结构方向值得投入。低分本身不能断言“数据和模型都不对”。
4. 接近 76.11 可以作为进入结构实验的阶段目标；还需看多个种子的波动、9 个任务的分项和独立验证集，才能判断是否只是对公开基准适配。不能由单个模型的一条曲线宣称验证了 scaling law 或数据已经充分。

当前交付是 **已下载的原始数据、训练划分清单、分类头选择与对照目标**；统一 schema、去重/去污染和训练尚未执行。
