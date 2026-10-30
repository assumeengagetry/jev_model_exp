# Qwen3-0.6B：五专科训练、全数据混合与模型融合

## 1. 实验目标与模型

按榜单五个方向划分专科：**Knowledge、Language、Retrieval、Tools、Arts**。棋类归入Knowledge，不再单独训练Games模型。

本轮回答两个问题：

1. 专科模型在自己的方向上，是否优于全数据混合训练的模型？
2. 将五个专科融合成一个模型后，相比混合训练，在五个方向上分别提升或损失多少？

| 模型 | 训练数据 |
| --- | --- |
| `SPECIALIST_KNOWLEDGE` | Knowledge专科数据 |
| `SPECIALIST_LANGUAGE` | Language专科数据 |
| `SPECIALIST_RETRIEVAL` | Retrieval专科数据 |
| `SPECIALIST_TOOLS` | Tools专科数据 |
| `SPECIALIST_ARTS` | Arts专科数据 |
| `MIXED_ALL` | 五个专科实际使用数据的合并池 |

**共训练6个模型，再复用五个专科权重生成融合模型。**

## 2. 模型与环境

- 基座统一使用 **`Qwen/Qwen3-0.6B`**，只需准备一份。
- 沿用候选匹配二分类：`Qwen3ForSequenceClassification`，匹配/不匹配两个输出。
- 五个专科与混合模型从同一份初始分类模型出发，使用相同分类头与输入格式。
- 训练采用 **BF16、causal attention、LoRA r32/alpha64**，分类头一起训练和保存。
- 复用集群已有的PyTorch、Transformers、PEFT、datasets、accelerate、pyarrow及FlashAttention环境。
- 模型和数据已有的直接使用，缺少的由集群下载；不需要为本计划重新搭一套训练框架。

## 3. 数据总体安排：专科数据尽量充分

### 通用底池

以 **`tasksource/zero-shot-label-nli`的default**作为底池，约190万原始训练pair；之前约181万是处理后的规模。按来源任务和内容划入五个专科，而不是平均切成五份。

还可补充Tasksource的`tasksource-jev-typed-decisions`完整池、`procedural-typed-decisions`和现有本地决策数据，统一转为同一种候选匹配格式，再按五科归类。

每个专科的数据组成：

```text
该方向的Tasksource数据
＋ 对应benchmark的训练部分
＋ 同能力的其他公开训练集
＋ 必要的程序生成数据
```

**不再沿用50万pair总上限，也不把正式专科限定为3M、17M或20M token短训。优先使用充分的训练数据，把五个专科本身训练好，再讨论融合。**

各来源合并时去掉重复内容。Benchmark补充使用train；没有train的项目，用同能力的训练数据补充。

## 4. 五个专科分别准备哪些数据

### 4.1 Knowledge：知识、数学、推理、代码与棋类

**Tasksource部分：**科学与学科问答、常识推理、数学、逻辑、因果、代码行为和棋类决策任务。

**优先补充：**

| 能力 | 训练数据 |
| --- | --- |
| 数学 | GSM8K、MATH的训练部分；现有数学选择与数值判断数据 |
| 科学与常识 | ARC-Easy、ARC-Challenge、OpenBookQA、CommonsenseQA、SciQ、WinoGrande |
| 专业知识 | MedMCQA等专业问答训练数据 |
| 代码 | MBPP、CodeContests等训练部分，转换为代码输出/正确性/候选解判断 |
| 棋类 | searchless_chess的训练数据，转换为给定局面的候选棋步选择 |
| 规则推理 | 程序生成的逻辑、数字关系、状态推演、因果与约束判断 |

知识类来源尽量保留各学科覆盖，不只集中在数学。GPQA、MMLU-Pro、BBH、HLE等作为对应能力的评测项目。

### 4.2 Language：语言理解、情感、立场、蕴含与长文判断

**Tasksource部分：**文本匹配、情感、讽刺、立场、文档理解、法律/临床蕴含、grounding等。

**优先补充：**

| 能力 | 训练数据 |
| --- | --- |
| 通用蕴含与匹配 | ANLI、MultiNLI、SNLI、WANLI、DocNLI |
| 常识续写 | HellaSwag |
| 情感与方面情感 | ACOS、SST-2、TweetEval、Yelp/商品评论训练数据 |
| 讽刺与立场 | iSarcasmEval、VAST |
| 法律与临床文本 | ContractNLI、NLI4CT |
| 幻觉与依据判断 | RAGTruth训练部分 |
| 金融实体情感 | SEntFiN等独立金融情感训练数据 |

现有ANLI、HellaSwag、ACOS、RAGTruth等数据继续利用，再扩充通用NLI与长文来源，避免专科只围绕几个小任务反复采样。

### 4.3 Retrieval：检索、相关性、意图和大标签分类

**Tasksource部分：**意图识别、主题分类、文档相关性、证据判断、垃圾邮件/钓鱼识别等。

**优先补充：**

| 能力 | 训练数据 |
| --- | --- |
| 意图识别 | BANKING77、CLINC150+OOS、MASSIVE |
| 查询与产品相关性 | Amazon ESCI训练部分，关联产品标题与描述 |
| 查询与文档相关性 | MS MARCO passage ranking训练数据及其他有train的检索语料 |
| 多跳证据核验 | HoVer训练部分及配套文档 |
| 大标签分类 | AG News、DBpedia及现有多类别分类任务 |
| 邮件分类 | 现有Tasksource与公开spam/phishing训练来源 |

除了二分类，还要充分覆盖多候选、大标签集和相关性排序任务。

### 4.4 Tools：工具选择、路由、调用和业务流程

**Tasksource部分：**工具/API选择、任务路由、政策执行、结构化状态和动作判断。

**优先补充：**

| 能力 | 训练数据 |
| --- | --- |
| 调用/澄清/拒绝决策 | When2Call的train_sft、train_pref |
| API与工具调用 | API-Bank训练对话、Glaive function-calling-v2、xLAM function-calling-60k、ToolBench训练部分 |
| 工具检索 | ToolRet-Training-20w |
| 路由与流程 | 现有路由数据、程序生成的工单/业务流程/状态机任务 |
| 结构化操作 | JSON字段判断、参数约束、动作合法性、多步状态变化 |

工具数据统一转换为决策时刻的候选匹配任务，不改成聊天生成式SFT。规则与状态任务可以程序扩量，增加工具专科的数据覆盖。

### 4.5 Arts：幽默、诗歌、音乐与人类偏好

**Tasksource部分：**幽默、诗歌、偏好、音乐及相关判断任务；普通主题分类仍归Retrieval，普通情感理解归Language。

**优先补充：**

| 能力 | 训练数据 |
| --- | --- |
| 幽默选择 | Humicroedit训练部分 |
| 描述与标题匹配 | New Yorker caption contest的matching训练部分，使用文本描述 |
| 群体偏好 | Habermas Machine的训练部分 |
| 色彩偏好 | cfcolor的训练ratings，构造palette偏好选择 |
| 回答质量与偏好 | HelpSteer、HelpSteer2、UltraFeedback、HH-RLHF等训练来源 |
| 诗歌判断 | 独立诗歌语料构造原诗/扰动诗、押韵与节奏选择 |
| 音乐判断 | 独立MIDI/和弦训练语料，程序生成音符、和弦与和声识别任务 |

Arts不只用Humicroedit与New Yorker两小份数据。优先把偏好、质量判断与程序生成的音乐/诗歌任务补起来，使这个专科也有充分训练量。

## 5. 训练配置与加速方式

### 推荐起点

| 项目 | 配置 |
| --- | --- |
| 微调 | LoRA r32、alpha64，分类头同时训练 |
| 精度 | BF16 |
| Attention | causal；优先复用集群已有的FlashAttention/varlen实现，没有则使用SDPA fused attention |
| 单卡microbatch | **32 pair** |
| 单卡梯度累计 | **4** |
| 有效batch | **128 pair** |
| 学习率 | **1e-4**起步，AdamW，cosine，weight decay 0.01 |
| 训练长度 | **4096**起步；长文数据较多的专科再增加长序列训练 |
| 梯度checkpointing | 显存充足时关闭以减少重算；长序列或显存吃紧时开启 |
| 数据组织 | 长度分桶＋动态padding＋token预算batch；有现成实现时使用padding-free/varlen packing |
| 训练量 | 每个专科先完整训练自身数据池一遍，再根据学习曲线决定是否追加 |

已有真实混长测量中，32×4比16×8快约19%，64×2没有更快，因此默认32×4，不以占满显存为目标。多卡时保持全局有效batch128，例如四卡32×1。

### 优先采用的加速手段

1. **长度分桶。** 普通batch优先放相近长度，减少短样本被补齐到长样本的浪费。
2. **动态token batch。** 短样本多放、长样本少放，梯度累计按实际pair数凑到目标batch。
3. **FlashAttention/varlen。** 复用集群已有的高效实现，减少attention中间量和显存读写。
4. **Padding-free packing。** 不同长度样本拼接时保持各自独立，每个pair仍单独读出分类结果。
5. **少做不必要重算。** 梯度checkpointing用于解决显存问题，不是默认开启得越多越快。
6. **复用预处理。** 数据只做一次tokenization，后续训练直接读取缓存，减少CPU与数据加载等待。

先用现成高效路径完成实验，不为追逐新框架反复重搭环境。训练速度看实际有效tokens/s和整轮完成时间。

## 6. 混合模型怎么训练

`MIXED_ALL`使用五个专科实际训练数据的合并池，按来源交错混合。

- 专科各自完整训练一遍，混合模型也完整训练合并池一遍。
- 混合模型在各方向见到的数据和重复次数，与对应专科保持一致。
- 不把五个专家各训一大轮，却只给混合模型相当于一个专家的训练量。
- 六个模型使用相同输入格式、分类目标与基础训练配置。

数据规模可能很大，不再按每轮两小时的小池方式组织正式实验。提速主要靠高效batch、attention、packing和缓存，而不是把专科训练压成很短的小跑。

## 7. 模型融合

五个专科训练结束后，复用它们的权重做以下融合，不再训练一遍专家：

| 方法 | 作用 |
| --- | --- |
| Simple Averaging | 等权平均，作为基础对照 |
| Task Arithmetic | 合并相对共同起点的参数增量 |
| TIES | 减少小增量与符号冲突造成的干扰 |
| DARE＋Task Arithmetic | 对增量随机稀疏化、重标定后做任务向量合并 |
| DARE＋TIES | 对增量做DARE处理后使用TIES |

融合使用LoRA合并后的有效权重，并包含训练过的分类头，最终仍部署为一个Qwen3-0.6B分类模型。

**第一轮是6个训练模型＋5种融合结果，共11个主要评测条件。** Fisher、RegMean可以在有现成实现时作为后续补充，不作为第一轮依赖。

## 8. 最后交付什么结果

所有模型都评同一套五维公共benchmark，输出五维雷达图、各benchmark分数和总分。

| 模型 | Knowledge | Language | Retrieval | Tools | Arts | 总分 |
| --- | --- | --- | --- | --- | --- | --- |
| Knowledge专科 | — | — | — | — | — | — |
| Language专科 | — | — | — | — | — | — |
| Retrieval专科 | — | — | — | — | — | — |
| Tools专科 | — | — | — | — | — | — |
| Arts专科 | — | — | — | — | — | — |
| 全数据混合 | — | — | — | — | — | — |
| 五种融合结果，各一行 | — | — | — | — | — | — |

重点给出三类差值：

- **专科优势：**本方向专科分数－混合模型本方向分数。
- **融合影响：**融合模型分数－混合模型分数，分别看五科和总分。
- **能力保留：**融合模型本方向分数－对应专科分数。

同时给出训练时间、有效tokens/s和单模型推理耗时。有价值的主结果再补训练seed；需要研究融合数量影响时，再复用已有权重比较2、3、5个专科的组合。

## 9. 集群执行顺序

1. 复用现有模型、数据、训练代码和高效实现，下载缺少的数据。
2. 将Tasksource与补充训练来源归入五科，扩充不足的专科数据。
3. 训练五个专科和一个全数据混合模型，每个先完成充分的一轮训练。
4. 复用五专家权重生成五种融合结果。
5. 完成统一五维评测，交付雷达图、分项表、差值和耗时。

**先把五个专科训练充分，再比较特化和融合；不再单列Games，不再以小池短训代替正式专科实验。**
