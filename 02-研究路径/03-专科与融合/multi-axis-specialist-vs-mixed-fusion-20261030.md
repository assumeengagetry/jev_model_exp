# 旧版六方向方案（已替代）

**当前训练计划请使用：[五专科训练、全数据混合与模型融合](five-area-training-plan.md)。** 新版按Knowledge、Language、Retrieval、Tools、Arts五科组织，本文仅保留为旧版。

日期：2026-10-30。本文针对本轮新实验，直接列模型/数据/环境与对照设计；不是继续原Language-only或Teacher优先队列。

**一句话方案：只下载一份Qwen3-0.6B基座；把允许训练的数据按能力来源划为六组，从同一个分类模型初始化训练六个专家和一个全数据混合模型，再对六专家做有效权重平均、Task Arithmetic和TIES，所有模型评同一套冻结测试。** 不需要下载大Teacher，不需要先找回旧Language专用权重。

本轮仅完成方案与资料核查，没有安装环境、下载大模型/数据或启动训练；以下配置是明确建议，不冒充已在新集群验收通过的完整锁文件。

## 1. 六个研究方向可以做，但不是六个独立官方计分维度

已在线读取Space固定revision `c130faf65f82e81d3bdefb01e758c712dc5c5199`的[methodology.json](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/c130faf65f82e81d3bdefb01e758c712dc5c5199/data/methodology.json)：

- `index.areas_count=5`。
- Knowledge、Language、Retrieval、Tools、Arts为五个活跃计分维度。
- Games虽然有类别定义，但`weight=0`、`folded_into=knowledge`、活跃`panel=[]`；ChessBench在官方Knowledge中。

**为保留本轮六专家目标，建议采用以下六个不重叠研究轴：**

| 专家 | 研究方向 | 固定评测分组 | benchmark数 |
|---|---|---|---:|
| `E_K` | Knowledge & Reasoning | GPQA Diamond、GSM8K、MuSR、SATA-Bench、CRUXEval、CLadder、HLE、MMLU-Pro、BBH、WinoGrande | 10 |
| `E_L` | Language Understanding | ContractNLI、ANLI、HellaSwag、ACOS、FinEntity、iSarcasmEval、VAST、NLI4CT、RAGTruth | 9 |
| `E_R` | Retrieval & Classification | BANKING77、CLINC150+OOS、BRIGHT、Amazon ESCI、PhishNChips、HoVer | 6 |
| `E_T` | Tools & Automation | BFCL、ToolRet、API-Bank、Home appliance simulator、When2Call | 5 |
| `E_A` | Arts & Human Taste | BPoMP、Humicroedit、POP909-CL、cfcolor、Habermas Machine、New Yorker | 6 |
| `E_G` | Games & Strategy，自定义研究轴 | ChessBench，从官方Knowledge单列 | 1 |
| 合计 | 六个研究轴 | 当前37个活跃benchmark，各计一次 | **37** |

研究矩阵中的Knowledge不含Chess，避免与Games重复。报告官方公共总分时，仍把Chess合回Knowledge并使用官方五维权重；六轴宏平均是另外的研究指标，不能叫官方总分。Games首轮只覆盖静态棋步选择，不能外推为Hanabi、扑克或所有游戏能力；交互游戏另立协议，不复用别的模型轨迹当本模型成绩。

## 2. 模型需要下载什么

### 必需：一份基座，不是六份不同模型

| 项目 | 固定值 |
|---|---|
| Hub模型 | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) |
| 本次查得revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| 必需文件 | `model.safetensors`、config、tokenizer及相关词表/配置文件；下载整个该模型仓库即可 |
| 实验模型类 | `Qwen3ForSequenceClassification`，不是用CausalLM生成答案 |
| 分类头 | `Linear(1024,2,bias=False)`，0=匹配、1=不匹配 |
| 共享初始化 | 初始化一次分类头和LoRA后保存`M0`；六专家与混合模型都加载这个完全相同的M0 |

官方基座config声明CausalLM是正常的；加载序列分类类会新建score。**不要为七个训练分别随机初始化七个分类头**，也不要以旧Language训练后的R0作为本轮默认起点，否则给Language预先训练优势。

首轮固定causal、BF16、LoRA r32/alpha64；训练并保存score。无需大Teacher、vLLM、DeepSpeed、FSDP或Unsloth才能启动这个0.6B研究。若以后改全参，另立适配容量对照，不与本轮混在一起。

## 3. Tasksource需要下载哪一个、怎么分

### 3.1 主数据就是这一个版本

| 项目 | 内容 |
|---|---|
| 数据集 | [`tasksource/zero-shot-label-nli`](https://huggingface.co/datasets/tasksource/zero-shot-label-nli) |
| revision | `9f0763b798363a6d4ca8f65de2dc80e0fcb48ae2`，本次在线API仍返回该版本 |
| config | **default**，不是triplet |
| 文件 | `data/train-00000-of-00005.parquet`至`00004-of-00005`，validation与test各1个Parquet；保留README |
| 原始规模 | train **1,899,993 pair**、validation24,971、test25,440；294个来源任务 |
| 主要字段 | premise、hypothesis、labels、task、license、license_use |
| 实际原始标签 | 0/2，匹配/不匹配；核对实载后映射`{0:0,2:1}` |

“180万”对应历史清洗后的1,813,810条训练pair，不是这个公开原始train的精确行数。新清洗与新分区后规模应重新统计，不能要求恰好等于旧数。这些是展开后的文本对，不是同样数量的原生多选题。

不需要额外再下Tasksource typed/full、filtered-full、procedural等所有大池才开始；它们是不同数据产品，不等于本轮这份约190万label-NLI。

### 3.2 不能随机六等分，也没有现成六分区

按`task`及上游任务语义建立一份**人工可审计的来源→方向表**，再按原题/文档分组分配：

| 方向 | Tasksource任务判定原则 |
|---|---|
| K | 科学/学科知识、数学、常识推理、代码行为等 |
| L | 蕴含/矛盾、情感、方面情感、立场、讽刺、医学/法律文本理解、grounding |
| R | 检索相关性、意图分类、大标签集分类、垃圾/钓鱼检测、证据检索/核验；边界任务按冻结规则归属 |
| T | 工具/API选择、路由、过程/政策执行、结构化状态操作 |
| A | 幽默、诗歌、音乐、审美/偏好等，按具体目标区分一般情感分类 |
| G | 棋步、棋盘状态、游戏策略/规划；没有对应来源就如实记零，不把逻辑题硬凑为游戏数据 |

同一来源可能有不同子任务；允许在子任务层确定唯一归属。来源不明或无法明确归类的记录进入`unassigned`，先审清，不默认复制到六组。**六组规模不会相等，Games/Arts甚至可能很小；没有完成task统计前不编造每组条数。**

首轮主研究使用互斥分区，避免一个原题跨专家重复后让混合控制的曝光失配。同题多视图、候选和近重复保留group映射；有歧义的跨能力任务登记规则，未来做敏感性分析，而不是宣称能力分类天然唯一。

## 4. 各方向还要补哪些benchmark训练数据

下面下载清单只允许**正式train或明确独立的替代源**。下载整个benchmark仓库不代表全部文件都能入训；所有来源都需与冻结37项测试按ID、正文和源组去污染。公开train与test之间也可能本来就有重复。

### 4.1 Knowledge：先下四个明确train，再补待验收项

| 下载ID | config / 只取的split | 原始train数 | 与活跃benchmark的关系 |
|---|---|---:|---|
| [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | `main / train` | 7,473 | 同benchmark原生train；自由回答须仅从train构造决策候选，不输入答案推导 |
| [`allenai/winogrande`](https://huggingface.co/datasets/allenai/winogrande) | `winogrande_xl / train` | 40,398 | 同benchmark原生train；其他大小config是嵌套池，不重复累计；validation留作评测 |
| [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | `ARC-Easy / train`及`ARC-Challenge / train` | 2,251＋1,119 | 科学知识独立补充；ARC不在当前37项活跃计分中；仍审跨库重复 |
| [`tau/commonsense_qa`](https://huggingface.co/datasets/tau/commonsense_qa) | `default / train` | 9,741 | 常识选择独立补充，不是所有知识benchmark的原生train |
| [`allenai/openbookqa`](https://huggingface.co/datasets/allenai/openbookqa) | `main / train` | 4,957 | **条件项**：train明确，但HF数据许可unknown，补查作者数据条款后再纳入；additional不能重复算独立题 |

**以下只作评测/去污染参照，不进训练白名单：**GPQA、MMLU-Pro、BBH、CRUXEval、MuSR、SATA-Bench、CLadder、HLE。MMLU-Pro只有validation70/test12,032；GPQA某些loader虽把题库叫`train`，也不是干净训练集；BBH/CRUXEval及其CoT/候选随题一起隔离。不能为了“每个benchmark补train”把原本没有train的题库改名入训。

### 4.2 Language：已有原始文件可复制到新集群，不必重下全部

当前本地[[research/language-axis/](../02-Language/language-axis)](../02-Language/language-axis)保留了已校验的原始文件；精确URL、revision、SHA256和选择器见[manifest](../../03-证据与图表/02-Language/language-axis/manifest.json)。这是原件，不是已转换训练缓存。

| 下载ID/官方源 | 只用的训练划分/选择器 | 原始训练单位/数量 |
|---|---|---:|
| [`facebook/anli`](https://huggingface.co/datasets/facebook/anli) | train_r1/r2/r3，保留三类语义后转候选匹配 | 162,865对原生premise/hypothesis |
| [`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag) | train；validation不得入训 | 39,905题 |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip) | ZIP中train.json，完整合同而非gold证据片段 | 423合同×17命题＝7,191判断 |
| [ACOS](https://github.com/NUSTM/ACOS) | laptop/restaurant的train TSV | 2,934＋1,530评论 |
| [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval) | 英文train；阿文train可独立登记加入 | 3,468英文tweet；可选3,102阿文 |
| [VAST](https://github.com/emilyallaway/zero-shot-stance) | vast_train.csv | 13,477条文本＋话题 |
| [NLI4CT](https://github.com/ai-systems/Task-2-SemEval-2024) | training_data.zip中train标签＋所需trial正文 | 1,700命题 |
| [RAGTruth](https://github.com/ParticleMedia/RAGTruth) | response中split=train，用source_id关联source_info | 15,090回答 |
| [SEntFiN](https://github.com/pyRis/SEntFiN) | 按Title分组自留dev，审重后展开实体 | 10,753标题/14,404实体标签候选池 |

FinEntity的979份未拆分文档是评测源，没有可直接切出的官方train；SEntFiN只是独立训练代理。ANLI不能把原三类ID直接送进二分类头；RAGTruth不得泄漏gold幻觉span，受限原始上下文不再分发。数据许可包括非商用等来源限制，保留每条上游条款，不用代码仓库license替代原始内容许可。

### 4.3 Retrieval：意图、产品相关性先接通

| 下载ID/官方源 | config / 训练选择器 | 原始规模与首轮范围 |
|---|---|---|
| [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) | `default / train` | 10,003；必须清除作者train/test重复 |
| [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) | `plus / train` | 15,250，150类＋OOS；不要与作者Full的15,100重复叠加 |
| [Amazon ESCI](https://github.com/amazon-science/esci-data/tree/7916cdf6ab75a462e77f20ab40428a10923998d5/shopping_queries_dataset) | examples内`large_version==1 AND split=="train"`，按locale＋product_id关联products | 全语言1,983,272，us为1,393,063个原始query-product pair；可在新实验manifest中预先固定query组子集，不能未经说明全量淹没其他来源 |
| [HoVer原始train](https://raw.githubusercontent.com/hover-nlp/hover/39b84697f196308f398a251a7aea9b82ae0f0562/data/hover/hover_train_release_v1.1.json) | train18,171；配[作者wiki库](https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db) | **条件项**：证据库未就绪则延后；claim-only不是榜单oracle-document任务的等价输入 |

ESCI需两个文件：[examples，约51.3MB](https://media.githubusercontent.com/media/amazon-science/esci-data/7916cdf6ab75a462e77f20ab40428a10923998d5/shopping_queries_dataset/shopping_queries_dataset_examples.parquet)＋[products，约1.11GB](https://media.githubusercontent.com/media/amazon-science/esci-data/7916cdf6ab75a462e77f20ab40428a10923998d5/shopping_queries_dataset/shopping_queries_dataset_products.parquet)。表内的test不能随机重切回train。

BRIGHT没有本轮可直接用的干净train；其documents是检索语料，不是允许把评测query/gold入训的证据。PhishNChips核心2,000邮件是榜单评测，auxiliary/validation也不是已核实train，首轮不混入。

### 4.4 Tools：When2Call先做，其他作明确补充

| 下载ID | config / 文件 | 训练用途与限制 |
|---|---|---|
| [`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call) | **`train_sft / train`**，15,000 | 首选原生训练数据；从问题、tools和目标行为转决策匹配，不是直接沿用聊天token SFT |
| 同上，可选 | `train_pref / train`，9,000 | chosen/rejected可作偏好候选；与SFT有关联，先去重，不当新增9,000独立任务 |
| [`liminghao1630/API-Bank`](https://huggingface.co/datasets/liminghao1630/API-Bank) | `training-data/lv1-train.json`、lv2、lv3 | 作者训练对话补充，约1,888对话的发布规模；各level不同视图选一种，不累加combined/api-only等副本 |
| [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | `default / train`，112,960 | 独立工具调用补充，不是BFCL原生train；剥离当前/未来调用答案，只用决策前可见上下文 |
| [`mangopy/ToolRet-Training-20w`](https://huggingface.co/datasets/mangopy/ToolRet-Training-20w) | `ToolRet-Training-20w / train`，208,826 | **条件项**：确有作者train，但上游许可和跨来源去重未完成，先不默认入训 |

When2Call的评测是**`config="test", split="mcq"`**，不是训练。BFCL评测题库、ToolRet-Queries、ToolRet-before-sample不可入训。Home appliance只有dev/test构造，不把公开dev当train；以后生成新场景也需状态去重和源组隔离。xLAM60k有门控，且与When2Call/ToolRet有来源关系，不作为无需授权的额外必下项。

### 4.5 Arts：先用Humicroedit和New Yorker，别为了覆盖硬切评测

| 下载ID/官方源 | 训练选择器 | 首轮决定 |
|---|---|---|
| [Humicroedit作者ZIP](https://zenodo.org/records/3969509/files/semeval-2020-task-7-dataset.zip?download=1)，或[`SemEvalWorkshop/humicroedit`](https://huggingface.co/datasets/SemEvalWorkshop/humicroedit) | `subtask-2 / train`，原始9,381 | 纳入候选；按原headline分组，对齐二选一时排除label0平局；评分/meanGrade不进入输入。HF卡license unknown，但[作者Zenodo metadata](https://zenodo.org/api/records/3969509)明确CC-BY-4.0，保留作者来源证据 |
| [`jmhessel/newyorker_caption_contest`](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest) | **`matching / train`，fold0**，9,792 | 纳入候选；用场景文字描述、entities与caption候选，不用图片。按contest_number审重；不混matching_1…4的train，否则可能覆盖fold0测试 |
| [Habermas Machine](https://github.com/google-deepmind/habermas_machine/tree/7923b71966c14136077c78d7841bc9e1a182dfe0) | `metadata.version=TRAINING_DATA_V1…V5`且`question.split=TRAIN` | 条件扩展；排除EVAL/IID_TEST/OOD_TEST，群组/问题/participant审重，群体标签按完整排名聚合 |
| [cfcolor](https://www.dgp.toronto.edu/~donovan/cfcolor/) | 原包确有train_vec/test_vec | 暂缓；train ratings还会成为评测用户历史，需预先定义用户/新palette泛化与冻结目标隔离，不说“完全没有train” |
| [POP909-CL](https://github.com/AndyWeasley2004/POP909-CL-Dataset/tree/be9094392903c471a930519e1c0bacf8b6be5d62) | 没有与榜单天然隔离的现成train清单 | 暂缓；先排除冻结测试全部song_id及同曲窗口/转调，剩余按曲划分；隐藏gold chord轨道；作者90/10划分不自动等于安全train |
| [BPoMP](https://zenodo.org/records/7299879) | 发布为测试框架，未核实原生train | 暂缓；若以后自划，按原诗及全部扰动家族隔离，不只删单个pair |

New Yorker的HF Parquet含图片列，模型不用图片不等于下载包无图片，train约548MB。可用作者提供的[纯文本tasks.zip](https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/tasks.zip)＋[annotations_per_split.zip](https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/annotations_per_split.zip)重建split0，但转换后与HF逐行一致性仍要验收，不把这条路线冒称本次已转换。

### 4.6 Games：Chess有官方train，但要单独防棋盘泄漏

官方源：[google-deepmind/searchless_chess](https://github.com/google-deepmind/searchless_chess/tree/90ae0e6b121673fc3079aaeffa047580bb600c0a)。

| 可下载资产 | 大小/格式 | 如何用 |
|---|---|---|
| [train behavioral_cloning_data.bag](https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag) | 本次HEAD确认36,316,203,219字节，约36.3GB/33.8GiB；`(fen,oracle_move)` | **首轮优先候选**：按官方Bagz/coder解码，确定性抽取去污染训练局面，配合法着候选；这是棋步模仿监督 |
| train action-value分片 | 2,148片；[首片](https://storage.googleapis.com/searchless_chess/data/train/action_value-00000-of-02148_data.bag)约1.34GB；`(fen,move,win_prob)` | 值监督候选；**单片未必含某棋盘全部合法着，不能在残缺动作中取argmax冒充最佳合法着** |
| [test action_value_data.bag](https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag) | 仅评测/构造排除表 | 不能入训；提取官方test全局面及冻结ChessBench题组进行排除 |

**不要直接运行下载脚本全量抓所有train。** 官方README称数据约1.1TB，而脚本还列出全部2,148片；这对本轮0.6B不是启动必需。首轮可下载单个BC文件，再固定一份有预算的训练子集，明确它是“所选全部训练池”，不是全部Chess语料。

最重要的风险：[原论文](https://arxiv.org/html/2402.04494v2)明确说明，最大train中出现了14.7%的IID test棋盘。**即使文件路径是train也必须排除测试棋盘**，按规范化棋盘键审计，而非只比含计步字段的完整FEN。这里只核对文档与对象大小，没有实际解码或完成棋盘排除。

如不愿承担约36GB下载，可另立程序生成的小型完全可解游戏训练源；训练/测试seed不同仍要状态去重。它不是ChessBench原生train，其对ChessBenchmark的迁移能力需要实测，不能用它悄悄替代本轮Games数据定义。

### 4.7 额外HF来源的冻结revision

除Language沿用本地manifest、原始文件型来源记录内容hash外，下列revision已由本轮只读API核对；加载时仍只取上述config/split：

```text
openai/gsm8k                         740312add88f781978c0658806c59bc2815b9866
allenai/winogrande                  01e74176c63542e6b0bcb004dcdea22d94fb67b5
allenai/ai2_arc                     210d026faf9955653af8916fad021475a3f00453
tau/commonsense_qa                  94630fe30dad47192a8546eb75f094926d47e155
allenai/openbookqa                  388097ea7776314e93a529163e0fea805b8a6454
PolyAI/banking77                    1846984189f2bc0ac51c6fc3a600fa9e1714e0c6
clinc/clinc_oos                     155b9c710419136e17307b80d0a13e68cd46b4ec
nvidia/When2Call                    0582f7749df63a96fdc3070932e83e72396ace53
liminghao1630/API-Bank               12e8158b7628c168f07e8f31fbbe3445e99f44cf
glaiveai/glaive-function-calling-v2  e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac
SemEvalWorkshop/humicroedit         2ac5775fe2cb97e6ff096aa5274b0a19a1ea3872
jmhessel/newyorker_caption_contest  d81cbab7d0392708d5371d3a4960e69261824db4
```

这些清单确认的是来源、划分和已知风险，不是已完成的数据转换/许可总审计/去污染。**本轮不为凑齐每项benchmark而强行下载全部源**：主Tasksource＋各方向优先项先形成闭环，条件项只有验收后才加入；冻结后混合模型使用六专家实际允许池的并集。

## 5. 环境配成什么样

### 5.1 核心环境：沿用已有成功记录，去掉不必要的加速框架

| 组件 | 建议固定版本/方式 |
|---|---|
| OS/架构 | Linux x86_64；以下已核对torch wheel为manylinux_2_28，系统glibc需满足要求 |
| Python | 3.11；可用时固定历史3.11.17 |
| PyTorch | **2.11.0+cu130** |
| transformers | **5.5.0** |
| peft | **0.18.1** |
| datasets | **4.3.0** |
| accelerate | **1.15.0** |
| huggingface_hub | **1.33.0** |
| pyarrow | **23.0.1** |
| 其他 | safetensors、numpy、scipy、scikit-learn、pandas、pyyaml、tqdm；兼容安装后导出完整freeze，不冒称已锁定所有传递依赖 |
| 训练执行路径 | 原生Transformers＋PEFT＋SDPA；causal、`use_cache=False` |
| Unsloth/flash-attn/vLLM | 首轮不要求安装，避免把kernel/框架差异混进数据比较 |

本次已在[官方cu130索引](https://download.pytorch.org/whl/cu130/torch/)确认torch2.11.0的Python3.11/Linux x86_64 wheel存在，但没有下载或安装它。按[NVIDIA官方兼容表](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)，CUDA13.x常规要求驱动分支≥580，具体功能仍须实测；若新集群驱动不满足，先选匹配驱动的官方wheel并另记环境，不自动升级宿主机驱动。

以下是**以后在新集群运行**的环境命令，不是本次已执行内容；假设`python3.11`已可用且驱动满足cu130：

```bash
export ROOT="$HOME/assumeengage"
python3.11 -m venv "$ROOT/envs/axis-train"
source "$ROOT/envs/axis-train/bin/activate"

python -m pip install --upgrade pip
python -m pip install 'torch==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install \
  'transformers==5.5.0' 'peft==0.18.1' 'datasets==4.3.0' \
  'accelerate==1.15.0' 'huggingface_hub==1.33.0' 'pyarrow==23.0.1' \
  safetensors numpy scipy scikit-learn pandas pyyaml tqdm
python -m pip check
python -m pip freeze > "$ROOT/envs/axis-train/freeze.txt"
```

可使用经过验证的可达pip/HF镜像，但不承诺旧镜像域名现在可达，也不把私有凭据写入命令或配置。不要把依赖装进base；不要为了数据列读取额外装torchvision。

### 5.2 融合环境独立

MergeKit本次查得commit `6abfb9955bf12e459b32745781cf11b9df949dda`，项目版本0.1.5；要求Python≥3.10、Transformers5.x、PEFT≥0.18等。建议单独环境，不让融合安装改动已经锁定的训练环境：

```bash
python3.11 -m venv "$ROOT/envs/axis-merge"
source "$ROOT/envs/axis-merge/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'torch==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install \
  'git+https://github.com/arcee-ai/mergekit.git@6abfb9955bf12e459b32745781cf11b9df949dda'
python -m pip check
python -m pip freeze > "$ROOT/envs/axis-merge/freeze.txt"
```

融合可在CPU上执行，保存后的模型回训练/评测环境重载评分。这里没有声称两个新环境已安装成功；完整验收还包括分类头更新/保存、合并重载和真实GPU更新。

### 5.3 评测工程

- 官方kit：[apolinario/decision-index](https://github.com/apolinario/decision-index)，本次在线main查得commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`。
- 固定同一commit、suite manifest、原始请求和scorer；全37项不再只是Language31,715请求。
- metadata中存在历史总行数字段，不硬编码某一旧总数作为本轮恢复完成条件；以实际冻结suite的manifest、排除表和评分器一致性为准。
- 本地仓库缺完整kit、自定义分类Engine和请求本体；即使装完上面pip包也还不能直接训练/评分。本轮仍需构建可运行的数据转换器、训练入口、分类Engine和统一调度/归档。
- 不编造`--area six`之类CLI参数；输入必须按真实kit接口构建并保留关联请求组。分类Engine需支持全部候选，不能拿CausalLM内置Engine直接替换分类权重。

## 6. 下载命令：先拿这两份主资产

环境准备后执行；模型和Tasksource都固定本次核对的commit：

```bash
source "$ROOT/envs/axis-train/bin/activate"

hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$ROOT/models/Qwen3-0.6B"

hf download tasksource/zero-shot-label-nli --repo-type dataset \
  --revision 9f0763b798363a6d4ca8f65de2dc80e0fcb48ae2 \
  --include 'data/*' --include 'README.md' \
  --local-dir "$ROOT/datasets/tasksource/zero-shot-label-nli"
```

不下`triplet/*`。各补充来源按第4节的config/split读取，下载时记录实际repo revision；不能把README里提到的全库全部转成训练行。训练/验证/测试缓存分别保存，补充数据先留独立dev，再候选展开。

## 7. 怎样训练，才能真的回答你的问题

### 7.1 最小模型集合：七次训练＋融合，不是六次训练

设`T_i`为Tasksource第i方向允许的训练数据，`B_i`为该方向审计通过的补充train，`D_i=dedup(T_i∪B_i)`。

| 模型 | 训练数据 | 是否需要训练 |
|---|---|---|
| `E_K/E_L/E_R/E_T/E_A/E_G` | 各自`D_i`，不看其他方向train | **6次** |
| `M_ALL` | 六方向允许训练数据的去重并集，交错混合 | **1次** |
| `MERGE_AVG` | 六专家有效权重等权平均 | 不新增训练 |
| `MERGE_TA` | 相对共同M0的Task Arithmetic | 不新增训练 |
| `MERGE_TIES` | 六专家相对共同M0的TIES | 不新增训练 |
| `MERGE_DARE_TA` | 论文式DARE预处理后做相同Task Arithmetic | 不新增训练；需要记录随机mask |
| `MERGE_DARE_TIES` | 论文式DARE预处理后做相同TIES | 不新增训练；需要记录随机mask |

M0仅为共同初始化，不作为已经有分类能力的基线。所有模型都用同一骨干、二分类head、标签解释、tokenizer、输入格式与attention；不能每方向换一个语义不兼容的分类头。

结合用户补充的文章，主方案由3种扩为**5种融合方法**：首个训练seed下是7个训练模型＋5个融合条件＝12个基础评测条件，不是12次训练。Fisher、RegMean列为第二阶段条件方法，不是启动依赖。DARE随机mask复验、2/4/6专家子集和后续训练seed会增加评测终点，不能仍声称总共只有12份产物；这些均为未来计划，不据此占用当前服务器资源。

### 7.2 主比较要匹配“各方向实际见过的数据”

**建议首轮按允许训练池各一遍构建冻结训练流：**专家i处理自己的全部训练流，混合模型处理六专家训练流的合并交错版本。原题/候选、重复次数、训练截断后的有效token和尾批提交均作回执。

令专家i实际提交的非padding token为`t_i`，则主混合控制应满足：

```text
M_ALL总token = t_K + t_L + t_R + t_T + t_A + t_G
M_ALL在方向i的源样本multiset与token曝光 = E_i在方向i的曝光
```

这样融合所使用的六模型**总训练数据曝光**与一个混合模型相同；不能六专家各50M、混合只50M后称计算公平。也不要求六方向`D_i`大小相等或给每个方向机械定相同epoch步数。

这比较的是独立优化后融合与混合优化的训练策略；六套optimizer和一套optimizer、warmup/更新顺序、GPU时间不同，不宣称优化轨迹完全相等。另报告训练FLOPs代理/实际GPU-hours、更新数和部署成本。

若还要问“每个单模型相同总token预算时谁更好”，可另加`M_ALL_SHORT`，匹配某一共同预算；它是第二种比较，不能替代上面的总曝光控制。领域很不均衡时再做一个域均衡混合对照，不在首轮混入采样网格。

### 7.3 首轮统一训练候选

| 配置 | 建议起点，待小pilot后统一冻结 |
|---|---|
| 精度/attention | BF16、causal SDPA、`use_cache=False` |
| LoRA | r32、alpha64、dropout0；相同投影模块列表 |
| score | 共同初始化；训练并保存，显式核对`modules_to_save=["score"]`或等价SEQ_CLS处理 |
| 优化器 | AdamW，lr1e-4、weight_decay0.01；与历史从基座起训设置相近，不称新最佳LR |
| batch | **历史实测优先起点：单卡microbatch32×累计4＝有效128 pair**；新方向最长batch不通过才退到16×8，不默认增到64/128 |
| 训练长度 | cap4096，保留候选描述；记录分方向截断率，不认为4096能完整覆盖所有方向 |
| checkpointing | 保守基线开启；最快候选须在新后端/真实长尾上证明可关闭且留显存余量，关闭不通过则继续开；真实padding mask不改全1 |
| 首轮seed | 共同seed3407，所有模型同一个M0；有结论后再做匹配seed复验 |
| packing | 第一优先长度分桶＋动态padding；再比较token-budget batching。FA变长/padding-free packing通过分类readout、序列隔离及loss验收后启用，不裸拼序列 |
| 调度 | 同类LR计划与归一化进度；记录不同run总长度导致的实际LR/optimizer差异 |

Tasksource pair直接沿用匹配目标；新多分类/choice来源机械转候选匹配，gold只进入loss，不进入state。保留全部正确选项/多标签语义，不把未标注候选随意当已证实负例。训练pair数、原题数与token分别记录；不能因某任务候选151个就无意给它151倍原题权重。

本轮明确使用跨方向Tasksource和补充train，不再默认套用先前Language-only、50万pair合计及整轮两小时快筛范围；这是**新实验的范围重定义**，不是已经批准任意算力消耗。实际数据规模与训练预算须在训练前统计和冻结，不先承诺一轮几小时完成。

### 7.4 “最佳配置”的证据：先以32×4起步，不把显存用满当目标

这里的“优先”指**当前历史证据支持的默认起点**，不是六方向新数据、新rank或新attention后端的全局最优证明。所有建议仅供未来新run，不改正在跑的任务。

来自[真实batch测量](../../01-研究结论/05-Attention与效率/batch-benchmark-20261028/results.md)及[mb32机器记录](../../03-证据与图表/05-Attention与效率/batch-benchmark-20261028/joint-real-mb32.json)：每个配置独立进程，固定相同数据窗口与有效batch128，2次预热更新＋6次计时更新，确有optimizer step。

| 单卡microbatch | grad accumulation | 有效batch，pair | 有效tokens/s | 峰值allocated/reserved GiB | padding |
|---:|---:|---:|---:|---:|---:|
| 16 | 8 | 128 | 8,728 | 11.98 / 14.67 | 20.1% |
| **32** | **4** | **128** | **10,347** | **22.59 / 24.43** | 37.2% |
| 64 | 2 | 128 | 9,036 | 43.89 / 47.54 | 57.8% |
| 128 | 1 | 128 | OOM | — | — |

32在该真实窗口比16快约18.6%，比64快约14.5%。全4096无padding压力测试中，microbatch128虽然能通过，但相对16吞吐只高约1.4%、显存大幅上升；不能据此把真实混长训练改成128。mb32的实际累计吞吐也不能用更好看的中位更新时间替代。

该batch归档记录了SDPA/BF16、原R0加载路径，但未独立展开loader内的全部checkpointing配置；**不从显存数值反推其GC状态**。第7.6节另用显式On/Off测量决定重算开关。新方向和r32须重验真实长尾，旧r16资源表不能当成新配置容量保证。

[8个batch/LR短试评审](../../01-研究结论/02-Language/lu-batch-lr-pilot-20261029/STUDY-REVIEW.md)另显示：r16、两续训seed、各3M预算下，batch128/lr1e-4的原生dev宏均值0.42561±0.01232；batch256并没有明显吞吐优势，质量更低。但这不是完整Language分数，也没有验证所有方向/r32，所以只支持把128与1e-4当候选，不按线性缩放规则盲目扩大batch。

### 7.5 梯度累计、物理batch与多卡的关系

固定大小microbatch时：

```text
global_effective_batch_pairs = microbatch_pairs * grad_accum * DDP_world_size

1 GPU : 32 * 4 * 1 = 128
2 GPUs: 32 * 2 * 2 = 128
4 GPUs: 32 * 1 * 4 = 128
```

六方向0.6B专家可在资源允许时独立单卡运行；不因为有四卡就必须把每个模型改DDP，也不假定当前有空闲卡。混合模型若改多卡，仍保持目标全局batch并核对同步、loss归一化和实际提交量。

**梯度累计主要是让较小物理batch实现较大有效batch，不是天然加速器。** 16×8比32×4有更多前后向调用/调度开销；反过来，盲目扩大物理batch可能使padding、长尾和显存开销更差。它也不消除模型和optimizer状态内存。

对于动态batch/packing，不能再把“四次backward”直接等同128个pair：必须累积到目标实际pair数/预先定义的更新单位。若某次只有16或8个pair，继续累积而不是提前step。pack数量、token数、pair数和原题数分别记录。

### 7.6 Gradient checkpointing：省显存，以重算换时间

梯度检查点/激活重算不是“多存一次checkpoint”，也不是免费加速。来自[长度与重算研究](../../01-研究结论/05-Attention与效率/checkpoint-length-scaling-20261029/RESEARCH-REPORT.md)，r16/BF16、causal SDPA、物理microbatch16、累计2次：

| 单条长度 | GC On allocated | GC Off allocated | On有效tokens/s | Off有效tokens/s |
|---:|---:|---:|---:|---:|
| 512 | 2.46 GiB | 16.91 GiB | 29,400 | 39,375 |
| 1024 | 3.66 GiB | 32.56 GiB | 29,798 | 40,922 |
| 2048 | 6.07 GiB | 63.85 GiB | 29,251 | 39,960 |
| 4096 | 10.90 GiB | OOM | 27,518 | — |
| 8192 | 20.55 GiB | OOM | 24,227 | — |

512–2048时，GC On耗时比Off多约34%–37%，不是On更快；真实2527–2807混长、microbatch16关闭时也OOM。显存不足时On能让工作完成，或允许更合理物理batch，因此仍可能改善整个可行方案，但不能把单次重算当作加速。

**注意表中是前向＋反向探针，optimizer更新为0。** 它可说明显存/重算代价，不能直接当完整训练token/s或r32性能。新FA/varlen后端可能使Off可行，但本轮没有实测，不能写“换Flash后肯定能关”。

建议：先有GC On的正确可运行参照；在独立新环境、同一真实最长/高padding trace上比较Off。只有无OOM、完整optimizer-step更快且reserved＋CUDA进程有余量，才把Off提升为速度配置；否则保留On。不要在一个正在训练的run中按临时显存状况切换，也不通过全1 mask伪造省显存。

### 7.7 以前有没有用FlashAttention？能省显存，也能加速，但不是注意力降阶

历史证据能回答的是：

- 后期资源实验实际为Transformers SDPA，profiler记录`aten::_scaled_dot_product_cudnn_attention`及其backward，是cuDNN fused attention，不是单纯Python/eager注意力。
- SDPA是调度入口，可能选择Flash、cuDNN、memory-efficient或math后端；`flash_sdp_enabled=true`只说明允许使用，不证明实际调用了Flash kernel。
- 没有足够安装/profiler证据认定此前使用独立`flash-attn`包；也不能把后期SDPA路径直接推成最初Unsloth R0所有阶段的实际kernel。

FlashAttention通过tiling、在线softmax与减少HBM中间矩阵读写，通常兼顾显存与速度，是精确attention的高效实现；**完整dense attention的理论主阶仍为O(L²d)，并非变成线性注意力**。实现可减少部分非矩阵计算和冗余工作，但不能把主要收益简单归结为计算复杂度降低。真正少算padding对应的位置还需看unpadding/varlen；某些cuDNN实现已经很快，所以FA2/FA3相对它的加速必须实测，不能照搬“比朴素attention快几倍”。

H800属于Hopper，值得比较FA2以及FA3的BF16训练路径；FA3不是“仅推理加速”。[官方README](https://github.com/Dao-AILab/flash-attention)给出H100/H800、CUDA≥12.3及FP16/BF16前反向支持，但这**不证明当前torch2.11/cu130的wheel、ABI和模型接入已验收**。不在活动环境安装/升级，也不为安装它擅自切换CUDA栈。先独立验证可复用后端，记录精确版本；不为追最新版顺带改attention语义或LR。

### 7.8 你提到的动态打包：分清三个层次

| 技术 | 怎么处理长度 | 减少什么 | 作为本项目第一步的风险 |
|---|---|---|---|
| 长度分桶＋动态padding | 相近长度放同一microbatch，只pad到本批最长，必要时对齐8 | 无效token和attention padding，减少形状波动 | 最低，原分类head可沿用 |
| Token-budget dynamic batching | 短样本多放、长样本少放，限制每批padded/有效token等预算 | 极端长批次的padding、显存峰值与拖尾 | 中等，需正确累积样本和loss |
| 真正packing / padding-free varlen | 不同长度序列展平拼接，用边界隔离，不计算其间的padding | attention padding；完整flatten路径还可减少MLP等逐token padding | 需适配边界与每pair分类readout，不能直接套聊天SFT |

**普通padding batch里尽量把相近长度放一起；只有正确的padding-free路径才适合把不同长度直接拼成一条。** 例如64、80、1024-token三个样本，pad到1024是3072 token，真实只有1168；真正varlen可以避开相应padding工作，但不能据token比预报端到端加速倍数。

优先实施的低风险方案是：**先固定一次optimizer更新应见到的128个pair及其源组，再只在这个更新窗口内按长度重排和组microbatch。** 随机化更新窗口/桶顺序，不全局从短到长制造课程，不丢弃长样本、不改变任务/标签配额。这样比跨整个数据集任意重排更利于保留既定曝光和比较语义；数值仍可能因BF16累加顺序变化，不能要求逐位相同。

动态padding下可把**16k/32k padded-token预算、每批最多32 pair**作为新的短测候选；它们尚未验证。预算按`B * round_up(max_length,8)`核算，同时约束最大长度/必要的attention成本。固定token预算不保证所有mask/backend的显存相同，因为临时张量、GQA路径和长度平方项仍会变化。最长样本不适配时拆小microbatch，不截断评测或丢样本。

变长microbatch的更新loss必须对应实际训练单位：本实验仍是`sum(CE_per_pair) / actual_pairs_in_update`。**不能先把各microbatch loss取mean，再对大小不同的microbatch等权平均**；否则长短样本或来源被意外重加权。DDP还要考虑rank默认梯度平均，用全局样本数正确缩放；确认训练器不会再重复除grad_accum。

### 7.9 真正varlen packing：序列分类比聊天SFT多一道验收

本模型是28层普通full self-attention；训练时同一序列的Q/K/V token长度对应一致，GQA的Q头与KV头数可以不同。`seqlen_q/k`和`max_seqlen_q/k`应来自真实序列，`cu_seqlens`编码每个样本的累积边界，不是能任意调小以“少算”的旋钮。causal=True只隔离当前序列的未来token，不自动保证拼接样本之间隔离。

本次只读核对了Transformers **v5.5.0**原件：

- [`DataCollatorWithFlattening`](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/data/data_collator.py)确有`return_flash_attn_kwargs`，可产生`cu_seq_lens_q/k`、`max_length_q/k`和position_ids；但其默认labels处理面向**token序列标签**，不是本项目每pair一个整数的分类标签，不能直接替换collator。
- [`modeling_flash_attention_utils.py`](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/modeling_flash_attention_utils.py)已有varlen路径；Transformers参数名与原生[flash_attn_varlen_func](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)的`cu_seqlens_q/k`、`max_seqlen_q/k`命名不同，要按实际接口传递，不混抄最新版示例。
- [`GenericForSequenceClassification`](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/modeling_layers.py)按物理batch每行最后一个非pad位置pooling。把N个pair展平为`[1,total_tokens]`后，原封不动的head路径只给一个序列级输出，不会自动返回N个分类结果。

因此正确packing至少需要：

1. 每pair的边界、有效长度、原标签和稳定ID；独立序列的position_ids按要求重置。
2. FA varlen的真实`cu_seqlens`或正确block-diagonal隔离；不以普通全1的2D mask＋causal代替，不仅靠EOS或重置position_ids猜隔离成功。
3. 在每个pair原来规定的readout位置收集hidden state/score，输出`[N_pairs,2]`，不能改成只取整个pack最后token。
4. 同权重、同完整更新样本下，对照unpadded/普通batch验证logits、分类CE、梯度和一次optimizer更新至预先记录的数值容差。
5. 改变同pack其他样本不能影响当前pair输出；覆盖有无padding、不同长度、序列顺序、保存重载与推理。正式评测的候选顺序和任务语义保持不变。

低风险中间步骤是先保留`[B,L]`分类输入/head，只切经过验收的FA后端，让attention内部unpad；它仍可能在MLP等步骤处理padding，不能冒称已经实现全模型padding-free。再决定完整flatten＋自定义多readout是否值得工程成本。

### 7.10 速度优化优先级与“尽快交付结论”的判据

| 顺序 | 独立新run里的候选 | 晋级条件 |
|---:|---|---|
| A0 | BF16、SDPA、32×4/有效128、GC On、原始真实mask | 正确、能完成更新的参照；保留已有同等产物，不重复训练 |
| A1 | 更新窗口内长度分桶＋动态padding，其他不变 | 实际非padding token/s提高，样本/标签曝光和更新单位不变 |
| A2 | 对照16k/32k token-budget microbatch，自适应累积到128 pair | 完整更新/尾批的loss归一化正确，真实长尾无OOM |
| A3 | 已验收FA2后端；资源允许再FA3；随后才做完整varlen packing | 实际kernel/边界/readout正确，训练更新和最终质量验收通过 |
| A4 | 在选定后端比较GC Off与On；必要时再试更大短序列microbatch | 包括optimizer、最长/高padding batch均稳定，完整更新时间确实更好 |

顺序不表示已有加速结果，不自动执行全部网格。先每次只改变一个因素；padding高优先A1/A2，kernel或显存成为瓶颈再投入A3/A4。不要一开始同时叠加FA3、packing、torch.compile、CUDA graphs和新优化器。编译/图捕获可能因变长形状反复开销，只有包含准备时间后值得才使用。

数据端先离线tokenize一次、复用Arrow/cache；未来新worker可将pin_memory、persistent_workers和少量DataLoader workers作为候选，按实际CPU/I/O预算选择，不把旧脚本`set_num_threads(8)`误当服务器总核数。不创建额外并发worker挤占当前实验，也不修改其缓存。

若未来明确有独立资源，可在同一真实trace上预热后测少量**完整optimizer更新**，包含collation/H2D/forward/backward/step/zero_grad；覆盖自然长度和长尾，不用全是短样本的平均值。验收用副本或独立clone，探针更新不能写回正式parent；预热/丢弃的额外token计入工程成本。记录：

- 实际非padding tokens/s、pairs/s、每完整update耗时与p95，而非padded-token吞吐。
- padding率、每批最大长度/真实shape、allocated/reserved和CUDA进程开销。
- torch/transformers/flash-attn版本、实际attention算子、GC实际启用状态、有效pair数和梯度累计步数；开关/包已安装不等于实际生效。
- `准备/编译/加载＋训练＋dev＋正式评分＋融合/审计`的端到端时间；最终目标是最快得到可信结论，不只是单kernel最快。

**当前可直接给出的优先起点：单卡32×4、有效128、BF16、causal、GC On作稳定参照，先优化分桶；速度上限候选是正确FA-varlen＋尽可能关闭不必要的重算，但是否胜出必须实测。** r32、训练cap4096和既定数据流先不因性能猜测改变；正式评测不截断。工程吞吐候选先做短测/dev验收，不为每个候选都完整训练七模型、跑全37项；选定实现后统一用于正式对照。不得通过少训token、跳过长样本或只给赢家延长预算来伪造加速/收益。

## 8. 怎么做模型融合，不把六个模型变成六模型投票

用户已在对话中提供[知乎文章](https://zhuanlan.zhihu.com/p/677088727)全文，先前网页403不再妨碍阅读这份正文。这里以文章提出的问题补充实验，并对照DARE、Fisher、TIES原论文及固定版本的[MergeKit方法说明](https://github.com/arcee-ai/mergekit/blob/6abfb9955bf12e459b32745781cf11b9df949dda/docs/merge_methods.md)区分定义、工具变体和经验观察。

**运行隔离约束：服务器已有其他OpenCode和实验正在运行。本节只补充本地方案，不授权连接服务器、安装/升级依赖、改配置/数据、抢占GPU、停止/重启任务或重训专家。** 已在跑的模型完成后，先只读核对其parent、head与训练流；不满足本设计时标记可比性限制，不为迁就新方案中断或热改它们。第5、6节命令仅是未来候选，也不能在已有同名环境/目录上直接执行。

### 8.1 必须融合有效权重和训练过的head

对每个LoRA专家先得到有效权重：

```text
W_i = W_0 + (alpha/r) * B_i * A_i
delta_i = theta_i - theta_0
```

`theta`同时包含骨干有效tensor和score。先合并每个专家的LoRA，验证合并前后同样输入的输出至数值容差，再执行权重融合。**不能直接分别平均A和B，也不能漏score或随机重建score。**

| 方法 | 定义/首轮固定候选 |
|---|---|
| AVG | `theta_AVG = (sum_i theta_i)/6`，最简单基线 |
| Task Arithmetic | `theta_TA = theta_0 + lambda * sum_i delta_i`；建议首轮lambda0.5，作为预先固定候选而非最优值 |
| TIES | 对delta做幅值裁剪、符号共识与同符号聚合；首轮density0.2、lambda1、`normalize=true`、六专家等权 |
| DARE＋Task Arithmetic | 先按第8.4节随机drop并做论文式rescale，再用与TA相同的lambda0.5相加；隔离DARE的额外作用 |
| DARE＋TIES | 先做同样的DARE，再执行与TIES相同的Trim→Elect→Disjoint Mean；TIES保留率0.2、lambda1不变 |

融合权重/λ/密度只能按独立dev选择，不能看公开测试后改参数追分。首轮先固定少数方法，不上大网格。若AVG等权和TA设lambda1/6，两者数学等价，不重复算成两种独立方法。

这些数值是首轮候选，不是文章或论文保证的最优值。不同方法若进入调参阶段，使用预先约定的同等候选/评测预算；不能只给某个方法大量搜索后宣称算法本身稳胜。

### 8.2 MergeKit有个本项目必须处理的接口问题

固定commit的[`qwen3.json`](https://github.com/arcee-ai/mergekit/blob/6abfb9955bf12e459b32745781cf11b9df949dda/mergekit/_data/architectures/qwen3.json)列的是`Qwen3ForCausalLM`和`lm_head`，没有本实验的`score.weight`。因此不要盲目把SEQ_CLS目录交给默认CausalLM模板后就声称head正确融合。

可采用同版本的[`mergekit-pytorch`](https://github.com/arcee-ai/mergekit/blob/6abfb9955bf12e459b32745781cf11b9df949dda/mergekit/scripts/merge_raw_pytorch.py)原始tensor路径：输入七份共同key集合的**有效完整权重文件**，包含score；FP32计算，BF16导出。缺tensor必须报错，不用`--tensor-intersection`静默丢掉head。输出还要配回正确的分类config/tokenizer，并在训练环境完整重载验收。

这条CLI在官方源码存在，但本地尚未实现本实验的导出、config打包和加载验收脚本；不是现在已经能直接运行的训练/融合工程。融合产物是一个0.6B分类模型，不是MoE，也不是运行六模型输出投票。

可额外保留六专家概率平均作为“输出集成”参考，但它需要六模型推理，不能与一个0.6B权重融合模型混称同一部署成本。

### 8.3 文章中的合理性解释：用于提出假设，不用于提前下结论

| 文章观点 | 本实验怎么落实 | 不能直接推出什么 |
|---|---|---|
| 多个专家来自同一Base/初始化 | 核对同一基座revision、M0、LoRA初值、score初值、tokenizer、标签和attention | 仅模型名同为Qwen3-0.6B不够；相同初始化也不保证融合成功 |
| SFT delta存在冗余 | 比较TA与DARE＋TA、TIES与DARE＋TIES，另留单专家drop诊断 | 不能默认0.6B分类LoRA也能drop90%–99%不掉分 |
| Task vector可能接近正交 | 计算本实验有效delta的相似度、逐层相对范数和符号冲突 | 不能把正交当已知前提；欧氏空间低cosine不保证非线性模型无功能干扰 |
| delta绝对值应较小 | 保存分位数和相对基座的变化尺度，关注分类头与骨干差异 | **0.005不是跨架构/层/参数化的通用安全阈值** |
| 融合数量增加可能带来退化 | 做2/4/6专家的预先固定子集分析 | “模型越多必然越差”及“六模型应保留85%”不是已知定律 |
| 融合可能改善OOD泛化 | 对同任务的新来源/文档群做独立留出验证 | 不能把本来就在训练里的其他benchmark或普通同源test直接叫OOD |

[DARE论文v3第4.6节](https://arxiv.org/html/2311.03099v3#S4.SS6)讨论的是特定模型/参考基座差异，举例为相对较小的delta通常在0.002以内；这也不是普适阈值。不同论文版本或文章中的绝对数值不能变成本项目的自动准入规则。本项目还增加了原基座没有的随机分类头，其训练增量不应与预训练骨干套同一个阈值。

“无需训练数据”只适用于部分方法在参数合并这一阶段的需求；调参和评分仍需要数据，Fisher/RegMean还需代表性样本统计。权重和统计量也可能携带信息，**不访问原始数据不等于获得形式化隐私保证**。

### 8.4 DARE：保留率、随机性与工具实现要写清

本实验优先定义**论文式DARE**，对有效delta逐元素操作：

```text
p = drop_rate
q = 1 - p                         # retention / density
mask_j ~ Bernoulli(q)
delta_DARE_j = mask_j * delta_j / q
theta_DARE = theta_0 + delta_DARE
```

推荐首轮只用一个保守候选：**p=0.5、q=0.5**。先在dev核对；p=0.9/0.99只作为后续有预算时的压力测试，不直接照搬大模型经验。DARE单独作用于一个专家、再评其本方向dev，是“drop本身是否伤能力”的辅助诊断，不是新的专家训练。

- `p=0.9`是丢弃90%，对应保留率/density **0.1**；TIES的density0.2是按幅值保留20%，二者不是同一干预。
- rescale作用于**delta**而非整个基座权重。严格保持的是随机参数增量的期望；固定输入的一层线性变换可以推导相应期望，不保证深层模型输出、准确率或F1严格不变。
- 主比较包含训练过的score，不能只在DARE组临时保留一份未稀疏head。如果要保护score，另做明确的head策略消融，且披露与全tensor处理的差别。
- 所有方法FP32计算、相同保存精度；记录checkpoint hash、每个专家/每个tensor的mask生成规则、随机seed和实际保留率，避免tensor遍历顺序改变mask。
- 首轮一个mask只作初筛；有信号后对同一训练checkpoint补至少3个mask seed，报告波动，不挑最有利mask。**mask seed不是训练seed**，不能充当三个独立训练重复。
- DARE＋TIES的主定义是先随机drop/rescale，再执行同一TIES的幅值Trim、Elect和Disjoint Mean；两阶段保留率分别记录，不能不说明分母就把它们简单相乘。

**固定MergeKit版本与论文式DARE有需要核对的差异：**本次只读检查commit `6abfb995…`的[`registry.py`](https://github.com/arcee-ai/mergekit/blob/6abfb9955bf12e459b32745781cf11b9df949dda/mergekit/merge_methods/registry.py)、[`generalized_task_arithmetic.py`](https://github.com/arcee-ai/mergekit/blob/6abfb9955bf12e459b32745781cf11b9df949dda/mergekit/merge_methods/generalized_task_arithmetic.py)和[`sparsify.py`](https://github.com/arcee-ai/mergekit/blob/6abfb9955bf12e459b32745781cf11b9df949dda/mergekit/sparsify.py)，发现：

1. `ties`默认normalize=true，而`dare_ties`默认normalize=false；不能沿用不同默认值后把所有差异归因于DARE。
2. 该版本`rescale=true`经通用路径使用原tensor/稀疏tensor的**L1范数比**重标定，不是对每次mask固定乘`1/q`；它不应冒称严格相同的论文式rescale。
3. `dare_ties`在该注册路径使用随机稀疏化＋符号共识，并非自动再执行一遍TIES的top-20%幅值Trim。

因此主表中的`MERGE_DARE_TA`、`MERGE_DARE_TIES`是上述**明确数学定义的候选**，尚需单独的预处理/单元验收，不是简单改CLI方法名就已实现。若选择直接使用该版工具变体，分别命名并记录`L1-rescale`、normalize、是否额外Trim；不能将其与严格论文式DARE混写。此处只记录差异，不修改或升级服务器MergeKit。

### 8.5 Fisher Averaging和RegMean：值得比较，但不放在第一轮启动依赖里

| 方法 | 相对简单平均增加了什么 | 需要的额外数据/计算 | 本轮定位 |
|---|---|---|---|
| Fisher Averaging | 用每个参数的重要性加权，而非所有参数同权 | 代表性输入、逐样本梯度、对角Fisher统计；记录期望/采样或empirical Fisher估计方式 | 第二阶段；不是仅凭checkpoint就能完成的参数运算 |
| RegMean | 使线性层融合后的输出在校准输入上尽量接近源模型 | 各专家线性层输入激活Gram矩阵`X^T X`、正则化/稳定求解；非线性层采用明确的后备规则 | 第二阶段；需要额外前向、存储和实现验收 |

Fisher常见对角近似形式为：

```text
theta_Fisher[j] = sum_i(w_i * F_i[j] * theta_i[j]) / sum_i(w_i * F_i[j])
```

分母近零时预先冻结回退规则，例如回退同权AVG；不能简单加epsilon而把该参数意外压向0。原论文可按模型预测分布对标签求期望/采样，**不一定需要人工标签**；若改用真实标签梯度平方，应登记为empirical Fisher，不冒称完全相同的估计。使用per-example统计，不以整个batch平均梯度的平方代替。LoRA坐标中的Fisher也不能直接当有效完整权重坐标中的Fisher使用。

本项目的RegMean应对融合目标的有效线性权重和score明确处理，padding不参与有效token激活统计，冻结公共权重可原样保留；病态Gram需要明确阻尼/非对角缩放等数值策略，不能用未经验收的求逆冒充闭式解已正确实现。

额外统计只能来自允许的训练/单独校准集，不能读取公开test或OOD留出数据来拟合。统计、调参、评测开销分别记录；没有统计文件时标记“未实施”，不能只下一个MergeKit包就把这两行填成已完成。

### 8.6 分阶段安排：不增加新的专家训练是第一原则

1. **第一阶段：**待现有任务自然完成、资源明确允许后，确认六专家与混合控制可比性；比较AVG、TA、TIES、DARE＋TA、DARE＋TIES五种固定候选。
2. **第二阶段：**只对有信息量的方法做mask复验、2/4/6专家数量曲线和机制诊断；均复用既有权重，不为此重新训练六专家。
3. **第三阶段：**有额外资源预算时，收集Fisher/RegMean代表性统计，作为带数据辅助的融合对照；不与第一阶段的纯参数方法混称同等成本。
4. **之后可选：**融合后再微调、同任务checkpoint soup或task-vector subtraction均是另一个问题，不能混入“融合本身相比混合训练”的主结果。

新增DARE/诊断不要求额外下载基座或大Teacher；Fisher/RegMean也可复用允许的现有训练样本作统计，不自动引入新数据集。不因为文章列了某方法就立即在当前服务器运行。

## 9. 最终比较什么：必须看矩阵与差值

### 9.1 主矩阵：特化收益、融合与混合的差别

每个专家、混合模型和五种主要融合方法的正式终点都评全部37个benchmark，而不是专家只评自己方向。Fisher、RegMean只有真正完成统计和验收后才追加，不填造分数。

| 模型 | K不含Chess | L | R | T | A | G/Chess | 官方五维公共总分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E_K | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| E_L | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| E_R | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| E_T | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| E_A | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| E_G | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| M_ALL | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| MERGE_AVG | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| MERGE_TA | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| MERGE_TIES | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| MERGE_DARE_TA | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| MERGE_DARE_TIES | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

两个核心问题对应两个差值：

```text
特化收益_i = Score(E_i, test_i) - Score(M_ALL, test_i)
融合影响_i = Score(MERGE, test_i) - Score(M_ALL, test_i)
专家能力保留差_i = Score(MERGE, test_i) - Score(E_i, test_i)
```

同时报告融合相对本方向专家的差值，衡量保留多少专用能力；若要“保留比例”，必须先定义基准，分母接近零时不报误导性的百分比。可以拼接“已知方向→指定专家”的oracle路由作为多专家路由参考，但它不是理论性能上限、单个融合模型或未知方向的实际路由系统，融合或混合也可能超过某个本方向专家。

- 各benchmark用原生accuracy/F1/nDCG等和官方Skill双报告；六轴Skill与官方五维总分分开。
- 正式评测不截断、不删候选，超上下文按Unsupported记录；每个模型对同一冻结测试全分母评分，不取“大家都答了”的成功交集。
- 用独立dev选方法，公开测试不负责调权；保留文档/题组级预测以做配对bootstrap。
- 首轮单seed只能得初步方向；关键专家/混合/胜出融合至少补匹配seed，报告均值、波动与分方向取舍。
- 训练总GPU-hours、总曝光、模型数量、单模型部署尺寸、实际推理时间一并展示；不先声称特化或融合一定优于混合。

### 9.2 按文章补充：融合2/4/6个专家，而不只看六专家一个点

为了检验“融合数量多了会不会更难”，优先在AVG、TIES上做以下辅助分析，复用同一批专家权重：

- 在看测试结果前冻结多个嵌套顺序；每个顺序取前2、前4、全部6专家。可先用`K,L,R,T,A,G`的六个循环移位作为一个**预先固定、各方向出现次数平衡的有限方案**，不是覆盖全部15对/15个四元组。
- k=6只有同一个全集，不把六个重复顺序算六个独立组合；同一训练seed的所有子集也不是独立训练重复。
- 主数量曲线在**参与融合的方向**上，按方向等权报告相对各自专家的分差；全六方向结果另列，不能因为2专家没有其他4方向就将全榜掉分全部解释成参数干扰。
- 记录组合身份和波动；某个有利的二元组不能代表所有二专家组合，也不能只挑有利子集。
- AVG/TIES采用明确、可比较的归一化；若补TA曲线，要预先冻结lambda随k的规则，不能把求和幅度随k增加导致的变化全部归因于“模型越多干扰越强”。
- 与`M_ALL`的主公平比较仍是**全部六专家总曝光匹配**。若要把k=2/4结果与相同子集混合训练作因果比较，需要另外的子集混合控制，不能拿全六方向`M_ALL`冒充。

文章中的约85%来自特定任务和归一化准确率口径，不预测本实验六轴结果。主报告以Skill点差、原生指标差和coverage为先；“保留百分比”分母接近零时不报，不把不同任务F1、nDCG和准确率随意相除求平均。

### 9.3 ID与OOD分开，不把普通测试包装成分布外泛化

| 评测层 | 要验证什么 | 边界 |
|---|---|---|
| 同源分组留出/ID | 专家在本方向是否优于混合，融合保留多少能力 | 原题/文档/标题/棋盘按组隔离，不按pair随机拆 |
| 官方37项公共评测 | 与冻结benchmark同协议的能力与覆盖 | 不等于私测Full，也不自动等于OOD |
| 源级留出/OOD | 同一能力在未用于训练的新来源/群组上是否泛化 | source在六专家、混合、Fisher/RegMean校准与融合调参中都不能出现 |

如果尚未训练，可在数据准备时为各方向预留完整来源作确认集；这会改变允许训练池，必须在冻结数据预算前明确。**如果服务器任务已使用了相关来源，不能回头移动、删除或重划其训练数据以迁就本节。** 保留现有运行记录，另核验真正未用过的来源，或者明确本轮没有可认证的源级OOD结论。最终确认集不同时用于选择融合lambda、密度或mask。

### 9.4 融合为什么成功/失败：增加少量机制诊断

以下都是待实施的离线诊断，不是已经观察到的性质：

1. **六乘六task-vector cosine矩阵**：使用相同坐标中的有效权重delta，分别看全局、逐层和score；零范数项记NA，不能假称正交。
2. **delta尺度**：记录绝对值分位数、`||delta_layer||/(||theta0_layer||+epsilon)`、各专家增量范数；骨干和新score分开看，不以0.005一刀切。
3. **符号冲突**：冻结小值阈值和分母，分别统计稀疏化前后、至少两个专家有有效更新的坐标冲突；单纯cosine低不证明没有符号干扰。
4. **稀疏化实际效果**：记录DARE每tensor保留率、重标定规则及单专家dev变化；若drop本身已伤专家，不把融合后全部损失归为跨任务冲突。
5. **专家质量检查**：专家本方向是否真的比混合控制或相关参照好；若专家尚未形成专用能力，不能用差融合结果证明“融合技术普遍无效”。

这些诊断支持解释和提出下一步假设，不单凭相关性声称已证明模型融合机制。没有原权重/共同M0时，不能从CSV分数反推cosine或delta分布。

### 9.5 计算成本、随机性与最终交付

- 将训练seed、DARE mask seed、子集组合和评测bootstrap seed分开保存；多个mask不增加训练独立性。
- 12个基础条件是每个方法一个固定候选的配置计数；若两种DARE各做3个mask、其余不变，则为**16个评测终点**，仍只有7个训练终点。子集、单专家drop诊断和其他seed另计。
- “融合不更新参数”不等于零资源：参数读写/CPU RAM、额外统计、dev搜索和完整公开评测都计时；Fisher有反向计算，但不因此冒称新增了专家训练。
- 优先dev小面板核对数值与明显失效，再按冻结规则做正式评测；未完成公开评测的候选标记dev-only，不填造完整分数。
- 主结论至少同时回答：**是否存在本方向特化优势、哪种融合保留最多专家能力、相对混合训练差多少、是否有独立OOD收益、节省了什么成本而付出什么损失**。

### 9.6 参考依据与本次核查范围

- 用户贴出的知乎正文：《模型融合(Model Merging)：合理性、常见技术及其特性》，用于提出本轮补充问题。
- [DARE原论文v3](https://arxiv.org/html/2311.03099v3)：drop/rescale定义、尺度与模型大小限制；文中既有成功也有失败情况。
- [Task Arithmetic](https://openreview.net/forum?id=6t0Kwf8-jrj)：task vector与缩放思想；不将推荐lambda当本项目最优值。
- [Fisher原论文v2](https://arxiv.org/html/2111.09832v2)：对角Fisher、逐样本梯度和近零权重处理。
- [RegMean原论文入口](https://openreview.net/forum?id=FCnohuR6AnM)：本次OpenReview页面被浏览器验证拦截，未宣称阅读全文；本节Gram矩阵/线性层说明同时依据用户正文及[TIES第5节对基线的定义](https://arxiv.org/html/2306.01708v2#S5)。
- [TIES原论文v2](https://arxiv.org/html/2306.01708v2)：三步定义、数量曲线、OOD及oracle-sign诊断。
- 固定commit的MergeKit源码见第8.2/8.4节；核对实现定义，不等于本地运行了这些算法。

本次只改本地说明文字并读取公开资料，没有读取服务器现场状态、运行模型、提取权重统计、安装依赖或写入其他会话目录。

## 10. 开跑前最后核对

- [ ] 主模型与Tasksource下载到本地，revision和文件完整性有记录。
- [ ] 294来源任务的归属表、unassigned、六组条数和token已经统计；不随机六等分。
- [ ] 各补充来源的真实train/dev/test、许可、原题分组及37项评测去污染通过。
- [ ] M0的基座、LoRA初值、score、tokenizer和config已保存；七个训练的起点hash一致。
- [ ] 数据转换和Engine语义一致，保存重载、真实mask、head更新、LoRA更新和一次optimizer step通过。
- [ ] 专家与混合控制的按来源样本multiset、实际token/更新预算已经冻结。
- [ ] 融合导出覆盖score、所有tensor的key/shape和分类config；输出可以真实重载。
- [ ] 五种主方法的lambda/归一化、DARE rescale定义与mask规则已冻结；论文式方法和工具变体不混称。
- [ ] ID/OOD、校准、调参和最终确认数据边界明确；不为新设计修改正在运行任务的数据与配置。
- [ ] 同一冻结37项suite、scorer和排除表可运行；数据准备与训练/评测预算分别登记。

**未来执行顺序：确认现有任务和资源边界 → 在不修改现有实验的前提下复用/补齐资产 → 六方向数据审计与共同M0 → 六专家＋一个匹配混合训练 → 五种主要权重融合 → 同一测试矩阵、数量曲线与可认证的OOD/机制分析。** 已完成的阶段复用，不因文档新增方法自动重训或抢占资源。
