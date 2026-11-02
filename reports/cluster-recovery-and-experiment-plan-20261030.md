# 集群环境重建、实验状态还原与后续规划

整理日期：2026-10-30。依据：当前 Git 仓库中的文档、JSON/CSV 结果、研究脚本与提交记录。

> **范围说明：本文是本地证据复盘和恢复方案，不代表已经恢复集群。** 本次未连接原集群、未安装训练依赖、未下载模型、未启动训练或评测。集群为什么丢失、卷是否还在、是否有外部备份，无法从本目录判断。

## 1. 先说结论

1. **有效工作主线是“小模型的 Language Understanding 实验”。** 运行记录指向4×H800 NVL、Qwen3-0.6B、二分类匹配头、LoRA，以及 Decision Index 的9项Language评测。本文仅据实际运行与后续产物还原；用户已作废的方案不纳入恢复依据或后续计划。
2. **这个仓库是实验研究和结果归档，不是完整可启动的集群工程备份。** 报告中写着某个 `/home/jovyan/.../final` 路径，不表示那个目录或权重已经同步到本地。恢复需要分开处理环境、代码、原始数据、处理缓存、模型权重、优化器/随机状态和评测记录。
3. **要先清点可救回的资产，再决定恢复还是重跑。** 找回原 adapter 和分类头，可以恢复模型函数；只有权重、没有 optimizer/scheduler/RNG/数据游标，只能开新续训，不是精确断点恢复；只剩分数和路径，不能反推出训练后的权重。
4. **不能把旧文档中的“正在训练”“等待 GPU”“尚未做比例实验”直接当作现在的待办。** 本目录后续还有 2026-10-29 的补充结果和融合报告，须以最新产物与具体证据交叉确认，详见实验台账。
5. **后续重点应是恢复一个可信参照、完成少量高信息量实验，而不是把历史搜索全部重跑。** 新环境改变、数据重建和训练方法改变，必须分别记账，不能把环境差异误算成方法收益。

**截至本地最新看板快照2026-10-30 02:56:38 UTC：**归档包含104个完整Language评测方案的汇总记录，历史上对应103个实体checkpoint终点和1个虚拟概率集成；另有8个仅dev pilot和49个资源测量单元。当前最充分复验的配方是r32四seed **35.0492±0.9593**，最高单次 **36.3303**。比例和固定融合首轮均已完成，未替换主线。Teacher有“9998对软标签已输出”的较晚派生记录，但标签本体和ready回执不在本地，三臂Student未见训练/评测完成证据。

### 阅读顺序

- 想知道以前是什么环境：看第 3 节。
- 想知道本地还剩什么、哪些东西必须找回：看第 4 节。
- 想知道哪些实验跑过、哪些没跑：看第 5、6 节。
- 想开始恢复：按第 7 节的验收关卡推进。
- 想继续研究：看第 8 节的实验顺序和第 9 节的资源约束。

## 2. 证据规则：什么能确定，什么不能

本文使用以下标记：

- **【归档证实】** 当前文件中存在相应结果或明确配置；仅证明历史归档所描述的状态，不证明原文件现在仍在集群。
- **【合理推断】** 多处记录支持，但缺原配置、原权重或现场验证。
- **【待核实】** 本目录没有足够证据；需新集群检查、旧存储/备份或补实验。
- **【新建议】** 本文基于已有结果提出的后续方案，不是声称以前已经决定或完成。

判定先后关系时，**具体运行结果、审计和后续 addendum 优先于早期计划和旧状态快照**；文件名日期不是唯一依据。报告提到的 `running`、GPU 编号、PID、锁和许可均是历史记录，不能拿来代表当前进程或授权。

### 主要证据入口

| 证据 | 用途 | 注意事项 |
|---|---|---|
| [原始环境与 R0 路线](../language_roadmap.md) | 2026-10-27 的硬件、软件版本、R0 配置和原始对照设计 | 截图/计划层证据，后续实际结果可能已更新 |
| [服务器执行清单](../opencode_language_checklist.md) | 数据标签、分类链路、评测契约与目录约定 | 不是本次启动训练的指令 |
| [实验总报告](language-understanding-study-20261029.md) | 历史阶段、质量结果、实验限制 | 应连同后续 addendum 阅读 |
| [历史实验 CSV](language-understanding-study-20261029/runs.csv) | 原 91 个完整评测终点 | 不包含后来所有新终点 |
| [比例实验合并台账](language-understanding-study-20261029/runs-through-pair-ratio.csv) | 95 个历史完整评测终点 | 不等于本地有 95 份权重，也不等于全部最新融合产物 |
| [比例补充报告](language-understanding-study-20261029/PAIR-RATIO-ADDENDUM.md) | 正负比例实验的正式结果和曝光审计 | 覆盖早期“尚未训练”的记录 |
| [融合报告](../research/lu-specialist-fusion-20261029T152144Z/FUSION-REPORT.md) | specialist、soup、merge、distillation 的推进状态 | 区分已评分候选和待完成阶段 |
| [融合评审](../research/lu-specialist-fusion-20261029T152144Z/FUSION-REVIEW.md) | 可归因性、曝光配平、候选是否晋级 | 负结果也需保留 |
| [实验看板](experiment-dashboard/index.html)及[数据](experiment-dashboard/data.json) | 离线查看九项分数、阶段和 seed | 是派生快照，不是权重存储或实时调度器 |
| [资源研究报告](checkpoint-length-scaling-20261029/RESEARCH-REPORT.md) | 长度、checkpointing、mask、OOM 的实测 | 部分测试没有 optimizer 更新 |
| [数据来源清单](../research/language-axis/manifest.json)及[审计](../research/language-axis/audit.md) | 训练来源、原始划分、revision、许可 | 下载文件中可能同时有 train/dev/test |

## 3. 能还原到什么程度：旧集群环境与实际训练配置

### 3.1 系统、硬件和软件

| 项目 | 历史值 | 证据强度与恢复说明 |
|---|---|---|
| 使用形态 | Linux Pod，用户`jovyan`，家目录`/home/jovyan` | 【归档证实】文档和实际任务路径支持；不是已掌握容器镜像 |
| 工程根 | `/home/jovyan/assumeengage` | 实际worker、run、数据路径共同支持；新节点可换根，但必须显式映射 |
| GPU | 4×NVIDIA H800 NVL | R0四卡DDP、后续GPU0–3独立任务记录；不是当前可用硬件承诺 |
| 单卡可见容量 | 93.0870 GiB，文档约写94 GiB | [逐case环境](checkpoint-length-scaling-20261029/cases/fixed-L1024-B16-on-r1.json)的`environment.gpu_total_GiB` |
| 训练环境 | `~/assumeengage/temp_env/train` | 历史Python入口为该目录`bin/python`；Conda激活脚本记录为`/opt/conda/etc/profile.d/conda.sh` |
| Python | 3.11.17 | 完成后的[R0报告](language-baseline-20261027.md)环境记录 |
| torch | 2.11.0+cu130 | 机器JSON实测字段；优先匹配原wheel构建，不仅匹配主版本 |
| Torch CUDA构建 | 13.0 | `environment.cuda`；**不是驱动或本机nvcc版本** |
| transformers | 5.5.0 | 机器JSON实测；已有该版attention源码研究材料 |
| peft | 0.18.1 | 机器JSON实测 |
| Unsloth / unsloth_zoo | 2026.10.9 / 2026.10.9 | R0完成报告记录；后续SDPA worker不可自动等同于Unsloth路径 |
| datasets | 4.3.0 | R0完成报告记录 |
| accelerate | 1.15.0 | R0完成报告记录 |
| huggingface_hub | 1.33.0 | 仅旧截图转述，缺相应完整运行时环境文件，证据较弱 |
| pyarrow | 23.0.1 | 同上；应作为恢复候选版本，安装后实际验收 |
| 后期attention | Transformers SDPA，causal，`use_cache=False`；测得cuDNN attention算子 | [manifest](checkpoint-length-scaling-20261029/manifest.json)及[路径诊断](checkpoint-length-scaling-20261029/MIXED-LENGTH-DIAGNOSIS.md) |
| 精度 | BF16 autocast，参数中仍有FP32部分 | 该r16测量中BF16参数596,054,016、FP32参数10,092,544；不能写“所有参数均BF16” |

**没有足够证据精确还原：**Linux发行版、kernel、容器镜像digest、驱动、系统CUDA toolkit、cuDNN/NCCL版本、CPU型号和核数、RAM/磁盘容量、挂载方式、网络/NVLink拓扑、Pod/作业配置、完整pip/Conda锁定文件。未见Slurm等调度配置，不能编造分区名、账号或`sbatch`脚本。

`flash_sdp_enabled=true`只说明PyTorch相应后端开关，不证明安装过第三方`flash-attn`包。基座config里的`transformers_version=4.51.0`也不是本项目运行环境版本。

### 3.2 R0：真正已经完成的第一条基线

| 项目 | 实际值 |
|---|---|
| 基座 | `Qwen/Qwen3-0.6B`，本地目录`~/assumeengage/models/Qwen3-0.6B` |
| 基座精确HF revision | 【待核实】归档主要保存旧`assets-audit.json`指针，没有展开实际commit；路径字符串不能当revision |
| 模型与readout | `Qwen3ForSequenceClassification`；`Linear(1024,2,bias=False)`，最后有效token处做序列分类 |
| 输入与标签 | premise＋hypothesis匹配判断，原始Tasksource`{0,2}`映射为`{0,1}`；0匹配，1不匹配 |
| 损失 | 普通二分类CE；不是生成式token loss，不是255-way整题头 |
| LoRA | r16/alpha32/dropout0，attention和MLP投影；score同时训练和保存；确切原始target_modules列表仍应找配置 |
| 训练 | BF16、4卡DDP；每卡batch32、累计1，有效batch128 |
| 优化 | lr=1e-4、cosine、warmup3%、weight_decay0.01 |
| 长度与seed | 训练max_length4096、seed3407、causal；已由后续完成记录补齐，不再沿用早期“待核验”状态 |
| 完成量 | 1 epoch、14,171次更新，Trainer记录6003.54秒；不是现在仍处于4500步 |
| 选出的产物 | validation最低的`checkpoint-13000`导出到`checkpoints/final`，不是简单取最后一步 |
| 权重依赖 | final是LoRA＋训练过的score，重载仍需正确基座和tokenizer |

R0旧位置：

```text
/home/jovyan/assumeengage/runs/language-baseline-20261027T061200Z/
  checkpoints/final/                 # adapter、训练过的score、配置/tokenizer
  prepared-data/                    # 实际清洗和编码产物
  config.json / environment.json / data-report.json
  training-complete.json / internal-test.json
  language-suite/ / evaluation/      # 冻结请求、计分与逐请求输出
```

**这是找回资产的索引，不是本地现存目录树。** 其中原adapter记录为40,426,368字节，SHA256：

```text
0b0334c5e6ac7919a0a5d7d3ad88b201f7127c516073bf7933687d8ae69af554
```

此hash来自[旧manifest](checkpoint-length-scaling-20261029/manifest.json)和内嵌R0摘要，可用于找回后的验真；本地没有该权重，**本次未重算这个权重hash**。

### 3.3 后续主线与评测接法

- 长预算研究：共享同一R0，fresh optimizer/scheduler，每个分支新增50M实际非padding token，主要lr5e-5、有效batch128；不是按R1→R2→R3串行把权重叠加。
- 正式容量对照：r16/alpha32与r32/alpha64保持`alpha/r=2`；r32保留原已训练LoRA切片、训练过的score，并以零贡献新增部分扩容。
- 当前已复验配方：**causal、r32/alpha64、D0:目标池20:80实际token、原题任务采样指数0.5、原生英语讽刺类别平衡＋目标池5%token配额**。这几个比例的分母不同。
- 小池候选：通常每卡一个模型，microbatch32×累计4=有效batch128；不能误写为四卡共同训练一个候选。
- 评测kit：**Decision Index 0.3**，commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`。榜单0.4.1快照不是本地kit版本。
- Engine：对state、instructions和每个criteria渲染文本对，取`logit(匹配)−logit(不匹配)`，温度1，在全部候选上归一化；不是把分类checkpoint直接当CausalLM运行。
- 正式评测真实上下文上限40,960，不截断、不删候选。九项31,715请求、131,360问题字段；只做Language，不能冒充全榜Full。

来源：[R0报告第2–5节](language-baseline-20261027.md)、[总报告“模型与推理”](language-understanding-study-20261029.md)。

### 3.4 下载与网络：只保留历史线索，不假定仍可用

历史文档记载：清华pip镜像；HF API/元数据走`https://d6108366.hf-mirror.com`，大文件走`https://hf-mirror.net`；没有本地代理。**本次没有验证域名可达性或镜像内容，不能保证仍可沿用。** 原自定义`download_tasksource.py`未同步；单设`HF_ENDPOINT`也不保证blob按另一域名下载。

未来补下载应固定repo ID、revision和需要的文件，不下载无关目录。下面仅为已核对[Hub v1.33.0官方语法](https://github.com/huggingface/huggingface_hub/blob/v1.33.0/docs/source/en/package_reference/cli.md)的示例；**不是本次执行命令**：

```bash
# 基座revision未知，必须先从备份查明；查不明则固定一个新commit并记为重建版本。
hf download Qwen/Qwen3-0.6B \
  --revision "<已核实的基座commit>" \
  --local-dir "$RECOVERY_ROOT/models/Qwen3-0.6B"

# Tasksource的原始数据revision有记录；下载清单仍应确认default所需分片。
hf download tasksource/zero-shot-label-nli --repo-type dataset \
  --revision 9f0763b798363a6d4ca8f65de2dc80e0fcb48ae2 \
  --include 'data/*' --include 'README.md' \
  --local-dir "$RECOVERY_ROOT/datasets/tasksource/zero-shot-label-nli"
```

`RECOVERY_ROOT`应显式设置为新恢复目录。示例包含占位符，不可原样当一键恢复脚本；不要把凭据写进命令、文档或分享链接。

## 4. 本地到底还剩什么：资产与缺口

### 4.1 现在可直接保留利用的东西

| 类别 | 本地内容 | 能恢复什么 |
|---|---|---|
| 实验台账 | 总报告、`experiment-facts.json`内嵌旧JSON、CSV和addendum | 阶段、seed、预算、聚合分数、历史产物路径和部分审计 |
| 原始目标数据 | `research/language-axis/`的14份manifest资产，含可选阿拉伯语iSarcasm | 有一批真实数据文件，不是全都要重新下载 |
| 构建参考 | Tasksource上游脚本片段、来源列表、构建摘要 | 理解标签语义和数据来源；不是本项目完整清洗器 |
| Attention研究 | Transformers5.5.0固定源码片段和audit | 诊断mask执行路径；不是可直接替代完整包的独立模块 |
| 资源/长度代码 | batch、长度、checkpointing测量代码和JSON | 复用统计方法、历史显存与吞吐证据；GPU重跑仍缺依赖资产 |
| 融合记录 | 17份报告/CSV/JSON，包括`MERGE-RESULT.json`、曝光审计 | 恢复方法和负结果；没有tensor权重 |
| 离线看板 | HTML/JS/CSS、`data.json`和`data.js` | 直接查看104方案；不需重新生成数据或启动训练 |

**本次实际校验：**已逐一重算14份原始数据资产的大小与SHA256，全部符合[manifest](../research/language-axis/manifest.json)；两份ZIP的CRC检查也通过。这个验证只说明文件与归档一致，**不等于已重新完成清洗、去污染、许可审批或标签正确性审计**；未全量重读Parquet验证语义。

### 4.2 关键缺失：这些不是装几个包就能补回的

| 缺什么 | 当前工作树核查结果 | 恢复影响 |
|---|---|---|
| 基座与训练权重 | 未发现任何`.safetensors/.bin/.pt/.pth/.ckpt`；没有`models/`、`runs/` | 不能立即评测、续训或恢复任何原模型 |
| 原R0与LR32 final | CSV保存的是旧绝对路径，非文件内容 | 必须找回adapter＋score；否则按第7节范围门槛建立新参照 |
| 优化器和精确断点状态 | 未见optimizer/scheduler/RNG/trainer state | 有权重也未必能断点恢复 |
| 处理缓存与采样表 | 未发现`.arrow/.npz`、`prepared-data/`、`schedule.npz` | 原题选择、token曝光、候选顺序和采样无法直接回放 |
| 训练核心代码 | 缺完整`language_baseline/`、`model_utils`、旧`code/`快照、`launch.sh` | 原训练/加载/恢复机制不能直接运行 |
| 评测工程 | 缺完整kit、自定义Engine、冻结Language请求及多数逐模型原始输出 | 不能仅凭汇总分重算官方逐请求评分或逐题一致性 |
| 环境锁定 | 缺完整requirements/Conda环境导出/lockfile和原environment内容 | 可近似配出候选环境，无法保证完全相同 |
| Teacher | 缺`research/lu-teacher-distillation-20261029/`、权重、ready、软标签、Student代码与结果 | 不能确认软标签资产可用，也不能启动既有三臂入口 |
| 更大模型资产 | 缺`lu-asset-prefetch-20261028/`及模型本体 | “资产已齐”只可作历史陈述，新环境仍须清点 |

缺失结论限定于当前工作树；本次没有穷尽外部备份、旧卷或所有Git历史对象。

**代码中的直接断点：**

- [checkpoint benchmark](checkpoint-length-scaling-20261029/benchmark.py)第96–99行依赖`language_baseline.common.PairEncoder`；第172–174行依赖旧`PILOT/code/model_utils`。
- 同脚本第99/118/121行依赖`pool-1.arrow`、R0 prepared validation和Tasksource validation原始Parquet；本地未保存那些实际输入。
- [长度诊断](length-label-benchmark-20261029/run_length_diagnostic.py)依赖旧`active-code.json`和加载器；固定阶梯所需`source-pairs.json`、`inputs-medium.json`等也缺。
- Tasksource的`build_label_nli.py`、`recast.py`还有未同步的上游模块，不能把片段当完整R0清洗/训练实现。
- [旧收集器](language-understanding-study-20261029/collect_facts.py)收集说明明确未复制巨型预测和权重；[看板生成器](experiment-dashboard/build_data.py)仍读取旧绝对`source`路径。**现在直接重跑生成器不是恢复手段，可能失败或漏掉旧结果。**

### 4.3 数据版本必须分开恢复

| 名称 | 已归档规模 | 恢复注意 |
|---|---:|---|
| 原始Tasksource default | train1,899,993 / val24,971 / test25,440 | revision见第3节；不是清洗后的D0 |
| R0/D0 | train1,813,810 / val24,609 / test25,004 | 排除重复/冲突/关联和若干来源后所得；原清洗实现缺失 |
| D1目标数据 | ANLI＋HellaSwag | D1是目标数据阶段；R3/R4实际使用D0＋目标数据 |
| D2目标数据 | train524,448 pair、clean-dev64,362 pair | 含ANLI/HellaSwag/ACOS/iSarcasm/RAGTruth；不是九项任务全部有原生train |
| 后续小训练池 | selected-D0 22,909＋selected-D2 94,444＝117,353 pair | 与历史D0大池不同；需找回选池映射 |
| 旧小dev / 补充面板 | 1,476 / 5,904 pair | 已多次用于选择；不能改名成全新独立测试 |
| 比例轮自然dev | 67,202 pair＝D0原文2,840＋D2 64,362 | 与宽D0长度诊断面板不同 |

R0已记录整任务排除`isarcasm/intended`、`ragtruth/grounding`及两域`semeval2014-absa`；不能下载原始Tasksource后直接把其train当“已还原的D0”。FinEntity的979份未拆分文档只作评测，金融训练可另审SEntFiN；ContractNLI、NLI4CT和RAGTruth下载包中的dev/test不能入训。RAGTruth受限原始上下文不随报告或模型重新发布。

## 5. 已经跑过什么：不要重复启动的实验台账

### 5.1 总数量与时间线

```text
R0 1 + 长预算分支47 + 正常小池39 + 应急短训4 = 原91终点
原91 + pair ratio4 = 95终点阶段快照
95 + 五专家5 + 匹配混合1 + 权重融合2 + 概率集成1 = 104评测方案
实体checkpoint终点103，虚拟集成1；不是104次独立训练，也不是本地有103份权重。
另外：batch/LR仅dev pilot8；资源测量49。它们不填造Language分数。
```

| 时间（UTC） | 归档状态 |
|---|---|
| 10-27 | R0训练、内部test、完整Language完成 |
| 10-27至10-29 | 数据×attention、D2、采样/混合、讽刺策略、JOINT/rank及小池探索完成 |
| 10-29 08:00 | 8个batch/LR仅dev pilot完成 |
| 10-29 14:30 | pair ratio四臂正式训练与完整评测结束，15:11补充验证 |
| 10-29 16:57 | 专家/混合/融合9方案全部结束，随后复核与Teacher资源交接 |
| 10-30 02:30 | 路线图记Teacher62.6GB、31文件下载校验完成；原验收文件本地缺失 |
| 10-30 02:56 | 最新看板记Teacher已输出9998 pair软标签；仍无Student终点 |

### 5.2 长预算受控研究：主线已经做了三/四seed

下表除R0外均为共享R0起点的续训对照；每个终点新增50M token。`±`为样本标准差，不是独立全流程重训方差。详细逐seed见[runs.csv](language-understanding-study-20261029/runs.csv)。

| 阶段/代号 | Language Skill，均值±标准差 | seed | 已得到的结论 |
|---|---:|---:|---|
| R0 | 17.7407 | 1 | 内部test accuracy82.4028%、Macro-F1 82.3905%，不能当Language分 |
| R1 D0 causal | 17.3999±0.7815 | 3 | 单纯多训原池没有稳定涨分 |
| R2 D0 bidirectional | 11.5106±0.3011 | 3 | 本次R0后切双向下降 |
| R3 D0＋ANLI/HS causal | 24.9242±0.2990 | 3 | 相对R1配对平均+7.5243 |
| R4 D0＋ANLI/HS bidirectional | 15.6718±1.1930 | 3 | 仍不如同数据causal，不再把2×2列为待跑 |
| R5_DATA扩充D2 | 30.7531±0.5275 | 3 | 相对R3约+5.8289，但有任务退化 |
| D2任务指数1 / 0.5 / 0 | 30.9290 / 30.7827 / 29.6455 | 各3 | 按clean-dev选0.5，不按公开分最高选1 |
| D0:目标80:20 / 50:50 / 20:80 | 27.7195 / 30.7827 / 31.8311 | 各3 | 采用20:80实际token；不是每seed都涨 |
| 原生SAR平衡 / 仅5%token / 两者组合 | 31.9832 / 33.3989 / 32.3525 | 各3 | 按dev选组合；与单平衡dev差很小，不称稳胜 |
| JOINT r16/alpha32 | 33.7402±0.5232 | 4 | 联合20:80＋SAR策略已做真实确认 |
| LR32 r32/alpha64 | **35.0492±0.9593** | **4** | 四个匹配seed均高于r16，配对平均+1.3090；保留为正式参照 |

**R0 token口径：**所选checkpoint曝光342,474,304；完整训练实际计算373,545,953；缓存token总量373,382,083。三者含义不同，不应混成“训练token”。

**历史最高单次权重索引：**seed3409、Language36.3303216375：

```text
/home/jovyan/assumeengage/research/lu-lora-capacity-20261028/
  runs/LR32/seed-3409/arms/LR32/main/final
```

这个最高单次不替代四seed均值，也不证明该权重已备份。

### 5.3 主线九项分项：还差在哪里

来自[总报告九项分解](language-understanding-study-20261029.md)；LR32为四seed均值，非最好单次：

| ID | 任务 | R0 Skill | LR32四seed平均Skill |
|---:|---|---:|---:|
| 11 | ContractNLI | 22.97 | 21.05 |
| 12 | ANLI | 0.00 | 17.71 |
| 29 | HellaSwag | 25.47 | 57.60 |
| 38 | ACOS | 4.92 | 42.09 |
| 39 | FinEntity | 65.10 | 69.17 |
| 40 | iSarcasmEval | 11.53 | 18.67 |
| 41 | VAST | 20.85 | 20.39 |
| 42 | NLI4CT | 10.62 | 16.83 |
| 59 | RAGTruth | 0.21 | 50.88 |

总体提高不表示每项提高。ContractNLI、VAST未改善，ANLI和临床任务仍弱；但不能仅凭低分判定容量瓶颈。Language是官方加权校正分，**不是上述九行算术平均**，Skill0也不是零题答对。

### 5.4 小池、pilot和比例首轮

- **小池快筛已完成：**39个正常终点（12×20M＋27×17M）及4个约1M应急终点。应急结果不用于选方向；正常短跑没有找到可稳定替换长预算主线的配方。
- **r64已试过，不是未跑：**17M下公开分有升，但SAR保护面板退步，未推广。D0比例、SAR配额、任务指数0.75/1的多轮筛选也已有负结果，不默认继续网格。
- **batch/LR pilot已完成：**8×3M，r16、两个seed、仅dev。batch128/lr1e-4的原生宏均值0.42561较好，但不是Language42.56，也未在r32/较长预算确认。
- 长度与资源测试已完成，其范围和工程结论单列第9节，不算新能力模型。

比例首轮共同seed3407、expanded-R0-r32、各**19,999,954实际token**，所有CE权重均`[1,1]`。来源：[增补报告](language-understanding-study-20261029/PAIR-RATIO-ADDENDUM.md)、[汇总CSV](language-understanding-study-20261029/pair-ratio-summary.csv)。

| 臂 | 实际正pair占比 | 实际正token占比 | 自然dev Macro-F1 | Language Skill |
|---|---:|---:|---:|---:|
| NEG2 | 36.4618% | 34.5108% | 90.8950% | 29.0525 |
| NATURAL | 45.5127% | 47.2135% | **90.9691%** | 29.4933 |
| POS2 | 55.3654% | 60.4611% | 90.1462% | **31.2347** |
| POS4 | 64.1143% | 71.6486% | 88.1054% | 30.6752 |

**结果解释：**提高正pair曝光提升匹配召回，却损害不匹配召回；POS2公开分提高但自然F1下降，三个处理均未通过共同晋级线。精确全局50:50和CE类别权重没有测；不能宣称1:1或55%最佳。固定token下原题draw与唯一样本覆盖也变了，不是纯类先验干预。

### 5.5 专家与固定融合首轮：已经完整交付的负结果

共同expanded-R0-r32、seed3407；专家每个约3M，匹配混合控制14,995,340 token为五专家之和。来源：[九方案CSV](../research/lu-specialist-fusion-20261029T152144Z/dashboard-runs.csv)、[评审](../research/lu-specialist-fusion-20261029T152144Z/FUSION-REVIEW.md)。

| 方案 | Language Skill | 训练/部署性质 |
|---|---:|---|
| ANLI专家 | 17.9212 | 新训练，180次更新 |
| HellaSwag专家 | 19.3337 | 新训练，219次更新 |
| ACOS专家 | 19.1991 | 新训练，197次更新 |
| 英语讽刺专家 | 16.7531 | 新训练，433次更新 |
| RAGTruth专家 | 19.2575 | 新训练，仅27次更新 |
| 相同总曝光混合控制 | **31.0712** | 新训练，1055次更新 |
| 等权有效权重平均 | 20.3541 | 无新增训练；历史生成单模型权重 |
| 固定TIES density0.2、λ1 | 24.3858 | 无新增训练；历史生成单模型权重 |
| 等权概率集成 | 20.6175 | 五模型输出组合；没有第九份新checkpoint |

九方案均完成31,715请求、100%coverage、零错误/Unsupported。融合按共同R0的有效delta计算，含训练过的score，**没有直接平均LoRA A/B**。概率集成成本是五模型，已有latency求和不是新测端到端延迟。

本轮三种固定融合都不如混合控制，不能继续把“跑首轮融合”列待办；也不能据弱专家、单seed、短预算宣布所有融合方法无效。本轮未推广新配方，不自动补密度/权重网格。`MERGE-RESULT.json`只存统计和旧路径，不是权重。

## 6. 还没跑完什么：真正的未完成清单

### 6.1 当前核心未交付：Teacher验收与三臂Student

| 环节 | 最新可用证据 | 正确登记状态 |
|---|---|---|
| Teacher选型 | 路线图称KnowLine31B，当时Full71.86 | 历史选型；缺精确repo ID/revision、原生推理代码和环境。不是本次核实的当前榜首 |
| 权重下载 | 路线图称10-30 02:30:52完成62.6GB/31文件验证 | 历史报告称完成；权重及原始校验文件不在本地 |
| 软标签 | 02:56:38看板称9998 pair已输出；[此前路线记录](experiment-roadmap-status-20261029.md)准备4166道问题，完成问题数与映射待原始manifest确认 | **有较晚完成线索，原始ready和标签尚未恢复验收** |
| HARD Student | 原文有CPU准备和入口描述 | 未见GPU训练或完整评测结果 |
| SOFT Student | 同上 | 未见GPU训练或完整评测结果 |
| HYBRID Student | 同上 | 未见GPU训练或完整评测结果 |

较晚看板文案不是随意猜测：[生成器第107–114行](experiment-dashboard/build_data.py)只有在旧`teacher-labels-ready.json`存在时才写“已输出”。但原ready已不在本地，且路线状态`running`是固定写入，**不能据此证明现在仍运行、标注资产完好或Student已启动**。因此优先找回标签，而不是默认重下Teacher、重标一遍。

### 6.2 明确未测、但不应一口气全部补跑

| 未完成/缺证据项 | 为什么还算缺口 | 建议优先级 |
|---|---|---|
| 三臂Student及必要复验 | 蒸馏核心问题尚无实际结果 | **最高：恢复验收后接续** |
| 1.7B/4B/8B同协议能力对照 | 只有“资产已齐”陈述，模型/训练/评分不在本地 | 蒸馏首轮后，先1.7B小预算 |
| 训练max_length对照 | 1024/2048来自固定权重推理诊断 | 有长尾/效率需求再独立比较 |
| pilot候选的r32确认 | r16/3M仅dev不能直接推广 | 若要调优化器，单独立匹配控制 |
| CE类权重 | 比例轮明确所有CE权重均1 | 可选诊断，不叫数据比例实验 |
| 精确全局50:50 pair比例 | 已测四档不含精确50:50 | 不因未测就自动追加网格 |
| HS30＋RAG50原20M第二seed | 原预算匹配复验未完成，17M不算补齐 | 暂不优先，保留未决状态 |
| VAST等新原生任务缓存 | 有原始源，不等于许可/标签/划分/转换验收 | 单独完成数据门槛后再入训 |
| 全流程独立seed和规模/数据量曲线 | 续训seed共享R0，池大小与重复曝光未全面控制 | 候选配方确定后做正式确认 |
| 独立适配容量/目标函数对照 | 当前没有全参能力对照或整题归一化目标对照 | 条件触发，不恢复已作废方案 |

**不列为待跑：**R1–R4、D2初轮、任务指数0/0.5/1、20:80等混合、SAR策略、r16/r32正式对照、已完成小池轮、比例四臂、五专家与固定融合、现有资源benchmark。需要时只做新环境最小复现，不重复整网格。

## 7. 恢复操作顺序：先恢复证据链，再恢复算力

以下是**以后在新集群执行的方案**，不是本次已执行内容。涉及下载、安装、写入和启动的操作，均应在确认目标机器和资产后执行；不要在旧目录上原地覆盖。

### G0：先分清“环境没了”还是“数据和权重也没了”

**第一步不是安装包，而是检查持久卷、旧主目录、快照和用户自己的其他备份。** 对所有找回文件先只读清点，确认文件大小、校验和与目录结构，复制到独立恢复目录后再使用。

| 找回情况 | 可以做什么 | 不能声称什么 |
|---|---|---|
| 完整 run、权重、优化器、随机状态和数据游标都在 | 按原训练器机制尝试断点恢复，并检查下一批采样与预算 | 未通过恢复验收前，不声称逐步一致 |
| adapter、训练过的 score、tokenizer/config 都在 | 用同一基座重载，恢复评测；另建 optimizer 开新续训 | 不能称为原训练断点无缝继续 |
| 只找回 merged 模型 | 验证其模型函数，作为新续训 parent | 不能凭空还原原 LoRA 分解或优化器 |
| 只有原始基座和原始数据 | 重建代码和允许范围内的数据，建立新的重建参照；完整历史R0重训须另确认范围 | 不能把新训练结果冒充原 checkpoint |
| 只有本目录 | 保留历史结论，实施“重建式复现” | 不能由分数、hash、路径还原训练权重 |

**G0 交付：** 资产清单，每项记录原路径、恢复路径、大小、SHA256、来源、是否完整，以及“可评测/可新续训/可断点恢复”的级别。不要保存或复制登录凭据到报告。

### G1：重建最小软件环境

1. 先记录新节点的 Linux/架构、驱动、GPU 型号与可见容量、CPU/内存、可用磁盘；不要假设仍有四卡。
2. 用第 3 节中的历史版本作为第一候选，在独立 Python/Conda 环境中安装；不要写入 base，不先升级所有依赖，也不要为了分类任务顺手安装整套视觉/音频依赖。
3. Torch 的 CUDA 构建与驱动兼容必须实际核验；`torch.version.cuda`、`nvidia-smi` 和 `nvcc --version` 是不同层次的信息。本地归档没有证明新集群具备哪一个驱动/toolkit。
4. 先复原后期实际使用的 Transformers + PEFT + SDPA 分类路径。Unsloth 属于 R0 历史环境的一部分，但不是凭名字就必须替换所有后期 worker 的加载路径。
5. 没有完整锁文件时，明确标记“候选恢复环境”。如果历史精确组合无法取得，建立单独兼容环境，记录变化；不可静默替换后仍称完全一致。

在**新集群**完成安装后，可用以下只读诊断核验版本，不在本次本地分析环境中把输出当作集群证据：

```bash
nvidia-smi
python -m pip check
python - <<'PY'
import importlib.metadata as m
import sys
import torch
print('python', sys.version)
for name in ('torch', 'transformers', 'peft', 'datasets', 'huggingface_hub',
             'accelerate', 'pyarrow', 'numpy', 'unsloth', 'unsloth_zoo'):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        print(name, 'NOT INSTALLED')
print('torch CUDA build', torch.version.cuda)
print('CUDA available', torch.cuda.is_available())
print('GPU count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory / 2**30, 'GiB')
PY
```

**G1 通过条件：** 依赖无已知冲突；实际模型能加载；GPU 上前向、分类 CE、反向和一次更新成功；保存重载输出一致。计划多卡时另验 DDP 同步和保存，不用“能 import torch”替代这些检查。

### G2：恢复基座、数据与训练工程

- 原基座能找回就先核对原 revision；找不回时固定新下载 commit，记录与原训练是否可比。
- 需要补原始Tasksource时恢复`default`对应划分；找回D0则核对原过滤/标签/编码证据。默认只为新的Language相关小参照构建允许范围内的训练子池，不因此启动完整历史D0重训；**行数相同不是内容一致的证明**，应核对按行或分组hash。
- 后续的 selected-D0、D1、D2、训练/自然 dev、原题分组、候选顺序、tokenization 与采样 schedule 分别恢复；不能把小池、原始大池和不同阶段处理缓存混成一个 D0。
- 在原题/文档/标题/源 prompt 层划分和去污染，再展开候选。`0=匹配`、`1=不匹配` 的 pair 标签与原生任务类别分别保存。
- 补回真正的训练器、模型加载器、PairEncoder、Engine、scorer、schedule 生成器和 worker；报告脚本并不包含这些模块的全部实现。
- 所有重建输出写新目录，记录原始/处理后条数、原题数、pair 数、非 padding token、重复曝光与截断率。

**G2 通过条件：** 独立 manifest 可复核来源与划分；无 gold 输入泄漏；模型 logits 为 batch×2；LoRA 与 score 确实更新；训练和 Engine 渲染一致。缺少原始转换代码时，给数据版本新名字，不宣称还原了原缓存。

### G3：先恢复评测标尺

1. 固定原 Decision Index kit commit、suite 版本、Engine、归一化和 scorer；本目录没有的部分需找回或重新构建。
2. Language catalog ID 固定为 `{11,12,29,38,39,40,41,42,59}`，目标为 **31,715 请求、九项齐全**。保留关联请求组；不要编造 `--area language` 参数。
3. 不截断正式评测输入，不删候选，不按 benchmark 名做答案规则。超过真实上下文的请求按协议记 Unsupported，报告 coverage 与失败数。
4. 先小样本契约验证，再完整 Language；若只有分类 checkpoint，不能直接交给加载 CausalLM 的内置 Engine。
5. 若旧逐题预测能找回，用同一冻结请求比较概率与决策；容差和数值差异先登记。若只能找回汇总分数，只能做汇总核对，不能证明逐题完全复现。

**G3 通过条件：** 九项、计分版本、request ID 和 coverage 对齐，错误/Unsupported 可解释；“Language 完整”与“全榜提交 complete”分开描述。

### G4：恢复一个主参照，而不是整个历史网格

- 能找回历史 LR32 或其他已验证 parent：先重评该权重，再确定新实验的共同起点。
- 找不到训练权重：默认从真实基座训练一个新的`rebuild-reference-*`，仅使用当前允许的Language相关训练池；它不是历史R0，也不要求复现历史LR32分数。不要从随机分类头直接声称接续了原R0。
- 对精确配方仍缺配置的阶段，先补证据；只能重建时用 `rebuild-*` 新实验 ID，明确与历史分数的比较限制。
- 只有基座、不完整的 adapter 或丢失的分类头，都不算原 parent 已恢复。

**G4 通过条件：** 一个可加载、可评分、可追溯、可新续训的参照 checkpoint；若要断点恢复，另需 optimizer/scheduler/RNG/游标验收。

**完整R0重建的范围关卡：**[后续快筛约定](experiment-timebox-2h.md)明确“不再默认使用完整181万条D0”，只用Language相关任务，所有候选训练池合计最多50万最终pair。资产丢失不自动豁免这项约束：

1. **未获得额外范围确认时：**按Language相关、合计≤50万pair建立新参照，重新冻结其对照分数；与历史R0/LR32只作带限制的历史比较。
2. **确需复现完整历史R0时：**未来启动前先请用户明确允许完整历史训练池及单独计算预算，再立`rebuild-r0-full-*`。恢复的wall time单列，并不等于已获得全池训练许可。

本次不启动训练，因此无需为了交付本文先作这个决定；执行者不得替用户默认选择完整历史R0重训。

## 8. 后续实验规划：先把未回答的问题做完

**本节均为【新建议】，不是已经执行或本轮授权启动的任务。** 排序承接已有路线中“Teacher/Student在前、规模在后”的安排；不续接已作废方案，不自动恢复已经暂停的采样/融合网格。

### 8.1 推荐队列与停机条件

| 顺序 | 工作 | 回答的问题 | 进入条件与交付 |
|---:|---|---|---|
| P0 | 第7节G0–G4：资产救回、工程/评测恢复 | 能否可信接续，还是必须重建 | 一个有明确parent和版本的有效参照；通过前不启动新方向 |
| P1 | Teacher资产/标签验收 | 历史标注是否可复用，软标签是否合法且有信息 | 真实标签本体、ready、问题映射、Teacher provenance和质量小审计 |
| P2 | HARD / SOFT / HYBRID匹配首轮 | Teacher监督是否比同批人工标签更有用 | 同parent、同样本/候选、同预算和seed的三臂结果 |
| P3 | 条件通过的候选＋HARD匹配复验 | 收益能否跨续训seed保留，是否以牺牲其他任务换来 | 先补第二seed；有一致信号再到至少3seed，否则停止扩算力 |
| P4 | 1.7B小预算规模迁移，必要时再4B | 当前配方在更大同家族基座上能否改善 | 先重扫少量LR/batch并固定；不先穷举所有尺寸 |
| P5 | 按错误类型选一个：新原生数据/长度/目标函数 | 剩余瓶颈来自哪一环 | 分项证据触发，每轮只改一个因素 |

**恢复可能比新实验更耗时，不能塞进一个“两小时实验轮”假装完成。** 先交付环境/资产验收，再为GPU轮独立计时。

### 8.2 P1：Teacher标签验收，先找回、后决定是否重标

至少核对以下内容：

1. **模型身份**：KnowLine31B的精确repo ID、revision、权重文件hash、tokenizer、原生readout、attention和作者推理实现。现在本地没有这些信息，不能凭模型名猜加载类。
2. **完成证据**：找回`teacher-labels-ready.json`及它指向的实际文件；9998 pair的完成记录、此前准备的4166原生问题、实际成功问题数及对应关系需要原始manifest确认，不能只验一个数字相等。
3. **训练数据边界**：每题能追溯source/split/question ID/候选ID；全部来自允许训练的划分；Teacher不能标注公开评测gold来反哺Student。
4. **概率与候选**：完整保留候选、有限非负概率、正确维度和归一化；明确noul的Yes语义。多标签/多真候选不能强制成唯一正确的多分类。
5. **覆盖与失败**：缺失、超长、重试、解析失败和unsupported逐项记录；三臂若只能用共同成功子集，应冻结同一个子集并报告排除量，不能各臂按结果过滤。
6. **Teacher信号**：按原生任务审计Teacher与gold的一致性、entropy、置信度和明显错误，保留错误而不是只挑Teacher答对题。较强榜单总分并不保证这批标签或此维度更好。
7. **可加载与隔离**：Teacher用独立环境；先小批与作者推理方法对齐，再决定补缺标注或重做。找回完整标签则不必为“恢复环境”重复全部Teacher推理。

**P1停止条件：**缺Teacher身份/完整候选映射、无法确认训练划分、标签概率不合法，或原生概率与Student任务语义不能对齐。此时先修数据/接口，不启动SOFT/HYBRID。

### 8.3 P2：三臂Student的最小公平设计

| 项目 | 首轮建议 |
|---|---|
| 共同起点 | 优先复用历史已准备三臂的准确parent；找不到原配置则新建并声明。建议使用同一恢复后的R0-r32或同一重建参照，不让三臂加载不同checkpoint |
| Student | Qwen3-0.6B，固定causal、二分类score、LoRA r32/alpha64、同一渲染与训练长度 |
| 输入流 | 同一批原始题、完整候选和顺序，同一采样计划、实际非padding token预算、有效batch与seed；D0回放如保留，三臂一致且单列监督来源 |
| HARD | 原始人工监督，不能复用另一预算/另一池的历史分数代替本轮控制 |
| SOFT | 同题Teacher软监督，正确对齐相同输出空间 |
| HYBRID | 同题人工＋Teacher监督；权重在看正式评分前冻结 |
| 预算 | 先基于恢复后的完整更新pilot统一选小预算；3M token可作起点建议，**不是已找回的原Student配置**。三臂同预算，记录重复率和唯一样本数 |
| 学习率 | 先沿用当前参照的已验证计划；不同时叠加仅r16/3M pilot得到的LR赢家 |
| 选择与评测 | 自然pair、原生任务dev、D0迁移代理用于判断；完整31,715请求Language用于报告；保存九项与coverage |

**监督对齐是启动前必须明确的实现决定：**Teacher的原生候选分布不等于每个pair独立的“匹配概率”。如果用完整候选上的Student margin归一化分布，三臂都在该空间使用one-hot/Teacher/混合target；如果用独立Bernoulli pair监督，则必须证明Teacher给出的就是该语义。不能直接把候选概率灌进原二分类类别ID，也不能把概率当原始logits。

对于确实处于同一分布空间的三臂，可预先冻结为：

```text
HARD:   H(y, p_student)
SOFT:   KL(q_teacher || p_student)
HYBRID: (1 - λ) H(y, p_student) + λ KL(q_teacher || p_student)
```

【新建议】只有现成概率、没有原始Teacher logits时首轮温度固定1，不臆造高温logits；HYBRID可先λ=0.5。若输出空间/目标从原pair CE改成整题loss，必须明确登记为新HARD匹配基线，不能将其与旧CE基线的差异全部归因于Teacher。以上是候选设计，仍需P1语义验收，不预先宣称已实现或最优。

**首轮禁止混加：**POS2正pair重采样、新LR、训练截断、r64、新任务池、融合。否则即使涨分，也无法回答Teacher监督的价值。

### 8.4 P3：晋级规则和负结果处理

- 在正式训练前写清主指标、任务保护线、自然pair两类recall、D0迁移约束、允许的coverage差异与时间预算；能找回原预注册阈值则沿用，找不回则重新冻结，不能看结果后改。
- 任一候选只提高公开Language、同时明显伤害自然dev或关键任务，不直接晋级；比例首轮已经出现过这种情况。
- 有可解释正信号，先增加一个匹配续训seed；一致后再补到至少3seed。候选与HARD同时重跑同预算，不只给赢家加训练。
- 共享R0的多seed仍只覆盖续训随机性；若要对正式配方给更强结论，另从基座开始做独立全流程确认。
- 有逐题预测时按文档/题组做配对分析，并报告选择偏差；历史反复复用的dev不能变成“独立最终测试”。可从允许的train来源按分组另留未参与选择的确认集，固定后不反复调参。
- 无信号则记录三臂负结果，转规模或数据瓶颈诊断；不自动追加λ/温度/Teacher网格维持占卡。

### 8.5 P4/P5：规模、数据和目标函数的顺序

**规模：**先0.6B对1.7B，保持已冻结数据、输出语义、token预算和评测；两档可按相同小调优预算选LR/batch。固定token的质量比较与固定GPU-hours的效率比较分别报告。每档从其自身真实基座出发，不把0.6B权重塞入大模型；只有1.7B有稳定收益或明确容量迹象，再进4B，8B留后。旧“资产已齐”不免除新节点的文件校验。

**数据：**优先看ContractNLI、ANLI、VAST、NLI4CT的自然dev与错误类型。原始源已在本地，不代表已可入训；先完成许可、标签、文档分组和去污染。FinEntity仍不能从979评测文档切train。新数据与困难负例分别试，不同时改数据配比和readout。

**长度：**若错误集中于长文，可独立比较训练cap2048与4096，模型/采样/实际token预算固定，报告长桶效果和截断量；正式评测全部不截断。1024留作更激进的速度候选，不因总体F1近似就默认裁掉所有长上下文。

**目标函数：**若候选竞争/校准而非输入理解成为主要错误，再比较pair CE与整题候选归一化目标，固定head和数据。CE类权重、pair重采样、原生类别平衡是三个不同实验。架构/readout修改另立编号，不算纯数据或规模收益。

### 8.6 每轮交付与预算约束

历史后续轮的约束包括Language范围、两小时交付、最终训练pair合计不超过50万、镜像下载和旧结果保留。它们是历史项目约束，不是本次执行授权；【新建议】恢复后的新GPU轮继续作为保守默认，并在轮前按新硬件实测确认：

- 预算表同时写**唯一池pair、跨候选保守合计、实际重复出现次数、实际token和原题draw**；不通过重复采样把小池包装成大规模新数据。
- 计划时间含准备、加载、训练、dev、完整公开评测、审计和备份；下载与环境重建单独计时。
- 三臂可各用一张卡，第四张卡不必强行安排任务；完整评测在明确资源释放后调度，不因旧handoff就假定GPU空闲。
- 先冻结所有组共同可完成的token预算，不临近截止只缩某一候选；未完成就如实标记，不能把应急短跑当计划预算结果。
- 一轮结束交付：可加载checkpoint、原始配置/版本/hash、数据manifest、采样与实际曝光回执、自然dev与九项分数、coverage/errors/unsupported、完整运行时长、失败记录和下一步决策。

不要承诺未经新节点测量的固定小时数。历史完整Language评测在不同worker/阶段有8.5分钟和约31–35分钟等差异，不能挑最快旧值保证新的端到端工期。

## 9. 已有资源实测：恢复后不要再踩同一批坑

### 9.1 Batch 不应按“显存用满”来选

[2026-10-28 batch 实测](batch-benchmark-20261028/results.md)采用真实 schedule 窗口，microbatch≤128 时有效 batch 固定 128：

| 真实混长 microbatch | 有效 tokens/s | 峰值 allocated | padding | 结果 |
|---:|---:|---:|---:|---|
| 16 | 8,728 | 11.98 GiB | 20.1% | 通过 |
| 32 | 10,347 | 22.59 GiB | 37.2% | 本窗口最快 |
| 64 | 9,036 | 43.89 GiB | 57.8% | 更占显存但更慢 |
| 128 | — | — | — | OOM |

这是短窗口测量，不是永恒最优配置。全 4096、无 padding 的压力测试中 microbatch128 成功，也不能推翻真实混长128 OOM。microbatch256 改变了有效 batch，只是容量探测。

### 9.2 Checkpointing 与长度必须一起看

来自[长度研究](checkpoint-length-scaling-20261029/RESEARCH-REPORT.md)及[机器汇总](checkpoint-length-scaling-20261029/summary.json)。历史设备为 H800 NVL，CUDA 可见约 93.09 GiB；Qwen3-0.6B、LoRA r16、BF16、causal SDPA，物理 microbatch16、累计两次：

| 每条长度 | Checkpoint On allocated | Off allocated | 解释 |
|---:|---:|---:|---|
| 1,024 | 3.66 GiB | 32.56 GiB | 整齐、无 padding 输入 |
| 2,048 | 6.07 GiB | 63.85 GiB | On 有效吞吐约29.3k，Off约40.0k tokens/s |
| 4,096 | 10.90 GiB | OOM | 不可用短输入经验直接外推 |
| 8,192 | 20.55 GiB | OOM | On reserved峰值约23.00 GiB |

关闭 checkpointing 的整齐输入搜索中，2944 通过、3072 OOM；真实混长2527–2807、microbatch16关闭时也会 OOM。保留真实 mask、降低到 microbatch8 后可通过；**不能把 mask 改成全1当作正式训练优化**。

已有定位表明，同 shape 的真实 padding mask 路径发生 K/V 展开与显式4D mask；microbatch8 下显存由44.29增至50.05 GiB。仅改 allocator 为 expandable_segments 未解决原 OOM，不能把问题简单归结为碎片。

**适用范围：** 长度阶梯共有40个主测量、9个诊断单元，另有 attention 路径追踪；该组前后向测量 **optimizer 更新为0、权重保存为0**。不能把它当作完整 optimizer-step 训练吞吐、LR32 显存、全参显存或跨模型通用公式。新 GPU、rank、长度、attention 或 worker 变动后要重测。

### 9.3 “1024 是甜点”不等于正式评测可以截断

[长度诊断](length-label-benchmark-20261029/BENCHMARK-REPORT.md)是固定权重、1667 pair 开发面板的推理干预：

- full 加权 Macro-F1：90.6331%。
- cap1024：90.4512%，下降约0.182个百分点。
- cap2048：90.6118%，下降约0.021个百分点。
- 原长1024–2047桶截到1024，Macro-F1 从86.200%降至74.086%，长尾损失被总体平均掩盖。

因此1024/2048只能作为**训练长度或内部速度实验的候选**。本轮尚未证明不同训练 max_length 的优劣；正式 Language 仍不截断。这些 F1 也不能和公开 Language skill 直接比较。

### 9.4 资源与时间预算的写法

- 可用几张卡就按几张安排；四卡方案是恢复后的候选，不是当前保证。
- 0.6B 的独立 seed/实验通常可一张卡一组，先测真实混长、真实 mask 与完整 optimizer-step；不为了四卡都忙而重复同一实验。
- 固定有效 batch：`物理 microbatch × 梯度累计 × DDP world_size`。改变其一时记录学习率、更新次数和 token 预算是否联动。
- 训练工期：`剩余实际非 padding tokens ÷ 同 worker 实测有效 tokens/s`；总工期另加加载、dev、31,715请求评测、审计和备份，不用纯前后向数值估算一整轮。
- 规模晋级先限于本项目后续实际准备过的1.7B/4B/8B候选，以新节点显存、实际加载路径和完整训练pilot决定资源；不引用已作废方案的容量与工期估算。

## 10. 恢复后必须补的归档，避免再次只剩分数

以下是【新建议】的交付结构，不是声称当前仓库已经存在：

```text
recovery-YYYYMMDD/                  # 与历史目录分开，不覆盖
  RECOVERY-MANIFEST.json           # 找回/重建/缺失、原路径→新路径、hash
  environment/
    system.txt                    # OS、驱动、GPU、容量，不含凭据
    pip-freeze.txt / conda-export.yml
    wheel-or-image-manifest.json  # 来源、构建版本、镜像digest
  code/                           # 真正可运行的训练、加载、数据和评测代码
  provenance/
    models.json                   # base/adapter/score/Teacher revision与hash
    datasets.json                 # 来源、许可、split、过滤、分组和样本映射
    evaluation.json               # kit commit、suite/request hash、Engine版本
  experiments/<new-run-id>/
    config.json / parent.json
    sampling-plan/ / actual-exposure/
    checkpoints/                  # adapter+score或完整权重
    resume-state/                 # optimizer/scheduler/RNG/trainer/data游标
    predictions/ / metrics/ / logs/
    artifact-checksums.json
    RUN-REPORT.md
```

**最低备份优先级：**

1. 基座revision＋tokenizer/config；每个关键parent的adapter和训练过的score。
2. 可执行代码与环境锁定、真实运行配置、来源与去污染/划分证据。
3. 数据选择映射、采样计划与worker成功提交后的曝光回执；可依法重建的原始数据可用revision/hash引用，但本地派生选择不能只留摘要。
4. 原始预测、计分版本和汇总指标；需要精确恢复的任务另保存完整resume state。
5. Teacher原生概率、候选映射、ready与身份/推理定义；不能只留“已输出9998对”的展示文案。

【新建议】关键阶段完成后，将允许备份的产物复制到独立持久存储，并在另一路径做一次真实重载验收。是否上传/公开另行决定；不自动上传私有数据、凭据或受限制上下文。

### 实验台账最低字段

```text
run_id / state / rebuilt_or_restored
base_repo / base_revision / parent_checkpoint / parent_hash
code_commit / environment_id / attention / head / precision
LoRA_rank / LoRA_alpha / target_modules / modules_to_save
data_version / split_hash / source_group_policy / sampling_plan_hash
unique_questions / unique_pairs / pair_draws / committed_nonpadding_tokens
stage_tokens / cumulative_exposure_tokens / actual_training_compute_tokens
seed / microbatch / accumulation / world_size / LR / scheduler
dev_population / native_metrics / pair_metrics / D0_transfer_metrics
Language_skill / coverage / requests / errors / unsupported / nine_skills
GPU_hours / end_to_end_wall_time / inference_time_scope
checkpoint_path / resume_state_path / predictions_path / artifact_hashes
decision / limitations / next_action
```

`state`建议明确区分`planned / assets_ready / smoke_passed / training / evaluating / completed / failed / incomplete / unverified_legacy`。只有文件路径、脚本生成或历史running文字都不等于已启动/完成。

## 11. 本次文档做了什么验证，仍有哪些限制

### 已实际执行的本地验证

- 读取当前Git提交、工作树、原始报告、机器JSON/CSV和看板生成逻辑；**文档开始时工作树干净**。
- 重新检查当前工作树：没有模型权重后缀文件、Arrow/NPZ采样缓存，以及关键模型/run/训练包/Teacher目录。
- 14份manifest原始数据文件大小和SHA256全部吻合；两份ZIP CRC检查通过。
- 已核对91＋4＋9＝104的阶段边界、103个历史实体checkpoint路径＋1个虚拟集成、看板104条记录及主线均值/单次最高、比例与融合分数。
- 本地已有资源统计单元测试：`test_benchmark.py`＋`test_scaling.py`，**7/7通过**；没有GPU执行。
- 本地看板数据/逻辑测试：`node --test reports/experiment-dashboard/test_dashboard.cjs`，**5/5通过**；未启用可选DOM依赖，不声称本次运行了额外DOM或浏览器测试。
- 137份归档JSON均可解析；文档3个Bash代码块和1个内嵌Python片段只做语法解析，未执行其中安装/下载/集群诊断命令。
- 最终交付前对本文件的本地Markdown链接、关键数值与Git差异作检查；只有本次新增文档，不改历史实验记录。

可重复的现有CPU测试命令（仓库根目录执行）：

```bash
(
  cd reports/checkpoint-length-scaling-20261029
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
    python -m unittest -v test_benchmark.py test_scaling.py
)
node --test reports/experiment-dashboard/test_dashboard.cjs
```

**不要原样运行旧`verify_pair_ratio_update.py`来验证恢复成功：**它还检查旧绝对checkpoint路径，并尝试写一个必须不存在的历史verification文件；本地已有验证快照，脚本并非可在缺失资产时直接重跑的无副作用检查。

### 本次没有验证或执行

- 没有访问旧集群，因此不能判断旧卷是否可救、进程最后状态或丢失原因。
- 没有安装候选训练环境，没有核实新GPU/驱动兼容，没有恢复/加载任何模型。
- 没有重新运行训练、Teacher标注、Student蒸馏、GPU资源测试或31,715请求官方评测。
- 没有重算旧权重hash、逐题预测一致性、统计区间或所有原始数据的语义/去污染。
- 外部官方文档仅用于核对Hub下载语法与CUDA可用性检查方法，不用于推断过去装了什么包；精确环境仍以上述本地证据及后续真实安装验收为准。

## 12. 最后给出一条可执行的恢复决策

**先找回R0/LR32权重＋score、原训练/评测代码和数据映射 → 建最小匹配环境 → 冻结0.3评测标尺 → 重评一个可信参照 → 核验既有Teacher软标签 → 完成HARD/SOFT/HYBRID三臂 → 有信号才补seed，再做1.7B规模迁移。**

若找不回权重和核心代码，就改为**有版本记录的重建实验链**：从真实基座重建分类与评测，在Language相关、合计≤50万pair的默认范围内训练新的`rebuild-reference-*`，再继续上述未完成问题。只有用户另行明确允许完整历史训练范围，才单独计划重训历史R0。历史104个方案的报告继续保留为历史证据，既不假装权重还在，也不为重建环境把全部历史搜索再跑一遍。
