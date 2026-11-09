# Jev 项目工作量明细台账

核查日期：2026-11-02。依据本地历史归档与指定集群 `/home/dev/assumeengage` 的只读记录。

## 统计原则

- 按研究问题、工程能力和实际交付分项；分项是工作清单，不是同等难度的工作次数或人天。
- 已完成训练、已实现但未跑通的模块、执行中的阶段、负结果和纯方案分别标注。
- 同一工作的代码、日志、检查点、备份和图表相互印证，不重复计算为新实验；第三方原始代码不算自行研发。
- 数据池大小、独立原题、候选匹配对、重复训练曝光、评分记录与 GPU 工时分开统计。
- 本轮没有启动、续跑、停止或修改远端任务，没有额外测试、烟测或文件哈希。台账只盘点已存在工作。

## 一、可直接量化的工作量

| 口径 | 数量 | 解释与依据 |
|---|---:|---|
| 完整参评方案 | 114 | 历史 Language 104＋后期跨领域 10；含融合、概率集成和 4 个工程验证终点，不等于独立训练次数 |
| 仅开发集的 batch／LR 短训 | 8 | 独立于上述完整评测；每个新增 300 万 token |
| 显存／速度资源单元 | 49 | 前后向探针，不更新权重；不计为新模型 |
| 真实 batch／长尾吞吐条件 | 9 | 7 个完成、2 个 OOM；含实际 optimizer 更新的短测 |
| 后期正式全量训练 | 5 | 四专家＋一个 MIXED_ALL；均有 `complete.json` |
| 后期正式训练累计 pair 曝光 | 约 3,634.74 万次 | 四专家合计与混合模型各见一次相同四科池，不是新增独立数据量 |
| 后期正式训练累计有效 token | 约 80.84 亿 | 五份完成记录合计 8,084,106,538 |
| 后期正式优化器更新 | 约 28.40 万次 | 五份完成记录合计 283,967 |
| 五训练任务时长之和 | 27.53 小时 | 各模型 `elapsed_seconds` 相加，不是整条流水线端到端墙钟；均记录为四卡 |
| 后期实际 OOM 回退 | 12 次 | 五份训练完成记录中的 `oom_retries` 合计；属于同批训练的恢复，不新增实验数 |
| 后期权重融合 | 5 种 | AVG、Task Arithmetic、TIES、DARE＋TA、DARE＋TIES；均有完成记录 |
| 后期完整公开评分记录 | 约 110.09 万条 | 10×110,087；是模型×请求记录，不是 110 万独立题目 |
| 当前 HARD 续训 | 执行中 | 09:24 UTC 快照：43,440／141,983 步、5,560,320 pair、1,236,846,499 有效 token；未完成，不并入已完成训练总量 |
| Teacher 大模型资产准备 | 部分完成 | 约 54.73／62.58 GB，13 片权重中 9 片为完整文件；无 Teacher／标签 READY |

已完成训练与评测原始字段保存于 [只读状态快照](../../../03-证据与图表/08-综合汇总与历史记录/cluster-work-inventory-20261102/remote-evidence.json)，代码功能位置保存于 [实现摘录](../../../03-证据与图表/08-综合汇总与历史记录/cluster-work-inventory-20261102/implementation-evidence.json)。上述数值是该次读取快照，不保证持续等于正在执行任务的实时进度。

## 二、前期研究、验证和探索工作

本节的证据主要在本地历史归档。它们已经出现在主工作记录，保留在全量台账以免只补新集群工程后漏掉前期研究。

| 编号 | 独立工作项 | 状态与产出 | 证据 |
|---|---|---|---|
| R01 | 决策模型路线与读出方案调研 | 比较候选匹配、字母 logits、255 路读出、复杂 head，以及训练／推理成本；属于调研，不是全部复现 | `jev.md`、[research/top-models/](../../../01-研究结论/04-数据与评测/top-models) |
| R02 | 榜单规则与领域边界核查 | 区分 Public／Language Skill／Full，确认 Games 不独立计分及五领域口径 | [research/language-axis/audit.md](../../../01-研究结论/02-Language/language-axis/audit.md)、`language_roadmap.md` |
| R03 | 社区训练来源可获取性研究 | 逐方案区分公开成品、来源清单、转换器、未公开完整池和仅评测资产 | `data_sources.md`、[research/notes-mapika-bobcat-jebadiah.md](../../../01-研究结论/04-数据与评测/notes-mapika-bobcat-jebadiah.md) |
| R04 | 目标任务 train 与标签核查 | 核实 Language 九项任务的真实输入、来源与可用训练划分 | [research/language-axis/manifest.json](../../../03-证据与图表/02-Language/language-axis/manifest.json)、`audit.md` |
| R05 | 初版数据清洗与标签映射 | Tasksource 处理后训练约 181.38 万对；验证、内部测试独立登记 | [reports/language-baseline-20261027.md](../../../01-研究结论/02-Language/language-baseline-20261027.md) |
| R06 | 分类训练链路验证 | 32 步四卡试跑；确认 LoRA／分类头更新、保存和重载，再进入正式训练 | 同上 |
| R07 | 基线完整训练与内外部对照 | 内部准确率 82.40%，外部 Language 17.74，明确两种能力指标差异 | 同上 |
| R08 | 数据×注意力受控研究 | 同新增预算、3 个匹配续训 seed，分离补数据与直接双向化的影响 | [reports/language-understanding-study-20261029.md](../../../01-研究结论/02-Language/language-understanding-study-20261029.md) |
| R09 | 双向注意力实现源码核查 | 检查 config、模块级因果开关与 mask／backend 路径，区分 0.6B 与 hybrid 大模型结构 | [research/attention/audit.md](../../../01-研究结论/05-Attention与效率/attention/audit.md) |
| R10 | 目标数据扩充与任务取舍分析 | ANLI、HellaSwag 后再接 ACOS、讽刺与 RAGTruth；总分与逐任务变化分别分析 | Language 阶段报告、`dev-metrics.csv` |
| R11 | 原题级任务采样研究 | 指数 0／0.5／1，对照原控制；按验证指标选择而非只选公开最高分 | 同上 |
| R12 | 通用回放与目标数据混合比例研究 | 实际 token 比例 80:20／50:50／20:80；分析收益及 seed 波动 | 同上 |
| R13 | 原生讽刺类别平衡与配额研究 | 单因素、组合与任务交互分开比较，不将 pair 平衡冒充原生类别平衡 | 同上 |
| R14 | 联合采样配方确认 | 重新训练验证组合，不将单因素增益直接相加；4 seed rank16 均值 33.74 | 同上 |
| R15 | LoRA rank16／32 容量验证 | 保留已训练函数起点与分类头，4 匹配 seed；rank32 为 35.05±0.96 | 同上 |
| R16 | 小池快筛与 rank64 探索 | 39 个正常终点；筛任务配额、回放、讽刺与容量，不稳定方向保留负结果 | `runs-through-pair-ratio.csv`、阶段报告 |
| R17 | 初筛信号复验与归因 | 对采样指数等补匹配 seed、不同面板和迁移代理；未通过者不晋级 | 同上 |
| R18 | 正负匹配比例独立消融 | 4 个固定预算终点，区分匹配召回、不匹配召回、自然 dev 与公开分数 | `PAIR-RATIO-ADDENDUM.md` |
| R19 | 实际训练曝光核对 | 按成功提交更新检查任务×标签、非 padding token 与源 pair multiset | 同上及融合 `ACTUAL-EXPOSURE-AUDIT.json` |
| R20 | 固定权重的输入长度诊断 | 1,667 对、六种长度；总体与原长桶分开观察，不将截断诊断用作正式评测 | `length-label-benchmark-20261029/BENCHMARK-REPORT.md` |
| R21 | 静态池、采样与 token 比例研究 | 解释原题类别、候选结构、采样重复及正类 token 份额不同 | 同目录 `exposure-ratios.md`、数据清单 |
| R22 | 真实 batch 与长尾吞吐研究 | 固定有效 batch；9 条件，证明更大物理 batch 不必更快 | `batch-benchmark-20261028/results.json` |
| R23 | Batch×学习率短预算研究 | 4 设置×2 seed，8 个 dev-only 训练；限制短预算外推 | `pilot-dev-only-runs.csv`、`STUDY-REVIEW.md` |
| R24 | 激活重算与长度阶梯 | 49 资源单元，量化显存／耗时与 OOM 边界 | `checkpoint-length-scaling-20261029/RESEARCH-REPORT.md` |
| R25 | Padding／GQA 路径定位 | 同 shape 改 mask；追踪 28 层实际调用，定位 K／V 展开与额外开销 | 同目录 `attention-dispatch.json` |
| R26 | 梯度累计与重算数值一致性 | 对照重算 On／Off、物理 batch 与累计拆分，并保留 BF16 数值差异 | [research/lu-batch-lr-pilot-20261029/STUDY-REVIEW.md](../../../01-研究结论/02-Language/lu-batch-lr-pilot-20261029/STUDY-REVIEW.md) |
| R27 | 显存估算关系与配置工具 | 拟合物理 batch×补齐长度的显存关系，形成估算脚本；不外推为普适公式 | `SCALING-ANALYSIS.md`、`estimate_memory.py` |
| R28 | Language 数据家族专科研究 | 五家族专家、同总曝光混合控制与本族 dev 交叉比较 | [research/lu-specialist-fusion-20261029T152144Z/](../../03-专科与融合/lu-specialist-fusion-20261029T152144Z) |
| R29 | 首轮融合数学与保存精度核查 | 有效 delta＋已训练 head；对比 AVG、TIES、输出概率集成与部署成本 | 同目录融合报告与 `MERGE-RESULT.json` |
| R30 | 研究负结果与实验保护线整理 | 区分未复现、局部涨分、旧任务退化和工程短训，避免最高分筛选偏差 | Language 汇总、各轮复核 |

## 三、后期训练、评分与报告交付

以下远端路径均相对 `/home/dev/assumeengage/`。已完成实验属于前述 114 个参评方案的一部分，不另加一次。

| 编号 | 独立工作项 | 状态与产出 | 证据 |
|---|---|---|---|
| T01 | 建立可公平比较的共同初始化 | 同一 Qwen3-0.6B、rank32／alpha64、同一分类头与类别配置 | [models/Qwen3-0.6B-common-pair-init-r32/config.json](../../../03-证据与图表/03-专科与融合/共同初始化/Qwen3-0.6B-common-pair-init-r32/config.json)、初始化脚本 |
| T02 | 知识专科完整一轮训练 | 完成约 307.35 万对、5.24 亿 token、24,012 更新 | `runs/five-specialists-20261030/SPECIALIST_KNOWLEDGE/complete.json` |
| T03 | 检索专科完整一轮训练 | 完成约 1,104.76 万对、15.72 亿 token、86,310 更新 | `SPECIALIST_RETRIEVAL/complete.json` |
| T04 | 工具专科完整一轮训练 | 完成约 157.97 万对、6.90 亿 token、12,342 更新 | `SPECIALIST_TOOLS/complete.json` |
| T05 | 艺术专科完整一轮训练 | 完成约 247.29 万对、12.56 亿 token、19,320 更新 | `SPECIALIST_ARTS/complete.json` |
| T06 | 四科混合公平控制 | 完成同池全部约 1,817.37 万对、40.42 亿 token、141,983 更新；曝光量等于四专家之和 | `MIXED_ALL/complete.json` |
| T07 | 跨领域五种有效权重融合 | 五种方法均导出完整单模型；包含已训练 head，不直接平均 LoRA A／B | `runs/five-specialists-20261030/merges/*/complete.json` |
| T08 | 专科优势与跨域迁移分析 | 四专家本方向与 MIXED_ALL 同方向对比；局部优势与综合分开解释 | `report/five-area-results.csv`、`differences-vs-mixed.json` |
| E01 | 分类模型接入公开评分 | 实现 logits(match)−logits(non-match)，还原原题候选概率与原始答案键 | `scripts/run-formal-evaluation.py:84–178` |
| E02 | 完整候选与长输入处理 | 不因候选多或输入长静默删除／截断；超能力标记 unsupported | 同文件 `margins_for_request`、worker 状态分支 |
| E03 | 四卡分片与请求级进度 | 每模型四 worker；保存逐请求输出、进度和完成标记 | 同文件 `worker`、`main` |
| E04 | 缺失请求与计分范围校验 | 缺请求拒绝完成；保留无 gold 辅助字段的正式计分规则 | 同文件 `score_model:249–298` |
| E05 | 部署官方计分逻辑并连接自定义模型 | 对接冻结公开面板与官方任务计分函数，不将内部准确率充当 Index | 同文件 `score_model`，`src/decision-index/` |
| E06 | 十方案完整公开评测 | 各 110,087 请求、37 项，完成记录均覆盖 100%；不称私测 Full | `runs/five-specialists-20261030/evaluation/*/complete.json` |
| E07 | 融合与混合的自动比较报告 | 产出逐领域 CSV、相对混合差值 JSON、专科与融合雷达 SVG | `scripts/run-formal-evaluation.py:302–361`、`report/` |
| E08 | 蒸馏评分与差值图表模块 | 已实现 baseline＋三臂复用评分、37 项差值热力图、柱图、五领域表与 Teacher 成本；三臂尚未完成，未产生成绩 | `scripts/run-distillation-evaluation.py` |

## 四、Teacher 蒸馏与下载恢复工程

| 编号 | 独立工作项 | 状态与产出 | 证据 |
|---|---|---|---|
| K01 | Teacher 选型与原生概率协议研究 | 固定 KnowLine 31B，核对原生 read-twice、标签 token 与温度折入方式 | `config/experiment-four-domain-distillation.json`、本地 Teacher 证据报告 |
| K02 | Teacher 与 Student 环境隔离 | Teacher Transformers 5.12.1，复用既有 Torch；环境 READY，Student 环境未改；不等于已加载完整 Teacher | `teacher-assets/ENVIRONMENT_READY.json` |
| K03 | 大模型资产清单与就绪门槛 | 固定模型版本及 29 个资产的大小／类型、进度和完整性门槛；不运行额外哈希 | `teacher-assets/ASSET_MANIFEST.json`、`ASSET_PROGRESS.json` |
| K04 | 磁盘不足条件下的独立资产承载 | Teacher 使用专用承载位置，经项目模型入口引用，保留既有成果 | 同上 `actual_model_path`；本地蒸馏 `EXECUTION.md` |
| K05 | 国内下载链路与分片保留 | 已实际切换，保留下载分片和 HARD 进程，切换回执记录了续传增长 | `source-switch-domestic-20261102/SWITCH_RECEIPT.json` |
| K06 | HTTP 分段提交错误修复 | 完整响应通过范围检查后才追加提交，避免失败响应将部分内容误当成功 | `RANGE_COMMIT_FIX_RECEIPT.json`；`prepare-knowline-teacher.py:211–263` |
| K07 | 下载链路再次切换与无损交接 | 已保留完整权重、部分分片及 HARD 原进程；后续仍有网络阻断，不称下载成功 | `OFFICIAL_PROXY_SWITCH_RECEIPT.json` |
| K08 | 同可见文本的 Teacher 请求构造 | 已实现 Student 实际裁剪文本重建，再转原生独立 yes／no；避免 Teacher 获得额外上下文 | `label-knowline-pairs.py:208–256,301–386` |
| K09 | 原生读两遍与温度语义保护 | 已实现短输入 read-twice、标签概率顺序和已折入温度／softcap 检查；无完整 Teacher GPU 执行证据 | 同文件配置与 `Renderer`、worker 模型加载分支 |
| K10 | 大规模标注的有界 CPU 内存与预取 | 已实现 512-pair 块、单块预取、长度微批与内存映射概率文件；尚未实际完成标注 | 同文件 `RENDER_BLOCK_PAIRS`、`run_worker` |
| K11 | 三 worker 切分与稳定全局索引 | 已实现 `global_index=rank+local_index×workers` 和各域覆盖检查 | 同文件 `rank_length`、`common_identity`、`run_merge` |
| K12 | 标注断点、持久化和并发保护 | 已实现 NaN 待处理位、逐 worker 身份、文件锁、flush／fsync、原子 READY；不覆盖不相容旧文件 | 同文件 `locked`、`durable_array`、`pending_blocks` |
| K13 | 标注 OOM 切批与错误概率防护 | 已实现切分微批；不截断替代、不用硬标签填补，概率越界／不完整拒绝完成 | 同文件 `scan_probabilities`、`infer_with_splitting` |
| K14 | 标注合并与完成门槛 | 已实现等待 worker 锁、按原顺序合并、域数与概率总量核对，成功后才发布 READY | 同文件 `run_merge:619–728`；当前无标签 READY |
| K15 | 三种监督训练入口 | 已实现 HARD CE、SOFT Bernoulli KL、HYBRID 0.5／0.5；同 parent／曝光／有效 batch | `scripts/train-four-domain-kd.py` |
| K16 | HARD 提前训练与已执行工作 | HARD 无需等待 Teacher，实际独立执行至约 30.60%；尚无完成能力结果 | `logs/distillation-KD_HARD.log`、快照 |
| K17 | 完整续训状态保存 | step35000、step40000 检查点包含 adapter、head 配置及 optimizer／scheduler 等训练状态文件，支持接续 | `KD_HARD/checkpoints/`、`latest.json`、训练脚本 save 分支 |
| K18 | 蒸馏控制器与进程身份接管 | 已实现 arm 锁、PID＋starttime、资产锁 owner 接管、缺失状态恢复和失败阻断；避免重复启动 | `run-four-domain-distillation.py`、`OWNER.json`、交接回执 |
| K19 | 标注—训练—评测阶段衔接 | 已实现 Teacher 就绪、全量标签合并、释放 Teacher 后安排 SOFT／HYBRID、最后评分的门槛；后半程尚未执行 | 控制器 main、`run-distillation-evaluation.py` |
| K20 | 8×A800 迁移与实验资源规划 | 完成 Teacher TP2 副本、三组 Student DDP、环境与数据迁移方案；仅规划，不记新机器实验结果 | 本地 [reports/a800-8gpu-distillation-20261031/](../../01-蒸馏/a800-8gpu-distillation-20261031) |

## 五、数据获取、转换、程序生成与缓存工程

本节已由主会话重新 SSH 读取当前清单核对，证据保存在 [数据清单快照](../../../03-证据与图表/08-综合汇总与历史记录/cluster-work-inventory-20261102/data-evidence.json)。各阶段 pair 存在包含、候选展开和去重关系，不能叠加成额外数据规模；“转换完成”不等于每个来源已通过全面污染或标注质量审计。

| 编号 | 独立工作项 | 状态与具体产出 | 远端相对证据 |
|---|---|---|---|
| D01 | canonical label-NLI 完整底池恢复 | 固定 default/train 的 5 个分片齐备，读取约 190 万原行；实现续传与回退，不把镜像重复算成新池 | `datasets/zero-shot-label-nli-http/DOWNLOAD.json` |
| D02 | typed 国内镜像快照接入 | 50 个文件条目；full 实载约 252.93 万 decision；与未完成 HF 快照隔离，full／default／filtered-full 不相加 | `datasets/tasksource-jev-typed-decisions-ms/{DOWNLOAD,TRAIN_COUNTS}.json` |
| D03 | 官方来源国内回退下载 | 回退清单 9 个文件／包条目，接通 GSM8K、WinoGrande、ANLI、When2Call、MMLU、HellaSwag 等原下载失败来源 | `datasets/benchmarks-train/MODELSCOPE_DOWNLOADS.json` |
| D04 | 跨域补充来源并行获取 | 18 个来源、22 个文件条目有下载／已存在记录；单独维护来源、版本与 train 标记 | `datasets/overnight-supplement/DOWNLOAD.json` |
| D05 | 从原包提取训练划分与文档 | 当前提取回执 10 项，涵盖 ACOS、英阿讽刺、MMLU auxiliary_train、HellaSwag、VAST、HoVer claim、Humicroedit、NLI4CT 等；只提取不代表均可入训 | `datasets/benchmarks-train/EXTRACTED_TRAIN.json` |
| D06 | 统一候选匹配数据协议 | 保留 state、question、criteria、candidate、软目标、domain、source、repo、group；比较型题保留双方和完整候选语境 | `scripts/pair_data_common.py`、两类转换脚本 |
| D07 | typed 全候选与概率规范化 | 648 个来源／子任务条目；转换约 252.93 万 decision，去重后约 1,607.51 万对；合并约 81.43 万重复，记录 18,978 条不可用记录 | `cache/pair-training/normalized/typed-full/MANIFEST.json` |
| D08 | SNLI／MultiNLI 三关系转换 | MultiNLI 约 39.27 万行→117.81 万对；SNLI 使用约 54.94 万行→164.81 万对，另记 785 行未转换 | `datasets/overnight-supplement/PAIRS.json` |
| D09 | 原生学科与常识多选来源接入 | MMLU auxiliary_train、ARC、WinoGrande、OpenBookQA、SciQ；分别约 39.94／1.35／8.08／1.98／4.67 万展开对；不是将各自 test 改称 train | `extra-pairs/MANIFEST.json`、补充 `PAIRS.json` |
| D10 | HellaSwag／ANLI 原生候选适配 | HellaSwag 39,905 题→159,620 对；ANLI 三轮 162,865 条→488,595 对，保持原生类别关系 | `cache/pair-training/extra-pairs/MANIFEST.json` |
| D11 | GSM8K／MATH 开放答案决策化 | 从答案标记提取目标，用其他训练答案构造干扰项；产出约 2.99／3.00 万对，未执行符号求解验证 | 同上；`prepare-main-extra-pairs.py` |
| D12 | MBPP 静态断言与输出候选转换 | 374 条原始 train→4,108 对；AST 读取已有断言与 literal 结果，没有执行程序或 MBPP 测试 | 同上 `mbpp`、转换脚本 AST 分支 |
| D13 | 高候选数意图分类展开 | BANKING77、CLINC150+OOS、MASSIVE 分别约 77.02／230.28／69.08 万对；类别名映射与原生标签分别处理 | extra／overnight 转换清单 |
| D14 | 新闻、主题与情感分类转换 | AG News 12 万行→48 万对，DBpedia 2,048 行→28,672 对，SST-2 约 13.47 万对，Tweet sentiment 约 13.68 万对 | `datasets/overnight-supplement/PAIRS.json` |
| D15 | Amazon ESCI 四级相关性转换 | 镜像约 41.97 万 query-product 行→167.86 万对；保留 E／S／C／I 四级关系，不冒充全量 ESCI 原始库 | 同上 `esci` |
| D16 | xLAM 多工具与无需工具适配 | 6 万条→228,584 对，其中 83,386 正对；支持多正确工具及 no-tool 候选 | 同上 `xlam` |
| D17 | Glaive 多轮工具决策提取 | 112,960 原行→429,356 对；另记 80,912 次调用解析失败事件，属于有损覆盖而非完整无损转换 | `extra-pairs/MANIFEST.json:glaive` |
| D18 | When2Call 双入口与多监督视角 | SFT＋preference 共 24,000 原行→113,394 对，覆盖响应适当性、偏好及工具／no-tool，不等于 24,000 独立题 | 同上 `when2call` |
| D19 | 临床 NLI 完整章节拼接 | NLI4CT 1,700 命题→3,400 对，关联主／次试验的指定正文 section | 同上 `nli4ct`；`prepare-main-extra-pairs.py` |
| D20 | 合同与立场标签接入 | ContractNLI 6,819 判断→20,457 对；VAST 13,477→40,431 对，保持 against／favor／neutral 语义 | 同上 `contractnli/vast` |
| D21 | 英阿语及多讽刺属性处理 | iSarcasm 6,570 行→23,542 对，处理有标注的 sarcasm／irony／satire 等属性 | 同上 `isarcasm` |
| D22 | ACOS 类别×情感全展开 | Laptop 2,934 review→1,003,428 对；Restaurant 1,530→55,080 对；体现候选结构带来的曝光放大 | 同上 ACOS 两域 |
| D23 | RAGTruth 回答与来源上下文连接 | 15,090 回答→30,180 对，按 source_id 获取正文；转换 group_id 使用回答 ID，不能据此称已按共享 prompt 完成隔离 | 同上 `ragtruth`、转换代码 |
| D24 | HH-RLHF 真实偏好比较 | 160,800 原行中使用 159,748→319,496 对；排除上下文不一致等不可转换项 | `datasets/overnight-supplement/PAIRS.json:hh_rlhf` |
| D25 | UltraFeedback 偏好与分数过滤 | 61,135 原行中使用 53,410→106,820 对；排除相同／非优响应等 7,725 行；轮换比较选项顺序 | 同上 `ultrafeedback` |
| D26 | HelpSteer／HelpSteer2 人类评分展开 | 3 项属性分别转 0–4 候选；产出 529,965／304,860 对，非同样数量独立回答 | 同上两来源 |
| D27 | Humicroedit 评分与幽默偏好 | 29,239 原行→95,476 对；连续评分分配相邻等级软概率，比较保留平局 | `extra-pairs/MANIFEST.json:humicroedit` |
| D28 | cfcolor 个性化历史与目标排除 | 158,500 个评分配对实例→367,264 对；历史最多 8 条、至少 4 条，排除目标 palette／相同颜色组合；不是用户人数 | 同上 `cfcolor`；转换脚本 |
| D29 | 工作流权限与审批程序生成 | 10 万场景、40 万对；成员状态、MFA、角色、操作、敏感性、审批等组合规则 | `generated-pairs/MANIFEST.json:workflow_policy` |
| D30 | 数值约束程序生成 | 5 万场景、20 万对；求和、最大值和阈值计数等规则 | 同上 `numeric_constraints` |
| D31 | 符号音乐结构程序生成 | 5 万场景、20 万对；MIDI 小节与音级和弦结构，不冒充人类审美 | 同上 `symbolic_music` |
| D32 | 诗歌尾韵结构程序生成 | 5 万场景、15 万对；AABB／ABAB／ABBA 三类模式，不冒充诗歌质量偏好 | 同上 `poetry_structure` |
| D33 | 象棋合法性多正例生成 | 3 万局面、12 万对；合法走棋形成局面，合法／非法候选；不是最佳着法或胜率监督 | 同上 `chess_move_legality` |
| D34 | 全池跨产品／来源精确去重 | 再合并 251,048 对重复，记录 14,483 次不同目标事件与 10 次跨域重复归属；不同目标取平均，不称已人工裁决 | `global-pool/MANIFEST.json:counts`、构建脚本 |
| D35 | 来源注册、分域映射与覆写 | 形成 983 个来源／子任务注册条目及 retained／duplicate 计数，支持显式 source-domain overrides；不等于独立设计 983 个转换器 | `global-pool/SOURCES.json`、`config/source-domain-overrides.json` |
| D36 | Prefix／候选分离 token 缓存 | typed：240.08 万 prefix／134.90 万 suffix；新增：499.94 万 prefix／84.06 万 suffix；原文分词不截断并复用旧 ID，不等于共享模型前向已实现 | `tokenized/{typed-full,global-new}/MANIFEST.json` |
| D37 | Token 缓存续接与互斥机制 | 已实现 PROGRESS／DONE、源行数／字节／mtime 身份校验、按行续接与 flock；未额外模拟断电或并发故障 | `scripts/tokenize-normalized-source.py` |
| D38 | 训练长度台账与头尾裁剪 | 四科共 10,451 对超过 4096，保留记录并按训练策略裁剪；不因长度丢 pair，另有 1 个艺术候选裁剪；不是训练无截断 | `global-pool/TRAINING_READY.json`、runtime `assemble` |
| D39 | 全量 mmap 运行时与长度组织 | pair 数组／prefix／suffix 只读 mmap，全池无放回顺序及分块长度整理，降低内存和 padding 成本 | `scripts/pair_training_runtime.py` |
| D40 | 训练／评测命名空间隔离 | evaluation-only 角色、路径拒绝和来源名保护；属于实际实现，不等于全来源内容级去污染已完成 | datasets README、转换与构建脚本 |
| D41 | 冻结公开请求本地重建 | 发布的冻结底稿＋官方 GSM8K answer-groups 语义生成 2,638 请求；不宣称与官方字节级一致 | `evaluation-only/decision-index-suite-0.3/LOCAL_PREPARATION.json` |
| D42 | 评测独立候选及分词缓存 | 全归档缓存 140,178 请求、2,484,515 pair，304,911 独立输入、215,099 候选；主榜选出 110,087 请求，两个规模不叠加 | 同目录 `candidate-cache/{normalized,tokenized}/MANIFEST.json` |
| D43 | 原生来源缺口排查与替代边界记录 | HoVer 缺 oracle 文档，ChessBench／ToolRet／New Yorker／Habermas 原生接入未解决；有下载与差距记录，不将替代规则源称为原生复现成功 | `extra-pairs/MANIFEST.json:gaps`、原始下载清单 |

来源注册的完整条目可见 [来源台账 CSV](../../../03-证据与图表/08-综合汇总与历史记录/cluster-work-inventory-20261102/source-register.csv)，typed 648 个任务条目可见 [typed 任务 CSV](../../../03-证据与图表/08-综合汇总与历史记录/cluster-work-inventory-20261102/typed-task-register.csv)。这些是逐来源追踪附件，不把每个上游任务自动算成一个新增研究项目。

## 六、环境、调度、兼容、恢复、备份与迁移工程

工程回执和关键代码位置见 [工程证据快照](../../../03-证据与图表/08-综合汇总与历史记录/cluster-work-inventory-20261102/engineering-evidence.json)。与前表重复涉及同一机制时给出交叉引用，不按两个视角重复增加成果数量。包、文件、分片和通知条目数都不是独立工作项数。

| 编号 | 工程工作项 | 状态与具体贡献 | 远端相对证据／边界 |
|---|---|---|---|
| G01 | 系统开发工具与长任务环境 | 已完成一次 70 个新系统包、7 个升级包安装事务，包含 fish、tmux、aria2、编译工具等 | `logs/apt-install.log`；不把 70 个包算作 70 项成果 |
| G02 | GPU Python 软件栈部署 | Python 3.11.18，195 个 Python 包安装事务；Torch 2.9.1、CUDA12.8 配套、TF4.57.6、PEFT、Unsloth 等落地并保存清单 | `logs/python-install-home.log`、`config/requirements.txt`；安装成功不等于所有组件经过独立测试 |
| G03 | Conda 迁至持久化工作区 | 离线 clone 失败后在线完成；迁移 26 个 Conda 包、18 个附加文件，更新激活与缓存前缀 | `logs/conda-relocate-online.log`；同一迁移的失败和重试不重复计数 |
| G04 | torchao 依赖兼容修复 | 实际从 0.18.0 调整至 0.14.1，并固化依赖约束 | `logs/torchao-pin.log`、requirements；不宣称另做过完整兼容测试 |
| G05 | 国内依赖源与统一运行目录 | Bash／fish 入口，Conda／pip／uv 国内源，HF／ModelScope 及模型、缓存、临时文件路径统一 | `config/condarc`、`pip.conf`、`env.sh` 等；本次未读取认证值 |
| G06 | 通用大文件网络适配 | IPv4／DNS／TLS 兼容、MSS、重定向、Range、续传、错误回退与完成改名，支持 D01–D04 的数据获取 | `scripts/download_utils.py`、模型／数据下载脚本；Xet 备选只有实现，不称已完整验证 |
| G07 | 单实例控制器和完成复用 | 文件锁防重复 owner，原子状态，串行训练后融合／评测，已完成阶段复用，最终 5 训练＋5 融合＋10 条件完整闭环 | `run-five-area-takeover.py`、正式 PIPELINE |
| G08 | 既有训练无重启接管 | 按 PID、命令、模型参数及 starttime 接管 Retrieval 原进程，日志有实际 adopted 记录 | takeover 控制器与 `scope-change-backup.../PIPELINE.json`；不是从非零 checkpoint 重启 |
| G09 | NCCL 受限下的四卡 Gloo 兼容 | 共享内存不足导致 NCCL 失败后，GPU 算梯度、pinned CPU FP32 SUM、再回传 GPU；5 个完成模型均使用此路径 | `formal-SPECIALIST_KNOWLEDGE.log`、`train-pair-model-takeover.py`、各完成记录 |
| G10 | 动态微批与尾批归一化 | token／样本双预算、动态 padding、异步拷贝，保持全局 128 pair，处理各 rank 不同微批数；与 D39 为同套运行时的训练侧功能 | runtime、takeover trainer；不把采样或长度处理再算新数据源 |
| G11 | 跨 rank OOM 一致回退 | 全局失败同步、清梯度、缩 token 预算后重算同一宏批；实际 12 次回退，最终五模型 pair 均未少训 | trainer、各 `oom_retries`；末级自动启用重算分支未实际触发 |
| G12 | 可续算状态、轮转与防误恢复 | 5 模型各保留 2 份 checkpoint／training_state；保存 optimizer、scheduler、随机状态、进度与微批预算，latest 原子更新 | trainer save 分支、各 checkpoints；正式五模型所见 resume_step=0，未证明做过非零步恢复演练 |
| G13 | 四科范围变更的跨组件联动 | 取消 Language 训练并从 MIXED_ALL 排除；同步 runtime、控制器、融合、监控、蒸馏队列；变更前保留 11 个文件，既有 Retrieval 不重启 | `ACTIVE_TRAINING_SCOPE.json`、`scope-change-backup-20261031T155002Z/`；评测仍五领域 |
| G14 | 冻结评测文件分段恢复下载 | 32 KiB range、16 并发、定位写入、chunk 进度与完整返回检查；2,176＋333 chunk 完成，约 82.17 MB | `download-frozen-evaluation-ranges.py`、resume 日志；不是额外评测任务 |
| G15 | 公开输入版本重建与隔离 | D41 的恢复工程补充：冻结底稿与 GSM8K 语义衔接、保存版本及 evaluation-only 角色 | `finalize-frozen-evaluation-data.py`；同 D41，不重复计数，不称官方字节级复现 |
| G16 | 评测 scorer 异常排查修复 | 处理 null gold／原生 ranking 路径及 edition metric；恢复时保留 Knowledge 模型已有 110,087 请求 | `evaluation-recovery-20261101/resumed.json`；不称新增模型或新增独立题 |
| G17 | 逐请求评测复用与尾部保护 | 稳定 worker ordinal，按 run_id 跳过终态，完成条件复用；仅允许处理无效尾部，中间损坏拒绝；与 E03／E04 为同一评测工程 | `run-formal-evaluation.py`；代码具有尾部保护不等于任意损坏都可恢复 |
| G18 | 盘满评测恢复与成果保全 | 历史恢复回执保留两分片 27,010＋27,003＝54,013 条，约 52.45 MB，另保留两个已完成 worker；后续流水线完成 | `evaluation-recovery.../diskfull-20261101T134030Z/RECOVERY.json`；本次未重新写入或校验字节，不能与 G16 相加成唯一请求数 |
| G19 | 阶段快照、观察与通知接口 | 实现唯一 milestone、完成产物／锁释放门槛和非干预 observer；本地 watcher 另有阶段受理状态 | `five-area-stage-snapshot.py`、`watch-training-session.py`；消息受理不等于用户已读 |
| G20 | 稳定资产私有云备份 | 初始 23 单元完成，后增至 25 单元、1,380 文件、约 21.53 GB 的云端核对回执 | `backup/modelscope-20261031/{FINAL-RECEIPT,SUMMARY}.json`；这是 11.01 归档回执，本次未新访问云端重验 |
| G21 | 增量上传监听与完成筛选 | 已实现单 worker、变更检查、稳定 checkpoint 和 complete 模型筛选；初始后实际增加 Arts／Retrieval 两单元 | `upload_backup_watch.py`、`upload-state.json`；总 complete=false，旧待传列表仍有 7 项且含取消的 Language，最终全产物备份未闭环 |
| G22 | 代码与小资产配套交付 | 469 个小资产文件打包约 16.36 MB，含约 97.05 KB 台账；另有 5 个执行材料／依赖／图等支持文件上传回执 | `SMALL-ASSET-UPLOAD-RECEIPT.json`、`GUIDE-UPLOAD-RECEIPT.json`；不要与大资产文件数直接去重相加 |
| G23 | 新机器恢复指南与仅补缺失策略 | 恢复指南在两个私有仓库发布，明确初始化／已训权重、依赖、缓存、源码顺序及不覆盖较新现场 | `NEW-MACHINE-RESTORE.md`、`RESTORE-GUIDE-RECEIPT.json`；完成文档发布，不代表跨机恢复演练完成 |
| G24 | 已有缓存发现浏览器工具 | 已实现 Chromium／Puppeteer 辅助终端与限定目录定位；没有可确认的缓存恢复成功回执 | `tools/cache-browser/`、`locate-existing-training-cache.cjs`；不读取 profile，不宣称浏览器安全加固 |
| G25 | 临时私有中转尝试与接收保护 | 实现私有／免费资源约束、路径穿越防护、冲突文件保留、限定自身资源清理；启动／回退／清理均有错误记录 | `scripts/private-transfer/`、对应日志；未证明中转或清理闭环，不披露账号／链接／凭据 |

备份限制另须保留：初始回执曾记录源清单覆盖缺口，后续小资产上传不自动证明全部缺口已消除。旧 watcher 的 pending 列表和云端路径布局仍需专项修正／核对；本轮只盘点，不擅自修改它们。

## 七、数据挖掘、可视化、台账与研究交付

| 编号 | 独立工作项 | 已完成内容与边界 | 证据 |
|---|---|---|---|
| V01 | 37 项输入形态全量画像 | 请求／问题／候选、state／joint／真实 pair 多种长度口径，主榜与归档展示项分开 | [reports/benchmark-data-mining-20261031/summary.json](../../../03-证据与图表/04-数据与评测/benchmark-data-mining-20261031/summary.json)、APPENDIX |
| V02 | 原生类别、匹配密度与未知标签研究 | 区分 yes/no 和 match/non-match，分析 ACOS 极稀疏原生正类及检索未判断候选 | 同目录标签表与分析报告 |
| V03 | 编码负担与权重错配研究 | 量化 pair／joint token 比、POP909／API-Bank 负担，形成共享编码优化依据；没有声称已获得对应加速 | 同目录计算负担表 |
| V04 | 来源组与跨 benchmark 关联研究 | RAGTruth prompt、GSM8K 原题、ACOS review 等聚类，以及 When2Call／BFCL 精确来源交集 | `source-overlap-summary.json`、来源组表 |
| V05 | 同输入异标签与表示语义诊断 | 检查 VAST、BBH 等可见输入歧义及 pair 表示的辅助字段信息损失 | 同目录 ambiguity 统计 |
| V06 | 构造线索与描述性基线研究 | 候选长度、位置和简单长度规则的关联；BPoMP 79.07% 是描述性参照，不是模型成绩 | 同目录候选偏差表、报告 |
| V07 | 数据画像与榜单表现关联研究 | 接入 247 个模型的既有公开逐项分数，观察跨任务数据特征与分数关联；不作因果解释 | 同目录 leaderboard 及 feature-correlation 表 |
| V08 | 数据分析网页与图表交付 | 汇报版、完整附件、HTML 总览／明细、CSV 和 PNG／SVG 图表，保留人类可读数字 | 同目录 `REPORT.md`、`APPENDIX.md`、`index.html`、`details.html`、figures |
| V09 | 实验对比看板 | 104 个历史 Language 方案的单次／seed 均值、九轴雷达、阶段／预算／rank 筛选、差值及 SVG／CSV 导出 | 本地 [reports/experiment-dashboard/](../../../03-证据与图表/02-Language/experiment-dashboard)；不是新增模型评测 |
| V10 | 实验方法与结果边界沉淀 | 完整精度 CSV／JSON、数据字典、阶段报告、复核、负结果、训练成本与分数口径分离 | 本地 Language 附录、teacher／fusion 研究报告及主工作记录 |

## 八、未完成部分与遗漏防护

不计入已完成成果：取消且未开训的 Language 专家、尚未产生的 SOFT／HYBRID 成绩、尚未生成的全量 Teacher 标签、官方私测 Full／线上名次、单卡延迟认证，以及尚未执行的新机器训练。

特别说明：typed 本轮使用 full，不将下载来的上游 filtered-full 过滤／LLM 审查报告计为本项目完成的全池审计；精确去重、来源路径保护、原生 train 标签都不能单独证明没有污染。已发生的来源调研、过滤机制和问题发现可以计为工作，但不夸大为未完成的质量结论。
