# Language Understanding 后续实验路线

更新日期：2026-10-27。当前四卡 LoRA 正式训练已启动，用户明确要求继续完成。本版替代上一版“先全参、再从预训练权重启动四组”的默认顺序：先收取现有训练与自动评测结果，再从该 checkpoint 做受控续训，最后研究适配容量与模型规模。

**执行优先级：当前训练及其自动评测继续完成；后续实验不得触发停止、重启、原地修改或重复启动当前任务。** 本文是后续计划，不是对当前进程的修改指令。

## 1. 当前起点，以用户最新截图为准

| 项目 | 实际起点 |
|---|---|
| 节点 | 用户独占 Linux Pod，用户 jovyan，主目录 /home/jovyan |
| GPU | 4 × NVIDIA H800 NVL，每张约 94 GiB；CPU 内存和存储充足 |
| 环境 | /home/jovyan/assumeengage/temp_env/train |
| 已下载模型 | /home/jovyan/assumeengage/models/Qwen3-0.6B，BF16 |
| 已下载数据 | /home/jovyan/assumeengage/datasets/tasksource/zero-shot-label-nli/data/，default 配置约 1.17 GB |
| 软件 | Python 3.11.17；torch 2.11.0+cu130；Transformers 5.5.0；datasets 4.3.0；huggingface_hub 1.33.0；Unsloth / unsloth_zoo 2026.10.9；accelerate 1.15.0；pyarrow 23.0.1 |
| 当前正式 run | /home/jovyan/assumeengage/runs/language-baseline-20261027T061200Z，本文命名 R0 |
| 正式配置 | Unsloth + PEFT SEQ_CLS；BF16；LoRA r16/alpha32；lr=1e-4；1 epoch；每卡 batch32，有效 batch128；四卡 DDP |
| 清洗后的数据 | train 1,813,810；validation 24,609；test 25,004；以当前 run 的 manifest 为准 |
| 最新截图状态 | 08:41 UTC，4500/14171 step，31.75%；loss 0.387；约 6.22 万非 padding tokens/s；这些是截图记录，不是实时状态 |
| 已准备评测 | suite 0.3 的 Language 9 项，共 31,715 请求；18 个输入输出契约检查通过 |
| 已安排流水线 | 训练完成 → 内部 test → 最终分类契约检查 → 四卡完整 Language 评测 |

当前模型是 **Qwen3-0.6B**。R0 是有效的第一条正式 LoRA 基线，先完整使用其结果。截图未明确实际 attention 模式、max_length、seed、LoRA target_modules；之后从原 run 的配置和代码快照只读核验，不能把默认 causal 当作已验证事实。

R0 的当前日志位于 `training-status.json`、`training-progress.jsonl`；截图时已有 `checkpoints/checkpoint-4400`，最终位置看 `last-checkpoint.json`。最终模型预计位于 `checkpoints/final`，完整评分预计位于 `evaluation/language-score.json`。不能在任务仍存活时再次运行 launch.sh；恢复只用于实际意外中断之后。

环境入口：

```bash
source /opt/conda/etc/profile.d/conda.sh
export TRAIN_ENV_DIR="$HOME/assumeengage/temp_env/train"
conda activate "$TRAIN_ENV_DIR"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_DEFAULT_INDEX="$PIP_INDEX_URL"
export HF_ENDPOINT="https://d6108366.hf-mirror.com"
export QWEN3_MODEL_DIR="$HOME/assumeengage/models/Qwen3-0.6B"
export TASKSOURCE_DATA_DIR="$HOME/assumeengage/datasets/tasksource/zero-shot-label-nli"
```

下载沿用服务器已实测的分工：API/仓库元数据使用 `https://d6108366.hf-mirror.com`；模型和数据大文件使用 `https://hf-mirror.net`。复用 `~/assumeengage/download_tasksource.py` 的下载处理方式。HF_ENDPOINT 的单一设置不意味着所有 blob 下载都会自动切到另一域名；下载器需实际区分 URL。没有本地代理；不把依赖装进 Conda base。这些是截图给出的服务器现状，开始执行时以一次实际检查为准。

## 2. 固定首轮 readout 和评测目标

首轮沿用 **Qwen3ForSequenceClassification 自带线性头**。Qwen3-0.6B 的 hidden_size=1024，当前数据用 `Linear(1024, 2, bias=False)`。每个“状态＋问题＋候选描述”形成匹配判断，多个候选可批处理。此阶段固定 readout 和分类 CE，不同时研究 MLP、pooling 或 255 路 head。

原始 Tasksource default 的已核对版本有 train 1,899,993、validation 24,971、test 25,440，294 个分类任务；R0 使用的是清洗后 train 1,813,810、validation 24,609、test 25,004。后续 D0 复用 R0 的实际清洗产物与划分，不另造一个同名数据池。构建代码只产生原标签 0/2；实际一致时映射 `{0:0, 2:1}` 并保存标签配置。元数据有三个 ClassLabel 名称不代表实际包含 neutral。继承当前 run 已验证的映射与序列化格式，不在本轮训练中途重做。

新数据也必须转换成相同的匹配任务：

- ANLI：把原始 premise 和 hypothesis 一起作为 state；“蕴含 / 矛盾 / 信息不足”作为候选关系描述，构造正确与错误候选的匹配句对。不能直接往 2 类头传原始 3 类 ID，也不能在不改变题义的情况下把 neutral 与 contradiction 混成同一关系类别。
- HellaSwag：上下文是 state，各续写为候选，正确/错误续写构成正负匹配。
- 其他任务：保留原始问题与候选语义，机械转换，不把金标证据、答案或标签名称误放进 state。
- 先按原始题目/文档采样，再生成固定比例的正负匹配。分别统计原始题数和展开后的句对数，防止选项多的任务仅因展开次数多而占据训练。

为所有来源固定统一的文本对序列化格式，训练和 Engine 适配器一致；真实多分类任务的全部候选必须在评测时保留。若以后为了推理效率改成整题一次前向的 255 路 head，将其登记为独立 readout/目标函数实验，不能把分数变化算作纯数据增益。

公开 Language 评测只看同版官方 kit 的 **9 项任务、官方 skill 和 coverage**：ContractNLI、ANLI、HellaSwag、ACOS、FinEntity、iSarcasmEval、VAST、NLI4CT、RAGTruth，catalog_id 为 `11,12,29,38,39,40,41,42,59`。这是公开 category.skill，不是普通准确率、Full 或 private combined。正式上榜完整提交属于后续阶段。

## 3. 对照目标按规模与基座分别记录

下面是 2026-10-26 15:30:58 UTC 的榜单快照，分数均为公开 Language skill ×100。基座、训练数据、上下文能力不同，分差不能全部解释为数据质量。

| 用途 | 参考模型 | Served 参数 | Language |
|---|---|---:|---:|
| 近似同 Qwen3-0.6B 家族起点 | Decision-2.0-Kai-0.6B | 0.600B | 11.56 |
| 约 0.6B 部署规模参照 | Sieve 0.8B | 约 0.590B，榜单估计 | 26.59 |
| <1B decoder 强目标 | StartLux v1.1 0.8B | 0.873B | **55.53** |
| 小参数量跨架构参照 | GLiClass v4 edge | 0.440B | **54.62** |
| 2B 级 | StartLux v1.1 2B | 2.274B | **62.39** |
| 名义 4B 级 | StartLux v1.1 4B | 4.660B | **71.19** |
| 较严格 4.2B 级 | Jeffan-4B-2010 | 4.206B | 69.78 |
| 9B 级 | elephant-1 | 9.410B | **73.76** |
| 本快照 Language 第一 | mammoth-1 | 27.357B | **76.11** |
| 31B 级 | Jeffan-31b-1710 | 31.273B | 76.00 |

先超过同家族弱参考，再向更强的相近规模模型追赶。追平 11.56 只能证明超过该参考，不能宣称数据已经充分；也不把必须达到新一代 0.8B 的 55.53 设为永远不能晋级的硬门槛。Language 里 encoder 的成绩很强，不能由其总榜成绩低就排除这类对照。

来源与精确参数记录在 `research/small-model-language-targets.md`；在线来源为 [榜单 index.json](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json)。

## 4. 阶段一：让 R0 与现有自动评测完整结束

R0 已通过真实分类 DDP、参数同步与保存重载验证，评测请求与 Engine 契约已准备，不再从零重做链路。

### 当前训练和自动评测期间

- 保持现有四卡训练、训练参数、代码、数据文件、环境和流水线调度。不要改成全参、切 mask、热更新依赖，或抢卡启动下一轮。
- 只读检查当前日志、配置、代码快照和 manifest；独立目录中下载 ANLI/HellaSwag train、处理数据与准备 dev。CPU/网络任务也避免明显拖慢现有训练与评测。
- 查看现有 audit 的去污染与划分证据，特别是 isarcasm 与 RAGTruth；不把相关修改原地应用到正在运行的数据。
- 双向实现先做代码与配置核查；需要 GPU 的 hidden-state 验证排到 R0 整条流水线完成后，在独立实验目录执行。

### 完成后先读结果，再决定下一批

1. 等待现有流水线完成内部 test、契约检查和 31,715 请求的完整 Language 评测，不重复创建评测任务。训练 loss 0.387 本身不能预测 Language 分数。
2. 保存 R0 的真实配置、最终 adapter、score、基座 revision、数据与代码快照、token 计数、日志和耗时。核验实际 attention、max_length、seed 与 LoRA target_modules。
3. 形成 9 分项、总 skill、coverage、unsupported、错误类型与长度桶报告。评测严格不截断；先区分接口、长度覆盖、候选打分和任务能力问题。
4. 固定独立开发集：优先官方 dev，否则按文档/标题/源 prompt 分组留出；保留现有 Tasksource validation。后续超参选择不以反复跑公开榜单为唯一依据。
5. 如果发现评测适配或污染问题，保留 R0 产物并标记受影响结果；修复在新目录进行。评测接口错误可重评同一权重；训练污染需要另列干净训练结果，不能只重评就宣称解决。

## 5. 阶段二：从 R0 最终 checkpoint 分支续训

这一阶段回答“在现有已训练模型上，下一步怎样改最有效”。**四组都继承 R0 的同一基座状态、最终 adapter 与分类头**，复用已经完成的训练。当前不自动切全参。

定义 D0 为 R0 实际清洗后的 Tasksource；D1 为 D0 加 ANLI 和 HellaSwag 的官方 train。先按当前评测弱项确认数据优先级。新增数据使用与 R0 兼容的序列化、标签映射和独立 dev。

下表以核验 R0 为 causal 为前提。若 R0 已经双向，则将“原模式”记为 bidirectional，另一 attention 水平设为 causal；不能把重复的双向组当成对照。

| 实验 ID | 起点 | 新阶段数据 | Attention | 回答的问题 |
|---|---|---|---|---|
| R1 | R0 最终模型 | D0 | 原模式 | 仅多训练一段能涨多少 |
| R2 | 同 R1 | D0 | bidirectional | 双向续训的额外收益 |
| R3 | 同 R1 | D1 | 原模式 | 目标数据续训的额外收益 |
| R4 | 同 R1 | D1 | bidirectional | 两者结合是否更好 |

首轮每组可从 **新增 50M 实际非 padding 输入 token** 开始，R0 已消耗 token 另列。统一重置新阶段 optimizer/scheduler，使用相同学习率计划；不沿用已走完 1 epoch、学习率可能接近零的旧 scheduler。LoRA r16/alpha32、target_modules、head、格式、长度与有效 batch 按 R0 固定。续训学习率可先以 5e-5 作 pilot 起点，确认稳定后统一锁定；这不是对正在运行的 lr=1e-4 的修改。

四组采用同一续训 seed，R1/R2 使用相同采样序列，R3/R4 同样匹配。D1 初始 Tasksource/目标任务各约 50% 实际 token，不把样本数比例当 token 比例。先测 100–200 step 的吞吐与稳定性，再锁定共同预算。短跑只筛候选，不代表已经收敛。

R0 自动评测结束后，四张 H800 可每卡一组；单卡通过梯度累积保持有效 batch128，microbatch 依据显存实测。若 2×2 卡或 DDP 更快，可据吞吐调整，但组间有效 batch 保持一致。记录 tokens/s、GPU-hours 与显存峰值。

数据贡献看 R3−R1，attention 贡献看 R2−R1，交互线索看 (R4−R3)−(R2−R1)。不能只用 R2/R3/R4 相对 R0 的涨分归因于新配方，因为其中包含新增训练量。这个结论限定为“R0 后的续训效果”，不冒充从预训练初始化的完整因果/双向训练比较。

胜出组合与关键对照至少补足 3 个续训 seed，报告均值与标准差；它们共享 R0，只覆盖续训随机性。最终正式配方再从各自预训练权重重跑完整训练链，必要时使用 3 个独立全流程 seed 验证，不能把共享 R0 的分支当成全流程独立重复。

### 全参训练何时做

把全参改为后续独立的适配容量实验。固定较好的数据与 attention，用相同模型函数起点分别继续 LoRA 和全参训练；全参分支可由同一 adapter 合并后生成，先验证合并前后 eval logits 一致。两组新增 token、head、数据与格式固定，学习率按相同调优预算分别选择。这样研究的是后续适配容量，而不是把更大训练量误算为全参收益。

若全参明显优于 LoRA，再决定主线是否转全参；若 LoRA 已有稳定数据收益，先继续利用现有实现。不能把某个全参短跑与 R0 的整轮成绩直接比较后解释为纯数据增益。

### 双向 attention 的实现验收

“值得优先实验”有公开证据；“改完肯定涨分”没有被这些组合配方单独证明。Perplexity 同时增数据并解除 causal mask；StartLux 还改变训练阶段并做 seed averaging。

当前 Qwen3-0.6B 是 **28 层普通 full attention**。后面 hybrid Qwen3.5/3.8 的“full-attention 双向、linear-attention 保持 causal”适用于不同架构。StartLux 的 one in four 指层的结构，不是每四个 token 只开放一个 token。

对截图中的 Transformers 5.5.0，不能只设置 config.is_causal=False 就认为修改完成：Qwen3Attention 的模块级 is_causal 和 backend 的 mask 路径也要核验，Unsloth 还可能替换 forward/kernel。

- 先用 Transformers + SDPA 建立可验证实现，再对齐 Unsloth 加速路径。
- 改变后部有效 token，检查早期 token hidden state：causal 早期位置应不变，bidirectional 应能受影响；仅看最后 token 无法区分两者。
- 同时验证有/无 padding、不同 batch 组成、保存重载和推理一致性。
- 首轮不 packing；后续 packing 必须用 block-diagonal mask 隔离样本。
- gold 仅参与 loss，不能进入可见输入。训练与评测使用同一 attention mode，分类 prefill 使用 use_cache=False。
- 双向前缀表征依赖后文，不能假定跨候选的 causal prefix KV cache 仍可复用。

固定版本源码与核查见 `research/attention/audit.md`。

## 6. 阶段三：把数据操作做成有顺序的研究

### 6.1 补任务覆盖

第一优先补 ANLI、HellaSwag。之后根据完整 9 项的错误分布和独立 dev 结果补 ContractNLI、ACOS、iSarcasmEval、VAST、NLI4CT、RAGTruth 的官方 train。

FinEntity 的原始 979 文档为榜单评测用的未拆分源，不能从中切 train；用独立 SEntFiN 金融实体情感作为训练补充。数据入口、规模和原始版本见 `research/language-axis/manifest.json`。该清单此前下载在 Mac 工作区；服务器上的 OpenCode 必须实际检查或下载，不能假设 Pod 已存在这些文件。

接入目标 train 时，以原始划分为准，按正文/文档/源 prompt 去重与去污染。相同文本的其他任务视图也一起检查。合同和临床任务保留题目所需上下文，不用金标证据裁剪输入。

### 6.2 改任务分布

对已固定的数据池，比较：`p(task) ∝ n_task^α`，α 依次取 1、0.5、0，对应自然比例、平方根平滑、任务均衡。n_task 按原始训练题统计，再记录匹配句对和 token 的实际占比。

选择较好采样法后，再比较通用 Tasksource 与目标任务数据的混合：80:20、50:50、20:80。采用相同训练 token 预算。不要同时改变采样指数、混合比、attention、head 和学习率。

这些比例都是起始搜索点，不是已知最优配比。独立 dev 决定候选；阶段公开分用于核对迁移效果，记录已进行的公开评测次数。

### 6.3 改难度、负例和长度

- 固定随机负例基线，再比较相近标签/相近续写/矛盾关系的困难负例。避免把未标注的正确答案误当负例。
- 单独加入改变实体、否定词、时间、数字或证据充分性后应翻转答案的对照对；逐步引入，不把它们和其他改动一起加。
- 分开看 <512、512–2k、2k–4k、4k–8k、>8k 的开发集表现。若错误集中在长文，加入真实长合同、临床或 grounding 训练样本，做短长混合；不要仅把 seq_len 数值调大就认为解决了长文问题。
- 保留关键细节不同、因此答案不同的对照题；去重不能把这类题当作完全重复内容删除，也不能按模型当前答题正确率筛掉难题。语义确实相同却标签冲突的记录另行核验。
- 若仍有需要，后续再研究 teacher 软标签；只对 train 生成，保留 gold 监督，作为独立蒸馏实验。

### 6.4 区分数据规模和重复训练

对最佳池做按任务与文档分层的嵌套子集：12.5%、50%、100%。主比较保持训练 token 预算一致，小池允许重复；记录实际采到的独立题数与 token，不能把可用池大小冒充实际已训练规模。辅以一遍数据的曲线，明确其训练计算量不同。

同时记录训练/验证 loss、任务分项、独立样本数、输入长度、实际见过 token、epoch、GPU-hours。这里研究后训练的样本效率；不能直接套预训练的 Chinchilla token/parameter 配比，也不能由一个模型追平一个公开分数宣称验证了 scaling law。

若主要错误表现为候选间比较或概率校准，而非输入理解，先做一个独立的 readout/目标函数小实验，例如固定数据下比较当前逐候选 binary CE 与按整题归一化的 choice loss。明确它改变了训练目标，不把收益归为数据或参数规模；255 路整题 head 可作为之后的效率研究，当前不作为启动依赖。

## 7. 阶段四：何时放大模型

进入下一规模的默认判据：

1. 分类头、评测协议、attention 模式、数据版本已固定，完整 9 项没有明显接口错误或未解释的覆盖缺口；
2. 最佳配方与关键对照已复验，至少有 3 seed 的主结果；
3. 已接近匹配规模目标，或者连续两轮固定预算的数据改进在独立 dev 上已没有稳定收益，且分项诊断支持容量/基座可能成为瓶颈。

“接近”可先用距指定参照 ≤2 个 skill 点作工程目标；这不是统计等价的证明。只有拿到对照逐题预测，才可按文档/关联题组做配对 bootstrap；只有榜单标量时，报告点差和我们自己的 seed 波动，不虚构对照置信区间。

不要为 Qwen3-0.6B 必须追平更新基座 0.8B 的 55.53 而无限停留。即使仍有差距，也可以安排一次 1.7B 或 4B 的小预算迁移：同一数据配方更大模型是否显著改善，是判断容量因素的直接证据。

### 两条规模路线分开记录

| 路线 | 建议规模 | 用途 |
|---|---|---|
| 同家族容量对照 | Qwen3 0.6B → 1.7B → 4B → 8B/14B → 32B | 尽量减少换代因素；中间档用于小预算筛选，非每档都做穷举 |
| 面向榜上强基座 | 选一个 Qwen3.5 0.8B/2B 与对应 Qwen3 比较 → 4B/9B → Qwen3.8-27B 或 Gemma-4-31B-it | 研究配方能否迁移到更强基座；同时记下基座/架构变化 |

当前榜上明确的 Qwen3.8 目标是 27B，Gemma 对照为 31B；若保持 Qwen3 同家族，32B 是独立选择。每个尺寸从其自身预训练 checkpoint 初始化，迁移的是数据、训练和推理实现，不直接把 0.6B 的权重矩阵装进大模型。

每升一档，先用该档未经本轮任务训练的基座、固定 readout 接法和小预算做 pilot；重扫少量学习率与有效 batch，比较前后训练增益及与该档参照的差距。新初始化分类头的起点仍需标注为随机头。

## 8. 4×94 GiB 的使用方案

| 阶段 | 资源分配 | 训练方式 |
|---|---|---|
| 当前 R0 及其自动评测 | 维持现有四卡流水线 | BF16 LoRA r16/alpha32，不修改 |
| R1–R4 数据/attention 续训初筛 | R0 全部结束后，优先每卡 1 个实验或 2 组×2 卡 | 保持同一 LoRA 设定 |
| 0.6B 最佳配方复验与容量对照 | 并行 seed；全参与 LoRA 单独比较 | 依据实验结果决定主线 |
| 1.7B/4B | 单卡、2 卡或 4 卡根据激活和实测吞吐选择 | 迁移已验证配方，必要时补适配方式对照 |
| 8B/14B | 4 卡分片与 activation checkpointing | 评估 FSDP/ZeRO；记录实现变化 |
| 27B/31B/32B | 4 卡配合大 CPU 内存 | 先跑 LoRA 基线；全参再评估 FSDP/ZeRO + optimizer/参数 CPU offload 或低内存优化器 |

常规含 FP32 主权重与 Adam 状态的混合精度全参方案约需 16–18 bytes/参数的状态空间。27B 对应约 432–486 GB（约 402–453 GiB），已超过 4×94=376 GiB，尚未算激活和临时张量。单纯 DDP 会复制状态；FSDP 只在 GPU 内切分也不能消除总状态需求。实际实现可能更省，须通过 pilot 测量，但不能直接沿用旧文档的 8×141 GiB 配置。充足 CPU 内存正好用于 offload。

先测每个配置 100–200 step 的实际 tokens/s，再按剩余训练 token 和实测完整评测耗时估计工期；不预先承诺未经测量的小时数。

## 9. OpenCode 下一批应交付的结果

1. 保持 R0 与自动评测运行，读取现有日志更新真实状态，不重复启动任务。截图里的剩余 68 分钟只是当时估计，后续以实际进度为准。
2. 收取 R0 最终 checkpoint、`evaluation/language-score.json`、9 分项与 coverage，以及配置、manifest、audit 与代码快照；复用已存在文件，不要求改当前 run 补文件。
3. 在独立目录准备 D1、dev 与 R1–R4 配置；当前流水线释放四卡后再执行 GPU 验证和续训。
4. 四组共同 parent checkpoint、LoRA 配置、有效 batch、续训 token、优化器重置规则与 attention 实现验收证据。
5. 一张统一台账，分别标记 R0 整轮结果、续训分支、全参容量实验和之后的大模型；每行记录 model/revision、parent、head、attention、数据池/比例、本阶段与累计 token、seed、dev、Language skill/coverage、9 分项、GPU-hours 与推理耗时。
6. 根据分项选择数据覆盖/分布的下一轮；另行评估全参实验和模型晋级，不因新计划覆盖现有结果。

本次根据用户最新训练截图修改后续计划，未直接连接或操作 Pod。服务器上的 OpenCode 按该顺序接续现有工作；当前训练继续完成是明确约束。

## 主要依据

- [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
- [Tasksource label-NLI](https://huggingface.co/datasets/tasksource/zero-shot-label-nli)
- [Perplexity 对数据与解除 causal mask 的说明](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
- [StartLux 0.8B 的训练和 attention 说明](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant)
- [Transformers 5.5.0 Qwen3](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/qwen3/modeling_qwen3.py)
- [Transformers 显存组成](https://huggingface.co/docs/transformers/model_memory_anatomy)
