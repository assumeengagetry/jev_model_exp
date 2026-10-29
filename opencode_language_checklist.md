# 给服务器 OpenCode 的后续清单：保留正在运行的 LoRA 基线

当前四卡正式训练已经启动，用户明确要求继续完成。**本清单用于接续现有工作，不得据此停止、重启、改配置或重复启动当前训练及其自动评测。** 先收取当前 run 的完整结果，再执行新实验。

这是用户独占的 Pod 节点，只有用户本人使用，内存和存储充足。全部 4 张 H800、CPU、内存和存储可用于本项目，常规下载、后续代码修改、依赖修复及新任务启动按已有授权自主执行。不设置资源配额或额外确认环节。当前流水线运行期间，保持其代码、环境、数据、调度和 GPU 分配；新准备工作放独立目录，不原地升级运行环境，也不抢卡启动新实验。

## 目标与已知位置（2026-10-27 截图）

- 只研究 Language Understanding，先完成当前 R0，再做从同一 R0 checkpoint 分支的受控续训，最后研究数据、适配容量和模型规模。
- 用户 jovyan，主目录 `/home/jovyan`。4 张 NVIDIA H800 NVL，每卡约 94 GiB，CPU 内存和存储充足。
- 基座为已下载 BF16 的 `Qwen/Qwen3-0.6B`，目录 `~/assumeengage/models/Qwen3-0.6B`；不要另换默认 4B。
- 数据为 `tasksource/zero-shot-label-nli` 的 default，目录 `~/assumeengage/datasets/tasksource/zero-shot-label-nli/data/`，约 1.17 GB，train/validation/test 已下载。
- 环境：`/home/jovyan/assumeengage/temp_env/train`。截图版本为 Python 3.11.17、torch 2.11.0+cu130、Transformers 5.5.0、datasets 4.3.0、huggingface_hub 1.33.0、Unsloth / unsloth_zoo 2026.10.9、accelerate 1.15.0、pyarrow 23.0.1；执行时记录实际版本。
- 当前 R0：`~/assumeengage/runs/language-baseline-20261027T061200Z`。Unsloth + PEFT SEQ_CLS，普通无 bias 线性 score，输出 batch×2；BF16 LoRA r16/alpha32、lr=1e-4、1 epoch、每卡 batch32、有效 batch128、四卡 DDP。后续不自动切换全参。
- 清洗后 train 1,813,810、validation 24,609、test 25,004；复用 R0 的实际 manifest 和缓存。
- 最新截图 08:41 UTC 为 4500/14171 step、loss0.387、约6.22万非padding tokens/s；仅为历史状态，以日志为准。实际 attention、seed、max_length、target_modules 从保存配置和代码只读核验，不能擅自声称 causal 已验证。
- R0 已通过真实分类 DDP、同步、保存重载；suite0.3 Language9项31,715请求和18项输入输出契约检查已准备。结束后已有“内部 test→最终契约检查→四卡完整 Language 评测”自动流水线，不另起重复任务。

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

下载源按截图实测分工：API/仓库元数据用 `https://d6108366.hf-mirror.com`，大文件用 `https://hf-mirror.net`。复用 `~/assumeengage/download_tasksource.py` 对两类源的处理；单设 HF_ENDPOINT 不保证 blob 自动切域名。没有本地代理，不假设代理端口；HF 官网和 hf-mirror.com 主域此前不可达，不反复原样重试。依赖装进现有训练环境，不装 Conda base。

## 1. 当前任务优先，按实际状态接续

- 先只读查看 R0 的 `training-status.json`、`training-progress.jsonl`、流水线日志和进程状态。截图中的 PID 与 ETA 可能已变化，不凭旧 PID 启停进程。
- 训练仍在推进：继续现有运行，只做不影响吞吐的独立 CPU/下载准备。不能因为已有 checkpoint-4400 就抢卡另起评测。
- 正在内部 test、契约检查或 Language 评测：沿用现有流水线，等待整条完成。
- 只有发生实际意外中断且确认原训练/流水线均已退出时，才按原 run 的恢复机制接续。不要主动制造中断。原恢复入口为 `bash "$RUN/launch.sh"`，恢复时先确认 RUN 指向该真实目录。
- 全部完成：读取 `checkpoints/final` 与 `evaluation/language-score.json`，再进入新阶段。缺结果时先查原流水线的实际失败原因，补跑缺失阶段，不重做已完成训练。
- 新阶段审计、配置和处理结果写独立目录；原 run 作为 R0 留档。

## 2. 环境和模型

- 当前四卡已用于 R0 及自动评测，保留现有分配。结束后的续训可每卡一组；采用 DDP/FSDP 时再验证新组合的实际通信。
- 复用已验证的 Unsloth + PEFT 分类路径，不为本版计划重装当前环境。后续双向验证先核对当前 kernel，必要时在独立环境用 Transformers + SDPA 建立参照；与原 run 隔离，记录后续最终实现。
- 检查实际基座权重、tokenizer、config 完整；缺少时下载真实基座并记录 revision。
- 在 run 目录保存最终配置、依赖版本、日志和 checkpoint。充分使用本地数据缓存、并行下载、CPU worker 和内存预处理，完成实际文件和数据加载验证。

## 3. 核对数据并建立可靠训练划分

以下用于读取既有证据及准备后续数据，不重新清洗、覆盖或替换当前 R0 正在使用的文件。

- 先看本地是 HF snapshot、save_to_disk 数据还是仅 manifest；核对实际 Parquet 分片是否齐全，不要仅凭目录或清单存在就判定下载完成。缺什么补什么。
- 使用 `default` 配置，不用 `triplet`。所需字段为 premise、hypothesis、labels；保留 task 做分层统计。
- 原始公开版本约有 train 1,899,993、validation 24,971、test 25,440；R0 已清洗为 train 1,813,810、validation 24,609、test 25,004。后续 D0 指 R0 实际清洗产物，不能把原始未过滤池替换成同名 D0。
- 统计实际标签。已知构建版本仅产生 `0=entailment` 和 `2=contradiction`，虽然元数据声明三个类别。若实际如此，重编码 `{0:0, 2:1}`，设 `num_labels=2`，保存正确 id2label/label2id。遇到别的标签先查版本和语义，不能静默丢弃或合并。
- train 用于训练，validation 用于选择 checkpoint，test 留作最终内部测试；这些不是榜单评测集。为新接入的目标数据准备独立 dev，按原始文档/题目/源 prompt 分组，不能只依赖公开榜单选择超参数。
- 先读取 R0 既有 audit 的重叠检查证据，关注 isarcasm/intended、ragtruth/grounding 等来源，检查正文、文档、源 prompt 或原始 ID。新增来源执行同样检查并写新 manifest；当前训练数据不热更新。
- 某个可疑来源的划分查不清时，先从本轮新训练清单排除该任务并记录，继续其余数据；无法核验的项不要声称已完成去污染。
- 将清洗后的现有 Tasksource 定义为 D0。并行补充 ANLI、HellaSwag 的官方 train，组成 D1；首轮 D1 的 Tasksource / 目标任务混合从各约 50% 实际 token 开始。不等待全部 9 项 train 齐备才建立闭环。
- 全部来源统一转为“state＋问题＋候选描述”的兼容/不兼容二分类。ANLI 将原始 premise+hypothesis 共同放入 state，蕴含/矛盾/信息不足作为关系候选；不要直接把原始三类 ID 传给两类头。HellaSwag 用上下文与候选续写组成正负匹配。gold 只生成监督标签，不进入输入正文。
- 先采样原始题目，再固定正负候选展开；分别记录原始题数、展开句对和实际 token，避免多选项任务被意外超采样。

## 4. 接通分类训练，而不是生成式聊天 SFT

R0 的分类链路已跑通。以下是后续实现需要继承的约束，不是要求重启 R0 再验证一次。

- 实际模型须为 Qwen3ForSequenceClassification，score 为 Linear(hidden_size, 2, bias=False)。基座缺少 score.weight 时新初始化是正常情况，不用另找分类头下载。
- 输入同时包含 premise 和 hypothesis，使用固定、明确分隔的格式；labels 为单个序列级整数，loss 为分类交叉熵。训练和推理使用同一套格式与标签解释。
- 使用分类 Trainer/训练循环，不能直接把 token-level 聊天 SFT 当作分类训练。
- 当前 R0 使用 BF16 LoRA SEQ_CLS，并训练/保存 score；沿用现有实现。新阶段固定 r16/alpha32、target_modules、head、格式和长度，保留当前有效 batch128。全参训练移到之后的独立容量实验，不作为当前任务的替代方案。
- 新阶段统一重置 optimizer/scheduler；续训 LR 可从5e-5作 pilot 起点，之后同组固定。不要沿用已经完成一轮、LR可能接近零的旧 scheduler。原 R0 的 lr=1e-4 不修改。记录本阶段与累计实际 token、tokens/s，use_cache=False；单卡分支可用梯度累积保持有效 batch128。
- 训练长度是初始配置，要记录长度分布、截断方式和比例，保留 hypothesis；不得把训练的截断设置带入正式榜单评测。
- R0 已做过分类与保存重载检查，不为了清单重跑当前训练。新代码或新 attention 组合只在后续独立 run 做必要 pilot：检查 batch×2、loss 有限、score 与 LoRA 更新，以及保存重载一致。
- 新初始化的随机 score 仅用于链路检查；不能把它的训练前分数称作原始 Qwen3 的能力。

## 5. 准备 Language 官方评测

R0 的评测已排入现有流水线。先复用并收取结果；以下约束用于检查报告和后续实验，不替换正在使用的评测代码。

- 官方工具：https://github.com/apolinario/decision-index 。先检查已安装版本和 CLI --help，缺少则安装所需依赖并记录 commit。
- R0 已准备 suite0.3 Language9项31,715请求与18项契约检查；先复用现有 suite、Engine、scorer、温度与归一化，不在等待期间重构当前评测。后续若修复接口，记录新版本并重评受影响结果。
- 只评测这 9 项，catalog_id：`11,12,29,38,39,40,41,42,59`，依次对应 ContractNLI、ANLI、HellaSwag、ACOS、FinEntity、iSarcasmEval、VAST、NLI4CT、RAGTruth。
- 优先复用校验通过的完整 suite；仅构建语言来源也可以用于局部开发，但必须保留官方样本选择、关联请求组和计分规则，记录验证范围，不能宣称通过全套 hash/complete 检查。
- 不要因其他维度的 HLE 等下载门槛卡住语言基线。不编造 `--area language` 参数；当前可将官方请求按 `_evaluation.catalog_id` 筛选后，通过自定义 `--rows` 输入。
- 实现自定义 Engine，或兼容 `/v1/systemone` 的服务。内置 transformers engine 加载 CausalLM，不能直接塞入分类 checkpoint。
- 统一使用 state、问题 instructions 和候选 criteria 渲染待判断文本对，不能读取 gold，也不按 benchmark 名写答案规则。使用固定的候选匹配打分和概率归一化，保存实现版本。
- choice 对全部候选返回有限、和为 1 的概率与 argmax；noul 返回 p_yes。候选可以 batch，但这是逐候选判别，不是 255 路共享读出。
- 评测不截断输入、不删除候选。超出真实支持长度则按 kit 记 Unsupported，计入覆盖率与错误。
- 先跑少量请求检查输出契约，再完整跑这 9 项。报告官方 Language skill、coverage、9 项分项、unsupported 和失败数，不拿内部准确率与榜单 skill 混比。
- 只跑 Language 时全榜 complete=false 是正常情况，需另证实这 9 项已跑齐。本轮交付是该维度的本地评分、模型 checkpoint 和可复现命令。

## 6. R0 全部结束后的四组分支续训

每组从同一个 R0 最终基座状态＋adapter＋score 开始。下面按原模式核验为 causal 写；若原本已经双向，则保留其真实标签，并用 causal 作为另一 attention 水平，避免两个重复双向组。

| 实验 | 新阶段数据 | Attention | 用途 |
|---|---|---|---|
| R1 | D0：R0 实际清洗后 Tasksource | 原模式 | 额外训练量控制 |
| R2 | 与 R1 相同 | bidirectional | 双向续训 |
| R3 | D1：D0 加 ANLI/HellaSwag 官方 train | 原模式 | 目标数据续训 |
| R4 | 与 R3 相同 | bidirectional | 两者结合 |

- 默认每组新增50M实际非padding输入token；R0已见token另列。先测100–200step，再统一锁定预算。不能只给某组更多训练量。
- 四组加载同一个 R0 最终模型，包括已训练分类头；都重置新阶段优化器和调度器。固定 seed、r/alpha、格式、有效 batch128、长度、学习率计划。R1/R2共享采样序列，R3/R4同样匹配。
- 比较 R2−R1 和 R3−R1，而不是把相对 R0 的全部涨分归因于 attention/数据；交互线索为(R4−R3)−(R2−R1)。结论限定为 R0 后的受控续训，不宣称是从预训练开始的完整2×2。
- R0 完整自动评测结束后才运行新GPU实验。优先每卡一组，必要时2×2卡；通过梯度累积保持有效batch。准备 D1 和 dev 可提前在独立目录推进。
- Qwen3-0.6B 有 28 层普通 full attention。先验证其 prompt 内双向自注意力；后续 hybrid 模型需区分 full attention 与 linear attention。不能把 StartLux 的 one in four layers 理解为每四个 token 开放一个。
- Transformers 5.5.0 不能仅改 config.is_causal=False 就视为完成；核验 Qwen3Attention 模块标志、mask 与 SDPA backend，Unsloth 的替换 kernel 同样核验。
- 以修改后部有效 token 后早期 hidden state 是否变化，验证真实双向：causal 早期应不变，bidirectional 应变化。最后 token 本来就能看到前文，不能用它作为唯一验证。覆盖有/无 padding、不同 batch 组成和保存重载。
- 首轮不 packing；后续 packing 必须隔离不同样本的注意力。训练和评测固定同一 attention mode。双向情况下不要假定跨候选的 causal prefix KV cache 仍可复用。
- “双向一定更好”是待检验假设。初筛单续训seed，胜出组合与关键对照至少3个续训seed，报告其共享R0的限定；最终配方从预训练权重重跑完整链时另做独立全流程复验。保存独立dev、Language分项、coverage、GPU-hours与推理耗时。

## 7. 真正启动并交付状态

- 先交付 R0 现有自动评测结果；不因为训练中间 checkpoint 已存在就另起评测抢卡。R0 整条流水线完成后，才调度 R1–R4 或后续 seed。
- 使用现有调度器、tmux 或可靠的后台方式运行，保存完整启动命令、日志、PID/作业号、GPU、输出目录和恢复命令。
- 确认进程存活且实际训练 step/loss 或评测请求数正在推进，再报告“已启动”；不能只生成脚本就停止。记录 checkpoint 的实际写出位置。
- 后续独立任务的安装、下载和代码问题自行修复，不为常规操作等待确认；保持正在运行的 R0 原环境与配置。若发现原评测/污染问题，保留原产物并标记受影响结果，在新 run 修正；训练污染不能仅靠重评同一权重消除。
- 最后给出：检查表、修改内容、实际运行命令、当前状态（训练中/评测中/已完成）、日志与产物路径、预计剩余量及其依据。已完成评测时附 Language 总分与 9 项分项；仍运行时给查看进度和恢复的命令，不编造结果。

## 8. 后续研究顺序与晋级条件

- 先读 R0 的9项分数、coverage、unsupported和错误桶。训练loss下降不能代替榜单提升；先区分接口/覆盖/校准/任务能力，再选下一轮数据。
- 全参训练改为独立容量对照：同一已训练模型函数起点，分别继续LoRA和全参；全参由同一adapter合并生成并核对eval logits一致。固定数据、head、attention、新增token与学习率调优预算，按结果决定是否切主线，不自动要求全参。
- 先根据独立 dev 和 9 项错误分布补缺失官方 train；FinEntity 未拆分的 979 文档属于评测源，不能从中切训练，用独立 SEntFiN 作金融实体情感训练代理。其他来源也须检查正文、文档与源 prompt 的重叠。
- 固定池与 token 预算，依次比较任务采样 p(task)∝n_task^α（α=1、0.5、0），再比较通用/目标 token 混合 80:20、50:50、20:80。n_task 按原题计数。每轮只改变一个因素，之后再研究困难负例、长文本和 teacher 蒸馏。
- 数据规模采用按任务/文档分层的嵌套 12.5%、50%、100% 子集，主比较同训练 token，记录实际见过的独立题数与重复率；一遍数据的结果另列，不混作同计算量。
- 参考 2026-10-26 15:30:58 UTC 公开 Language skill：同 Qwen3-0.6B 家族 Kai=11.56；相近规模强目标 StartLux v1.1 0.8B=55.53；跨架构 GLiClass v4 edge 0.44B=54.62。不能追平弱参考就宣称数据已充分。
- 最佳配方复验、评测覆盖有效后，接近指定同档目标，或连续两轮数据改进无稳定增益并有容量瓶颈证据，即安排更大模型小预算验证。不因旧 0.6B 未追平新基座 0.8B 而无限停留。
- 同家族路线：Qwen3 0.6B→1.7B→4B→8B/14B→32B；强基座路线另记 Qwen3.5 0.8B/2B/4B/9B→Qwen3.8-27B 或 Gemma-4-31B-it。换架构和换代的效果不能全算作参数规模收益。每档从各自预训练权重初始化，迁移配方。
- 27B 常规含主权重与 Adam 的全参状态约需 16–18 bytes/参数，即约 402–453 GiB，超过四卡合计 376 GiB，尚未算激活。先跑 LoRA 基线；全参需测 FSDP/ZeRO 加 CPU offload 或低内存优化器，不能认为 GPU 分片即可消除总状态需求。当前不等待大模型阶段准备完毕才启动小模型。
- 建立实验台账，每行至少含 base/revision、head、attention、数据池/采样、训练 token、seed、独立 dev、Language skill/coverage、9 分项、GPU-hours 与推理耗时。

## 已核验参考

- 数据：https://huggingface.co/datasets/tasksource/zero-shot-label-nli
- 标签构建：https://github.com/sileod/tasksource/blob/2796c32a346733e14b8c19b47e2fd9aade5aa03b/src/tasksource/recast.py
- Unsloth 分类加载：https://github.com/unslothai/unsloth/blob/63939abd1453968496f08ca7a1a59cfa5b8f8a9b/unsloth/models/loader.py
- Engine 接口：https://github.com/apolinario/decision-index/blob/main/docs/engines.md
- 评测数据源：https://github.com/apolinario/decision-index/blob/main/docs/suite.md
