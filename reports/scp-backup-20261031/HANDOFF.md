# 小资产 SCP / ModelScope 大资产交接

## 首轮已知稳定资产备份交接（2026-10-31；以最新验收报告为准）

- SSH 别名：`sh-09-gpu`；主机名：`jev-gpu-dev-0`；当前 UID 为 0（root）。
- **实际工作区 root：`/home/dev/assumeengage`。**
- 此前成功只读观察时，`/home/jovyan/assumeengage`、`/tmp_home/jovyan/assumeengage` 均不存在；本轮未重新连接验证实时状态。
- 本目录仅由小资产 SCP 执行方维护，不修改主任务目录 `reports/modelscope-backup-20261031/`。
- 本轮不提交、不推送、不上传，不停止或重启任何实验/服务，不复制凭据。
- **小资产已安放：467 个源文件，约 38.19 MB；其中 433 个稳定文件、34 个明确标记的非最终快照。**
- **主任务最新独立核对：ModelScope CN 已有 1372 个稳定大资产，约 21.37 GB；首轮已知稳定资产备份已完成。** 此结论来源于主任务验收，不是本执行方访问云端复测。
- 主任务还逐项核对本地 467 个 installed 源文件，存在性/大小 467/467 匹配、无凭据模式命中；并将这 467 个文件加 2 个环境观察文件打成 **469 文件、约 16.36 MB** 的归档，放入私有数据仓库 `support/` 路径。
- 本执行方本轮再次对源机的 **201 个脚本、官方评测源码及配置文件**做 hash 对照，全部与本地已存版本一致，实际新增复制 0 文件。
- 首轮完成仅指当前已识别的稳定资产集合，不表示所有实验已经结束、未来模型/评测结果已经生成，或新机已通过恢复训练验证。历史丢失资产和后续新结果仍按下文缺口/通知处理。
- 此前已拉回的两份 final、日志/回执及共同初始化共 24 个文件已在本地重验，不重复 SCP。
- **实际落盘的小文件完成清单：`SCP-COMPLETED.json` / `SCP-COMPLETED.csv`，仅列本目录内存在且 hash 核验通过的 467 个源文件。** `LEDGER.json` / `LEDGER.csv` 仅是本执行方受限范围的盘点/排除/传输台账，不是旧机完整盘点，不能据其行数声称已备份；`COPIED-FILES.csv` 还含此前已有的 24 个阶段资产，不是新增小文件数。
- 大资产准确路径与优先级：`LARGE-ASSET-PATHS.csv`；关键目录汇总：`KEY-ASSETS.json`；传输核验：`VERIFICATION.json`。

## 此前成功只读观察的大资产路径（不是上传或完整性证明）

| 资产 | 路径 | 大小/状态 | 分工 |
|---|---|---:|---|
| 公共 Qwen3-0.6B 基座 | `/home/dev/assumeengage/models/Qwen3-0.6B/model.safetensors` | 1,503,300,328 bytes；文件存在，未加载验证 | ModelScope 方按需要处理，不 SCP |
| 共同 r32 初始化 | `/home/dev/assumeengage/models/Qwen3-0.6B-common-pair-init-r32/` | adapter 80,796,648 bytes | 已在本地 `models/Qwen3-0.6B-common-pair-init-r32/`；本轮不重复 |
| Knowledge 最终模型 | `/home/dev/assumeengage/runs/five-specialists-20261030/SPECIALIST_KNOWLEDGE/final/` | `complete.json` 为 true；adapter 80,800,744 bytes | 已 SCP 到本地阶段目录，本轮不重复 |
| Knowledge 最新完整训练状态 | `/home/dev/assumeengage/runs/five-specialists-20261030/SPECIALIST_KNOWLEDGE/checkpoints/step-000024012/` | `training_state.pt` 161,845,679 bytes；同目录有 adapter | 已交接主任务大文件队列；没有据此声称已上传，本轮未 SCP |
| Tools 最终模型 | `/home/dev/assumeengage/runs/five-specialists-20261030/SPECIALIST_TOOLS/final/` | `complete.json` 为 true；adapter 80,800,744 bytes | 已 SCP 到本地阶段目录，本轮不重复 |
| Tools 最新完整训练状态 | `/home/dev/assumeengage/runs/five-specialists-20261030/SPECIALIST_TOOLS/checkpoints/step-000012342/` | `training_state.pt` 161,845,615 bytes；同目录有 adapter | 已交接主任务大文件队列；没有据此声称已上传，本轮未 SCP |
| Arts 已存在中间 checkpoint | `/home/dev/assumeengage/runs/five-specialists-20261030/SPECIALIST_ARTS/checkpoints/step-000005000/` | adapter 80,800,744 bytes；training_state.pt 161,845,615 bytes | 按主任务更新仅标中间状态，不冒称最终模型；不干预训练 |

`training_state.pt` 内实际字段尚未反序列化检查；上述“训练状态”不等于已验证可无缝断点恢复。

## 必须一起备份的大训练/评测依赖

以下均以 `/home/dev/assumeengage/` 为 root。目录大小是盘点清单汇总，不是大文件内容/完整性校验；锁、`.part`、`.writing`、下载进度文件不应被当作完成产物。

| 目录 | 盘点大小 | 不能遗漏的原因 |
|---|---:|---|
| `cache/pair-training/global-pool/` | 约 2.22 GB | 实际全局训练池；`pairs/*_{state,candidate,target,source,group,length}.npy` 是逐 pair 映射；原文/候选在 inputs/candidates Parquet |
| `cache/pair-training/tokenized/typed-full/` | 约 2.69 GB | 共享 typed 前后缀 token IDs 和 offsets；仅备份 global-pool 不包含它 |
| `cache/pair-training/tokenized/global-new/` | 约 5.41 GB | 新增原文/候选的 token IDs 和 offsets；同样必须保留 |
| `cache/pair-training/normalized/typed-full/` | 约 0.81 GB | typed 基础原文、候选、各 area 的规范化来源与映射 |
| `cache/pair-training/extra-pairs/` | 约 0.16 GB | 附加来源转换后的训练对与元数据 |
| `cache/pair-training/generated-pairs/` | 约 0.01 GB | 已生成训练来源及其 manifest；没有把这些数据本体重复 SCP |
| `datasets/evaluation-only/decision-index-suite-0.3/` | 约 0.73 GB | 冻结请求、原始候选键、评分/group 元数据、规范化 field-options 和 eval token cache |

**大文件方最稳妥的恢复单位是整个 `cache/pair-training/` 和冻结 suite 目录，而非只上传两个最大的 bin 或只上传 manifest。** 逐行 NPY/Parquet 映射与 token offsets 未由本任务复制；准确文件路径、大小见大资产 CSV。原始数据源仍在 `datasets/`，其数据卡、来源和下载回执已在小备份中保存；大资产方按研究备份范围及原许可处理其本体。

全局 pool manifest 自报 `complete=true`。主任务最新确认的 `TRAINING_READY` 为 **32,787,042 pair、6,527,647,613 有效 tokens**；与本地真正落盘的 `files/cache/pair-training/global-pool/TRAINING_READY.json` 一致。source/area 元数据及精确计数保存在同目录 `{MANIFEST.json,SOURCES.json,TRAINING_READY.json}`，逐行二进制映射仍由大资产方处理。这不等于本轮重新验证或备份了全部大数据内容。

主任务现已明确确认 `reports/modelscope-backup-20261031/remote-inventory.json` 为真实落盘的逐文件元数据清单：1438 个文件、约 21.72 GB，范围是其选择的模型、数据、训练缓存和 runs 资产，不是旧机所有目录。其快照内旧的认证状态字段不代表当前上传状态；**元数据盘点也不等于上传或本地落盘证明**。

此前源文件状态反复更新时的“未生成/冲突/0 上传”记录仅作为历史更正保留，已被最新的清单确认和首轮验收结果替代。当前以小文件的实际 hash 清单、主任务独立核对的大资产结果为完成依据，不复用早先有争议的完整覆盖推断。

主任务确认源清单中未进云的 6 个根目录小元数据已在本 SCP 目录；64 个 `.part` 残片有意不作为完整数据备份。本执行方不把目录、归档内文件、碎片和已上传稳定资产的不同计数口径强行相加。

## MIXED_ALL / 融合 Student / 软标签：发现与缺口

- 在此前成功取得的受限工作区盘点快照中，**未发现最终 `MIXED_ALL/final`、五种融合模型的实体产物、`KD_HARD/KD_SOFT/KD_HYBRID` 训练目录，或 Teacher 软标签文件/ready 回执**。这不是当前全机不存在这些资产的证明；完整盘点和远端实时增量确认尚未完成。
- 已找到后续蒸馏计划：`config/next-distillation-REPORT.md` 与 `config/queued-distillation.json`，均已 SCP。
- queued 配置指定的未来 Student parent 是 `/home/dev/assumeengage/runs/five-specialists-20261030/MIXED_ALL/final`，未来新 run root 是 `/home/dev/assumeengage/runs/distillation-after-five-specialists-20261031`；Teacher 为 `PelaAI/KnowLine-31B-Gen1`。**这些是计划路径/选型，不是已存在模型或标签的证明。**
- 当前 run 清单含 Knowledge、Tools 的完成回执以及 Arts 中间 checkpoint；没有把 Arts 快照或共同初始化冒充最终混合 Student。
- 搜索范围是当前工作区 models/runs/config/scripts/reports/datasets 与非 Hub 的训练 cache；没有穷尽外部卷或公共 Hub 缓存，也没有持续监视后续训练阶段。新完成产物应按后续通知再增量备份。
- `/home/jovyan/assumeengage` 的历史 Language/Teacher 工程当前不存在；**旧 R0/LR32 权重、旧 9998 pair 标签本体和历史 PairEncoder 的完整恢复，不能由这份当前重建工程备份保证。**

## 源码、PairEncoder、加载器与自定义评测

源码原样保存在本目录 `files/` 下，不修改原机，也不在本地执行它们：

**新机恢复代码索引：`MIGRATION-CODE-INDEX.md`。** 包含输入模板、三份配置一致的 id2label、NCCL 与 CPU-Gloo 原路径、所有 prepare 脚本和评测加载器。源码/配置的本轮源机复核记录在 `checks/PRIORITY-CODE-RECHECK.json`。

| 用途 | 旧机相对路径 / 本地 `files/` 下同路径 |
|---|---|
| 正式训练入口 | `scripts/train-pair-model.py`、`scripts/train-pair-model-takeover.py` |
| 调度/接续入口 | `scripts/run-formal-experiment.py`、`scripts/run-five-area-takeover.py` |
| 共享输入渲染及 source→area 规则 | `scripts/pair_data_common.py`：`render_pair`、`domain_for` |
| Token cache、pair/source/group 读取和拼批 | `scripts/pair_training_runtime.py`：`TokenStore`、`PairPool`、`assemble`、`collate` |
| 共同初始化 | `scripts/initialize-pair-classifier.py` |
| 固定融合实现 | `scripts/merge-formal-models.py` |
| 自定义模型加载/推理/评测 | `scripts/run-formal-evaluation.py`：`load_classifier`、`margins_for_request`、`answers_for_request` |
| 官方 scorer / suite / engines / fixtures | 实体源码目录 `src/decision-index-main/` |
| 参数、标签、模型与融合条件 | `config/experiment-five-specialists.json` |
| 缓存构建及冻结评测准备 | `scripts/build-global-pair-pool.py`、`finalize-training-cache.py`、`prepare-evaluation-candidate-cache.py`、`finalize-frozen-evaluation-data.py` 等 |

**没有在已备份的 Python 源码中找到字面上的 `PairEncoder` 定义。** 当前重建流水线用共享 `render_pair` 与 runtime 函数实现对应职责；不能宣称找回了旧 `language_baseline.common.PairEncoder`。`src/decision-index` 在旧机解析到 `src/decision-index-main`，本任务备份实体源码，不创建/改写别名或系统链接。

迁移必须保留本轮实际语义：当前 manifest/config 的 entailment target 为 **1.0**，neutral/contradiction 为 **0.0**，评测使用 `logit[1]-logit[0]`。不要直接套用历史“类 0 匹配”的旧接口。源码内多处绝对路径指向 `/home/dev/assumeengage`，新机需要由迁移执行方显式映射；本轮没有改写这些代码/路径。

共同初始化及 Knowledge/Tools final 已核对为 `id2label={0:not_match,1:match}`、`label2id={not_match:0,match:1}`。原 `train-pair-model.py` 是 **NCCL＋手工 FP32 梯度 SUM all-reduce**，没有 `DistributedDataParallel` 包装；`train-pair-model-takeover.py` 保留 **CPU pinned FP32 缓冲＋Gloo SUM** 的临时路径。**新机采用 NCCL/DDP 是迁移目标，不是本轮已经实现/验证的配置；两份原源码均原样保存，未修改源训练。**

## 冻结 suite 与环境

- 冻结评测 manifest、excluded-question 清单、原镜像 manifest、candidate-cache 的规范化/tokenized manifest 和 DONE/PROGRESS 回执均已 SCP。
- suite 目录名虽然带 `0.3`，其 manifest 具体写的是 `release-v3`、base `release-v2.1` 及固定源 revision；规范化缓存自报 **140178 requests、2484515 candidate pairs、312320 fields**。恢复时必须保留原有替换/退休/exclusion/subset 语义，不能仅凭目录名重拉“最新 suite”，也不能拿旧九项 Language 的 31715 请求替代。
- 原 manifest 明确未声称官方 payload 的字节级一致；本任务的 SHA-256 只证明小文件传输一致，没有额外验证大冻结 payload 或重新评分。
- `environment/ENVIRONMENT-OBSERVED.json` 是在旧机现有 `envs/jev` 中通过只读 `importlib.metadata` 导出的包名/版本及 Python 版本，非环境变量/secret 导出。`files/config/installed-packages.txt`、`requirements.txt` 和 Accelerate 配置也已保留。
- 观察到 torch 2.9.1、Transformers 4.57.6、PEFT 0.18.1、datasets 4.3.0、Accelerate 1.15.0、PyArrow 26.0.0、NumPy 2.4.6、ModelScope 1.40.2。没有加载模型、运行 GPU 测试或安装依赖；这些版本不是已验证可在新机直接运行的完整镜像锁。

## ModelScope 账户 / 仓库 / 可用性

- 旧机 SDK 已安装（ModelScope **1.40.2**）。
- **当前按主任务最新确认：CN 已登录、两个私有仓库可写，且现有稳定大资产首轮已独立核对完成。** 模型仓库为 `assumeengage/qwen3-06b-experiment-backup-20261031`；数据仓库为 `assumeengage/jev-distill-data-20261031`。账户登录和云端可写性由主任务确认，本执行方未读取凭据或访问仓库复测。
- 1372 文件/约 21.37 GB 是主任务最新验收报告；469 文件/约 16.36 MB 的小代码归档已由主任务放入数据仓库 `support/`，具体归档文件名/版本以主任务恢复说明为准，本执行方不编造云端路径。
- 支持归档中的两个环境观察文件是打包时的快照；不能用包内历史认证观察推翻当前主任务验收。最新 HANDOFF、LEDGER 和本地增量工具是本目录的独立生成材料，不冒称它们全部都在该 469 文件归档中。
- 旧 HTTP 400 / Code `10010103009`、0 上传以及此前被撤回的上传数量，仅作为有时间/来源的历史记录，不是当前阻断或当前完成量。
- **后续认证交由主任务处理。本执行方不再尝试登录、查找凭据、读取凭据内容或调用认证/仓库探测接口。**
- 详情及来源归属见 `environment/MODELSCOPE-STATUS.json`。本任务未调用 login、创建仓库或上传接口，没有回传 Token/Cookie 值、私钥或环境 secret。

## 完成状态、快照与限制

- 467 个源文件均已在独立备份目录安放并核对 SHA-256；源码/配置在复制前后做了稳定性检查和常见凭据模式筛查，未发现命中。
- 收到主任务更新后，再次逐文件检查实际本地存在性、大小与 SHA-256；完成清单只从通过这些检查的本地文件生成，不把远端 inventory、排除项或计划中的路径纳入已备份数量。
- 对传输中变化的文件曾延后处理，之后仅在能取得一致版本时重拉。Arts 训练日志已保存为 **非最终快照**，没有为了备份停止或干预训练。34 个快照的时间、hash、完成状态都在台账中，不冒称最终结果。
- 当前源码备份是盘点时的工作区版本；它不自动证明与每个历史训练进程启动时的内存代码完全相同。未运行训练、模型加载、评分、pickle/训练状态反序列化或补实验。
- 不修改其他报告草稿，也不修改 `reports/modelscope-backup-20261031/`。没有 git add/commit/push，没有上传公网，没有复制凭据、权重、原始训练缓存或大预测文件。
- 大资产一侧必须另行验证上传与下载恢复；只有这份小备份不能声称完整模型/数据已备份。上游数据卡、来源与许可信息已保留，未进行新的公开再分发许可审批。

## 当前 SSH 与后续增量分工

- 先前网关关闭和旧控制套接字失效是历史情况。主任务通知恢复后，本执行方已实际用 `ClearAllForwardings=yes`、`ControlMaster=no`、`ControlPath=none` 成功完成只读源码盘点及 201 文件的源机 hash 对照。
- 未改全局 SSH 配置，未建立持久监视连接，未查找/使用他人的连接。源文件均与本地一致，本轮不重复 SCP。
- 主任务的大文件 watch 由主任务维护、最长 72h，并已挂完成通知；本执行方不另起 watcher、不修改或停止它，也不让本 session 一直等实验结束。

## 后续增量入口：一次运行、指定文件、完成即退出

首轮已知选定集合中的 467 个文件已真实落盘且无本地缺失；该范围可以收尾。历史未找到的实现/权重及尚未完成的未来模型/评测不冒充已备份，也不让本轮无限等待。主任务后续通知新结果/准确小文件路径后，在项目根目录手动单次运行：

```bash
# 仅检查指定源文件是否有变化；不 SCP，不更改台账。
python3 reports/scp-backup-20261031/tools/incremental_scp.py \
  --source scripts/pair_training_runtime.py \
  --source scripts/run-formal-evaluation.py \
  --dry-run

# 实际增量复制；每个 --source 都必须是明确的小文件，不接受目录或通配符。
python3 reports/scp-backup-20261031/tools/incremental_scp.py \
  --source scripts/pair_training_runtime.py \
  --source scripts/run-formal-evaluation.py
```

- 只用 SSH/SCP，并固定 `ClearAllForwardings=yes`、`ControlMaster=no`、`ControlPath=none`；没有 ModelScope SDK、认证/仓库调用或上传逻辑。
- 单文件上限 10 MiB；拒绝 hidden/secret/认证缓存、私有传输配置、权重、原始数组/数据缓存、越界路径和 symlink。
- 相同来源/hash 且现有本地文件仍核验通过时跳过；新版本保存在 `incremental/<时间-唯一ID>/files/<原相对路径>`，不覆盖首轮文件或用户工作。
- 传输 hash 一致才登记 `SCP-COMPLETED.json/CSV` 与 `LEDGER.json/CSV`；日志/运行状态明确标为非最终快照。完成清单可能包含同一来源的多个已验证版本，`unique_sources` 可区分来源数。
- 一次运行处理明确文件后退出，无监视、无限重试、训练控制、Git 操作或大件处理；源文件不存在/变化或连接失败会退出，收到稳定产物后再手动运行。请串行运行，不同时启动多个写台账实例。
- 后续本地实际数量以 `SCP-COMPLETED.json` 为准；本说明中的 **467 文件/38.19 MB** 仅是已知文件集合的落盘基线，不是全源覆盖或云端完成证明。
