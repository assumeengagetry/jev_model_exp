# 五维蒸馏参榜候选 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按最高单次 E24 蒸馏配方保留四科、补入语言，重训一个五维 Qwen3-0.6B，并生成真实可复现的参榜材料。

**Architecture:** 独立命名空间复用旧四科数据／Teacher 缓存和已审计关系，仅新增原生语言训练数据与对应 Teacher 标注。单模型联合重训，推理与官方 Engine 可移植化；完整公共运行和发布分别受用户授权约束。

**Tech Stack:** 现有 Python／NumPy／PyTorch／Transformers／PEFT／显式 FlashAttention-3／KnowLine Teacher 环境、Decision Index 官方 kit；不升级共享环境。

**Spec:** `docs/superpowers/specs/2026-11-09-five-dimension-distilled-submission-design.md`

## Global Constraints

- 当前扩大续训保持暂停、`complete=false`、不自动恢复；旧权重／partial／结果不改写。
- 只做一个正式候选；不加测试、烟测、消融、重复训练、额外 hash；官方公共运行及强制套件校验需用户批准。
- 不自动上传、发布、commit／push，不启用付费 HF Jobs，不重启服务。
- 原四科每科 250,000 pair 和 E24 损失不变；语言约 250,000 pair，保留完整组、原始标签和来源身份。
- 语言 Teacher 权重建议 0.5，K/R/Tools/Arts 为 0.5/0.25/0.5/0；Tools logit bias -1.2826099976040561；关系 gamma0.5／lambda0.05。
- rank32／alpha64、LR1e-4、batch128、weight_decay0.01、warmup0.03、seed42、一遍；严格数值控制与真实 FA3 前后向，不回退 SDPA。

## Review Focus

- 五维拼接后的 Teacher 概率与 3,647 条关系索引不能错位；旧百万条必须逐索引对应。
- 语言不得被广泛 `typed_full` 语料淹没；来源 train／英语标记／原题组／原生类平衡需可恢复。
- Teacher partial、Student resume、final 与评测必须绑定同一冻结池与配方；失败不伪造完成。
- 新训练不得误恢复已取消的第十波或已暂停的扩大训练，不抢外部进程。
- 官方 Engine 不截断、不过滤选项；旧本地分与新模型成绩严格分开，未获正式结果不填 `complete=true`。

---

### Task 1: 暂停资源与锁定方案

**Files:** 本 spec／plan；`reports/five-dimension-distilled-submission-20261109/PAUSE-HANDOFF.json`。

**Interfaces:** 消费旧训练会话的暂停回执；产出精确暂停与 GPU 空闲证据、经审阅的训练范围。

- [x] 通知原实施会话停止自动恢复与训练，保留 checkpoint／优化器。
- [x] 独立读取远端暂停状态并查询 GPU；旧 coordinator／launcher／workers 均已退出，GPU 无计算进程。
- [x] 归档暂停回执；独立只读 builder 审查设计（session `ses_e40efd041ffeMgZhnwoqvtS22h`），已核实原始 converter 并明确任务单位／原生视图／库存不足规则与语言专科不等价的边界。
- [x] 用户已确认本会话直接实施本方案，并批准一次正式 public0.3 运行及官方强制套件校验；不加测试／烟测／额外哈希，不授权自动发布。

### Task 2: 冻结五维训练池与新增语言 Teacher 标注

**Files:** 新 namespace 的 `code/prepare_five_dimension.py`、`code/label_language.py`、`config.json`、`pool/`、`teacher-labels/`、`DATA_READY.json`。

**Interfaces:** 消费旧 `PairPool`／`TokenStore`、E24 配置／有效目标／原关系及原生语言训练元数据；产出冻结池、唯一索引、语言 Teacher 概率与 provenance、五维有效目标与原子组批计划。

- [x] 只读核对原生语言 train／英语／group 元数据；落实 spec 的9任务定义、非讽刺库存封顶重分配、核心英语讽刺5%原生类平衡和 RAGTruth 回答级二分类。已修复本实验的旧ANLI跨轮group_id碰撞，正式v2准入完成。
- [x] 复用原四科百万 pair 的顺序与旧 Teacher 概率，语言追加；正式池1,250,308pair／366,985,783tokens，语言250,308pair，unique244,710pair；关系端点的有效目标／计划最终准入仍在Teacher完成后的targets阶段。
- [ ] 使用原 labeler 的 worker／partial／恢复机制仅标新增语言；启动前检查 GPU 空闲和存储余量，任何不完整标注不得生成 READY。
- [ ] 构建语言0.5混合目标、原四科不变目标和原关系辅助，按实际池大小重建原子计划；记录真实 pair／token／尾批与重复曝光。

### Task 3: 一次正式五维联合重训

**Files:** 新 namespace 的 `code/train_five_dimension.py`、`code/run_pipeline.py`、`checkpoints/`、`progress.json`、`final/`、`TRAINING_COMPLETE.json`。

**Interfaces:** 消费 Task2 的冻结目标和计划；产出本任务独有的 final 与训练证据。复用严格数值 trainer 和 FA3 helper，不改共享实现。

- [x] 将原训练入口的四科／百万／旧 namespace 限制最小 fork 为本任务真实五维身份；保留原 BCE 与关系损失归一、完整尾批及 OOM 恢复机制，独立只读审查已完成；没有新增测试。
- [x] 固定 spec 的优化配置与单卡原子训练；存储锁、进程身份、暂停 guard、checkpoint 身份与数值证据路径已实现并部署。Student实际恢复／FA3前后向仍待真实训练验证。
- [ ] 实际开训并读回优化步、五维曝光与 FA3 前后向，记录启动 PID；不以启动命令退出0代替真实训练证据。
- [ ] 监视到 final 实际导出，检查真实曝光／完整步数／来源配置；失败或暂停维持 incomplete，保留现场。

### Task 4: 可移植推理包与正式公共材料

**Files:** `artifacts/five-dimension-distilled-20261109/` 的 model／source／evidence／README；新 namespace 的 `code/decision_engine.py`、`submission/`。

**Interfaces:** 消费 Task3 的固定 final 与原真实推理渲染；产出维护者可启动的 Engine、依赖和精确设置；获准后产出官方 `runs/<name>/` 完整运行目录。

- [ ] 整理分类头、adapter、base revision、tokenizer、推理渲染和 choice／noul 概率转换，移除个人绝对路径依赖；不改本模型读出方式。
- [ ] 固定可声明容量，完整请求不截断、所有候选给概率，超容量使用 `Unsupported`；保存模型与输入／输出设置。
- [ ] 仅在用户批准后，使用官方 public0.3 suite／pipeline 完整运行一次及官方强制输入校验，不额外跑旧 wrapper 或小样本评测。
- [ ] 保存未经修改的结果、官方 scores／environment／status；只有官方实际完成时才记录完整，缺材料明确列出。
- [ ] 核对 base／Teacher／训练源许可，本地完成模型卡和 PR 草稿；上传、访问方式、模型名称和 PR 发布等用户最终确认，不自动发布。

## 计划自审

数据、训练、推理与发布边界均对应 spec；新语言权重与采样是明确建议，不称已验证。用户禁止额外测试，本计划以生产准入、真实训练及获准的正式公共运行验证，不添加测试步骤。模型分数、耗时、实际 token 数未执行前不预填。实施方式建议本会话直接执行，避免多个实现者改同一组实验脚本。

## 2026-11-09 08:17 UTC 实际状态

- 正式根目录 `/tmp/opencode/jev-five-dimension-distilled-20261109-v2`；coordinator323791与三个Teacher worker均存活，语言真实持久化标注29,760／244,710，OOM0。
- Student训练将由同一队列衔接，目标9,769步、尾批4pair；尚未执行Student优化，不声称FA3新训练已经通过。
- 官方Engine与队列经独立只读审查后，已修复单候选OOM整请求Unsupported、受保护子进程启动及继承锁、模型/实际基座/Engine只读冻结快照与启动/完成复核、完成后幂等跳过；这些路径的运行验证仍以未来真实正式运行作证据。
- 官方coordinator324276实际阶段为waiting_for_completed_five_dimension_final，独立查活；不会抢Teacher资源。只安排获准的一次官方完整public0.3，不上传、不启用付费Jobs。
- 官方suite已按固定upstream commit严格导入match=true；全套件不是旧37项wrapper，最终complete以官方scores.json为准。
- 本地只读监视器已启动，完成/暂停/阻塞时通知，不恢复任何队列。实施未结束，以上未勾选步骤继续等待真实产物。
