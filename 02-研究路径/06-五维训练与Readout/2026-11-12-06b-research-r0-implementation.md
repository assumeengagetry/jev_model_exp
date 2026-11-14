# 0.6B研究R0：执行路径与读出精度机制实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用现有5个checkpoint和冻结1100题＋150换序题，隔离封存v2、标准PEFT BF16读出、同hidden严格FP32读出；产出实际机制证据后继续R1／R2，不把该诊断当正式Public分数。

**Architecture:** 实际新独立root `/tmp/opencode/jev-06b-research-20261112-r0-v2`；首版r0仅完成CPU准备、因Python缺少pidfd wrapper被启动门禁拒绝，原封存目录保留。v2只改控制兼容层，用已核实Linux syscall／pidfd落实自有进程截止；数值方案不变。CPU准备冻结来源、5个adapter／tokenizer快照及相同完整pair token输入。单GPU一个模型诊断任务，最多4个并行，按固定队列覆盖5模型；每任务先封存v2 A、再标准PEFT B与同一次hidden的FP32 C。完整逐题输出／metrics／来源receipt后合并给出条件式决策；只有此后实际冻结的候选才做完整public。

**Tech Stack:** 已有Torch2.9.1、Transformers4.57.6、PEFT0.18.1、Triton3.5.1／H800，所有依赖离线，不安装包。Context7额度耗尽，已回到相应版本原始源码核对hook／autocast／Qwen3 mask与pooling。

**Spec:** [已批准研究设计](2026-11-12-06b-top-public-research-design.md)。用户已明确自主执行，不等待逐任务确认。

## Global Constraints

- 不训练、不改旧权重／旧源码／暂停任务，不调用Teacher，不读public答案作训练或候选选择；现有24.91冠军保留。
- R0是本轮授权的真实机制实验，不是额外smoke／unit；沿用no-extra-tests、no-extra-model-or-corpus-hash，不以静态审查冒充通过。
- 研究Git只增加中文报告／计划／聚合证据；代码和原始题／标签／概率留在独立scratch。用户已批准此隔离方式，不新建Gitworktree或改主分支执行源码，不commit/push。
- 数据为现存readout完整题：dev1100、reverse150；明确历史trunk暴露未排除，只用于机制判断，不选泛化赢家。禁止从旧菜单input_ids直接当pair输入。
- 模型固定KD final、旧joint final、高LRstep0、step1000、final。checkpoint补齐的tokenizer只来自同KD tokenizer，config／adapter来源分别绑定，不伪造旧COMMITTED。
- 每题保留全部K、完整state/question/options、目标分布及语义安全换序映射；超过真实context时记录unsupported并阻止“完整成功”的声明，不截断或删候选。
- A复用4个封存v2核心逐字一致。B使用未改标准Qwen3／PEFT full-pair、causal、右pad；mask mapping full_attention=None，只允许未来右pad，不改变真实EOS可见输入。B/C共享hidden；C禁用autocast与TF32，只FP32 head projection，未改权重。
- 每正式输出绑定code、source、模型、数据、runtime及condition；未知partial不接管、不覆盖。进程只控制本轮精确lifetime；有外部GPU占用等待／停止本轮，不抢占。
- 每模型诊断正式工作时限3600秒（含加载）；超限保留partial、不把partial当全量，不据其误判机制。失败不自动改变权重／学习率或按答案重试。

## Review Focus

1. 新checkpoint与tokenizer／模型名绑定不能串用；step0必须与KD全部393保存参数相等，来源全程只读。
2. pool位置必须是实际EOS最后非pad，不将padding、旧菜单最后token或某个候选边界当同hidden；C必须关TF32且不重复计算另一份骨干hidden。
3. score/noul/choice目标按原完整分布解释，native score的概率诊断不冒称官方choice指标；reverse不重复计入主dev。
4. 全1250题／全部候选恰一次；A/B/C保持相同token输入，A真实v2调度与B完整pair可能有浮点差异，不能预设谁正确。
5. manager死亡／signal／OOM／断网／未知partial不会让孤儿GPU任务长时间失控；code／DATA_READY封存后不原地改动。

## Task 1：冻结来源和完整题输入

**Files:** 库外`code/research_common.py`、`code/prepare_r0.py`；新root `CODE_BINDING.json`、`SOURCE_BINDING.json`、`models/{kd,old_joint,high_step0,high_step1000,high_final}/`、`data/diagnostic.jsonl`、`DATA_READY.json`。

**Interfaces:** `prepare_r0.py` CPU-only；输出每题`rid,split,domain,source,native_kind,target,prefix_ids,suffix_ids,original_rid,option_mapping`及raw原字段到私有数据。common提供`ROOT/read/atomic/metadata/verify_ready/identity/live/require/stopped/own_lock`。

- [x] 保存来源既有完成receipt／tokenizer／5模型／完整题及4核心源码binding，核对旧完整产物后copy2到独立快照。
- [x] 原完整raw数据由同v2 renderer生成prefix、独立candidate分词及EOS；实际长度、K、目标分布、reverse映射准入，保存1100／150计数和逻辑token预算。
- [x] 正式CPU准备并直接回读DATA_READY；1250题、11879候选pair、3212543pair token／condition，5snapshot、4核逐字一致、315源文件binding通过。KD与step0全部393保存参数逐项相等。

## Task 2：A/B/C诊断与完整题metrics

**Files:** 库外`code/r0_worker.py`、`code/r0_metrics.py`、`code/inference/{kernel_engine,kernel_model,kernel_runtime,fused_kernels}.py`。

**Interfaces:** `r0_worker.py --model MODEL_KEY --gpu PHYSICAL_GPU`；输出`runs/MODEL/{A,B,C}.jsonl`，每行带rid、ordered margins、概率及条件辅助统计；完成receipt绑定文件集合。`r0_metrics.summarize(rows,outputs)`产生dev／reverse分开统计。

- [x] A使用封存`Field/score_request`数值路径及原默认warmup；不改kernel或根据正确性fallback。B逐key/dtype/value加载393参数，standard full-pair，同一实际EOS hidden通过forward pre-hook取得；C只FP32 head读出，BF16骨干不重跑。
- [x] 同全候选做float64 softmax，全部finite；完整source/condition原始输出后处理补齐native_kind／K分层和top-tie，dev/reverse分离，不冒称官方Index。
- [x] 真实机制运行前独立只读代码审阅并修正；实际每model每condition1250输出、rid及K精确覆盖，step0 saved params等KD，没有额外unit／smoke。

## Task 3：持锁调度、全量对账与机制决策

**Files:** 库外`code/run_r0.py`、库外只读watcher；root `PIPELINE.json`、`WORKERS.json`、`SUMMARY.json`、`COMPLETE.json`。

**Interfaces:** `run_r0.py`启动已admit模型队列，最多4自有GPU进程；`SUMMARY`读取所有condition实际完整文件后计算A-B／B-C配对变化和KD-step0身份对照，不自动修改模型。

- [x] 每worker独立lock、manager精确lifetime和全链marker，manager逐child登记；异常只停止自己children；3600秒预算／pidfd cutoff／parent-death保留现场实现，异常路径未额外故障注入。
- [x] 正式新鲜GPU准入后启动；读回模型exactload、实际已完成题而不是仅PID，再发布运行中状态。
- [x] 04:37:24UTC主代理直接核对5×3×1250题和source/code/runtime，18750输出全部finite及softmax／预算一致，KD-step0三条件逐题零差异，全部自有进程退出，4卡显存0。
- [x] 基于完整结果进入R1：高LRstep1000/final标准B、FP32 C仍接近随机NLL，FP32改善残余tie/reverse但不恢复识别；不额外跑失败FP32模型完整public。该判断不证明LR是唯一根因，R1继续隔离四卡路径。

## 执行裁决

- 已有用户“自主迭代”授权，选择主代理inline实现＋全R0一次fresh只读审阅，不再要求设计／方法二次确认。
- 无额外测试、无自动commit、无workspace删除；以正式生产准入、独立静态审查及本轮真实机制实验补证，并披露异常故障路径未经动态注入。
- 验证集隔离／R1／R2按已批准设计独立分轮，避免在本轮不依赖结果前把后续实现写死。该分轮不是到24.91就停止长期目标。

## 实际部署／启动

- 独立review核实四项Important来源／异常收口问题及一项observer超时问题，均在正式v2源码binding前修正；没有修改4个数值核心，没有追加unit／smoke／hash。来源清单exclusive＋只读＋DATA_READY元数据绑定；精确pidfd TERM→KILL截止和PDEATHSIG仅作用本轮worker，日志写失败不绕过清理；正常退出竞态重新poll。
- 首版CPU准备后启动因远端Python/libc未导出pidfd API被拒绝，未生成启动intent、未启动任何GPUworker。保留首版不覆盖；核实内核5.15／x86_64 syscall434/424后新建v2，prepare和launch对自身pidfd用signal0作实际OS能力准入。
- v2于2026-11-12T04:29:59UTC部署9份Python源码、逐字回读一致；04:30:45UTC新鲜四卡空闲准入后真实启动coordinator。启动时scratch约144.96GiB，全部旧资产保留；后续不得原地修改sealed code／ready。
- tokenizer沿用原权重配套资产与原默认行为，现有regex兼容警告保留；没有在机制对照中偷偷设置fix flag改变token IDs。
