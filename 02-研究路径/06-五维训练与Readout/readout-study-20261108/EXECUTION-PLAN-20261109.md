# 已训五维主干Readout适配 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development with isolated file ownership, or superpowers:executing-plans for controller-owned integration. Steps use checkbox (`- [ ]`) syntax for tracking. User's no-tests/no-smoke/no-hash/no-publish constraints override generic test/commit workflows.

**Goal:** 实际推进同一保留stage2-final起点的两层四臂Readout有界对照，不影响五维蒸馏参榜主任务。
**Architecture:** 原始完整题CPU编译为共享冻结manifest，GPU3短窗口提取共同冻结特征／适配独立LoRA分支，独立保存与恢复；质量与成本比较预先限定，不跑额外完整官方榜单。
**Tech Stack:** 既有Python3／Transformers4.57.6／PEFT0.18.1／Torch2.9.1／严格FA3／NumPy／PyArrow；不安装或升级共享环境。
**Spec:** [reports/readout-study-20261108/EXPERIMENT-DESIGN.md](EXPERIMENT-DESIGN.md) 第9节及 `HANDOFF-20261109.md`。

## Global Constraints

- 不停止、修改、重启、抢占主任务；不恢复原始标签扩大续训，不上传／发布／commit／push。
- 无测试、烟测、额外hash；只做生产准入、真实有界训练和登记的正式质量／成本比较。
- 所有新代码在独立 `/tmp/opencode/readout-adaptation-20261109/code/`，远端独立 `/tmp/opencode/jev-readout-adaptation-20261109`；不得修改共享脚本或其它实施者的文件。
- 同一stage2-final快照；A冻结选定LM行；Bcopy复制；Brand整体norm匹配；Cptr d128；两层各分支从原快照和未适配head独立起步。
- 同题／同token／同输入、原生causal、T1／无额外softcap／零dropout、FP32 head，原输入embedding冻结；最大255候选。
- 两层每臂最多5M完整prompt tokens，macro16/micro1，head LR1e-4／LoRA LR5e-5，seed42/headseed20261108；节点1M/3M/5M完整macro前缀。
- 数据20:80按token，五维任务等token；禁止dedup pair重建；留出仅称适配留出。
- 质量1000五维题＋100通用题；成本125完整请求；换序150题一次逆序；不扩张网格。
- GPU3仅短窗口600秒，主Student稳定200步至最后500步前且官方等待；每次重查无外部占用/预约，边界变化自有worker保存让路。

## Review Focus

- 原始noul／软标签／多工具不被强制单选，case/state隔离不泄漏到适配train。
- codebook及选项结束／decide索引真实绑定当前tokenizer，目标码不进入输入。
- stage2 LoRA完整加载、score头不混入新出口、两层/四臂独立初始化与恢复身份。
- 主任务phase切换、future GPU、暂停、重复启动和孤儿进程都不得导致抢占／误杀。
- 质量缓存与完整请求实测成本分开，K>255/context超限不隐藏，负结果如实记录。

---

### Task 1: 协议、共同起点与调度门槛

**Files:** controller owns `code/readout_common.py`、`code/run_readout_queue.py`、`code/admit_readout.py`。
**Interfaces:** common `build_codebook(tokenizer)->dict`、`compile_record(record,tokenizer,book)->dict`、`reverse_record(record)->dict`；记录字段见任务2。
- [x] 实时核对主任务：09:39 UTC仍Teacher，GPU0/1/2，GPU3空闲；官方等待，未动主任务。
- [ ] 快照核对stage2 adapter/base来源、rank32/config/tensor形状及原训练完成身份，不算模型hash。
- [ ] 实现固定共同模板／codebook／索引与只允许自有短窗口的资源门槛；当前Teacher阶段只做CPU。

### Task 2: 原始完整题CPU准备（独立实施者）

**Files:** implementer owns only `code/prepare_readout_data.py` and `task2-report.md`。
**Consumes:** Task1 common API，配置 `root/config.json`。
**Produces:** `data/train.jsonl`、`data/dev.jsonl`、`data/reverse.jsonl`、`data/DATA_READY.json`、`data/schedule.json`、`data/CODEBOOK.json`。

每条JSON记录字段固定：`id, case_id, source, domain, kind, state, question, options, target, source_location`；`options`为完整字符串列表，`target`为同长度合法原生概率分布，noul统一false/true顺序。编译后增加`input_ids, option_end_indices, decide_index, tokens`，或`unsupported_reason`；不支持记录不删除原候选。domain为knowledge/language/retrieval/tools/arts/generic。

- [ ] 从原始 `/home/dev/assumeengage/datasets/tasksource-jev-typed-decisions-ms/full/train-*.parquet` 和 `/home/dev/assumeengage/datasets/zero-shot-label-nli-http/data/train-*.parquet` 读取，禁止global-pool反拼。typed原字段state/kind/id/options/target/question/source/variant/split/group_id/question_id/license/license_use；原schema已读取。
- [ ] 每typed shard最多65,536行、每domain最多6,000完整题 reservoir；canonical总扫描最多100,000行、最多20,000题 reservoir；最多2 CPU线程和nice19/idle I/O。记录扫描上限与源文件元数据，不声称全库代表性。
- [ ] 合并同state与同来源case/group进行适配切分，raw官方test/dev/validation不进入适配训练；typed n=1 noul目标保留，不把多正choice强归一化；语义不明按整题登记拒绝。
- [ ] 任务每维200题留出、generic100；逆序每维30题，timing每维25题在dev标记。训练20:80按完整prompt token、每维0.8M/generic1M预算，完整题前缀不超5M；超4096训练整题拒绝但登记coverage，留出完整输入上限40960/K255。
- [ ] 同一物化训练清单供两层所有分支；以macro16写节点1M/3M/5M的真实题数/token/更新数。原始文本与token只留私有scratch，不进Git或公开报告。

### Task 3: 两层四臂工作器与有界正式比较

**Files:** controller integration/new implementation owns `code/readout_worker.py`；Task2实施者不得编辑。
**Consumes:** Task2数据、共同起点、codebook；产出每phase/arm的head/adapter/optimizer/RNG、完整曝光和质量／成本记录。
- [ ] 实现冻结特征生产与head-only适配；A固定0更新，物理共享前向如实统计，B/C新head可训练。
- [ ] 实现四个LoRA独立分支，从共同原快照和未训练head初值开始，不继承frozen后参数；原SEQ_CLS score不参与新出口。
- [ ] 每macro检查低优先级让路，保存自身状态并退出75；不操作主任务PID。600秒窗口上限，共同节点可恢复。
- [ ] 正式质量1000+100，换序150，同硬件完整请求成本125；pair参照作为完整方案另报，不称纯head消融或官方Index。

### Task 4: 审查、部署、阶段报告

**Files:** 本plan与 `EXECUTION-STATUS-20261109.md`，远端独立控制文件。
- [ ] 数据实现任务审查与整体只读代码审查；按用户要求不跑测试，以生产CPU准入和真实执行补证。
- [ ] 实际低优先级CPU准备；读取完整计数与共同身份，未完成不标ready。
- [ ] 挂接独立GPU3窗口队列并核对真实等待状态；合适窗口实际推进，不把排队当训练。
- [ ] 分别报告已准备/已训练/已评分、预算实际数、负结果与资源使用；不预填胜者。
