# Clean State / Rank Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. 用户已授权自主排队，显式no-extra-unit/smoke/hash、no-commit/push、保留旧资产优先于技能默认测试/提交/确认步骤。

**Goal:** 接入两个隔离的fresh探索臂，正式macro0外部验收后才续跑同99824365-token完整训练。

**Architecture:** 以远端v2实际冻结8份源码为基线，新namespace新增state_boundaries.py；JPAIR纯loss与state reader独立实现和review，再由主代理集成严格配置/Pair/实际dispatch记账。外部saved-artifact primary与GPU2/3 side queue不改active controller。

**Tech Stack:** Python3.11、已装Torch2.9.1/CUDA12.8、Transformers4.57.6、PEFT0.18.1、native FA3、H800；不安装。

**Spec:** `2026-11-12-followup-state-rank-design.md`，含builder-check后的有限科学续跑裁定。

## Global Constraints

- STATE_NL_PB_P_CE=prefix_block/packed/CE；JPAIR_P_BCE_RANK=causal/packed/BCE_RANK_HARD，λ=τ=1。
- 新root `/tmp/opencode/jev-clean-trainer-20261112-followup-v1/`，本地 `/tmp/opencode/jev-clean-followup-20261112-local/`，原v2绝不修改。
- fresh重新从stock生成，之后才对比v2 oracle393逐键bytes；fresh_init/checkpoints原样复用，不放宽trainer_code绑定。
- 16-case宏、原slot/micro/order/IDs/position/EOS、unknown不forward、F_known/C、3557calls/3556positive-LR、三select节点不变。
- 所有实际失败/partial/checkpoint保留；无额外unit/smoke/模型语料hash或自动commit/push/发布/清理。独立源码review不等于runtime passed。
- 不开confirm/public；原controller GPU0/1 BI/PB预留不动，四基础臂完整退出后才考虑GPU2/3。

## Review Focus

- host near-one soft q经FP32变1：硬集合只在原host值判断；Task1独立源码review和Task4真实margin验收，未出现分支记未覆盖。
- 重复case occurrences／unknown／K1：不按cid合并F/C，不把unknown作负例；Task1/2/4核对完整原字段与首macro覆盖，不制造toy分支。
- train/select roster和原split不同：reader显式train→train、select→validation，坐标/ordinal/wholefields硬匹配；Task2全量真实store readback。
- 首macro两遍共同错mask：Task3记录实际FA3 cu/QKV/flags，Task4独立预期分区核对；不冒充跨后端/邻居证据。
- GPU空闲检查竞态及磁盘旧队列预留：Task4租约持有、前后进程身份、全已知预算核算，非合作外部竞态只暂停自有工作、不抢卡。

### Task1：纯BCE＋硬标签排序目标

**Files:** 新本地 `source/objectives.py`；不编辑基线。**Consumes:** `coefficients(actual_margins, fields, cases, kind)` 实际CPU缓存、原host target、F_known；**Produces:** 原三元组coef/loss/bookkeeping，新增kind BCE_RANK_HARD，保留BCE/CE。

- [ ] 先写清实际验收断言与精确component/coverage/coef尺度JSON合同；由Task4使用，禁止额外unit/smoke。
- [ ] 用λτ1、host硬集合、零rank保留分母实现objective/coefficients；component autograd只作用实际margin leaf，不引入模型forward或顶层torch。
- [x] 独立readonly spec＋quality review通过，主代理直接读源码；只读源码通过不签新模型macro。
- [x] 新目标一次真实saved-mechanism：原控制完整macro0实际缓存16case/19field/60candidate，18rankactive/1zero，2soft；独立loss/总coef逐位相同。0模型forward/0更新/0新曝光；不是synthetic/unit/smoke，也不代替新臂正式macro0。

### Task2：无损state边界reader

**Files:** 新本地 `source/state_boundaries.py`。**Consumes:** cfg `state_boundary`={root,complete,primary}三个明确成员；原CaseStore train/select。**Produces:** `bind_prepared(cfg)->dict`含完整守护pins/code；`BoundaryStore(binding, train, select)`；`for_case(case, *, dev=False)->list[int|None]`全部字段切点；`check()`；`close()`。

- [x] 实现COMPLETION/primary/source/config/input/sourcebytes全身份链验证，固定v2policy及完整ready；仅CPU/json，无torch/tokenizer。
- [x] 只读逐条索引实际sidecar，与原全CaseStore case/ordinal/coordinate/known/candidate/length/EOS对账；known cut必须0<cut<len(originalP)，unknown四项null且不解释raw；重复/缺失/额外行硬失败。
- [x] 独立readonly spec＋quality review无actionable finding；主代理新reader真实全量验收16.8386秒，24600train/375select及全部known/unknown/candidate/row完整通过；实际16个正式train查询各重复＋全部375select查询通过。无Torch/tokenizer/model/confirm，不重跑边界投影；人为损坏分支未注入，未声称全分支运行覆盖。

### Task3：新配置、Pair共享入口与正式调用证据

**Files:** 新 `source/bindings.py,case_runtime.py,train_case.py,evaluate_native_dev.py,model_runtime.py`；fresh_init/checkpoints与基线逐字节相同。**Consumes:** Task1/2接口。**Produces:** 仅两新arm可启动的Context；`slots(cases,*,dev=False,boundary_store=None)`；runtime.boundary_store显式None/实例；正式macro0 dispatch receipt。

- [x] 从远端冻结v2复制并逐字节核对仓库snapshot，保留原8文件快照单独目录。
- [x] bindings源码集增加reader、version及精确arm/λτ/policy验证；state_binding进入snapshot，Context.check持续守护。未建fresh时仅允许原pending-init生成配置，训练仍validate_init严格。
- [x] main在torch前初始化reader，train/select共用slots；known无损重切且row/EOS/length断言，未知不动；rank边界显式None。源码接线通过独立review，实际新模型尚未运行。
- [x] 正式macro0两pass的CountedNative捕获代码已独立review：实际QKV shape、cuQ/cuK/max/flags，独立dispatch文件，平时不堆积per-call日志；原count/runtime和two-pass/actualstep逻辑不改。实际macro捕获仍待运行，不能以此勾模型机制通过。
- [x] 5文件676行集成diff独立review无P0/P1/P2，9源AST语法通过；先封存source并真实CPU fresh0forward，之后加载旧fresh393oracle逐键bytes完全一致，finalconfig冻结。独立primary源码也已完成并review，但其真实saved-artifact验收仍须等正式macro0。

### Task4：外部primary、空卡队列、正式macro与冻结dev

**Files:** 新本地/远端 `ops/verify_first.py,side_queue.py,deploy.py`及progress.md。**Consumes:** 两新臂scope/fresh、旧C_P_BCE/C_P_CE首macro产物、基础四臂完整完成和退出证据。**Produces:** 外置每臂FIRST_PRIMARY_VERIFICATION/最新cursor1 pin，QUEUE状态/OWNER/STOP/失败留痕及全保留训练链。

- [ ] 先实现saved-only独立primary：完整输入/几何/初态、JPAIR旧logits相同、STATE精确cut与actual FA3分区、独立loss/coef/scales、393raw/clipped/delta/Adam/scheduler/checkpoint/cursor核对；未覆盖分支显式列出，不从finite推断梯度数学等价。
- [ ] 实现inotify/pidfd等待原四臂COMPLETE及exact退出；GPU2/3租约跨首macro/审核/续跑持有并记录UUID/exact身份；外部竞争只停自有任务，无自动重试。
- [ ] 资源准入：现存文件不删；保留旧BI/PB各25GiB、两新各25GiB、fresh/evidence5GiB及20GiB共享reserve，不能同时容纳则串行或BLOCKED，不能把own RUN.lock叫GPU锁。
- [ ] 新source/config/ops全部readback封存；先正式macro0 pause cursor1/exit75，主代理直接验收外部报告与关键实际产物，再显式resume index1，禁止回填cfg。
- [ ] 三节点完整select+100M退出后独立saved-only回算及配对case分析，只归档聚合/实际源码；无confirm/public或winner声明。

## Self-review

设计每节均映射Task1–4；类型/接口统一；五项Review Focus在对应真实reader/macro/readback gate覆盖，未出现分支仅源码review不称测试；无synthetic测试或额外模型调用。状态以progress.md逐项实际证据为准，不预勾选未执行步骤。
