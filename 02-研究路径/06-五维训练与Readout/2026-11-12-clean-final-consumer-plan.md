# Frozen clean final consumer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development；独立文件组可按dispatching-parallel-agents并行，主代理集成。用户自主授权/no-extra-unit-smoke-hash/no-commit-publish-clean优先。

**Goal:** 对已实际封存baseline CPCE与challenger JPAIR做input/loader准入、一次完整250case确认及各140620完整public，不再选模。

**Architecture:** 现有frozen ROOT下新独占source目录，四模块分工common/backend与evaluate/ops；固定API降低跨代理共享状态。先source review/真实install/codec，再confirm与独立回算，再同一预声明两模型依次四卡public。所有层级和失败分别保留。

**Tech Stack:** 原Python3/Torch2.9.1/CUDA12.8/Transformers4.57.6/PEFT0.18.1/nativeFA3/官方SDK source snapshot与Suite0.3，现有H8000–3；不安装。

**Spec:** `2026-11-12-clean-final-consumer-design.md`；selection规则见已冻结`2026-11-12-terminal-candidate-confirm-public-policy.md`；builder-check无未决重要问题。

## Global Constraints

- ROOT `/tmp/opencode/jev-clean-final-eval-20261112-v1`已存在policy/freezer/FROZEN_CANDIDATES，禁止重建/修改；LOCAL `/tmp/opencode/jev-clean-final-eval-20261112-local`。
- 只manifest baseline C_P_CE与challenger JPAIR、causal/packed/stock＋trained393一次载入、pureaftercompare/nooptimizer/noTeacher/no跨benchmark路由。
- confirm全250/nativevalidation/各域50，unknown不forward、unsupported保留敏感性，不能改名select绕过旧接口。
- public每role完整140620/4×35155，官方排除442后140178全ok；resume=False/warm=False/limit=None/keep=None，官方Suite strict+exclusions校验保留。
- 正式首用成本计入真实请求，no syntheticwarmup/unit/smoke/extrahash/Graph/cache/fusion/dummyT，不复用旧模型readiness/预测。
- source安装独占source目录，现有候选不改；GPU4 cooperative known-successor/UUID/exactpidfd/actualwait，旧metadata不写、无外来信号。
- 新cap20GiB＋全程固定额外future20GiB＋shared20GiB门禁（currentfree已经含实际已用，故保守）；所有旧文件保留。

## Review Focus

- SDK.run先importTorch：worker必须先backend.configure_role，Engine只接受同进程预配置role，不在constructor再调用strict初次配置。
- 一进程混入8/9source模块：每GPU worker单role，manager/codec不importproducerTorch；helper路径/bytes验证且producer Context不改。
- resume=False仍append：所有SDK输出目录/results必须新建，禁止已有partial复用；fullmerge逐条payload/modelid/runid核对。
- 确认不能重选public列表：仅一个冻结contrast，confirm与public结果不改变两角色/checkpoints/runtime。
- SDK status.json原子replace：cap独立uniqueinode与精确tmp消失规则，不递归把mutablestatus/historyledger当静态身份，不以ownused减少未来预留。

### TaskA：common/backend（独立文件所有权）

**Files:** LOCAL/source/common.py、backend.py、task-A-report.md。**Consumes:** 已封存候选及原producer code/checkpoints，现有SDK public source。**Produces:** shared API（见LOCAL/interface.md）、纯stdlib Guard/install_config、role预配置、trained393模型与purefield codec、SDK ducktype engine。

- [x] 断言先写，身份/static/history/known-successor区别及四源/SDK snapshot源码实现；install不读confirm/public数据。实际source安装/封存仍待TaskC。
- [x] 单role context→首次Torch配置→trained393一次load→nativefullrow，purebefore/after/no.grad/0optimizer；统一renderer/micro/fieldsoftmax与APIchoice/noul的源码实现完成，尚未实际运行。
- [x] 科学/fullrow/dev来源与新consumer/public资格分开，不把producerallow_training当public-ready。
- [x] TaskA独立源码审查与MAIN AST通过；热路径历史body放大P1在fix1完成分层后scoped复审判ADDRESSED。最终四源接口集成及actualsizefit/runtime仍由TaskC验证，不以源码结论声称latency或consumer通过。

### TaskB：evaluate/ops（独立文件所有权）

**Files:** LOCAL/source/evaluate.py、ops.py、task-B-report.md。**Consumes:** 同一固定API，不改A文件。**Produces:** CPUcodec全原select ID对账、confirm GPU/独立CPU回算、Suite/shard/officialrun/merge-score、install-only/one-shot manager。

- [x] install-only源码实现：已有ROOT/source独占claim、四public源/main-review acknowledgement/byte readback，候选不改。实际source安装仍待TaskC整体门禁。
- [x] codec615/5190/90与独立confirm250/50quota/真实分母/paired10000bootstrap源码实现；不public换候选。实际输入/确认尚未执行。
- [x] 官方Suite strict/exclusions、四完整shard、SDK调用顺序和fresh results的源码实现；无warm/resume/limit/keep、无partial成功伪装。实际Suite/public尚未执行。
- [x] 四GPU known-successor/预算/登记/actualwait及阶段调度源码实现；资源扫描、重复信号和durability/temporary名单在fix2补齐，尚未实际取得lease或启动worker。
- [x] TaskB独立spec/quality及资源安全review：hostfix1、resourcefix2 scoped均PASS；MAIN AST通过。最终四源整体集成正在TaskC审查，所有源码结论不冒称runtime。

### TaskC：集成、一次真实确认与完整public

- [x] 全四源3133行broad review无新P0/P1/P2，所有scoped已收齐；actualinstall/readback通过，config47824bytes/SOURCE_SNAPSHOT1802947bytes/小源码1206155bytes实际fit，候选及SDK/source/model来源冻结。0forward/worker/confirm/publicpayload，未新增unit/smoke。
- [ ] 主代理显式one-shot启动，读真实codec/393 loader/250确认与独立分析回执；必要能力阻断保留而非换model。
- [ ] 两predeclaredrole各完整public、官方140178计分与独立合并/元数据/唯一性对账；score/系统效率/单seed限制分开。
- [ ] 聚合与actualsource归档，privatepayloads/q/predictions/tokens/weights留远端；未通过层级保持unchecked，不publish/commit/cleanup。

## Self-review

任务A/B只共享冻结API不共享编辑文件；所有source正向安装/身份/模型/输入/250/140620/140178/资源退出层级都有独立实际门禁，五项Review Focus分别映射A/B。source/候选先冻结、Dconfirm只确认、公测不再选模型；无未授权外部副作用。
