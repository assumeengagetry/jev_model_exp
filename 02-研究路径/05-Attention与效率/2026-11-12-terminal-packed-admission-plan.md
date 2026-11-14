# Terminal weights packed inference admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. 用户已授权自主真实研究／生产准入，不逐臂追问；no-extra-unit/smoke/hash/no-commit/push/cleanup优先。

**Goal:** 以原denseBCE/CE真实terminal权重各一次完整冻结D_select证明packed consumer的身份、完整性、旧数值合同和实际质量，不增加训练曝光。

**Architecture:** 新namespace两public模块：GPU consumer只读复用原8-source函数、独立packed context view和纯eval后权重比较；CPU ops install/run管理两GPU0/1租约/总预算/精确退出。完整execution与numeric readiness/eligibility分离，失败保留，不改变旧结果。

**Tech Stack:** 原Python/Torch2.9.1/Transformers4.57.6/PEFT0.18.1/nativeFA3/H8000/1，仅已安装依赖。

**Spec:** `2026-11-12-terminal-packed-admission-design.md`，已独立buildercheck无关键问题。

## Global Constraints

- Producers仅original-v2的C_D_BCE/C_D_CE-seed42真实COMPLETE.final_checkpoint，固定m3557/99824365token；旧source/cfg/checkpoints/optimizer/predictions不编辑。
- 新ROOT `/tmp/opencode/jev-terminal-packed-admission-20261112-v1`；LOCAL `/tmp/opencode/jev-terminal-packed-admission-20261112-local`。
- 只GPU0/1空闲+UUID+lease；不改/signal2/3或其新controller；新输出5GiB、保留旧future两25GiB+旧evidence5GiB+shared20GiB。
- 配置精度/stock/source/native/version/393FP32/served616237056/realT/head-before-EOS固定；0optimizer/backward/新学习token，无fresh训练语义冒充。
- 375case/615known/5190candidate/90unknown/full originaltokens/position/EOS/micro与producer完全一致，旧frozen数值阈值不改。
- 不开Dconfirm/public/新seed/Graph/cache/fusion/unit/smoke/hash，不发布/提交/删除资产。

## Review Focus

- audit393依赖requires_grad：保留flags，no_grad、.grad None；post-eval只比较不重载权重。
- producer arm与cfg/snapshot共享引用：consumer arm独立dict并单独packed bind_admission，原Context.check仍守护dense。
- evaluator main会进入trainer resume：仅调用evaluate_select函数，不构造Run或optimizer。
- 数值失败不是execution失败：所有375保留，logits/margin逐case4谓词+0带外flip才eligible，不能全局RMS掩盖坏case。
- 并发/资源：Popen抗中断登记+actualwait后关lease，postflock stat，不改已有nonemptylease，reserve不借用activeevidence。

### Task1：两模块source与独立review

**Files:** LOCAL/consumer.py、ops.py、implementation-report.md；ROOT同名部署。**Consumes:** 原8-source函数、真实terminal与原100Mreference、唯一stock与data readiness。**Produces:** install-only config/source封存；consumer CLI `--config --arm --output`；ops CLI `--action install|run`，完整execution/numeric/quality/weights receipts。

- [x] 先写真实合同断言，不造synthetic；实现严格terminal/source/config/checkpoint/currentnative/data身份链与sourcebytes/stat。
- [x] 无Torch提前import的consumer configureCUDA→build_stock→loadtrained393一次→独立packedview FullRow→完整evaluate_select；before/after纯weightbyte比较，0grad/optimizer，实际tokens/nativecalls/参数统计。
- [x] 原dense已存reference逐题对齐、原数值合同、原q质量独立回算；negative保留且eligibleFalse、不变batch或retry。
- [x] install-only ops新root独占+publicsafeFDupload，run仅0/1lease/UUID/总预算/exactpidfd/actualwait，源和配置冻结后不改。
- [x] 独立whole spec/quality+资源安全review，publication transient/hardlink预算两P2仅ops修复后scoped复审通过；主代理directread/AST及真实安装回读，未用sourcePASS代替运行。

### Task2：真实完整准入与归档

- [x] 实际install/readback、19:55:03UTC显式启动；0/1真实lease/UUID/预算准入后每producer一次完整375cases，2/3不动；不rerundense、不warmup/probe。
- [x] 完整输出/393参数before-after purebyte identity、native13132calls/1554484realT及quality回算真实通过。数值准入两臂均NEGATIVE：各34bad cases，BCE1/CE0flip，0带外flip；main从savedraw独立重数一致，eligibleFalse。14聚合和2actual源归档，private预测/weights不下载。
- [ ] 未经过新的candidate/confirm/public冻结协议不能自动选winner/openconfirm/public；numeric失败不放宽，保留原全部资产。

## Self-review

设计每段映射Task1/2；接口只围绕同一consumer闭环，两个文件共同一次review。Requires-grad与pureaftercompare、shared Context引用、eval主入口、逐casegate及GPUownership五风险均有真实运行断言。所有toy/commit/cleanup默认步骤显式不适用。
