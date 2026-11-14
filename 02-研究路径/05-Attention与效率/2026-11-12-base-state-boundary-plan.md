# Base-inspired state-boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax. 用户已授权自主排队、不逐臂确认；显式no-extra-unit/smoke/hash及no-commit/push约束优先。

**Goal:** 用严格输入不变的state后切点对照已跑的Candidate切点，研究0.6B两块双向mask的学习效果。

**Architecture:** 在新`/tmp/opencode/jev-base-state-boundary-20261112-local/`实现CPU-only准备器与独立sidecar；仅准备train/select，不改DATA_READY。准入后另建trainer/source/config，保持当前v2及其队列完全不动，复用原可微FA3 core，正式首macro验收后才完整训练。

**Tech Stack:** 已装Transformers4.57.6 fast tokenizer／Python3.11，Torch2.9.1／PEFT0.18.1／nativeFA3，仅现有H800；不安装。

**Spec:** `2026-11-12-base-state-boundary-design.md`。

## Global Constraints

- 原stock+fresh r32/alpha64/FP32二类head，393keys，不能沿旧adapter/optimizer/Teacher缓存warmstart。
- 所有row token／position／EOS／candidate order／完整case／q／F_known/C不变；无精确BPE切点则整版不ready，不能删除case/field或round边界。
- 只读原24600train/375select、原helper/tokenizer；D_confirm/public不打开，unknown不tokenize或forward。
- 完整训练56912case occurrences/99824365logicalpairtokens/3557optimizer calls，最后LR0；不能重吃首macro。
- 新源／sidecar／输出独立stat+source bytes绑定，无额外unit/smoke/model-corpus hash；原失败全部保留，无自动提交、清理、推送或付费。

## Review Focus

- BPE token跨state字符末端、Unicode重复offset、special-token offset：不得把非精确token边界签ready。
- 原prefix是独立tokenize而不是整pair tokenize：必须逐ID核对原prefix，不能引入新的联合分词。
- 非空state但无question、criteria条件附加、unknown字段：只按原renderer位置切，不能搜索用户正文中的Question/Candidate字符串。
- 恢复source/checkpoint后scope身份、实际pair重拼与micro几何不变，新增sidecar必须持续stat守护。
- 新state-P不能沿Candidate-P cached-serve准入；P可训练图、GQA/cu隔离与首正式macro验收必须保留。

### Task1：实际完整输入的精确切点准备

**Files:** 新`prepare_boundaries.py`、`progress.md`；远端新`/tmp/opencode/jev-base-state-boundary-20261112-v1/`；不修改v2 source。

**Consumes:** 原v2冻结config/data pins，原helper`text/criteria_context/render_pair`、stock fast tokenizer、原train/select JSONL。

**Produces:** `state_boundary(prefix_ids:list[int],prefix_text:str,state_end:int,offsets:list[tuple[int,int]]) -> int`，接受谓词严格采用Spec“Builder检查后的接口裁定”；私有`{train,select}-boundaries.jsonl`版本base-state-boundary-v1，关联`(roster,case_id,field_index,coordinate)`，包括unknown null条目及全部known候选数/原pair_lengths；`PREPARATION.json`／`COMPLETE.json`，complete与ready独立且后者仅全量精确切点及回读通过才为true。

- [x] 实际输入验收断言已实现，renderer/prefixIDs/精确offset/row重拼/EOS/全部field合同；没有toy/smoke。
- [x] 独立review的3P2修正后，完整CPU投影实际100.6339s／峰值RSS201367552bytes；ready=false：train28676/41674 knownfield、select427/615跨BPE。完整结果与所有失败身份保留，没有forward/训练。
- [x] 主代理独立端点枚举算法实际全量回读47.5031s，确认全部原case/sidecar/失败身份/计数/source/config/data/output绑定，ready仍false。仅聚合及实际源码归档。

**Task1实际裁定：v1扫描完成但科学notready，本版本Task2/3阻断，不删除任何case、不给切点取整、不改写此失败。**

### Task2：新state边界训练接线与fresh

**Files:** 另建新trainer版本，不改当前8份source/config；sidecar reader单一职责，持续stat守护。

**Consumes:** Task1真实ready及精确字段接口；原C_P_CE同曝光控制、v2共同fresh张量oracle（只比较，不沿训练权重起跑）。

**Produces:** `STATE_PB_P_CE`独立可调用入口、源绑定、fresh manifest、正式macro0 artifact合同。

- [ ] Task1不ready时停止本版并记录最小科学替代方案；不猜接口继续写不可执行配置。
- [ ] 同seed新fresh393张量对账；原CaseStore/Exposure/objective/checkpoint精确合同与两遍autocast/replay保留。
- [ ] first macro验收器先写完整原真实16case rows/position/slots/micros只改boundary的断言，正常正式执行到cursor1暂停。Expected：验收主代理实际回读通过，再显式恢复index1；失败保留，无阈值放宽。

### Task3：空卡队列、冻结dev与最终报告

- [ ] 只在已有基础四臂真正完整退出且自己的目标GPU空闲后启动新臂；不得重启/修改原controller。首macro准入是完整训练前提。
- [ ] 同schedule三个D_select节点完整375case评测、真实计数及独立paired-case分析；候选冻结后才D_confirm/public。Expected：真实节点或明确failure；没有partial覆盖冒充完整public/方法提分。
- [ ] 一次whole-new-trainer独立review、实际验证证据归档；未完成步骤保持unchecked，不commit/push/清理。
