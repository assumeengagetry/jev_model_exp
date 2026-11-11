# 通用＋五维联合续训 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: executing-plans for controller integration; isolated implementation delegation is allowed for independent files. User's no-tests/no-smoke/no-hash/no-commit/no-publish instructions override generic skill defaults.

**Goal:** 保持当前五维KD训练完整结束，从其final联合续训通用原标签+五维原KD目标，再对新固定final运行唯一一次官方public0.3。
**Architecture:** 新独立命名空间、六路数据、旧五路逐索引完整回放、独立原始generic流；复用原严格原子trainer并保留权重，用来源完成/GPU空闲门槛串行训练，最后评测绑定本段final。
**Tech Stack:** 既有Python3、NumPy、PyArrow、Transformers4.57.6、PEFT0.18.1、Torch2.9.1+cu128、strict FA3、官方Decision Index固定commit。
**Spec:** `docs/superpowers/specs/2026-11-09-joint-generic-continuation-design.md`

## Global Constraints

- 不停止当前训练；只延后旧official等待器，不恢复原始标签扩大训练，不抢外部GPU。
- 只读旧pool/目标/关系/模型；新代码/状态在独立namespace，不改共享实现。
- task完整1250308pair/366985783token；generic<=91746445token；新段总<=458732228token；有效token20:80，非按pair或loss权重。
- generic原始标签、无Teacher；五任务原有效KD目标不变；原3647关系/gamma0.5/lambda0.05不变。
- retained source final，rank32/alpha64，LR2e-5，batch128，micro32/16384，AdamW0.01，warmup.03+cosine，strict FA3，GPU0。
- 不额外测试/烟测/hash/评测网格；正式public完整一次绑定新final，不发布上传提交。

## Review Focus

- 六路runtime索引/sentinel与旧五域顺序/target关系不会错位；不改共享DOMAINS。
- generic原始target与token组装、重复/冲突/任务重叠处理透明，不均值化或阈值化原标签。
- 来源final与本段resume身份严格，不能误用共同未训练init或其它模型/优化器。
- 旧评测不自动跑，新评测消费本段fixed final，状态/锁/取消/孤儿不误接旧scope。
- 完整样本预算、真实tail/更新数/FA3前后向/完成标记，NFS/磁盘不足保留阻塞不清理旧模型。

---

### Task 0: 后续队列优先级移交

- [x] 读取实际状态13:33UTC：当前Student5780/9769仍活跃，official仅等待且无child/结果。
- [x] 13:34UTC核实旧official324276身份/cmd后SIGTERM并确认退出，原训练327929身份未变；archive和明确deferred记录已写，不伪造完成。
- [x] 13:54UTC原official入口deferral guard已部署并逐字比对，原源码备份保留；不修改原trainer/config。

### Task 1: 六路runtime与原始generic CPU准备

**Files:** `/tmp/opencode/joint-generic-continuation-20261109/code/joint_data.py`、`prepare_joint.py`（controller内联实现，独立namespace）；生产结果记录到本段报告和ledger。
**Interfaces:** `STREAMS`、`PairPool`、`TokenStore`与旧同签名；`prepare_joint.py`生产入口，输出pool/config/targets/plan/READY。

- [ ] 原五路逐索引只读复用数据/有效目标/关系，真实token总量核对。
- [ ] 原始canonical采样/标签/组装/任务输入重叠与重复准入；generic有效token完整前缀<=task/4。
- [ ] 六路数组及独立token offsets/sentinel、真实N/尾批/步数、完整原子plan与targets READY。
- [x] 14:16UTC生产CPUprepare343155实际nice19/CPU0-1/CUDA不可见，独立身份/锁/原子写入；没有运行额外模型/测试。准备仍进行中，DATA_READY未完成。

### Task 2: 保留final的低LR训练与调度

**Files:** `code/train_joint.py`、`run_joint.py`、`joint_control.py`（controller所有）；后者集中只读来源绑定和完成校验。
- [ ] 最小fork当前strict trainer：父权重为source fixed final，不再要求untrained init；新fresh optimizer/scheduler，参数不重置。
- [ ] 六stream计数/计划目标身份/尾批/关系归一，保留checkpoint/执行计数/RNG和自己的暂停/锁。
- [ ] coordinator等待source真实完成/原worker退出/GPU0空闲后开本段，进程身份与原source永久绑定。

### Task 3: 新final官方完整评测与本地材料

**Files:** `code/run_official_joint.py`、提交包/证据/监视器。
- [ ] 复用已审 portable Engine与旧strict-import suite，来源绑定改为本段final，不用旧candidate分。
- [ ] 实际基座/adapter/tokenizer/code只读快照、启动/完成来源复核、一次完整pipeline/完成幂等。
- [x] 独立只读审查所有变更并核实修正2Important；仅生产准备/真实训练/正式公共输出作验证，不加测试或smoke。当前真实父模型/Engine加载尚未发生。
- [x] 14:16UTC部署并启动新coordinator343153，归档review-v1、逐字比对最终9源码；只读监视器已移交，发布仍待用户确认。全链训练和正式评分结果尚未完成。
