# 0.6B研究R1：四卡低LR因果桥实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** R0完整核验后，用相同KD父权重、相同数据／targets／宏顺序和新四卡执行路径，完整训练joint LR2e-5并完整public，隔离此前50倍LR与多卡实现混杂；这是突破24.91之前的因果桥，不是长期研究终点。

**Architecture:** 全新root `/tmp/opencode/jev-06b-research-20261112-r1-bridge`；只读复用原冻结数据／四份官方请求输入，独立KD快照、优化器和checkpoint。旧LR1e-3四卡trainer／eval控制源码显式版本化为JOINT_BRIDGE及joint_bridge，新增只读训练观测和rank一致性门禁，不改目标／clip／归约算法；GPU阶段持锁严格训练→完整public。只有主代理在读取R0完整实际结果后生成的R0_DECISION允许启动。

**Tech Stack:** 原Torch2.9.1／PEFT0.18.1／Transformers4.57.6／FA3 deterministic／Triton3.5.1／4×H800；无安装／外部付费资源。

**Spec:** [已批准研究设计](2026-11-12-06b-top-public-research-design.md)，用户明确自主迭代。此计划在R0通知等待期准备，不意味着已决定跳过数值问题或已启动R1。

## Global Constraints

- R0未完整核验、KD-step0执行对照未检查、或存在尚未排除的重大路径异常时，不启动本轮训练；已有R0部分结果不作选择依据。
- 主代理先作明确条件式决策：若BF16／v2是主要可恢复问题，先处理统一推理候选；若各路径均低且控制成立，再执行本桥接。不得根据想要的结果改写R0证据。
- 唯一训练超参对照是LR从1e-3恢复2e-5；同KD snapshot、1667242pair／458731913token／13026步／3647关系／global128／seed42／微批≤32及16384padded token、原0.03warmup+cosine／decay0.01／clip1不变。
- 本轮不重跑KD public；引用既有完整KD receipt作为source-bound前置，不把拷贝旧receipt伪装成新评测。新训练scope绑定真实KD_PUBLIC_REFERENCE、R0_DECISION及新config／CODE／SOURCE／DATA_READY。
- 完整public为140620执行、排除442后140178计分，全候选完整state，无预测混用；只有joint_bridge一个新stage。四核继续封存v2，模型名／来源独立，官方kit不修改。
- 不恢复任何旧暂停任务或旧optimizer；新root不覆盖／删除旧权重／checkpoint／partial。失败保留，有限值异常不降LR、不丢batch。
- 代码库外隔离，研究Git只增计划／报告／聚合证据，不commit／push／发布。沿用no-extra-unit/smoke/hash，实际训练／对照／完整public为研究本体。

## Review Focus

1. 旧KD public只作原始参考而非新root假完成；新训练不能误用旧joint或高LR final初始化。
2. 新observability不能改变计算图目标、FP32梯度SUM／global分母、数据顺序或scheduler；最终不能只靠有限loss声称所有rank一致。
3. point／relation分量、head差分／LoRA变化、clip前范数按同一实际global macro解释，不能把稀疏macro loss与旧窗口loss混称。
4. 所有train rank及eval worker有精确parent-death／有界取消；log／STOP写失败不阻止清理，正常退出竞态不误杀其它自有臂。
5. SOURCE清单自身、外部引用及CODE封存绑定，不接管未知partial；public合并／原生score必须绑定当前joint_bridge结果。

## Task 1：独立生产准入与父参数／R0引用

**Files:** 库外R1 `code/{experiment_common.py,prepare_bridge.py}`，root `SOURCE_BINDING.json`、`KD_PUBLIC_REFERENCE.json`、`R0_DECISION.json`、`kd-frozen/`、`train/config.json`、`DATA_READY.json`。

**Interfaces:** common `verify_ready()`、`verify_kd_reference()`、`verify_training_complete()`；config路径仍`ROOT/train/config.json`，arm为JOINT_BRIDGE。四份Suite rows引用原已verified shard输入，绝不引用旧predictions。

- [ ] CPU核实原代码／数据／父参数／既有KD receipt和R0完整SOURCE／DATA binding，独占创建新来源清单。
- [ ] Freeze同KD父快照与旧配方，LR精确2e-5；验证N／tokens／宏／关系、rank原子覆盖，保存实际预算和所有引用metadata。
- [ ] 正式CPU准入通过才生成DATA_READY；Expected：原same math／同global128、KD父、R0决策可查，来源无变化。

## Task 2：四卡trainer与只读观测

**Files:** 库外R1 `code/{train_bridge.py,atomic_ranks.py}`。

**Interfaces:** `python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 train_bridge.py --root ROOT --config ROOT/train/config.json`；输出train/JOINT_BRIDGE及TRAINING_COMPLETE。

- [ ] 从实际成功四卡trainer显式版本化ROOT／arm／LR／scope，保存exact393父参数，重建AdamW；仍局部loss sum除global实际macro后FP32 SUM，再clip／step／scheduler。
- [ ] 正式progress保持原总loss，并增加point／加权relation sum、clip前范数；checkpoint节点记录head差分范数、LoRA变化和全rank参数一致性（非hash），不改变loss函数权重。
- [ ] 每1000步／最终保存所有rank RNG／FA3证据；参数不一致或非有限停止并保留，不发布完整receipt。正常无OOM训练预计约1.5小时，只是旧实测预算锚。

## Task 3：单新stage完整public与全链收口

**Files:** 库外R1 `code/{eval_public.py,run_bridge.py,inference/experiment_engine.py}`及4个原样数值核心。

**Interfaces:** `eval_public.py --stage joint_bridge [--worker 0..3]`；train完整验证后才启动，输出eval/joint_bridge/COMPLETE与ROOT/COMPLETE。

- [ ] 来源／OWNER／scoring提交门禁、逐完整请求shard合并和官方原生计分沿用已修正流程；stage名与源final不能串用。
- [ ] R1实现整体fresh只读审阅、修正Important后封存新CODE；不让代码在DATA_READY后原地变化。
- [ ] 正式新鲜四卡准入启动；只有真实优化步／正式已完成请求后报运行中。
- [ ] 全量训练及public完成后直接验证N／tokens／steps／relations、rank一致性、140620／140178与旧源身份，再决定R2具体冻结数据和loss；不把24.91附近恢复当最高目标已达。

## 自主执行边界

本计划只授权既定桥接的一遍训练与一次完整public，不包含自动LR网格、checkpoint择优public扫描或调参至侥幸高分。R0完成前可以准备计划／资源，但R1实际启用需要主代理写下基于完整证据的R0_DECISION。后续R2及结构对照继续按获批设计自主分轮，不向睡眠中的用户逐项请求确认。
