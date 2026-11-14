# 0.6B级Public最高分探索：前期研究

**状态：R1完整训练与新public已完成，07:21:39UTC直接全量复核Public24.67／Raw43.84，未超过历史24.91。另发现历史xLAM已选训练request与94个公开ToolRet query完全同文，涉及98请求；这是输入案例暴露，不能把旧谱系称为已证明干净的冲榜基线。R2尚未开训，下一参榜分支需改为原基座上的全新LoRA／head。**

用户目标是持续“找原因—做对照—完整public验证—再迭代”，不是只把高LR降回去。当前可保留的较优模型仍为原joint的Public24.91；KD24.38，高LR新joint0.81均已有完整证据。

**公平性限定：**以上历史分数仍是其实际模型在完整public上的原生计分，不删改、不虚构扣分；但不再把它们说成“本项目微调未暴露评测案例”的结果。94个同文query尚未穷尽比较候选／schema／label，不能据此断言整榜分数虚高或人为作弊，也不能生成所谓“纠正后的clean分数”。

## 当前靶标不是Kai的16.68

主代理直接解析官方固定快照的298条总分／元数据，筛得44个公开`served_params<=0.7B`且>0的条目，其中最高Public为 **GLiClass v4.1 edge，51.99**。Sieve实际约0.59B、Public32.26是可用的跨架构中间目标，不是同规模最高。参数未知的Jev条目不能擅自放入此分组。

现有Qwen分类模型的逻辑服务参数为596049920骨干＋20185088 LoRA＋2048 active head＝**616237056（0.616B）**。文件中还保存未被分类模型使用的LMhead；不能把这个unused head加进去误称0.772B，也不能只报20M trainable参数。直接下载来源是ModelScope的`Qwen/Qwen3-0.6B/master`，不是已证明相同初始化的Kai `Qwen3-0.6B-Base`。

官方0.4.1主榜按私有Full排序，public权重为0；本地自主研究只能验证public目标，正式线上名次仍需用户另行授权发布、维护者私测。不能承诺凭本地public高分证明private冠军。

## 原因研究的三条新证据

1. **目前未找到足以单独解释0.81的明显训练实现bug。** 独立只读架构检查支持数学global归一化、原子关系、effective target和父参数加载闭合，但没有动态等价证明。稀疏loss有限并不证明候选区分能力正常。
2. **原readout dev不是历史未见集。** 其manifest明确写历史trunk暴露未排除。新的native validation15550题经初步精确state／已存group扫描，107题碰到已选训练state、150题碰到public state；闭包排除257题后剩15293题仍未正式准入，旧dedup丢失的多来源group和语义近重复仍需核查。
3. **高LR很早已大幅重写参数，但尚未证明唯一根因。** 只读CPU参数统计显示，高LRstep1000的head两行差分范数由KD0.80818降到0.17634，final0.24075；旧低LRjoint仍0.80834。新final LoRA B总范数为KD的3.566倍，旧joint为1.045倍。同时训练／v2均是在BF16投影之后才转FP32作两类差分，FP32保存权重不等于FP32执行，须用执行路径／head精度对照排除数值交互。

## 拟定首轮，不先盲扫学习率

- **R0，不重训：** KD、旧joint、高LRstep0／1000／final，同完整题诊断集比较封存v2、标准PEFT BF16读出、同hidden的FP32读出。区分权重／表示损坏与部署精度损坏；既有dev只作机制诊断，不宣称泛化。
- **R1，因果桥：** 从同KD起点，新四卡原完整joint配方LR2e-5，单独隔离四卡路径与50倍LR；补point／relation、梯度范数、head／LoRA变化和rank一致性证据。
- **R2，突破方向：** 当前较优joint上，同合法完整题、同预算，对照题均pair BCE与完整field CE；多工具仍是独立真假字段，unknown不当负例。候选先经新非public验证选择，再冻结做完整public。

后续才根据证据晋级排列等变候选交互、受控attention、rank扩展或低LR全参数训练；不把已做且换序不稳的Cptr原样重跑，也不编造强参照未公开的训练配方。

设计在[研究设计](../../../02-研究路径/06-五维训练与Readout/2026-11-12-06b-top-public-research-design.md)；聚合证据为[资产／暴露／参数侦察](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/PREFLIGHT.json)和[榜单靶标](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/LEADERBOARD_TARGET.json)。这些不含原始题目、标签、模型权重、凭据或编码源码包。

全部旧资产保留；R0所有自有进程已退出、四卡显存0，无新Public分数、无提交／推送／发布。R0实施计划见[执行路径／读出精度诊断](../../../02-研究路径/06-五维训练与Readout/2026-11-12-06b-research-r0-implementation.md)，启动证据见[R0_START_OBSERVATION.json](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/R0_START_OBSERVATION.json)。

实际root为`/tmp/opencode/jev-06b-research-20261112-r0-v2`。首版准备目录因Python／libc未导出pidfd API被启动门禁拒绝，未运行GPU；保留不覆盖。独立review提出的source-manifest封存、退出竞态、异常取消和watcher超时问题在v2定稿前修正。v2使用已对照内核／头文件的Linux pidfd syscall，仅对本轮精确进程做有界截止；4份数值核心不变。新版本真实CPU准入通过1250题／11879候选／3212543pair token每condition、5快照、315源文件binding和step0全部393参数exact equality。旧tokenizer兼容警告保留，不在对照里改tokenizer。

首轮机制实验和后续研发不是“保证达到数学最高分”的承诺；完整R0后须直接逐输出／来源复核，再作下一轮决策，不只信watcher。异常故障路径未另跑被禁止的unit／smoke注入验收。

## R0完整实际结果与R1决策

5模型×3条件×1250题＝18750条输出全部完整，全部候选margin／概率有限、同原输入预算、原softmax一致，source／code未变化。KD与high-step0在A/B/C各1250题的margin、probability和argmax逐项完全相等；这次不是仅比较聚合数。

下表为五task均衡dev **CE/lnK，越低越好，均匀预测约1**，不是Public Index。既有dev历史trunk暴露未排除，不能据此选泛化冠军。

| 模型 | A：v2 | B：标准PEFT/BF16 head | C：同hidden/FP32 head |
|---|---:|---:|---:|
| KD final | 0.634780 | 0.633808 | 0.633816 |
| 旧joint | 0.618499 | 0.618854 | 0.618804 |
| 高LR step0 | 0.634780 | 0.633808 | 0.633816 |
| 高LR step1000 | 0.998716 | 0.998750 | 0.998738 |
| 高LR final | 1.003253 | 1.003292 | 1.003316 |

精度作用不能一概否认：高LRfinal的reverse一致率由B52%增至C84%，step1000由66%增至95.33%；舍入和tie确实影响残余argmax。但是高LRfinal的五域均衡“预测argmax处原target质量”仅从B0.3508到C0.3532，仍远低旧joint C0.6779，且NLL没有恢复。因此不将0.81归为纯v2bug，也不为无竞争价值的FP32失败final再跑全套public；进入同KD／同四卡路径LR2e-5桥接。

三份完整证据：[直接全量复核](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/results/r0/FINAL_VERIFICATION.json)、[条件／领域汇总](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/results/r0/SUMMARY.json)、[真实完成receipt](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/results/r0/COMPLETE.json)。R1实施计划：[四卡低LR桥接](../../../02-研究路径/06-五维训练与Readout/2026-11-12-06b-research-r1-bridge-implementation.md)。R0本身没有更新权重或产生新public分数。

## R1真实启动与数据审计边界

- 新root`/tmp/opencode/jev-06b-research-20261112-r1-bridge`，11份部署Python逐字回读一致、426源文件绑定、CPU真实准入与原官方强制suite校验通过。固定同KD／同数据／同targets／同四卡数学路径，仅恢复峰值LR2e-5；原KD public只引用真实receipt，不伪造为一次新运行。
- 实际120步／15360pair／3743059token／44关系。初始checkpoint全rank参数MAX差为0，head差分范数0.8081767。此时只证明初始节点一致，不能预填全部训练节点一致。
- 新训练记录point／weighted relation／clip前范数，并在实际checkpoint／final检查rank一致性及head／LoRA变化；没有改变优化目标。独立review发现的rank收养者误判和public退出快照竞态已在封存前修正。
- 只读watcher后台观察训练→完整public，失败只报告，不降LR／重启。启动证据：[R1_START_OBSERVATION.json](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/R1_START_OBSERVATION.json)。
- 数据源编码新核查：[GROUP-PROVENANCE-DISCOVERY.md](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/GROUP-PROVENANCE-DISCOVERY.md)。旧dedup group并非native case group，原侦察的0group匹配不得当成独立性证据；15293条仍为待审候选，不用于R1选checkpoint。

## R1完成：恢复能力，但没有提分

| 模型 | Public Index | Raw Index | 限定 |
|---|---:|---:|---|
| 历史五维KD | 24.38 | 43.62 | 历史暴露谱系，保留原计分 |
| 历史单卡joint2e-5 | **24.91** | 44.07 | 当前历史最好，不是已证实clean榜首 |
| 四卡joint1e-3 | 0.81 | 22.63 | 严重退化 |
| R1四卡joint2e-5 | **24.67** | 43.84 | 同数据低LR因果桥，完整恢复到原区间 |

R1训练1667242pair／458731913token／13026步／3647关系、OOM0，15个保存节点rank参数MAX差0，final head差分范数0.8087664。完整public140620执行／140178计分全ok，全部payload／metadata／stage／顺序／候选预算经直接复核。恢复public管线960.922秒。

原训练协调器在训练收尾GPU监控处停止；存在rank正常退出快照竞态，当时PID未记录，不能唯一排除短暂外部占用。训练回执已验证完整后，新建eval-only root补跑原本尚未开始的public；原stop／blocked状态／sealed code均保留，没有重训或伪造原pipeline完成。

R1结果支持“高LR是关键致败条件”，并排除了已观测保存节点的rank分叉；仍不是逐位纯LR唯一因果证明。证据：[R1直接复核](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/results/r1-bridge/FINAL_VERIFICATION.json)、[原生scores](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/results/r1-bridge/public/scores.json)。

## Case审计完成，但不是新模型分数

实际扫描2529342条train＋15550条validation，5561805图节点；608.773秒，私有文件约5.164GiB。06:11:59UTC逐原行、native group、SQLite与UF root flags核验，323源文件未变。

保守政策下10770条候选val／8336component：Arts640、Retrieval547、Knowledge1920、Language7063、Tools600。107行历史case关联、5行public输入case关联、149行身份不完整及4560行provenance quarantine受影响（拒绝原因可重叠）。此前字符串初筛的150public匹配混有空输入问题，不能直接当150个真实case泄漏，更不能据旧group0匹配宣布独立。**D_select／D_confirm尚未构造或评模型。**

证据：[case直接复核](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/results/case-audit/FINAL_VERIFICATION.json)、[实际汇总](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/results/case-audit/SUMMARY.json)。精确字符串／原生case政策不证明基座预训练未见、所有未知别名或语义近重复均不存在。

## 新发现：旧谱系的ToolRet输入案例暴露

官方ToolRet来源含ToolACE/APIGen，原HF文件叫train也不能直接放行。按实际旧`prepare-overnight-pairs.py`的JSON state结构，将已选`overnight/xlam`36092pair对应27763个input ID全部解码`request`，与public ToolRet1013请求／936个规范化query比较，得到**94个同文query、98条执行请求**。没有读取public gold作标签或据正确性选样；候选／schema／label完全一致性及具体得分影响尚未量化。

此前整段JSON state对plain query的0匹配是表示错位，不能作为未暴露证明。为不把这项限制带入后续公平冲榜，保留旧权重与分数，仅作历史／机制参考；新排名分支须从同一未经过本项目微调的原0.6B基座新建LoRA/head，不接旧已训adapter/head或旧teacher-label-cache。基座原始预训练暴露仍未知，不能声称绝对从未见过。

证据：[历史ToolRet案例输入交集](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/HISTORICAL_TOOLRET_INPUT_EXPOSURE.json)、[完整题Tools元数据](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/NATIVE_TOOLS_METADATA.json)。原native Tools池以procedural为主、浏览器视图归Language不等于整个旧global-pool无真实函数调用训练；现有未恢复case映射的浏览器验证仍quarantine，不能洗白为新dev。
