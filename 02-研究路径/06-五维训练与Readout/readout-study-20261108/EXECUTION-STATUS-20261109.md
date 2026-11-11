# Readout适配阶段状态

> **最终状态已更新：两层四臂及完整比较实际完成，20来源闭包重新核验通过。请先读[最终报告](../../../01-研究结论/06-五维训练与Readout/readout-study-20261108/FINAL-REPORT-20261110.md)及[聚合结果](../../../03-证据与图表/06-五维训练与Readout/readout-study-20261108/final-20261110/RESULTS.json)。下文保留各次历史观察，不应再按“尚未完成／尚未评分”理解当前状态。**

最近证据读取：2026-11-10 03:13 UTC。运行目录 `/tmp/opencode/jev-readout-adaptation-20261109`；本地源码与ledger `/tmp/opencode/readout-adaptation-20261109`。GPU1/2/3已实际运行并有提交的优化更新，不是仅有队列。

| 阶段 | 状态 | 依据 |
|---|---|---|
| 共同已训stage2起点／原分类头 | 已CPU准入及独立快照 | 实际393-key布局、196对rank32 LoRA、score2×1024；配置/来源/文件身份核对 |
| 原tokenizer的255码共同协议 | 已CPU生产构造 | A至JT、固定答案槽、无答案码入输入；无模型前向 |
| 原始完整题与共同曝光清单 | 已准备 | 16992题／4999571token／计划1062macro，生产退出0，READY双标志true |
| 共享冻结特征 | GPU3实际续接中 | 已从3149题/931987token接至3821题/1122560token，updates0，尚无features COMPLETE |
| 冻结主干的新head训练 | 尚未训练 | 尚无head训练完成标记 |
| 四个独立LoRA适配分支 | GPU1/2实际训练Cptr/Bcopy | 各已提交256题/76635token/16次优化更新，真实确定性FA3前后向；Brand/A待卡位，完整分支未完成 |
| 质量／换序／完整请求成本 | 尚未评分 | 固定1100／150／125范围，未得新结果 |
| readout资源调度 | 已实际使用GPU1/2/3 | 新coordinator361386；旧等待队列330314精确退出；GPU0正式评测未动 |

主任务Teacher已完成，五维蒸馏Student真实优化920／9769步、35816310token，确定性FA3前后向已执行；主任务及其官方运行仍优先。没有停止、改动或重启主任务，没有恢复被暂停的原始标签扩大训练。

净新增仅为本研究的完整题CPU准备、共同已训起点/协议准入和独立实现；没有重复2026-11-05零训练原生LM诊断，没有另跑大规模官方榜单，没有预填胜者。

当前保留边界：内部是适配留出，不代表主干未见来源；逆序风险筛查保守但非完备；缺group_id回退命名空间可能保守误合并（未确认实际发生，minor暂缓）。CPU完成不等于模型训练或评分完成。

工作器整体静态审查发现并登记两项Important，现进入定向修复：完整请求成本须把pair额外安全复核耗时并列量化；同一请求足额新窗口反复无进展须有界停止。修复不追加模型请求/样本/烟测，不改变共同数据与初始化。GPU审阅批准尚未写入。

更新：上述计时／有界无进展修复已定向静态复审通过。最终集成发现的queue skip/总完成来源链已实现修复，正在唯一一次定向复审。12:45 UTC实际部署前校验确认8个源码文件逐字一致、已冻结数据文件元数据及共同起点v2身份均未改变；审批仍false，没有readout GPU计算。完成验证新行政入口只读取现有来源，不加载模型或Tensor payload，不增加实验菜单。

12:49 UTC更新：来源链定向静态复审已关闭原Important，无新增阻断项。控制器只核实并停止了自身旧等待队列327760（无worker），保留旧LAUNCH，启动最终代码队列330314并绑定审阅批准；未给任何主任务PID发信号。12:50:24实际状态仍为等待新鲜Student进度和合格GPU3窗口，无研究GPU worker；这不是已训练或已评分。

## 首次真实GPU推进及保存

随后GPU3实际准入，features worker330362执行真实完整题前向；首个已核实让路回执保存32,433token、退出75，无强制回收。后续窗口已续接：累计提交**264题／78,941token／17个特征chunk**，确定性FA3前向累计7,392次，backward0、无SDPA。

因此阶段状态更新为：**完整题数据已准备；共同冻结特征已开始GPU提取并保存／续接；新head与LoRA尚未训练，正式质量／成本尚未评分。** A固定参照和frozen阶段不应伪称产生主干更新；该特征阶段updates实际为0。

安全准入曾短暂以student_step=null拒绝窗口，worker按约定保存让路；未放宽门槛、未强占GPU，也未操作主任务。主任务绑定观察进度为4080／9769步、约1.56亿token，确定性FA3反向仍真实执行。当前累计快照见 `execution-20261109/GPU_FEATURE_PROGRESS_OBSERVED.json`，激活记录见 `ACTIVATION.json`。

## 用户变更主流程后的保守等待

13:56UTC实际读取：readout coordinator330314仍存活，`worker:null`、pending `features`；准入以`main_resource_plan_changed_recheck_needed`等待，具体为主任务旧official入口按用户新指令增加deferral guard、与原研究资源绑定不同。该等待符合主任务优先级，**不是工作器崩溃、研究完成或新质量结论**。未改研究common／共同stage2起点／冻结数据来绕过门槛，也未强占空闲GPU。

主流程现为当前五维KD完成→通用原标签＋五维原KD联合续训→仅新final正式评测；用户已确认继续并离线休息。研究之前已保存的features／partial保留；新头、LoRA和质量成本结果仍不能声称完成。

## 2026-11-10 GPU1/2/3显式授权移交

用户明确要求直接用GPU1、2、3推进。02:35UTC实查GPU0正式public占用，GPU1/2/3显存和利用率均0；readout停等是旧资源策略不匹配，不是评测占满卡。02:51UTC新行政启动层3文件及GPU123_POLICY已私有暂存，原8个数值源码全文、原review元数据、common/config/数据身份均核对未变，旧3149题checkpoint保留。独立行政层静态审查进行中，尚未写新批准或启动worker。

计划GPU3接续features，GPU1/2从同一原stage2及未适配head独立启动LoRA；特征完成后GPU3也接独立臂。frozen头仅用CPU。成本比较在研究训练全部退出后统一GPU3串行，不能用并行占卡的时延判胜。原预算和质量清单不增加，GPU0任务及代码不动。详细移交见`GPU123-HANDOFF-20261110.md`；有实际GPU进程和续接计数后再更新为“运行中”。

**03:09/03:10UTC已实际激活并独立核实：**旧330314按精确pid/starttime/boot/cmd、无worker状态退出，旧LAUNCH归档。新361386持研究QUEUE锁，GPU1/Cptr361397、GPU2/Bcopy361401、GPU3/features361393；原共同数值scope和8源码未变。LoRA两臂都出现实际FA3 deterministic forward/backward，features真实存至3213题/949797token，证明不是从零重跑。最初LoRA保存点仍为0步，不把前后向事件冒称某个已提交更新数；后续以各slot COMMITTED为准。

同次GPU0仍是原official352088，公开请求从此前36710继续到36940，未重启或改Engine。每卡独立控制slot/分支锁，原head/feature文件不重写；完整成本仍等训练退出后GPU3统一串行。研究尚未完成、没有质量胜者，也尚未push结论。

**03:13UTC进一步证据：**Cptr/Bcopy各16个optimizer update已真实提交，非仅模型加载或反向事件；GPU3特征已保存3821题/1122560token。原review/code/config、旧feature锚点文件、未适配HEADS、INIT_READY和全部冻结数据元数据均保持一致；实际QUEUE及3卡FLOCK可见。GPU0原进程352088仍活跃，official已到36990ok。详见`execution-20261109/GPU123_START_OBSERVED.json`。仍未得到两层完整质量/成本结论，不满足push结论条件。
