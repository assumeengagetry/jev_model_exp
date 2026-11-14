# 五维KD单独public → joint LR1e-3重训 → 新public执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补出五维KD final在完整官方public0.3的实际分数，再从同一个KD final重新训练joint段、仅将峰值LR由2e-5改为1e-3，并完成第二次完整public。

**Architecture:** 用户已明确确认三阶段严格串行；阶段内充分使用4张空闲H800。评测采用已完成完整public的v2 bounded-JIT推理核心，四卡按固定run_id分片、仍调用未修改官方runner，合并全量结果后用未修改官方scorer计分。训练保持原数据/目标/原子plan/有效batch128，以四卡FP32梯度SUM同步并行；关系两端留在同rank和同micro内，优化器和scheduler从新阶段重建。

**Tech Stack:** 已有Torch2.9.1 / Transformers4.57.6 / PEFT0.18.1 / Triton3.5.1 / FA3 deterministic / 4×H800 NVL。固定kit commit25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d。无依赖安装。

**Spec:** 用户本轮确认：“现在先用五维KD跑一遍测试，跑完之后再用1e-3重训一遍joint，再跑一次public jev index”；允许充分使用计算资源。不是重训五维KD，不是从现有joint final上再接一段，也不新增学习率网格。

## 已核实起点

- 2026-11-11T16:50:05UTC实际只读核实：原五维KD `KD_HYBRID/final/adapter_model.safetensors`存在，80,800,744字节，393张量，196对LoRA，FP32 score[2,1024]，header数据范围与文件大小匹配。
- 原KD complete/TRAINING_COMPLETE一致：1,250,308pair、366,985,783token、9769步、3647关系。
- 原joint完整数据仍ready：1,667,242pair、458,731,913token、13026步、尾42；旧LR2e-5完整结果不改。
- 同次4卡均无计算进程、显存0；每次正式启动仍须新鲜准入，不能把此观察当作长期预约。
- 当前同joint权重已有v2完整public24.91/3436.6秒；原24.97/v1 24.99也全部保留。v2存在已记录输出漂移，本轮两次使用同一个封存v2核心，不预称与旧SDPA逐位等价。

## Global Constraints

- 三阶段顺序：`KD_PUBLIC_COMPLETE` → `JOINT_LR1E3_TRAINING_COMPLETE` → `JOINT_LR1E3_PUBLIC_COMPLETE`；前项未完整验证不能启动后项。
- 原KD final、原joint final、所有旧结果/源码/cache只读；独立远端根`/tmp/opencode/jev-kd-public-joint-lr1e3-20261111`，本地执行资产在库外scratch；不恢复旧暂停训练、不停止他人进程。
- joint唯一父参数为同一个KD final的固定快照；不是共同初始化，不是现有joint final。加载时逐key/dtype/value核对393个保存参数。
- 完整复用原joint已冻结六stream、effective targets、generic标签、3647关系及宏plan；N=1667242、tokens458731913、steps13026、seed42、global batch128、micro≤32/16384 padded token、warmup0.03+cosine、decay0.01、clip1.0；唯一算法超参变化LR=0.001。
- 4卡梯度FP32 SUM改变历史单卡归约执行顺序，不承诺逐位复现。与旧joint比较须披露并行数值路径，不把同数学loss说成严格同浮点消融。
- 每个完整评测必须140620条执行记录唯一覆盖，排除442后官方计分140178；不截断、不筛候选、不按正确性重试、不读旧预测冒充新推理。
- 四卡分片只为质量评测吞吐；并行请求时延不是维护者单进程/单请求硬件验收。不能将分片耗时直接宣称为单卡加速。
- 只使用固定官方Suite的完整输入与原runner/scorer；不修改kit，不使用--no-verify。沿用官方强制套件校验，不额外模型/语料hash、unit/smoke/synthetic model benchmark。
- LR1e-3是原joint的50倍；非有限logit/loss/gradient即全rank同步停止，不悄悄降LR/裁剪标签/丢batch以伪造完成。普通OOM可按原有完整macro重试机制减micro预算。
- 不上传、commit/push、发布权重/原始语料或参榜PR。本轮只执行已确认训练/两次public及必要生产准入，研究报告按01/02/03新结构保存，不动其它会话未提交材料。

## Review Focus

1. KD模型、joint模型及两个public的模型名/快照/binding不能串用；完成重用必须核对实际结果文件和完整输入身份。
2. 4个eval分片必须互斥且并集恰为原suite执行集；排除项仅由官方scorer处理，不能在worker提前删去。
3. 原子关系不因rank或micro切分而断开；每rank局部sum除实际global macro，再FP32 all_reduce SUM，不能额外除world_size。
4. OOM、非有限值、信号、worker异常不能让其余rank卡死或在部分梯度上更新；只发布已提交检查点，不采用未知partial。
5. 后台进程需来源/锁/精确pid生命绑定，所有失败材料保留；原代码/data/final在整链运行前后保持身份一致。

## Task 1：源资产与生产数据准入

**Files:** 库外`prepare_experiment.py`，新root下`SOURCE_BINDING.json`、`kd-frozen/`、`train/config.json`、`DATA_READY.json`。

**Interfaces:** 输出不可变`kd-frozen`、原joint只读data_source_root、已验证完整eval分片清单、4卡rank原子分配规则；训练脚本读取`--root ROOT --config ROOT/train/config.json`。

- [x] 核实KD文件/header/训练终点和4卡当前占用。
- [x] 对照实际官方Suite/runner/scorer接口及现存源码；核实原joint数据和父参数binding，不重tokenize或重新抽generic。
- [x] 在新root只读绑定并冻结KD、原data/targets/plan/base和v2代码；完整枚举/核对所有宏的rank关系完整性，保存实际预算。
- [x] 固定4个eval分片；按原Suite ordinal%4分配，不读取正确性来筛样本，保存完整run_id覆盖和无重叠检查。

## Task 2：完整四卡public

**Files:** 库外`eval_public.py`（manager与worker入口合一）；实际v2数值核心4文件保持不变，外层只改阶段模型标识与生命周期控制。

**Interfaces:** `eval_public.py --stage kd|joint_lr1e3 [--worker 0..3]`；固定独立ROOT与按stage选择的模型，不开放任意model覆盖；成功写`ROOT/eval/STAGE/COMPLETE.json`和官方原生scores；只有已核验receipt允许复用。

- [x] 调用未修改官方runner逐分片执行，模型加载、默认warmup、validate及device-error停止语义保留；每进程只见分配的单卡。
- [x] 每分片独立锁、输出、来源、runtime和恢复计数；两次完成后来源核对无模型／版本混用。异常恢复路径未故障注入测试。
- [x] 按原suite顺序合并完整唯一结果；逐run_id/payload/field/criteria覆盖核对，不改原生概率；再调用原scorer。
- [x] 两次各140620执行/140178计分全ok；保存并行总墙钟与明确非官方单请求验收的延迟口径。

## Task 3：四卡joint LR1e-3训练

**Files:** 库外`train_joint_lr1e3.py`；输出`ROOT/train/JOINT_LR1E3/{checkpoints,final,complete.json}`和`ROOT/train/TRAINING_COMPLETE.json`。

**Interfaces:** `torchrun --standalone --nnodes=1 --nproc-per-node=4 train_joint_lr1e3.py --root ROOT --config ROOT/train/config.json`；全rank同一配置/初值/宏游标，rank0保存规范adapter/head及所有rank RNG/优化器状态。

- [x] 从原已审查trainer复用model load、FA3、point+relation loss、flat FP32梯度和完整macro归一化，不改effective targets。
- [x] 完整原子unit分配给rank，逐宏验证全部selected行恰好一次和全部3647关系不拆分；NCCL SUM后统一clip/step/scheduler，实际跑满13026步。
- [x] 实现对OOM、非有限和控制信号先同步状态；全macro重试／保留现场停止。实际OOM0，未动态注入验证异常路径。
- [x] 每1000步及终点保存独立检查点；resume同config/data/parent/world/code门禁；完整终点计数1667242/458731913/13026/3647后发布TRAINING_COMPLETE。

## Task 4：审阅、串行协调与监视

**Files:** 库外`run_experiment.py`与只读watcher；报告`01-研究结论/06-五维训练与Readout/kd-public-joint-lr1e3-20261111/README.md`及对应证据目录。

- [x] 独立只读审阅eval分片与distributed loss/恢复/来源；主代理核实并修正3项Important及提升后的完成重入项；不以审查当运行通过。
- [x] 三阶段全链协调器精确持锁，启动前新鲜GPU准入；仅控制本轮自有child/worker。2026-11-11T17:30:17UTC已在四卡空闲时启动。
- [x] 真实启动KD public并读回4卡worker/已完成请求证据；17:32:01UTC实际2140条全ok，模型来源／stage／PID精确匹配；观察器不代替完成核验。
- [x] 基线真正完成后自动训练并评测；19:32:40UTC整链完成，19:36:35UTC直接全量复核。KD24.38，新joint1e-3为0.81，按三阶段分别报告，不将低分当执行失败重跑。

## 执行方式

用户已明确要求直接推进并确认三阶段顺序，不重复设计审批。主代理统筹来源/资源/串行协调，独立文件可委派实现和只读审阅；不得让多个实现者同时改同一文件。普通TDD服从已保留的no-extra-tests边界，以完整生产准入、独立静态审查和两次真实public/真实训练补证，不预填成功。

## 实际生产准入与部署记录

- 2026-11-11T17:29:03UTC新root部署11份Python源码，逐字回读一致；4份v2数值核心与既有成功bounded-JIT封存文本一致。
- 正式CPU准入已生成DATA_READY：140620完整请求，四份各35155；官方强制校验通过，排除442后计分目标140178。
- 全冻结plan的rank训练行数为416842/416841/416837/416722，关系数936/885/944/882，总计1667242行及3647关系；rank宏最大33/33/33/32行，micro仍由原子microbatch按≤32行切分。
- SOURCE_BINDING实际包含281份资产／源码文件元数据；包括旧joint控制、官方kit全部Python和共享runtime。无额外模型／语料hash。
- 首次部署传输包装字符串语法错误发生在远端解析阶段，未创建root；修正库外传输包装后首次实际部署成功。CPU准入SSH会话末尾断开，未重复准备；直接回读DATA_READY并重新执行生产来源门禁，证实已完成且prepare进程／锁已释放。
- 17:30:34UTC四个KD worker真实存活且在封存v2模型加载／Graph初始化；尚不以该加载观察宣称已完成请求或新分数。
- 17:32:01UTC直接复核四个正式worker及已写出的完整2140条全ok，默认warmup已结束；来源均为kd-frozen，未开始训练且无scores。本地只读watcher已后台启动，依据完成通知跟踪，不能将进度当完成。

## 最终实际结果

- 2026-11-11T19:32:40UTC（北京时间11月12日03:32:40）整链完成，总墙钟7343.768秒；只读watcher随后正常退出，没有自动恢复或补跑。
- KD public24.38／Raw43.62，完整pipeline994.238秒；joint训练5324.349秒；新joint public0.81／Raw22.63，完整pipeline987.129秒。
- 两次各140620执行／140178计分全ok；原Suite、两份merged与八份worker结果逐条核对run_id／metadata／payload／stage及顺序，无重复或缺失；两次输入逻辑预算完全一致。
- 新joint全量1667242pair／458731913token／13026步／3647关系，global128／world4／OOM0，最终393参数张量header完整；不是早停、截断或错用旧joint父参数。
- 19:36:35UTC新鲜核对281份源资产元数据及部署源码／4份v2核心仍一致，所有自有进程退出，4卡均显存0。14份最终聚合／原生JSON与中文完整报告已归档。
- 高LR配方严重退化，不替换原2e-5模型；历史单卡与本轮四卡路径不同，不宣称逐位纯LR消融；不自动追加其它LR、checkpoint评测、commit／push或上传。
