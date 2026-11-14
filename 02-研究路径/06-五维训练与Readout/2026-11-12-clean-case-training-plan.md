# Clean-origin完整case匹配训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** 用真实准入的完整native case，从同一stock base＋fresh LoRA/head比较padding布局、BCE/Field-CE和后续mask方法的学习质量。

**Architecture:** 新`/tmp/opencode/jev-clean-trainer-20261112-local/`源码；原stock只读，先生成种子成对共享的全新393-key LoRA/head初始化。每臂单独GPU/namespace、相同16-case曝光macro，使用两遍margin VJP保持完整field CE且显存有界；保存原PeftModel，不保存执行wrapper替代参数格式。

**Tech Stack:** 原Torch2.9.1/CUDA12.8、Transformers4.57.6、PEFT0.18.1、原生FA3/H800；不安装依赖。

**Spec:** R2数据计划顶部有效修订、packed-mask-native-design及本页冻结。

## Global Constraints

- 用户已授权夜间自主排队；不再问逐臂确认，不publish／commit／push／租卡／抢占外来进程。
- 数据必须实际DATA_READY=true、完整COMPLETE与其stat匹配、无BLOCKED；空审批模板/metadata complete不代替ready。
- 模型源仅`/home/dev/assumeengage/models/Qwen3-0.6B`原ModelScope stock，fresh r32/alpha64/7projection LoRA＋FP32两类head；不得载项目已训adapter/head/optimizer/teacher缓存，不切成另一个-Base。
- 原Tensor格式393keys/196LoRApairs、原renderer/tokenizer/真实EOS/全候选保留；unknown fields按双方统一skip-forward policy留raw但不造负例。
- 每臂100M为逻辑pair-token ceiling；真实16-case曝光清单、10M/30M/100M最大完整macro前缀、重复/尾部/未用额度原样消费并报告。无public选checkpoint。
- 数值flags必须来自实际机制准入记录并显式冻结；当前历史R1跨batch/source失败不偷偷改成通过。Mask源码必须有真实train-layout/实际不串样本的证明；未准入则trainer halt。
- 不增加synthetic/unit/smoke/model/corpus hash；初始化检查、实际首macro、节点dev及完整public是研究本体。

## First matched wave (seed42)

| Arm | Mask | Tokenwise layout | Objective |
|---|---|---|---|
| C_D_BCE | causal | dense control | case/known-field/known-pair-balanced BCE |
| C_P_BCE | causal | real packed | same BCE |
| C_D_CE | causal | dense control | full known-field soft distribution CE |
| C_P_CE | causal | real packed | same CE |

这是科学对照，dense只存在于声明的控制臂；production packed仍须realT。第一质量波固定相同micro组成（共同padded-admission规划、packed执行只算realT），避免把微批切分与layout混归因。之后production real-token预算组批的吞吐/同GPU时间额外曝光另臂记录，不称同曝光学习算法收益。

随后复用实际causal packed CE控制，fresh同seed初始化／同曝光另跑bidirectional和prefix_block CE；Base state-boundary及Jina子研究各有匹配输入/位置控制，不用训练好的causal adapter继续冒充fresh。是否扩seed43由预定dev/确认协议和实际差异决定；多seed结论不足则报告不确定，不保证51.99。

## Loss/VJP contract

每macro C个完整case，case内F_known个有监督field：BCE每field known候选均值再F_known/C平均；CE为完整field`-sum(q*log_softmax(m))`再F_known/C平均，q不事后归一化。已知K1 CE零梯度仍记账，unknown-only不贡献监督分母。

1. 第一遍无梯度同权重得到macro全部原候选margin与稳定slot映射；每micro独立autocast作用域，不能与第二遍共享no_grad污染的autocast权重cache。
2. 用这些实际cached margins通过torch.autograd取得dL/dmargin，固定全部case/field分母；不是手推／micro平均。
3. 第二遍同输入/顺序/波次/权重/零dropout重算；要求first/second margin replay精确一致（否则停止并保留，不偷偷称exact VJP），逐micro反传`sum(m_i*coeff_i)`，不再次除C/世界数/micro数。
4. FP32 trainable梯度全有限、clip1、AdamW fused2e-5/weight_decay.01、3%warmup cosine一次macro step。单GPU每臂，无DDP或“同时4臂”等同4rank归约的假声明。

首实际macro在正式曝光中计数，保留coef/logits/393grad/update/optimizer证据；tiny-autograd检查仅消费实际cached margins，不另造样本或跑模型smoke。BCE可用同实际margin目标直接求导对照coef；全macro模型图不强行保留到OOM。

## Files / Tasks

- [x] `fresh_init.py`：实际从stock构造全新r32/FP32score，393keys／196pairs／B-zero及served616237056已直接回读，0forward，不载旧adapter。
- [x] `case_runtime.py`：实际完整data/complete/source/code pins、JSONLoffset/exposure回读与四臂共同首macro组成核对通过；后续全曝光仍进行中。
- [x] `objectives.py`：首正式macro实际cached margin的BCE/CE及autogradcoef由主代理独立重算精确相同，F_known/C无重复平均。
- [ ] `train_case.py`：模式/数据/初态/源码/numerical admission配置完全绑定；两遍actual replay、finite梯度、一次更新、全部checkpoint和optimizer/RNG/cursor/exposure ledger保留，禁止复用别mode complete。
- [x] `evaluate_native_dev.py`：只在10M/30M/100M观察节点评D_select；四臂三节点现已全部实际完成并独立回算；case统计单位五域known-field argmax target mass，NLL/Brier/coverage并报。D_confirm不在trainer中打开，public不在此入口调用。
- [x] readonly review、源封存、实际首macro／cursor1暂停与显式续跑已由主代理直接核验；四臂完整99824365token/3557calls和三个dev节点已完成，原worker退出、各43checkpoint全部保留。BI/PB后续仍在跑，不等于整研究结束。

## Review Focus

- fresh-init不能偷偷读取旧common已训head/teacher/optimizer，served与physical备用head参数区分。
- two-pass不能重复除macro分母、跨候选softmax不能跨field、unknown标签不能当0。
- checkpoint必须含loss/mask/layout/seed/data/precision/source身份，旧完成结果不得按另mode复用；STOP只保完整macro与自有任务。
- D_confirm/public不得在训练选择循环中读取；dev长题不截断、不补短题，覆盖如实。
- active model admission与真实数据commit门禁必须阻断模板、部分产物、未解释数值/串扰问题，不由源码存在直接签通过。

## 执行依赖实际回执（2026-11-12）

- [x] clean数据实际准备＋必需回读＋主代理直接复核。远端`/tmp/opencode/jev-clean-origin-20261112-v1/prepared-01/`DATA_READY=true，无BLOCKED；24600train、375select、250confirm。
- [x] 同核原macro布局比较：三mode logitsMAX0，393grad/实际AdamW更新全过原合同。原单独/混批及prefixBlock fork失败仍保留。
- [x] 进一步真实同几何内容隔离：原128pair／45216T／4micro目标输出与393参数VJP换等长同P/S真实邻居后逐位相同，实际28层realT。结果`/tmp/opencode/jev-fullrow-admission-20261112-v1/FINAL_PRIMARY_VERIFICATION.json`。
- [x] 主代理据实际证据签fullrow科学训练-only准入，保持原TF32=true/BF16red=true；非预测等价、非prefixcache/official serve准入。首正式macro的fresh模型精确两遍replay/有限梯度/一步更新仍需真实验证。
- [x] 接收独立trainer、主代理review、源封存与实际fresh共同初始化及四臂首macro／续跑已完成；见下节。10M/30M/100M D_select及完整训练均已独立回读；fresh public仍未执行，不回写旧机制成绩为clean。

Ruling: 将“同几何下内容依赖”和“不同批几何的浮点稳定”分别判定，不把HF本身重现的几何漂移冒充padding串样本或用放宽RMS偷过原门禁；原结果不改，只给full-row科学训练有限准入 — cost if wrong: 正式首macro replay与后续dev必须停止并保留异常，正式serve/prefix另有未通过门禁。

### 训练源封存前的审查处置

- 8份trainer源整体独立readonly review未发现确定P0/P1阻断；P2持续native stat漏项已补到Context.check，封存/fresh前完成，不会让旧fresh manifest身份失配。
- 固定受控checkpoint周期100macro，另存step0／首macro／观察点／final／pause；全部保留。实际scratch约156.9GB，四臂各25GiB预算及共享20GiB free reserve，不淘汰旧checkpoint。
- 明确保留所有臂相同的warmup107＋cosine-to-zero：实际3557次optimizer调用中，末次LR0，推进梯度／Adam状态／曝光但无参数或decoupled decay更新。报告3556次positive-LR调用，不把它与3557次非零学习混称。

## 已启动的实际fresh训练（2026-11-12T13:05UTC直接准入）

- v1 CPU fresh已通过，但四臂在load_saved_adapter的逐键对账前报相同KeyError，尚无训练forward/更新/曝光。主代理直接检查installed PEFT0.18.1：setter会原地改写传入mapping的modules_to_save键。只改为传`dict(saved)`保留oracle并核对canonical键集，v1root/失败全保留；新v2重新签源/fresh manifest，未放宽逐键byte比较。
- v2四臂正式macro0均完成并按cursor1/exit75暂停。主代理实际回读全部证据：共同fresh393键、相同16case/field/candidate/micro组成、first/replay logits和margin逐位一致、系数独立重算精确、393组未clip/已clip梯度有限、实际before/after/delta与Adam state及checkpoint权重一致。
- 每臂首macro逻辑token20947，两pass实际forward41894、backward20947；不是额外smoke。四臂正式首macro logits dense/packed MAX0，393grad/actualupdate均过原冻结合同（BCE最差gradient/update相对L2 .000266524/.00453116，CE .00115246/.01928358）。
- 13:03:16UTC续跑总队列启动；13:05:09UTC一次性直接核对四个exact worker存活且完成45／53／46／52macro，均从index1恢复、macro0仅开始一次，无failure。这是当时准入证据，不是持续轮询或最终完成。
- 远端实际执行root：`/tmp/opencode/jev-clean-trainer-20261112-v2/`；已冻结`seed42.json`和`source/`，四臂目录为`C_D_BCE-seed42`、`C_P_BCE-seed42`、`C_D_CE-seed42`、`C_P_CE-seed42`。first/optimizer/weights全部留远端，不下载到Git。
- 四臂完整退出后自动从**同fresh起点**跑BI_P_CE／PB_P_CE（非causal warmstart），三观察点仍按冻结schedule。任一基础臂失败则保留，不静默rewind/重试，不把未配对的后续方法伪称完成。
- exact pidfd completion观察器与只读inotify/pidfd的10M select观察器已接入；不会通过轮询读取进度，也不会另启动dev。当前没有fresh public分数、没有D_confirm或public执行。

## 冻结10M D_select：已独立回算（2026-11-12T13:27UTC）

- [x] 四臂更新365／5840case occurrences／9988659逻辑token的预定观察全部完成；相同375case（各域75）、615known field、5190完整candidate、90unknown保留不forward；known coverage1.0，完整onehot exact仅259case子集。
- [x] 主代理从保存的实际logits／margin＋原始pinned select roster独立回算每field／case／域argmax target mass、NLL、Brier及coverage，最大field指标误差1.56e-15以内；节点前后checkpoint同cursor、scope／候选顺序／target均核对，无新forward／eval／确认集读取。

| 臂 | 五域case target mass ↑ | NLL ↓ | Brier sum ↓ | 完整onehot exact（259子集） |
|---|---:|---:|---:|---:|
| C_D_BCE | 0.4754056282 | 1.1893012354 | 0.5864355014 | 114/259 |
| C_P_BCE | 0.4968982522 | 1.2002228905 | 0.5840132656 | 121/259 |
| C_D_CE | 0.5387655985 | 1.0677630039 | 0.5537759034 | 136/259 |
| C_P_CE | 0.5208058597 | 1.0740614145 | 0.5574684269 | 129/259 |

- 同曝光CE−BCE target mass：dense＋6.3360百分点，packed＋2.3908百分点；两布局等权＋4.3634百分点。独立paired-case五域分层10000次bootstrap描述区间分别[＋2.3639,＋10.3571]／[−1.1402,＋5.8235]／[＋1.2903,＋7.5392]，未多重比较校正、不覆盖训练seed不确定性。
- Packed−dense：BCE＋2.1493百分点，CE−1.7960百分点，两case区间都跨0。当前不是稳定padding学习收益或长期数值等价；不能把D_select53.88%当成超过Public Index51.99。
- 四臂实际两pass训练forward19977318 real token、backward9988659；dense累计tokenwise输入行数26812992、packed19977318。系统计数与学习质量分开，不据节点完成先后给稳定吞吐结论。
- 结果详见仓库`03-证据与图表/05-Attention与效率/attention-queue-20261112/results/clean-trainer-v2/D_select-10000000/README.md`和`PRIMARY_VERIFICATION_AND_PAIRED.json`。原private case-metrics／target／logits不进Git，归档view明确elide重复scope并保原身份指针。
- 10M观察器已退出；30M只读文件事件观察器已接上，整队列exact pidfd观察器保留。没有改8份active trainer source／seed42 config／LR／曝光，没有重启或重跑模型，没有选择赢家或打开D_confirm/public。

## 冻结30M D_select：已独立回算（2026-11-12T13:47UTC）

- [x] 更新1122／17952case occurrences／29985465logicaltokens四臂节点完整结束。原375case/615knownfield/5190candidate/90unknown及coverage1.0全核对；首尾checkpoint同cursor，独立raw-logits/target/argmax/NLL/Brier回算最大误差1.89e-15以内，未新增model/eval/confirm/public。

| 臂 | 五域case target mass ↑ | NLL ↓ | Brier sum ↓ | 完整onehot exact（259子集） |
|---|---:|---:|---:|---:|
| C_D_BCE | 0.5398824406 | 1.1453914519 | 0.5648176083 | 132/259 |
| C_P_BCE | 0.5313842060 | 1.1624516934 | 0.5816323764 | 136/259 |
| C_D_CE | 0.5488940973 | 1.0326026563 | 0.5327917567 | 139/259 |
| C_P_CE | 0.5567535822 | 1.0365784468 | 0.5288095160 | 141/259 |

- CE−BCE target mass dense＋0.9012／packed＋2.5369百分点，两布局等权＋1.7191（描述性95%case区间[−1.5341,＋4.8883]）；各区间都跨0。NLL/Brier两布局仍优于BCE，不能把10M较大的dense优势当稳定结论。
- Packed−dense为BCE−0.8498／CE＋0.7859百分点，都与10M方向相反；继续不声称packing本身稳定提分或数值等价。denseCE Arts相对10M降10百分点，packedCE Arts降4.6667，其它域进展分开报告，不中途重配数据／LR。
- 原协议不变、继续冻结100M；30M观察器已退出，新100M只读事件观察器接入，已有整队列pidfd观察器未重复启动。详细报告与9份聚合归档于`results/clean-trainer-v2/D_select-30000000/`，分析源与10M为同一封存版本。

## 第一波完整结束及100M D_select（2026-11-12T15:32UTC独立回算）

- 原四臂各3557optimizer calls/56912case occurrences/99824365logicaltokens及三观察节点全部complete；末步LR0，3556positive-LR calls。15:34:08UTC实际回读确认原四worker退出，各43checkpoint保留；原controller已按冻结合同fresh启动BI_P_CE/GPU0、PB_P_CE/GPU1，不改原源/配置/队列。
- 末次375case/615knownfield/5190candidate/90unknown全覆盖，主代理同CPU分析器独立回算field/case/domain、真实候选顺序/skip以及节点前后同cursorcheckpoint，最大误差1.7902346272e-15。0新forward/optimizer/eval，未开confirm/public。
- D_BCE/P_BCE/D_CE/P_CE mass56.4260%/57.5694%/61.6935%/59.5955%；NLL1.2816409/1.2835494/1.1710422/1.1938659；Brier.5703201/.5733636/.5372062/.5406977。完整onehot exact140/146/155/145，分母259，不混为375case主指标或官方Public Index。
- CE平均mass优势+3.6468pp，五域整case配对10000bootstrap描述区间[+.4123,+6.8833]pp；未经多重比较调整且不含训练seed方差。Packing在BCE/CE上+1.1434/-2.0980pp，两case区间仍跨0，未证明稳定学习收益。
- 相对30M四臂mass都提高，但NLL/Brier全部变差；不能只取单指标称质量单调进步，亦不回选public最高checkpoint或修改曝光/LR。候选选择/确认需后续冻结，原fullrow-only/数据来源/工具procedural等局限不解除。
- 两passreal训练forward199648730/backward99824365、各16977wave；dense tokenwise physical266503232 vs packed199648730。系统计数与质量分开，不冒称稳定吞吐或官方serve准入。
- 详细报告、9份节点聚合、4份完整退出view及实际分析源均归档`results/clean-trainer-v2/`，private case/logits/targets/tensors及含cid的schedule留远端。100M观察器已正常退出；原整队列观察器继续等BI/PB，不重启eval或watcher。
