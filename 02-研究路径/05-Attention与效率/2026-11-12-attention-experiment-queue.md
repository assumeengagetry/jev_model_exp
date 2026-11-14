# Attention实验队列：padding-free／链接方法／commit2e334fb

**用户授权：** 2026-11-12用户明确要求把padding、给定链接和刚pull的latest commit排队自主跑，尽可能充分利用现有计算资源，睡觉期间不再逐项确认。此授权不扩大为发布、付费租卡、抢占他人GPU或丢弃旧资产。

## 队列与当前已验证状态

| 项目 | 实际状态 | 不能冒称的结论 |
|---|---|---|
| PF v1 | 真实128pair macro完成但数值门禁失败，保留 | 不是paddingfree已通过 |
| PF v2 | 同macro45216真实token；原HF对dense logits/gradient/update均MAX0；packed全部393组gradient/update过原合同，28层realT hooks通过 | 不是四卡训练或提分 |
| PF v3完整public | 两臂全量已完成并主代理逐条对账；dense24.71／packed24.70，各140620输出、140178计分；原逐题数值合同未过 | 固定权重没有提分，不能称clean排名或预测等价 |
| 最新commit causal | 真实FA3 forward112/backward112；128pair完整macro；FA3对SDPA整模型数值合同未过 | 不是causal mask bug证明／质量结果 |
| 最新commit bidirectional | 真实FA3 forward112/backward112；完整macro；数值合同未过 | 不是双向训练提分 |
| 最新commit prefix_block | 真实FA3 forward224/backward224（每层两次）；完整macro；数值合同未过 | 不是已证明GPU train/serve数值等价 |
| Flex causal | 真实compiled1uniquegraph，forward112/backward112，首生产调用计入9.7665秒；数值合同未过 | API能跑／能反传不等于与FA3训练轨迹等价或涨分 |
| Fresh原六臂 | 原四臂＋BI/PB均完整99824365token/3557calls及三D_select节点结束，独立回算通过，原worker/controller退出，全部checkpoint保留 | 后续两探索臂也已完成；D_select不是Public Index，未confirm/public |
| Base严格state字符末切点 | 真实全train/select CPU投影＋主代理独立回读完成；28676/41674 train、427/615 select knownfield跨BPE，版本ready=false，不训练 | 不是Base质量失败；不round/删case/冒充Candidate边界已复现 |
| Base固定已有换行v2 | 完整99824365token/3557calls/三节点且真实退出，终点mass44.3620%，独立回算通过，43checkpoint保留 | 不是严格v1成功、完整Base复现、新Pbackend/neighbor或serve准入；未见稳定边界收益 |
| JPAIR固定BCE＋hard-rank | 完整99824365token/3557calls/三节点且真实退出，终点mass61.2437%，独立回算通过，43checkpoint保留；已冻结challenger | 不等于改善已确认、完整Jina/listwise/cosine复现或Public Index提分 |
| Terminal dense→packed | 两固定终端权重各完整375case/393before-after bytes不变，均34case数值违约，执行完整但eligible=false | 0choiceflip不代替raw数值准入；denseCE61.69%不能冒称packed生产候选 |
| 最终两候选消费者 | baseline=C_P_CE、challenger=JPAIR已封存；四模块源码/修复/最终集成审查及实际install/readback通过，22:19:51UTC显式启动one-shot | 候选manifest不事后改写ready；codec/confirm/clean public完成仍须实际分层回执 |
| Jina-inspired交互／loss | J0/J1/J2真实16case机制接线已执行，fresh匹配训练仍待接入 | 不直接换权重，不把causal listwise误称双向 |
| 知乎文章 | 403，已告知用户需正文／截图 | 内容未读，不猜方法 |

## 资源安排与故障策略

1. 第一波GPU0/1/2分别运行exact commit三个mode；GPU3运行Flex。四个任务均已实际完成、保存raw／393组gradient／actualupdate／optimizer。
2. 单方法超容差只标该方法未通过，独立的已准入PF完整public继续，避免整队列空等。
3. 完整public集中四卡，每臂四个35155不相交shard，完整合并140620、官方排除442后140178计分；不得跨臂补预测或按正确性重试。
4. CPU clean数据准备草案与只读审查同步；新排名模型stock base＋fresh LoRA/head，语义请求去污／许可／完整case／D_select/D_confirm未准入就不沿旧暴露谱系继续冲榜。
5. 原长SSH连接返回255后，直接核对`/proc`和新队列exact PID/starttime/boot，证明远端队列仍在继续、四卡public叶进程实际运行；**没有重启或重试GPU任务**。用只读pidfd阻塞观察器等待exact队列退出，不轮询完成状态；transport重连只重附观察器。
6. 所有旧root、失败、checkpoint、optimizer、partial保留。原R1停止标记不改；不自动commit／push／上传。

## 明确未通过项的下一步

- 先看同mode的FA3与SDPA数值漂移，连causal基线也未过，因此不能直接把漂移归罪于新mask。区分mask语义、BF16后端数值和首步AdamW近零梯度的符号放大；不看结果放宽原合同。
- 后续mask质量训练保持同后端与train/serve一致，不能把旧causal权重切非因果开关的分数当mask重新训练收益。
- 新的100M逻辑pair-token／10M、30M、100M观察点训练必须有同起点／完整case／相同曝光对照；相同GPU时间带来的额外曝光另报。
- clean数据实现由独立工作流准备，主代理检查源码／schema与实际产物后准入，不以子代理“已实现”代替DATA_READY证据。

## 运行与证据定位

- 远端总队列：`/tmp/opencode/jev-attention-queue-20261112-v1/QUEUE.json`，`ACTIVE.json`，每任务独立log。
- 远端commit三臂：`/tmp/opencode/jev-attention-2e334fb-20261112-v1/runs/`。
- 远端Flex：`/tmp/opencode/jev-flex-20261112-v1/results/`及独立Inductor/Triton cache、FAILURE.json。
- 远端PF已通过真实macro：`/tmp/opencode/jev-padding-free-20261112-v2/mechanism/`；完整public新namespace：`/tmp/opencode/jev-padding-free-20261112-v3/eval/`。
- 已完成机制原始JSON22份已归档：`03-证据与图表/05-Attention与效率/attention-queue-20261112/results/`；主代理已直接复核完整组数／原生调用数／通过与失败状态。
- 实际冻结源码37份已回读归档于仓库：`2026-11-10-attention-mask-modes/code/experiments/20261112/`。其中padding-v1/v2/v3与方法独立分目录；不是把这些版本混合使用的安装包。

**记录边界：** 本页是队列账本，不是“所有实验已完成”声明。旧固定权重public和八臂fresh clean训练／三节点独立回算均已完成；新候选已封存，但最终consumer准入、一次D_confirm与clean新权重完整public尚未执行。以顶部当前状态和末尾实际回执为准，时间章节保留其当时事实。

## 后续接线与新数据依赖

- 新坐标metadata阶段已实际完成：59.011954秒、2544892完整坐标、667sources；主代理复核全部1868994种finalroot/flag组合和332继承source身份。这个阶段本身不签DATA_READY；后续有界clean准备现已真正完成，见下节。
- 数据准入收敛为有限native source/shape正向白名单＋input-only负向membership＋已知request包装对称解包。未使用来源无需先穷尽全部考据，但未知兄弟case／未决许可／known public/native-dev排除／75+50配额等门禁不解除，详见R2计划顶部修订。
- 同native FA3的三mode packed执行层已实际执行：与exact-commit HF/nativeFA3的logits MAX0，全部393组gradient／actualAdamW通过原合同。整体原marker仍false：单独/混批数值合同均失败，prefix_block fork亦失败。不得只取布局通过当整体通过。
- 后续队列v1因训练Python没有os.pidfd_open在开GPU前退出，原partial保留；v2使用已核实syscall pidfd，实际等前序退出后自动执行三臂，全部结果已回读归档。总public队列与同核队列均已结束，不再等待或重启它们。
- Jina-inspired布局／同Flex隔离与交互／ranking loss组件已交付及静态review，并于12:29UTC实际完成真实16case机制接线（见下节）。新的BCE及rank共同冻结F_known/C分母；与旧pair128目标分开。physical field超40960仍是正式全覆盖前必须解决的能力缺口，不能删候选／丢题冒称完整public。

## 最新直接验证事实（截至2026-11-12T12:19UTC）

### 完整public：回答固定权重下packing是否提分

- 主代理重新逐条核对两个merged所有payload、官方元数据、唯一run_id和model标识；每臂4×35155完整shard，140620全ok，官方计分140178全ok。
- dense24.71／packed24.70，没有提分。旧谱系机制结果，不是clean模型成绩。
- 14801请求违反原数值合同；378 field choice翻转、涉及376请求，21带外。平均绝对概率差0.000161998、最大0.151963，最大margin差0.486328125。完整覆盖≠预测等价。
- 单次完整A/B mean118.2795→101.9751ms，p80 148.1435→129.1743ms；四卡pipeline4263.0214→3705.3197s。只是这一轮实测，不是稳定多重复提升或官方单卡latency admission。

### 精度研究：将批几何与跨样本内容依赖分开

- precision-v1因原macro内未找到同时满足缓存组和等长邻居的fixture，在模型加载前退出，保留；v2把旧失败target/cache位置32/33与真实等几何邻居7/9分开，CPU先冻结长度/P/S，无synthetic token、无答案选择。
- 2×2 TF32／BF16 reduction四卡实际均执行完成，每worker约5.6s，三mode共1064次native deterministic forward，无backward／更新／public计分。
- 主代理直接读取四份ACTUAL_TENSORS.pt：同几何换邻居在cos/sin、norm0、q_proj0、attention context0、layer0后hidden、EOS hidden和目标logits逐位不变；HF和packed的solo/mixed也各自逐位相同。四旗标的所观察fullrow hidden一致，开关未消除形状漂移。
- 首观察差异在attention context0之后、layer1 input norm之前，尚未分清o_proj或MLP具体哪步；不能归因给RoPE／TF32，更不能称所有输入绝无串扰。
- 新扩展实际控制覆盖完整原128pair／45216真实目标token／4micro：每micro附等几何B或C，三mode全部128 logits与393参数VJP逐位不变；28层realT全核对。不开optimizer，不重跑public或旧训练。
- 结合原同核393grad/actualstep，主代理只签发**full-row科学训练准入**；原solo/mixed失败保留，不签预测等价／prefix-cache／正式serve。fresh初始化与正式首macro的两遍精确replay和一步更新仍须实际验证。原精度flags不变，避免把新精度策略夹入学习对照。

### 有界clean数据：ready不是模型已训

- 实际准备623.7405s，峰值RSS815398912bytes、最终2439673916bytes；无BLOCKED，COMPLETE和DATA_READY的身份／最终预算核对通过。
- train24600cases：arts4971、knowledge4976、language4950、retrieval4937、tools4766；D_select375（各75）、D_confirm250（各50）。48571字段中42678known、5893unknown原样保留且统一不forward。
- 两seed同一曝光多集：56912case occurrences、3557个16-case更新、99824365逻辑pair token；两遍199648730为预计forward token，**未训练实测**。
- seed42观察点更新365／1122／3557，实际逻辑token9988659／29985465／99824365；D_confirm未用来调参。
- 原字段回读／known完整pair重tokenize／跨split完整输入精确检查／曝光回读已实际完成；主代理另行直接复核。28202 recognized native closure对回读，unrestored sibling veto实际0；这只是声明的有限source/shape与精确表示覆盖，不证明所有语义近重复或预训练暴露不存在。
- 本节截至12:19UTC时trainer尚在实现；其后已交付、审查并实际启动（见末节）。不得将数据ready或训练-only机制准入写成已完成100M学习实验。

聚合回执及实际源码已回读至仓库证据`results/{full-public,native-mask,precision-v2,clean-ready}/`及源码快照相应目录；未回读私有原文、token序列或任何.pt权重／激活／optimizer到Git。

## 12:24～12:30UTC补充真实执行

- 原生paged候选GPU3实际可执行3D packed-Q/cuQ/page1，P333／S11+12，28层全prefix不变、换序/重复逐位相同。但prefixBlock full/paged raw相对RMS5.56%／margin4.90%仍超原合同；causal在此组过。整体serve不准入，停止盲目换cache kernel，fullrow科学训练不依赖它。
- Jina-inspired J0/J1/J2在三卡实际运行完整clean曝光第一macro：16case／19field／60candidate，物理realT10342，field-cu19但EOS60正确接线；两pass logits逐位一致，393组梯度与实际更新finite。每臂56次compiled Flex forward提交／28次backward hook；首次生产编译计时9.82～10.64s。
- 这次Jina仅用**历史诊断adapter**做机制接线，不是fresh学习／不计新clean曝光，没有新public分。主代理从保存的实际logits和原known target独立重算F_known/C系数，最大差7.45e-9。rank λ1的clip前norm约13.29，J0约3.50；尺度变化如实记录，不能拿不同目标loss比较提分。
- 这些macro未覆盖unknown字段GPU路径、候选排列/跨field换邻居鲁棒性或>40960完整serving；不将一次接线签成所有能力通过。
- 新聚合和源已归档至`results/{fullrow-admission,native-paged,jina-macro}/`及对应源码snapshot。只读证据传输遇一次SSH255后重附读取，没有重启任何GPU任务或替换预测。

## Fresh训练正式推进（13:05UTC已直接核验）

- 8份trainer源完成整体独立只读review；持续native pin的P2补齐后封存。CPU fresh393FP32／196LoRApairs／B-zero／FP32 2×1024 head直接回读通过；actual served616237056，不载旧adapter/head/optimizer/teacher。
- v1在任何forward/更新前发现PEFT0.18.1 setter原地改写mapping导致对账KeyError。主代理读installed源码后仅复制输入mapping保留原oracle，保留全部v1失败；v2源和fresh manifest独立重签，无训练曝光重放。
- 四臂v2首个**正式**macro0实际完成并暂停cursor1，主代理逐实际tensor/coef/393grad/actualstep/Adam/checkpoint核验。相同16case、相同field/candidate/微批构成；20947逻辑pair token／41894实际forward token／20947 backward token。两遍逐位一致；dense/packed首macro logitsMAX0，BCE和CE各393gradient/actualupdate全过原合同。
- 续跑队列13:03:16UTC启动，一次直接准入13:05:09UTC证明四个精确worker均存活、从index1连续推进45/53/46/52macro、macro0各只出现一次，无失败。不据此推算最终速度或新分数。
- 当前root`/tmp/opencode/jev-clean-trainer-20261112-v2/`，四个C_*各占一张H800；完成后自动fresh同起点BI_P_CE／PB_P_CE，全程fullrow不使用失败的prefixcache。100M为99824365逻辑token／3557optimizer calls，其中末步LR0，3556positive-LRcalls，所有臂一致。
- 所有checkpoint保留，周期100macro＋step0/first/预设观察/final/pause；每臂25GiB及共享20GiB reserve。不替换旧checkpoint、不抢占、不发布。
- 已归档`results/clean-trainer-v{1,2}/`聚合与实际源码，私有案例/权重/梯度/optimizer留远端；10M dev用只读文件事件观察，整队列用exact pidfd阻塞观察，均不轮询、不额外启动eval。新质量结论等待冻结D_select/D_confirm/public协议，不能提前宣称超过51.99。

## 首个新质量中间观察：10M D_select（13:27UTC独立回算）

- 四臂预定更新365均已完整评375case／615known field／5190candidate，90unknown保留不forward，coverage1.0。不是“仍没有dev结果”；也不是训练最终结果或public分数。
- Target mass（五域case等权）：denseBCE0.4754056282、packedBCE0.4968982522、denseCE0.5387655985、packedCE0.5208058597；NLL分别1.1893012354／1.2002228905／1.0677630039／1.0740614145。CE方向较好，但不提前选checkpoint／赢家或改学习率。
- CE−BCE为dense＋6.3360／packed＋2.3908百分点；packed−dense为BCE＋2.1493／CE−1.7960百分点，packing方向不一致、两paired-case描述区间均跨0，尚无稳定学习收益证据。单训练seed的case区间不代表跨seed置信，也不证明数值等价。
- 主代理只读原实际logits／margin＋pinned select独立重算field/case/domain指标，最大差1.56e-15以内；原COMPLETE、scope、真实候选／顺序、未知不forward、节点前后checkpoint及计数均核对。没有重跑模型、eval、D_confirm或public。
- 聚合及实际分析源已归档`results/clean-trainer-v2/D_select-10000000/`，重复scope只在标注view中省略、原证据仍保留。完整报告含每域指标／成对case分析和单次系统计数，不能把53.88%的自建dev当Public Index超过51.99。
- 原10M事件观察器已完成退出，新30M只读inotify/pidfd观察器等待预设更新1122；当前训练和后续BI/PB队列协议不变。Base精确state切点及Jina fresh匹配训练仍待接线，不能称链接实验全完成。

## 第二个中间观察：30M D_select（13:47UTC独立回算）

- 四臂更新1122／29985465logicaltokens，375case/615knownfield/5190candidate/90unknown完整一致、coverage1.0。主代理复用同一封存CPU分析器从实际logits/原target独立回算，最大指标差1.89e-15以内，无新model/eval/confirm/public。
- DenseBCE／packedBCE／denseCE／packedCE target mass依次0.5398824406／0.5313842060／0.5488940973／0.5567535822，NLL1.1453914519／1.1624516934／1.0326026563／1.0365784468，Brier0.5648176083／0.5816323764／0.5327917567／0.5288095160。
- CE的NLL/Brier优势仍在，mass两布局等权优势缩为＋1.7191百分点且case描述区间跨0；不按10M强信号或30M暂时最高者选winner。Packing相对dense的mass方向在两个loss上都从10M反转，尚无稳定学习收益。
- 同期记录domain退步：denseCE Arts相对10M−10百分点、packedCE−4.6667，与Knowledge/Language/Retrieval增长并存。仍沿冻结数据/曝光/LR，不用某域中间结果做public路由或checkpoint拼接。
- 9份聚合view/primary证据和详细表归档`results/clean-trainer-v2/D_select-30000000/`；原完整privateartifact不改。30M观察器已退出，100M只读文件事件等待接入；原整队列控制器/active8source/config不动，Base仍先做新namespace CPU切点投影。

## Base严格state切点：CPU实证阻断（14:04UTC）

- 三个review P2（sidecar生成版pin、config自绑定、preflight BLOCKED留痕）封存前已修正。独立CPU准备器全24600train/375select实际100.6339s、峰值RSS201367552bytes，不载Torch/model或打开D_confirm。
- 41674train knownfield中28676跨state最后字符切点BPE，615select knownfield中427跨界；其余12998/188可精确切。不能用子集偷偷继续原causal控制，整版ready=false且未启动state-P训练。
- 主代理独立相邻offset端点算法全量重新分词核对原prefixIDs／切点、sidecar／失败清单／计数及source/config/data/output绑定，实际47.5031s确认该结果。原干净数据DATA_READY不受影响，v1complete只指扫描结束，非训练准入。
- 原所有source/sidecar/failure保留，聚合与实际源码已归档`results/base-state-boundary-v1/`。不round、不删候选、不改模板掩盖结果；任何固定separator所有权变体需独立设计，不能回写本版为已过。

## Base固定已有换行v2：输入边界ready（14:19UTC）

- policy固定`state-plus-one-existing-newline-v2`，全体field统一将state后的一个已有LF归P，不新增token、不按样本fallback，不含后续Question／criteria／Candidate标头字符。
- 新namespace完整CPU投影102.2980秒、峰值RSS201121792bytes；全部24600train／375select、41674／615known field精确，无失败；unknown5753／90不tokenize。独立primary全量相邻offset端点算法48.7907秒回读通过，verified UTC14:19:32。
- 3份聚合及实际producer/primary源码已归档`results/base-state-boundary-v2/`；原v1失败不变。`allow_new_state_boundary_training=true`仅是输入产品准入，不是393fresh、新mask实际gradient/update、质量或prefixcache准入。
- 新无损sidecar reader另经独立spec/quality源码review及真实全量CPU验收16.8386秒：24600train/375select全部case/field/candidate/EOS/row及来源链通过，真实16train各重复查询和全部375select查询通过；无Torch/tokenizer/model/confirm。聚合及4份实际源归档`results/state-reader-admission-v1/`，不是重跑已完成投影，也不签新模型首macro。
- 后续`STATE_NL_PB_P_CE`只改边界、`JPAIR_P_BCE_RANK`只改目标，均需新source/config/fresh、正式macro0消费一次及外部cursor1审核。两者属于看到10M／30M之后的探索臂，不追记为原四臂预注册比较；原controller／GPU0/1 BI/PB预留不动。

## 后续新目标：真实saved-macro机制（无新模型运行）

- 新BCE＋hard-rank纯目标模块已独立只读spec/quality review通过；原host q精确0/1分类先于FP32，不把soft或unknown作rank负例，λτ1及F_known/C冻结。
- 主代理用原C_P_BCE完整正式macro0缓存一次验证新目标：16case/19field/60candidate，18rank-active/1无rank，2soft；独立loss/总coef逐位一致，0模型forward/0optimizer/0新训练曝光。不是重跑旧model或synthetic/unit/smoke。
- BCE1.5607420、rank1.0335380、total2.5942800；margin总coef L2/BCE2.5531，明确目标同时改变方向/尺度，后续需报clip频率。K1/near-rounding/soft-only未由本组覆盖，不冒称全分支已测。
- 仅聚合及3份实际源归档`results/jpair-real-objective-v1/`。此机制时新fresh尚未生成；其后已在独立followup scope真实生成并与原393初态对账（见下节）。新正式macro0/实际更新和100M质量仍未完成，未打开confirm/public，不称Jina完整复现。

## 第一波完整训练与100M D_select（15:32～15:34UTC）

- 四臂真实完成3557optimizer calls／56912case occurrences／99824365logical token及10M/30M/100M三节点，最后LR0／3556positive-LR calls；原四worker已真实退出，各43checkpoint全部保留。
- 100M末次完整375case／615known field／5190candidate／90unknown／coverage1.0。主代理同封存CPU分析器独立回算最大误差1.7903e-15以内，无新模型/eval/confirm/public；9份聚合及详细报告归档`results/clean-trainer-v2/D_select-100000000/`。
- D_BCE/P_BCE/D_CE/P_CE mass依次56.4260%／57.5694%／61.6935%／59.5955%；NLL1.2816409／1.2835494／1.1710422／1.1938659；Brier.5703201／.5733636／.5372062／.5406977。**不是Public Index，不能据61.69%声称超过51.99。**
- CE两布局等权mass优势+3.6468pp，单seed整case配对分域bootstrap描述区间[+.4123,+6.8833]pp，未经多重比较调整且不含seed方差；packing在BCE/CE方向仍相反，未证明学习提分。相对30M四臂mass提高但NLL/Brier全变差，不挑单指标或回头用public选节点。
- 原controller按冻结队列从同fresh启动BI_P_CE/GPU0、PB_P_CE/GPU1，15:34:08UTC实际exact PID回读均存活；未修改原controller。新STATE/JPAIR的独立stock fresh393逐键byte比对已通过，GPU2/3后续队列与正式macro0外部门禁仍在实现，尚未训练。

## Followup正式macro0已过、资源guard保留失败（16:44UTC后）

- 新9源/fresh/cfg未变，ops修复4项异常/竞争/上传边界问题后，经task/spec/security/finalbroad只读review及AST语法，真实install-only/readback通过。16:44:17UTC main显式启动queue，原0/1任务不变。
- STATE与JPAIR各实际正式macro0并exit75/cursor1；各16case/19field/60candidate、20947logical/41894forward/20947backward。独立primary实际重算完整coef/book、actualnative分区/positions/EOS、393grad/clip/delta/Adam/CKPT通过；JPAIR首遍与旧C_P_BCE logits/margins逐位同。State336calls、rank168calls，各6wave。有限fullrow科学证据，不签newP backend/neighbor/serve或完整Base/Jina复现。
- PRIMARY_READY后、MAINauth前，磁盘guard记录free115269525504bytes<required116379385959bytes，差1109860455bytes，queue安全退出。全部4个first/primary child实际wait，原failure/paused/checkpoints/lease metadata保留；**0resume/0AUTH，macro0未重做**。
- MAIN直接回读在rootBLOCKED启动检查拒绝，未读模型tensor或放行。7份正式/primary/资源失败聚合与失败MAIN源已归档。原queue不重启，不降低25GiB每臂或20GiBshared，不删除checkpoint。
- 按原计划预留的post-BLOCKED分支，另冻显式staged controller：旧模型scope完全不改，STATE先、容量fit再JPAIR（若启动时两臂都fit可并发），pending透明未预留，需精确绑定旧资源失败/退出/0resume及MAIN实际回读后新授权。当前该新外部控制器在实现，未续跑。

## 原六臂全部完成；三mask质量负结果（17:37～17:47UTC）

- 原BI/PB各完整3557calls/99824365token/三D_select并exit0，原controller17:37:42UTC全部六臂complete并退出；原完成观察器已返回，不重启任务或观察器。
- 主代理新standalone CPU保存结果回算三mask×三节点，375case/615knownfield/5190candidate/90unknown全覆盖，最大误差1.8874e-15以内；原C_P_CE控制回算等于既有结果。0新模型/eval/optimizer/confirm/public。
- 10M/30M/100M mass：causal52.0806/55.6754/59.5955%；BI37.5653/43.4677/48.1857%；Candidate-PB41.1644/41.0244/43.2434%。100M BI−causal−11.4098pp、PB−causal−16.3521pp，描述性单seed case区间均负；不是跨seed或多重比较调整结论。
- 100M NLL causal1.1938659/BI1.3311914/PB1.4399255，Brier.5406977/.6351701/.6852793。当前冻结stock＋LoRA/head协议下非因果mask没有提分；不因此临时改LR/曝光、不泛化为所有双向模型或完整Base/Jina失败。
- 三节点15份聚合+2完成view+1phase回执及actualanalysis源归档`results/mask-matched-v1/`，privatepayload留远端。新state边界与rank仍仅正式首macro/primary阶段，不把Candidate-PB负结果当它们已完整训练。

## Followup显式恢复已实际准入（17:59～18:02UTC）

- staged-v1 MAIN coef精确门失败被保留；实际60candidate单因素诊断定位为校验器先rank后BCE的forward图创建顺序，12coef最大3.7253e-9差。训练producer/独立primary保持exact。新external v2只按冻结BCE→rank顺序修正MAIN，位级门禁不放宽，模型源/权重/optimizer/data/fresh不改、macro0不重做。
- 新v2 install/readback及MAIN真正cache/replay/coef/nativepartitions/393raw-clipped/update/Adam/checkpoint回读通过，0forward/更新；main另发12字段AUTH后，新controller17:59:32UTC启动，旧失败queue从未restart。
- 18:02:03UTC一次增量事件回读确认controller＋两worker exactlive：STATE/GPU2已macro50/1217745logicaltoken，JPAIR/GPU3已macro63/1595242token；都从原cursor1恢复，macro0及explicitresume各1。
- 启动时两臂容量已fit并发；25GiB每臂、5GiB evidence及20GiB shared不降，旧文件全部保留。仅同inode旧lease bytes不改的successor journal，不冒称系统级GPU/磁盘预留或强sameUID认证。
- 14份失败/成功/AUTH/launch/resume/lease聚合及两版本actualexternal源已归档`results/clean-followup-staged-v{1,2}/`；源和完整产物远端保留。10M只读事件及newcontroller exactpidfd完成观察器已接入，full100M/三节点仍未完成，未开confirm/public，不按中间分数改协议。

## Followup冻结10M已独立回算（18:34UTC）

- 同m365/9988659token/5840case occurrences，五臂actualcfg/source/native/data/fresh393 bridge、完整375case/615knownfield/5190candidate/90unknown/259exact分母及前后同cursorcheckpoint核对，maxmetricerror1.77636e-15。
- State mass40.7510% vsCandidate-PB41.1644%，差−.4134pp/case区间[−5.0458,+4.2709]；JPAIR50.9312% vsBCE49.6898%，差+1.2414pp/区间[−2.0663,+4.4769]，vsCE52.0806%差−1.1494pp/区间[−3.9424,+1.7228]。均描述性单seed、未经多重比较调整，不选winner或早停。
- JPAIR NLL1.0805442/Brier.5545982，STATE1.3610731/.6380916；不是Public Index或完整Base/Jina复现。
- 真实saved前365macro尺度诊断：五臂记录scalar clip<1均100%；rank7485active/739zero/8224knownfield（91.0141%）、1784softcandidate occurrences；margin total/BCE L2比均值3.6355不是参数梯度分量比，不单独归因排序方向/尺度。
- 六份聚合与两actualsource归档`results/followup-matched-v1/D_select-10000000/`。0新forward/optimizer/eval/confirm/public；10M观察器退出、30M事件观察器已接入，原100M协议不变。

## Followup冻结30M已独立回算（18:46UTC）

- 同一封存quality/scaling source实际回算m1122/29985465token/17952case occurrences，375/615/5190/90/259完整不变，maxmetricerror1.8874e-15；0模型/优化/eval/confirm/public。
- JPAIR55.7496% vsCE55.6754%：+.0743pp/case区间[−3.1424,+3.3691]；vsBCE53.1384%：+2.6112pp/区间[−.7468,+5.8556]。NLL1.0606366/Brier.5333580仍略差于CE1.0365784/.5288095；没有稳定赢家。
- STATE41.6300% vsCandidate-PB41.0244%：+.6056pp/区间[−4.4214,+5.5657]；vsCE低14.0454pp。仍无稳定边界收益。
- 实际前1122macro rank22953active/2204zero/25157knownfield（91.2390%）、5290softcandidate occurrences，margin total/BCE L2比均值3.5671；五臂记录clip<1均100%，不把方向/尺度/clip影响混为单因果。
- 六份聚合归档`results/followup-matched-v1/D_select-30000000/`，源码未改；30M观察器退出、100M事件观察器已接入。另在空闲0/1准备终端dense权重packed准入source，不新开训练/seed，不提前读confirm/public，2/3原训练及预算不动。

## 八臂训练全部结束、终点回算与候选已封存（20:27UTC后）

- 新STATE/JPAIR各真实m3557/99824365token/56912case occurrences/三D_select、43checkpoint保留，actualexit0/wait、controllercomplete退出，macro0/ownresume各1；原所有失败与权重scope不改，原八个预定seed42训练臂均已结束。
- 同封存源码100M独立quality/scaling20:27:17UTC通过，完整375/615/5190/90/259，maxerror1.79023e-15。JPAIR mass61.2437%/NLL1.1480786/Brier.5336582；CE59.5955%/1.1938659/.5406977，Δ+1.6482pp/描述性caseCI[−1.9076,+5.1159]，未确认稳定提升。vsBCE Δ+3.6743pp/CI[+.0821,+7.3262]，单seed且未经多重比较调整。
- STATE44.3620% vsCandidate-PB43.2434%，Δ+1.1186pp/CI[−3.7838,+5.9899]；仍比causal CE低15.2335pp。新边界不是完整Base复现，固定rank组合不是完整Jina。
- full3557实际positive-LR3556、五臂记录clip<1均100%，rank active72430/79599knownfield（90.9936%）、soft17293；margin total/BCE L2均值3.3008不是参数梯度比，不能单独归因排序方向。
- 最后9份聚合及完成view归档`results/followup-matched-v1/D_select-100000000/`/completion；actualCPUfreezer按已冻结policy独占封存C_P_CE baseline和JPAIR challenger，两个dense→packed数值失败项保持排除。消费者尚未生产准入，未开D_confirm/public，不能把61.24%的D_select当Public Index或超过51.99。

## Terminal dense→packed真实准入负结果（19:55UTC后）

- 新source经独立whole spec/quality、安全及两项发布/硬链接预算P2的scoped修复复核、AST和install/readback后，19:55:03UTC仅0/1各执行一次原terminal权重的完整375case；不rerundense、不warmup/probe、不训练，2/3未动。
- 每臂完整375/615/5190/90，1554484realT/469wave/13132nativecalls，393FP32 before/after purebytes一致且.gradNone，0optimizer/backward/新训练token，未eval后reload。两workerexit0并actualwait，原全部文件保留。
- 固定旧逐case logits/margin四数值谓词两臂各34/375bad；BCE1choiceflip、CE0，带外均0。Main另从savedraw独立重数一致；两`eligible_packed_candidate=false`，不因CE选择不变或mass61.69%跳过raw门禁。
- 实际packed质量BCE mass56.3371%/NLL1.2799205/Brier.5699076；CE61.6935%/1.1661483/.5350556，仅fixed-weight layout delta不是学习增益/公共成绩。14聚合与2actual源归档`results/terminal-packed-admission-v1/`，原模型/预测不改，不调整阈值或重跑。
- 后续terminal候选/一次确认/完整public协议已独立builder检查并冻结，但候选manifest及实际consumer准入仍是前提；当前未打开D_confirm/public，不借旧历史暴露模型的readiness。
