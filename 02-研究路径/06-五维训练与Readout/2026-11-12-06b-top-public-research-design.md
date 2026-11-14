# Qwen3-0.6B冲击同规模JEV Public最高分：研究设计

**状态：前期只读研究及fresh builder检查已完成（nothing important left）；用户已确认“自主迭代，我睡觉去了”，进入自主执行。授权不包含发布、外部付费资源或更大模型。**

## 1．目标与不能偷换的口径

用户要求“研究探索、找原因、跑实验、跑完整public验证，再迭代”，目标为“直到0.6B级别榜单上最高分”，不是只补一次学习率网格。

- 主模型保持真实Qwen3-0.6B量级；允许研究数据、蒸馏目标、训练参数覆盖、attention和head，但不能把更大模型、额外在线Teacher、查表预测或按benchmark切专家冒充单个0.6B。
- 必须先取得当前官方榜单的日期、commit、edition、实际部署参数和总Public Index；严格同Qwen3-0.6B、相近实际部署参数、<1B跨架构分开列，不能只凭模型名或单领域分数判定第一。
- 现有固定public为edition0.3、43个执行／报告benchmark、37个进入Index、140620执行／140178官方计分，kit commit`25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`。官方README明确0.4榜仍原样接受public0.3；不把不同版本或private Full混比。
- “最高”是超过一份可核验、同版本、同参数口径的当前参考，不是证明全世界所有未公开模型或所有可能训练方案的数学最优。正式线上名次需维护者验收；用户尚未授权上传、发布、参榜PR。
- Public被反复用于调参后只能称公开调参指标，不能继续称独立泛化验证。保留非public验证与最终未参与选择的保留集，不读private任务或用public答案训练。

### 当前可核验靶标

榜单固定commit`44626fbe70223a2b92dffcab72a864178321a5d1`，结果generated_utc为2026-11-11T14:28:34Z，标签Decision Index0.4.1。主代理直接解析并联结当前v03及index全部298条，得到44个公开served_params>0且≤0.7B的模型；该口径内最高为GLiClass v4.1 edge，Public Index **51.99**、服务参数约439.79M。高于它但参数缺失的条目为Jev1.13／1.14，不能假定它们也≤0.7B。44条是官方metadata筛选，不是对所有第三方代码的独立参数审计。

| 层级 | 参考 | Public Index | 使用方式 |
|---|---|---:|---|
| Qwen3-0.6B-Base近亲 | Decision-2.0-Kai-0.6B | 16.68 | 已低于本地24.91，但直接初始化不同，不能称我们已正式同级第一 |
| 实际约0.59B跨基座中间目标 | Sieve0.8B（前16/24层） | 32.26 | 中间里程碑，不是最终最高标杆 |
| 公开参数≤0.7B主目标 | GLiClass v4.1 edge | **51.99** | 主线冲击其public；不更换成该模型直接冒充研发成果 |
| <1B扩展参考 | PKU-KCL-Meteor0.8B | 58.77 | 约0.853B，超出主线预算，不偷换成0.6B目标 |

工程达标要求：模型冻结后完整public超过当前同口径最高参考，并优先以约0.5点余量和独立训练seed／未参与选择的audit holdout复核小增益，避免舍入或单次波动被误报第一。0.5点是主代理提出的工程余量，不是显著性证明或官方Full并列规则；达标时重新检查榜单快照。

**官方主榜Full并不由public决定。** 0.4.1 text Full为60%私有same-skills＋40%私有new-domains（经stock模型等尺度映射），public权重为0。用户此前明确要求跑public，本设计将“自主实验达标”具体化为上述public目标；正式Full榜排名只能在另获发布授权后由维护者私测，不承诺本地能计算Full。

### 本模型的真实参数与初始化身份

主代理直接读取现有base／adapter的safetensors header和下载记录：分类骨干596049920参数＋LoRA20185088＋active score2048＝**616237056（0.616237B）**。base文件还存有未用于序列分类的155582464参数LMhead；该unused LMhead不计实际分类模型，但不能因磁盘有重复张量误称0.772B。后续新增head或rank按完整服务参数重新计账，主线总逻辑参数上限0.7B，不用active／trainable数量偷换。

下载记录为`Qwen/Qwen3-0.6B`、ModelScope、revision master；不是已证明等同HF不可变commit的Qwen3-0.6B-Base。所有后续主线从这份已实际绑定的原基座／现有checkpoint延续，不悄悄替换成新HF main。

主源：

- [固定当前总public／Full数据](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/44626fbe70223a2b92dffcab72a864178321a5d1/data/v03.json)
- [固定完整模型元数据](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/44626fbe70223a2b92dffcab72a864178321a5d1/data/index.json)
- [固定官方kit规则／0.4与0.3关系](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/README.md)
- [GLiClass固定参榜结果](https://huggingface.co/datasets/knowledgator/decision-index-results/raw/664d77bce9149d8758d8c8e51bd5ab74e0d4ac43/runs/gliclass-v4.1-edge/index.json)

公开配方取舍：Kai的endpoint＋global-query头、分离LR及proper losses可作为机制参考，但其public低于当前模型；GLiClass4.1和Meteor没有完整公开训练配方，不能凭推理代码编造teacher／训练数据。Sieve披露前16层全参、低LR5e-6、独立校准，但其Qwen3.5结构不可直接当Qwen3结构。训练期Teacher没有官方一律禁令，若后续使用需逐项验证来源／许可；本项目不允许推理时借大模型、未披露ensemble或benchmark-ID路由来伪装0.6B。

## 2．当前已验证基线

| 模型 | Public Index | Raw Index | 状态 |
|---|---:|---:|---|
| 五维KD final | 24.38 | 43.62 | 两次新实验中的第一次完整public |
| 原joint LR2e-5 final | 24.91 | 44.07 | 已有完整v2 public，当前保留较优模型 |
| 从同KD起点joint LR1e-3 | 0.81 | 22.63 | 完整训练＋完整public，但严重退化 |

高LR实验不是早停、漏题或分数串用：1667242pair／458731913token／13026步／3647关系完整；OOM0；两次各140620执行、140178计分全ok；两次逻辑候选及输入token完全一致。近均匀概率field由0.63%增至25.99%，只证明已有输出区分能力下降，尚未定位唯一机制。

旧joint单卡与新joint四卡归约路径不同；必须控制这个混杂，不能直接将所有下降归结为LR。现有adapter r32／alpha64、196对LoRA＋FP32两类head，不代表已比较全参数训练或更合适的完整题目标。

## 3．2026-11-12新鲜资产／验证集侦察

- 02:48UTC实际四张H800均free、显存0，无计算进程；scratch可用约249.03GiB、CPU affinity176。正式每次启动仍重新准入，不抢占他人任务，不租新GPU或调用付费外部服务。
- 原typed完整题train、validation、test的parquet均还在；只检查validation的schema/source/split，没有读取test目标。
- 原typed validation14977题＋tasksource addition validation573题，共15550题，native split均为dev，group_id均存在；应用现有source-domain-overrides后：Knowledge2552、Language10733、Retrieval890、Tools600、Arts775。**这些是隔离核查前的候选库存，不是已准入独立dev。**
- 已有readout data/dev1100题和reverse150题的manifest明确写着“historical trunk exposure is not excluded”；只能用于适配诊断，不能称整个KD／joint历史未见集。
- 02:58UTC进一步CPU只读扫描了实际选中的818069个task state、763252个task group、416934条generic原始来源及完整public输入（不读取public答案用于训练）：15550条候选validation中，107条state匹配历史选中训练、150条state匹配public；在validation内作state／source-group闭包后排除257条，尚余Arts732、Retrieval858、Knowledge2542、Language10561、Tools600。**这仍只是精确字符串／已存group的侦察结果**；旧dedup可能丢失别名／兄弟group，未知source alias及语义近重复尚未排除，不能将剩余15293题直接宣布已准入。
- global-pool仍有32787042pair、约6.527B token及完整state/question/criteria表、source/group映射；不等于应无差别全量训练，也不意味着与public天然隔离。
- KD、旧joint和高LR新joint的中间checkpoint均保留；高LR从step0到每1000步及13026有COMMITTED marker。旧checkpoint没有新marker，若用于诊断需按原保存协议核实来源，不能伪造新marker或恢复旧暂停任务。

03:00UTC只读读取既有参数张量，未构造／前向／训练模型：

| 参数现象 | KD | 旧joint final | 高LR step1000 | 高LR final |
|---|---:|---:|---:|---:|
| head两行差分L2范数 | 0.80818 | 0.80834 | 0.17634 | 0.24075 |
| LoRA A总范数／KD | 1.000 | 1.006 | 1.362 | 1.648 |
| LoRA B总范数／KD | 1.000 | 1.045 | 2.711 | 3.566 |
| head相对KD参数变化范数 | 0 | 0.0275 | 0.6496 | 0.6870 |

所读6个snapshot均393张量、参数全部有限；高LR step0与KD统计完全一致。高LR到1000步已发生大幅参数重写，差分head范数降至KD约21.8%；这加强了优化破坏的调查价值，**不等于已证明功能坍塌发生的精确节点或已排除数值执行交互**。

## 4．实验前的数据隔离门禁

1. 以实际已选KD／joint训练state及source-family/group为暴露闭包，对候选native validation做全state／语义关联组隔离。不能只检查readout train/dev，也不能只按文件名train/dev判断。
2. 排除与完整public输入相同或有明确关联group的候选；尊重源数据的官方split、许可和NO_TRAIN规则。public标签不进入训练／合成／选择原始候选的代码。
3. 验证集由准入后的合法原生完整题构成，保持所有候选、noul/choice/score语义，不从已拆散pair杜撰完整题；若某类不支持，显式统计不能悄悄转为另一任务。
4. 新dev按source／family／group冻结；另保留未用于选择的audit holdout。源数、题数、长度、K、类别、拒绝原因和历史暴露范围必须记录。group证据不完备的样本不进入“历史未见”主分析。
5. 小规模dev用于机制诊断与候选筛选；完整官方public仅验证预先选定的候选。两者分数不能混为Public Index。

## 5．首轮最小机制对照（主代理拟定，待设计确认）

### R0：已有权重的执行路径／读出精度诊断，不先重训

- 固定KD final、旧joint final、高LR step0／1000／final，诊断集先用已有1100题＋150个语义安全reverse；明确标注历史骨干暴露未排除，只定位机制，不据此宣布泛化涨分。
- 同一原始完整题重新按当前pair渲染取得相同IDs／EOS／候选；不能直接把旧菜单prompt的input_ids当pair输入。
- A为封存v2；B为标准PEFT完整pair前向、BF16投影；C复用B同一hidden，仅将分类投影改FP32。不得同时改变head权重、prompt、attention语义或温度。先检验step0与KD来源及执行行为，再解读后续节点。
- 记录完整field归一NLL、软目标得分、每题指标、field内margin方差／tie率、两类logit公共量级、K≥2分层、reverse一致性；只有原始数据完整覆盖case所有兄弟field时才报case-exact，不把1100个单题当完整BFCL cases。
- B恢复而A不恢复→调查v2对新权重的敏感交互；C恢复而B不恢复→调查head执行精度；三者均差且FP32 margin仍近常量→转向表示／优化破坏。均为可被数据推翻的假设，不预填结论。
- 诊断预算预先有界，只读现存checkpoint，不更新权重；若要宣称修复或完整public涨分，需另外冻结候选再跑完整public，不能把诊断子集称正式分数。

### R1：四卡低LR桥接，隔离本轮执行路径混杂

- 从同KD snapshot使用新四卡路径、原1667242pair／458731913token／13026步、global128／相同顺序、target、relation、warmup／cosine、精度，只将峰值LR改为2e-5，独立输出。
- 补齐预定节点的point／relation分量、clip前范数、head差分范数、LoRA更新尺度、rank参数一致性与固定诊断集的区分能力；诊断观测不能自动替换预注册终点。
- 若四卡低LR恢复到原joint相近能力，支持高LR是新四卡regime内的关键致败条件，但不宣称排除了precision／relation交互；若低LR仍坏，暂停新LR网格，转为相同真实macro的单卡／四卡梯度与一步更新机制对照。
- 实测旧四卡训练约5.92 GPU·h、完整public约1.10 GPU·h，可作为预算锚，不承诺逐次相同耗时。此桥接用于补因果证据，不把“恢复24.91”当研究终点。

### R2：当前较优joint上，题均BCE与完整field CE

- 从当前24.91 joint起步，不从高LR失败final或旧原标签stage2起步；使用通过新暴露／许可门禁的同一份原始完整题和监督。
- 控制臂为每case等权、题内平均pair BCE；实验臂保持相同case权重，对互斥field全部候选margin作分布CE。模型、head、attention、pair编码、样本顺序、训练预算和监督相同，先只研究loss概率空间。
- 多工具是多个独立真假field，不对工具目录强行做单选CE；verified positive／verified negative／unknown分别记账，unknown不当负例；同case字段平均。
- 先每臂上限10M**实际组装pair token**、seed42、同一完整case顺序、峰值LR2e-5，余下AdamW／warmup3%／cosine／decay0.01／clip1沿用已验证配方；head与LoRA不在此轮额外引入不同LR。对完整候选／长题的训练可用性逐case统计，不与历史5M菜单prompt token混称同预算。具体有效case数由准备后的冻结manifest决定，不伪造预算；实际终点取不超预算的完整case／macro前缀，两臂完全一致。
- 候选只按新D_select选择，D_confirm核验后冻结；只晋级预先选定版本做完整public。若两臂同涨同跌，优先考虑数据／续训效应，不能归为CE独有效应。

### 既有研究不重复与后续路线

已有500万完整prompt token的frozen／LoRA四臂readout、KD温度／软硬标签和3647关系消融，不原样重跑。尤其Cptr历史NLL改善伴随换序一致率79.33%，不能直接部署冒充已验证赢家。

R0／R1排除执行故障、R2证明有值得继承的方向后，下一轮才考虑当前pair表示上的排列等变集合交互、受控attention适配、函数保持扩LoRA rank和低LR全参数训练。改变attention后原共享prefix KV的条件可能失效，必须新Engine／缓存语义验收，不能原样套封存causal v2。

## 6．每轮研究循环

每轮先写“事实—假设—可推翻结果—实际预算”，再启动GPU：

1. **定位瓶颈：** 区分训练实现／归约、优化破坏、标签与目标不匹配、历史数据覆盖、pairwise读出局限和容量／适配参数不足。能由已有日志／输出／checkpoint判断的，不重复重训。
2. **最小对照：** 同一父权重、冻结题组／顺序／预算、同一数值路径，单独比较被研究变量；不同时更换数据、head、LR然后把收益归给其中之一。
3. **非public筛选：** 除总体外报告五域、source、K、长度、类别先验、swap语义安全题的一致性、概率区分／校准；不按benchmark ID运行不同策略。
4. **完整public：** 只有明确选择的候选进入140620全量执行，官方排除后140178计分；不按结果选择重试、删候选或拼接不同checkpoint的逐题赢家。
5. **复现与继承：** 实际涨分后保留原冠军和新候选，独立seed验证可复现性；不能只拿单次最高波动称稳定提升。若无增益，先分析证据、淘汰假设，再决定下一轮，不自动铺无限网格。

四卡分配按任务效率决定：独立小模型对照可一卡一个，长预算训练可用已核实的四卡global-batch不变路径；完整public优先四卡分片。充分利用资源不等于强制每次训练四卡或无依据占满四卡。

## 7．参数、资源和停止边界

- 用户给出的长期目标不等于无限上下文、无限磁盘或新外部费用授权。每轮必须有限、可检查、可恢复，保留实验账本、来源和完成通知；长链不依赖会话记忆。
- 达到同版本可核验靶标后做完整复核和复现；不能未经许可发布。若靶标版本变更、排名口径不明、数据隔离失败、无可信改进假设、非有限值、外部GPU占用或磁盘不足，保留现场并报告，不猜测补齐。
- 不覆盖／删除任何旧权重、checkpoint、optimizer、partial或原始结果，不恢复旧暂停训练。新实验一律新namespace，未知partial不认领。
- 旧的“不额外unit／smoke／模型或语料hash”边界保留。用户本轮明确要求的机制实验、非public对照与完整public属于研究任务本体，不冒称之前已授权运行；以具体首轮设计获确认后实施。

## 8．研究与实现分工

- 独立只读资料研究核实当前榜单／模型公开配方／公平规则。
- 独立只读机制／架构研究检查高LR训练及既有读出／数据研究，给出可证伪的前三个最小实验。
- 主代理负责原始资产／验证集暴露核查、综合设计、唯一实现者、GPU调度、独立审阅集成和实际验证；不让多个实现者改同一文件。
- 设计定稿后再由fresh builder检查缺失的关键决策；设计确认前只做阅读／探索，不写新训练器、不安装依赖、不启动新GPU实验。按研究仓库01/02/03结构保存中文方案与聚合证据，执行源码留库外。
- 本设计一经确认，主代理自主执行已界定的R0／R1／R2及有证据支持的后续有限轮次，不逐个LR／checkpoint要求用户批准；但未经授权不更换更大模型、不租外部GPU、不扩大外部费用、不发布／推送，或在无可信路径时把无穷预算当默认。

## 9．实现计划需要落实的可检验项（不是已知成功）

- 当前主标杆已固定51.99／≤0.7B，但上线正式排名和private泛化仍未知。
- 高LR灾难退化静态未发现足以单独解释的明显实现bug；FP32读出、v2交互与优化破坏须通过R0／R1区分。
- 工程实现者在GPU启动前写死R0预定节点／诊断时限、R1正式观测节点与粗失败门禁，以及R2的D_select度量／bootstrap case单位／D_confirm解封次数；这些合理细节由实现计划决定，不在看过新分数后改规则。
- 新native validation最终隔离后是否有足够可靠五域题组，仍须恢复历史dedup的多来源关联证据；不足则保守排除或合法补源，不用public补训练题。

上述未知不以猜测写成“已验证”，也不使用历史Language单域榜单替代当前总Public靶标。
