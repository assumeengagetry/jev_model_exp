# Readout三路线：原理澄清与受控实验草案

**状态：2026-11-09已获准作为低优先级适配实验实际推进；正在准备独立实现与完整题数据，未有新训练／评分结果。禁止额外测试／烟测／hash。**

当前优先任务是五维蒸馏参榜模型及其获准的官方公共运行。本研究不改动、暂停或抢占主任务；2026-11-09用户通过跨会话交接明确授权见缝插针实施。历史大预算建议不自动生效，本轮有界范围与资源门槛以第9节为准。

## 1. 来源与已纠正事项

已读取同项目其他会话中关于完整候选分布、pair监督与整题监督边界的可见答复，以及共享产物`jev.md:126–145`，再回到作者源码核对。会话是线索，不是模型实现的一手证据；未提取／保存其他会话的隐藏推理或完整转录。

### 单token码并非普遍成立

- `AA–ZZ`共有676种字符串，不能断言全部在所有tokenizer和前后缀下都是单token。
- Mapika、MORJ旧版、Sieve codes、Perplexity的已核实实现主要是枚举`A–Z`再枚举双大写字母，按实际tokenizer筛选，取前255个有效码；不是这些模型统一先用62个字符。
- 62字符`A–Z/a–z/0–9`再扩展属于引用的Intern-Decision lift。该methodology明确是562个“仍为单token”的双字母码，不是全部676个。
- 独立encode单token还不足以证明答案槽／选项行中的边界BPE一致。选项码是token ID与展示字符串的绑定，必须固定目标模型及模板；不能把Gemma/Qwen3.5的codebook直接迁到当前Qwen3-0.6B。

### 14,493/151,024的精确含义

该数出自methodology的`notes.lifts`中Intern-Decision 0.8B/2B/4B的特定运行配置，约9.60%的原始请求行需要解除某个serving guard，不是所有readout都必然损失该比例。

原文包含三种修改：

1. 上下文guard由8192放宽到该模型宣称的262144窗口；其中1681行超过8192。
2. 16问题的校验限制按16题分块校验放开，仍按同一skeleton一次forward。
3. 原62码字母表增加经tokenizer筛选的双字母码；模型原训练只覆盖62码。

大K、大Q、长上下文的行可能重叠，不能把各数字直接相加。lift是解除已知实现的容量／部署限制，不是证明模型已学会新增码位，也不是通用的分块softmax拼接算法。

## 2. 三类readout分别做什么

共同输入：state、一个question、K个完整候选；共同输出：本题K维分布。255是每题最多255个候选位置，不是跨任务固定255种语义类。

### A. 答案槽hidden × lm_head选项码行

令答案槽最终hidden为h，code i的词表token ID为t_i，原语言出口W_LM形状[V,H]：

`z_i = W_LM[t_i] · h`，`p_i = softmax(z_1,...,z_K)`。

不需要generate或自回归解码。A可直接gather所需的W行再投影，不必先计算全部词表logits。Mapika/Sieve slots这样做；MORJ旧运行时则先得到一步全词表logits再取码位，语义相同但工程成本不必相同。

答案槽应是输入后缀，例如`Answer: (`最后一个输入token；真实答案码不能进入输入。仅对当前题合法K项归一化。

### B. 答案槽hidden × 独立固定容量head

`U ∈ R^(255×H)`，`z=U h`，取当前题合法K行softmax。U可随机初始化，也可令`U_0=W_LM[token_ids]`。

同样的h、码表、bias、dtype、温度/softcap和mask下，A与copy-init B在初始化时数学上相同。若冻结策略、参数共享和有效更新都匹配，可以继续等价。真正需要比较的是head可训练性、初始化、tie和优化约束，而不是名称本身。

B的输出是候选位置的255个实数通道，不是生成255种字符串，因此**B并不在结构上要求255个单token答案码**；单token码主要用于复制LM行初始化或兼容既有模板。本实验统一使用同一码表是为了公平控制，不是B本身的必需条件。C同样不依赖把答案名称编码成一个输出token。

当前Qwen3-0.6B配置H=1024、V=151936、tie_word_embeddings=true；B仅261120个weight，约26.11万参数。复制成独立U后不会自动与输入embedding tied。删除lm_head引用也不能删除仍被模型使用的输入embedding。

Perplexity版本要分开：v1.1单题末位pooling；v1.2是多题及每题`<|fim_suffix|>` marker。不能把v1.2的joint模板归给v1.1。

### C. 指向／打分候选表示

这一族至少要分三种实现：

1. **共享scalar：**`z_i = wᵀ h_i`，h_i来自候选位置／候选分支末尾。
2. **Sieve pointer：**`q=W_q h_decide`、`k_i=W_k h_opt_i`，`z_i=k_iᵀq/sqrt(d_p)`。最终decide状态可见完整候选列表，作为查询去匹配候选内容表示。
3. **NanoJev：**独立候选路径`state→question→candidate→readout`批量编码，末位表示用共享scalar打分；choice还可加无候选位置编码的set-attention残差，再按本题softmax。它不是简单地在一条拼接序列上读每个选项末尾。

C可以在架构上支持动态K，但仍受上下文／显存／API上限限制。独立分支若不看候选集合，且没有set head，就无法让新增选项改变旧候选间的log-odds；不能把这种IIA限制误当成head没训好。

“一次读出”不必等于“一条短序列”：NanoJev的flat batch可以一次调用处理多个完整候选路径，但其state前缀仍重复，计算量与答案槽式不同。prefix KV/tree共享属于另一个工程变量，不能同时混进首轮readout比较。

## 3. 主研究问题

1. 把预训练词表出口作为固定读出，是否不如允许独立head适应决策分布？
2. 独立head收益是预训练行初始化还是最终可训练性？
3. 在近似相同head参数量、完整同一prompt下，query-key pointer是否优于固定位置head？
4. 大K下的变化来自码的tokenizer支持、训练覆盖、位置偏差还是模型能力？
5. 变长候选、长state、多题打包对覆盖率和延迟的影响是否改变最优方案？

## 4. 推荐第一阶段：四臂、相同输入与训练预算

本轮已确认起点：保留的五维原始标签 [artifacts/all-dimension-supervised-20261108/stage2-final/](../../../03-证据与图表/06-五维训练与Readout/all-dimension-supervised-20261108/stage2-final) 对应的同一实际基座与adapter；实施前核实远端身份并独立快照。不使用正在变化的蒸馏参榜模型。原始Qwen起点对照后置，不与本轮迁移适配结果混排名。

首轮每个输入只含一题及全部候选，统一同一文本、token IDs、候选码、选项闭合标记和答案槽，关闭thinking/生成；统一原生causal attention，不同时加bidirectional、KV/tree共享、calibration、rotation averaging、set-attention或Teacher。

| 臂 | 读出 | head初始化／更新 | 要识别的因素 |
|---|---|---|---|
| R-A | 选项码lm_head行 | 原行冻结，backbone LoRA可训 | 固定预训练词表读出参考 |
| R-Bcopy | 独立255×H | 拷贝同一原行；head可训 | 对R-A主要是head适应能力，不称为纯架构差 |
| R-Brand | 同一独立255×H | 随机初始化，匹配copy头的初始整体norm；head可训 | 比较整套预训练行初始化与norm匹配随机初始化，不单独识别语义／几何／逐行尺度 |
| R-Cptr | query-key pointer | W_q/W_k可训，d_p=128、无bias | 对固定位置head的内容寻址方案 |

H=1024时，R-B为261120参数，R-Cptr为2×1024×128=262144参数，容量接近；原Sieve默认d_p=256，本计划128是为匹配容量的实验选择，不冒称完整复现Sieve。

四臂冻结相同输入embedding，backbone采用相同LoRA rank/alpha/target modules和LR；不让R-A的tied embedding产生额外更新。默认rank32/alpha64，LoRA LR5e-5、可训练head LR1e-4只是起始建议，后续是否调参应给各方案相同调参预算。骨干初始化及数据顺序seed42；先锁定共同backbone/LoRA初值，再用独立head-init seed20261108初始化随机头，避免head参数形状不同改变骨干或数据的RNG消耗。

不加额外head LayerNorm，不仅给某一臂加softcap；先固定T=1、零dropout，softmax只在合法K项。pointer使用最终答案槽query，避免将只看较早前缀的scalar与看完整列表的答案槽模型混为“同上下文比较”；其key位置仍存在causal顺序差异，因此第二阶段单独研究attention交互。

明确的读出接口：`decide_index`为最终答案槽最后一个有效**输入**token；`option_end_indices[i]`为第i个完整候选正文之后、下一候选之前的固定闭合序列的最后一个token，不取候选码起点。所有臂都包含完全同一闭合序列与答案槽，按显式索引取状态，不靠EOS或pad推断。Cptr的W_q/W_k分别采用独立非零Xavier-uniform初始化、无bias；不宣称其初始logit尺度已与B严格匹配。骨干BF16计算，可训练参数沿用FP32存储；首轮所有readout统一用FP32 hidden/weight投影、FP32 logits/loss，A也仅将选定原行转FP32，不改变其冻结属性。

### 数据与预算（建议值，尚未准备或执行）

- 原草案每臂2亿token只是历史建议，本轮每层每臂上限500万有效输入token，具体边界见第9节；四臂用完全相同的完整题集合／展示顺序／曝光次数，不消耗或改变主任务预算。
- 本草案明确选择**原标签决策监督，不加LM辅助loss**。通用部分用项目canonical zero-shot-label NLI的原始premise/hypothesis/label，转换成完整二元noul决策题：entailment映为true，neutral/contradiction映为false，与项目现有0/1原标签语义一致；有合法软目标t则保留`[1-t,t]`。它不是普通文本next-token训练，也不拿现有dedup pair均值当新整题gold。该选择是主代理基于当前原始标签任务提出的默认，未声称用户已确认；若要保留LM目标，需另立统一LM出口与损失配比实验，不能悄悄混入此轮。
- 20%通用＋80%五维任务、任务五维均衡均按**完整prompt的有效输入token配额**计量；四臂使用同一已固定的完整题曝光清单，实际题数按原始train库存和token长度决定。20:80是数据token占比，不意味着按题平均CE的数值权重也恰为20:80。若最后一题将超过预算，四臂采用相同完整题前缀截止并报告实际token数，不截断该题凑整。按完整state／source case／group划分训练与保留集，防同题不同选项串入不同split。
- **不能把去重后的pair缓存硬拼成完整题。** 原构池会平均冲突target、只保留首次source/group，按pair抽样又会缺候选。必须回到原始typed决策或可靠native转换记录，保留完整候选集和原始目标。
- `choice`使用合法单选gold／原生目标分布；多工具／多标签任务保持多个独立noul问题，不能把未调用工具都当负例后强制成互斥softmax。
- `noul`首轮统一false/true两项，避免有的模型一个语义路径、有的两个，混入计算与语义差；`score`保留等级映射，先统一分类CE，不额外加入只对某一臂有利的ordinal loss。
- 全部loss按题平均，不按展平候选平均；合法来源软目标保留原值，不阈值化。候选换序时同步逆映射gold和输出，不能在答案槽插入正确码。
- 训练样本超长应按共同训练入选规则记录，不静默截断；未来正式比较覆盖全输入，不能缩减候选／删困难题来提高分数。
- 记录每个K区间和码位置的监督曝光，尤其63–255，区分“可服务”与“训练过”。

若筛选有明确优势，两个优选方案再用**相同完整训练配方与每臂至少62.5亿新增有效token**确认趋势，并保留相同checkpoint节点。此数是建议确认预算，不表示现在有授权启动或保证某个小时数完成；完整菜单的attention成本不同，不能套用当前pair训练速度估算。

### 预先锁定的选拔规则

- 本轮只比较第9节共同有界预算的最终节点（完整题前缀的实际总token相同），不为某一臂事后挑最有利训练步；原2亿建议不自动执行。
- 开发集来自训练来源自身隔离出的完整题，按五维等权。主筛选量为`D=(1/5)*sum_domain mean_question[CE(y,p)/log(K)]`，越小越好；均匀预测的归一CE为1。通用NLI保留集单独报告保持能力，不重复计入五维主指标。
- 先要求共同合法输入开发集合（K≤255且在声明上下文内）的完整覆盖，失败候选不靠删行晋级；再按D选前二。若第二名边界差距≤0.01，先按预先固定的完整请求median时延择优；若四臂没有超出0.01的清晰信号，不宣布胜者，按事先指定的Bcopy与Cptr进入长预算确认。
- 正式公共37项及最终保留集用于冻结方案后的报告，不用于本轮开发集选优。若用户之后决定直接按公共37项选型，则必须将该分数标作自适应选型依据，不能再称同一份分数为独立确认；不同seed重跑也不会自动产生独立测试集。

## 5. 未来正式研究指标（现在不运行）

本轮仍不新增测试／烟测／hash；仅实施第9节登记的有界适配质量／成本比较，不为本研究追加大规模官方公共运行。本节的完整榜单与更多网格仍需另行授权。

### 质量与校准

- 主指标：冻结版本的37项五维公共评分，完整coverage、每维skill、每基准结果；不把public分数冒称私测Full。
- 分类/检索选择补充：accuracy／MRR等任务原指标；NLL、Brier及固定bin ECE区分挑对与概率可信。
- score保留原指标，并报告等级误差／分布质量；不能仅看总准确率或训练loss。

### 大K与多题容量

- K分层：2、3–10、11–26、27–62、63–128、129–255，检查26/62是否出现断崖及训练未覆盖码的泛化。
- Q分层：1、2–4、5–16、17–32；Q容量与K容量是两个独立维度，首轮主实验Q=1，之后再研究多题结构，不把限制混为head=255。
- 按模型/codebook的原生支持与lift后的支持分别报告coverage；lift不等于已训练能力，不拼接各块独立softmax冒充全局分布。
- 选项重排后映回原ID，对正式预先选定的研究样本报告决策一致率、概率距离和位置偏差；不是根据正确性重试选择最好答案。

### 工程效率

- 相同硬件、dtype、相同完整请求，报告有效输入token、前向调用数、候选路径数、峰值显存、median/p80/p95时延和吞吐。
- A也采用selected-row投影，不能强迫A算全词表再宣称B更快。B/C首轮使用相同骨干输入，prefix/KV/tree优化留到独立工程阶段。
- 同题／同token曝光是主机制预算；若后续比较NanoJev独立分支，再额外按GPU时间／实际处理token报告，不能把“一次API调用”当作同计算成本。

## 6. 第二阶段：有信号后再扩展，不一次堆机制

1. 对固定位置head与pointer优选方案做causal／prompt-bidirectional的交互对照；mask是单独变量，不能只给pointer解锁全上下文。
2. pointer对共享scalar作消融，分辨最终query的价值；再对NanoJev式分支scalar、有/无set-attention作同分支输入对照。该阶段的重复prefix成本单独计量。
3. large-K训练覆盖消融：同题数／token预算下比较原生完整大K覆盖与仅小K；不要把扩codebook后的coverage当成训练收益。
4. 最后才研究多题joint、question isolation、共享prefix/cache与calibration，并单列速度／质量变化。

## 7. 判定规则

- Bcopy胜A，只能先说明额外head适应能力有用；不能直接称255-way“结构更强”。Bcopy胜Brand才支持初始化有用。
- Cptr若只在大K获益而小K不变，才进一步检验内容寻址／位置容量解释；不能从不同榜单模型横向分数推因果。
- 任一方案质量增益须同时看coverage、长输入、大K以及时间成本；固定head不保证快、pointer也不自动消除所有顺序偏差。
- 新候选最终结论应有另一次独立正式训练或不同数据顺序的确认；不是同seed重复多次就自动得到跨seed统计显著。

## 8. 一手资料

1. Mapika selected-row答案槽：[decider/model.py v1.6](https://github.com/Mapika/decider/blob/23579f7a7e8f10e1045be492af3c1c05a005d67c/decider/model.py#L18-L27)。
2. Sieve独立codes head：[sieve/codes/model.py](https://github.com/sthanika-ai/Sieve/blob/1e64657c930f91db7119b5638774c0f11a77e441/sieve/codes/model.py)。
3. Perplexity v1.1：[autojev/model.py](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/source/src/autojev/model.py)；[v1.2 card](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/cce0f3f80033e9d89d80831d5281fd72124aacce/README.md)。
4. Sieve query-key pointer：[sieve/model.py](https://github.com/sthanika-ai/sieve/blob/main/sieve/model.py)。
5. NanoJev候选分支scalar＋set-attention：[train_toy_decisions.py](https://github.com/TianyuCodings/NanoJev/blob/main/scripts/train_toy_decisions.py)，`DecisionModel.forward`；[README](https://github.com/TianyuCodings/NanoJev/blob/main/README.md)。
6. 真实codebook选择：[Mapika label_table](https://github.com/Mapika/decider/blob/23579f7a7e8f10e1045be492af3c1c05a005d67c/decider/prompt.py#L38-L55)，[Sieve codes/prompt.py](https://github.com/sthanika-ai/Sieve/blob/1e64657c930f91db7119b5638774c0f11a77e441/sieve/codes/prompt.py)。
7. lift原始metadata：[Decision Index data/methodology.json](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/methodology.json)，`notes.lifts`下Intern-Decision条目；本地也有[research/2026-10-27/raw/leaderboard-methodology.json](../../../../jev-辅助归档/2026-11-09整理/辅助文件/research/2026-10-27/raw/leaderboard-methodology.json)快照，已对照线上内容。

## 未执行／未知边界

- 未读取模型权重张量，shape来自源码／配置，未验证任一发布checkpoint张量值。
- 未运行当前Qwen tokenizer确认255码实际集合及上下文分词；实施时必须作为输入构造准入，不能照抄别的tokenizer表。
- 2026-11-09授权后开始独立准备；截至该修订仍无新GPU训练／评分结果。没有下载大权重或新训练语料，不改动已运行主任务，不上传、提交或推送。
- 独立只读设计审查已完成：修正了通用数据目标接口、pointer索引／初始化、初始化收益解释和预先选拔规则。没有因审查追加任何实测；数值预算与默认设置仍是供用户决策的研究草案。

## 9. 2026-11-09已授权的低优先级执行范围

授权原文与边界见本目录 `HANDOFF-20261109.md`，不覆盖其中主任务优先、无额外测试／烟测／hash、不发布的限制。以下有界数值是实施者在授权的窗口裁量内固定的首轮范围，不是用户要求铺开长预算。

### 两层、同一起点、四臂

1. `frozen`：共同已有主干完全冻结；A固定参照（0次参数更新），Bcopy／Brand／Cptr仅更新新head。可以一次提取共同完整输入的hidden缓存，明确区分每臂逻辑曝光与共享实际前向成本，不把缓存后的head耗时当成完整请求延迟。
2. `lora`：A／Bcopy／Brand／Cptr各自重新加载同一初始快照和各自**未适配的新head初值**，仅原rank32 LoRA范围及允许的head训练；不继承frozen层的训练后head或其它分支的权重。保留pair classifier为单独完整方案参照。

每层每臂上限5,000,000有效完整prompt token；原标签监督、无Teacher、无LM辅助loss，20%通用、80%任务，任务五维按token等分。按完整题前缀截止，真实共同曝光以生产manifest为准；batch16题、micro1题，head LR1e-4、LoRA LR5e-5、seed42；head独立初始化seed20261108，不扰动主干／数据RNG。节点为不超过100万／300万／500万预算的共同完整macro边界。不得自行扩成2亿、62.5亿或更多seed。

### 原始题数据与有界正式研究比较

- 原始typed `full/train-*.parquet`、原始canonical NLI train；原生kind/options/target必须保留，不能读取global dedup pair池反拼。typed `noul`保留独立二元判断；互斥choice／score仅接受原生合法分布；遇到语义不明的多正choice不擅自归一化成单选，登记拒绝原因。
- 按相同state与来源case/group隔离适配train／留出，来源样本和扫描量有明确上限；不从官方评测集取训练题。由于共同已有主干历史上训练过相关来源，留出仅称“适配留出”，不冒称主干从未见过的独立测试。
- 训练统一context入选上限4096，不截断整题；正式研究留出按各方案声明context完整输入，K>255等不支持计coverage。A只gather码行，不算全词表；pointer仍有因果选项顺序差异。
- 固定质量范围：五维各200题，共1000题，通用保持100题；不足库存真实报告。四臂报告归一NLL、期望正确率、Brier、10-bin ECE、score等级误差、K分层与coverage。不得把这些内部适配留出指标写成官方Index。
- 固定成本范围：上述留出每维25题，共125个完整请求，在同GPU、同dtype无输出缓存路径测median/p80/p95、tokens／前向路径／峰值显存；frozen共享feature仅用于head训练/质量，不替代实际完整请求成本。
- 固定换序范围：每维30题，共150题，一次预先登记的逆序并映回原选项ID；不按正确性重试、不加旋转平均。Pair参照单独报告输入包装和重复prefix成本，不把它与整题路线的差全归因head。

### 资源与保存门槛

- 独立根目录 `/tmp/opencode/jev-readout-adaptation-20261109`；本地实现 `/tmp/opencode/readout-adaptation-20261109`，旧模型／代码只读。
- CPU准备最多2线程、nice19／idle I/O，不启动模型推理。先核实主任务当前phase、进程身份及实际future GPU使用，不能凭旧观察称空闲。
- GPU只准3号卡，且必须当前无计算进程、主任务Student进入稳定期（至少200步且不在最后500步）、官方仍等待；主任务切换/阻塞/暂停或3号出现外部进程即不新启动。若主任务已完成官方运行，可在再次确认无外部预约和占用后继续短窗口。
- 每个GPU窗口最多600秒，worker在完整macro边界保存本任务optimizer／RNG／head／adapter并让路；任何控制只作用本研究自有进程，不给主任务发信号。每次新窗口重新准入，不能长期锁住主任务之后要用的卡。
- 不对主任务修改资源配置或追加任务；任一阶段没实际完成时保持incomplete。阶段报告列“已准备／已训练／已评分”和净新增，不预填胜者。
