# R2完整题目标对照：数据冻结实施计划

> **2026-11-12 修订（取代旧起点和小预算条款）：** 已证实旧 selected xLAM 与 ToolRet 有94个完全同文请求、涉及98条执行请求，旧case图未对称解析JSON `request`。因此下文早期“从较优joint起跑”“HISTORY可replay”“最多5M”不再作为新公平冲榜训练的准入依据。旧图审计结果保留，但仅是其明确精确state/native-case政策下的候选库存，不是完全去污证明。

## 当前有效的覆盖修订

1. **起点：** 同一原 ModelScope Qwen3-0.6B 基座，新建并冻结成对相同的LoRA r32/alpha64与FP32 head；不加载项目已训adapter/head/optimizer或旧Teacher标签缓存。不宣称基座预训练绝无评测暴露。
2. **对称输入去污：** 对历史／public／native train/dev两侧按任务语义提取已知请求锚点，包括已查明的JSON `request` 包装；以membership taint覆盖原case component及其派生view，再冻结训练／D_select／D_confirm。通用schema词（如user/string）不作为任务case锚点盲目union，避免巨型伪component。
3. **历史标记：** clean分支不因为HISTORY或来源unknown可供旧机制replay就放行；来源、许可、完整case及语义覆盖不足的记录先隔离。原历史权重／数据可继续用于明确标记的机制诊断，但不作为clean训练初态。
4. **曝光合同：** BCE／Field-CE每臂预先冻结延伸至100M逻辑pair token，10M／30M／100M为预定观察点；完整macro截止、尾部、重复次数及五域组成在数据准备完成后固定，两臂同一有序曝光。双pass实际forward token单独记账，不偷换成逻辑token。此条取代原第9、10条的5M/10M总上限。
5. **布局归因：** padding-free同权重完整public另见Attention与效率计划。clean训练首先做同初始化／数据／loss／曝光的dense与packed比较；同GPU时间多训练的token及更新另报，不把数据清理／新初始化／CE变化归给packing。
6. **当前状态：** 新语义覆盖层、D_select／D_confirm、100M曝光清单和clean训练尚未实现／运行。旧R1恢复public已完成24.67，不再等待它，也不重跑已完成R0/R1/完整审计。

### 2026-11-12执行中的有限准入细化

- 独立架构核查后采用：**旧图封存只读＋完整native坐标census＋有限source/input-shape正向白名单＋input-only负向membership＋已知JSON request对称解包＋完整native case**。不是等所有未使用来源恢复后才开始，也不是只沿旧flags就训练。
- 正向准入必须证明当前快照的source/shape/adapter与真实完整请求、native group/variant及监督方向；未知兄弟source/shape/许可会隔离整个component，不能删字段放行。
- 负向public/history/native-dev只播种taint，不需要逐来源先证明它们是“完整任务”才能用于保守排除。旧input-only文本view可作为ban-set，**不因同属public请求而union其叶子**，不能把位置丢失的文本view称作已完整恢复public任务。
- 封存旧`state` key已有原公开provided文本／history／native-dev库存；可查旧key后沿parent到finalroot读taint，新增SQLite仅存明确wrapper解包alias，避免复制数百万原文或重建旧图。新增转换必须有完整遍历/失败分类回执，不能拿旧count冒充新解包完成。
- 未审阅native-dev本身quarantine，不代表同请求train已排除；负向边界仍需覆盖它们原state及明确可解包表示。已知无法覆盖的本轮相关布局须局部补ban或隔离该native scope。
- 许可只审实际拟准入scope的exact(source,license,license_use)组合；标准明确许可别名及有一手证据的来源例外可显式冻结，未知/Custom/Request/Other不自动放行。不能要求给所有未使用源作绝对完整provenance背书，也不能凭source级attestation替代实际兄弟坐标完整性。
- 训练库存与dev准备分开记账；dev缺额不阻止CPU安全库存准备，但正式成对实验仍须规定配额。首轮每域最多5000候选，编译后合法完整train库存至少2500，否则明确incomplete，不默许一个case重复凑100M。unknown字段保留，已知合法K1也保留；unknown/unrenderable字段只mask、不为制造输入而造候选，两个loss臂统一forward规则。
- 100M为完整macro/domain quota的**ceiling**，必须报告实际逻辑token、尾部、未用配额与重复；family权重以合法编译库存的case数平方根为基数。10M/30M/100M为不超上限的最大完整16-case前缀，不把上限写成实际已消费。
- `example_id`保留原始provenance但不替代group；noul q→Yes的方向须引用实际typed importer合同，不只靠`[q]`形状认定。
- 新坐标metadata阶段已实际完成并直接复核2544892坐标／1868994finalroot-flag组合／332绑定文件；667sources许可库存齐全，但**DATA_READY仍false**。原flags-only potentialdev多出的149行均缺原始输入身份，不能因此绕开旧roster门禁。

这里的“clean-origin”只表示fresh项目未训stock起点和本轮声明的保守输入准入，不证明全来源所有语义近重复／未知别名或基座预训练暴露绝不存在。最终ready仍须原tokenizer/renderer、编译跨split输入碰撞、case配额和完整产物回读证据。

以下原文保留为研究决策历史；与上述修订冲突时以上述条款为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已直接验证的case暴露图上，冻结同一份合法完整case训练数据、新D_select与D_confirm，供题均BCE／完整field CE的同预算对照；不使用旧readout dev选赢家，不把审计eligible库存直接称独立性能结果。

**Architecture:** 数据准备CPU-only、新namespace`/tmp/opencode/jev-06b-research-20261112-r2-data`。只读已完成validation-audit-v2的SQLite／parent／flags及原始完整题，使用原KD同一tokenizer／pair renderer编译，不加载模型。训练父权重在R1恢复public全量复核后另行冻结，不影响先行的数据身份。

**Spec:** [已批准研究设计](2026-11-12-06b-top-public-research-design.md)R2、[已完成case审计](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/results/case-audit/FINAL_VERIFICATION.json)。用户已授权自主迭代；发布／付费资源／更大模型不在授权内。

## 已有事实与不能偷换的范围

- 审计完整读取2529342条原typed train＋15550条validation、恢复5561805图节点；10770条val在精确来源政策下eligible，8336component。D_select／D_confirm尚未选取或评模型。
- “Tools600行”不是BFCL真实工具验证：当前registry将100001条typed train Tools归为5个procedural来源；原global-pool另有native/flat/canonical，不能说旧模型没见过真实工具。浏览器5596条train被旧registry归Language，已有browser验证view因历史case来源未知而quarantine。
- 本轮先保持旧domain registry作为对照的数据分层，映射修正／新增真实tool来源另立同数据控制，不改全局override文件。两个loss臂共用全部新数据变化；任何相对父模型提升不能全归为CE，必须看匹配BCE控制。
- 原native score可能是回归／独立多标签，而非互斥分布。没有来源证明时不归一化、不偷转choice；只mask未知监督并记录完整case结构。仅全部字段均有已知onehot标签的case才可报完整case-exact。

## 冻结规则

1. **按最终component分组。** 候选validation component必须所有原行均满足审计eligible；任一兄弟行缺身份／历史／public／不完整／未恢复来源连接，整个case不入选择集。case之间以图root作身份，不用旧dedup行ID。
2. **监督语义。** choice保留所有原options，target须有限、范围合法、长度一致且sum≈1，不重新归一化；noul只接受已声明原生单概率，转Yes./No.及[q,1-q]以匹配当前pair服务协议；未知score/非法target字段保持unknown mask，不造负例。没有已知字段的case拒绝。
3. **许可。** 首轮私有研究允许明确商业或非商业研究用途、且有可识别许可的来源；unspecified/other/需另行申请及不能确认的混合条款先隔离。记录原license／license_use，不宣称已完成全部上游法律审计，不公开原文或权重。
4. **选择集与确认集。** 每主domain固定75个component为D_select、另50个为D_confirm，data-seed20261112，按source-family均衡的固定抽取；跨domain component只归一个预定primary domain，其全部已知字段保留并另报field-domain指标。无法凑齐不静默减少配额，准备标incomplete。
5. **主domain。** 已知字段domain计数最多者为case主domain，平票按固定knowledge/language/retrieval/tools/arts顺序；不看模型预测或答案类别决定。domain来自原未改动的registry。
6. **验证长题。** 已抽取的case保留完整候选；超共同40960上下文字段预先记静态unsupported，所有模型同样计覆盖缺失，不截断、不在模型跑后删题。主指标明确已知字段覆盖，不把masked score当官方支持。
7. **新训练。** 只原native train、case图不含PUBLIC/FORBIDDEN/NATIVE_DEV/INCOMPLETE；HISTORY可合法replay。历史来源unknown标记只在已保守隔离验证的前提下允许原生已知训练case，不据此解除任何validation quarantine。
8. **训练长度与预算。** 单完整pair上限8192，超过则拒绝整个训练case，不head-tail裁剪或删候选。先准备最多2500～5000候选case/domain，再按case原生source-family平方根加权，不穷尽tokenize全2.5M行；所有过滤／缺额／K／长度实际计数保存。
9. **曝光同一。** 5个domain各上限约1M supervised pair token，合计≤5M，取完整16-case macro前缀；末尾不足macro不训练，两臂/两seed使用完全相同case多集。seed42/43仅改变已冻结case次序；paired BCE/CE同seed次序相同。不用模型正确性重采样。
10. **计算预算单位。** 若采用内存有界双passVJP，两臂都同样两遍：每臂≤5M监督逻辑pair token、≤10M实际forward输入token（backward另外记录）。不能与旧5M菜单prompt token混称同预算。

## Review Focus

- 图root必须逐级寻根再读flags；读child旧flags或把来源unknown/quarantine误解成可用dev均阻断。
- component全部兄弟字段／来源都记账；mask未知监督不是删除已知field候选，也不是伪称支持全case。
- no-target-in-input、Native q顺序与Yes/No顺序、soft-target sum容差及全部option保留。
- 许可／长度／source-case库存只在模型评测前确定，不根据D_select／public选择更容易的数据；D_confirm只作冻结候选确认，不能反复筛选。
- Source／code／audit产物／原tokenizer自身绑定、私有raw/token数据不进Git，不改正在评测的R1和旧global配置。

## Tasks与接口

- [ ] 独占创建新数据root，绑定完整audit原件、原17份train/validation及tokenizer；核对Node/flag字节数和Source／CODE。
- [ ] metadata-only census及case sampling；按原字段坐标恢复完整case，保存known／unknown mask、许可与拒绝原因。不把同parent-view短id当一个case。
- [ ] 冻结D_select／D_confirm及互斥component检查，保存实际raw行数、case数、field数、K／长度、source/domain组成。输出私有`data/{select,confirm,train}.jsonl`及`DATA_READY.json`，公开仅聚合。
- [ ] 以同tokenizer重新pair编译，原state/instructions/criteria_context，prefix和candidate分别分词，EOS实际末尾；无旧菜单input_ids复用。
- [ ] 独立只读review后正式CPU准备，主代理直接复核来源／split／预算／全部候选。此完成不等于任何模型训练或新public分数。

## 后续模型实验（须另有冻结执行计划）

R1实际public完成后从可核验较优joint初始化。计划BCE/CE×seed42/43四臂、每GPU一臂，所有数据／起点／预算／macro／optimizer一致，只有loss不同。主选择指标预先固定为case统计单位、五域加权的已知字段argmax target mass，归一NLL／完整onehot-case exact及覆盖作并列解释，不能只因CE自身NLL改善就声称Public提升。D_confirm不用于换臂／换seed；冻结赢家才跑完整public，并保留所有失败臂。

本文件是数据工程计划，不是已开始R2，也不保证达到51.99。当前没有新D_select/D_confirm或新训练分数。
