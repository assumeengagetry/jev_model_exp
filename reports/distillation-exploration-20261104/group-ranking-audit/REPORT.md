# 排序蒸馏可行性：可信关系比表面组数少

**当前100万缓存中，按已审查的闭集来源筛选后，Knowledge／Retrieval可保留3647条正负关系，只覆盖全池约0.73%的pair。** 因此，不能把此前机械统计的33849个K/R/T单正组直接当成可信排序监督；排序仍可探索，但不能假定其覆盖充分或一定优于现有逐pair蒸馏。

2026-11-05北京时间12:09，已通过独立SSH对实际百万池的metadata、未截断token长度索引和现成Teacher概率做统计；没有GPU前向、测试、烟测、哈希、重新抽训练数据或新增Teacher标注。原有第三波训练未改动。

## 实际能留下多少

| 关系来源 | 可用关系数 | 涉及pair | 占百万池pair | Teacher方向与来源一致 |
|---|---:|---:|---:|---:|
| Knowledge＋Retrieval闭集标注 | 3647 | 7294 | 0.73% | 94.16% |
| Arts的HH-RLHF原始偏好 | 1599 | 3198 | 0.32% | 56.35% |
| 两类合计 | 5246 | 10492 | 1.05% | 不将其解释成Teacher真实准确率 |

闭集关系按域分别为Knowledge 2306条、Retrieval 1341条。上述Teacher一致率只比较缓存q的相对方向与训练来源标签，不是人工复核后的真实准确率；没有按Teacher方向筛选这批关系。

| 闭集来源 | 关系数 | 负例含义 |
|---|---:|---|
| native/arc | 69 | 非原始正确答案的选项 |
| native/mmlu | 1635 | 非原始正确答案的选项 |
| native/winogrande | 275 | 非标注正确的填空选项 |
| overnight/openbookqa | 117 | 非原始正确答案的选项 |
| overnight/sciq | 210 | 原始distractor选项 |
| native/banking77 | 190 | 非gold意图类别 |
| native/clinc150 | 330 | 非gold意图类别 |
| overnight/massive | 193 | 非gold意图类别 |
| overnight/esci | 628 | 同一query-product的错误ESCI关系类别，不是负商品 |

HH-RLHF的chosen／rejected是相对偏好，不能把rejected解释为绝对错误。Teacher在这部分来源方向上的一致率较低，不能因为Arts存在1599组就直接加入Teacher排序辅助项。

## 本次筛选规则

1. source精确匹配上表及`overnight/hh_rlhf`，不按整个domain放行。
2. 从全局`groups.parquet`解码原始group，拒绝空白、空值占位；核对group对应source。
3. 以`(domain, source_id, global_group_id, state_id)`分组；state ID代表完整state／question／criteria，不跨组借负例。
4. 整组来源标签必须严格为0或1，保留至少一个正例和负例，candidate不同；不阈值硬化软标签。
5. 每组用seed42均匀选择一个正例和一个负例，选择不看Teacher。允许原组有多个正例，但不比较正—正。
6. 两个端点的原始prefix＋suffix＋EOS均不超过4096，避免裁剪导致同state在两个候选上看到不同上下文。
7. 百万池其余样本不删除、不改标签；此处只是确定潜在排序辅助关系，没有改变实际训练。

去重和按pair抽样会使组残缺。已有可信正负两端仍能做pairwise，但不能把“缓存只剩一个正例”说成“原始完整单正组”，也不能直接当完整listwise集合。

## 不应直接纳入的来源

- **Tools：** xLAM／Glaive／When2Call-select中的0主要表示参考调用集合外的工具，未确认逐个不适用；目前没有无条件的非合成Tools白名单。When2Call-pref另有228条保留关系，但标注生成与审查来源尚未确认，不并入主集合。
- **typed_full：** 统一展开与去重不能证明每个源的负例语义；在逐源回溯前不整体放行。
- **canonical_nli：** group来自独立pair行号，不是原候选组。
- **GSM8K／MATH：** 其他题答案的字符串差异不保证数学不等价，暂不作为最保守排序负例。
- **合成来源：** 已另行统计，保持独立，不冒充真实业务或人工负例。workflow_policy可保留6321条、numeric_constraints 917条、chess_move_legality 749条；是否采用及如何配对控制另作决定。
- **unjudged检索候选与evaluation-only数据：** 不补作训练负例。

## 对后续实验的影响

- 保持第三波4组真实训练和自动正式评分不变；先取得本波结果，再决定下一波GPU分配。
- 排序支线若启动，优先检验上述K/R闭集关系，明确辅助项只有0.73%的pair覆盖，不强凑Tools或加入当前Teacher较弱的Arts偏好。
- 必须共用atomic二行unit与sampler plan，配置组批HARD／HYBRID以及来源排序／Teacher排序控制；不能让改变batching的收益冒充Teacher收益。
- **Teacher方向过滤不是默认步骤。** 如果以后只保留Teacher与来源同向关系，来源排序对照也必须使用完全相同的筛选集合；不能只在Teacher臂筛选后直接宣称蒸馏更有效。
- 关系数、来源一致率均不能替代真实训练和37项正式benchmark的效果验证。

## 证据边界与产物

- 原始统计：[SEMANTIC_GROUP_RELATION_AUDIT_20261105.json](SEMANTIC_GROUP_RELATION_AUDIT_20261105.json)。
- 已保留的两端selected-index：[SEMANTIC_GROUP_RELATIONS_20261105.npy](SEMANTIC_GROUP_RELATIONS_20261105.npy)，包含K/R闭集与HH-RLHF两类，不能把整份数组当作已经决定使用的Teacher训练关系。
- 来源语义证据来自本地缓存快照的`build-global-pair-pool.py`、`prepare-main-extra-pairs.py`、`prepare-overnight-pairs.py`及`pair_training_runtime.py`。主代理已阅读关键闭集转换、隐式工具负例、全局去重及截断实现；未证明本地源码快照与远端历史转换版本完全一致。
- 当前没有排序训练结果；不把源码审计或这些计数称作排序蒸馏成功。
