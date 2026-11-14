# Base-inspired 精确state边界：冻结设计

**状态：严格state末字符版本已实际完整CPU投影及主代理独立回读，ready=false；未执行新模型训练。** 用户已授权自主排队链接方法、不逐臂确认。本研究只迁移Base的state/question两块边界，不下载Base权重、不换成9B/27B、不冒充原slot/LM-head配方完整复现。

## 为什么单独做

- 现`PB_P_CE`只在原`Candidate:\n`后切两块，P中包含question与可能的criteria。Base报告把system+state置P、question+全部options+answer-slot置S，二者不是同一个mask。
- 2026-11-12再次读取[官方报告2.2/2.3](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)及[PR63114固定head的slot_prompt.py](https://github.com/simon-veitner-redhat/vllm/blob/3eded8038a1c0c55eb981e0d38fd301a1eec62bf/vllm/entrypoints/openai/decisions/slot_prompt.py)。PR仍open，作者的大模型分数／候选过滤／缓存验证不是本项目的结果或授权。
- 当前骨干为28层全attention Qwen3-0.6B，读出是每candidate真实EOS两类head，不是Qwen3.5 hybrid/GDN或answer-slot option-letter LM head。因此严格称**Base-inspired state-boundary-only迁移**。

## 最小可归因选择（主代理裁定）

优先**不改变任何输入token**：在已经DATA_READY的原完整pair中恢复精确的state结束token坐标，仅把两块边界从Candidate移到state后。原causal C_P_CE可作为同起点／同曝光控制，因为causal完全不读这个边界。

- P的字符定义固定为原helper的`'Decide whether the candidate is a correct match.\nInput:\n' + text(state)`，**不含其后的换行Question／Available criteria／Candidate标头**。
- 只用state／question／全部options及原helper/tokenizer恢复，target不进入边界计算。用fast-tokenizer offset_mapping核对原prefix IDs，再定位P结束的**真实token边界**，不能用字符串搜索猜长度、取LCP、ceil/floor跨界token或依据答案挑case。
- 若任何已监督train/select field存在跨越该字符切点的BPE token、空P／空S、renderer不匹配、offset不完备，记录完整受影响case的私有身份和聚合原因，**整个此版本准备不ready**。不删field／candidate／case以凑ready，不自动改变tokenizer分段。下一种重新分词/模板方案将需新的匹配causal控制，不夹带到本版。
- sidecar不修改原DATA_READY／任何原case／token／exposure／schedule；所有unknown field仍保留原mask、不恢复监督或额外tokenize/forward。只准备train和D_select，D_confirm不打开。
- 新runtime依sidecar把原`prefix_ids+suffix_ids`无损重切为state-P／problem-S；**重拼后的每个完整pair与原列表逐元素相同**，长度、position0..L−1、真实EOS、slot／micro组成、候选顺序、q、F_known/C均不变。
- 保留原native FA3 prefix_block数学，P双向独立、S双向读P+自身；不同pair／field／case仍cu隔离。训练完整图不能detach P。所有tokenwise运算真实packed，不用失败的cache/fork或SDPAfallback。

## 训练与验收边界

1. 准备全部原train24600／select375的sidecar并直接回读，与原stat／helper／tokenizer／代码绑定。准备器无模型／Teacher／旧adapter，不重建旧去污图，不跑unit/smoke/hash。
2. 只有sidecar全ready才创建新namespace的trainer/config；当前v2八份source/config和已排队BI/PB不变。fresh LoRA/head从同stock/seed42重新生成并与v2共同fresh393张量逐键byte比对，不载已训权重warmstart。
3. 新臂`STATE_PB_P_CE`同原56912case occurrences／99824365logicaltokens／3557calls、warmup107/cosine-to-zero/末步LR0、每100macro及观察点保存全部checkpoint。正式首macro按同合同保存replay、coefficient、393gradient／actualAdamW／optimizer／cursor1证据，不额外吃macro0。
4. 正式首macro还必须验证全部row/position/slot/micro与原C_P_CE完全一致，只有prefix_lengths不同；native调用及全28层realT、精确two-pass replay／393finite／actualupdate均实证后才续跑。当前Candidate-P的数值准入不自动冒充state-P实测。
5. 另一个事件调度器只能在基础四臂完整结束、GPU确实空闲后使用剩余自有空卡；不改原continuation controller，不抢卡，不跨failure自动rewind/replay。必须先完成实测首macro验收，不能为了接队列跳过门禁。
6. D_select仍只三个冻结节点；冻结候选后再按原D_confirm/public协议。新边界的cached serve、稳定速度、完整public数值合同均未准入；不得用旧24.70或dev53.88%当本方法分数。

## 非目标与代价

- 本版不加入Base的CE+Brier λ1、不修改optimizer β、重排／增加options、不做数据合成／重复public选择；这些与mask效应分开。
- 原pair renderer只有条件满足时才附全部criteria，并非Base每question全部slot选项；本版保留这一差异，不称完整Base输入迁移。
- BPE切点可能使本版无法准入。宁可明确失败，也不把不精确边界、删case或新分词冒充mask-only对照。
- 未有跨seed/确认集/public证据前不宣称超过51.99；所有资产、失败、partial保留，不commit/push/发布/租卡。

## Builder检查后的接口裁定（实施前冻结）

1. offset坐标只使用实际`prefix_text`的Python字符索引，不用UTF-8字节长度。每offset为`0<=start<end<=len(prefix_text)`，字符跨度的有序并集必须覆盖原prefix全文，重叠／重复跨度允许（Unicode一个字符可对应多个token），零长／不可解释special offset不准入。所有token必须唯一分到`end<=state_end`或`start>=state_end`，前者在序列中构成非空真前缀；任一`start<state_end<end`记crossing。P最后跨度end与S第一跨度start均须等于state_end，禁止跨同一字符的token组切分、有缺口取整或LCP。重算IDs逐元素等于原prefix_ids是必要但非充分条件。
2. sidecar版本`base-state-boundary-v1`；外层`roster`为`train/select`，`native_split`为`train/validation`，以`(roster,case_id,field_index,coordinate)`四元组核对身份。case原顺序及全部field/known mask完整覆盖，不能按当前读取行号join。known保存`state_prefix_tokens/original_candidate_prefix_tokens/candidate_count/pair_lengths`；unknown的这些值全为null，既不解释raw_options为候选数，也不tokenize raw。恢复必须通过train/select共享的Pair构造入口；缺sidecar不fallback旧boundary。
3. Task1是独立CPU投影器，手动验证实际data/helper/tokenizer/source pins，不构造新模型Context，也不绕过它。Task2新fork的namespace规则、reader源码bytes/sidecar pins进入冻结scope；独立生成fresh后才与旧fresh oracle字节比较。最终config必须在macro0前冻结，macro0主代理报告是外部cursor1续跑门禁，不回填改config。只要求原输入/几何/曝光/初态相同，不要求state-P的logits/梯度等于causal/Candidate-P。
4. 保留原continuation对GPU0/1的BI/PB排队预留；只有核实部署映射一致、基础四臂真正完整结束后，才考虑空闲自有GPU2/3。不抢0/1的短暂空窗。
5. `COMPLETE.complete`仅表示全量扫描+回读完成；`ready`独立判断，任一known失败则ready=false，并保留完整私有失败身份清单。下游必须同时核对二者与所有bindings。stat不是内容hash，不扩大完整性声明。

## v1实际阻断（14:04UTC已主代理独立验证）

完整24600train/375select、41674/615knownfield中，分别28676/427个跨state字符末端的BPE token。producer实际100.6339s，独立primary端点枚举回读47.5031s；所有原字段、unknown、candidate/EOS及身份绑定核对，ready=false，不是原干净数据failed。v1不得继续Task2，原报告/源码/所有private sidecar保留，后续任何固定分隔符定义都必须作为独立显式变体，不改本版失败。
