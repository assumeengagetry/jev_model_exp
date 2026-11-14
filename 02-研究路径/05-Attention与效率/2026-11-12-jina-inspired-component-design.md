# Jina-inspired候选交互／ranking loss组件设计

**状态：独立组件及真实完整macro接线已实际执行；没有fresh学习或新分数。** 用户要求链接方法也尝试；学习研究对象固定为原ModelScope Qwen3-0.6B＋fresh LoRA r32/alpha64＋EOS二类head，不下载Jina权重代答、不把下面的组件迁移称完整Jina复现。

## 原方法边界

Jina v3确为同量级Qwen3系列，公开实现保持causal，多个doc同窗，末端query表征经projector/cosine与doc表征匹配。其读出／特殊token／辅助loss／131K训练与当前模型不同；本阶段只研究候选交互及排序目标，保留原EOShead，不夹带这些变化。

## 最小可归因三臂

- J0：候选隔离、原EOS两类head、case-balanced BCE。
- J1：同field候选按原提供顺序causal可见，head/loss同J0；不同field/request绝不相见。
- J2：attention同J0，增加请求内ranking loss，head不变。其与原BCE／Field-CE研究有数学重合部分必须说明，不能包装成全新独立收益。
- 可选J3为J1＋J2，首轮不为凑grid自动启动。

所有臂同stock/fresh初始化、数据、多集/顺序、mask backend、optimizer/scheduler/曝光。J0/J1都采用相同的“prefix只出现一次＋各candidate原suffix/EOS”的物理token布局，只有candidate可见边不同；不能拿重复prefix的旧pair baseline直接归因。

## 输入与位置接口

独立source：`/tmp/opencode/jev-jina-components-20261112-local/code/`，只读复用已冻结PF的TokenBatch，不修改运行中的代码。

- `build_candidate_layout(fields, device, pad, position_mode='sequential') -> CandidateLayout`。
- `fields`每项包含一个完整`prefix_ids`和所有原顺序`suffix_ids`（每项真实EOS末尾），另有case_id/field_id／稳定candidate keys；不得读labels排序或在mask中编码known/unknown。
- hidden `[T,H]`，field的cu隔离；EOS gather可以每field多个offset，不误用`cu[1:]-1`只得到最后候选。
- 两种matched-control位置：首轮固定`sequential`（field内0..physicalL-1），J0/J1完全相同；`candidate_local`可选保留每candidate从P开始，不把它与interaction同时改后归因。
- 前缀每field一次；不向业务hidden补tile/graph假行。

## 精确mask

每token有field owner和candidate id，prefix candidate_id=-1；再叠加真实界限与causal `q>=k`。

- interaction：同field＋causal。
- isolation：同field＋causal＋（key在prefix 或query/key属于同candidate）。prefix query自然不读后续candidate。

使用同一个Flex backend表达两臂，GQA Q16/KV8，不能让J0用FA3、J1用Flex后把换核算成交互效果。只构造BlockMask metadata，无T×T token mask；同batch复用28层，有界cache，生产首次编译记账。

## Ranking项与unknown

保留原known-mask BCE；ranking项仅使用已知硬标签：P为q==1，N为q==0，soft-q或unknown不伪造负例。

`Lrank = logsumexp(m_known_hard/tau) - logsumexp(m_positive/tau)`，预定lambda=1、tau=1。没有正例／没有有效对比对象／K1时该项为0并记账，不称零对比信号提升。

它是适配原二类margin的多正例目标，不是照抄Jina cosine InfoNCE。所有字段/case分母与原研究一致，记录clip前梯度norm，不能忽略新loss尺度变化。

**执行分母冻结：** 新研究每macro有C个完整case，每case有F_known个含已知监督的field；BCE按每field known候选平均，再F_known平均，再C平均。ranking亦按同F_known／C平均；known但K1／无正／无负／soft-only字段的rank为0仍计F_known，unknown-only保留raw但不提供监督分母。不按有rank信号的field/case重新归一化。这个决定不改旧128-pair机制实验的原目标。

## 数据、验收与后续

- Jina子研究预先冻结同一完整case子集：每pair沿8192训练上限，且每field物理prefix＋全部suffix≤16384；超过拒整个训练case，两臂相同，不截断候选。验证保留全部原case并静态报超共同模型40960覆盖，不能事后补短题。
- J0/J1位置／token／head／backend一致；真实单例／混field／换邻居的跨请求隔离必须通过；候选排列允许模型变化，但需反向／固定随机排列恢复ID后实际报告，而非宣称天然排列不变。
- J0的prefix与独立pair在sequential位置下并非自动同函数，因此必须与J1匹配训练，不能把旧模型分数当J0。
- 真实macro／梯度／一步更新门禁、clean数据ready、同曝光100M逻辑pair-token及10M/30M/100M观察点、完整public和必要paired seeds均在实际执行后填结果。当前组件源交付不解除这些门禁。

## 已执行的机制接线，不能当学习收益

- 使用实际clean曝光第一macro的全部16case、19known field、60candidate，逻辑pair token20947、prefix-once物理真实token10342；3臂相同布局/顺序/位置/head。该macro没有unknown字段，不能借此宣称unknown GPU路径已覆盖。
- **仅载历史诊断R1权重**验证新组件接线，未作为fresh初始化，不计入新clean模型曝光。J0/J1/J2各独立GPU：56次Flex forward提交（两pass）、28次实际输出梯度hook、393组finite梯度与实际AdamW更新；全28层realT，19个field-cu对应60个candidate EOS，无EOS漏接。
- 三臂的no-grad第一遍与可微第二遍logits逐位相同。主代理从保存的真实logits及原known targets独立重算F_known/C loss和系数，最大coef差7.45e-9，无额外模型forward/synthetic/unit。
- 首生产调用含编译10.19／9.82／10.64s；allocated约22.59GB。只是一轮真实接线，不是稳定性能比较。
- rank λ1将clip前norm由J0约3.50推至J2约13.29，尺度变化必须保留，不能把J2 loss1.34与J0 BCE0.52直接对比成质量好坏；J1 loss0.51也不是重新训练的提分。
- v1入口漏FLASH_ATTENTION_DETERMINISTIC，严格注册门禁在模型加载前拒绝；只补launcher该env进入v2，旧失败保留，不放宽gate、不重试official请求。
- 真实排列/换邻居及跨field内容隔离、fresh匹配训练、长field>40960全覆盖能力仍未完成；本回执不签全public或正式serve。

远端`/tmp/opencode/jev-jina-components-20261112-v2/`，实际聚合及直接回读`FINAL_PRIMARY_VERIFICATION.json`；源码和聚合归档但私有输入/tensor留远端。
