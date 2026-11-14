# Native FA3同后端的packed mask执行层

**状态：独立实现／真实机制对照准备，不是已训练或提分。** 最新commit三个mode在真实Hopper均执行了forward/backward，但FA3对SDPA连causal基线也未通过冻结合同。原失败原样保留。本分支不放宽合同，而是将“跨后端迁移”与“同一个实际FA3算法的dense→packed布局迁移”分开，后者也按原容差验收。

## 固定不变与接口

- 新源码namespace `/tmp/opencode/jev-packed-mask-20261112-local/code/`，不编辑用户commit、已冻结PF v1/v2/v3或全局scripts。
- 以已通过PF v3的可微原Qwen3/PEFT tokenwise运算为基础；head必须先计算全部实际token行再EOS gather，不回退v1的EOS-before-head形状。训练不使用forward-only融合。
- `MaskedBackbone(backbone,head,native_fa3,mode)`，`forward(TokenBatch,prefix_lengths, prefix=None,capture_prefix=False)`返回logits与真实P只读KV。所有整数cu／position／EOS／关系映射沿PF原合同。
- `causal`／`bidirectional`／`prefix_block`精确沿用户commit语义，分界在`Candidate:\n`之后；不擅自替换为Base精确state/problem切点或Jina列表布局。

## 原生可微mask运算

- causal：native FA3 varlen，causal=True。
- bidirectional：同序列native FA3 varlen，causal=False；不同cu序列始终隔离。
- prefix_block：同一层投影QKV后，真实prefix queries只读prefix KV、真实suffix queries读本row完整P+S KV，两次causal=False的原生varlen；两个query区间互补且完整，经可微index_copy恢复T顺序。不是两次独立softmax输出相加，suffix attention只有一个完整KV softmax。
- prefill prefix_block时只有一个完整prefix block，允许S为空并仅执行prefix self-attention；训练完整pair必须有真实suffix＋EOS。
- 实际prefix_len元数据在batch开始绑定一次，跨28层只读；不能按物理flatten offset使用RoPE，不能缓存可训练prefix跨optimizer step。

## 推理依赖与复用

- causal保留原token-exact common-prefix复用。
- prefix_block只允许复用**完整且完全相同的双向prefix block**；禁止把不同field的最长公共token前缀当已完成双向prefix。原engine的跨field common-prefix拆分必须在此mode关闭，或要求整个prefix完全相同。
- prefix_block suffix fork读取本owner prefix和本branch全部suffix，非因果；不能读其它branch，也不能更新prefix。prefix KV仅真实P行，suffix QKV与MLP仅真实T行。
- bidirectional必须完整P+S重算，不使用immutable prefix复用。不能为了速度偷偷把它变成prefix_block。
- Triton fork如从已核实的causal版扩展，只有suffix三角mask这一语义参数不同；仍为inference-only，不拿它做训练反传。

## 验收与后续

1. 先实现可微核心与真实QKV/metadata接口；只读review后，用同一真实原macro逐mode对exact commit HF+nativeFA3 dense与新packed比较logits/loss/393grad/actualAdamW更新，原合同不变。SDPA失败不被覆盖成通过。
2. 真实full-row对prefix复用、单独/混批/换邻居/换序、prefix A→B→A及全层realT继续执行；不存在的测试证据不填通过。
3. 核心准入后，clean-origin训练使用同FA3 backend、同初态/数据/loss/曝光；保存mode和源绑定，拒绝旧完成臂复用。训练与正式serve必须用同mask。
4. 100M训练及完整public必须另有真实数据ready和执行回执；这个执行层不解除clean数据／许可／确认集门禁，不产生新的质量结论。
