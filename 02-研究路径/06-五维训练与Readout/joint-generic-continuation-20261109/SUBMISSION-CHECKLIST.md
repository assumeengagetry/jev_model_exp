# 正式提交前检查项（仅准备，不执行发布）

- [ ] 联合续训本段`complete.json`、`TRAINING_COMPLETE.json`均为自己的真实完成；全六stream、token、relation、optimizer步数与冻结清单一致。
- [ ] 新final、tokenizer、实际基座和Engine不可变快照来源完整；仅以此运行完整官方public0.3，不混入旧final／旧wrapper结果。
- [ ] 官方`results.jsonl.gz`／`benchmark-summary.json`／`index.json`／`scores.json`／`environment.json`／`status.json`原样保留，`scores.json`为complete且完成140,178请求；不得手工改成绩或补完成标记。
- [ ] 固定官方kit commit、Engine代码、硬件、精确推理设置和依赖随包提供。当前设定：独立候选二分类margin→候选softmax，全部criteria、不截断／不筛候选，SDPA推理、BF16基座、max_microbatch_pairs32／max_microbatch_tokens32768；最终以实际冻结绑定和environment为准。
- [ ] 实际基座LICENSE和NOTICE按原件保留。已只读观察当前基座目录存在Apache2.0 LICENSE；这**不代表**全部训练语料／Teacher／派生权重发布条件已审清。
- [ ] 训练语料、token流、私有suite、原始行来源和鉴权资料不进公开提交包；官方重建suite保持私有，遵守各来源条款。
- [ ] 用户确认模型名、权重和结果可见性、维护者访问、具体Hub仓库及参榜PR范围后，才执行上传／发布／PR。

固定kit README当前规则：0.4主榜仍接收完整public0.3；Full由维护者私测计算（同技能60%＋新域40%），public不计Full。维护者在单RTX PRO6000、单进程逐请求测median／mean／p80，均须≤1000ms；本机H800的完整请求成本仅作参考，不能自报官方通过。
