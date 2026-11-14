# 2026-11-12真实GPU实验源码快照

这些文件由远端实际冻结namespace逐字节回读，未改写用户的`2e334fb`源码；没有模型权重、私有原始语料或凭据。

- `padding-v1/`：首次真实机制失败版本，保留，不应部署成通过版本。
- `padding-v2/`：通过真实128pair macro的数值实现；all-token head后gather EOS，禁用未通过的额外数学融合。
- `padding-v3/`：与v2四份数值核心完全相同；只调整原runner warmup对margin sidecar的记账，以及已验证mechanism的独立引用／源绑定，供完整public。
- `commit-modes/`：exact用户commit三个mode的真实macro harness；原始commit模块在远端`commit-code`下，来自本目录上层已跟踪的原源码。
- `flex/`：真实packed causal Flex后端及完整macro实验；本轮数值验收未通过，保留编译和原始结果。
- `queue/`：四卡调度，独立机制失败不会伪装成功，也不取消其它已准入方法；完整public四卡集中运行。

**这些是实际版本证据，不是可混装的快捷安装脚本。** 原始绝对路径和源绑定保留；不要将其覆盖到`/home/dev/assumeengage/scripts`或旧实验root。全局训练脚本的路径优先级与completed-arm mode复用存在已记录风险，正式mask训练须独立bootstrap／准入。

计划与状态：[实验总队列](../../../../2026-11-12-attention-experiment-queue.md)。原始JSON：[证据目录](../../../../../../03-证据与图表/05-Attention与效率/attention-queue-20261112/results/)。

当前只PF真实macro通过；最新commit三mode和Flex均已实际执行但未过冻结数值合同。没有以这些源码快照声称新public分数或clean排名。
