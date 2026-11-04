# 蒸馏改进第二轮：校准和样本级Teacher信息

**2026-11-04 15:49已实际启动step275711.66，四卡并行；15:51各arm均已完成第1/7,813步。当前没有本轮最终质量分数。**

第一轮普通LS0.1的19.79超过最好KD19.17，因此本轮同时以HARD18.46、HYBRID19.10、T2 19.17及LS19.79为参考，不能只凭超过HARD就宣称Teacher独有收益。

## 四组

| 条件 | 内容 | 目的 |
|---|---|---|
| KD_CAL50 |每域拟合Teacher概率的正slope/intercept，0.5CE＋0.5KL|去掉分域概率偏差后是否改善|
| KD_CALSOFT |同一校准概率纯KL|校准是否能救回原来的纯SOFT|
| KD_ADAPT_LS |按Teacher置信度rank分配epsilon0.05–0.15，每(domain,exact target)组均值0.1|从Teacher提取具体样本的难度信息|
| KD_SHUFFLE_LS |在相同组内打乱同一epsilon分布|保留平滑强度，去掉Teacher与样本的对应关系|

ADAPT与SHUFFLE成对比较，能比单独对比普通LS更直接地检验样本级Teacher信息。它们是Teacher-guided smoothing，不冒充vanilla KL。CALSOFT的Student loss没有CE，但校准阶段用原训练target，不声称完全label-free。

全部仍从原始Qwen＋未训练LoRA/head开始，同1M样本/order、同超参/训练量和冻结评测；不丢样本，不重标数据，不使用评测输入/标签校准，原Teacher和旧规模系列不恢复。

## 已观测的preprocessing

8个校准scalars只用1.23秒拟合。Arts的Teacher均值从75.05%变成47.11%，与原训练target均值对齐；对应训练target CE从0.8954变成0.6537。**这是训练标签上的拟合诊断，不是本轮评测成绩。**

Aligned和shuffled两个条件在每个exact-target组内保留相同epsilon分布/均值0.1；不会把不同原标签强行二值化。该轮使用已知固定子集，属于探索性实验，不是独立held-out确认。

完成/失败通知已挂上，最终report、所有分数和PNG/SVG图会自动保存到本目录。
