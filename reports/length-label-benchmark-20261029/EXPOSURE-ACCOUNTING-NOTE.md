# 归档计划与真实训练曝光：后续worker的记录要求

本说明只补充CPU准备，不修改batch/LR pilot、长度诊断或历史训练产物。

**后续状态：首轮比例worker已实际接入该接口并完成训练/评测，14:58 UTC完成逐更新task×标签行数/token、源pair计数与D0 multiset核对。** 结果见`research/lu-pair-ratio-screen-20261029T133342Z/RATIO-REVIEW.json`。本说明末尾“尚未接入”仅保留CPU准备时刻身份，不适用于已完成的新比例轮；旧17M/20M归档数仍不是worker真实回放。

## 已有数值的分母

| 来源 | 正pair出现次数占比 | 正token占比 | 证据范围 |
|---|---:|---:|---|
| 唯一小训练池117353对 | 41.14% | 46.07% | 静态池，不含重复抽样 |
| BASE-17M归档计划 | 45.39% | 47.16% | 完整schedule复原，非逐step worker审计 |
| BASE-20M归档计划 | 45.54% | 47.21% | 完整schedule复原，非逐step worker审计 |

17M与20M是两份不同计划，不能合并为一次曝光比例。完成记录的总行数/token与计划相等，也只构成端点计数核对；
在没有逐step记录时，不把它提升为逐行顺序、标签和实际attention长度的完整回放证明。
不同配方的得分差不能直接解释为这些比例差的因果效果。

## 后续比例worker必须记录什么

- 元数据包括pool、task、pair标签、原始row index、原题/源组ID、重复draw ID；不把这些字段传入模型输入。
- 每个成功完成forward/backward的microbatch，从**实际attention_mask逐行求和**记录长度，不用计划length替代。
- optimizer.step成功后再提交本次累积的task×label行数与非padding token；失败或未提交的累积单列，不计入已提交训练曝光。
- 每step写JSONL，结束时报告累计实际正/负行数、token、每task预算偏差、unique原题/源组、重复率，以及计划/实际差异。
- 原生类别按原题draw单独计，不按被重复的正pair行数倒推。跨更新未完整提交的候选组和尾部应单列；原生core平衡策略不变不等于实际全流恰好50:50。
- 所有候选CE权重相同；若未来另测CE权重，保留自然采样表逐数组一致，不能把加权损失说成数据行数比例变化。

CPU模块`research/lu-pair-ratio-prep-20261029/exposure_receipts.py`提供提交/未提交分账的接口和合成测试，
**尚未接入或启动任何真实worker**。自然dev/test、完整采样表与超时处理仍需在本轮启动前冻结。
