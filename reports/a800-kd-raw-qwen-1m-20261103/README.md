# 从原始 Qwen 起点重新做 KD 对照

## 状态

**2026-11-03 15:01 CST更新：本轮已实际启动，不再等待。** SAMerging结束且step退出0后，raw对照自动启动为275711.64。HARD/SOFT/HYBRID三组训练与RAW BASELINE评测正在四卡并行；三组均已完成第1/7,813个optimizer step。

实际初始化receipt确认：196个LoRA B矩阵全部为0，加载同一份[2,1024]未训练head，未使用训练过的MIXED_ALL adapter/head。最终质量结果仍待训练和评测完成。

2026-11-03 11:46 CST：实验代码、数据链接和自动调度已准备。**当前等待优先级更高的SAMerging成功完成，尚未启动本轮训练，也没有本轮效果分数。** SAMerging结束且step正常退出后，会自动在同一个四A800 allocation275711内运行；旧的warm-start规模系列继续保持暂停。

## 唯一主动改变：初始化

- 旧实验：已四域训练好的 `MIXED_ALL/final`，包括其训练过的LoRA与分类head。
- 新实验：原始预训练 `Qwen3-0.6B` backbone＋保存的未训练r32/alpha64 LoRA初始化及同一份seed42随机二分类head。
- **不带入旧trained LoRA或trained classification head。** 实际作业启动后会记录原始初始化来源、LoRA B全零、共用head存在等必要初始化事实。
- 原始Qwen没有此二分类head，因此RAW BASELINE只是未训练任务head的起点参考，不能当作原始Qwen语言能力的zero-shot分数。

## 四组

| 组别 | 训练 |
|---|---|
| RAW BASELINE | 原始Qwen＋共用未训练head，直接评测，不复用旧BASELINE预测 |
| RAW HARD | 从同一原始起点用原始标签训练1M pairs、一遍 |
| RAW SOFT | 从同一原始起点用KnowLine完整概率做KL蒸馏，1M pairs、一遍 |
| RAW HYBRID | 从同一原始起点训练0.5CE＋0.5KL，1M pairs、一遍 |

严格复用上一轮1M的样本ID、epoch顺序、Teacher概率快照、训练超参和冻结评测完整case。Teacher仍是PelaAI/KnowLine-31B-Gen1，T1；BF16/SDPA、LoRA r32/alpha64、batch128、lr5e-5、AdamW/cosine、warmup3%、wd0.01均不变。

## 结果解读

主要看新起点的SOFT−HARD和HYBRID−HARD，再与旧1M起点的−1.13和−0.11比较；报告同时给出初始化×监督方式的交互差值。不能仅因RAW SOFT超过随机head BASELINE就宣称蒸馏优于原始标签训练。

最终报告将位于 `report/REPORT.md`；`LIVE_STATUS.json`由后台watcher刷新；原始分数、逐题预测、训练日志和checkpoint完成记录会随实际完成后拉回。
