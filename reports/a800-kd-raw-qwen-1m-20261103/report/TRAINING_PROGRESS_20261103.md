# 原始Qwen蒸馏：进行中进度快照

**观测时间：2026-11-03T17:55:37+08:00。本文件不是最终效果报告。**

原始Qwen3-0.6B＋共用未训练LoRA/head；严格复用原1M样本、顺序、Teacher概率和冻结评测。

| 组别 | 当前阶段 | steps／请求 | pairs |
|---|---|---:|---:|
| KD_HARD | 训练中 | 6,600/7,813 | 844,800 |
| KD_SOFT | 训练中 | 6,600/7,813 | 844,800 |
| KD_HYBRID | 训练中 | 6,600/7,813 | 844,800 |
| RAW BASELINE | 评测已完成，Index 1.76 | 5,935/5,935 | — |

RAW BASELINE接的是未训练随机分类头，其分数不能当作原始Qwen的zero-shot语言能力。

## 训练曲线

![训练loss快照](figures/raw-qwen-training-loss-progress.svg)

三条曲线对应不同loss定义，不能据此直接比较模型质量。最终结论等待训练后评测，以SOFT−HARD/HYBRID−HARD及与warm-start的交互差值为主。

完整日志提取与进度JSON：[training-progress-snapshot.json](training-progress-snapshot.json)。
最终报告完成后由原watcher保存至同目录的REPORT.md；本快照保留其明确的观测时间。
