# Clean 八臂官方评测：聚合证据

原始运行目录：`/home/dev/assumeengage/runs/jev-public-eight-fourway-20261113-v1/`。

## 文件

- [官方八模型汇总](SCORES.json)：保留服务器原始字节，含全部八行分数、状态、耗时、四片运行计数及证明指针。
- [独立全量核验](FINAL_OFFICIAL_VERIFIED.json)：保留原始字节；记录全部原序输入和分片字节对账、32 份 canonical393 after-proof、实际退出及 lease 核对。
- `scores/<arm>.json`：8 份对应权重的官方 SDK `scores.json`，不包含逐请求预测。
- [逐 benchmark 表](official-benchmarks.csv)：8×43＝344 行，保留原生 metric、score、Index raw/skill/chance、coverage/status 和是否进入 Index。
- [五域表](official-areas.csv)：8×5＝40 行，保留 raw/skill 及 skill×100。
- **[完整学习报告](../../../01-研究结论/06-五维训练与Readout/clean-eight-official-20261113/REPORT.md)**：解释材料、实际改动、数学、过程、结果及局限。

## 统一口径与边界

- 每权重完整执行 140620、官方排除 442、实际计分 140178，八臂全部 `ok`。
- 最终队列为四卡同一权重分片，各 35155 请求；恢复全量原序后调用未改 SDK 计分。
- headline 是 public0.3 的 `balanced_skill`，不是总体准确率或 `breadth_skill`；没有 private Full 结果。
- `C_D_CE`、`C_D_BCE` 为 dense 训练、packed 消费的科学测量，原数值准入 34/375 失败不被全量完成抹去。
- 原始 checkpoint、模型、raw case、token、激活和 JSONL 逐请求预测不纳入此附件组。JSON 内部绝对路径表达当时服务器来源，不因仓库归档而机械改写；以后迁移必须另核对身份。
