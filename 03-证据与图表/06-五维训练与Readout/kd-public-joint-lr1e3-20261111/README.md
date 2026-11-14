# KD public → joint LR1e-3 → public：聚合证据

本目录只放本轮真实执行的聚合状态与完成证据；不放模型权重、原始题目／预测、凭据或编码源码包。

- `START_VERIFICATION.json`：部署及正式CPU准入后的只读生产状态复核。其`experiment_complete=false`时只证明阶段进度，不能当最终分数。
- `FINAL_VERIFICATION.json`：2026-11-11T19:36:35UTC最终直接复核，逐条对齐两份merged、八个shard与原Suite，全部来源／覆盖／训练终点检查通过。
- `results/kd/`：第一次完整public的官方原生scores、index、benchmark-summary及两个完成／计分来源receipt，Public Index24.38。
- `results/joint_lr1e3/`：第二次完整public的同类原生文件，Public Index0.81，严重退化但执行／计分均完整。
- `results/TRAINING_COMPLETE.json`、`results/TRAINING_PROGRESS_FINAL.json`：新joint实际1667242pair／458731913token／13026步／3647关系的终点与来源。
- `results/COMPLETE.json`：整链完成receipt，2026-11-11T19:32:40UTC完成。含上述文件共14份最终JSON，另保留启动时点JSON。

本次最终复核只读取既有结果／源码／元数据／日志，没有新增模型执行、tests／smoke、scorer重跑或额外模型／语料hash。四卡并行墙钟不代表官方单进程时延验收；仅为public结果，不是private Full。

[结果报告](../../../01-研究结论/06-五维训练与Readout/kd-public-joint-lr1e3-20261111/README.md) · [执行计划](../../../02-研究路径/06-五维训练与Readout/2026-11-11-kd-public-joint-lr1e3.md)
