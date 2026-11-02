# A800 蒸馏快速实验：本地报告入口

## 先看这些

- [实验报告与结论](REPORT.md)
- [四模型五维雷达图](four-model-five-area-radar.png)
- [逐 benchmark 差值热力图](benchmark-deltas-heatmap.png)
- [四模型五维结果 CSV](four-model-five-area.csv)
- [逐 benchmark 原生指标 CSV](native-benchmark-metrics.csv)
- [SOFT−HARD、HYBRID−HARD 等差值](differences.json)
- [训练时间与吞吐](timing-throughput.json)

## 目录内容

- `evaluation/{BASELINE,KD_HARD,KD_SOFT,KD_HYBRID}/`：逐题预测 `results.jsonl`、原生 benchmark 指标、五维分数、coverage 和完成记录。未复制内容重复的 `part-0.jsonl`。
- `training/`：三组训练的完成记录及末次进度。
- `selection/`：训练抽样配置、10 万条样本索引、冻结评测子集配置。
- `run-records/`：共同训练顺序、最终完成记录、训练退出码及评分修复记录。
- `logs/`：训练、评测及调度日志。
- `huggingface/`：模型、数据和报告上传回执及仓库地址，不含登录凭据。
- `download-record.json`：本次取回的文件清单与大小，不含文件 hash。

这是 **100k pilot**：三组同起点、同样本、同训练量；四模型共用冻结 37 项中的 4,786 个完整 case、5,935 个请求和 163,671 个完整候选 pair。不是完整公共榜单成绩，也不是私测 Full。

日志保留了曾经的 PhishNChips 不计分字段评分报错。该问题已修复，且只从已有预测重算分数，没有重复推理；最终状态以 `REPORT.md`、`run-records/pilot-complete.json` 和各模型的 `evaluation/*/complete.json` 为准。

## Hugging Face 备份

- [四组 checkpoint、报告与图](https://huggingface.co/engageassume/qwen3-06b-knowline-kd-pilot-100k-20261102)
- [全部已完成 Teacher 标注及 100k 子集索引](https://huggingface.co/datasets/engageassume/knowline-teacher-labeled-pairs-20261102)

以上仓库均为 private。本地本目录没有下载 checkpoint、大训练数据或原始冻结评测题。
