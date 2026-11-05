# Qwen native-LM 读出诊断：已完成

**原始 Qwen 8.01，HARD backbone 接回原 LM head 10.72，LS backbone 接回原 LM head 12.84。** 三组各 5,935 请求全部成功，coverage 100%；没有新增训练。

| 同一 backbone 历史 | 原 classifier pipeline | native-LM pipeline | native−classifier |
|---|---:|---:|---:|
| 原始 Qwen，无本项目训练 | 1.76（随机分类头） | 8.01 | +6.25 |
| HARD，raw 1M | 18.46 | 10.72 | −7.74 |
| LS0.1，raw 1M | 19.79 | 12.84 | −6.95 |

## 结论与边界

- 直接接回原 LM head 并切换 native 协议，没有找回优于现有 classifier 的隐藏高分；不是一个可直接替代现有 pipeline 的捷径。
- 在相同 native 协议内，HARD/LS 比原始 Qwen 分别提高 +2.71/+4.83。这个总分变化不等于所有能力无遗忘，也不是 KD 独有收益。
- 与 classifier 相比同时改变输入包装、读出位置和 head，不能把差值单独归因于 head 权重。三组未做 native 协议 SFT，不能据此排除 native 协议训练的价值，或断言 0.6B 的容量上限。
- 结合 Teacher native 68.37 / pair 63.95，当前更应区分 Student 的 target fitting 和泛化/训练覆盖，而不是继续盲扫 loss；尚无 Teacher 方案超过匹配的无 Teacher LS0.1 对照 19.79。

## 范围与运行依据

同一冻结 37 项、4,786 完整 cases、5,935 请求、全部候选；不是完整公共榜单或私测 Full。原始 Qwen 自带 tokenizer/chat template，thinking off，native label-token scoring；后两组从既有训练 checkpoint 导出 backbone，保留原 LM head，分类头不参与。实际 step `275711.69` 于 2026-11-05 02:40–02:50（北京时间）运行，`STEP_EXIT.json` 为 exit 0；三个 `evaluation/*/complete.json` 均为 5,935 ok、coverage 1.0。

- [最终报告与五方向结果](report/REPORT.md)
- [对比图 PNG](report/native-readout-comparison.png) · [SVG](report/native-readout-comparison.svg)
- [结果 CSV](report/native-readout-comparison.csv)
- 原生预测、评分、服务日志和实际代码均保存在本目录下；没有自动上传或发布。
