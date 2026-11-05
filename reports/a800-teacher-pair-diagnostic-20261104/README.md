# Teacher 原生协议与 pair 改写：已完成

**同一 Teacher：原生协议 Index 68.37，generic candidate-pair 协议 63.95，差值 −4.42；coverage 均为 100%。本实验没有新 Student 训练。**

2026-11-04 22:09 提交，实际 step `275711.68`，四张现有 A800 运行两个 TP2 副本；2026-11-05 02:17 完成并退出 0。5,935 个原始请求全部 ok，163,148 次 unique pair 调用覆盖全部 163,671 个 field-option rows。

- [最终报告](report/REPORT.md)
- [对比图 PNG](report/teacher-protocol-comparison.png) · [SVG](report/teacher-protocol-comparison.svg)

## 保持一致与改变的部分

- 同一PelaAI/KnowLine31B权重、BF16/triton/fp32LMhead、frontendT1、read-twice≤3000及官方per-type温度文件。
- 同一冻结37项/4,786完整cases/5,935请求，全部163,671个field-option候选；523个完全相同输入复用后，需要163,148次独立pairAPI调用。
- 使用同一render_pair函数构造输入，沿用原标注的genericnoul正确匹配问题，不向Teacher发送expected/gold/scoring标签。
- 保留完整eval输入，不用训练4096token截断；所以长输入上不是逐字节训练label管线重放。语义、序列化、questiontype/readout及聚合属于本次整体protocol变化，不单独归因某一项。
- 概率按原annotation两列float32[Pnotmatch,Pmatch]保存，margin用log(Pmatch)−log(Pnotmatch)，不从单个round-to1的Pmatch重建负类概率。候选聚合/评分与Student一致。
- 私有cache只在本实验目录，不写原canonical训练概率矩阵；这些评测回答不会用于训练。无新环境、smoke/unit测试或hash。

## 判读

Pair Teacher 仍比 raw Student 最佳 LS0.1 的 19.79 高 **44.16**，因此不支持“pair 改写毁掉 Teacher 大部分能力”作为主要解释。Student 容量、优化、训练覆盖和训练长度差异仍未定位，不能从这个实验直接断言容量不足，也不能据此保证 KD 获益。

最终 report、原生评分记录、预测及 PNG/SVG 图已保存到本目录。后续 [Qwen native-LM 读出诊断](../a800-qwen-native-readout-20261105/README.md) 也已完成；旧规模系列和 Teacher 训练标签生产保持暂停。
