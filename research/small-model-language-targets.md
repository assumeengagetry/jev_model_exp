# Language 同规模目标核验（2026-10-27）

来源：[Decision Index 原始 index.json](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json)。本次重新下载到 `research/2026-10-27/raw/leaderboard-index-small-audit.json`；仍是 **2026-10-26 15:30:58 UTC** 快照，共 229 个模型条目。

表中指标为 **公开 Language category.skill × 100**，不是普通 accuracy、private combined、Full 或总榜 Public index。参数为榜单 `meta.served_params`，不能仅按名称判断模型实际部署大小。

| 用途 | 模型 | Served params | Language skill | 基座 |
|---|---|---:|---:|---|
| Qwen3-0.6B 近似同基座参考 | [Decision-2.0-Kai-0.6B](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B) | 0.600B | 11.56 | Qwen3-0.6B-Base |
| 约 0.6B 部署规模参考 | [Sieve 0.8B](https://huggingface.co/sthanika-ai/Sieve-0.8B) | 0.590B | 26.59 | Qwen3.5-0.8B-Base |
| <1B decoder 较强目标 | [StartLux-Decision v1.1 0.8B](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant) | 0.873B | **55.53** | Qwen3.5-0.8B |
| 小参数量跨架构参考 | [GLiClass v4 edge](https://huggingface.co/knowledgator/gliclass-v4-edge) | 0.440B | **54.62** | DeBERTa-v3-large |
| 约 1B 跨架构参考 | [Crivo-1B](https://huggingface.co/tardellirs/crivo-1b) | 1.031B | **56.26** | ettin-reranker-1b-v1 |
| 2B 级参考 | [StartLux-Decision v1.1 2B](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant) | 2.274B | **62.39** | Qwen3.5-2B |
| 4B 名义规模参考 | [StartLux-Decision v1.1 4B](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant) | 4.660B | **71.19** | Qwen3.5-4B |
| 较严格约 4.2B 部署规模参考 | [Jeffan-4B-2010](https://huggingface.co/tattrongvu/Jeffan-4B-2010) | 4.206B | 69.78 | Qwen3.5-4B |
| 9B 级参考 | [baselabs elephant-1](https://huggingface.co/baselabs/elephant-1) | 9.410B | **73.76** | Qwen3.5-9B |
| 27B 级及本快照 Language 第一 | [baselabs mammoth-1](https://huggingface.co/baselabs/mammoth-1) | 27.357B | **76.11** | Qwen3.5-27B |
| 31B 级参考 | [Jeffan-31b-1710](https://huggingface.co/tattrongvu/Jeffan-31b-1710) | 31.273B | 76.00 | Gemma-4-31B-it |

## 对路线的影响

- 用户截图明确当前已经下载的是 **Qwen/Qwen3-0.6B**，不是 Qwen3-0.5B，也不是尚未准备的 4B。不要沿用上一版 checklist 的默认 4B 覆盖现有模型。
- “同参数规模最好”要分严格同基座、相近部署参数量和跨架构参考。Qwen3-0.6B-Base 的 11.56 只是较弱的同家族参考，追平它不等于小模型数据已经优化充分。
- 对当前 0.6B，可以先追平同家族参考，再向约 0.6B 部署模型的 26.59 和 <1B 模型的 55.53 看齐；后者使用更新基座、模型稍大，不能把差距全部归因为数据或实现。
- 此前把 Jeffan 的 69.78 作为“4B 级最好”过于粗糙：按名义 4B 或 served≤5B，StartLux 4B 的 Language 为 71.19。若实验严格限制 served≤4.5B，则 Jeffan 69.78 才是该桶最高。
- 不宜从总榜弱就断言 encoder 不适合这个维度：GLiClass 0.44B 的 Language 54.62 与 StartLux 0.873B 的 55.53 接近。保留 decoder 路线是用户当前实验选择，不能说已被榜单证明唯一可行。
- 若研究纯规模效应，优先同模型家族做 0.6→1.7→4→8/14→32B；迁移到 Qwen3.5/Qwen3.8 的 0.8/2/4/9/27B 是另一个基座变量，需在实验表中标明，不能称严格 scaling law 拟合。

## StartLux 小模型的直接证据

[0.8B 模型卡](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant)说明：

- 每题单独含完整 state，在最后 prompt 位置读取选项字母的 logits，然后按题型温度归一化。
- “one in four” 指混合架构中每四层有一层 full attention；该 full-attention 层的 prompt tokens 也能看后文，linear-attention 层保持 causal。不能误写成“每四个 token 只开放一个 token”。
- Stage 1 约 2.7M decision examples full FT；Stage 2 约 1.7M 加 bidirectional 的 adapter 训练后合并；三个种子的权重平均。
- 所以模型卡证明组合配方有效，并不能单独证明用户将 Qwen3-0.6B 的 causal mask 改成 noncausal 后必然涨分。其最后位置本来就能看完整前文；新增双向信息主要改变中间 token 表征，训练适配和 pooling 方式会影响收益。

模型卡的 53.75 是其自行运行的**总 Public index**，不是本表 Language 55.53；两者不能互换。
