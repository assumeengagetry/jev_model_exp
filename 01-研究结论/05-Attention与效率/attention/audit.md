# Qwen3-0.6B 双向注意力：计划核查

核查对象：截图中的 Transformers 5.5.0、Qwen/Qwen3-0.6B；日期 2026-10-27。

## 结论

双向注意力值得作为早期独立实验，但“改完肯定好”并不成立。Perplexity v1.1 同时扩大数据并解除 causal mask；其卡片将增益归于两者，不能据此把全部增益归于 attention。StartLux 同时用了两阶段数据、双向训练和多 seed 平均。应固定模型、head、训练数据、初始化、采样、token 预算和种子，比较 causal 与 bidirectional；不能只在推理时切换已按 causal 训练的权重就宣布有无增益。

来源：[Perplexity](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)、[StartLux](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant)。

## 当前 0.6B 与榜上 27B hybrid 的差异

已读取 Qwen3-0.6B 的 config：28 层、hidden_size=1024、use_sliding_window=false，普通 softmax full attention；没有 Qwen3.5/3.8 hybrid 的 Gated DeltaNet 层。因此这一模型的实验是将这 28 层 prompt self-attention 改为双向，而非照搬“仅每四层中的 full-attention 层双向，linear-attention 层不动”。后者适用于对应 hybrid config，扩大模型时须重新核对架构。

Config：[Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)；[d3 的真实 hybrid 实现](https://huggingface.co/vllm-sr/d3-turbo/blob/main/d3_runtime.py)。

## Transformers 5.5.0 的具体陷阱

1. Qwen3Attention.__init__ 把模块级 self.is_causal 硬编码为 True。
2. masking_utils.create_causal_mask 会在 config.is_causal=False 时转到 create_bidirectional_mask。
3. SDPA 的无 padding 场景中，双向 mask 可能优化成 None；SDPA backend 随后仍会读取模块的 is_causal，并按 True 走 causal kernel。
4. 因而仅写 config.is_causal=False 不能在该组合上盲目信任。需核对 mask 创建与每层 backend 实际收到的 is_causal；必要时同时设置 config 与 Qwen3Attention 模块的 is_causal=False，或使用显式双向 mask/自定义前向，并验证。
5. Unsloth 可能替换模型 forward 和 kernel，该行为还需针对安装版本实测。先以 Transformers+SDPA 获得正确实现作为参照，再验证 Unsloth 加速后输出/梯度一致。

对应固定版本源码：

- [Qwen3 model](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/qwen3/modeling_qwen3.py)
- [masking_utils](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/masking_utils.py)
- [SDPA backend](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/integrations/sdpa_attention.py)
- [FlashAttention backend](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/integrations/flash_attention.py)

最新 Transformers 文档已有更便捷的 config/forward is_causal API，不能未核对版本就假定截图的 5.5.0 与最新文档行为相同：[官方 attention 文档](https://huggingface.co/docs/transformers/attention_interface)。

## 最低验证

- 取同一序列，只改变后面的有效 token：causal 的早期位置 hidden state 不应变化，bidirectional 的早期位置应能随之变化。检查早期位置，不能只检查最后 token，因为最后 token 在 causal 下本来就能看全部前文。
- 对同一样本改变 padding 长度与 batch 组成，eval logits 应保持一致至容差；既测无 padding，又测有 padding，防止仅有 padding 的路径是真双向。
- 首轮用一题一序列、普通 padding batch，不做 packing。以后 packing 必须用 block-diagonal mask，严格隔离样本；改变另一样本不应影响本样本输出。
- gold 标签只在 loss 侧，不进入双向可见输入；训练、验证、评测、保存重载和部署使用同一 attention mode。
- use_cache=False，整段输入的 prefill 分类即可；bidirectional 下不要假定跨候选的 prefix KV cache 可复用，因为前缀表征依赖后文。

## 实验顺序建议

第一轮 causal 线性分类基线与评分器跑通后，立刻在同一 100k–200k 分层样本、相同 token/优化设置下跑 causal/bidirectional 两个实验。小模型可独立单卡运行多个对照以并行获取结果，不必为填满 GPU 强制每个 0.6B 实验都四卡 DDP。正式扩大数据后，只把在独立 dev、多个 seed 和完整 Language 分项均可复现的赢家带入更大模型。

两种 attention 不一定吞吐相同，因此同时报告固定训练 token 下的质量对照和实测 GPU-hours/tokens-per-second，避免把更多计算混入 attention 的因果归因。
