# 少见参数量 attention 补充（2026-11-09，只读静态核查）

核查当前 index.json/qwen-inventory.json、作者模型卡、发布源码；没有运行模型、tokenizer或测试，没有下载权重。

| 榜单条目 | 量级与实际底座 | 判断 | 直接证据 |
|---|---|---|---|
| Qwen-2.5-1B-RLCD (harsha) | 1.5B；卡实际为 Qwen/Qwen2.5-1.5B-Instruct，榜单省略了 Instruct | 原生causal路径，非双向改造 | core/engine_torch.py L106 直接原生prefill；L129–132 prefix+suffix构造2D padding mask，直接原生模型续算；engine_mlx同样共享prefix cache。 |
| CLM-v0.1-8B | 8B；作者卡/脚本为 Qwen/Qwen3-8B，榜单写 Qwen3-8B-Base | 作者公开路径为原生Qwen3 causal编码+末位pooling；没有双向mask改造 | serve_qwen3_8b.sh L16–25 为原生 vllm serve Qwen/Qwen3-8B --runner pooling；无mask override；train/embed_utils.py L63–65亦同。模型卡中的 bidirectional InfoNCE 指训练损失的双向对比，不是 attention mask。 |
| Cloudflare clef-omni (30B-A3B) | 30B总参/3B激活；Qwen3-Omni-30B-A3B-Instruct | 语言骨干保持原生causal，额外joint schema head使用无因果mask的全局attention | joint_schema_model.py L636–643 标准Qwen3Omni加载；L598–605骨干只传2D padding mask，未替换attention；L380–387 head的MultiheadAttention不传attn_mask；L533–534 TransformerDecoderLayer(fields,memory)也不传causal mask。视觉/音频编码器是独立组件，不将它们与语言骨干方向混计。 |

作者链接：

- [Harsha engine_torch.py](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/engine_torch.py#L93-L132)
- [Harsha模型卡](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
- [CLM serve_qwen3_8b.sh](https://github.com/Contrastive-LM/CLM/blob/main/serve_qwen3_8b.sh#L16-L25)
- [CLM embed_utils.py](https://github.com/Contrastive-LM/CLM/blob/main/train/embed_utils.py#L53-L66)
- [CLM模型卡](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
- [Clef-Omni joint_schema_model.py](https://huggingface.co/Cloudflare/clef-omni/blob/main/joint_schema_model.py#L591-L613)
- [Clef-Omni head routing](https://huggingface.co/Cloudflare/clef-omni/blob/main/joint_schema_model.py#L380-L389)

当前库存没有 Qwen 14B、32B、72B 底座条目；35B-A3B为MoE总参，不是3B dense。

以上为作者当前公开路径的静态核查。CLM原生causal判断依赖作者直接调用标准Qwen3 vLLM pooling的实现路径，不将bidirectional InfoNCE解释成mask。

## CLM pooling默认mask追查

进一步追到vLLM模型实现，而非仅凭`--runner pooling`判断：

- [vLLM v0.10.2 qwen3.py L180–187](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/qwen3.py#L180-L187)：源码明确说明Qwen3默认causal；`getattr(config, "is_causal", True)`为真时使用`AttentionType.DECODER`，只有显式`is_causal=False`才转成`ENCODER_ONLY`双向。
- [vLLM v0.11.0同一分支](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/models/qwen3.py#L179-L186)一致；当前main L193–200亦一致。
- [原Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)没有`is_causal=False`，使用上述默认True。
- [vLLM pooling适配器](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/adapters.py)继承原模型、保留原模型构造；没有设置`is_causal=False`。
- CLM的[pyproject.toml](https://github.com/Contrastive-LM/CLM/blob/main/pyproject.toml)仅要求`vllm>=0.6`，没有锁精确版本。因此可确定其作者公开标准路径保持causal；不声称还原了榜单运行的精确vLLM安装版本。
