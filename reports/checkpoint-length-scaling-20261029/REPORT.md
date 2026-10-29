# Gradient checkpointing：具体输入长度、显存与速度

## 原来的medium和long_real_pairs到底多长

| 原profile | 实际长度范围 | 均长 | 中位长度 | 两次microbatch的实际shape | 有效tokens / padded tokens | padding | 原On峰值 | 原Off峰值 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| medium | 192–205 | 197.94 | 197.5 | 16×208 + 16×208 | 6,334 / 6,656 | 4.84% | 1.77 GiB | 8.22 GiB |
| long_real_pairs | 2527–2807 | 2608.31 | 2579.0 | 16×2576 + 16×2808 | 83,466 / 86,144 | 3.11% | 8.39 GiB | OOM |

medium的筛选条件写的是192–384，但实际取的是命中条件的前32对，最终仅192–205；long_real_pairs取目标池最长32对，全部来自RAGTruth Summary。medium全部来自ANLI r1。两组都采用物理microbatch16、累计两次形成32对；不是32条同时驻留。

## 固定长度阶梯：checkpoint On / Off

设备：NVIDIA H800 NVL，CUDA可见容量93.09 GiB。模型：Qwen3-0.6B分类模型，LoRA r16/alpha32、原R0分类头、BF16、causal SDPA。物理microbatch固定16，累计2次；每档32条真实长文本裁到相同精确长度，padding为0。每个组合在两个独立进程中重复，各预热1次、计时3次前向+反向，权重不更新。

| 每条sequence tokens | 每个microbatch tokens | On峰值allocated GiB | Off峰值allocated GiB | On有效tokens/s | Off有效tokens/s | On节省显存 |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 4,096 | 1.85 | 9.09 | 17,024 | 31,236 | 79.6% |
| 512 | 8,192 | 2.46 | 16.91 | 29,400 | 39,375 | 85.5% |
| 1,024 | 16,384 | 3.66 | 32.56 | 29,798 | 40,922 | 88.8% |
| 2,048 | 32,768 | 6.07 | 63.85 | 29,251 | 39,960 | 90.5% |
| 4,096 | 65,536 | 10.90 | OOM | 27,518 | — | — |
| 8,192 | 131,072 | 20.55 | OOM | 24,227 | — | — |

### OOM位置

- Checkpoint Off：最长通过2,944 tokens；下一档3,072 tokens发生OOM（128-token步长）。
- Checkpoint On：已通过全部阶梯，最长8,192 tokens。

### 两个旧profile的独立进程重测

| 输入profile | 开关 | 状态 | 峰值allocated GiB | 有效tokens/s |
|---|---|---|---:|---:|
| medium | On | OK | 1.77 | 13,293 |
| medium | Off | OK | 8.21 | 22,353 |
| long_real_pairs | On | OK | 8.39 | 21,902 |
| long_real_pairs | Off | OOM | — | — |

## 推广到其他batch：看物理microbatch，不看累计后的batch

以下把有效batch都固定为32对。microbatch8/16/32分别累计4/2/1次。

| sequence tokens | 物理microbatch | On峰值allocated GiB | Off峰值allocated GiB | On有效tokens/s | Off有效tokens/s |
|---|---:|---:|---:|---:|---:|
| 1,024 | 8 | 2.46 | 16.91 | 29,112 | 39,388 |
| 1,024 | 16 | 3.66 | 32.56 | 29,798 | 40,922 |
| 1,024 | 32 | 6.04 | 63.82 | 30,447 | 41,436 |
| 2,048 | 8 | 3.66 | 32.56 | 28,595 | 39,218 |
| 2,048 | 16 | 6.07 | 63.85 | 29,251 | 39,960 |
| 2,048 | 32 | 10.86 | OOM | 29,595 | — |

实际选配置时，先查每个microbatch补齐后的sequence长度，再与物理batch一起查上表。累计次数控制一次更新看多少样本；显存主要由单次驻留的物理batch和补齐长度决定。混长输入优先按长度分桶，减少短文本被补到长文本长度的开销。

## 分配显存与allocator保留量

| 长度 | On allocated / reserved GiB | Off allocated / reserved GiB |
|---|---:|---:|
| 256 | 1.85 / 1.95 | 9.09 / 9.17 |
| 512 | 2.46 / 2.62 | 16.91 / 17.04 |
| 1,024 | 3.66 / 3.96 | 32.56 / 32.88 |
| 2,048 | 6.07 / 6.75 | 63.85 / 64.41 |
| 4,096 | 10.90 / 12.25 | OOM |
| 8,192 | 20.55 / 23.00 | OOM |

## 环境与原始数据

- Torch 2.11.0+cu130，CUDA 13.0，transformers 5.5.0，PEFT 0.18.1。
- 总参数606,146,560，可训练参数10,094,592；参数dtype计数：`{'torch.bfloat16': 596054016, 'torch.float32': 10092544}`。
- 实际SDPA算子：`['aten::_scaled_dot_product_cudnn_attention', 'aten::_scaled_dot_product_cudnn_attention_backward', 'aten::scaled_dot_product_attention']`。
- 吞吐计时：CPU拼batch、H2D、前向、分类loss、反向及梯度清零；没有optimizer step。
- 每组都由独立进程启动，显存不继承上一组的allocator缓存。
- `original-profiles.json`保存旧profile全部长度和原始row索引；`source-pairs.json`保存固定阶梯输入；`manifest.json`保存配置和hash。
- `cases/*.json`保存逐次耗时、前向/反向耗时、allocated/reserved、padding、OOM发生位置；`results.json`和`summary.json`为汇总。
- `worker-logs/`为完整进程输出；`benchmark.py`和`test_benchmark.py`可复用。
