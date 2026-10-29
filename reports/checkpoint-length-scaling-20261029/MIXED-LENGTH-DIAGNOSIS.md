# 为什么固定2816能跑，真实混长2807却OOM

原混长组是同32条2527–2807-token输入，两个物理batch分别补齐至2576和2808。默认allocator下，第二个batch前向OOM：当时allocated74.82 GiB、reserved但未使用17.41 GiB，GPU仅剩100 MiB。

| 对照 | 实际shape | Allocator | 开关 | 结果 | 峰值allocated / reserved GiB | 有效tokens/s |
|---|---|---|---|---|---:|---:|
| dense2808-default | 16×2808 + 16×2808 | default | Off | complete | 87.29 / 88.22 | 39,457 |
| mixed-long-expandable | 16×2576 + 16×2808 | expandable_segments:True | Off | oom | 92.25 / 92.31 | — |
| mixed-long-fixedpad2808 | 16×2808 + 16×2808 | default | Off | oom | 91.88 / 92.29 | — |
| mixed-long-batch8 | 8×2552 + 8×2576 + 8×2632 + 8×2808 | default | Off | complete | 49.94 / 67.91 | 29,885 |
| mixed-medium-kernel | 16×208 + 16×208 | default | Off | complete | 8.21 / 8.33 | 22,853 |
| mixed-long-on-kernel | 16×2576 + 16×2808 | default | On | complete | 8.39 / 10.90 | 21,944 |

## 实际attention算子

- dense2808-default：`['aten::_scaled_dot_product_cudnn_attention', 'aten::_scaled_dot_product_cudnn_attention_backward', 'aten::scaled_dot_product_attention']`。
- mixed-long-batch8：`['aten::_scaled_dot_product_cudnn_attention', 'aten::_scaled_dot_product_cudnn_attention_backward', 'aten::scaled_dot_product_attention']`。
- mixed-medium-kernel：`['aten::_scaled_dot_product_cudnn_attention', 'aten::_scaled_dot_product_cudnn_attention_backward', 'aten::scaled_dot_product_attention']`。
- mixed-long-on-kernel：`['aten::_scaled_dot_product_cudnn_attention', 'aten::_scaled_dot_product_cudnn_attention_backward', 'aten::scaled_dot_product_attention']`。

各对照的逐步OOM位置、输入长度与完整错误均保存在本目录的JSON和log中。

## Padding mask定位结果

### 同样的tokens和shape，只改变padding mask

用原来同32条真实长输入，全部补齐至2808，保持token IDs和物理batch不变：

| 物理microbatch | Mask | Checkpoint | 结果 | 峰值allocated GiB | 峰值reserved GiB |
|---|---|---|---|---:|---:|
| 16 | 真实padding mask | Off | OOM | OOM | OOM |
| 16 | 全1 mask定位对照 | Off | 通过 | 87.29 | 88.22 |
| 8 | 真实padding mask | Off | 通过 | 50.05 | 50.82 |
| 8 | 全1 mask定位对照 | Off | 通过 | 44.29 | 44.78 |

microbatch8仅切换mask就增加5.76 GiB，即13.0%。全1 mask用于定位执行开销；正式训练保留真实padding mask。

### 已跟踪到的执行路径

| 运行路径 | Q头数 | SDPA接收的K/V头数 | 原生GQA | SDPA mask |
|---|---:|---:|---|---|
| 全1 mask，整齐路径 | 16 | 8 | 开启 | None，使用is_causal=True |
| 真实padding mask | 16 | 16 | 关闭，先repeat_kv | 显式4D mask |

运行时对同一组2×256输入抓取了全部28层的SDPA参数。真实padding路径中的K/V已由8头展开到16头；对应transformers 5.5.0的`integrations/sdpa_attention.py:28–62`。两种路径实测都调用cuDNN attention，额外开销出现在mask与K/V展开这一执行路径。[实际调用形状](attention-dispatch.json)已保存。

### allocator和物理batch对照

| 真实2527–2807-token输入 | 物理microbatch | 结果 | allocated GiB | reserved GiB | 有效tokens/s |
|---|---:|---|---:|---:|---:|
| Checkpoint Off，默认allocator | 16 | OOM | OOM | OOM | — |
| Checkpoint Off，expandable_segments | 16 | OOM | OOM | OOM | — |
| Checkpoint Off，减小microbatch | 8 | 通过 | 49.94 | 67.91 | 29,885 |
| Checkpoint On | 16 | 通过 | 8.39 | 10.90 | 21,944 |

默认allocator在第二个microbatch前向OOM时，reserved但未使用的显存有17.41 GiB；改expandable_segments后仍OOM，统一padding至2808后也仍OOM。本轮直接解决方法是开启checkpointing，或者把物理microbatch从16降到8。

