# 激活重算、具体长度与padding执行路径：协作实测增补

**来源：协作会话已完成的49个测量单元及28层SDPA参数追踪，本会话只读引用，未重复测试。**
完整报告见[RESEARCH-REPORT.md](../checkpoint-length-scaling-20261029/RESEARCH-REPORT.md)，原测量、输入与日志均保留。
这是前向/反向资源研究，optimizer更新次数0，未保存或修改模型权重，不增加Language参评模型数。

## 旧profile的实际输入

| Profile | 实际长度范围 | 来源与样本数 | 两个物理microbatch shape | 有效 / padded tokens | 整profile padding |
|---|---:|---|---|---:|---:|
| medium | 192—205 | ANLI r1，32对 | 16×208、16×208 | 6334 / 6656 | 4.84% |
| long_real_pairs | 2527—2807 | RAGTruth Summary，32对 | 16×2576、16×2808 | 83466 / 86144 | 3.11% |

medium的筛选条件是192—384，但所取前32对实际集中在192—205，不能用筛选上限384代替实长。
旧long是2527—2807，不能称为固定4096输入。64条逐项长度与原row索引见[original-inputs.csv](../checkpoint-length-scaling-20261029/original-inputs.csv)。

## 无padding长度阶梯

固定Qwen3-0.6B/R0、LoRA r16/alpha32、BF16、causal SDPA，物理microbatch16、累计2次；H800 NVL的CUDA可见容量93.09 GiB。
同32对真实长输入裁到各固定长度，padding为0。每格两个独立进程，每次预热1个effective batch、测3个前向+反向。
On/Off指gradient checkpointing（激活重算），不是磁盘模型checkpoint保存。

| 每条长度 | On峰值allocated GiB | Off峰值allocated GiB | On有效token/s | Off有效token/s | On耗时/Off耗时 |
|---:|---:|---:|---:|---:|---:|
| 256 | 1.85 | 9.09 | 17024 | 31236 | 1.83 |
| 512 | 2.46 | 16.91 | 29400 | 39375 | 1.34 |
| 1024 | 3.66 | 32.56 | 29798 | 40922 | 1.37 |
| 2048 | 6.07 | 63.85 | 29251 | 39960 | 1.37 |
| 4096 | 10.90 | OOM | 27518 | — | — |
| 8192 | 20.55 | OOM | 24227 | — | — |

2048时allocated从63.85降至6.07 GiB，节省约90.5%，代价是本工况耗时约增加37%。
Off的128-token步长搜索中2944通过、3072 OOM；这限定了本配置与搜索步长下的观察边界，不是所有输入/模型的普适上限。
On已测到8192，allocated20.55 GiB、reserved23.00 GiB；没有据此证明8192是On的最大长度。
计时包含拼batch、H2D、前向、分类loss、反向与梯度清零，不含optimizer更新；不能直接当完整训练step或模型能力测量。

## 为什么真实混长2807会OOM，而整齐输入2944能通过

同32条真实2527—2807-token输入，Off、micro16在第二个microbatch前向OOM；On、micro16可跑，峰值allocated8.39 GiB。
Off降为micro8可跑，allocated49.94、reserved67.91 GiB。仅改`expandable_segments`或统一pad至2808，仍未解决Off16 OOM。

同token IDs、同2808 shape、同micro8，仅切换mask：

| mask定位条件 | Off allocated GiB | Off reserved GiB |
|---|---:|---:|
| 全1 mask定位对照 | 44.29 | 44.78 |
| 真实padding mask | 50.05 | 50.82 |

该对照增加5.76 GiB，约13.0%。全1 mask只是定位开销，**不证明预测等价，正式训练不能为省显存忽略真实padding mask**。

协作会话另以相同2×256 token IDs抓取全部28层的实际SDPA参数：

| 当前TF5.5.0路径 | Q头数 | SDPA接收K/V头数 | enable_gqa | SDPA mask | is_causal |
|---|---:|---:|---|---|---|
| 全1 mask进入整齐快路径 | 16 | 8 | True | None | True |
| 真实padding mask | 16 | 16，已repeat_kv | False | bool，[2,1,256,256] | False |

带mask路径的因果约束由显式4D mask承载，`is_causal=False`不表示模型改成了双向注意力。
两种路径实际都调用cuDNN attention；不能将本结果写成“padding使attention回退到math实现”。
在当前transformers 5.5.0的SDPA包装路径，真实padding触发K/V从8头展开到16头，结合相同shape的mask对照，支持这一路径有额外显存开销。
这也解释了为什么不能仅按3.11%的padding量或allocator碎片单独判断混长OOM。
实际调用见[attention-dispatch.json](../checkpoint-length-scaling-20261029/attention-dispatch.json)，混长对照见[MIXED-LENGTH-DIAGNOSIS.md](../checkpoint-length-scaling-20261029/MIXED-LENGTH-DIAGNOSIS.md)。

## 整齐输入的经验拟合与适用范围

令**B为物理microbatch，而非梯度累计后的effective batch**，L为补齐后的sequence长度，x=B×L/16384，M单位GiB：

- On：M≈1.243＋2.412x。
- Off：M≈1.266＋31.291x。

已测8×2048与16×1024的allocated几乎相同：On约3.66、Off约32.56 GiB。
公式描述本模型/精度/实现的**整齐、无padding路径allocated**；不包含AdamW等optimizer状态，也不是reserved或整个进程容量公式。
真实混长且有显式padding mask时应使用相应实测或另留余量，不能直接套无padding直线；换模型、精度或attention实现应重新校准。
计算器和拟合细节见[estimate_memory.py](../checkpoint-length-scaling-20261029/estimate_memory.py)、[SCALING-ANALYSIS.md](../checkpoint-length-scaling-20261029/SCALING-ANALYSIS.md)。

## 对比例研究的处理

GPU0完成/释放通知已收到，新资源测试预约解除；这不自动使尚未生成的比例采样表训练就绪。
后续各比例候选继续共用模型、长度、batch/LR、真实padding mask与checkpointing设置，不把资源探针的开关变化混入比例因素。
完整采样流、自然dev/test和实际worker曝光日志校验是启动条件。后续比例首轮已通过准备检查并完成训练/评测与实际曝光复核，见[比例研究增补](PAIR-RATIO-ADDENDUM.md)；没有重做本资源研究。全部原始49单元见[all-measurements.csv](../checkpoint-length-scaling-20261029/all-measurements.csv)，本会话未改其产物。
