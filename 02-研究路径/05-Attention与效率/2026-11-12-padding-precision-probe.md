# Padding形状数值漂移：真实输入的有界精度机制对照

**状态：真实2×2对照已完成并直接回读；不是学习质量或新public实验。** 完整public已直接核对：dense24.71／packed24.70，各140620输出／140178官方计分；有14801请求超原数值合同、378field choice翻转（21带外），不能称预测等价。三mode同FA3训练布局393gradient/update均过，但单独/混批原隔离RMS门禁未过。

## 待验证假设，不当作事实

1. 单独与混批同时改变GEMM外形，可能是库的batch/slice数值差异，而不是序列attention串扰；必须增加**同几何、等长真实邻居替换**控制。
2. Torch原Qwen3 RoPE在禁autocast的FP32 matmul下算频率，但原全局allow_tf32=True仍可能改变FP32路径；比较真实first-row cos/sin及layer0/final hidden定位。
3. BF16 GEMM默认允许截断中间累加，可能放大形状误差；单独关闭与TF32关闭分开比较。

PyTorch2.9官方数值文档已实际读取：`https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html`。文档明确batch/slice非bitwise保证、TF32读取10bit尾数、BF16 reduced-precision reduction默认开启。以上说明合理研究变量，不证明本机误差来源已经确定。

## 四卡有界2×2合同

每卡一个(TF32允许,BF16 reduced reduction允许)：(1,1)/(0,1)/(1,0)/(0,0)，同一实际R1权重、同一原训练macro中按长度/重复prefix元数据选A/B/C，不更新任何权重、不读teacher target或public gold、不重跑旧训练/旧public。

- 原macro128pair/45216tokens不改，只对其真实行做Forward机制切片；保留所有原数据引用和选择索引。
- 每mode causal/bidirectional/prefix_block比较A单独、A+B、A+C（B/C实际等长且内容不同），另有DenseHF/packed/dense-same-kernel控制。
- 所有ctx/position/cu/EOS/模型/种子/FA3 deterministic不变，TF32与BF16累加是仅有新变量；原fail结果不覆盖、不改容差。
- 捕获实际RoPE first-row cos/sin、layer0后的真实hidden、EOS hidden和两类logits，分层保存；等长换邻居应无跨样本内容影响，混批几何漂移单独报告。
- 共享prefix full/fork沿原raw-logits和margin合同做同组比较，bidi明确无cache；原prefixBlock失败不会因新probe自动改成通过。
- 每worker15分钟上限，所有结果独立namespace保留。生产准入仍需真实梯度/一步更新及完整Suite，不用一次forward改称修复或提分。

## 实际结果与有限结论

- v1在模型加载前因fixture约束缺覆盖退出，保留；v2将旧失败target/cache位置32/33固定，只从原macro元数据选择实际邻居7/9，L340/P337/S3，CPU先冻结，未造token或按答案挑选。原计划不再假定缓存组本身能提供等长替换邻居。
- 四旗标各自执行三mode，1064次native deterministic forward，无学习更新；耗时5.59～5.75s。原始SUMMARY/ACTUAL_TENSORS保留，主代理逐tensor回读而非只信pass布尔。
- 同几何换邻居所有8个观察边界及目标logits逐位相同。HF和packed各自solo/mixed输出也逐位相同，二者均超过原RMS门禁；不是packing独有误差。
- 四旗标的fullrow所观察hidden逐位相同；TF32或BF16中间归约开关没有消除这个形状漂移。不要归因给RoPE、TF32或宣称其已修复。
- 首个观测差异在attention context0之后、layer1 input norm之前，具体o_proj/MLP仍未分解定位。原逐题数值失败不据此重评分为通过。
- 新扩展实际控制对完整原128pair／45216目标token／4micro执行等几何邻居B/C：三mode全部128目标logits及393参数VJP逐位相同，28层realT核对通过，无optimizer更新。结合已完成同核梯度/实际step，仅签fullrow科学训练准入；不是几何不变性、cached serve或完整public数值准入。
- 原Triton prefixBlock full/fork raw约3.17%／margin约5.23%仍失败；另一次原生paged候选同样未过，不用换cache后端掩盖几何误差。

远端证据：`/tmp/opencode/jev-padding-precision-20261112-v2/FINAL_PRIMARY_VERIFICATION.json`及`/tmp/opencode/jev-fullrow-admission-20261112-v1/FINAL_PRIMARY_VERIFICATION.json`；已回读仓库聚合证据，不将私有实际激活.pt下载到Git。
