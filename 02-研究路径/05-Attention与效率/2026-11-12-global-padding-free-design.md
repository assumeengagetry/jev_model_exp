# Qwen3-0.6B训练／推理全局padding-free设计与质量验证

**状态：用户明确要求尽可能全局padding-free，并验证是否提高分数；沿用自主执行授权。旧代码／权重／结果保留，新实现进入独立namespace，尚无padding-free实测结果。**

## 1．已经实查的现状

- 训练`pair_training_runtime.collate`创建`[B,ceil(maxL/8)*8]`，budget用`B*longest`；Embedding／QKV／Norm／MLP／LoRA仍处理pad，只有FA3 wrapper在attention内unpad。
- v2推理`KernelRuntime._inputs`按Graph的batch/width向上取整，prefix和suffix也有补齐；不是端到端packed execution。
- 已安装Hopper FA3的实际`flash_attn_varlen_func`接受`q/k/v[sumL,heads,128]`、CUDA int32 cu_seqlens、实际maxlen，支持causal/GQA和已核实deterministic backward；不需要新增依赖或更换Torch。

## 2．“全局”的可验证定义

1. 直接将真实序列拼成`flat_ids[sumL]`；不先分配padded CPU/GPU矩阵再伪称packed。
2. 主干hidden始终`[sumL,1024]`，Embedding／所有QKV/o与LoRA／两级RMSNorm／MLP只接收真实token行。RoPE可有只含真实T的singleton batch/head view，不存在业务dummy row。
3. cu_seqlens严格划分独立序列，position IDs在每条完整序列重置；不把拼接后的一条普通causal attention当正确packing。
4. 分类读出使用真实EOS offsets=`cu[1:]-1`；case/field/option映射保持，softmax仍逐完整field，不跨样本。
5. micro预算使用实际token和；global macro与loss分母仍按真实逻辑pair/case，不按token或rank私自改权重。关系原子unit、case边界保留。
6. CUDA Graph不能为了复用bucket补假token。先用动态真实shape eager；以后只允许真实shape完全匹配且有界的Graph缓存，不把效率目标倒退成padding。
7. Tensor Core内部tile的无效lane、allocator对齐和KV内存容量不等于业务padding；分别记账，不能宣称硬件所有FLOP绝无对齐开销。

## 3．共享packed骨干

直接复用已加载PEFT Qwen3模块／参数，不改保存格式或更新全局库文件：

```text
flat ids → embed[T,H]
每层：RMSNorm → QKV/LoRA[T,*] → Q/K norm+真实RoPE
     → native FA3 varlen（train/full）或packed prefix-fork（infer）
     → o/LoRA → residual → RMSNorm → gate/up/down/LoRA → residual
真实EOS gather → final norm/head → 2类margin → 原完整field softmax
```

训练保留原PyTorch可微Norm／MLP／PEFT linear；不能拿旧forward-only Triton fused norm冒充支持训练。严格FA3 deterministic wrapper保留实际forward/backward证据，直接使用varlen入口，不经过HF重新unpad／pad。

推理可复用原CachedLoRA／FusedRMS／FusedMLP的算术边界，但输入是真实T；只在无梯度推理使用forward-only融合。head保持原存储及执行精度，第一轮不夹带FP32head优化。

## 4．padding-free且不放弃prefix复用

- 前端仍固定同一pair renderer、独立prefix/candidate分词、EOS、全部criteria，逻辑group／request恢复顺序不变。
- prefix prefill按实际P执行、每层保存真实`K/V[P,8,128]`，不为context上限预分配业务行或算prefix pad。多个prefix以后可以packed唯一owner预填；首版允许逐owner独立真实P，仍无padding。
- suffix波次将所有真实suffix token拼成T，RoPE位置为`P+local_suffix_position`。packed fork attention读取一份immutable prefix，以及本branch的causal suffix；cu和owner metadata防跨branch读取。不为每候选repeat/cat prefix KV。
- full-row波次使用nativeFA3 varlen；FFN／LoRA等全程只处理真实token。fork kernel的末端tile mask仅是GPU tile，不产生隐藏层假token张量。
- 原v2 Graph／SDPA／fused与新FA3的差异不能全部归给padding，因此同权重quality A/B使用一个显式dense-same-kernel控制：相同attention和前缀策略，仅tokenwise层保留旧padding。原v2历史分数另作参考。

## 5．质量与效率验证顺序

### PF0：真实原始macro的数值／隔离实验

固定同一冻结诊断权重和相同完整pair token：原padded HF+FA3与新packed可微骨干比较EOS margin、全局point/relation loss、FP32梯度方向／范数和同一真实macro的一步更新；先固定相同micro组成，再测按real-token重新切micro的性能。只使用已批准真实机制实验，不新增synthetic pytest/smoke或模型／语料hash。

覆盖真实长短混合、单序列、多个case、同field多候选、关系端点、顺序置换及换邻居隔离；tokenwise hook记录真实／padded行数，防止只attention去pad。容差和失败解释在执行计划中先冻结，不看结果挑。

### PF1：同权重完整public推理A/B

dense-same-kernel与packed各自完整140620执行、442排除后140178计分；原runner/scorer不改、全部候选不截断。保存逐题probability／choice／native scores差异，报告墙钟／显存／真实处理token行。

既有24.91／24.67谱系有94个已知ToolRet请求文本重合，仅可用作数值／效率诊断；即使packing得到更高该分数，也不称干净SOTA。第一轮固定权重只回答是否数值／质量变化，不把浮点+0.02当系统性学习收益。

### PF2：clean-origin训练质量对照

新公平分支从同一原基座重建LoRA／head，匹配初始化／数据／loss／global macro／scheduler／曝光，比较padded与packed；不得用旧已训权重或换了数据的分数证明packing提分。可在BCE／Field-CE的四臂factorial中安排dense/packed各一臂，必要时独立seed复验。

相同曝光比较“packing是否保持或改善最终质量”；相同GPU时间比较“节约padding后多训练真实样本的收益”，额外token／更新数必须披露，两种结论不混写。

## 6．保留的边界与成功条件

- 不改旧sealedcode／数据／停止标记，不恢复旧暂停任务、不删除partial、不触碰他人GPU进程、不发布／push。
- 本项目已知public输入暴露的旧谱系保留历史标签；新干净分支另有去污数据／case隔离，基座预训练暴露不作绝对无泄漏承诺。
- 成功必须有实际shape／tokenwise行数、nativeFA3 backward、隔离／EOS、梯度／一步更新、真实完整public和训练证据。只改batch阈值、只attention-unpad、或只写设计均不算完成。
- padding-free主要是效率／显存优化，数学上不应自动改变判别规则。能否涨分由上述受控实验回答，不预填结果。
