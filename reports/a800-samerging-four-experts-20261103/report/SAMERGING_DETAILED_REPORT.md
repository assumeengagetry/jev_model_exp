# SAMerging 四专家合并实验报告

**实验日期：2026-11-03；时间均为北京时间（CST，UTC+8）。**

**报告范围：** Dalili、Mahdavi 的 SAMerging 方法在现有四个 Qwen3-0.6B candidate-pair classifier 上的原理、实现适配、实验配置、运行过程和实测结果。本报告整理已完成实验，不触发新的训练或评测。

> **核心结论：本次 SAMerging 的完整公共 Index 为20.21，coverage为100%；相对MIXED_ALL低2.61分、相对average低1.25分、相对分层SLERP低1.34分，高于TIES 1.21分。本配置没有取得优于简单平均的合并收益。**

这是一轮SAMerging核心算法的Qwen适配实测，**不是官方GPT-2/GLUE或CLIP实验的无修改复现，也不是对该方法所有配置的否定**。

## 1. 实验目标与对照

目标是把Knowledge、Retrieval、Tools、Arts四个已训练expert合并成一个通用模型，检验少量无标签校准数据学习出的merge coefficients是否优于固定的平均、TIES及SLERP等合并方式。

| 对照 | 含义 |
|---|---|
| MIXED_ALL | 原先使用四领域混合数据训练的单模型；只作对照，不参与SAMerging权重构造 |
| MERGE_AVERAGE | 四个领域expert的有效权重等权平均 |
| MERGE_TIES | 原已完成的TIES合并条件 |
| MERGE_SLERP_HIERARCHICAL | 原已完成的固定分组、分层SLERP条件 |
| MERGE_SAMERGING | 本次通过multi-teacher KL和adaptive SAM学习系数的合并模型 |

SAMerging的teachers是四个Qwen expert，**不是PelaAI/KnowLine-31B-Gen1**。KnowLine的百万级软标签缓存没有用于本次SAMerging校准。

## 2. 方法原理

### 2.1 从固定平均到可学习task-vector组合

设四个expert的共同初始化为 $\theta_0$，第 $t$ 个expert的有效参数为 $\theta_t$，定义task vector：

$$
\tau_t=\theta_t-\theta_0.
$$

对于参数tensor $k$，合并模型为：

$$
\theta_\lambda^{(k)}=\theta_0^{(k)}+
\sum_{t=1}^{4}\lambda_{t,k}\tau_t^{(k)}.
$$

每个expert在每个tensor上的贡献由一个可学习标量控制。expert原权重、共同模型权重及task vectors均冻结，只优化这些系数。最终模型仍受限于各tensor的task-vector affine span，不能像自由微调那样学习任意新的参数方向。

本次系数不要求非负、不要求和为1。因此它不是概率加权或convex model soup，也不是推理时的ensemble。

### 2.2 Multi-teacher distillation目标

对于领域 $t$ 的输入 $x$，使用该领域expert的预测分布：

$$
p_t(y\mid x)=\operatorname{softmax}(f_{\theta_t}(x)/T),\qquad
q_\lambda(y\mid x)=\operatorname{softmax}(f_{\theta_\lambda}(x)/T).
$$

本次 $T=1$，目标为：

$$
L_{\mathrm{KD}}(\lambda)=\frac14\sum_{t=1}^{4}
\mathbb E_{x\in\mathcal B_t}
D_{\mathrm{KL}}\big(p_t(\cdot\mid x)\Vert q_\lambda(\cdot\mid x)\big).
$$

Knowledge输入由Knowledge expert提供监督，其余领域同理。不是对每个输入平均四个expert的输出。本任务的label space为`[not_match, match]`，蒸馏的是两个logits对应的完整二分类分布，不是词表级LM distillation、CoT或hidden-state matching，也不把Teacher概率取argmax。

校准不使用原始标签CE。KL变小只说明更接近对应expert，不保证公共Index变高；expert自身的错误和分布差异仍可能传递到合并模型。

### 2.3 SAM与本次实际adaptive扰动

普通SAM的直观目标是：

$$
\min_\lambda\max_{\|\epsilon\|_2\le\rho}L_{\mathrm{KD}}(\lambda+\epsilon).
$$

先在当前系数计算梯度，构造增大loss的扰动，再在扰动后计算梯度，恢复原系数并执行优化器更新。一次SAM step需要两轮forward/backward。

本次跟随官方NLP代码设置`adaptive=True`，实际扰动为：

$$
\epsilon=\rho\frac{\lambda^2\odot g}
{\left\||\lambda|\odot g\right\|_2+10^{-12}},
\qquad g=\nabla_\lambda L_{\mathrm{KD}}(\lambda).
$$

它与普通Euclidean SAM的 $\rho g/\|g\|_2$ 不完全相同，实际扰动的普通L2 norm不必等于 $\rho$。本次两遍forward使用完全相同的输入及expert targets，每遍都根据当前系数重新构造merged weights。

论文以flatness-aware PAC-Bayes分析和student–teacher分布差异提供理论动机，但本次没有进行独立的loss-landscape、Hessian或泛化界测量；不能把执行了SAM更新等同于实证证明最终解更平坦。

## 3. 模型、共同起点与LoRA处理

### 3.1 模型与共同初始化

使用Qwen3-0.6B sequence classifier，28个Transformer blocks、hidden size 1024、统一二分类label space。四个expert为：

```
SPECIALIST_KNOWLEDGE
SPECIALIST_RETRIEVAL
SPECIALIST_TOOLS
SPECIALIST_ARTS
```

共同起点是原始Qwen3-0.6B backbone加上四专家当初共用的LoRA初始化和初始随机二分类head，即`Qwen3-0.6B-common-pair-init-r32`。**不是已经四域训练后的MIXED_ALL。**

### 3.2 先展开有效权重，不直接平均LoRA因子

每个expert使用LoRA `r=32, alpha=64`。对于一个linear weight：

$$
W_t=W_{\mathrm{base}}+\frac{\alpha}{r}B_tA_t
=W_{\mathrm{base}}+2B_tA_t.
$$

本次通过PEFT的`merge_and_unload()`在FP32中得到effective weights，再计算 $W_t-W_0$。不分别平均A/B矩阵，因为通常：

$$
\sum_t\lambda_tB_tA_t\ne
\left(\sum_t\lambda_tB_t\right)
\left(\sum_t\lambda_tA_t\right).
$$

### 3.3 实际参数粒度：197个tensor、788个系数

每个block的`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`七个weight tensors发生变化：

$$
28\times7=196.
$$

再加上一个`score.weight`，共197个可合并tensor。每个tensor学习四个expert系数，共：

$$
4\times197=788.
$$

所以这里的layer-wise实际上是**per-weight-tensor**，不是每个Transformer block仅一个系数。没有发生变化的embedding、norm等权重保留共同起点。

四个expert的score head均为`[2, 1024]`，本次也作为task vector合并成单个共享head。这不同于官方GPT-2/GLUE示例的task-specific heads，是本次部署统一pair classifier的适配选择。

## 4. 实现与梯度数据流

`run_samerging.py`首先加载共同模型和四个expert的effective FP32 state，检查tensor keys一致，随后保存各变化tensor的共同值和四个task vectors。

每次forward计算：

```python
merged_weights[key] = (
    initial[key]
    + torch.tensordot(coefficients[:, k], task_vectors[key], dims=([0], [0]))
).to(torch.bfloat16)

logits = torch.func.functional_call(
    model, merged_weights, (), inputs,
    tie_weights=True, strict=False,
).logits.float()
```

`functional_call`让临时权重保留与系数的autograd联系，梯度沿`loss → logits → merged weights → coefficients`传播。模型原参数冻结，但替换权重是可微表达式，因此仍能得到系数梯度。

优化循环的核心为：

```python
optimizer.zero_grad()
backward_on_fixed_batch()       # 当前系数，每个microbatch重建weights
optimizer.first_step(zero_grad=True)
backward_on_same_fixed_batch()  # 扰动后系数，同输入、同targets、重建weights
optimizer.second_step(zero_grad=True)
```

每个microbatch的loss按`16 * 4`归一化，再累积梯度。没有对expert权重或LoRA A/B执行optimizer update。788个trainable parameters不等于近乎零计算成本：求系数梯度仍需经过整个模型，SAM还需要两次forward/backward。

## 5. 校准数据与batch schedule

### 5.1 固定无标签pair采样

从原四领域训练池中每域随机抽1,600个不同row indices，seed42，不按Teacher置信度筛选，不读取原始target。为了兼容旧collate接口使用的dummy target不进入loss。

| 领域 | 原pool pairs | 校准pairs | 校准中的不同state数 | 平均有效tokens | 最大有效tokens |
|---|---:|---:|---:|---:|---:|
| Knowledge | 3,073,453 | 1,600 | 1,598 | 173.25 | 2,252 |
| Retrieval | 11,047,617 | 1,600 | 1,592 | 128.93 | 1,891 |
| Tools | 1,579,715 | 1,600 | 1,594 | 433.61 | 1,906 |
| Arts | 2,472,921 | 1,600 | 1,598 | 512.29 | 4,096 |

6,400指pair rows，不是6,400个保证互不相关的完整case。校准沿用原训练的4,096-token head-tail policy；没有用评测数据调系数。

**设定边界：** 这里使用原训练池输入并丢弃标签，属于label-free calibration，但并非严格复现完全无法访问原训练输入的场景。本次没有新增跨数据集去重审计。

### 5.2 对齐的expert logits缓存

每个领域的输入仅由所属expert提供targets。通过one-hot系数按数学构造恢复相应expert的有效权重，预计算并缓存`[4, 1600, 2]`的FP32 logits。输入位置与缓存位置一一对应，避免shuffle后错配。

### 5.3 600步的实际含义

- 每step每域16 pairs，四域合计64 pairs。
- 每域1,600/16=100个batch；固定样本集合独立shuffle六轮，构成600步schedule。
- 共38,400次逻辑pair exposure；SAM两遍对应76,800次pair级forward/backward exposure。
- 每域16 pairs按8,192 padded-token budget拆分microbatches，不改变pair数量或loss权重。
- 四域loss各占25%，不是按原pool约60.8%的Retrieval比例加权。

精确indices/schedule保存在`../calibration/calibration.npz`，expert logits保存在`../calibration/expert-logits.npy`。

## 6. 实际超参数、软件与精度

| 项目 | 实际设置 |
|---|---|
| Optimized parameters | 788个merge coefficients |
| Coefficient initialization | 全部0.5 |
| Coefficient constraints | 不clamp，不softmax，不要求和为1 |
| Steps / seed | 600 / 42 |
| Batch | 每域16 pairs，合计64；同batch执行两遍SAM |
| Objective | 四域平均KL(expert‖merged)，T=1 |
| Optimizer | 官方SAM实现，SGD base |
| Learning rate / momentum | 1e-3 / 0.99 |
| Weight decay | 5e-4 |
| SAM radius / adaptive | 0.07 / true |
| Coefficients / task vectors | FP32 |
| Forward / exported weights | BF16 |
| Attention | causal SDPA；未安装独立flash-attn包，未做内部kernel profiling |
| TF32 | 允许CUDA matmul使用TF32 |
| PyTorch / Transformers | 2.8.0 / 5.5.0 |
| PEFT / safetensors | 0.18.1 / 0.8.0 |

初始0.5不是等权average。average对应每个系数0.25；本次初始模型为 $\theta_0+0.5\sum_t(\theta_t-\theta_0)$，允许外推。

实际safetensors header确认导出311个tensors全部为BF16，总参数596,051,968。`config.json`的`dtype`字段残留`float32`，属于元数据未同步；评测显式BF16加载，不能仅凭该字段判断实际运行精度。本报告没有事后改动完成实验的权重。

## 7. 与论文及官方示例的差异

固定上游commit：`a194e8ed8ab790dcae9c02a8d1467e00e86b53e5`。

| 项目 | 论文/官方GPT-2示例 | 本次处理 |
|---|---|---|
| 模型和任务 | GPT-2/GLUE，另有CLIP实验 | Qwen3-0.6B四个统一二分类expert |
| Head | 示例使用task-specific heads | 合并为单个共享binary head |
| 校准输入 | 示例从task test loader读取 | 固定抽取训练池输入，不读原始labels，不用评测输入校准 |
| Input/logit alignment | 预计算与训练再次推进shuffle loader | 固定input position与logits对应 |
| SAM perturbed forward | 系数扰动后未显式刷新merged-weight缓存 | 每次forward重新构造权重 |
| Base optimizer | 论文附录描述Adam；实际NLP代码使用SGD | 跟随实际NLP代码的SGD配置 |
| Adaptive geometry | 论文简单公式为L2邻域；NLP代码adaptive=true | 跟随adaptive实现，披露实际扰动公式 |
| Loss总量 | GPT-2代码逐域累加 | 按论文任务权重四域平均 |
| 数值形式 | 示例softmax后取log | 稳定的log_softmax，KL方向不变 |

两个代码问题的依据仅针对该固定版本的GPT-2示例：预计算与实际优化分别继续取shuffle batches会失去样本对应；缓存的merged weights若不在SAM first_step后刷新，第二遍forward不能可靠地反映扰动系数。本次不沿用这两个行为，官方SAM optimizer自身原样保留。

**“跟随官方NLP配置”不等于逐行等价。** 尤其四域平均相比逐域累加多了整体1/4；固定lr、weight decay和600步时，会改变优化轨迹及相对正则强度，不能称为无关紧要的常数。

## 8. 校准完成情况与系数

校准及导出耗时3,232.54s，即53.9分钟，完成600/600 steps。

第一步实际记录：clean KL1.136297、perturbed KL1.429486、系数gradient norm6.856123、实际系数扰动norm0.035。这说明发生了真实的非零系数更新及不同的perturbed forward，不是额外smoke run，也不是泛化能力证据。

最终系数统计：

| 统计 | 数值 |
|---|---:|
| Shape | 4×197 |
| Minimum | −0.623078 |
| Maximum | 4.611129 |
| Mean | 0.487413 |
| 负系数个数 | 4 |
| 大于1的系数个数 | 2 |

最小值对应Knowledge expert的`model.layers.26.mlp.up_proj.weight`；最大值对应Retrieval expert的`model.layers.27.mlp.up_proj.weight`。这只描述实际学到的系数，未通过ablation证明这些极值是效果下降原因。

分类头四系数按Knowledge/Retrieval/Tools/Arts顺序为`[0.261279, 0.366905, 0.393795, 0.325494]`，和约1.347474。它们不是概率或投票比例。

## 9. 导出、推理和评分

### 9.1 单模型导出

将最终系数代入task-vector表达式，保存完整BF16 sequence-classifier权重。部署不需要保留expert副本、task vectors、SAM optimizer或runtime router。结构上没有ensemble开销；本次未做独立latency benchmark，不报告未经测量的加速倍数。

### 9.2 完整公共37项评测

复用原有本地重建公共0.3 suite及native evaluator：110,087个请求，保留全部candidates，不应用训练时的4,096-token截断规则。这不是私测Full，也不冒充公开服务端榜单提交。

对candidate $i$，沿用已有margin及候选归一化逻辑：

$$
m_i=z_{i,\mathrm{match}}-z_{i,\mathrm{not\_match}},\qquad
P(c_i\mid\text{question})=\frac{\exp(m_i)}{\sum_j\exp(m_j)}.
$$

之后使用各benchmark的原生指标和既有Index聚合，不把所有指标统称accuracy，不自行对五个方向做简单算术平均。显式声明为`unscored_fields`的null-gold诊断字段不进入评分，预测保留；没有针对SAMerging筛题或筛候选。

原MIXED_ALL/average/TIES/SLERP复用已完成的同口径评分，不重复运行baseline inference。

### 9.3 冻结KD子集

从完整预测筛出原KD冻结run IDs后直接重算：37项、4,786个完整cases、5,935个请求。没有重新推理。完整公共Index20.21与冻结子集Index20.09是不同口径。

## 10. 运行过程、调度问题与恢复

使用既有allocation275711中的A800，不新增allocation。用户要求优先SAMerging后，暂停了原Teacher/Student及自动推进控制器，并保留其数据/checkpoints。

| 阶段 | 实际过程 |
|---|---|
| 校准 | 11:08:33启动，单卡完成effective weights构造、logits缓存、600步更新和导出 |
| 首次评测 | 主step执行shard0；计划中的三卡sidecar因CPU binding未启动 |
| 故障原因 | 嵌套srun继承8核主step的binding，无法满足24核sidecar的绑定要求；不是OOM或校准失败 |
| 恢复 | 保留校准模型和已完成shard0，从登录节点提交独立三卡step，仅补shards1–3 |
| 完成 | 14:59:26生成完整报告，14:59:27评测step退出0 |

因此实际不是四张卡从头同时评测：先完成shard0，再用三卡补其余shards。修复只改变调度，不改变checkpoint、四分片定义或评测样本；没有重复校准或重跑shard0。

| Shard | 请求数 | Input tokens | 实际worker耗时 |
|---|---:|---:|---:|
| 0 | 27,522 | 253,951,309 | 85.38min |
| 1 | 27,522 | 253,602,711 | 84.96min |
| 2 | 27,522 | 250,904,707 | 84.24min |
| 3 | 27,521 | 254,814,089 | 86.62min |
| **合计** | **110,087** | **1,013,272,816** | 不将并行worker时间相加冒充墙钟时间 |

从首step开始到成功退出的墙钟时间约3h51min，包含调度故障和恢复，不能当作理想四卡并行耗时。

## 11. 实测结果

### 11.1 完整公共结果

| 模型 | Knowledge | Language | Retrieval | Tools | Arts | 公共Index | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| MIXED_ALL | 9.69 | 17.44 | 46.26 | 26.08 | 17.78 | **22.82** | 100% |
| MERGE_AVERAGE | 8.47 | 16.60 | 35.24 | 37.85 | 10.10 | 21.46 | 100% |
| MERGE_SLERP_HIERARCHICAL | 8.56 | 16.50 | 35.29 | 38.25 | 10.13 | 21.55 | 100% |
| MERGE_TIES | 7.14 | 14.40 | 40.81 | 24.29 | 8.27 | 19.00 | 100% |
| **MERGE_SAMERGING** | **8.21** | **15.57** | **33.43** | **35.33** | **9.08** | **20.21** | **100%** |

全部110,087个请求为`ok`，无缺失或`unsupported`请求。

| 对照 | SAMerging的Index差值 |
|---|---:|
| MIXED_ALL | −2.61 |
| MERGE_AVERAGE | −1.25 |
| MERGE_SLERP_HIERARCHICAL | −1.34 |
| MERGE_TIES | ＋1.21 |

相对average，五个方向均下降。相对MIXED_ALL，Tools提高9.25分，但Retrieval下降12.83、Arts下降8.70，总体没有补回。

### 11.2 与原1M KD同一冻结子集的分数

| 模型 | 冻结子集Index |
|---|---:|
| 已训练MIXED_ALL起点BASELINE | 22.39 |
| KD_HARD（1M） | 22.49 |
| KD_SOFT（1M） | 21.36 |
| KD_HYBRID（1M） | 22.38 |
| MERGE_SAMERGING | 20.09 |

本表只对齐评测口径，不是等训练预算或相同teachers的因果对照：SAMerging只训练合并系数，KD训练LoRA/head，而且Teacher来源不同。原始Qwen起点的新KD实验独立进行，不包含在本报告结果中。

## 12. 结论及解释边界

1. **已验证事实：** 本次四Qwen专家SAMerging适配的Index20.21，未超过average、SLERP或MIXED_ALL，只超过TIES；coverage完整。
2. **方法链路事实：** 已实际执行固定输入的multi-teacher KL、两遍adaptive SAM、非零系数更新及单模型导出，不是只做静态加权或将训练loss当质量结果。
3. **不能外推：** 未做初始化、optimizer、SAM/no-SAM、共享head、校准比例/数据量等ablation或大规模搜索，不能断言某一项导致下降，也不能据此否定该方法在其他模型/配置中的效果。
4. **可能的归因线索但未验证：** task-vector span限制、expert自身质量与校准/评测分布差异、共享head处理及优化尺度都可能相关；没有相应实验，不将它们写成已证实原因。

最准确的结论是：**少量无标签输入＋逐tensor系数＋multi-teacher KL＋adaptive SAM这条合并链路已完成实测，但本配置在现有四expert上没有取得优于平均合并的收益。**

## 附录A：代码、数据和完成证据

路径以下均相对本报告所在`report/`目录：

| 产物 | 路径 |
|---|---|
| 自动生成的简版结果报告 | [REPORT.md](REPORT.md) |
| 完整公共对比CSV | [public-comparison.csv](public-comparison.csv) |
| 分方向差值JSON | [differences.json](differences.json) |
| 合并实现、校准和导出 | [../code/run_samerging.py](../code/run_samerging.py) |
| 官方SAM optimizer副本 | [../code/official_sam_utils.py](../code/official_sam_utils.py) |
| 评测、恢复、子集重算 | [../code/evaluate_and_report.py](../code/evaluate_and_report.py) |
| 官方GPT-2配置快照 | [../code/upstream-gpt2-config.yaml](../code/upstream-gpt2-config.yaml) |
| 官方GPT-2实现快照 | [../code/upstream-gpt2-layerwise.py.txt](../code/upstream-gpt2-layerwise.py.txt) |
| 校准manifest | [../calibration-manifest.json](../calibration-manifest.json) |
| 精确校准indices和schedule | [../calibration/calibration.npz](../calibration/calibration.npz) |
| 对齐的expert logits | [../calibration/expert-logits.npy](../calibration/expert-logits.npy) |
| 系数tensor及tensor-key映射 | [../merging-coefficients.pt](../merging-coefficients.pt)、[../coefficient-map.json](../coefficient-map.json) |
| 校准完成记录 | [../MERGE_COMPLETE.json](../MERGE_COMPLETE.json) |
| 完整公共原生完成记录 | [../outputs/evaluation/MERGE_SAMERGING/complete.json](../outputs/evaluation/MERGE_SAMERGING/complete.json) |
| 完整预测 | [../outputs/evaluation/MERGE_SAMERGING/results.jsonl](../outputs/evaluation/MERGE_SAMERGING/results.jsonl) |
| 冻结子集完成记录 | [../outputs/subset/evaluation/MERGE_SAMERGING/complete.json](../outputs/subset/evaluation/MERGE_SAMERGING/complete.json) |
| 最终完成记录 | [../COMPLETE.json](../COMPLETE.json) |

代码阅读入口：`run_samerging.py:44–88`为effective weights/系数与functional_call，`106–162`为采样和logits缓存，`165–239`为SAM训练/导出；原始代码快照保留了实际运行实现。

完整checkpoint仍在集群，不为生成本报告重复下载约1.19GB的模型权重：

```text
/data/run01/scwb721/jev-kd-20261101/experiments/samerging-four-experts-20261103/outputs/MERGE_SAMERGING/final
```

原始训练数据、base权重及expert adapters复用已有项目资产；本报告附带的indices/schedule是对这些资产的引用，不是重新发布原始数据。

## 附录B：原始资料

- Dalili, S. A.; Mahdavi, M. **Model Merging via Multi-Teacher Knowledge Distillation**. arXiv:2512.21288v1. [论文](https://arxiv.org/abs/2512.21288)；[方法章节](https://arxiv.org/html/2512.21288v1#S3.SS3)。
- [固定官方代码版本](https://github.com/arshandalili/SAMerging/tree/a194e8ed8ab790dcae9c02a8d1467e00e86b53e5)。
- [官方GPT-2配置](https://github.com/arshandalili/SAMerging/blob/a194e8ed8ab790dcae9c02a8d1467e00e86b53e5/config/method/samerging/layer_wise_gpt2.yaml)。
- [官方GPT-2 SAMerging实现](https://github.com/arshandalili/SAMerging/blob/a194e8ed8ab790dcae9c02a8d1467e00e86b53e5/fusion_bench/method/samerging/gpt2_layer_wise_samerging.py)。
