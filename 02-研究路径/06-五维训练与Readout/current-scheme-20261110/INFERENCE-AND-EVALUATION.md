# 一次请求如何评分、如何加速、最终怎样计分

[返回总说明](README.md)。这里只描述已保存的两个正式运行，不把待做的 Mac 单文件共享前缀草稿当作已部署代码。

## 1. 原正式 Engine 的每一步

实际源码：decision_engine.py（库外源码），类 `PairClassifierEngine`。

### 1.1 模型加载

1. 从 final 读取保存的 tokenizer、模型 config。
2. 用实际基座构建 `AutoModelForSequenceClassification`，BF16、`attn_implementation='sdpa'`、`trust_remote_code=False`。
3. **先 `base.score.float()` 再加载 PEFT**，保留 FP32 保存头；加载为不可训练。
4. 与 `adapter_model.safetensors` 逐 key、dtype、tensor 精确比较；缺少头或加载舍入会阻止运行。
5. `.to(cuda).eval()`，关闭通用 `use_cache`，读取40,960上下文、pad和EOS。
6. 四个 CPU 计算线程，允许 TF32；模型一次加载，计入加载时间而不是每请求重新加载。

依赖锁定为Torch2.9.1、Transformers4.57.6、PEFT0.18.1、NumPy2.4.6；原 `requirements-inference.txt` 未单独pin safetensors。本次从加速运行原binding补齐[实际运行环境](../../../03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results/RUNTIME.json)：**NVIDIA H800 NVL，95,830MiB显存，驱动590.44.01，Python3.11.18，safetensors0.8.0**。原GPU UUID已去掉；该记录不是另一时期的旧环境快照。执行源码位置见[库外归档说明](SOURCE-ARCHIVE.md)。

### 1.2 规范化与渲染

遍历 `questions` 的原字段顺序，读取原候选字典的顺序。`noul` 没有显式 criteria 时使用 `{'true':'Yes.','false':'No.'}`。

对每个字段，按[训练文档](DATA-AND-TRAINING.md)的**同一模板和正则**构造 prefix；独立 tokenize prefix 和每个 candidate，均 `add_special_tokens=False,truncation=False`，最后加EOS。

CPU `lru_cache(maxsize=2048)` 只缓存文本对应 token IDs，**不是预测概率缓存**；不会跨请求复用答案。候选列表引用只在原正则条件下加入 prefix，不按任务名调 prompt。

### 1.3 原版在整个请求范围合批

原版先收集一个 request 的**所有字段的全部候选**，再一次调用 `score(rows)`。这允许不同短字段的候选合成一个 microbatch。

- 按完整候选长度稳定排序，保留逆映射恢复原顺序。
- microbatch 最大32个候选。
- 按向上取8倍数的 padded width 限制32,768 padded token。
- 每个 singleton 仍允许越过批预算，先按完整候选尝试；批预算不是上下文截断阈值。
- 右填充 pad，显式2D attention mask。
- `torch.inference_mode()`；完整分类 forward 在 BF16 autocast 内。

超过40,960的任何完整候选会让请求明确 Unsupported，**不会截断它、删它或只给其它候选归一化**。

多候选批 OOM 后释放张量、GC、`empty_cache()`，减半 token budget 并重试当前未完成位置；单候选仍 OOM 就整请求 Unsupported，不拿部分候选答案充当完整结果。

### 1.4 哪个位置的 hidden 用来分类

HF sequence-classification 路径先在各 token 投影，再按最右非 pad 位置取最终 logits；本方案附加的终末EOS不是pad，因此正常情况下是该EOS位置。

候选正文包含什么、实际 pad/eos ID是什么都重要，不能用“取数组最后一列”处理右填充，也不能只凭 `attention_mask.sum()` 对任意模型重写 pooling。

### 1.5 从二分类分数变成整题分布

对字段 f 的每个候选 i：

```text
m_i = float(logit_i[match]) - float(logit_i[not_match])
p_i = exp(m_i-max(m)) / sum_j exp(m_j-max(m))
```

需要区分三个量：

1. 训练中单个 pair 的 `sigmoid(m_i)`；
2. 正式一题各候选之间的 `softmax(m)`；
3. Teacher 给某个 pair 的独立 `P(match)`。

它们不相同。正式不能先把每个 m sigmoid 成概率，再 softmax这些概率；也不能把所有字段的候选混在同一个 softmax。

- `choice` 返回 argmax 对应原候选 key 和所有候选概率。
- `noul` 要求恰好两个候选、且唯一一个 key 是 true/yes/1，返回该候选概率。
- 二候选 `noul` 有 `P(true)=sigmoid(m_true-m_false)`，**不是单独 `sigmoid(m_true)`**。
- 当前没有额外推理温度、截距、按 K 校准、旋转平均、投票或正确性重试。

概率必须覆盖全部原始候选，有限并归一化；不筛选、top-k或把多个小softmax拼成一个完整分布。

### 1.6 usage 的含义

```text
usage.input_tokens = sum_over_all_candidates(len(prefix)+len(candidate)+1 EOS)
```

这是原模型的**逻辑输入曝光**。共享前缀后实际送入计算的 token 可减少，但 usage 仍按原逻辑算，不把缓存节省偷算成输入变少。padding不计入 usage；GPU实际计算会受padding影响。

## 2. 加速版没有改权重，但改了执行

实际源码：

- fast_pair_engine.py（库外源码）
- prefix_graph_runtime.py（库外源码）
- run_public.py（库外源码）
- compare_results.py（库外源码）

### 2.1 已保留的部分

同一 joint final、实际基座、未合并LoRA、保存分类头、独立分词边界、EOS、causal语义、整题完整候选、margin及softmax公式。没有新训练、LM字母读出、指针头、量化或prompt改写。

### 2.2 LoRA 推理包装

`resolve_lora` 将196个已加载 vanilla PEFT Linear包装为 `InferenceLoRA`，提前解析唯一活跃的default adapter，仍计算：

```text
result = base(x)
x = x.to(A.weight.dtype)
output = (result + B(A(x))*scale).to(result.dtype)
```

不 merge 低秩权重，也不是把FP32分支偷偷换成BF16常量。它拒绝多活跃adapter、merged、非vanilla variant或非零dropout等不匹配状态。

### 2.3 同字段共享 prefix

当候选数>1且 `(N-1)*prefix_length >= 1024` 时走共享路径；否则走完整候选路径。**1024是这份已执行版本的阈值，不是另一个待实现草稿中4096的默认值。**

1. prefix单独前向一次，将每层KV写入固定buffer。
2. 每个候选后缀含正文＋EOS，分批接在同一个prefix之后。
3. 每波通过 `expand` 只读视图广播prefix KV，再 `torch.cat` 与新后缀KV组成独立分支，不修改共享prefix。
4. 显式4D causal mask屏蔽物理prefix填充、后缀padding和后缀未来位置。
5. 后缀RoPE语义位置为真实prefix长度＋局部后缀位置，不是物理桶宽＋后缀位置。
6. 使用真实后缀length选EOS位置，继续保存head和原autocast策略。

这里用了 `PrefixWriter/PrefixFork` 自定义cache接口包装，**不是已实现 HF DynamicCache 的跨问题 LCP/crop方案**。

当前buffer覆盖一个字段，下一字段重写；没有同请求相邻问题的LCP复用、没有跨请求内容KV字典、也没有输出缓存。

### 2.4 固定CUDA Graph池

CUDA Graph把固定形状的CUDA执行预先捕获，减少每次Python/launch调度成本；不是改变模型的数学任务。

| 路径 | 预定义形状 |
|---|---|
| full | batch=(2,4,8)，width=(64,128,256,512)，12种 |
| prefix | width=(64,128,256,512,1024,2048,4096)，7种 |
| fork | physical prefix=(256,1024,4096)，batch=(8,32)，suffix=(8,32)，12种 |
| 总数 | 最多31种；本次实际31种全部捕获 |

- 初始化时每形状先执行3次准备，再capture一次；它们是Graph初始化，不是额外公共评测样本。
- `graph_reserved_gib=32` 是启动捕获前检查reserved阈值，不是总显存严格硬上限：一个shape的捕获仍可越过阈值。
- 捕获后shape池封存；miss走eager，不在请求中偷偷compile/capture。
- 固定输入buffer，replay输出及时clone，防后续波次覆盖尚未读取的输出。
- autocast `cache_enabled=False`，避免借用短命临时weight-cast buffer。
- prefix固定KV buffer按声明context分配，生命周期为runtime；不存在“缓存无限增长”。

### 2.5 已知实现差别与限制

**新引擎逐字段调用 `score_field`，丢掉了原版跨字段合批。** ACOS/BRIGHT/SATA这类多短字段请求因此可能变慢。共享减少某字段内重复前缀，不自动补偿所有调用碎片化。

新版本也**没有实现与原版完全相同的共享OOM整题fallback／对半重试路径**；原源码中共享prefill及fork没有统一捕获并退回原Engine。此次实际完整public全成功，不等于证明所有外部输入／显存压力下的失败集合不扩大。若要把它作为通用稳健替代版，必须另立有界验收；本次报告不顺手修改正在归档的数值代码。

## 3. 两次正式 public 的事实

两次都使用固定 upstream commit：

```text
25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d
edition: 0.3
```

原冻结套件基础110,201请求＋新增30,419请求＝**140,620**；固定排除442后计分**140,178**。有43个展示benchmark，其中37个计入该public指数；不能把新展示项误算成指数额外得分项。

| 指标 | 原 Engine | 加速 Engine |
|---|---:|---:|
| Public Index | 24.97 | **24.99** |
| Raw Index | 44.15 | 44.18 |
| 完整执行请求／成功 | 140,620／140,620 | 140,620／140,620 |
| 计分请求／成功 | 140,178／140,178 | 140,178／140,178 |
| 实际推理墙钟 | 33,609.1秒 | **12,157.2秒** |
| 请求median | 70.5ms | 16.6ms |
| 请求mean | 238.7ms | 86.2ms |
| 请求p95 | 676.5ms | 328.4ms |
| Knowledge Skill | 9.24 | 9.27 |
| Language Skill | 32.43 | 32.42 |
| Retrieval Skill | 36.06 | 36.14 |
| Tools Skill | 31.93 | 31.94 |
| Arts Skill | 11.42 | 11.43 |

旧运行读取模型7.57秒，只加载一次；新模型加载含Graph初始化15.029秒，其中Graph捕获6.085秒。新pipeline含加载／初始化／评分总墙钟12,220.829秒。不能把3小时说成“模型启动慢”，也不能把评分耗时混进request latency。

原生结果：[旧scores](../../../03-证据与图表/06-五维训练与Readout/joint-generic-continuation-20261109/scores.json)、[新scores](../../../03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results/scores.json)。完整耗时解释：[原版](../../../01-研究结论/05-Attention与效率/joint-generic-continuation-20261109/OFFICIAL-TIMING-EXPLAINED-20261110.md)、[新版](../../../01-研究结论/05-Attention与效率/inference-prefix-graphs-20261110/README.md)。

### 3.1 不能叫逐位无损

两次已有输出按完整run_id/payload/字段/候选对齐，逻辑输入token差为0，无旧预测复用：

- 308,559个choice字段，2,575个选择变化，**一致率99.1655%**。
- 14,700个noul字段，82个跨越0.5阈值。
- 2,506,535个概率，平均绝对差0.001205717，最大绝对差**0.225062558**。

数学评分函数相同，不保证BF16在不同batch、mask、cache和kernel路径下相同。**也没有逐项隔离证明所有漂移都只是舍入**，不能把最大22.5个百分点的差别用“小数误差”一笔带过。

0.02总分差只说明这两份实际输出的本次评分很接近，不是统计显著提升、跨seed验证或通用等价证明。首500条完全相同只对那个短题子集成立。

### 3.2 为什么快了仍要3小时

整体加速 **2.765×**；若目标30分钟，还需要在新版本上约 **6.754×**，不是再优化几个毫秒就自然达到。

| 任务 | 旧请求总耗时 | 新请求总耗时 | 旧／新 |
|---|---:|---:|---:|
| RouterBench | 1h58m57.7s | 51m48.4s | 2.30× |
| POP909 | 2h19m22.6s | 32m51.0s | 4.24× |
| API-Bank | 1h08m08.5s | 5m30.0s | 12.39× |
| ACOS | 7m22.2s | 16m05.0s | 0.46×，变慢 |
| SATA | 6m06.2s | 9m04.9s | 0.67×，变慢 |
| BRIGHT | 5m44.9s | 6m53.5s | 0.83×，变慢 |

原始模型每候选重算prefix的成本是真实大头；修复一部分后，新的长尾与逐field碎片化又显现。没有额外profiler，不虚构GPU/CPU占比或断言恢复跨field合批就一定30分钟。

## 4. Public Index到底是什么

按本次固定kit，不同benchmark先计算自己的原生指标，做coverage调整，再按其chance／track规则得到Skill；常见形式为：

```text
skill = clip((score - chance)/(1 - chance), 0, 1)
```

不能把所有原生指标都叫accuracy：例如有F1、相关性、整case准确率和不同track聚合。area内部gold任务权重1.2，其余1.0，再按固定area权重合成指数：

| Area | 固定权重 |
|---|---:|
| Knowledge | 0.258494 |
| Language | 0.258494 |
| Retrieval | 0.200229 |
| Tools | 0.182783 |
| Arts | 0.1 |

`Index = 100 * sum(weight_area*skill_area)`。因此一个小任务raw涨10个百分点，不等于总分涨10分；接近随机的任务raw看着50%，Skill可能约0。

**24.99 是这个固定 public0.3 的结果。** 当前线上榜单private Full怎么加权、是否已变版，不能从早期调研或该kit旧note外推；我们没有private Full、官方名次或维护者指定硬件验收结果。

官方的完整输入校验和suite来源记录照常保留；本次文档没有额外跑模型／语料hash，也没有修改kit绕过其内置校验。

## 5. 三种“速度”不能混着比较

1. **正式public整套墙钟：** 原9h20m，新3h22m，实际整套收益2.765×。
2. **readout内部完整请求：** 同GPU3同125题；包含各自包装和安全检查，非同一正式模型。
3. **冻结feature后的head训练耗时：** 已缓存hidden，只训头；绝不是用户真实请求端到端延迟。

新的readout/Cptr不能直接拿内部94ms去对比正式16.6ms中位数，样本、起点、attention、精度和控制开销都不同。详见 [readout最终报告](../../../01-研究结论/06-五维训练与Readout/readout-study-20261108/FINAL-REPORT-20261110.md)。
