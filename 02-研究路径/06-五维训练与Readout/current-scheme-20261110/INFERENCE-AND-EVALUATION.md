# 推理、加速与计分

[返回总览](README.md)。本文描述已保存的两次正式 public 运行；文中提到的 `.py` 文件均在[库外辅助归档](SOURCE-ARCHIVE.md)。

## 结论

- **同一权重，加速 Engine 把 public 整套从 9h20m 缩短到 3h22m，约 2.8 倍；Public Index 从 24.97 到 24.99，基本不变。**
- **两版输出不是逐位一致。** choice 字段 99.2% 选择相同，单个概率最大相差 0.23。
- **43 项 benchmark 中有 6 项变慢，都是每请求多字段的任务。** 加速版逐字段打分，丢掉了原版的跨字段合批；ACOS 的耗时增至原来的约 2.2 倍。
- **距离 30 分钟的目标还差约 6.8 倍。** 加速版的耗时集中在 RouterBench 和 POP909，两项合计约占 42%。

## 1. 原 Engine：整个请求的候选一起合批，逐个打分后题内 softmax

源码为 `decision_engine.py` 的 `PairClassifierEngine`。

### 1.1 加载

1. 从 final 读取 tokenizer 和模型 config。
2. 用实际基座构建 `AutoModelForSequenceClassification`：BF16、`attn_implementation='sdpa'`、`trust_remote_code=False`。
3. 先执行 `base.score.float()` 再加载 PEFT，保住 FP32 的分类头。
4. 与 `adapter_model.safetensors` 逐 key、dtype、tensor 比较，不一致则拒绝运行。
5. `.to(cuda).eval()`，关闭 `use_cache`，读取 40,960 的上下文上限以及 pad、EOS。
6. 使用 4 个 CPU 线程，允许 TF32。模型只加载一次。

依赖版本为 Torch 2.9.1、Transformers 4.57.6、PEFT 0.18.1、NumPy 2.4.6。加速运行记录的[环境](../../../03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results/RUNTIME.json)是 NVIDIA H800 NVL（95,830 MiB）、驱动 590.44.01、Python 3.11.18、safetensors 0.8.0。

### 1.2 渲染与分词

- 按 `questions` 的原字段顺序和候选字典的原顺序处理。`noul` 没有显式 criteria 时使用 `{'true':'Yes.','false':'No.'}`。
- 模板和 `criteria_context` 的触发正则与[训练](DATA-AND-TRAINING.md)相同。prefix 和每个 candidate 各自分词，`add_special_tokens=False, truncation=False`，末尾加 EOS。
- CPU 上的 `lru_cache(maxsize=2048)` 只缓存文本对应的 token ID，不缓存预测结果。

### 1.3 合批

原版先收集一个 request 中所有字段的全部候选，再一次性调用 `score(rows)`，不同短字段的候选可以进入同一个 microbatch。

| 项目 | 设置 |
|---|---|
| 排序 | 按完整候选长度稳定排序，打分后恢复原顺序 |
| Microbatch | 最多 32 个候选，且不超过 32,768 padded token（宽度向上取 8 的倍数） |
| 单个超长候选 | 允许超出批预算，单独尝试 |
| Padding / mask | 右填充，显式 2D attention mask |
| 执行 | `torch.inference_mode()`，整个分类 forward 在 BF16 autocast 内 |
| 超过 40,960 token | 整个请求返回 `Unsupported`，不截断、不丢弃候选 |
| OOM | 释放并减半 token 预算后重试；单候选仍 OOM 则整个请求 `Unsupported` |

### 1.4 分类位置

HF 的 sequence-classification 路径先在每个 token 上做投影，再取最右的非 pad 位置。本方案在末尾附加的 EOS（151645）与 pad（151643）不同，因此取到的是 EOS 位置。

### 1.5 从 margin 到题内分布

对字段 f 的每个候选 i：

```text
m_i = float(logit_i[match]) - float(logit_i[not_match])
p_i = exp(m_i-max(m)) / sum_j exp(m_j-max(m))
```

三个容易混淆的量：

| 量 | 计算 | 用途 |
|---|---|---|
| pair 概率 | `sigmoid(m_i)` | 训练时的 BCE |
| 题内分布 | 本题全部候选的 `softmax(m)` | 正式输出 |
| Teacher 概率 | 每个 pair 独立的 `P(match)` | 训练目标 |

正式输出直接对 margin 做 softmax，不经过 sigmoid；softmax 的范围是一个字段的全部候选，不跨字段。

- `choice` 返回 argmax 对应的原候选 key 和全部候选的概率。
- `noul` 要求恰好两个候选，其中一个 key 为 true/yes/1，返回该候选的概率。两个候选的 softmax 等价于 `P(true) = sigmoid(m_true − m_false)`。
- 没有推理温度、截距、按候选数校准、旋转平均、投票或重试。

### 1.6 usage

```text
usage.input_tokens = sum_over_all_candidates(len(prefix)+len(candidate)+1 EOS)
```

usage 统计逻辑输入量，不含 padding。加速版共享 prefix 后实际计算的 token 减少，usage 仍按这个公式计。

## 2. 加速 Engine：同字段共享 prefix，固定形状走 CUDA Graph

源码为 `fast_pair_engine.py`、`prefix_graph_runtime.py`、`run_public.py`、`compare_results.py`。

权重、基座、未合并的 LoRA、分类头、分词边界、EOS、causal 语义、margin 和 softmax 公式都与原版相同。没有新训练，没有量化，也没有改 prompt。

### 2.1 LoRA 推理包装

`resolve_lora` 把 196 个 PEFT Linear 包装为 `InferenceLoRA`，提前解析唯一活跃的 default adapter，计算保持不变：

```text
result = base(x)
x = x.to(A.weight.dtype)
output = (result + B(A(x))*scale).to(result.dtype)
```

低秩分支仍以 FP32 单独计算，不 merge。遇到多个活跃 adapter、已 merge、非 vanilla 变体或非零 dropout 时拒绝运行。

### 2.2 同字段共享 prefix

候选数大于 1 且 `(N-1)*prefix_length >= 1024` 时走共享路径，否则走完整候选路径。步骤：

1. prefix 单独 forward 一次，把每层的 KV 写入固定 buffer。
2. 每个候选的后缀（正文 + EOS）分批接在同一个 prefix 之后。
3. 每一波用 `expand` 的只读视图广播 prefix KV，再与新后缀的 KV 拼接，共享的 prefix 不被修改。
4. 显式 4D causal mask 屏蔽 prefix 的物理填充、后缀 padding 和后缀的未来位置。
5. 后缀的 RoPE 位置为“真实 prefix 长度 + 后缀内位置”，与物理桶宽无关。
6. 按真实后缀长度取 EOS 位置，分类头和 autocast 策略不变。

缓存接口是自定义的 `PrefixWriter/PrefixFork`。buffer 只覆盖一个字段，下一个字段重写。同一请求的相邻问题之间、不同请求之间都不复用 KV，也没有输出缓存。

### 2.3 固定的 CUDA Graph 池

CUDA Graph 把固定形状的 CUDA 执行预先捕获下来，减少每次的 Python 和 launch 调度开销，计算内容不变。

| 路径 | 预定义形状 | 数量 |
|---|---|---:|
| full | batch = (2, 4, 8)，width = (64, 128, 256, 512) | 12 |
| prefix | width = (64, 128, 256, 512, 1024, 2048, 4096) | 7 |
| fork | physical prefix = (256, 1024, 4096)，batch = (8, 32)，suffix = (8, 32) | 12 |
| 合计 | 本次全部捕获 | 31 |

- 初始化时每个形状先执行 3 次再捕获一次，这些执行不属于评测样本。
- `graph_reserved_gib=32` 是开始捕获前检查 reserved 的阈值，不是显存硬上限。
- 捕获完成后形状池封存；没有命中的形状走 eager，请求期间不再捕获。
- 输入 buffer 固定，replay 的输出立即 clone，避免被下一波覆盖。
- autocast 设 `cache_enabled=False`。prefix 的 KV buffer 按声明上下文一次分配，随 runtime 存在，不会增长。

### 2.4 与原版的两处差别

- **没有跨字段合批。** 加速版逐字段调用 `score_field`。共享 prefix 减少了字段内的重复计算，但一个请求含很多短字段时，调用次数变多。
- **共享路径没有原版的 OOM 处理。** 共享 prefill 和 fork 阶段没有对半重试，也没有退回原 Engine 的 fallback。本次 public 全部成功；其他输入或显存压力下的表现没有验证。

## 3. 两次 public：分数基本不变，耗时 9h20m → 3h22m

两次都使用 upstream commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`，edition 0.3。套件共 140,620 个请求（原冻结套件 110,201 + 新增 30,419），固定排除 442 个后计分 140,178 个。展示 43 个 benchmark，其中 37 个计入 Index。

| 指标 | 原 Engine | 加速 Engine |
|---|---:|---:|
| Public Index | 24.97 | 24.99 |
| Raw Index | 44.2 | 44.2 |
| Knowledge Skill | 9.2 | 9.3 |
| Language Skill | 32.4 | 32.4 |
| Retrieval Skill | 36.1 | 36.1 |
| Tools Skill | 31.9 | 31.9 |
| Arts Skill | 11.4 | 11.4 |
| 执行请求 / 成功 | 140,620 / 140,620 | 140,620 / 140,620 |
| 计分请求 / 成功 | 140,178 / 140,178 | 140,178 / 140,178 |
| 推理耗时 | 9h20m | 3h22m |
| 请求 median | 70.5 ms | 16.6 ms |
| 请求 mean | 238.7 ms | 86.2 ms |
| 请求 p95 | 676.5 ms | 328.4 ms |

推理耗时不含模型加载。原版加载模型约 7.6 秒；加速版加载加初始化约 15.0 秒，其中 Graph 捕获约 6.1 秒。加速版 pipeline 含加载、初始化和计分的总耗时约 3h24m。

原生分数：[原版 scores](../../../03-证据与图表/06-五维训练与Readout/joint-generic-continuation-20261109/scores.json)、[加速版 scores](../../../03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results/scores.json)。耗时说明：[原版](../../../01-研究结论/05-Attention与效率/joint-generic-continuation-20261109/OFFICIAL-TIMING-EXPLAINED-20261110.md)、[加速版](../../../01-研究结论/05-Attention与效率/inference-prefix-graphs-20261110/README.md)。

### 3.1 两版输出有 0.8% 的选择不同

两次输出按 run_id、payload、字段、候选逐项对齐，逻辑输入 token 完全相同。

| 对比项 | 数量 | 差异 |
|---|---:|---|
| choice 字段 | 308,559 | 2,575 个选择改变，一致率 99.2% |
| noul 字段 | 14,700 | 82 个跨过 0.5 阈值 |
| 概率值 | 2,506,535 | 平均绝对差 0.001，最大 0.23 |

两版的评分公式相同，但 batch、mask、cache 和 kernel 路径不同，BF16 下的数值不保证一致。这些差异没有逐项归因，最大 0.23 的概率差不能只当作舍入误差。前 500 个请求两版完全一致，这只在该短题子集上成立。

总分相差 0.02，两版的整体得分接近，没有体现出能力变化。

### 3.2 长任务大幅提速，多字段短请求变慢

| Benchmark | 每请求字段数 | 原 Engine | 加速 Engine | 原 / 加速 | 占加速版请求耗时 |
|---|---:|---:|---:|---:|---:|
| RouterBench | 2.0 | 119 分钟 | 52 分钟 | 2.3× | 26% |
| POP909 | 1.0 | 139 分钟 | 33 分钟 | 4.2× | 16% |
| API-Bank | 1.0 | 68 分钟 | 5.5 分钟 | 12.4× | 3% |
| ACOS | 58.2 | 7.4 分钟 | 16.1 分钟 | 0.46×（变慢） | 8% |
| SATA-Bench | 9.4 | 6.1 分钟 | 9.1 分钟 | 0.67×（变慢） | 4% |
| BRIGHT | 32.0 | 5.7 分钟 | 6.9 分钟 | 0.83×（变慢） | 3% |

原版每个候选都重算一遍 prefix，这是原先耗时的主要来源，共享 prefix 后 API-Bank 快了约 12 倍。变慢的 6 项（上表 3 项，加 PhishNChips、Home appliance simulator、BFCL）都是每请求多字段的任务；ToolRet、ContractNLI 等多字段任务仍然变快。

整套加速约 2.8 倍。要达到 30 分钟，还需要在加速版基础上再快约 6.8 倍。本轮没有做 profiler 分析，GPU 与 CPU 的耗时占比未知。

## 4. Public Index：原生指标 → Skill → area 加权

每个 benchmark 先算自己的原生指标并做 coverage 调整，再按 chance 和 track 规则换算成 Skill，常见形式为：

```text
skill = clip((score - chance)/(1 - chance), 0, 1)
```

原生指标包括 accuracy、F1、相关性和整 case 准确率等。area 内 gold 任务的权重为 1.2，其余为 1.0；area 之间按固定权重合成：

| Area | 权重 |
|---|---:|
| Knowledge | 0.258494 |
| Language | 0.258494 |
| Retrieval | 0.200229 |
| Tools | 0.182783 |
| Arts | 0.1 |

```text
Index = 100 * sum(weight_area * skill_area)
```

两点直接影响对分数的理解：

- 单个小任务的原生指标提高 10 个百分点，经 area 内平均和 area 权重后，对总分的贡献远小于 10 分。
- 接近随机水平的任务，原生指标看起来有 50%，Skill 可能接近 0。

24.99 是固定 public 0.3 套件上的结果。目前没有 private Full 分数、官方名次或指定硬件上的验收结果。

## 5. 三种速度口径各自独立

| 口径 | 数值 | 测量范围 |
|---|---|---|
| 正式 public 整套 | 9h20m → 3h22m，约 2.8 倍 | joint final，140,620 个请求的推理耗时 |
| readout 内部完整请求 | median 约 94 ms | 原标签 stage2 起点，GPU3 上 125 题，含各自的包装和安全检查 |
| 冻结 feature 后的 head 训练 | — | hidden 已缓存，只训 head，不是请求延迟 |

readout 的 94 ms 与正式加速版的 16.6 ms median 来自不同的模型起点、样本、attention 实现和精度设置，两者不能直接比较。详见 [readout 最终报告](../../../01-研究结论/06-五维训练与Readout/readout-study-20261108/FINAL-REPORT-20261110.md)。
