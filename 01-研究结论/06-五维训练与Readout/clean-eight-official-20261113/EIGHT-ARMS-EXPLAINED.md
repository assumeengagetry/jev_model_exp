# 八个实验分支：名字、改动和结果

> 2026-11-15 按学习用途重写。实验于 2026-11-12～13 完成，原版整理于 2026-11-14（服务器副本 `/home/dev/assumeengage/reports/jev-eight-arms-explained-20261114.md`）。
>
> 官方数字已与[证据目录](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/README.md)里的 `SCORES.json`、`scores/*.json` 和两张 CSV 逐项核对，保留官方文件的两位小数。本地开发集和训练记录来自服务器上的汇总文件，仓库里没有原件，取一位小数。
>
> 数据、训练流程、评测规则、推理后端和四卡调度见[完整学习报告](REPORT.md)。

## 先看结论

- **八个分支从同一个起点出发：**原 Qwen3-0.6B，加上 seed 42 新初始化的 LoRA 和二分类 head；训练数据、曝光顺序和预算完全相同。它们是并列的对照组，不是一个接一个改进出来的版本。
- **名字编码三件事：**attention 可见性、训练时的 token 布局、训练目标。`STATE_NL` 另外移动了 prefix 与 suffix 的分界。
- **官方成绩分成两档。**保留 causal 的五个分支在 12.83～14.42；改成非 causal 的三个分支（BI、PB、STATE_NL）只有 1.97～3.09。
- **本地开发集的排名不能预测官方排名。**官方最高的 `C_P_BCE`（14.42）在开发集上只排第四。

<a id="names"></a>
## 1. 名字怎么读

```text
C _ P _ BCE
│   │   └─ 训练目标：BCE / CE / BCE_RANK
│   └───── 训练时的 token 布局：D = dense（补 padding）/ P = packed（只有真实 token）
└───────── attention：C = causal / BI = 全双向 / PB = 两块双向
```

| 缩写 | 含义 | 容易误解成 |
|---|---|---|
| `C` | causal：每个 token 只看自己和前面 | — |
| `D` | dense：训练时张量按批内最长补 padding | double 精度、蒸馏、全参数训练 |
| 中间的 `P` | packed：真实 token 首尾相接，不补 padding | prefix、prefix cache |
| `BI` | bidirectional：一条 pair 内全双向 | bi-encoder 双塔 |
| `PB` | prefix-block：分成 prefix、suffix 两块，两块内部都双向 | 常见的 prefix-LM（那里 suffix 仍是 causal） |
| `BCE` | 每个候选单独做二元交叉熵 | — |
| `CE` | 同一个 field 的全部候选一起做交叉熵（field-CE） | 把二分类 head 换成 K 类 head |
| `RANK` | 额外加一项排序损失 | LoRA 的 rank（仍是 32） |
| `STATE_NL` | PB 的分界移到 state 后面那个原有的换行（newline）之后 | 新增一层或新 token |
| `JPAIR` | 借鉴 Jina 的排序目标，但保留独立 pair | 完整的 Jina reranker |
| `†` | 结果表的脚注：dense 训练、packed 评测，迁移时有数值偏差（第 9 节） | 配置名的一部分 |

本文出现三种 "P"：名字中间的 P 是 packed；attention 公式里的 P 是 prefix 块；排序损失里的正例集合写作 `I+`，以免混淆。

<a id="overview"></a>
## 2. 八个分支一览

| 分支 | attention | 训练布局 | 训练目标 | 本地开发 mass（100M） | 官方 Index |
|---|---|---|---|---:|---:|
| `C_P_BCE` | causal | packed | BCE | 57.6% | **14.42** |
| `JPAIR_P_BCE_RANK` | causal | packed | BCE＋排序 | 61.2% | 13.60 |
| `C_D_CE` † | causal | dense | field-CE | 61.7% | 13.07 |
| `C_P_CE` | causal | packed | field-CE | 59.6% | 12.95 |
| `C_D_BCE` † | causal | dense | BCE | 56.4% | 12.83 |
| `BI_P_CE` | 全双向 | packed | field-CE | 48.2% | 3.09 |
| `PB_P_CE` | 两块双向，分界在 `Candidate:` 后 | packed | field-CE | 43.2% | 2.59 |
| `STATE_NL_PB_P_CE` | 两块双向，分界在 state 后 | packed | field-CE | 44.4% | 1.97 |

官方评测时八个分支**都用 packed 推理**，名字里的 D/P 只表示训练时的布局。两个 D 分支的开发集数字是在 dense 推理下测的。本地 mass 和官方 Index 是两种不同的算法，见报告[第 8 节](REPORT.md#dev)和[第 10 节](REPORT.md#index)。

八个分支的关系：

```text
原 Qwen3-0.6B ＋ seed 42 新建的 LoRA/head ＋ 同一套数据和曝光顺序
  ├─ causal / dense  / BCE        → C_D_BCE
  ├─ causal / packed / BCE        → C_P_BCE
  ├─ causal / dense  / field-CE   → C_D_CE
  ├─ causal / packed / field-CE   → C_P_CE
  ├─ 全双向 / packed / field-CE   → BI_P_CE
  ├─ PB（Candidate 分界）/ packed / field-CE → PB_P_CE
  ├─ PB（state 分界）    / packed / field-CE → STATE_NL_PB_P_CE
  └─ causal / packed / BCE＋排序  → JPAIR_P_BCE_RANK
```

每条分支都从头训练完整预算。BI、PB 没有拿训练好的 causal 权重直接改 mask；后加的 STATE_NL 和 JPAIR 也先从原模型重新生成初始权重，再把 393 个初始张量和前六个分支逐字节核对。

前四个分支组成"布局 × 目标"的 2×2 对照；BI、PB 以 `C_P_CE` 为控制组研究 mask；STATE_NL 和 JPAIR 是看到 10M、30M 开发集结果后追加的探索分支，不是事先计划的。

<a id="comparisons"></a>
### 每个问题该和谁比

下表的差值都是"后者 − 前者"的官方 Index 点数。

| 想回答的问题 | 正确的对照 | 官方 Index 差 | 常见的错误比较 |
|---|---|---:|---|
| packing 有没有影响 | `C_D_BCE` → `C_P_BCE`；`C_D_CE` → `C_P_CE` | +1.59；−0.12 | packed BCE 对 dense CE（目标也变了） |
| BCE 还是 field-CE | `C_D_BCE` → `C_D_CE`；`C_P_BCE` → `C_P_CE` | +0.24；−1.47 | 拿不同布局的两组比 |
| 双向 mask 有没有用 | `C_P_CE` → `BI_P_CE`；`C_P_CE` → `PB_P_CE` | −9.86；−10.36 | BI 对 causal BCE（loss 也变了） |
| state 分界有没有用 | `PB_P_CE` → `STATE_NL_PB_P_CE` | −0.62 | STATE_NL 对 C_P_BCE（三处都变了） |
| 排序损失有没有用 | `C_P_BCE` → `JPAIR_P_BCE_RANK` | −0.82 | 只和 C_P_CE 比（基线 loss 不同） |
| 数据清理是否导致掉分 | 无法单独回答 | — | 新的 14.42 对旧谱系 24.91（起点、数据、预算、后端都不同） |

两个注意点：

- packing 那一行的 D 分支带 †，它们的官方分数是在 packed 推理下测的（第 9 节），所以不是"dense 训练 + dense 推理"对"packed 训练 + packed 推理"的干净对照。
- 只有 seed 42 一次训练，不知道 1～2 点的差距换个 seed 是否还在。causal 与非 causal 之间约 10 点的差距，比其它所有差距都大得多。

<a id="shared-model"></a>
## 3. 共同起点：模型结构

### 3.1 从语言模型变成候选分类器

共同起点是本地的原 ModelScope `Qwen3-0.6B`，没有另下载 Jina 或 Base 的模型，也没有加载项目历史上训练过的 adapter。构造方式的节选如下（以 `fresh_init.py` 为准）：

```python
config.num_labels = 2
config.id2label = {0: 'not_match', 1: 'match'}
config.pad_token_id = 151643
config.eos_token_id = 151645
config.use_cache = False

# 保留原 Qwen3 backbone，套上序列分类接口
model = AutoModelForSequenceClassification.from_pretrained(STOCK, config=config, dtype=torch.bfloat16, ...)

# 丢掉 loader 临时建的 score，换成八个分支共用的新二分类 head
model.score = Linear(1024, 2, bias=False, dtype=float32)
normal_init(model.score.weight, mean=0, std=0.02)

model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0,
    target_modules=[q, k, v, o, gate, up, down],
    modules_to_save=['score'], task_type='SEQ_CLS', bias='none'))
```

实际前向不靠 HF 的 `attn_implementation` 字符串，而是由共享的 `MaskedBackbone` 调用原 Qwen3/PEFT 模块和 native FA3，mask 也在这里决定。

### 3.2 结构

| 项目 | 八个分支共同的值 |
|---|---|
| backbone | 原 Qwen3-0.6B，28 层，全是 softmax attention |
| hidden / MLP intermediate | 1024 / 3072 |
| Q heads / KV heads / head_dim | 16 / 8 / 128（GQA） |
| attention 实现 | 冻结的 native FA3 |
| LoRA | r=32，alpha=64，缩放 2，dropout 0，七种投影每层都加 |
| 分类 head | 一个无 bias 的 `Linear(1024, 2)`，只有一层 |
| 可训练参数 | 20,187,136（196 对 LoRA A/B ＋ 1 个 head ＝ 393 个张量） |
| 推理时实际用到的参数 | 616,237,056，其中冻结的原 backbone 596,049,920 |

物理上注册的参数还多 2048 个，是 loader 留下、推理时不用的原 score 副本。改 mask、改分界、改 loss 都不增加参数，所以八个分支的参数量完全相同。

### 3.3 LoRA 怎样改每个投影

$$W_{eff}=W_0+\frac{\alpha}{r}BA=W_0+2BA.$$

按 PyTorch 的 `[out, in]` 约定，A 是 `[r, in]`，B 是 `[out, r]`。Q 的输出维度是 16×128 = 2048，比 hidden 1024 大，所以各投影形状不同：

| 投影 | 原权重 `[out,in]` | LoRA A / B | 每层新增参数 |
|---|---|---|---:|
| q | `[2048,1024]` | `[32,1024]` / `[2048,32]` | 98,304 |
| k、v | `[1024,1024]` | `[32,1024]` / `[1024,32]` | 各 65,536 |
| o | `[1024,2048]` | `[32,2048]` / `[1024,32]` | 98,304 |
| gate、up | `[3072,1024]` | `[32,1024]` / `[3072,32]` | 各 131,072 |
| down | `[1024,3072]` | `[32,3072]` / `[1024,32]` | 131,072 |

每层合计 720,896，28 层是 20,185,088，加上 head 的 2,048，正好 20,187,136。B 初始化为零，所以训练开始时 LoRA 不改变任何输出；head 是新的随机初值。可训练参数只有约 20.2M，但前向仍要走完整个 backbone，反向也要沿每层传回到各个 LoRA。

### 3.4 精度：存储、计算、读出各不相同

- 原 backbone 参数以 BF16 加载并冻结；LoRA 和 head 以 FP32 保存和训练。
- 前向在 BF16 autocast 下运行，所以 head 虽然是 FP32 参数，矩阵乘法走的是 BF16。
- head 先作用在所有 token 行上，再取出 EOS 位置的 logits。先挑 EOS 再乘 head 在数学上相同，但不同形状的矩阵乘法舍入可能不同，所以沿用冻结时的做法。
- logits 转 FP32 后相减得到 margin；字段内归一化用双精度。

### 3.5 训练条件也一样

| 项目 | 八个分支共同的设置 |
|---|---|
| 训练集 | 24,600 个完整 case，41,674 个有标签的 field |
| 曝光量 | 每个分支 99.8M 逻辑 pair token，56,912 次 case 出现，3,557 次 optimizer step |
| 优化器 | fused AdamW，峰值 LR 2e-5，weight decay 0.01，梯度 clip 1.0，107 步 warmup 后 cosine 降到 0 |
| 每次更新 | 16 个完整 case；按显存切成若干 micro，两遍前向（第一遍算 margin，第二遍反传） |
| 硬件 | 每个分支单独用一张 GPU，不是多卡 DDP |
| 终点 | 最后一个 checkpoint（第 3557 步），没有按官方分数挑 checkpoint |

为什么要两遍前向、数据怎样清理，见报告[第 6、7 节](REPORT.md#data)。起点相同不代表终点相同：loss 或 mask 一变，梯度和权重的轨迹就分开了。

<a id="readout"></a>
## 4. 输入和读出

### 4.1 case、field、candidate、pair

下面是教学例子，不是训练样本：

```text
case：一个客户请求
  field 1：应该转给哪个部门？
    candidate A：退款
    candidate B：物流
    candidate C：技术支持
  field 2：要不要升级处理？
    candidate Yes
    candidate No
```

每个候选单独构成一条完整的 pair：上面这个 case 是 5 条 pair，每条都包含完整的题目和一个候选。不同 pair 在 attention 里互相看不见。

### 4.2 模板

```text
Decide whether the candidate is a correct match.
Input:
{state}
Question:
{question}
Available criteria:
{按原规则需要时才出现的完整候选列表}
Candidate:
{当前 candidate}
<EOS>
```

`Question` 和 `Available criteria` 按原条件出现。遇到 "all/none of the above" 这类引用其它选项的写法，或候选不多的相对比较题，模板会按原顺序列出全部候选；所以独立 pair 的文本里有时能看到其它选项的文字，但不同 pair 的 hidden state 之间没有任何连接。

prefix 和 candidate 分别 tokenize，不加特殊 token、不截断；candidate 后面追加真实 EOS（151645），pad 是 151643。八个分支使用完全相同的模板和 token。

### 4.3 两个 logit 怎样变成多个候选的概率

每条 pair 在 EOS 位置得到两个 logit：$z_{k,0}$（不匹配）和 $z_{k,1}$（匹配）。取差得到 margin：

$$m_k=z_{k,1}-z_{k,0}.$$

field 有 K 个候选，就得到 K 个 margin，在这 K 个之间做 softmax：

$$p_k=\frac{e^{m_k-\max_jm_j}}{\sum_je^{m_j-\max_jm_j}}.$$

所以 `num_labels=2` 不限制候选数量，255 个候选的 field 也用同一个 head。

sigmoid 和字段 softmax 是两回事。以 margin `[2, 0, −1]` 为例：

| 候选 | sigmoid(m) | 字段 softmax(m) |
|---|---:|---:|
| A | 0.88 | 0.84 |
| B | 0.50 | 0.11 |
| C | 0.27 | 0.04 |
| 合计 | 1.65 | 1.00 |

sigmoid 是每个候选各自的"匹配概率"，加起来不等于 1；字段 softmax 才是候选之间的分布。**八个分支推理时都输出字段 softmax**，BCE 分支也不例外。

<a id="layouts"></a>
## 5. 第一条轴：D 和 P（token 布局）

### 5.1 Dense

三条 pair 长度分别为 4、2、5，dense 宽度向 8 对齐：

```text
ids[3, 8]
A A A A _ _ _ _
B B _ _ _ _ _ _
C C C C C _ _ _
```

Embedding、Norm、QKV/LoRA、MLP、head 这些逐 token 的模块要处理 24 行，其中 13 行是 padding。attention 这一步会先取出真实 token，再用 FA3 的 varlen 接口按序列隔开，所以 **padding 从不作为 key 被看到**；D 只意味着其它模块在 padding 上做了无用计算。

### 5.2 Packed

```text
ids = A A A A | B B | C C C C C     共 11 个真实 token
cu  = [0, 4, 6, 11]                 每条序列的边界
pos = [0 1 2 3 | 0 1 | 0 1 2 3 4]   每条序列内各自从 0 开始
```

hidden 是 `[11, 1024]`，所有模块只算 11 行。`cu` 决定 attention 的边界，`pos` 决定 RoPE 位置；第二条 pair 的位置要重新从 0 开始，不能接着上一条数下去。

EOS 的物理下标也随布局变：dense 展平后是 `[3, 9, 20]`，packed 是 `cu[1:] − 1 = [3, 5, 10]`。两者必须指向同一条 pair 的同一个 EOS。

### 5.3 Packing 不会让候选互相看见

只要 `cu` 正确，A 看不到 B、C。错误的做法是把所有 token 拼成一条后套一个全局下三角 mask，那样后面的序列会看到前面无关的题。

### 5.4 两种布局用同一份 micro 规划

两种布局都按相同规则切 micro：按长度稳定排序、每个 micro 最多 32 条 pair、按 dense 补齐后的大小算 16,384 token 的预算。packed 只是执行时不补 padding。这样做是为了只改变布局这一个变量；如果 packed 同时换成更大的 micro，就混进了另一个变化。

结果：两遍前向合计，逐 token 模块要处理的行数 dense 为 266.5M，packed 为 199.6M，少了约 25%。行数少不等于同比例的提速，更不等于提分（报告[第 3 节](REPORT.md#padding-free)有旧权重上的同权重 A/B）。

<a id="masks"></a>
## 6. 第二条轴：C、BI、PB（attention 可见性）

### 6.1 记号

attention 对每个 query $i$ 计算：

$$A_{ij}=\frac{\exp(q_i^Tk_j/\sqrt{128}+M_{ij})}{\sum_t\exp(q_i^Tk_t/\sqrt{128}+M_{it})},$$

允许时 $M_{ij}=0$，禁止时视为 $-\infty$。FA3 不会真的分配这张 T×T 的矩阵，公式只是描述可见性。三种模式都先限制在同一条 pair 内。

### 6.2 三种 mask

以 P（prefix）、S（suffix）各 2 个 token 为例，行是 query，列是 key，1 表示可见：

```text
causal          全双向 BI       两块双向 PB
1 0 | 0 0       1 1 | 1 1       1 1 | 0 0
1 1 | 0 0       1 1 | 1 1       1 1 | 0 0
----+----       ----+----       ----+----
1 1 | 1 0       1 1 | 1 1       1 1 | 1 1
1 1 | 1 1       1 1 | 1 1       1 1 | 1 1
```

- **causal**：$j\le i$ 才可见。实现上给 FA3 传 `causal=True`。
- **BI**：pair 内所有位置互相可见。state 开头的 token 也能看到后面的问题和候选。
- **PB**：P 内部双向，P 看不到 S；S 看完整的 P 和 S，而且 **S 内部也是双向**，这一点和常见的 prefix-LM 不同。

### 6.3 EOS 在 causal 下本来就看得到整条 pair，为什么结果还会不同

EOS 在最后，causal 下第一层的 EOS 已经能看到全部前文。改 mask 改变的是其它 token 的表示：BI 下 state 里的 token 在第一层就读到了问题和候选，表示和 causal 不同；从第二层起，EOS 读到的 K/V 就是这些已经不同的表示。所以多层之后输出不同。

### 6.4 PB 怎样在 FA3 上实现

每层只算一次 Q/K/V 投影，然后按 query 分两次调用：

```python
prefix_out = FA3(Q_P, K_P,         V_P,         causal=False)
suffix_out = FA3(Q_S, K_full_pair, V_full_pair, causal=False)
out = scatter_to_original_positions(prefix_out, suffix_out)
```

第二次调用对 S 的每个 query 在完整 P+S 上做**一次** softmax，不能分别算 S→P 和 S→S 再相加（两次 softmax 相加不等于一次 softmax）。

causal 和 BI 每个 micro 是 28 次 attention 调用，PB 是 56 次；QKV、MLP 只算一遍，所以总计算量远小于两倍。官方评测里 PB 两个分支平均每条请求比 causal 慢约 22%（报告[第 13 节](REPORT.md#results)）。

<a id="losses"></a>
## 7. 第三条轴：BCE、CE、RANK（训练目标）

### 7.1 统一的分母

一次更新有 C = 16 个完整 case；case $c$ 有 $F_c$ 个有标签（known）的 field；field $f$ 有 $K_f$ 个候选，目标是 $q_k\in[0,1]$。三种目标都这样平均：

$$L=\frac1C\sum_c\frac1{F_c}\sum_{f\in known(c)}L_{cf}.$$

先在 field 内算、再对 field 平均、再对 case 平均，于是每个 case 权重相同，字段多的 case 不会支配目标。没有标签的 field 不当负例，也不进分母。

### 7.2 BCE：每个候选单独做二分类

$$B_f=-\frac1{K_f}\sum_k\left[q_k\log\sigma(m_k)+(1-q_k)\log(1-\sigma(m_k))\right]$$

```python
F.binary_cross_entropy_with_logits(m, q, reduction='mean')   # 没有 pos_weight
```

$$\frac{\partial B_f}{\partial m_k}=\frac{\sigma(m_k)-q_k}{K_f}.$$

BCE 让 $\sigma(m_k)$ 接近 $q_k$，不是让 $m_k$ 接近 $q_k$。软标签（$0<q<1$）的最优 margin 是 $\log\frac{q}{1-q}$；硬标签 1 或 0 要把 margin 推向正负无穷，实际被共享参数、weight decay 和 clip 限制住。

### 7.3 field-CE：同一 field 的候选互相竞争

$$CE_f=-\sum_kq_k\log p_k,\qquad p_k=\operatorname{softmax}(m)_k$$

```python
-(q * torch.log_softmax(m, dim=0)).sum()    # dim=0 是这个 field 的全部候选
```

$$\frac{\partial CE_f}{\partial m_j}=p_j\sum_kq_k-q_j.$$

目标和为 1 时就是熟悉的 $p_j-q_j$。q 不在 loss 里重新归一化；和 BCE 不同，这里没有除以 $K_f$。

### 7.4 BCE 和 CE 学的东西不一样

- 给同一 field 的所有 margin 加一个常数，softmax 和 CE 都不变，BCE 会变。所以 CE 只约束候选之间的相对大小，BCE 还约束每个候选 margin 的绝对位置。
- BCE 的最优解经过字段 softmax 后，一般不等于目标分布。例：$q=[0.7,0.3]$，BCE 最优 margin 约为 $[0.85,-0.85]$，softmax 后是 $[0.84,0.16]$，不是 $[0.7,0.3]$；CE 的最优解是 $m=\log q+c$，softmax 后正好是 q。

这只说明两种目标的概率空间不同。本轮官方结果反而是 packed BCE 高于 packed CE。

### 7.5 RANK：正例集合的排序损失

按原始 Python 的 q 值（不是转成 FP32 之后的值）分出：

```text
I+ = {k : q_k == 1}     正例
I− = {k : q_k == 0}     负例
H  = I+ ∪ I−            软标签不在 H 里
```

$$R_f=\begin{cases}\log\sum_{k\in H}e^{m_k}-\log\sum_{k\in I+}e^{m_k}, & I+\text{ 和 }I-\text{ 都非空}\\ 0, & \text{否则}\end{cases}$$

$$L_{JPAIR}=\frac1C\sum_c\frac1{F_c}\sum_f\left(B_f+R_f\right),\qquad \lambda=1,\ \tau=1.$$

- 它衡量"正例集合在所有硬标签候选里占多少概率"，不是把每个正例和每个负例两两比较的 RankNet。
- 严格 one-hot 且 K ≥ 2 时，$R_f=\log\sum_ke^{m_k}-m_{positive}$，正好等于 field-CE。所以在硬标签字段上 JPAIR 就是 BCE＋CE；纯软标签字段上 rank 为 0，只剩 BCE。
- 按原始值判断 0 和 1，是为了防止 0.9999999 这类软标签转成 FP32 后被当成正例。

### 7.6 三种目标的梯度大小

教学例子：K = 4，$q=[1,0,0,0]$，所有 margin 为 0，还没乘外层分母：

| 目标 | 正例的 dL/dm | 每个负例的 dL/dm |
|---|---:|---:|
| BCE | −0.125 | +0.125 |
| CE | −0.750 | +0.250 |
| BCE＋rank | −0.875 | +0.375 |

加 rank 不只是"多了排序信息"，正负例的相对权重和梯度大小也一起变了，进而影响 clip 和 Adam 的轨迹。

### 7.7 loss 里的竞争不是 attention 里的交互

```text
pair A ──独立前向── m_A ─┐
pair B ──独立前向── m_B ─┼─ 同一个 field 的 loss
pair C ──独立前向── m_C ─┘
```

CE 和 RANK 让一个候选的梯度依赖其它候选的分数，但算 $m_A$ 的那次前向完全看不到 B、C。head 也还是 `Linear(1024, 2)`，没有换成 K 类。

<a id="arms"></a>
## 8. 逐个分支

<a id="arm-c-p-bce"></a>
### `C_P_BCE`：本轮最高分

- **配置**：`{"mode": "causal", "layout": "packed", "loss": "BCE"}`
- **作用**：packed 下的 BCE 基线，也是 JPAIR 的控制组。相对 `C_D_BCE` 只改布局，相对 `C_P_CE` 只改 loss。
- **结果**：本地开发集 57.6%，八个中第四；官方 **14.42**，第一。比 `C_P_CE` 高 1.47，比 JPAIR 高 0.82。领先 `C_P_CE` 的部分大多来自 Tools 区域的 BFCL（报告[第 13.3 节](REPORT.md#tools-gap)）。

<a id="arm-jpair"></a>
### `JPAIR_P_BCE_RANK`：BCE 加排序损失

- **配置**：`{"mode": "causal", "layout": "packed", "loss": "BCE_RANK_HARD"}`。仍是 causal、仍是独立 pair，LoRA rank 仍是 32。
- **相对 `C_P_BCE` 只改了 loss**：

```python
positive = [i for i, value in enumerate(original_host_q) if value == 1]
negative = [i for i, value in enumerate(original_host_q) if value == 0]
bce = binary_cross_entropy_with_logits(m, q, reduction='mean')
if positive and negative:
    rank = logsumexp(m[positive + negative]) - logsumexp(m[positive])
else:
    rank = (m * 0.0).sum()      # 保持计算图连接
field_loss = bce + rank
```

- **细节**：软标签只进 BCE；没有标签的 field 不进 rank；没有正例或没有负例时 rank 为 0，但这个 field 仍算在分母里。候选顺序、pair、EOS、micro 划分都和 `C_P_BCE` 相同，也没有按标签把正例挪到前面。
- **训练中的观察**：rank 项在约 91% 的有标签 field 上生效；JPAIR 和它的对照组在所有记录里都触发了梯度 clip；JPAIR 的总 margin 系数与其中 BCE 部分的 L2 比平均约 3.3。这是 margin 层的系数，不是参数梯度。它说明加 rank 后梯度的方向和大小都变了，所以结果不能单独归因于排序信息。
- **和 Jina 的关系**：只借了排序目标的思路。完整的 Jina v3 还有多文档同窗、query 出现两次、projector 和 cosine 打分，这里都没有。另做的 J0/J1/J2 机制接线（报告[第 5 节](REPORT.md#jina)）不是这个分支。
- **结果**：本地开发集 61.2%，比 `C_P_BCE` 高 3.7 个百分点；官方 13.60，反而比 `C_P_BCE` 低 0.82。它比 `C_P_CE` 高 0.65，但正确的对照是 `C_P_BCE`。

<a id="arm-c-p-ce"></a>
### `C_P_CE`：mask 研究的控制组

- **配置**：`{"mode": "causal", "layout": "packed", "loss": "CE"}`
- **作用**：BI、PB 用 field-CE，所以它们的控制组也必须用 field-CE；拿 causal BCE 去比，差距里就混进了 loss 的变化。它相对 `C_P_BCE` 只改 loss，相对 `C_D_CE` 只改布局。
- **结果**：本地开发集 59.6%；官方 12.95。四卡评测时它第一个跑完（1 小时 43 分），这只是排队顺序。

<a id="arm-c-d-ce"></a>
### `C_D_CE` †：dense 训练的 field-CE

- **配置**：`{"mode": "causal", "layout": "dense", "loss": "CE"}`
- **作用**：相对 `C_D_BCE` 只改 loss，相对 `C_P_CE` 只改训练布局。dense 训练时 attention 仍然只看真实 token。
- **结果**：本地开发集 61.7%（八个中最高，dense 推理下测得）；官方 13.07（packed 推理）。同一权重换成 packed 推理后，375 个开发 case 中有 34 个的数值超出容差，见第 9 节。

<a id="arm-c-d-bce"></a>
### `C_D_BCE` †：dense 训练的 BCE

- **配置**：`{"mode": "causal", "layout": "dense", "loss": "BCE"}`
- **作用**：补齐 2×2 对照。没有它，就只能拿 packed BCE 对 dense CE，布局和 loss 同时在变。

| 训练布局 | BCE | field-CE |
|---|---:|---:|
| dense | `C_D_BCE` 12.83 † | `C_D_CE` 13.07 † |
| packed | `C_P_BCE` 14.42 | `C_P_CE` 12.95 |

- **结果**：本地开发集 56.4%（dense 推理），同一权重改用 packed 推理后 56.3%，这是执行路径不同带来的差，不是又学了一轮；官方 12.83。

<a id="arm-bi"></a>
### `BI_P_CE`：pair 内全双向

- **配置**：`{"mode": "bidirectional", "layout": "packed", "loss": "CE"}`。相对 `C_P_CE` 只改可见性。
- **代码**：

```python
native_fa3(..., cu_q=batch.cu, cu_k=batch.cu,
           causal=(mode == 'causal'), deterministic=True)   # BI 时传 causal=False，每层一次
```

- **新增的可见关系**：state 的 token 能看到后面的问题和当前候选；问题能看到候选；一条 pair 里所有位置互相可见。不同候选仍是不同的 pair，没有变成"所有候选一起编码"。
- **训练方式**：从共同的初始权重按双向完整训练了同样的预算，不是把训练好的 causal 模型直接切换开关。
- **结果**：本地开发集 48.2%，比 `C_P_CE` 低 11.4 个百分点；官方 3.09，低 9.86。

<a id="arm-pb"></a>
### `PB_P_CE`：两块双向，分界在 `Candidate:` 之后

- **配置**：`{"mode": "prefix_block", "layout": "packed", "loss": "CE"}`。第一个 PB 是 attention 模式，第二个 P 是 packed。
- **两块的内容**：

```text
P = 固定说明 + Input/state + Question（若有）+ Available criteria（若有）+ "Candidate:\n"
S = 当前候选 + EOS
```

  P 里已经包含了问题，这和 Base 原文的分法（P 只有 state，问题和选项在后块）不同。这里沿用了你给的 commit `2e334fb` 的分界，先在相同输入上只比较 mask。
- **实现**：`bind_mask` 根据每条 pair 的 prefix 长度生成 P、S 的下标和各自的 `cu`，28 层共用；每层 Q_P 读 K_P/V_P，Q_S 读整条 pair 的 K/V，再用 `index_copy` 写回原位置。训练时 prefix 的计算图没有被截断。
- **结果**：本地开发集 43.2%；官方 2.59，比 `C_P_CE` 低 10.36。这不是 Base 完整配方的测试（没有 Qwen3.5 混合结构、answer-slot LM head、全参训练和 CE+Brier）。

<a id="arm-state"></a>
### `STATE_NL_PB_P_CE`：只移动分界，不改任何 token

- **配置**：同 `PB_P_CE`，另外绑定分界文件和规则 `state-plus-one-existing-newline-v2`。NL 指模板里 state 后面原本就有的那个换行，没有新增 token，也没有删掉问题。
- **改了什么**：

```text
PB_P_CE：        [说明 + state + Question + criteria + "Candidate:"]P   [候选 + EOS]S
STATE_NL_PB_P_CE：[说明 + state + 一个原有换行]P   [Question + criteria + "Candidate:" + 候选 + EOS]S
```

  问题和 criteria 现在属于 S，可以双向读到当前候选；在 PB 里它们属于 P，读不到候选。完整的 `input_ids` 一个都没变，但可见性变了，所以这是真实的模型变化。
- **实现是无损重切，不是重新分词**：

```python
new_prefix = old_prefix[:cut]
new_suffix = old_prefix[cut:] + old_suffix
assert new_prefix + new_suffix == old_prefix + old_suffix
```

  还逐条核对 EOS、长度、slot、候选顺序和 case/field 身份。找不到分界文件或分界不合法就直接报错，不退回旧分界。完整 pair 长度不变，所以 micro 划分也和 PB 相同。
- **为什么不按 state 的字符长度切**：BPE 可能把 state 最后一个字符和后面的换行编成同一个 token。最早的严格版本（分界放在 state 最后一个字符）检查发现，训练集 41,674 个有标签 field 中 28,676 个、开发集 615 个中 427 个都跨了这个边界（约七成），所以那一版没有训练。v2 统一把那个换行归入 P，全部 field 都能精确切开，才开始训练。
- **借鉴 Base 的程度**：只借了"state 作前块、问题作后块"的分界思路，仍是单候选 pair 加 EOS 二分类 head。
- **结果**：本地开发集 44.4%，比 PB 高 1.1 个百分点，但 bootstrap 区间 [−3.8, +6.0] 跨过 0；官方 1.97，比 PB 低 0.62。

<a id="dagger"></a>
## 9. † 的含义：dense 训练、packed 评测

`C_D_CE` 和 `C_D_BCE` 是 dense 训练的，官方评测时和其它六个分支一样用 packed 推理。正式评测前，先拿同一个最终权重，在全部 375 个开发 case 上对比"保存的 dense 输出"和"新的 packed 输出"的 logits 与 margin。每个 case 要满足：

$$RMS(\Delta)\le0.01\max(RMS(ref),10^{-3}),\qquad \max|\Delta|\le0.02+0.02\max|ref|,$$

并且不能出现带外的选项翻转（带外的定义见报告[第 9 节](REPORT.md#numerics)）。

| dense 训练的权重 → packed 推理 | 超出容差的 case | 选项变化 | 带外翻转 | 最大概率差 |
|---|---:|---:|---:|---:|
| `C_D_CE` | **34 / 375** | 0 | 0 | 0.08 |
| `C_D_BCE` | **34 / 375** | 1 | 0 | 0.11 |

两组都判定为未通过。393 个参数张量对比前后字节不变，所以偏差来自 dense 和 packed 两条执行路径的浮点差异，不是权重被改了。`C_D_CE` 的选项一个都没变，但概率最多差 0.08，因此也不算数值等价。

你要求八个都测，所以这两个权重仍然跑了完整官方评测，记为"仅作科学测量"。13.07 和 12.83 是"dense 训练的权重在 packed 推理下"的实测分数，不能用来判断 dense 训练和 packed 训练孰优孰劣。

<a id="sources"></a>
## 10. 代码位置和自测问题

### 10.1 八个分支在代码里只差几个开关

```python
# model_runtime.FullRow
core = MaskedBackbone(backbone, head, native, arm['mode'])     # causal / bidirectional / prefix_block
batch = make_batch(rows, layout=arm['layout'])                 # dense / packed

# objectives
coef = coefficients(all_complete_field_margins, fields, cases, arm['loss'])   # BCE / CE / BCE_RANK_HARD

# 只有 STATE_NL 额外做
pair = Pair(..., original_prefix[:cut], original_prefix[cut:] + suffix)
```

开关虽少，背后要求布局隔离、完整 field 的目标、精确分界、两遍前向逐位一致，以及初始化、数据和曝光完全相同。

### 10.2 仓库里能直接读的源码

- [基础 BCE / field-CE](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/objectives.py)：第 7～35 行。
- [JPAIR 和按原始 q 分正负](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-followup-v1/source/objectives.py)：第 7～74、135～146 行。
- [三种 mask 和前向执行层](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/native-mask/packed_mask/model.py)：第 73～102 行是三种 mask，146～183 行是 Qwen3/PEFT 逐 token 计算和 head。
- [dense / packed 布局和 EOS 下标](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/native-mask/packed_mask/layout.py)：第 70～109 行。
- [mode 和 layout 怎样接进训练](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/model_runtime.py)：第 68～126 行。
- [STATE_NL 的无损重切](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-followup-v1/source/case_runtime.py)：第 195～232 行。
- [新建 LoRA 和 head](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/fresh_init.py)：第 35～81 行。
- [两遍前向的训练循环](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/train_case.py)：第 60～158 行。
- [warmup 和 cosine 学习率](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/checkpoints.py)：第 68～104 行。

仓库里的是源码快照，用来理解代码。真实运行以服务器上冻结的配置和源码为准：

```text
前六个分支的训练：/tmp/opencode/jev-clean-trainer-20261112-v2/
后两个分支的训练：/tmp/opencode/jev-clean-trainer-20261112-followup-v1/
八个分支的官方评测：/home/dev/assumeengage/runs/jev-public-eight-fourway-20261113-v1/
最终权重：各 <arm>-seed42/checkpoint-m0003557-s00043/
```

官方汇总和独立核验的副本在[证据目录](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/README.md)。

### 10.3 检查自己是否读懂

1. `PB_P_CE` 里的第一个 PB 和中间的 P 为什么不是一个意思？
2. 为什么 `D` 不能理解成 double 精度或全参数训练？
3. head 只有 2 个输出，为什么能回答有 255 个候选的 field？
4. 为什么 CE 能让候选在梯度上互相竞争，却不让它们在 attention 里互相看见？
5. 名字里的 `RANK` 和 LoRA 的 r=32 有什么区别？
6. causal 下 EOS 已经能看到全部前文，为什么改成双向后深层输出仍会不同？
7. STATE_NL 一个 token 都没改，为什么仍是模型变化？
8. 为什么两个 D 分支的官方分数可以报告，却不能说 dense 和 packed 等价？
9. 为什么 JPAIR 应该先和 `C_P_BCE` 比，而不是 `C_P_CE`？
10. 官方评测用了 32 个推理进程，为什么不是 32 个新模型，也不是 ensemble？
