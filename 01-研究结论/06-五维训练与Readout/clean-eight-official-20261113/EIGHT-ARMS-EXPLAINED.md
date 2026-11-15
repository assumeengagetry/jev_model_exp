# 八个实验名字到底是什么意思：逐模型、逐变量、逐公式详解

> 实验完成：2026-11-13；本文整理：2026-11-14。
>
> 本文专门解释 `C_P_BCE`、`JPAIR_P_BCE_RANK`、`C_D_CE`、`C_P_CE`、`C_D_BCE`、`BI_P_CE`、`PB_P_CE`、`STATE_NL_PB_P_CE`。它不是只有缩写翻译的结果表，而是说明每一组从什么共同起点开始、具体改变哪条计算路径、目标函数和梯度是什么、哪些东西没有改，以及结果应怎样解读。
>
> **先记住：这是同一个 Qwen3-0.6B 基座上的八个独立匹配训练分支，不是下载的八种基础模型，也不是从上一个训练结果接着改成下一个。你列出的顺序是按官方分数排序，不是模型演化顺序。**
>
> 服务器副本：`/home/dev/assumeengage/reports/jev-eight-arms-explained-20261114.md`。本文不新增训练、模型评测或模型下载，不修改既有权重/配置；本次未自动提交或推送。

## 目录

1. [一分钟读懂命名和八组关系](#names)
2. [八组共同的模型改造](#shared-model)
3. [输入、二类 head、margin 和字段 softmax](#readout)
4. [D 与 P：改变的是 token 布局](#layouts)
5. [C、BI、PB：改变的是注意力可见性](#masks)
6. [BCE、CE、RANK：改变的是训练目标](#losses)
7. [逐组解释：C_P_BCE](#arm-c-p-bce)
8. [逐组解释：JPAIR_P_BCE_RANK](#arm-jpair)
9. [逐组解释：C_D_CE 与 †](#arm-c-d-ce)
10. [逐组解释：C_P_CE](#arm-c-p-ce)
11. [逐组解释：C_D_BCE 与 †](#arm-c-d-bce)
12. [逐组解释：BI_P_CE](#arm-bi)
13. [逐组解释：PB_P_CE](#arm-pb)
14. [逐组解释：STATE_NL_PB_P_CE](#arm-state)
15. [共同的数据、曝光、优化器与两遍训练](#training)
16. [如何做正确的成对比较](#comparisons)
17. [结果、局限和不能混淆的结论](#results)
18. [实际代码位置与自查问题](#sources)

---

<a id="names"></a>
## 1. 一分钟读懂命名和八组关系

### 1.1 这些不是行业统一模型名称

它们是这次实验用来追踪“注意力模式、训练布局、训练目标”的分支标识，也叫实验臂（arm）。不要把它们当成类似 BERT、Qwen 那样不同的基础架构名称。

大多数名字可以拆成：

```text
注意力模式 _ 训练时 token 布局 _ 训练目标
```

| 缩写 | 本实验含义 | 不是指什么 |
|---|---|---|
| `C` | causal，因果注意力 | 不是 classification 标签数 |
| `D` | dense/padded，训练 tokenwise 张量含 padding | 不是 double/FP64、蒸馏、全参数微调或 DDP |
| 中间的 `P` | packed，真实 token 直接串接执行 | 不是 positive、prefix cache、剪枝或新预训练模型 |
| `BCE` | 对每个候选 margin 做二元交叉熵 | 不是最后直接拿各 sigmoid 当字段分布 |
| `CE` | 完整 field 内所有候选 margin 的分布交叉熵 | 不是把二类 head 改成固定 K 类 head |
| `BI` | bidirectional，一条完整 pair 内全双向 | 不是 bi-encoder 双塔 |
| `PB` | prefix-block，两块注意力可见性 | 不是通常“prefix 双向、suffix 因果”的 prefix-LM |
| `STATE_NL` | P 边界移到 state 加一个已有 newline/LF 后 | NL 不是新增层，也不是 natural-language 专家 |
| `JPAIR` | 标记 Jina-inspired、保留独立 pair 的排名目标分支 | 不是完整 Jina 模型，也不是新增列表交互架构 |
| 尾部 `RANK` | 排名损失项 | **不是把 LoRA 的 rank 从 32 改大** |
| `†` | 结果表的脚注警告 | 不是实际配置或 checkpoint 名的一部分 |

三个容易撞名的 P 要区分：名字中间的 `P` 是 packed；注意力公式中的 P 是 prefix；ranking 中的正例集合是另一层数学记号。下文尽量把正例集合写成 `I+`，避免混淆。

### 1.2 八组一览

| 名字 | 注意力 | 训练布局 | 目标 | 官方消费者布局 | 官方 Index |
|---|---|---|---|---|---:|
| `C_P_BCE` | causal | packed | BCE | packed | 14.42 |
| `JPAIR_P_BCE_RANK` | causal | packed | BCE + hard-rank | packed | 13.60 |
| `C_D_CE` † | causal | dense | field CE | packed † | 13.07 |
| `C_P_CE` | causal | packed | field CE | packed | 12.95 |
| `C_D_BCE` † | causal | dense | BCE | packed † | 12.83 |
| `BI_P_CE` | full bidirectional | packed | field CE | packed | 3.09 |
| `PB_P_CE` | Candidate 边界 prefix-block | packed | field CE | packed | 2.59 |
| `STATE_NL_PB_P_CE` | state+LF 边界 prefix-block | packed | field CE | packed | 1.97 |

**特别注意：D/P 在名字里记录训练布局。最终这次官方测分，八组消费者全是 packed；两个 D 组因此带有额外的迁移数值限制。**

### 1.3 它们不是串行继承训练权重

正确关系是：

```text
同一原始 Qwen3-0.6B
  + 相同 seed42 fresh LoRA/head 初态
  + 相同冻结 case/候选/监督/曝光流
  │
  ├─ causal / dense  / BCE      → C_D_BCE
  ├─ causal / packed / BCE      → C_P_BCE
  ├─ causal / dense  / field CE → C_D_CE
  ├─ causal / packed / field CE → C_P_CE
  ├─ BI     / packed / field CE → BI_P_CE
  ├─ PB(Candidate) / packed / CE→ PB_P_CE
  ├─ PB(state+LF) / packed / CE → STATE_NL_PB_P_CE
  └─ causal / packed / BCE+rank → JPAIR_P_BCE_RANK
```

这里的分叉表示配方对照。不是先训练 `C_P_CE`，再把其已训练权重切成 BI/PB；也不是训练完 BCE 再在其终点加 rank 继续训。后续两组 STATE/JPAIR 也先从 stock 独立生成 fresh，再核对初态与前六组一致。

前四组组成布局×目标的 2×2 对照；BI/PB 是匹配的 mask 对照；STATE/JPAIR 是看到早期 dev 之后提出的探索性跟进，不追记成八组全预注册。

<a id="shared-model"></a>
## 2. 八组共同的模型改造

### 2.1 从因果语言模型资产到候选匹配分类模型

共同起点是本地原 ModelScope `Qwen3-0.6B`，不是另下载一个 Jina/Base 模型，也不是用历史已经训练的 adapter 做新基线。

共同构造大意如下，细节以 `fresh_init.py` 为准：

```python
config.num_labels = 2
config.id2label = {0: 'not_match', 1: 'match'}
config.pad_token_id = 151643
config.eos_token_id = 151645
config.use_cache = False

# 保留原 Qwen3 backbone，构造序列分类接口。
model = AutoModelForSequenceClassification.from_pretrained(
    STOCK, config=config, dtype=torch.bfloat16, ...
)

# 丢弃 loader 临时创建的 score，建立真正共同 fresh 的二类 head。
model.score = Linear(1024, 2, bias=False, dtype=float32)
normal_init(model.score.weight, mean=0, std=0.02)

model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0,
    target_modules=[q, k, v, o, gate, up, down],
    modules_to_save=['score'], task_type='SEQ_CLS', bias='none'
))
```

这是说明性节选，不是本文新执行的代码。底层执行通过共享的 `MaskedBackbone` 调用实际 Qwen3/PEFT 模块和 native FA3，而不是仅靠 HF 的 `attn_implementation` 字符串决定全部注意力行为。

### 2.2 保留了哪些结构

| 项目 | 八组共同值 |
|---|---|
| backbone | 原 Qwen3-0.6B，28 层 |
| hidden size | 1024 |
| MLP intermediate | 3072 |
| Q heads / KV heads / head_dim | 16 / 8 / 128 |
| attention | 原冻结 native FA3，GQA 不展开 KV |
| LoRA | r32、alpha64、scale2、dropout0 |
| active head | 一个无 bias 的 `Linear(1024,2)` |
| trainable 参数 | 20,187,136 |
| active served 参数 | 616,237,056 |
| 保存张量 | 196 对 A/B + 1 个 head = 393 |

**“二类 head”不是“两层 head”。**本轮只有一层线性分类头，不是历史 readout 研究的两层 MLP，也不是 Jina 的 projector。

### 2.3 LoRA 实际怎样改变每个投影

对于原线性层权重 `W0`：

$$W_{eff}=W_0+\frac{\alpha}{r}BA=W_0+2BA.$$

按 PyTorch `[out,in]` 约定：A 是 `[r,in]`，B 是 `[out,r]`。输入 X 的额外路径等价于 `X A^T B^T × 2`；不是把一个 r×r 方阵直接塞到 attention 中。

| 投影 | 原权重 `[out,in]` | LoRA A / B | 每层新增参数 |
|---|---|---|---:|
| q | `[2048,1024]` | `[32,1024]` / `[2048,32]` | 98304 |
| k | `[1024,1024]` | `[32,1024]` / `[1024,32]` | 65536 |
| v | `[1024,1024]` | `[32,1024]` / `[1024,32]` | 65536 |
| o | `[1024,2048]` | `[32,2048]` / `[1024,32]` | 98304 |
| gate | `[3072,1024]` | `[32,1024]` / `[3072,32]` | 131072 |
| up | `[3072,1024]` | `[32,1024]` / `[3072,32]` | 131072 |
| down | `[1024,3072]` | `[32,3072]` / `[1024,32]` | 131072 |

合计每层720896，28层 LoRA 为20185088，再加 head2048，得到20187136。

原 stock backbone 的596049920参数冻结；加 LoRA 与活动 head 后服务参数为616237056。物理注册量另多2048，是不参与服务的原 score 副本，不应算成第二个分类模型。

LoRA B 的 fresh 初值是零；A 按原初始化生成，所以初始 LoRA 增量为零。head 是新的随机初态。八组不是“完全随机训练0.6B”：预训练 backbone 已存在，只重新训练约20.19M的 LoRA/head 参数。

### 2.4 改 mask 并没有改变参数数量

C、BI、PB 改的是 attention 中“允许哪些 Q/K 配对”，没有新增一套 Transformer block。STATE 改边界，没有新增层；JPAIR 改损失，没有新增 head。因此八组活动服务参数量相同。

trainable 少也不代表计算仅需20M参数模型的成本：前向仍经过完整 backbone，训练反向还需沿 hidden/attention 传播到各个 LoRA 分支。

### 2.5 参数精度不等于算子执行精度

- stock 规范参数以 BF16 加载，保持冻结。
- LoRA/head 规范参数是 FP32，保存和前后 byte 比对沿这个 dtype。
- forward 位于 BF16 autocast，head 的 GEMM 也可按 BF16 路径执行；不是全 FP32 计算。
- 先对全部 tokenwise 行做 head，然后 gather EOS logits；packed 是全部 realT，dense 控制的 head 还包含其 padded 物理行。
- logits 转 FP32，再相减得 margin；最终字段 softmax 使用双精度读出。

不先 gather EOS hidden 再缩小 head GEMM，是为了保留已冻结的运算形状边界；数学上的线性等式，不保证不同 GEMM 形状在浮点库里逐位相同。

<a id="readout"></a>
## 3. 输入、二类 head、margin 和字段 softmax

### 3.1 case / field / candidate / pair

下面是教学例子，不是私有训练样本：

```text
case：一个客户请求
  field 1：该交给哪个部门？
    candidate A：退款
    candidate B：物流
    candidate C：技术支持
  field 2：是否需要升级处理？
    candidate Yes
    candidate No
```

一个 case 可以有多个 field；一个 field 可能有很多 candidate。每个候选都构成一条完整 pair。上面是5条 pair，不是2条，也不是把5个答案都塞进同一条无隔离序列。

### 3.2 原始模板

```text
Decide whether the candidate is a correct match.
Input:
{state}
Question:
{question}
Available criteria:
{按原规则需要时保留的完整候选上下文}
Candidate:
{当前 candidate}
<真实 EOS>
```

Question/criteria 段按条件出现。遇到“all/none of the above”等选项引用，或小候选集的相对比较问题，原 renderer 会保留原顺序的完整 criteria。独立 pair 并不等于文本里永远没有其它选项信息，但它们没有额外的跨 pair hidden-state attention。

prefix 和 candidate 分别 tokenize，不自动加特殊 token、不截断；candidate 末尾追加 EOS151645。后文的 STATE 不重新分词，也不换模板。

### 3.3 二类 head 不等于只有两个候选

对每个候选 k，EOS head 返回两个数：

$$z_{k,0}=\text{not-match logit},\quad z_{k,1}=\text{match logit}.$$

再取：

$$m_k=z_{k,1}-z_{k,0}.$$

有 K 个候选，就分别得到 K 个 margin；随后组合成动态长度 K 的字段分布：

$$p_k=\frac{e^{m_k-\max_jm_j}}{\sum_je^{m_j-\max_jm_j}}.$$

所以 `num_labels=2` 不代表 API 只能回答两个选项，也不需要为每个不同 K 新建一个 K 类 head。

### 3.4 sigmoid 和最终 softmax 是不同层级

若三候选 margin 是 `[2,0,-1]`：

| 候选 | sigmoid(m) | field softmax(m) |
|---|---:|---:|
| A | 0.880797 | 0.843795 |
| B | 0.500000 | 0.114195 |
| C | 0.268941 | 0.042010 |
| 和 | 1.649738 | 1.000000 |

sigmoid 是各自独立的二分类概率；字段 softmax 才是所有给定候选之间的归一分布。**所有八组正式输出都用字段 softmax，并不是 BCE 组输出 sigmoid、CE 组输出 softmax。**

候选多时，一条请求会组装多条完整 pair；`usage.input_tokens` 是这些 pair 长度和，包含重复 prefix，而不是只数一次 state。没有生成答案文本再解析选项的步骤。

<a id="layouts"></a>
## 4. D 与 P：改变的是 token 布局

### 4.1 Dense 训练怎样排布

设三条真实长度为4、2、5。实际 dense 宽度向8对齐：

```text
ids[3,8]
A A A A _ _ _ _
B B _ _ _ _ _ _
C C C C C _ _ _
```

Embedding、Norm、QKV/LoRA、MLP 等 tokenwise 模块处理24个物理行，其中13个是 pad。

但 attention 会 gather 真实 Q/K/V，再用 native FA3 varlen 和 cu 隔离序列。因此 **D 不表示把 pad 作为有效 key，也不表示使用全连接的“dense attention”**；causal 的 D 组仍然是因果可见性。

### 4.2 Packed 训练怎样排布

```text
ids[11] = A A A A | B B | C C C C C
cu      = [0,4,6,11]
pos     = [0,1,2,3 | 0,1 | 0,1,2,3,4]
```

hidden 为 `[11,1024]`，全部 tokenwise 运算只处理11个真实行。

注意 cu 和 positions 分工不同：cu 决定各序列的边界，positions 决定各序列内 RoPE 的位置。不能因为物理串接，就让后一条 pair 的 position 从4或6继续增长。

### 4.3 EOS 索引也随物理布局变化

上例中每行最后一个真实 token 是 EOS：

- dense flatten 下 EOS indices 为 `[3,9,20]`；
- packed 下为 `[3,5,10]`，即 `cu[1:]-1`。

虽然物理索引不同，必须指向同一条原 pair 的同一个真实 EOS。

### 4.4 Packing 不产生候选交互

只要 cu 正确，A pair 看不到 B/C pair 的 token。错误地只把所有 token 拼起来，再套一张全局 causal 下三角，才会让后面的无关题看到前面的题，那是实现错误，不是正确 packing。

独立 pair 也不自动保证整个服务对候选排列严格不变：条件 criteria 文本保留原顺序，tie 规则也按原序取首个；本轮没有用“独立 pair”冒充排列不变性证明。

### 4.5 为什么 matched-control 的 micro 还按 padded 预算规划

两布局采用相同 stable length 排序、max32 pairs、向8对齐的共同预算16384，先保证 micro 成员/顺序一致。packed 只是执行 realT，不按这个 admission 预算真的补 pad。

如果一边改布局、一边把 packed 的 micro 组成和大小也换了，就额外改变了浮点几何。当前是在研究“同曝光布局效果”，不是“同GPU时间让packed多学一些样本”的另一种实验。

<a id="masks"></a>
## 5. C、BI、PB：改变的是注意力可见性

### 5.1 统一的数学记号

attention 每个 query 行先计算：

$$A_{ij}=\frac{\exp(q_i^Tk_j/\sqrt{128}+M_{ij})}{\sum_t\exp(q_i^Tk_t/\sqrt{128}+M_{it})},$$

允许时 `M=0`，禁止时可视作 `M=-∞`。实际 FA3 不必真的分配这张 T×T 矩阵；公式只是表示可见性。

所有模式先限制在同一独立 pair 内，下面再讨论 pair 内允许哪些 token 相互看见。

### 5.2 Causal

$$\text{allow}(i,j)=[j\le i].$$

四 token 示意：

```text
query\key  0 1 2 3
0          1 0 0 0
1          1 1 0 0
2          1 1 1 0
3          1 1 1 1
```

每个位置只能看自己和之前的位置。真实实现传 `causal=True` 给 native FA3，而不是只在配置里写一个字段。

### 5.3 BI 全双向

$$\text{allow}(i,j)=1\quad\text{（同一pair内）}.$$

```text
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
```

state 内 token 可以看 question/candidate，candidate 前面的 token 也能看后面的 token。不同候选仍是不同 pair，不互相串联。

### 5.4 PB 两块双向

把一条 pair 分成 P（prefix）和 S（suffix）：

$$\begin{array}{c|cc}
 &K_P&K_S\\\hline
Q_P&允许&禁止\\
Q_S&允许&允许
\end{array}$$

P、S块内都双向。P不能看S；S看自己的完整P+S。P长2、S长2时：

```text
1 1 | 0 0
1 1 | 0 0
----+----
1 1 | 1 1
1 1 | 1 1
```

**与常见 prefix-LM 的区别：这里 suffix 不是因果的。**

### 5.5 EOS最后都能读整条pair，为什么结果仍不同

causal 的 EOS 本来就在最后，能读到之前所有 token。改变 mask 主要改变其它位置的中间表示。

在理想实数算术、同输入与同权重下，第一层末端 EOS 的可见 key 集合可以相同；但第一层其它 token 的表示已经不同。第二层的 EOS 看到的是这些不同的 K/V，所以多层结果不再相同。实际不同 GPU kernel 的浮点路径连第一层也不承诺逐位一致。

### 5.6 为什么 PB 每层有两次 native attention

共享 QKV 投影后，按 query 分区：

```python
prefix_out = FA3(Q_P, K_P, V_P, causal=False)
suffix_out = FA3(Q_S, K_full_pair, V_full_pair, causal=False)
out = scatter_to_original_positions(prefix_out, suffix_out)
```

第二个 softmax 对完整 P+S 的 key 集合一次归一化，不能分别算 `attend(S,P)`、`attend(S,S)` 后直接相加。

causal/BI 每个 wave 是28次 attention；PB是56次。但 QKV/MLP只执行一遍，**不是整个模型的计算成本简单乘2**。

### 5.7 八组没有按名字决定是否开启KV缓存

中间的P是packed，不是prefix cache。最终八组全部 full-row、无prefix KV快路径。

理论上causal前缀可满足复用条件；BI前缀读suffix不能保持immutable；PB虽不读suffix，但其完整双向prefix必须完全相同，不能任意按LCP命中。实际prefix-block fork/paged曾有数值失败，本轮没有为了速度打开未准入路径。

<a id="losses"></a>
## 6. BCE、CE、RANK：改变的是训练目标

### 6.1 统一分母

macro有C=16个完整case；case c有F_c个known field；field f有K_f个完整候选，目标q来自原始合法监督。

所有目标按：

$$L=\frac1C\sum_c\frac1{F_c}\sum_{f\in known(c)}L_{cf}.$$

不是把全macro所有候选混在一起平均，也不是平均每个micro loss。unknown field不作为负例，不进入监督分母。

### 6.2 BCE的实际公式和梯度

$$B_f=-\frac1{K_f}\sum_k\left[q_k\log\sigma(m_k)+(1-q_k)\log(1-\sigma(m_k))\right].$$

原代码：

```python
F.binary_cross_entropy_with_logits(m, q, reduction='mean')
```

它同时计算sigmoid和稳定的BCE表达，不是先手工sigmoid再做容易出现log(0)的计算；没有额外`pos_weight`。

$$\frac{\partial B_f}{\partial m_k}=\frac{\sigma(m_k)-q_k}{K_f}.$$

**拟合对象是sigmoid(m)，不是m直接接近q。**软标签0<q<1、独立无约束时最优margin是log(q/(1−q))；硬标签1/0的损失下确界分别在margin趋向正/负无穷时取得。实际共享参数、有限预算、weight decay和clip会限制它。

### 6.3 Field-CE的实际公式和梯度

$$CE_f=-\sum_k q_k\log p_k,\qquad p_k=\operatorname{softmax}(m)_k.$$

原代码：

```python
-(q * torch.log_softmax(m, dim=0)).sum()
```

`dim=0`是当前field切片中的全部候选，不是整个macro。没有额外除K。

$$\frac{\partial CE_f}{\partial m_j}=p_j\sum_kq_k-q_j.$$

目标和为1时为`p_j-q_j`。q只在准入时检查合法性与和容差，不在loss里重新归一化。

### 6.4 CE并没有修改输出head

`C_P_CE`仍是每个pair一个`Linear(1024,2)`。先从每个候选得到margin，再把同field的K个margin放在一起算CE。它不是“将二类head换成K类head”。

### 6.5 loss竞争不等于attention交互

```text
pair A ──独立forward── mA ─┐
pair B ──独立forward── mB ─┼─ 同field loss
pair C ──独立forward── mC ─┘
```

CE使一个候选的梯度依赖其它候选的分数；但计算mA的这次forward没有额外读取pair B/C的hidden。改变loss不会自动生成Jina式候选同窗结构。

### 6.6 共同平移不变性不同

对同field所有margin都加常数c，softmax与CE不变；BCE通常会变。CE主要约束相对竞争关系；BCE也约束各候选的绝对二分类概率标尺。

一个容易忽略的理论细节：若每个soft候选都能独立达到BCE最优，则`m_k=logit(q_k)`；再做服务用的field softmax，得到的分布一般不是q。

教学例子q=[0.7,0.3]：BCE独立最优m≈[0.847298,−0.847298]，但softmax(m)≈[0.844828,0.155172]，不是[0.7,0.3]。CE的无约束最优则可取`m=log(q)+c`。

这说明两种训练概率空间不完全一样；**不证明BCE实际一定更差或CE一定更好**。真实模型共享参数、跨样本训练、正则化与任务分布都使它不能简单等同这个孤立最优例子。本轮官方结果恰好是packed BCE较高。

### 6.7 RANK的精确定义

按原始host q取：

```text
I+ = {k : q_k == 1}
I− = {k : q_k == 0}
H  = I+ ∪ I−
```

soft q不进H，unknown更不能当负例。先在原host值上分类，避免把接近1的soft值转FP32后误变成精确1。

$$R_f=\begin{cases}
\log\sum_{k\in H}e^{m_k}-\log\sum_{k\in I+}e^{m_k},&I+,I-均非空,\\
0,&否则.
\end{cases}$$

本轮固定lambda=1、tau=1，所以：

$$L_{JPAIR}=\frac1C\sum_c\frac1{F_c}\sum_f(B_f+R_f).$$

无rank信号的known字段仍在F_c分母里，不重新按active rank字段计平均。实现的零项是`(m*0).sum()`，保持图连接并避免先求和巨大有限margin造成溢出。

### 6.8 为什么不是“所有正负pair两两做RankNet”

它没有枚举每个正例与每个负例的差值再平均，而是用两个logsumexp表达“正例集合相对所有已知硬候选的概率质量”。因此不要只凭名字中的PAIR/RANK就套另一个论文配方。

当前合法归一分布通常只有一个精确q=1。严格one-hot且K≥2时：

$$R_f=\log\sum_ke^{m_k}-m_{positive}=CE_f.$$

因此在这些硬字段上，JPAIR可理解为BCE＋CE；纯soft字段上rank为0，只做BCE。若q虽近似归一但还含soft残差，不能无条件说rank等于完整field CE。

### 6.9 一个梯度尺度的手算例子

教学例子：K=4、q=[1,0,0,0]、所有m=0，尚未乘case/field外层分母：

| 目标 | 正候选dL/dm | 每个负候选dL/dm |
|---|---:|---:|
| BCE | −0.125 | 0.125 |
| CE | −0.750 | 0.250 |
| BCE＋rank | −0.875 | 0.375 |

所以加rank不仅改变名字，还会改变梯度方向、正负相对权重和尺度，进而影响clip/Adam轨迹。不能把效果单独归因为“增加排序信息”。

<a id="arm-c-p-bce"></a>
## 7. C_P_BCE：因果＋packed＋逐候选BCE

### 7.1 名字与实际配置

```json
{"mode":"causal","layout":"packed","loss":"BCE"}
```

- C：pair内因果注意力。
- P：训练全层tokenwise真packed。
- BCE：field内先对候选BCE取mean，再field/case平均。

### 7.2 具体改了哪些代码路径

这不是单独写一个新模型类，而是共享构造器读取这个arm：

```python
batch = make_batch(rows, layout='packed', ...)
core = MaskedBackbone(backbone, head, native_fa3, mode='causal')
# native call: causal=True
loss = binary_cross_entropy_with_logits(field_margins, q, reduction='mean')
```

相对dense BCE控制，改变`make_batch`布局；相对packed CE，改变目标函数分支。base/head/LoRA形状、监督、完整pair和曝光都不改。

### 7.3 在这组中一条题怎样处理

field有K候选就构建K条完整pair。每条都从position0开始，末端真实EOS经过同一二类head。虽然K条可能在一个packed微批里，它们的cu隔离不允许相互attention。BCE根据各自q拉高/压低相应sigmoid(m)，服务时再对K个margin做softmax。

### 7.4 为什么保留它

它是packed因果BCE基线，用来观察去padding后的原候选判别目标，也给JPAIR提供匹配目标控制。没有它，就无法判断JPAIR的变化到底比BCE本身好不好。

### 7.5 实际结果与边界

- 100M D_select mass：57.5694%。
- 完整官方Public Index：**14.42**，本轮八个终点最高。
- 比C_P_CE高1.47个Index点，比JPAIR高0.82点。

只能说本轮固定seed、数据与终点中它最高；不能说BCE普遍优于CE，也不能说packing必然带来这个提升。

<a id="arm-jpair"></a>
## 8. JPAIR_P_BCE_RANK：独立pair路径上的BCE＋硬标签排名

### 8.1 名字与实际配置

```json
{"mode":"causal","layout":"packed","loss":"BCE_RANK_HARD"}
```

另有冻结合同：hard label只认原host精确0/1，lambda=tau=1，known-field mean后case mean。

**它仍然是causal，也仍然是独立pair。LoRA仍r32，不是把低秩rank加大。**

### 8.2 相对C_P_BCE到底改了什么

原先：

```python
bce = binary_cross_entropy_with_logits(m, q, reduction='mean')
field_loss = bce
```

新分支：

```python
positive = [i for i, value in enumerate(original_host_q) if value == 1]
negative = [i for i, value in enumerate(original_host_q) if value == 0]
bce = binary_cross_entropy_with_logits(m, q, reduction='mean')
if positive and negative:
    rank = logsumexp(m[positive + negative]) - logsumexp(m[positive])
else:
    rank = (m * 0.0).sum()
field_loss = bce + rank
```

原field/case分母不改；对组合后的完整loss求`dL/dmargin`，再用原两遍训练机制。

### 8.3 软标签、unknown和候选顺序怎样处理

- q=0.7/0.3等soft目标只参与BCE，不伪装成hard正负。
- unknown字段不造q=0，不进入rank。
- 没有正或负时rank为0，但字段不从分母移除。
- 不按标签把正例移到序列前面，不把正确答案编码进mask。
- 输入候选顺序、pair、EOS、slot、微批和全部学习参数都与匹配BCE控制一致。

### 8.4 借鉴Jina的程度

借鉴的是排序目标思路，保留本项目pair/EOS分类体系。完整Jina还包含多doc同窗causal交互、末端query、特殊token、projector/cosine读出等；本组没有引入这些结构。

另做过J0/J1/J2机制接线，不等于本组已经完成J1候选交互的fresh训练。**给JPAIR的13.60不能当完整Jina的复现分数。**

### 8.5 实际结果应该跟谁比

- 本地mass：61.2437%，比C_P_BCE高3.6743个百分点。
- 官方Index：**13.60**，却比C_P_BCE低0.82点。
- 对C_P_CE的差是+0.65点，但最匹配的“是否加rank”控制是C_P_BCE，不应只挑CE这个参照来包装成功。

训练中还记录了rank/BCE/total的margin梯度系数尺度；所比较臂clip factor<1的记录均100%，JPAIR总/BCE margin系数L2比均值约3.3008。它不是参数梯度分量比，也不是排序方向独立收益证明。

<a id="arm-c-d-ce"></a>
## 9. C_D_CE †：dense训练的因果field-CE控制

### 9.1 配置与D的含义

```json
{"mode":"causal","layout":"dense","loss":"CE"}
```

训练时tokenwise张量是向8对齐的`[B,width]`；attention仍gather真实QKV、同native FA3 causal，不让真实token读取pad。

### 9.2 相对哪个控制改了什么

- 与C_D_BCE比：只改变loss分支，从各候选BCE变成完整field CE。
- 与C_P_CE比：训练目标和causal一致，只改变训练tokenwise布局。
- 不是改成全参数训练，也不是用FP64精度。

### 9.3 CE发生在哪里

不是在`Linear(1024,2)`处对两个logit直接用一个K类标签；仍先得到每候选`m=z1−z0`，然后对该field全部K个margin做`log_softmax`。

若field有6候选，head仍是2行权重，loss的field向量长度才是6。

### 9.4 为什么名字后面有†

`†`只是一条警告：这组**训练是dense，最终public统一改用packed消费者**。固定终端权重从dense迁移到packed的375case比较中，34个case未通过原logits/margin数值合同。

虽然该开发集上choice变化0、带外flip0，但最大概率差约0.076288；不能据argmax不变说数值等价。原`numerical_acceptance=false`没有被修改。

所以正式13.07是“dense-trained权重在这次packed消费者上的完整实际分数”，不是原dense执行路径自己的完整public分数，更不是迁移准入通过证明。

### 9.5 实际结果

- 原训练布局100M本地mass：61.6935%，本地八组最高。
- 统一packed消费者的官方Index：**13.07**。
- 与packed CE的12.95差0.12点，可描述实际消费者分差；不能据此说dense数学上等价packed，或dense训练稳定更好。

<a id="arm-c-p-ce"></a>
## 10. C_P_CE：packed因果field-CE，以及mask研究的控制组

### 10.1 实际配置

```json
{"mode":"causal","layout":"packed","loss":"CE"}
```

它与C_P_BCE的网络和layout相同，仅改变目标概率空间；与C_D_CE的causal/目标相同，仅改变训练layout。

### 10.2 为什么BI/PB以它为主要控制

为了研究mask，必须固定同样的loss。BI/PB用field CE，所以控制也用field CE。不能拿BCE因果模型与CE双向模型对比后，把全部差异归为mask。

这不表示C_P_CE是看过public后选出来的最佳模型；它是研究控制，本轮最终分数并非最高。

### 10.3 与BCE最关键的差异

```python
# 共同：独立pair → EOS二类logits → FP32 margin
# BCE目标：
bce(m, q, reduction='mean')
# field-CE目标：
-(q * log_softmax(m, dim=0)).sum()
```

候选之间在loss层竞争，但attention仍独立。CE对该field整体margin平移不敏感；K和field/case分母的规则见第6节。

### 10.4 实际结果

- 本地mass：59.5955%。
- 官方Index：**12.95**。
- 它也是四卡同权重调度中第一个完整完成的模型，约1h43m16s。先测它是执行顺序/首分目标，不是判断其必然最好。

<a id="arm-c-d-bce"></a>
## 11. C_D_BCE †：dense训练的因果BCE控制

### 11.1 实际配置

```json
{"mode":"causal","layout":"dense","loss":"BCE"}
```

它是四基础臂中最直接的dense/BCE控制。与C_P_BCE仅改变训练layout；与C_D_CE仅改变目标。

### 11.2 它为什么必须保留

若只做packed BCE与dense CE，layout和loss同时变化，无法区分哪一项相关。加入C_D_BCE及另外两个C臂，才能形成完整2×2表：

| 训练布局 | BCE | field CE |
|---|---|---|
| dense | C_D_BCE | C_D_CE |
| packed | C_P_BCE | C_P_CE |

这种对照比随便挑两个结果更有解释力；但最终D组迁移packed的数值限制仍要单独保留。

### 11.3 †警告与实际结果

- 原dense开发mass：56.4260%。
- fixed-weight packed消费者开发mass：56.3371%，这是布局运行时差异，不是又学习了一轮。
- 原数值合同34/375case失败；1个field choice flip，带外0，最大概率差约0.110492。
- 官方packed消费者Index：**12.83**。

可以把这次13.07和12.83描述为两个dense-trained权重在同一packed消费者下的实测差；不能把它们升级为dense/packed数值等价或跨seed稳定结论。

<a id="arm-bi"></a>
## 12. BI_P_CE：整条pair内全双向，不是双塔

### 12.1 实际配置

```json
{"mode":"bidirectional","layout":"packed","loss":"CE"}
```

相比C_P_CE，设计上保持输入、LoRA/head、数据、曝光、optimizer、loss和layout不变，只改变pair内可见性。

### 12.2 代码怎样改变

共享执行层的非PB分支：

```python
native_fa3(...,
    cu_q=batch.cu, cu_k=batch.cu,
    causal=(mode == 'causal'),
    deterministic=True)
```

mode为bidirectional时实际传`causal=False`，每层调用一次。不是仅改模型卡说明，也不是只设置一个没有接到kernel的config开关。

### 12.3 新增了哪些可见边

- state前面的token能看到后面的question和当前candidate。
- question前面的token能看到candidate。
- 每个位置都能读取同一pair里所有位置。

没有新增跨candidate hidden边；各候选仍分别复制完整输入并由cu隔离。所以不是“所有候选组成一个双向集合编码器”。

### 12.4 是否拿训练好的causal权重直接开双向

不是。本组从共同fresh初态按bidirectional完整训练了同一预算。只切已训练causal模型的开关，不能作为本轮方法比较。

### 12.5 为什么没有缓存immutable prefix

因为prefix读suffix，candidate一变，深层prefix表示也可能改变。不能把causal缓存原封不动搬过来。

### 12.6 实际结果与解释边界

- 本地mass：48.1857%，比C_P_CE低11.4098个百分点。
- 官方Index：**3.09**，比C_P_CE低9.86点。

这证明的是当前骨干、低秩适配、数据和预算下的这个改法没有收益，不证明双向attention普遍差。因果预训练转双向是否需要不同参数覆盖/数据/预算，仅是待隔离的假设。

<a id="arm-pb"></a>
## 13. PB_P_CE：Candidate边界的两块双向

### 13.1 实际配置

```json
{"mode":"prefix_block","layout":"packed","loss":"CE"}
```

第一个PB是attention模式，第二个P是packed布局，不是两个prefix。

### 13.2 P和S到底包含什么

```text
P = 固定说明 + Input/state
  + Question（若有）
  + Available criteria（按原规则若有）
  + Candidate:\n

S = 当前candidate文本 + EOS
```

P里包含问题，并不是“只有state”。这点与Base原文的state/question分界不同。

### 13.3 为什么这样设计

首先沿用户给定commit的Candidate边界语义做同输入mask比较，而不是同时换成Base的新模板或LM-head。P不看S，S读完整P+S；两块内部双向。

### 13.4 实现细节

`bind_mask`从每条完整pair的prefix长度生成P/S真实索引和各自cu。28层复用同一份当前batch的BoundMask。每层QP读KP/VP，QS读本pair完整K/V；index_copy把互斥query结果写回原realT顺序。

没有把P/S分开各自softmax后相加，没有让另一个candidate的suffix进入当前key集合，也没有在训练中detach可训练LoRA的prefix图。

### 13.5 与C_P_CE、BI的不同

- 对C_P_CE：改变mask，同目标/布局。
- 对BI：阻止P读S，但S仍双向。
- 对STATE：PB的P截止Candidate标头，STATE的P只截止state后的已有LF。

**如果同一组sequence上P等于0或L，PB会退化，不能据一个不具代表性的长度说所有模式等价。训练使用完整非空P/S；本轮不是拿空suffix的prefill来代替完整候选训练。**

### 13.6 实际结果

- 本地mass：43.2434%。
- 官方Index：**2.59**，比C_P_CE低10.36点。
- 它不是Base整个配方的测试：没有Base的Qwen3.5混合结构、answer-slot LM-head、全参训练、CE+Brier和其数据配方。

<a id="arm-state"></a>
## 14. STATE_NL_PB_P_CE：只移动两块边界，不删除文本

### 14.1 实际配置与额外输入绑定

```json
{"mode":"prefix_block","layout":"packed","loss":"CE"}
```

再绑定精确boundary sidecar及policy：`state-plus-one-existing-newline-v2`。

**NL就是那一个原来已有的换行。没有新增special token、没有把问题删掉、没有增加一个state网络层。**

### 14.2 相对PB改变的只是P/S归属

```text
旧PB：
[说明 + state + Question + criteria + Candidate标头]P
[candidate + EOS]S

STATE_NL：
[说明 + state + 一个已有LF]P
[Question + criteria + Candidate标头 + candidate + EOS]S
```

因而问题/criteria现在在suffix块中，可以双向读取当前candidate；旧PB里这些token属于prefix，不能读candidate。这是真实的可见性变化，即使完整input_ids丝毫未改。

### 14.3 代码里不是重新tokenize，而是无损重切

```python
new_prefix = old_prefix[:cut]
new_suffix = old_prefix[cut:] + old_suffix
assert new_prefix + new_suffix == old_prefix + old_suffix
```

实际还核对EOS、完整长度、slot、candidate index/order、case/field身份与原pair_lengths。0<cut<len(old_prefix)，缺sidecar、重复或不可用known field就硬失败，不fallback到旧Candidate切点。

prefix长度变了；完整pair长度没变，所以stable-length micro排序和common admission不因这个边界变化而换组成。

### 14.4 为什么不是直接按state字符串长度切token

BPE token可能横跨state最后字符与后面的LF；字符数不是token数，单独tokenize一个更短prefix也不保证原IDs不变。

先前严格“仅state最后字符”版本实际发现train28676/41674、select427/615个known field跨界，因此整版notready，未训练。没有四舍五入切点、删除失败case或按答案挑成功版本。

新版本给所有field统一规定：state后的一个已有LF归P。重新完整offset投影，全部41674/615 known field可精确切，才准入训练。不是把原失败报告的ready改true。

### 14.5 它借鉴Base到什么程度

借鉴的是“state作为前块、problem作为后块”的边界思想。仍保留原单candidate pair和EOS二类head，不是Base每question全部options+answer-slot的输入/读出，也没有加入CE+Brier。

因此称**Base-inspired state-boundary-only**迁移。不能用本组1.97说Base论文被完整复现且失败。

### 14.6 实际结果

- 本地mass：44.3620%，比旧PB高1.1186个百分点，单seed描述性case区间跨0。
- 官方Index：**1.97**，反而比旧PB低0.62点。

它给出的实际结论是：这个固定边界变体未在本轮完整官方评测中带来收益。边界可精确恢复、真实kernel按新分区执行、训练完成，不等于学到了更强的Public能力。

<a id="training"></a>
## 15. 共同的数据、曝光、优化器与两遍训练

### 15.1 哪些条件八组都一样

| 项目 | 共同设置 |
|---|---|
| stock | 原本地ModelScope Qwen3-0.6B |
| fresh初态 | seed42、393张量匹配；不载旧项目训练adapter/head/optimizer/Teacher |
| train库存 | 24600完整case，41674 known / 5753 unknown field |
| D_select | 375case，615 known field，5190候选，90 unknown |
| D_confirm | 250case准备库存，但旧两候选仅各215条partial，未完整确认 |
| 逻辑曝光 | 99824365 pair tokens/arm |
| case occurrences | 56912/arm，16-case macro |
| optimizer calls | 3557，其中3556次LR>0，最后一次used_lr=0 |
| optimizer | fused AdamW，峰值2e-5，decay0.01，clip1，β=(0.9,0.999)，eps1e-8 |
| scheduler | 107次warmup，之后cosine到0 |
| micro | stable length、max32、共同padded预算16384、pad_multiple8 |
| 训练pair长度 | ≤8192；超限拒整个训练case，不截候选 |
| 开发/服务上限 | 40960，不截断 |
| 保存 | 每臂43checkpoint；最终官方固定m3557，不回选逐题赢家 |
| native/精度 | 同FA3与GQA，BF16autocast，原TF32/BF16reduction flags保持 |

“相同初始化”不等于训练完相同。loss或mask一变，梯度和后续权重轨迹会变；dense/packed即使实数数学相同，也可能因浮点几何而产生长期分歧。

### 15.2 数据clean的有限含义

按完整native component/兄弟字段、source/shape准入、许可和input-only membership检查。训练排除PUBLIC/FORBIDDEN/INCOMPLETE/UNKNOWN/NATIVE_DEV；dev另外排除HISTORY。合法HISTORY-only native train可从fresh stock训练，不等于八组继承了历史权重。

这不证明所有语义近重复、未知alias或Qwen预训练暴露绝无；tools主要procedural，也不能称为完整真实工具使用训练。

unknown是监督编译状态：保留raw，但不把unknown字段造为负例/forward。**正式public engine不读取gold/这些监督mask，仍回答全部提供的问题和候选。**

### 15.3 一次macro为什么需要两遍forward

CE/rank需要完整field全部margin，但一个大case的所有候选激活不宜同时长驻显存。因此：

1. 无梯度跑全部原micro，按slot保存FP32 logits/margins。
2. 在完整macro/field上计算`a_k=dL/dm_k`，含全部固定分母。
3. 同权重、同输入/顺序/微批重放，每个micro反传`sum(stopgrad(a_k)*m_k)`。
4. 每macro核对两遍logits/margins逐位相同、RNG未变；覆盖完整后clip并AdamW一次。

链式法则：

$$\nabla_\theta L=\sum_k a_k\nabla_\theta m_k.$$

每field的候选可能被执行微批拆开，但目标不是对每个micro单独softmax。不能在第二遍再除C、F或micro数。

### 15.4 预算单位

逻辑token是每个候选完整prefix+suffix+EOS只记一次；两遍forward实际为199648730 real tokens/arm，backward覆盖99824365。它不等于“GPU只前向处理了100M”，也不等于把24600case整齐看一轮。

实际冻结流56912 occurrences涉及17622不同case，最多重复17次。所有臂相同；没有为了某个较好分数重新采样容易的题。

同曝光也不等于同GPU时间：PB多一次attention分区，dense多算tokenwise pad；这些是应记录的系统代价，不是多学了额外样本的收益。

### 15.5 实际学习率公式

令即将执行的更新序号t为1…3557，w=107，峰值η=2e-5：

$$lr_t=\begin{cases}
\eta t/w,&t\le w,\\
\frac\eta2\left[1+\cos\left(\pi\frac{t-w}{3557-w}\right)\right],&t>w.
\end{cases}$$

最后t=3557用lr0；它仍计算梯度、推进Adam状态和曝光，但没有非零LR权重更新。不能把3557次step都称为3557次非零学习。

### 15.6 训练单卡与推理四卡不是一回事

每训练臂独立单GPU，macro16，没有DDP归约。曾经四卡同时训练四个臂，不等于一个模型四rank训练。

最终public是一个权重的四个推理副本，各处理35155条不相交请求；恢复140620原序后统一计分。不是四个模型投票/概率ensemble，也没有产生32套训练权重。

<a id="comparisons"></a>
## 16. 如何做正确的成对比较

下面箭头只表示比较配方，不表示加载左边已训练权重继续训右边。

```text
研究packing：C_D_BCE ↔ C_P_BCE；C_D_CE ↔ C_P_CE
研究loss：   C_D_BCE ↔ C_D_CE；C_P_BCE ↔ C_P_CE
研究mask：   C_P_CE  ↔ BI_P_CE；C_P_CE ↔ PB_P_CE
研究边界：   PB_P_CE ↔ STATE_NL_PB_P_CE
研究rank：   C_P_BCE ↔ JPAIR_P_BCE_RANK
```

| 想回答的问题 | 不合适的比较 | 为什么 |
|---|---|---|
| 加rank有没有好处 | 只拿JPAIR和C_P_CE比 | baseline loss也不同，应该先比C_P_BCE |
| 双向mask有没有好处 | BI CE比causal BCE | 同时改变mask和loss |
| state边界有没有好处 | STATE比C_P_BCE | 同时改loss、mask与边界 |
| packing有没有好处 | packed BCE比dense CE | 同时改layout和loss |
| 新clean是否因清理掉分 | 新fresh最高14.42直接比旧joint24.91 | 起点、数据/目标/预算/后端均有差异 |

最终D组官方消费者又由dense转packed，故public布局归因还要保留迁移失败的限制。可以比较本次消费者实测分数，不应假设原dense执行路径已被证明等价。

<a id="results"></a>
## 17. 结果、局限和不能混淆的结论

### 17.1 本地mass不是官方Index

本地D_select用五域case等权的argmax target mass：选中候选j，字段贡献q_j。soft标签上不是0/1准确率；官方Index还使用不同任务指标、机会基线和区域权重。

| 臂 | 原训练布局100M开发mass% | 完整Public Index |
|---|---:|---:|
| C_P_BCE | 57.5694 | 14.42 |
| JPAIR_P_BCE_RANK | 61.2437 | 13.60 |
| C_D_CE † | 61.6935 | 13.07 |
| C_P_CE | 59.5955 | 12.95 |
| C_D_BCE † | 56.4260 | 12.83 |
| BI_P_CE | 48.1857 | 3.09 |
| PB_P_CE | 43.2434 | 2.59 |
| STATE_NL_PB_P_CE | 44.3620 | 1.97 |

两个dense开发值来自原训练布局；不是把它们冒充最终packed数值准入已过。

### 17.2 †究竟保留什么限制

每case比较原dense终点输出与同权重新packed输出的logits、margin，使用冻结RMS/maxabs合同：

$$RMS(\Delta)\le0.01\max(RMS(ref),10^{-3}),$$

$$\max|\Delta|\le0.02+0.02\max|ref|.$$

还要求零带外choice翻转。两组均34/375 case违反数值合同；不是“只有34条运行error”，它们当时都完整执行且权重不变，是输出数值偏差违约。

最终用户要求八个都测，因此保留scientific-only标签跑完整public。全ok、分数有限、没有更新参数，都不能反向证明原迁移合同通过。

### 17.3 官方结果是完整执行，不是错误处罚造成的假低分

八组各完整140620条，官方排除442后计分140178条，全ok。32个分片正常退出并actual-wait；原序/完整输入/shard bytes与393权重after-proof核验均有凭据。

all-ok不是答对；Index3.09/2.59/1.97不能因此被解释为GPU没有运行。另一方面，执行完整也不等于模型所有语义路径都已跨后端数值等价。

### 17.4 这次只能得到有限结论

- packed causal BCE是这一组seed42终点的最高分。
- JPAIR没有比其匹配BCE控制带来官方收益。
- STATE边界没有比PB带来官方收益。
- BI/PB在当前stock+LoRA/head、数据和预算下明显较低；不能证明所有双向模型或完整Base/Jina无效。
- 只有单seed，没有跨seed稳定性证明。case-bootstrap不是训练seed区间。
- 低分具体由数据分布、适配参数、预算、优化还是数值路径敏感交互造成，尚无唯一因果结论。

正式headline是public0.3的加权`balanced_skill`，不是总体accuracy或几何`breadth_skill`。没有private Full、维护方正式名次或生产latency准入结论。

<a id="sources"></a>
## 18. 实际代码位置与自查问题

### 18.1 最关键的“代码开关”只有几条

```python
# model_runtime.FullRow
core = MaskedBackbone(backbone, head, native, arm['mode'])
batch = make_batch(rows, layout=arm['layout'])

# objectives
coef = coefficients(all_complete_field_margins, fields, cases, arm['loss'])

# STATE additionally
pair = Pair(..., original_prefix[:cut], original_prefix[cut:] + suffix)
```

但“开关少”不代表只改三行就完成实验：还需要布局隔离、字段目标、精确边界、梯度重放、初始化/数据/曝光一致与完整官方测分。八组不是八份互不相关的手写网络。

### 18.2 仓库内可直接读的源码

- [基础 BCE/Field-CE](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/objectives.py)：第7～35行。
- [JPAIR 与原始host q分类](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-followup-v1/source/objectives.py)：第7～74、135～146行。
- [共享 attention/forward 执行层快照](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/native-mask/packed_mask/model.py)：第73～102行三mask，146～183行Qwen3/PEFT tokenwise与head。
- [dense/packed与EOS索引](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/native-mask/packed_mask/layout.py)：第70～109行。
- [FullRow对接mode/layout](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/model_runtime.py)：第68～126行。
- [精确state无损重切](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-followup-v1/source/case_runtime.py)：第195～232行。
- [fresh LoRA/head](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/fresh_init.py)：第35～81行。
- [两遍实际训练](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/train_case.py)：第60～158行。
- [精确warmup/cosine](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/checkpoints.py)：第68～104行。

仓库快照用于理解代码；真实运行绑定以服务器冻结T/F/P配置、source与回执为准，不建议直接重启旧root来“复现”。

### 18.3 真实运行与结果路径

```text
原六臂生产者：/tmp/opencode/jev-clean-trainer-20261112-v2/
后两臂生产者：/tmp/opencode/jev-clean-trainer-20261112-followup-v1/
正式八臂消费者：/home/dev/assumeengage/runs/jev-public-eight-fourway-20261113-v1/
```

- 实际配置：生产者`seed42.json`与消费者`config.json`。
- 真实终点：各`<arm>-seed42/checkpoint-m0003557-s00043/`。
- 原数据、sidecar和全部checkpoint留在服务器；本文不复制私有样本。
- [官方汇总](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/SCORES.json)。
- [完整独立核验](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/FINAL_OFFICIAL_VERIFIED.json)。
- [总体研究报告](REPORT.md)包含材料引用、完整过程、推理优化、失败记录与五域分解。
- PyTorch语义参考：[2.9 BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html)。本次Context7额度不可用，回到官方文档；执行事实以安装/冻结源码为准。

### 18.4 判断是否读懂的十个问题

1. `PB_P_CE`的第一个PB和中间P为什么不是一个意思？
2. 为什么`D`不能解释成double precision或全参数dense模型？
3. head只有2个输出，为什么仍能回答有255候选的field？
4. 为什么CE可以改变候选之间的梯度竞争，却不让候选hidden互相attention？
5. `RANK`与LoRA的r32有什么区别？
6. 为什么同一个EOS在三种mask都能看到前文，深层输出却仍不同？
7. STATE保留了全部token，为什么仍是模型语义变化？
8. 为什么D组的官方数值可以报告，却不能说原dense→packed等价已通过？
9. 为什么JPAIR应该先与C_P_BCE比较，而不是只与C_P_CE比较？
10. 为什么“一套八臂、32个推理worker”不是“32个新训练模型或一个ensemble”？

---

**结论：这八个名字主要编码三条实验轴——怎样排布token、允许token看谁、用什么目标更新参数；STATE再单独移动prefix/suffix边界。共同骨干、LoRA/head、完整pair、数据与曝光大体固定。它们是八个独立训练对照，不是一串连续升级版本。**
