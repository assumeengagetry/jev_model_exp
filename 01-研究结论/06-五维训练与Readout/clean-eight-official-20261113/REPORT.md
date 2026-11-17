# Qwen3-0.6B clean 八臂实验：从你给的材料到官方成绩

> 2026-11-15 按学习用途重写。实验于 2026-11-12～13 完成，原报告整理于 2026-11-14（服务器原件 `/home/dev/assumeengage/reports/jev-clean-eight-detailed-learning-report-20261113.md`）。
>
> 官方数字已与[证据目录](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/README.md)里的 `SCORES.json`、`scores/*.json`、[344 行逐 benchmark 表](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/official-benchmarks.csv)和[40 行五域表](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/official-areas.csv)逐项核对，保留官方文件的两位小数。本地开发集、训练账本和后端计时来自服务器记录，仓库里只有本文的转述。
>
> 八个分支的名字、mask 和 loss 公式在 [EIGHT-ARMS-EXPLAINED.md](EIGHT-ARMS-EXPLAINED.md)，本文不重复。

## 结论

1. **八个新训练的权重都完整跑完了官方 Public 0.3。**每个权重 140,620 条请求全部 `ok`，官方排除 442 条后计分 140,178 条。
2. **最高的是 `C_P_BCE`，Index 14.42。**离当初设定的参考 51.99 差 37.57 点，冲榜目标没有达到。它也低于旧谱系的 24.91，但两者的起点、数据、预算和后端都不同，不能直接比。
3. **改 attention 可见性的三个分支分数很低。**五个 causal 分支在 12.83～14.42；全双向和两种两块双向只有 1.97～3.09。
4. **本地开发集排名和官方排名对不上。**开发集最高的 `C_D_CE`（61.7%）官方排第三；官方第一的 `C_P_BCE` 在开发集排第四。JPAIR 和 STATE_NL 在开发集上都比各自的对照高，官方分数反而更低。
5. **`C_P_BCE` 领先 `C_P_CE` 的 1.47 点里，约 1.37 点来自 Tools 区域，主要是 BFCL**（case-exact 原始分 0.44 → 0.64）；Knowledge 区域它反而略低。这一条是本次重写时从证据 CSV 补算的，原报告没有。
6. **packing 去掉了约 25% 的无效计算，但没有改变分数。**在旧权重上做的同权重 A/B：dense 24.71，packed 24.70。
7. **推理后端的第二版优化让前 1,000 条请求的平均延迟降低约 10%～11%；第一版反而慢了 8%～9%。**
8. **四卡跑同一个权重后，第一个完整成绩 1 小时 43 分就出来了；八个全部跑完用了 14 小时 38 分。**
9. 只有 seed 42 一次训练；没有跑 private Full，也没有提交榜单。

## 阅读顺序

| 想知道 | 读哪里 |
|---|---|
| 结果和结论 | 上面的结论 → [第 13 节结果](#results) → [第 14 节没解释的问题](#open-questions) |
| 八个模型各改了什么 | [EIGHT-ARMS-EXPLAINED.md](EIGHT-ARMS-EXPLAINED.md)，再读本文第 1、3～5 节（改动的来历） |
| 数据和训练 | 第 6～8 节 |
| 数值一致性、评测规则 | 第 9～10 节 |
| 推理工程和多卡调度 | 第 11～12 节 |
| 对照源码 | [第 16 节](#code)和附录 |

文中的小例子和伪代码只用来解释概念，不是真实样本，也不是另外跑的实验。

<a id="materials"></a>
## 1. 你给的材料，各变成了什么改动

这次没有把几份材料混成一个大模型，而是拆成可以单独归因的变量。

| 材料或要求 | 想回答的问题 | 实际做了什么 | 没有做到的 |
|---|---|---|---|
| 尽量全局 padding-free | 不只是 attention，Embedding、LoRA、MLP 是不是也在算 padding？ | 建真正的 packed 布局，整个主干只算真实 token；在同一个 kernel 上做 dense / packed 对照 | packing 本身不提分，也不和 dense 逐位相同 |
| 你给的 commit `2e334fb`（本仓库 attention-mask-modes 补丁） | causal、bidirectional、prefix_block 能否在 H800 / FA3 上真跑、真反传？ | 原样保留该版本做三模式机制实验；之后改成同一个 native FA3 的 packed 执行层 | 该 commit 本身不是完整 padding-free；FA3 对 SDPA 的整模型数值对照没有全部通过 |
| PyTorch FlexAttention 博客和 2.9.1 文档 | 任意可见性能否不建巨大的稠密 mask？ | packed causal 的编译和前反向机制实验；接好 Jina 交互组件 | 没有进入最终官方后端，数值对照没有全部通过 |
| Base Labs 的 Mammoth / Elephant 报告和 vLLM PR | state 与 question 两块双向、prefix 不看 suffix，能否迁移过来？ | 实现 PB；另做把分界放在 state 后的 STATE_NL | 不是 Qwen3.5 混合架构、answer-slot LM head、CE+Brier 和原数据配方的复现 |
| Jina reranker v3 模型卡、源码和论文 | 候选交互和排序目标，哪部分能单独迁移？ | J0 / J1 / J2 机制接线；八臂里完整训练的只有排序目标 JPAIR | 候选同窗交互的 J1 没有完整训练和官方评测 |
| 知乎文章 | 正文方法能否拿到？ | 当时 HTTP 403，没读到正文 | 没有凭标题猜内容 |
| 另一个 session 的推理优化记录 | 哪些工程手段能复用？ | 词表长度缓存、CPU 端元数据、autocast 生命周期、请求末一次回传、正确的 PEFT helper | 没有恢复未通过数值核对的 KV fork、融合算子、padding Graph、量化和 merge |
| 你要求"八个都测""先尽快看到一个完整成绩" | 验收看全量官方分数，还是小开发集？ | 停掉只晋级两个候选的旧流程；八个全测；四卡同权重分片，逐个计分 | — |

外部数字不能和我们的直接比：

- Base 原文在 9B、同数据、10k 条 public 样本上的消融：causal 44.8，block-bidirectional 48.6。这不是 0.6B 的全量结果。
- Jina 报告的是 BEIR 上的 nDCG，不是 JEV Public Index。
- 51.99 是当时冻结榜单上 served_params ≤ 0.7B 的 GLiClass v4.1 edge，不是重新核对过的最新第一名。

<a id="task"></a>
## 2. 任务形状和运行环境

每个候选变成一条独立的"是否匹配"输入，在 EOS 位置读一个二分类 head；同一个 field 的候选 margin 做 softmax 得到概率。模型不生成文本，也不从文本里解析答案。官方 runner 只把 `state` 和 `questions` 交给模型，gold 不作为输入。模板、margin 和 softmax 的细节见 [EIGHT-ARMS 第 4 节](EIGHT-ARMS-EXPLAINED.md#readout)。

不同 case、field、候选的 pair 之间没有 attention 连接，物理上放进同一个张量也一样。候选之间的显式交互是另一个建模变量。

官方评测时保存的运行环境：

| 项目 | 值 |
|---|---|
| GPU | 四张 NVIDIA H800 NVL，每个 worker 独占一张 |
| PyTorch / CUDA | 2.9.1+cu128 / 12.8 |
| Transformers / PEFT | 4.57.6 / 0.18.1 |
| attention | 冻结的 native FA3，没有换别的后端 |
| 确定性 | deterministic algorithms 开；cudnn deterministic 开、benchmark 关 |
| world_size | 每个 worker 都是 1，不是四卡 DDP |
| prefix cache | 关 |

原模型来自 ModelScope `Qwen/Qwen3-0.6B`，下载记录的 revision 是 master，没有核对它等同于 HF 上的哪个 commit。

<a id="padding-free"></a>
## 3. Padding-free：去掉的是无效计算

### 3.1 只在 attention 里去掉 padding 不够

旧的 collate 构建 `[B, ceil(maxL/8)*8]` 的张量。FA3 wrapper 虽然在 attention 内部去掉了 padding，但 Embedding、Norm、QKV/LoRA、MLP 和 head 仍在处理 padding 行。例子见 [EIGHT-ARMS 第 5 节](EIGHT-ARMS-EXPLAINED.md#layouts)。

### 3.2 改成了什么

```text
flat_ids[T]                                 T = 所有 pair 的真实长度之和
  ↓ embedding
hidden[T, 1024]
  ↓ 每层：Norm / QKV+LoRA / QK norm / RoPE
native FA3 varlen（用 cu 划分每条 pair）
  ↓ o+LoRA / residual / Norm / MLP+LoRA / residual
所有真实 token 过 head → 取各自 EOS 的 logits → margin → 字段 softmax
```

位置在每条 pair 内从 0 重新开始，不能用全局偏移当 RoPE 位置。`cu = [0, L1, L1+L2, …]` 是序列边界；EOS 下标是每条序列最后一个真实 token。"全局 padding-free"指的是没有假 token 进入这些模块，不是指硬件 tile 或显存分配器没有任何对齐。

### 3.3 为什么 packed 不顺便把 batch 加大

四个基础分支用同一套 micro 规划（按长度稳定排序、每个 micro 最多 32 条、按 dense 补齐后的 16,384 token 预算）。如果 packed 同时把 micro 加大，就既改了布局又改了浮点计算的形状。"同样曝光下的学习质量""同样 GPU 时间能多学多少""吞吐"是三个不同的问题，要分开比。

两遍前向合计，逐 token 模块的输入行数 dense 为 266.5M，packed 为 199.6M，少了约 25%。行数少不等于同比例的墙钟提速。

### 3.4 旧权重上的同权重 A/B

在旧谱系的同一个权重上，分别用 dense 和 packed 推理跑完整 public：

| | dense | packed |
|---|---:|---:|
| Public Index | 24.71 | 24.70 |
| 平均每条请求 | 约 118 ms | 约 102 ms |
| 四卡完整流程 | 4263 s | 3705 s |

两边都是 140,620 条执行、140,178 条计分、全部 ok，分数没有变化。但逐题看并不等价：14,801 条请求超出数值容差，376 条请求里有 378 个 field 的选项翻转，其中 21 个是带外翻转。总分只差 0.01，不代表每题的概率都一样。这组旧权重有已知的历史数据暴露，它的分数不能当作 clean 分数。

<a id="base"></a>
## 4. Attention 改动的来历：借鉴 Base

三种 mask 的定义、矩阵和 PB 在 FA3 上的两次调用见 [EIGHT-ARMS 第 6 节](EIGHT-ARMS-EXPLAINED.md#masks)。这里讲分界是怎么来的。

### 4.1 PB 的分界和 Base 原文不同

```text
本项目的 PB：P = 固定说明 + state + Question + 必要的 criteria + "Candidate:"    S = 当前候选 + EOS
Base 原文：  P = system + state                                               S = question + 全部选项 + answer-slot
```

Base 还用了 Qwen3.5 的 GDN / attention 混合结构、选项字母的 LM head 和 CE+Brier 配方；本项目是 28 层全 attention 的 Qwen3-0.6B，每个候选一条独立 pair，EOS 二分类 head，LoRA 训练。只迁移 mask 或分界，不算复现 Base。

### 4.2 分界放在 state 最后一个字符：没法精确切

最初想让 P 截止在 `模板头 + state` 的最后一个字符。但 BPE 可能把 state 末尾的标点和后面的换行编成一个 token，字符位置不一定落在 token 边界上。CPU 上完整检查后发现：训练集 41,674 个有标签 field 中 28,676 个、开发集 615 个中 427 个都跨了这个边界（约七成）。这一版判为不可用，没有训练；也没有四舍五入切点、删题或悄悄退回旧分界。这是"不改 token 就无法精确实现这个分界"的准备失败，不是 Base 方法训练失败。

### 4.3 最终的 STATE_NL

模板里 state 后面本来就有一个换行。新规则统一把它归入 P：

```text
P = 模板头 + state + 那个原有的 '\n'
S = 其余的 Question / criteria / "Candidate:" / 候选 / EOS
```

没有新增换行，没有重新分词，完整 `input_ids` 不变，只是把一个已有的分隔符划给 P。所有训练和开发样本用同一规则，不逐题挑容易成功的版本。用 tokenizer 的 offset 找到精确的 token 边界后无损重切（`newP = oldP[:cut]`，`newS = oldP[cut:] + oldS`），并核对每条 pair 的 ID、EOS、长度、位置、slot、micro 和监督都不变。全部 41,674 / 615 个 field 都能精确切开，才训练 `STATE_NL_PB_P_CE`。

### 4.4 为什么最后没有加回 prefix KV cache

- causal 的 prefix 不读后文，在满足原数值要求时可以复用。
- BI 的 prefix 会读 suffix，候选一变，深层的 prefix 表示就变了，不能缓存。
- PB 的 prefix 不读 suffix，但 prefix 内部双向，只能复用**完整且完全相同**的 P 块，不能像 causal 那样按最长公共前缀命中。
- STATE_NL 改了 P 的定义，原来针对 Candidate 分界做的缓存验证不能直接套用。

本轮 prefix-block 的 full / paged / fork 缓存路径都有没解决的数值失败，所以八个分支统一用不带缓存的整行 native FA3。代价是每个候选都要重算一遍 prefix。

<a id="jina"></a>
## 5. 借鉴 Jina：做了哪些，没做哪些

### 5.1 Jina v3 本身是什么

核对固定版本的源码和模型卡：Jina v3 基于 Qwen3-0.6B，**保持 causal**；多个文档按顺序放在同一个窗口里，query 在开头和末尾各出现一次；后面的文档能读前面的，末尾的 query 能读全部文档；文档和 query 末端的 hidden 经共享 projector 后算 cosine 分数。

模型卡写 projector 是 1024→512→256，固定版本的源码实际是 1024→512→512，本文以源码为准（没有加载权重核对）。它的 131K 上下文、特殊 token 和读出方式都不能不经训练直接搬过来。本轮没有下载 Jina 权重；它的 CC-BY-NC-4.0 许可也不因为"借鉴思路"而变成可商用。

### 5.2 拆成 J0 / J1 / J2

- **J0**：布局、位置和后端都匹配、但候选互相隔离的控制组。
- **J1**：同一 field 的候选按原顺序在一个窗口里 causal 可见；保留 EOS head 和原 BCE，只研究交互。
- **J2**：attention 同 J0，只加排序目标。

三者都用 prefix-once 布局，在同一个 Flex 后端上比较，避免"J0 用 FA3、J1 用 Flex"时分不清变化来自交互还是换了 kernel。

实际做了一次真实 macro 的接线（16 个 case、19 个有标签 field、60 个候选）：三组两遍 logits 逐位一致，393 组梯度和更新都是有限值；物理 token 10,342，逻辑 pair token 20,947。这次加载的是历史诊断权重，只用来验证机制，不是新训练的结果。

### 5.3 八臂里的 JPAIR 是哪一部分

`JPAIR_P_BCE_RANK` 保持独立 pair、causal、native FA3 和 EOS head，只把目标改成 BCE＋硬标签排序项。它能回答"固定数据、初始化和曝光时，这个组合目标有没有收益"，不能回答"完整的 Jina 方法是否更好"。J1 的完整训练、候选换序实验和超过 40,960 token 的 field 都没有做。

<a id="data"></a>
## 6. 数据：为什么重建起点，怎样清理

### 6.1 为什么不沿用旧权重

之前较好的 joint 模型 Public 约 24.91，但历史上选进训练的 xLAM 和 ToolRet 之间有 94 条完全相同的请求，涉及 98 条执行请求；旧的 case 图也没有对称地拆开 JSON 里的 `request` 包装。所以旧谱系还能做机制和效率诊断，但不能当新对照的起点。

新八臂全部从原模型出发，用 seed 42 重新生成 LoRA 和 head，没有加载项目历史上的 adapter、head、optimizer 或旧 Teacher 标签缓存。也正因如此，新结果低于 24.91 不能只归因于清理：起点、监督、数据范围、预算和后端都变了。

### 6.2 为什么不能只按文件去重

一个原始 case 可能有多个 field、变体或派生视图。如果一个 field 进了训练、兄弟 field 进了开发集，文本不完全相同也可能共享同一个任务内容。

实际做法：完整的原始坐标普查、组件和兄弟字段闭包、来源和输入形状的白名单、许可核对、只看输入的负向成员检查，以及对已知 JSON `request` 包装的对称拆包。兄弟坐标不明或来源未决时，不能只删掉那一个麻烦的 field 就放行整个 case。源码里的禁止标记：

```text
所有 split 都禁止：PUBLIC | FORBIDDEN | INCOMPLETE | UNKNOWN
训练集另外禁止：  NATIVE_DEV
开发集另外禁止：  HISTORY
```

只带 HISTORY 标记的合法训练 case 可以用来从头训练，但不能进开发集。所以"clean"不等于"训练样本从未在本项目历史中出现过"。public 输入只用来做负向排除，public 的 gold 从未用于训练。

### 6.3 最终冻结的数据

| split | case | field | 有标签 | 无标签 |
|---|---:|---:|---:|---:|
| train | 24,600 | 47,427 | 41,674 | 5,753 |
| D_select（开发） | 375 | 705 | 615 | 90 |
| D_confirm（确认） | 250 | 439 | 389 | 50 |

训练集五个领域的 case 数：Knowledge 4,976、Language 4,950、Retrieval 4,937、Tools 4,766、Arts 4,971。开发集每个领域 75 个，确认集每个领域 50 个。Tools 的训练数据主要来自程序化生成的来源，不是原生 BFCL 训练，也不能证明真实工具使用的泛化。

- `choice` 保留完整的原选项和原目标分布 q。
- 合法的 `noul` 按已证明的语义方向展开成 Yes. / No. 两个候选，目标是 `[q, 1−q]`。
- 分数方向不明或目标不合法的 field 记为无标签（`target=None`），不分词、不前向，也不造成全零的负例；同一 case 的其它合法 field 照常保留。
- 有标签 / 无标签只影响训练和开发集的监督。**正式评测不读 gold，所有问题和候选都要回答。**

训练 pair 超过 8,192 token 时整个 case 都不进训练，不截断、不删候选。开发集保留完整原题，本次开发集的有标签 field 全部在 40,960 的上限内。

### 6.4 100M 曝光量是怎样排的

- 每个领域的逻辑 pair token 上限是 20M。
- 每个 family 的权重是其合法 case 数的平方根，family 内均匀抽 case。
- data seed 20261112 冻结了曝光序列；训练 seed 42 和 43 只改变顺序、不重新抽样。本轮只训练了 seed 42。
- 每次更新取 16 个完整 case；预算尾部放不下一个完整 macro 就停，不临时挑短 case 凑满。

实际消费了 56,912 次 case 出现，涉及 17,622 个不同 case，最多的重复 17 次。不是 24,600 个 case 各看相同次数。八个分支用同一个曝光流。

### 6.5 clean 证明到什么程度

证明了：声明范围内的来源和形状准入、完整的坐标和组件、已知的请求包装、精确的表示，以及完整 pair 的跨 split 检查。没有证明：所有来源的语义近重复、未知别名、Qwen 预训练时的数据暴露都不存在。许可检查也只覆盖声明的范围。

<a id="training"></a>
## 7. 训练流程

### 7.1 为什么要两遍前向

field-CE 和排序项的分母需要同一个 field 的全部候选。如果一半候选在 micro 1、一半在 micro 2，各自 softmax 再平均，就已经不是原来的目标了。可是把一个大 macro 的全部激活一直留在显存里又会 OOM。所以用两遍前向加 margin 的 VJP（向量—雅可比积）。

### 7.2 实际流程

```python
# 说明性伪代码：同一个 macro，更新前的同一份权重
m_first = no_grad_forward_all_micros()          # 第一遍：只保存每个候选的 margin
coef = autograd_d_loss_d_margin(m_first)        # 在完整 field 和 case 上算 dL/dm，包含全部分母

optimizer.zero_grad()
for micro in exactly_the_same_micro_plan:
    m_replay = forward_with_grad(micro)         # 第二遍：逐个 micro 留计算图
    require_replay_exact(m_replay, m_first)     # 两遍的 logits / margin 必须逐位相同
    (m_replay * stop_gradient(coef_for_micro)).sum().backward()

clip_grad_norm_(trainable_parameters, 1.0)
optimizer.step()                                # 每个 macro 只更新一次
```

依据是链式法则：

$$\nabla_\theta L=\sum_k\frac{\partial L}{\partial m_k}\nabla_\theta m_k.$$

只要两遍用同一份权重、同样的输入、顺序和形状，dropout 为 0，并且实际重放逐位一致，就可以先拿到完整目标的 dL/dm，再分 micro 累加参数梯度。分母已经在 coef 里了，第二遍不能再除以 micro 数、case 数或 DDP world size。每个分支是单卡独立训练，不是四个分支一起做四 rank 的梯度归约。

### 7.3 训练不能照搬推理的 autocast 缓存

第一遍无梯度，第二遍要梯度。如果两遍之间保留了无梯度阶段生成的 autocast 权重缓存，可能污染可微路径。所以训练时两遍用各自独立的 autocast 作用域；第 11.4 节讲的"worker 全程 no_grad 加 autocast 缓存"只适用于没有 backward 的推理。

### 7.4 训练参数

| 参数 | 值 |
|---|---|
| macro | 16 个完整 case |
| micro | 按长度稳定排序，最多 32 条 pair，共同的补齐预算 16,384 token，按 8 对齐 |
| 优化器 | fused AdamW，峰值 LR 2e-5，weight decay 0.01，β = (0.9, 0.999)，eps 1e-8 |
| 梯度 clip | 全局范数 1.0 |
| 学习率 | 前 3%（107 步）线性 warmup，之后 cosine 降到 0 |
| 精度 | LoRA / head 以 FP32 保存；BF16 autocast 前向；dropout 0 |
| 数值设置 | matmul / cudnn TF32 开；BF16 reduced-precision reduction 开；float32 matmul precision = high |
| 环境 | `CUBLAS_WORKSPACE_CONFIG=:4096:8`，FA3 严格确定性 |
| checkpoint | 第 0 步、第一个 macro、每 100 个 macro、观察点和终点，每个分支 43 个，全部保留 |

这些数值设置在所有分支里保持不变，没有把"改精度"夹进 loss 或 mask 的比较里。

学习率公式（t = 1…3557，w = 107，η = 2e-5）：

$$lr_t=\begin{cases}\eta\,t/w, & t\le w\\[2pt] \dfrac\eta2\left[1+\cos\left(\pi\dfrac{t-w}{3557-w}\right)\right], & t>w\end{cases}$$

### 7.5 预算账本

| 名义上限 | 实际逻辑 pair token | case 出现次数 | optimizer step |
|---|---:|---:|---:|
| 10M | 9.99M | 5,840 | 365 |
| 30M | 29.99M | 17,952 | 1,122 |
| 100M（终点） | **99.82M** | **56,912** | **3,557** |

逻辑 token 对每条完整 pair 只记一次，包括每个候选重复的 prefix 和 EOS。两遍前向实际输入 199.6M 个真实 token，反向覆盖 99.8M；dense 分支还要多算 padding 行。

3,557 次 step 里只有 3,556 次 LR 大于 0：最后一步 LR 为 0，仍计算梯度、更新 Adam 状态、计入曝光。八个分支都一样，没有为了账本好看把最后一步去掉。第一个 macro 只消费一次，保存完两遍输出、系数、393 组梯度、Adam 状态和 checkpoint 后暂停再续跑；后来的资源失败和控制器替换都没有重做它。

<a id="dev"></a>
## 8. 本地开发集：指标和三个观察点

### 8.1 主指标：argmax 目标质量（mass）

$$M=\frac15\sum_d\frac1{|D_d|}\sum_{c\in D_d}\frac1{F_c}\sum_{f\in known(c)}q_{cf,\arg\max_km_{cfk}}$$

模型选中哪个候选，这个 field 就得到该候选的目标值；先在 case 内平均，再在每个领域内平均，最后五个领域等权平均。例：q = [0.7, 0.3]，模型选第一个，贡献 0.7。它不是 `sum(q·p)`，只有 one-hot 时才等于正确率。另外还报 NLL、Brier、NLL / log K 和覆盖率；所有 field 都有标签且都是 one-hot 的 case 只有 259 个，在这 259 个上另报 case 全对率。

### 8.2 三个观察点的结果

每次都是完整的 375 个 case、615 个有标签 field、5,190 个候选；90 个无标签 field 不前向。数字来自训练时保存的 logits 和 margin，之后在 CPU 上独立回算核对过。

| 分支 | 10M | 30M | 100M |
|---|---:|---:|---:|
| `C_D_BCE` | 47.5% | 54.0% | 56.4% |
| `C_P_BCE` | 49.7% | 53.1% | 57.6% |
| `C_D_CE` | 53.9% | 54.9% | 61.7% |
| `C_P_CE` | 52.1% | 55.7% | 59.6% |
| `BI_P_CE` | 37.6% | 43.5% | 48.2% |
| `PB_P_CE` | 41.2% | 41.0% | 43.2% |
| `STATE_NL_PB_P_CE` | 40.8% | 41.6% | 44.4% |
| `JPAIR_P_BCE_RANK` | 50.9% | 55.7% | 61.2% |

### 8.3 观察

- 四个基础分支从 30M 到 100M，mass 都上升，NLL 都变差；Brier 有三个变差，只有 `C_P_BCE` 从 0.58 降到 0.57。不同指标的走向不一致。
- packed 相对 dense 的高低随 loss 和观察点变化，看不出"去掉 padding 就更会做题"。
- 100M 时 JPAIR 比 `C_P_CE` 高 1.6 个百分点，case bootstrap 区间 [−1.9, +5.1]；STATE_NL 比 PB 高 1.1 个百分点，区间 [−3.8, +6.0]。两个区间都跨过 0。
- 这里的 bootstrap 是在同一次训练的固定输出上，对每个领域 75 个完整 case 有放回重采样 10,000 次，不是训练 seed 的置信区间，也没有做多重比较校正。

开发集的 61.7% 和官方 Index 不是一个东西，不能拿来和 51.99 比。第 13 节会看到两者排名并不一致，原因之一是算法不同：开发集五个领域等权，官方区域权重不等，而且官方用各任务原生指标、机会校正和覆盖率。

### 8.4 D_confirm 为什么没有结论

后期曾定下"`C_P_CE` 作基线、JPAIR 作挑战者"的两候选确认流程。实际两个目录各只跑了 215 / 250 个 case，没有完整的汇总。你后来明确要求八个权重全部直接跑官方评测，于是停掉了这个流程，保留了不完整的结果。之前"最多两个候选、先确认再 public"的安排和你要验收全部权重的目标不一致，这是流程上的纠正。所以本轮没有 D_confirm 结论。

<a id="numerics"></a>
## 9. 数值一致性：哪些通过，哪些失败

### 9.1 "能跑"和"等价"是两回事

```text
语法通过
  ≠ GPU 上真的执行和反传了
  ≠ 输出和梯度在容差内
  ≠ 换了 batch 形状仍逐位相同
  ≠ 学得更好
  ≠ 官方分数更高
  ≠ 生产或官方 latency 准入
```

同一个数学式在不同 batch、kernel、切片或矩阵形状下，不保证逐位相同；反过来，数值不同也不能证明发生了跨样本泄漏。

### 9.2 容差事先冻结

| 对象 | 判据 |
|---|---|
| hidden / logits / margin 的 RMS 误差 | ≤ 0.01 × max(参考 RMS, 1e-3) |
| 最大绝对误差 | ≤ 0.02 + 0.02 × 参考的最大绝对值 |
| macro loss | ≤ 1e-3 × (1 + 参考 loss 的绝对值) |
| 每组梯度 / 更新的 L2 误差 | ≤ 0.02 × 参考范数 + 1e-7 × sqrt(元素数) |
| 非近零的梯度 / 更新方向 | cosine ≥ 0.999，近零组单独记 |
| 选项敏感带 | 参考的 top1 − top2 差 ≤ 0.04 + 0.04 × 该 field margin 的最大绝对值 |

落在敏感带内的选项翻转叫"带内翻转"，说明这个 field 对规定范围内的数值漂移很敏感，不能因此算作预测等价；敏感带外的叫"带外翻转"。只有一个候选的 field 没有 top2，不造假的差值。

### 9.3 实际的失败

- padding-free 第一版在真实 macro 上没过，保留记录；修正版没有回写成"第一版通过"。
- commit `2e334fb` 的 FA3 对 SDPA，三种模式都没过整模型对照，连 causal 也有漂移，所以不能只怪新 mask。
- Flex causal 编译、前向、反向、393 组梯度和一步更新都执行了，但 logits、梯度、更新没有全部过容差，没有用作最终后端。
- 同一个 native FA3、同一个 macro 形状下的 dense / packed 对照，梯度和更新都在容差内；换成同形状的真实邻居样本时，目标输出和 VJP 逐位不变，支持"内容互相隔离"这一有限判断。
- 单独跑与混合跑的形状稳定性、prefix-block 的 fork / paged 缓存，都有失败，保留记录。

所以后来允许的是"整行科学训练"：在同一个 FA3、明确的输入、位置和分界下做匹配的学习对照。这不等于把早期失败都改成通过。

dense 训练的两个权重迁移到 packed 推理也有失败（各 34 / 375 个 case），见 [EIGHT-ARMS 第 9 节](EIGHT-ARMS-EXPLAINED.md#dagger)。

<a id="index"></a>
## 10. 官方 Public Index 怎么算

### 10.1 这次固定的东西

- public suite 版本 **0.3**；执行并报告 43 个 benchmark，其中 37 个进入 Index。
- 每个权重完整执行 140,620 条，官方排除 442 条，计分 140,178 条。
- MMLU、ARC-Easy、ARC-Challenge、SimpleBench、RouterBench、SGD 这 6 个照常执行和显示，但不进 Index；不能因为它们分数高就推断 headline 高。
- 复用已严格验证过的 suite 和排除清单。SDK 来源声明的 revision 是 `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`，冻结的 SDK 文件本身不是 git checkout。
- 模板、候选、EOS、FP32 margin 和双精度字段归一化都不变；没有按对错重试、删候选、混入旧预测、按题路由或挑 checkpoint。
- 原生的 unsupported、abstained、普通 error 会按 SDK 规则受覆盖率惩罚。本次八个分支恰好全部 ok，不是事先删掉了非 ok 的结果。

### 10.2 Index 不是统一的准确率

每个 benchmark 先按自己的原生指标算分，可能是 accuracy、macro-F1、nDCG、整个 case 全对等。BFCL 的 case 全对和"单个 field 对了一半"不是一回事。然后做机会校正：

$$r_b=\text{原生分数（经覆盖率调整）},\qquad s_b=\operatorname{clip}_{[0,1]}\frac{r_b-c_b}{1-c_b}.$$

例：四选一，机会水平 0.25，原始分 0.40，skill = (0.40 − 0.25) / (1 − 0.25) = 0.20。这个 20 是相对机会水平的提升，不是 20% 的准确率。机会基线按任务结构定义，不一定是 1/K；已经在指标内部给失败组记零的 track 类任务，不再重复乘覆盖率。具体以冻结的 SDK 为准。

### 10.3 五个区域怎样汇总

区域内 gold★ benchmark 权重 1.2，其余 1.0，归一化后对 skill 加权平均得到区域分 $S_a$。五个区域的权重和 benchmark 数：

| 区域 | 权重 $w_a$ | 进入 Index 的 benchmark |
|---|---:|---:|
| Knowledge & Reasoning | 0.258494 | 11 |
| Language Understanding | 0.258494 | 9 |
| Retrieval & Classification | 0.200229 | 6 |
| Tools & Automation | 0.182783 | 5 |
| Arts & Human Taste | 0.100000 | 6 |

（benchmark 数是本次重写时从官方 `scores.json` 读出的。）

$$\text{Public Index}=100\sum_aw_aS_a=\texttt{balanced\_skill}.$$

这是加权算术平均，不是几何平均。SDK 另外报告 `breadth_skill`（一种几何平均）和用原始分代替 skill 的 `raw_index`，两者都不是 headline。SDK 中间有舍入，用已经四舍五入的展示值重算出略不同的数字是正常的。

### 10.4 Public 0.3 和榜单的 Full 0.4

固定版本的 kit README 写明：0.4 榜仍接受不变的 public 0.3；0.4 的 text Full 由 60% 私有同技能集和 40% 私有新领域组成，**public 不计入 Full**。旧 0.3 时代 Full 是 20% public + 50% + 30%，不能套到 0.4。

本地没有跑 private Full，没有上传或参榜。四卡 H800 的墙钟和本地延迟也不是维护方在单卡 RTX PRO 6000 上的官方 latency 验收。

<a id="backend"></a>
## 11. 推理后端的优化

### 11.1 两个"v2"不是一回事

- 历史的 `inference-kernel-runtime` v2：旧权重，用了 prefix 缓存、融合算子和 CUDA Graph，单卡跑完 57 分 16.6 秒，Index 24.91；它有概率漂移和 3,814 个选项变化。
- 本轮的 `public-eight-opt-v2`：新八权重、整行 native FA3 后端，优化了 CPU 端元数据、缓存和输入类型转换；之后的四卡调度复用它。

两者权重不同、方案不同。57 分钟不能当本轮的预计时间，也不能当"老师的 4B 半小时跑完"的本地证明，那边的硬件、卡数和候选工作量没有同条件记录。

### 11.2 词表长度只查一次（v1 之前就已完成）

旧的推理代码在逐 token 检查时反复调用 `len(tokenizer)`。改成初始化时调用一次，保存含新增 token 的长度 151,669，之后只读这个整数。安装版的 FastTokenizer `__len__` 会调用 `get_vocab_size(with_added_tokens=True)`；没有重新剖析它的 Rust 内部开销。这项改动在 public-eight-v1 里已经有了，不算 opt-v1 / v2 的收益。注意 embedding 的 `vocab_size` 是 151,936，和 tokenizer 长度不是同一个数。

### 11.3 元数据在 CPU 上生成和校验

旧的通用 layout 校验会把 `cu`、真实下标、EOS 位置、position 等 CUDA 张量读回 CPU 检查。新的私有构造器只接受本 worker 内部生成的 Pair：

1. 在 CPU 上检查：整数类型、ID 范围、P / S 非空、真实 EOS、完整长度 ≤ 40,960、int32 累计边界。
2. 用这份已检查的数据一次生成 `cu`、行内位置、EOS 和真实下标。
3. GPU 上仍检查 shape、dtype、device、连续性；PB 的 P / S 划分互斥且覆盖全部。
4. 生成私有的 BoundMask，绑定具体的 batch、模式、角色和无缓存状态；核心代码仍拒绝来源不对的输入。

校验没有删掉，只是放到了数据的可信来源处，省掉了"刚传上 GPU 又读回来"。

### 11.4 autocast 缓存贯穿 worker 的整个生命周期

原来每个 micro 结束都退出最外层 autocast，PyTorch 会清空类型转换缓存；不变的 FP32 LoRA 和 head 在下一个 micro 又要重新转一次。改成加载、移到 GPU、审计、393 个张量的初始比对都完成后，进入长期作用域：

```text
no_grad
  └─ BF16 autocast(cache_enabled=True)
       └─ 官方 PEFT helper
            └─ 普通 LoRA 的实际输入检查
```

原来每个 micro 内嵌套的 autocast 保留。规范参数不永久转成 BF16，不手写 BF16 副本，不替换 PEFT 的 forward。缓存要求权重、adapter、设备和执行状态都不变，否则可能用到过期的转换结果；推理期间没有任何更新参数的接口，另有版本、指针和槽位检查，结束后再逐字节比对权重。

### 11.5 opt-v1 为什么反而变慢

opt-v1 除了上面的元数据、缓存和请求末回传，还在每条请求上构建整个模型的递归身份快照（参数和模块名、dtype、device、adapter 和缩放等）。同样的前 1,000 条正式请求，四个模型的平均延迟从约 104 ms 变成约 113 ms，慢了 8%～9%。这是速度回归。没有做单项计时，所以不能确定每一项各占几毫秒。

于是只做一次窄修正：请求热路径改用冷启动时缓存的对象、模块槽位、版本、data_ptr、梯度标志、eval 状态和 adapter 引用检查；完整的递归身份检查只在启动和关闭时做。

### 11.6 PEFT helper：先写错了名字

**我最初的简报写错了接口名 `disable_lora_input_dtype_casting`**，所以 opt-v1 没有接入。查看已安装的 PEFT 0.18.1 源码，正确的是：

```python
from peft.helpers import disable_input_dtype_casting
with disable_input_dtype_casting(model, active=True):
    ...
```

普通 LoRA 里会出现 `BF16 激活 → 转成 LoRA A 权重的 FP32 → 再进 BF16 autocast 的 Linear`，在限定条件下这个来回转换是多余的。不是无条件关闭所有转换：

- 冷启动时必须正好有 196 个普通 PEFT Linear，default adapter，r32 / alpha64 / 缩放 2，未 merge，没有 DoRA、变体、bias 或 dropout。
- 每层的输入检查确认输入是 BF16 或 FP32、在 CUDA 上、属于当前线程和请求，并且处于 eval、no_grad、BF16 autocast 和 helper 生效状态。
- helper 只临时改变类型转换标志，不改 LoRA 的缩放、加法、形状或参数。

退出时按后进先出：移除 hook → helper 恢复标志 → 检查已恢复 → 退出 autocast 和 no_grad。

### 11.7 请求末一次回传

每个 micro 按原来的 slot 把 FP32 margin 写进同一个 GPU 结果包，同时记录 logits 和 margin 是否有限、margin 等式是否成立。请求结束时一次把结果包传回 CPU，验证后才记为成功。"一次回传"只指结果包，SDK 原有的请求前后同步、启动和结束时的权重比对都还在。

### 11.8 opt-v2 的效果

同样是每个模型的前 1,000 条正式请求，平均延迟（ms）：

| 模型 | 原版 | opt-v1 | opt-v2 | 原版 / opt-v2 |
|---|---:|---:|---:|---:|
| `C_P_CE` | 104.5 | 113.6 | 93.2 | 1.12× |
| `JPAIR_P_BCE_RANK` | 104.3 | 113.1 | 93.7 | 1.11× |
| `C_P_BCE` | 104.6 | 112.9 | 93.1 | 1.12× |
| `C_D_CE` | 104.1 | 113.0 | 93.8 | 1.11× |

和两个基线相比，1,000 个答案完全一致：选项变化 0，最大概率差 0，usage 变化 0。这是对已有正式输出的被动比较，没有另跑模型。前 1,000 条偏短，后面的长上下文和多候选请求更慢，所以这不是全语料的提速比例。helper 和请求热路径检查是同时改的，没有分别测各自的贡献。

### 11.9 为什么不把旧的快路径全搬回来

历史 57 分钟的后端还用了 prefix KV、融合的 Norm / RoPE / MLP / LoRA 小算子、Graph 和有界 JIT；它的 Index 从 24.99 变成 24.91，有 3,814 个 field 选项变化，不是无损的参考。本轮明确不恢复：padding / 假 token Graph、未通过核对的 prefix 缓存、改变运算边界的融合、LoRA merge、量化和新的 micro 预算。跨 field 合批早就有了，不算本次收益。优化到 opt-v2 为止，之后只改了调度。

<a id="scheduling"></a>
## 12. 四卡调度：同一个权重分四片

### 12.1 两种并行解决不同的问题

起初四张卡各跑一个不同的权重，八个分两波。你希望"尽快先看到一个完整官方成绩"后，改成四张卡各加载同一个权重的独立副本，八个依次评。假设每个模型单卡要 T，且完全均衡：

| 方式 | 第一个完整成绩 | 八个全部结束 |
|---|---:|---:|
| 四卡各跑不同模型，两波 | T | 2T |
| 四卡同模型分片，八个依次 | T / 4 | 8 × T / 4 = 2T |

所以改善的主要是**等第一个成绩的时间**，八个的总时间不会变成四分之一。实际还受 CPU、I/O、长短题、模式、加载和尾部等待影响；旧的两波方案没有跑完，没有总吞吐的对照数据。

### 12.2 不是 tensor parallel，也不是 DDP

0.6B 的完整模型单卡放得下，每张卡独立处理自己的请求，不把一条请求的 28 层拆到四张卡上通信，也没有梯度归约。每个 worker 把自己那张物理 GPU 映射为 CUDA:0，是独立的模型进程。

### 12.3 怎样分片

官方的 140,620 行按原顺序编号（从 0 开始），用未修改的 SDK 原生 `keep` 接口交错分配：

```text
GPU0: 0, 4, 8, 12, ...
GPU1: 1, 5, 9, 13, ...
GPU2: 2, 6, 10, 14, ...
GPU3: 3, 7, 11, 15, ...
```

每个 worker 遍历全部 140,620 行，正好选出 35,155 行。交错分配让不同任务和长短请求大致均匀，不需要先估难度，也不读 gold。参数保持 `limit=None, resume=False, warm=False`。**SDK 的 `resume=False` 不会覆盖旧结果，底层仍以追加模式打开文件**，所以新的根目录和每个分片的输出目录都必须是新建的，不能往旧的半成品里续写。

### 12.4 只在四片都完整后才计分

四个子进程都实际等待退出、退出码 0、各 35,155 条完整、来源 / 配置 / 权重证明都通过后：

1. 按原编号 `% 4` 从对应分片依次读取原始 JSONL 字节。
2. 交错写回完整的 140,620 行，而不是把四个文件简单拼接。
3. 检查唯一的 run_id、完整的内容、官方元数据、模型名、状态、usage 和有限的延迟，并确认四个流都恰好读完。
4. 对这个完整文件调用一次原 SDK 的 `score_run`，必须得到版本 0.3、140,178 条计分和有限的 Index。
5. 立即写入 `FIRST_OFFICIAL_SCORE.json`，然后才启动下一个权重；最后汇总八个。

计分在所有 GPU 分片退出后、在一个只用标准库和冻结 SDK 的管理进程里进行，它明确拒绝导入 Torch，不会顺带再跑推理。

### 12.5 安全地替换正在运行的版本

旧版本一直运行，直到新源码审查完、独占安装好、CPU 准备完成、实际字节读回核对过、并确认 0 个 worker 和 0 个模型在跑，才给旧版本写 STOP。接替前还要核实：旧进程的 PID / 启动时间 / boot 身份、唯一的终止状态、所有子进程都实际等待并退出、四张卡没有未知占用、全局 lease 文件的 inode 和内容不变且能拿到排他锁。不能靠删除或改写 lease 文件来"解锁"。所有新增的缓存、临时文件、日志和结果都在 `/home/dev/assumeengage`；保留 80 GiB 输出上限和 100 GiB 剩余空间的门槛。

### 12.6 被中止的运行

public-eight-v1、opt-v1、opt-v2 前三次运行都是因为主动 STOP 被接替，半成品保留，Index 为空；不是因为答题结果不好而重试。最终的四卡运行没有接用任何旧的半成品，每个权重的 140,620 条都在新目录里独立完成。

之前给过的 20～30 小时是早期的粗估，不是测量，不能拿它和 14 小时 38 分相除来说提速了多少。

<a id="results"></a>
## 13. 官方结果

### 13.1 八个权重

| 分支 | 训练布局 / attention | 官方 Index | raw_index | 完整耗时 | 平均每条请求 |
|---|---|---:|---:|---:|---:|
| `C_P_BCE` | packed / causal | **14.42** | 35.28 | 1:43:17 | 172.5 ms |
| `JPAIR_P_BCE_RANK` | packed / causal | 13.60 | 35.05 | 1:43:58 | 172.9 ms |
| `C_D_CE` † | dense / causal | 13.07 | 34.78 | 1:43:43 | 172.8 ms |
| `C_P_CE` | packed / causal | 12.95 | 34.47 | 1:43:16 | 172.3 ms |
| `C_D_BCE` † | dense / causal | 12.83 | 33.87 | 1:43:10 | 172.1 ms |
| `BI_P_CE` | packed / 全双向 | 3.09 | 25.01 | 1:47:38 | 178.6 ms |
| `PB_P_CE` | packed / 两块双向 | 2.59 | 25.09 | 2:06:17 | 210.8 ms |
| `STATE_NL_PB_P_CE` | packed / 两块双向 | 1.97 | 24.77 | 2:06:31 | 211.2 ms |

- 所有权重**都用 packed 推理**；"dense"指训练布局。带 † 的两个有 dense → packed 的迁移偏差（[EIGHT-ARMS 第 9 节](EIGHT-ARMS-EXPLAINED.md#dagger)）。
- raw_index 用原始分代替 skill 计算，也不是总体准确率。
- 平均每条请求的延迟是本次重写时从 `SCORES.json` 补的。PB 两个分支比 causal 慢约 22%，和它们每层两次 attention 调用相符；没有做计时拆分。
- 每个权重执行 140,620 条、排除 442 条、计分 140,178 条，全部 ok，覆盖率 1。八个合计执行 1,124,960 条，计分 1,121,424 条。

**all ok 只表示每条请求都按协议执行成功，不是 140,620 题全对，也不是生产准入。**

时间线（UTC）：2026-11-13 07:46:28 开始，22:24:31 结束，合计 **14 小时 38 分 03 秒**，不含之前中止的尝试。第一个完成的是 `C_P_CE`，用了约 1 小时 43 分，之后自动启动 JPAIR。

完成后的独立核验：32 个 worker 都退出码 0 并被实际等待，管理进程已退出；32 份 393 张量的前后字节一致，helper 标志已恢复、hook 为 0，native 调用和真实 token 计数对得上；八个完整结果逐条核对了原顺序、完整输入、元数据和分片原始字节；源码固定点未变，四个 lease 文件的 inode 和内容未变且退出后能拿到排他锁，GPU 进程快照为空。

### 13.2 五个区域

各区域的官方 skill × 100（不是区域准确率）：

| 分支 | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|
| `C_P_BCE` | 4.82 | 9.54 | 20.19 | **32.42** | 7.35 |
| `JPAIR_P_BCE_RANK` | 6.19 | 9.21 | 21.05 | 25.86 | 6.82 |
| `C_D_CE` † | 5.63 | 9.10 | 20.21 | 24.62 | 7.16 |
| `C_P_CE` | 5.26 | 9.22 | 19.74 | 24.92 | 7.03 |
| `C_D_BCE` † | 5.36 | 9.12 | 15.04 | 29.49 | 6.83 |
| `BI_P_CE` | 1.10 | 1.16 | 2.66 | 8.60 | 4.07 |
| `PB_P_CE` | 0.74 | 0.62 | 3.40 | 5.77 | 5.01 |
| `STATE_NL_PB_P_CE` | 0.93 | 0.65 | 1.69 | 5.17 | 2.77 |

所有分支在 Knowledge 和 Language 上都很低（causal 分支 4.8～9.5），分数主要来自 Retrieval 和 Tools。

<a id="tools-gap"></a>
### 13.3 `C_P_BCE` 的领先来自哪里

（本节是重写时用证据 CSV 补算的。）按区域权重拆开 `C_P_BCE` − `C_P_CE` 的 1.47 点：

| 区域 | 区域 skill 差 × 权重（Index 点） |
|---|---:|
| Tools | +1.37 |
| Retrieval | +0.09 |
| Language | +0.08 |
| Arts | +0.03 |
| Knowledge | −0.11 |

Tools 的五个 benchmark：

| benchmark | 指标 | 机会水平 | `C_P_CE` 原始分 / skill | `C_P_BCE` 原始分 / skill |
|---|---|---:|---:|---:|
| BFCL | case 全对率 | 0.26 | 0.44 / 0.24 | **0.64 / 0.51** |
| API-Bank | accuracy | 0.02 | 0.36 / 0.34 | 0.41 / 0.40 |
| When2Call MCQ | accuracy | 0.25 | 0.42 / 0.22 | 0.43 / 0.24 |
| ToolRet | nDCG@10 | 0.13 | 0.49 / 0.42 | 0.50 / 0.42 |
| Home appliance simulator | case 全对率 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

所以 `C_P_BCE` 的最高分主要是 BFCL（1,694 条请求）上整题全对率从 0.44 升到 0.64，不是各方面普遍更强；Knowledge 区域它反而低于 `C_P_CE`。为什么 BCE 在 BFCL 上好这么多，本轮没有诊断。

### 13.4 为什么 raw 约 25，Index 却只有 2～3

BI、PB、STATE_NL 的 raw_index 在 24.77～25.09，但它们在 Knowledge、Language 等区域的机会校正 skill 接近 0：原始分只比随便猜好一点点，减去机会水平后几乎不剩。所以 raw 约 25 和 headline 只有 1.97～3.09 并不矛盾。这只说明分数低在哪里，不说明为什么低。

<a id="open-questions"></a>
## 14. 还没解释的问题

### 14.1 为什么不能直接说"后端坏了"或"数据清理导致掉分"

全量的输入、输出、权重和计数协议都完整，排除了漏题、混入旧预测、没正常跑完这类执行层面的解释；但 all ok 不能证明没对照过的路径在语义上也等价。

新的 clean 模型和历史 24.91 的模型，起点、数据、监督、预算和执行路径都不同，原模型预训练时见过什么也不完全可知。多处差异同时存在，不能把分数差单独归给数据清理、loss、packing、精度或某个 kernel。

### 14.2 值得研究、但还没证实的假设

| 假设 | 相关观察 | 为什么还不是结论 |
|---|---|---|
| causal 预训练的骨干改成非 causal 后，LoRA 和现有预算不够它适应 | BI / PB / STATE_NL 在开发集和官方都低 | 没有同数据下全参训练或更大适配容量的对照 |
| clean 训练数据的分布和 Public 任务不匹配 | 开发集五域抽样和 43 个任务不同；Tools 主要是程序化来源；开发集和官方排名不一致 | 缺逐来源的误差分析和匹配数据的消融 |
| BCE / rank 的梯度尺度和 clip 改变了优化轨迹 | rank 改变了 dL/dm 的尺度；比较的分支都频繁触发 clip | 没有分别控制方向和尺度的实验 |
| 独立 pair 的读出缺少候选间交互，限制了复杂比较 | 和 Base 的 answer-slot、Jina 的 listwise 输入输出差异明确 | J1 没有完整训练和官方评测 |
| BCE 在 BFCL 这类整题全对的任务上有优势 | `C_P_BCE` 的领先几乎都来自 BFCL | 单 seed；没有看 BFCL 的逐题错误 |
| 数值路径和权重之间有敏感的相互作用 | 跨形状和 dense → packed 都有失败；前 1,000 条一致不覆盖全量 | 没有独立参考后端在全部权重、全量 public 上的成对输出 |

这张表是后续研究问题，不是已经启动的计划。

<a id="failures"></a>
## 15. 过程中的失败和纠正

| 事件 | 发生了什么 | 怎样处理 | 要点 |
|---|---|---|---|
| padding-free 第一版数值失败 | 真实 macro 没过容差 | 新版本修正执行边界，旧失败留档 | 真跑过不等于通过 |
| FA3 / SDPA 和 Flex 对照没全过 | 连 causal 基线也有漂移 | dense / packed 布局和内容隔离分开研究；最终不用 Flex | 不把后端差异误算到新 mask 头上 |
| 严格 state 分界跨 BPE token | 训练 28,676 / 41,674、开发 427 / 615 个 field 切不开 | v1 一直不可用；另定义 v2，把已有换行归入 P | 改定义就要显式出新版本，不靠取整掩盖 |
| 新训练器 v1 的 PEFT setter 报 KeyError | setter 原地改写了传入的映射；还没开始前向和更新 | 改传 `dict(saved)` 保护参考对象，重新封存 v2 | 验证代码不能破坏被比较的原始对象 |
| 跟进训练第一个 macro 后磁盘检查失败 | 剩余空间比规定少约 1.11 GB | 保留断点；新的分阶段控制器在空间够了之后显式恢复 | 不删 checkpoint、不降门槛来凑通过 |
| 分阶段验证器系数差 3.7253e-9 | 验证图先算 rank 后算 BCE，生产代码顺序相反 | 按冻结的 BCE → rank 顺序修正验证器，没放宽逐位门槛 | 浮点和反传的累加顺序也会影响末位 |
| 两候选、先确认的安排不符合验收目标 | 确认集两个分支各只到 215 条 | 按你的要求停掉，八个都跑完整 public | 内部流程不能代替用户的实际目标 |
| opt-v1 提速失败 | 前 1,000 条约 104 → 113 ms | 只做一次窄修正成 opt-v2 | 要用真实正式输出测，不凭"代码应该更快"判断 |
| helper 名字写错 | 简报里的接口名不存在，导致漏用 | 查已安装 0.18.1 的源码，换成正确的 helper，加输入和恢复检查 | 以安装版本的真实 API 为准 |
| 四卡源码生成时缩进错误 | 在内存阶段被 AST 检查拒绝，尚未部署 | 统一去缩进后重新生成，AST 和只读审查通过 | 生成的代码先做语法检查，不碰正在运行的版本 |
| 三次旧官方运行被中止 | 主动 STOP 接替后端或调度，Index 为空 | 半成品不删、不拼进新运行，实际等待退出后再接管 GPU | 主动停止不等于 OOM，也不等于答题失败 |
| 预计时间偏早 | 短请求开头不代表长任务的分布 | 撤回不可靠的承诺，最终按实际 14 小时 38 分报告 | 估计不是测量 |

你明确要求之后，没有再增加 seed、补训练、额外烟测或自建确认来阻塞正式 public；必要的数据身份、资源、原生 SDK 和完整性检查保留。

<a id="code"></a>
## 16. 怎样读代码和复查结果

### 16.1 沿数据流读

路径缩写见[附录 A](#appendix-paths)。

| 顺序 | 文件 / 入口 | 要看懂的问题 |
|---|---|---|
| 1 | `D/case_pairs.py`：`normalized_field / make_case / compile_case` | 原始 case、有标签 / 无标签、q 的顺序和完整候选怎样确定？ |
| 2 | `pair_data_common.py:render_pair` | 实际的 prefix 和候选字符串是什么？ |
| 3 | `T/source/case_runtime.py`：`Pair / slots / micro_plan` | case → field → 候选 → slot → micro 怎样对应？ |
| 4 | `PK/layout.py`：`make_batch / TokenBatch` | ids、位置、`cu`、EOS 怎样隔离每条序列？ |
| 5 | `PK/model.py`：`MaskedBackbone` | 原 Qwen3 / PEFT 模块怎样只在真实 token 上计算？三种 attention 怎样分区？ |
| 6 | `T/source/objectives.py`、`F/source/objectives.py` | loss 分母、软 / 硬 q、rank 的零项、完整 field 的梯度系数？ |
| 7 | `T/source/train_case.py` | 两遍前向、逐位重放、一次 clip / AdamW 和断点推进在哪里？ |
| 8 | `T/source/evaluate_native_dev.py` | mass、NLL、Brier、覆盖率和 case / 领域聚合怎样实现？ |
| 9 | `P/source/backend.py`：`PublicEngine` | 最终加载、batch、autocast / helper、结果包、读出和事后权重比对怎样闭合？ |
| 10 | `P/source/run_public.py`：`worker / check_results / Manager.finalize_arm` | 分片、原序合并、140,620 条完整性验证和 140,178 条官方计分怎样串起来？ |
| 11 | `SDK/pipeline.py`、`SDK/scoring/index02.py` | headline 到底是哪个字段？ |
| 12 | `P/FINAL_OFFICIAL_VERIFIED.json` | 哪些完整性陈述有实际证据？ |

部分源码在仓库里有快照，链接见 [EIGHT-ARMS 第 10 节](EIGHT-ARMS-EXPLAINED.md#sources)。

### 16.2 服务输出

`choice` 返回原候选的 key 和概率字典；`noul` 返回正候选的概率和概率字典。同时返回实际的模型名和 `usage.input_tokens`，即所有独立完整 pair 的长度之和，包含重复的 prefix 和 EOS；raw output 为 None，不从生成文本里解析答案。

一次完整 public 的四个分片合计约 1.56B 个真实 pair token，这是每个候选都重复 prefix 之后的工作量，和训练的 99.8M 逻辑 pair token 不是同一个口径。

### 16.3 只读复查

下面只读取已有的官方汇总，不启动模型、训练或计分：

```python
from pathlib import Path
import json

root = Path('/home/dev/assumeengage/runs/jev-public-eight-fourway-20261113-v1')
summary = json.loads((root / 'SCORES.json').read_text())
for row in summary['models']:
    print(row['producer_arm'], row['decision_index'], row['executed'], row['scored'], row['status_counts'])
```

仓库里有同一个 `SCORES.json` 的副本（[证据目录](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/README.md)）。服务器上还可以打开 `public/<arm>/scores.json`、`COMPLETE.json`、`shards/<arm>/shard-N/WEIGHTS.json`、`<arm>-shardN-EXIT.json`，分别看官方分数、聚合身份、393 张量的事后比对和退出记录。

**不要直接重启旧的管理进程来"复现"。**旧目录是排他的，配置、源码、父进程和输入身份都已冻结，同名结果不能覆盖或追加。正式复现要用新目录，并重新核对输入、源码和资源。

### 16.4 可追溯，但不是一条命令就能在空机器上复现

完整复现需要合法的原始数据、原 ModelScope 模型和 tokenizer、当前环境、准备好的 case 和分界文件、冻结的源码和 native FA3。私有原始样本、checkpoint、optimizer、激活和逐请求预测都没有公开。部分冻结依赖还在 `/tmp/opencode`，是之前就有的资产；以后迁移要单独做映射和身份核对。

<a id="lessons"></a>
## 17. 要点、术语和自测

### 17.1 最重要的八条

1. **布局、mask、loss、读出是四件不同的事。**packing 改的是物理表示，mask 决定谁看谁，loss 决定学什么，readout 决定怎样输出。
2. **完整 case 很重要。**数据隔离、监督分母、bootstrap 和 BFCL 式的整题全对，都不能按拆散的 pair 处理。
3. **移动一个分界也是模型变化。**STATE_NL 和 PB 的输入 ID 完全相同，但可见性不同。
4. **参数的存储精度不等于计算精度。**autocast、矩阵形状和后处理精度要分开说清楚。
5. **数值一致、预测一致、总分相同是三个不同的命题。**选项一个没变也不代表概率等价；总分只差 0.01 也可能有大量逐题变化。
6. **优化要用真实输出测。**opt-v1 退步了，opt-v2 只在前 1,000 条上有小幅提升；没有单项计时就不编开销占比。
7. **并行方式服务于目标。**四卡同权重缩短的是等第一个成绩的时间，不会让八个模型的总时间变成四分之一。
8. **借鉴要标明边界。**"借鉴 Base""借鉴 Jina"不是完整复现；没做完和没通过的部分也要写进报告。

### 17.2 小词典

| 词 | 在本实验里的意思 |
|---|---|
| case / field / candidate | 一道完整的原始关联题 / 其中一个问题 / 一个备选答案 |
| pair | 固定 prefix ＋ 一个候选 ＋ 真实 EOS 构成的一条独立序列 |
| realT | 当前这批所有真实 token 的行数，不含 padding |
| `cu` | 累计序列长度，用来划分各自独立的 attention 区间 |
| margin | match logit − not_match logit，在 FP32 里相减 |
| VJP | 向量—雅可比积：先求 dL/dm，再分 micro 反传参数梯度 |
| macro / micro | 一次 optimizer 更新用的 16 个完整 case / 为了显存切开的实际执行批 |
| 逻辑 token | 曝光预算对每条完整 pair 记一次，不含第二遍前向 |
| 事后权重比对 | 不重新加载，直接把评测后的权重和原保存张量逐字节比较 |
| all ok | 每条响应都按协议执行成功，不是每条都答对 |
| 仅作科学测量 | 可以报告完整的实测结果，但不代表数值等价或生产准入 |
| Index 点 / 百分点 | 官方 Index 的差用"Index 点"；本地 mass 的差用"百分点" |

### 17.3 自测问题

- 把三条独立序列直接拼成一条 causal 序列，为什么不是正确的 packing？
- 为什么 PB 的 suffix 要对完整的 P+S 做一次 softmax，不能把两次 attention 的输出相加？
- 为什么全双向的 prefix 不能按最长公共前缀复用另一个请求的缓存？
- BCE 有 1/K，field-CE 没有，这会怎样改变不同候选数的 field 的梯度大小？
- 为什么 JPAIR 在 one-hot field 上包含了 CE，又不能把收益单独归因于"排序"？
- 同样是 FP32 head，为什么先取 EOS 再过 head 可能改变输出？
- 为什么 SDK 的 `scores.complete=true` 还不足以证明 140,620 条都执行且全部 ok？
- 为什么开发集五域等权的 mass 排名，不一定和官方 Index 排名一致？
- 四个分片各 35,155 条，直接拼接为什么不满足原顺序的完整性？
- 一张卡只占几 GB 显存，为什么不代表把同一个模型拆到四张卡上一定更快？

如果能用自己的话回答这些问题，就掌握了本轮改造的主要逻辑：数据结构、attention 的数学、目标的概率空间、浮点计算、评测指标和调度目标。

<a id="appendix"></a>
## 附录

<a id="appendix-paths"></a>
### A. 服务器路径缩写

| 缩写 | 路径 |
|---|---|
| T | `/tmp/opencode/jev-clean-trainer-20261112-v2`（前六个分支的训练） |
| F | `/tmp/opencode/jev-clean-trainer-20261112-followup-v1`（STATE_NL 和 JPAIR 的训练） |
| D | `/tmp/opencode/jev-clean-origin-20261112-v1`（数据准备） |
| PK | `/tmp/opencode/jev-packed-mask-20261112-v2/code/packed_mask`（布局和 mask 执行层） |
| S | `/tmp/opencode/jev-base-state-boundary-20261112-v2`（state 分界） |
| N | `/tmp/opencode/jev-terminal-packed-admission-20261112-v1`（dense → packed 迁移检查） |
| E | `/tmp/opencode/jev-clean-final-eval-20261112-v1`（被停掉的确认流程） |
| P | `/home/dev/assumeengage/runs/jev-public-eight-fourway-20261113-v1`（官方评测） |
| SDK | `/tmp/opencode/jev-five-dimension-distilled-20261109/decision-index/decision_index` |
| SITE | `/home/dev/assumeengage/envs/jev/lib/python3.11/site-packages` |

### B. 材料来源

- **padding-free 设计和执行记录**：本仓库 [`2026-11-12-global-padding-free-design.md`](../../../02-研究路径/05-Attention与效率/2026-11-12-global-padding-free-design.md)、[`2026-11-12-global-padding-free-implementation.md`](../../../02-研究路径/05-Attention与效率/2026-11-12-global-padding-free-implementation.md)、[`2026-11-12-attention-experiment-queue.md`](../../../02-研究路径/05-Attention与效率/2026-11-12-attention-experiment-queue.md)。最早的设计图写的是"先取 EOS 再过 head"，后来实际实现改成了"所有真实 token 过 head"，本文以实现为准。
- **你给的 commit 和三模式实验**：`2e334fb76c4cd4826df37424450b2734101db9f8`；本仓库 [PATCH.md](../../../02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/PATCH.md) 和 [`2026-11-12-commit-2e334fb-experiment.md`](../../../02-研究路径/05-Attention与效率/2026-11-12-commit-2e334fb-experiment.md)。
- **Base Labs 原文**：[Mammoth-1 and Elephant-1](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)，重点看 2.2、2.3、2.6 节。原文是 9B / 27B 全参训练、6.21M 条数据、Qwen3.5 混合结构、CE+Brier。
- **Base 的输入和分界**：[vLLM PR 63114](https://github.com/vllm-project/vllm/pull/63114) 和当时固定的 [slot_prompt.py](https://github.com/simon-veitner-redhat/vllm/blob/3eded8038a1c0c55eb981e0d38fd301a1eec62bf/vllm/entrypoints/openai/decisions/slot_prompt.py)；本仓库 [`2026-11-12-base-state-boundary-design.md`](../../../02-研究路径/05-Attention与效率/2026-11-12-base-state-boundary-design.md)、[`2026-11-12-base-state-fixed-separator-design.md`](../../../02-研究路径/05-Attention与效率/2026-11-12-base-state-fixed-separator-design.md)。当时 PR 还是 open 状态。
- **Jina v3**：固定 revision `d7d7e73b6ea138ced340b83865931b5dfb6c97aa` 的[模型卡](https://huggingface.co/jinaai/jina-reranker-v3/blob/d7d7e73b6ea138ced340b83865931b5dfb6c97aa/README.md)、[modeling.py](https://huggingface.co/jinaai/jina-reranker-v3/blob/d7d7e73b6ea138ced340b83865931b5dfb6c97aa/modeling.py#L67-L109)、[论文 v4](https://arxiv.org/html/2509.25085v4)。
- **方法拆分和跟进设计**：本仓库 [`2026-11-12-flex-block-bidirectional-reranker-design.md`](../../../02-研究路径/05-Attention与效率/2026-11-12-flex-block-bidirectional-reranker-design.md)、[`2026-11-12-jina-inspired-component-design.md`](../../../02-研究路径/05-Attention与效率/2026-11-12-jina-inspired-component-design.md)、[`2026-11-12-followup-state-rank-design.md`](../../../02-研究路径/05-Attention与效率/2026-11-12-followup-state-rank-design.md)。
- **没读到的知乎文章**：`https://zhuanlan.zhihu.com/p/1897675709696149020`。
- **官方 kit 规则**：固定 revision 的 [README](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/README.md#what-changed-in-04)，第 5、13、20、136、138～144 行写明 0.4、public 0.3、Index 和 Full 的规则。
- **研究目标和数据修订**：本仓库 [`2026-11-12-06b-top-public-research-design.md`](../../../02-研究路径/06-五维训练与Readout/2026-11-12-06b-top-public-research-design.md)、[`2026-11-12-r2-complete-case-data-plan.md`](../../../02-研究路径/06-五维训练与Readout/2026-11-12-r2-complete-case-data-plan.md)、[`2026-11-12-clean-case-training-plan.md`](../../../02-研究路径/06-五维训练与Readout/2026-11-12-clean-case-training-plan.md)。早期的 warm-start、5M、双 seed、只晋级候选等草案后来被修订或被你的要求取代，本文以实际执行为准。
- **浮点数值**：[PyTorch 2.9 numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html)。
- **autocast**：[PyTorch 2.9 AMP](https://docs.pytorch.org/docs/2.9/amp.html)；嵌套退出时清缓存见安装源码 `SITE/torch/amp/autocast_mode.py:409–414`。
- **PEFT helper**：[0.18.0 helper 文档](https://huggingface.co/docs/peft/v0.18.0/package_reference/helpers)；本机是 0.18.1，以 `SITE/peft/helpers.py:217–251` 为准。

### C. 代码位置

| 主题 | 文件与行 |
|---|---|
| 文本、监督和候选编译 | `/home/dev/assumeengage/scripts/pair_data_common.py:27–44`；`D/case_pairs.py:64–109, 226–293` |
| head、margin 和概率读出 | `T/source/model_runtime.py:98–126`；`PK/model.py:146–183`；`P/source/backend.py:520–523, 645–658` |
| 初始权重和加载 | `T/source/fresh_init.py:14–20, 35–146`；`T/init-seed42/FRESH_MANIFEST.json`；`F/FRESH_PRIMARY_VERIFICATION.json` |
| 真实 token、隔离、mask 和可微核心 | `PK/layout.py:70–109, 133–181`；`PK/model.py:73–102, 126–183` |
| 训练、重放、optimizer 和断点 | `T/source/train_case.py:60–158, 254–256`；`checkpoints.py:68–104`；`case_runtime.py`；`T/seed42.json` |
| 精确分界 | `S/prepare_boundaries.py:101–160`；`F/source/case_runtime.py:195–232` |
| 数据准入和曝光 | `D/case_pairs.py:112–140, 296–360`；`D/semantic_taint.py:193–198, 242–278`；`D/integration_driver.py:280–304` |
| loss 和梯度系数 | `T/source/objectives.py:7–35`；`F/source/objectives.py:7–74, 135–146` |
| 最终推理引擎 | `P/source/backend.py:234–729`，重点 258–262、295–421、423–539、606–658、692–729 |
| 分片、进程、资源和计分 | `P/source/run_public.py`，重点 157–196、334–428、464–489、518–624、626–879 |
| dense → packed 迁移 | `N/consumer.py:439–504` 及两个分支的 `NUMERICAL.json`、`QUALITY.json` |
| 官方计分 | `SDK/runner.py:18–107`；`pipeline.py:13–19, 57–118`；`scoring/index02.py:25–106, 123–155`；`scoring/index.py:31–116, 151–156`；`data/index-0.3.json` |
| tokenizer 和 PEFT | `SITE/transformers/tokenization_utils_fast.py:287–291`；`SITE/peft/helpers.py:217–251`；`SITE/peft/tuners/tuners_utils.py:1538–1553`；`SITE/peft/tuners/lora/layer.py:793–818` |

JSON 文件大多是单行，按字段名查找。行号以当时的冻结文件为准，其它目录的同名文件行号可能不同。

### D. 运行证据

| 主题 | 位置 |
|---|---|
| 数据和曝光 | `D/prepared-01/AGGREGATES.json`、`FINAL_VERIFICATION.json`、`SEMANTIC_MEMBERSHIP.json`、`data/schedule-seed42.json` |
| 初始权重和训练完成 | T、F 的初始化清单，各分支的 `COMPLETE.json`、`ledger.jsonl` 和 43 个 checkpoint |
| 开发集指标 | T、F 各分支 `D_select-cap{10000000,30000000,100000000}-m{0000365,0001122,0003557}/SUMMARY.json`，以及保存的 case、logits、margin |
| 旧权重的 dense / packed A/B | `/tmp/opencode/jev-padding-free-20261112-v3/eval/`；本仓库 `03-证据与图表/05-Attention与效率/attention-queue-20261112/results/full-public/` |
| state 分界 | `/tmp/opencode/jev-base-state-boundary-20261112-v1/`（原失败）；S 的 `prepared-01/PREPARATION.json`、`COMPLETE.json`、`FINAL_PRIMARY_VERIFICATION.json` |
| 机制实验和有限准入 | `/tmp/opencode/jev-fullrow-admission-20261112-v1/`、`jev-attention-2e334fb-20261112-v1/runs/`、`jev-flex-20261112-v1/results/`、`jev-jina-components-20261112-v2/` |
| dense → packed | `N/{C_D_BCE,C_D_CE}-packed/{NUMERICAL,QUALITY}.json` |
| 被停掉的确认流程 | `E/confirm/{baseline,challenger}/case-metrics.jsonl`（各 215 条）、`E/STOP_FINAL.json` |
| 官方原件 | P 的 `config.json`、`SCORES.json`、`public/<arm>/scores.json`、`COMPLETE.json` 和全部 `results.jsonl` |
| 另一个 session 的历史优化 | `/home/dev/assumeengage/reports/jev-inference-prior-session-20261113.md`；本仓库 [inference-kernel-runtime-20261111](../../05-Attention与效率/inference-kernel-runtime-20261111/README.md) |
| 本轮优化的时间线和前缀比较 | `/home/dev/assumeengage/runs/` 下的 `jev-public-eight-20261113-v1/`、`jev-public-eight-opt-20261113-v1/`、`jev-public-eight-opt-20261113-v2/` |
| 完整独立核验 | P 的 `FINAL_OFFICIAL_VERIFIED.json`、`FINAL_OFFICIAL_REPORT.md`、`FIRST_OFFICIAL_SCORE_VERIFIED.json`、32 个 `<arm>-shardN-EXIT.json`、32 个 `shards/<arm>/shard-N/WEIGHTS.json` |

### E. 本轮没有证明的事

- 完整复现了 Base、Jina 或 Flex 的配方。
- 实现了知乎文章的方法。
- packing 或双向 attention 普遍没用。
- 找到了本轮低分的原因。
- 多 seed 稳定、全量新旧后端逐位等价、dense / packed 等价或生产准入。
- private Full 成绩、维护方名次或维护方单卡 latency 验收。
- 全部来源数据没有语义重复、许可问题或预训练暴露。
