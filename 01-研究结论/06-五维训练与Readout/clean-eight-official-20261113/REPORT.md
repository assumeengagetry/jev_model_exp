# 从给定资料到八臂官方评测：Qwen3-0.6B 改造全过程与学习报告

> **仓库阅读副本。** 服务器原报告为 `/home/dev/assumeengage/reports/jev-clean-eight-detailed-learning-report-20261113.md`。本副本只增加仓库导航并调整附表链接；原实验事实、公式和数值不变。
>
> 附件：[344 行逐 benchmark 表](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/official-benchmarks.csv) · [40 行五域表](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/official-areas.csv) · [官方计分与独立核验](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/README.md)。

> **实验范围：2026-11-12～2026-11-13；整理日期：2026-11-14（UTC）；报告版本：v1.0（原始来源核实、定点更正后）。** 文件名中的 20261113 指实验完成日。
>
> 本文回答：你之前给出的 padding-free、FlexAttention、Base Labs、Jina reranker 与 attention-mask commit 材料，我分别理解成什么问题，实际修改了什么，为什么这样改，怎样核对，以及最后得到了什么。
>
> **最终事实：八个新训练终端权重均完成官方 Public 0.3 全量评测；本轮最高 C_P_BCE = 14.42。最终四卡逐模型队列耗时 14 小时 38 分 03 秒。没有达到当时研究设定的 51.99 参考目标，也没有证明新方案优于历史旧谱系。**

## 阅读方式与证据等级

- **想先掌握整体：**读第 1、2、9、14、15 节。
- **想理解为什么这样改模型：**读第 3～8 节，重点看 mask、损失和两遍梯度重放。
- **想学推理优化与多卡：**读第 10～13 节。
- **想自己对应源码：**读第 17 节和附录来源表。
- 本文的“小例子”“伪代码”用于解释概念，**不是实际训练样本或新增实验**；写报告没有运行新模型、重新评分、烟测或计算文件哈希。

本文区分四类陈述：

1. **材料提出的思路**：外部作者的方法，不等于我已完整复现。
2. **代码确实实现**：可以在冻结源码中定位，不等于训练或数值验收成功。
3. **实际观测结果**：有正式日志、保存的中间结果、完成回执或官方计分文件。
4. **解释性假设**：可以讨论，但没有因果隔离证据时不写成已证实原因。

本报告以现存冻结代码和回执为准。旧计划中的“未训练”“先确认后 public”等文字是历史状态，不能覆盖后续真实完成记录。正文中的 `[M…]` 为材料、`[C…]` 为代码、`[E…]` 为结果证据，完整定位见附录。

## 目录

1. [你的材料与实际改动的对应关系](#c1)
2. [模型、输入、读出：我们到底在训练什么](#c2)
3. [Padding-free：去掉的是无效行，不是改变题目](#c3)
4. [三种 attention 和 Base-inspired 边界](#c4)
5. [Jina-inspired：哪些做了，哪些没有完整做](#c5)
6. [数据清理与完整 case：为什么重建训练起点](#c6)
7. [BCE、Field-CE、JPAIR 的数学区别](#c7)
8. [训练器与两遍 VJP：显存有界但字段完整](#c8)
9. [八臂设计、本地开发指标和三个观察点](#c9)
10. [数值合同、失败结果与有限准入](#c10)
11. [官方 Public Index 的准确含义](#c11)
12. [推理后端的逐版修改与真实收益](#c12)
13. [四卡同模型分片：为什么能更早出首分](#c13)
14. [八个官方结果、五域分解和实际耗时](#c14)
15. [能得出的结论与尚未证实的原因](#c15)
16. [关键失败、纠正和决策变化](#c16)
17. [按文件阅读代码与复查结果](#c17)
18. [学习总结、术语和自测问题](#c18)
19. [附录：材料、代码、回执与边界](#c19)

---

<a id="c1"></a>
## 1. 你的材料与实际改动的对应关系

这次不是把几篇材料“混成一个大模型”，而是拆成若干尽量可归因的变量。

| 输入材料或你的要求 | 我提取的问题 | 实际实施 | 没有冒充完成的部分 |
|---|---|---|---|
| 尽量全局 padding-free | 不仅 attention，Embedding/LoRA/MLP 是否仍在计算 pad？ | 建真实 packed 布局，整个主干处理 realT；做 dense/packed 同核对照 | packing 不自动带来学习提分，也不自动逐位等价 |
| 你 pull 的 `2e334fb76c4cd4826df37424450b2734101db9f8` | causal、bidirectional、prefix_block 是否真的在 H800/FA3 上执行、反传？ | 精确保存该版本；实际三模式机制实验；后续改为同 native FA3 的 packed 执行层 | 该 commit 本身不是完整 padding-free；FA3 与 SDPA 的整模型数值合同没有全部通过 |
| PyTorch FlexAttention 博客/2.9.1 文档 | 任意可见性是否能用 block metadata 表达而不物化巨大稠密 mask？ | 真实 packed causal 编译/前反向机制实验；另做 Jina 交互组件接线 | 没有把 Flex 替换进最终八模型官方后端，也没有宣称它已数值等价 |
| Base Labs Mammoth/Elephant 技术报告、vLLM PR | state 与 question 两块双向、但 prefix 不看 suffix，是否可迁移？ | 实现 prefix-block；再单独实现 state＋一个已有换行的精确边界臂 | 不是 Qwen3.5 混合架构、answer-slot LM-head、CE+Brier 和其完整数据配方的复现 |
| Jina reranker v3 模型卡/源码/论文 | 候选交互和排序目标，哪一部分可以单独迁移？ | J0/J1/J2 真实 macro 接线；八臂中完整训练的是独立 pair 的 BCE＋硬标签排名项 JPAIR | JPAIR 不是完整 Jina，也不是候选列表交互模型；完整 J1 fresh 训练/public 未完成 |
| 知乎链接 | 其正文方法是否能实际取得？ | 当时 HTTP 403，保留未读取状态 | 不根据标题猜文章内容，不写“已复现知乎方法” |
| 另一个 session 的推理优化记录 | 哪些工程方法能在本轮数学/realT 边界内复用？ | 词表标量缓存、CPU-owned metadata、autocast 生命周期、请求末 D2H、正确 PEFT helper | 没有直接恢复未准入的 KV fork、融合、padding Graph、量化或 merge |
| 你要求“八个都测”“先尽快看到一个完整成绩” | 验收目标是全量官方分数，还是小开发集筛选？ | 停止只晋级两候选的旧流程；八臂全测；最终四卡同权重分片、逐模型计分 | 不用开发集百分比冒充 Public Index，不等八个全结束才交首分 |

**外部结果和我们的结果必须分开：**Base 原文给出的 9B、同数据、单次 10k public 样本消融为 causal 44.8 / block-bidirectional 48.6；它不是我们的 0.6B 全量结果。Jina 的 BEIR nDCG 不是 JEV Public Index；它们的训练数据、读出、规模/上下文也不完全相同。材料来源见 `[M1]～[M8]`。

研究最初的 51.99 参考来自当时冻结榜单中公开 served_params≤0.7B 的 GLiClass v4.1 edge 条目。**这是当时参考快照，不是本文重新核验的最新榜单第一名。**本轮 14.42 与该参考相差 37.57 个 Index 点；没有完成冲榜目标。[M9]

<a id="c2"></a>
## 2. 模型、输入、读出：我们到底在训练什么

### 2.1 不是生成答案的聊天模型路径

本任务把每个候选转成一条“是否匹配”的判别输入，末尾读一个二分类 head。不让模型逐 token 写答案，再从自然语言中解析选项。

层级关系：

```text
case：一个原生完整题/关联组
 ├─ field 1：一个待回答问题
 │   ├─ candidate 1 → 独立完整 pair → EOS 二分类 logits → margin 1
 │   ├─ candidate 2 → 独立完整 pair → EOS 二分类 logits → margin 2
 │   └─ ...                                      ↓
 │                                     本 field 内 softmax
 └─ field 2：另一个问题，仍有自己的候选和 softmax
```

**不同 case、field 或 candidate pair 不会仅因物理 packing 而共享 attention。**候选间显式交互是另外的建模变量，不是“把 token 放进同一张量”自然得到的能力。

### 2.2 原始 renderer 保持不变

实际模板的固定部分为：

```text
Decide whether the candidate is a correct match.
Input:
{state}
Question:
{question}
Available criteria:
{按原规则需要时保留的候选上下文}
Candidate:
{candidate}
```

`Question` 和 `Available criteria` 都按原条件出现，而不是每次必有。`text()` 会对字符串做 `strip()`；非字符串状态按 `ensure_ascii=False, sort_keys=True` 序列化 JSON。

`criteria_context()` 遇到“all/none/both…above”等选项指代，或小候选集的相对比较问题时，保留**原顺序的完整候选上下文**。因此“独立 pair”不等于任何情况下都看不到其它选项文字；但这仍不是 Jina 的多 document 同窗交互。

prefix 与 candidate 分别 tokenize，均不自动添加特殊 token、不截断；candidate 后显式追加真实 EOS **151645**。pad ID 为 **151643**。完整 public 保留全部给定候选，不靠删除候选缩减计算。[C1]

### 2.3 二分类 logits 如何成为字段概率

对于候选 k：

$$z_k=(z_{k,0},z_{k,1}),\qquad m_k=z_{k,1}-z_{k,0}.$$

0 代表 not_match，1 代表 match。最后在同一个 field 的全部候选之间归一化：

$$p_k=\frac{\exp(m_k-\max_jm_j)}{\sum_j\exp(m_j-\max_jm_j)}.$$

**不是把 `sigmoid(m_k)` 直接当作多选候选概率。**BCE 的 sigmoid 是训练代理目标中的二分类概率；最终服务的 choice 分布仍是字段 softmax。二者区别是本轮损失对照的重要动机。

最终读出保留原 tie 规则和候选顺序；`noul` 经过合法映射时对应 Yes./No. 两候选，不按 benchmark ID 切换模型。正式 runner 只把 `state` 与 `questions` 传给 engine，官方 expected/gold 不作为模型输入。[C2,C9,C12]

### 2.4 基座、LoRA 和 head 的精确结构

| 项目 | 实际值 |
|---|---|
| stock 来源 | 原 ModelScope `Qwen/Qwen3-0.6B` 本地资产，不悄悄换另一个 `-Base` 或新 revision |
| 层数 / hidden / MLP intermediate | 28 / 1024 / 3072 |
| Q heads / KV heads / head_dim | 16 / 8 / 128，GQA |
| context 上限 | 40960；训练单 pair 准入上限 8192 |
| LoRA | r=32，alpha=64，scaling=2，dropout=0，bias=none |
| target modules | q/k/v/o/gate/up/down 七种投影，每层都覆盖 |
| LoRA 对数 | 28×7=196 对 A/B |
| 分类 head | 无偏置 `Linear(1024,2)`，初始化 std=0.02 |
| trainable | 20,187,136 参数，393 个保存张量 |
| active served | 616,237,056 参数，约 0.616B |
| physical registered | 616,239,104；多 2048 是未参与服务的原 score 副本 |

LoRA 使用：

$$W_{eff}=W_0+\frac{\alpha}{r}BA=W_0+2BA.$$

stock BF16 参数冻结；LoRA A/B 和活动 head 的规范参数以 FP32 保存、训练。B 初始化精确为零，意味着一开始 LoRA 增量为零；head 则是新的随机分类头。393＝196×2＋1，不是393层。

实际投影形状不是全都 1024×1024：

| 投影 | 原权重 `[out,in]` | LoRA A | LoRA B |
|---|---|---|---|
| q | `[2048,1024]` | `[32,1024]` | `[2048,32]` |
| k、v | `[1024,1024]` | `[32,1024]` | `[1024,32]` |
| o | `[1024,2048]` | `[32,2048]` | `[1024,32]` |
| gate、up | `[3072,1024]` | `[32,1024]` | `[3072,32]` |
| down | `[1024,3072]` | `[32,3072]` | `[1024,32]` |

Q 输出维度是 16×128=2048，虽然 hidden 为1024。不要用“hidden/头数”替代实际 config 的 `head_dim`。旧 causal-LM 磁盘资产中未使用的词表 LM head，不计入当前二分类服务参数。[C3,E2]

### 2.5 实际运行环境

下面来自最终正式分片保存的environment，而不是凭安装命令推测：

| 项目 | 已保存运行值 |
|---|---|
| GPU | 四张 NVIDIA H800 NVL；每worker独占一个确切UUID |
| PyTorch | 2.9.1+cu128 |
| CUDA build | 12.8 |
| Transformers | 4.57.6 |
| PEFT | 0.18.1 |
| Attention | 原有冻结native FA3实现，未安装/替换另一套后端 |
| deterministic algorithms | true；cudnn deterministic=true、benchmark=false |
| world_size | 每worker为1，不是四卡DDP训练 |
| prefix_cache | false |

历史Flex组件文档记录Triton3.5.1，但最终environment没有单列该版本；本文不把历史文档当作一次新的运行时Triton版本审计。精度flags见第8.4节，source身份见附录。stock下载记录为ModelScope revision master，不代表已核实等同某个HF不可变commit。[C3,C9,E9]

### 2.6 保存精度、计算精度、读出精度是三回事

- **规范参数**：LoRA/head FP32，不 merge、不量化、不在评测期间更新。
- **forward**：BF16 autocast，使用原 Qwen3/PEFT 运算。**head 参数是 FP32，并不代表 head GEMM 全程 FP32。**
- **head 的形状合同**：先对所有实际 token 行做 head，再 gather EOS logits；不是先 gather EOS hidden 再用较小矩阵做 head。
- **margin**：head logits 转 FP32 后做 `z1-z0`。
- **最终字段归一化**：开发指标用 FP64，正式服务以 Python 双精度 `math.exp/math.fsum` 归一化。不是把整个 backbone 改成 FP64。

数学上“先挑行再做线性层”等于“先做线性层再挑行”；浮点库的 GEMM 选择和舍入却可能随矩阵形状变化。因此本轮保留 head-all-realT 的操作边界。[C2,C4,C9,M10]

<a id="c3"></a>
## 3. Padding-free：去掉的是无效行，不是改变题目

### 3.1 为什么只在 attention 里 unpad 不够

旧 collate 构建 `[B,ceil(maxL/8)*8]`。即使 FA3 wrapper 在 attention 内去掉 pad，Embedding、Norm、QKV/LoRA、MLP 和 head 仍可能处理无效行。

教学例子：三条长度为4、2、5的序列：

```text
dense（暂不考虑8对齐）         packed
A A A A _                    A A A A | B B | C C C C C
B B _ _ _                    共11个实际token
C C C C C
15行，其中4行无效
```

实际旧实现还向8对齐，这个例子可能变成24个 tokenwise 行。区别应计算在各算子的真实输入上，而不是只看 attention 里是否使用 varlen。

### 3.2 真正改成什么张量

```text
flat_ids[T]
   ↓ embedding
hidden[T,1024]
   ↓ 每层 Norm / QKV+LoRA / QK norm / RoPE
native FA3 varlen（由 cu 划分每条独立 pair）
   ↓ o+LoRA / residual / Norm / MLP+LoRA / residual
所有实际token head → 各自真实EOS logits → margin → 字段softmax
```

`T=sum(length_i)`。位置在每条 pair 内重置为0…L−1，不能直接用全局 flatten offset 当 RoPE 位置。`cu=[0,L1,L1+L2,…]` 是序列边界；EOS offsets 为各序列最后的真实 token，不能落到 pad 或下一条序列中。

“全局 padding-free”不表示没有任何硬件 tile、allocator 对齐或 metadata 容量冗余。要求是**业务 token 行**不送假 token 进入 Embedding/LoRA/MLP 等算子。[C4,M1]

### 3.3 为何保持同 micro，而不是立即把 batch 加大

四个基础臂的 micro 规划使用共同规则：stable length、max32 pairs、向8对齐的共同 padded-admission 预算16384。packed 实际执行仍只计算 realT；共同规划中的 padding 预算不等于真的补齐 token。

这样是为了先隔离“布局”变量。若 packed 一边改变 micro 构成、一边增大 batch，再与 dense 比学习质量，就同时改变了浮点几何和执行划分。等曝光质量、等 GPU 时间多训练、吞吐三类问题应分别比较。

原四臂记录中，两遍 forward 的 tokenwise 输入行数为 dense **266,503,232**、packed **199,648,730**。这说明计算行数减少；**不直接等于相同比例的墙钟加速或提分。**[C5,E3]

### 3.4 已做的固定权重实验与结论

历史旧谱系的同权重完整 public A/B：dense **24.71**，packed **24.70**；每臂全量140620/140178、全ok，没有提分。该轮 mean request 从约118.28ms到101.98ms，四卡 pipeline 从4263.02s到3705.32s，是该次实验的效率观察。

但原逐题数值合同有14801个请求违约、378个 field choice翻转（376个请求）、其中21个带外翻转。总分只差0.01，**不意味着所有概率或预测等价**。这组旧模型有已知历史输入暴露，不能作为新 clean 模型分数。[E4]

<a id="c4"></a>
## 4. 三种 attention 和 Base-inspired 边界

### 4.1 Causal、BI、PB 的区别

以下 row=query、column=key；1表示允许注意力。以两段 P/S 各2个 token 为教学例子：

```text
causal            full bidirectional      prefix-block
1 0 | 0 0         1 1 | 1 1               1 1 | 0 0
1 1 | 0 0         1 1 | 1 1               1 1 | 0 0
----+----         ----+----               ----+----
1 1 | 1 0         1 1 | 1 1               1 1 | 1 1
1 1 | 1 1         1 1 | 1 1               1 1 | 1 1
```

- **Causal**：同一 pair 内 `key_position<=query_position`。
- **BI**：整条 pair 内全双向。这里 BI 是 bidirectional，**不是 bi-encoder 双塔**。
- **PB**：P 内双向，P 不看 S；S 看完整 P 和完整 S。**S 内也是双向**，不同于“P双向、S因果”的通常 prefix-LM。
- 所有模式都受序列 owner/cu 隔离约束；不是把不同候选或 case 全连起来。

注意：causal 下，最后 EOS 本来就能读到之前整条 pair。改双向主要改变的是多层内部各 token 如何相互编码，不只是“让 EOS 多看到一些前面的字”。[C4]

### 4.2 原生 FA3 怎样实现 PB

同一层只投影一次 Q/K/V，然后做：

```python
out_P = attention(Q_P, K_P, V_P, causal=False)
out_S = attention(Q_S, K_full_row, V_full_row, causal=False)
out = scatter_back_to_original_positions(out_P, out_S)
```

P/S query indices 互斥且穷尽。第二个调用对 S 的每个 query 做一次覆盖完整 P+S 的 softmax；**不是分别对 P、S 做 attention 再把两个 softmax 输出相加**。

所以 causal/BI 每个 micro-wave 有28次 native attention 调用，PB为56次；但不是整个28层模型跑两遍，QKV/MLP并未因此整体复制一遍。[C4,C9]

### 4.3 原 PB 的分界与 Base 原文不同

当前模板的原 PB：

```text
P = 固定说明 + state + Question + 必要criteria + Candidate标头
S = 当前candidate + EOS
```

Base 原文：

```text
P = system + state
S = question + 全部options + answer-slot
```

Base 还使用 Qwen3.5 的混合 GDN/attention 结构、option-letter LM head 和 CE+Brier 配方；本项目是28层全 attention Qwen3-0.6B、每候选独立 pair、EOS二类head、LoRA。**只迁移 mask 或切点，不能说完整复现 Base。**[M3,M4]

### 4.4 严格 state 字符末尾为什么失败

最初定义 P 截止在 `HEADER + text(state)` 的最后字符；不能用字符串长度猜 token 数，因为 BPE token 可能跨过这个字符切点，例如把末尾标点和后面的换行编码为同一个 token。

实际完整 CPU 投影发现：train 的41674个 known field 中28676跨界，select 的615中427跨界。因此这个严格版本 `ready=false`，**没有启动该严格版本的学习训练**。没有取整、删题或偷偷回退 Candidate 边界。

这是“不能在不改 token 的前提下精确实现这个边界”的准备失败，不是“Base 方法已经训练失败”。[E5]

### 4.5 最终 STATE_NL 改了什么

原 renderer 在 state 后本来就追加一个 LF。新独立版本统一定义：

```text
P_text = HEADER + text(state) + 一个原来就存在的 '\n'
S      = 原来的其余Question/criteria/Candidate/candidate/EOS
```

它没有新增换行、重分词或改变完整 input_ids，只把既有分隔符归入 P。全部 train/select 用同一规则，不能逐题在两种边界间挑容易成功的版本。

sidecar 以原 tokenizer offset 验证精确 token 边界，再无损重切：`newP=oldP[:cut]`、`newS=oldP[cut:]+oldS`。重拼后每个完整 pair 的 ID、EOS、长度、position、slot、micro 和监督必须不变。v2 全部41674/615 known field精确可切，才训练 `STATE_NL_PB_P_CE`。[C6,E5]

### 4.6 为什么最终没有直接加回 prefix KV cache

- Causal 的前缀表示不读未来，可以在满足原合同的条件下复用。
- BI 的 P 会读取 S，S一变，深层 P 表示也会变，不能缓存成 immutable prefix。
- PB 的 P 不读 S，但 P 内双向，必须复用**完整且完全相同的 P block**；不能像 causal 一样把任意最长公共 token 前缀视为相同深层 KV。
- STATE 改了 P 的定义，原 Candidate-P 的 cache/邻居证据不能自动覆盖新边界。

本轮 prefix-block full/paged、fork 等路径存在未解决的数值合同失败，最终八臂统一使用 full-row native FA3，不启用这些 cache。它增加了重复前缀计算成本，但避免拿未准入的快路径冒充同一个已核实函数。[M1,M3,E6]

<a id="c5"></a>
## 5. Jina-inspired：哪些做了，哪些没有完整做

### 5.1 Jina v3 并不是“换成双向 attention”

固定源码/模型卡核实：Jina v3 基于 Qwen3-0.6B，保持 causal，多 document 顺序同窗，query在开头和末尾各一次；后面的 document 能读前面的，末端 query 能读全部 document。document/query 的末端 hidden 经共享 projector 后做 cosine 打分。

模型卡说 projector 1024→512→256，但固定 revision 源码实际为 **1024→512→512**，本文以源码构造为准，**没有加载其权重来核对 checkpoint 的 projector 张量**。源码还显式 `use_cache=False`。它的131K上下文、special-token 语义与读出，不能无训练地移植给本项目。

本轮没有下载 Jina 权重来代答；CC-BY-NC-4.0 的来源边界也没有因“借鉴思路”变成任意商用许可。[M5]

### 5.2 拆成 J0/J1/J2 的理由

- **J0**：匹配布局、位置和 backend 的候选隔离控制。
- **J1**：同 field 候选按原顺序 causal 可见，保留 EOS head 与原 BCE，研究 interaction-only。
- **J2**：attention 与 J0 一致，只加 ranking objective。

机制组件采用 prefix-once 布局，在同 Flex backend 下比较，而不是 J0用FA3、J1用Flex。否则变化既可能来自交互，也可能来自位置、模板或换 kernel。

已实际做的是：一组真实16-case/19-known-field/60-candidate macro的 J0/J1/J2 接线；每臂两遍 logits 逐位一致、393组梯度/实际更新 finite；physical realT10342，逻辑 pair token20947。该次载的是历史诊断权重，仅用于机制接线，**不是 fresh 完整训练结果**。[M6,E6]

### 5.3 八臂中完整训练的 JPAIR 是哪个部分

最终 `JPAIR_P_BCE_RANK` 保持原独立 pair、causal、native FA3 和 EOS head，只把目标改为 **BCE＋硬标签集合 ranking**。它没有加入 J1 的候选同窗交互，也没有改成 projector/cosine 读出。

因此能回答的是：“固定数据/初始化/曝光下，这个组合目标是否有收益？”不能回答“完整 Jina 是否优于原方法”。完整 J1 的 fresh匹配训练、候选换序与>40960物理field的全覆盖等，仍不能填成已完成。

<a id="c6"></a>
## 6. 数据清理与完整 case：为什么重建训练起点

### 6.1 为什么不继续沿旧 joint 权重做所有对照

之前较优 joint 的 Public 约24.91，但历史 selected xLAM 与 ToolRet 发现94个完全同文请求，涉及98条执行请求；旧 case 图没有对称解析 JSON `request` 包装。因此旧谱系仍能做机制/效率诊断，却不能直接作为新 clean 对照的起点。

新八臂统一使用原 stock，重新生成同 seed42 的 LoRA/head；没有加载项目历史已训练 adapter、head、optimizer 或旧 Teacher 标签缓存。后续 STATE/JPAIR 也是先独立生成 fresh，再把393个初始张量与共同起点逐字节核对，不是从已经训练的 causal 模型继续换 mask。[M9,C3,E2]

**这也意味着新结果低于旧24.91，不能只归因于 packing 或清理：起点、监督、数据范围、预算与后端均有差异，不是单变量实验。**

### 6.2 为什么不能只按 train/dev 文件名去重

一个原生 case 可能包含多个字段、变体或派生 view。若某一字段进训练、兄弟字段进开发集，文本不完全相同也可能共享同一任务内容。

实际采用：完整 native 坐标 census、component/sibling闭包、source/input-shape正向白名单、许可核实、input-only负向 membership，以及已知 JSON request 的对称解包。未知兄弟坐标或未决来源，不能仅删除那个麻烦字段就放行整个 case。

实际源码中的禁止位很具体：

```text
所有split：PUBLIC | FORBIDDEN | INCOMPLETE | UNKNOWN
train另外禁止：NATIVE_DEV
validation另外禁止：HISTORY
```

**HISTORY alone 的合法 native train case 可以由 fresh stock 重新训练；但不能进入“历史未见”dev。**因此不能把“clean-origin”说成“所有训练样本从未在本项目历史里出现”。公共输入仅用于负向排除，不把 public gold拿来训练。[C7]

### 6.3 最终冻结库存

| split | case数 | field数 | known | unknown |
|---|---:|---:|---:|---:|
| train | 24600 | 47427 | 41674 | 5753 |
| D_select | 375 | 705 | 615 | 90 |
| D_confirm | 250 | 439 | 389 | 50 |

训练五域case数：Knowledge4976、Language4950、Retrieval4937、Tools4766、Arts4971。select每域75，confirm每域50。tools 主要是 procedural来源，不能称为原生BFCL训练或真实工具使用泛化证明。

`choice`保留完整原options与原q；合法 `noul` 按已证明的语义方向展开为 Yes./No. 和 `[q,1-q]`；分数方向不明或非法目标的字段保留 raw/unknown，`target=None`，不 tokenize/forward、不造全零负例。同一 case 其它合法字段保留。

**这里的known/unknown仅指训练与开发监督编译。正式public服务不读取gold或这些监督mask，仍须回答全部给定question和candidate，不能借unknown政策删减官方输入。**

训练 pair 超8192时拒绝整个训练case，不截断、不删候选；开发集保留原完整题，超过共同40960能力时应明确unsupported/coverage，而不能模型跑后再补容易的题。本次八臂 select 的已知字段均覆盖完整。[C1,C7,E1]

### 6.4 100M 不是“把24600题各看一遍”

- 每域逻辑 pair-token ceiling 为20M。
- family 权重为合法case数的平方根，family内均匀抽case。
- data seed20261112冻结曝光多重集；训练seed42/43只改变冻结顺序，不重新抽样。本轮完整八臂实际仅训练了seed42。
- 16-case macro不能在预算尾部完整放入就停止，不临时挑短case补满。

实际消费56912次 case occurrence，涉及17622个不同case，最多重复17次；并不是24600个case恰好平均循环同样次数。八臂使用同一曝光流。[C7,E1]

### 6.5 clean 的证据边界

已证明的是声明范围的 source/shape 准入、完整坐标/组件、已知请求包装、精确表示及完整 pair 跨split检查。没有证明全来源的所有语义近重复、未知别名或 Qwen预训练暴露绝不存在。许可检查也不是一份无限范围法律审计。[C7]

<a id="c7"></a>
## 7. BCE、Field-CE、JPAIR 的数学区别

设一个 macro 有 C=16个完整case；case c 有 F_c个known field，field f 有 K_f个完整候选，监督 q_k∈[0,1]。合法分布允许1e-6和误差，但**不事后重新归一化q**。

### 7.1 BCE：每个候选是一个软二分类

$$B_{cf}=-\frac1{K_f}\sum_k[q_k\log\sigma(m_k)+(1-q_k)\log(1-\sigma(m_k))].$$

实际调用 `binary_cross_entropy_with_logits(...,reduction='mean')`，比先 sigmoid 再取log更稳定。BCE促使每个候选自身的二分类概率 `sigmoid(m_k)` 接近标签 `q_k`，从而约束margin的绝对标尺；不是让 `m_k=q_k`。独立考察一个候选：

$$\frac{\partial B_{cf}}{\partial m_k}=\frac{\sigma(m_k)-q_k}{K_f}.$$

软标签0<q_k<1的无约束最优margin为 `log(q_k/(1-q_k))`；硬标签1/0的损失下确界分别在margin趋向正/负无穷时取得。实际共享网络、正则化、有限训练和梯度clip当然会限制它们。

### 7.2 Field-CE：完整字段内候选竞争

$$CE_{cf}=-\sum_kq_k\log\operatorname{softmax}(m)_k.$$

这里没有额外 `1/K_f`。对所有margin加同一个常数不改变softmax/CE，但会改变BCE。BCE与CE不仅“函数名字”不同，候选数的尺度和概率空间也不同。

CE的梯度为 `p_j*sum(q)-q_j`；目标和恰好1时是熟悉的 `p_j-q_j`。不重新归一化q，就应保留这一实际细节。K=1时CE梯度为0，但该known字段仍在分母里；这是实现合同，实际已准入库存中 known K1为0，不冒称覆盖了实际不存在的分支。[C8]

### 7.3 case/field 归一化必须固定

两种目标都采用：

$$L=\frac1C\sum_c\frac1{F_c}\sum_{f\in known(c)}L_{cf}.$$

BCE已经在field内平均候选；CE使用完整分布和式。然后字段平均、case平均。不能把每个micro再平均一次，也不能因为某些field没有rank信号，就用更小的active分母重算。

这样每个case的权重相同，字段多的case不会天然支配目标；但并不等于每个token或每个候选全局等权。

### 7.4 JPAIR 的精确定义

在**原始Python q**上定义：`P={k:q_k==1}`，`N={k:q_k==0}`，`H=P∪N`。

$$R_{cf}=\begin{cases}
\log\sum_{k\in H}e^{m_k}-\log\sum_{k\in P}e^{m_k},&P,N均非空,\\
0,&否则.
\end{cases}$$

$$L_{JPAIR}=\frac1C\sum_c\frac1{F_c}\sum_f(B_{cf}+R_{cf}).$$

固定λ=1、τ=1。soft标签只进BCE，不进rank；无正/无负/纯软字段的rank为0，但仍保留原field分母。先在原host q判硬标签，是为防止近1的soft q转FP32后被舍入成精确1。

它是**正例集合的logsumexp排名概率损失**，不是所有正负margin差值的RankNet平均。严格one-hot、K≥2时R就是同一field的CE，所以硬标签上可理解为BCE＋CE，而非凭空出现的全新目标；当前合法schema也不表达多个q=1的归一分布。[C8]

### 7.5 为什么不能把改进只归为“排序方向”

加rank会同时改变梯度方向、尺度与clip行为。真实训练记账显示rank-active字段约91%，全程所比较臂的记录中clip factor<1均为100%；JPAIR的margin总系数/BCE系数L2比均值约3.3008。

这是**margin层的系数尺度**，不是参数梯度分量范数比。正式macro0中组合目标与BCE目标loss数值大小也不能直接比较成“质量变好/变坏”。本轮只能回答固定组合目标的净结果，不能独立证明排序方向就是原因。[E3]

<a id="c8"></a>
## 8. 训练器与两遍 VJP：显存有界但字段完整

### 8.1 为什么不把一个字段的候选随micro任意拆开算CE

字段softmax分母需要同一field的全部候选。若一半候选在micro1、一半在micro2，分别softmax再平均，训练的已经不是原完整field目标。

又不能为了保留一个大macro的所有候选激活，强行把整图一直驻显存直到OOM。因此采用两遍margin VJP（向量—雅可比积）。

### 8.2 实际执行流程

```python
# 说明性伪代码，不是另外运行的测试
# 同一个macro，更新前的同一份权重
m_first = no_grad_forward_all_micros()          # 第一遍：保留完整margin/slot
coef = autograd_d_loss_d_margin(m_first)       # 全field、全case的固定分母

optimizer.zero_grad()
for micro in exactly_the_same_micro_plan:
    m_replay = forward_with_grad(micro)         # 第二遍：逐micro留图
    require_replay_exact(m_replay, m_first)     # logits/margins逐位核对
    (m_replay * stop_gradient(coef_for_micro)).sum().backward()

clip_grad_norm_(trainable_parameters, 1.0)
optimizer.step()                               # 全macro仅一次
```

链式法则：

$$\nabla_\theta L=\sum_k\frac{\partial L}{\partial m_k}\nabla_\theta m_k.$$

只要两遍同权重、同输入/顺序/几何、dropout=0且实际replay一致，就可先拿到完整目标的 `dL/dm`，再分micro累计参数梯度。**不能再除以micro数、case数或虚构的DDP world size。**每个训练臂是独立单GPU模型，不是四臂一起做四rank梯度归约。[C5,C8]

### 8.3 为什么训练不能照搬推理的跨请求no_grad缓存

第一遍无梯度、第二遍需要梯度。如果跨两遍保留由no_grad阶段产生的autocast权重缓存，可能污染可微路径。因此训练保留相互独立的autocast作用域；后面介绍的“worker终生no_grad＋autocast cache”只用于**没有optimizer/backward的推理**，不能原样抄进训练器。[C5,C9]

### 8.4 冻结训练参数

| 参数 | 实际设置 |
|---|---|
| 完整case/macro | 16 |
| micro | stable length，max32 pairs，共同padded-admission16384，pad_multiple8 |
| AdamW | fused，LR峰值2e-5，weight_decay0.01，β=(0.9,0.999)，eps=1e-8 |
| 梯度clip | global norm1 |
| scheduler | 3%线性warmup，107 calls；之后cosine到0 |
| LoRA/head | FP32规范参数；BF16 autocast forward；dropout0 |
| 数值flags | matmul/cudnn TF32=true，BF16 reduced-precision reduction=true，float32 matmul precision=high |
| 关键环境 | `CUBLAS_WORKSPACE_CONFIG=:4096:8`，严格FA3 deterministic |
| checkpoint | step0、首macro、每100macro、观察点、final/pause，实际各43份，均保留 |

这些flags沿冻结机制与匹配对照保留，没有把“改精度”夹在loss/mask比较中。开关曾被实际研究，但没有证明它们解释了所有几何漂移。[C5,E2,E3]

### 8.5 预算账本，避免“100M”误读

| 名义观察cap | 实际逻辑pair tokens | case occurrences | optimizer calls |
|---|---:|---:|---:|
| 10M | 9,988,659 | 5,840 | 365 |
| 30M | 29,985,465 | 17,952 | 1122 |
| 100M终点 | **99,824,365** | **56,912** | **3557** |

逻辑token只对一次完整pair输入计账，包含每个候选重复的prefix和真实EOS。两遍训练forward实际输入 **199,648,730 real tokens**；反传覆盖99,824,365；dense还承担额外pad的tokenwise行。

**3557次optimizer.step不等于3557次非零学习率更新。**最后一次used_lr=0，仍计算梯度、推进Adam状态和曝光，实际只有3556次LR>0。八臂全部一样；没有把最后一个零LR macro丢掉来美化账本。

正式macro0只消费一次，保存first/replay、coef、393组梯度/clip/delta、Adam与checkpoint证据后，cursor1暂停并显式续跑。后续资源失败和外部控制器替换没有偷偷重做macro0。[C5,E2,E3]

<a id="c9"></a>
## 9. 八臂设计、本地开发指标和三个观察点

### 9.1 八个名字怎样读

| 臂 | attention | 训练tokenwise布局 | 训练目标 | 主要匹配对照 |
|---|---|---|---|---|
| C_D_BCE | causal | dense | BCE | 与C_P_BCE只改布局；与C_D_CE只改目标 |
| C_P_BCE | causal | packed | BCE | packed因果BCE控制 |
| C_D_CE | causal | dense | Field-CE | dense目标对照 |
| C_P_CE | causal | packed | Field-CE | BI/PB的causal控制 |
| BI_P_CE | full bidirectional | packed | Field-CE | 与C_P_CE改可见性 |
| PB_P_CE | Candidate边界prefix-block | packed | Field-CE | 与C_P_CE改可见性 |
| STATE_NL_PB_P_CE | state＋LF边界prefix-block | packed | Field-CE | 与PB_P_CE只改分界 |
| JPAIR_P_BCE_RANK | causal | packed | BCE＋hard rank | 与C_P_BCE改目标 |

D/P分别是dense/packed；C是causal；CE不是“输出两个类别的普通CE”，而是本报告第7节完整field的候选分布CE。

原四个C臂是布局×目标的2×2对照。BI/PB从同fresh起点另训，不从训练完的causal warm-start。STATE/JPAIR是在看到10M/30M后提出的**探索臂**，不能事后称八臂全部预注册。[C3,C5,C6,C8]

### 9.2 开发集主指标到底算什么

D_select主指标是五域case等权的“argmax target mass”：

$$M=\frac15\sum_d\frac1{|D_d|}\sum_{c\in D_d}\frac1{F_c}\sum_{f\in known(c)}q_{cf,\arg\max_km_{cfk}}.$$

本次已知字段都支持，因此这里F_c与实际supported分母相同；未知字段仍不进入监督/指标分母。若有不支持字段，还需另报coverage和按原known分母计零的敏感性，不能只报被挑出的易题。

教学例子：q=[0.7,0.3]，模型选第一个，字段贡献为0.7；不是 `sum(q*p)`，也不一定是0/1准确率。只有one-hot时才像字段正确率。另报NLL、Brier sum、NLL/logK及覆盖。完整one-hot case-exact只对259个全部字段known且one-hot的case计算，不能把259子集分母写成375。

### 9.3 三节点完整结果——这里的数字是本地mass百分比

| 臂 | 10M mass% | 30M mass% | 100M mass% |
|---|---:|---:|---:|
| C_D_BCE | 47.5406 | 53.9882 | 56.4260 |
| C_P_BCE | 49.6898 | 53.1384 | 57.5694 |
| C_D_CE | 53.8766 | 54.8894 | 61.6935 |
| C_P_CE | 52.0806 | 55.6754 | 59.5955 |
| BI_P_CE | 37.5653 | 43.4677 | 48.1857 |
| PB_P_CE | 41.1644 | 41.0244 | 43.2434 |
| STATE_NL_PB_P_CE | 40.7510 | 41.6300 | 44.3620 |
| JPAIR_P_BCE_RANK | 50.9312 | 55.7496 | 61.2437 |

每次都是完整375case、615known field、5190候选；90unknown保留、不forward。训练时已保存实际logits/margins；后续独立CPU分析从保存输出回算，不额外跑模型。最大指标差约1.89e-15量级。[C5,E3]

几个不能漏掉的观察：

- 四基础臂从30M到100M，mass均上升、NLL均变差；Brier在其中三臂变差，C_P_BCE则从0.5816323764降到0.5733636277，是改善。因此不能挑单一指标说质量单调改善。
- Packing相对dense的mass方向随loss和观察点变化，没有稳定“去padding就更会做题”的结论。
- 100M时JPAIR比C_P_CE高1.6482个百分点，描述性case区间约[-1.9076,+5.1159]；STATE比PB高1.1186个百分点，区间约[-3.7838,+5.9899]。两者都不能称已确认稳定提升。
- 这些case-bootstrap只是在同一训练seed的固定输出上，对五域各75个**完整case**有放回重采样，10000次；不是训练seed的置信区间，也没有多重比较校正。

**本地61.69%不等于Public Index61.69，更不能拿来宣称超过51.99。**后面的正式结果会显示开发集排名与官方排名并不一致。

### 9.4 D_confirm为什么没有被写成“确认通过”

最初后期制定了baseline=C_P_CE、challenger=JPAIR的两候选确认流程。实际confirm两个目录各只有215条case-metrics，没有完整250条的COMPLETE/SUMMARY。你后来明确要求优先八个权重全部直接跑官方，我停止并保留该partial，没有把它冒充完整确认。

这也是流程上的纠正：**我先前的“最多两个候选、先确认再public”安排，与当时你要验收全部已训练权重的目标不一致。**最终按你的新要求测全八臂，不再让自建确认成为官方测分前的冗长关卡；因此本轮没有独立完成的D_confirm结论。[E8]

<a id="c10"></a>
## 10. 数值合同、失败结果与有限准入

### 10.1 为什么“能跑”和“等价”不是一件事

需要区分：

```text
能解析/语法通过
  ≠ GPU确实执行和反传
  ≠ 输出/梯度与参考在原容差内
  ≠ 换batch几何时逐位相同
  ≠ 学习质量提高
  ≠ 全量官方分数提高
  ≠ 生产/官方latency准入
```

同一数学式在不同batch、kernel、切片或GEMM形状下，不保证bitwise相同。反过来，数值不同也不自动证明跨样本内容泄漏。PyTorch数值文档明确讨论了这些边界。[M10]

### 10.2 冻结容差不是看到结果后再定

| 对象 | 原冻结判据 |
|---|---|
| hidden/logits或margin的RMS误差 | ≤0.01×max(reference RMS,1e-3) |
| 最大绝对误差 | ≤0.02＋0.02×maxabs(reference) |
| macro loss | ≤1e-3×(1＋abs(reference loss)) |
| 每组梯度/实际更新L2误差 | ≤0.02×norm(reference)＋1e-7×sqrt(numel) |
| 非近零梯度/更新方向 | cosine≥0.999，近零组单独记账 |
| choice敏感带 | reference top1−top2 gap≤0.04＋0.04×maxabs(reference field margins) |

“带内flip”表示该field对规定数值漂移敏感，不代表这个flip可以称作预测等价。K1没有top2，不制造假gap。[M1,C11]

### 10.3 实际失败怎样保留

- PF v1真实macro失败，保留；修正版本不回写v1为通过。
- exact commit的FA3对SDPA三模式均未过整模型合同，连causal基线也有漂移，不能只责怪新mask。
- Flex causal实际compiled、forward/backward、393组梯度和一步更新都执行了，但logits/gradient/update没有全部过原合同；没有装成最终官方后端。
- 同native FA3、同macro几何的dense/packed布局对照通过原梯度/更新容差；换同几何真实邻居的目标输出/VJP逐位不变，支持有限的内容隔离判断。
- solo/mixed形状稳定性、prefix-block fork/paged等失败保留；没有因此签发“任意几何、任意缓存都等价”。

后续允许的是**full-row科学训练**：相同实际FA3、明确输入/位置/边界和有限证据下跑匹配学习对照；不是把所有早期失败变成通过。[E4,E6]

### 10.4 训练完的dense权重迁移packed仍有失败

这一步没有重新训练，也没有重跑dense参考，只拿已保存的原dense终点开发结果，对同一终端权重的新packed消费者做完整375case比较：

| dense训练权重→packed推理 | 数值失败case | choice变化 | 带外flip | 最大概率差 |
|---|---:|---:|---:|---:|
| C_D_CE | **34/375** | 0 | 0 | 0.0762880 |
| C_D_BCE | **34/375** | 1 | 0 | 0.1104917 |

两者numerical_acceptance=false，393参数前后字节未变。CE虽然argmax没变，raw logits/probabilities仍违约，不能说等价。

你要求八个全部测后，仍对两者测完整packed官方分数，但配置和报告一直保留 **scientific only / 原数值准入失败**。官方完成、全ok不会反向抹去这项限制；也不能从其public差异单独归因dense/packed训练布局优劣。[C11,E7]

<a id="c11"></a>
## 11. 官方 Public Index 的准确含义

### 11.1 这次固定了哪些东西

- public suite edition **0.3**；执行/报告43个benchmark，其中37个进入Index。
- 每权重完整执行140620条；按官方排除442条，计分140178条。
- MMLU、ARC-Easy、ARC-Challenge、SimpleBench、RouterBench、SGD这6个benchmark仍执行/展示，但不计入本次Index；不能只因常见任务分数高就推断headline高。
- 复用已经实际严格验证的Suite及exclusions receipt，不反复导出/哈希语料。
- SDK来源声明revision `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`；冻结SDK资产本身不是git checkout，不能把这串字当一次新的git核验。
- 原renderer、候选、EOS、FP32margin→双精度field归一化不变；无按正确性重试、删候选、混旧预测、按题路由或回选public checkpoint。

原生unsupported/abstained/普通error按SDK状态保留并受官方coverage惩罚；设备错误和“五次error且尚无success”的SDK原生门槛仍可中止。**本次最终八臂刚好全部ok，不是靠预先把非ok删除。**[C12,E9]

### 11.2 为何Index不是统一accuracy

各benchmark先按自己的原生指标计算，可能是accuracy、macro-F1、每review F1、retrieval nDCG、整case所有字段全对等。BFCL的case-exact和单字段对一半不是同一件事。

常见chance路径（chance c_b<1）可概括为：

$$r_b=\text{native score经coverage调整},\qquad
s_b=\operatorname{clip}_{[0,1]}\frac{r_b-c_b}{1-c_b}.$$

旧track-scored路径已在metric内部给失败/不完整组计零，**不能再重复乘一次coverage**。机会基线可能按query/track/case和任务结构定义，不是所有任务都统一1/K。历史loss-rule分支也有自己的baseline/best换算；具体以冻结SDK为准。[C12]

教学例子：四选一机会水平0.25，raw=0.40，skill=(0.40−0.25)/(1−0.25)=0.20。这个20只说明该任务相对机会水平的提升，不是该任务accuracy20%。

### 11.3 五域怎样汇总

区域内gold★ benchmark权重1.2，其余1.0，归一化后求skill加权平均S_a；五区域权重：

| 区域 | w_a |
|---|---:|
| Knowledge & Reasoning | 0.258494 |
| Language Understanding | 0.258494 |
| Retrieval & Classification | 0.200229 |
| Tools & Automation | 0.182783 |
| Arts & Human Taste | 0.100000 |

本次headline准确地说是：

$$\boxed{Public\ Index=100\sum_aw_a S_a=balanced\_skill}.$$

不是几何平均。SDK另报：

$$breadth\_skill=\frac{100}{0.9}\left[\prod_a(0.1+0.9S_a)^{w_a}-0.1\right],$$

以及使用raw而非skill的`balanced_raw/raw_index`，但二者都不是本次headline。SDK还有中间舍入与末端两位舍入，不应拿已经四舍五入的展示表重算一个略不同数字就认定bug。[C12]

### 11.4 Public 0.3与榜单Full 0.4不要混淆

固定kit README明确：0.4榜仍接受未改变的public0.3；**0.4 text Full由60%私有same-skills＋40%私有new-domains组成，public不计入Full。**旧0.3时代Full曾是20%public＋50%＋30%，不能把旧比例套到当前0.4规则。

本地没有运行private Full，没有上传、提交参榜或获维护方正式排名。四卡H800墙钟与本地request latency，也不是维护方单进程单卡RTX PRO 6000上的官方latency验收。[M8]

<a id="c12"></a>
## 12. 推理后端的逐版修改与真实收益

### 12.1 先分清两个名字里的“v2”

- 历史`inference-kernel-runtime` v2：旧权重、prefix/fusion/Graph方案，完整单卡57分16.6秒，Index24.91；有概率漂移与3814个choice变化。
- 本轮`public-eight-opt-v2`：新八权重的full-row native FA3后端，CPU元数据/缓存/输入cast优化；随后fourway调度复用它的完整PublicEngine。

二者不是同一个v2，也不是同一权重。57分钟不能直接作为本轮ETA或老师4B半小时的本地证明。老师的硬件、卡数和实际候选工作量没有同条件本地记录。[E10]

### 12.2 词表长度只查询一次——早于opt-v1

旧最终消费者在逐token检查中重复调用`len(tokenizer)`。最终初始化只调用一次，保存含added tokens的长度151669；encode/请求校验只读一个整数。

本次读到的安装版FastTokenizer `__len__`委托给`get_vocab_size(with_added_tokens=True)`。**能确定的是热循环重复调用被一次缓存取代；本文没有重新剖析Rust内部，不把“每次一定复制完整15万词表”写成本次独立证实的实现细节。**这一修正已经在原public-eight-v1中存在，不再重复算成opt-v1/v2的新收益。

模型embedding config的vocab_size151936与tokenizer含新增项的长度151669不是同一指标，不能混用。[C9,C13]

### 12.3 CPU-owned metadata：不再“刚传上去又读回来检查”

旧通用layout校验会把cu、real_indices、EOS、positions等CUDA张量读回CPU。新私有构造器只接受本worker内部的Pair：

1. 在CPU验证整数非bool、ID范围、非空P/S、真实EOS、完整长度≤40960和int32累计边界。
2. 从这份已经验证的数据一次生成cu、行内position、真实EOS和real indices。
3. GPU仍检查shape、dtype、device、contiguity；PB的P/S分区互斥且穷尽。
4. 生成私有BoundMask，绑定确切batch/mode/role/full/no-cache；原core仍拒绝错误owner。

这不是把“任何外部CUDA张量”都当可信输入，也不是删除所有校验，而是让校验发生在数据的可信来源处。[C9]

### 12.4 Worker生命周期的autocast cache

原每micro退出最外层autocast，PyTorch会清空cast cache。对不可变FP32 LoRA/head，下一micro或请求可能再次做执行dtype转换。

最终顺序：加载→GPU move→audit→393 pure-before比较，之后才进入长期：

```text
no_grad
  └─ BF16 autocast(cache_enabled=True)
       └─ 官方PEFT helper
            └─ 普通LoRA actual-input guards
```

原每波嵌套autocast保留。规范参数不永久转BF16，不手写BF16权重副本，不替换PEFT.forward，不改requires_grad标志；所有forward位于no_grad，grad必须始终None。[C9,M11]

缓存意味着必须保证权重、adapter、设备和执行状态不变，否则可能复用过期执行转换。推理期没有optimizer/参数更新API；版本/指针/槽位守卫和结束后纯byte比较共同防护。

### 12.5 Opt-v1为什么反而慢

opt-v1除了上述元数据、lifetime cache、请求末D2H，还在每请求构造整模型递归身份快照：参数/模块名字、dtype/device字符串、adapter/scaling等。

同样的前1000条真实正式请求，四模型平均从约104ms变为113ms，只有约0.92×：**这是速度回归，不能称优化成功。**这个观察说明组合方案退步，但没有单项计时实验把某一项确定归因成几毫秒。

因此只做一次窄纠正：请求热路径用冷启动缓存的对象/模块槽位/版本/data_ptr/gradflag/eval/adapter引用检查，完整递归身份保留在启动/关闭；原source/resource周期守卫仍保留。[C9,E11]

### 12.6 PEFT helper的接口名纠正及数值理由

**我最初简报写错了`disable_lora_input_dtype_casting`这个名字**，opt-v1按当时范围没有接入。实际安装的PEFT0.18.1支持：

```python
from peft.helpers import disable_input_dtype_casting
with disable_input_dtype_casting(model, active=True):
    ...
```

原普通LoRA中可出现：

```text
BF16 activation → 对齐FP32 LoRA A权重dtype → BF16 autocast Linear
```

BF16值扩为FP32再转回BF16，在限定情形下是冗余往返；FP32输入的第一次对齐本来就是no-op。不是无条件关闭所有输入转换：

- 冷启动必须恰好196个普通PEFT Linear、default adapter、r32/alpha64/scaling2、未merge、无DoRA/variant/bias/dropout。
- 每层actual pre-hook检查输入为BF16或FP32、CUDA、所属线程/请求、eval、no_grad、BF16 autocast/cache与helper flag。
- helper只暂时改变cast flag；不改LoRA的scale/add、矩阵形状或规范参数。

退出时按LIFO：移除hooks→helper finally恢复flags→检查恢复→退出autocast/no_grad。stack先摘除再关闭，防止异常路径重复退出。[C9,C13,M12]

### 12.7 请求末统一D2H：到底统一了什么

各micro仍按原Pair.slot把FP32margin写入同一GPU packet，并携带logits finite、margin finite及margin等式flags。请求末一次传回结果packet，验证后才记成功wave/token/pair。

这里的“一次D2H”指**结果packet**，不是宣称整个进程只有一次GPU→CPU操作或SDK不再同步。SDK原有请求前后`synchronize`、启动/结束权重比较等仍保留。

原native completed_calls只表示调用返回，pre-hook次数只表示输入检查，不是GPU已完成。最终还要对齐成功D2H/finite检查、196×waves、28或56×waves和realT计数；传输完成与整个请求成功也分开记账。[C9,C12]

### 12.8 Opt-v2测到了什么

同一正式前1000条/模型，四组如下：

| 模型 | 原版mean ms | opt-v1 | opt-v2 | opt-v2/原版加速 |
|---|---:|---:|---:|---:|
| C_P_CE | 104.4867 | 113.5560 | 93.1935 | 1.1212× |
| JPAIR | 104.2620 | 113.1289 | 93.6781 | 1.1130× |
| C_P_BCE | 104.6491 | 112.8668 | 93.1189 | 1.1238× |
| C_D_CE | 104.1226 | 112.9861 | 93.7962 | 1.1101× |

各组与两个基线比较，1000个answers完全一致，choice变化0、最大概率差0、usage变化0。只是被动比较现有正式输出，没有新增模型推理。

**这是首批约1.11～1.12×的结果，不是全语料比例或全量逐位等价证明。**前1000条主要偏短，后续长上下文/多候选更慢。guard与helper同时改动，没有独立ablation可说各贡献多少。[E11]

### 12.9 为什么不把旧快路径全搬回来

历史57分钟后端还用了prefix KV、融合Norm/RoPE/MLP/LoRA小算子、Graph和有界JIT。它同时有Index24.99→24.91及3814个field choice变化；不是无损参考。

本轮明确不恢复：padding/fake-token Graph、未准入prefix cache、改变运算边界的fusion、LoRA merge、量化、新micro预算。跨field合批早已有，不能当本次新增收益。优化在opt-v2收束；随后变化只涉及调度。[E10,E11]

<a id="c13"></a>
## 13. 四卡同模型分片：为什么能更早出首分

### 13.1 两种并行回答不同目标

起初四张卡分别跑四个不同权重，八个分两波。你明确希望“尽快先看到一个完整官方成绩”后，改成四张卡加载同一权重的四个独立副本，依次评完八个。

假设各模型单卡完整耗时都为T，且完全理想均衡：

| 方式 | 第一个完整成绩 | 八个全部结束 |
|---|---:|---:|
| 四卡各不同模型，两波 | T | 2T |
| 四卡同模型分片，八个串行 | T/4 | 8×T/4＝2T |

所以主要改善**首分等待**，不能把八模型总耗时直接除4。实际还受CPU/I/O、长短题、模式、加载和波次末尾等待影响；本次旧两波运行没有完整结束，**没有最终总吞吐4倍对照证据**。

### 13.2 不是tensor parallel，也不是DDP训练

0.6B完整模型已经能放进单卡，每张卡独立处理请求，不把一条请求的28层模型拆成四段通信；也没有四rank梯度归约。每个worker看到自己的单物理GPU UUID映射为CUDA:0，仍是独立模型进程。[C10]

### 13.3 精确分片规则

原始官方140620行，ordinal从0开始：

```text
GPU0: 0,4,8,12,...
GPU1: 1,5,9,13,...
GPU2: 2,6,10,14,...
GPU3: 3,7,11,15,...
```

使用未改SDK的native `keep`接口；冻结版本对每个原始row按原顺序调用一次keep。每worker须完整遍历140620行，恰好选35155行。交错分片使不同任务/长短序列较均匀，不另跑模型估算题目难度，也不读取gold决定分配。

保持`limit=None,resume=False,warm=False`。**SDK的resume=False并不保证覆盖旧results，底层仍以append打开；因此新root和每个shard输出目录必须排他新建。**这也是为什么不能向已有partial随便续写。[C10,C12]

### 13.4 只在全量恢复后计分

每个shard都有唯一arm-shard grant、log、OWNER/EXIT、输出目录和393 after-proof；shard COMPLETE本身不是Public分数。

四个child全部实际wait、exit0、35155完整、同source/config/权重证明通过后：

1. 按原ordinal `%4`，从对应shard读取下一行原始JSONL字节。
2. 交错写回全部140620行，不是把四个文件简单concatenate。
3. 检查唯一run_id、完整payload、官方metadata、model、status、usage与finite latency，四个stream均恰好耗尽。
4. 对这个完整文件调用一次原SDK `score_run`；必须得到edition0.3、140178 scored和有限Index。
5. 立即提交`FIRST_OFFICIAL_SCORE.json`，然后才启动下一个权重；最终再汇总八个。

计分在所有GPU分片退出后的SYSTEM管理进程里进行；实际调用链只需标准库和冻结SDK资源，明确拒绝意外导入Torch，不借大模型环境另跑推理。[C10]

### 13.5 安全切换不靠“看起来已经没进程”

旧版保持运行，直到新源码审查、排他安装、CPU prepare、实际字节readback和0worker/0模型安装回执齐备。MAIN才写旧版STOP。

接替必须核实：旧PID/starttime/boot身份、唯一terminal、所有创建过的子进程actual wait/exit/死亡；四卡无未知占用；原global lease **inode与bytes不变**、UUID一致、`LOCK_EX|LOCK_NB`可取得。不能删/truncate/重写旧lease来“解锁”。Popen/信号登记、pidfd、parent-death guard与异常actual-wait barrier沿用，lease关闭严格晚于actual waits。

全部新增cache/tmp/log/结果在`/home/dev/assumeengage`。独立80GiB输出cap与固定100GiB free门槛保留，不拿已用输出抵扣未来reserve，也不冒称操作系统级磁盘配额或对任意同UID进程的强认证。[C10]

### 13.6 中止记录怎样理解

原public-eight-v1、opt-v1、opt-v2前三次都因明确记录的主动STOP被接替，partial保留、Index为null；不是按答题正确性重试，也不是把不理想预测偷偷换掉。最终fourway没有接用任何旧partial，全部140620/权重在新namespace独立完成。

此前20～30小时是早期低置信度排期估计，不是已测完整baseline；**不能拿它与14h38相除来宣称实际speedup**。老师4B半小时的参考也没有等条件本地测量，不能承诺照搬。[E11,E12]

<a id="c14"></a>
## 14. 八个官方结果、五域分解和实际耗时

### 14.1 最终权威结果

| 权重 | 训练布局 / attention | 官方Index | raw_index（也不是总体准确率） | 四卡完整耗时 |
|---|---|---:|---:|---:|
| `C_P_CE` | packed / causal | **12.95** | 34.47 | 1:43:16 |
| `JPAIR_P_BCE_RANK` | packed / causal | **13.60** | 35.05 | 1:43:58 |
| `C_P_BCE` | packed / causal | **14.42** | 35.28 | 1:43:17 |
| `C_D_CE` | dense / causal | **13.07** | 34.78 | 1:43:43 |
| `C_D_BCE` | dense / causal | **12.83** | 33.87 | 1:43:10 |
| `BI_P_CE` | packed / bidirectional | **3.09** | 25.01 | 1:47:38 |
| `PB_P_CE` | packed / prefix_block | **2.59** | 25.09 | 2:06:17 |
| `STATE_NL_PB_P_CE` | packed / prefix_block | **1.97** | 24.77 | 2:06:31 |

**所有模型的consumer均为packed。**两个dense行指训练布局，不是正式评测时仍dense；它们保留第10.4节的34/375原数值失败限制。

每模型140620执行、442排除、140178计分，全ok、coverage1。总执行 **1,124,960**，总计分 **1,121,424**。

最终四卡逐模型队列时间（UTC）：

- 开始：2026-11-13 07:46:28.100458。
- 完成：2026-11-13 22:24:31.528708。
- 合计：**52683.42825秒＝14小时38分03.428秒**，不包括之前中止尝试。
- 首个C_P_CE：约1小时43分16秒；之后自动启动JPAIR。

32个worker均exit0并actual-wait，manager已退出；32份规范393张量before/after字节一致、未重载掩盖变更，helper flags恢复、hooks=0、native/realT计数通过。独立最终验收逐条对照八个完整结果的原序/完整输入/metadata和shard原始bytes；源pin和小源码bytes未变，四原lease inode/bytes未变且退出后EXNB可取，GPU进程快照为空。[E9,E12]

**all-ok只是协议/执行成功，不是140620题全答对，也不是生产准入。**

### 14.2 五区域skill分解

下表为各area的官方skill×100，不是area accuracy；取自官方文件已舍入area值。

| 权重 | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|
| `C_P_CE` | 5.26 | 9.22 | 19.74 | 24.92 | 7.03 |
| `JPAIR_P_BCE_RANK` | 6.19 | 9.21 | 21.05 | 25.86 | 6.82 |
| `C_P_BCE` | 4.82 | 9.54 | 20.19 | 32.42 | 7.35 |
| `C_D_CE` | 5.63 | 9.10 | 20.21 | 24.62 | 7.16 |
| `C_D_BCE` | 5.36 | 9.12 | 15.04 | 29.49 | 6.83 |
| `BI_P_CE` | 1.10 | 1.16 | 2.66 | 8.60 | 4.07 |
| `PB_P_CE` | 0.74 | 0.62 | 3.40 | 5.77 | 5.01 |
| `STATE_NL_PB_P_CE` | 0.93 | 0.65 | 1.69 | 5.17 | 2.77 |

C_P_BCE相对C_P_CE的Tools区域从24.92到32.42，是总Index差异的重要组成；但它Knowledge从5.26到4.82，并非五域都提高。不能只看“总分较高”就概括成所有能力普遍提高。

BI/PB/STATE在Knowledge/Language等区域的机会校正skill接近0。这解释了raw_index约25但headline只有1.97～3.09并不矛盾；**不代表已定位表示、优化或数据方面的唯一致败原因**。

另附八模型×43 benchmark的344行CSV，保留native metric/score、index raw/skill/chance、coverage、status与是否进入Index；方便逐项学习，未复制私有逐请求样本。见附录文件索引。

<a id="c15"></a>
## 15. 能得出的结论与尚未证实的原因

### 15.1 当前证据支持的结论

1. **这组终端模型中，C_P_BCE最高14.42。**不是世界最优，也没有超过当时51.99参考。
2. packed BCE比packed CE高1.47个Index点；但开发mass反而57.5694%<59.5955%。开发主指标的排序不能代替官方排序。本地mass每域等权20%，官方区域权重并不等权，且使用任务原生指标、机会校正、case/track规则与coverage；两种总体指标本来就不是同一个函数。
3. JPAIR比CE高0.65点，却比匹配的packed BCE低0.82点；“本地mass较高”没有转成对BCE的官方收益。
4. STATE本地mass比PB高1.1186个百分点，正式Public却低0.62点。改边界没有在本轮官方口径得到收益。
5. 当前同基座、fresh LoRA/head、数据、预算和目标下，非causal mask明显低于causal；**不能推广成所有双向模型、完整Base或Jina方法失败**。
6. packing的明确价值首先是减少无效tokenwise工作；本轮没有稳定学习提分结论。dense→packed准入失败也限制了正式表中对布局的因果解读。
7. 八臂仅seed42；case-bootstrap不替代训练seed复现，不能把小分差称统计显著或跨seed稳定。

### 15.2 为什么不能直接说“后端坏了”或“数据清理导致掉分”

本轮全量输入/输出与权重/计数协议完整，排除了漏题、混预测、未正常完成等执行层解释；但all-ok并不能证明语义函数在任何未对照过的路径都等价。

同时，新clean模型与历史24.91模型不同起点、数据/监督、预算和执行路径；原始stock预训练暴露也未完全可知。**多项差异同时存在，不能把分数差单独归给数据清理、loss、packing、dtype或某个kernel。**

### 15.3 值得研究、但尚未证实的假设

| 假设 | 相关观察 | 为什么还不是结论 |
|---|---|---|
| 因果预训练骨干转非因果，LoRA/预算不足以适应 | BI/PB/STATE在dev与public都低 | 没有同数据的全参/不同适配参数覆盖等隔离对照 |
| 有限clean训练来源与完整Public任务分布不匹配 | select五域抽样与43任务不同，tools主要procedural；dev/Public排名不一致 | 还缺充分逐来源误差与匹配数据消融，不能直接判某来源是原因 |
| BCE/rank的尺度、clip改变了优化轨迹 | rank改变dL/dmargin尺度，所比较臂clip均频繁 | 未做保持方向/尺度分别控制的实验 |
| 独立pair读出缺少全候选交互，限制复杂比较 | 与Base answer-slot、Jina listwise的输入/读出差异明确 | J1 fresh完整匹配训练/public未完成 |
| 数值路径与权重存在敏感交互 | 历史跨形状与dense→packed有失败，前1000后端相同不覆盖全量 | 没有所有权重/全public的独立参考后端成对全量输出证据 |

这张表是帮助理解的后续研究问题，**不是新的执行授权或已经启动的训练计划**。本次写报告没有新增实验。

<a id="c16"></a>
## 16. 关键失败、纠正和决策变化

| 事件 | 实际发生了什么 | 修正/保留方式 | 学习要点 |
|---|---|---|---|
| PF v1数值失败 | 真实macro未过合同 | 新独立版本修正执行边界，旧失败留存 | 真实运行过≠通过 |
| FA3/SDPA及Flex对照不全通过 | causal基线也有漂移 | 同native FA3布局与内容隔离分开研究；最终未用Flex | 不把backend差异错归新mask |
| strict-state BPE跨界 | 28676/41674 train、427/615 select不可精确切 | v1一直notready；另定义统一已有LF归P的v2 | 改定义要显式版本，不用round掩盖 |
| fresh trainer v1 PEFT setter KeyError | setter原地改写传入mapping；尚无forward/更新 | 传`dict(saved)`保护oracle，独立v2重新封存 | 验证代码不能破坏被比较的原始对象 |
| followup首macro后磁盘guard失败 | free比规定需求少约1.11GB；0resume/0AUTH | 保留cursor1/partial；新staged controller在容量fit后显式恢复 | 不删checkpoint/降底线凑通过 |
| staged验证器系数差3.7253e-9 | 主验证图先rank后BCE，producer顺序相反 | 按冻结BCE→rank构图顺序修校验器，未放宽位级门槛 | 浮点/反传累加顺序也能影响末位 |
| 两候选/confirm-first安排不合用户验收目标 | confirm两臂各215条partial | 按你的明确要求停旧流程，八个都完整public | 内部流程不能代替实际用户目标 |
| opt-v1提速失败 | 前1000约104→113ms | 仅一次窄纠正为opt-v2，未无休止追新优化 | 要用正式真实输出测，不凭“代码更快”判断 |
| helper名写错 | 初始简报不存在的接口名导致漏用 | 查已安装0.18.1源码，用正确helper并加实际输入/恢复守卫 | 对照安装版本的真实API |
| fourway源码生成时缩进错误 | stdlib AST在内存阶段拒绝，尚未部署 | 对完整行统一dedent后生成；最终AST/只读审查通过 | 生成代码应先语法检查，不能碰正在跑的版本 |
| 三次旧official中止 | 明确STOP接替后端/调度，Index为null | partial不删、不拼新run，actual-wait后再接GPU | 主动停止≠OOM≠答题质量失败 |
| ETA偏早 | 短前缀不代表长任务分布 | 撤回不可靠承诺，最终按完整实际14h38报告 | 估计不是测量，不能拿粗估算speedup |

真正实施的是多层证据流程，而不是无限堆门禁：必要的数据身份、资源、原生SDK和完整性检查保留；你明确要求后，没有再扩seed、补训练、额外烟测/自建确认来阻塞正式public。

<a id="c17"></a>
## 17. 按文件阅读代码与复查结果

### 17.1 建议沿数据流读，不要先陷入manager细节

正文路径缩写在附录展开。

| 阅读顺序 | 文件/入口 | 要看懂的问题 |
|---|---|---|
| 1 | `D/case_pairs.py`：`normalized_field / make_case / compile_case` | native case、known/unknown、q顺序、过滤和完整候选怎样确定？ |
| 2 | `pair_data_common.py:render_pair` | 实际prefix/candidate字符串是什么？为什么不能随便换模板？ |
| 3 | `T/source/case_runtime.py`：`Pair/slots/micro_plan` | case→field→candidate→slot→micro怎样对应？ |
| 4 | `PK/layout.py`：`make_batch/TokenBatch` | ids、positions、cu、EOS如何隔离每条序列？ |
| 5 | `PK/model.py`：`MaskedBackbone` | 原Qwen3/PEFT模块怎样在realT上计算？三种attention怎样分区？ |
| 6 | `T/source/objectives.py`和`F/source/objectives.py` | loss分母、soft/hard q、rank零项、完整field梯度系数是什么？ |
| 7 | `T/source/train_case.py` | 两遍forward、exact replay、一次clip/AdamW和cursor推进在哪里？ |
| 8 | `T/source/evaluate_native_dev.py` | mass/NLL/Brier/coverage及case/domain聚合怎样实现？ |
| 9 | `P/source/backend.py`：`PublicEngine` | 最终加载、owned batch、autocast/helper、packet、读出和after-proof怎样闭合？ |
| 10 | `P/source/run_public.py`：`worker/check_results/Manager.finalize_arm` | ordinal分片、原序合并、完整140620验证和官方140178计分如何串联？ |
| 11 | `SDK/pipeline.py`与`SDK/scoring/index02.py` | 官方headline究竟是哪一个字段，为什么不是accuracy/geometric？ |
| 12 | `P/FINAL_OFFICIAL_VERIFIED.json` | 哪些运行和完整性陈述已经有实际证据，而不是配置愿望？ |

### 17.2 最终服务输出的细节

`choice`返回原candidate key、概率字典；`noul`返回正候选对应概率及概率字典。服务还返回实际producer model名和`usage.input_tokens`，即全部独立完整pair长度之和，包含重复prefix和EOS；raw output为None，没有通过生成文本解析答案。源码没有凭空添加Base式confidence或score-level输出字段。[C9:645–658]

最终一次完整public的四个shard合计真实full-pair token约1,556,124,407。它是大量候选重复prefix后的工作量，不能拿140620当“只有140620个短token”。也不能与每臂99,824,365的**训练逻辑pair-token预算**、两遍forward计数混为同一口径。

### 17.3 只读复查示例

以下只读取已经存在的官方汇总，不启动模型/训练/评分：

```python
from pathlib import Path
import json

root = Path('/home/dev/assumeengage/runs/jev-public-eight-fourway-20261113-v1')
summary = json.loads((root / 'SCORES.json').read_text())
for row in summary['models']:
    print(row['producer_arm'], row['decision_index'],
          row['executed'], row['scored'], row['status_counts'])
```

进一步打开`public/<arm>/scores.json`、`COMPLETE.json`、`shards/<arm>/shard-N/WEIGHTS.json`、`<arm>-shardN-EXIT.json`，可分别看官方分数、聚合身份、393张量纯after-proof、actual-wait/exit记录。

**不要直接重启旧manager当复现。**旧root排他、配置/源码/父进程/输入身份都已冻结；同名结果不得覆盖或append。新的正式复现必须有新namespace和重新核对的输入/source/资源绑定，而不是改旧COMPLETE或删除锁文件。

### 17.4 为什么说“可追溯”，不说“空机器一条命令复现”

完整复现仍依赖合法原数据、原ModelScope stock/tokenizer、当前环境、prepared case/sidecar、冻结source和native FA3。本文没有公开私有原始样本、checkpoint、optimizer、激活或逐请求预测，没有把不确定许可的数据打包发布。

已保存的393张量键、source/stat/byte绑定、数据抽样/曝光账本、checkpoint与官方结果可用于本工作区复查；但不是一份对外发布的独立安装包。

一些冻结历史依赖仍位于`/tmp/opencode`，它们是之前的既有资产，不是本报告新建文件。后续迁移必须有独立映射和新身份核对；兼容symlink不等于原resolved stat pin仍自动通过。本文没有迁移、清理它们，也没有修改本轮结果。

<a id="c18"></a>
## 18. 学习总结、术语和自测问题

### 18.1 最重要的八条

1. **布局不等于模型语义。**packing改变物理表示；mask决定谁看谁；loss决定学什么；readout决定怎样输出，四者不能混称同一种改进。
2. **完整case很重要。**数据隔离、监督分母、统计重采样、BFCL式case-exact都不能随便按拆散pair处理。
3. **一处边界也是模型变化。**Candidate-P与state-P即使输入ID不变，可见性图也不同；不能拿一个成功替代另一个失败。
4. **head存储dtype不等于执行dtype。**autocast、GEMM形状和后处理精度都应单独写清。
5. **数值一致、预测一致、总分相同是三种不同命题。**零choice flip不证明概率等价，分数差0.01也可伴随大量逐题变化。
6. **优化必须测真实输出。**opt-v1退步、opt-v2仅在首1000有小幅收益；没有单项因果计时证据就不编开销占比。
7. **并行方式服务目标。**四卡同权重缩短首分等待；同样总工作量下不自动把八模型总时长缩4倍。
8. **学术借鉴要标边界。**Base-inspired、Jina-inspired不是完整复现；未完成和未准入项也属于报告内容，不应隐藏。

### 18.2 小词典

| 词 | 在本实验里的含义 |
|---|---|
| case / field / candidate | 完整原生关联题 / 一个问题 / 一个备选答案 |
| pair | 固定prefix＋某candidate＋真实EOS的一条独立模型序列 |
| realT | 当前波全部真实token行之和，不含业务pad/dummy |
| cu | 累计序列长度，划分独立attention区间 |
| margin | match logit−not_match logit，FP32相减 |
| VJP | 向量—雅可比积：先求dL/dmargin，再分micro反传参数梯度 |
| macro / micro | 一次optimizer更新的完整16case / 为显存分开的实际执行波 |
| logical token | 曝光预算对完整候选pair的一次计账，不含第二遍forward倍数 |
| pure after-proof | 不重新加载权重，直接把评测后状态与原保存张量逐字节比较 |
| all-ok | 每条响应通过执行/协议，不是每条预测正确 |
| scientific only | 可以报告完整实际测量，但不等于数值等价或生产准入 |
| frozen pin | 对源/输入/回执的路径、resolved位置、size/mtime及必要小源码bytes绑定；不是一律内容hash |
| Index point / 百分点 | 正式Index差值用“Index点”；本地mass百分比之差用“百分点”，不混写 |

### 18.3 可用来检验自己是否真正理解的问题

- 将三条独立序列直接拼成一条causal序列，为什么不是正确packing？
- 为什么PB的suffix必须对完整P+S做一次softmax，不能把两个attention输出相加？
- 为什么完整双向P不能任意复用与另一请求相同的最长公共前缀？
- BCE的`1/K`与Field-CE没有额外`1/K`，会如何改变不同候选数field的梯度尺度？
- 为什么JPAIR在one-hot字段上包含CE，又不能把组合收益单独归因“排序方向”？
- 同样都是FP32 head，为什么先gather EOS再head仍可能改变输出？
- 为什么SDK scores.complete=true还不够证明原始140620条都执行且all-ok？
- 为什么D_select每域等权20%的mass排名，不一定与不同区域权重/任务指标/chance校正的官方Index排名一致？
- 四卡各35155条但简单concatenate，为什么不满足原序完整性？
- 一张卡显存只占几GB，为什么不代表把同模型拆成四卡就一定更快？

这些问题分别对应“数据结构、attention数学、目标概率空间、浮点计算、评测指标、调度目标”。如果能用自己的话回答，已掌握本轮改造的主要逻辑。

<a id="c19"></a>
## 19. 附录：材料、代码、回执与边界

### 19.1 实际目录缩写

| 缩写 | 完整路径 |
|---|---|
| T | `/tmp/opencode/jev-clean-trainer-20261112-v2` |
| F | `/tmp/opencode/jev-clean-trainer-20261112-followup-v1` |
| D | `/tmp/opencode/jev-clean-origin-20261112-v1` |
| PK | `/tmp/opencode/jev-packed-mask-20261112-v2/code/packed_mask` |
| S | `/tmp/opencode/jev-base-state-boundary-20261112-v2` |
| N | `/tmp/opencode/jev-terminal-packed-admission-20261112-v1` |
| E | `/tmp/opencode/jev-clean-final-eval-20261112-v1` |
| P | `/home/dev/assumeengage/runs/jev-public-eight-fourway-20261113-v1` |
| SDK | `/tmp/opencode/jev-five-dimension-distilled-20261109/decision-index/decision_index` |
| SITE | `/home/dev/assumeengage/envs/jev/lib/python3.11/site-packages` |
| REPO | 本地研究资料库 `/home/assumeengage/profiler/jev_model_exp`，不是远端安装目录 |

### 19.2 材料来源

- **[M1] Padding-free设计与执行账本**：REPO下`02-研究路径/05-Attention与效率/2026-11-12-global-padding-free-{design,implementation}.md`、`2026-11-12-attention-experiment-queue.md`。最早设计图含EOS-before-head表述，**本报告已按后续实际实现纠正为head-all-realT**，不机械照抄旧草案。
- **[M2] 用户commit与三模式实验**：hash `2e334fb76c4cd4826df37424450b2734101db9f8`；REPO下`02-研究路径/05-Attention与效率/2026-11-10-attention-mask-modes/PATCH.md`、`2026-11-12-commit-2e334fb-experiment.md`。
- **[M3] Base Labs原文**：[Mammoth-1 and Elephant-1](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)，重点2.2/2.3/2.6。原文为9B/27B全参、6.21M数据、Qwen3.5 hybrid、CE+Brier，不能与本轮LoRA配方混同。
- **[M4] Base输入与边界设计**：[vLLM PR63114](https://github.com/vllm-project/vllm/pull/63114)及[当时固定slot_prompt.py](https://github.com/simon-veitner-redhat/vllm/blob/3eded8038a1c0c55eb981e0d38fd301a1eec62bf/vllm/entrypoints/openai/decisions/slot_prompt.py)；REPO同主题`2026-11-12-base-state-boundary-design.md`、`2026-11-12-base-state-fixed-separator-design.md`。当时PR为open，不代表现成发行能力。
- **[M5] Jina v3固定资料**：[模型卡](https://huggingface.co/jinaai/jina-reranker-v3/blob/d7d7e73b6ea138ced340b83865931b5dfb6c97aa/README.md)、[modeling.py](https://huggingface.co/jinaai/jina-reranker-v3/blob/d7d7e73b6ea138ced340b83865931b5dfb6c97aa/modeling.py#L67-L109)、[论文v4](https://arxiv.org/html/2509.25085v4)。固定revision `d7d7e73b6ea138ced340b83865931b5dfb6c97aa`；卡/代码projector维度差异见第5节。
- **[M6] 方法分解及跟进设计**：REPO下`02-研究路径/05-Attention与效率/2026-11-12-flex-block-bidirectional-reranker-design.md`、`2026-11-12-jina-inspired-component-design.md`、`2026-11-12-followup-state-rank-design.md`。
- **[M7] 当时无法读取的知乎链接**：`https://zhuanlan.zhihu.com/p/1897675709696149020`；本报告不转述未获取正文。
- **[M8] 官方kit固定规则**：[README固定revision](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/README.md#what-changed-in-04)，第5/13/20/136/138～144行明确0.4与public0.3、Index及Full规则。不要把其中“what changed in0.3”的20%public误套0.4。
- **[M9] 原研究目标/数据修订**：REPO下`02-研究路径/06-五维训练与Readout/2026-11-12-06b-top-public-research-design.md`、`2026-11-12-r2-complete-case-data-plan.md`、`2026-11-12-clean-case-training-plan.md`。早期warmstart/5M/双seed/只晋级候选等草案被有效修订或后续用户要求取代，本文以实际执行为准。
- **[M10] 浮点数值边界**：[PyTorch2.9 numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html)，batched/slice、TF32与BF16 reduction；文档解释风险，不证明本机失败原因。
- **[M11] Autocast**：[PyTorch2.9 AMP](https://docs.pytorch.org/docs/2.9/amp.html)；具体nesting归零清cache见安装源码`SITE/torch/amp/autocast_mode.py:409–414`。
- **[M12] 正确PEFT helper**：[0.18.0同系列官方helper文档](https://huggingface.co/docs/peft/v0.18.0/package_reference/helpers)；本机实际版本0.18.1，以`SITE/peft/helpers.py:217–251`源码为权威。Context7查询因额度不可用，回到官方源，没有为此安装或升级依赖。

### 19.3 代码定位

| 证据ID | 文件与重点行/函数 |
|---|---|
| C1 | `/home/dev/assumeengage/scripts/pair_data_common.py:27–44`；`D/case_pairs.py:64–109,226–293`：文本/监督/候选编译 |
| C2 | `T/source/model_runtime.py:98–126`；`PK/model.py:146–183`；`P/source/backend.py:520–523,645–658`：head、margin和概率读出 |
| C3 | `T/source/fresh_init.py:14–20,35–146`；`T/init-seed42/FRESH_MANIFEST.json`；`F/FRESH_PRIMARY_VERIFICATION.json`：初态与加载 |
| C4 | `PK/layout.py:70–109,133–181`、`PK/model.py:73–102,126–183`：realT、隔离、mask与可微core |
| C5 | `T/source/train_case.py:60–158,254–256`、`checkpoints.py:68–104`、`case_runtime.py`；`T/seed42.json`：训练/replay/optimizer/cursor |
| C6 | `S/prepare_boundaries.py:101–160`；`F/source/case_runtime.py:195–232`与独立state reader：精确边界 |
| C7 | `D/case_pairs.py:112–140,296–360`、`D/semantic_taint.py:193–198,242–278`、`D/integration_driver.py:280–304`：准入/曝光 |
| C8 | `T/source/objectives.py:7–35`；`F/source/objectives.py:7–74,135–146`：真实loss/coef |
| C9 | `P/source/backend.py:234–729`，特别258–262/295–421/423–539/606–658/692–729：最终完整引擎 |
| C10 | `P/source/run_public.py`，特别157–196/334–428/464–489/518–624/626–879：分片、进程、资源和计分 |
| C11 | `N/consumer.py:439–504`及两臂`NUMERICAL.json/QUALITY.json`：terminal迁移负结果 |
| C12 | `SDK/runner.py:18–23,25–107`、`pipeline.py:13–19,57–118`、`scoring/index02.py:25–106,123–155`、`scoring/index.py:31–116,151–156`、`data/index-0.3.json` |
| C13 | `SITE/transformers/tokenization_utils_fast.py:287–291`、`SITE/peft/helpers.py:217–251`、`SITE/peft/tuners/tuners_utils.py:1538–1553`、`SITE/peft/tuners/lora/layer.py:793–818` |

JSON多为单行，核查时应按字段名找，不虚构漂亮的JSON内部行号。当前live文件行号与历史其他namespace可能不同，引用前先确认完整路径。

### 19.4 运行证据索引

- **[E1] 数据/曝光**：`D/prepared-01/AGGREGATES.json`、`FINAL_VERIFICATION.json`、`SEMANTIC_MEMBERSHIP.json`、`data/schedule-seed42.json`；state边界的41674/615等数目来自同一prepared roster。
- **[E2] fresh与训练完成**：T/F的fresh manifest、每臂`COMPLETE.json`、`ledger.jsonl`、真实checkpoint COMPLETE；每臂43checkpoint。`F/FRESH_PRIMARY_VERIFICATION.json`记录393初态桥接。
- **[E3] 开发指标**：T/F各臂`D_select-cap{10000000,30000000,100000000}-m{0000365,0001122,0003557}/SUMMARY.json`、保存的case/logits/margins；T的`D_SELECT_PRIMARY-cap*.json`及followup独立CPU回算/尺度报告。未知字段/私有payload不在本文复制。
- **[E4] 旧同权重PF A/B**：`/tmp/opencode/jev-padding-free-20261112-v3/eval/`，REPO中`03-证据与图表/05-Attention与效率/attention-queue-20261112/results/full-public/`和队列账本聚合。
- **[E5] state边界**：`/tmp/opencode/jev-base-state-boundary-20261112-v1/`原失败，S的`prepared-01/PREPARATION.json`、`COMPLETE.json`和`FINAL_PRIMARY_VERIFICATION.json`；严格v1 notready没有回写。
- **[E6] 机制/有限准入**：`/tmp/opencode/jev-fullrow-admission-20261112-v1/FINAL_PRIMARY_VERIFICATION.json`、`/tmp/opencode/jev-attention-2e334fb-20261112-v1/runs/`、`/tmp/opencode/jev-flex-20261112-v1/results/`、`/tmp/opencode/jev-jina-components-20261112-v2/FINAL_PRIMARY_VERIFICATION.json`及REPO聚合。
- **[E7] dense→packed**：`N/{C_D_BCE,C_D_CE}-packed/{NUMERICAL,QUALITY}.json`与WEIGHTS/COMPLETE；两臂34/375保留为失败。
- **[E8] 被停止的两候选确认**：`E/confirm/{baseline,challenger}/case-metrics.jsonl`各215条，无完整SUMMARY/COMPLETE；`E/STOP_FINAL.json`保留决策变化。
- **[E9] 最终官方原件**：P的`config.json`、`SCORES.json`、`public/<arm>/scores.json`、`public/<arm>/COMPLETE.json`及全部`results.jsonl`。
- **[E10] 另一个session的历史优化**：`/home/dev/assumeengage/reports/jev-inference-prior-session-20261113.md`；REPO的`01-研究结论/05-Attention与效率/inference-kernel-runtime-20261111/README.md`。它不是老师4B实现，也不是本轮权重。
- **[E11] 本轮opt时间线/前缀比较**：`/home/dev/assumeengage/runs/jev-public-eight-20261113-v1/`、`jev-public-eight-opt-20261113-v1/`、`jev-public-eight-opt-20261113-v2/`的MANAGER_START、BLOCKED及`FIRST_OPTIMIZED_PUBLIC_PROGRESS.json`；对应work目录保留review.diff/report。
- **[E12] 完整独立验收**：`P/FINAL_OFFICIAL_VERIFIED.json`、`FINAL_OFFICIAL_REPORT.md`、`FIRST_OFFICIAL_SCORE_VERIFIED.json`、COMPLETE、32个`<arm>-shardN-EXIT.json`、32个`shards/<arm>/shard-N/WEIGHTS.json`。全量审计发生在最终验收；本次写报告是读回，不重新调用模型或官方评分器。

### 19.5 本次报告与附表文件

| 文件 | 用途 |
|---|---|
| `REPORT.md` | 本文的仓库阅读副本；服务器原文件名为 `jev-clean-eight-detailed-learning-report-20261113.md` |
| [official-benchmarks.csv](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/official-benchmarks.csv) | 8×43＝344行官方逐benchmark结果；原生score/metric、Index raw/skill/chance、coverage/status、是否计入Index |
| [official-areas.csv](../../../03-证据与图表/06-五维训练与Readout/clean-eight-official-20261113/official-areas.csv) | 8×5＝40行五域raw/skill及skill×100 |

三个服务器原文件同在`/home/dev/assumeengage/reports/`；仓库副本按“01研究结论／03证据与图表”分类，使用上表链接。CSV是从各臂已有官方`scores.json`逐项抄录，不是重新评分；`native_score`量纲由metric决定，`index_raw/index_skill/chance`通常为0～1；展示Index为0～100。未进入Index的6个benchmark也保留，`in_index=false`，不混入headline。

### 19.6 本文有意不声称的内容

- 不是完整Base/Jina/Flex新模型配方全部复现。
- 不是知乎文章正文实现。
- 不是“packing或双向attention普遍无用”的证明。
- 不是已找到本轮低分唯一原因。
- 不是多seed稳定收益、全量新旧后端bitwise等价、dense/packed等价或production准入。
- 不是private Full、维护方正式名次、维护方单卡latency验收。
- 不是对全部源数据语义重复/许可/预训练暴露的无限范围保证。
- 本次只写新报告与聚合附表，不修改模型、配置、源码、旧报告或原回执，不自动训练、重测、迁移、清理、commit/push、上传或发布。

---

### 19.7 最终事实审阅的三项更正

独立只读审阅不是无条件PASS：它指出两处事实/数学表述和一处导航名称问题。主代理按原始SUMMARY、实际loss源码与AST函数列表逐项核实后修正：

1. BCE的匹配对象是sigmoid(margin)，不是margin直接接近q；补充了梯度和soft-label最优logit公式。
2. 30M→100M四基础臂NLL都变差；Brier只有三臂变差，packed BCE改善，已写明原始数值。
3. 数据入口名称为normalized_field/make_case/compile_case，不存在make_field别名。

没有修改旧历史报告、源数据、模型或实验回执来迎合新文字。CSV仍直接对应现有官方文件。本次最终化采用一次审阅后的原始来源核实与定点修正，没有让审阅者重新运行模型或评分。

**最终一句话：**这轮把你提供的多种思路拆成了可追溯的布局、attention边界、监督目标和执行调度对照，确实完成了八个fresh模型与全部官方测分；系统完整性和首分等待目标得到实际证据支持，但学习质量并未达到冲榜目标。正确的学习结论，是理解每个变量和证据边界，而不是把所有改动包装成成功提分。
