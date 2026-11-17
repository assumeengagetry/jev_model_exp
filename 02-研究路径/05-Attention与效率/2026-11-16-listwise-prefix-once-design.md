# 一题多候选与"题目只算一次"：设计草案

> 2026-11-16。**状态：设计草案，没有实现，也没有运行。**
>
> 起因：现在每个候选都单独拼一遍完整题目，你问能不能一题多候选一起打分，并在 BI 或 PB 下这样做。本文的背景数字都来自仓库里已有的结果，出处见第 8 节。

## 结论先行

把"省算力"和"让候选互相比较"分成两件事，按顺序做：

1. **第 1 步，省算力，不改模型。**causal，题目只算一次，候选之间互相看不见，每个候选的位置都从题目末尾重新开始。这和现在的 `C_P_BCE` 是同一个函数，不用训练，拿现有权重做数值对照即可。
2. **第 2 步，测候选交互，需要训练。**从头训练两个分支：J0（候选互相看不见）和 J1（同一题的候选按原顺序 causal 可见）。两者都用 BCE，数据和预算相同，只差候选能否互相看见。
3. **第 3 步，只在第 2 步有收益时做。**让候选全部互相可见，或把题目块改成 PB 式的块内双向。

不做全双向 BI：BI 的题目要读候选，没法只算一次；本轮 BI 只有 3.09 分。

## 1. 背景

### 1.1 现在的做法

每个候选拼成一条独立的 pair：`[说明 + state + 问题 + "Candidate:"] + [候选 + EOS]`。一个 field 有 K 个候选，题目部分就算 K 遍。

field-CE 已经让同一题的候选在 loss 里一起做 softmax（`C_P_CE`、`C_D_CE`），所以"多个候选一起算 loss"已经有了，缺的是输入端的交互。现在训练要两遍前向，原因是 `micro_plan` 把一个 macro 里所有 pair 按长度排序后切 micro，同一个 field 的候选会被分到不同的 micro。

### 1.2 浪费有多大

- 训练时一个真实 macro（16 个 case、19 个 field、60 个候选）：每个候选复制题目共 20,947 个 token，题目只算一次是 10,342 个，约一半是重复计算。
- 官方完整评测一次约 1.56B 个 pair token，其中重复题目占多少没有统计过。
- 旧权重上，推理内核 v1 让题目只算一次并加了 CUDA Graph，完整评测从 9 小时 20 分降到 3 小时 22 分，Index 从 24.97 到 24.99。两项一起做的，没法拆开各自的贡献。

### 1.3 哪些 mask 允许题目只算一次

| mask | 能否只算一次 | 原因 |
|---|---|---|
| causal | 能，结果和现在相同 | 题目的 token 不看后面的候选 |
| PB | 能 | P 块不看候选 |
| BI | 不能 | 题目的 token 会读候选，换一个候选表示就变了 |

在 causal 下，复制题目只浪费算力，不改变学到的东西：每条 pair 里题目的表示都一样，梯度也就是 K 份相加。

### 1.4 本项目已有的相关结果

- **BI 和 PB 在独立 pair 上效果很差。**同样的 field-CE、数据和预算：BI 3.09、PB 2.59、STATE_NL 1.97，causal 的 `C_P_CE` 是 12.95。
- **causal 的题目共享在 native FA3 上数值通过了，PB 没有。**native paged 缓存在一个真实 case 组（题目 333 个 token，2 个候选）上：causal 的 margin 相对 RMS 0.0060，在 0.01 的容差内；prefix_block 是 0.049，没过。那只是推理用的缓存接口，没有反向，只测了一组。
- **J0/J1 的布局代码已经有了。**它在一个真实 macro 上跑通了两遍前向和一步更新，但加载的是历史诊断权重，只用来验证机制，没有从头训练过。
- **本轮 causal 五个分支之间相差最多 1.59 点**（12.83～14.42），都只有一个 seed。

## 2. 布局和三种候选可见性

一个 field 是一条序列：

```text
[说明 + state + Question (+criteria) + "Candidate:\n"] [候选1 EOS] [候选2 EOS] … [候选K EOS]
 └──────────────────── 题目块 P，只出现一次 ─────────────┘ └────────── 候选块 ──────────────┘
```

**读出：**每个候选末尾的 EOS 位置接现有的 `Linear(1024, 2)`，得到 margin 后在 field 内做 softmax。head、loss 和读出公式都不变。模板文字也不变，候选之间直接用原来的 EOS（151645）隔开，不新增分隔符。

候选块内的可见性有三种：

| 方案 | 候选 i 能看到 | 位置编号 | 在 native FA3 上怎么算（每层） | 和现在的关系 |
|---|---|---|---|---|
| **L-A 隔离** | 题目 ＋ 自己（causal） | 每个候选都从题目末尾开始 | 两次调用：① 题目块 causal；② 候选的 query 对"题目 K/V ＋ 自己的 K/V"做 causal | 位置用候选各自编号时，和现在的 pair 是同一个函数 |
| **L-B 顺序可见** | 题目 ＋ 前面的候选 ＋ 自己（causal） | 整条序列依次编号 | 一次调用：整个 field 当一条 causal 序列，和现在的 causal 路径完全相同 | Jina v3 的做法；有顺序偏差 |
| **L-C 全互见** | 题目 ＋ 所有候选 | 两种都可以 | 两次调用：① 题目块；② 所有候选的 query 对整个 field 非 causal | 现有 PB 路径把 S 换成"全部候选" |

几点说明：

- **L-A 的第二次调用**要把题目的 K/V 按候选各取一份，拼上候选自己的 K/V。FA3 在 query 比 key 短、`causal=True` 时按右下角对齐，所以候选第 j 个 token 能看到整段题目和自己的前 j 个 token，正好是 pair 的 causal 语义。这个对齐方式在训练用的 varlen 接口上必须先在 GPU 上验证。拼出来的 K/V 和现在 pair 布局里的 K/V 一样多；hidden、Q、MLP 这些激活只算一次题目，所以变少。
- **L-B 不需要新的 attention 代码**：`cu` 按 field 划分，调用现有的 causal 路径；改的只是布局和 EOS 下标。
- **L-C 配合每个候选从题目末尾编号**，候选之间就没有先后顺序。这是我的想法，没有验证过：多个 token 共用同一个位置，预训练时没见过，必须训练才能用。
- 三种方案的题目块都可以从 causal 换成 PB 式的块内双向（第 1 次调用传 `causal=False`），题目仍只算一次。

### 2.1 代码改动点

下面的路径是仓库里的源码快照，在 `2026-11-10-attention-mask-modes/code/experiments/20261112/` 下；真实运行以服务器上的冻结源码为准。

| 改动 | 位置 | 内容 |
|---|---|---|
| 布局 | [jina-macro/list_layout.py](2026-11-10-attention-mask-modes/code/experiments/20261112/jina-macro/list_layout.py) `build_candidate_layout`（第 311 行） | 已有：题目只出现一次、每个候选的 EOS 下标、`sequential` 和 `candidate_local` 两种位置编号（第 380 行）、按 key 恢复候选顺序。基本可以直接用 |
| attention | [native-mask/packed_mask/model.py](2026-11-10-attention-mask-modes/code/experiments/20261112/native-mask/packed_mask/model.py) `_attention`（第 81 行） | 新增 L-A 和 L-C 两个分支，沿用 `_native`（第 73 行）调用 native FA3；L-B 直接走现有 causal 分支 |
| 绑定下标 | [native-mask/packed_mask/layout.py](2026-11-10-attention-mask-modes/code/experiments/20261112/native-mask/packed_mask/layout.py) `bind_mask`（第 185 行）旁边 | 为 L-A 生成题目下标和 `cu`、候选下标和 `cu`、"题目＋自己"的 K/V 下标和 `cu`；L-C 生成题目和全部候选的下标 |
| 读出 | `model.py` 第 183 行 | 已经按 `batch.eos_indices` 取 logits，换成候选布局的 EOS 下标即可 |
| 训练 | [clean-trainer-v2/case_runtime.py](2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/case_runtime.py) `slots` / `micro_plan`（第 195、213 行） | micro 改成以 field 为单位，按 field 的物理长度切，一个 field 不拆开 |
| 训练循环 | [clean-trainer-v2/train_case.py](2026-11-10-attention-mask-modes/code/experiments/20261112/clean-trainer-v2/train_case.py) `train_macro`（第 60 行） | 两遍前向保持不变，margin 按候选 key 对回原来的 slot 顺序 |
| 推理 | 服务器 `P/source/backend.py` 的 `PublicEngine` | 每个问题建一个候选布局，按 field 物理长度组 batch |

不用 Flex：本项目的 Flex causal 没有通过整模型数值对照，而 native FA3 是已经核对过的路径。现有的 Flex 版 J0/J1 mask（[jina-macro/list_attention.py](2026-11-10-attention-mask-modes/code/experiments/20261112/jina-macro/list_attention.py)）可以拿来交叉核对可见性。

## 3. 第 1 步：省算力，不改模型

**做什么：**L-A ＋ causal ＋ 每个候选从题目末尾编号，用在现有 `C_P_BCE` 的最终权重（第 3557 步）上。不训练。

**怎样验证：**

1. **CPU 静态统计。**训练集（24,600 个 case）、开发集和官方 public（按 benchmark 分）各自的"每个候选复制题目"和"题目只算一次"的 token 数，以及训练集里物理长度超过 16,384 的 field 有多少。只读数据、只做 tokenize，不跑模型。
2. **前向数值对照。**在开发集 375 个 case 上，比较 L-A 和已保存的 pair 输出的 logits 与 margin，用已冻结的容差表（RMS、最大绝对误差、带外翻转为 0）。
3. **反向数值对照。**同一权重、同一个真实 macro，分别用两种布局做两遍前向和反向，按已冻结的梯度和更新容差比较。L-A 的训练路径用 varlen 反向，和只做推理的缓存接口不是一回事。
4. **可选：完整官方评测。**用 L-A 跑一次 `C_P_BCE` 的完整 public。预期 Index 接近 14.42，耗时短于现在的 1 小时 43 分 17 秒。

**判定：**第 2、3 项通过，就可以用于训练和推理；不通过就像之前 PB 的缓存那样记为失败，不放宽容差。

第 2、3 项本身就是这一步要回答的问题；第 4 项是额外的 GPU 时间，由你决定要不要跑。

## 4. 第 2 步：测候选交互，需要训练

### 4.1 两个新分支

| 分支 | 布局 | 位置编号 | 其它 |
|---|---|---|---|
| `J0_BCE` | L-A（候选隔离） | 整条依次编号 | causal，BCE |
| `J1_BCE` | L-B（候选按原顺序可见） | 整条依次编号 | causal，BCE |

两者都和本轮八臂保持一致：同一个原模型，seed 42 新建的 LoRA 和 head，同一个曝光流（56,912 次 case 出现、3,557 次 optimizer step），以及同样的优化器、学习率和精度设置。

**为什么两个都依次编号：**J1 的候选要能看到前面的候选，位置必须往后增长。J0 也用同样的编号，两者就只差可见性。`C_P_BCE`（pair，每个候选各自编号）作第三个参照：

| 比较 | 回答的问题 |
|---|---|
| `J1_BCE` − `J0_BCE` | 候选互相可见有没有用 |
| `J0_BCE` − `C_P_BCE` | 只换位置编号和 micro 组成有没有影响 |
| `J1_BCE` − `C_P_BCE` | 两者合起来，相对本轮最好分支 |

loss 固定用 BCE，因为它是本轮官方最好的目标；field-CE 和排序损失留作另一个变量，等布局有结论后再测。

### 4.2 数据：特别长的 field

一个 field 变成一条序列后，候选多的 field 会很长。J 设计里的训练上限是每个 field 的物理长度不超过 16,384，超过就整个 case 不进训练，而 pair 布局下不会因此排除。这样 J0 和 J1 的训练集就会比 `C_P_BCE` 少一些 case。

先用第 1 步的静态统计看会排除多少：

- **很少：**J0 和 J1 用同样的排除规则，彼此严格对照，和 `C_P_BCE` 的比较注明数据略有不同。
- **不少：**把长 field 切成多个窗口，每个窗口带一份题目。这对 J0 是精确的，但会改变 J1 能看到的候选范围，需要你决定。

### 4.3 候选顺序

训练时用原始顺序。J1 训练完后，在开发集上把每个 field 的候选顺序倒过来再跑一遍，报告选项变化的比例，不重新训练。训练时随机打乱顺序是另一个变量，留到之后。

### 4.4 判定标准（2026-11-16 已确认）

- **主指标用官方 Public Index。**本轮开发集排名和官方排名不一致，开发集 mass 只用来监控训练过程。
- **只有一个 seed**，本轮 causal 分支之间已经相差 1.59 点。判定线：
  - J1 − J0 ≥ +2 点：值得跑第二个 seed；
  - −2 到 +2 之间：看不出差别，是否加 seed 由你决定；
  - ≤ −2 点：停止这个方向。

**代价：**两次训练和两次完整官方评测。每次训练的计算量比本轮的分支少，少多少看第 1 步的统计。

## 5. 第 3 步：只在第 2 步有收益时做

- **L-C 全互见：**候选之间完全可见。配合每个候选从题目末尾编号可以去掉顺序影响，但这种位置用法没有验证过。
- **题目块双向：**题目块内部改成双向（PB 的 P 规则），候选部分沿用第 2 步胜出的方案。题目块仍不看候选，所以仍能只算一次。不过本轮 PB 只有 2.59，PB 的缓存路径也没过数值对照，优先级低。
- **换 loss：**在胜出的布局上试 field-CE 或 BCE＋排序。

## 6. 不在本方案内

- **BI 全双向：**题目块要读候选，无法只算一次；本轮 3.09。
- **Base 式的选项字母读出**（LM head 读 A/B/C）：字母不够上百个候选用，标签映射也要改。它是另一个变量。
- **Jina 的 projector 和 cosine 打分、query 在开头和末尾各出现一次。**
- **改模板文字。**

## 7. 风险和未决问题

- **右下角对齐的 causal：**FA3 varlen 在 `cu_q ≠ cu_k`、`causal=True` 时按右下角对齐，必须先在 GPU 上验证前向和反向。之前只在推理缓存接口上验证过一组。
- **`usage.input_tokens`：**现在报的是所有 pair 长度之和。题目只算一次后，是继续报这个逻辑长度（和历史可比），还是改报实际计算的 token 数，正式评测前要定。
- **超长 field 的推理：**物理长度超过 40,960 时，L-A 可以精确地切窗口，L-B 和 L-C 不行。是报为不支持（受覆盖率惩罚），还是切窗口，要定。
- **训练还能更省：**一个 field 不再被拆到多个 micro 后，每个 micro 的 loss 可以直接算（case 和 field 的分母事先已知），一遍前向加反向就是精确的，两遍前向里的第一遍可以省掉。为了少改训练器，第 2 步先不做。
- **数值不会逐位相同：**换布局后矩阵形状不同。之前题目共享的推理内核也有选项变化。所以第 1 步只承诺"在冻结容差内一致"，不承诺逐位相同。

## 8. 参考

- 本轮八臂的结果和解释：[完整学习报告](../../01-研究结论/06-五维训练与Readout/clean-eight-official-20261113/REPORT.md)、[八个分支逐个讲](../../01-研究结论/06-五维训练与Readout/clean-eight-official-20261113/EIGHT-ARMS-EXPLAINED.md)。
- J0/J1/J2 的原设计和那次接线：[2026-11-12-jina-inspired-component-design.md](2026-11-12-jina-inspired-component-design.md)。
- native paged 缓存的 causal / PB 数值结果：[2026-11-12-native-paged-prefix-fork-design.md](2026-11-12-native-paged-prefix-fork-design.md)。
- 推理内核 v1 / v2 的题目共享和计时：[inference-kernel-runtime-20261111](../../01-研究结论/05-Attention与效率/inference-kernel-runtime-20261111/README.md)。
- Base 的分界设计：[2026-11-12-base-state-boundary-design.md](2026-11-12-base-state-boundary-design.md)。
