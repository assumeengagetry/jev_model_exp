# 当前方案总览

结果截至 2026-11-10。配置和完成计数以 [`CURRENT_STATE.json`](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/CURRENT_STATE.json) 为准。

## 结论

- **方案是一个候选匹配模型。** Qwen3-0.6B 加 LoRA rank32 和二分类头，对“材料 + 问题 + 一个候选”输出 match 分数，再在同一题的全部候选之间做 softmax。正式推理不生成文本，也不调用 Teacher。
- **当前唯一的正式权重是 joint final。** 它经过两段训练：先从未训练的 LoRA/head 做五维 KD，再保留参数、加入 generic NLI 续训。两段合计约 292 万次 pair 曝光、8.26 亿 token、22,795 步。
- **Public Index 为 24.99。** 原 Engine 得 24.97，整套耗时 9h20m；加速 Engine 得 24.99，耗时 3h22m。两次使用同一权重，分数基本不变，整套加速约 2.8 倍。
- **距离 28 还差约 3 分。** 相对只训四维的受控 E31，Language Skill 从 13.8 提高到 32.4，Tools 从 43.3 降到 31.9，总分只从 22.49 增至 24.97。
- **Readout 研究没有得到可替换的方案。** Cptr 的内部 NLL 最低，但换序后决策一致率约 79%，原 pair 路线约 99%。

这条路线满足 Decision Index 的输入输出约定，不是 TypeSafe 闭源 Jev 的架构复现。

## 项目内名称

| 名称 | 含义 |
|---|---|
| request | 一次请求：一个 `state` 加一组问题 |
| field（完整题） | 一个问题及其全部候选 |
| pair | 一题中的一个候选与材料、问题拼成的一条输入，是训练和推理的最小单位 |
| margin | 分类头两个输出之差：`logit[match] − logit[not_match]` |
| Teacher、KD | Teacher 是 KnowLine-31B，给每个 pair 打 match 概率；KD 指用这个概率参与监督 |
| generic | 第二段加入的通用 NLI 原标签数据，不属于五个计分域 |
| KD final、joint final | 第一段、第二段训练的终点权重；joint final 是当前正式权重 |
| E24、E31 | 历史四维蒸馏实验。E24 单次最高 23.96；E31 是受控重跑的参照，22.49 |
| Engine | 正式推理实现，有原版和 prefix + CUDA Graph 加速版 |
| Skill、Public Index | Skill 是各 benchmark 原生指标按 chance 校正后的 0–100 分；五个 area 的 Skill 加权得到 Index |
| readout、Cptr | 另一条研究线：把整题全部候选放进一条 prompt 后读出；Cptr 是其中的 pointer 读出臂 |

## 一题拆成多个 pair，逐个打分后在题内归一化

输入分三个层次：

- **request**：一个 `state` 加一组问题，同一个 state 可以有很多字段。
- **field**：一个问题及其全部候选。
- **pair**：从一题中取一个候选，形成一次匹配判断。

四选一的题变成 4 个 pair。判断五个工具是否适用的题是 5 个相互独立的真假字段，可以同时为真。因此 pair 数不等于题数，也不等于 state 数；文中的 token 指实际组装后的输入 token 曝光量。

```json
{
  "state": "包裹已付款，但尚未发货。",
  "questions": {
    "status": {
      "type": "choice",
      "instructions": "包裹现在处于哪个状态？",
      "criteria": {
        "paid": "已付款，待发货",
        "delivered": "已送达"
      }
    }
  }
}
```

模型分别给两个候选打分。假设 margin 为 2 和 −1，题内 softmax 得到约 [0.95, 0.05]，选择 `paid`（数字为人工示意）。

正式 Engine 接受 `choice` 和 `noul`（真假）两种字段，其他 type 返回 `Unsupported`。public 0.3 套件的全部请求均成功处理。readout 研究内部保留了原生 `score` 等级，正式 Engine 没有这个分支。

## 骨干不更新，只训练 LoRA 和二分类头

模型是 `Qwen3ForSequenceClassification`，配置取自已冻结的正式模型：

| 配置 | 实际值 |
|---|---|
| 模型族 | Qwen3-0.6B |
| Transformer 层数 | 28 |
| Hidden size | 1,024 |
| MLP intermediate | 3,072 |
| Attention heads / KV heads | 16 / 8 |
| Head dimension | 128 |
| Vocab | 151,936 |
| 上下文上限 | 40,960 token（每个完整候选序列） |
| Attention | 全层 full attention，causal mask |
| RoPE theta | 1,000,000，无 rope scaling |
| Pad / EOS | 151643 / 151645 |
| 输出类别 | 0 = `not_match`，1 = `match` |

配置中的 full attention 指非滑窗的层类型，mask 仍是 causal。基座是当时从 ModelScope 下载的 `Qwen/Qwen3-0.6B`（来源标记 `master`），没有对应到 HF 的某个固定 commit，复现时使用评测时另存的基座快照。

LoRA 在原线性投影旁增加一条低秩分支：

```text
W(x) + (alpha/r) × B(A(x))
r = 32，alpha = 64，缩放为 2
```

每层的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` 都带这条分支，共 28 × 7 = 196 个投影、392 个 LoRA 张量。再加一个 `score.weight`（形状 [2, 1024]，无 bias），导出共 393 个张量。导出配置的 `modules_to_save` 里 `score/classifier` 名称重复出现，实际只有一个分类头，以张量清单为准。

精度和推理形态：

- LoRA 和分类头用 FP32 存储，骨干为 BF16。
- 正式推理的整个分类 forward 在 BF16 autocast 内；readout 研究另用 FP32 投影，两条路径的数值不同。
- 正式推理不 merge LoRA，不量化。

## 六条路线中只有 joint final 是正式权重

| 名称 | 起点与训练内容 | 用途 | 已有结果 |
|---|---|---|---|
| 旧四维 E24 / 受控 E31 | 原始 Qwen + 共同未训练 LoRA/head，训 Knowledge、Retrieval、Tools、Arts | 历史蒸馏参照 | E24 单次 23.96，同配方重跑 20.51；E31 22.49 |
| 五维 KD final | 同一个未训练初始化，加入 Language 后五维重新联合训练 | 当前权重的第一段 | 训练完成，未单独跑 public |
| **joint final** | **保留 KD final 的参数，再训 generic 原标签 + 五维 KD 回放** | **当前正式权重** | **原 Engine 24.97；加速 Engine 24.99** |
| 原标签 stage2 final | 另一条路线：通用分类 → 五维原标签 | readout 研究的起点 | 未跑完整 public |
| 扩大原标签训练 | 从原标签 stage2 继续扩大 | 已暂停，checkpoint 保留 | 不属于当前主线 |
| Readout frozen / LoRA 四臂 | 从原标签 stage2 分别适配 A、Bcopy、Brand、Cptr | 比较整题读出方式 | 内部留出集结果，不是 Public Index |

第一段复用了 E24 的数据和配方，但从共同未训练初始化重新训练，没有继承 E24 的 23.96 权重。第二段才保留第一段的参数继续训练。

## 两段训练的流程

```text
公开 train 来源、原生任务转换
  ├─ 原四维固定池：每域 250,000 pair
  └─ Language 九任务：250,308 pair（unique 244,710）
             ↓
KnowLine Teacher：原四维标签复用，只新增 Language 标签
             ↓
原标签与 Teacher 概率按域混合，加 3,647 条 Knowledge/Retrieval 关系
             ↓
第一段：共同未训练 LoRA/head → 五维 KD final
  1,250,308 pair，367.0M token，9,769 步，LR 1e-4
             ↓ 保留全部 LoRA/head 参数，optimizer 重建
第二段：五维完整回放 + generic NLI 原标签
  1,667,242 pair，458.7M token，13,026 步，LR 2e-5
             ↓
joint final（当前正式权重）
  ├─ 原 Engine：public 24.97，9h20m
  └─ prefix + CUDA Graph Engine：public 24.99，3h22m
```

五维数据在两段中各训一遍，所以合计的 292 万 pair 是曝光量，不是去重样本数。标签、系数和采样数量见 [数据与训练](DATA-AND-TRAINING.md)；pair 分数如何变成一题的分布见 [推理与评测](INFERENCE-AND-EVALUATION.md)。

## pair loss 下降不等于榜单分数上升

训练优化的是每个 pair 的二元 BCE，外加少量 Knowledge/Retrieval 关系项。榜单按整题选择、多字段 case 全对、F1、相关性等指标计分，两者的单位不同：

- 工具 case 只要一个字段判错，整个 case 就算错，pair 平均 loss 体现不出这一点。
- 五域的 pair 数接近，token 数相差约 3.6 倍，独立题数也不相等。ACOS laptop 的 12,996 个 pair 只来自 38 条 review。
- 五个域共享同一组 LoRA 参数。相对只训四维的 E31，五维 joint final 的 Language 提高、Tools 下降。

各域的得失和 BFCL 的失败形态见 [瓶颈分析](../../../01-研究结论/06-五维训练与Readout/current-scheme-20261110/BOTTLENECKS.md)。

## 阅读顺序与证据

| 要回答的问题 | 文档或证据 |
|---|---|
| 数据、监督、采样和训练是什么 | [DATA-AND-TRAINING.md](DATA-AND-TRAINING.md) |
| 一次请求如何得到输出，两版 Engine 差在哪 | [INFERENCE-AND-EVALUATION.md](INFERENCE-AND-EVALUATION.md) |
| 为什么没到 28 | [BOTTLENECKS.md](../../../01-研究结论/06-五维训练与Readout/current-scheme-20261110/BOTTLENECKS.md) |
| 人工如何复现，缺哪些资产 | [REPRODUCTION.md](REPRODUCTION.md) |
| 实际配置和完成计数 | [CURRENT_STATE.json](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/CURRENT_STATE.json) |
| 源码清单与存放位置 | [SOURCE_MANIFEST.json](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/SOURCE_MANIFEST.json)、[SOURCE-ARCHIVE.md](SOURCE-ARCHIVE.md) |
| 加速版的原生分数和两版对比 | [results/](../../../03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results) |
| Readout 各臂的聚合结果 | [最终报告](../../../01-研究结论/06-五维训练与Readout/readout-study-20261108/FINAL-REPORT-20261110.md) |

执行源码保存在库外辅助归档，不随 Git 推送；配置 JSON 做了路径脱敏和字段摘录；模型和原始题目不在 Git 中。这组文档根据已有结果整理，没有重新训练或评测。
