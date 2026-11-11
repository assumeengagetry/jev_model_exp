# Readout两层四臂：最终结果与边界

**状态：已实际完成。** 总完成回执时间 **2026-11-10 10:03:03 UTC（北京时间18:03:03）**。本次为文档发布重新只读读取已有结果，并通过原行政completion reader重建／核验**20个来源节点**；读取时间见 [RESULTS.json](../../../03-证据与图表/06-五维训练与Readout/readout-study-20261108/final-20261110/RESULTS.json) 的 `observed_utc`。

**先给结论：** 本轮500万token有界预算下，冻结骨干只训新head不足；LoRA适配后Cptr内部主指标最好，但换序一致率明显弱于原pair，不能直接替换正式方案。没有新的官方Public Index或private Full分数，也没有启动长预算确认。

## 1. 这轮比较的对象

完整设计见[EXPERIMENT-DESIGN.md](../../../02-研究路径/06-五维训练与Readout/readout-study-20261108/EXPERIMENT-DESIGN.md)，实际数值／数据代码位置见[库外源码说明](../../../02-研究路径/06-五维训练与Readout/current-scheme-20261110/SOURCE-ARCHIVE.md)，不在本次Git发布中。

**共同起点是原标签路线的stage2 final，不是当前24.97／24.99的joint final。** 两者不能共用结果标签，也不能把本轮pair参考当成当前正式模型。

统一每题全部候选的一条causal prompt，tokenizer实际筛出255个有效单token码（A开始，最后JT；中间跳过不合法码，不是所有双字母都单token）：

```text
Decide the answer using the state, question and every supplied option.
State:
{state}
Question:
{question}
Options:
A: {完整候选0}
[/option]
B: {完整候选1}
[/option]
...
Answer:
```

真实正确答案码不进入输入。`decide_index`是答案槽最后一个输入token；`option_end_indices`是每个完整候选后的闭合序列末尾token，按实际offset取，不靠EOS/pad猜。

| 臂 | 读出 | 初始化／更新 |
|---|---|---|
| A | 255个选项码对应的原LM行乘最终答案槽hidden，只取本题前K行 | tied输入embedding原行，作为冻结buffer |
| Bcopy | 独立[255,1024] head | 拷贝A相同原行；head可训 |
| Brand | 同形状独立head | Gaussian随机，匹配copy头整体Frobenius norm；head可训 |
| Cptr | `q=Wq*h_decide`，`k_i=Wk*h_option_i`，`z_i=k_i·q/sqrt(128)` | Wq/Wk均[128,1024]、非零Xavier uniform、无bias |

B有261,120参数，Cptr262,144参数，容量近似；这不是完整Sieve复现。首轮无额外LayerNorm、温度、softcap、bidirectional、set-attention、旋转平均或Teacher。

两层彼此独立起步：

- **frozen：** 共同已训主干完全冻结，一次共享feature提取；A零更新，另三臂只训新head。
- **lora：** 每臂重新加载相同原stage2及各自**未适配的新head**，训练相同rank32 LoRA和允许的新head，不继承frozen训练后权重。

所有readout FP32投影／logits／loss，骨干BF16、显式FA3、TF32关闭；embedding冻结。pair参考也明确采用FA3骨干＋FP32分类投影，不声称等同旧正式SDPA＋BF16 autocast路径。

## 2. 实际训练数据与预算

从原始typed完整题和canonical NLI原始train构建，按state／source case/group隔离适配train与holdout，**不把dedup pair池反拼成完整题**。

| 域 | 完整题曝光 | 有效prompt token |
|---|---:|---:|
| Knowledge | 4,103 | 799,896 |
| Language | 3,281 | 799,935 |
| Retrieval | 1,231 | 799,957 |
| Tools | 1,281 | 799,987 |
| Arts | 1,669 | 799,913 |
| generic | 5,427 | 999,883 |
| **合计／每臂每层** | **16,992** | **4,999,571** |

该实际清单无重复曝光。generic约20%token，任务每域约16%；loss是**每完整题一个CE、按题平均**，不等于按token比例加权。

- 训练完整prompt上限4096，不截断题；评估声明上下文40960，候选2–255。
- batch16完整题、micro1；共1062个macro。
- head LR1e-4、LoRA LR5e-5；AdamW decay0.01、clip1.0；ceil(3%steps) warmup后cosine。
- 数据／骨干seed42，新head独立seed20261108；所有臂同一曝光顺序。
- 节点：998,583token／212macro；2,999,833／633；4,999,571／1062；本轮只比较最终节点，不事后挑最好的中间checkpoint。
- 监督为原生完整分布CE；noul展开false/true，原生软目标保留；无LM辅助loss、无KD或ranking项。

骨干历史可能见过相关来源，因此这里只叫**适配留出集**，不是“从未见过的独立测试”。

## 3. 完成性核验

| 项目 | 实际结果 |
|---|---|
| 特征提取 | 完成；共享物理成本不重复算四遍 |
| frozen四臂 | 完成；A=0更新，其余各1062 |
| LoRA四臂 | 完成；每臂1062更新 |
| 两层比较 | 均完成；LoRA阶段复用已完成pair参考，不再执行pair一遍 |
| 调度slot | 11/11 returncode0；forced=false；保存确认均true |
| 来源闭包 | **20/20实际重建核验通过**，已有global receipt一致 |
| 评估 | pair＋八臂各1250题＝1100原题＋150换序题 |
| 完整覆盖 | 各臂原题1100/1100回答；换序150/150；计时125/125 |

其中20来源包含features、8训练、2层compare、1份共享pair评估、8份分支评估，不是只看一个`complete=true`。

frozen头在CPU训练；LoRA训练利用获准GPU1/2/3。完整成本比较在研究训练退出后，统一GPU3串行执行；不拿并行训练占卡时的结果判速度。

## 4. 质量：主指标与补充指标分开

预注册主指标：

```text
D = mean_over_5_domains(mean_question[-sum(y_i*log(p_i))/log(K)])
```

每域200题，共1000；generic100题单独看保持能力。均匀分布的归一NLL为1，D越低越好。

下表`全1100期望正确率`是对hard／soft原生目标统一取`y[argmax(p)]`的平均，不全部是硬标签accuracy，更不是Public Index。

| 层／臂 | 主指标D ↓ | 全1100归一NLL ↓ | 全1100期望正确率 ↑ | generic归一NLL ↓ |
|---|---:|---:|---:|---:|
| pair参考 | 0.61497 | 0.63483 | 68.88% | 0.83343 |
| frozen A | 2.23396 | 2.22761 | 50.64% | 2.16417 |
| frozen Bcopy | 1.20358 | 1.17547 | **57.00%** | 0.89434 |
| frozen Brand | **1.18576** | 1.17005 | 44.72% | 1.01300 |
| frozen Cptr | 2.64804 | 2.65556 | 54.94% | 2.73075 |
| LoRA A | 0.61643 | 0.61450 | 68.25% | 0.59519 |
| LoRA Bcopy | 0.61726 | 0.61480 | 67.63% | 0.59014 |
| LoRA Brand | 0.64862 | 0.64621 | 66.58% | 0.62216 |
| **LoRA Cptr** | **0.60335** | **0.60012** | **69.42%** | **0.56782** |

观察：

1. frozen所有新方案D明显弱于pair；仅从已训pair骨干抽feature再适配小head，在本轮预算下没有解决包装迁移。
2. LoRA适配显著缩小差距；Cptr是本轮D最低分支，比pair低0.01162；未有跨seed或独立正式确认。
3. A与Bcopy D差只有约0.00083，不能宣称独立copy head已经质量胜出；Brand较弱也不能单独识别“预训练语义”的纯效应，初始化整体几何与尺度等一起变化。
4. 按预注册晋级规则，Cptr第一；A/Bcopy在第二名边界差≤0.01时按完整请求median，Bcopy93.56ms低于A94.08ms，故**后续候选为Cptr与Bcopy**。这是规则下的有界研究候选，不是部署胜者；约0.52ms差异未另做统计显著性确认，也没有自动启动长预算训练。

## 5. 逆序：最重要的风险信号

每域预先选30题，一次逆序，输出映回原候选ID后比较；不挑有利顺序，不平均，不按正确性重试。

换序集合来自支持的五域留出题，预先按固定规则排除位置／选项引用等语义不安全题，并要求逆序后完整输入仍受支持；筛选**不删除原质量集合中的题**。规则见实际 `prepare_readout_data.py` 的换序准入与选样逻辑，不能把此一致率外推成任意题随机逆序的结果。

| 臂 | 150题决策一致率 ↑ | 平均概率total variation ↓ |
|---|---:|---:|
| pair | **98.67%** | **0.01194** |
| frozen A | 46.67% | 0.51545 |
| frozen Bcopy | 45.33% | 0.47396 |
| frozen Brand | 30.67% | 0.50285 |
| frozen Cptr | 51.33% | 0.47310 |
| LoRA A | 80.00% | 0.12935 |
| LoRA Bcopy | **82.00%** | 0.13289 |
| LoRA Brand | 75.33% | 0.17392 |
| LoRA Cptr | 79.33% | **0.10130** |

Cptr的质量小幅优势伴随明显换序风险：约五分之一逆序题决策变化。不能把“pointer内容寻址”自动理解成“对选项排列不敏感”。causal下较早key无法看到后面候选；此外仍有位置／模板／训练分布等因素，未逐一消融。

## 6. 完整请求成本：双口径同时报告

同GPU3、五域各25个预注册完整请求，共125；模型加载排除，渲染／分词／完整前向／读出保留。原pair每微批还有安全准入检查；检查次数和wall time已单独计量，不取消原安全保护。

| 路径 | raw median | raw mean | raw p80 | raw p95 | 辅助median：扣实测准入 |
|---|---:|---:|---:|---:|---:|
| pair | 248.46ms | 386.45ms | 371.56ms | 811.08ms | **104.91ms** |
| LoRA A | 94.08ms | 96.85ms | 98.07ms | 103.29ms | 94.08ms |
| LoRA Bcopy | 93.56ms | 96.32ms | 97.46ms | 102.37ms | 93.56ms |
| LoRA Brand | 93.50ms | 95.85ms | 96.53ms | 102.62ms | 93.50ms |
| LoRA Cptr | 94.27ms | 96.53ms | 97.69ms | 100.19ms | **94.27ms** |

frozen四臂完整请求raw median94.19–94.47ms，详表见 [summary.csv](../../../03-证据与图表/06-五维训练与Readout/readout-study-20261108/final-20261110/summary.csv)。

- pair→Cptr raw median约2.64×，**不能全部宣传成模型计算加速**。
- 扣已测准入开销后，辅助median104.91→94.27ms，约1.11×；辅助mean159.98→96.53ms，约1.66×。
- 两种口径使用完全相同125题及相同成功集合；辅助不是kernel-only，也不跨窗口拼接部分候选时延。
- 这组样本pair平均1.496次前向／2747.456输入token，整题路线1次／711.856token；token减少不保证同倍延迟减少。
- 125题最大peak allocated：pair约2.318GiB，整题约2.322GiB；是该研究负载，不是训练峰值或40k/255-way极限显存保证。

不能把本轮94ms与当前正式加速版16.6ms直接排名：模型起点、样本、attention、FP32投影和调度都不同。

## 7. 大K证据与未覆盖范围

1100质量题中，K27–62有23题、63–128有14题、129–255有20题，K>255无题。支持255码和这些有限样本，不等于已证明对任意大K语义泛化。

本轮没有多题joint／question isolation研究，没有bidirectional，没有跨问题prefix cache，没有rotation averaging，没有长预算、多seed、当前joint final迁移或官方整榜复测。

## 8. 发布材料与限制

- [RESULTS.json](../../../03-证据与图表/06-五维训练与Readout/readout-study-20261108/final-20261110/RESULTS.json)：配置、实际码表、共同schedule、训练终点、逐域／来源／K分层指标、计时、逆序及聚合计算成本；不含原始题目、训练文本、请求ID或逐行预测。
- [summary.csv](../../../03-证据与图表/06-五维训练与Readout/readout-study-20261108/final-20261110/summary.csv)：pair＋8臂共9行的可筛选大表，精度不裁成报告中的两位数。
- [库外源码说明](../../../02-研究路径/06-五维训练与Readout/current-scheme-20261110/SOURCE-ARCHIVE.md)：实际8个准备／数值／行政验证源码保留在相邻辅助目录，未上传Git；不代表已经补齐模型／数据的公开release。
- 来源完整性核验是只读completion reader；没有模型构造、张量payload读取、新GPU请求或新增实验测试。

**最终判断：有界研究已完成，Cptr/Bcopy可作为后续研究候选；尚未找到一个能同时保住质量、换序稳定性并明显减少完整计算成本的已验收正式替代方案。**
