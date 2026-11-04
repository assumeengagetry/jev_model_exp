# 蒸馏为什么可能变差：研究、实际线索与第一轮在跑实验

日期：2026-11-04。用户授权自行研究、规划和直接运行实际实验；本轮不加单测、smoke test、hash或额外review gate，不重新下载环境，不重标Teacher数据。

## 1. 当前证据

同1M原始Qwen起点实验：HARD18.46、SOFT17.54、HYBRID0.5 19.10。纯SOFT比HARD低0.92；HYBRID高0.64。因此不是“蒸馏完全没用”，而是纯软监督不如原标签、混合监督有小幅增益。

本轮目标：寻找实测超过HARD、进一步超过现有HYBRID19.10的配置，并区分真正Teacher信息与普通label smoothing。不能预设哪组必须获胜，所有四组结果完整保留。

## 2. 数学上并没有KD必然优于CE的结论

令Student的binary match probability为 $p=\sigma(m)$，原target为 $y$，Teacher probability为 $q$。T1时：

$$
\frac{\partial CE(y,p)}{\partial m}=p-y,\quad
\frac{\partial KL(q\Vert p)}{\partial m}=p-q.
$$

Hybrid梯度为：

$$
p-[(1-\alpha)y+\alpha q].
$$

也就是说，它等价于对 $r=(1-\alpha)y+\alpha q$ 做CE，差一个不依赖Student的常数。Teacher提供了样本相关softening，也可能引入偏差。模型更大不意味着在当前pair判断格式和目标分布上一定更准确或更calibrated。

如果真实条件概率是 $\eta(x)$，无限容量下纯KD拟合的是 $q(x)$，而不是真实 $\eta(x)$。相对Bayes log-loss的excess是 $\mathbb E KL(\mathrm{Bern}(\eta)\Vert\mathrm{Bern}(q))$，不保证为0。有限数据中soft targets又可能降低方差，因此bias与regularization之间需要实验平衡。

本任务只蒸馏两个概率，不是LM词表级distillation，不能直接套用丰富词表类别关系的“dark knowledge”解释。

## 3. 已实际完成的Teacher-target诊断

在本轮真实训练作业启动时，直接分析既有1M概率和原target，没有新增Teacher/Student forward。它是科学数据分析，不是软件测试；以下agreement/Brier/CE对照的是原训练target，**不是Teacher在目标benchmark上的准确率**，原target也可能有噪声。

| 领域 | pairs | 原target均值 | Teacher概率均值 | 判断冲突率 | 高置信冲突占全部pairs比例 |
|---|---:|---:|---:|---:|---:|
| Knowledge | 169,116 | 21.45% | 32.34% | 15.98% | 1.74% |
| Retrieval | 607,890 | 3.00% | 5.67% | 1.92% | 0.32% |
| Tools | 86,923 | 19.48% | 20.91% | 3.90% | 0.22% |
| Arts | 136,071 | 47.11% | 75.05% | 46.09% | 6.68% |
| 全部 | 1,000,000 | 13.56% | 20.95% | 10.48% | 1.41% |

冲突用Teacher与原target在0.5两侧的方向不一致定义；高置信要求Teacher最大类概率至少0.9。连续target保留原值，方向诊断不把原监督强制argmax。

**最强线索是Arts的系统性target偏移。** 固定alpha0.5会把Arts的平均有效训练target从47.11%推到约61.08%。这与此前Arts在SOFT/HYBRID下降的现象相容，但冲突率不等于Teacher错误率，也没有单独证明它就是因果原因，必须看本轮干预结果。

另一个有用的控制关系：全局HYBRID0.5的平均有效target为17.2507%；epsilon0.1 uniform label smoothing对应17.2006%，两者只差约0.05个百分点。该control因而能检验部分收益是否来自target softening/整体prior变化；这不意味着二者的逐样本targets或分域分布相同。

完整科学诊断数据见`teacher-diagnostic.json`，包括概率分位数、entropy、Brier、CE和校准bins。

## 4. 四卡并行的第一轮实际实验

| 条件 | 实际loss | 重点验证 |
|---|---|---|
| KD_MIX25 |0.75CE＋0.25KL(T1)|Teacher有用但0.5权重可能过强|
| KD_AGREE50 |(1−0.5w)CE＋0.5wKL(T1)，$w=1[(y-.5)(q-.5)>0]$|冲突Teacher targets可能引起负迁移；冲突时回完整CE，所有样本仍训练|
| KD_TEMP2_50 |0.5CE＋0.5×4KL(q_T‖p_T)，T2|改变confidence和梯度geometry，检验T1是否过尖锐|
| KD_LS10 |CE against0.9×原始分布＋0.1×uniform|区分Teacher-specific信息和一般smoothing收益|

T2对缓存的二类概率做softmax(log(clamp(q))/2)，Student logits同温并乘T²。它不是恢复KnowLine原始vocabulary logits，也不保证错误Teacher梯度变小。

agreement gating改善还可能来自平均KD权重降低；若原标签错误，也会屏蔽Teacher的纠错方向。单个LS epsilon不能代表所有label-smoothing配置。这些边界在结果报告中保持明确。

## 5. 保持不变的条件与固定参考

原始Qwen3-0.6B＋共用未训练LoRA/head，r32/alpha64，BF16/SDPA，seed42，batch128，lr5e-5，AdamW/cosine，warmup3%，wd0.01，一遍1M训练。四组使用完全相同的样本索引和epoch顺序，复用同一KnowLine31B概率快照。

固定参考直接复用已完成的raw1M HARD18.46、SOFT17.54、HYBRID0.5 19.10和BASELINE1.76，不重跑它们的训练或inference。评测使用同37项、4,786完整cases、5,935请求、163,671完整候选pair，不缩减候选。

本轮继续使用已经看过的固定子集，属于**探索性实验与模型选择**；完整保留正负结果，不把选出的赢家直接宣称为独立held-out或多seed确认。新一轮根据本轮实际结果再定，不无限盲扫。

## 6. 实际运行状态

2026-11-04 11:12:30在现有allocation275711提交四卡step，实际为275711.65。没有新增allocation，没有恢复原18M/3M Teacher生产，也没有修改既有模型/缓存。

截至11:18:31，四组均已完成实际第100/7,813个optimizer step，各训练12,800 pairs、3,036,350 effective tokens，吞吐约18.2K tokens/s。这里是实际训练进度，不是准备、排队、单测或smoke run。

按前一轮实测速度，本轮预计需要约3.5小时训练＋约1小时并行评测；精确时间随实际运行变化，不提前承诺效果。

完成/失败observer已挂上，最终`report/REPORT.md`、全部对比分数、PNG/SVG图、native metrics、预测和日志会自动保存到本目录。

## 7. Primary references

- Yuan et al., **Revisiting Knowledge Distillation via Label Smoothing Regularization**, CVPR2020：分析KD与learned label smoothing的关系，实验说明Teacher更强不是收益的唯一来源。[论文](https://arxiv.org/abs/1909.11723)
- Cho & Hariharan, **On the Efficacy of Knowledge Distillation**, ICCV2019：更大/更准确的Teacher不必然更适合Student，讨论capacity mismatch；这是视觉实验提供的动机，不是本任务确诊。[论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html)
- Hinton et al., **Distilling the Knowledge in a Neural Network**, 2015：teacher/student同温和T²缩放的经典来源。[论文](https://arxiv.org/abs/1503.02531)
- Guo et al., **On Calibration of Modern Neural Networks**, ICML2017：calibration与accuracy不同，temperature scaling的效果依赖分布。[论文](https://arxiv.org/abs/1706.04599)

这些文献不构成“本实验KD一定赢”的证明。真正结论来自上面四个已开跑、与固定参考对齐的实际结果。
