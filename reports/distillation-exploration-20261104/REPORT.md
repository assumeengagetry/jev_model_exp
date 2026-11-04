# 蒸馏快速探索：第一波已启动

**目标：超过当前100万pair HYBRID的公共总分22.63。第一波四组已真实并行训练，尚无新的正式评分，不提前宣称哪种方法有效。**

## 当前在跑的四个想法

| 实验 | 改什么 | 依据与待验证点 |
|---|---|---|
| E01_KD25 | Teacher权重0.25，原始来源监督0.75 | 看当前0.5是否过强；不是假定权重越小越好 |
| E02_DOMAIN_GUARD | Knowledge/Tools权重0.5，Retrieval 0.25，Arts 0 | 保护Teacher与来源冲突较多、上一轮易退化的领域；不删除任何样本 |
| E03_SAMPLE_TRUST | 每条样本权重β=0.5×exp(−2 KL(y‖q)) | 与来源冲突越大越少信Teacher，软来源标签保留原值 |
| E04_INTERCEPT_CAL | 每科拟合Teacher logit截距，再0.5混合 | 校正训练来源上的平均概率偏置，保留科内排序；不声称Teacher排序被修好 |

所有实验均使用同一个原始Qwen3-0.6B、同一未训练分类头/LoRA初始化、同一100万pair、seed42、LR1e-4、batch128、一遍和显式FlashAttention 3。没有新增Teacher标注，也没有从已训练模型续跑。

在T=1和forward Bernoulli KL下，混合损失对Student的梯度等价于目标`t=y+β(q−y)`的BCE。因此先计算有效目标，再进行正常训练；不把不同配方的loss数值直接当效果排名。

## 训练缓存中的线索

| 方向 | 来源目标均值 | Teacher平均P(match) | 与二值来源标签一致率 |
|---|---:|---:|---:|
| Knowledge | 0.25 | 0.34 | 86.51% |
| Retrieval | 0.07 | 0.11 | 93.91% |
| Tools | 0.35 | 0.43 | 89.63% |
| Arts | 0.34 | 0.52 | 59.46% |

上述一致率**不是Teacher真实准确率**，来源也可能有偏差。截距校准只使用当前训练池的q/y，不使用公共评测gold拟合。约1.4%的原始来源标签是软分数，全部保留，不阈值化、不筛掉。

样本门控实际平均Teacher权重约0.35；四科分别约0.34、0.44、0.41、0.21。领域保护平均Teacher权重约0.31。两者并不是同一个固定权重实验。

## 实际启动与评分安排

- 2026-11-04北京时间15:59：四张H800各启动一个独立训练。
- 16:07实查：四组均到360/7813步，处理46080 pair；实际FA3前后向计数均非零，无SDPA回退；GPU利用率95%～100%。
- 每组完成全部100万pair后，自动按同一37项公共benchmark评分；优先校准、样本门控、领域保护、低权重。
- 不跑额外测试、烟测或哈希；训练和正式benchmark本身是idea验证。
- 远端目录：`/home/dev/assumeengage/runs/distillation-explore-20261104`。
- 已注册后台完成／阻塞通知，避免跑完无人处理。

## 资料依据与方法边界

- [Hinton等：Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)：硬/软目标混合与KD温度；不提供本项目最优权重。
- [Confidence Conditioned Knowledge Distillation](https://arxiv.org/abs/2107.06993)：按Teacher对来源类别的支持度调整信任。
- [Teacher’s pet](https://arxiv.org/abs/2106.10494)：蒸馏偏差、子群可靠性；本波领域保护是针对当前问题的改造。
- [Guo等：On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)：仿射logit校准；这里校准到来源监督，不等于真实正确率校准。
- [Weighted Soft Labels](https://arxiv.org/abs/2102.00650)：支持后续Student/Teacher相对困难度动态加权研究。
- [Annealing KD](https://aclanthology.org/2021.eacl-main.212/)：支持后续温度/课程研究方向，不将普通warmup称为其算法复现。

二分类DKD的非目标类分布只有一个元素，非目标KL恒为0，不单独浪费一张卡做等价实验。新的KD温度与Teacher已折入权重的1.44是不同概念，不重复应用原生温度。

## 下一波准备

优先准备LoRA r64/alpha128和全参数微调，各自成对跑HARD与原HYBRID，区分“Student本身更强”和“KD净收益更大”。容量实现正在独立准备，尚未启动。温度2、温度退火、渐进KD和WSL留作后续候选，不同时叠加所有机制。

本次是快速单seed探索；以真实正式评分决定下一步，不把论文结论直接当成本项目已验证收益。
