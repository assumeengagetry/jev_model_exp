# 四专家 SAMerging 实测

**SAMerging 公共总分：20.21；相对 MIXED_ALL：-2.61；相对 average：-1.25；相对 SLERP：-1.34。**

## 完整公共37项对比

| 模型 | Knowledge | Language | Retrieval | Tools | Arts | 公共总分 | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| MIXED_ALL | 9.69 | 17.44 | 46.26 | 26.08 | 17.78 | 22.82 | 100.00% |
| MERGE_AVERAGE | 8.47 | 16.60 | 35.24 | 37.85 | 10.10 | 21.46 | 100.00% |
| MERGE_TIES | 7.14 | 14.40 | 40.81 | 24.29 | 8.27 | 19.00 | 100.00% |
| MERGE_SLERP_HIERARCHICAL | 8.56 | 16.50 | 35.29 | 38.25 | 10.13 | 21.55 | 100.00% |
| MERGE_SAMERGING | 8.21 | 15.57 | 33.43 | 35.33 | 9.08 | 20.21 | 100.00% |

使用同一重建公共0.3评测的37项、110,087个请求和全部候选；不是私测 Full。原基线直接复用已完成预测/评分，不重跑推理。

## 与当前 KD 的冻结子集对照

以下只比较同一冻结子集分数，不能把它当成相同训练量/监督来源的因果对照；SAMerging学习系数，KD训练LoRA和head。

| 模型 | 冻结子集 Index |
|---|---:|
| BASELINE | 22.39 |
| KD_HARD | 22.49 |
| KD_SOFT | 21.36 |
| KD_HYBRID | 22.38 |
| MERGE_SAMERGING | 20.09 |

## 方法与实际配置

- Knowledge/Retrieval/Tools/Arts四专家；以它们实际共同的pair-classifier初始化为W0。每个有效权重tensor学习每专家一个系数，包含共享二分类head。不是直接平均LoRA的A/B矩阵。
- 每域1,600个无标签输入；不读原始target、不用评测题校准。KL(expert||merged)，T1，任务等权。
- 788个可训练系数，600步，每步每域batch16；init0.5，不clamp；官方SAM实现，SGD base，lr1e-3，momentum0.99，weight decay5e-4，rho0.07，adaptive=True。
- 校准总耗时53.9分钟。FP32合并运算、BF16 forward及单模型导出；推理不保留多专家或额外adapter。

## 与论文/示例的差异

- 这是SAMerging在现有Qwen3二分类专家上的适配实测，不是复现论文的GPT2/GLUE或CLIP数值。论文附录写Adam，但官方NLP代码实际用SGD；本次跟随代码并明确披露。
- 官方GPT2示例分别读取两轮shuffle batch，预计算logits与后续输入可能错位。本次按固定input IDs准确配对。
- 官方GPT2示例在SAM first_step后没有刷新缓存的merged weights。本次每次forward都重建，包括perturbed pass，确保真正执行SAM。
- 所有专家的label space相同，因此合并出一个共享head；不使用GPT2/GLUE示例的每task独立head。

官方代码：https://github.com/arshandalili/SAMerging/tree/a194e8ed8ab790dcae9c02a8d1467e00e86b53e5
论文：https://arxiv.org/abs/2512.21288
