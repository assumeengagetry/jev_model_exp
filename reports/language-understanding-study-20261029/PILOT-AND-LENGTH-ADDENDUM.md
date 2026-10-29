# 协作交付增补：长度长尾与已完成batch/LR短试

本增补仅只读引用新收到的记录。没有停止、重启或覆盖pilot，也未重跑长度诊断。
原报告的91个终点指完整Language参评终点；本节另列8个**仅dev评测**的3M短试终点，不把它们算成完整Language实验或填写虚构Language Skill。

## 长度总体折中必须与长尾一起解释

同一支持面板上，1024相对full的加权Macro-F1下降约0.182个百分点，2048下降约0.021个百分点。
但原始长度1024—2047桶的Macro-F1从full的86.200%降至1024的74.086%，下降12.114个百分点；2048在该桶保持86.200%。
因此1024只满足总体工作阈值；对这一长尾，2048是更保守的推理候选。更长路径或动态路由没有在本轮测量，不能据此指定训练最优长度。

差值按原JSON计算，可能与四位小数显示值相减相差末位舍入。本诊断使用1667对加权支持面板，不是Language Skill。
面板计时包含collation、H2D和同步推理，排除模型加载和原文tokenization；长尾过采样、单次测量、allocator缓存继承等边界见协作完整报告。

来源：`reports/length-label-benchmark-20261029/BENCHMARK-REPORT.md`、`length-results.json`。

## batch/LR pilot已自然完成，结论只限于短程设置

只读`queue-state.json`标记全部完成，结束于2026-10-29 08:00:08 UTC；8个主run各3M、两个续训seed，共享R0，r16/LoRA alpha32，microbatch32。
原始报告和复核报告保持原样：`research/lu-batch-lr-pilot-20261029/STUDY-REPORT.md`、`STUDY-REVIEW.md`。

| 有效batch | 峰值LR | 10任务native dev宏均值±样本标准差 | D0宏pair accuracy | 训练token/s |
|---:|---:|---:|---:|---:|
| R0参考 | — | 0.38136 | 0.80038 | — |
| 128 | 5e-5 | 0.35255±0.01799 | 0.80180 | 12732 |
| 128 | 1e-4 | 0.42561±0.01232 | 0.80099 | 12620 |
| 256 | 5e-5 | 0.32010±0.02777 | 0.79910 | 12576 |
| 256 | 1e-4 | 0.33696±0.03263 | 0.80120 | 12822 |

在这四个r16/3M设置中，batch128＋1e-4在两个seed均最好；batch256＋1e-4没有复现batch128＋5e-5的端点质量，吞吐也没有明显提高。
D0准确率变化小，但每组CE均高于R0，不能写成完全没有遗忘。梯度裁剪发生在93.3%—100%的更新上，
batch改变还改变Adam/weight-decay的更新频次，这些结果不支持通用线性LR缩放规则。

技术检查中，同权重checkpoint开/关的样本相对L2差为0；physical32与16×2的BF16差约0.23%。
原float32 cosine略超1的结果是无效数值估计，原复核已明确说明，不能把它当更强的一致性证据。
该pilot没有新增save/reload前向等价测量，没有公开Language评测；不能直接将短程r16结果外推为r32/20M或完整训练预算的最优设置。

## 对后续比例研究的约束

当前batch/LR pilot已完成且已有报告交付，不补跑、不替换旧产物。后续比例候选应共用一个冻结的模型起点、batch、LR、长度、task预算和自然dev/test。
若根据pilot另选共同设置，需在本轮自然比例控制与所有比例候选上同时执行，不把LR/长度变化与比例变化并成一次因果比较。
本会话仍只做CPU准备，完整训练采样表和worker真实曝光日志尚未部署，不声称比例研究已有结果。

完整数值与版本来源：[只读增补快照](pilot-and-length-update.json)、[8个dev-only端点训练账本](pilot-dev-only-runs.csv)。
原始复核报告：[STUDY-REVIEW.md](../../research/lu-batch-lr-pilot-20261029/STUDY-REVIEW.md)。

## 后续资源诊断已交付（无权重更新）

协作会话已从旧worker/cache复原原profile：medium实际192—205 tokens，两批micro16均pad到208，整体padding4.84%；
long_real_pairs实际2527—2807，两批pad到2576/2808，整体padding3.11%。这些是复原记录，不是新跑的固定长度结果；旧long不能称作固定4096测试。
固定256—8192的checkpoint On/Off及OOM边界测试已由协作会话完成并释放GPU0，共49个单元且无optimizer更新。本会话没有重复测量。
具体结果和运行时padding/原生GQA路径见[资源研究增补](CHECKPOINT-RESOURCE-ADDENDUM.md)；[协作记录](../length-label-benchmark-20261029/GPU-COORDINATION.md)保留预约与本次释放的关系。
