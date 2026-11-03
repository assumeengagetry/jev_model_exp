# 原始Qwen3-0.6B：首个100万pair蒸馏节点

> 按用户本轮“现在开始标注”的授权直接执行首档；执行方式为主代理原地实施。后续扩量不自动执行。

**目标：**不用经过四科微调的MIXED_ALL；从原始Qwen3-0.6B和共同未训练分类头／LoRA初始化出发，比较100万pair上的HARD、SOFT、HYBRID。

**设计：**复用已下载的KnowLine-31B、既有四科pair与编码缓存。首档100万pair；默认四科均衡，每科25万，固定seed 42且不重复抽样。三个训练分支使用完全相同的样本、顺序和初始化。Language不参与训练，公共评测仍保留五维37项。

**节点：**本次只授权累计100万pair。保存0样本起点及100万训练后的结果；以后按累计200万、300万等节点比较，但不自动增加标注或训练量。

**模型：**基座`models/Qwen3-0.6B`，Student起点为未训练的`models/Qwen3-0.6B-common-pair-init-r32`，不是任何训练后的final。Teacher固定为已就绪的`PelaAI/KnowLine-31B-Gen1`。

**配置：**BF16、LoRA r32/alpha64、分类头可训练；新基座微调默认LR 1e-4，AdamW/cosine、warmup 3%、weight decay 0.01、有效batch 128、首档完整一遍。HARD=CE，SOFT=Teacher KL，HYBRID=0.5 CE+0.5 KL。

**来源监督说明：**实际训练池包含合法的0～1概率型来源标签，沿用原工程BCE监督语义，不擅自阈值化或丢弃。HARD在本轮指原始来源监督，不声称所有样本都是人工二值gold。

## 执行顺序

- [x] 保存旧MIXED_ALL起点HARD的105000步断点并隔离旧轮，不混用为新实验对照。
- [x] 创建独立100万pair索引池和配置，四科各25万，共314094515个Student输入tokens；复用原token缓存。
- [x] 在GPU 1–3启动三个Teacher标注副本；GPU 0评测未训练起点。2026-11-03 12:07已记录3010条真实概率。
- [ ] 标注完成后合并100万条独立P(match)，从同一未训练初始化并行训练三个新分支。
- [ ] 完成同一37项评测，输出首个100万节点的SOFT−HARD、HYBRID−HARD及相对起点的差值。

## 文件分工

- `prepare-fresh-qwen-pool.py`：均衡抽样、源索引、pair表和本档数据量。
- `label-fresh-qwen-milestone.py`：复用原生KnowLine标注器，只对独立首档池工作。
- `train-fresh-qwen-milestone.py`：复用三种loss与加载器，独立配置、未训练起点。
- `run-fresh-qwen-evaluation.py`：起点评测及三个新分支的正式公共评测与汇总。
- `run-fresh-qwen-milestone.py`：资源调度、断点接管、只执行首个100万节点。

## 约束与核对重点

不删除旧结果，不更换Teacher，不重新编码全量数据，不误标Language，不用旧续训HARD冒充新HARD。抽样索引、概率数组和训练顺序一一对应；首档最多100万pair。遵从现场约定，不额外运行测试、烟测或全量哈希；以正式标注、实际训练日志和正式评测结果核对进展。不自动提交或推送。
