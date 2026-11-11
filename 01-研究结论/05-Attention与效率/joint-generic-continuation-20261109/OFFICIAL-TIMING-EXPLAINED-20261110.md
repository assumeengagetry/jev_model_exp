# 正式public评测耗时审计（非额外评测）

数据依据：`OFFICIAL_TIMING_AUDIT_20261110.json`。只读统计既有results/logs，不加载模型、不重新分词、不运行新forward、profiler、测试或hash；没有改动正在运行的Engine、权重、设置或结果。

**05:57:47UTC最新只读核验：**runner已于05:57:05.717249UTC结束全部140620请求，counts全部ok，推理阶段elapsed33609.1秒＝9小时20分9.1秒。此刻原生scores.json和OFFICIAL_COMPLETE尚未生成，属于“请求执行完成、官方汇总评分/来源复核未完成”，不是正式总流程完成。

**06:01:12UTC完成核验补记：**原生scores已于05:57:48UTC生成，官方完成回执05:58:05UTC发布。最终计分140178条全部ok，public index24.97、raw index44.15。8项实际完成/来源核对均为true：完整score、精确计分数、完成回执、训练binding、receiptbinding、score文件元数据、冻结snapshot文件清单、原final导出元数据。最终官方报告latency median70.5ms、mean238.7ms、p95676.5ms；不是维护者RTX PRO6000私有样本测速，也不是private Full。没有上传、公开、push或创建PR。

## 2026-11-10 05:54:40 UTC快照

- 原runner总请求140620；评分排除442后140178。已落盘完整结果138546，全部ok；status每10条刷新，显示138540。
- 模型加载7.5708557秒，loading/ready各一次，无重启证据。
- ready20:36:56.632739UTC至快照最后请求05:54:40.247054UTC为33463.614315秒。
- 逐请求`total_wall_ms`求和33425.3298695秒；差额38.2844455秒，是逐请求计时之外的读取、记录、进度等合计，不是数小时排队。
- 已执行2490325条逻辑候选pair，usage累计1554056650非padding输入token；重复题干在每候选中重复计数，不是去重语料token。微批并行，不能把pair数当成同样多次独立forward。
- 当前中间原始请求统计mean241.258ms、median70.474ms、p80 124.744ms、p95 678.352ms；不是维护者指定硬件的私有验收结果。

## 已完成任务的主要耗时

| Benchmark | 请求 | 累计请求秒 | 平均每请求ms | 候选pair总量 | 输入token |
|---|---:|---:|---:|---:|---:|
| POP909 | 2000 | 8362.639 | 4181.320 | 258000 | 565000478 |
| RouterBench | 10000 | 7137.708 | 713.771 | 220000 | 484511674 |
| API-Bank | 508 | 4088.501 | 8048.231 | 26924 | 181538030 |
| CLINC150+OOS | 5500 | 1912.936 | 347.807 | 830500 | 40031055 |
| ToolRet | 1013 | 1588.520 | 1568.134 | 60094 | 48389276 |
| MMLU | 14042 | 982.545 | 69.972 | 56168 | 4964514 |

前三项12508请求占此快照约9%，耗时19588.85秒、约5.44小时，占总请求路径约58.6%。API-Bank每请求53候选、平均357358token（每候选约6743）；POP909每请求129候选、平均282500token（每候选约2190）。最长已见POP909请求约76.2秒；长尾不能用平均题长解释。

## 计时实现

固定kit `decision_index/runner.py:70–91`在请求前记录perf_counter，GPU同步，执行Engine，GPU同步，验证返回协议后计算elapsed。`total_wall_ms`和`model_request_wall_ms`赋的是同一个elapsed，不是两个独立的GPU/端到端计时。渲染、分词、CPU tensor构造、H2D、全部候选前向、D2H/softmax、协议校验和等待同步都在请求时间内；模型加载和之后JSONL编码/flush、进度写入不在该请求timer内。全局elapsed包括后者，但开始于加载/warmup/旧结果读取之后。最终原生评分另算，不在推理阶段进度elapsed内。

## 实现因素与不确定性

当前Engine是SEQ_CLS pair模型，不是生成式LM：每候选拼接相同prefix+自己的suffix+EOS，取match/nonmatch margin后对该field候选softmax。max_microbatch_pairs32是请求内部候选上限，不是32请求并发；max_microbatch_tokens32768限制padding后的微批。40960是单候选上下文上限，不截断。LRU2048只缓存分词结果；use_cache=False，没有前缀KV共享。PEFT以未merge形式推理；通用eager调度、张量构造及每微批.cpu().numpy同步有实现开销。没有跨请求continuous batching。

训练strictFA3与正式推理SDPA是不同路径；SDPA可自动选择fused/kernel后端，不能仅凭配置断言实际运行math attention或归因于某个kernel。未做profiler，所以没有CPU/GPU/分词/PEFT各部分的百分比分解。快速题阶段GPU0曾采样util32%、主线程累计CPU时间近walltime，只能作为调度/小batch开销线索，不是GPU始终32%或单纯CPU瓶颈的证明。没有可据此承诺的优化倍数。

## ETA口径

约01:59UTC时两个粗代理外推：剩余候选pair工作量比例×已耗时≈3.55小时，剩余重复字符工作量比例×已耗时≈2.83小时，随后人工保守给4–8小时。该范围不是置信区间，不是官方时限；没有精确建模每个未知benchmark的attention平方项/固定请求成本/最终汇总时间。

更合理的形式是`T_remaining = sum(各剩余benchmark请求数 × 对应单请求平均成本) + T_scoring + T_overhead`。05:54:40UTC最近5/15/60分钟约13.6–14请求/秒，status剩2080请求；按相同快速题混合仅需约149–153秒推理，之后仍需官方汇总与完成来源复核。ETA随剩余任务组成变化，不应按已完成请求比例线性推整程。

官方要求完整请求、全候选和不截断，但不强制作者使用这种重复prefill实现。当前可移植忠实适配尚非吞吐优化后端，存在作者实现效率成本；不能都归因于官方套件。当前运行不混入新后端/新数值设置，以保持提交版本和结果一致。
