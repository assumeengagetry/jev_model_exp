# 联合 final 推理加速与新 public 重跑

> **新目录阅读说明：实际执行源码按研究库约定保存在[库外辅助归档](../../../02-研究路径/06-五维训练与Readout/current-scheme-20261110/SOURCE-ARCHIVE.md)，不随报告推送；原生JSON已放在[证据目录](../../../03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results)。下文历史命令／远端路径不机械改写。完整推理机制与已知OOM保护缺口另见[当前方案说明](../../../02-研究路径/06-五维训练与Readout/current-scheme-20261110/INFERENCE-AND-EVALUATION.md)。**

**新完整public已完成：Public Index24.99，推理12157.2秒（3小时22分37.2秒）。旧版33609.1秒（9小时20分9.1秒），整体加速2.765倍，未达到约30分钟目标。** 用户2026-11-10指示按先前建议修改推理并重跑完整public；没有授权上传或第三次public。

最终结论依据13:41:44UTC直接SSH复核，不仅依赖监视器通知。14项完成/覆盖/旧来源/新源码字节绑定/回执元数据检查均为true，见`artifacts/inference-prefix-graphs-20261110/results/FINAL_VERIFICATION.json`。

## 最终结果

| 指标 | 旧路径 | 新路径 |
|---|---:|---:|
| Public Index | 24.97 | 24.99 |
| Raw Index | 44.15 | 44.18 |
| 完整执行请求 | 140620，全部ok | 140620，全部ok |
| 官方排除442后计分 | 140178，全部ok | 140178，全部ok |
| 推理墙钟 | 9h20m09.1s | 3h22m37.2s |
| 请求中位数 | 70.5ms | 16.6ms |
| 请求均值 | 238.7ms | 86.2ms |
| 请求p95 | 676.5ms | 328.4ms |

新pipeline含加载/Graph初始化/官方评分的墙钟12220.829秒＝3h23m40.8s，之后还有约19秒只读全量比较和完成复核；协调器13:37:46UTC（北京时间21:37:46）发布完成回执。31个Graph初始化自身6.085秒，模型加载含捕获15.029秒，数小时成本不是启动捕获造成。GPU0在13:38:56UTC已无显存占用。

五维skill百分数：Knowledge9.27（旧9.24）、Language32.42（32.43）、Retrieval36.14（36.06）、Tools31.94（31.93）、Arts11.43（11.42）。这些都是public，不是private Full；0.02的总分差不宣称质量显著提升。

## 全量输出并非逐位相同

- 140620个run_id与payload、字段/完整候选覆盖一致，逻辑输入token计数差0；两边各140620条实际尝试，无error重试或旧预测复用。
- 308559个choice字段中2575个选择变化：一致率99.1655%，变化率0.8345%。
- 14700个noul字段中82个跨越0.5阈值。
- 2506535个概率值的平均绝对差0.001205717（0.1206个百分点），最大绝对差0.225062558（22.5063个百分点）。不能把平均值小表述成每条差异都很小。
- 模型/输入/读出公式保持，但执行形状、缓存与kernel路径改变后，**这不是输出逐位不变的无损加速**。没有另加模型forward来隔离所有漂移来源，不能把全部差异已证实归因为舍入；首500条完全相同的结论仅适用于该首批样本。

## 为什么仍需3小时多

下面是两次真实请求计时之和，非额外benchmark；细表见`artifacts/inference-prefix-graphs-20261110/results/TIMING_AUDIT.json`。

| 任务 | 旧耗时 | 新耗时 | 旧/新 |
|---|---:|---:|---:|
| RouterBench | 1h58m57.7s | 51m48.4s | 2.30x |
| POP909-CL | 2h19m22.6s | 32m51.0s | 4.24x |
| API-Bank | 1h08m08.5s | 5m30.0s | 12.39x |
| ACOS | 7m22.2s | 16m05.0s | 0.46x，退步 |
| SATA-Bench | 6m06.2s | 9m04.9s | 0.67x，退步 |
| BRIGHT | 5m44.9s | 6m53.5s | 0.83x，退步 |
| BFCL完整任务 | 3m23.6s | 3m43.0s | 0.91x，退步 |

代码中存在明确的调度退步：旧引擎先收集一个request的所有field，再统一候选合批；新`FastPairClassifierEngine.__call__`逐field调用`score_field`，没有保留跨field合批。ACOS1565请求含91092个field，平均58.21field/请求；BRIGHT平均32field、SATA9.40field。短小field被拆成许多小调用，与这些任务的实测退步相符。这是本次实现的问题，不是官方强制逐field执行。

另外，full Graph的batch桶只到8，fork Graph的suffix桶只到32、prefix桶只到4096；未覆盖形状仍走eager。最终运行计数含官方默认warmup：full Graph230306波/full eager27253波、prefix Graph63385次/prefix eager2317次、fork Graph104875波/fork eager15618波。计数本身不能分解各kernel占时；没有跑额外profiler，不宣称精确GPU/CPU瓶颈占比。

因此首500条4.68x不能替代整体2.765x，部分多field任务甚至变慢。要到30分钟，仍需在当前基础上再提速约6.754倍；优先恢复跨field合批及补齐有价值的Graph形状，但单修这些退步也不能保证解决RouterBench/POP909剩余长尾。没有自动启动另一轮修改或public。

## 不变的部分

- 同一已完成joint final、实际训练base、独立prefix/candidate分词、EOS、位置、全候选、FP32保存的真实score头与margin→field softmax。
- 不merge/量化、不换LM/指针读出、不重训、不改温度/prompt、不截断、不丢候选。
- 旧24.97结果及其`official-frozen`只读复用且不覆盖；新代码、新运行目录、新输出和对比回执分开。
- 仅GPU0；不停止/改动GPU1/2/3 readout。不额外测试、烟测、模型/语料hash，不上传/commit/push。

## 实现与执行记录

- 计划：`docs/superpowers/plans/2026-11-10-inference-prefix-graphs.md`。
- 可审计源码：`artifacts/inference-prefix-graphs-20261110/code/`。
- 独立远端根：`/tmp/opencode/jev-inference-prefix-graphs-20261110/`。
- 保留原Qwen3实现；单默认LoRA分支预解析但保留原未合并算术；同field共享prefix、只读candidate fork、有限启动graph池及长尾eager。
- 数值一致不预先假定；两次真实public输出已按run_id逐条对齐，结果如上；原始对比见本地`results/COMPARISON.json`。
- 独立静态审查及修正见`artifacts/inference-prefix-graphs-20261110/review-findings.md`。4项Important恢复/来源问题已按固定upstream源码修正，源码绑定进一步改为真正的字节比较；没有额外测试通过声明。
- 10:13:46UTC启动新生产协调器618837、上游child618846，仅执行新授权完整public。四文件最终部署回读字节完全相同；13:37:46UTC完成，未上传、commit、push或创建PR。

## 验证口径

正式执行必须140620请求全部ok，官方排除442后计分140178全部ok；使用原固定kit和完整0.3 suite。报告推理墙钟与包含graph初始化/加载/评分的总墙钟，两者不混。只读对比不额外执行模型；没有独立测试通过声明。

## 10:14:45UTC 首批生产证据（历史快照，不外推全程）

- 新上游child618846，协调器618837；31个Graph全部实际捕获并封存。模型加载＋捕获15.029秒，启动reserved12.211GiB，非预估值。
- 上游status：770请求全部ok，推理elapsed38.0秒；仍在完整public的开头，不是新正式分数。
- 同ID/同payload首500条BFCL：旧mean70.962ms，新mean15.161ms，约4.681倍；500个field的choice变化0，概率差精确0，逻辑输入token计数差0。
- 这个短题样本尚不足证明共享前缀长任务的总体收益，也不证明140620条都完全等值；**约30分钟整套目标尚未证实**。当前版本封存运行，不中途改代码或拼接旧预测。
- 本地只读监视器`artifacts/inference-prefix-graphs-20261110/watch_public.py`于13:38:29UTC收到完成信号后退出；随后直接核对全部来源和输出，不仅采用通知布尔值。

## 已知未验证与保留项

中断/恢复修正仅静态核对，未另做故障注入。若发生逻辑token不一致，原始结果会完整保留并阻止completion，但当前错误路径不会先保存比较摘要JSON；该小项未扩展实现。本次实际token一致，比较摘要已生成。实际数值漂移来源未单独隔离，未达30分钟，也没有private Full结论。

本地最终证据：`artifacts/inference-prefix-graphs-20261110/results/`内保存原生`scores.json`、`status.json`、`environment.json`、`COMPLETE.json`、`COMPARISON.json`、直接复核`FINAL_VERIFICATION.json`及只读`TIMING_AUDIT.json`；未复制逐请求大文件，远端原始文件完整保留。

沿用官方pipeline本身的suite完整性核验和runner来源记录，不另跑模型/语料hash命令，也不为避免官方内置校验改动kit。按旧33609.1秒推理时间，30分钟目标需要约18.7倍整体加速；不能用单个短任务的加速比宣称整套已达到。

## 实现依据

Context7月额度已耗尽，回退到版本匹配一手源码：

- [Transformers4.57.6 Qwen3](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py)：复用原backbone、Q/K norm、rotary和SDPA；显式mask和位置下提供只读fork。
- [PEFT0.18.1 vanilla LoRA](https://github.com/huggingface/peft/blob/v0.18.1/src/peft/tuners/lora/layer.py)：保持base＋B(A(x))×scale及dtype转换，不执行merge。
- [Torch2.9 CUDA Graph约束](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs)、[2.9.1 graph实现](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/cuda/graphs.py)：保留固定地址，独立pool，禁用autocast临时weight cache；启动后不再捕获新形状。
