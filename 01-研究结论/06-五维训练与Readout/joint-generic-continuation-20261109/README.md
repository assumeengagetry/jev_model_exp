# 通用原标签＋五维原蒸馏目标联合续训

> **当前阅读入口：[从零完整方案](../../../02-研究路径/06-五维训练与Readout/current-scheme-20261110/README.md)。本目录记录原正式Engine的24.97；同一joint final后续加速版完整public为24.99，见[加速版最终报告](../../05-Attention与效率/inference-prefix-graphs-20261110/README.md)。下文较早“不发布／不push”等是原执行阶段授权记录；本次用户已明确授权将相关报告提交并推送现有GitHub仓库，不包含模型／原始数据发布。**

## 授权与结果边界

用户于2026-11-09明确要求：**当前五维蒸馏继续训完，再从其final接通用原标签＋五维原蒸馏目标联合续训，最后只评测续训后的固定版本**。用户随后确认继续训练，离线休息。

- 设计：`docs/superpowers/specs/2026-11-09-joint-generic-continuation-design.md`。
- 计划：`docs/superpowers/plans/2026-11-09-joint-generic-continuation.md`。
- 本地实现／ledger：`/tmp/opencode/joint-generic-continuation-20261109/`。
- 远端独立根目录：`/tmp/opencode/jev-joint-generic-continuation-20261109`。
- 唯一父模型：当前五维KD的真正完成final；不使用旧原始标签模型或未训练共同初始化替代。
- 不额外测试、烟测或模型／语料hash；不恢复已暂停的原始标签扩大训练；不上传、发布、commit、push、建PR或启用付费HF Jobs。

## 固定有界预算

五维已有1,250,308pair／366,985,783有效token完整回放一次，原有效KD目标及3,647条K/R原子关系保持。generic只读原始canonical train标签，0→match1、1/2→match0；上限91,746,445token，完整样本前缀截止；总新增不超过458,732,228token。实际pair、token、尾批和步数以真实`DATA_READY.json`为准。

保留rank32／alpha64 LoRA与分类头，以新AdamW／cosine、LR2e-5、batch128、micro32／16384token、seed42、strict deterministic FA3在GPU0续训。generic不新增Teacher、不重复Tools校准、不作为新的榜单得分维度。

## 当前证据（尚非新模型成绩）

- 13:34UTC：旧official仅等待、无child或结果，经身份核实后退出；当前主Student身份未变。明确deferred回执保留，旧LAUNCH归档。
- 13:53UTC：当前五维KD实际6,580／9,769步、842,240pair／249,419,233token，FA3确定性反向真实执行；仍在训练。
- 13:54UTC：新实现19个源码／参考文件已部署并逐字比对；旧official入口deferral guard已部署；复用81个已准入官方kit／suite／回执文件的元数据绑定。**新生产队列尚未启动**，独立静态审查进行中。
- 静态审查、排队和数据准备不能冒充新模型训练或评分；尚无当前五维KD／联合续训的新正式公共分。

模型与结果发布范围、模型名和维护者访问仍需最终确认。官方Full由维护者私测，本机公共运行不能代替官方指定硬件的时延门槛。

## 已激活的真实串行队列

**14:16UTC新队列已实际启动，不再只是设计／暂存。** coordinator343153，CPUprepare343155；生产状态为`preparing_joint_original_generic_data`。实际进程nice19、仅CPU0-1、CUDA_VISIBLE_DEVICES为空；日志已准入knowledge250,000条／42,371,145实际组装token。数据准备尚未完成，不预填generic真实N或步数。

同次独立读取：来源Student327929身份未变，已到**7,540／9,769步、965,120pair／283,507,929token**，实际确定性FA3反向为true；GPU上仍只有此来源训练进程。新DATA_READY和TRAINING_COMPLETE尚未完成，新official尚未启动。

队列现在会自主执行CPU准备→等待来源真正完成／旧Student退出／GPU0空闲→冻结父权重→实际联合续训→新fixed final的唯一完整官方公共运行。未知数据／来源／partial冲突保持现场并阻塞，不重启来源或改用别的checkpoint。

独立静态审查发现的两项Important已针对实际PEFT0.18.1／官方runner源码核实，并修正在最终部署代码中：先把分类头置FP32再加载PEFT且保留全键／dtype／tensor精确准入；拒绝认领无binding的非空官方目录。另绑定实际GPU／驱动／包版本，复核snapshot原来源。最终9个代码文件已逐字部署比对，旧暂存版备份保留。**没有运行被禁止的测试／烟测／额外hash，真实父模型加载和正式Engine加载仍待后续生产发生，静态修正不冒充已通过。**

只读监视器已移交至新链，原本地旧watcher的SIGTERM退出是计划内观察器替换，不是训练中断。readout因主资源绑定变更而安全等待，旧features／partial保留，不放宽门槛抢跑。

**14:18:55UTC进一步生产证据**：新coordinator与prepare的实际FLOCK分别由343153／343155持有；9个最终文件仍逐字匹配CODE_BINDING。五维**全部1,250,308条／366,985,783实际组装token**已完成输入索引准入，与旧曝光精确一致；canonical原始train已有3片／1,139,997行完成索引。当前仍是CPU数据准备，不是联合权重更新。来源KD已到7,640／9,769步、977,920pair／287,001,542token，GPU仍仅来源Student327929。真实原样摘要见`START_OBSERVED.json`；新DATA_READY、TRAINING_COMPLETE、OFFICIAL_COMPLETE仍无完成标记。

## 2026-11-10 最终正式public结果

**两段训练及唯一一次官方public0.3已真实完成。** 最终候选是联合续训`JOINT_HYBRID/final`，不是来源旧KD final。联合段实际1667242pair/458731913token/13026步，generic416934pair/91746130token，来源任务和3647关系完整保留。

正式runner于05:57:05UTC完成140620请求、全部ok，推理墙钟33609.1秒（9小时20分9.1秒）；官方评分排除442后140178条全部ok。原生scores05:57:48UTC生成，05:58:05UTC发布完成回执。**Public Index24.97，raw index44.15**；没有private Full或官方名次结论。原生报告本机latency median70.5ms/mean238.7ms/p95676.5ms，不冒充维护者指定硬件私测通过。

06:01:12UTC独立实际核对score完成/精确计分数/完成回执/训练binding/receiptbinding/score元数据/冻结snapshot文件清单/原final导出元数据8项均一致。监视器完成通知仅作触发，最终声明以本次直接核对为依据。未上传、公开、commit、push或建PR；发布范围和维护者访问仍待用户确认。

运行耗时、重复前缀、微批和计时口径详见`OFFICIAL-TIMING-EXPLAINED-20261110.md`，原始结果的只读聚合见`OFFICIAL_TIMING_AUDIT_20261110.json`。该审计没有重跑forward/分词/hash/profiler或变更推理后端。
