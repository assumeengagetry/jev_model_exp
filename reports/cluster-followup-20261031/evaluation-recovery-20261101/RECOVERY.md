# 2026-11-01 公开评分汇总恢复

## 范围

只修复本轮 `scripts/run-formal-evaluation.py` 的官方 0.3 评分对接；保留四科训练预算、原缓存、模型、融合结果、已完成 GPU 推理与原始预测。不修改第三方评分库，不恢复 Language，不操作其他任务。

## 已确认的原因

- 控制器在 `score_model -> benchmark_summary -> score` 触发 `Null gold may only be handled by ranking scorer`。
- PhishNChips（benchmark 56）的 2000 个请求含按规范不计分的信号字段，其 gold 为 null；只有 verdict 应进入该任务的主评分。
- 现有对接把全部 37 个任务传入旧通用评分器，再单独调用 added.report，顺序错误。
- 官方 `decision_index.pipeline.score_run_v02` 先按 spec.added 拆开基础任务与新增任务，新增任务交给 `added.report`；还会把 spec.metrics 传给基础汇总（ACOS 使用 per-review F1）。本次按同样方式修复。
- 读取原始与加载后结果：均为 110,087 条且均保留 catalog_id，全部 status=ok；无需修复元数据，也无需重新推理。

## 执行清单

- [x] 读取当前配置、ACTIVE_TRAINING_SCOPE.json、日志和官方参考流程。
- [x] 保存当前评测脚本，准备最小任务分流及官方指标映射修复。
- [x] 原子部署修复，并以原控制器从完成标记恢复正式流程。
- [x] 核实首个模型正式评分成功、四个 worker 完成标记被复用，下一模型开始实际推理。
- [x] 核实后台控制锁、watcher 与十条件完成门槛维持有效。

项目明确不要求额外测试、烟测或哈希；使用已失败的正式评分及修复后的正式续跑作为验证依据，不额外启动模型测试。

## 实际恢复依据

- 2026-11-01 09:00:28 UTC：仅原子替换本轮评测脚本，备份到远端本轮目录的 `evaluation-recovery-20261101/`，控制器以 PID 98053 恢复；正式评测父进程 PID 98070。
- 09:01:37 UTC：Knowledge 正式完成标记已生成，110,087 / 110,087 请求 status=ok，37 个 benchmark，覆盖率 1.0，公共总分 17.83（仅此单模型，不代表混合或融合对比结论）。
- PhishNChips 按官方只计 verdict：2000 条请求、2000 个计分字段、accuracy 0.5425；ACOS 指标明确为 per-review F1，不再误用 case-exact accuracy。
- Knowledge 四个 worker 完成标记均保留恢复前的时间，未启动新的 Knowledge GPU worker。
- 同时观测到 Retrieval 四个正式评测 worker（98101—98104）运行，四卡显存已分配，GPU 利用率分别为 82%、93%、91%、96%；控制器持有本轮排他锁。
- 原四专家、MIXED_ALL、五融合完成标记全部复用；有效四科范围、完整一轮曝光、十个公开条件和后续蒸馏门槛未改动。
- 本地 watcher 2765321 仍运行；新评分成果由既有 watcher 按原 milestone key 通知 SCP，会话通知不由本次修复手工重复发送。

## 操作附记

首次恢复命令的排他锁检查使用只读文件描述符，在集群文件系统上返回 EBADF；该命令在任何文件部署或进程启动前退出。改用与原控制器一致的 a+ 打开方式后恢复成功，没有放宽锁约束。
