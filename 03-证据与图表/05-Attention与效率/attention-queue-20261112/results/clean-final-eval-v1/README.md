# 最终固定候选与新消费者：已实际安装，one-shot已启动

CPU freezer按冻结的terminal-only规则实际生成候选manifest；主代理再次回读核实其原始stat为37044bytes／mtime_ns1794517053881585813，policy和freezer源码bytes保持不变。

| 固定role | 终端模型 | D_select mass | NLL | Brier |
|---|---|---:|---:|---:|
| baseline | C_P_CE，seed42/m3557，causal/packed | 59.5955% | 1.1938659 | .5406977 |
| challenger | JPAIR_P_BCE_RANK，seed42/m3557，causal/packed | 61.2437% | 1.1480786 | .5336582 |

- 仅m3557/99824365logicaltoken；没有按10M/30M节点择优，原43checkpoint均保留。
- 两个dense→packed转换虽执行完整，均因34/375case违反原数值合同不eligible，不用其更高点分数绕过门禁。
- 选择规则在看过原六臂100M及新增两臂30M后制定，明确不是原四臂预注册；D_select选择本身不证明未选择总体的稳定提升。JPAIR−baseline描述性case区间仍跨0。
- public角色列表已固定为这两个，不依D_confirm结果换权重、runtime、早期checkpoint或域路由。
- 候选manifest保持其冻结时的 **consumer_ready=false，D_confirm_opened=false，public_run=false**，不事后改写。其后真实consumer层的准入由独立执行回执表述；候选封存本身不代替输入codec、trained393、250-case确认或140620/140178 public证明。

`FROZEN_CANDIDATES.json`仅模型/源/协议来源与汇总指标，不含private case、q、tokens或权重。`ACTUAL_CANDIDATE_FREEZE.json`是实际CPU执行回执。SDK源码是既有snapshot而非git checkout；`SDK_SOURCE_PROVENANCE.json`记录五个实际公开文件与原source pins一致，不继承历史暴露模型的readiness或预测。

## 新consumer实际安装与启动

- 四模块源码经任务审查、限定修复复审和最终3133行独立集成审查通过，MAIN四源与内嵌安装脚本AST语法通过；没有额外unit/smoke或性能试跑。
- 原热路径大历史body放大P1已修，持续stat与小源码exact保留；实际新config47824bytes、SOURCE_SNAPSHOT1802947bytes、小源码总1206155bytes满足冻结上限，不把这些尺寸当latency实测。
- 已实际install-only及MAIN独立四源码逐字节回读，候选原stat/bytes不变；安装回执为0forward、0worker、未读confirm/publicpayload。
- 22:19:51UTC主代理显式启动SYSTEM one-shot manager（`MANAGER_START.json`），固定先codec→确认→独立分析→Suite→baseline fullpublic→challenger fullpublic。此launch不是阶段完成证据，不能先行声称codec/confirm/public通过。
- 四份安装/compact来源/launch聚合和四实际源码已归档；ROOT与配置此后不编辑，旧失败/模型/checkpoint保留。phase事件＋exact-pidfd只读观察器已附着，不轮询或重启任务。

SDK普通non-ok按每10条进度事件检测，每worker可再执行最多9条才拒绝；严格全量all-ok/不重试不变，未冒称首错即时停止。资源遍历failclosed、整体signal-mask/actualwait-before-lease-release、完成依赖fsync及精确临时名单的修复均已源码复核，实际资源与后续分支以真实回执为准。

所有旧失败、模型和产物保留；未发布、提交、上传或清理。新consumer后续实际codec、393/confirm和public结果将逐层追加独立回执，不回写候选manifest。
