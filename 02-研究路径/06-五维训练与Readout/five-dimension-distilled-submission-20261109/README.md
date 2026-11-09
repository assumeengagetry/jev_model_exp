# 五维蒸馏参榜候选：准备记录

用户于 2026-11-09 将最高分蒸馏配方的五维补全模型提升为优先任务，并要求先暂停原始标签扩大续训。

- 训练设计：[docs/superpowers/specs/2026-11-09-five-dimension-distilled-submission-design.md](../2026-11-09-five-dimension-distilled-submission-design.md)。
- 实施计划：[docs/superpowers/plans/2026-11-09-five-dimension-distilled-submission.md](../2026-11-09-five-dimension-distilled-submission.md)。
- `PAUSE-HANDOFF.json` 为旧扩大训练的暂停状态原样副本，不是新模型训练完成回执。

用户已批准本会话直接实施及一次官方公共运行和强制输入校验。旧任务暂停已完成；修正ANLI原题身份后的正式数据准备完成，语言Teacher标注已实际开始，Student与官方公共运行尚未完成。不预填新模型成绩，不自动恢复旧训练，不上传或发布。

独立只读设计审查已完成；据此明确了语言9个任务视图、非讽刺库存封顶重分配、核心英语讽刺的原生类平衡，以及本方案不等同于旧35分语言专科。审查不运行测试或GPU任务，不作为训练通过证据。

训练代码审查发现并修正旧ANLI跨轮组号碰撞、coordinator SIGTERM清理和锁竞争状态覆写问题。旧CPU准备原件保留，正式根目录为 `/tmp/opencode/jev-five-dimension-distilled-20261109-v2`；配置与最终数据量为1,250,308pair／366,985,783tokens，语言250,308pair、unique244,710pair。此次修正通过正式数据准入执行核实，不谎称运行了被用户禁止的测试。

2026-11-09 08:05 UTC 独立读取：coordinator323791，Teacher workers323799／323805／323811，真实概率累计592条，3张卡各约74,652MiB，未见OOM；不是Student已经执行优化的声明。完整结果应以远端当前控制文件为准。

更新至08:17 UTC：Teacher已持久化29,760／244,710条（约12.2%），OOM0；训练协调器仍存活。官方协调器324276已部署并存活，阶段waiting_for_completed_five_dimension_final；不会提前加载Student或启动额外评测。`START_OBSERVED.json`保存此次独立观测，`OFFICIAL_LAUNCH.json`／`OFFICIAL_STATUS.json`保存真实等待队列证据。

推理／官方队列的独立静态审查已完成；据此补齐单候选OOM整体拒绝、信号受保护启动与子进程继承锁、只读冻结模型/实际基座/Engine及运行前后来源复核、完整结果幂等跳过。没有运行用户禁止的额外测试。正式结果、最终导出和后续实际执行路径不能以静态审查冒充通过。
