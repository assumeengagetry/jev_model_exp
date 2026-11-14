# R1已完成训练的public-only受控恢复

用户已授权自主迭代。R1于2026-11-12T06:37:30UTC左右完成完整训练后，原协调器在训练GPU监控阶段报告外来进程并停止，未启动public。

## 直接证据

06:42:49UTC调用原封存common的`verify_training_complete()`及`verify_ready()`，真实验证1667242pair／458731913token／13026步／3647关系、LR2e-5、OOM0；15个节点0/1000…13000/13026的rank参数MAX差均0；final393张量header/payload范围一致；所有旧自有children/ranks已退出，old public目录不存在。原NO_AUTOMATIC_RESUME保留。

原训练监控只把当时仍live的rank放入允许列表，随后取GPU PID快照，存在正常退出时被误报外来的竞态。日志没有保存触发PID，不能证明当时完全没有短暂外部占用；不把“竞态可能”写成已排除所有外部解释。

## 恢复范围

- 新独立root`/tmp/opencode/jev-06b-research-20261112-r1-eval-recovery`，只读原R1的TRAINING_COMPLETE及JOINT_BRIDGE/final，不复制旧预测，不执行torchrun／optimizer。
- 原R1 code/SOURCE/DATA_READY/停止标记／PIPELINE均不改，原COMPLETE仍不伪造成功。新root记录真实训练引用和新public结果，最终跨namespace聚合说明。
- 新helper仅通过原只读验证器核实训练终点；自己的stop/code/source/input/output在新root封存。源码显式import新eval_common，不monkeypatch旧模块ROOT。
- 四核、stage语义、模型响应名、所有tokenizer／概率规则不变；完整140620请求，官方排除442后140178计分；四片各35155，仍原runner/scorer，model_for指向真实原训练final。
- source清单含原R1全部封存源码／数据来源／训练回执／final／observations和旧停止标记，未知partial不认领。官方强制suiteverify保留，不新增模型smoke/unit/hash。
- fresh只读审阅后生产CPU准入，新鲜GPU空闲准入，只跑一次此前未开始的完整public。3600秒评测预算、精确parent-death及pidfd自有任务清理，不触碰其它任务。

这不是按正确性重试或挑checkpoint；模型固定原完整final。若再阻断，保留现场，不清旧标记、改学习率或重训来掩盖控制失败。
