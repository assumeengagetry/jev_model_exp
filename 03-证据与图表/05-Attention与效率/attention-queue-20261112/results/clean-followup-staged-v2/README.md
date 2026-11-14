# 显式分阶段续跑v2：MAIN真实通过，已从cursor1连续恢复

## 保留原失败，不重做训练

原followup queue在两正式macro0及独立primary通过之后、MAIN授权之前，因完整磁盘预留差约1.11GB安全退出。原QUEUE_BLOCKED、两cursor1checkpoint、全部first/primary产物和非空globallease metadata均保留；原queue未重启。

external staged-v1 MAIN在rank系数逐位检查失败。实际saved-cache单因素诊断证明：其rank-before-BCE forward节点创建顺序产生12个FP32 coef最大3.7252902985e-9差异；训练producer/独立primary一直exact，BCE-before-rank时MAIN也MAX0。v2只在新控制namespace修正该节点顺序与hostmask-before-tensor，**未改trainer/权重/optimizer/数据/容差、未重跑macro0**。prior-v1源、失败与诊断都保留并被v2 provenance stat绑定。

## 已实际验证

- 3公开external源AST＋枚举delta独立review通过；真实install-only及精确byte readback，不写原DATA/model scope。
- v2 MAIN真正读取两臂实际cache/replay/coef、native partitions/positions/EOS、393raw/clipped/before-after/delta/Adam与latestcursor1checkpoint，独立loss/coef均逐位相同；0model forward/step，未开confirm/public。
- 主代理读回成功报告后，另行原子发布mode444、12字段的外置AUTH。AUTH只在新controlroot，原DATA root没有AUTH；不是producer自报ready就自动续跑。
- 新controller于2026-11-12T17:59:32UTC显式启动。18:02:03UTC增量inotify/pidfd回执确认exact controller/worker均live：STATE/GPU2为macro50／1217745logicaltoken，JPAIR/GPU3为macro63／1595242logicaltoken；两臂都实际恢复原cursor1／20947token，macro0 starts和explicit ownresume都恰好1。
- 当时两臂都能准入，所以并发；原BI/PB已complete，其未来预留为0。真实free157050576896bytes，两新臂剩余25GiB cap＋5GiB evidence＋20GiB shared要求78733088367bytes。**没有降低25/20规则、删除任何checkpoint或声称磁盘容量被系统级锁定。**
- 原global lock inode及旧metadata bytes不改，flock取得后新root successor journals明确绑定旧已退出owner及新owner。same-owner/advisory流程不是抵御恶意同UID或非合作GPU竞争的OS隔离。

## 证据文件

- `INSTALL_COMPLETE.json`、`MAIN_DIRECT_ACTUAL_REVIEW.view.json`、`AUTHORIZE_RESUME.view.json`、`QUEUE_LAUNCH.json`；view明确省略重复完整scope，原全产物保留。
- `ACTUAL_RESUME_ADMISSION.json`是上述一次18:02实际状态，不冒充当前持续进度。
- 两个`*-ADMITTED.view.json`及gpu2/3 successor journals保留准入预算、原inode/metadata及新owner证据。
- actual public source三文件已逐字节回读归档；private case/target/logit/token/tensor均未下载。

**仍未完成：** 两臂完整99824365token/3557calls及三select、独立节点回算、冻结确认与完整public。新P独立backend/neighbor/serve准入及模型梯度独立数学证明仍是明确gap；本阶段仅有限科学fullrow train/dev。只读10M节点与exact-pidfd完成观察器已接入，不额外启动模型/evaluator，也不以中间结果改LR/曝光。
