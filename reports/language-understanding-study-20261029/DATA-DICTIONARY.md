# 附件数据字典与来源口径

- `runs.csv`：91个已有Language参评终点，含起始模型1、50M续训47、正常短预算续训39和应急验证4。复用控制不重复计数；训练中未参评的中途checkpoint、试跑、batch测试和只读dev诊断不计入91。
- `new_tokens`：该阶段参评权重的实际新增训练曝光。R0行是其所选checkpoint曝光342474304；`training_compute_tokens=373545953`另记完整R0训练计算。其他主训练取固定新增token终点，不能直接把这些列当全项目GPU成本或不同独立语料量求和。
- `skill_11`等9列：官方Language0.3各catalog的Skill/100。`language_skill`是官方维度聚合，不是9列的简单平均；原始指标保留在JSON中，不能都解释为accuracy。
- `requests=31715`依据完整Language协议和已有coverage记录；本次没有重新遍历/评分逐请求预测。重复模型评测不会增加独立问题数。
- `dev-metrics.csv`：面板分别标识。`questions_or_fields`是原记录的问题/字段数，不自动等于独立源组数，ACOS尤其不能把字段计成review。不同面板的宏均值不能直接比较。
- `D2-clean-64362-reaudit`使用另存的早期对照clean-dev复核；未拿旧10448对面板的`independent_dev`字段冒充新clean-dev。原字段原样保留在JSON源文档中，以便追溯历史。
- `D2-confirmation-5904`是原生类别分层面板，与原小dev源组不交叉，但不是新的独立测试，且不适合作为新比例研究未经校正的自然分布主验证集。
- `D0-Language-pair-proxy-2006`只记缓存文本对分类准确率，来源validation已被旧研究使用，task/标签分层，既不是Language Skill也不是新独立测试。
- `round-timing.csv`区分主报告、所有评测终点和最终有效attempt启动。首轮47.3分钟为初次尝试字段；85.8分钟才是最终成功attempt发起时刻，不能择较小者声称准备达标。
- `all_evaluations_completed_minutes`与`aggregate_evaluation_minutes`取主报告、辅助验证和守护记录中的较晚终点，避免辅助卡提前结束被误作整轮结束。微小时间差包含报告写入/轮询开销，不是逐kernel延迟。
- `experiment-facts.json`保存归一化终点、原有聚合源文档及补充判断，完整精度不以正文舍入值替代。源文件路径存在性、算术、coverage/error和重复缓存副本一致性经过只读检查；本次未重跑模型、官方评分或权重hash校验。
- `collaboration-and-cpu-prep.json`单独保存协作长度诊断和本会话CPU比例准备；这些不是新增参评模型或比例训练效果，不能加入91个训练终点或宣称新最佳配方。
- `pilot-and-length-update.json`及`pilot-dev-only-runs.csv`是后续收到的协作增补：8个r16/3M、仅dev评测的batch/LR终点与完整长度/归档曝光资料。它们不加入原91个完整Language终点；归档计划的比例不是逐step worker实际曝光。
- `reference_training_exposure`是CPU初版字段名，在`cpu-preparation-v2.json`中明确限定为完整归档计划统计，另列端点行数/token核对。端点一致不自动证明逐step task/label顺序，未来实际曝光须由已提交更新日志记录。

所有原权重、逐请求预测、旧报告与失败记录保留在原实验目录。本报告及附件是新的整理版本，没有覆盖它们。
