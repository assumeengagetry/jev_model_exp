# Terminal候选／一次确认／完整public：后续冻结口径

**状态：协议已冻结；独立builder/source接口检查无重要政策问题遗留，但此文件不代替候选封存及真实consumer运行准入，也不授权立即打开D_confirm或public。** 它在已经看过原六臂100M及后续两臂30M之后制定，不能冒称原四臂预注册；冻结时D_confirm尚未打开，clean新权重尚无public结果。沿用户自主排队授权，不发逐臂确认请求，不发布/上传/commit。

## 1. 有限terminal池，不做public checkpoint搜索

只允许已计划的seed42、完整m3557/99824365token/三D_select全部完成的terminal checkpoint。10M/30M始终只作描述，不挑最高点、不拼接checkpoint、不依据public选择权重。

生产推理一律nativeFA3 fullrow packed、原precision、原二类EOS head、原完整输入/候选、position reset、无Graph假token/未准入fusions/prefixcache。固定字段内margin softmax读出与原engine一致，不能借此改变概率规范。每个predictor统一处理所有benchmark/domain，不能按题/benchmark路由或调用大模型。

有限pool最多8项：

1. C_P_BCE、C_P_CE、BI_P_CE、Candidate-PB_P_CE、STATE_NL_PB_P_CE、JPAIR_P_BCE_RANK的真实terminal packed结果；
2. C_D_BCE/C_D_CE的**终端权重packed consumer**，仅当新完整375case迁移执行、权重未变、旧逐case raw/margin合同及0带外flip全部通过时eligible，并只用新实际packed D_select质量，不拿旧dense分数冒充生产。

任何未完成／loader或coverage失败／参数改变／不完整候选／数值迁移未准入的项不可悄悄补预测进入pool。所有失败原样保留；不新增seed、LR网格或额外训练以填卡。

## 2. D_select选择最多两个具体predictor

所有pool项完整terminal evidence与独立saved-only D_select回算齐全之后，冻结：

- **baseline**固定C_P_CE terminal packed；
- **challenger**为其余eligible pool中terminal primary mass最高者；没有eligible challenger则只baseline。

Primary为原375case/615knownfield、五域case等权的native q[argmax margin]；coverage必须1。排序并列容差1e-12时，依次低NLL、低Brier，再按本文件pool顺序。dense consumer名称沿原dense条目顺序附在最后，不能为打破并列引入另一指标。checkpoint/source/core/stock/precision/renderer/tokenizer/micro/readout、fullrow-only和每个predictor的producer→consumer身份全部pin到独占`FROZEN_CANDIDATES.json`；不从中间点或未准入runtime补候选。

Challenger即使点估计低于baseline，也可作为这一已冻结方法比较的唯一challenger，但必须明确“不构成dev提分”；不能换一批候选反复开confirm。选择本身是开发集选择，不能将所选D_select case区间当作未选择的总体置信。

## 3. D_confirm只确认这份固定候选，不再调参或换权重

只有candidate manifest独占封存、source review和资源准入后，才为这一个固定baseline/challenger集合各执行**一次完整250case（各域50）**的D_confirm。原roster外层名confirm、原case native split为validation，不能假装select绕过旧trainer接口；必须另立只读确认入口，不修改原evaluate_native_dev/main/训练cfg。

完整case/全部known field/全部原候选及q不归一化，unknown raw保留且不forward，不删case。所有known supported时primary等于原known-field mass；若任何known静态不支持，仍保留全250 records、报告conditional与zero-credit known-denominator sensitivity及coverage，**不冒称完整确认通过**，不制造输入或按错误重试。state方法需要confirm输入only精确cut产品，但只有候选冻结后才准备；不能round/fallback/drop。

一次整case配对、五域各50、固定10000bootstrap/seed20261112，原native候选tie顺序。报告mass/NLL/Brier/coverage与完整onehot exact的真实分母。改善确认仅指challenger−baseline primary点估计>0且单一冻结contrast的描述性95%case区间下界>0、覆盖完整；否则明确“未确认”。不把它扩张为训练seed置信或独立证明预训练无暴露。不得根据confirm重训、改LR/clip/λτ、换某个早期checkpoint、挑域或再次换candidate开confirm。

## 4. Public作为固定两predictor的最终测试，不依据其结果继续选择

是否一个或两个public arms，由candidate manifest提前固定：baseline始终；若有challenger则也固定执行它。D_confirm不改变这份public列表（除明确的安全/输入能力阻断，须保留为blocked而非换model）。即使确认不显著也如实标注，不能为了public好看删比较臂。

每臂独立新输出namespace、四个35155不相交shard（若四卡可用），完整**140620**请求全执行；官方排除442后**140178**计分。不得截断、删候选、取子集、跨model补旧预测、按正确性重试或以partial冒充complete。若任一request不ok，保留全部partial并明确本臂未完整通过；不自动换模型补洞。

旧official kit固定`25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`，Suite0.3及官方强制`Suite.verify(strict=True)`/exclusion验证保留；该官方必要corpus校验是既有允许项，不新增模型/语料hash或其他unit/smoke。只从官方payload取state/questions给统一predictor，不让_evaluation/gold/benchmark名进入模型路由。单卡latency与四卡pipeline、原生调用/实际token计数单独报告；完整score不自动等于官方单卡latency或cached-serving准入。

Source loader只原ModelScope stock＋冻结terminal393FP32 adapter/head，before/after pureweight校验；不合并降精度LoRA，不用历史暴露adapter/Teacher/答案查表。模型大小报active served616237056（含LoRA/head，剔除unused原head），不能冒称Full0.4.1维护者私测。

Report所有固定arms的真实Public Index，标记best point仅描述这次预先冻结对照，**不据public再选checkpoint、推理runtime、域路由或继续训练同一研究的候选**。51.99只作为已核验同规模Public标杆比较，不把D_select/D_confirm百分比与其直接比较，也不保证一定超过。不自动发布/上传/推送。

## 5. 失败与计算资源

原所有checkpoints、optimizers、训练／部署／校验失败、部分预测、paused run、source/cfg/manifest均留存，不restart旧failed队列。新scope出错先定位而非改阈值，任何新修复/consumer版本都另namespace并显式source provenance。

0/1终端consumer只短时准入，不干扰2/3训练；确认/public在对应source/候选冻结后才申请实际空闲自有GPU，lease/UUID/exactpidfd/actualwait及总磁盘reserve沿现有review过的模式，不抢占外来任务或租卡。没有把“尽可能充分利用”扩张为自动复制seed43或新增算法。

## 冻结前接口检查的执行边界

独立builder明确：D_select结束选择，D_confirm不二次选模；完整比较的两个public arms在候选manifest中先固定，不因确认/public分数换模型。确认入口必须新立，不改名或放宽旧select-only evaluator。原`scientific-full-row-train-dev-only`不自动升级为生产/public准入；新source/loader/输入能力/资源及官方Suite校验仍需真实证据。旧public流程只能借鉴校验模式，不能继承历史暴露模型的readiness、权重或预测。
