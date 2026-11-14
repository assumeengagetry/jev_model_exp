# 终端dense控制权重 → packed推理准入设计

## 意图、授权与非目标

延续用户已授权的真实机制／生产准入／冻结dev／完整public研究，保持全局production packed。原六臂已全部完成，dense CE在原dense D_select的终点mass最高，但**不能把dense推理结果直接冒称packed生产质量**。主代理选择在新namespace对两个终端dense控制checkpoint各执行一次完整冻结100M节点D_select的packed consumer验证；这是固定权重运行时迁移，不是新增训练、提分目标或第43seed，也不重跑原dense参考模型。

不改原模型source/config/fresh/checkpoint/optimizer/原预测，不加unit/smoke/hash/依赖，不启用失败prefix-cache/fusions/Graph/dummy token，不开confirm/public、不选赢家、不发布/commit/push/清理。不为占卡复制训练或扩seed。GPU2/3现有STATE/JPAIR持续按原100M训练，不动其控制器或预算。

## 固定对象与作用域

- Producers：原`/tmp/opencode/jev-clean-trainer-20261112-v2/C_D_BCE-seed42`与`C_D_CE-seed42`的真实COMPLETE及其中final_checkpoint，必须完整m3557/56912cases/99824365logicaltokens/三select、allcheckpointsretained、无arm blocker。
- Consumer：新`/tmp/opencode/jev-terminal-packed-admission-20261112-v1/`；本地`/tmp/opencode/jev-terminal-packed-admission-20261112-local/`；每producer独立输出目录，complete/readiness/numerical/eligibility分别记账。
- 每臂原393FP32 trained tensor逐key/shape/dtype/byte载入相同stock＋r32/alpha64＋二类EOS head架构；唯一pretrained loader仍原ModelScope stock，不读旧暴露谱系adapter或Teacher。不是沿已训练权重开始新学习，optimizer根本不构造。
- 明确producer科学scope与consumer`causal/packed/nativeFA3/fullrow`scope分离。consumer不修改producer Context.arm/cfg/snapshot；可用独立只读context view给原FullRow，原Context仍验证真实dense生产者源/准入/数据。
- stock596049920＋LoRA20185088＋activehead2048＝served616237056；unused注册head不当LM head或served参数。

## 执行路径与完整输入

复用实际冻结原8文件中`configure_cuda/build_stock/load_saved_adapter/audit_model/FullRow/CaseStore/slots/micro_plan/evaluate_select`的已核实接口，仅只读import，source/core/native/stock/data/producer checkpoint及new consumer源持续stat与小源码exactbytes绑定。

configure_cuda必须在首次Torch import前，以producer已冻结precision/CUBLAS/原FA3包配置；model CPU build_stock(seed42)后只加载该producer final adapter/head，`fresh=False`，loader逐393byte reload核对。audit before requires_grad改动；eval/no_grad、0backward/0optimizer/0训练曝光。可保持参数requires_grad=True但所有forward在no_grad中，grad必须始终None；不合并或降精度LoRA/head。

原375case（各75）、615knownfield、5190candidate及90unknown统一不forward，原完整prefix/suffix IDs、position reset、EOS151645、candidate/slot/micro顺序完全不变；新推理只把tokenwise layout改packed，实际每operator仅realT。P仍原Candidate边界且modecausal，无state sidecar或目标loss迁移。复用同stable-length/common-padded-admission micro几何预算16384、max32、padmultiple8只是规划，绝不实际补token。

新scope下调用原完整evaluate_select，只在原frozen m3557节点，输出全case metrics、SUMMARY、COMPLETE及实际runtime。其checkpoint_identity标明原final_checkpoint和fixed-weight-consumer，不伪造fresh训练身份。模型权重在GPU move后、eval前与eval后都对原saved393做exactbyte核对，证明无参数更新；保存比较回执而非下载weights。

## 冻结数值与质量合同（先于实际执行）

仅读取producer**已保存**的原100M dense D_select全部375case/logits/margins，不重新运行dense。新packed完整输出与原case/component/field/coordinate/known/nativecandidate/q/order一一对齐；unknown仍90且无logit/假target。

对每case全部known logits `[K_total,2]`和margin分别采用既有PF合同：

- error RMS ≤ `.01 * max(reference RMS,1e-3)`；
- maxabs error ≤ `.02 + .02 * maxabs(reference)`。

原field选择变化只允许原band内：reference top1−top2 gap ≤`.04 + .04*maxabs(reference field margins)`；所有带外flip必须0。K1不制造top2，Native tie顺序保持。

全375case必须两个数值谓词均通过且0带外flip才`numerical_acceptance=true`；否则完整raw/失败保留、`eligible_packed_candidate=false`，不删case、放宽tol、换micro/batch或按正确性retry。有限D_select数值准入不签全public几何等价、缓存serve或官方latency。

质量单独从新实际packed logits/margins与原select q独立回算（原1e-12核对）、全部coverage/case/domain/NLL/Brier/mass；固定权重的layout变化不得冒称学习收益。原dense训练质量报告不回写；consumer结果作为独立后续部署资格。只有数值准入通过，才可在未来冻结候选时引用实际packed质量，不能提前选public模型。

## GPU与保留预算

原BI/PB已实际complete/exited，GPU0/1无本队列预留；只在实时NVIDIA UUID/process查询为空、取得同全局目录gpu0/1 cooperative lease后分别执行两producer。gpu2/3/STATE/JPAIR不改不signal。持lease至own worker实际退出，exactpidfd/父死亡guard/TERM-INT登记critical section沿已修复模式；foreign只停自有，unknown lock记录不接管。若0/1被外部使用则明确BLOCKED/等待，不抢占。

新输出合计5GiB上限，shared20GiB底线不降，另保留正在跑STATE/JPAIR各25GiB减实际用量的未来预算，**以及原staged controller已有5GiB evidence额度**；新export5GiB不暗借原额度。GPU0/1仅一次真实完整375case eval，不加warmup/synthetic/model probes；所有首调用成本如实计入。CPU/source/fresh构造不是forward或训练曝光。

## 完成层级与后续

source review／真实loader393身份／完整375case execution／旧合同数值准入／独立packed质量／资源退出分别记录，任何单层不代替其余。成功/失败都保留全部产物与原checkpoint。以后确认/public协议需另冻结有限terminal候选和统一生产runner，强制Suite验证、每臂140620全执行/140178计分口径仍不变；本任务不开这些入口。

**主代理裁定与代价：** 不直接丢弃可能较好的dense-trained终端权重，也不直接用dense分数进production；改做严格packed真实准入。若错误，只增加两次冻结完整dev的推理成本，原学习轨迹/所有旧证据不变，数值失败就不生产准入。

## Builder-check接口裁定

独立builder-check无重要问题遗留。最小实现保留393参数requires_grad=True，仅eval/no_grad并核对.grad始终None，不构造optimizer。**eval后的权重核对是纯adapter_state与immutable saved oracle的byte比较，不得再次调用load_saved_adapter“验证”而覆盖可能的意外变更。** Consumer arm独立dict，不浅拷贝后修改producer共享arm引用；另调用bind_admission(producer.cfg, consumer_arm)核对packed布局。仅用evaluate_select函数，不调用其main或trainer.Run；函数finally切回train不是更新，外层再eval后不forward。
