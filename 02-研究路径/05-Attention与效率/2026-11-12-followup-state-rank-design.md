# Clean后续探索：固定state边界与硬标签排序目标

## 授权、问题与非目标

沿用户已明确给出的自主排队研究授权，不逐臂追问。主代理选择两个窄消融，保持现有v2四臂与后续GPU0/1的BI/PB不变；不重跑旧审计、不安装、不额外unit/smoke/compile/模型语料hash、不自动提交/发布/清理。

两臂均为**看过10M/30M后提出的探索性研究**，不是原四臂的预注册比较。单seed结果及case-bootstrap不能证明跨seed稳定性。输入、曝光、初始化、训练目标及系统时间分别记账，不把dev百分比当Public Index。

1. `STATE_NL_PB_P_CE`：原packed prefix_block/CE，P改为state＋一个已有LF。与`C_P_CE`只差mask，与原`PB_P_CE`只差mask边界；不改任何完整pair token、候选、position、slot、micro、q、曝光、optimizer或head。
2. `JPAIR_P_BCE_RANK`：原packed causal/nativeFA3独立pair，仅改为field-mean BCE＋hard-label ranking。与`C_P_BCE`对照，保留原完整case/field，不套prefix-once的物理field长度筛选。

不宣称完整Base配方、Jina cosine/listwise/多正例复现，不启用失败的prefix cache/serve，不选择public checkpoint、不读confirm/public训练或按题路由。未知监督保留raw且不forward/作负例。

## 不变合同与独立scope

新本地`/tmp/opencode/jev-clean-followup-20261112-local/`、远端`/tmp/opencode/jev-clean-trainer-20261112-followup-v1/`，源码从远端v2实际冻结8文件取回并与原封存bytes比较，原版本绝不编辑。新增独立`state_boundaries.py`，共9文件全部source/stat绑定；construction/fresh validator不放宽。

从唯一ModelScope stock重新独立生成same-seed42的393FP32初态，不先加载v2 oracle。生成后CPU逐键dtype-preserving bytes比较v2共同fresh；两个artifact不同、张量必须相同。绝不沿旧已训练adapter/head/optimizer/teacher。served616237056，r32/alpha64，FP32二类EOS head，原Torch2.9.1/Transformers4.57.6/PEFT0.18.1/FA3及precision不变。

原16-case macro、稳定长度micro、seed42完整56912case occurrences、3557optimizer calls、99824365logical pair tokens及warmup107/cosine-to-zero不变，3556positive-LR calls。三个固定select节点365/1122/3557完整375case；同原NLL/Brier/mass/coverage定义，不按组合训练loss选赢家。

## State sidecar契约

只接受实际ready的`/tmp/opencode/jev-base-state-boundary-20261112-v2/prepared-01/COMPLETE.json`及独立`FINAL_PRIMARY_VERIFICATION.json`，版本`base-state-boundary-fixed-separator-v2`、policy`state-plus-one-existing-newline-v2`。绑定prepared全部文件、producer/primary源码、projection config与它的完整inputs/source bytes；持续stat守护，不能仅凭布尔ready。

reader只用train/select原CaseStore与已投影sidecar，不新tokenize，不打开confirm；按`(roster,case_id,field_index,coordinate)`定位，校验ordinal/native split/component/known/所有字段/候选数/pair_lengths/原Candidate prefix长度。unknown条目必须null，不解释raw。无匹配/重复/任何known unavailable均硬失败，不回退Candidate切点。

train和select同一Pair构造入口：对每个known field的原P和每个S，精确重切为`P[:cut]`、`P[cut:]+S`，逐row保持原IDs/EOS/length；`0<cut<=len(originalP)`。不去重重复曝光case，保持slot/position/micro几何。rank臂保持原Pair入口，不启用sidecar。

## JPAIR精确定义与记账

每macro C=16；每case的所有known field分母F_known不变。每field：

`B_f = mean_i BCEWithLogits(m_i,q_i)`；在**原host q**上精确判断`P={i:q_i==1}`、`N={i:q_i==0}`、`H=P∪N`，不能先转FP32再判硬标签。

若P和N均非空，`R_f=logsumexp(m[H])-logsumexp(m[P])`，否则`R_f=0`；λ=τ=1。`L=Σ_c Σ_known_f (B_f+R_f)/F_known(c)/C`，q不归一化，不除active fields或额外K。soft候选只参与BCE，不进入rank；unknown没有margin slot。

原schema q总和约1，严格one-hot K≥2时R就是原field CE，常规soft无q==1则rank0；当前schema不表达多q==1。目标同时改变方向与尺度/clip频率，只能回答固定组合目标是否改善质量，不独立归因为“排序方向”。

纯目标放独立objectives.py，禁止顶层torch import或引入原CandidateLayout。仍以实际完整缓存margin求coef，第二遍严格replay同输入并用原coef，不能按micro重新归一化。保留BCE/CE原实现。

每次macro账本记录BCE/rank/total、各component `dL/dmargin` 范数与总coef尺度、硬正/硬负/soft数、active/zero/K1/无正/无负/soft-only原因（可重叠，不加总当分母）、case/known/unknown覆盖；field细节限正式macro0证据，避免整轮日志无界。实际393参数preclip norm、clip factor、LR及actual update沿原路径，component coef不是component参数梯度。

## 正式首macro与恢复门禁

实现独立saved-artifact验收器，先于GPU启动封存；不额外模型forward。每臂正式macro0消费一次，cursor1完整checkpoint后exit75。primary核验：

- 同v2原控制完整exposures/fields/slot/micro、393before=fresh；State row/pair/position完全不变，只改变P长度；JPAIR第一遍logits/margins必须与旧`C_P_BCE`实际macro0逐位相同。
- 第一遍/replay逐位相同；独立从原host q与实际margin重算目标、component/scales、coef及固定F_known/C；393 raw/clipped梯度、delta、Adam/scheduler、cursor1checkpoint逐项核对，不将finite冒充梯度数学全验证。
- 当前scope绑定到外部报告及最新checkpoint COMPLETE pin，检查macro0仅出现一次且next index1。报告不写回冻结cfg；没有外部primary就不能resume。
- State正式macro0记录每wave实际送入FA3的Q/K/V shape、cuQ/cuK、maxQ/maxK、causal与deterministic；外部primary从原pair/精确cut独立构造预期28层P→P及S→全row非因果调用，逐项匹配两pass真实调用。不额外模型forward；这些调用证据不是另一个数值后端参考。
- **Builder-check裁定：允许有限full-row科学训练续跑。** 当前已签core、独立sidecar、实际新P调用分区、正式macro0重放/目标/393grad/actual-update全部通过即可外部放行；新P-specific跨后端数值和邻居隔离未验证是明确保留的gap，**不阻断这一有限科学训练**，但不许扩签为新P数值等价、全几何不变、cached-serve或production准入。若实际分区、row或首macro合同失败则阻断。原Candidate边界邻居证明不自动覆盖新P，不以自我replay冒充独立mask证明。

失败/partial/所有checkpoint原样保留，不阈值放宽、不静默rewind/retry/重吃macro0。新源码变化一律新namespace与fresh，不修改已消费曝光的版本。

## 资源、队列与总预算

先等待原四臂都完成3557macro/3节点并真实退出；原controller仅预留GPU0/1给BI/PB。新side queue只考虑GPU2/3；原空闲进程检查与own RUN.lock都不是GPU租约。

主代理实现独立GPU lease协调：固定新scope的GPU2/3租约锁跨首macro/审核/续跑全程持有，记录UUID与exact进程身份，启动前后用nvidia-smi确认无外部进程，不修改旧controller，不抢占他人；非合作外部进程仍可能竞争，发现后只暂停本scope自有任务，明确非系统调度器的强互斥保证。监听inotify/pidfd，不通过重启任务探测完成。

每臂25GiB、共享20GiB reserve。启动前核对全盘现存用量与旧BI/PB尚未消费的保留预算（各25GiB），再加两个新臂各25GiB与fresh/evidence5GiB；若并发容不下则串行/阻断，不能删除任何checkpoint。free检查只防本次已知队列，不能冒称系统级容量租约。

独立新queue先正式macro0→外部审核→显式cursor1 resume，完整100M及同三select节点，不开放任意checkpoint/confirm/public入口。后续候选选择/确认与完整public必须另冻结，不自动沿本计划打开。

## 验收层级

设计/源码review、输入边界ready、fresh身份、正式macro机制、完整训练、独立dev回算、冻结确认、完整public分别记录。只有实际发生且主代理直接验证的层级才勾选；缺口不会由“子代理完成”或布尔配置填补。
