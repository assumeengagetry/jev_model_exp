# 五维KD public与joint LR1e-3：完整结果

**三阶段已完整结束。五维KD final的Public Index为24.38；从同一KD final以joint峰值LR1e-3重训一遍后，Public Index降至0.81。新高LR配方严重退化，不应替换原2e-5 joint final。**

整链2026-11-11T17:30:17UTC启动、19:32:40UTC完成，即北京时间2026-11-12 **01:30:17～03:32:40**，总墙钟 **2小时2分23.8秒**。19:36:35UTC已直接完成两次全量逐请求对账和来源复核，不是仅采用监视器通知。四张GPU均已释放。

## 1．实验问题与结果

本轮先评测保留的原四维＋Language五维KD final，完整结束后从**同一个KD final**重新训练原joint段，将该段峰值LR由2e-5改为1e-3，再做第二次完整public。不是重训KD，也不是从旧joint final上追加一段。

| 模型 | joint峰值LR | Public Index | Raw Index | 结果来源 |
|---|---:|---:|---:|---|
| 五维KD final | 不适用 | **24.38** | 43.62 | 本轮第一次完整public |
| 原joint final，v2推理 | 2e-5 | **24.91** | 44.07 | 已有历史完整public，本轮没有重跑 |
| 同KD起点的新joint | 1e-3 | **0.81** | 22.63 | 本轮第二次完整public |

- 历史原joint相对KD高 **0.53个Public Index点**。
- 新joint相对KD低 **23.57点**，相对原joint低 **24.10点**。
- 以上均是官方public0.3，不是private Full，不是训练集准确率，也不是0.81%的简单准确率。

### 五个领域的官方skill（原生值×100）

| 领域 | 五维KD | 原joint 2e-5（历史） | 新joint 1e-3 |
|---|---:|---:|---:|
| Knowledge | 9.12 | 9.20 | 0.64 |
| Language | 31.11 | 32.26 | 0.14 |
| Retrieval | 36.28 | 36.06 | 0.85 |
| Tools | 30.99 | 31.80 | 0.69 |
| Arts | 10.54 | 11.57 | 3.09 |

退化覆盖全部领域，不是单个benchmark缺失。各领域coverage均为1.0。各领域skill与总Public Index应按官方原公式解释，不能自行把此表算术平均当总分。

## 2．完整执行与官方计分闭合

两次各自完整执行 **140620条请求**，官方排除442后各自计分 **140178条**，全部status=ok。四个固定分片各35155条，原官方runner逐片推理，再按原Suite顺序合并并调用原官方scorer；不是平均分片分数。

最终直接逐条复核了两份merged results、八份worker结果与原Suite：

- 每条run_id、全部官方metadata、完整payload、当前stage的response.model和分片原始结果精确对应；无重复、缺失或额外记录。
- 两模型均覆盖 **323259个field、2506535个逻辑候选pair、1556124407个逻辑输入token**，不存在截断、删候选或更换输入预算。
- worker environment、阶段模型快照、SCORING_COMPLETE及当前merged results绑定一致，未混用KD与新joint的预测／分数。
- 11份新执行源码与部署CODE_BINDING一致；4份v2数值核心与既有成功bounded-JIT封存源码逐字一致；281份源资产／运行时代码的元数据仍匹配。

## 3．新joint确实完成了一遍训练

| 项目 | 实际终点 |
|---|---:|
| 训练pair | 1667242 |
| 有效student token | 458731913 |
| optimizer更新步 | 13026 |
| 完整关系 | 3647 |
| generic pair | 416934 |
| global batch | 128，尾批42 |
| world size | 4 |
| 峰值LR | 0.001 |
| OOM重试 | 0 |
| 最终micro token预算 | 16384 |

四个task领域各250000pair、Language250308pair，generic416934pair，与冻结清单完全一致。原targets、关系及宏plan只读复用，没有新增Teacher、重抽generic、重复施加Tools偏置或改有效batch。

训练从保留KD的独立`kd-frozen`快照加载全部393个LoRA／head参数，重建AdamW及原warmup／cosine调度；不是使用旧joint final或旧optimizer。逐rank局部loss sum除实际global macro，再NCCL FP32 SUM，不额外除world。所有rank均保存实际确定性FA3 backward证据。

新final的adapter文件仍为80,800,744字节，393张量、196对LoRA，FP32 score[2,1024]；header数据范围与文件大小一致。检查点、optimizer、final和失败／partial保留策略未清理。没有触发自动降LR或丢batch。

## 4．退化现象与因果边界

实际训练正常走到终点，不代表训练质量正常。653条正式progress记录中的loss均有限，但首条loss0.45956、末条0.52664，最后50个记录均值0.64347，不能说训练已收敛到更好结果。

仅对两次**已有完整输出**作同字段统计：候选最大／最小概率差≤1e-6的field，从KD的 **2034/323259（0.63%）** 增至新joint的 **84023/323259（25.99%）**；choice选择首个候选的比例从35.88%增至52.62%。这里包含全部相同field集合，含单候选字段，不是新增模型验收，也不是官方准确率。这支持“新模型候选区分能力明显退化”的现象描述，**不单独证明某一层或某一步发生了何种机制性坍塌**。

LR1e-3是原joint的50倍，当前结果足以否定这份高LR配方的替换价值。但历史原joint为单卡，新训练采用四卡归约，浮点执行路径也不同；因此不将本轮宣称为逐位相同的纯LR消融，不凭分数断言唯一根因。未额外启动检查点评测、模型诊断、学习率网格或补跑实验。

## 5．实际耗时与资源口径

| 阶段 | 实际时间 | 口径 |
|---|---:|---|
| KD完整public | **16分34.2秒** | 四卡并行，含加载／默认warmup／合并／官方评分 |
| 新joint训练 | **1小时28分44.3秒** | trainer累计elapsed，13026步及例行checkpoint |
| 新joint完整public | **16分27.1秒** | 同上四卡public口径 |
| 整链 | **2小时2分23.8秒** | 协调器启动至最终receipt，含阶段切换及终点保存等余量 |

原joint历史单卡训练耗时5小时23分30.8秒，本轮训练elapsed约快3.65倍；这不是相同浮点路径的性能对照。四卡public墙钟只表示质量评测吞吐，不能冒称维护者单进程／单请求延迟验收。

完成复核时coordinator、两个manager与八个eval worker均已退出，四卡显存／利用率均为0；未停止其它任务。

## 6．执行决定与验证边界

- **隔离执行资产：** 新代码／权重放独立库外scratch，研究Git只增计划、报告和聚合证据；代价是执行源码依赖不随此研究仓库公开。原KD、原joint、旧评测、暂停训练和其它会话材料不动。
- **四卡但不扩大算法batch：** 保留global128和完整原子关系，以FP32 SUM利用四卡；代价是历史单卡浮点路径不同，评测并发延迟也不是官方单请求门槛。
- **遵守no-extra-tests：** 没有追加unit／smoke／模型诊断／模型或语料hash；正常完整生产已实际验证，异常故障路径未动态注入验证，不能把静态审阅说成tests通过。
- **完成状态重入项提升处理：** 独立审阅中的成功receipt重入误标blocked问题，由Minor提升Important，在同一控制层修正pass中改为核验后复用原完成时间；其受控重入／故障路径仍未动态测试。另3项Important的未知分片、未知分数和孤儿worker门禁在部署前修正；无其余延期Minor。
- **不自动开展后续实验：** 新高LR结果完整保留，但不覆盖原模型，不自动恢复暂停任务、不继续调LR、不上传／commit／push。

## 7．证据与资产定位

- [最终全量复核](../../../03-证据与图表/06-五维训练与Readout/kd-public-joint-lr1e3-20261111/FINAL_VERIFICATION.json)
- [KD官方原生scores](../../../03-证据与图表/06-五维训练与Readout/kd-public-joint-lr1e3-20261111/results/kd/scores.json)
- [新joint官方原生scores](../../../03-证据与图表/06-五维训练与Readout/kd-public-joint-lr1e3-20261111/results/joint_lr1e3/scores.json)
- [新joint完整训练receipt](../../../03-证据与图表/06-五维训练与Readout/kd-public-joint-lr1e3-20261111/results/TRAINING_COMPLETE.json)
- [整链完成receipt](../../../03-证据与图表/06-五维训练与Readout/kd-public-joint-lr1e3-20261111/results/COMPLETE.json)
- [启动时点聚合证据](../../../03-证据与图表/06-五维训练与Readout/kd-public-joint-lr1e3-20261111/START_VERIFICATION.json)，仅为历史进度，不是当前状态。
- [执行计划](../../../02-研究路径/06-五维训练与Readout/2026-11-11-kd-public-joint-lr1e3.md)

独立远端根：`/tmp/opencode/jev-kd-public-joint-lr1e3-20261111`。

```text
kd-frozen/                       本轮两阶段共用的原KD固定快照
eval/kd/                         第一次完整public原始输出与官方评分
train/JOINT_LR1E3/final/          新高LR退化final，保留不替换原模型
train/JOINT_LR1E3/checkpoints/    本轮全部已提交checkpoint与optimizer状态
eval/joint_lr1e3/                第二次完整public原始输出与官方评分
COMPLETE.json                    整链source-bound完成receipt
```

本地执行源码、只读复核脚本及ledger在`/tmp/opencode/kd-public-joint-lr1e3-20261111-local/`。已归档14份最终原生／聚合JSON，不包含原始题目／预测、模型权重、凭据或编码源码包。建议继续保留原joint 2e-5 final作为当前较优模型；本轮未自动部署任何模型。
