# 三mask同曝光CE：完整100M后独立回算

## 实际完成与分析边界

原controller于2026-11-12T17:37:42UTC完成BI_P_CE和PB_P_CE并退出；连同原四臂共六臂完整训练结束。每臂同seed42共同fresh、相同完整曝光56912case occurrences／3557optimizer calls／99824365logical pair token，最后LR0／3556positive-LR calls，10M/30M/100M三固定select齐全。所有原checkpoint保留。

主代理于17:47:14～17:47:19UTC，用独立完整standalone CPU分析器从原375-case select与三臂实际logits/margins重算全部field/case/domain、native calls、pre/post checkpoint及配对case指标。每臂每节点375case／615known field／5190candidate／90unknown，coverage1.0；最大metric误差1.8874e-15以内。0模型forward／新eval／optimizer／confirm／public。

分析器是原已执行四臂分析的枚举独立移植：仅更换三mask arms/contrasts/output，PB两nativecall每层，margin校验精确复现FP32相减；原metric公式／1e-12数值核对／原q不归一化／完整候选与unknown规则／10000分层case bootstrap均不变。没有改预测或放宽容差。

## 三节点target mass（五域case等权，不是Public Index）

| 冻结节点 | Causal packed CE | Bidirectional packed CE | Candidate-prefix-block packed CE |
|---|---:|---:|---:|
| 10M，m365/9988659token | 52.0806% | 37.5653% | 41.1644% |
| 30M，m1122/29985465token | 55.6754% | 43.4677% | 41.0244% |
| 100M，m3557/99824365token | 59.5955% | 48.1857% | 43.2434% |

## 100M概率指标与完整onehot exact

| 臂 | NLL↓ | Brier sum↓ | 完整onehot exact，分母259 |
|---|---:|---:|---:|
| C_P_CE | 1.1938659447 | .5406976948 | 145/259 |
| BI_P_CE | 1.3311914062 | .6351701171 | 119/259 |
| PB_P_CE | 1.4399255032 | .6852792573 | 100/259 |

100M成对target-mass差：

| 差值 | 百分点 | 95%描述性case区间（百分点） |
|---|---:|---:|
| BI−causal | −11.4098 | [−16.4473,−6.2375] |
| Candidate-PB−causal | −16.3521 | [−21.5861,−11.1202] |
| Candidate-PB−BI | −4.9423 | [−9.9607,+.0462] |

单seed、whole-case配对、五域分层10000bootstrap，**未作多重比较调整，不覆盖训练seed不确定性**。PB−BI区间仍跨0，不能只看点估计宣称稳定胜负。

**本冻结协议下，两种非因果mask没有带来提分，三个节点均低于相同packed causal CE；100M NLL/Brier也更差。** 这是该stock＋LoRA/head、数据、曝光与优化协议的实际负结果，不是“所有双向模型无效”或完整Base/Jina方法失败。不能把旧causal权重临时开mask的机制结果替代本次重新训练对照，也不因结果差临时加LR／曝光或择public节点。

相对30M，BI/PB的mass提高，NLL/Brier却变差；仍须同时报告概率质量，不能挑单指标称单调进步。

## 系统计数与科学解释分开

每臂两pass训练real forward199648730／backward99824365，first/replay各16977wave；三dev节点合计4663452real token／1407wave，所有tokenwise路径为packed realT。BI共990108nativeFA3 forward提交，PB1980216；PB query总量与causal/BI相同，但KV提交总量更大、每层两次attention。backward仍只计实际完成autograd wave，不冒称逐C++kernel统计。

末次dev单次ledger墙时：causal41.7371s、BI42.8774s、PB50.9699s。不是稳定多重复吞吐或官方单卡latency准入；不用不同耗时产生的额外曝光混淆同曝光学习质量。

## 证据与尚未完成部分

- 三节点各5份primary/BI/PB SUMMARY/COMPLETE聚合共15份，加两完整退出view及原phase receipt，共18份。重复scope/含private cid的schedule明确省略，原全证据及private case/target/logits/token/tensor留远端。
- Causal控制的原节点view在`../clean-trainer-v2/`；本次从保存预测重新回算与既有结果一致，没有重跑控制模型。
- 此PB仍是`Candidate:\n`边界，不是新的state＋LF边界；新STATE/JPAIR仅正式macro0及独立primary通过，续跑控制层还在验收，不把本结果当其完整训练质量。
- 未开D_confirm/public、未选winner、未发布；D_select百分比不能与Public Index51.99直接比较。
