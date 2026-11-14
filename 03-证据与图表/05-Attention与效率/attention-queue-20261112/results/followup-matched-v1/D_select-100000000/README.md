# 后续探索完整100M：终点同曝光比较

STATE与JPAIR各已完成3557 optimizer calls／56912case occurrences／99824365logical pair tokens及三个冻结开发集节点，最后LR0、3556positive-LR calls。两resume worker实际exit0并wait，controller完成退出；各43checkpoint全部保留。macro0只消费一次，未重启旧failed队列或改模型scope。

同一封存quality/scaling源码于2026-11-12T20:27:17UTC实际独立回算全部五臂，完整375case／615knownfield／5190candidate／90unknown，coverage1、onehot exact真实分母259。source/config/data/native、独立生成的393初态byte桥接、MAIN实际macro0/更新身份及目标前后checkpoint/曝光前缀均核对；最大metric误差1.7902346272e-15。0新模型forward/optimizer/eval/confirm/public。

| 全部packed终端臂 | target mass↑ | NLL↓ | Brier sum↓ |
|---|---:|---:|---:|
| Causal CE | 59.5955% | 1.1938659 | .5406977 |
| Causal BCE | 57.5694% | 1.2835494 | .5733636 |
| Candidate-PB CE | 43.2434% | 1.4399255 | .6852793 |
| State＋已有LF PB CE | 44.3620% | 1.3294748 | .6411995 |
| JPAIR BCE＋hard-rank | 61.2437% | 1.1480786 | .5336582 |

## 成对case差异

| 差值 | 百分点 | 95%描述性case区间（百分点） |
|---|---:|---:|
| STATE−Candidate-PB | +1.1186 | [−3.7838,+5.9899] |
| STATE−causal CE | −15.2335 | [−20.4103,−10.0615] |
| JPAIR−BCE | +3.6743 | [+.0821,+7.3262] |
| JPAIR−CE | +1.6482 | [−1.9076,+5.1159] |

10000整case配对、五域各75分层，单训练seed、未作多重比较调整；不覆盖训练seed方差。JPAIR相对BCE三质量指标方向较好，相对CE的终点点估计也较好，但case区间仍跨0，不能宣称稳定／已确认提升。STATE相对Candidate-PB的mass差仍不确定，虽NLL/Brier较好，仍明显落后causal CE。两方法是在原10M/30M后提出的探索，不是原四臂预注册比较，更非完整Jina/Base复现。

## 完整优化轨迹诊断

前3557实际macro，五臂记录scalar clip_factor<1比例全部100%；实际used_lr>0调用均3556，不能称3557次非零学习。clip前参数梯度L2均值CE15.2990／BCE8.8064／PB19.8323／STATE15.7798／JPAIR21.4442。

JPAIR的79599known field occurrences中72430active、7169zero（active90.9936%）；hardpositive72430／hardnegative303260／soft17293；5794case occurrences无rank信号。zero无正7169、无负5117、soft-only5117，理由重叠不可相加当分母；K1实际前缀计数0。margin total/BCE L2比均值3.3008，不是393参数梯度分量比，不能将方向／尺度／clip影响分离成单因果；这些训练loss/梯度统计不作质量指标。

## 候选与public边界

独立terminal dense→packed迁移两臂均34case违约，虽然denseCE的packed点分数61.6935%，仍按冻结数值门禁eligible=false，不因分数更高放宽合同。

已冻结有限terminal规则随后仅用实际eligible packed结果封存baseline=C_P_CE、challenger=JPAIR。此处只描述D_select选择，**不是D_confirm已确认、生产consumer已准入或新Public Index**。新consumer source/codec/loader/一次250case确认及每模型完整140620/140178public仍须实际执行；不能据public再换checkpoint/runtime/域路由。

本节点6份primary/scaling/newSUMMARY/COMPLETE聚合，另2完整退出view＋controller完成view共9份；源码与前两节点一致。view明确省略scope/含private cid的schedule；所有private case/q/logits/token/tensor与原失败完整保留在远端。
