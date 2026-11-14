# 第一波100M：完整训练后的冻结D_select（不是Public Index）

## 实际完成与独立回算

四臂均完成3557optimizer calls／56912case occurrences／99824365逻辑pair token；末步LR0，3556positive-LR calls。四个原worker已退出，各保留43个checkpoint，原控制器已从同fresh启动GPU0的BI_P_CE和GPU1的PB_P_CE（15:34:08UTC一次准入回读确认）。不是整队列所有方法已结束。

每臂末次D_select完整375case（各域75）、615known field、5190candidate、90unknown不forward，coverage1.0。主代理于2026-11-12T15:32:16UTC使用同一封存CPU分析器，从原pinned select与实际保存logits/margins独立回算field/case/domain和节点前后同cursor checkpoint；最大指标误差1.7902346272e-15，0模型forward／optimizer／新eval，无confirm/public。

| 臂 | 五域case等权target mass↑ | NLL↓ | Brier sum↓ | 完整onehot exact（分母259） |
|---|---:|---:|---:|---:|
| C_D_BCE | 0.5642603233 | 1.2816408913 | 0.5703201440 | 140/259 |
| C_P_BCE | 0.5756943622 | 1.2835494087 | 0.5733636277 | 146/259 |
| C_D_CE | 0.6169353843 | 1.1710422157 | 0.5372061919 | 155/259 |
| C_P_CE | 0.5959548961 | 1.1938659447 | 0.5406976948 | 145/259 |

Target mass是原q[argmax margin]，不是所有字段0/1 accuracy，更不是官方public计分。259分母的完整onehot exact不与375case主指标混用。

## 同曝光比较与不确定性

整case配对、五域分层10000bootstrap；下表为描述性单seed case区间，**未作多重比较调整、不覆盖训练seed方差、不能证明数值等价**。

| target mass差值 | 百分点 | 95%描述性case区间（百分点） |
|---|---:|---:|
| CE−BCE，dense | +5.2675 | [+1.2548,+9.2795] |
| CE−BCE，packed | +2.0261 | [−1.9162,+5.9887] |
| packed−dense，BCE | +1.1434 | [−2.6739,+4.9033] |
| packed−dense，CE | −2.0980 | [−4.8263,+0.5704] |
| CE−BCE，两布局等权 | +3.6468 | [+0.4123,+6.8833] |

- 本节点CE的mass/NLL/Brier方向均优于相应BCE，但不能据单seed case区间签跨seed稳定提升。
- Packing在BCE/CE的mass方向仍相反；CE packed低于dense约2.10百分点，不能称packing学习提分。
- 与30M相比四臂mass都上升，**NLL和Brier却都变差**。例如denseCE NLL1.0326027→1.1710422、Brier0.5327918→0.5372062；packedCE NLL1.0365784→1.1938659、Brier0.5288095→0.5406977。不能只挑mass称所有质量单调改善，也不因此追选30M checkpoint或改LR。

## 域别mass

| 臂 | Arts | Knowledge | Language | Retrieval | Tools |
|---|---:|---:|---:|---:|---:|
| C_D_BCE | .5933333 | .5458108 | .6231869 | .5288889 | .5300817 |
| C_P_BCE | .6333333 | .5374603 | .6131076 | .5777778 | .5167928 |
| C_D_CE | .6066667 | .5535221 | .6222203 | .7244444 | .5778234 |
| C_P_CE | .5666667 | .5168555 | .6649081 | .6555556 | .5757886 |

Tools train/dev主要procedural，不冒称BFCL/真实工具泛化；不按域选择checkpoint或public路由。

## 系统计数，单独于学习质量

每臂实测两pass训练real forward199648730、backward99824365，first/replay各16977wave；三节点dev合计4663452real token／1407wave。每臂完整科学fullrow共990108次nativeFA3 forward提交；backward计完成autograd wave，不冒称逐C++ kernel计数。

Dense两pass tokenwise物理rows266503232；packed199648730，说明本轮tokenwise路径没有dense padding，但不是独立稳定吞吐或官方单卡latency证明。末次D_select训练ledger墙时依D_BCE/P_BCE/D_CE/P_CE为46.9566／41.6486／46.1319／41.7371秒，仅本轮节点实测，不含完整部署成本，不夸称多重复加速。

## 证据与后续

`PRIMARY_VERIFICATION_AND_PAIRED.json`及各臂SUMMARY/COMPLETE聚合view共9份；view明确省略重复scope，private case/logits/target/tensor留远端。四臂完整退出view另见`../first-wave-completion/`（原scope及含private cid的schedule被明确省略，全部原产物仍保留）。

未选winner、未打开D_confirm/public、不宣称超过51.99。原BI/PB继续；新STATE/JPAIR独立fresh393已实际生成后对账一致，但正式macro0/外部审核/续跑仍待接线。后续候选冻结与确认规则另行封存，不能用public选择checkpoint。
