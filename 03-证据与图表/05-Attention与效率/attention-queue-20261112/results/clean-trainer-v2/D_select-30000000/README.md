# Fresh 四臂：冻结30M D_select 中间观察

**范围：** seed42、更新1122、17952次完整case曝光、29985465逻辑pair token。主代理2026-11-12T13:47:22UTC从已保存logits／margin＋原select独立回算完成，无新model／eval／optimizer／D_confirm／public工作。不是Public Index、不与51.99数值比较。

## 同曝光聚合

主指标仍为原冻结case内known field的`q[argmax margin]`均值、case/域均值、五域等权。

| 臂 | Target mass ↑ | NLL ↓ | Brier sum ↓ | 完整onehot exact（259子集） | 对10M的mass变化（百分点） |
|---|---:|---:|---:|---:|---:|
| C_D_BCE | 0.5398824406 | 1.1453914519 | 0.5648176083 | 132/259 | +6.4477 |
| C_P_BCE | 0.5313842060 | 1.1624516934 | 0.5816323764 | 136/259 | +3.4486 |
| C_D_CE | 0.5488940973 | 1.0326026563 | 0.5327917567 | 139/259 | +1.0128 |
| C_P_CE | 0.5567535822 | 1.0365784468 | 0.5288095160 | 141/259 | +3.5948 |

四臂375case／每域75、615known field／5190完整candidate、90unknown保留不forward；coverage1.0，所有原候选顺序、target／真实EOS合同不变。exact只在259个完整onehot case子集统计，不把其余case当错或删掉。

### 每域target mass

| 臂 | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|
| C_D_BCE | 0.5329844934 | 0.5639428488 | 0.5288888889 | 0.4802626384 | 0.5933333333 |
| C_P_BCE | 0.5093640663 | 0.4968036623 | 0.5955555556 | 0.4485310791 | 0.6066666667 |
| C_D_CE | 0.5656419985 | 0.5580493701 | 0.6311111111 | 0.4830013401 | 0.5066666667 |
| C_P_CE | 0.5278642185 | 0.5533033480 | 0.6266666667 | 0.5159336779 | 0.5600000000 |

不要只看总体：denseCE的Arts从10M的0.6066667降至0.5066667（−10百分点），packedCE的Arts降4.6667百分点；它们在Knowledge／Language／Retrieval上同时增长。仍沿原曝光/LR，不因某域中途较差调配方；Tools主要procedural，不冒称BFCL泛化。

## 成对case差异（百分点）

仍使用整case、四臂共同draw、五域分层10000次bootstrap，seed20261112。区间仅描述单训练seed的case样本不确定性，未校正多重比较，不代表跨seed置信或等价性检验。

| 对照（前者−后者） | Mass差 | 描述性95% case区间 |
|---|---:|---:|
| CE−BCE，dense | +0.9012 | [−2.9088, +4.6563] |
| CE−BCE，packed | +2.5369 | [−1.1705, +6.1541] |
| Packed−dense，BCE | −0.8498 | [−3.9642, +2.2624] |
| Packed−dense，CE | +0.7859 | [−1.4887, +3.0578] |
| CE−BCE，两布局等权 | +1.7191 | [−1.5341, +4.8883] |
| loss×layout交互 | +1.6358 | [−2.1934, +5.3424] |

- CE的NLL／Brier相对同曝光BCE仍低：dense分别−0.112789／−0.032026，packed−0.125873／−0.052823。target mass方向仍正，但本节点各case区间都跨0，10M的较大dense优势未稳定保持。
- Packing两种loss的方向都与10M相反：BCE由+2.1493变−0.8498，CE由−1.7960变+0.7859；不能宣称稳定padding学习收益。BCE有127field／108case argmax差异，CE有61field／59case差异，不能称长期逐题数值等价。
- 不按10M/30M中间最高分选checkpoint；100M、后续BI/PB、冻结确认／完整public仍待真实结果。

## 实际核验和系统计数

- 原case顺序／field coordinate／known mask／全部candidate及q与pinned select逐条一致，保存的二类logits对margin一致。独立stdlib log-sum-exp回算field/case/domain argmax、NLL、Brier，最大差1.89e-15以内。
- 节点前后checkpoint同cursor1122、冻结scope/seed/mode/layout/loss/fresh绑定一致，dev未修改历史训练计数；原COMPLETE+所有产物stat核对。
- 四臂累计训练real forward59970930（两pass），backward29985465；tokenwise输入行数计数器两pass累计dense80864400、packed59970930。不是额外曝光或逐kernel性能证据。
- 本次D_select都完整执行1554484real token／469wave；ledger单次wall denseBCE47.35、packedBCE41.81、denseCE46.32、packedCE42.04s。没有多重复单卡吞吐合同，不把一次速度或完成时刻当算法提分。
- 归档9份聚合view/primary report，view显式elide重复scope/checkpoint_identity并携原artifact身份。私有case-metrics／案例文本／target／logits／token／权重仍留远端，actual分析源码与10M同一封存版本。
- 30M事件观察器已完成退出，100M只读inotify/exact-pidfd观察器接入；原controller及8份训练source/config未改／未重启，不打开D_confirm或public。
