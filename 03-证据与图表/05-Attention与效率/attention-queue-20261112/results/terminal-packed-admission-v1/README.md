# Terminal dense权重的packed consumer：完整执行，数值准入负结果

本次只把原C_D_BCE/C_D_CE在m3557的**固定terminal权重**各载入一次packed fullrow consumer，对原冻结375-case select各执行一次。没有新增训练/seed/backward/optimizer、没有重跑dense参考、没有warmup/probe或打开confirm/public。

## 实际执行与身份

- 19:55:03UTC显式manager启动，仅GPU0/1；GPU2/3训练与controller不变。
- 每consumer完整375case／615knownfield／5190candidate／90unknown不forward，1554484real token、469wave、13132次nativeFA3 forward；全部tokenwise为realT、head在所有真实token后才gather EOS。
- 原393FP32 tensor在GPU move后、eval前后与immutable saved oracle逐key/shape/dtype/byte一致；requires_grad保留393但所有.grad None，eval后没有reload掩盖变更。active served616237056，unused原head2048不算served。
- 两worker实际exit0并wait后managercomplete，leases随后释放；数值negative是完整科学结果，不是允许重试的编程失败。所有原checkpoints和本次raw都保留。

## 原冻结逐case合同没有通过

对每case全部known logits及margins分别检查RMS≤1%×max(refRMS,1e-3)、maxabs≤.02+.02×refMax；field flip的reference band仍为.04+.04×maxabs(field margin)，没有改阈值或batch。

| 终端producer → packed | 违反四谓词的case数 /375 | choice flips | 带外flips | eligible |
|---|---:|---:|---:|---|
| C_D_BCE | 34 | 1 | 0 | **false** |
| C_D_CE | 34 | 0 | 0 | **false** |

主代理另外从原已保存dense raw与本次packed raw独立重数，bad/flips/outside计数完全一致；未做任何额外模型forward或tensor审计。不能因为CE选择未翻转，就跳过raw numeric合同或声称预测/数值等价。

- BCE logits最大abs差.328125，margin最大abs差.65625，最大field概率差.1104917。
- CE logits最大abs差.296875，margin最大abs差.56640625，最大field概率差.0762880。

## 固定权重质量，单独于学习收益

| producer | 原dense mass | 实际packed mass | 实际packed NLL | 实际packed Brier |
|---|---:|---:|---:|---:|
| C_D_BCE | 56.4260% | 56.3371% | 1.2799205 | .5699076 |
| C_D_CE | 61.6935% | 61.6935% | 1.1661483 | .5350556 |

原q不重归一化、field/case/domain独立回算1e-12核对通过；但两个consumer仍因数值准入失败**不进入后续packed生产候选池**。NLL/Brier的小变化或61.69%的点分数不能改写失败，也不是新学习提分，更不是Public Index。

共14份install/launch/manager/weights/numeric/quality/独立重数聚合及两actual源码已回读归档。`NUMERICAL.view.json`明确省略private per-case记录，`scope`/per-key shape metadata也在标注view中省略；完整原证据和private case/logits/token/weights均留远端。未选择winner，未发布或修改旧预测。
