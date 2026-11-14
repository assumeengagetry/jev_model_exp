# JPAIR新目标：完整真实saved-macro机制

本次仅把新BCE＋hard-rank目标作用到已保留的原`C_P_BCE`正式macro0实际margin缓存：16case／19known field／60candidate。没有重跑旧控制模型、没有synthetic输入、没有新模型forward／optimizer／训练曝光。

- 18rank-active字段、1无rank字段；18硬正／40硬负／2soft候选；本组无K1，near-rounding和soft-only等分支未运行覆盖。
- 主代理独立从原host q先精确分类0/1，再重算BCE、rank及总coef，与新模块实际结果逐位相同。
- macro BCE1.5607420206069946，rank1.0335379838943481；按field先加再除F/C的总loss2.5942800045013428。
- `dL/dmargin` L2：BCE0.0989706453、rank0.1798223624、total0.2526782262；total/BCE2.5530623。这不是393参数component梯度，不能把后续clip变化仅归因排序方向。
- λ=τ=1，soft不作rank负例，inactive字段不删除分母。严格onehot K≥2时rank在代数上就是原field CE；本臂不是完整Jina复现。

`COMPLETE.view.json`明确省略source bytes，完整原产物留远端；3份实际源经远端逐字节回读核对后归档。**本结果不签新臂fresh身份、正式macro0/replay/393gradient/actualupdate、完整训练、dev质量或public成绩。**
