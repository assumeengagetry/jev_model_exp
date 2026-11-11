# 通用原标签＋五维原蒸馏目标联合续训设计

## 当前授权与目标

用户明确授权：当前五维蒸馏继续训完；从其final保留权重，接通用原标签＋五维原蒸馏目标联合续训；只对续训后的固定版本运行已授权的那一次完整官方public0.3。用户未授权额外测试／烟测／模型或语料hash、上传／发布／commit／push／付费HF Jobs。

当前来源根目录 `/tmp/opencode/jev-five-dimension-distilled-20261109-v2`。13:33UTC观察主Student327929在5780/9769步、218964639token，仍真实运行。13:34UTC只核实并SIGTERM本任务尚未启动评测child的旧official等待器324276，已观测退出；训练coordinator323791/Student327929未动。旧official LAUNCH保留，`OFFICIAL_DEFERRED_TO_JOINT_CONTINUATION.json`和明确deferred状态记录了用户指令，未伪造完成。

## 训练定义（实施者固定的有界选择）

- 新根目录 `/tmp/opencode/jev-joint-generic-continuation-20261109`；本地代码 `/tmp/opencode/joint-generic-continuation-20261109`。不修改仍运行的原trainer/config/pool；保留旧模型、优化器、partial与全部结果。
- 唯一Student父权重为本次当前五维KD完整final，不能用共同未训练初始化或旧原始标签stage2替代；来源完成、精确pid生命、final文件清单和配置在实际交接时核实并快照。
- 五维当前1,250,308pair与366,985,783有效token完整回放一遍，K/R/Tools/Arts/Language顺序、数据、原始标签及已烘焙的有效KD目标逐索引保持。Tools校准不重复施加，language0.5／K0.5／R0.25／Tools0.5／Arts0保持。
- 新generic域仅原始canonical zero-shot-label NLI train，labels0->match1、1/2->match0；不增加Teacher或LM辅助loss，不把它当榜单新增得分维度。
- generic目标有效token上限为task/4，即91,746,445，按完整样本共同前缀截止；总新增不超过458,732,228token（约4.59亿），实际以生产DATA_READY为准。五任务绝对曝光不因通用混入而减少；这是下一段新训练，不改当前段预算。
- generic样本seed42，不放回抽原始记录；核对并排除与task实际模型输入相同的样本及已观察重复/冲突，保留原标签，不用dedup均值target替代原标。不声称完成了全库/私有集去污染。
- generic用相同pair renderer、prefix/suffix分开分词+EOS、与旧训练相同4096训练组装规则；本段依旧二分类margin BCE。原始样本与新token只留私有scratch，不进公开报告。
- 保留既有3647条K/R原子关系及相对target、gamma0.5/lambda0.05；generic没有新排序关系。新全池原子计划按实际N、尾批、步数归一，不混用旧Sampler身份。
- 保留rank32/alpha64与显式strict FA3、Torch/cuBLAS控制；单GPU0、global batch128、micro32/16384tokens。新阶段重建AdamW／调度（不重置参数），LR2e-5、weight_decay0.01、warmup0.03+cosine、seed42、一遍本段冻结清单。所有可训参数FP32、主体BF16，禁止SDPA/FA2静默回退。

## 原始generic数据与实现接口

数据源 `/home/dev/assumeengage/datasets/zero-shot-label-nli-http/data/train-*.parquet`；列premise/hypothesis/labels/task，读取原始train，沿项目NO_TRAIN和既有generic保护排除规则，不借官方评测行补数据。

可只读复用共享TokenStore/assemble/collate，但不要改共享DOMAINS或缓存。新runtime的stream顺序为K/R/Tools/Arts/Language/generic，模型输入不包含stream标签。旧task低位state/candidate ID继续用全局TokenStore；新generic采用独立prefix/suffix流及offset，ID>=2**31映到本段流；保存前检查uint32边界和真实组装长度。generic可预先按同一规则clip prefix/suffix，必须证明重复组装不再改变长度，不在正式评测使用训练clip。

生产准备输出新pool六域数组、独立token streams、原标签/effective targets、group plan、config和DATA_READY/目标READY。旧task数据可只读链接但元数据/身份须绑定；generic只读原始标签；所有已有不一致产物保留并阻塞，不擅自覆盖。

## 后续评测与调度

- 先CPU准备，可用nice19/idle、2核亲和性，不抢正在训练的GPU0。
- 只有来源当前五维训练真正完成、final导出完且原Student退出/GPU0空闲后，才从该fixed final开本段。只启动一个候选，不追加消融网格。
- 联合续训完成后冻结自己的final、实际基座与portable Engine，只对该版本用固定upstream commit25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d的完整官方public0.3 pipeline运行一次；复用已严格验证的suite只读副本，不跑旧wrapper作额外评测。
- 不得用旧final的分数、snapshot、marker充作新final结果。所有resume、model/code/settings/hardware来源绑定与完成幂等规则保持；普通OOM/Unsupported语义沿正式Engine，不截断请求/过滤候选。
- 低优先级readout允许保存让路；不换其研究起点，不改其已冻结common/data来放宽门槛。主任务流程改变后，原准入可能自然等待，这是预期而非研究完成。
- 新监视器跟踪来源五维段→本段→正式公共结果；旧暂停原始标签扩大训练永不恢复。状态区分训练完成与官方完成，非完整不写complete=true。

## 发布边界

模型名称、权重/结果可见性、维护者访问、上传和PR仍需用户最终确认。本段实际训练、唯一正式公共运行和本地提交包整理已获授权，无须再次询问原始还是已训。变更研究目标、明显扩大预算或不可逆/发布动作才询问。
