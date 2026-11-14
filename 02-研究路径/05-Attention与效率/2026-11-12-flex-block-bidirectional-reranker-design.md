# FlexAttention／block-bidirectional／reranker 方法扩展

**授权与状态：** 用户于2026-11-12要求“这几个方法你也都尝试一下”，作为正在进行的padding-free与0.6B提分研究的增量，不替换原任务。当前仅已获取官方Flex资料并启动另外两项的只读原始资料核对；**没有这些方法的训练／public新分数**。

## 链接可达性（本次实际读取）

- 知乎 `https://zhuanlan.zhihu.com/p/1897675709696149020`：HTTP403，已告知用户请求正文／关键截图；不能猜文章内容。
- PyTorch博客 `https://pytorch.org/blog/flexattention/`：第一次传输错误，第二次成功读取。
- 版本匹配文档 `https://docs.pytorch.org/docs/2.9/nn.attention.flex_attention.html` 及 `https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/nn/attention/flex_attention.py`：成功读取；Context7月额度耗尽，回到官方源。
- Base Labs mammoth-1／elephant-1的GitHub条目用户仅给标题，精确链接定位中。
- Jina `https://huggingface.co/jinaai/jina-reranker-v3`：官方模型卡／代码／论文核对中，不预设其参数规模或许可。

## 分开控制变量

1. **Flex causal backend：** 在同权重、同真实token、同cu／position／EOS和causal语义下对FA3与Flex做forward/backward／真实一步更新对照。这首先是后端与mask表达实验，不是新学习能力；可以与全层packed结合。
2. **block-bidirectional训练：** 精确mask以Base Labs原文／实际实现为准，不能把“同块双向”误写成全case全双向。改变可见性需要训练适配和train/serve一致，不能只切旧causal权重开关就把结果当方法复现。固定stock base、成对fresh LoRA/head、clean完整case、loss／曝光／scheduler，仅改mask。
3. **Jina可迁移方法：** 先区分网络结构、列表交互／晚交互、读出和loss，不能凭reranker名字就当完整field CE。迁移到当前0.6B骨干，模型大小另记；不直接把更大模型作为最终服务或推理代答。多候选交互若采用，必须验证variable K和候选排列，不跨无关case。
4. **知乎方法：** 收到可读正文后再界定，未获取前不编造方法，也不凭标题宣称已尝试。

## Flex 2.9.1已核实实现要求

- `flex_attention(query,key,value,...,block_mask=...,enable_gqa=True)`接收4D；全层packed hidden仍可`[T,H]`，仅attention将真实QKV view/transpose为`[1,Hq/TKV,T,D]`，singleton batch不是业务padding。
- document/sample mask必须同时限制同序列与序列内mask；不能仅对flatten后token做普通causal。
- 博客中的“长度必须128倍／未支持GQA”等为旧限制，不能当成本机2.9.1现状；版本文档已经提供GQA及nested/jagged接口，实际非128长度路径仍需真实样本验证。
- 高性能路径需要`torch.compile(flex_attention)`；不编译的2.9.1路径会使用物化score的unfused执行。新增生产内核编译属于用户本轮明确要求的方法实现，**不是额外compile benchmark或smoke**；首次编译、重编译次数／时间必须实际记账，不从耗时中隐去。
- `create_block_mask`默认可能物化`[1,1,T,T]`布尔mask；不能把它作为长packed训练的无界实现。采用版本匹配的编译构造或精确的block-metadata构造，并在一次batch构造后跨28层复用；缓存必须有界且绑定真实长度／mask结构。
- 没有需要业务pad的理由；底层tile mask和BlockMask索引容量不送入Embedding／LoRA／MLP。动态shape和编译缓存问题单独测量。
- 若prefix会读suffix，则不同候选改变prefix表示，不能继续使用原immutable共享prefixKV；只有精确mask保证prefix独立才保留复用。该取舍不能靠错误缓存掩盖。

## 顺序与验收

- 先完成原padding-free的真实机制门禁和同权重完整public；这些方法不混入该A/B。
- Flex先复用真实原始macro／完整case验收，不新增synthetic/unit/smoke大网格、不模型／语料hash；保留严格确定性设置，若后端不支持则报告，不偷偷降级。
- mask／reranker质量臂先用合法clean数据和已冻结未见dev，达到预定节点后完整public；每次140620执行／140178计分，不截断候选、不混旧预测。
- 相同曝光与相同GPU时间分开；失败臂、checkpoint、optimizer、partial全部保留，不发布／push，不增加付费资源。

后续用原始资料核对结果补充精确结构和可执行任务；本文件不是实验完成回执。

## 已完成的一手资料核对与后续实验轴

### Base Labs

- 精确PR：[vLLM #63114](https://github.com/vllm-project/vllm/pull/63114)，2026-10-22创建，本次读取仍Open，不能当已发行能力。
- [官方技术报告](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)明确同数据9B单次消融causal44.8／block-bidirectional48.6；主代理已实际读取原表并核对“single run”。这不是本项目0.6B结果。
- 两块mask（row=query、column=key）：`[[P→P允许,P→S禁止],[S→P允许,S→S允许]]`。同块内部双向，不是整段全双向；不同问题suffix之间仍隔离。
- 当前commit `prefix_block`在`Candidate:\n`处分界，prefix含问题；Base按system＋state为prefix、问题＋全options＋answer-slot为suffix。两者必须标记为不同边界／输入结构，不能冒充完整复现。
- prefix只有在所有层不读suffix时可只读复用；双向prefix须完整相同，不能沿用causal的任意最长公共前缀cache命中。训练不能detach可训练LoRA的prefix图。
- 后续mask质量控制为causal／prefix-LM（P双向S因果）／两块双向／全双向；第一轮优先用户最新commit的三个模式，Base精确边界另作匹配控制，不同时改loss。

### Jina reranker v3

- 固定revision：`d7d7e73b6ea138ced340b83865931b5dfb6c97aa`，[模型卡](https://huggingface.co/jinaai/jina-reranker-v3/blob/d7d7e73b6ea138ced340b83865931b5dfb6c97aa/README.md)、[实际modeling.py](https://huggingface.co/jinaai/jina-reranker-v3/blob/d7d7e73b6ea138ced340b83865931b5dfb6c97aa/modeling.py)、[论文v4](https://arxiv.org/html/2509.25085v4)。主代理已直接读卡／代码核对以下关键项。
- 本身是Qwen/Qwen3-0.6B系列，非大模型前提；公开权重CC-BY-NC-4.0。迁移思想不等于使用其权重／代码获得任意商业授权；本轮不下载它充当答题模型。
- **保持causal**，query在开头和末尾各一次，多个doc顺序同窗，后doc可读前doc，末端query可读全部doc；不是candidate对称双向，也不是独立pair加softmax就等价。
- 文档末标记和query末标记hidden经共享projector、cosine打分，不是当前EOS二类head。卡写1024→512→256，但固定源码实际为1024→512→512；差异必须保留，未读权重张量前不声称checkpoint形状已核验。
- 源码强制`use_cache=False`；没有提供本项目式immutable prefix-fork的通过证据。131K长上下文与新的special-token语义也不能无训练继承。
- 迁移先拆为：J0原候选隔离＋原loss；J1同请求候选causal交互-only（EOShead和loss不变，增加位置／模板匹配的隔离控制）；J2请求内ranking-loss-only（attention不变，unknown不作负例、多正例与K=1规则预先固定）；随后才考虑J3组合。以上是Jina-inspired，不冒充完整Jina训练配方复现。
- 候选顺序、variable K和pool变化属于建模鲁棒性而非缓存小节；必须以原case为统计单位，跨case绝不互看，标签不进入输入／mask。

## 实际执行记录（2026-11-12）

- 用户随后要求将padding、链接方法与最新commit排队自主运行，充分利用计算资源。
- Flex causal真实原macro已在H800执行：112层次forward、112个实际Q-backward hook、1个unique compiled graph，包含首生产调用编译总计9.7665秒。全部原始grad／update／optimizer与编译缓存保留。
- 该次**未通过冻结数值合同**：logits相对L2约.0103544、梯度最差组相对L2约.254330、更新约.537647；不放宽合同、不称提分。严格确定性下实际能执行不等于与FA3模型训练轨迹等价。
- 最新commit三个mask真实FA3均已执行相应112/112/224 forward及deterministic backward，FA3对SDPA整模型数值合同均未通过，连causal基线亦有漂移，需分开诊断后端数值与mask语义。原始失败保留。
- padding-free修正后的原HF／dense／packed真实macro及原kernel backward已通过原容差；完整public两臂进入独立四卡队列，**尚无最终新分数**。详情见`2026-11-12-commit-2e334fb-experiment.md`。
