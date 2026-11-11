# Tools定向修复与最小推理适配研究

日期2026-11-10。用户优先要求研究Tools回退，再找Transformers或其他框架的小改动路径。本轮只读现成训练数组、已保存预测/官方gold及版本源码；**没有新增模型forward、tokenizer调用、校准拟合/阈值扫描、训练、测试、smoke、hash、安装或发布**。原24.97权重、Engine和结果不改；readout独立实验不动。

## 1. BFCL具体是漏选，而不是过选

同一1694个case/4768个yes-no字段上的现成预测，与受控E31参考逐ID/field对应。只用原固定决策，不重新拟合：

| 指标 | 受控E31 | 当前24.97 |
|---|---:|---:|
| 完整case准确率 | 68.77% | 32.94% |
| 字段准确率 | 88.07% | 75.86% |
| 需要调用的召回率 | 69.60% | 33.00% |
| 预测调用的精确率 | 95.62% | 97.22% |
| TP / FN | 1179 / 515 | 559 / 1135 |
| FP / TN | 54 / 3020 | 16 / 3058 |
| 全部工具都判No的case | 485 | 1120 |
| 固定输出ROC-AUC | 0.95370 | 0.91042 |
| Brier | 0.09147 | 0.19763 |

GT每case均有一个正工具；这只是当前套件子集属性，**不能硬编码任意Tools请求至少选一个**。旧正确而新错误622case，新正确而旧错误15case。当前1135个FN中有684个P(yes)≤0.1，并非都在0.5附近。

旧/新需要调用字段的Pyes中位数0.7868→0.2071；负字段0.00869→0.00282。配对log-odds变化：正字段均值−2.31、负字段−1.14，不是统一常数。AUC是正单调校准不改变的量，它也下降，说明不能声称只改一个bias即可完全恢复旧输出。校准可能修一部分召回，必须由独立train-dev验证，不在public上选阈值。

When2Call总分69.25%→69.00%看似稳定，但现成分轨道准确率：cannot_answer69.88%→76.68%，tool_call61.24%→55.83%，request_for_info78.25%→75.71%。这与更保守的调用判断相符，仍是描述现象，不证明某一训练阶段造成。

机器依据：`../joint-generic-continuation-20261109/TOOLS_ERROR_AUDIT_20261110.json`。旧/新训练、精度加载与wrapper不同，故不是同条件因果消融。

## 2. 当前Tools数据不等于完整BFCL式工具选择监督

实际250000pair/109292378token，原正85191/负159022/软5787。selected来源：

| 来源 | selected pair |
|---|---:|
| generated/workflow_policy | 63014 |
| typed procedural五类合计 | 82036 |
| native/glaive | 52463 |
| overnight/xlam | 36092 |
| native/when2call | 16395 |

约58.02%pair是生成workflow/程序化决策，42%是上述原生调用来源。没有BFCL命名来源；这不证明所有上游数据都无BFCL相关痕迹，也不否定程序化数据本身价值。

以source/group/prefix为单位，相对**现有去重global pool**，selected中的partial单位：Glaive27756/49880，xLAM27326/27763，When2Call10503/13730。这只证明当前pair采样没有完整保留许多global组；global去重池本身也不等于原始完整case，不能反拼后冒充原生完整目录。

源码确认训练候选多是整个工具文档；正式BFCL则把目标工具写进问题，候选为`Invoke this tool.`/`Do not invoke this tool.`。native next-action与完整请求required-tools也不是同一时间范围。这是明确的监督/呈现差别，长期存在，不能独自解释本轮回退。

## 3. 补Tools的优先候选

**先修工具适用性正例召回和字段读出对齐，不先扩一轮随机Tools pair或扫旧校准系数。**

1. 从固定24.97参数续接，不能再次以共同未训练head重训来声称保参修复。
2. 从已有xLAM/Glaive/When2Call的**原始train**恢复完整请求、历史和目录，按case分组保留，不从残缺pair重建。
3. 标签分positive/verified-negative/unknown。原轨迹未调用≠逐个证明不适用；unknown不可统一变成ranking负例。无functioncall marker也不能一概视为no-tool，需复核来源解析。
4. 按训练来源构造与实际读出一致的工具字段：一个目标工具，两个完整响应候选Invoke/Do-not-invoke；多工具保留多个正字段，不做工具间单正softmax。
5. 对已确认字段，用`d=margin(Invoke)-margin(Not)`的BCE（等价两候选CE），并按case归一；其它原任务回放/必要的函数保留锚点独立计账，预先约定非劣保护。完整case与标签mask的干预分开，避免同时改多项后无法归因。
6. 首个有界修复候选应选完整可见输入，不能让head/tail裁剪移除目标工具或必要条件；超4096 case要记录不足，不能静默改变上下文预算。

这是**待批准的研究配方**，未启动、未设定新增token预算、没有分数保证。只训共享head也会影响Language；保留其它域数据也不自动保留函数。若需冻结parent输出作retention，需要明确新增forward预算，不能假称已有缓存。

仅给所有候选margin统一加截距会在softmax中抵消；正温度缩放不改choice。旧−1.2826是已烘焙Teacher训练校准，不能重复施加或当成当前字段阈值。若独立dev支持差值截距校准，那是一个单独读出候选，不能按BFCL名字/测试gold特例处理。

## 4. 不能把BFCL公开测试叫作训练集

固定kit只用BFCL simple/live_simple/live_multiple，工具名集合选择、排除空gold set，不做参数/顺序/执行评分。所查Gorilla固定提交明确是evaluation/test categories，未见独立BFCL train/dev；HF文件叫train或许可Apache2.0也不等于可用其benchmark题训练。

现有可研究来源：xLAM60k train（CC-BY4.0）、Glaivev2 train（card Apache2.0）、When2Call train（CC-BY4.0，合成自动标注）。When2Call官方说明test来自BFCL、train来自xLAM/APIGen，必须只用train；公开生成源码有不一致警报，但本轮没证明发布/selected数据实际受影响，不能直接丢旧数据或放行pref ranking。

## 5. 当前最小可用框架路径：不换框架

**Transformers4.57.6 + PEFT0.18.1的现有完整SEQ_CLS模型可接受并返回KV cache。**

- `GenericForSequenceClassification.forward`传递past/use_cache，继续执行原score，并返回`SequenceClassifierOutputWithPast.past_key_values`。
- PEFT普通LoRA SEQ_CLS透传kwargs并直接返回base输出；不是prompt-learning分支，不丢cache。
- 可只在Engine保留field/prefix边界，新增可选cached scorer：prefix通过完整self.model prefill一次，各批candidate+EOS用独立workingcache继续同一self.model。原full scorer仍作fallback。
- 不必换CausalLM、不必拆score、不必mergeLoRA、不改依赖/训练代码。

必须保留：prefix/candidate分别分词后的实际IDs；pad151643/EOS151645；rightpadding；past+new长度的2Dmask；prefixoffset绝对positions/cache_position；完整P+suffix的context校验；原bf16autocast覆盖完整scoreforward；两列logits各转float再减、字段softmax。

**FP32头权重存储≠FP32头投影。**当前scoreforward在bf16autocast内，退出autocast手写FP32投影会改变数值路径。优先继续完整PEFT分类器，可避免误取`score.original_module`丢掉保存头。

DynamicCache没有专用fork/clone；update/crop/batch_repeat_interleave会修改cache，repeat还会物理复制KV。每字段保留干净template，每microbatch独立workingcache，设prefixKV和workspace字节预算；第一版继续原B≤32/完整paddedtoken≤32768保守限制。OOM/异常丢弃工作cache，释放后走原全量scorer，不删候选/截断/复用半更新状态。短prefix不一定更快，数值和速度收益都未验证。

## 6. 为什么其它包装不比这条更小

| 框架/包装 | 判断 |
|---|---|
| Transformers pipeline | 重新分词、默认softmax，且classification路径设置use_cacheFalse；不是自动KV分支复用 |
| SentenceTransformers CrossEncoder5.1.2 | 底层仍是HF；predict默认truncationTrue、无prefixbranch；rank要求1label。定制后不更简单 |
| vLLM0.25.1 pooling | 有Qwen3分类/LoRA能力，但loader明确不支持非空modules_to_save；我们必须保留保存score，不能删配置丢头。默认FP32head计算也不同于现有autocast。还需Torch2.11/Transformers≥5.5.3新环境与部署产物 |
| SGLang分类路径 | 有Qwen3分类实现，但所查版本score默认bias=True、savedhead/LoRA分发及精度不同；较新版本依赖更大改动，不是最省事首选 |

## 主要版本来源

- [kit BFCL adapter](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/decision_index/suite/build/adapters_mechanical.py#L126-L162)
- [固定BFCL数据README](https://github.com/gorilla-llm/gorilla/blob/916260dfc116bf06793a1af79b4ec8195b0453b6/berkeley-function-call-leaderboard/data/README.md)
- [When2Call来源说明](https://github.com/NVIDIA/When2Call/blob/ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc/README.md)
- [HF4.57.6 generic classification](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_layers.py)、[cache_utils](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/cache_utils.py)、[cache mask说明](https://github.com/huggingface/transformers/blob/v4.57.6/docs/source/en/cache_explanation.md)
- [PEFT0.18.1 SEQ_CLS forward](https://github.com/huggingface/peft/blob/v0.18.1/src/peft/peft_model.py)
- [vLLM0.25.1 modules_to_save拒绝](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/lora/peft_helper.py)、[classifier head计算](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/pooler/seqwise/heads.py)
- [CrossEncoder5.1.2](https://github.com/huggingface/sentence-transformers/blob/v5.1.2/sentence_transformers/cross_encoder/CrossEncoder.py)

代码/API链经独立researcher与controller一手源读取核实；不是运行验证或数值parity声明。所有新路径作为独立候选，不覆盖24.97冻结结果；任何验收/Tools续训/新公开评测都需明确有界范围，不默认增加整榜或长训练。
