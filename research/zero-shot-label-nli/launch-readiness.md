# Qwen3 分类基线：训练前核对

核查日期：2026-10-27。当前工作区无法访问用户所述 H800 服务器上的 `~/assumeengage/`；以下依据用户截图、公开数据与官方源码，尚未在该服务器实测。

## 结论

现有数据足够启动第一轮通用分类基线，无须先继续下载大批通用训练数据。环境截图验证的是微型 Qwen3 的生成式 LoRA/QLoRA 路径；正式开始需要接好序列分类的输入、损失、分类头保存和榜单评测适配器。

## 数据事实

- Hugging Face 数据集：`tasksource/zero-shot-label-nli`。
- 核查 revision：`9f0763b798363a6d4ca8f65de2dc80e0fcb48ae2`。
- `default` 配置字段为 `premise/hypothesis/labels/task/license/license_use`；不要误用 `triplet` 配置。
- train 1,899,993；validation 24,971；test 25,440；混合 294 个源任务。
- 它将分类标签转成 `This example is <label>.`，并非完整的原生 NLI 训练合集。构建脚本排除了标签已经包含 entailment 的任务。
- 元数据声明三个 ClassLabel 名称，但构建函数实际上只产生 `0 = entailment`、`2 = contradiction`。第二种在这里表示抽到错误分类标签，并不是另行标注的自然语言矛盾。
- 仅用该来源开二分类基线时，重新编码 `0 -> 0`、`2 -> 1`，设置 `num_labels=2`；保存对应 id2label/label2id。先核对远端 manifest 对应版本及实际标签分布；遇到其他值应明确处理，不能静默映射。
- 此数据的 train/validation/test 是它自己的划分，不能代替 Decision Index Language 评测集。源清单包含 isarcasm/intended、ragtruth/grounding 等，需对冻结的榜单评测记录做重叠检查，不能仅凭 split 名认定互不重叠。

来源：[数据卡](https://huggingface.co/datasets/tasksource/zero-shot-label-nli)、[数据转换函数](https://github.com/sileod/tasksource/blob/2796c32a346733e14b8c19b47e2fd9aade5aa03b/src/tasksource/recast.py)、[构建脚本](https://github.com/sileod/tasksource/blob/2796c32a346733e14b8c19b47e2fd9aade5aa03b/scripts/build_label_nli.py)。

## 分类训练接法

固定使用 Qwen3ForSequenceClassification 的 `score = Linear(hidden_size, 2, bias=False)`，无需再下载一个独立分类头。基座通常没有 score 权重，加载时新初始化是正常行为。它读最后一个非 padding token，不要求另加自定义 marker。

输入同时保留 premise 和 hypothesis，以一致、明确分隔的文本格式编码；labels 是一个序列级整数。使用分类交叉熵和分类 Trainer 路径，不能直接沿用生成式聊天 SFT 的 token-level loss。

Unsloth 当前官方实现支持传 `num_labels` 加载 AutoModelForSequenceClassification。采用通用 FastModel 路径时，可用 SEQ_CLS 类型并明确训练/保存 score。注意 FastLanguageModel 优化路径不接受任意显式 modules_to_save；不能把 FastModel 的参数组合直接照搬过去。

最低核查项：实际模型类为 Qwen3ForSequenceClassification；分类 logits 为 batch×2；label 映射正确；pad_token_id 与 tokenizer 一致；score 及 LoRA 参数有梯度、确实更新；checkpoint 保存后重载能复现同一批 eval logits。

来源：[Unsloth 模型加载器](https://github.com/unslothai/unsloth/blob/63939abd1453968496f08ca7a1a59cfa5b8f8a9b/unsloth/models/loader.py)、[Qwen3 文档](https://huggingface.co/docs/transformers/en/model_doc/qwen3)、[PEFT SEQ_CLS 实现](https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py)。

## 还需要下载或准备什么

1. 真正要训练的 Qwen3 小基座权重和 tokenizer：如果服务器只缓存了微型测试模型，仍需下载实际基座。
2. 官方 decision-index kit 与 Language 维度的冻结评测请求：9 项为 ContractNLI、ANLI、HellaSwag、ACOS、FinEntity、iSarcasmEval、VAST、NLI4CT、RAGTruth。训练池本身不含这套榜单评测请求。
3. 自定义 Engine 适配器或兼容 `/v1/systemone` 的服务。kit 的内置 transformers engine 加载的是 AutoModelForCausalLM，不能直接拿分类 checkpoint 替换。

序列分类器的适配方式可以固定为：将 state、问题和每个候选渲染成待判断的文本对，逐候选得到匹配分数，再按问题类型输出选项概率。候选可批量计算，但不是一次共享 hidden state 直接输出所有候选的 255 路方案；推理成本需单独测。

kit 没有 `--area language` 参数，不要编造该命令。应使用官方冻结 suite 的请求格式，按 `_evaluation.catalog_id` 选取 `{11,12,29,38,39,40,41,42,59}`，保留完整关联请求组，再通过 `--rows` 驱动自定义 Engine，用官方 scorer 查看这 9 项的 Language 分数。局部开发评测不是完整上榜提交；正式提交要求完整 public suite，Full 私测由维护者运行。

来源：[官方 kit](https://github.com/apolinario/decision-index)、[Engine 接口](https://github.com/apolinario/decision-index/blob/main/docs/engines.md)。

## 建议执行顺序

先跑同一模型的训练前评测；再用分层小样本训练几十到几百步，验证分类头更新、保存和重载；随后完成一次 Language 端到端评测，再扩大训练量。无需等 9 项训练数据全部补齐才得到第一条基线。

该池缺少多个目标任务的直接训练覆盖，第一轮分数低不能直接归因为小模型结构错误。之后根据任务分项补数据；ANLI、HellaSwag 等官方 train 的来源已记录在 `/Users/apple/Workspace/jev/research/language-axis/README.md`。这些文件此前下载在本机工作区，不代表已传到 H800 服务器。
