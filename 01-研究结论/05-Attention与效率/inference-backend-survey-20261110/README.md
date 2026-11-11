# Decision Index参赛者推理后端调研

> **本文为当时只读后端调研。后续同joint final的原生HF共享前缀＋Graph实现已完成一次完整public，实际结果24.99及数值漂移见[最终报告](../inference-prefix-graphs-20261110/README.md)。本文其它后端的“未实施／未测收益”边界仍保留，不视为这些框架已通过我方验收。**

日期2026-11-10。只读查官方kit、榜方methodology、作者固定版本源码和许可证；未下载权重、安装后端、改变模型/Engine、运行测试/compile/smoke/forward/hash、重跑public或发布。两名独立quality-researcher分查原生分类与加速LM路径；controller另读主源验证关键实现。

## 结论

**能借鉴实现，但不能把别人的读出当作我们二分类模型的等价后端。** 大量参赛者仍是原生PyTorch/Transformers，自行做prefix复用、有界graphs、批处理和LoRA执行优化；不是统一vLLM。HTTP只是接口，Triton kernel DSL也不是Triton Inference Server。

当前我方是Qwen3ForSequenceClassification、rank32/alpha64、FP32保存的二分类score，独立prefix/candidate分词后拼EOS，margin后按field全候选softmax。原始24.97结果/模型/设置必须保留，未经数值验证不能将新runtime结果冒认为这份结果。

## 已核实代表

| 参赛者 | 可核实实际路径 | 值得借鉴 | 不能直接搬的部分 |
|---|---|---|---|
| Decider2B/4B | 自建PyTorch/Transformers；4B 1.2.1，有限shape CUDAgraphs；部分独立测速FP8+compile | 启动预捕获/封存graph池、长尾eager、受预算限制prefixfork | LM字母行读出、schema-first、Qwen3.5 recurrent kernels、FP8非无损 |
| reflex27B | 作者前端+SGLang0.5.20，radix cache；RTX测量用Triton attention | tokenIDs传输、共享prefix、分支隔离 | 其API取LM标签logprobs、两种顺序平均，不是我们的binaryscore |
| Winnow12B Q8 | llama.cpp/GGML CUDA，自定义server；合并LoRA后Q8_0GGUF | prefix一次prefill、容量感知分支waves、cache生命周期 | Gemma4专用selected LM head、量化、该patch不能直接加载Qwen3 classifier |
| Kev | PyTorch+合并BF16backbone+FP32pointer、SDPA/nativeprefixcache；作者另有有界graphs | cache条目/token双预算、单执行线程、graphfallback | Qwen3.5 DeltaNet/FLA、pointerhead、temperature均非我们的等价读出 |
| Decision2.0 Kai0.6B | PyTorch+HFQwen3Model+FP32bilinear/MLPhead；不是vLLM serving | 张量核心/调度分离、受限shape执行、轻量unmergedLoRA思路 | 原fast.py受ROCm+Transformers5.17 gate，fused限gfx942；当前CUDA/4.57.6不能直接启用 |
| GLiNER2.5-Decide | PyTorchDeBERTa jointencoder+labelmarker分类MLP，lab记录eager | 一次组织labels、合并回传、Python与编译core分离 | 双向joint编码不是可缓存独立causalprefix；换模型/头需要新训练 |
| Metask-Jev4B | lab记录原生PyTorchserve.py+LM字母logits | 有界shape/staticbuffers可借思路 | repo另有graph/KV服务不等于lab用了；hybridKV实现存在静态风险，不能照抄 |
| mini-jev | 原生HFPyTorch，主last_hidden默认use_cacheFalse；另有prefixbroadcast | suffix positions/mask/真实末尾gather | 字母评分不同，方法存在不证明board主路径使用；不能改我方分词边界 |
| vllm-pr57250、openjev/razorback | vLLM上的DiffusionGemma；openjev记录NVFP4/TRITON_ATTN/one-read | 调度设计可参考 | diffusioncanvas不是Qwen3二分类器的后端替换 |

Decider35B-NVFP4是另一条vLLM0.29路径，不能把Decider整族统称vLLM；该运行记录反而关闭prefixcache，说明缓存并非所有模型/负载必赚。榜方methodology混合历史public/latency记录，不能把其中每条配置冒充当前0.4.1全部私有run的最终commit。

## 优先可移植的方向（未实施、未测收益）

1. **同field prefix一次prefill + 独立candidate/EOS suffix**：最直接减少长题多候选的重复计算。严格保留原tokenIDs、positions、causal隔离、原末尾pooling、FP32score和margin→softmax。不能只给候选换A/B回答码。
2. **有限shape buckets/CUDAgraph**：更针对短题约70ms/微批的固定调度成本；启动预捕获、有界显存、长尾eager。不在请求中偷偷编译导致冷形状时延。
3. **LoRA执行优化**：合并或lean-unmerged都是可研究方向，但FP32低秩分支与BF16合并权重的有限精度运算可能不同。必须保留真实保存的score，不能漏头或自动BF16舍入。
4. **vLLM pooling迁移候选**：v0.25.1文档列Qwen3ForSequenceClassification及LoRA路径，causal LAST pooling可用prefixcache；但文档也明确pooling目前不保证比HF更快。我们是2类，不应直接套仅num_labels1的/score或替换成Qwen官方reranker LM yes/no head。必须取得raw二分类logits再做原margin，而非对P(match)再softmax。FP32head/PEFTmodules_to_save/实际base/输入边界仍须适配和验证。

controller实际只读核实我方最终config：pad151643、EOS151645，二者不同；id2label0not_match/1match。迁移仍应按实际GenericForSequenceClassification的最右非pad行为处理，不把任意模型的EOS假设移植过来。

这些方向原则上可保留数学模型，但缓存、批形状、融合、merge、量化会改变浮点执行。无我方同硬件/同负载的实测，不给加速倍数，不声称概率/argmax完全一致。速度优化不会自动把24.97提升至28；若刻意改变prompt/head/校准/多顺序集成，须标为新模型/新实验。

## 主要来源与许可

- 榜方固定快照[methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/e05cd04cad2b4f8d6c34aaa486ff88f1a9584942/data/methodology.json)、[当前public/Full数据](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/e05cd04cad2b4f8d6c34aaa486ff88f1a9584942/data/v03.json)。不混比public与Full。
- [Decider1.2.1 serving](https://github.com/Mapika/decider/blob/v1.2.1/docs/SERVING.md)、[shared_prefix](https://github.com/Mapika/decider/blob/v1.2.1/decider/shared_prefix.py)：Apache2.0。
- [reflex SGLang](https://github.com/kshetrajna12/reflex/blob/231f896d818a62b94fec305ed553df1088486dcb/src/reflex/backends/sglang.py)：MIT；v2归档overlay未逐行解包复核。
- [Winnow implementation](https://github.com/EldanRing/winnow-inference/blob/6c2b3c04e248a319f2cb43832628eba03e55fe38/docs/IMPLEMENTATION.md)、[runtime lock](https://github.com/EldanRing/winnow-inference/blob/6c2b3c04e248a319f2cb43832628eba03e55fe38/runtime.lock.json)：代码MIT，模型条款另查。
- [Kev checkpoint](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/checkpoint.py)、[CUDAgraphs](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/cuda_graphs.py)：Apache2.0，Qwen3.5/库版本不能硬套。
- [Decision2Kai fast.py](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/cc7f30cda9e7c7b28c5f71ef33f01e09385f1c4b/decision2/fast.py)、[head源码](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/cc7f30cda9e7c7b28c5f71ef33f01e09385f1c4b/decision2/_vendor/dev2model/decision_model.py)：manifest/许可证声明Apache2.0。
- [GLiNER2模型代码](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/gliner2/models/span/model.py)：Apache2.0；[mini-jev broadcast](https://github.com/r-ms/mini-jev/blob/ca612198bfb69f538f029a4615f6d0a18b4f814c/minijev/engine.py)：MIT。
- [Metask scorer](https://github.com/metask-ai/metask-jev/blob/6cce2276d7898a9ed0122afbc49d0f0b74ccda98/inference/jev_scorer.py)：README/card声明Apache2.0，所查tree未见独立LICENSE，复制前补核vendor条款。
- [官方kit高效参考engine](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/decision_index/engines/transformers_engine.py)：MIT；可借branchmask/KVcrop，不可照搬LM-key评分函数。
- [vLLM0.25.1 pooling](https://docs.vllm.ai/en/v0.25.1/models/pooling_models/)、[classification](https://docs.vllm.ai/en/v0.25.1/models/pooling_models/classify/)、[feature兼容矩阵](https://docs.vllm.ai/en/v0.25.1/features/)。Context7仍额度耗尽，使用版本匹配的一手文档；未安装该库。

复制代码时保留原版权/许可/NOTICE并区分权重、基座、数据的独立条款。公开源码的注释/benchmark/能力声明不等于我们已经验证；不照搬截断、丢候选、限制26标签、改prompt布局或受不匹配硬件gate保护的kernel。真实benchmark收益和输出漂移均留待用户批准的单独有界验收。
