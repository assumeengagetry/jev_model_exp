# 推理调度与计算内核 v2 执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留当前joint final权重、逐候选模型及评分规则，重做调度/热点内核，修复v1多field退步并以一轮新的完整public验证。

**Architecture:** 小候选field跨field合批；适合复用的field在原有token ID拼接结果上构造公共前缀，可跨field共享，不重新分词。全序列与prefix用无显式dense mask的因果SDPA；suffix使用直接读取单份prefix KV的Triton online-softmax kernel，避免重复cat/expand KV和4D mask。融合RMSNorm、Q/K norm+RoPE、SiLU×up、LoRA的scale/add；保留未merge的LoRA三次线性算术。

**Tech Stack:** 已有Torch2.9.1 / Triton3.5.1 / Transformers4.57.6 / PEFT0.18.1 / H800 GPU0。无新增依赖安装；原upstream kit固定25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d。

**Spec:** 用户2026-11-11明确确认“保留当前权重和评分规则，重做推理调度与计算内核”，并不要求寻找没有代码的4B对标。沿用本会话完整public验收范围，不复用旧预测。

## Global Constraints

- 同一原joint official-frozen/model与实际base；原独立prefix/candidate分词、EOS、语义位置、最右非pad读出、两类margin和field全候选float64 softmax不变。
- 不训练、不merge/量化、不换head、不改prompt/温度、不截断/删候选。保留保存的FP32 LoRA/head原张量；BF16运行副本仅缓存原autocast本已执行的转换，不写回权重。
- 融合内核保留BF16中间舍入边界、FP32归一化/online softmax累积；浮点归约顺序/attention实现可导致输出漂移，必须报告而不预称逐位等价。
- 延续此前不额外tests/smoke/model-or-corpus-hash限制；Triton首次JIT与CUDA Graph warmup是实际引擎启动所需，不另跑synthetic benchmark/模型诊断。验证使用独立只读审查、真实新public以及既有输出对比。
- 仅GPU0；不动旧两次public或readout，不停止他人进程。不上传/commit/push/重启服务。
- 当前研究库已由用户整理并提交：只保留文档和聚合证据，不强行把执行源码加回Git。初版库外源码/远端根完整保留；当前有界JIT修正版源码`/tmp/opencode/jev-inference-kernels-20261111-bounded-local/code/`、独立远端`/tmp/opencode/jev-inference-kernels-20261111-bounded-jit/`。本计划和后续证据跟随当前中文目录结构。
- 半小时是争取目标，不是承诺；不使用短样本代替全程吞吐，不在运行中改封存代码。

## Review Focus

- 仅右padding且严格causal时，真实EOS前没有未来padding；full/prefix可去除padding mask，不能搬到左padding/双向网络。
- 跨field分组只移动既有token边界；每条完整token序列、位置、候选索引、field softmax范围保持原样。
- prefix只写一次、suffix只读；GQA query head→KV head为h//2，online softmax覆盖完整真实prefix和本分支此前suffix，不读其它分支或bucket空隙。
- RMS与LoRA等融合不能把原BF16乘法/加法中间舍入悄悄合成FP32 FMA；图内禁止CPU同步；所有输入、运行副本、cache和输出地址必须存活。
- pinned异步H2D源在该request最终D2H同步前保留，不重用或提前释放；Graph只启动捕获，长尾eager仍用同一正确内核。

## Task 1: 内核与静态模型执行

**Files:** 库外源码`fused_kernels.py`、`kernel_model.py`。

**Interfaces:** `install_kernels(backbone, head)`返回运行head及安装说明；backbone仍调用原Qwen3层结构，attention使用`PrefixWriter`或`SharedFork`上下文。

- [x] 核对当前仓库整理、远端原冻结代码、GPU及实际依赖；Context7额度耗尽，读版本匹配Torch/Triton/Qwen3主源。
- [x] 实现BF16 staged舍入的RMS、Q/K norm+RoPE、gate/up、LoRA scale/add；实际生产startup/capture成功，长尾全覆盖待完成。
- [x] 实现只读prefix+独立suffix GQA online softmax，不autotune/量化。首生产发现长度特化JIT膨胀，已改S/stride为显式非特化runtime参数，在新根从头执行。
- [x] 安装持久autocast副本；实际保存参数精确准入成功，保存原参数且不执行LoRA merge。

## Task 2: 请求级调度与有界图执行

**Files:** 库外源码`kernel_runtime.py`、`kernel_engine.py`。

**Interfaces:** Engine协议不变；`score_request(field token plans)`返回按原flat候选顺序的margin及完整逻辑token数。

- [x] 全部field先渲染；短field统一长度分组/合批；完整BFCL实际v1 223.05秒→38.62秒，非全程加速承诺。
- [x] 对符合预算的公共前缀跨field或单field复用；基于suffix而不是重复prefix长度计算wave激活预算。
- [x] full图batch覆盖1/2/4/8/16/32，prefix覆盖到8192，fork不按prefix扩展KV；67图实际全部捕获，长尾完整eager。
- [x] 一次request最终D2H，异步H2D保持源存活；已执行首1831条全ok，最终calls/token分解待全量结束。

## Task 3: 审查、独立部署与完整public

**Files:** 库外源码`run_public.py`、`compare_results.py`复用已审查v1控制逻辑，适配新root/Engine/version/dependencies；研究报告放`01-研究结论/05-Attention与效率/inference-kernel-runtime-20261111/README.md`，证据放对应`03-证据与图表`目录。

- [x] 独立只读审查全部6源文件，修正完成重用证据缺口和提前称数值已测的Important；不伪称测试通过。
- [x] 新源码字节准入并封存；首attempt因实际JIT特化膨胀主动停止并保留1942条ok及源码/cache；有界JIT版新根，不混用或原地改代码。
- [x] GPU0当前新根`/tmp/opencode/jev-inference-kernels-20261111-bounded-jit/`15:37:16UTC启动；17.75秒加载/67图/6kernel版本，首1831条全部ok；无额外smoke/benchmark。
- [x] 16:38:12UTC直接核对、16:41:13UTC归档12组检查全部true；140620执行/140178计分全部ok。Public24.91、推理3436.6s，比v1快3.538x/原始快9.780x；全任务较v1累计耗时均下降。对v1 choice一致率98.7639%、概率均差0.0017041/最大0.206437，token计数差0；全部源/结果保留，GPU释放。30分钟目标未达，仍需约1.909x；无下一轮自动运行。

## 执行记录

用户已明确选择直接推进，不重复设计审批。沿用库外独立运行目录隔离，不创建/改动当前研究分支或旧资产。普通TDD工作流服从本会话保留的no-extra-tests约束；主代理实现，最终独立只读审查。
