# Global Padding-Free Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在真实 Qwen3-0.6B 分类模型上实现训练可微／推理 prefix 复用的全层 packed 执行，并以同权重完整 public 回答是否涨分。

**Architecture:** 新建独立 `/tmp/opencode/jev-padding-free-20261112-local/code`，远端运行另用 `/tmp/opencode/jev-padding-free-20261112-v1`。共享原 PEFT 参数和原 Qwen3 tokenwise 模块，手动调用分层 forward 与 FA3 varlen；推理使用真实长度只读 prefix KV 和 packed suffix fork。dense 控制保留 tokenwise padding，但在同一位置 gather 出相同真实 QKV、调用同一 attention kernel，再 scatter 回 padded hidden，因此不混入 attention 后端变化。

**Tech Stack:** 现有 Python、Torch 2.9.1/CUDA 12.8、Transformers 4.57.6、PEFT 0.18.1、Triton 3.5.1、已装 FA3；不安装依赖。

**Spec:** `02-研究路径/05-Attention与效率/2026-11-12-global-padding-free-design.md`

## Global Constraints

- 延续用户自主迭代授权；不再要求逐实验确认。代码／数据／历史停止标记原样保留，不 commit／push／发布。
- 不新增 synthetic/unit/smoke、模型或语料 hash；本计划使用用户要求的真实 padding-free 机制实验、真实一步训练及完整 public。Suite 强制校验保留。
- 原 joint／R1 谱系有历史输入暴露，只作机制对照；不把分数解释为 clean SOTA。
- 完整 public 每臂执行 140620、官方计分 140178，全部候选和上下文保留、不重试坏题、不复用旧预测。
- packed 主干 hidden `[sumL,1024]`；LoRA／Norm／MLP 不算业务 padding，真实 EOS offsets；训练 relation unit／macro／分母不改变。
- 所有启动先确认目标 GPU 无外来进程；失败保留 partial，不强制抢占或杀他人任务。
- clean-origin 同曝光／同预算训练另行执行；旧 warm-start／5M 不能作为新公平提分计划。

## Review Focus

- dense／packed 的位置、gather/scatter 与 EOS 对齐：真实训练 macro 的逐序列／置换实验负责。
- varlen 不能退化为跨序列 causal：真实 A 单独／混批／换邻居负责。
- prefix 缓存不得被 suffix 污染、suffix 使用绝对位置：A→B→A 和 full-pair 对照负责。
- FP32 LoRA/head 梯度和 loss 分母：原始完整 macro 的梯度／一步 AdamW 负责。
- public 全覆盖和序号恢复：原 Suite merge 按 `_evaluation` 和完整 payload 核对负责。

## 执行前冻结的数值合同（不是已通过结果）

- 整数 token／cu／position／EOS／unit／候选恢复必须精确相同，所有输出有限。
- 对齐 hidden/logits：相对 RMS `<=0.01`（分母 max(reference RMS,1e-3)），最大绝对误差 `<=0.02+0.02*maxabs(reference)`；同时保存实际误差，不只保存 pass。
- macro loss：绝对误差 `<=1e-3*(1+abs(reference_loss))`。
- 每个 LoRA/head 梯度及参数更新：L2 误差 `<=0.02*norm(reference)+1e-7*sqrt(numel)`；reference RMS >1e-7 的组要求 cosine >=0.999。近零组单独报告而不填伪 cosine。
- 数值漂移预测带：dense top1-top2 margin `<=0.04+0.04*maxabs(dense candidate margins)`。带外翻转须调查；带内分歧也不称逐题预测一致／学习提分。
- 同一数据 dense 重复若噪声超合同，先调查，不看 packed 后放宽；原始失败材料保留。
- 相同曝光质量目标首先是保持；没有多 paired seeds／干净确认集，不将微小分差称系统性提升。

### Task 1: 真实布局和可微骨干

**Files:** Create `code/packed_layout.py`, `code/packed_model.py`, `code/mechanism.py`。

**Interfaces:** `make_batch(rows, device, layout, pad, prefix_length=0) -> TokenBatch`，持有 ids/positions/cu/real_indices/eos_indices/lengths。`PackedBackbone(backbone, head, fa3).forward(batch, prefix=None, capture_prefix=False) -> (logits, prefix_or_none)`；不改变原模型注册参数或保存键。

- [ ] 先实现真实原始训练 macro 的验收入口：原 HF+FA3、dense-same-kernel、packed 的 forward／loss／梯度／一步更新和逐层行数；实现缺失时记录门禁未通过，不运行合成 smoke。
- [ ] 实现直接 packed CPU 拼接、真实 position reset、FA3 varlen 与原可微 Qwen3/PEFT 模块，真实 token 预算组批不拆 relation unit。
- [ ] 以冻结原 plan 第一个含 relation 的完整 macro，以及长度元数据选出的长短覆盖 macro 验证；保存原始成员索引、所有误差和 native FA3 backward 计数。

### Task 2: 真实 prefix 与 packed suffix 推理

**Files:** Create `code/packed_fork.py`, `code/packed_runtime.py`, `code/packed_engine.py`。

**Interfaces:** `fork_attention(q[T,16,128], k/v[T,8,128], prefix_k/v[P,8,128], cu, max_length) -> output[T,16,128]`；`PackedRuntime.prefill(prefix)`, `.branched(P, rows)`, `.full(rows)`；兼容原 engine render／field softmax。

- [ ] 真实 public metadata 预选单候选、变 K、长短 suffix、共享 prefix case；冻结 run_id，不按结果挑。
- [ ] 实现只读真实 P KV、真实 suffix T，动态 eager 无 Graph 假行；dense 与 packed 使用相同 fork／FA3 和相同波次。
- [ ] 真实样本验证 full-pair 对 prefix-fork、A 单独／混批／换序／换邻居、prefix A→B→A 完整复用；hook 覆盖各层和所有 tokenwise 子模块，包括 LoRA。

### Task 3: 冻结部署与完整 public A/B

**Files:** Create `code/pf_common.py`, `code/evaluate.py`, `deploy.py`, `verify.py`；执行账本 `progress.md`。

**Interfaces:** `evaluate.py --layout dense|packed [--worker 0..3]`；原 runner、原 Suite、原 scorer不改。

- [ ] 独立只读 review 后核实修正；封存代码字节／源文件 stat binding，不创建模型或语料 hash。
- [ ] Task1/2 通过后，dense 和 packed 各跑四个不相交 shard（可以同 GPU 先后、同一波次），分别完整 merge/score；一臂失败不从另一臂补题。
- [ ] 直接核对每臂全部 140620 输出与完整 payload、官方 140178 计分，逐题报告概率／choice／margin 分歧、mean/p80／墙钟／显存及实际 tokenwise 行；不是官方单卡 latency admission。
- [ ] 归档真实 JSON 证据及结果；回答同权重是否涨分，并将数值差异、效率收益、尚未进行的 clean 训练分开。

### Task 4: clean 训练后续边界

- [ ] 修订 R2：stock base＋fresh LoRA/head、同数据/初始化/曝光，预定每臂 100M 逻辑 pair token、10M/30M/100M 观察点；先完成对称任务请求去污和 D_select/D_confirm 冻结，禁止沿暴露谱系冲榜。
- [ ] 在 Task3 机制结果明确后独立安排同曝光与同 GPU 时间训练；未运行前不给训练质量结论。

## 自审

共享骨干和推理是同一模型的依赖链，先做可微正确性再 prefix，再完整 public；任务 4 为后续明确边界，不阻塞用户当前同权重验证。实际实现源位于新 namespace，仓库现有未提交材料不改写，不创建新 worktree／安装依赖／跑额外基线套件。
