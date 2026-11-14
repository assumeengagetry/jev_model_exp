# Commit 2e334fb attention-mask 实验执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** 实际运行用户刚pull的`2e334fb76c4cd4826df37424450b2734101db9f8`中的causal／bidirectional／prefix_block，补上其明确未做的真实Hopper FA3前向、反向与真实数据实验，再按准入结果推进训练／完整public。

**Architecture:** 精确复制该commit的`attention_modes.py`及`patched/`，进入新`/tmp/opencode/jev-attention-2e334fb-20261112-v1`，不覆盖集群全局scripts或旧任务。首先3张卡各一mode，对同一实际完整训练macro与相同冻结诊断权重，以原HF tokenwise全形状分别做FA3路径和SDPA显式mask参考、实际loss／393组梯度及一步AdamW；第四卡留给独立基线／后续任务，不伪造已充分占满。后续质量训练必须fresh stock起点和clean数据，不拿旧因果权重切mask的分数冒充训练收益。

**Tech Stack:** 当前Torch2.9.1/CUDA12.8、Transformers4.57.6、PEFT0.18.1、真实已装FA3、H800，不新增依赖。

**Spec:** `2026-11-10-attention-mask-modes/PATCH.md`及用户2026-11-12“跑这个commit的实验”。

## Global Constraints

- 不改用户commit源码或全局运行文件；基准与补丁按字节绑定，所有偏离单独记录。
- 不重跑CPU仿真／synthetic/unit/smoke/hash；本轮跑真实模型／数据／原生kernel。
- 该commit仍是padded tokenwise布局，不把它声称全层padding-free；packing和Flex后端不夹入第一轮mask实验。
- prefix_block边界精确沿commit：`Candidate:\n`之后，不擅自改成Base Labs的state／question边界；与原方法的差异如实报告。
- 原R1权重有已知历史输入暴露，只作机制诊断。fresh训练不载它或旧Teacher标签缓存，不宣称其public分数clean。
- 不覆盖／恢复旧停止任务，保留失败资产，不发布／commit／push、不买资源、不抢占他人GPU。
- 正式public如执行，每臂完整140620请求／140178计分，全候选，不按正确性重试／换权重／混旧预测。

## Review Focus

- register_student_fa3后才能install_prefix_block_flash；注册顺序不能覆盖补丁。
- prefix_block的context必须跨forward和backward，cuQ≠cuK及noncausal必须真的执行nativeFA3 deterministic backward。
- 推理SDPA非因果必须显式4Dmask，不能只改config／is_causal后随padding变义。
- head、dtype、输入token／prefix边界、loss分母完全一致；不能混入前一PF手写骨干／融合形状变化。
- 禁止将普通bidirectional的深层prefix当immutable缓存，prefix_block与causal的缓存也不能混用。

### Task 1: 精确源码冻结与真实机制三臂

**Files:** 独立实验`code/run_modes.py`，只读`commit-code/attention_modes.py`及`commit-code/patched/`。

**Interfaces:** `run_modes.py --mode causal|bidirectional|prefix_block`；输入冻结原plan第一个含relation的完整macro，使用原PairPool/TokenStore、patch collate返回真实prefix_len；输出每mode的原始logits／gradient／update／optimizer及JSON指标。

- [ ] 核实git HEAD、工作区源码与commit字节一致；对比远端基准依赖，不覆盖已有scripts。
- [ ] 在每个独立进程先注册严格deterministicFA3，再应用exact commit mode；同一初态FA3／SDPA分别真实forward/backward／一步更新。
- [ ] 固定原合同：logits相对RMS≤1%、maxabs≤.02+.02×参考maxabs；loss≤1e-3×(1+|Lref|)；每组梯度／update相对L2≤2%＋近零绝对项1e-7√n、cos≥.999。所有实际误差均保存；不事后放宽，更新不通过可分析但不填通过。
- [ ] 独立只读review并处理具体问题，exclusive部署及真实GPU准入；主代理直接读取回执与失败／通过证据。

### Task 2: 机制与质量结论分离

- [ ] 若原生kernel失败，保留exact commit失败路径，定位最小适配后另namespace复验；不偷偷修改已冻结原件。
- [ ] 同权重切mask仅机制诊断，可在真实完整case留出诊断分布变化；不是mask训练收益。
- [ ] mask匹配的clean-origin训练：同一stock base/fresh LoRA/head，causal／bidirectional／prefix_block同clean完整case／顺序／loss／LR／曝光，原定100M逻辑pair token和10M/30M/100M观察点；待对称语义输入去污与D_select/D_confirm准入后冻结实际训练清单。
- [ ] 质量结论须实际训练和完整public，当前未运行前不填写结果。

本计划优先用户刚pull的源码；前一PF v1完整保留为失败机制，v2修正尚未部署，Flex代码审查可完成但不争抢本轮GPU。
