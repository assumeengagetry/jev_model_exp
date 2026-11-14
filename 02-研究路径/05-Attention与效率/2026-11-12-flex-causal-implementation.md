# Flex causal packed backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** 在相同真实训练macro上实际比较FA3和Flex causal后端的forward、393个LoRA/head梯度、一次更新及首编译时间，作为用户新增方法的第一个独立实验。

**Architecture:** 独立`/tmp/opencode/jev-flex-20261112-local/code`，只读复用冻结PF源码与同一原始macro。Flex仅接收真实packed QKV、singleton batch／GQA，序列隔离与原causal语义不变。block-bidirectional与Jina训练臂等精确原始资料后单独定义，不夹入此实验。

**Tech Stack:** 已装Torch2.9.1、Triton3.5.1，生产Flex编译；不安装、下载或改全局配置。

**Spec:** `2026-11-12-flex-block-bidirectional-reranker-design.md`。

## Global Constraints

- 同PF数值合同、相同原macro、权重、loss和optimizer，不看结果放宽。
- 旧谱系仅数值机制，不声称clean排名提分；不训练public答案。
- 不新增synthetic/unit/smoke/hash，不额外compile benchmark；生产首次编译计入墙钟，严格确定性，不fallback。
- 不使用默认无界`T×T`mask；按序列区间直接构造block metadata，同batch跨28层复用，缓存仅当前batch。
- 输出、失败与编译材料均在独立namespace保留；不commit／push／发布。

## Review Focus

- 非128倍真实长度的mask尾部不能越界索引；只允许tile mask，无业务padding。
- GQA保持Q16/KV8，不repeat成16份KV。
- 跨doc屏蔽与序列内causal同时成立，位置／EOS沿用PF。
- 同mask metadata跨层复用但不能沿用上一个不同batch。
- 生产编译缓存／首次编译时间、严格确定性／backward是否真实成功，不能仅API导入成功就记尝试通过。

### Task 1: 实际完整macro的backend对照

**Files:** Create `code/experiment.py`, `code/flex_backend.py`；独立`progress.md`。

**Interfaces:** `FlexBackend.bind(lengths,device)`构造当前batch元数据；`.flash_attn_varlen_func(q,k,v,cu_q,cu_k,max_q,max_k,softmax_scale,causal,deterministic)`仅接受匹配self-attention／strict causal合同，返回`[T,16,128]`。

- [ ] 先写真实macro验收入口，缺失实现时明确fail；不生成假token。
- [ ] 实现区间精确partial/full block目录、`BlockMask.from_kv_blocks(...,seq_lengths=(T,T))`及`torch.compile(flex_attention,fullgraph=True,dynamic=True)`，绑定当前真实序列owner mask。
- [ ] 等PF实际macro验收完成后，加载其绑定索引并原样重建；同初态分别FA3与Flex forward/backward与一次AdamW，保存逐组误差、编译时间／计数、显存、actual T证据。
- [ ] 只读审查、冻结部署后实际执行；不支持或超容差均保留fail结果，不自动降级／删样本。

**执行顺序说明：** 与PF完整public争用相同4卡，实际准入必须排队，不抢占已经运行的作业。mask／reranker后续训练计划不以本数值机制代替。
