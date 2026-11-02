# 新机恢复所需小代码：原样备份索引

本轮用新 SSH 连接参数核对了 201 个 `scripts/`、`src/`、`config/` 文件，均返回 `UNCHANGED_ALREADY_VERIFIED`：源机 hash 与已落盘本地文件一致，新增复制 0 文件。没有修改源训练、配置或已有备份版本。

## 1. 编码、输入格式与加载器

以下路径均相对本备份目录，源机对应 `/home/dev/assumeengage/` 下的同名路径。

| 职责 | 已落盘路径 / 符号 |
|---|---|
| 共享输入渲染与 source→area | `files/scripts/pair_data_common.py`：`render_pair`、`domain_for` |
| token 流、逐 pair 映射读取 | `files/scripts/pair_training_runtime.py`：`TokenStore`、`PairPool` |
| 输入组装、长度处理、EOS、padding | 同上：`assemble`、`clip_tokens`、`collate`、`microbatches` |
| 共同分类初始化与标签命名 | `files/scripts/initialize-pair-classifier.py` |
| 实际模型/adapter 加载 | `files/scripts/train-pair-model.py`、`train-pair-model-takeover.py`；`files/scripts/run-formal-evaluation.py:load_classifier` |
| 原生自定义推理/解码/评分入口 | `files/scripts/run-formal-evaluation.py` |
| 官方 scorer、suite、engines、fixtures | `files/src/decision-index-main/` |

在当前源机脚本和已备份 Python 源码中仍未找到名为 `PairEncoder` 的原始类；上述是当前重建流水线的共享实现，不能冒称旧 `language_baseline.common.PairEncoder` 已找回。

实际输入模板来自 `render_pair`：固定提示 `Decide whether the candidate is a correct match.`，随后是 `Input:`；非空时追加 `Question:` 与 `Available criteria:`，再接 `Candidate:`。前缀/候选后缀由缓存保存，runtime 在 `assemble` 中组合并追加 EOS。非字符串输入采用共享 `text()` 的 JSON 序列化规则；不能在新机任意改提示、criteria、token 边界、截断或 padding 后仍声称输入一致。

## 2. id2label / label2id：已核对三份配置一致

共同初始化及 Knowledge/Tools final 均为：

```json
{
  "id2label": {"0": "not_match", "1": "match"},
  "label2id": {"not_match": 0, "match": 1}
}
```

- 共同初始化：本项目 `models/Qwen3-0.6B-common-pair-init-r32/config.json`。
- Knowledge / Tools：本项目 `artifacts/five-specialists-20261030/train/<模型名>/final/config.json`。
- 初始化记录也在共同目录 `initialization.json`；已有资产不重复 SCP。
- 训练 BCE 和评测 margin 使用 `logit[1]-logit[0]`。不得沿用历史“类 0 匹配”的旧约定，也不得加载 adapter 时重新随机初始化已训练分类头。

## 3. 分布式与 CPU 通信：两份原代码都保留

| 原文件 | 静态代码事实 | 备份/迁移边界 |
|---|---|---|
| `files/scripts/train-pair-model.py` | `init_process_group('nccl')`；FP32 平坦梯度缓冲留在 GPU；`all_reduce(..., SUM)` 手工同步 | 这是 NCCL 手工同步路径，不是 `DistributedDataParallel` 包装实现 |
| `files/scripts/train-pair-model-takeover.py` | `init_process_group('gloo')`；梯度复制到 CPU pinned FP32 缓冲后 `all_reduce(..., SUM)`，再回到 GPU；失败标志/损失也有 CPU 通信路径 | 作为源机临时接续代码原样保留，没有替换、删除或在源机切换 backend |
| `files/scripts/run-five-area-takeover.py` | 接续调度与 Gloo 通信标记 | 保留原调度上下文，不启动/重启它 |

**新机采用 NCCL/DDP 是用户的迁移要求，不是本轮已实现或已验证的训练配置。** 新机若改用标准 DDP，需另行处理现有手工梯度同步、全局 batch/微批与 loss 缩放，不能叠加 DDP 和原手工 all-reduce 导致双重同步。这里仅指出原代码机制和迁移边界，不修改任何训练源码或做训练测试。

## 4. 所有 prepare 脚本与配套构建入口

以下均已在 `files/scripts/`：

- `prepare-benchmark-train.py`
- `prepare-typed-pairs.py`
- `prepare-overnight-pairs.py`
- `prepare-main-extra-pairs.py`
- `prepare-official-evaluation-data.py`
- `prepare-evaluation-candidate-cache.py`
- 配套 `build-global-pair-pool.py`、`generate-training-rules.py`、`finalize-training-cache.py`、`finalize-frozen-evaluation-data.py`、tokenize 脚本及下载辅助源码也保留在同目录。

配置和规则在 `files/config/`；训练 manifest、source/area 元数据在 `files/cache/pair-training/`。逐行 NPY/Parquet、共享 token IDs/offsets 和冻结请求 payload 属于主任务的大资产备份，本代码包不重复拉取。

## 5. 范围

这里只证明原样小代码已落盘且本轮与源机 hash 一致，没有导入执行模型、跑评测、做训练恢复或验证新机 NCCL/DDP。源码中的绝对 root 仍为 `/home/dev/assumeengage`，迁移方需显式映射。
