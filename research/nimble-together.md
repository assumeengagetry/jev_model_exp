# Nimble、Together、Tasksource 与 typed-decisions 数据核查

核查日期：2026-10-27。仅阅读一手仓库、数据卡、Hub 元数据及数据首行；没有运行远程构建脚本，也没有批量下载训练集。

## 可以直接拿的训练集

| 数据 | 当前可验证的 train 大小 | 内容、格式 | 获取方式 |
|---|---:|---|---|
| tasksource/tasksource-jev-typed-decisions，default | 999,832 个 decision | 统一 `state/question/kind/options/target`；choice、score、noul；15 列 | `load_dataset(..., "default", split="train")`，约 673 MB Parquet |
| 同仓库，full | 2,550,012 个 decision | 更大的文本池；不同 config 大量重叠，不能直接与 default 相加 | `load_dataset(..., "full", split="train")`，约 1.93 GB Parquet |
| 同仓库，filtered-full | 当前 size API 为 1,304,043；数据卡历史数为 1,305,478 | 经过许可证、重叠、LLM 保守审核筛选的历史版本；13 列，无后续增量 | 当前 API 与卡片计数不一致，实际取数后确认；不要把它当成最新版 full 的等量新增数据 |
| tasksource/procedural-typed-decisions，all | 360,000 个 state，每个 state 多题 | 18 个程序任务，规则确定答案；`id/task/level/state/questions/answers` | `load_dataset(..., "all", split="train")`，约 266 MB Parquet |
| tasksource/synthetic-typed-decisions | 6,666 个 state，21,190 个 decision | 多题文本判断，soft labels；`state/questions/answers/teachers/skills/...` | `load_dataset(..., split="train")`，约 12.1 MB Parquet |
| LocalLLaMA/typed-decisions，all | 1,200 个 state，6,000 个 decision | 4 个工作流，各 300 train/100 test；每 state 5 题；包含完整 gold 概率 | `load_dataset(..., "all", split="train")`，约 599 KB Parquet |
| Bespoke Nimble 原始发布训练集 | 2,676 个 record，1,338 对 base/counterfactual | 每条一个 typed question，输入在 `input`，监督在 `reference` | 直接下载 `data/train.jsonl`，约 30.5 MB |

Hub 的 dataset 总行数会把 `all` 与各分任务 config 重复计入；上表只取指定 config/train。Tasksource 主混合已包含 procedural 来源，混合时仍需按来源和样本去重。Synthetic 是额外可取得的好数据，但不能仅凭公开时间就认定某个榜单模型用了该发布版本。

数据卡与计数来源：

- https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions
- https://datasets-server.huggingface.co/size?dataset=tasksource/tasksource-jev-typed-decisions
- https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/blob/main/sources.yaml
- https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/blob/main/filtered-full/README.md
- https://huggingface.co/datasets/tasksource/procedural-typed-decisions
- https://datasets-server.huggingface.co/size?dataset=tasksource/procedural-typed-decisions
- https://huggingface.co/datasets/tasksource/synthetic-typed-decisions
- https://datasets-server.huggingface.co/size?dataset=tasksource/synthetic-typed-decisions
- https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- https://datasets-server.huggingface.co/size?dataset=LocalLLaMA/typed-decisions
- https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md

### 直接文件链接

- Nimble train（完整 JSONL）：https://raw.githubusercontent.com/bespokelabsai/nimble/main/data/train.jsonl
- Nimble holdout，324 条，仅评测：https://raw.githubusercontent.com/bespokelabsai/nimble/main/data/eval.jsonl
- LocalLLaMA all/train：https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/train-00000-of-00001.parquet
- Tasksource default/train 第一片，六片之一：https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/resolve/main/data/train-00000-of-00006.parquet
- Procedural all/train 第一片，三片之一：https://huggingface.co/datasets/tasksource/procedural-typed-decisions/resolve/main/all/train-00000-of-00003.parquet
- Synthetic train：https://huggingface.co/datasets/tasksource/synthetic-typed-decisions/resolve/main/data/train-00000-of-00001.parquet

```python
from datasets import load_dataset

# streaming=True 可以先预览，无需先下载整个数据池。
main = load_dataset("tasksource/tasksource-jev-typed-decisions", "full", split="train", streaming=True)
print(next(iter(main)))

rules = load_dataset("tasksource/procedural-typed-decisions", "all", split="train", streaming=True)
soft = load_dataset("tasksource/synthetic-typed-decisions", split="train", streaming=True)
workflows = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
```

下载 Tasksource 时要选文本 config 或文件 pattern：这个仓库还有约 140 GB 的 `vision`，不要为取文本全仓下载。代码仓库目前写 1,100+ decision tasks（570 English、550 multilingual、45 vision），原始文档中的“600+”是较旧口径。

## Nimble 实际训练了什么

原始公开 Bespoke-Nimble-9B 用的是上述 2,676 条对照式合成数据，另有 324 条冻结 holdout。10 个领域，34 个训练 source families 与 held-out families 不相交。choice 856、noul 888、score 932。标签由模型核验，未人工复核，训练目标使用 hard reference labels，非 Jev 蒸馏。仓库中训练数据与 holdout 已直接提交。

实际首行 schema 已抽样核查：

```text
domain, evidence_certificate, family, id,
input: {state: string|array, questions: {decision: {criteria, instructions, type}}},
method, provenance, quality_status,
reference: {human_reviewed, source, target},
source_family, split, variant
```

适合先跑通分类头训练和验证“改少量证据，答案应翻转”的对照能力；体量不足以当 27B 的全部训练语料。当前仓库没发现清晰的数据集许可声明，不能由代码或模型许可直接推定数据许可。

**重要纠正：Nimble 的 public_sources converters 在官方文档中是外部评测套件，不是证明原模型在这些集合上训练。** 它们大量使用 validation/test，包括 BoolQ validation、MASSIVE test、PAWS test、Civil Comments test、Aegis test、HelpSteer2 validation 等。可以参考转换实现后改用上游 train，但不能原样把公开评测子集混入训练。

来源：

- https://github.com/bespokelabsai/nimble
- https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md
- https://github.com/bespokelabsai/nimble/blob/main/docs/PUBLIC_BENCHMARKS.md
- https://github.com/bespokelabsai/nimble/tree/main/nimble/datasets/public_sources

## Together tev1：可复现完整配方，成品数据需要本地构建

官方保存的最终 new-v1 manifest：37,840 train，4,568 dev；train 约 16.93M tokens。不是“约 38k”猜测，有逐来源确切计数：

| train 来源 | 条数 | 原始入口 |
|---|---:|---|
| MultiNLI | 5,000 | https://huggingface.co/datasets/nyu-mll/multi_nli |
| BoolQ | 3,000 | https://huggingface.co/datasets/google/boolq |
| Banking77 | 3,000 | https://huggingface.co/datasets/legacy-datasets/banking77 |
| AG News | 1,500 | https://huggingface.co/datasets/fancyzhx/ag_news |
| SST-5 | 2,000 | https://huggingface.co/datasets/SetFit/sst5 |
| 原始 policy 合成 | 1,500 | `build_dataset.py` |
| 新增 policy 合成 | 12,000 | `build_v2.py` |
| priority routing 合成 | 6,000 | `build_v2.py` |
| 虚构研究摘要分类 | 3,840 | `build_v21.py` + `configs/research-taxonomy.json` |

最终 train 含 14,500 个原始公开标注样本 + 23,340 个合成样本；合成 policy 13,500，routing 6,000，research classification 3,840。

复现入口（本次仅核查，没有执行）：

```sh
git clone https://github.com/togethercomputer/tev1
cd tev1
uv sync --locked
uv run python fetch_sources.py
uv run python build_all.py
```

产物：`data/new-v1/records/train.jsonl`、`sft/train.jsonl`、`instruction/train.jsonl`。`fetch_sources.py` 下载锁定 revision 的五个数据源及 tokenizer，不下载模型权重。最终成品 JSONL 不在当前 Git tree，需要自己构建。没有 Jev 蒸馏或 teacher soft probabilities。

record schema：`id/source/split/kind/state/question/options/answer/answer_key/provenance`；SFT 导出 chat messages；instruction 导出 prompt/completion。

来源：

- https://github.com/togethercomputer/tev1
- https://github.com/togethercomputer/tev1/blob/main/docs/DATASET.md
- https://github.com/togethercomputer/tev1/blob/main/DATA_SOURCES.md
- https://github.com/togethercomputer/tev1/blob/main/sources.lock.json
- https://github.com/togethercomputer/tev1/blob/main/runs/new-v1/dataset-manifest.json
- https://github.com/togethercomputer/tev1/blob/main/fetch_sources.py

## 各数据的取舍

- 要大规模现成文本池：先 Tasksource full/train，255 万 decision，而不是从 600+ 源逐个重写转换器。
- 要可验证的规则/状态/路由任务：Procedural all/train；18 类包括 policy、graph、planning、evidence、table、tracking；18 类各 20,000 train state，all 是它们的合并。
- 要学概率和不确定性：Synthetic train 的完整分布，另用 LocalLLaMA train 做适配；test 留出。
- 要最小可复现跑通：Nimble 2,676 条或 Together 37,840 条。
- Tasksource 主混合是 mixed license，保留 `license`、`license_use`；procedural/synthetic 与 LocalLLaMA 卡片声明 Apache-2.0。Together 数据源卡片明确 AG News、SST-5 许可元数据不完备，不能直接按仓库 MIT 重发全部数据。

版本快照：Tasksource 主仓库 `8df2e5c9a33f19a2c2b8febf291c3acb0a91a8a8`；procedural `05ca1d0388f57461a634a7caed67145d28e2eca7`；synthetic `7ec5a61d5db158f637c30adeb6e66660f1dfee87`；LocalLLaMA `9ee69423ce0b0568c2d2d86e92c72ed7952fd084`。获取后应按 revision 固定，并保留真实 split；本次没有判断它们与榜单私测的实际交叠。
