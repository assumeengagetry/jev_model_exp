# 决策模型训练数据：核查与可获取清单

核查日期：2026-10-27（北京时间）。依据作者模型卡、技术报告、训练 manifest、构建代码及 Hugging Face API。榜单快照为 Decision Index 0.4.1，`generated_utc=2026-10-26T15:30:58Z`。本文关注决策后训练数据，不追溯基座预训练语料。

已保存数据来源与元数据，并通过 HF first-rows API 获取三个 Tasksource 数据集各两条真实训练样例；没有批量下载完整训练池、运行远端构建脚本或启动训练。样例在 [training-examples.jsonl](/Users/apple/Workspace/jev/research/2026-10-27/training-examples.jsonl)。

## 榜单模型实际披露了什么

| 模型/版本 | 一手披露的数据 | 当前能拿到什么 |
|---|---|---|
| Mammoth-1 / Elephant-1 | Mix 8 共 6.21M examples：核心技能 3.62M、安全/弃权 1M、评判/专业领域 1M、公开决策数据 0.59M；公开部分包含 61 数据集的 93 任务及 25 社区数据集 | 技术报告已公开；没有完整 Mix 8 下载或穷尽数据集名字表 |
| Perplexity v1.1 | 626,033 rows：530,103 Tasksource + 95,930 original。manifest 的 349 个 source/task 条目只覆盖 Tasksource 部分 | 逐来源清单与公开上游；未发现完整同版本训练 JSONL |
| Perplexity v1.2 | 新增部分基准的官方 train split，去污染；另加大量其他决策数据，致谢 Tasksource | 模型卡披露方向，未披露完整混合及精确条数；不能套用 v1.1 的规模 |
| StartLux 27B | Stage 1 约 0.8M、Stage 2 约 1.7M，包含公开 train split、程序/模型生成、teacher 标签及回放 | 论文附录有较早 preview 来源表；不是最终穷尽表，模型生成示例未重新分发；阶段有重叠 |
| Mapika Decider | 公开配方每 epoch 1.47M examples / 455M tokens，含公开源、程序规则、teacher 生成与格式增强；部分 loader 为 heldout | 转换器、mixture 构建器、部分 teacher JSONL；mixture-v2 builders 未公开，作者称公开配方覆盖约 60% |
| Bobcat 1.3 | 54,732 个保留旧问题，加 27k/54k 新技能题训练不同 arms；新增技能来自 17 个明确来源 | `dc13_skills.py` 列 HF ID、revision、文件与转换代码；17 个条目的指定文件均已核对公开可用；未找到最终完整训练包 |
| Jebadiah v2.1 | 最终 16,622 train records / 21,190 question presentations；另有 103 个 source-pool 条目 | data-summary、部分生成器和公开合成池。最终冠名数据仓库匿名 API 401，不能声称可直接取得 |
| Bespoke Nimble 原始公开版 | 2,676 对照式合成 records，1,338 对 base/counterfactual；324 holdout | 原始完整 train/eval JSONL 在 GitHub。不能把该配方自动套用到后续 Nimble v4 |
| Together tev1 | 37,840 train：MultiNLI 5,000、BoolQ 3,000、Banking77 3,000、AG News 1,500、SST-5 2,000、policy 13,500、routing 6,000、研究摘要分类 3,840 | 锁版本的源下载代码、构建代码和 manifest；成品 train 需要本地构建 |
| Sieve-31B | 190,314 questions、76.1M tokens、155 tracks；公开 train split、程序任务、合成/模型标签 | 模型卡披露类别和数量；仓库主要发布推理代码，未发现可直接拿的完整同版本训练池 |
| d3 / d3-turbo / d3-ultra | 检查的模型卡没有完整训练来源表；作者 Vela 文章提到 d3 的 1.31M 软标签混合 | 未发现完整公开训练混合；评测结果仓库不能当训练数据 |

出处：

- [Mammoth 作者技术报告 §2.4](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)
- [Perplexity v1.1 数据 manifest](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/3b45dead91dfa6d95aad6b95764a606fab2bf7a6/training/data-manifest.json)、[v1.2 模型卡](https://huggingface.co/perplexity-ai/pplx-decider-v1.2)
- [StartLux 论文附录 A](https://arxiv.org/html/2610.17309v1#A1)
- [Mapika 数据代码](https://github.com/Mapika/decider/tree/main/decider/data)、[teacher_data](https://github.com/Mapika/decider/tree/main/teacher_data)
- [Bobcat 新技能数据转换器](https://github.com/foxl-ai/bobcat/blob/main/src/bobcat/dc13_skills.py)、[模型卡](https://huggingface.co/sanghwa-na/bobcat-1.3)
- [Jebadiah 最终数据统计](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1/blob/main/data-summary.json)
- [Nimble 数据说明](https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md)
- [Together 逐来源 manifest](https://github.com/togethercomputer/tev1/blob/main/runs/new-v1/dataset-manifest.json)
- [Sieve-31B Training](https://huggingface.co/sthanika-ai/Sieve-31B#training)
- [d3 作者文章](https://vllm-sr.com/blog/vela-3-0-name-it-vela-finds-it/)、[d3-ultra 卡](https://huggingface.co/vllm-sr/d3-ultra)

## 优先获取的成品训练数据

同一 state 可以有多道 question；下面明确区分 state/record 与 decision/question。各数据包可能交叠，不能把行数直接相加。

| 数据集 / config | train 数量 | 监督与用途 | 获取 |
|---|---:|---|---|
| Tasksource typed / `full` | 2,550,012 decisions | 广覆盖分类、NLI、评分、规则等，已统一为 `state/question/kind/options/target` | [数据集](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)，文本 Parquet 约 1.95 GB |
| 同仓库 / `default` | 999,832 decisions | 体积更小的精选池 | 同一数据集，约 694 MB；与 full 重叠 |
| 同仓库 / `filtered-full` | 约 1.30M decisions | 历史冻结池的许可/来源/重叠过滤及 GLM 辅助审查；原标签保留 | [该 config 说明](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/blob/main/filtered-full/README.md)；不是 Perplexity 的那份 530,103 rows |
| Tasksource procedural / `all` | 360,000 多题 states | 18 类程序规则任务，含 policy、状态追踪、表格、证据、规划等；程序确定答案 | [数据集](https://huggingface.co/datasets/tasksource/procedural-typed-decisions)，约 266 MB |
| Tasksource synthetic / `default` | 6,666 states / 21,190 decisions | 多工作流及教师概率分布；样例 teachers 含 Jev、DeepSeek、Solar | [数据集](https://huggingface.co/datasets/tasksource/synthetic-typed-decisions)，约 12 MB |
| Jebadiah synth v2 | 3,253 states / 14,714 questions | 24 家族，choice/noul，两个教师的概率；为实验合成池，不等于最终 v2.1 train，也不保证加入后涨分 | [数据集](https://huggingface.co/datasets/frontier-infra/jebadiah-synth-v2)，24 个原始 JSONL |
| Nimble 原始 train | 2,676 records | 对照式少量改词翻转答案；hard reference；适合先验证训练流程 | [train JSONL 直链](https://raw.githubusercontent.com/bespokelabsai/nimble/main/data/train.jsonl)，约 30.5 MB |
| LocalLLaMA / `all` | 1,200 states / 6,000 decisions | 4 工作流，每 state 5 题；完整 gold 分布 | [数据集](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)、[train Parquet 直链](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/train-00000-of-00001.parquet)，约 599 KB |

数量来自当日数据卡/API。Tasksource `filtered-full` 的 README 记 1,305,478，而 size API 为 1,304,043，可能为缓存或版本差异；使用时固定 revision 并复核实载行数。Tasksource full/default 是另一组配置，不能与 filtered-full 累加。

Tasksource 主混合已经含 procedural 来源；原始多题池转换后的单题版本可能再次出现。保留 source、group_id、源 ID，按来源和原始样本去重。同一个 state 的问题与对照 siblings 应留在同一分割。

### 直接预览

```python
from datasets import load_dataset

main = load_dataset(
    "tasksource/tasksource-jev-typed-decisions",
    "full",
    revision="8df2e5c9a33f19a2c2b8febf291c3acb0a91a8a8",
    split="train",
    streaming=True,
)
row = next(iter(main))
print({k: row[k] for k in ("state", "question", "kind", "options", "target")})
```

`streaming=True` 为逐条读取；改成 `False` 才会缓存完整所选 split。选择文本 config，不要对整个 Tasksource 仓库运行无 pattern 的 snapshot_download，仓库另有约 140 GB 的 vision。

Jebadiah synth 的原始 JSONL 已确认公开。HF viewer 当前存在嵌套 schema 转换错误，优先下载原始 JSONL，再用标准库 `json.loads` 逐行读：

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="frontier-infra/jebadiah-synth-v2",
    repo_type="dataset",
    revision="d87c06dc1e89e931bd54fc9c663714bab424ede5",
    allow_patterns=["data/synth-*.jsonl", "README.md"],
    local_dir="data/jebadiah-synth-v2",
)
```

### 与分类头对接

Tasksource choice/score 的 `options[i]` 与 `target[i]` 对齐，可直接作为候选类别和监督分布。noul 的 `target` 往往是 `[p_true]`、`options` 为空，需要显式展开成约定顺序的两类概率，例如 `[1-p_true, p_true]`。不同来源的 questions/answers 有时为 JSON 字符串，有时为对象，转换时先统一。

## 需要按技能补充的原始数据

Bobcat 的 17 个正式新增源已逐项核查 pinned revision 和文件存在：MASSIVE、CommonsenseQA、QASC、MedMCQA、MedQA、ScienceQA、AQuA-RAT、GAP、WiCE、VitaminC、FaithDial、IBM argument quality、sarcasm headlines、MAUD、LEDGAR、UNFAIR-ToS、ColBERT Humor。精确 HF ID、文件路径、revision 和来源许可证见 [详细表](/Users/apple/Workspace/jev/research/notes-mapika-bobcat-jebadiah.md)。原始源不会自动变成统一 typed schema，可复用 dc13_skills.py 的转换器。

Perplexity manifest 中值得复用的来源包括 FOL-nli、doc-nli、WANLI、DynaSent、SPARTQA、MathQA、HelpSteer 1–3、civil_comments、医疗/法律 NLI、安全/注入检测。13 个 procedural 子任务合计 119,363 rows。原始 manifest 已保存到 [本地](/Users/apple/Workspace/jev/research/top-models/pplx-v1.1-data-manifest.json)。

## 对旧计划的必要修正

1. Tasksource 已有百万级现成 typed 数据，不必先从全部原始任务重写转换器。
2. Mammoth 已公开技术报告与数据统计，旧文档“其余不公开”已过时；完整 Mix 8 仍未提供。
3. Nimble `public_sources` 主要是外部评测 converters，包含大量 validation/test；不能原样当作训练池。Perplexity 参考其转换实现，并不代表可把冻结评测行拿来训练。[官方说明](https://github.com/bespokelabsai/nimble/blob/main/docs/PUBLIC_BENCHMARKS.md)
4. Mapika 的 loader 数包含 heldout，不能把所有下载的任务当作训练源。`scripts/train.sh full` 构建后直接启动训练；只取数据应单独调用 data 模块。
5. `pngwn/typed-decisions-v2-system-one` 的 MMLU-Pro test 重叠已被榜单明确记录，排除。不要将它与不同作者的 `ZefanCai/Open-Jev` 因名字相似而混为一谈。[榜单 methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/methodology.json)
6. Tasksource 主池按行保留 `license/license_use`，不要用仓库代码许可替代上游数据条款。固定 revision、保留 source/group_id，并把自己的评测文本与训练池做重叠检查。

建议第一轮使用 Tasksource default 或 filtered-full 跑通训练；根据独立 dev 的薄弱技能补 procedural 或 Bobcat 来源。要扩大覆盖再使用 full。上述选择是基于可获取性和已有模型证据，未做这几份数据对你基座的收益实验。

## 调研明细

- [顶部模型与来源证据](/Users/apple/Workspace/jev/research/top-models/top-model-training-data-2026-10-27.md)
- [Mapika / Bobcat / Jebadiah：逐来源清单](/Users/apple/Workspace/jev/research/notes-mapika-bobcat-jebadiah.md)
- [Nimble / Together / Tasksource：数据与构建用法](/Users/apple/Workspace/jev/research/nimble-together.md)
