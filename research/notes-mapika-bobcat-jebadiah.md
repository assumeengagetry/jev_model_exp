# Mapika / Bobcat / Jebadiah 训练数据核查

核查日期：2026-10-27（北京时间）。仅获取公开仓库、模型卡、HF API 元数据和少量源代码；没有运行仓库脚本、没有下载大数据集。下列“许可证”若标为作者清单，是一手项目的声明，不等于对镜像上游权利的独立法律确认。

## 最有价值的结果

1. **现成 typed decision 数据**：`frontier-infra/jebadiah-synth-v2` 确认是匿名可访问的公开 HF dataset，24 个 `data/synth-*.jsonl` 文件、3,253 states / 14,714 questions，Apache-2.0，含完整 teacher soft distributions。它确实是一个可直接下载的数据包，但不能冒充最终 v2.1 冠名训练集；作者报告过该合成池使 9B 变差。
2. **大量转换代码**：Mapika 的 `decider/data/` 是从公共 HF 源转统一 Example/Q 的现成起点，teacher_data 中还有随仓库公开的生成后 JSONL。最终 mixture pickle 不在仓库中，需要本地构建。**不要运行 `scripts/train.sh full` 来“只取数据”**：它在构建后马上开始训练。
3. **17 个有明确 revision / files / license 的技能源**：Bobcat 1.3 的 `dc13_skills.py` 可以逐项下载原始数据、转换成 typed rows，做 overlap audit，再按技能配额选择；没有找到官方打包发布的最终完整训练 JSONL。
4. **Jebadiah v2.1 的公开信息比旧文档细**：模型的 `data-summary.json` 列 **103 个 source-pool 条目**，最终训练是 16,622 records / 21,190 question presentations，calibration 280 / 512。source-pool counts 不能相加当成最终训练规模。最终完整训练数据仓库 `frontier-infra/jebadiah-data-v2-1` 匿名 API 返回 HTTP 401；只能确认目前未能匿名取得，不能断言私有或不存在。`...-item47` 同样 401。

## 一手证据与版本

- [Mapika repo](https://github.com/Mapika/decider)，检查时 main `91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc`。
- [Bobcat repo](https://github.com/foxl-ai/bobcat)，main `538169365291d7c46d2c116f8ee09ad9f4b3ac5e`；[Bobcat 1.3 card](https://huggingface.co/sanghwa-na/bobcat-1.3)，HF main `4f25c70ece312ec4e74c47f1c5264cb2aeea956a`。
- [Jebadiah repo](https://github.com/getainode/jebadiah)，main `a7e8108fbedaa08faa016bf4aeaa7c281b068639`；[27B v2.1 card](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1)，HF main `b1df30c3cd9ad97c31ddb0b7c6884b580876bcdd`。

## Mapika：公开 mixture 及边界

[数据主转换器](https://github.com/Mapika/decider/blob/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/decider/data/core.py)、[扩展任务](https://github.com/Mapika/decider/blob/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/decider/data/tasks_extended.py)、[mixture 构建器](https://github.com/Mapika/decider/blob/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/decider/data/mixture.py)、[一键训练脚本](https://github.com/Mapika/decider/blob/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/scripts/train.sh)。

- README 报告公开重现配方为 1.47M examples / 455M tokens 每 epoch；不能把这些全视为独立原始题，mixture 会复制、拆成单题、换格式、加 distractors。
- 普通任务默认 `TRAIN_CAP=20000`、`EVAL_CAP=1500`，大选项集合的普通 general 数据先缩到 10 项；mixture 又补全选项/长选项样本。
- **`heldout=True` 即整个任务不入训练**：`load_all()` 会读取源数据，但不把该任务的 tr 加入 train。静态检查当前下载的四个任务模块得到 94 个带直接 HF 调用的任务 loader，其中 27 个标为 heldout；这不是 94 个独立 HF 数据集，更不是 94 个都训练。
- 公开仓库内已有 [teacher_data](https://github.com/Mapika/decider/tree/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/teacher_data)：custom_questions、routing_messages、routing_terse、commands、situations 的 JSONL，以及 label_descriptions.json。来自本地 Qwen3.5-27B teacher（以各版本说明为准）。
- 程序合成：[rules.py](https://github.com/Mapika/decider/blob/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/decider/data/rules.py)，规则/状态双胞胎翻转标签，mixture 默认生成 90,000；teacher_contrastive/questions/situations 负责模型写题。contrastive_pairs.jsonl 在代码中可选，当前 tree 没有该文件，不应把它计作已公开数据。
- README 明说 **mixture-v2 builders 尚未公开**，`scripts/train.sh full` 重现其 public 60%；RL stage 也依赖未放出的外部研究训练环境。不能宣称这个仓库完整复现所有最新榜单模型的数据。
- Repo code Apache-2.0；各原始数据保留自身许可证，不能一概算 Apache。HF 源加载大多没有 revision pin，要另外锁版本。

只构建数据的命令（供复现参考，本次未执行）：

```bash
python -m decider.data.core --out data/tasks.pkl
python -m decider.data.mixture --base data/tasks.pkl --mode full --out data/mixture_full.pkl --probes data/probes.pkl
```

### Mapika loader 清单（直接解析源代码，heldout 请保留隔离）

`_split_pair(ID, config, TRAIN_SPLIT, EVAL_SPLIT)`；`_ld(...)` 中只有一个 split 的，具体拆分还应查看该函数，不能自动把全部行训练。以下是代码中的下载调用，不等于本次已验证每个远端可用。

| 任务 | 角色 | 代码中的数据调用 | 源文件位置 |
|---|---|---|---|
| helpsteer2 | 训练+评测（由转换器拆分） | `_split_pair("nvidia/HelpSteer2", None, "train", "validation")` | tasks_extended.py:24 |
| helpsteer3_pref | 训练+评测（由转换器拆分） | `_split_pair("nvidia/HelpSteer3", "preference", "train", "validation")` | tasks_extended.py:52 |
| stsb | 训练+评测（由转换器拆分） | `_split_pair("SetFit/stsb", None, "train", "test")` | tasks_extended.py:69 |
| hate_speech_scales | 训练+评测（由转换器拆分） | `_ld("ucberkeley-dlab/measuring-hate-speech", None, "train")` | tasks_extended.py:80 |
| liar2 | 训练+评测（由转换器拆分） | `_split_pair("chengxuphd/liar2", None, "train", "test")` | tasks_extended.py:109 |
| prosocial_safety | 训练+评测（由转换器拆分） | `_split_pair("allenai/prosocial-dialog", None, "train", "validation")` | tasks_extended.py:119 |
| ultrafeedback_pref | 训练+评测（由转换器拆分） | `_split_pair("HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs", "test_prefs")` | tasks_extended.py:140 |
| shp | 训练+评测（由转换器拆分） | `_split_pair("stanfordnlp/shp", None, "train", "test")` | tasks_extended.py:156 |
| hh_rlhf | 训练+评测（由转换器拆分） | `_split_pair("Anthropic/hh-rlhf", None, "train", "test")` | tasks_extended.py:170 |
| arena_pref | 仅评测 | `_ld("lmarena-ai/arena-human-preference-55k", None, "train")` | tasks_extended.py:187 |
| reward_bench | 仅评测 | `_ld("allenai/reward-bench", "default", "filtered")` | tasks_extended.py:207 |
| glaive_tools | 训练+评测（由转换器拆分） | `_ld("glaiveai/glaive-function-calling-v2", None, "train")` | tasks_extended.py:235 |
| toolace | 训练+评测（由转换器拆分） | `_ld("Team-ACE/ToolACE", None, "train")` | tasks_extended.py:267 |
| hermes_tools | 仅评测 | `_ld("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train")` | tasks_extended.py:301 |
| copa | 训练+评测（由转换器拆分） | `_split_pair("aps/super_glue", "copa", "train", "validation")` | tasks_extended.py:329 |
| wic | 训练+评测（由转换器拆分） | `_split_pair("aps/super_glue", "wic", "train", "validation")` | tasks_extended.py:343 |
| multirc | 训练+评测（由转换器拆分） | `_split_pair("aps/super_glue", "multirc", "train", "validation")` | tasks_extended.py:351 |
| cb | 仅评测 | `_split_pair("aps/super_glue", "cb", "train", "validation")` | tasks_extended.py:359 |
| fever | 训练+评测（由转换器拆分） | `_split_pair("copenlu/fever_gold_evidence", None, "train", "validation")` | tasks_extended.py:368 |
| wiki_qa | 训练+评测（由转换器拆分） | `_split_pair("microsoft/wiki_qa", None, "train", "test")` | tasks_extended.py:380 |
| msmarco_rel | 训练+评测（由转换器拆分） | `_split_pair("microsoft/ms_marco", "v1.1", "train", "validation")` | tasks_extended.py:393 |
| medmcqa | 训练+评测（由转换器拆分） | `_split_pair("openlifescienceai/medmcqa", None, "train", "validation")` | tasks_extended.py:409 |
| quality | 仅评测 | `_split_pair("emozilla/quality", None, "train", "validation")` | tasks_extended.py:416 |
| xstory_cloze | 仅评测 | `_ld("juletxara/xstory_cloze", "en", "eval")` | tasks_extended.py:426 |
| agenttraj | 训练+评测（由转换器拆分） | `load_dataset("AgentGym/AgentTraj-L", split="train")` | tasks_agents.py:15 |
| mind2web | 训练+评测（由转换器拆分） | `load_dataset("osunlp/Mind2Web", split="train")` | tasks_agents.py:67 |
| clinc_oos | 训练+评测（由转换器拆分） | `_split_pair("clinc/clinc_oos", "plus", "train", "test")` | core.py:98 |
| banking77 | 训练+评测（由转换器拆分） | `_split_pair("mteb/banking77", None, "train", "test")` | core.py:109 |
| massive_intent | 训练+评测（由转换器拆分） | `_split_pair("SetFit/amazon_massive_intent_en-US", None, "train", "test")` | core.py:118 |
| massive_scenario | 仅评测 | `_split_pair("SetFit/amazon_massive_scenario_en-US", None, "train", "test")` | core.py:129 |
| bitext_support | 训练+评测（由转换器拆分） | `_ld("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None, "train")` | core.py:139 |
| support_tickets | 训练+评测（由转换器拆分） | `_ld("Tobi-Bueck/customer-support-tickets", None, "train")` | core.py:156 |
| ag_news | 训练+评测（由转换器拆分） | `_split_pair("fancyzhx/ag_news", None, "train", "test")` | core.py:176 |
| dbpedia | 训练+评测（由转换器拆分） | `_split_pair("fancyzhx/dbpedia_14", None, "train", "test")` | core.py:184 |
| yahoo_topics | 训练+评测（由转换器拆分） | `_split_pair("community-datasets/yahoo_answers_topics", None, "train", "test")` | core.py:192 |
| newsgroups | 训练+评测（由转换器拆分） | `_split_pair("SetFit/20_newsgroups", None, "train", "test")` | core.py:200 |
| bbc_news | 仅评测 | `_split_pair("SetFit/bbc-news", None, "train", "test")` | core.py:208 |
| trec | 仅评测 | `_split_pair("SetFit/TREC-QC", None, "train", "test")` | core.py:216 |
| student_questions | 仅评测 | `_split_pair("SetFit/student-question-categories", None, "train", "test")` | core.py:224 |
| dolly_category | 仅评测 | `_ld("argilla/databricks-dolly-15k-curated-en", None, "train")` | core.py:232 |
| imdb | 训练+评测（由转换器拆分） | `_split_pair("stanfordnlp/imdb", None, "train", "test")` | core.py:241 |
| sst2 | 训练+评测（由转换器拆分） | `_split_pair("stanfordnlp/sst2", None, "train", "validation")` | core.py:248 |
| sst5 | 训练+评测（由转换器拆分） | `_split_pair("SetFit/sst5", None, "train", "test")` | core.py:255 |
| yelp | 训练+评测（由转换器拆分） | `_split_pair("Yelp/yelp_review_full", None, "train", "test")` | core.py:263 |
| amazon_stars | 训练+评测（由转换器拆分） | `_split_pair("SetFit/amazon_reviews_multi_en", None, "train", "test")` | core.py:271 |
| emotion | 训练+评测（由转换器拆分） | `_split_pair("dair-ai/emotion", "split", "train", "test")` | core.py:279 |
| go_emotions | 训练+评测（由转换器拆分） | `_split_pair("google-research-datasets/go_emotions", "simplified", "train", "test")` | core.py:287 |
| tweet_sentiment | 训练+评测（由转换器拆分） | `_split_pair("cardiffnlp/tweet_eval", "sentiment", "train", "test")` | core.py:296 |
| tweet_emotion | 训练+评测（由转换器拆分） | `_split_pair("cardiffnlp/tweet_eval", "emotion", "train", "test")` | core.py:303 |
| tweet_irony | 仅评测 | `_split_pair("cardiffnlp/tweet_eval", "irony", "train", "test")` | core.py:310 |
| fin_sentiment | 仅评测 | `_split_pair("zeroshot/twitter-financial-news-sentiment", None, "train", "validation")` | core.py:317 |
| cr_reviews | 仅评测 | `_split_pair("SetFit/SentEval-CR", None, "train", "test")` | core.py:324 |
| counterfactual | 训练+评测（由转换器拆分） | `_split_pair("SetFit/amazon_counterfactual_en", None, "train", "test")` | core.py:331 |
| subjectivity | 训练+评测（由转换器拆分） | `_split_pair("SetFit/subj", None, "train", "test")` | core.py:338 |
| tweet_offensive | 训练+评测（由转换器拆分） | `_split_pair("cardiffnlp/tweet_eval", "offensive", "train", "test")` | core.py:346 |
| tweet_hate | 训练+评测（由转换器拆分） | `_split_pair("cardiffnlp/tweet_eval", "hate", "train", "test")` | core.py:353 |
| hate_offensive | 训练+评测（由转换器拆分） | `_split_pair("SetFit/hate_speech_offensive", None, "train", "test")` | core.py:360 |
| civil_comments | 训练+评测（由转换器拆分） | `_ld("google/civil_comments", None, "train[:400000]"); _ld("google/civil_comments", None, "test[:60000]")` | core.py:368 |
| toxic_chat | 训练+评测（由转换器拆分） | `_split_pair("lmsys/toxic-chat", "toxicchat0124", "train", "test")` | core.py:385 |
| sms_spam | 训练+评测（由转换器拆分） | `_ld("ucirvine/sms_spam", None, "train")` | core.py:399 |
| enron_spam | 训练+评测（由转换器拆分） | `_split_pair("SetFit/enron_spam", None, "train", "test")` | core.py:406 |
| insincere_questions | 训练+评测（由转换器拆分） | `_split_pair("SetFit/insincere-questions", None, "train", "test")` | core.py:413 |
| ade | 仅评测 | `_split_pair("SetFit/ade_corpus_v2_classification", None, "train", "test")` | core.py:420 |
| snli | 训练+评测（由转换器拆分） | `_split_pair("stanfordnlp/snli", None, "train", "test")` | core.py:436 |
| mnli | 训练+评测（由转换器拆分） | `_split_pair("nyu-mll/multi_nli", None, "train", "validation_matched")` | core.py:443 |
| rte | 训练+评测（由转换器拆分） | `_split_pair("nyu-mll/glue", "rte", "train", "validation")` | core.py:450 |
| qnli | 训练+评测（由转换器拆分） | `_split_pair("nyu-mll/glue", "qnli", "train", "validation")` | core.py:457 |
| qqp | 训练+评测（由转换器拆分） | `_split_pair("nyu-mll/glue", "qqp", "train", "validation")` | core.py:464 |
| mrpc | 训练+评测（由转换器拆分） | `_split_pair("nyu-mll/glue", "mrpc", "train", "validation")` | core.py:471 |
| paws | 仅评测 | `_split_pair("google-research-datasets/paws", "labeled_final", "train", "test")` | core.py:478 |
| cola | 训练+评测（由转换器拆分） | `_split_pair("nyu-mll/glue", "cola", "train", "validation")` | core.py:485 |
| boolq | 训练+评测（由转换器拆分） | `_split_pair("google/boolq", None, "train", "validation")` | core.py:493 |
| strategyqa | 仅评测 | `_split_pair("ChilleD/StrategyQA", None, "train", "test")` | core.py:500 |
| pubmedqa | 仅评测 | `_ld("qiaojin/PubMedQA", "pqa_labeled", "train")` | core.py:507 |
| arc | 训练+评测（由转换器拆分） | `_ld("allenai/ai2_arc", "ARC-Challenge", "train"); _ld("allenai/ai2_arc", "ARC-Easy", "train"); _ld("allenai/ai2_arc", "ARC-Challenge", "test")` | core.py:534 |
| commonsense_qa | 训练+评测（由转换器拆分） | `_split_pair("tau/commonsense_qa", None, "train", "validation")` | core.py:544 |
| qasc | 训练+评测（由转换器拆分） | `_split_pair("allenai/qasc", None, "train", "validation")` | core.py:551 |
| openbookqa | 训练+评测（由转换器拆分） | `_split_pair("allenai/openbookqa", "main", "train", "test")` | core.py:558 |
| sciq | 仅评测 | `_split_pair("allenai/sciq", None, "train", "validation")` | core.py:565 |
| hellaswag | 训练+评测（由转换器拆分） | `_split_pair("Rowan/hellaswag", None, "train", "validation")` | core.py:577 |
| piqa | 训练+评测（由转换器拆分） | `_split_pair("ybisk/piqa", None, "train", "validation", rev="refs/convert/parquet")` | core.py:584 |
| social_iqa | 仅评测 | `_split_pair("allenai/social_i_qa", None, "train", "validation", rev="refs/convert/parquet")` | core.py:591 |
| winogrande | 训练+评测（由转换器拆分） | `_split_pair("allenai/winogrande", "winogrande_xl", "train", "validation")` | core.py:598 |
| race | 训练+评测（由转换器拆分） | `_split_pair("ehovy/race", "all", "train", "test")` | core.py:605 |
| mmlu | 训练+评测（由转换器拆分） | `_ld("cais/mmlu", "all", "auxiliary_train"); _ld("cais/mmlu", "all", "test")` | core.py:612 |
| medqa | 训练+评测（由转换器拆分） | `_split_pair("GBaker/MedQA-USMLE-4-options", None, "train", "test")` | core.py:619 |
| truthfulqa | 仅评测 | `_ld("truthfulqa/truthful_qa", "multiple_choice", "validation")` | core.py:626 |
| fin_phrasebank | 仅评测 | `_ld("AdaptLLM/finance-tasks", "FPB", "test")` | core.py:635 |
| bias_in_bios | 训练+评测（由转换器拆分） | `_split_pair("LabHC/bias_in_bios", None, "train", "test")` | core.py:643 |
| hwu64 | 仅评测 | `_ld("FastFit/hwu_64", split="test")` | tasks_heldout.py:20 |
| dbpedia_l2 | 仅评测 | `_ld("DeveloperOats/DBPedia_Classes", split="test")` | tasks_heldout.py:25 |
| dbpedia_l3 | 仅评测 | `_ld("DeveloperOats/DBPedia_Classes", split="test")` | tasks_heldout.py:30 |
| trec_fine | 仅评测 | `_ld("CogComp/trec", split="test")` | tasks_heldout.py:49 |
| quality_full | 仅评测 | `_ld("emozilla/quality", split="validation")` | tasks_heldout.py:56 |

## Bobcat：17 个正式新增技能源

[dc13_skills.py](https://github.com/foxl-ai/bobcat/blob/538169365291d7c46d2c116f8ee09ad9f4b3ac5e/src/bobcat/dc13_skills.py) 的 SOURCES 为每个源给出 pinned revision、具体文件、作者声明许可证；[THIRD_PARTY.md](https://github.com/foxl-ai/bobcat/blob/538169365291d7c46d2c116f8ee09ad9f4b3ac5e/THIRD_PARTY.md#bobcat-13) 有归属。

Bobcat 1.3：在剔除了 ARC/BANKING77 及命中评测内容后的 Bobcat 1.2 数据（54,732 questions）上，加入 27k 或 54k 新技能问题训练三个 LoRA arms，再平均权重；765 新问题被 overlap audit 去掉。这个规模是模型卡的披露，**不等于下面每个源都被完整使用**。原始 source 的 validation/test 在部分项也被拿来建训练候选池；复用到自己的评测时必须重新隔离，不能按原始文件名猜测角色。

| 功能源键 | 精确 HF dataset | 使用文件 / split | 作者声明许可证 | revision |
|---|---|---|---|---|
| massive | [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) | `en-US/train/0000.parquet` | CC-BY-4.0 | `ed58ac423a2f4121720918bf5301577edce4ffd3` |
| csqa | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | `data/train-00000-of-00001.parquet`; `data/validation-00000-of-00001.parquet` | MIT | `94630fe30dad47192a8546eb75f094926d47e155` |
| qasc | [allenai/qasc](https://huggingface.co/datasets/allenai/qasc) | `data/train-00000-of-00001.parquet`; `data/validation-00000-of-00001.parquet` | CC-BY-4.0 | `a34ba204eb9a33b919c10cc08f4f1c8dae5ec070` |
| medmcqa | [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) | `data/train-00000-of-00001.parquet` | Apache-2.0 | `91c6572c454088bf71b679ad90aa8dffcd0d5868` |
| medqa | [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | `phrases_no_exclude_train.jsonl` | CC-BY-4.0 (upstream MIT) | `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18` |
| scienceqa | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | `data/train-00000-of-00001-1028f23e353fbe3e.parquet` | CC-BY-SA-4.0 | `f18b0a70359ebfb41f658fd564208d0355b013f4` |
| aqua | [deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat) | `raw/train-00000-of-00001.parquet` | Apache-2.0 | `33301c6a050c96af81f63cad5562cb5363e88971` |
| gap | [coref-data/gap_raw](https://huggingface.co/datasets/coref-data/gap_raw) | `data/train-00000-of-00001.parquet`; `data/validation-00000-of-00001.parquet`; `data/test-00000-of-00001.parquet` | Apache-2.0 | `b3cb9f80cfade3374e73ecea63da8957eb125b3c` |
| wice | [tasksource/wice](https://huggingface.co/datasets/tasksource/wice) | `train.jsonl` | CC-BY-SA-4.0 | `49055086c199701ee46260f687f4fa4f836fb751` |
| vitaminc | [tals/vitaminc](https://huggingface.co/datasets/tals/vitaminc) | `train.jsonl` | CC-BY-SA-3.0 | `be6febb761b0b2807687e61e0b5282e459df2fa0` |
| faithdial | [McGill-NLP/FaithDial](https://huggingface.co/datasets/McGill-NLP/FaithDial) | `data/train.json` | MIT | `7a414e80725eac766f2602676dc8b39f80b061e4` |
| argq | [ibm-research/argument_quality_ranking_30k](https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k) | `train.csv` | CC-BY-3.0 | `590726b3765b1b90c5e53a17e3b1f77d92d3aa8a` |
| headlines | [Lots-of-LoRAs/task1488_sarcasmdetection_headline_classification](https://huggingface.co/datasets/Lots-of-LoRAs/task1488_sarcasmdetection_headline_classification) | `data/train-00000-of-00001.parquet`; `data/valid-00000-of-00001.parquet`; `data/test-00000-of-00001.parquet` | Apache-2.0 | `00a5979fea2e080fdfafd939ed67b83f6e7f4068` |
| maud | [theatticusproject/maud](https://huggingface.co/datasets/theatticusproject/maud) | `MAUD_v1/MAUD_train.csv` | CC-BY-4.0 | `37d5c3b95d18dcd8404cc5ce3fd5069be062392f` |
| ledgar | [coastalcph/lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue) | `ledgar/train-00000-of-00001.parquet` | CC-BY-4.0 | `c23fdff1a6bf74e0e1a71cb86f1e781d37da888c` |
| unfair_tos | [coastalcph/lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue) | `unfair_tos/train-00000-of-00001.parquet` | CC-BY-4.0 | `c23fdff1a6bf74e0e1a71cb86f1e781d37da888c` |
| humor | [CreativeLang/ColBERT_Humor_Detection](https://huggingface.co/datasets/CreativeLang/ColBERT_Humor_Detection) | `dataset.csv` | CC-BY-2.0 | `2bb7d6bce15e42c2a3cf2be8305fa3049929d3ac` |

对上表 17 个 source 条目逐一请求 HF `api/datasets/<ID>/revision/<REV>`，均确认匿名可用（`private=false, gated=false`），全部脚本指定文件均存在于 pinned revision；这包含同一个 lex_glue repo 的两个子数据集。没有下载 parquet 正文。

可复用的流水线入口：`python -m bobcat.dc13_skills fetch` / `build` / `audit` / `select`。每一步的具体路径参数见脚本 argparse；不应该直接跳过审计而拿全部 build pool 训练。`fetch` 仅拉 pinned 原始文件，`build` 输出 pool.jsonl/build.json，`audit` 输出 kept.jsonl/audit.json，`select` 再按技能配额产 skill.jsonl。

同一个 SOURCES 表还列了后续 H-cycle 的 civil_comments、DBpedia、WANDS、ETHICS commonsense；代码存在不证明它们全部进了最终 Bobcat 1.3。GoEmotions、StrategyQA、QuaRTz、PAWS、ETHICS justice、SNIPS 在代码中明确 `held_out=True`，用于泛化检查，不能列作已用训练数据。

Bobcat 旧数据和合成脚本：[student_data_v11.py](https://github.com/foxl-ai/bobcat/blob/538169365291d7c46d2c116f8ee09ad9f4b3ac5e/src/bobcat/student_data_v11.py) 做 planted-answer、long padding、缺证据 abstain，SNLI train、SQuAD2 train 等输入；[flash_families.py](https://github.com/foxl-ai/bobcat/blob/538169365291d7c46d2c116f8ee09ad9f4b3ac5e/src/bobcat/flash_families.py) 为原生程序决策族。商业源许可声明还涉及 CC-BY-SA 等，各源衍生决策不统一改成仓库 Apache 许可。

## Jebadiah：可下载的合成包、可重建 v1、v2.1 披露

### 直接可拿：jebadiah-synth-v2

- [数据集](https://huggingface.co/datasets/frontier-infra/jebadiah-synth-v2)，匿名 HF API 验证 `private=false, gated=false`，revision `d87c06dc1e89e931bd54fc9c663714bab424ede5`。
- 24 个 `data/synth-*.jsonl`，3,253 records/states、14,714 questions；card 的 configs 使用 `data/synth-*.jsonl`，没有官方 test split，应按 state/family 划分 train/calib。
- Apache-2.0；DeepSeek V4 Flash 写 states，DeepSeek V4 Flash 与 Qwen3.8 27B 为每题打概率，两者分布均保留，gold 为均值。choice 2–12 项，noul yes/no，无 score。
- 24 家族覆盖客服、工具选择、发票、告警、邮件路由、审核、表单校验、事实证据、摘要忠实性、翻译质量、JDE coverage 等。
- [生成/过滤代码](https://github.com/getainode/jebadiah/tree/a7e8108fbedaa08faa016bf4aeaa7c281b068639/data/pool-v2)。card 与源码公开；不需自己调用 teacher 才能获得既有数据。

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

HF viewer 当前存在嵌套 schema casting 错误，因此优先下载原始 JSONL，用标准库 json 逐行读取。作者未替你完成 train/calib 分割；读记录中的 questions/gold/provenance，按 state/family 做自己的分割。

### 可重建：v1 pool

[convert_data.py](https://github.com/getainode/jebadiah/blob/a7e8108fbedaa08faa016bf4aeaa7c281b068639/data/convert_data.py)、[manifest](https://github.com/getainode/jebadiah/blob/a7e8108fbedaa08faa016bf4aeaa7c281b068639/data/manifests/convert-data-v1.json) 可重建约 11,013 records / 15,813 questions，再按 family hash 95/5 划 train/calib。built pool 不在 Git 仓库。

| 来源 | 使用范围 | v1 pool 用量 | 许可证/说明 |
|---|---|---|---|
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | train，revision `ea9306458d6e9563628369a3d1e72e362fb381d2` | 1,200 states（多题） | Apache-2.0，test 400 states / 2,000 questions 隔离 |
| [jaredpalmer/kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites) | `v7/decision-v7/train.jsonl`，revision `a88f56db5341397299137cb68775c2ea6e3f68cb`，仅 BoolQ/MNLI/DBpedia/程序合成 contrastive & composition 允许子集 | 5,576 records | 上游不同许可证，非整个 Kev 包都入训练 |
| [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | train，revision `990b2711a36180dd19d9c94b8627844866f8982a`；validation 全隔离 | 2,573 score records | CC-BY-4.0 |
| [mteb/summeval](https://huggingface.co/datasets/mteb/summeval) | 原源 split=test，但排除 Nimble 评测文章后分自己训练/校准 | 1,664 score questions（416 summaries × 4 维） | 作者 manifest 为 MIT；仅剩余文章使用，按 article family 分割 |

v1 明确排除 Banking77（作 zero-shot eval）；Kev 中 AG News、Yelp、IMDb、Amazon reviews、SST5、TREC 因许可问题或缺许可而不训练；Nimble train 在当时因未声明许可证不训练。它的 eval 构建器也下载 PubMedQA、Banking77、Nimble 等，所以“代码下载了某源”绝不等于“模型用其训练”。

### v2.1 实际披露

[模型卡](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1#training-data)、[data-summary.json](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1/blob/b1df30c3cd9ad97c31ddb0b7c6884b580876bcdd/data-summary.json)。

- final train=16,622 records / 21,190 question presentations，calib=280 / 512；choice=11,708，noul=3,696，score=5,786。
- `sources` 有 103 条 source-pool 记录，既有精确 HF ID，也有别名（如 `prost`, `agent_action_safety`, `item25-oracles.py`），不能全视为可直接 `load_dataset(source)` 的 ID。
- 13,291 条 oracle 原始池由 `item25-oracles.py, seed=20261008` 生成；当前 public repo tree 未找到这个确切脚本，不能说这部分完整已公开。repo 另有 item33/47/49/54/55 等公开生成器，属于后续/不同实验，不与 v2.1 13,291 条自动等同。
- 公开后续脚本是有价值的程序合成起点，例如 [item33_skill_data.py](https://github.com/getainode/jebadiah/blob/a7e8108fbedaa08faa016bf4aeaa7c281b068639/data/item33_skill_data.py)：HWU 原始人类意图数据+工具 timing 世界+evidence 世界，manifest 报 4,050 train / 450 calib，按 family 分割；不能把它当作 v2.1 的全部训练池。
- `frontier-infra/jebadiah-data-v2-1` 与 `frontier-infra/jebadiah-data-v2-1-item47` 匿名 API 返回 401，当前可交付物应标注“sources/manifest 公开，最终包未能公开下载”。
- [许可证纠正 overlay](https://github.com/getainode/jebadiah/blob/a7e8108fbedaa08faa016bf4aeaa7c281b068639/data/manifests/item46-v21-license-corrections.json) 把 x-stance 改成 CC-BY-NC-4.0，明确训练不许、不可作为商业训练/公开打包。它属于历史 holdout，不是实际训练；说明不能只看镜像 code license。

### v2.1 完整 source-pool 条目

以下逐项来自 data-summary.json，不代表精确最终 split 抽样数；带斜杠也可能是项目内部别名。数据可用性、原始 split、revision 未在该 summary 逐项给出；未自行猜测。

| source / alias | 作者声明许可证 | source-pool records | source-pool questions |
|---|---|---:|---:|
| `12ml/e-CARE` | BSD 2-Clause License (DPI) | 986 | 986 |
| `agent_action_safety` | Apache-2.0 | 687 | 687 |
| `allenai/qasc` | cc-by-4.0 | 885 | 885 |
| `allenai/quartz` | cc-by-4.0 | 827 | 827 |
| `AndyChiang/cloth` | mit | 1116 | 1116 |
| `AndyChiang/dgen` | mit | 764 | 764 |
| `bogdanminko/wildguardmix-cleaned` | odc-by | 1468 | 1468 |
| `cambridgeltl/xcopa` | cc-by-4.0 | 391 | 391 |
| `community-datasets/offenseval_dravidian` | cc-by-4.0 | 2006 | 2006 |
| `community-datasets/proto_qa` | cc-by-4.0 | 798 | 798 |
| `declare-lab/cicero` | mit | 1082 | 1082 |
| `deepset/prompt-injections` | Apache-2.0 | 177 | 177 |
| `docred` | mit | 201 | 201 |
| `dynabench/dynasent` | CC BY 4.0 (DPI) | 1767 | 1767 |
| `ethics/commonsense` | mit | 358 | 358 |
| `ethics/deontology` | mit | 398 | 398 |
| `ethics/justice` | mit | 415 | 415 |
| `fancyzhx/amazon_polarity` | Apache-2.0 | 838 | 838 |
| `flaitenberger/wnut_17` | cc-by-4.0 | 895 | 895 |
| `GGLab/PARADISE` | mit | 1068 | 1068 |
| `google-research-datasets/go_emotions` | Apache-2.0 | 89 | 89 |
| `google/civil_comments` | cc0-1.0 | 10748 | 10748 |
| `HannahRoseKirk/HatemojiBuild` | cc-by-4.0 | 817 | 817 |
| `headline_cause/en_simple` | cc0-1.0 | 830 | 830 |
| `hendrycks/ethics` | mit | 441 | 441 |
| `hendzh/PromptShield` | Apache-2.0 | 862 | 862 |
| `hope_edi/english` | cc-by-4.0 | 801 | 801 |
| `item25-oracles.py, seed=20261008` | CC0-1.0 | 13291 | 13291 |
| `jaredfern/codah` | odc-by | 884 | 884 |
| `jaredpalmer/kev-suites` | Apache-2.0 | 2177 | 2177 |
| `LabHC/moral_stories` | mit, MIT License (DPI) | 979 | 979 |
| `LocalLLaMA/typed-decisions` | Apache-2.0 | 1200 | 6000 |
| `marcov/health_fact_promptsource` | mit | 835 | 835 |
| `mteb/AfriSentiClassification` | cc-by-4.0 | 2293 | 2293 |
| `mteb/amazon_counterfactual` | cc-by-4.0 | 798 | 798 |
| `mteb/MassiveIntentClassification` | cc-by-4.0 | 156 | 156 |
| `multilingual-discourse-hub/disrpt` | Apache-2.0 | 769 | 769 |
| `multilingual/tweet_sentiment_multilingual` | cc-by-3.0 | 1956 | 1956 |
| `nedjmaou/MLMA_hate_speech` | mit | 837 | 837 |
| `neurae/dnd_style_intents` | Apache-2.0 | 1891 | 1891 |
| `neuralchemy/Prompt-injection-dataset` | Apache-2.0 | 831 | 831 |
| `nightingal3/fig-qa` | mit | 957 | 957 |
| `nvidia/HelpSteer` | cc-by-4.0, CC BY 4.0 (DPI) | 1857 | 1857 |
| `nvidia/HelpSteer2` | CC BY-4.0 (nvidia/HelpSteer2) | 2531 | 2531 |
| `nvidia/HelpSteer3` | cc-by-4.0 | 3533 | 3533 |
| `openlifescienceai/medmcqa` | Apache-2.0 | 1080 | 1080 |
| `pkavumba/balanced-copa` | cc-by-4.0, BSD 2-Clause License (DPI) | 332 | 332 |
| `propsegment/nli` | cc-by-4.0 | 827 | 827 |
| `prost` | Apache-2.0 | 945 | 945 |
| `pushpdeep/UltraFeedback-paired` | mit | 1039 | 1039 |
| `qbao775/PARARULE-Plus` | mit | 843 | 843 |
| `S-Labs/prompt-injection-dataset` | mit | 810 | 810 |
| `Samsoup/cosmos_qa` | cc-by-4.0 | 1069 | 1069 |
| `SemEvalWorkshop/sem_eval_2010_task_8` | CC0 1.0 (DPI) | 852 | 852 |
| `sileod/discovery` | Apache-2.0 | 158 | 158 |
| `sileod/mindgames` | Apache-2.0 | 834 | 834 |
| `sonos-nlu-benchmark/snips_built_in_intents` | cc0-1.0, CC0 1.0 (DPI) | 123 | 123 |
| `tasksource/acceptability-prediction` | Apache-2.0 | 3422 | 3422 |
| `tasksource/arct` | Apache-2.0 | 472 | 472 |
| `tasksource/babi_nli` | bsd | 1894 | 1894 |
| `tasksource/blog_authorship_corpus` | Apache-2.0 | 615 | 615 |
| `tasksource/clcd-english` | Apache-2.0 | 808 | 808 |
| `tasksource/commonsense_qa_2.0` | cc-by-4.0 | 806 | 806 |
| `tasksource/corr2cause` | mit | 1544 | 1544 |
| `tasksource/cycic_classification` | Apache-2.0 | 809 | 809 |
| `tasksource/cycic_multiplechoice` | Apache-2.0 | 921 | 921 |
| `tasksource/defeasible-nli` | Apache-2.0 | 1138 | 1138 |
| `tasksource/equate` | Apache-2.0 | 685 | 685 |
| `tasksource/feasibilityQA` | mit | 811 | 811 |
| `tasksource/hh-rlhf` | MIT License (DPI) | 3740 | 3740 |
| `tasksource/hlgd` | Apache-2.0 | 805 | 805 |
| `tasksource/I2D2` | Apache-2.0 | 800 | 800 |
| `tasksource/jigsaw_toxicity` | Apache-2.0 | 851 | 851 |
| `tasksource/logical-entailment` | Apache-2.0 | 782 | 782 |
| `tasksource/lonli` | mit | 858 | 858 |
| `tasksource/math_qa` | Apache-2.0 | 2220 | 2220 |
| `tasksource/monotonicity-entailment` | Apache-2.0 | 823 | 823 |
| `tasksource/multilingual-sentiments` | Apache-2.0 | 2005 | 2005 |
| `tasksource/naturallogic` | Apache-2.0 | 815 | 815 |
| `tasksource/numer_sense` | mit | 856 | 856 |
| `tasksource/patent-phrase-similarity` | cc-by-4.0 | 822 | 822 |
| `tasksource/procedural-typed-decisions` | Apache-2.0 | 54414 | 54414 |
| `tasksource/ruletaker` | Apache-2.0 | 812 | 812 |
| `tasksource/scone` | cc0-1.0 | 823 | 823 |
| `tasksource/sherliic` | CC BY 4.0 (DPI) | 354 | 354 |
| `tasksource/social_i_qa` | CC BY 4.0 (DPI) | 1073 | 1073 |
| `tasksource/SpaceNLI` | mit | 888 | 888 |
| `tasksource/spartqa-mchoice` | mit | 1152 | 1152 |
| `tasksource/spartqa-yn` | Apache-2.0 | 897 | 897 |
| `tasksource/starcon` | CC0 1.0 (DPI) | 556 | 556 |
| `tasksource/stepgame` | mit | 837 | 837 |
| `tasksource/sts-companion` | Apache-2.0 | 2405 | 2405 |
| `tasksource/subjectivity` | mit | 247 | 247 |
| `tasksource/temporal-nli` | Apache-2.0 | 867 | 867 |
| `tasksource/tracie` | Apache-2.0 | 268 | 268 |
| `tomngdev/shell-safety-v2` | mit | 1570 | 1570 |
| `Touche23-ValueEval` | cc-by-4.0, CC BY 4.0 (DPI) | 647 | 647 |
| `TuringBench` | Apache-2.0 | 859 | 859 |
| `ucirvine/sms_spam` | CC BY 4.0 (DPI) | 1133 | 1133 |
| `unigram/FOL-nli` | Apache-2.0 | 1003 | 1003 |
| `utilitarianism` | mit | 680 | 680 |
| `webis/args_me` | cc-by-4.0, CC BY 4.0 (DPI) | 791 | 791 |
| `zapsdcn/hyperpartisan_news` | CC BY 4.0 (DPI) | 174 | 174 |


## 额外匿名可用性验证

对以下精确 HF ID 获取元数据，均 `private=false, gated=false`：LocalLLaMA/typed-decisions、jaredpalmer/kev-suites、nvidia/HelpSteer2、AmazonScience/massive、tasksource/wice、tals/vitaminc、mteb/summeval。Kev 的当前 HF card 无统一 license 字段，必须沿子数据上溯；不是可以统一当 Apache-2.0。

对 Bobcat 任一源只下载一个固定原始文件的写法（本次未执行）：

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="AmazonScience/massive",
    repo_type="dataset",
    revision="ed58ac423a2f4121720918bf5301577edce4ffd3",
    filename="en-US/train/0000.parquet",
)
```

这些文件是原始任务数据；转换成 `state + questions + gold` 的方法在 dc13_skills.py，不应直接把 parquet 原字段当统一 schema。
