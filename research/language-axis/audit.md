# Language Understanding axis audit — 2026-10-27

The live index snapshot lists **six category definitions**, but only **five active scored categories**; Games is not independently scored and carries `folded_into: knowledge`.

## Actual Language Understanding benchmarks (9)

ContractNLI, ANLI, HellaSwag, ACOS, FinEntity, iSarcasmEval, VAST, NLI4CT, RAGTruth.

The category is heterogeneous: entailment, multiple-choice completion, aspect sentiment, entity sentiment, sarcasm, stance, clinical NLI, and groundedness. Merely training NLI does not reproduce the full category.

Snapshot leader is **mammoth-1 (27B)** with public chance-corrected category skill **0.7611**, and combined category score **0.6709**. Those are different metrics. A local public evaluation should compare with public skill, not with a private/combined headline. Source: `research/2026-10-27/raw/leaderboard-index.json`, snapshot generated 2026-10-26T15:30:58Z, Decision Index 0.4.1.

## Verified official train counts and direct files

| Benchmark | Official train data | Source | Notes |
|---|---:|---|---|
| ANLI | 162,865 pairs = 16,946 + 45,460 + 100,459 | https://huggingface.co/datasets/facebook/anli | train_r1/r2/r3; 3-way NLI; CC BY-NC 4.0 |
| HellaSwag | 39,905 questions | https://huggingface.co/datasets/Rowan/hellaswag | train; 4 choices; leaderboard uses validation |
| ACOS laptop | 2,934 reviews | https://raw.githubusercontent.com/NUSTM/ACOS/45d179a3dcc6a3dedd848d81b16f2552454805fe/data/Laptop-ACOS/laptop_quad_train.tsv | Verified file downloaded locally |
| ACOS restaurant | 1,530 reviews | https://raw.githubusercontent.com/NUSTM/ACOS/45d179a3dcc6a3dedd848d81b16f2552454805fe/data/Restaurant-ACOS/rest16_quad_train.tsv | Verified file downloaded locally |
| iSarcasmEval English | 3,468 tweets | https://raw.githubusercontent.com/iabufarha/iSarcasmEval/dfc708b53bde1bb571abfb5692f63231c2232195/train/train.En.csv | Verified csv records; board headline is English Task A F1 |
| VAST | 13,477 records | https://raw.githubusercontent.com/emilyallaway/zero-shot-stance/e7c4775182b184730f350995f8260579c9e066fe/data/VAST/vast_train.csv | Verified csv records; topic-conditioned 3-way stance |
| RAGTruth | 15,090 responses | https://github.com/ParticleMedia/RAGTruth | response.jsonl split=train; join source_info.jsonl on source_id; binary label is bool(labels); official test 2,700 |
| ContractNLI | 423 contracts × 17 statements = 7,191 decisions | https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip | Use train only; evaluate full contracts, not gold evidence-selected snippets |
| NLI4CT | 1,700 training statements | https://github.com/ai-systems/Task-2-SemEval-2024 | training_data.zip; preserve complete trial sections |
| FinEntity | **No official train split** | https://github.com/yixuantt/FinEntity | Board uses whole 979-document unsplit source (one span quarantined). Do not train this corpus. Use separate financial entity-sentiment data, with overlap filtering. |

All counts above verified from dataset API or exact original files in this directory. ContractNLI splits contain 423/61/123 documents, with 17 fixed hypotheses; NLI4CT contains 1,700 train and 200 dev statements.

## Cautions relevant to choosing a first axis

Language is usable but nine components require different input formatting and output labels. FinEntity has no disjoint official train source. ContractNLI and NLI4CT are long-document tasks, so small-model context/readout handling can dominate data gains. Public board test/dev examples must remain held out, and aggregate Tasksource generated train splits must be audited against original benchmark splits before treating them as safe.

A gap to the category leader alone does not identify whether data or model is wrong; use increasing train-data fractions with the same model and head, multiple seeds, and a clean per-benchmark validation breakdown. Changing one factor at a time supplies the relevant evidence.
