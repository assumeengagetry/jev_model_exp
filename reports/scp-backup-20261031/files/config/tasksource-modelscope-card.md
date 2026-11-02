
# tasksource-jev-typed-decisions

**2.5 million typed decisions (choices, ratings and probabilities) from 670 sources.**

## Why use it

- **Real supervision.** Labels, ratings, and annotator votes come from
  established datasets, not a teacher model. Every row names its `source`.
- **Breadth.** Over 300 dataset families: NLI and reasoning, QA and
  commonsense, sentiment, intent and topic, toxicity and safety, preference
  pairs, fact checking, entity tagging, and dozens of languages. GLUE,
  SuperGLUE, HellaSwag, PIQA, ScienceQA, Banking77, CoNLL-2003, MasakhaNEWS,
  HelpSteer, ChaosNLI, and many more, with no task allowed to dominate.
- **Three decision types in one schema.** `choice` (pick one option), `score`
  (an ordered scale), and `noul` (the probability that the answer to a yes/no
  question is yes). `noul` holds only probabilities: entailment likelihoods, and the
  share of annotators who answered yes. Mean ratings and similarity are `score`
  distributions whose expected level is the mean (3.4 on 1–5 puts 0.6 on 3 and 0.4
  on 4). Ordinal label sets appear as both `choice` and `score`, split
  deterministically per row, so a model learns both requests for the same scale.
  Soft targets are kept wherever the source has mean ratings or votes from at
  least five annotators per item (vote shares from fewer are too noisy): STS,
  ChaosNLI, civil_comments, Measuring Hate Speech, WouldYouRather, ProtoQA,
  LeWiDi and more. They make up the graded share.
- **Built so position, repeated eval data, and question choice give nothing away.**
  - Multiple-choice options are shuffled per row, so the answer's position carries no signal.
  - Validation and test rows whose content appears in train are removed.
  - Derived questions are chosen without looking at their answers.
  - Annotations were reviewed task by task. Inverted, unanswerable, and garbled labels were fixed or dropped.
- **Multi-question states.** Related decisions share a `group_id` and can be
  asked together. Packed states test reasoning over several items at once, and
  [procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions)
  adds exact counting, arithmetic, retrieval, state tracking, routing among
  up to 60 options, and exact posteriors when a policy applies to a requester
  whose role is uncertain.

## Quick start

Coding agent? Read [AGENTS.md](AGENTS.md): row semantics, rebuilding multi-question requests from `group_id`, filtering, and evaluation caveats.

```python
from datasets import load_dataset

ds = load_dataset("tasksource/tasksource-jev-typed-decisions")          # steered 1M-row mix
# full = load_dataset("tasksource/tasksource-jev-typed-decisions", "full")  # every row of the build
row = ds["train"][0]
print(row["state"], row["question"], row["options"], row["target"])
```

```json
{"state": "My body cast a shadow over the grass. What was the cause of this?",
 "question": "Choose the criterion that best answers the question.",
 "kind": "choice", "options": ["The sun was rising.", "The grass was cut."],
 "target": [1.0, 0.0], "source": "super_glue/copa"}
```

## Configs

- `default`: a steered mix of about 1M train rows. Sources are first gated on label correctness, then weighted by how interesting they are and how close they sit to the zone of proximal development (judged by decision models). Two-option tasks get fewer rows, and procedural generators get 12%. No row is repeated. Validation and test are the full eval splits, restricted to the mixed sources. Buckets and shares are in [`jev_mixes.py`](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/jev_mixes.py) and per-source scores in `jev_source_scores.csv`.
- `full`: every row that passed the build, with per-source caps (about 2.5M train rows).

- `filtered-full`: 1,305,478 retained `train` decisions from the frozen `full` revision `d2ab1d12be4463fb7ac1f877ed2193b3bb59642a`. Applies commercial-license metadata gating, benchmark-overlap/structural filters, and conservative GLM-5.3-Flash review. Uncertain cases are retained; original row values and soft targets are preserved. No validation or test split is added. This is a filtered historical snapshot and excludes additions made after that revision. See [filtering methods and counts](filtered-full/README.md).

## Format

| field | meaning |
|---|---|
| `state` | The text to decide about |
| `question` | What to decide |
| `kind` | `choice`, `score`, or `noul` |
| `options` | Runtime criteria; empty for `noul` |
| `target` | Distribution over `options`, or `[p]` for `noul` |
| `id`, `group_id`, `question_id` | Link decisions over the same source example |
| `example_id` | Stable hash of the source example's input and gold; the same across releases unless the label changes |
| `source`, `split`, `variant` | Originating task, original split, and recast variant |
| `license`, `license_use` | The source's license(s), and `commercial`, `non-commercial` or `unspecified` (see below) |

Splits: train (about 1M rows in `default`, 2.5M in `full`), 15,000 validation (`dev` in `split`), and 15,000 test,
following each source's own train/dev/test splits where it has them.

## How it is built

- **Canonical recasts.** Each Tasksource task is converted deterministically.
  - Criteria are the source's own label names and answer options.
  - Multiple-choice rows keep every option in a per-row order.
  - A final "all/none of the above" reads "all/none of the other options".
  - Options that cite other options by letter or number keep their order.
  - The question is the task's own when its inputs alone do not say what to predict
    ("What stance does the tweet take on feminism?"), and a generic instruction otherwise.
    Label-verification and packed questions carry it too.
- **Variants.** Low-frequency, deterministic variants cover label verification as `noul`, criterion order, and instruction wording.
- **Packing.** Up to 10% of each classification task's examples are packed, two to four at a time, into `packed_derived` states. Their questions (an item's label, agreement, existence, counts) follow exactly from the gold labels.
- **Mixing (`full`).** Formats get fixed shares of the train rows (47% classification, 30% multiple choice, 3% token labeling, 10% graded (soft-label sources), 10% procedural). Within a format, dataset families get equal shares, scaled by [hand-set weights](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/weights.py) (more for adversarial NLI, long documents and preference pairs; less for templated probes), times [audit weights](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/audit_weights.py) from a per-task check of Jev on 200 examples: ×1.5 for hard tasks whose gold is right by construction (synthetic logic, theory of mind, spatial reasoning), ×0.5 for near-solved tasks and for hard tasks whose gold is a judgment call (ratings, preferences, crowd sentiment). Sources with many options get slightly more room. Related questions are kept together.
- **Mixing (`default`).** About 1M rows drawn from `full`, never repeating a row. Buckets of related sources get set shares ([jev_mixes.py](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/jev_mixes.py): 15% logic, 10% NLI, 10% knowledge QA, 9% long documents and fact-checking, 8% intent and routing, 12% procedural, ...). Inside a bucket, sources get rows by score × √size. The [score](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/jev_source_scores.csv) multiplies:
  - correctness: sources that failed label review get 0;
  - the zone of proximal development: how much probability the decision models (Jev, Liquid D1) give the gold answer. Near-solved sources and sources the models miss outright both get less;
  - interest and cleanliness, from Jev yes/no checks for transferable skills, trivial examples and malformed rows;
  - ×0.6 for two-option tasks (about a third of the rows);
  - ×2 for sources picked by reading them.
- **Order and coverage.**
  - The first 1,000 train rows are interleaved to show variety in the Dataset Viewer; the rest is shuffled. Questions of a group stay adjacent throughout.
  - Evaluation benchmarks (BIG-bench, MMLU, BLiMP, MATH test, ...) are left out so they stay clean for evaluation.
- **Sources.** [sources.yaml](sources.yaml) lists every source with its rows, the Hub dataset and revision it was loaded from, the original dataset behind each tasksource copy, and its licenses.
- **Audit trail.** The [source mix](release-audit.json), [failed source list](failed-tasks.json), and [build manifest](build-manifest.json) ship with the data.
- **Reproducible.** The [build runbook](https://github.com/sileod/tasksource/blob/main/docs/jev/README.md) rebuilds the release from [Tasksource](https://github.com/sileod/tasksource)'s [task catalog](https://github.com/sileod/tasksource/blob/main/tasks.md).

## License and scope

Tasksource harmonizes datasets from many publishers; their original licenses
and terms still apply, hence `license: other`.

Each row carries its source's license, to help filter:

```python
ds = ds.filter(lambda use: use == "commercial", input_columns="license_use")
```

- `license` lists the `license` of the Hub dataset card the source was loaded from,
  and of the original dataset behind a tasksource copy. It also lists licenses recorded
  by the [Data Provenance Initiative](https://www.dataprovenance.org/), marked `(DPI)`.
- `license_use` takes the most restrictive of those: `non-commercial` if any is
  non-commercial or academic-only, `commercial` if one allows commercial use (share-alike
  and copyleft included), and `unspecified` otherwise. That covers missing licenses and
  `other`, bare `cc`, and no-derivatives licenses.
- [sources.yaml](sources.yaml) records each card and DPI license per source.

This is a best-effort aid, not legal advice. Licenses on cards can be wrong or
incomplete, and a source's terms may differ from its card's. Check the
original terms before relying on them. This recast is independent of
TypeSafe and OpenJev.

## Citation

```bibtex
@inproceedings{sileo-2024-tasksource,
  title = {tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework},
  author = {Sileo, Damien},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  year = {2024},
  pages = {15655--15684},
  url = {https://aclanthology.org/2024.lrec-main.1361/}
}
```


## Incremental filtered WebInstruct addition

Filtered `tasksource/webinstruct` configs `mc` and `binary`, pinned at `05e8b528d68c628c1af682d47bdb53f85535876a`, are appended as standalone shards to both `default` and `full`. The original train/test split assignment is preserved; validation is unchanged. All choices of included rows are retained, with canonical Jev option permutation and one-hot targets; 47 MC rows with duplicate option text are omitted under the existing choice contract. binary questions use explicit no/false and yes/true choices. No repairs or derived variants are included. Existing shards and mixture sampling are unchanged. See `webinstruct-addition.json` for exact counts, transformations and provenance.


## Reusable selection evidence

The [exclusion evidence resource](quality/filtered-full-v1/README.md) records historical filtered-full selection membership separately from licensing and benchmark policies. Remaining candidates are unattributed exclusions, not verified errors. Canonical decision keys support evidence reuse across recasts.

## Vision config

Load `load_dataset("tasksource/tasksource-jev-typed-decisions", "vision")` for the multimodal pilot (66,085 train, 475 test, 972 validation).
It covers ai2d, aokvqa, bapps/preference, clevr/color, clevr/count, clevr/material, clevr/shape, clevr/size, clevr/yesno, coco/panoptic-region, doclaynet/region, figureqa, intergps, lvis/region, mapqa/yesno, nlvr2, rico-widget/element, rico-widget/grid7, scienceqa-img, spair71k/grid7, superclevr/color, superclevr/count, superclevr/material, superclevr/shape, superclevr/size, superclevr/yesno, tallyqa/count, tqa, visual7w, vsr/yesno, with genuine source labels.
`state`, `question`, `kind`, `options`, and `target` retain their existing decision semantics.
`images` is an ordered sequence of Hugging Face Image features. Source bytes are preserved
unless `metadata.augmentation` records rendering (SoM, region highlighting or source keypoint marks); images decode on access.
Feed the images alongside `state` through your model's image adapter.
NLVR2 retains its two images in source order. Answer options remain text.

`metadata` is a JSON string preserving available source identifiers, pinned provenance,
and separate Hub-card/DPI license evidence. `license` and `license_use` are also available
as columns; missing or conflicting evidence is retained rather than replaced by an inferred license.
See [vision/sources.yaml](vision/sources.yaml), [vision/release-audit.json](vision/release-audit.json),
[vision/build-manifest.json](vision/build-manifest.json), and [vision/quality-audit.json](vision/quality-audit.json)
when the materialized release was audited.

Sampling: full-source uniform reservoirs, shared per-family export caps, with per-task checkpoint provenance. Sampling caps and checkpoint provenance are recorded in the manifests.
Visual task views share one export budget per dataset family, including CLEVR, TallyQA, and RICO;
individual task IDs remain in `source`, and family totals appear in `release-audit.json`.
Only native labeled splits are used. Source action identities group related GUI decisions;
other decisions sharing the same ordered image set and text state share `group_id`.
`metadata.image_group_id` links source image groups across questions and augmentation variants.
Evaluation rows sharing any image with training are excluded.
This initial config contains direct decisions; text packing and derived variants are disabled.
Typed request renderers preserve images; model-specific image transport remains the caller's responsibility.

The 2026-10-09 selective update adds at most 1,000 train rows per new family while preserving the existing rows and caps. Mind2Web remains excluded. HTML-only browser tasks are not part of this config.

### Vision prompt and license correction (2026-10-09)

Known Cauldron answer-style suffixes were removed from 18,353 training rows;
complete original questions remain in metadata. Twelve identical MapQA requests
with identical gold answers were deduplicated. Images and retained gold targets
are unchanged. There are **66,085 train, 972 validation and 475 test** decisions.

Original dataset terms for [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/),
[MapQA](https://github.com/OSU-slatelab/MapQA/blob/main/license), and
[TQA](https://registry.opendata.aws/allenai-tqa/) fill missing Hub-card evidence.
Training license-use counts are **26,987 commercial, 7,000 non-commercial, and
32,098 unspecified**. This remains a **mixed-license research pilot**, not a
commercial-cleared dataset. Unspecified does not grant permission. NLVR2 photo
rights and other unresolved sources remain unresolved; repository software
licenses do not establish image rights. LVIS/COCO classification conservatively
reflects the NC images included in their source mirrors.

Filter `license_use` when loading; new Tasksource vision builds also accept
`--license-use commercial` to exclude unresolved/NC sources before loading them.
See [repair-audit.json](vision/repair-audit.json) and the reproducible
[repair script](vision/repair.py); both record the pinned parent release.

### Component-level license review (2026-10-09)

This review supersedes earlier license-use counts. All sources remain included.
Training: **26,987 commercial, 13,000 non-commercial,
26,098 unresolved complete-row coverage**. Counts describe scoped
license evidence, not legal clearance of each image.

**FigureQA's official archive contains Microsoft Research Open Data License terms:
non-commercial research/testing only; dataset redistribution and standalone hosting
are prohibited.** Its generator's MIT license covers code. The retained FigureQA
rows are explicitly flagged `redistribution: prohibited`; their inclusion here does
not grant redistribution rights. BAPPS training patches inherit Adobe / Adobe-MIT
research image terms; its software BSD license does not clear the images.

NLVR2 annotations and VSR annotation cards specify CC BY 4.0. TallyQA and A-OKVQA
repositories specify Apache 2.0; InterGPS's data/code repository specifies MIT.
These terms do not clear third-party images. COCO per-image grants, NLVR2 photo
rights, imported TallyQA QAs and SPair PASCAL/Flickr terms still need row-level
resolution. No explicit Visual7W annotation grant was found in checked original
project/toolkit documentation. These are documented coverage gaps, not a claim
that every upstream component has no license.

Every reviewed row has `metadata.licenses.license_review` with separate annotation
and image terms, evidence URLs, review status and unresolved reasons. See
[license-review.json](vision/license-review.json), [sources.yaml](vision/sources.yaml)
and [the reproducible update script](vision/update_vision_license_metadata.py).
Images, questions, choices, targets, IDs and source splits are unchanged.

### English task correction and benchmark exclusion (2026-10-09)

CLadder is evaluation-only and removed from `default`, `full` and `filtered-full`.
Missing-item decisions now separate the list and query and explicitly ask about
membership; the previous query was at the end of the state. WikiHow order now
asks which step comes first. QuaRel uses native answer texts as MC options with
permuted, correctly remapped golds, replacing fixed A/B classes. Its old
classification-only packs and obsolete generic instruction-paraphrase variants
were removed. Source examples and splits remain; unrelated tasks and vision
are unchanged. Current counts are in the config metadata. See
[rebuild audit](english-task-rebuild.json) and [reproducible script](english-task-rebuild.py).


## HTML/browser text additions

WebLINX action/DOM-element and WebSRC yes/no/element tasks retain their source partitions. WebLINX named test subsets are combined into the hosted test split, with native split names in row provenance. `mind2web/action` and `mind2web/dom-element` use only the original public training trajectories. Tasksource-derived train/validation/test holdouts keep whole trajectories and identical DOM snapshots together (80/10/10 deterministic group hashing). These are not the official benchmark test partitions. The same partition is used for both annotations. Original annotation/action IDs and page hashes are recorded in [row provenance](additions/167c5ad6fb79/rows.json). Only past actions enter context; four-way choices use native DOM candidates. Direct decisions are appended to default/full; existing shards, vision and filtered-full are unchanged. Alignment with the Decision Index private evaluation rows has not been independently verified.
