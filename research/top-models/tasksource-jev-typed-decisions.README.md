---
pretty_name: tasksource-jev-typed-decisions
language:
- en
- multilingual
license: other
task_categories:
- zero-shot-classification
- text-classification
- question-answering
- token-classification
tags:
- tasksource
- jev
- system-one
- runtime-defined-decisions
- decision-models
- multiple-choice
- decision-model
- extraction
- information-extraction
- extractive-qa
- span-extraction
- named-entity-recognition
- structured-extraction
- clairic
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
- config_name: extraction
  data_files:
  - split: train
    path: extraction/train-*
  - split: validation
    path: extraction/validation-*
  - split: test
    path: extraction/test-*
- config_name: filtered-full
  data_files:
  - split: train
    path: filtered-full/train-*.parquet
- config_name: full
  data_files:
  - split: train
    path: full/train-*
  - split: validation
    path: full/validation-*
  - split: test
    path: full/test-*
- config_name: vision
  data_files:
  - split: train
    path: vision/train-*
  - split: test
    path: vision/test-*
  - split: validation
    path: vision/validation-*
dataset_info:
- config_name: default
  features:
  - name: state
    dtype: string
  - name: kind
    dtype: string
  - name: id
    dtype: string
  - name: options
    list: string
  - name: target
    list: float64
  - name: question
    dtype: string
  - name: source
    dtype: string
  - name: variant
    dtype: string
  - name: split
    dtype: string
  - name: source_index
    dtype: string
  - name: group_id
    dtype: string
  - name: question_id
    dtype: string
  - name: example_id
    dtype: string
  - name: license
    dtype: string
  - name: license_use
    dtype: string
  splits:
  - name: train
    num_bytes: 1534153675
    num_examples: 999832
  - name: validation
    num_bytes: 23419061
    num_examples: 15000
  - name: test
    num_bytes: 21143086
    num_examples: 15000
  download_size: 694326150
  dataset_size: 1578715822
- config_name: extraction
  features:
  - name: id
    dtype: string
  - name: source
    dtype: string
  - name: split
    dtype: string
  - name: state
    dtype: string
  - name: fields
    list:
    - name: name
      dtype: string
    - name: type
      dtype: string
    - name: instruction
      dtype: string
  - name: answers
    list:
    - name: 'null'
      dtype: bool
    - name: values
      list:
      - name: text
        dtype: string
      - name: start
        dtype: int32
      - name: end
        dtype: int32
  - name: metadata
    dtype: string
  - name: license
    dtype: string
  - name: license_use
    dtype: string
  splits:
  - name: train
    num_bytes: 1195502112
    num_examples: 584912
  - name: validation
    num_bytes: 96145060
    num_examples: 45705
  - name: test
    num_bytes: 91948894
    num_examples: 46612
  download_size: 938851089
  dataset_size: 1383596066
- config_name: filtered-full
  features:
  - name: state
    dtype: string
  - name: kind
    dtype: string
  - name: id
    dtype: string
  - name: options
    list: string
  - name: target
    list: float64
  - name: question
    dtype: string
  - name: source
    dtype: string
  - name: variant
    dtype: string
  - name: split
    dtype: string
  - name: group_id
    dtype: string
  - name: question_id
    dtype: string
  - name: license
    dtype: string
  - name: license_use
    dtype: string
  splits:
  - name: train
    num_bytes: 1633306939
    num_examples: 1305478
  download_size: 335270018
  dataset_size: 1633306939
- config_name: full
  features:
  - name: state
    dtype: string
  - name: kind
    dtype: string
  - name: id
    dtype: string
  - name: options
    list: string
  - name: target
    list: float64
  - name: question
    dtype: string
  - name: source
    dtype: string
  - name: variant
    dtype: string
  - name: split
    dtype: string
  - name: source_index
    dtype: string
  - name: group_id
    dtype: string
  - name: question_id
    dtype: string
  - name: example_id
    dtype: string
  - name: license
    dtype: string
  - name: license_use
    dtype: string
  splits:
  - name: train
    num_bytes: 3871839670
    num_examples: 2550012
  - name: validation
    num_bytes: 23419061
    num_examples: 15000
  - name: test
    num_bytes: 21143086
    num_examples: 15000
  download_size: 1950002463
  dataset_size: 3916401817
- config_name: vision
  features:
  - name: images
    list: image
  - name: metadata
    dtype: string
  - name: id
    dtype: string
  - name: kind
    dtype: string
  - name: options
    list: string
  - name: target
    list: float64
  - name: state
    dtype: string
  - name: question
    dtype: string
  - name: source
    dtype: string
  - name: variant
    dtype: string
  - name: split
    dtype: string
  - name: group_id
    dtype: string
  - name: question_id
    dtype: string
  - name: example_id
    dtype: string
  - name: license
    dtype: string
  - name: license_use
    dtype: string
  splits:
  - name: train
    num_bytes: 236883226428
    num_examples: 722314
  - name: test
    num_bytes: 146575133
    num_examples: 475
  - name: validation
    num_bytes: 274444703
    num_examples: 972
  download_size: 140122509371
  dataset_size: 237304246264
task_ids:
- extractive-qa
- named-entity-recognition
---

# tasksource-jev-typed-decisions

Typed decisions from hundreds of curated [Tasksource](https://github.com/sileod/tasksource) tasks: classification, multiple choice, ordered ratings, probabilities and visual reasoning. Targets come from source annotations or deterministic transformations of them. Native annotation noise and subjective judgments remain.

## Load

```python
from datasets import load_dataset

text = load_dataset("tasksource/tasksource-jev-typed-decisions")
vision = load_dataset("tasksource/tasksource-jev-typed-decisions", "vision")
extraction = load_dataset("tasksource/tasksource-jev-typed-decisions", "extraction")
row = text["train"][0]
print(row["state"], row["question"], row["options"], row["target"])
```

| Config | Contents |
|---|---|
| `default` | Steered text training mix, plus reviewed incremental additions; validation and test included. |
| `full` | Larger text pool with per-source caps, including incremental additions; validation and test included. |
| `filtered-full` | Filtered, training-only historical snapshot. Later additions are not included. See [selection policy and counts](filtered-full/README.md). |
| `vision` | Multimodal decisions with ordered images and text choices. See [visual sources, sampling and provenance](vision/README.md). |
| `extraction` | A text and several typed fields (questions, entity types) answered with character spans or null, Clairic-style. See [extraction format and sources](extraction/README.md). |

Exact split sizes are in the Dataset Viewer metadata. Task coverage and counts are in [sources.yaml](sources.yaml) and [vision/sources.yaml](vision/sources.yaml).

## Row format

| Field | Meaning |
|---|---|
| `state` | Text to reason about. |
| `question` | The decision to make. |
| `kind` | `choice`, `score` or `noul`. |
| `options` | Criteria in their presented order; empty for `noul`. |
| `target` | Distribution aligned with `options`, or `[p_true]` for `noul`. |
| `id`, `group_id`, `question_id` | Decision identity and grouping of related questions. |
| `example_id` | Content-derived example identity used by the build. |
| `source_index` | Original row of the example in its source dataset, as `split:row` (the split before any re-splitting); empty for packed and generated rows. |
| `source`, `split`, `variant` | Task identity, split annotation and recast variant. |
| `license`, `license_use` | Recorded source terms and their usage classification. |
| `images` | **Vision only:** ordered Hugging Face image features, decoded on access. |
| `metadata` | **Vision only:** JSON-string source identity, geometry, augmentation and license evidence. |

```json
{"state": "My body cast a shadow over the grass. What was the cause of this?",
 "question": "Choose the criterion that best answers the question.",
 "kind": "choice", "options": ["The sun was rising.", "The grass was cut."],
 "target": [1.0, 0.0], "source": "super_glue/copa"}
```

`choice` selects a criterion; `score` uses an ordered scale. Soft targets preserve annotator disagreement or interpolate numerical ratings between neighboring scale anchors. `noul` represents the probability that a proposition is true. Scale meaning is determined by the question and criteria.

Rows sharing `group_id` can be assembled into a multi-question request. Vision images stay separate from text and must be passed through the model's image adapter; answer options remain text. See [AGENTS.md](AGENTS.md) for request assembly and row semantics.

## Construction and evaluation

Tasksource standardizes native labels and choices, adds task-specific questions where needed, and applies deterministic option permutations and low-frequency text variants. Some text rows pack related decisions over a shared state. A decision's majority label keeps at most three rows per row of its other labels (graded answers count as yes above 0.5; a question with a single label is dropped), so skewed binary sources such as answer selection or toxicity no longer reward always answering the common label; procedural questions keep each label under 60%. The visual config contains direct decisions and shares sampling budgets across related task views.

Native source partitions are retained where available; derived group-based holdouts and overlap exclusions are documented per source. These checks do not establish absence of all cross-dataset or benchmark contamination. For example, Mind2Web text tasks use derived holdouts from public training trajectories, not official benchmark tests; alignment with private Decision Index evaluation rows is unverified. Mind2Web is excluded from the visual config.

Source and recast reviews are coarse checks, not corpus-wide correctness guarantees. [English examples](https://github.com/sileod/tasksource/blob/main/docs/examples.md) illustrate the exported and canonical representations.

## Licenses

This collection has **mixed source terms**, hence `license: other`. Original annotation and image terms still apply. `license_use` is a best-effort classification: `commercial`, `non-commercial` or `unspecified`; it is not legal clearance. Missing or scoped evidence is retained explicitly.

```python
text = text.filter(lambda use: use == "commercial", input_columns="license_use")
```

Visual annotation licenses may not cover third-party images. The visual license review records unresolved image rights and explicit restrictions, including FigureQA's prohibition on redistribution. See [component-level evidence](vision/license-review.json) and the [vision documentation](vision/README.md) before reuse.

## Documentation and provenance

- [Text sources and revisions](sources.yaml), [release audit](release-audit.json), [build manifest](build-manifest.json) and [failed sources](failed-tasks.json).
- [Vision documentation](vision/README.md) and [filtered-full selection evidence](quality/filtered-full-v1/README.md).
- [Build runbook](https://github.com/sileod/tasksource/blob/main/docs/jev/README.md), [base build recipe](BUILD.md) and [task catalogs](https://github.com/sileod/tasksource#task-catalogs).
- [Release history](CHANGELOG.md). Addition manifests and sampled reviews retain their original snapshot context.

## Citation

If you use tasksource, its standardized task annotations, or datasets derived from them in published research, please cite the tasksource paper (Sileo, LREC-COLING 2024). The procedural data is generated with Reasoning Core; please also cite the Reasoning Core paper (Sileo et al., 2026).

```bibtex
@inproceedings{sileo-2024-tasksource,
    title = "tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework",
    author = "Sileo, Damien",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1361",
    doi = "10.63317/2h4pkr685mkq",
    pages = "15655--15684",
}

@article{sileo2026reasoning,
  title={Reasoning Core: Designing Broad Procedural Data for Completion-Supervised Reasoning Training},
  author={Sileo, Damien and Lacombe, Valentin and Kachler, Dimitri},
  journal={arXiv preprint arXiv:2608.05148},
  year={2026},
  url={https://arxiv.org/abs/2608.05148}
}
```
