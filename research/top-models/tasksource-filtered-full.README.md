# filtered-full

`filtered-full` contains **1,305,478 training decisions** retained from the `full/train`
split of `tasksource/tasksource-jev-typed-decisions` at revision
[`d2ab1d12be4463fb7ac1f877ed2193b3bb59642a`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/tree/d2ab1d12be4463fb7ac1f877ed2193b3bb59642a).
The cleanup ran October 3–4, 2026. This configuration is a filtered historical
snapshot; it does not include rows added to the repository after that revision.

```python
from datasets import load_dataset

train = load_dataset(
    "tasksource/tasksource-jev-typed-decisions", "filtered-full", split="train"
)
```

Only `train` is supplied. Existing `default` and `full` data, configurations, and
validation/test splits are unchanged.

## Selection counts

| Stage | Decisions |
|---|---:|
| Frozen source `full/train` | 2,500,000 |
| Excluded before LLM review | 1,171,377 |
| Eligible for LLM review | 1,328,623 |
| Retained as keep | 1,208,448 |
| Retained as uncertain | 97,030 |
| Rejected after review and retention checks | 23,138 |
| Unresolved API errors, excluded | 7 |
| **Published training decisions** | **1,305,478** |

The run has finished, but seven decisions never obtained a successful review.
They are excluded; this is not a claim of complete LLM review coverage.

## Preprocessing

The selection reflects a specific training-data policy as well as label quality.
It is not a pure GLM label-error filter over all 2.5 million rows.

| Pre-review exclusion reason | Decisions |
|---|---:|
| Source `license_use` was not `commercial` | 1,025,115 |
| Explicit benchmark-source holdout | 122,480 |
| Positional references in options | 8,108 |
| Protected exact text match | 8,156 |
| Protected near text match | 1,526 |
| Protected contained text match | 3,692 |
| Group exceeded the prompt token budget | 1,128 |
| Conflicting targets for an identical decision | 1,172 |

These checks operated on source groups. Token filtering used the training prompt
and tokenizer with an 8,192-token budget, including a 64-token reserve. Text
overlap checks protected the frozen evaluation panels and Tasksource heldouts
used in the training workflow. The matching thresholds and explicit source
holdouts are recorded in [filtering-report.json](filtering-report.json).
Lexical checks are approximate and do not establish universal benchmark
decontamination. Preserved license fields describe the upstream metadata; this
configuration does not introduce a new license or replace source terms.

## GLM review

The reviewer was **GLM-5.3-Flash**, model identifier `glm_5_3_flash`, accessed
through llm-api with 768 workers. Review considered the input, question, ordered
options, and original target distribution. It sought demonstrable defects rather
than replacing source labels with a model's preferred answer.

An initial reviewer and a reasoning-enabled verifier both had to demonstrate an
obvious defect and recommend rejection. Uncertain cases were retained. Final
conservative retention checks also restored 1,221 neutral/non-entailment
disagreements, 89 emotion-classification disagreements, and one ambiguous
title-versus-procedural-advice case. These 1,311 cases are included in the final
retained counts above.

Identical rows already reviewed during the earlier `default`-configuration
cleanup reused their review evidence after matching the complete serialized
review payload. Remaining rows were reviewed separately. Inherited reviews keep
their original prompt provenance; the later full review added a clarification
that the state is the input. This is an LLM-assisted cleanup, not human-certified
ground truth. Model disagreement alone is not proof of an incorrect label.

## Row preservation and export validation

The exported Parquet files select original rows by upstream `id` from the frozen
source files. They preserve all **13 columns present in that historical revision**,
including text, option order, `kind`, IDs, soft targets, and license metadata.
Newer repository revisions added an `example_id` column; it is not synthesized
for this historical configuration.

The intermediate training representation used numeric option keys and mapped
`score` requests to its internal choice interface. The published dataset restores
the original upstream schema and values. For 7082 retained rows
whose option wording had been normalized for training, export also restores the
original option strings. No labels or probabilities are rewritten.

Preprocessing retained or removed groups, while the final LLM selection acts on
individual decisions. Consequently, some `group_id`s contain fewer decisions
than in the original snapshot. Original row order is retained; do not assume
every surviving group is contiguous. Group by both `group_id` and `state` when
reconstructing related requests.

Validation checks the frozen input hashes, unique IDs, every retained row against
the cleanup selection, and every exported value after reading all Parquet shards
back. Counts, source distributions, selection fingerprints, and shard checksums
are available in [filtering-report.json](filtering-report.json).
