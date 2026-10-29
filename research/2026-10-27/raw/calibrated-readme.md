---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: source
    dtype: string
  - name: source_license
    dtype: string
  - name: source_id
    dtype: string
  - name: type
    dtype: string
  - name: instructions
    dtype: string
  - name: state
    dtype: string
  - name: options
    list: string
  - name: option_descriptions
    list: string
  - name: candidate
    dtype: string
  - name: gold
    dtype: string
  - name: gold_index
    dtype: int64
  - name: display_perm
    list: int64
  - name: teacher_option_logprobs
    list: float64
  - name: teacher_temperature
    dtype: float64
  - name: teacher_shift
    dtype: float64
  - name: teacher_probs
    list: float64
  - name: teacher_letter_mass
    dtype: float64
  - name: human_probs
    list: float64
  - name: soft_label
    list: float64
  - name: soft_label_source
    dtype: string
  - name: corrected_soft_label
    list: float64
  - name: soft_label_corrected
    dtype: bool
  splits:
  - name: train
    num_bytes: 12924237
    num_examples: 16995
  - name: validation
    num_bytes: 1379863
    num_examples: 1799
  - name: test
    num_bytes: 1377093
    num_examples: 1797
  - name: heldout
    num_bytes: 1915788
    num_examples: 1000
  download_size: 11013876
  dataset_size: 17596981
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
  - split: heldout
    path: data/heldout-*
language:
- en
license: cc-by-4.0
task_categories:
- text-classification
pretty_name: Calibrated Typed Decisions
tags:
- calibration
- soft-labels
- distillation
- typed-decisions
---

# calibrated-typed-decisions (v2.1)

Typed decisions (multiple choice and yes/no) over short English texts. Every row has a human
gold label and a calibrated soft label, plus the raw ingredients to rebuild the soft label
your own way. Earlier versions stay available: `load_dataset("m-lagnajit/calibrated-typed-decisions", revision="v1.0")` and `revision="v2.0"`.

## What changed in v2.1

- **New column `corrected_soft_label`**: identical to `soft_label`, except on rows where the calibrated
  teacher's top pick disagreed with the human answer. There, the teacher's value on its top pick and its
  value on the right answer are swapped: the teacher's numbers are kept, but the right answer gets the
  largest one. If the right answer is already tied for the teacher's largest value, the row is
  unchanged. Rater-labelled GoEmotions rows are unchanged; held-out rows stay empty.
- **New column `soft_label_corrected`**: `true` on exactly the rows where that swap changed the values.
- Nothing else changed. `soft_label`, `teacher_probs` and every other value are exactly as in v2.0.

How often the teacher's top pick was wrong (train, validation and test):

| source | type | rows changed | teacher-labelled rows | share |
|---|---|---|---|---|
| amazon_massive_intent | choice | 220 | 2516 | 8.7% |
| amazon_massive_intent | noul | 108 | 1076 | 10.0% |
| amazon_polarity | choice | 93 | 2496 | 3.7% |
| amazon_polarity | noul | 36 | 1104 | 3.3% |
| banking77 | choice | 284 | 2484 | 11.4% |
| banking77 | noul | 146 | 1116 | 13.1% |
| clinc_oos | choice | 123 | 2461 | 5.0% |
| clinc_oos | noul | 104 | 1139 | 9.1% |
| scicite | choice | 408 | 1832 | 22.3% |
| scicite | noul | 205 | 768 | 26.7% |

## What changed from v1

- **GoEmotions soft labels come from the human raters**, not the teacher. On 600 GoEmotions validation and test questions, the teacher's top pick disagreed with the human answer 38.5% of the time, which is why GoEmotions soft labels in v2 come from the raters instead.
- **Options carry short descriptions** (`card arrival: the customer is waiting for a card`).
  Some are hand-written; the rest were written by the teacher from real training examples.
- **Harder distractors:** about half the wrong options are the labels closest to the right
  answer (by embedding similarity of label and description), so questions aren't trivially easy.
- **More data:** 3,000 training questions per source (2,000 for SciCite), plus SciCite as a
  new source.
- **A held-out split** from PubMedQA (see below).

## Splits

| split | rows | use |
|---|---|---|
| train | 16995 | training |
| validation | 1799 | choosing epochs, fitting dials |
| test | 1797 | final evaluation on the same sources |
| heldout | 1000 | **test only**: a new source and a new kind of decision |

**Do not train on `heldout`, and do not fit dials on it.** It's 1,000 expert-labelled PubMedQA
questions (yes / no / maybe from an abstract), there to measure how well a model decides on
a label set and domain it has never seen.

## Fields

| field | meaning |
|---|---|
| `type` | `choice` or `noul` (yes/no; options are always `["false", "true"]`) |
| `instructions`, `state` | the question, and the text being judged |
| `options`, `option_descriptions` | option labels in canonical order, and their descriptions |
| `candidate` | yes/no rows: the label being asked about |
| `gold`, `gold_index` | the human label from the source dataset |
| `soft_label` | **the recommended training target**: rater votes for GoEmotions, the calibrated teacher otherwise |
| `soft_label_source` | `human_raters`, `teacher`, or `none (held out)` |
| `corrected_soft_label` | `soft_label` with the teacher's wrong top picks corrected (see v2.1 above) |
| `soft_label_corrected` | `true` where `corrected_soft_label` differs from `soft_label` |
| `human_probs` | GoEmotions only: rater votes per option, plus 0.1 pseudo-votes per option |
| `teacher_option_logprobs` | the teacher's raw log-probability of each option's answer letter |
| `teacher_temperature`, `teacher_shift` | the dial and nudge fit on validation for this row's group |
| `teacher_probs` | the calibrated teacher distribution |
| `teacher_letter_mass` | how much probability the teacher put on answer letters at all |
| `display_perm` | the option order the teacher saw |
| `source_id` | the row's id in its source dataset, where one exists |

## Which training target to use

We trained ModernBERT-base students on the train split with 6 different targets, each with
**3 random seeds** (3 epochs; best epoch chosen on validation log loss). Every student then gets
**2 temperature dials** fitted on validation: one for choice questions, and one plus a yes/no shift for
yes/no questions. We compared this with one dial per source and question type, and with no dial, by
cross-fitting on validation: 2 dials gave the lowest log loss for all five targets.

| run | target |
|---|---|
| A | one-hot `gold` |
| B | `gold`, label-smoothed (0.1) |
| C | `soft_label` |
| D | `0.5 * one_hot(gold) + 0.5 * soft_label` |
| E | `corrected_soft_label` |
| F | `0.5 * one_hot(gold) + 0.5 * corrected_soft_label` |

### Single students: mean over 3 seeds (lowest–highest)

| run | test accuracy | test ECE | test log loss | held-out accuracy | held-out ECE | held-out log loss |
|---|---|---|---|---|---|---|
| A | 0.912 (0.909–0.917) | 0.011 (0.006–0.019) | 0.247 (0.230–0.267) | 0.580 (0.538–0.614) | 0.130 (0.097–0.171) | 0.970 (0.945–0.997) |
| B | 0.918 (0.913–0.923) | 0.022 (0.016–0.026) | 0.281 (0.268–0.289) | 0.579 (0.568–0.585) | 0.092 (0.020–0.149) | 0.987 (0.957–1.005) |
| C | 0.892 (0.886–0.898) | 0.021 (0.020–0.023) | 0.310 (0.307–0.313) | 0.583 (0.576–0.592) | 0.055 (0.027–0.106) | 0.940 (0.926–0.952) |
| D | 0.915 (0.909–0.918) | 0.016 (0.015–0.019) | 0.248 (0.241–0.256) | 0.574 (0.548–0.595) | 0.044 (0.028–0.064) | 0.947 (0.878–1.017) |
| E | 0.912 (0.908–0.915) | 0.015 (0.014–0.016) | 0.261 (0.252–0.267) | 0.596 (0.591–0.605) | 0.086 (0.048–0.107) | 0.935 (0.924–0.946) |
| F | 0.919 (0.917–0.921) | 0.015 (0.015–0.016) | 0.245 (0.241–0.249) | 0.595 (0.584–0.607) | 0.046 (0.019–0.068) | 0.914 (0.888–0.953) |

### Ensembles: each run's 3 calibrated students averaged

| run | test accuracy | test ECE | test log loss | held-out accuracy | held-out ECE | held-out log loss |
|---|---|---|---|---|---|---|
| A | 0.922 | 0.018 | 0.221 | 0.595 | 0.153 | 0.964 |
| B | 0.925 | 0.009 | 0.249 | 0.582 | 0.052 | 0.948 |
| C | 0.895 | 0.013 | 0.292 | 0.595 | 0.065 | 0.932 |
| D | 0.920 | 0.008 | 0.231 | 0.589 | 0.054 | 0.928 |
| E | 0.918 | 0.013 | 0.241 | 0.587 | 0.081 | 0.926 |
| F | 0.928 | 0.006 | 0.225 | 0.595 | 0.051 | 0.899 |

On held-out, always answering "yes" scores 0.552 accuracy, and always predicting
the label frequencies gives a log loss of 0.937.

### What held up under both checks

Listed only if it held in **every seed** *and* in a paired bootstrap over questions for the ensembles.

**Test split** (the six training sources):
- C worse than A on accuracy
- C worse than A on log loss
- C worse than B on accuracy
- C worse than B on log loss
- D better than B on log loss
- F better than B on log loss
- F better than E on accuracy
- F better than E on log loss
- F better than C on accuracy
- F better than C on log loss

**Held-out PubMedQA** (never trained on):
- C better than A on ECE
- C better than A on log loss
- D better than A on ECE
- E better than A on log loss
- E better than B on log loss
- F better than A on ECE
- F better than A on log loss
- F better than B on log loss

### In short

- **Recommended target: F**, `0.5 * one_hot(gold) + 0.5 * corrected_soft_label`, with 2 dials,
  and an ensemble of 3 seeds if you can afford the training.
- **Why F:** against every target it was compared with (A, B, C, D and E), F was never reliably
  worse, on either split, under either check. On held-out it was reliably better than both
  `gold`-only targets on log loss. On the test split, it had the smallest seed-to-seed spread in
  accuracy and ECE.
- **F against its closest rivals (D and E) is only partly settled.** The average gap
  favoured F against D, in 5 of 6; against E, in 4 of 6 comparisons. Under both checks, F was better on: E test accuracy and E test log loss; the rest are within training luck.
- **On data like the six training sources, one-hot `gold` with a dial is about as good as any soft
  target.** `soft_label` alone (C) is the weakest there, because it inherits the teacher's mistakes.
- **Transfer to the held-out domain is weak for every target.** F's ensemble had the best held-out
  log loss (0.899, against 0.937 for predicting the label
  frequencies) and 0.595 accuracy (against 0.552 for always
  answering "yes").
- **Judge targets on several seeds.** Retraining alone moved one run's held-out accuracy by
  0.076 (run A).

## Teacher calibration on the test split

| source | type | n | acc raw | acc calibrated | T | shift | fit | ECE raw | ECE calibrated |
|---|---|---|---|---|---|---|---|---|---|
| `allenai/scicite` | choice | 214 | 0.799 | 0.799 | 5.46 | +0.00 | own | 0.190 | 0.030 |
| `allenai/scicite` | noul | 86 | 0.674 | 0.744 | 4.89 | +0.70 | own | 0.237 | 0.067 |
| `clinc/clinc_oos` | choice | 202 | 0.896 | 0.896 | 3.92 | +0.00 | own | 0.096 | 0.067 |
| `clinc/clinc_oos` | noul | 98 | 0.776 | 0.867 | 3.22 | +2.00 | own | 0.211 | 0.038 |
| `google-research-datasets/go_emotions` | choice | 216 | 0.569 | 0.569 | 8.86 | +0.00 | own | 0.398 | 0.088 |
| `google-research-datasets/go_emotions` | noul | 84 | 0.619 | 0.738 | 9.46 | +0.60 | own | 0.353 | 0.110 |
| `mteb/amazon_massive_intent` | choice | 212 | 0.901 | 0.901 | 5.46 | +0.00 | own | 0.089 | 0.028 |
| `mteb/amazon_massive_intent` | noul | 85 | 0.859 | 0.871 | 4.10 | +1.35 | own | 0.142 | 0.092 |
| `mteb/amazon_polarity` | choice | 190 | 0.963 | 0.963 | 5.46 | +0.00 | own | 0.037 | 0.020 |
| `mteb/amazon_polarity` | noul | 110 | 0.955 | 0.955 | 1.98 | -2.10 | own | 0.052 | 0.022 |
| `mteb/banking77` | choice | 216 | 0.921 | 0.921 | 5.11 | +0.00 | own | 0.070 | 0.020 |
| `mteb/banking77` | noul | 84 | 0.905 | 0.905 | 4.68 | -0.05 | own | 0.093 | 0.067 |

## Sources

- CLINC150 (Larson et al., 2019), CC-BY-3.0
- GoEmotions (Demszky et al., 2020), Apache-2.0; soft labels from its raw per-rater annotations
- BANKING77 (Casanueva et al., 2020), PolyAI/banking77, CC-BY-4.0; loaded via mteb/banking77
- MASSIVE (FitzGerald et al., 2022), English subset, CC-BY-4.0; loaded via mteb/amazon_massive_intent
- Amazon Review Polarity (Zhang et al., 2015), Apache-2.0 per fancyzhx/amazon_polarity; loaded via mteb/amazon_polarity
- SciCite (Cohan et al., 2019), Apache-2.0 (authors' repository); downloaded from the authors' official link
- PubMedQA (Jin et al., 2019), expert-labelled PQA-L subset, MIT. Held out: never used for training or calibration.

## Limitations

- Teacher soft labels reflect one 4-bit 7B teacher's judgement, calibrated on average per group.
- GoEmotions rater soft labels come from 3 raters per comment (5 when they disagreed), so they are coarse.
- Teacher-written option descriptions were reviewed by sampling, not one by one.
- Distractor difficulty depends on the embedding model (`BAAI/bge-small-en-v1.5`).
- English only. Gold labels inherit any noise in the source datasets.
