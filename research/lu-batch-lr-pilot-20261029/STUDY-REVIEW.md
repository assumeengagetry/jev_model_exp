# Batch / learning-rate pilot: reviewed results

All eight3M-token runs completed. Inputs and frozen worker hashes remain unchanged.
All saved adapters contain finite LoRA and score tensors. No save/reload forward-equivalence measurement was added.

## Native quality and retention

Native is the10-task mean on the reused5904-pair panel, **not Language Skill or a single accuracy**.
Retention is macro pair accuracy on the reused2006-pair D0 Language-validation proxy.

| Effective batch | Peak LR | Native macro, mean ± sample std | D0 macro accuracy | D0 pair loss | Train token/s |
|---:|---:|---:|---:|---:|---:|
| R0 baseline | — | 0.38136 | 0.80038 | 0.38937 | — |
| 128 | 5e-05 | 0.35255 ± 0.01799 | 0.80180 | 0.40039 | 12732 |
| 128 | 0.0001 | 0.42561 ± 0.01232 | 0.80099 | 0.40610 | 12620 |
| 256 | 5e-05 | 0.32010 ± 0.02777 | 0.79910 | 0.39668 | 12576 |
| 256 | 0.0001 | 0.33696 ± 0.03263 | 0.80120 | 0.40252 | 12822 |

## Interpretation

- B128_L10 is the best of these four settings on this short reused native panel, in both seeds.
- B128_L10 improves over both B128_L5 and the common R0 native baseline; the other three settings are below R0 in the mean.
- B256_L10 does not reproduce B128_L5 endpoint quality in these two seeds; no evidence of a meaningful single-GPU throughput gain.
- Retention accuracy barely moves but retention CE rises vs R0 in every arm; do not claim zero forgetting.
- Gradient clipping occurs on93.3–100% of updates; optimizer dynamics and clipping limit simple scaling-rule interpretations.
- Native improvements are task-dependent, particularly RAGTruth QA/Summary; not a uniform gain across all tasks.
- Only3M tokens, two continuation seeds, shared R0 and reused validation; no universal scaling law or full-budget winner established.

## Native task breakdown (two-seed means)

| Task | 128 /5e-5 | 128 /1e-4 | 256 /5e-5 | 256 /1e-4 |
|---|---:|---:|---:|---:|
| acos/Laptop-ACOS | 0.07976 | 0.07412 | 0.06074 | 0.07266 |
| acos/Restaurant-ACOS | 0.65893 | 0.64794 | 0.59643 | 0.52976 |
| anli_r1 | 0.32570 | 0.37378 | 0.30951 | 0.32364 |
| anli_r2 | 0.24559 | 0.27224 | 0.24355 | 0.25207 |
| anli_r3 | 0.29983 | 0.35752 | 0.30869 | 0.30198 |
| hellaswag | 0.39062 | 0.42969 | 0.39844 | 0.39062 |
| isarcasm/intended-en | 0.47600 | 0.54420 | 0.50000 | 0.47163 |
| ragtruth/Data2txt | 0.66844 | 0.68966 | 0.67018 | 0.67196 |
| ragtruth/QA | 0.30864 | 0.55309 | 0.11348 | 0.31049 |
| ragtruth/Summary | 0.07197 | 0.31389 | 0.00000 | 0.04478 |

## Checkpoint memory / time

Microbatch16, effective32, forward/backward only; no optimizer updates. One warmup and three measured calls, two opposite-order repeats.

| Profile | Checkpoint | Successful / OOM repeats | Mean token/s | Peak allocated / reserved GiB |
|---|---|---:|---:|---:|
| medium | True | 2 / 0 | 13475 | 1.77 / 1.86 |
| medium | False | 2 / 0 | 17939 | 8.22 / 8.33 |
| long_real_pairs | True | 2 / 0 | 21899 | 8.39 / 10.70 |
| long_real_pairs | False | 0 / 2 | OOM | — |

## Gradient consistency / numerical caveat

- Same-weight checkpoint on/off: relative-L2 difference0 on this sample.
- Physical32 vs16x2: BF16 relative-L2 difference 0.00227239 (about0.23%).
- The CPU float64 toy-model tests separately verified normalization, partial batches, checkpointing and matching AdamW updates.
- Raw full-vector float32 cosine values slightly exceed1 and are invalid numerical estimates; do not interpret them. The raw artifacts and original auto-report are preserved, with this issue explicitly recorded in `technical/numerical-review.json`.

## Boundaries

Only two continuation seeds sharing R0; reused validation; only four ACOS reviews per domain; short3M horizon. No public Language evaluation, full-finetuning comparison, new threshold fitting, automatic recipe change or follow-up training.
