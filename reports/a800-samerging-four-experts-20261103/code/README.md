# SAMerging: four existing Qwen3 specialists

The user requested an actual SAMerging trial, not a literature-only answer. This experiment reuses the existing A800 Student environment, base model, four expert adapters, shared token caches and native evaluation suite.

- Source: https://github.com/arshandalili/SAMerging at `a194e8ed8ab790dcae9c02a8d1467e00e86b53e5`; paper https://arxiv.org/abs/2512.21288.
- Inputs: Knowledge/Retrieval/Tools/Arts specialist checkpoints from the same common initialization. These are not the KD HARD/SOFT/HYBRID arms, and KnowLine31B weights are not merged.
- Learn only effective-weight-tensor merge coefficients, including one shared binary classification head. Materialize FP32 effective weights before task-vector construction; do not average LoRA factors.
- Official NLP defaults:600 SAM steps,init0.5,lr1e-3,batch16 per task,T1,unclamped. Official code uses adaptive SAM with SGD(momentum0.99,weight_decay5e-4),rho0.07; paper appendix instead describes Adam. Follow the actual NLP code and disclose the distinction.
- Calibration:1,600 fixed unlabeled inputs per task,seed42,task-balanced KL(expert||merged). No gold labels or evaluation inputs used for calibration.
- Correct official example's shuffled input/logit misalignment and stale merged-weight cache after SAM perturbation. The official SAM optimizer itself is vendored unchanged.
- Evaluation: full public37-benchmark reconstruction,110,087 requests, all candidates, same native scoring; additionally derive the KD frozen-subset score from saved predictions, with no repeated inference.
- One bounded GPU step queues inside existing allocation275711. Do not interrupt the Student2M/Teacher3M steps or request extra GPUs.

Remote root: `/data/run01/scwb721/jev-kd-20261101/experiments/samerging-four-experts-20261103`.

Local result root: `../jev_model_exp/reports/a800-samerging-four-experts-20261103`.

Plan/ledger: `docs/superpowers/plans/2026-11-03-samerging-four-experts.md`.
