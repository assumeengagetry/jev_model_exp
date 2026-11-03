# KD scale series:1M →2M →3M

Teacher: `PelaAI/KnowLine-31B-Gen1` (unchanged official frontend, BF16 TP2, read-twice, T1).

Each nested training size starts HARD/SOFT/HYBRID from the same frozen trained `MIXED_ALL/final`, for one equal epoch and identical sample/order. The already completed100k pilot is not rerun.

Evaluation reuses the same frozen37-benchmark complete-case subset and baseline predictions. Reported indices are subset estimates, not complete public leaderboard or private Full scores.

## Latest verified status —2026-11-03 02:40 CST

- **1M is complete**, with all three arms trained and all selected evaluation cases scored. Index: BASELINE22.39, HARD22.49, SOFT21.36, HYBRID22.38. SOFT−HARD−1.13; HYBRID−HARD−0.11. See `1m/REPORT.md` and `SCALE_CURVE.md`.
- Teacher2M remains active at1,997,980/2,000,000 committed labels;2,020 targets remain. The controller has no blocked role and will automatically start2M Student work and3M Teacher production once ready.
- Each1M arm completed7,813 steps and228,021,289 effective tokens. Evaluation coverage is100% of the frozen selected subset, not100% of the full benchmark collection.
- The local watcher was rearmed for the next report/failure event; no experiment configuration was changed in response to1M results.

## Initial launch record

Started2026-11-02 18:45:30 CST within the existing four-A800 allocation275711:

- Student1M: step275711.54; HARD/SOFT launched in parallel, with HYBRID queued next; each arm was evaluated after training. This step has now completed successfully.
- Teacher2M: step275711.55; fills only the256,157 missing targets, then advances to the3M scope.
- Controller automatically materializes ready stage snapshots and advances1M→2M→3M Student milestones. This round ends after3M.

Actual productive startup was observed: HARD/SOFT both reached100/7,813 optimizer steps, while Teacher committed519 new labels by18:51 CST. These are progress records, not new evaluation results.

## Results

- `LIVE_STATUS.json`: latest read-only metadata snapshot, refreshed by the local watcher.
- `SCALE_CURVE.md`, `scale-curve.csv`, `scale-curve.json`: completed-stage comparisons; initially contain only the completed100k pilot.
- `1m/REPORT.md`, `2m/REPORT.md`, `3m/REPORT.md`: created only after that stage's actual three-arm training and evaluations finish and results are downloaded.
- Each completed stage also includes native metrics, saved predictions, selection metadata, training logs, timing records and the code used. Checkpoints/probability matrices remain on the cluster; no inference is repeated for report copying.

Implementation/controller records: `/home/assumeengage/profiler/a800/devlocal/jev-kd-stages-20261102/` and `/home/assumeengage/profiler/a800/docs/superpowers/plans/2026-11-02-kd-scale-series.md`.
