# Pair classifier shared-prefix and bounded graphs implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate the already trained joint final's inference and execute one new complete official public run without replacing its scoring model.

**Architecture:** Add a separate inference snapshot, leaving all completed training/evaluation artifacts untouched. Reuse the installed Qwen3 backbone and actual saved FP32 score head; cache a field prefix once, fork independent candidate suffixes, and replay a finite startup-captured CUDA graph set for common shapes. Keep the unmerged LoRA arithmetic and use intact eager fallbacks for long tails.

**Tech Stack:** Existing Torch 2.9.1, Transformers 4.57.6, PEFT 0.18.1, upstream Decision Index 25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d, GPU0 H800.

**Spec:** User's 2026-11-10 instruction to implement the prior inference recommendations and rerun public; `reports/inference-backend-survey-20261110/README.md:27–36` defines the accepted semantics and priorities.

## Global Constraints

- Same frozen joint final and actual training base, same tokenizer boundaries, rendering, EOS, positions, rightmost-nonpad pooling, two-class margin and per-field full softmax.
- No retraining, head replacement, LoRA merge, quantization, new calibration, truncation, filtering, upload, commit, push, or service restart.
- Keep the prior no-extra-tests/smoke/model-or-corpus-hash boundary. Graph initialization is intrinsic engine startup, not a separate benchmark; numerical comparison reads the two actual official result files, without additional inference.
- GPU0 only; readout GPU1/2/3 allocation and all its processes stay unchanged.
- Existing dirty/untracked files are other sessions' work. Implementation is additive under `artifacts/inference-prefix-graphs-20261110/code/`, deployed to a new `/tmp/opencode/jev-inference-prefix-graphs-20261110/` root, not edited in the old frozen snapshot. No worktree or baseline test installation is needed for this separate run artifact.
- Thirty minutes is a target, not a guarantee. Report actual progress and timings, and do not extrapolate short tasks over long task families without evidence.

## Review Focus

- Padding between cached prefix and suffix must be masked; suffix rotary positions start at the real prefix length, never the bucket width.
- Candidate KV branches never update or contaminate the shared prefix; subsequent fields overwrite the appropriate prefix region.
- Empty candidates still include EOS; original pad and EOS differ; all candidate counts and original order survive microbatch sorting.
- Graph inputs/output addresses remain live; private pools are independent, bounded, sealed after startup; misses use eager execution without recapture.
- New run binds existing official corpus/model and new source byte-for-byte; rejects unknown partials and never resumes with changed code.

## Task 1: Inference engine and graph/cache core

**Files:** Create `artifacts/inference-prefix-graphs-20261110/code/fast_pair_engine.py` and `prefix_graph_runtime.py`.

**Interfaces:** `FastPairClassifierEngine(model, base_model, device='cuda:0', ...)` implements the existing Engine protocol. `PrefixGraphRuntime(backbone, head, ...)` provides `full(rows) -> Tensor` and `branched(prefix, suffixes) -> Tensor`, with margins restored to input candidate order by the engine.

- [x] Read actual saved configuration, installed upstream implementation and current GPU allocation; use version-matched primary sources after Context7 quota failure.
- [x] Write no-merge LoRA wrapper after exact saved-tensor admission; use original backbone layers/norms/rotary/SDPA rather than rewriting the network. Runtime execution remains pending.
- [x] Write immutable prefix KV fork, explicit prefix-padding and causal suffix mask, true position IDs, and full candidate order restoration. Runtime execution remains pending.
- [x] Write startup capture of a fixed bounded graph set, retained inputs/outputs and independent pools; eager for uncaptured shapes. Actual capture remains pending.
- [x] Have an independent read-only reviewer check masks, pooling, dtype, lifecycle and long-tail fallbacks. Four Important provenance/resume issues were source-inspected and fixed; exact source byte binding also tightened. No extra test commands under the existing boundary.

## Task 2: New official orchestration and result comparison

**Files:** Create `artifacts/inference-prefix-graphs-20261110/code/run_public.py`, `compare_results.py`, and `reports/inference-prefix-graphs-20261110/README.md`.

**Interfaces:** `run_public.py` runs the untouched upstream pipeline on the complete existing edition 0.3 suite in its own root; after success, writes its own completion receipt and calls `compare_results.py` to join by run ID, compare coverage/choices/probabilities/tokens/latencies, and report original and optimized scores separately.

- [x] Bind old official completion/snapshot/input metadata and exact new source bytes; before-launch and actual post-completion direct revalidation passed. Unbound outputs fail closed.
- [x] Use a new run lock and the old official GPU0 lock, wait for GPU0 availability, launch the complete upstream pipeline without a limit/upload option, preserve output on failure.
- [x] Deploy only new files, read them back for byte equality, then launch the newly authorized full production run.
- [x] Verify real child identity, engine ready/progress and initial production throughput. At10:14:45UTC:770 requests allok; first500 mean15.16ms vs70.96ms,4.68x, probabilities/choices/tokens identical; 31 startup graphs,15.03sload,12.21GiBreserved. This is not complete coverage or a full-suite ETA.
- [x] Actual completion independently verified13:41:44UTC:140620executed/140178scored allok;14provenance/coverage/source/receiptchecks true. Public24.99, inference12157.2s,2.765xvsold, not30minutes; choice agreement99.1655%, mean/max probability difference0.0012057/0.225063. Original frozen artifacts unchanged; complete comparison archived, no upload or third run.

## Execution authorization and evidence

The user explicitly directed implementation of the previously recommended inference-only approach and a new public run; this document records execution details, not another proposed model experiment. Execute inline, with a fresh read-only code review as required by the workflow; no automatic commits. Runtime tests remain unperformed by explicit retained user boundary; final acceptance is the authorized production pipeline plus read-only result comparison.
