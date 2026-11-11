# Independent static review and dispositions

Reviewer session: `ses_e3b17f15bffeWKqvei8sKZ3czC`; four complete source files, no tests/model/compile/hash or edits. No Critical findings; no definite prefix-mask/RoPE/KV-isolation/LoRA arithmetic bug found. This is not runtime verification.

## Important — one implementation fix pass

1. Anchor current official input list to `OFFICIAL_COMPLETE.binding.official_inputs`. Implemented exact equality; direct remote read confirmed the current lists actually match before launch.
2. Ignore previous-attempt stale runner status until the status file metadata changes after child launch. Implemented; legitimate old error counts cannot terminate a new child during CLI input verification.
3. Runner counts are attempt-local on resume. Full acceptance now validates actual final unique results rather than requiring this attempt alone to execute 140620 successes; scored count remains140178.
4. Comparison now joins final records by run_id, not line order. Duplicate records require identical request identity/payload and a preceding error, preserving upstream last-attempt semantics without silently accepting non-error rewrites. Attempt/failed latency is separately audited.
5. Regraded the reviewer's source-byte-binding minor as Important because byte preservation is explicit in this plan. Source and frozen source are now Base64 of exact `read_bytes()`, with no new hashing.

These changes are source-inspected, not dynamically tested under the retained no-extra-tests boundary. Production capture/replay, numerical agreement, complete coverage and runtime remain pending.

## Deferred minor

- A logical-token mismatch raises before `COMPARISON.json` is saved. Both complete raw result files remain preserved and no completion receipt is emitted, but there is not yet a saved summary diagnostic for that failure. The production report must not imply a comparison summary exists in that case.
