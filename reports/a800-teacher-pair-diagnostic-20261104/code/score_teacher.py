#!/usr/bin/env python3
"""Use original candidate scoring on exactly the same complete pilot cases."""
import argparse
import json
from pathlib import Path
import types
import pyarrow as pa

from kd_runtime import ROOT, WORKSPACE, PARENT, atomic_json, load_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=["TEACHER_PAIR"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    selected = json.loads((ROOT / "selection/evaluation.json").read_text())
    run_ids = set(selected["run_ids"])
    checkpoint = args.checkpoint or (PARENT if args.name == "BASELINE" else ROOT / "outputs" / args.name / "final")
    source_path = WORKSPACE / "scripts/run-formal-evaluation.py"
    source = source_path.read_text().replace("HOME = Path('/home/dev/assumeengage')", f"HOME = Path({str(WORKSPACE)!r})")
    condition = "if evaluation['catalog_id'] not in allowed:"
    if source.count(condition) != 1:
        raise RuntimeError("Original evaluation worker changed; its filter was not replaced")
    source = source.replace(condition, "if evaluation['catalog_id'] not in allowed or evaluation['run_id'] not in PILOT_RUN_IDS:")
    source = source.replace("torch.set_num_threads(4)", "torch.set_num_threads(2)")
    source = source.replace("formal_public_score_complete", "pilot_score_complete").replace("'public_score':", "'pilot_score':")
    formal = types.ModuleType("restored_pilot_evaluation")
    formal.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), formal.__dict__)
    original_suite = formal.Suite

    class PilotSuite:
        def __init__(self, *a, **kw):
            self.base = original_suite(*a, **kw)

        def __getattr__(self, key):
            return getattr(self.base, key)

        def rows(self, *a, **kw):
            for row in self.base.rows(*a, **kw):
                if row["_evaluation"]["run_id"] in run_ids:
                    yield row

    formal.Suite = PilotSuite
    formal.PILOT_RUN_IDS = run_ids
    formal.ROOT = ROOT / "outputs"
    formal.load_classifier = lambda path: load_classifier(path, trainable=False)
    native_summary = formal.benchmark_summary

    def summary_with_scored_fields(suite, results, *a, **kw):
        # PhishNChips exposes diagnostic fields with null gold, explicitly marked
        # unscored. Keep all predictions, but do not treat those fields as golds.
        rows = kw.get("rows")
        if rows is not None:
            scored_rows = []
            for row in rows:
                excluded = set(row.get("scoring", {}).get("unscored_fields", []))
                if excluded:
                    row = dict(row)
                    row["questions"] = {k:v for k,v in row["questions"].items() if k not in excluded}
                    row["expected"] = {k:v for k,v in row["expected"].items() if k not in excluded}
                scored_rows.append(row)
            kw["rows"] = scored_rows
        return native_summary(suite, results, *a, **kw)

    formal.benchmark_summary = summary_with_scored_fields
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    folder = formal.ROOT / "evaluation" / args.name
    if formal.done(folder / "complete.json"):
        print("Pilot evaluation already complete:", args.name, flush=True)
        return
    worker_args = argparse.Namespace(worker=0, workers=1, model=args.name, checkpoint=str(checkpoint))
    inference_complete = formal.done(folder / "worker-0-complete.json")
    if args.score_only and not inference_complete:
        raise RuntimeError("Cannot score incomplete inference outputs")
    if not inference_complete:
        formal.worker(worker_args)
    scores = formal.score_model(args.name, folder)
    scores.update(scope="pilot_subset", official_full_public_run=False, private_full_score=False,
                  selected_cases=selected["selected_cases"], selected_requests=selected["selected_requests"])
    atomic_json(folder / "scores.json", scores)
    completion = json.loads((folder / "complete.json").read_text())
    completion.pop("public_score", None)
    completion.update(scope="pilot_subset", pilot_score=scores["index"],
                      selected_cases=selected["selected_cases"],
                      candidate_pairs=selected["candidate_pairs"],
                      full_public_requests=selected["full_public_requests"],
                      official_full_public_run=False, private_full_score=False,
                       latency_scope="Two TP2 Teacher replicas; actual concurrent paired-adapter wall time, not isolated latency-v1",
                      evaluation_inputs="Same frozen complete pilot cases and ALL candidates; no training-length truncation")
    atomic_json(folder / "complete.json", completion)
    print(json.dumps({"event":"pilot_evaluation_complete", "model":args.name,
                      "pilot_score":scores["index"], "coverage":scores["coverage"],
                      "requests":selected["selected_requests"]}), flush=True)


if __name__ == "__main__":
    main()
