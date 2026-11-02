"""Read-only backup inventory, executed on the source host via SSH stdin.

No tensor loading or full-file hashing. Credential directories and browser/cache
profiles are intentionally outside the allowlist. Reports file metadata only.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/dev/assumeengage")
ROOTS = ["models", "datasets", "cache/pair-training", "runs/five-specialists-20261030"]
IGNORE_DIRS = {".git", ".secrets", "__pycache__", ".cache", ".ipynb_checkpoints"}
now = time.time()
files, skipped_symlinks = [], []
totals = defaultdict(lambda: {"files": 0, "bytes": 0})
completed = {}
run_root = ROOT / "runs/five-specialists-20261030"
if run_root.is_dir():
    for folder in run_root.iterdir():
        if not folder.is_dir():
            continue
        marker = folder / "complete.json"
        complete = False
        if marker.is_file():
            value = json.loads(marker.read_text())
            complete = value.get("complete") is True
        completed[folder.name] = complete

for area in ROOTS:
    start = ROOT / area
    if not start.exists():
        continue
    for directory, dirs, names in os.walk(start, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS and not d.startswith("."))
        for name in sorted(names):
            path = Path(directory) / name
            if name.startswith(".") or name.endswith(".lock") or name == "EXPERIMENT.lock":
                continue
            if path.is_symlink():
                skipped_symlinks.append(str(path.relative_to(ROOT)))
                continue
            if not path.is_file():
                continue
            stat = path.stat()
            relative = str(path.relative_to(ROOT))
            kind, priority, completion = "source_data", 2, "stable_source"
            if area == "models":
                kind, priority = "model_dependency", 0
                if "common-pair-init" in relative:
                    kind = "common_initialization_not_final_student"
            elif area == "cache/pair-training":
                kind = "prepared_training_data"
                priority = 0 if any(p in relative for p in ["global-pool/", "tokenized/global-new/", "tokenized/typed-full/", "normalized/typed-full/"]) else 1
            elif area.startswith("runs/"):
                parts = Path(relative).parts
                run = parts[2] if len(parts) > 2 else None
                completion = "completed_model" if completed.get(run) else "in_progress_snapshot_only"
                if "/final/" in relative:
                    kind, priority = "trained_final_weights", 0
                elif "/checkpoints/" in relative:
                    kind, priority = "training_checkpoint_state", 1
                else:
                    kind, priority = "run_metadata", 1
            files.append({"path": relative, "bytes": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns, "age_seconds_at_inventory": round(now-stat.st_mtime, 1),
                          "group": area, "kind": kind, "priority": priority, "status": completion})
            totals[area]["files"] += 1
            totals[area]["bytes"] += stat.st_size

pipeline_path = run_root / "PIPELINE.json"
pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.is_file() else {}
ready_path = ROOT / "cache/pair-training/global-pool/TRAINING_READY.json"
ready = json.loads(ready_path.read_text()) if ready_path.is_file() else {}
result = {
    "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "source_root": str(ROOT), "source_alias": "sh-09-gpu", "read_only": True,
    "full_file_hashing": False, "totals": dict(totals), "file_count": len(files),
    "total_bytes": sum(f["bytes"] for f in files), "completed_run_markers": completed,
    "observed_pipeline": {k:pipeline.get(k) for k in ["stage", "model", "updated_unix"]},
    "prepared_pool": {k:ready.get(k) for k in ["complete", "mixed_all_pairs", "mixed_all_effective_tokens"]},
    "selected_final_student": "not_yet_identified; do not substitute common initialization or an arbitrary specialist",
    "modelscope_upload_status": "blocked_by_rejected_access_token",
    "excluded": [".secrets", "credential files", "environment package caches", "browser profiles", "Git internal metadata", "lock files"],
    "skipped_symlinks": skipped_symlinks,
    "files": files,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
