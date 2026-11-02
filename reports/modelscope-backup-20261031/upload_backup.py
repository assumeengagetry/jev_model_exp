"""Upload project assets to the user's private ModelScope CN repositories.

Operational backup worker: no GPU use, no model loading, no source deletion,
no manual full-file hashing. The Hub SDK performs its normal upload integrity
work. Credentials are read only from the source machine's private secret file.
"""

import argparse
import fcntl
import glob
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from modelscope_hub.api import HubApi
from modelscope_hub.config import HubConfig


ROOT = Path("/home/dev/assumeengage")
WORK = ROOT / "backup/modelscope-20261031"
ENDPOINT = "https://www.modelscope.cn"
OWNER = "assumeengage"
MODEL_REPO = OWNER + "/qwen3-06b-experiment-backup-20261031"
DATA_REPO = OWNER + "/jev-distill-data-20261031"
EXPECTED_MODELS = [
    "SPECIALIST_KNOWLEDGE", "SPECIALIST_LANGUAGE", "SPECIALIST_RETRIEVAL", "SPECIALIST_TOOLS", "SPECIALIST_ARTS", "MIXED_ALL",
    "MERGE_AVERAGE", "MERGE_TASK_ARITHMETIC", "MERGE_TIES", "MERGE_DARE_TASK_ARITHMETIC", "MERGE_DARE_TIES",
]
IGNORE_DIRS = {".git", ".secrets", ".cache", "__pycache__", ".ipynb_checkpoints"}
IGNORE_NAMES = {"token", "git_token", "cookies", "credentials", "session", ".env", "EXPERIMENT.lock", "TOKENIZATION.lock"}
SENSITIVE = re.compile(r"ms-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----|(?:X-Amz-Signature|x-oss-signature|oss-access-key-id)=")
logging.disable(logging.CRITICAL)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class Backup:
    def __init__(self):
        self.lock = threading.RLock()
        self.token = (ROOT / ".secrets/modelscope_token").read_text().strip()
        self.api = HubApi(config=HubConfig(
            endpoint=ENDPOINT, token=self.token,
            cache_dir=ROOT / "cache/modelscope-backup",
            config_dir=ROOT / ".secrets/modelscope-backup-cn"))
        assert self.api.whoami().username == OWNER, "Authenticated owner differs from the authorized repository owner"
        self.state_path = WORK / "upload-state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {
            "endpoint": ENDPOINT, "started_utc": now(), "repositories": {"model": MODEL_REPO, "dataset": DATA_REPO},
            "units": {}, "excluded_files": [], "complete": False, "initial_snapshot_complete": False}

    def save(self, **updates):
        with self.lock:
            self.state.update(updates)
            self.state["updated_utc"] = now()
            atomic_json(self.state_path, self.state)

    def private(self, repo, kind):
        info = self.api.get_model(repo) if kind == "model" else self.api.get_dataset(repo)
        visibility = info.get("visibility") if isinstance(info, dict) else getattr(info, "visibility", None)
        visibility = getattr(visibility, "value", visibility)
        assert str(visibility).lower() in {"1", "private"}, "Private repository visibility not confirmed"
        if isinstance(info, dict):
            assert info.get("private") is not False, "Repository private flag conflicts with visibility"

    def collect(self, folder):
        expected = {}
        for directory, directories, names in os.walk(folder, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in IGNORE_DIRS and not d.startswith("."))
            for name in sorted(names):
                path = Path(directory) / name
                if name.startswith(".") or name in IGNORE_NAMES or name.endswith((".lock", ".tmp", ".part", ".partial", ".writing", ".incomplete", ".pem", ".key")):
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                stat = path.stat()
                relative = str(path.relative_to(folder))
                # Operational metadata is small. Exclude credential-bearing
                # metadata without altering the source or the training corpus.
                if path.suffix.lower() in {".json", ".md", ".txt", ".yaml", ".yml"} and stat.st_size <= 5_000_000:
                    content = path.read_text(errors="replace")
                    if self.token in content or SENSITIVE.search(content):
                        excluded = str(path.relative_to(ROOT))
                        if excluded not in self.state["excluded_files"]:
                            self.state["excluded_files"].append(excluded)
                        continue
                expected[relative] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return expected

    def upload(self, relative, kind, role):
        folder = ROOT / relative
        if not folder.is_dir():
            return
        expected = self.collect(folder)
        if not expected:
            return
        previous = self.state["units"].get(relative, {})
        if previous.get("status") == "verified" and previous.get("source_files") == expected:
            return
        repo = MODEL_REPO if kind == "model" else DATA_REPO
        self.private(repo, kind)
        record = {"repo": repo, "repo_type": kind, "role": role, "path_in_repo": "workspace/" + relative,
                  "status": "uploading", "files": len(expected), "bytes": sum(v["bytes"] for v in expected.values()),
                  "started_utc": now(), "source_files": expected}
        self.state["units"][relative] = record
        self.save(current_unit=relative, phase="uploading")
        print(json.dumps({"event": "uploading", "unit": relative, "files": record["files"], "bytes": record["bytes"]}), flush=True)
        last_progress = [0.0]
        def progress(event):
            with self.lock:
                if time.monotonic() - last_progress[0] < 5:
                    return
                last_progress[0] = time.monotonic()
                record["numeric_progress"] = {k:v for k,v in event.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
                self.save()
        try:
            self.api.upload_folder(repo, kind, folder, path_in_repo=record["path_in_repo"],
                allow_patterns=[glob.escape(p) for p in expected], max_workers=2, use_cache=True,
                disable_tqdm=True, sync_remote_repo=False,
                tracker_path=WORK / "trackers" / (relative.replace("/", "__") + ".json"), progress_callback=progress)
            for name, before in expected.items():
                stat = (folder / name).stat()
                assert (stat.st_size, stat.st_mtime_ns) == (before["bytes"], before["mtime_ns"]), "Source changed during upload"
            remote = {getattr(f, "path", None): f for f in self.api.list_repo_files(repo, kind, recursive=True)}
            for name, before in expected.items():
                path = record["path_in_repo"] + "/" + name
                assert path in remote, "Uploaded file absent from remote repository tree: " + path
                size = getattr(remote[path], "size", None)
                assert size is not None and int(size) == before["bytes"], "Uploaded file size does not match: " + path
            record.update(status="verified", verified_utc=now(), remote_files_checked=len(expected))
            self.save()
            print(json.dumps({"event": "verified", "unit": relative, "bytes": record["bytes"]}), flush=True)
        except Exception as error:
            message = str(error).replace(self.token, "<redacted>").split("\n")[0][:700]
            record.update(status="error", error_type=type(error).__name__, error=message)
            self.save(phase="blocked")
            print(json.dumps({"event": "error", "unit": relative, "error_type": type(error).__name__, "message": message}), flush=True)
            raise RuntimeError("Backup unit failed; consult sanitized upload-state.json") from None

    def models(self, checkpoints=False):
        run = ROOT / "runs/five-specialists-20261030"
        if run.is_dir():
            for folder in sorted(run.iterdir()):
                marker = folder / "complete.json"
                if not folder.is_dir() or not marker.is_file():
                    continue
                if json.loads(marker.read_text()).get("complete") is True:
                    role = "completed_weight_merge_no_new_training" if folder.name.startswith("MERGE_") else "completed_trained_model"
                    self.upload(str((folder / "final").relative_to(ROOT)), "model", role)
                    # Completion metadata is also retained separately by SCP.
                    self.api.upload_file(MODEL_REPO, "model", marker,
                        "workspace/" + str(marker.relative_to(ROOT)), disable_tqdm=True)
        if checkpoints and run.is_dir():
            for folder in sorted(run.glob("*/checkpoints/*")):
                if not folder.is_dir():
                    continue
                files = [p for p in folder.rglob("*") if p.is_file() and not p.is_symlink()]
                if files and min(time.time()-p.stat().st_mtime for p in files) >= 120:
                    self.upload(str(folder.relative_to(ROOT)), "model", "stable_checkpoint_not_necessarily_final")

    def publish_status(self):
        verified = [v for v in self.state["units"].values() if v.get("status") == "verified"]
        pending = [name for name in EXPECTED_MODELS if self.state["units"].get("runs/five-specialists-20261030/"+name+"/final",{}).get("status")!="verified"]
        summary = {"endpoint": ENDPOINT, "updated_utc": now(), "scope": "large_asset_backup; small SCP assets recorded separately",
            "initial_snapshot_complete": self.state["initial_snapshot_complete"],
            "complete": self.state.get("complete",False), "pending_final_models": pending,
            "verified_units": len(verified), "verified_files": sum(v["files"] for v in verified),
            "verified_bytes": sum(v["bytes"] for v in verified),
            "note": "Only remotely verified stable assets are counted. Pending source models are not claimed complete.",
            "units": [{k:v[k] for k in ["repo", "repo_type", "path_in_repo", "role", "status", "files", "bytes"]} for v in verified]}
        atomic_json(WORK / "SUMMARY.json", summary)
        payload=json.dumps(summary,ensure_ascii=False,indent=2).encode()
        for repo, kind in [(MODEL_REPO,"model"),(DATA_REPO,"dataset")]:
            self.private(repo,kind)
            self.api.upload_file(repo,kind,payload,"BACKUP_STATUS.json",disable_tqdm=True)

    def initial(self):
        self.models()
        self.upload("models/Qwen3-0.6B-common-pair-init-r32", "model", "initialization_and_tokenizer_not_final_student")
        self.upload("models/Qwen3-0.6B", "model", "adapter_base_dependency")
        for relative in ["cache/pair-training/global-pool", "cache/pair-training/tokenized/global-new",
                         "cache/pair-training/tokenized/typed-full", "cache/pair-training/normalized/typed-full",
                         "datasets/evaluation-only"]:
            self.upload(relative, "dataset", "distillation_migration_dependency")
        self.publish_status()

        for relative in ["cache/pair-training/extra-pairs", "cache/pair-training/generated-pairs",
                         "datasets/zero-shot-label-nli-http", "datasets/zero-shot-label-nli",
                         "datasets/tasksource-jev-typed-decisions", "datasets/tasksource-jev-typed-decisions-ms",
                         "datasets/benchmarks-train", "datasets/benchmark-sources", "datasets/overnight-supplement"]:
            self.upload(relative, "dataset", "source_and_preparation_backup")
        self.models(checkpoints=True)
        self.save(initial_snapshot_complete=True, phase="initial_snapshot_verified", current_unit=None)
        self.publish_status()

    def watch(self, hours, interval):
        assert self.state.get("initial_snapshot_complete"), "Initial snapshot must finish before incremental watching"
        deadline=time.monotonic()+hours*3600
        while time.monotonic()<deadline:
            self.models(checkpoints=False)
            run=ROOT/"runs/five-specialists-20261030"
            evaluation=run/"EVALUATION_COMPLETE.json"
            done=evaluation.is_file() and json.loads(evaluation.read_text()).get("complete") is True
            if done:
                self.upload("datasets/evaluation-only", "dataset", "frozen_suite_and_final_candidate_cache")
                self.upload("runs/five-specialists-20261030/evaluation", "dataset", "completed_evaluation_predictions_and_scores")
                self.upload("runs/five-specialists-20261030/report", "dataset", "completed_experiment_report")
            pending=[name for name in EXPECTED_MODELS if self.state["units"].get("runs/five-specialists-20261030/"+name+"/final",{}).get("status")!="verified"]
            self.save(phase="watching_completed_source_models",pending_final_models=pending,source_evaluation_complete=done)
            if not pending and done:
                self.save(complete=True,phase="large_experiment_assets_verified")
                self.publish_status()
                return
            self.publish_status()
            time.sleep(interval)
        self.save(phase="watch_window_ended_pending_sources",complete=False)
        self.publish_status()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["initial","models","watch"],default="initial")
    parser.add_argument("--watch-hours",type=float,default=72)
    parser.add_argument("--interval",type=float,default=300)
    parser.add_argument("--wait-for-lock",action="store_true")
    args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    (WORK / "trackers").mkdir(exist_ok=True)
    process_lock = (WORK / "worker.lock").open("w")
    fcntl.flock(process_lock, fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB))
    atomic_json(WORK / "worker.json", {"pid":os.getpid(),"started_utc":now(),"mode":args.mode})
    backup=Backup()
    if args.mode=="initial":backup.initial()
    elif args.mode=="models":
        backup.models(checkpoints=True)
        backup.publish_status()
    else:backup.watch(args.watch_hours,args.interval)
    print(json.dumps({"event":"worker_finished","mode":args.mode,"time":now()}),flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never format SDK exceptions at the top level: their request details
        # may contain an authentication payload. Sanitized metadata is above.
        print(json.dumps({"event":"worker_stopped","reason":"see sanitized upload-state.json"}),flush=True)
        sys.exit(1)
