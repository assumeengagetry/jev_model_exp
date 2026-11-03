"""Exclusive state-local work claims for the existing canonical probability cache."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from kd_runtime import ROOT, SOURCE_ROOT, DOMAINS, atomic_json


def atomic_npy(path, values):
    temporary = path.with_name(path.stem + ".writing.npy")
    np.save(temporary, values)
    temporary.replace(path)


class AnnotationQueue:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.folder = self.cache / "dynamic-schedule"
        self.tasks = self.folder / "tasks"
        self.state_path = self.folder / "queue.json"
        self.tasks.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.folder / "queue.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def snapshot(self):
        return json.loads(self.state_path.read_text())

    def task_path(self, task_id):
        return self.tasks / f"{task_id:06d}.json"

    def require_no_writers(self):
        for directory in Path("/proc").iterdir():
            if not directory.name.isdigit():
                continue
            try:
                args = (directory / "cmdline").read_bytes().decode().split("\0")
                producer = any(arg.startswith(str(SOURCE_ROOT)+"/") and arg.endswith("/code/annotate_teacher.py") for arg in args)
                if producer and "--prepare" not in args:
                    raise RuntimeError(f"Annotation writer {directory.name} is still active; ownership cannot change")
            except (FileNotFoundError, PermissionError, UnicodeDecodeError, ProcessLookupError):
                continue

    def timing(self):
        path = self.cache / "timing.json"
        if path.exists():
            return json.loads(path.read_text())
        stage_starts = []
        annotation_starts = []
        current_stage = None
        log = ROOT / "logs/pipeline-driver.log"
        if log.exists():
            with log.open() as source:
                for line in source:
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if value.get("stage") == "teacher_annotation":
                        current_stage = value["unix_time"]
                        stage_starts.append(current_stage)
                    if value.get("event") == "teacher_annotation_started" and current_stage is not None:
                        annotation_starts.append(current_stage + value.get("setup_seconds", 0))
        value = dict(first_stage_started_unix=min(stage_starts) if stage_starts else time.time(),
                     first_annotation_started_unix=min(annotation_starts) if annotation_starts else None)
        atomic_json(path, value)
        return value

    def prepare(self, dataset, order, probabilities):
        self.require_no_writers()
        if len(order) > len(dataset) or probabilities.shape != (len(dataset), 2):
            raise RuntimeError("Target order and full canonical probability cache disagree")
        # Flush after the old producers have stopped, including their uncommitted blocks.
        probabilities.flush()
        finite = np.isfinite(probabilities).all(axis=1)
        recovered = int(finite[order].sum())
        ranges_path = self.folder / "ranges.npy"
        with self.locked():
            previous = self.snapshot() if self.state_path.exists() else None
            if previous and previous["total_pairs"] != len(order):
                raise RuntimeError("Existing work ownership belongs to a different pool")
            floor = previous["cached_pairs"] if previous else 0
            if previous is None:
                for replica in [0, 1]:
                    path = self.cache / f"replica-{replica}-progress.json"
                    if path.exists():
                        value = json.loads(path.read_text())
                        if value.get("mode") != "dynamic-state-local-chunks-v1":
                            floor += value.get("completed_pairs", 0)
                            archive = self.folder / f"legacy-replica-{replica}-progress.json"
                            if not archive.exists():
                                atomic_json(archive, value)
            if recovered < floor:
                raise RuntimeError("Fewer complete cached targets than previously committed; cache preserved")
            if not ranges_path.exists():
                states = np.concatenate([dataset.tables[d]["state"] for d in DOMAINS])[order]
                ranges = []
                begin = 0
                while begin < len(order):
                    end = min(len(order), begin + 4096)
                    if end < len(order):
                        group_end = int(np.searchsorted(states, states[end - 1], side="right"))
                        if group_end - begin <= 8192:
                            end = group_end
                        else:
                            group_begin = int(np.searchsorted(states, states[end - 1], side="left"))
                            if group_begin > begin:
                                end = group_begin
                    ranges.append((begin, end))
                    begin = end
                atomic_npy(ranges_path, np.asarray(ranges, dtype=np.int64))
            ranges = np.load(ranges_path, mmap_mode="r")
            pending = []
            for task_id, (begin, end) in enumerate(ranges):
                complete = bool(finite[order[int(begin):int(end)]].all())
                path = self.task_path(task_id)
                prior_task = json.loads(path.read_text()) if path.exists() else None
                if prior_task and prior_task.get("state") == "complete" and not complete:
                    raise RuntimeError("A committed task has unfinished targets; cache preserved")
                if complete:
                    if not prior_task or prior_task.get("state") != "complete":
                        atomic_json(path, dict(task_id=task_id, start=int(begin), end=int(end),
                                              state="complete", recovered_only=True))
                else:
                    pending.append(task_id)
            atomic_npy(self.folder / "pending.npy", np.asarray(pending, dtype=np.int64))
            timing = self.timing()
            value = dict(mode="dynamic-state-local-chunks-v1", run_id=time.time_ns(),
                         total_pairs=len(order), cached_pairs=recovered, recovered_pairs=recovered,
                         total_tasks=len(ranges), completed_tasks=len(ranges)-len(pending),
                         pending_task_count=len(pending), next_pending=0, active={},
                         new_pairs_by_replica={"0": 0, "1": 0}, reused_pairs_by_replica={"0": 0, "1": 0},
                         run_started_unix=time.time(), updated_unix=time.time(), **timing)
            atomic_json(self.state_path, value)
        return value

    def start_annotation_clock(self):
        with self.locked():
            state = self.snapshot()
            if state["first_annotation_started_unix"] is None:
                state["first_annotation_started_unix"] = time.time()
                timing = self.timing()
                timing["first_annotation_started_unix"] = state["first_annotation_started_unix"]
                atomic_json(self.cache / "timing.json", timing)
                atomic_json(self.state_path, state)

    def claim(self, replica, run_id):
        key = str(replica)
        with self.locked():
            state = self.snapshot()
            if state["run_id"] != run_id or key in state["active"]:
                raise RuntimeError("Stale run or duplicate annotation owner")
            if state["next_pending"] >= state["pending_task_count"]:
                return None
            pending = np.load(self.folder / "pending.npy", mmap_mode="r")
            ranges = np.load(self.folder / "ranges.npy", mmap_mode="r")
            task_id = int(pending[state["next_pending"]])
            begin, end = (int(x) for x in ranges[task_id])
            task = dict(task_id=task_id, start=begin, end=end, state="active", owner=replica,
                        owner_pid=os.getpid(), run_id=run_id, completed_rows=0,
                        new_pairs=0, reused_pairs=0, updated_unix=time.time())
            atomic_json(self.task_path(task_id), task)
            state["next_pending"] += 1
            state["active"][key] = task_id
            state["updated_unix"] = time.time()
            atomic_json(self.state_path, state)
            return task

    def require_owner(self, state, task):
        if state["run_id"] != task["run_id"] or state["active"].get(str(task["owner"])) != task["task_id"]:
            raise RuntimeError("Annotation task ownership changed while its writer was active")
        current = json.loads(self.task_path(task["task_id"]).read_text())
        if current.get("owner_pid") != os.getpid() or current.get("state") != "active":
            raise RuntimeError("Annotation worker does not own this task")
        return current

    def commit(self, task, new_pairs=0, reused_pairs=0):
        # Caller has flushed probability writes BEFORE publishing any progress here.
        with self.locked():
            state = self.snapshot()
            current = self.require_owner(state, task)
            current["new_pairs"] += new_pairs
            current["reused_pairs"] += reused_pairs
            current["completed_rows"] += new_pairs + reused_pairs
            current["updated_unix"] = time.time()
            if current["completed_rows"] > current["end"]-current["start"]:
                raise RuntimeError("Task completion exceeded its exclusive range")
            atomic_json(self.task_path(task["task_id"]), current)
            state["cached_pairs"] += new_pairs
            state["new_pairs_by_replica"][str(task["owner"])] += new_pairs
            state["reused_pairs_by_replica"][str(task["owner"])] += reused_pairs
            state["updated_unix"] = time.time()
            if state["cached_pairs"] > state["total_pairs"]:
                raise RuntimeError("Canonical target count exceeded the unchanged pool")
            atomic_json(self.state_path, state)

    def finish(self, task):
        with self.locked():
            state = self.snapshot()
            current = self.require_owner(state, task)
            if current["completed_rows"] != current["end"]-current["start"]:
                raise RuntimeError("Unfinished task cannot be committed complete")
            current.update(state="complete", completed_unix=time.time())
            atomic_json(self.task_path(task["task_id"]), current)
            del state["active"][str(task["owner"])]
            state["completed_tasks"] += 1
            state["updated_unix"] = time.time()
            atomic_json(self.state_path, state)

    @staticmethod
    def complete(state):
        return (state["cached_pairs"] == state["total_pairs"]
                and state["completed_tasks"] == state["total_tasks"]
                and state["next_pending"] == state["pending_task_count"]
                and not state["active"])

    def progress(self):
        state = self.snapshot()
        now = time.time()
        elapsed = max(0.001, now-state["run_started_unix"])
        return dict(mode=state["mode"], run_id=state["run_id"], total_pairs=state["total_pairs"],
                    completed_pairs=state["cached_pairs"], recovered_pairs=state["recovered_pairs"],
                    completion_fraction=state["cached_pairs"]/state["total_pairs"],
                    new_pairs_this_run=state["cached_pairs"]-state["recovered_pairs"],
                    new_pairs_per_second=(state["cached_pairs"]-state["recovered_pairs"])/elapsed,
                    run_elapsed_seconds=elapsed, annotation_elapsed_seconds=now-state["first_annotation_started_unix"] if state["first_annotation_started_unix"] else 0,
                    completed_tasks=state["completed_tasks"], total_tasks=state["total_tasks"],
                    active_tasks=state["active"], new_pairs_by_replica=state["new_pairs_by_replica"],
                    updated_unix=state["updated_unix"])
