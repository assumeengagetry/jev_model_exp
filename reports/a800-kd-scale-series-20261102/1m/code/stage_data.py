#!/usr/bin/env python3
"""Freeze nested scale milestones and materialize only fully labeled snapshots."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from kd_runtime import SOURCE_ROOT, WORKSPACE, POOL, DOMAINS, PARENT, legacy, atomic_json

PILOT = SOURCE_ROOT / "pilots/idea-100k-20261102"
SERIES = SOURCE_ROOT / "staged/scale-series-20261102"
sys.path.append(str(PILOT / "code"))
from prepare_pilot import allocate


def stage_root(size):
    return SERIES / "stages" / f"{size//1000000}m"


def select_rows(dataset, domain, candidates, count, offset, rng):
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == len(candidates):
        return candidates.astype(np.int64)+offset
    source = dataset.tables[domain]["source"][candidates].astype(np.int64)
    labels = (dataset.tables[domain]["target"][candidates] >= 0.5).astype(np.int64)
    codes = source*2+labels
    order = np.argsort(codes, kind="stable")
    keys, starts, capacities = np.unique(codes[order], return_index=True, return_counts=True)
    amounts = allocate(capacities, count, rng, minimum_one=True)
    chosen = []
    for begin, capacity, amount in zip(starts, capacities, amounts):
        if amount:
            rows = candidates[order[int(begin):int(begin+capacity)]]
            chosen.append(rng.choice(rows, size=int(amount), replace=False)+offset)
    return np.concatenate(chosen).astype(np.int64)


def define():
    marker = SERIES / "definitions.json"
    if marker.exists():
        print(json.dumps({"event":"stage_definitions_reused", **json.loads(marker.read_text())}), flush=True)
        return
    dataset = legacy.PairPool("MIXED_ALL", root=POOL)
    probabilities = np.load(SOURCE_ROOT / "outputs/teacher/probabilities.npy", mmap_mode="r")
    finite = np.isfinite(probabilities).all(axis=1)
    used = np.zeros(len(dataset), dtype=np.bool_)
    pilot_ids = np.load(PILOT / "selection/train_indices.npy")
    used[pilot_ids] = True
    rng = np.random.default_rng(20261102)
    counts = [len(dataset.tables[d]["state"]) for d in DOMAINS]
    definitions = []
    for size in [1000000,2000000,3000000]:
        destination = stage_root(size)
        for name in ["selection","outputs/teacher","outputs/evaluation","logs"]:
            (destination/name).mkdir(parents=True, exist_ok=True)
        if not (destination/"code").exists():
            (destination/"code").symlink_to(SERIES/"code",target_is_directory=True)
        targets = allocate(counts, size, rng)
        offset = 0
        domain_meta = {}
        for domain, total, target in zip(DOMAINS, counts, targets):
            local_used = used[offset:offset+total]
            need = int(target)-int(local_used.sum())
            available = np.flatnonzero(finite[offset:offset+total] & ~local_used)
            chosen = select_rows(dataset, domain, available, min(need,len(available)), offset, rng)
            used[chosen] = True
            need -= len(chosen)
            if need:
                available = np.flatnonzero(~finite[offset:offset+total] & ~used[offset:offset+total])
                chosen = select_rows(dataset, domain, available, need, offset, rng)
                used[chosen] = True
            domain_meta[domain] = int(target)
            offset += total
        ids = np.flatnonzero(used).astype(np.int64)
        if len(ids) != size:
            raise RuntimeError("Nested sample size differs from the declared milestone")
        rng.shuffle(ids)
        np.save(destination / "selection/train_indices.npy", ids)
        ready = int(finite[ids].sum())
        value = dict(complete=True, scope="staged_scale_experiment", pairs=size,
                     domains=domain_meta, seed=20261102, nested_pilot_pairs=len(pilot_ids),
                     initial_labeled_pairs=ready, initial_missing_teacher_pairs=size-ready,
                     effective_training_tokens=int(dataset.fetch(ids)["length"].astype(np.uint64).sum()),
                     parent=str(PARENT), teacher="PelaAI/KnowLine-31B-Gen1", epochs=1,
                     sampling="Nested; full-pool domain proportions; prefer cached rows, then source/original-label strata",
                     selected_by_teacher_confidence=False, created_unix=time.time())
        atomic_json(destination / "selection/train-manifest.json", value)
        shutil.copy2(PILOT / "selection/evaluation.json", destination / "selection/evaluation.json")
        baseline = destination / "outputs/evaluation/BASELINE"
        if not baseline.exists():
            baseline.symlink_to(PILOT / "outputs/evaluation/BASELINE", target_is_directory=True)
        definitions.append(dict(size=size,root=str(destination),initial_labeled_pairs=ready,
                                initial_missing_teacher_pairs=size-ready,domain_pairs=domain_meta))
        print(json.dumps({"event":"stage_definition_ready", **definitions[-1]}), flush=True)
    for size in [2000000,3000000]:
        root = SERIES / "producer" / f"{size//1000000}m"
        (root / "outputs/teacher").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        link = root / "outputs/teacher/probabilities.npy"
        if not link.exists():
            link.symlink_to(SOURCE_ROOT / "outputs/teacher/probabilities.npy")
        atomic_json(root / "scope.json", dict(target_indices=str(stage_root(size)/"selection/train_indices.npy"),
                    size=size,global_cache_rows=len(dataset),teacher="PelaAI/KnowLine-31B-Gen1",
                    initial_labeled_pairs=next(x["initial_labeled_pairs"] for x in definitions if x["size"]==size)))
    atomic_json(marker,dict(complete=True,milestones=definitions,initial_global_labeled_pairs=int(finite.sum()),
                upper_bound_pairs=3000000,teacher="PelaAI/KnowLine-31B-Gen1",student_parent=str(PARENT)))


def materialize(size):
    destination = stage_root(size)
    marker = destination / "outputs/teacher/complete.json"
    if marker.exists():
        print(json.dumps({"event":"stage_snapshot_reused","pairs":size}), flush=True)
        return True
    ids = np.load(destination / "selection/train_indices.npy", mmap_mode="r")
    probabilities = np.load(SOURCE_ROOT / "outputs/teacher/probabilities.npy", mmap_mode="r")
    snapshot = np.asarray(probabilities[ids], dtype=np.float32)
    ready = int(np.isfinite(snapshot).all(axis=1).sum())
    if ready != len(ids):
        print(json.dumps({"event":"stage_waiting_teacher","pairs":size,"ready_pairs":ready,"missing_pairs":size-ready}), flush=True)
        return False
    path = destination / "outputs/teacher/probabilities.npy"
    temporary = path.with_suffix(".writing.npy")
    np.save(temporary, snapshot)
    temporary.replace(path)
    atomic_json(marker,dict(complete=True,pairs=size,model="PelaAI/KnowLine-31B-Gen1",temperature=1,
                columns=["P(not_match)","P(match)"],reused_cache=True,new_teacher_calls=0,
                source_cache=str(SOURCE_ROOT/"outputs/teacher/probabilities.npy"),
                note="Snapshot creation makes no Teacher calls; upstream producer time is reported separately"))
    print(json.dumps({"event":"stage_training_snapshot_ready","pairs":size,"probability_rows":len(snapshot)}), flush=True)
    return True


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--define",action="store_true");p.add_argument("--materialize",type=int)
    args=p.parse_args()
    if args.define:define()
    if args.materialize is not None:
        raise SystemExit(0 if materialize(args.materialize) else 2)
