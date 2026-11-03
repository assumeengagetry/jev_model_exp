#!/usr/bin/env python3
"""Independent official binary probabilities with resumable state-local work claims."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import aiohttp
import numpy as np
import pyarrow.parquet as pq

from kd_runtime import ROOT, WORKSPACE, POOL, DOMAINS, TokenStore, atomic_json, legacy, pool
from pair_data_common import render_pair
from annotation_queue import AnnotationQueue

CACHE = ROOT / "outputs/teacher"


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    dataset = pool()
    order_path = CACHE / "annotation-order.npy"
    if not order_path.exists():
        scope = json.loads((ROOT / "scope.json").read_text())
        canonical = np.load(scope["target_indices"])
        states = np.concatenate([dataset.tables[d]["state"] for d in DOMAINS])[canonical]
        order = canonical[np.argsort(states, kind="stable")].astype(np.uint32)
        np.save(order_path, order)
    path = CACHE / "probabilities.npy"
    if not path.exists():
        raise RuntimeError("Scoped producer requires the existing shared canonical probability-file symlink")
    order = np.load(order_path, mmap_mode="r")
    probabilities = np.load(path, mmap_mode="r+")
    recovered = AnnotationQueue(CACHE).prepare(dataset, order, probabilities)
    which = np.searchsorted(dataset.ends, order, side="right")
    atomic_json(CACHE / "manifest.json", dict(model="PelaAI/KnowLine-31B-Gen1", precision="BF16", domains=DOMAINS,
                domain_pairs={d:int((which==i).sum()) for i,d in enumerate(DOMAINS)}, pairs=len(order),global_cache_pairs=len(dataset),
                row_order="Concatenated domain arrays in domains order; unchanged within each domain",
                columns=["P(not_match)", "P(match)"], temperature=1, frontend="Official knowline_server.py",
                question_type="noul", read_twice=True, independent_pairs=True,
                scheduling="dynamic-state-local-chunks-v1", target_chunk_pairs=4096,
                input_format="Original candidate-pair-v1; same head-tail policy for training pairs over 4096 Student tokens"))
    print(json.dumps({"event": "teacher_cache_prepared", "pairs": len(order), "domains": DOMAINS,
                      "recovered_pairs": recovered["recovered_pairs"], "pending_tasks": recovered["pending_task_count"],
                      "total_tasks": recovered["total_tasks"]}), flush=True)


class RawTexts:
    def __init__(self):
        self.states = []
        self.candidates = []
        for folder in [WORKSPACE / "cache/pair-training/normalized/typed-full", POOL]:
            self.states.append(pq.read_table(folder / "inputs.parquet", columns=["state", "question", "criteria"]).combine_chunks())
            self.candidates.append(pq.read_table(folder / "candidates.parquet", columns=["candidate"])["candidate"].combine_chunks())
        self.prefix_boundary = self.states[0].num_rows
        self.suffix_boundary = len(self.candidates[0])
        self.tokenizer = None
        self.store = None

    def pair(self, state_id, candidate_id, length):
        state_stream = int(state_id >= self.prefix_boundary)
        row = int(state_id) - (self.prefix_boundary if state_stream else 0)
        table = self.states[state_stream]
        state, question, criteria = (table[key][row].as_py() for key in ["state", "question", "criteria"])
        stream = int(candidate_id >= self.suffix_boundary)
        local = int(candidate_id) - (self.suffix_boundary if stream else 0)
        candidate = self.candidates[stream][local].as_py()
        if length >= 4096:
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(WORKSPACE / "models/Qwen3-0.6B-common-pair-init-r32", local_files_only=True)
                self.store = TokenStore()
            tokens = legacy.assemble(self.store, state_id, candidate_id, 4096, self.tokenizer.eos_token_id)
            return self.tokenizer.decode(tokens[:-1], skip_special_tokens=False)
        prefix, suffix = render_pair(state, question, candidate, criteria)
        return prefix + suffix


class CacheOwnershipError(RuntimeError):
    pass


async def annotate(replica, concurrency):
    dataset = pool()
    order = np.load(CACHE / "annotation-order.npy", mmap_mode="r")
    probabilities = np.load(CACHE / "probabilities.npy", mmap_mode="r+")
    queue = AnnotationQueue(CACHE)
    run_id = queue.snapshot()["run_id"]
    texts = RawTexts()
    endpoint = f"http://127.0.0.1:{18100 + replica}/v1/systemone"
    started = time.monotonic()
    new_total = reused_total = completed_tasks = 0
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=900)
    connector = aiohttp.TCPConnector(limit=concurrency)

    def progress(task_id=None):
        value = dict(mode="dynamic-state-local-chunks-v1", run_id=run_id, replica=replica,
                     new_pairs_written=new_total, reused_pairs=reused_total, completed_tasks=completed_tasks,
                     active_task=task_id, elapsed_seconds=time.monotonic()-started,
                     new_pairs_per_second=new_total/max(0.001, time.monotonic()-started))
        atomic_json(CACHE / f"replica-{replica}-progress.json", value)
        print(json.dumps({"event": "teacher_progress", **value}), flush=True)
        return value

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def score(index, state_id, candidate_id, length):
            async with semaphore:
                if np.isfinite(probabilities[index]).all():
                    return False
                state = texts.pair(state_id, candidate_id, length)
                payload = {"model": "m", "state": state, "questions": {"pair": {
                    "type": "noul",
                    "instructions": "Is the candidate a correct match for the input, question, and supplied criteria? Judge this pair independently. Answer Yes for match and No for not_match.",
                    "criteria": {"true": "The candidate is a correct match.", "false": "The candidate is not a correct match."}}}}
                for attempt in range(6):
                    try:
                        async with session.post(endpoint, json=payload) as response:
                            if response.status != 200:
                                raise RuntimeError(f"Official frontend returned HTTP {response.status}")
                            answer = await response.json()
                        yes = float(answer["answers"]["pair"]["noul"])
                        if not np.isfinite(yes) or not 0 <= yes <= 1:
                            raise RuntimeError("Official frontend returned an invalid binary probability")
                        if np.isfinite(probabilities[index]).all():
                            raise CacheOwnershipError(f"Canonical pair {index} was completed by another writer")
                        probabilities[index] = (1.0 - yes, yes)
                        return True
                    except CacheOwnershipError:
                        raise
                    except Exception as error:
                        print(json.dumps({"event": "teacher_request_retry", "replica": replica,
                                          "canonical_index": index, "attempt": attempt+1,
                                          "error_type": type(error).__name__}), flush=True)
                        if attempt == 5:
                            raise RuntimeError(f"Teacher failed on canonical pair {index}; no pair dropped") from None
                        await asyncio.sleep(min(30, 2 ** attempt))

        while True:
            task = queue.claim(replica, run_id)
            if task is None:
                break
            indices = np.asarray(order[task["start"]:task["end"]], dtype=np.int64)
            already_done = np.isfinite(probabilities[indices]).all(axis=1)
            reused = int(already_done.sum())
            queue.commit(task, reused_pairs=reused)
            reused_total += reused
            missing = indices[~already_done]
            batch = dataset.fetch(missing)
            # Create tasks in sorted-position order before as_completed can reorder results.
            tasks = [asyncio.create_task(score(int(index), int(state), int(candidate), int(length)))
                     for index, state, candidate, length in zip(missing, batch["state"], batch["candidate"], batch["length"])]
            pending_new = pending_reused = 0
            try:
                for result in asyncio.as_completed(tasks):
                    if await result:
                        pending_new += 1
                    else:
                        pending_reused += 1
                    if pending_new+pending_reused >= 512:
                        probabilities.flush()
                        queue.commit(task, new_pairs=pending_new, reused_pairs=pending_reused)
                        new_total += pending_new
                        reused_total += pending_reused
                        pending_new = pending_reused = 0
                        progress(task["task_id"])
                probabilities.flush()
                queue.commit(task, new_pairs=pending_new, reused_pairs=pending_reused)
                new_total += pending_new
                reused_total += pending_reused
                if not np.isfinite(probabilities[indices]).all():
                    raise RuntimeError("Task still has unfinished probabilities; no task committed complete")
                queue.finish(task)
                completed_tasks += 1
                progress()
            except BaseException:
                for pending in tasks:
                    if not pending.done():
                        pending.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                probabilities.flush()
                raise
    value = progress()
    atomic_json(CACHE / f"replica-{replica}-complete.json", dict(complete=True, **value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--replica", type=int, choices=[0, 1])
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        asyncio.run(annotate(args.replica, args.concurrency))


if __name__ == "__main__":
    main()
