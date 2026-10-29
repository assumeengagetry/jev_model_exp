"""Reconstruct real profiles and measure checkpoint memory/time at explicit lengths."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = Path.home() / "assumeengage"
PILOT = PROJECT / "research/lu-batch-lr-pilot-20261029"
R0 = PROJECT / "runs/language-baseline-20261027T061200Z"
REFERENCE = PROJECT / "research/lu-joint-confirmation-20261028/runs/JOINT/seed-3407"
LENGTHS = [256, 512, 1024, 2048, 4096, 8192]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded(length):
    return (length + 7) // 8 * 8


def profile_stats(rows, microbatch=16):
    import numpy as np
    lengths = [len(row["input_ids"]) for row in rows]
    batches = []
    for start in range(0, len(rows), microbatch):
        current = lengths[start:start + microbatch]
        batches.append({"shape": [len(current), rounded(max(current))],
                        "actual_lengths": current, "nonpadding_tokens": sum(current)})
    padded = sum(batch["shape"][0] * batch["shape"][1] for batch in batches)
    return {"pairs": len(rows), "microbatch": microbatch, "effective_batch": len(rows),
            "accumulation": len(batches), "length_min": min(lengths), "length_max": max(lengths),
            "length_mean": sum(lengths) / len(rows), "length_p50": float(np.median(lengths)),
            "length_p90": float(np.percentile(lengths, 90)), "lengths": lengths,
            "nonpadding_tokens": sum(lengths), "padded_tokens": padded,
            "padding_fraction": 1 - sum(lengths) / padded, "microbatches": batches}


def fixed_padding_stats(stats, length):
    if length is None:
        return stats
    assert length >= stats["length_max"] and length % 8 == 0
    for batch in stats["microbatches"]:
        batch["shape"][1] = length
    stats["padded_tokens"] = stats["pairs"] * length
    stats["padding_fraction"] = 1 - stats["nonpadding_tokens"] / stats["padded_tokens"]
    return stats


def encode_pair(pair, length, prefix, middle, eos):
    assert isinstance(eos, int)
    premise, hypothesis = pair["premise_ids"], pair["hypothesis_ids"]
    room = length - len(prefix) - len(middle) - len(hypothesis) - 1
    if room < 0 or len(premise) < room:
        raise ValueError("Source pair cannot fill requested length while preserving hypothesis")
    head = (room + 1) // 2
    tail = room // 2
    body = premise[:head] + (premise[-tail:] if tail else [])
    ids = prefix + body + middle + hypothesis + [eos]
    assert len(ids) == length
    return {"input_ids": ids, "attention_mask": [1] * length, "labels": pair["label"]}


def prepare():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    sys.path.insert(0, str(PROJECT))
    from language_baseline.common import PairEncoder

    table = pa.ipc.open_file(pa.memory_map(str(PILOT / "pool-1.arrow"), "r")).read_all()
    lengths = table["length"].to_numpy()
    selections = {"medium": np.flatnonzero((lengths >= 192) & (lengths <= 384))[:32],
                  "long_real_pairs": np.argsort(lengths, kind="stable")[-32:]}
    profiles = {}
    old_measurements = json.loads((PILOT / "technical/performance.json").read_text())["measurements"]
    for name, indices in selections.items():
        view = table.take(pa.array(indices))
        rows = view.select(["input_ids", "attention_mask", "labels"]).to_pylist()
        profiles[name] = {**profile_stats(rows), "source_row_indices": indices.tolist(),
                          "tasks": dict(Counter(view["task"].to_pylist())),
                          "selection": "first32 matching192<=length<=384" if name == "medium" else "longest32, stable ascending argsort",
                          "original_measurements": [row for row in old_measurements if row["profile"] == name]}
        write(ROOT / f"inputs-{name}.json", rows)
    write(ROOT / "original-profiles.json", {"time": now(), "source": str(PILOT / "pool-1.arrow"),
          "source_sha256": sha(PILOT / "pool-1.arrow"), "source_worker_sha256": sha(PILOT / "worker.py"), "profiles": profiles})

    tokenizer = AutoTokenizer.from_pretrained(R0 / "checkpoints/final", local_files_only=True)
    encoder = PairEncoder(tokenizer)
    ds = load_from_disk(str(R0 / "prepared-data/validation"))
    lengths = ds.select_columns(["original_length"]).with_format("arrow")[:]["original_length"].to_numpy()
    indices = np.flatnonzero((lengths >= 8192) & (lengths <= 40960))
    raw = pq.read_table(PROJECT / "datasets/tasksource/zero-shot-label-nli/data/validation-00000-of-00001.parquet")
    pairs = []
    for index in indices:
        row = ds[int(index)]
        original = raw.slice(int(row["raw_index"]), 1).to_pylist()[0]
        p = tokenizer.encode(original["premise"], add_special_tokens=False)
        h = tokenizer.encode(original["hypothesis"], add_special_tokens=False)
        overhead = len(encoder.prefix) + len(encoder.middle) + len(h) + 1
        if overhead > 256 or len(p) + overhead < 8192 or tokenizer.pad_token_id in p + h:
            continue
        label = {0: 0, 2: 1}[original["labels"]]
        assert label == row["labels"]
        pairs.append({"row": int(index), "raw_index": int(row["raw_index"]), "task": row["task"],
                      "label": label, "original_length": len(p) + overhead, "premise_ids": p, "hypothesis_ids": h})
        if len(pairs) == 32:
            break
    assert len(pairs) == 32, "Need32 intact long real source pairs"
    payload = {"pairs": pairs, "prefix": encoder.prefix, "middle": encoder.middle, "eos": encoder.eos[0]}
    for length in LENGTHS:
        rows = [encode_pair(pair, length, encoder.prefix, encoder.middle, encoder.eos[0]) for pair in pairs]
        assert all(isinstance(token, int) for row in rows for token in row["input_ids"])
        assert profile_stats(rows)["padding_fraction"] == 0
    write(ROOT / "source-pairs.json", payload)
    manifest = {"time": now(), "state": "prepared", "checkpoint": str(R0 / "checkpoints/final"),
                "checkpoint_sha256": sha(R0 / "checkpoints/final/adapter_model.safetensors"),
                "model": "Qwen3-0.6B sequence classification, LoRA r16/alpha32 plus trained score head",
                "attention": "causal, transformers SDPA, use_cache=False", "precision": "BF16 autocast, loader-original parameter dtypes",
                "main_microbatch": 16, "effective_batch": 32, "accumulation": 2,
                "lengths": LENGTHS, "independent_process_repeats": 2, "warmup_effective_batches": 1,
                "timed_effective_batches_per_repeat": 3, "optimizer_updates": 0, "weight_saves": 0,
                "timing": "CPU collation, H2D, forward, CE loss, backward, gradient zeroing; synchronized; no optimizer steps",
                "construction": "Same32 real raw-text long pairs at every cap; whole hypothesis, premise head/tail; each dense sequence exactly cap tokens, zero padding",
                "source_tasks": dict(Counter(pair["task"] for pair in pairs)),
                "sources": [{k: v for k, v in pair.items() if k not in ("premise_ids", "hypothesis_ids")} for pair in pairs],
                "source_pairs_sha256": sha(ROOT / "source-pairs.json"),
                "loader_sha256": sha(PILOT / "code/model_utils.py"),
                "additional": "Fresh-process replay of both original profiles; microbatch8/32 anchors at1024/2048; OOM refinement on128-token grid",
                "gpu_requested": 0, "total_timebox_minutes": 110}
    write(ROOT / "manifest.json", manifest)
    print(json.dumps({"event": "inputs_prepared", "profiles": {k: {f: v[f] for f in ("length_min", "length_max", "nonpadding_tokens", "padded_tokens", "padding_fraction")} for k, v in profiles.items()}, "fixed_lengths": LENGTHS}), flush=True)


def case_name(kind, length, enabled, microbatch, repeat):
    return f"{kind}-L{length}-B{microbatch}-{'on' if enabled else 'off'}-r{repeat}"


def worker(args):
    import importlib.metadata
    import torch
    from transformers import DataCollatorWithPadding, set_seed

    os.environ["LU_RESEARCH_ROOT"] = str(REFERENCE)
    sys.path.insert(0, str(PILOT / "code"))
    from model_utils import load_model
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(3407)
    if args.kind in ("medium", "long_real_pairs"):
        rows = json.loads((ROOT / f"inputs-{args.kind}.json").read_text())
    else:
        source = json.loads((ROOT / "source-pairs.json").read_text())
        rows = [encode_pair(pair, args.length, source["prefix"], source["middle"], source["eos"]) for pair in source["pairs"]]
    inputs_stats = fixed_padding_stats(profile_stats(rows, args.microbatch), args.padding_length)
    enabled = args.checkpoint == "on"
    result = {"state": "running", "time": now(), "kind": args.kind, "length": args.length,
              "checkpointing": enabled, "microbatch": args.microbatch, "repeat": args.repeat,
              "input": inputs_stats, "pid": os.getpid(), "optimizer_updates": 0,
              "fixed_padding_length": args.padding_length, "all_ones_mask_control": args.all_ones_mask}
    write(args.output, result)
    started = time.perf_counter()
    model, tokenizer = load_model(R0 / "checkpoints/final", mode="causal", device=0, trainable=True)
    if enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.train()
    assert bool(model.is_gradient_checkpointing) == enabled
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in name or ".score." in name for name in params)
    prop = torch.cuda.get_device_properties(0)
    dtype_counts = Counter()
    for p in model.parameters():
        dtype_counts[str(p.dtype)] += p.numel()
    result["environment"] = {"gpu_name": prop.name, "gpu_total_GiB": prop.total_memory / 2**30,
                             "torch": torch.__version__, "cuda": torch.version.cuda,
                             "transformers": importlib.metadata.version("transformers"), "peft": importlib.metadata.version("peft"),
                             "parameter_count": sum(p.numel() for p in model.parameters()),
                             "trainable_parameter_count": sum(p.numel() for p in params.values()),
                             "parameters_by_dtype": dict(dtype_counts),
                             "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
                             "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                             "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                             "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled()}
    collate = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if args.padding_length is None else DataCollatorWithPadding(
        tokenizer, padding="max_length", max_length=args.padding_length, pad_to_multiple_of=8)
    progress = {"phase": "warmup", "microbatch_index": 0, "operation": "initializing"}

    def effective_batch(events=False):
        model.zero_grad(set_to_none=True)
        timings = []
        losses = []
        for offset in range(0, len(rows), args.microbatch):
            progress["microbatch_index"] = offset // args.microbatch
            batch = collate(rows[offset:offset + args.microbatch])
            progress["operation"] = "H2D"
            labels = batch.pop("labels").cuda()
            batch = {key: value.cuda() for key, value in batch.items()}
            if args.all_ones_mask:
                batch["attention_mask"].fill_(1)
            if events:
                begin, fwd, bwd = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                begin.record()
            progress["operation"] = "forward"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum") / len(rows)
            if events:
                fwd.record()
            progress["operation"] = "backward"
            loss.backward()
            if events:
                bwd.record()
                timings.append((begin, fwd, bwd))
            losses.append(float(loss.detach()))
            del loss, logits, labels, batch
        return sum(losses), timings

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        effective_batch()
        for key, text in (("head", ".score."), ("lora", "lora_B")):
            matching = [p for name, p in params.items() if text in name and p.grad is not None]
            assert matching and all(bool(torch.isfinite(p.grad).all()) for p in matching)
            assert any(bool(p.grad.abs().sum() > 0) for p in matching)
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        result["baseline_allocated_GiB"] = torch.cuda.memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        wall_seconds = forward_ms = backward_ms = 0.0
        losses = []
        progress["phase"] = "measured"
        for step in range(3):
            progress["timed_step"] = step
            torch.cuda.synchronize()
            clock = time.perf_counter()
            loss, events = effective_batch(events=True)
            torch.cuda.synchronize()
            wall_seconds += time.perf_counter() - clock
            for begin, fwd, bwd in events:
                forward_ms += begin.elapsed_time(fwd)
                backward_ms += fwd.elapsed_time(bwd)
            losses.append(loss)
        result.update(state="complete", ok=True, seconds=wall_seconds,
                      seconds_per_effective_batch=wall_seconds / 3,
                      forward_seconds=forward_ms / 1000, backward_seconds=backward_ms / 1000,
                      nonpadding_tokens_per_second=inputs_stats["nonpadding_tokens"] * 3 / wall_seconds,
                      padded_tokens_per_second=inputs_stats["padded_tokens"] * 3 / wall_seconds,
                      peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                      losses=losses, checked_finite_gradients=True)
        if args.profile_kernel or (args.kind == "fixed" and args.length == 256 and args.repeat == 0 and not enabled):
            progress["phase"] = "kernel_profile"
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                effective_batch()
            result["attention_kernel_ops"] = sorted({item.key for item in prof.key_averages()
                                                    if "scaled_dot_product" in item.key or "flash_attention" in item.key})
    except torch.cuda.OutOfMemoryError as exc:
        result.update(state="oom", ok=False, oom_at=dict(progress), error=str(exc),
                      peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)
    result.update(time=now(), process_wall_seconds=time.perf_counter() - started)
    write(args.output, result)
    print(json.dumps({"event": "case_complete", **{key: result.get(key) for key in
          ("kind", "length", "microbatch", "checkpointing", "repeat", "state", "peak_allocated_GiB", "nonpadding_tokens_per_second", "oom_at")}}), flush=True)


def wait_free(gpu, deadline):
    while time.monotonic() < deadline:
        response = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=10)
        if not any(line.strip().isdigit() for line in response.splitlines()):
            return
        write(ROOT / "status.json", {"state": "waiting_for_free_gpu", "gpu": gpu, "time": now()})
        time.sleep(15)
    raise TimeoutError("GPU wait exceeded timebox")


def controller(args):
    started = time.monotonic()
    deadline = started + 110 * 60
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "manifest.json").exists():
        prepare()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert sha(R0 / "checkpoints/final/adapter_model.safetensors") == manifest["checkpoint_sha256"]
    records = []

    def run(kind, length, enabled, microbatch=16, repeat=0):
        name = case_name(kind, length, enabled, microbatch, repeat)
        output = ROOT / "cases" / f"{name}.json"
        if output.exists():
            existing = json.loads(output.read_text())
            if existing.get("state") in ("complete", "oom"):
                records.append(existing)
                return existing
            raise RuntimeError(f"Incomplete existing cell requires inspection: {output}")
        wait_free(args.gpu, deadline)
        write(ROOT / "status.json", {"state": "benchmarking", "time": now(), "case": name,
                                     "completed_cells": len(records), "gpu": args.gpu})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="8",
                   TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--kind", kind,
                   "--length", str(length), "--microbatch", str(microbatch), "--repeat", str(repeat),
                   "--checkpoint", "on" if enabled else "off", "--output", str(output)]
        log = ROOT / "worker-logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("x") as stream:
            process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=min(600, max(1, deadline - time.monotonic())))
        if process.returncode != 0:
            raise RuntimeError(f"Cell failed; preserved diagnostic log: {log}")
        row = json.loads(output.read_text())
        assert row["state"] in ("complete", "oom")
        records.append(row)
        write(ROOT / "results.json", records)
        print(json.dumps({"event": "checkpoint_scaling_cell", "case": name, "state": row["state"],
                          "peak_allocated_GiB": row.get("peak_allocated_GiB"),
                          "tokens_per_second": row.get("nonpadding_tokens_per_second")}), flush=True)
        return row

    for length in LENGTHS:
        for repeat in range(2):
            for enabled in ([True, False] if repeat == 0 else [False, True]):
                run("fixed", length, enabled, repeat=repeat)
    thresholds = {}
    for enabled in (False, True):
        grouped = {length: [row for row in records if row["kind"] == "fixed" and row["length"] == length
                            and row["checkpointing"] == enabled] for length in LENGTHS}
        failed = [length for length, rows in grouped.items() if any(row["state"] == "oom" for row in rows)]
        if not failed:
            thresholds[str(enabled)] = {"passed_through": max(LENGTHS), "first_oom": None, "grid_tokens": 128}
            continue
        high = min(failed)
        passed = [length for length, rows in grouped.items() if length < high and all(row["state"] == "complete" for row in rows)]
        low = max(passed, default=0)
        while high - low > 128:
            middle = ((low + high) // 2) // 128 * 128
            row = run("threshold", middle, enabled)
            if row["state"] == "complete":
                low = middle
            else:
                high = middle
        thresholds[str(enabled)] = {"passed_through": low, "first_oom": high, "grid_tokens": 128}
    write(ROOT / "oom-thresholds.json", thresholds)
    for profile in ("medium", "long_real_pairs"):
        for enabled in (True, False):
            run(profile, 0, enabled)
    for microbatch in (8, 32):
        for length in (1024, 2048):
            for enabled in (True, False):
                run("batch-anchor", length, enabled, microbatch=microbatch)
    assert sha(R0 / "checkpoints/final/adapter_model.safetensors") == manifest["checkpoint_sha256"]
    render(records)
    write(ROOT / "status.json", {"state": "complete", "time": now(), "completed_cells": len(records),
                                 "wall_seconds": time.monotonic() - started, "gpu0_released": True,
                                 "report": str(ROOT / "REPORT.md")})
    print(json.dumps({"event": "checkpoint_length_scaling_complete", "cells": len(records),
                      "report": str(ROOT / "REPORT.md"), "gpu0_released": True}), flush=True)


def aggregate(rows):
    successful = [row for row in rows if row["state"] == "complete"]
    if not successful:
        return {"state": "OOM", "successful_repeats": 0, "oom_repeats": len(rows)}
    total_seconds = sum(row["seconds"] for row in successful)
    tokens = sum(row["input"]["nonpadding_tokens"] * 3 for row in successful)
    return {"state": "OK" if len(successful) == len(rows) else "mixed",
            "successful_repeats": len(successful), "oom_repeats": len(rows) - len(successful),
            "peak_allocated_GiB": max(row["peak_allocated_GiB"] for row in successful),
            "peak_reserved_GiB": max(row["peak_reserved_GiB"] for row in successful),
            "tokens_per_second": tokens / total_seconds,
            "seconds_per_effective_batch": total_seconds / (3 * len(successful)),
            "forward_seconds_per_effective_batch": sum(row["forward_seconds"] for row in successful) / (3 * len(successful)),
            "backward_seconds_per_effective_batch": sum(row["backward_seconds"] for row in successful) / (3 * len(successful))}


def render(records=None):
    if records is None:
        records = json.loads((ROOT / "results.json").read_text())
    profiles = json.loads((ROOT / "original-profiles.json").read_text())["profiles"]
    manifest = json.loads((ROOT / "manifest.json").read_text())
    thresholds = json.loads((ROOT / "oom-thresholds.json").read_text())
    groups = {}
    for length in LENGTHS:
        groups[str(length)] = {str(enabled): aggregate([row for row in records if row["kind"] == "fixed"
                              and row["length"] == length and row["checkpointing"] == enabled]) for enabled in (True, False)}
    write(ROOT / "summary.json", {"fixed_lengths": groups, "oom_thresholds": thresholds})
    environment = next(row["environment"] for row in records if row.get("environment"))
    lines = ["# Gradient checkpointing：具体输入长度、显存与速度", "",
             "## 原来的medium和long_real_pairs到底多长", "",
             "| 原profile | 实际长度范围 | 均长 | 中位长度 | 两次microbatch的实际shape | 有效tokens / padded tokens | padding | 原On峰值 | 原Off峰值 |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for name, profile in profiles.items():
        old = {enabled: [row for row in profile["original_measurements"] if row["checkpointing"] == enabled] for enabled in (True, False)}
        memory = lambda values: f"{max(row['peak_allocated_GiB'] for row in values):.2f} GiB" if all(row["ok"] for row in values) else "OOM"
        shapes = " + ".join(f"{b['shape'][0]}×{b['shape'][1]}" for b in profile["microbatches"])
        lines.append(f"| {name} | {profile['length_min']}–{profile['length_max']} | {profile['length_mean']:.2f} | {profile['length_p50']:.1f} | {shapes} | {profile['nonpadding_tokens']:,} / {profile['padded_tokens']:,} | {profile['padding_fraction']:.2%} | {memory(old[True])} | {memory(old[False])} |")
    lines += ["", "medium的筛选条件写的是192–384，但实际取的是命中条件的前32对，最终仅192–205；"
              "long_real_pairs取目标池最长32对，全部来自RAGTruth Summary。medium全部来自ANLI r1。"
              "两组都采用物理microbatch16、累计两次形成32对；不是32条同时驻留。",
              "", "## 固定长度阶梯：checkpoint On / Off", "",
              f"设备：{environment['gpu_name']}，CUDA可见容量{environment['gpu_total_GiB']:.2f} GiB。"
              "模型：Qwen3-0.6B分类模型，LoRA r16/alpha32、原R0分类头、BF16、causal SDPA。"
              "物理microbatch固定16，累计2次；每档32条真实长文本裁到相同精确长度，padding为0。"
              "每个组合在两个独立进程中重复，各预热1次、计时3次前向+反向，权重不更新。", "",
              "| 每条sequence tokens | 每个microbatch tokens | On峰值allocated GiB | Off峰值allocated GiB | On有效tokens/s | Off有效tokens/s | On节省显存 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for length in LENGTHS:
        on, off = groups[str(length)]["True"], groups[str(length)]["False"]
        mem = lambda row: f"{row['peak_allocated_GiB']:.2f}" if "peak_allocated_GiB" in row else "OOM"
        rate = lambda row: f"{row['tokens_per_second']:,.0f}" if "tokens_per_second" in row else "—"
        saved = f"{1-on['peak_allocated_GiB']/off['peak_allocated_GiB']:.1%}" if "peak_allocated_GiB" in off and "peak_allocated_GiB" in on else "—"
        lines.append(f"| {length:,} | {16*length:,} | {mem(on)} | {mem(off)} | {rate(on)} | {rate(off)} | {saved} |")
    lines += ["", "### OOM位置", ""]
    for enabled in (False, True):
        info = thresholds[str(enabled)]
        text = f"最长通过{info['passed_through']:,} tokens；下一档{info['first_oom']:,} tokens发生OOM（128-token步长）" if info["first_oom"] else f"已通过全部阶梯，最长{info['passed_through']:,} tokens"
        lines.append(f"- Checkpoint {'On' if enabled else 'Off'}：{text}。")
    lines += ["", "### 两个旧profile的独立进程重测", "",
              "| 输入profile | 开关 | 状态 | 峰值allocated GiB | 有效tokens/s |", "|---|---|---|---:|---:|"]
    for profile in ("medium", "long_real_pairs"):
        for enabled in (True, False):
            value = aggregate([row for row in records if row["kind"] == profile and row["checkpointing"] == enabled])
            memory = f"{value['peak_allocated_GiB']:.2f}" if "peak_allocated_GiB" in value else "—"
            rate = f"{value['tokens_per_second']:,.0f}" if "tokens_per_second" in value else "—"
            lines.append(f"| {profile} | {'On' if enabled else 'Off'} | {value['state']} | {memory} | {rate} |")
    lines += ["", "## 推广到其他batch：看物理microbatch，不看累计后的batch", "",
              "以下把有效batch都固定为32对。microbatch8/16/32分别累计4/2/1次。", "",
              "| sequence tokens | 物理microbatch | On峰值allocated GiB | Off峰值allocated GiB | On有效tokens/s | Off有效tokens/s |",
              "|---|---:|---:|---:|---:|---:|"]
    for length in (1024, 2048):
        for microbatch in (8, 16, 32):
            kind = "fixed" if microbatch == 16 else "batch-anchor"
            on, off = [aggregate([row for row in records if row["kind"] == kind and row["length"] == length and row["microbatch"] == microbatch and row["checkpointing"] == enabled]) for enabled in (True, False)]
            mem = lambda row: f"{row['peak_allocated_GiB']:.2f}" if "peak_allocated_GiB" in row else "OOM"
            rate = lambda row: f"{row['tokens_per_second']:,.0f}" if "tokens_per_second" in row else "—"
            lines.append(f"| {length:,} | {microbatch} | {mem(on)} | {mem(off)} | {rate(on)} | {rate(off)} |")
    lines += ["", "实际选配置时，先查每个microbatch补齐后的sequence长度，再与物理batch一起查上表。"
              "累计次数控制一次更新看多少样本；显存主要由单次驻留的物理batch和补齐长度决定。"
              "混长输入优先按长度分桶，减少短文本被补到长文本长度的开销。", "",
              "## 分配显存与allocator保留量", "",
              "| 长度 | On allocated / reserved GiB | Off allocated / reserved GiB |", "|---|---:|---:|"]
    for length in LENGTHS:
        values = []
        for enabled in (True, False):
            row = groups[str(length)][str(enabled)]
            values.append(f"{row['peak_allocated_GiB']:.2f} / {row['peak_reserved_GiB']:.2f}" if "peak_allocated_GiB" in row else "OOM")
        lines.append(f"| {length:,} | {' | '.join(values)} |")
    kernels = sorted({key for row in records for key in row.get("attention_kernel_ops", [])})
    lines += ["", "## 环境与原始数据", "", f"- Torch {environment['torch']}，CUDA {environment['cuda']}，transformers {environment['transformers']}，PEFT {environment['peft']}。",
              f"- 总参数{environment['parameter_count']:,}，可训练参数{environment['trainable_parameter_count']:,}；参数dtype计数：`{environment['parameters_by_dtype']}`。",
              f"- 实际SDPA算子：`{kernels}`。",
              "- 吞吐计时：CPU拼batch、H2D、前向、分类loss、反向及梯度清零；没有optimizer step。",
              "- 每组都由独立进程启动，显存不继承上一组的allocator缓存。",
              "- `original-profiles.json`保存旧profile全部长度和原始row索引；`source-pairs.json`保存固定阶梯输入；`manifest.json`保存配置和hash。",
              "- `cases/*.json`保存逐次耗时、前向/反向耗时、allocated/reserved、padding、OOM发生位置；`results.json`和`summary.json`为汇总。",
              "- `worker-logs/`为完整进程输出；`benchmark.py`和`test_benchmark.py`可复用。"]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"event": "report_rendered", "path": str(ROOT / "REPORT.md")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--kind", default="fixed")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--checkpoint", choices=("on", "off"), default="on")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--padding-length", type=int)
    parser.add_argument("--profile-kernel", action="store_true")
    parser.add_argument("--all-ones-mask", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.worker:
        worker(args)
    elif args.prepare:
        prepare()
    elif args.render:
        render()
    else:
        try:
            controller(args)
        except Exception as exc:
            write(ROOT / "status.json", {"state": "failed", "time": now(), "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
