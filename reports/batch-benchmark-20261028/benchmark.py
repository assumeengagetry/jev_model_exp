"""Isolated SDPA/LoRA throughput measurements; never saves trained weights."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PROJECT = Path.home() / "assumeengage"
JOINT = PROJECT / "research/lu-joint-confirmation-20261028"
RUN = JOINT / "runs/JOINT/seed-3407"
R0 = PROJECT / "runs/language-baseline-20261027T061200Z"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--profile", choices=("joint-real", "4096-tail"), required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import DataCollatorWithPadding, set_seed
    source = json.loads((JOINT / "active-code.json").read_text())["path"]
    sys.path.insert(0, source)
    os.environ["LU_RESEARCH_ROOT"] = str(RUN)
    from model_utils import load_model
    name = f"{args.profile}-mb{args.microbatch}"
    destination = OUT / f"{name}.json"
    effective = max(128, args.microbatch)
    record = {"profile": args.profile, "microbatch": args.microbatch,
              "effective_batch": effective, "gradient_accumulation": effective // args.microbatch,
              "warmup_updates": args.warmup, "measured_updates": args.steps,
              "backend": "transformers_sdpa_reference", "precision": "bf16",
              "checkpoint": str(R0 / "checkpoints/final"),
              "comparison": "fixed effective batch 128" if effective == 128 else "throughput/capacity only; effective batch differs"}
    try:
        torch.set_num_threads(8)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        set_seed(3407)
        d0 = load_from_disk(str(R0 / "prepared-data/train"))
        d2 = load_from_disk(str(PROJECT / "research/lu-target-expansion-20261027/D2-final-targets/train"))
        pools = [d0, d2]
        schedule = np.load(RUN / "schedule.npz", allow_pickle=False)
        updates = args.warmup + args.steps
        batches = []
        if args.profile == "joint-real":
            starts = np.linspace(0, len(schedule["index"]) - effective, updates, dtype=np.int64)
            for start in starts:
                start = int(start) // 128 * 128
                rows = [pools[int(schedule["pool"][i])][int(schedule["index"][i])]
                        for i in range(start, start + effective)]
                batches.append(rows)
            record["data_policy"] = "Identical deterministic real schedule windows for mb16/32/64/128; original row order retained; dynamic padding, no packing or new truncation"
        else:
            # Arrow avoids the optional torchvision VideoReader import in the
            # datasets NumPy formatter; no environment/package changes needed.
            table = d0.select_columns(["length"]).with_format("arrow")[:]
            lengths = table.column("length").combine_chunks().to_numpy(zero_copy_only=False)
            indices = np.flatnonzero(lengths == 4096)
            if not len(indices):
                raise RuntimeError("No actual 4096-token rows available")
            for update in range(updates):
                batches.append([d0[int(indices[(update * effective + i) % len(indices)])]
                                for i in range(effective)])
            record["data_policy"] = "Real cached 4096-token rows; tail capacity stress test, NOT representative of mean training length"
        model, tokenizer = load_model(R0 / "checkpoints/final", mode="causal", device=0, trainable=True)
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.01, fused=True)
        collate = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        prepared = [[collate([{k: row[k] for k in ("input_ids", "attention_mask", "labels")}
                              for row in rows[start:start + args.microbatch]])
                     for start in range(0, effective, args.microbatch)] for rows in batches]
        seconds, real_tokens, padded_tokens = [], 0, 0
        losses = []
        for update, cpu_batches in enumerate(prepared):
            if update == args.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for cpu in cpu_batches:
                batch = {k: value.cuda(non_blocking=True) for k, value in cpu.items()}
                labels = batch.pop("labels")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**batch).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum") / effective
                loss.backward()
                loss_value += float(loss.detach())
                del batch, labels, logits, loss
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            if update >= args.warmup:
                seconds.append(duration)
                losses.append(loss_value)
                real_tokens += sum(int(batch["attention_mask"].sum()) for batch in cpu_batches)
                padded_tokens += sum(batch["attention_mask"].numel() for batch in cpu_batches)
        record.update(state="complete", samples_per_second=effective * args.steps / sum(seconds),
                      nonpadding_tokens_per_second=real_tokens / sum(seconds),
                      padded_tokens_per_second=padded_tokens / sum(seconds),
                      padding_fraction=1 - real_tokens / padded_tokens,
                      update_seconds=seconds, median_update_seconds=statistics.median(seconds),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                      measured_nonpadding_tokens=real_tokens, losses=losses,
                      gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda)
    except torch.cuda.OutOfMemoryError as error:
        record.update(state="oom", error=str(error))
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
