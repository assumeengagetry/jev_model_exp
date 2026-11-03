#!/usr/bin/env python3
"""Equal one-epoch LoRA/classification-head continuation for the three KD arms."""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import set_seed

from kd_runtime import ROOT, PARENT, DOMAINS, TokenStore, atomic_json, legacy, load_classifier, pool, shared_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["HARD", "SOFT", "HYBRID"], required=True)
    parser.add_argument("--micro-tokens", type=int, default=32768)
    parser.add_argument("--attention", default="auto")
    args = parser.parse_args()
    name = "KD_" + args.arm
    out = ROOT / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        print("Already complete:", name, flush=True)
        return
    set_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset, store = pool(), TokenStore()
    order = shared_order(dataset)
    macro = 128
    total_steps = math.ceil(len(dataset) / macro)
    probabilities = None
    if args.arm != "HARD":
        if not (ROOT / "outputs/teacher/complete.json").exists():
            raise RuntimeError("Full four-subject Teacher cache must be complete before SOFT/HYBRID")
        probabilities = np.load(ROOT / "outputs/teacher/probabilities.npy", mmap_mode="r")
        if probabilities.shape != (len(dataset), 2):
            raise RuntimeError("Teacher cache does not match the full four-subject pool")
    state = None
    checkpoint = PARENT
    latest = out / "latest.json"
    if latest.exists():
        checkpoint = out / json.loads(latest.read_text())["path"]
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        if state["pool_pairs"] != len(dataset) or state["arm"] != args.arm:
            raise RuntimeError("Existing continuation belongs to a different pool/arm")
    model, tokenizer = load_classifier(checkpoint, trainable=True, attention=args.attention)
    args.attention = model.config._attn_implementation
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.01, fused=True)
    warmup = max(1, int(total_steps * 0.03))

    def schedule(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    completed = pairs_seen = tokens_done = 0
    prior_seconds = 0.0
    micro_tokens = args.micro_tokens
    checkpointing = False
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        completed, pairs_seen, tokens_done = state["step"], state["pairs_seen"], state["tokens_done"]
        prior_seconds, micro_tokens = state["elapsed_seconds"], state["micro_tokens"]
        checkpointing = state["gradient_checkpointing"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        del state
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    started = last_log = time.monotonic()
    last_pairs, last_tokens = pairs_seen, tokens_done
    log_loss = torch.zeros(3, device="cuda", dtype=torch.float64)
    window_pairs = 0
    print(json.dumps({"event": "continuation_started", "arm": args.arm, "parent": str(PARENT), "pairs": len(dataset), "domains": DOMAINS, "steps": total_steps, "resume_step": completed, "effective_batch": macro, "epochs": 1}), flush=True)

    def save(step, final=False):
        # Two alternating owned slots retain an intact previous snapshot during writes.
        old_slot = json.loads(latest.read_text()).get("slot", 1) if latest.exists() else 1
        slot = 1 - old_slot
        folder = out / f"resume-{slot}"
        folder.mkdir(exist_ok=True)
        model.save_pretrained(folder, safe_serialization=True)
        model.config.save_pretrained(folder)
        elapsed = prior_seconds + time.monotonic() - started
        payload = dict(arm=args.arm, pool_pairs=len(dataset), step=step, pairs_seen=pairs_seen,
                       tokens_done=tokens_done, elapsed_seconds=elapsed, micro_tokens=micro_tokens,
                       gradient_checkpointing=checkpointing, optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state())
        torch.save(payload, folder / "training_state.pt")
        atomic_json(latest, {"path": folder.name, "step": step, "slot": slot})
        if final:
            final_dir = out / "final"
            model.save_pretrained(final_dir, safe_serialization=True)
            model.config.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            summary = dict(complete=True, model=name, parent=str(PARENT), domains=DOMAINS,
                           pool_pairs=len(dataset), pairs_seen=pairs_seen, epochs=1, optimizer_steps=step,
                           effective_batch=128, learning_rate=5e-5, warmup_ratio=0.03, weight_decay=0.01,
                           optimizer="AdamW_fused", scheduler="cosine", precision="BF16", temperature=1,
                           loss={"HARD": "CE", "SOFT": "KL(teacher||student)", "HYBRID": "0.5 CE + 0.5 KL"}[args.arm],
                           effective_tokens=tokens_done, elapsed_seconds=elapsed,
                           effective_tokens_per_second=tokens_done / max(1, elapsed),
                           pairs_per_second=pairs_seen / max(1, elapsed), attention=args.attention,
                           micro_token_budget=micro_tokens, gradient_checkpointing=checkpointing,
                           peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30)
            atomic_json(out / "complete.json", summary)
            print(json.dumps({"event": "continuation_complete", **summary}), flush=True)

    for step in range(completed, total_steps):
        indices = np.asarray(order[step * macro:min((step + 1) * macro, len(dataset))], dtype=np.int64)
        batch = dataset.fetch(indices)
        if probabilities is not None:
            batch["teacher"] = np.asarray(probabilities[indices], dtype=np.float32)
            if not np.isfinite(batch["teacher"]).all():
                raise RuntimeError("Unfinished Teacher probabilities encountered")
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        while True:
            optimizer.zero_grad(set_to_none=True)
            losses = torch.zeros(3, device="cuda", dtype=torch.float64)
            step_tokens = 0
            inputs = targets = logits = ce = kl = loss = None
            try:
                for micro in legacy.microbatches(batch, 128, micro_tokens):
                    inputs, targets, token_count = legacy.collate(store, micro, tokenizer.pad_token_id, tokenizer.eos_token_id, "cuda:0")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(**inputs).logits.float()
                        # Native labels include averaged conflicts/continuous original targets.
                        # Two-class CE preserves them without inventing an argmax label.
                        hard = torch.stack((1 - targets, targets), dim=-1)
                        ce = F.cross_entropy(logits, hard, reduction="sum")
                        kl = torch.zeros((), device="cuda")
                        if probabilities is not None:
                            soft = torch.as_tensor(micro["teacher"], device="cuda", dtype=torch.float32)
                            kl = F.kl_div(F.log_softmax(logits, dim=-1), soft, reduction="sum")
                        loss = ce if args.arm == "HARD" else kl if args.arm == "SOFT" else 0.5 * (ce + kl)
                    (loss / len(indices)).backward()
                    losses += torch.stack((loss.detach(), ce.detach(), kl.detach())).double()
                    step_tokens += token_count
                    inputs = targets = logits = ce = kl = loss = None
                break
            except torch.cuda.OutOfMemoryError:
                inputs = targets = logits = ce = kl = loss = None
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                if micro_tokens > 4096:
                    micro_tokens = max(4096, micro_tokens // 2)
                elif not checkpointing:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    checkpointing = True
                else:
                    raise
                print(json.dumps({"event": "actual_training_oom_retry", "step": step, "micro_tokens": micro_tokens, "checkpointing": checkpointing}), flush=True)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=True)
        optimizer.step()
        scheduler.step()
        pairs_seen += len(indices)
        tokens_done += step_tokens
        log_loss += losses
        window_pairs += len(indices)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == total_steps:
            torch.cuda.synchronize()
            now = time.monotonic()
            values = (log_loss / window_pairs).tolist()
            progress = dict(event="continuation_progress", model=name, step=step + 1, total_steps=total_steps,
                            pairs_seen=pairs_seen, effective_tokens=tokens_done, loss=values[0], ce=values[1], kl=values[2],
                            learning_rate=scheduler.get_last_lr()[0], elapsed_seconds=prior_seconds + now - started,
                            tokens_per_second=(tokens_done-last_tokens)/max(0.001, now-last_log),
                            pairs_per_second=(pairs_seen-last_pairs)/max(0.001, now-last_log))
            atomic_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            log_loss.zero_()
            window_pairs = 0
            last_log, last_pairs, last_tokens = now, pairs_seen, tokens_done
        if (step + 1) % 2000 == 0:
            save(step + 1)
    save(total_steps, final=True)


if __name__ == "__main__":
    main()
