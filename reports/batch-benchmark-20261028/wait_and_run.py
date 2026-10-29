"""Wait for genuine idle capacity. Only our own benchmark subprocess may be stopped."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
PROJECT = Path.home() / "assumeengage"
PYTHON = str(PROJECT / "temp_env/train/bin/python")


def status(state, **fields):
    (OUT / "status.json").write_text(json.dumps(
        {"state": state, "time": datetime.now(timezone.utc).isoformat(), **fields}, indent=2) + "\n")


def running_queues():
    busy = []
    for path in (PROJECT / "research").glob("*/queue-state.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            busy.append(str(path))
            continue
        if value.get("state") not in ("complete", "completed"):
            busy.append(str(path))
    return busy


def gpu_pids(gpu):
    result = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=10)
    return {int(line.strip()) for line in result.splitlines() if line.strip().isdigit()}


def main():
    deadline = time.monotonic() + 8 * 3600
    results_path = OUT / "results.json"
    records = json.loads(results_path.read_text()) if results_path.exists() else []
    existing = {(record["profile"], record["microbatch"]): record for record in records}
    for profile in ("joint-real", "4096-tail"):
        for batch in (16, 32, 64, 128, 256):
            name = f"{profile}-mb{batch}"
            prior = existing.get((profile, batch))
            if prior and prior.get("state") == "oom":
                break
            if prior and prior.get("state") == "complete":
                continue
            while time.monotonic() < deadline:
                if (OUT / "STOP").exists():
                    status("cancelled"); return
                queues = running_queues()
                if queues or any(gpu_pids(i) for i in range(4)):
                    status("waiting_for_idle_gpu", blocked_by=queues, next_case=name,
                           policy="Do not interrupt any training/evaluation; do not benchmark on occupied GPUs")
                    time.sleep(60); continue
                # Observe an idle window rather than using a transient utilization dip.
                time.sleep(5)
                if running_queues() or any(gpu_pids(i) for i in range(4)):
                    continue
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "8",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                with (OUT / f"{name}.log").open("w") as log:
                    process = subprocess.Popen([PYTHON, "-u", str(OUT / "benchmark.py"),
                        "--profile", profile, "--microbatch", str(batch)], env=env, stdout=log, stderr=subprocess.STDOUT)
                    status("benchmarking", case=name, own_pid=process.pid, gpu=0)
                    contended = False
                    while process.poll() is None:
                        time.sleep(3)
                        if gpu_pids(0) - {process.pid}:
                            # Never signal another session's process.
                            process.terminate()
                            process.wait(timeout=20)
                            contended = True
                            status("waiting_after_gpu_contention", case=name)
                            break
                if contended:
                    time.sleep(60); continue
                path = OUT / f"{name}.json"
                if process.returncode != 0 or not path.exists():
                    status("failed", case=name, log=str(OUT / f"{name}.log"))
                    raise SystemExit(1)
                result = json.loads(path.read_text()); records.append(result)
                (OUT / "results.json").write_text(json.dumps(records, indent=2) + "\n")
                print(json.dumps({"event": "batch_case_complete", "profile": profile,
                                  "microbatch": batch, "state": result["state"]}), flush=True)
                break
            else:
                status("idle_wait_timeout", case=name); return
            if records[-1]["state"] == "oom":
                break
    lines = ["# Batch 吞吐与显存实测", "", "后端：与当前联合训练一致的 Transformers SDPA、BF16、causal LoRA。",
             "microbatch≤128 时固定有效 batch=128；256 仅作容量/吞吐探测，不是保持原训练语义的替换。",
             "各配置独立进程、相同 R0 起点，预热2次更新后计时6次更新。真实分布取固定 schedule 窗口，4096-tail 单独做长尾压力测试。", "",
             "| Profile | Microbatch | 有效batch | 状态 | 非padding tokens/s | samples/s | 峰值allocated GiB | 峰值reserved GiB | padding比例 |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for record in records:
        if record["state"] == "complete":
            lines.append(f"| {record['profile']} | {record['microbatch']} | {record['effective_batch']} | 完成 | {record['nonpadding_tokens_per_second']:.0f} | {record['samples_per_second']:.1f} | {record['peak_allocated_gib']:.2f} | {record['peak_reserved_gib']:.2f} | {record['padding_fraction']:.1%} |")
        else:
            lines.append(f"| {record['profile']} | {record['microbatch']} | {record['effective_batch']} | OOM | — | — | — | — | — |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    status("complete", report=str(OUT / "results.md"), cases=len(records))
    print(json.dumps({"event": "batch_benchmark_complete", "report": str(OUT / "results.md")}), flush=True)


if __name__ == "__main__":
    main()
