"""A/B the padding-mask execution path with identical tokens and padded shapes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark import ROOT, now, wait_free, write


def main():
    folder = ROOT / "mask-diagnostics"
    folder.mkdir(exist_ok=True)
    records = []
    deadline = time.monotonic() + 20 * 60
    for batch, all_ones in ((16, True), (8, False), (8, True)):
        name = f"mixed-fixed2808-B{batch}-{'ones-mask' if all_ones else 'real-mask'}"
        path = folder / f"{name}.json"
        if path.exists():
            raise RuntimeError(f"Preserve existing diagnostic: {path}")
        wait_free(0, deadline)
        command = [sys.executable, "-u", str(ROOT / "benchmark.py"), "--worker", "--kind", "long_real_pairs",
                   "--checkpoint", "off", "--microbatch", str(batch), "--padding-length", "2808",
                   "--profile-kernel", "--output", str(path)]
        if all_ones:
            command.append("--all-ones-mask")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        with (folder / f"{name}.log").open("x") as stream:
            p = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        if p.returncode:
            raise RuntimeError(f"Mask control failed: {name}")
        row = json.loads(path.read_text())
        row["diagnostic_name"] = name
        records.append(row)
        write(folder / "results.json", records)
        print(json.dumps({"event": "mask_control", "name": name, "state": row["state"],
                          "peak_allocated_GiB": row.get("peak_allocated_GiB"), "peak_reserved_GiB": row.get("peak_reserved_GiB"),
                          "kernel": row.get("attention_kernel_ops")}), flush=True)
    write(folder / "status.json", {"state": "complete", "time": now(), "gpu0_released": True})
    print(json.dumps({"event": "mask_diagnostics_complete", "results": str(folder / "results.json"), "gpu0_released": True}), flush=True)


if __name__ == "__main__":
    main()
