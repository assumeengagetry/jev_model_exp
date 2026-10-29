"""Explain mixed-length OOM with allocator, padding-shape and batch controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark import ROOT, now, wait_free, write


def main():
    definitions = [
        {"name": "dense2808-default", "kind": "diagnostic-dense", "length": 2808, "checkpoint": "off", "microbatch": 16, "profile_kernel": True},
        {"name": "mixed-long-expandable", "kind": "long_real_pairs", "length": 0, "checkpoint": "off", "microbatch": 16, "allocator": "expandable_segments:True", "profile_kernel": True},
        {"name": "mixed-long-fixedpad2808", "kind": "long_real_pairs", "length": 0, "checkpoint": "off", "microbatch": 16, "padding_length": 2808, "profile_kernel": True},
        {"name": "mixed-long-batch8", "kind": "long_real_pairs", "length": 0, "checkpoint": "off", "microbatch": 8, "profile_kernel": True},
        {"name": "mixed-medium-kernel", "kind": "medium", "length": 0, "checkpoint": "off", "microbatch": 16, "profile_kernel": True},
        {"name": "mixed-long-on-kernel", "kind": "long_real_pairs", "length": 0, "checkpoint": "on", "microbatch": 16, "profile_kernel": True},
    ]
    records = []
    folder = ROOT / "mixed-diagnostics"
    folder.mkdir(exist_ok=True)
    deadline = time.monotonic() + 30 * 60
    for definition in definitions:
        path = folder / f"{definition['name']}.json"
        if path.exists():
            raise RuntimeError(f"Preserve existing diagnostic: {path}")
        wait_free(0, deadline)
        command = [sys.executable, "-u", str(ROOT / "benchmark.py"), "--worker", "--kind", definition["kind"],
                   "--length", str(definition["length"]), "--checkpoint", definition["checkpoint"],
                   "--microbatch", str(definition["microbatch"]), "--output", str(path)]
        if definition.get("padding_length"):
            command += ["--padding-length", str(definition["padding_length"])]
        if definition.get("profile_kernel"):
            command += ["--profile-kernel"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
        if definition.get("allocator"):
            env["PYTORCH_CUDA_ALLOC_CONF"] = definition["allocator"]
        with (folder / f"{definition['name']}.log").open("x") as stream:
            p = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        if p.returncode:
            raise RuntimeError(f"Diagnostic failed: {definition['name']}")
        row = json.loads(path.read_text())
        row["diagnostic_name"] = definition["name"]
        records.append(row)
        write(folder / "results.json", records)
        print(json.dumps({"event": "mixed_diagnostic", "name": definition["name"], "state": row["state"],
                          "peak_allocated_GiB": row.get("peak_allocated_GiB"), "peak_reserved_GiB": row.get("peak_reserved_GiB"),
                          "kernel": row.get("attention_kernel_ops")}), flush=True)
    lines = ["# 为什么固定2816能跑，真实混长2807却OOM", "",
             "原混长组是同32条2527–2807-token输入，两个物理batch分别补齐至2576和2808。"
             "默认allocator下，第二个batch前向OOM：当时allocated74.82 GiB、reserved但未使用17.41 GiB，GPU仅剩100 MiB。", "",
             "| 对照 | 实际shape | Allocator | 开关 | 结果 | 峰值allocated / reserved GiB | 有效tokens/s |", "|---|---|---|---|---|---:|---:|"]
    for row in records:
        shapes = " + ".join(f"{b['shape'][0]}×{b['shape'][1]}" for b in row["input"]["microbatches"])
        memory = f"{row['peak_allocated_GiB']:.2f} / {row['peak_reserved_GiB']:.2f}"
        rate = f"{row['nonpadding_tokens_per_second']:,.0f}" if row.get("ok") else "—"
        lines.append(f"| {row['diagnostic_name']} | {shapes} | {row['environment']['allocator_config']} | {'On' if row['checkpointing'] else 'Off'} | {row['state']} | {memory} | {rate} |")
    lines += ["", "## 实际attention算子", ""]
    for row in records:
        if row.get("attention_kernel_ops"):
            lines.append(f"- {row['diagnostic_name']}：`{row['attention_kernel_ops']}`。")
    lines += ["", "各对照的逐步OOM位置、输入长度与完整错误均保存在本目录的JSON和log中。"]
    (ROOT / "MIXED-LENGTH-DIAGNOSIS.md").write_text("\n".join(lines) + "\n")
    write(folder / "status.json", {"state": "complete", "time": now(), "gpu0_released": True})
    print(json.dumps({"event": "mixed_length_diagnosis_complete", "report": str(ROOT / "MIXED-LENGTH-DIAGNOSIS.md"), "gpu0_released": True}), flush=True)


if __name__ == "__main__":
    main()
