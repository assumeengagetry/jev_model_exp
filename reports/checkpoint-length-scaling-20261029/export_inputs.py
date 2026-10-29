"""Readable and CSV exports of the exact historical32-pair profiles."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    profiles = json.loads((ROOT / "original-profiles.json").read_text())["profiles"]
    lines = ["# 旧checkpoint benchmark的实际输入", "",
             "物理microbatch16，累计2次，共32对。以下长度均包括格式、premise、hypothesis和EOS。", "",
             "| Profile | 实际token范围 | 均长 | 中位长度 | p90 | padding |", "|---|---:|---:|---:|---:|---:|"]
    for name, p in profiles.items():
        lines.append(f"| {name} | {p['length_min']}–{p['length_max']} | {p['length_mean']:.2f} | {p['length_p50']:.1f} | {p['length_p90']:.1f} | {p['padding_fraction']:.2%} |")
    rows = []
    for name, p in profiles.items():
        inputs = json.loads((ROOT / f"inputs-{name}.json").read_text())
        lines += ["", f"## {name}", "", f"输入任务：{p['tasks']}。", "",
                  "| 物理batch编号 | Tensor shape | 最短 / 最长实际tokens | 有效tokens | padded tokens | padding |", "|---|---|---:|---:|---:|---:|"]
        for i, batch in enumerate(p["microbatches"]):
            actual, shape = batch["actual_lengths"], batch["shape"]
            padded = shape[0] * shape[1]
            lines.append(f"| {i+1} | {shape[0]} × {shape[1]} | {min(actual)} / {max(actual)} | {sum(actual):,} | {padded:,} | {1-sum(actual)/padded:.2%} |")
        lines += ["", "32条输入的完整长度：", "", "```text", ", ".join(map(str, p["lengths"])), "```"]
        task = next(iter(p["tasks"]))
        for position, (index, length, item) in enumerate(zip(p["source_row_indices"], p["lengths"], inputs)):
            batch_index = position // 16
            padded_length = p["microbatches"][batch_index]["shape"][1]
            rows.append({"profile": name, "position": position, "pool_row_index": index,
                         "task": task, "label": item["labels"], "physical_batch": batch_index+1,
                         "sequence_tokens": length, "padded_sequence_tokens": padded_length,
                         "padding_tokens": padded_length-length})
    with (ROOT / "original-inputs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines += ["", "逐条输入、原始row索引、标签和padding见[original-inputs.csv](original-inputs.csv)。"]
    (ROOT / "ORIGINAL-INPUTS.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "ORIGINAL-INPUTS.md")


if __name__ == "__main__":
    main()
