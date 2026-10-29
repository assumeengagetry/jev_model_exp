"""Read archived schedules only; distinguish unique-pool and sampled exposure ratios."""
import json
from pathlib import Path
import numpy as np
import pyarrow as pa

PROJECT = Path.home() / "assumeengage"
OUT = Path(__file__).resolve().parent
CASES = [
    ("BASE-17M", "lu-fastscreen-20261028T195840Z-capacity", "C_BASE_R32"),
    ("R50-17M", "lu-fastscreen-20261028T195840Z-capacity", "C_R50_R32"),
    ("BASE-20M", "lu-fastscreen-20261028T164315Z", "F_BASE"),
    ("HS30-20M-seed3408", "lu-fastscreen-20261028T175106Z", "F_HS30"),
]


def main():
    records = []
    for name, round_name, arm in CASES:
        root = PROJECT / "research" / round_name
        schedule_path = root / "runs" / arm / "schedule.npz"
        if not schedule_path.exists():
            records.append({"case": name, "state": "schedule-not-found"}); continue
        schedule = np.load(schedule_path, allow_pickle=False)
        pools = []
        for pool in (0, 1):
            table = pa.ipc.open_file(pa.memory_map(str(root / f"pool-{pool}.arrow"), "r")).read_all()
            pools.append({key: table.column(key).combine_chunks().to_numpy(zero_copy_only=False)
                          for key in ("length", "labels")})
        if not {"pool", "index"}.issubset(schedule.files):
            records.append({"case": name, "state": "unrecognized-schedule", "keys": schedule.files}); continue
        counts = [0, 0]
        tokens = [0, 0]
        by_pool = []
        for pool, arrays in enumerate(pools):
            indices = schedule["index"][schedule["pool"] == pool]
            labels = arrays["labels"][indices]
            lengths = arrays["length"][indices]
            current_counts = [int((labels == label).sum()) for label in (0, 1)]
            current_tokens = [int(lengths[labels == label].sum()) for label in (0, 1)]
            counts = [a + b for a, b in zip(counts, current_counts)]
            tokens = [a + b for a, b in zip(tokens, current_tokens)]
            by_pool.append({"pool": pool, "positive_presentations": current_counts[0], "negative_presentations": current_counts[1],
                            "positive_tokens": current_tokens[0], "negative_tokens": current_tokens[1]})
        records.append({"case": name, "state": "complete", "schedule": str(schedule_path),
                        "planned_presentations": sum(counts), "planned_tokens": sum(tokens),
                        "positive_presentations": counts[0], "negative_presentations": counts[1],
                        "positive_row_fraction": counts[0] / sum(counts),
                        "positive_token_fraction": tokens[0] / sum(tokens), "by_pool": by_pool})
    (OUT / "exposure-ratios.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 训练计划中的正负曝光", "", "正=兼容/entailment，负=不兼容/contradiction。以下按完整计划中的出现次数计，不是唯一样本数；不是原生讽刺标签比例。",
             "最终更新若在token cap前结束，实际最后少量曝光以worker完成记录为准。", "",
             "| 已完成参考计划 | 计划样本出现次数 | 计划tokens | 正类行数占比 | 正类token占比 |", "|---|---:|---:|---:|---:|"]
    for row in records:
        if row["state"] == "complete":
            lines.append(f"| {row['case']} | {row['planned_presentations']:,} | {row['planned_tokens']:,} | {row['positive_row_fraction']:.2%} | {row['positive_token_fraction']:.2%} |")
    (OUT / "exposure-ratios.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(records, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
