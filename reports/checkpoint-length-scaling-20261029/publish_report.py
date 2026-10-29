"""Publish measured conclusions, all49 cells, and exact attention dispatch evidence."""
import csv
import hashlib
import json
from pathlib import Path

from benchmark import ROOT, LENGTHS, aggregate, now, write
from analyze_scaling import formula


def main():
    main_rows = json.loads((ROOT / "results.json").read_text())
    mixed = json.loads((ROOT / "mixed-diagnostics/results.json").read_text())
    masks = json.loads((ROOT / "mask-diagnostics/results.json").read_text())
    trace = json.loads((ROOT / "attention-dispatch.json").read_text())
    fits = json.loads((ROOT / "memory-fit.json").read_text())
    profiles = json.loads((ROOT / "original-profiles.json").read_text())["profiles"]
    thresholds = json.loads((ROOT / "oom-thresholds.json").read_text())
    by_mixed = {row["diagnostic_name"]: row for row in mixed}
    by_mask = {row["diagnostic_name"]: row for row in masks}
    groups = {length: {enabled: aggregate([row for row in main_rows if row["kind"] == "fixed" and row["length"] == length and row["checkpointing"] == enabled]) for enabled in (True, False)} for length in LENGTHS}
    stats = lambda row: f"{row['peak_allocated_GiB']:.2f}" if "peak_allocated_GiB" in row else "OOM"
    rate = lambda row: f"{row['tokens_per_second']:,.0f}" if "tokens_per_second" in row else "—"
    lines = ["# 输入长度、checkpointing与显存：实测研究报告", "",
             "## 主要结果", "",
             "- 旧medium实际为192–205 tokens，long_real_pairs为2527–2807；对应padding为4.84%和3.11%。",
             "- 物理microbatch16时，2048 tokens的峰值allocated由63.85降到6.07 GiB，节省90.5%；8192 tokens开启checkpointing后为20.55 GiB。",
             "- 整齐输入关闭checkpointing时，2944 tokens通过、3072发生OOM。真实混长输入还会触发padding-mask执行路径；"
             "同shape的对照显示，这一路径把microbatch8的峰值从44.29升到50.05 GiB。",
             "- 配置迁移主要看物理batch×补齐长度；两组实测8×2048和16×1024的显存几乎相同。有效batch通过梯度累计保持。", "",
             "## medium和long_real_pairs的实际数据", "",
             "| Profile | 实际长度范围 | 均长 | 中位长度 | 物理batch tensor shapes | 有效 / padded tokens | Padding |",
             "|---|---:|---:|---:|---|---:|---:|"]
    for name, p in profiles.items():
        shape = " + ".join(f"{b['shape'][0]}×{b['shape'][1]}" for b in p["microbatches"])
        lines.append(f"| {name} | {p['length_min']}–{p['length_max']} | {p['length_mean']:.2f} | {p['length_p50']:.1f} | {shape} | {p['nonpadding_tokens']:,} / {p['padded_tokens']:,} | {p['padding_fraction']:.2%} |")
    lines += ["", "medium的代码先筛192–384，再取前32对，结果实际集中于192–205，全部来自ANLI r1。"
              "long_real_pairs取目标池最长32对，全部来自RAGTruth Summary。每组物理microbatch16、累计2次。"
              "[逐条64条输入及padding](ORIGINAL-INPUTS.md)与[CSV](original-inputs.csv)已保存。", "",
              "## 256–8192长度阶梯", "",
              "设置：NVIDIA H800 NVL，CUDA可见93.09 GiB；Qwen3-0.6B分类模型、LoRA r16/alpha32、BF16、causal SDPA、物理microbatch16，累计2次。"
              "每档同32对真实长文本裁到精确长度，padding为0。每格两个独立进程，各预热1次、计时3次前向+反向。", "",
              "| 每条长度 tokens | On峰值allocated GiB | Off峰值allocated GiB | On有效tokens/s | Off有效tokens/s | On节省显存 | On耗时 / Off耗时 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for length in LENGTHS:
        on, off = groups[length][True], groups[length][False]
        saving = f"{1-on['peak_allocated_GiB']/off['peak_allocated_GiB']:.1%}" if "tokens_per_second" in off else "—"
        time_ratio = f"{on['seconds_per_effective_batch']/off['seconds_per_effective_batch']:.2f}×" if "tokens_per_second" in off else "—"
        lines.append(f"| {length:,} | {stats(on)} | {stats(off)} | {rate(on)} | {rate(off)} | {saving} | {time_ratio} |")
    lines += ["", "512–2048 tokens时，开启checkpointing后的耗时增加约34–37%，显存节省85.5–90.5%。"
              "256-token档耗时增加83%；长输入下用重算换显存的收益更大。", "",
              "关闭checkpointing的128-token步长搜索：2560、2816、2944通过，3072发生OOM；开启后已通过8192。"
              "8192档的allocated峰值20.55 GiB、allocator reserved峰值23.00 GiB。", "",
              "![显存随长度增长](memory-scaling.svg)", "",
              "## 为什么真实2807 tokens会OOM，而整齐2816能通过", "",
              "### 同样的tokens和shape，只改变padding mask", "",
              "用原来同32条真实长输入，全部补齐至2808，保持token IDs和物理batch不变：", "",
              "| 物理microbatch | Mask | Checkpoint | 结果 | 峰值allocated GiB | 峰值reserved GiB |",
              "|---|---|---|---|---:|---:|"]
    fixed_mask_failed = by_mixed["mixed-long-fixedpad2808"]
    controls = [(16, "真实padding mask", fixed_mask_failed),
                (16, "全1 mask定位对照", by_mask["mixed-fixed2808-B16-ones-mask"]),
                (8, "真实padding mask", by_mask["mixed-fixed2808-B8-real-mask"]),
                (8, "全1 mask定位对照", by_mask["mixed-fixed2808-B8-ones-mask"])]
    for batch, mask, row in controls:
        memory = f"{row['peak_allocated_GiB']:.2f}" if row["ok"] else "OOM"
        reserved = f"{row['peak_reserved_GiB']:.2f}" if row["ok"] else "OOM"
        lines.append(f"| {batch} | {mask} | Off | {'通过' if row['ok'] else 'OOM'} | {memory} | {reserved} |")
    delta = by_mask["mixed-fixed2808-B8-real-mask"]["peak_allocated_GiB"] - by_mask["mixed-fixed2808-B8-ones-mask"]["peak_allocated_GiB"]
    percent = delta / by_mask["mixed-fixed2808-B8-ones-mask"]["peak_allocated_GiB"]
    lines += ["", f"microbatch8仅切换mask就增加{delta:.2f} GiB，即{percent:.1%}。全1 mask用于定位执行开销；正式训练保留真实padding mask。", "",
              "### 已跟踪到的执行路径", "",
              "| 运行路径 | Q头数 | SDPA接收的K/V头数 | 原生GQA | SDPA mask |",
              "|---|---:|---:|---|---|",
              "| 全1 mask，整齐路径 | 16 | 8 | 开启 | None，使用is_causal=True |",
              "| 真实padding mask | 16 | 16 | 关闭，先repeat_kv | 显式4D mask |", "",
              "运行时对同一组2×256输入抓取了全部28层的SDPA参数。真实padding路径中的K/V已由8头展开到16头；"
              "对应transformers 5.5.0的`integrations/sdpa_attention.py:28–62`。"
              "两种路径实测都调用cuDNN attention，额外开销出现在mask与K/V展开这一执行路径。"
              "[实际调用形状](attention-dispatch.json)已保存。", "",
              "### allocator和物理batch对照", "",
              "| 真实2527–2807-token输入 | 物理microbatch | 结果 | allocated GiB | reserved GiB | 有效tokens/s |",
              "|---|---:|---|---:|---:|---:|"]
    default_oom = next(row for row in main_rows if row["kind"] == "long_real_pairs" and not row["checkpointing"])
    recommendations = [("Checkpoint Off，默认allocator", default_oom),
                       ("Checkpoint Off，expandable_segments", by_mixed["mixed-long-expandable"]),
                       ("Checkpoint Off，减小microbatch", by_mixed["mixed-long-batch8"]),
                       ("Checkpoint On", by_mixed["mixed-long-on-kernel"])]
    for label, row in recommendations:
        memory = f"{row['peak_allocated_GiB']:.2f}" if row["ok"] else "OOM"
        reserved = f"{row['peak_reserved_GiB']:.2f}" if row["ok"] else "OOM"
        speed = f"{row['nonpadding_tokens_per_second']:,.0f}" if row["ok"] else "—"
        lines.append(f"| {label} | {row['microbatch']} | {'通过' if row['ok'] else 'OOM'} | {memory} | {reserved} | {speed} |")
    lines += ["", "默认allocator在第二个microbatch前向OOM时，reserved但未使用的显存有17.41 GiB；"
              "改expandable_segments后仍OOM，统一padding至2808后也仍OOM。"
              "本轮直接解决方法是开启checkpointing，或者把物理microbatch从16降到8。", "",
              "## 怎么推广到其他输入、batch和显卡", "",
              "### 用物理驻留tokens估算整齐输入显存", "",
              "对本套Qwen3-0.6B/LoRA配置，令B为物理microbatch、L为补齐后的sequence tokens，x=B×L/16384，M单位GiB：", "",
              f"- Checkpoint On：M ≈ **{formula(fits['True'])}**。",
              f"- Checkpoint Off：M ≈ **{formula(fits['False'])}**。", "",
              "已测的8×2048和16×1024都驻留16384 tokens：On均为3.66 GiB，Off均为32.56 GiB。"
              "1024长度从microbatch8增到16再到32时，On为2.46/3.66/6.04 GiB，Off为16.91/32.56/63.82 GiB。", "",
              "### 配置选择", "",
              "- 保持有效batch32：microbatch16累计2次；microbatch8累计4次。",
              "- 对真实混长输入，使用上节保留padding mask的测量列；按长度分桶减少padding，并减少shape切换。",
              "- 显卡容量对照reserved峰值和CUDA进程开销；allocated数值用于比较张量开销。",
              "- 换模型时，用同脚本测几档长度重新生成曲线；把层数、隐藏维度、精度和attention实现一起记录。", "",
              "附带显存计算器，例如：", "",
              "```bash", "python estimate_memory.py --microbatch 8 --length 2048 --checkpoint off --effective-batch 32", "```", "",
              "输出估算allocated为32.557 GiB，累计4次。前向/反向分项和完整系数见[SCALING-ANALYSIS.md](SCALING-ANALYSIS.md)。", "",
              "## 数据与执行记录", "",
              "本轮共49个显存/速度测量单元，加一组28层attention参数追踪。主阶梯与batch对照耗时10.6分钟。"
              "测量覆盖拼batch、H2D、前向、分类loss、反向、梯度清零；optimizer更新次数为0，未保存模型权重。", "",
              "- [原始测量总表](REPORT.md)：全部长度阶梯、batch对照、allocated/reserved。",
              "- [混长对照](MIXED-LENGTH-DIAGNOSIS.md)：allocator、固定padding、microbatch对照。",
              "- [49个单元CSV](all-measurements.csv)：状态、长度、padding、显存和吞吐。",
              "- `cases/`、`mixed-diagnostics/`、`mask-diagnostics/`：逐次结果、OOM阶段和完整日志。",
              "- `original-profiles.json`、`original-inputs.csv`：旧profile逐条长度及原始row索引。",
              "- `manifest.json`、`publication.json`：输入与代码hash、环境和输出文件。"]
    report_path = ROOT / "RESEARCH-REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    excerpt = "\n".join(lines[lines.index("### 同样的tokens和shape，只改变padding mask"):lines.index("## 怎么推广到其他输入、batch和显卡")])
    mixed_report = ROOT / "MIXED-LENGTH-DIAGNOSIS.md"
    base = mixed_report.read_text().split("\n## Padding mask定位结果\n")[0]
    mixed_report.write_text(base + "\n## Padding mask定位结果\n\n" + excerpt + "\n")
    all_rows = main_rows + mixed + masks
    exports = []
    for row in all_rows:
        exports.append({"name": row.get("diagnostic_name", f"{row['kind']}-L{row['length']}-B{row['microbatch']}-{row['checkpointing']}-r{row['repeat']}"),
                        "checkpointing": row["checkpointing"], "physical_microbatch": row["microbatch"],
                        "effective_batch": row["input"]["effective_batch"], "length_min": row["input"]["length_min"],
                        "length_max": row["input"]["length_max"], "padded_shapes": json.dumps([b["shape"] for b in row["input"]["microbatches"]]),
                        "nonpadding_tokens": row["input"]["nonpadding_tokens"], "padded_tokens": row["input"]["padded_tokens"],
                        "padding_fraction": row["input"]["padding_fraction"], "state": row["state"],
                        "peak_allocated_GiB": row["peak_allocated_GiB"], "peak_reserved_GiB": row["peak_reserved_GiB"],
                        "nonpadding_tokens_per_second": row.get("nonpadding_tokens_per_second"),
                        "allocator": row["environment"]["allocator_config"], "all_ones_control": row.get("all_ones_mask_control", False)})
    with (ROOT / "all-measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exports[0]))
        writer.writeheader()
        writer.writerows(exports)
    assert len(all_rows) == 49
    write(ROOT / "publication.json", {"state": "complete", "time": now(), "measurement_cells": len(all_rows),
          "attention_layer_calls": 56, "report": str(report_path), "gpu0_released": True,
          "source_code_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in ROOT.glob("*.py")}})
    print(json.dumps({"event": "checkpoint_research_published", "report": str(report_path), "cells": len(all_rows), "gpu0_released": True}), flush=True)


if __name__ == "__main__":
    main()
