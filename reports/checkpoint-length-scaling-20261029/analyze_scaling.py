"""Turn measured length/batch cells into reusable memory equations and charts."""
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def design(microbatch, length):
    return [1.0, microbatch * length / 16384, (microbatch / 16) * (length / 1024) ** 2]


def nonnegative_fit(x, y, columns):
    best = None
    for size in range(1, len(columns) + 1):
        for active in combinations(columns, size):
            coefficients = np.linalg.lstsq(x[:, active], y, rcond=None)[0]
            if np.any(coefficients < -1e-10):
                continue
            full = np.zeros(x.shape[1])
            full[list(active)] = np.maximum(coefficients, 0)
            error = float(np.mean((x @ full - y) ** 2))
            if best is None or error < best[0]:
                best = error, full
    assert best is not None
    return best[1], math.sqrt(best[0])


def fit_memory(records, enabled):
    cells = {}
    for row in records:
        if row["kind"] not in ("fixed", "batch-anchor", "threshold") or row["checkpointing"] != enabled:
            continue
        if row["state"] != "complete":
            continue
        key = row["microbatch"], row["length"]
        cells[key] = max(cells.get(key, 0), row["peak_allocated_GiB"])
    keys = sorted(cells)
    x = np.asarray([design(*key) for key in keys])
    y = np.asarray([cells[key] for key in keys])
    linear, linear_rmse = nonnegative_fit(x, y, [0, 1])
    quadratic, quadratic_rmse = nonnegative_fit(x, y, [0, 1, 2])
    use_quadratic = quadratic_rmse < .8 * linear_rmse and linear_rmse > .03
    coefficient, rmse = (quadratic, quadratic_rmse) if use_quadratic else (linear, linear_rmse)
    return {"checkpointing": enabled, "terms": ["1", "B*L/16384", "(B/16)*(L/1024)^2"],
            "coefficients_GiB": coefficient.tolist(), "rmse_GiB": rmse,
            "measured_cells": [{"microbatch": key[0], "length": key[1], "memory_GiB": cells[key],
                                "fitted_GiB": float(np.asarray(design(*key)) @ coefficient)} for key in keys]}


def formula(fit):
    a, b, c = fit["coefficients_GiB"]
    result = f"{a:.3f} + {b:.3f} × x"
    return result + (f" + {c:.3f} × y" if c > 1e-8 else "")


def memory_chart(records):
    from benchmark import LENGTHS, aggregate
    left, right, top, bottom = 75, 690, 45, 350
    height = bottom - top
    successful = [row["peak_allocated_GiB"] for row in records if row["kind"] == "fixed" and row["state"] == "complete"]
    ymax = max(100, math.ceil(max(successful) / 10) * 10)
    sx = lambda length: left + (math.log2(length) - 8) / 5 * (right - left)
    sy = lambda memory: bottom - memory / ymax * height
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420">',
             '<rect width="760" height="420" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#222">',
             '<text x="75" y="25" font-size="17">Qwen3-0.6B LoRA r16 · microbatch16 · BF16</text>']
    for memory in range(0, ymax + 1, 20):
        y = sy(memory)
        lines += [f'<line x1="{left}" x2="{right}" y1="{y}" y2="{y}" stroke="#ddd"/>',
                  f'<text x="{left-10}" y="{y+5}" text-anchor="end">{memory}</text>']
    lines.append('<text x="18" y="230" transform="rotate(-90 18 230)">Peak allocated (GiB)</text>')
    for length in LENGTHS:
        x = sx(length)
        lines.append(f'<text x="{x}" y="{bottom+25}" text-anchor="middle">{length}</text>')
    lines.append(f'<text x="{(left+right)/2}" y="{bottom+52}" text-anchor="middle">Exact sequence length (tokens)</text>')
    for enabled, color, label in ((True, "#007a57", "Checkpoint On"), (False, "#bc3c29", "Checkpoint Off")):
        points = []
        for length in LENGTHS:
            cell = aggregate([row for row in records if row["kind"] == "fixed" and row["length"] == length and row["checkpointing"] == enabled])
            x = sx(length)
            if cell["state"] == "OOM":
                lines.append(f'<text x="{x}" y="{top+18}" text-anchor="middle" fill="{color}">OOM</text>')
            else:
                y = sy(cell["peak_allocated_GiB"])
                points.append(f"{x},{y}")
                lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = 70 if enabled else 94
        lines.append(f'<text x="105" y="{legend_y}" fill="{color}">{label}</text>')
    lines += ['</g>', '</svg>']
    (ROOT / "memory-scaling.svg").write_text("\n".join(lines) + "\n")


def main():
    from benchmark import aggregate
    records = json.loads((ROOT / "results.json").read_text())
    fits = {str(enabled): fit_memory(records, enabled) for enabled in (True, False)}
    (ROOT / "memory-fit.json").write_text(json.dumps(fits, ensure_ascii=False, indent=2) + "\n")
    memory_chart(records)
    lines = ["# 把长度benchmark用于配置选择", "",
             "## 显存随输入长度和物理batch怎样变化", "",
             "用本轮固定长度和物理batch8/16/32的实测值拟合，M的单位为GiB：", "",
             "- B = 单次同时驻留GPU的sequence条数。",
             "- L = padding之后的单条sequence长度。",
             "- x = B × L / 16384。",
             "- y = (B / 16) × (L / 1024)²。", "",
             "| 开关 | 峰值allocated显存计算式 | 拟合RMSE GiB |", "|---|---|---:|"]
    for enabled in (True, False):
        fit = fits[str(enabled)]
        lines.append(f"| {'On' if enabled else 'Off'} | M ≈ {formula(fit)} | {fit['rmse_GiB']:.3f} |")
    lines += ["", "这里的常数项对应模型与梯度的固定开销；x项对应随总驻留tokens增长的开销；"
              "y项对应随单条长度平方增长的开销。计算时使用物理microbatch，而不是梯度累计之后的有效batch。", "",
              "## 开启checkpointing付出的时间", "",
              "| sequence tokens | On前向 / 反向 秒 | Off前向 / 反向 秒 | On总体耗时 / Off总体耗时 |", "|---|---:|---:|---:|"]
    for length in (256, 512, 1024, 2048, 4096, 8192):
        on, off = [aggregate([row for row in records if row["kind"] == "fixed" and row["length"] == length and row["checkpointing"] == enabled]) for enabled in (True, False)]
        times = lambda row: f"{row['forward_seconds_per_effective_batch']:.3f} / {row['backward_seconds_per_effective_batch']:.3f}" if "tokens_per_second" in row else "OOM"
        ratio = f"{on['seconds_per_effective_batch']/off['seconds_per_effective_batch']:.2f}×" if "tokens_per_second" in on and "tokens_per_second" in off else "—"
        lines.append(f"| {length:,} | {times(on)} | {times(off)} | {ratio} |")
    lines += ["", "前向/反向采用CUDA events计时，每行对应累计32条sequence；反向包含checkpoint触发的重算。总体耗时另含拼batch和H2D。", "",
              "## 换任务、batch或显卡时怎么选", "",
              "1. 先统计每个物理microbatch的最大长度及padding。不同文本内容先换算成tokens，不使用字数代替。",
              "2. 以B和补齐后的L查主报告的实测表；使用上面的计算式估算其他组合。",
              "3. 以主报告中的reserved峰值加上CUDA进程开销，对照目标GPU显存容量。",
              "4. 想保持有效batch又降显存，降低物理microbatch并增加梯度累计。",
              "5. 换模型时，用同一脚本测几档长度后重新拟合；模型层数、隐藏维度、精度和attention算子一并记下。", "",
              "## 输入长度与显存图", "", "![实测显存曲线](memory-scaling.svg)", "",
              "`memory-fit.json`保存完整系数和每个拟合点；主报告`REPORT.md`保存全部测量表。"]
    (ROOT / "SCALING-ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(ROOT / "SCALING-ANALYSIS.md"), "formulas": {key: formula(value) for key, value in fits.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
