#!/usr/bin/env python3
"""Recompute report tables and figures from archived results; no model execution.

Only Python's standard library and the installed rsvg-convert are used.
Run from any directory: python /path/to/this/generate_figures.py
"""

import csv
import hashlib
import json
import math
import shutil
import subprocess
from html import escape
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "research/lu-specialist-fusion-20261029T152144Z"
EXPERTS = ["ANLI", "HELLASWAG", "ACOS", "ISARCASM", "RAGTRUTH"]
FUSIONS = ["TIES", "PROBABILITY_MEAN", "LINEAR"]
MODELS = EXPERTS + ["MIXED", "TIES", "LINEAR", "PROBABILITY_MEAN"]
BENCHMARKS = [12, 29, 38, 40, 59, 11, 39, 41, 42]
OWN = dict(zip(EXPERTS, BENCHMARKS[:5]))
NAMES = {
    "ANLI": "ANLI专家", "HELLASWAG": "HellaSwag专家", "ACOS": "ACOS专家",
    "ISARCASM": "英语讽刺专家", "RAGTRUTH": "RAGTruth专家", "MIXED": "混合训练",
    "TIES": "TIES融合", "LINEAR": "权重平均", "PROBABILITY_MEAN": "概率集成",
}
HEADERS = {
    "ANLI": ["ANLI", "专家"], "HELLASWAG": ["HellaSwag", "专家"],
    "ACOS": ["ACOS", "专家"], "ISARCASM": ["英语讽刺", "专家"],
    "RAGTRUTH": ["RAGTruth", "专家"], "MIXED": ["混合", "训练"],
    "TIES": ["TIES", "融合"], "LINEAR": ["权重", "平均"],
    "PROBABILITY_MEAN": ["概率", "集成"],
}
TASKS = {
    11: "ContractNLI", 12: "ANLI", 29: "HellaSwag", 38: "ACOS",
    39: "FinEntity", 40: "iSarcasmEval", 41: "VAST", 42: "NLI4CT", 59: "RAGTruth",
}
FAMILY_TASKS = {
    "ANLI": ["anli_r1", "anli_r2", "anli_r3"], "HELLASWAG": ["hellaswag"],
    "ACOS": ["acos/Laptop-ACOS", "acos/Restaurant-ACOS"],
    "ISARCASM": ["isarcasm/intended-en"],
    "RAGTRUTH": ["ragtruth/Summary", "ragtruth/Data2txt", "ragtruth/QA"],
}
INK, MUTED, GRID = "#172d43", "#566d80", "#d9e4ec"
COLORS = {"MIXED": "#126c97", "TIES": "#ca6b1c", "LINEAR": "#7254a2", "PROBABILITY_MEAN": "#20856f"}


def read_csv(name):
    with (SOURCE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (a, b)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def blend(a, b, weight):
    a, b = a.lstrip("#"), b.lstrip("#")
    channels = [round(int(a[i:i+2], 16) * (1-weight) + int(b[i:i+2], 16) * weight) for i in (0, 2, 4)]
    return "#" + "".join(f"{c:02x}" for c in channels)


def sequential(value):
    assert 0 <= value <= 100
    if value <= 50:
        return blend("#f0f7fc", "#509dc6", value / 50)
    return blend("#509dc6", "#073b65", (value-50) / 50)


def diverging(value):
    assert -50 <= value <= 50
    return blend("#fafafa", "#aa343f" if value < 0 else "#217caa", abs(value) / 50)


class SVG:
    def __init__(self, width, height, title, description):
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<style>text{font-family:"Noto Sans CJK SC","Source Han Sans CN",sans-serif;font-variant-numeric:tabular-nums}</style>',
        ]
        self.rect(0, 0, width, height, "#ffffff")
        self.text(54, 57, title, 29, weight=700)
        self.text(54, 93, description, 17, color=MUTED)

    def rect(self, x, y, width, height, fill, stroke="none", sw=1, radius=0):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, value, size=18, color=INK, anchor="start", weight=400):
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')

    def lines(self, x, y, values, size=18, color=INK, anchor="middle", weight=400):
        for i, value in enumerate(values):
            self.text(x, y + i*(size+8), value, size, color, anchor, weight)

    def line(self, x1, y1, x2, y2, color=GRID, sw=1, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{sw}"{extra}/>')

    def circle(self, x, y, radius, fill, stroke="none", sw=1):
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def polygon(self, points, stroke, fill="none", sw=1, opacity=1, dash=None):
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polygon points="{points}" stroke="{stroke}" fill="{fill}" fill-opacity="{opacity}" stroke-width="{sw}"{extra}/>')

    def save(self, stem):
        path = OUT / "figures" / f"{stem}.svg"
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")
        converter = shutil.which("rsvg-convert")
        if converter is None:
            raise RuntimeError("rsvg-convert is required to render the PNG figures")
        subprocess.run([converter, "--output", str(path.with_suffix(".png")), str(path)], check=True, timeout=30)


def heatmap(scores, records, delta=False):
    columns = [c for c in MODELS if c != "MIXED"] if delta else MODELS
    title = "相对混合训练：专科优势并不集中在对角线" if delta else "模型 × benchmark：五个专家、混合训练与三种融合"
    subtitle = "每格 = 当前模型 Skill − 混合模型 Skill；蓝色更好、红色更差，零点居中。" if delta else "Language 维度内的九项公开评测；每格为 Skill / 100，金色框标出专家的对应任务。"
    chart = SVG(1640, 1050, title, subtitle)
    x0, y0, cw, rh = 270, 240, 132 if not delta else 148, 61
    chart.text(x0 + cw*2.5, 145, "五个数据家族专家（每个约 3M tokens）", 18, anchor="middle", weight=700)
    chart.text(x0 + cw*(5+(len(columns)-5)/2), 145, "相同总曝光控制 / 融合" if not delta else "三种融合 / 集成", 18, anchor="middle", weight=700)
    chart.line(x0, 160, x0+cw*5, 160, MUTED)
    chart.line(x0+cw*5+6, 160, x0+cw*len(columns), 160, MUTED)
    for j, code in enumerate(columns):
        chart.lines(x0+(j+.5)*cw, 193, HEADERS[code], 19, weight=700 if code == "MIXED" else 400)
    for i, bid in enumerate(BENCHMARKS):
        y = y0+i*rh
        if i == 5:
            chart.line(52, y-2, x0+cw*len(columns), y-2, MUTED, 2)
        chart.text(x0-18, y+38, TASKS[bid], 20, anchor="end")
        for j, code in enumerate(columns):
            value = scores[code, bid] - (scores["MIXED", bid] if delta else 0)
            chart.rect(x0+j*cw, y, cw, rh, diverging(value) if delta else sequential(value), "#ffffff", 2)
            if code in OWN and OWN[code] == bid:
                chart.rect(x0+j*cw+4, y+4, cw-8, rh-8, "none", "#d29a26", 3)
            color = "#ffffff" if (abs(value) >= 32 if delta else value >= 55) else INK
            chart.text(x0+(j+.5)*cw, y+38, f"{value:+.1f}" if delta else f"{value:.1f}", 20, color, "middle", 700 if code == "MIXED" else 400)
    y = 824
    chart.text(x0-18, y+30, "Language 总分", 20, anchor="end", weight=700)
    chart.text(x0-18, y+54, "官方加权聚合", 14, MUTED, "end")
    mixed = records["MIXED"]["public"]["language_skill"]
    for j, code in enumerate(columns):
        value = records[code]["public"]["language_skill"] - (mixed if delta else 0)
        chart.rect(x0+j*cw, y, cw, 67, diverging(value) if delta else sequential(value), "#ffffff", 2)
        chart.text(x0+(j+.5)*cw, y+41, f"{value:+.1f}" if delta else f"{value:.1f}", 21, INK, "middle", 700)
    for j in (5, 6) if not delta else (5,):
        chart.line(x0+j*cw, y0-12, x0+j*cw, y+67, "#90a5b5", 3)
    legend_x, legend_y, legend_w = 410, 935, 780
    for i in range(100):
        value = i-49.5 if delta else (i+.5)
        chart.rect(legend_x+legend_w*i/100, legend_y, legend_w/100+1, 15, diverging(value) if delta else sequential(value))
    chart.text(legend_x-20, legend_y+15, "−50" if delta else "0", 15, anchor="end")
    chart.text(legend_x+legend_w+20, legend_y+15, "+50" if delta else "100", 15)
    if delta:
        chart.text(legend_x+legend_w/2, legend_y+39, "0 = 与混合训练相同", 15, MUTED, "middle")
    chart.text(54, 1012, "同一 R0-r32、单个续训 seed；每方案 31,715 请求，覆盖 100%。这里只展示 Language 内任务，不是六大维度。", 16, MUTED)
    chart.save("02-relative-to-mixed" if delta else "01-model-benchmark-heatmap")


def specialization_figure(specialization):
    chart = SVG(1540, 840, "专科模型是否更擅长本方向？只有 HellaSwag 出现一致优势", "同一行比较同一个专家与混合训练；两面板指标不同，不能合并成一个平均收益。")
    panels = [(360, 400, -40, 5, "本族原生 dev 差值（百分点）", "native_difference_pp"),
              (1015, 360, -18, 4, "对应公开 benchmark 差值（Skill 分）", "public_skill_difference")]
    for x0, width, low, high, label, key in panels:
        scale = lambda value: x0 + (value-low)/(high-low)*width
        chart.text(x0+width/2, 166, label, 20, anchor="middle", weight=700)
        chart.text(x0+width/2, 199, "正值 = 专家更好；负值 = 混合训练更好", 16, MUTED, "middle")
        ticks = [-40, -30, -20, -10, 0, 5] if key == "native_difference_pp" else [-15, -10, -5, 0]
        for tick in ticks:
            chart.line(scale(tick), 225, scale(tick), 671, MUTED if tick == 0 else GRID, 2 if tick == 0 else 1)
            chart.text(scale(tick), 704, f"{tick:g}", 16, MUTED, "middle")
        for i, row in enumerate(specialization):
            y, value = 262+i*85, row[key]
            x1, x2 = sorted([scale(0), scale(value)])
            color = "#237d9c" if value > 0 else "#b94d57"
            chart.rect(x1, y-15, max(x2-x1, 1), 30, color, radius=3)
            chart.circle(scale(value), y, 4, color)
            chart.text(scale(value)+(12 if value >= 0 else -12), y+7, f"{value:+.1f}", 21, color, "start" if value >= 0 else "end", 700)
    for i, row in enumerate(specialization):
        chart.text(290, 269+i*85, NAMES[row["code"]], 21, anchor="end")
    chart.rect(54, 748, 1432, 54, "#f2f6f9", radius=8)
    chart.text(76, 781, "英语讽刺：原生英文 dev 仅 +0.3 个百分点，但公开 iSarcasmEval 复合 Skill 下降 7.0 分，不能判为稳定专科收益。", 17)
    chart.save("03-specialization-gains")


def fusion_figure(summary):
    chart = SVG(1540, 800, "固定融合没有超过混合训练：TIES 最接近，但仍损失约 22%", "比较的是相同训练数据曝光总量：混合模型约 15M tokens，等于五个专家各约 3M 的总和。")
    x0, width = 300, 710
    chart.text(655, 163, "Language Skill / 100", 21, anchor="middle", weight=700)
    chart.text(1190, 163, "相对混合训练", 21, anchor="middle", weight=700)
    for tick in [0, 10, 20, 30, 40]:
        x = x0+width*tick/40
        chart.line(x, 190, x, 610)
        chart.text(x, 640, tick, 17, MUTED, "middle")
    for i, code in enumerate(["MIXED"]+FUSIONS):
        row = summary[code]
        y = 245+i*102
        chart.text(270, y+8, NAMES[code], 23, anchor="end", weight=700 if code == "MIXED" else 400)
        chart.rect(x0, y-23, width*row["language_skill"]/40, 47, COLORS[code], radius=4)
        chart.text(x0+width*row["language_skill"]/40+14, y+8, f'{row["language_skill"]:.1f}', 24, COLORS[code], weight=700)
        if code == "MIXED":
            chart.text(1190, y+8, "匹配控制", 22, MUTED, "middle")
        else:
            chart.text(1190, y-2, f'{row["language_skill_delta"]:+.1f} 分', 24, "#ad3d48", "middle", 700)
            chart.text(1190, y+28, f'相对下降 {abs(row["relative_skill_change_percent"]):.0f}%', 18, MUTED, "middle")
    chart.line(x0+width*summary["MIXED"]["language_skill"]/40, 190,
               x0+width*summary["MIXED"]["language_skill"]/40, 610, COLORS["MIXED"], 2, "6 6")
    chart.rect(54, 685, 1432, 72, "#f2f6f9", radius=8)
    chart.text(76, 715, "TIES、权重平均：各部署一个 0.6B 模型；概率集成：同一问题需要五个专家输出，不是单模型权重融合。", 18)
    chart.text(76, 743, "本图仅为当前单 seed、短预算实验的观察结果，不把弱专家与融合干扰解释成同一个原因。", 16, MUTED)
    chart.save("04-fusion-language-skill")


def radar_figure(scores, summary):
    chart = SVG(1540, 1090, "能力轮廓：融合损失主要落在 HellaSwag、ACOS 和 RAGTruth", "九条轴均为公开 benchmark Skill，统一 0–100；雷达面积不是 Language 总分。")
    cx, cy, radius = 600, 565, 352
    ids = [11, 12, 29, 38, 39, 40, 41, 42, 59]
    angles = [-math.pi/2 + i*2*math.pi/len(ids) for i in range(len(ids))]
    def point(angle, value):
        return cx+radius*value/100*math.cos(angle), cy+radius*value/100*math.sin(angle)
    for level in [20, 40, 60, 80, 100]:
        chart.polygon([point(a, level) for a in angles], GRID, "#fafcfd" if level == 100 else "none")
    # Redraw inner grids above the outer background.
    for level in [20, 40, 60, 80]:
        chart.polygon([point(a, level) for a in angles], GRID)
    for angle, bid in zip(angles, ids):
        x, y = point(angle, 100)
        chart.line(cx, cy, x, y)
        x, y = point(angle, 116)
        anchor = "start" if math.cos(angle) > .2 else "end" if math.cos(angle) < -.2 else "middle"
        chart.text(x, y+7, TASKS[bid], 21, anchor=anchor)
    for level in [20, 40, 60, 80, 100]:
        chart.text(cx+12, cy-radius*level/100-5, level, 14, MUTED)
    for code in ["LINEAR", "PROBABILITY_MEAN", "TIES", "MIXED"]:
        points = [point(a, scores[code, bid]) for a, bid in zip(angles, ids)]
        chart.polygon(points, COLORS[code], COLORS[code], 3.5 if code == "MIXED" else 2.5, .07, "7 4" if code == "LINEAR" else None)
        for x, y in points:
            chart.circle(x, y, 4.5 if code == "MIXED" else 3.5, COLORS[code])
    for i, code in enumerate(["MIXED", "TIES", "PROBABILITY_MEAN", "LINEAR"]):
        y = 340+i*107
        chart.line(1120, y, 1170, y, COLORS[code], 4)
        chart.text(1190, y+7, NAMES[code], 23, COLORS[code], weight=700)
        chart.text(1190, y+38, f'Language {summary[code]["language_skill"]:.1f}', 18, MUTED)
    chart.lines(1110, 827, ["TIES 的 FinEntity 略有提升；", "概率集成的 VAST 有所提升。", "局部收益没有抵消整体下降。"], 18, MUTED, "start")
    chart.text(54, 1043, "Qwen3-0.6B / LoRA r32；同一完整评测协议，31,715 请求。没有用未测方向或缺失分数补齐雷达图。", 17, MUTED)
    chart.save("05-fusion-radar")


def main():
    report = json.loads((SOURCE / "fusion-report.json").read_text())
    records = report["records"]
    score_rows = read_csv("nine-benchmark-scores.csv")
    scores = {(r["code"], int(r["catalog_id"])): float(r["skill"]) for r in score_rows}
    assert len(score_rows) == len(scores) == 81
    assert set(scores) == {(c, b) for c in MODELS for b in BENCHMARKS}
    for (code, bid), value in scores.items():
        close(value, records[code]["public"]["per_benchmark"][str(bid)]["skill_0_100"])
    for record in records.values():
        assert record["public"]["language_complete"]
        assert record["public"]["coverage"] == 1
        assert record["public"]["counts"] == {"ok": 31715}
        assert record["natural"]["scored_pairs"] == 67202
    tokens = sum(records[c]["train"]["actual_new_tokens"] for c in EXPERTS)
    assert tokens == records["MIXED"]["train"]["actual_new_tokens"] == 14995340
    assert report["actual_exposure_audit"]["mixed_actual_source_row_multiset_exactly_expert_sum"]
    summary = {}
    mixed_skill = records["MIXED"]["public"]["language_skill"]
    for code in MODELS:
        record, natural = records[code], records[code]["natural"]
        skill = record["public"]["language_skill"]
        summary[code] = {
            "code": code, "name": NAMES[code], "language_skill": skill,
            "language_skill_delta": skill-mixed_skill,
            "relative_skill_change_percent": 100*(skill-mixed_skill)/mixed_skill,
            "natural_pair_macro_f1": natural["pair_metrics"]["macro_f1"],
            "native_10task_macro": natural["native_question_metrics"]["macro_native_independent_dev"],
            "D0_task_macro_pair_f1": natural["D0_task_macro_f1"],
            "new_training_tokens": record["train"]["actual_new_tokens"] if record["train"] else 0,
            "optimizer_steps": record["train"]["optimizer_steps"] if record["train"] else 0,
            "inference_models": 5 if code == "PROBABILITY_MEAN" else 1,
        }
    for row in read_csv("fusion-deltas.csv"):
        code = row["code"]
        for source_key, calculated_key in [
            ("Language_Skill_delta", "language_skill_delta"),
            ("pair_Macro_F1_delta", "natural_pair_macro_f1"),
            ("native_10task_macro_delta", "native_10task_macro"),
            ("D0_task_macro_pair_F1_delta", "D0_task_macro_pair_f1"),
        ]:
            value = summary[code][calculated_key]
            if calculated_key != "language_skill_delta":
                value -= summary["MIXED"][calculated_key]
            close(value, float(row[source_key]))
    native = {(r["code"], r["task"]): float(r["value"]) for r in read_csv("native-dev-metrics.csv")}
    specialization = []
    for row in read_csv("specialization-dev.csv"):
        code = row["code"]
        expert = mean(native[code, task] for task in FAMILY_TASKS[code])
        mixed = mean(native["MIXED", task] for task in FAMILY_TASKS[code])
        close(expert, float(row["specialist_native_family_mean"]))
        close(mixed, float(row["mixed_same_family_mean"]))
        close(expert-mixed, float(row["difference"]))
        specialization.append({
            "code": code, "expert_native": expert, "mixed_native": mixed,
            "native_difference_pp": 100*(expert-mixed),
            "expert_public_skill": scores[code, OWN[code]],
            "mixed_public_skill": scores["MIXED", OWN[code]],
            "public_skill_difference": scores[code, OWN[code]]-scores["MIXED", OWN[code]],
        })
    delta_rows = [dict(code=c, benchmark_id=b, benchmark=TASKS[b], skill=scores[c, b], mixed_skill=scores["MIXED", b], delta=scores[c, b]-scores["MIXED", b]) for c in MODELS for b in BENCHMARKS]
    write_csv("model-summary.csv", list(summary.values()))
    write_csv("specialization-comparison.csv", specialization)
    write_csv("benchmark-differences.csv", delta_rows)
    files = ["fusion-report.json", "nine-benchmark-scores.csv", "specialization-dev.csv", "native-dev-metrics.csv", "fusion-deltas.csv"]
    facts = {
        "scope": "Archived Language-only aggregate verification; no model loading, training or official per-request rescoring.",
        "source_sha256": {str((SOURCE/f).relative_to(ROOT)): hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in files},
        "summary": summary, "specialization": specialization,
        "benchmark_scores_checked": 81, "native_task_rows_checked": len(native),
        "actual_training_tokens_experts_and_mixed": tokens,
        "all_schemes_requests": 31715, "natural_dev_pairs": 67202,
        "errors": 0, "coverage": 1.0,
    }
    heatmap(scores, records)
    heatmap(scores, records, delta=True)
    specialization_figure(specialization)
    fusion_figure(summary)
    radar_figure(scores, summary)
    (OUT / "analysis-facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": True, "score_cells_checked": 81, "native_task_rows_checked": len(native), "figures": 5, "formats": ["svg", "png"], "matched_tokens": tokens}, ensure_ascii=False))


if __name__ == "__main__":
    main()
