#!/usr/bin/env python3
"""Generate an evidence report's figures without loading or training any model."""

import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from html import escape
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
INK, MUTED, GRID = "#172d43", "#566d80", "#d9e4ec"


class SVG:
    def __init__(self, width, height, title, subtitle):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
            '<style>text{font-family:"Noto Sans CJK SC","Source Han Sans CN",sans-serif;font-variant-numeric:tabular-nums}</style>',
        ]
        self.rect(0, 0, width, height, "#ffffff")
        self.text(54, 58, title, 29, weight=700)
        self.text(54, 96, subtitle, 17, color=MUTED)

    def rect(self, x, y, w, h, fill, stroke="none", radius=0):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" rx="{radius}"/>')

    def text(self, x, y, text, size=18, color=INK, anchor="start", weight=400):
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(text))}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, sw=1):
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{sw}"/>')

    def circle(self, x, y, r, fill):
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}"/>')

    def polygon(self, points, color, fill="none", opacity=1, sw=1):
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(f'<polygon points="{points}" fill="{fill}" fill-opacity="{opacity}" stroke="{color}" stroke-width="{sw}"/>')

    def save(self, stem):
        folder = OUT / "figures"
        folder.mkdir(exist_ok=True)
        path = folder / f"{stem}.svg"
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")
        converter = shutil.which("rsvg-convert")
        if not converter:
            raise RuntimeError("rsvg-convert is required for PNG rendering")
        subprocess.run([converter, "--output", str(path.with_suffix(".png")), str(path)], check=True, timeout=30)


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def progress_figure(facts, snapshot):
    chart = SVG(1540, 640, "蒸馏推进到了 Teacher 标注记录，尚缺 Student 效果对照", "颜色表示证据来源，不是完成百分比；历史下载和标注记录不等于当前资产可用。")
    cards = [
        ("Teacher 选型", f"Full {snapshot['leaderboard']['top_three'][0]['full']:.1f}", "线上榜单已核对", "KnowLine 31B Gen1", "数值第一，属于第一并列组", "#eaf5f2", "#267c68"),
        ("Teacher 权重", f"{facts['historical_download_gb']:.1f} GB", "历史报告记载下载完成", f"{facts['historical_download_files']} 个文件完成校验", "原权重与验收文件本地缺失", "#fbf3e8", "#aa6b20"),
        ("Teacher 软标签", f"{facts['reported_labelled_pairs']:,} pair", "较晚看板记载已输出", f"原准备 {facts['planned_teacher_questions']:,} 道问题", "标签本体与 ready 回执缺失", "#fbf3e8", "#aa6b20"),
        ("Student 三组对照", f"{facts['student_result_rows']} 份成绩", "本地完整评测台账未见", "HARD / SOFT / HYBRID", "没有可计算的蒸馏收益", "#f0f3f6", "#637787"),
    ]
    for i, (title, value, status, line1, line2, fill, color) in enumerate(cards):
        x, y, width = 54+i*365, 175, 337
        chart.rect(x, y, width, 300, fill, radius=12)
        chart.text(x+24, y+45, title, 23, weight=700)
        chart.text(x+24, y+111, value, 34, color, weight=700)
        chart.text(x+24, y+157, status, 19, color, weight=700)
        chart.text(x+24, y+214, line1, 17)
        chart.text(x+24, y+251, line2, 17, MUTED)
        if i < 3:
            chart.text(x+width+14, y+147, "→", 23, MUTED, "middle")
    chart.text(54, 546, f"本地看板快照：{facts['dashboard_generated']}；完整评测 {facts['full_result_count']} 个，其中 Student 蒸馏结果 {facts['student_result_rows']} 个。", 18)
    chart.text(54, 587, "9,998 是 pair 数；4,166 是原准备问题数，不能相除当作标注完成率。0 份成绩不代表模型得分为 0。", 17, MUTED)
    chart.save("01-distillation-evidence-status")


def baseline_figure(baselines, student_rows):
    chart = SVG(1540, 830, "已有硬标签成绩不能代替本轮蒸馏的 HARD 控制", "左侧是已完成历史模型；右侧是需要同起点、同样本、同预算比较的三组 Student，缺失成绩不画零分柱。")
    chart.text(440, 168, "已完成：历史 Language Skill", 22, anchor="middle", weight=700)
    chart.text(1175, 168, "待补：本轮配平 Student 对照", 22, anchor="middle", weight=700)
    x0, width = 300, 450
    for tick in [0, 10, 20, 30, 40, 50]:
        x = x0+width*tick/50
        chart.line(x, 210, x, 587)
        chart.text(x, 624, tick, 16, MUTED, "middle")
    for i, row in enumerate(baselines):
        y = 272+i*125
        color = ["#96b0c3", "#5398b7", "#146e96"][i]
        chart.text(270, y+7, row["plot_name"], 22, anchor="end")
        end = x0+width*row["score_mean"]/50
        chart.rect(x0, y-23, end-x0, 46, color, radius=4)
        if row["seed_count"] > 1:
            error = width*row["sample_std"]/50
            chart.line(end-error, y, end+error, y, INK, 2)
            chart.line(end-error, y-8, end-error, y+8, INK, 2)
            chart.line(end+error, y-8, end+error, y+8, INK, 2)
        label = f'{row["score_mean"]:.1f}'
        if row["seed_count"] > 1:
            label += f' ± {row["sample_std"]:.1f}'
        chart.text(end+22, y+8, label, 21, color, weight=700)
        detail = "起始模型，单个终点" if row["code"] == "R0" else f'新增 50M tokens，{row["seed_count"]} 个续训 seed'
        chart.text(x0, y+56, detail, 16, MUTED)
    chart.line(867, 195, 867, 634, GRID, 2)
    for i, row in enumerate(student_rows):
        y = 228+i*125
        chart.rect(923, y, 540, 91, "#f2f5f8", radius=8)
        chart.text(947, y+36, row["arm"], 23, weight=700)
        chart.text(1083, y+36, row["supervision"], 19)
        chart.text(1428, y+36, "未测", 23, MUTED, "end", 700)
        chart.text(947, y+69, "未见本轮训练、dev 或九项完整评测成绩", 16, MUTED)
    chart.rect(54, 679, 1432, 99, "#f2f6f9", radius=8)
    chart.text(76, 712, "误差线为 4 个续训 seed 的样本标准差，不是置信区间；它们共享同一个 R0。", 18)
    chart.text(76, 748, "历史 r32 的 35.0 分是参考水平，不是这批 9,998 pair 上的配平 HARD 分数，不能直接用来计算蒸馏增益。", 17, MUTED)
    chart.save("02-existing-baselines-and-missing-students")


def teacher_radar(snapshot):
    categories = snapshot["leaderboard"]["teacher_public_categories"]
    teacher = snapshot["leaderboard"]["top_three"][0]
    chart = SVG(1460, 940, "Teacher 的五方向能力：这是线上大模型成绩，不是蒸馏后成绩", "KnowLine 31B Gen1 的公开 Skill；五条轴统一 0–100。Full 排名来自私测，另列而不混入雷达轴。")
    cx, cy, radius = 540, 492, 292
    angles = [-math.pi/2+i*2*math.pi/5 for i in range(5)]
    def point(angle, value):
        return cx+radius*value/100*math.cos(angle), cy+radius*value/100*math.sin(angle)
    chart.polygon([point(a, 100) for a in angles], GRID, "#fafcfd")
    for level in [20, 40, 60, 80, 100]:
        chart.polygon([point(a, level) for a in angles], GRID)
        chart.text(cx+12, cy-radius*level/100-6, level, 14, MUTED)
    for angle, category in zip(angles, categories):
        x, y = point(angle, 100)
        chart.line(cx, cy, x, y)
        x, y = point(angle, 120)
        anchor = "start" if math.cos(angle) > .2 else "end" if math.cos(angle) < -.2 else "middle"
        value = category["skill_fraction"]*100
        chart.text(x, y, category["name"], 22, anchor=anchor)
        chart.text(x, y+31, f"{value:.1f}", 23, "#146e96", anchor, 700)
    points = [point(a, c["skill_fraction"]*100) for a, c in zip(angles, categories)]
    chart.polygon(points, "#146e96", "#146e96", .14, 3.5)
    for x, y in points:
        chart.circle(x, y, 5, "#146e96")
    chart.rect(1010, 242, 385, 362, "#f2f6f9", radius=10)
    for i, (label, value) in enumerate([("Full（私测）", teacher["full"]), ("Public（全部公开方向）", teacher["public"]), ("Language（公开）", teacher["public_language_skill"])]):
        y = 286+i*106
        chart.text(1035, y, label, 19, MUTED)
        chart.text(1035, y+47, f"{value:.1f}", 32, "#146e96", weight=700)
    chart.text(1010, 671, "Full 数值最高，但不等于", 19)
    chart.text(1010, 706, "每个方向或训练标签都最好。", 19)
    chart.text(54, 882, "来源：2026-10-30 读取的官方榜单快照（生成于 2026-10-29）；未在本地重新评测 Teacher。", 17, MUTED)
    chart.save("03-teacher-public-profile-radar")


def main():
    source_paths = [
        ROOT / "reports/experiment-dashboard/data.json",
        ROOT / "reports/experiment-dashboard/build_data.py",
        ROOT / "reports/experiment-roadmap-status-20261029.md",
        ROOT / "reports/language-understanding-study-20261029/runs-through-pair-ratio.csv",
        ROOT / "research/lu-specialist-fusion-20261029T152144Z/dashboard-runs.csv",
        OUT / "teacher-snapshot.json",
    ]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    dashboard = json.loads(source_paths[0].read_text())
    old_runs = csv_rows(source_paths[3])
    fusion_runs = csv_rows(source_paths[4])
    snapshot = json.loads(source_paths[5].read_text())
    assert len(old_runs) == 95 and len(fusion_runs) == 9
    assert len(dashboard["runs"]) == dashboard["counts"]["full"] == len(old_runs)+len(fusion_runs) == 104
    def is_student(row):
        return row.get("stage") == "student_distillation" or row.get("code", "").upper() in {"HARD", "SOFT", "HYBRID"}
    student_runs = [row for row in dashboard["runs"] + old_runs + fusion_runs if is_student(row)]
    assert not student_runs
    teacher_detail = next(r["detail"] for r in dashboard["roadmap"] if r["title"] == "榜首Teacher → 0.6B蒸馏")
    labelled_pairs = int(re.search(r"已输出(\d+)对", teacher_detail).group(1))
    roadmap = source_paths[2].read_text()
    prepared_pairs, planned_questions = map(int, re.search(r"(\d+)对/(\d+)道训练问题", roadmap).groups())
    assert prepared_pairs == labelled_pairs == 9998 and planned_questions == 4166
    assert "62.6GB" in roadmap and "31个文件均已验证" in roadmap
    assert 'labels_path.exists()' in source_paths[1].read_text()
    teacher_root = ROOT / "research/lu-teacher-distillation-20261029"
    assert not teacher_root.exists()
    baselines = []
    for code, name in [("R0", "R0"), ("JOINT", "硬标签 r16"), ("LR32", "硬标签 r32")]:
        rows = [r for r in old_runs if r["code"] == code]
        assert len(rows) == (1 if code == "R0" else 4)
        scores = [float(r["language_skill"]) for r in rows]
        assert all(int(r["requests"]) == 31715 and float(r["coverage"]) == 1 for r in rows)
        if code != "R0":
            assert {int(r["new_tokens"]) for r in rows} == {50000000}
        baselines.append({"code": code, "plot_name": name, "score_mean": statistics.mean(scores),
                          "sample_std": statistics.stdev(scores) if len(scores)>1 else 0,
                          "seed_count": len(rows), "tokens_per_endpoint": int(rows[0]["new_tokens"]),
                          "seed_scores": scores})
    for row in baselines:
        if row["code"] != "R0":
            group = next(g for g in dashboard["groups"] if g["code"] == row["code"])
            assert math.isclose(group["score"], row["score_mean"], abs_tol=1e-12)
            assert math.isclose(group["std"], row["sample_std"], abs_tol=1e-12)
    student_rows = [{"arm": arm, "supervision": label, "language_skill": None, "native_dev": None,
                     "trained_checkpoint_found": False, "status": "not_found_in_local_completed_results"}
                    for arm, label in [("HARD", "人工标签"), ("SOFT", "Teacher 软标签"), ("HYBRID", "人工＋Teacher")]]
    facts = {"scope": "Evidence in the current local archive; no training or Teacher/Student execution.",
             "source_sha256": source_hashes, "dashboard_generated": dashboard["generated"],
             "full_result_count": len(dashboard["runs"]), "student_result_rows": len(student_runs),
             "reported_labelled_pairs": labelled_pairs, "planned_teacher_questions": planned_questions,
             "completed_teacher_questions_verified": None, "historical_download_gb": 62.6,
             "historical_download_files": 31, "historical_download_timestamp": "2026-10-30T02:30:52+00:00",
             "local_teacher_root_exists": teacher_root.exists(), "labels_present_locally": False,
             "teacher_label_quality_measured_locally": False, "baselines": baselines, "students": student_rows}
    progress_figure(facts, snapshot)
    baseline_figure(baselines, student_rows)
    teacher_radar(snapshot)
    save_csv("historical-hard-label-baselines.csv", [{k: v for k, v in r.items() if k != "seed_scores"} for r in baselines])
    save_csv("student-comparison-status.csv", student_rows)
    save_csv("teacher-public-areas.csv", [{"area": c["id"], "skill": c["skill_fraction"]*100, "coverage": c["coverage"]} for c in snapshot["leaderboard"]["teacher_public_categories"]])
    (OUT / "analysis-facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for relative, digest in source_hashes.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    print(json.dumps({"passed": True, "complete_archive_results_checked": 104, "student_results_found": 0,
                      "baseline_endpoint_rows_checked": sum(r["seed_count"] for r in baselines),
                      "source_hashes_checked": len(source_hashes), "figure_pairs": 3, "model_executions": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
