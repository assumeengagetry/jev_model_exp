#!/usr/bin/env python3
"""Render experiment-design diagrams; no training or model execution."""

import importlib.util
import json
import sys
from pathlib import Path


OUT = Path(__file__).resolve().parent
HELPER = OUT.parent / "teacher-distillation-evidence-20261030/generate_figures.py"
spec = importlib.util.spec_from_file_location("report_svg_helper", HELPER)
helper = importlib.util.module_from_spec(spec)
previous_bytecode_setting = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    spec.loader.exec_module(helper)
finally:
    sys.dont_write_bytecode = previous_bytecode_setting
helper.OUT = OUT
SVG = helper.SVG
INK, MUTED, GRID = helper.INK, helper.MUTED, helper.GRID


def arrow(chart, x1, y1, x2, y2, color="#71899c"):
    chart.line(x1, y1, x2, y2, color, 2)
    if abs(y2-y1) >= abs(x2-x1):
        direction = 1 if y2 >= y1 else -1
        chart.polygon([(x2, y2), (x2-6, y2-direction*10), (x2+6, y2-direction*10)], color, color)
    else:
        direction = 1 if x2 >= x1 else -1
        chart.polygon([(x2, y2), (x2-direction*10, y2-6), (x2-direction*10, y2+6)], color, color)


def workflow(plan):
    chart = SVG(1640, 1010, "融合结束后，只新增 HARD／SOFT／HYBRID 三组蒸馏续训", "实验设计图：复用已完成阶段的 Student 和数据；不重训专科，不再次模型融合。")
    chart.rect(54, 151, 730, 142, "#edf4f9", radius=12)
    chart.text(80, 193, "继承同一个已有 Student", 24, "#176e98", weight=700)
    chart.text(80, 232, "融合实验结束后，固定一个 0.6B checkpoint。", 21)
    chart.text(80, 270, "三组继承完全相同的权重、分类头与适配结构。", 18, MUTED)
    chart.rect(850, 151, 736, 142, "#fcf2e7", radius=12)
    chart.text(876, 193, "沿用训练池＋Teacher 离线标注", 24, "#a8651b", weight=700)
    chart.text(876, 232, "KnowLine 31B 输出每个 pair 的匹配／不匹配概率。", 20)
    chart.text(876, 270, "数据缓存与软标签复用，不重新建设专科数据。", 18, MUTED)
    arrow(chart, 419, 293, 419, 349)
    arrow(chart, 1218, 293, 1218, 349)
    chart.rect(54, 349, 1532, 92, "#f2f6f9", radius=10)
    chart.text(820, 386, "同一个 Student 起点＋同一批输入／顺序＋同样的续训量，只改变监督", 24, anchor="middle", weight=700)
    chart.text(820, 422, "BF16、r32／alpha64、microbatch32 × grad acc4、有效 batch128；续训 LR 5e-5。", 19, MUTED, "middle")
    for x, title, loss, detail, fill, color in [
        (54, "HARD", "原始标签 CE", "等量续训控制，不能直接拿旧模型代替", "#eaf5f1", "#267c68"),
        (586, "SOFT", "Teacher 概率 KL", "只学习 Teacher，数据与 HARD 一致", "#fcf2e7", "#a8651b"),
        (1118, "HYBRID", "0.5 CE＋0.5 KL", "原始标签与 Teacher 混合监督", "#f1edf8", "#7455a2"),
    ]:
        arrow(chart, x+234, 441, x+234, 508)
        chart.rect(x, 508, 468, 166, fill, radius=12)
        chart.text(x+24, 548, title, 28, color, weight=700)
        chart.text(x+24, 591, loss, 23, color, weight=700)
        chart.text(x+24, 638, detail, 18)
        arrow(chart, x+234, 674, x+234, 738)
    chart.rect(54, 738, 1532, 145, "#edf4f9", radius=12)
    chart.text(80, 780, "沿用相同五方向／37 项评测：起始 Student＋三组续训结果", 25, "#176e98", weight=700)
    chart.text(80, 825, "主比较：SOFT − HARD、HYBRID − HARD；另列三组相对起始 Student 的增益。", 22)
    chart.text(80, 861, "交付雷达图、逐 benchmark 差值热力图、总体柱状图，以及训练与标注成本。", 19, MUTED)
    chart.text(54, 951, "本阶段新增 3 次等量续训，共 3T tokens；上一阶段已有模型只作起点和参照，不计为新训练。", 20, weight=700)
    chart.save("01-experiment-workflow")


def data_matrix(plan):
    chart = SVG(1640, 930, "蒸馏数据分配：三组共用同一训练池，只改变监督", "沿用上一阶段的五方向数据，不重新训练五个专科；颜色不是得分或完成状态。")
    x0, y0, cw, rh = 450, 243, 360, 82
    chart.text(x0+cw*1.5, 151, "共同起点：上一阶段固定的同一个 0.6B Student", 23, anchor="middle", weight=700)
    chart.line(x0, 171, x0+cw*3, 171, MUTED)
    for j, model in enumerate(plan["training_models"]):
        for k, label in enumerate(model["header"]):
            chart.text(x0+(j+.5)*cw, 203+k*27, label, 21, anchor="middle", weight=700)
    palettes = {
        "hard": ("#d5ece3", "#267c68"),
        "soft": ("#f8dfbd", "#a8651b"), "hybrid": ("#e5dcf1", "#7455a2"),
    }
    for i, area in enumerate(plan["areas"]):
        y = y0+i*rh
        chart.text(x0-22, y+40, area["name"], 22, anchor="end", weight=700)
        chart.text(x0-22, y+68, area["dataset"], 17, MUTED, "end")
        for j, model in enumerate(plan["training_models"]):
            x = x0+j*cw
            assert area["id"] in model["areas"]
            fill, ink = palettes[model["kind"]]
            chart.rect(x+1, y+1, cw-2, rh-2, fill)
            chart.text(x+cw/2, y+32, "沿用 "+area["dataset"], 21, ink, "middle", 700)
            chart.text(x+cw/2, y+62, model["loss"], 19, ink, "middle")
    for j, (kind, name) in enumerate([("hard", "原始标签 HARD"), ("soft", "Teacher 软标签 SOFT"), ("hybrid", "原始＋Teacher HYBRID")]):
        x = 240+j*430
        chart.rect(x, 711, 25, 25, palettes[kind][0])
        chart.text(x+38, 731, name, 19)
    chart.rect(54, 786, 1532, 88, "#f2f6f9", radius=8)
    chart.text(78, 821, "每组完整训练同一份 D_DISTILL 一遍：HARD、SOFT、HYBRID 各 T tokens，本轮共新增 3T。", 21, weight=700)
    chart.text(78, 856, "上一阶段 Student 仅作未续训参照；不能将它直接登记为本轮已经完成的 HARD 训练。", 18, MUTED)
    chart.save("02-training-data-matrix")


def main():
    plan = json.loads((OUT / "plan-data.json").read_text())
    areas = {a["id"] for a in plan["areas"]}
    training = plan["training_models"]
    assert len(areas) == 5 and len(training) == 3 and plan["new_fusion_models"] == 0
    assert sum(a["benchmarks"] for a in plan["areas"]) == 37
    assert {m["id"] for m in training} == {"KD_HARD", "KD_SOFT", "KD_HYBRID"}
    assert all(set(m["areas"]) == areas for m in training)
    assert plan["microbatch"] * plan["gradient_accumulation"] == plan["effective_batch"] == 128
    assert plan["performance_scores"] is None
    assert plan["reuse_previous_stage_model_as_hard_result"] is False
    assert plan["stage"] == "distillation_after_fusion"
    workflow(plan)
    data_matrix(plan)
    print(json.dumps({"passed": True, "formal_training_models": len(training), "new_fusions": 0,
                      "result_curves_with_existing_parent": len(training)+len(plan["existing_references"]), "public_benchmarks": 37,
                      "figure_pairs": 2, "training_or_model_execution": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
