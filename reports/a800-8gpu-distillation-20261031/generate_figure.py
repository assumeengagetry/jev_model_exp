#!/usr/bin/env python3
"""Render the proposed A800 allocation; does not contact a GPU or launch a job."""

import importlib.util
import json
import sys
from pathlib import Path


OUT = Path(__file__).resolve().parent
HELPER = OUT.parent / "teacher-distillation-evidence-20261030/generate_figures.py"
spec = importlib.util.spec_from_file_location("a800_report_svg", HELPER)
module = importlib.util.module_from_spec(spec)
previous = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
finally:
    sys.dont_write_bytecode = previous
module.OUT = OUT
SVG, MUTED = module.SVG, module.MUTED


def main():
    figure = SVG(1560, 1000, "8×A800：先并行标注，再并行训练三个 Student", "按 80GB 卡规划；这是 GPU 分配设计，不是已运行结果。40GB 卡采用报告中的 TP4 替代配置。")
    x1, x2, width = 260, 895, 565
    y0, row = 216, 72
    figure.text(x1+width/2, 160, "阶段 A：Teacher 离线标注", 24, anchor="middle", weight=700)
    figure.text(x2+width/2, 160, "阶段 B：Student 等量续训", 24, anchor="middle", weight=700)
    figure.text(x1+width/2, 196, "4 副本 × 每副本 TP2", 19, MUTED, "middle")
    figure.text(x2+width/2, 196, "3 组 × 每组两卡 DDP＋两卡评测", 19, MUTED, "middle")
    for gpu in range(8):
        figure.text(210, y0+gpu*row+43, f"GPU {gpu}", 23, anchor="end", weight=700)
    teacher_fills = ["#f8dfbd", "#f5e8d1", "#f8dfbd", "#f5e8d1"]
    for replica in range(4):
        y = y0+replica*2*row
        figure.rect(x1, y, width, row*2-6, teacher_fills[replica], radius=9)
        figure.text(x1+26, y+49, f"Teacher 副本 {replica} · TP2 · BF16", 24, "#a8651b", weight=700)
        figure.text(x1+26, y+95, f"GPU {replica*2}、{replica*2+1} · 不同训练分片 · 共享磁盘权重", 18)
    groups = [
        ("HARD", "原始标签 CE", "#d5ece3", "#267c68"),
        ("SOFT", "Teacher 概率 KL", "#f8dfbd", "#a8651b"),
        ("HYBRID", "0.5 CE＋0.5 KL", "#e5dcf1", "#7455a2"),
        ("起始 Student 评测／辅助", "旧模型的新机参照，不是 HARD 新训练", "#e5edf5", "#176e98"),
    ]
    for group, (name, loss, fill, color) in enumerate(groups):
        y = y0+group*2*row
        figure.rect(x2, y, width, row*2-6, fill, radius=9)
        figure.text(x2+26, y+49, name, 24, color, weight=700)
        figure.text(x2+26, y+95, loss, 19)
    figure.text(842, y0+row*4, "→", 36, MUTED, "middle", 700)
    figure.rect(54, 830, 1452, 112, "#f2f6f9", radius=10)
    figure.text(78, 868, "Teacher 标注完成后，先释放本轮 Teacher 显存，再运行三组 Student。", 23, weight=700)
    figure.text(78, 905, "每组：microbatch32／卡 × grad acc2 × 2 卡＝有效 batch128；BF16、LoRA r32、LR 5e-5。", 20, MUTED)
    figure.save("a800-gpu-allocation")
    assert 4*2 == 8 and 3*2+2 == 8 and 32*2*2 == 128 and 16*4*2 == 128
    print(json.dumps({"passed": True, "gpu_count": 8, "teacher_replicas": 4, "tp_per_replica": 2,
                      "student_groups": 3, "gpus_per_student": 2, "effective_batch": 128,
                      "new_training_jobs_started": 0, "figure_pairs": 1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
