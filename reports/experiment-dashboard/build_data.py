"""Build an offline dashboard from completed local reports; never run models."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
CATALOG = [(11, "ContractNLI"), (12, "ANLI"), (29, "HellaSwag"), (38, "ACOS"),
           (39, "FinEntity"), (40, "iSarcasmEval"), (41, "VAST"), (42, "NLI4CT"), (59, "RAGTruth")]
STAGES = {"initial": "首轮基线", "controlled": "数据 × attention", "D2": "目标数据与任务采样",
          "mixture": "数据池混合比例", "native-sampling": "原生讽刺采样", "joint": "联合配方确认",
          "rank": "LoRA容量", "engineering": "应急链路验证", "fastscreen": "小池快筛", "pair-ratio": "正负pair比例",
          "specialist": "数据专用模型", "matched_mixed": "总曝光匹配混合训练", "weight_fusion": "权重融合", "output_ensemble": "输出概率集成", "student_distillation": "Teacher蒸馏对照"}


def main():
    rows = list(csv.DictReader((PROJECT / "reports/language-understanding-study-20261029/runs.csv").open()))
    runs = []
    for i, row in enumerate(rows):
        runs.append({"id": f"run-{i}", "code": row["code"], "name": row["name"], "stage": row["stage"],
                     "round": row["round"], "seed": int(row["seed"]), "rank": int(row["rank"]),
                     "tokens": int(row["new_tokens"]), "score": float(row["language_skill"]),
                     "skills": [float(row[f"skill_{cid}"]) for cid, _ in CATALOG],
                     "coverage": float(row["coverage"]), "requests": int(row["requests"]),
                     "checkpoint": row["checkpoint"], "source": row["source"], "kind": "run", "n": 1})
    for path in sorted((PROJECT / "research").glob("lu-pair-ratio-screen-*/round-report.json")):
        report = json.loads(path.read_text())
        for arm in report["arms"]:
            name = arm["arm"]
            score_path = path.parent / "runs" / name / "arms" / name / "evaluation/language-score.json"
            score = json.loads(score_path.read_text())
            if not score.get("language_complete"):
                continue
            exposure = arm["actual_exposure"]
            runs.append({"id": f"ratio-{path.parent.name}-{name}", "code": name,
                         "name": f"正pair {exposure['positive_row_fraction']:.2%}" + ("（自然比例）" if name == "NATURAL" else ""),
                         "stage": "pair-ratio", "round": path.parent.name, "seed": report["seed"], "rank": 32,
                         "tokens": arm["train"]["actual_new_tokens"], "score": score["language_skill_0_100"],
                         "skills": [score["benchmarks"][str(cid)]["skill_0_100"] for cid, _ in CATALOG],
                         "coverage": score["language_coverage"], "requests": score["requests"],
                         "checkpoint": str(score_path.parent.parent / "main/final"), "source": str(score_path),
                         "kind": "run", "n": 1, "positive_fraction": exposure["positive_row_fraction"],
                         "dev_f1": arm["natural_validation"]["pair_metrics"]["macro_f1"]})
    for path in sorted((PROJECT / "research").glob("*/dashboard-runs.csv")):
        for i, row in enumerate(csv.DictReader(path.open())):
            assert float(row["coverage"]) == 1 and int(row["requests"]) == 31715
            public = json.loads(Path(row["source"]).read_text())
            assert public.get("language_complete") is True
            namespace = "fusion" if path.parent.name.startswith("lu-specialist-fusion-") else "experiment"
            runs.append({"id": f"{namespace}-{path.parent.name}-{row['code']}", "code": row["code"], "name": row["name"],
                         "stage": row["stage"], "round": row["round"], "seed": int(row["seed"]), "rank": int(row["rank"]),
                         "tokens": int(row["new_tokens"]), "score": float(row["language_skill"]),
                         "skills": [float(row[f"skill_{cid}"]) for cid, _ in CATALOG], "coverage": 1., "requests": 31715,
                         "checkpoint": row["checkpoint"], "source": row["source"], "kind": "run", "n": 1,
                         "deployment": "ensemble" if row["stage"] == "output_ensemble" else "dense_merged" if row["stage"] == "weight_fusion" else "lora"})
    # A model can finish before the four-model stage publishes its aggregate CSV.
    # Import only its complete official JSON, so the page updates at each real endpoint.
    seen_sources = {row["source"] for row in runs}
    specialist_names = {"ANLI": "ANLI专用", "HELLASWAG": "HellaSwag专用", "ACOS": "ACOS专用", "ISARCASM": "英语讽刺专用",
                        "RAGTRUTH": "RAGTruth专用", "MIXED": "相同总曝光混合训练", "LINEAR": "等权有效权重平均",
                        "TIES": "固定TIES，保留20%", "PROBABILITY_MEAN": "五专家等权概率集成"}
    for folder in sorted((PROJECT / "research").glob("lu-specialist-fusion-*")):
        for path in sorted(folder.glob("runs/*/arms/*/evaluation/result.json")):
            if str(path) in seen_sources: continue
            score = json.loads(path.read_text()); arm = path.parent.parent.name
            if not score.get("language_complete") or arm not in specialist_names: continue
            assert score["coverage"] == 1 and score["counts"] == {"ok": 31715}
            training = path.parent.parent / "main/complete.json"
            train = json.loads(training.read_text()) if training.exists() else {}
            stage = "matched_mixed" if arm == "MIXED" else "output_ensemble" if arm == "PROBABILITY_MEAN" else "weight_fusion" if arm in ("LINEAR", "TIES") else "specialist"
            runs.append({"id": f"fusion-{folder.name}-{arm}", "code": arm, "name": specialist_names[arm], "stage": stage,
                         "round": folder.name, "seed": 3407, "rank": 32, "tokens": train.get("actual_new_tokens", 0),
                         "score": score["language_skill"], "skills": [score["per_benchmark"][str(cid)]["skill_0_100"] for cid, _ in CATALOG],
                         "coverage": 1., "requests": 31715, "checkpoint": score.get("checkpoint") or "", "source": str(path),
                         "kind": "run", "n": 1, "deployment": "ensemble" if stage == "output_ensemble" else "dense_merged" if stage == "weight_fusion" else "lora"})
            seen_sources.add(str(path))
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["stage"], run["round"], run["code"], run["tokens"], run["rank"])].append(run)
    groups = []
    for group_key, members in grouped.items():
        first = members[0]
        stable_id = hashlib.sha256(json.dumps(group_key, ensure_ascii=False).encode()).hexdigest()[:16]
        group = dict(first, id=f"group-{stable_id}", kind="group", n=len(members), seed=None,
                     score=statistics.mean(r["score"] for r in members),
                     std=statistics.stdev(r["score"] for r in members) if len(members) > 1 else 0,
                     skills=[statistics.mean(r["skills"][axis] for r in members) for axis in range(9)],
                     member_ids=[r["id"] for r in members])
        groups.append(group)
    weight_paths = [r["checkpoint"] for r in runs if r["checkpoint"]]
    assert len(set(weight_paths)) == len(weight_paths)
    assert len({r["id"] for r in runs}) == len(runs)
    assert all(len(r["skills"]) == 9 and all(0 <= v <= 100 for v in r["skills"]) for r in runs)
    inventory = json.loads((PROJECT / "reports/length-label-benchmark-20261029/data-inventory.json").read_text())
    lengths = json.loads((PROJECT / "reports/length-label-benchmark-20261029/length-results.json").read_text())
    fusion_paths = sorted((PROJECT / "research").glob("lu-specialist-fusion-*/fusion-report.json"))
    latest_fusion = json.loads(fusion_paths[-1].read_text()) if fusion_paths else {}
    fusion_done = latest_fusion.get("complete") is True
    fusion_detail = "五个数据家族专家与混合控制已启动；随后比较输出集成、权重平均和TIES。"
    if fusion_done:
        names = {"MIXED": "混合", "TIES": "TIES", "LINEAR": "权重平均", "PROBABILITY_MEAN": "概率集成"}
        fusion_detail = "首轮9/9完成：" + "、".join(f"{name}{latest_fusion['records'][code]['public']['language_skill']:.2f}" for code, name in names.items()) + "。"
    teacher_root = PROJECT / "research/lu-teacher-distillation-20261029"
    teacher_detail = "已固定榜首KnowLine31B（Full71.86）并开始镜像下载；9998对训练数据、4166道Teacher问题已准备。"
    labels_path = teacher_root / "teacher-labels-ready.json"
    if labels_path.exists():
        label_data = json.loads(labels_path.read_text())
        teacher_detail = f"Teacher已输出{label_data['labelled_pairs']}对完整候选软标签，接续人工标签/软标签/混合监督的Student对照。"
    elif (teacher_root / "teacher-ready.json").exists():
        teacher_detail = "榜首KnowLine31B权重已下载验证；Teacher原生概率标注阶段接续中。"
    best = max(runs, key=lambda r: r["score"])
    data = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "edition": "Decision Index 0.3 · Language Understanding",
            "axes": [{"id": cid, "name": name} for cid, name in CATALOG], "stages": STAGES,
            "runs": runs, "groups": groups, "best_id": best["id"], "counts": {"full": len(runs), "dev_only": 8, "resource_cells": 49},
            "inventory": inventory["combined"], "length_results": [{"cap": r["cap"], "macro_f1": r["weighted"]["macro_f1"],
                 "seconds": r["inference_wall_seconds"], "allocated_gib": r["peak_allocated_gib"]} for r in lengths],
            "roadmap": [
                {"title": "分类基线与完整九项评测", "status": "done", "detail": "R0 Language 17.74；训练、保存重载、九项计分已跑通。"},
                {"title": "数据、attention、采样与LoRA容量", "status": "done", "detail": "当前主线r32，四seed均值35.05；历史最高单次36.33。"},
                {"title": "小池快筛与正负比例", "status": "done", "detail": "117353对缓存；最新四组比例首轮已完整评测，55.37%档Language 31.23。"},
                {"title": "输入长度、batch与checkpointing", "status": "done", "detail": "长度分布、256–8192资源阶梯及49个测量单元已完成。"},
                {"title": "雷达图实验页面", "status": "done", "detail": "当前页面；支持九项Skill、seed均值、单次结果、筛选和导出。"},
                {"title": "每数据集专用模型 → 模型融合", "status": "done" if fusion_done else "running", "detail": fusion_detail},
                {"title": "榜首Teacher → 0.6B蒸馏", "status": "running", "detail": teacher_detail},
                {"title": "更大基座对照", "status": "later", "detail": "1.7B/4B/8B资产已备齐；优先完成融合与蒸馏路线。"}]}
    text = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (ROOT / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "data.js").write_text("window.EXPERIMENT_DATA = " + text + ";\n")
    print(json.dumps({"runs": len(runs), "groups": len(groups), "best": best["score"], "page": str(ROOT / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
