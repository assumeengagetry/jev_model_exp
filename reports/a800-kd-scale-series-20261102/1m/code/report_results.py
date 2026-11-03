#!/usr/bin/env python3
"""Four-model pilot comparison; never advertise a sampled score as an official run."""
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from kd_runtime import ROOT, SOURCE_ROOT, atomic_json

NAMES = ["BASELINE", "KD_HARD", "KD_SOFT", "KD_HYBRID"]
AXES = ["knowledge", "language", "retrieval", "tools", "arts"]


def main():
    out = ROOT/"outputs/report"
    out.mkdir(parents=True,exist_ok=True)
    training=json.loads((ROOT/"selection/train-manifest.json").read_text())
    scores = {}
    native = {}
    for name in NAMES:
        folder = ROOT/"outputs/evaluation"/name
        completion = json.loads((folder/"complete.json").read_text())
        if not completion.get("complete") or completion.get("scope")!="pilot_subset":
            raise RuntimeError("Four identical-scope pilot evaluations must finish before comparison")
        scores[name] = json.loads((folder/"scores.json").read_text())
        native[name] = {str(x["catalog_id"]):x for x in json.loads((folder/"benchmark-summary.json").read_text())["benchmarks"]}
        native[name].update(json.loads((folder/"added-reports.json").read_text()))
    values = {name:{area["id"]:100*area["skill"] for area in scores[name]["areas"]} for name in NAMES}
    with (out/"four-model-five-area.csv").open("w",newline="") as file:
        writer=csv.writer(file);writer.writerow(["model",*AXES,"pilot_index","pilot_coverage"])
        for name in NAMES:writer.writerow([name,*(values[name][a] for a in AXES),scores[name]["index"],scores[name]["coverage"]])
    rows=[]
    for name in NAMES:
        for bid,b in native[name].items():
            point=scores[name]["benchmarks"].get(bid,{})
            rows.append(dict(model=name,catalog_id=bid,dataset=b["dataset"],native_metric=b.get("metric"),native_score=b.get("score"),
                             selected_requests=b["requests"],answered=b["answered"],coverage=point.get("coverage"),pilot_skill=point.get("skill")))
    with (out/"native-benchmark-metrics.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparisons=[("KD_SOFT","KD_HARD"),("KD_HYBRID","KD_HARD"),("KD_HARD","BASELINE"),("KD_SOFT","BASELINE"),("KD_HYBRID","BASELINE")]
    differences={}
    for candidate,reference in comparisons:
        differences[candidate+" - "+reference]=dict(pilot_index=scores[candidate]["index"]-scores[reference]["index"],
            areas={a:values[candidate][a]-values[reference][a] for a in AXES},
            benchmarks={bid:100*(b["skill"]-scores[reference]["benchmarks"][bid]["skill"]) for bid,b in scores[candidate]["benchmarks"].items() if b.get("in_index")})
    atomic_json(out/"differences.json",differences)
    angles=np.linspace(0,2*np.pi,len(AXES),endpoint=False);closed=np.append(angles,angles[0])
    fig,ax=plt.subplots(figsize=(8,7),subplot_kw={"projection":"polar"})
    ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1)
    for name in NAMES:
        points=[values[name][a] for a in AXES];ax.plot(closed,points+[points[0]],label=name,linewidth=2)
    ax.set_xticks(angles,[a.title() for a in AXES]);ax.set_ylim(0,100)
    ax.set_title(f"KD {training['pairs']:,} pairs: frozen-subset five-area Skill",pad=24)
    ax.legend(loc="upper right",bbox_to_anchor=(1.4,1.1))
    fig.savefig(out/"four-model-five-area-radar.png",dpi=180,bbox_inches="tight");fig.savefig(out/"four-model-five-area-radar.svg",bbox_inches="tight");plt.close(fig)
    ids=[str(bid) for area in scores["BASELINE"]["areas"] for bid in area["benchmarks"]]
    matrix=np.array([[entry["benchmarks"][bid] for entry in differences.values()] for bid in ids])
    fig,ax=plt.subplots(figsize=(13,max(10,len(ids)*0.32)))
    sns.heatmap(matrix,ax=ax,cmap="RdBu_r",center=0,annot=True,fmt="+.2f",xticklabels=list(differences),
                yticklabels=[f"{bid}: {native['BASELINE'][bid]['dataset']}" for bid in ids],cbar_kws={"label":"Pilot benchmark Skill delta (points)"})
    ax.set_title(f"KD {training['pairs']:,} pairs: per-benchmark deltas, not a full leaderboard run")
    fig.savefig(out/"benchmark-deltas-heatmap.png",dpi=180,bbox_inches="tight");fig.savefig(out/"benchmark-deltas-heatmap.svg",bbox_inches="tight");plt.close(fig)
    evaluation=json.loads((ROOT/"selection/evaluation.json").read_text())
    times={name:json.loads((ROOT/"outputs"/name/"complete.json").read_text()) for name in NAMES[1:]}
    production={}
    for size in [2000000,3000000]:
        path=SOURCE_ROOT/"staged/scale-series-20261102/producer"/f"{size//1000000}m/outputs/teacher/complete.json"
        if size<=training["pairs"] and path.exists():
            production[f"{size//1000000}m"]=json.loads(path.read_text())
    new_pairs=sum(phase["newly_written_pairs_total"] for phase in production.values())
    atomic_json(out/"timing-throughput.json",dict(teacher_snapshot_materialization_calls=0,
                upstream_new_teacher_pairs=new_pairs,upstream_producer_phases=production,students=times))
    lines=[f"# 蒸馏规模阶段：{training['pairs']:,} pairs","",f"训练每组 {training['pairs']:,} pairs，一遍；同一起点、同样本和顺序、同有效 batch128。",
           f"评测采用冻结37项的 {evaluation['selected_cases']:,} 个完整 case、{evaluation['selected_requests']:,} 个请求、{evaluation['candidate_pairs']:,} 个完整候选 pair。",
           "本报告是固定子集上的 pilot，不是完整公共榜单成绩，不是私测 Full；原生指标不能统一叫 accuracy。", "",
           "| 模型 | Knowledge | Language | Retrieval | Tools | Arts | Pilot Index | 子集 coverage |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:lines.append("| "+name+" | "+" | ".join(f"{values[name][a]:.2f}" for a in AXES)+f" | {scores[name]['index']:.2f} | {100*scores[name]['coverage']:.2f}% |")
    lines += ["","## 首要比较",""]
    for label,entry in differences.items():lines.append(f"- **{label}**：Pilot Index **{entry['pilot_index']:+.2f}**；"+"，".join(f"{a.title()} {entry['areas'][a]:+.2f}" for a in AXES)+"。")
    lines += ["","SOFT−HARD 才回答纯软监督是否优于等量原标签续训；不能只凭相对起点提升就判定蒸馏有效。各档按四域全池比例构造嵌套样本，优先复用已有标签；评测为固定子集，弱差异应视为趋势不明确，不能外推为完整榜单结论。", "", "## 收益最大的 benchmark",""]
    for label in list(differences)[:2]:
        lines += ["### "+label,"","| benchmark | Pilot Skill 差值 | 原生指标 | 续训模型 | HARD |","|---|---:|---|---:|---:|"]
        candidate=label.split(" - ")[0]
        for bid,change in sorted(differences[label]["benchmarks"].items(),key=lambda x:x[1],reverse=True)[:6]:
            a,b=native[candidate][bid],native["KD_HARD"][bid]
            lines.append(f"| {a['dataset']} | {change:+.2f} | {a.get('metric')} | {a.get('score')} | {b.get('score')} |")
        lines.append("")
    lines += ["## 运行时间与吞吐","",f"- Teacher：本档训练快照直接读取共享概率缓存；本轮截至该档上游累计新标注 **{new_pairs:,} pairs**。快照物化本身不调用 Teacher，不代表标签生成成本为0。"]
    for phase,t in production.items():
        lines.append(f"- Teacher {phase} producer：本阶段新标注 {t['newly_written_pairs_total']:,} pairs，annotation wall time {t['annotation_seconds']/3600:.2f} 小时，{t['newly_written_pairs_per_second']:.2f} new pairs/s。")
    for name,t in times.items():lines.append(f"- {name}：{t['elapsed_seconds']/60:.1f} 分钟，{t['pairs_per_second']:.1f} pairs/s，{t['effective_tokens_per_second']:.0f} effective tokens/s。")
    lines += ["","![四模型五维雷达图](four-model-five-area-radar.png)","","![逐 benchmark 差值热力图](benchmark-deltas-heatmap.png)"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    atomic_json(ROOT/"outputs/pilot-complete.json",dict(complete=True,scope="staged_scale_experiment",training_pairs=training["pairs"],report=str(out/"REPORT.md"),differences=differences))
    print(json.dumps({"event":"pilot_report_complete","report":str(out/"REPORT.md"),"soft_minus_hard":differences["KD_SOFT - KD_HARD"]["pilot_index"],"hybrid_minus_hard":differences["KD_HYBRID - KD_HARD"]["pilot_index"]}),flush=True)


if __name__ == "__main__":main()
