"""Render completed measurements only; no training, inference or experiment mutation."""
from collections import Counter
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    inventory = json.loads((OUT / "data-inventory.json").read_text())
    cases = json.loads((OUT / "length-results.json").read_text())
    exposure = json.loads((OUT / "exposure-ratios.json").read_text())
    manifest = json.loads((OUT / "sample-manifest.json").read_text())
    by_cap = {row["cap"]: row for row in cases}
    reference = by_cap["full"]["weighted"]["macro_f1"]
    combined = inventory["combined"]
    weights = Counter()
    counts = Counter()
    for sample in manifest["samples"]:
        weights[sample["bucket"]] += sample["weight"]
        counts[sample["bucket"]] += 1
    total_weight = sum(weights.values())
    lines = [
        "# 输入长度与正负样本比例：快速诊断报告",
        "",
        "## 结论",
        "",
        "1. 在固定模型、同一批dev样本的推理对照中，1024 tokens是偏速度的总体折中候选；2048更保守。"
        "1024相对不额外截断的加权Macro-F1下降0.182个百分点，2048下降0.021个百分点。"
        "但原长1024–2047桶截到1024后下降12.114个百分点，不能仅按总体均值统一裁短所有长文本。",
        "2. 当前117,353对训练池的长度中位数为123 tokens，p95为1066，最长2895。"
        "1024可完整覆盖94.30%，2048覆盖99.75%，4096全部覆盖；当前两池均没有已截断样本。",
        "3. 静态训练池的兼容/不兼容标签比例为41.14%/58.86%；"
        "按BASE-17M归档采样计划，正类出现次数占45.39%、token占47.16%。"
        "候选展开、任务采样和原生类别分布分别影响这些比例，目前没有正负比例训练消融结果，尚不能确定最优配比。",
        "",
        "## 输入长度对效果与速度的影响",
        "",
        "使用历史完整Language最高的LR32/seed3409 checkpoint，只做推理、不更新权重。"
        "固定1667对分层dev样本，来源为D0 validation和D2 dev，按来源×原始长度桶×pair标签抽样并加权。"
        "D0从原始文本恢复，D2使用未截断缓存。长度包含格式、premise、hypothesis和EOS；"
        "截断保留完整hypothesis，premise保留首尾，不是简单地删除尾部。",
        "",
        "| 输入上限 | 加权Macro-F1 | 相对full变化（百分点） | 本面板推理耗时（秒） | 相对4096的耗时比 | 峰值allocated（GiB） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cap in ("256", "512", "1024", "2048", "4096", "full"):
        row = by_cap[cap]
        lines.append(
            f"| {cap} | {row['weighted']['macro_f1']:.4%} | "
            f"{100 * (row['weighted']['macro_f1'] - reference):+.4f} | "
            f"{row['inference_wall_seconds']:.2f} | "
            f"{by_cap['4096']['inference_wall_seconds'] / row['inference_wall_seconds']:.2f}× | "
            f"{row['peak_allocated_gib']:.2f} |"
        )
    lines += [
        "",
        "以上所有长度的面板预测覆盖率均为100%。计时覆盖collation、H2D和同步推理，"
        "不含模型加载、原文tokenization。使用最多32对/批、16384 padded-token预算并按长度排序；"
        "这是单次测量，full在各有限上限之前运行。长尾过采样使本面板远长于自然训练分布，"
        "这些耗时比只描述同面板、同策略的对照，不代表生产吞吐、单请求延迟或训练加速。",
        "",
        "显存表仅为PyTorch峰值allocated。full之后未逐档清空allocator缓存，"
        "reserved峰值各档均约68.92 GiB并继承前档保留量，因此不能把1.8 GiB称作整个进程的显存占用，"
        "也不能用本表推断独立进程所需的最小GPU容量。",
        "",
        "### 自然长度桶与截断干预",
        "",
        "| 原始长度 | 面板对数 | 加权总体占比 | full Macro-F1 | 截到1024 Macro-F1 | 截到2048 Macro-F1 | 截到4096 Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in by_cap["full"]["by_original_length_bucket"]:
        values = [by_cap[cap]["by_original_length_bucket"][bucket]["macro_f1"] for cap in ("full", "1024", "2048", "4096")]
        lines.append(
            f"| {bucket} | {counts[bucket]} | {weights[bucket] / total_weight:.3%} | "
            + " | ".join(f"{value:.3%}" for value in values) + " |"
        )
    lines += [
        "",
        "本dev框架中短于256的加权质量占约92.94%，因此总体指标会弱化长尾退化。"
        "不同自然长度桶同时包含任务/来源难度差异；只在同一桶比较不同上限，才是在本诊断中观察截断的影响。"
        "部分稀少长桶的结果非单调，不能断言每增加长度都必然改善。",
        "",
        "工作定义为：相对full总体Macro-F1下降不超过0.5个百分点、且预测覆盖完整的最短上限。"
        "按此规则为1024。若要保留当前训练池的绝大多数长样本，建议把2048作为更保守的候选；"
        "对极长文本可另设更长路径，但动态路由策略本轮没有测量。"
        "本轮只改变推理截断上限，尚未比较不同训练max_length，不能据此证明训练长度最优。",
        "",
        "完整dev框架为88,971对。采样时有3对超出模型40,960上下文上限被剔除，"
        f"纳入指标的面板最长{max(sample['original_length'] for sample in manifest['samples']):,} tokens；"
        "原抽样权重保留，纳入质量为88,966.65625，指标在此质量内归一化。"
        "100%预测覆盖只针对1667对支持面板，不声称涵盖超出模型上下文的原文。"
        "所有分数均为复用dev上的pair级加权诊断，不是原生任务宏平均或完整Language Skill，也没有独立测试/显著性结论。",
        "",
        "## 当前训练池的长度分布",
        "",
        "| 统计量 | tokens |",
        "|---|---:|",
    ]
    lines += [f"| {key} | {value:g} |" for key, value in combined["length_quantiles"].items()]
    lines += ["", "| 上限 | 无需截断的训练对占比 |", "|---|---:|"]
    lines += [f"| {cap} | {share:.2%} |" for cap, share in combined["coverage_at_cap"].items()]
    lines += [
        "",
        "这里统计的是当前小训练池，不是上节dev面板，二者的自然长度分布不同。"
        "旧大D0缓存中的截断现象与当前两池零截断属于不同数据版本。",
        "",
        "## 正负比例：静态池、采样计划与原生类别",
        "",
        "正=pair标签0（匹配/entailment），负=pair标签1（不匹配/contradiction），不表示交换premise/hypothesis。",
        "",
        "| 静态池 | pair数 | 正pair数 | 负pair数 | 正pair占比 | 正token占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pool in inventory["pools"]:
        lines.append(
            f"| {pool['source']} | {pool['rows']:,} | {pool['positive_rows']:,} | {pool['negative_rows']:,} | "
            f"{pool['positive_rows'] / pool['rows']:.2%} | {pool['positive_tokens'] / pool['total_tokens']:.2%} |"
        )
    lines.append(
        f"| 合计 | {combined['rows']:,} | {combined['positive_rows']:,} | {combined['negative_rows']:,} | "
        f"{combined['positive_row_fraction']:.2%} | {combined['positive_token_fraction']:.2%} |"
    )
    lines += [
        "",
        "目标池中ANLI每题3个候选，正负为1:2；HellaSwag每题4个候选，为1:3；"
        "其他二选一目标任务为1:1。目标池总体38.88%的正pair占比主要反映这种结构，不能直接视为错误类别偏斜。"
        "目标池正pair平均359.47 tokens、负pair平均271.26，因此行数占比与token占比也不同。",
        "",
        "原生英语讽刺问题为664讽刺/2062非讽刺，即24.36%/75.64%；"
        "它展开成两候选后的5452个pair仍是2726正/2726负。"
        "平衡原生讽刺类别和改变pair正负比例是两种不同干预。",
        "",
        "### 归档采样计划中的曝光",
        "",
        "| 参考计划 | 样本出现次数 | 非padding tokens | 正类出现次数占比 | 正token占比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in exposure:
        if row["state"] == "complete":
            lines.append(
                f"| {row['case']} | {row['planned_presentations']:,} | {row['planned_tokens']:,} | "
                f"{row['positive_row_fraction']:.2%} | {row['positive_token_fraction']:.2%} |"
            )
    lines += [
        "",
        "曝光按完整schedule.npz的pool/index复原，包含重复采样，不能当作新增唯一样本数量。"
        "本统计没有逐步回放worker实际执行日志，若任务提前结束或末步有截断，真实曝光应再以完成记录核对。"
        "表中配方同时有其他采样差异，因此不能把它们的得分差解释为正负比例的因果效果。",
        "",
        "### 比例研究尚未完成的部分",
        "",
        "目前完成的是分布和归档曝光统计，没有正负比例训练消融的已确认结果。"
        "后续可在训练pair层固定每task token预算、模型起点、长度上限、seed和总预算，"
        "比较自然比例、约45%正类、50%正类，保留自然分布dev/test。"
        "若采用pair级重采样，会有意打破训练时完整候选单位，应单独标明；"
        "若只调CE类别权重，则应标为loss-weight实验，不能称作数据比例变化。"
        "不在本报告中预设1:1最优，也不改动正在执行的batch/LR实验。",
        "",
        "## 可追溯文件",
        "",
        "- `run_length_diagnostic.py`、`sample-manifest.json`：面板构建与推理程序。",
        "- `length-results.json`、`length-*.json`：各长度、来源、自然长度桶及逐类precision/recall/F1。",
        "- `data-inventory.json`：训练池完整逐task正负行数与逐长度桶计数。",
        "- `summarize_exposure.py`、`exposure-ratios.json`：归档schedule曝光统计。",
        "- `summarize_results.py`：本报告生成程序，仅CPU读取已完成产物。",
    ]
    (OUT / "BENCHMARK-REPORT.md").write_text("\n".join(lines) + "\n")
    print(OUT / "BENCHMARK-REPORT.md")


if __name__ == "__main__":
    main()
