"""CPU length/label inventory and fixed-checkpoint, same-example length sweep."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PROJECT = Path.home() / "assumeengage"
R0 = PROJECT / "runs/language-baseline-20261027T061200Z"
SHORT = PROJECT / "research/lu-fastscreen-20261028T164315Z"
D2 = PROJECT / "research/lu-target-expansion-20261027/D2-final-targets"
CHECKPOINT = PROJECT / "research/lu-lora-capacity-20261028/runs/LR32/seed-3409/arms/LR32/main/final"
EDGES = [0, 128, 256, 512, 1024, 2048, 4096, 8192, float("inf")]
BUCKETS = ["<128", "128-255", "256-511", "512-1023", "1024-2047", "2048-4095", "4096-8191", ">=8192"]


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def status(state, **fields):
    write(OUT / "status.json", {"state": state, "time": datetime.now(timezone.utc).isoformat(), **fields})


def arrow_file(path):
    import pyarrow as pa
    return pa.ipc.open_file(pa.memory_map(str(path), "r")).read_all()


def array(table, name):
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False)


def inventory():
    import numpy as np
    import pyarrow as pa
    tables = [arrow_file(SHORT / name) for name in ("pool-0.arrow", "pool-1.arrow")]
    reports = []
    for name, table in zip(("selected-D0", "selected-D2"), tables):
        lengths = array(table, "length").astype(np.int64)
        original = array(table, "original_length").astype(np.int64)
        labels = array(table, "labels")
        tasks = table.column("task").to_pylist()
        per_task = {}
        for task in sorted(set(tasks)):
            mask = np.array([value == task for value in tasks])
            per_task[task] = {"rows": int(mask.sum()), "positive_rows": int(((labels == 0) & mask).sum()),
                              "negative_rows": int(((labels == 1) & mask).sum())}
        report = {"source": name, "rows": len(table),
                  "length_quantiles": dict(zip(("p50", "p90", "p95", "p99", "max"),
                                                np.percentile(lengths, [50, 90, 95, 99, 100]).tolist())),
                  "original_max": int(original.max()), "truncated_rows": int(array(table, "truncated").sum()),
                  "total_tokens": int(lengths.sum()),
                  "positive_rows": int((labels == 0).sum()), "negative_rows": int((labels == 1).sum()),
                  "positive_tokens": int(lengths[labels == 0].sum()), "negative_tokens": int(lengths[labels == 1].sum()),
                  "coverage_at_cap": {str(cap): float((original <= cap).mean()) for cap in (128, 256, 512, 1024, 2048, 4096)},
                  "buckets": [{"bucket": label, "rows": int(((original >= lo) & (original < hi)).sum()),
                               "positive_rows": int(((original >= lo) & (original < hi) & (labels == 0)).sum()),
                               "negative_rows": int(((original >= lo) & (original < hi) & (labels == 1)).sum())}
                              for label, lo, hi in zip(BUCKETS, EDGES, EDGES[1:])],
                  "tasks": per_task}
        reports.append(report)
    combined = pa.concat_tables([table.select(["length", "original_length", "truncated", "labels"]) for table in tables])
    length = array(combined, "length").astype(np.int64)
    original = array(combined, "original_length").astype(np.int64)
    labels = array(combined, "labels")
    combined_report = {"rows": len(combined), "positive_rows": int((labels == 0).sum()),
                       "negative_rows": int((labels == 1).sum()), "positive_row_fraction": float((labels == 0).mean()),
                       "positive_token_fraction": float(length[labels == 0].sum() / length.sum()),
                       "length_quantiles": dict(zip(("p50", "p90", "p95", "p99", "max"),
                                                     np.percentile(length, [50, 90, 95, 99, 100]).tolist())),
                       "coverage_at_cap": {str(cap): float((original <= cap).mean()) for cap in (128, 256, 512, 1024, 2048, 4096)}}
    target = tables[1]
    native = Counter()
    for row in target.select(["task", "candidate_id", "labels"]).to_pylist():
        if row["task"] == "isarcasm/intended-en" and row["labels"] == 0:
            # Native preparation fixes candidate_id0=no, candidate_id1=yes.
            native["sarcastic" if row["candidate_id"] == 1 else "not_sarcastic"] += 1
    result = {"labels": {"0": "entailment/compatible", "1": "contradiction/incompatible"},
              "combined": combined_report, "pools": reports, "native_english_sarcasm_original_questions": dict(native)}
    write(OUT / "data-inventory.json", result)
    lines = ["# 当前训练池：长度与正负比例", "", "正负指pair兼容/不兼容标签，不是原生任务情感类别，也不是文本方向翻转。", "",
             f"- 合计 {combined_report['rows']:,} 对；正样本 {combined_report['positive_rows']:,}，负样本 {combined_report['negative_rows']:,}。",
             f"- 正样本行数占比 {combined_report['positive_row_fraction']:.2%}；有效token占比 {combined_report['positive_token_fraction']:.2%}。",
             f"- 长度分位数：{combined_report['length_quantiles']}。", "",
             "| 最大长度 | 原始长度无需截断覆盖率 |", "|---|---:|"]
    lines += [f"| {cap} | {share:.2%} |" for cap, share in combined_report["coverage_at_cap"].items()]
    lines += ["", "## 原生类别与展开后标签不同", "", f"- 原生英语讽刺问题：{dict(native)}。",
              "- 完整k选1候选展开通常带来1:(k-1)的兼容/不兼容结构；全局改成1:1不等于纠正原生类别偏斜。",
              "- 行数比例、token曝光比例、任务混合比例、类别loss权重需要分开报告。", "",
              "## 分池统计", ""]
    for report in reports:
        lines += [f"### {report['source']}", f"- {report['rows']:,} 对；正/负={report['positive_rows']:,}/{report['negative_rows']:,}。",
                  f"- 长度分位数：{report['length_quantiles']}；已截断样本 {report['truncated_rows']}。"]
    (OUT / "data-inventory.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"event": "length_label_inventory_ready", "combined": combined_report, "native": dict(native)}, ensure_ascii=False), flush=True)


def prepare_examples(tokenizer):
    import numpy as np
    import pyarrow.parquet as pq
    from datasets import load_from_disk
    sys.path.insert(0, str(PROJECT))
    from language_baseline.common import PairEncoder
    encoder = PairEncoder(tokenizer)
    sources = [("D0-validation", R0 / "prepared-data/validation"), ("D2-dev", D2 / "dev")]
    rng = np.random.default_rng(3407)
    examples = []
    unsupported = 0
    population = 0
    for source, path in sources:
        ds = load_from_disk(str(path))
        population += len(ds)
        metadata = ds.select_columns(["length", "original_length", "labels"]).with_format("arrow")[:]
        lengths = array(metadata, "original_length")
        labels = array(metadata, "labels")
        raw = pq.read_table(PROJECT / "datasets/tasksource/zero-shot-label-nli/data/validation-00000-of-00001.parquet") if source.startswith("D0") else None
        for bucket, lo, hi in zip(BUCKETS, EDGES, EDGES[1:]):
            for label in (0, 1):
                candidates = np.flatnonzero((lengths >= lo) & (lengths < hi) & (labels == label))
                if not len(candidates):
                    continue
                chosen = rng.choice(candidates, size=min(64, len(candidates)), replace=False)
                weight = len(candidates) / len(chosen)
                for index in chosen:
                    row = ds[int(index)]
                    if raw is not None:
                        original = raw.slice(int(row["raw_index"]), 1).to_pylist()[0]
                        p = tokenizer.encode(original["premise"], add_special_tokens=False)
                        h = tokenizer.encode(original["hypothesis"], add_special_tokens=False)
                        if {0: 0, 2: 1}[original["labels"]] != label:
                            raise RuntimeError("Raw-label alignment mismatch")
                    else:
                        ids = list(row["input_ids"])
                        positions = [i for i in range(len(encoder.prefix), len(ids) - len(encoder.middle))
                                     if ids[i:i + len(encoder.middle)] == encoder.middle]
                        if not positions or row["truncated"]:
                            raise RuntimeError("Target reference is not an intact encoded pair")
                        middle = positions[-1]
                        p = ids[len(encoder.prefix):middle]
                        h = ids[middle + len(encoder.middle):-1]
                    full, original_length, _ = encoder.from_ids(p, h)
                    if original_length > 40960:
                        unsupported += 1
                        continue
                    examples.append({"source": source, "task": row["task"], "row": int(index),
                                     "bucket": bucket, "label": label, "weight": weight,
                                     "premise_ids": p, "hypothesis_ids": h, "original_length": original_length})
    manifest = {"sampling": "Up to64 examples per(source, original-length bucket, label), seed3407",
                "population_pairs": population, "selected_supported_pairs": len(examples),
                "sampled_over_context_limit": unsupported,
                "reweighting": "Stratum population/sample weights restore D0validation+D2dev combined-pair distribution; not training-token-mixture weights",
                "scope": "Diagnostic dev panel, not official Language Skill; repeated validation panel, not an independent final test"}
    write(OUT / "sample-manifest.json", {**manifest, "samples": [{k: v for k, v in item.items() if k not in ("premise_ids", "hypothesis_ids")} for item in examples]})
    return examples, encoder, manifest


def metrics(examples, predictions, weighted=True):
    cm = [[0.0, 0.0], [0.0, 0.0]]
    available = 0
    mass = 0.0
    for item, prediction in zip(examples, predictions):
        weight = item["weight"] if weighted else 1.0
        mass += weight
        if prediction is None:
            continue
        available += weight
        cm[item["label"]][prediction] += weight
    total = sum(map(sum, cm))
    classes = {}
    for label in (0, 1):
        tp = cm[label][label]
        fp = cm[1 - label][label]
        fn = cm[label][1 - label]
        classes[str(label)] = {"precision": tp / max(tp + fp, 1e-15), "recall": tp / max(tp + fn, 1e-15),
                               "f1": 2 * tp / max(2 * tp + fp + fn, 1e-15)}
    return {"accuracy": (cm[0][0] + cm[1][1]) / max(total, 1e-15),
            "macro_f1": sum(value["f1"] for value in classes.values()) / 2,
            "prediction_coverage": available / max(mass, 1e-15), "classes": classes, "weighted_confusion": cm}


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    status("cpu_inventory")
    inventory()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    examples, encoder, manifest = prepare_examples(tokenizer)
    status("waiting_for_gpu0", panel_pairs=len(examples))
    while True:
        if time.monotonic() - started > 110 * 60:
            status("timeout_waiting_for_gpu"); return
        response = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=10)
        if not any(line.strip().isdigit() for line in response.splitlines()):
            break
        time.sleep(30)
    import torch
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    source = json.loads((PROJECT / "research/lu-joint-confirmation-20261028/active-code.json").read_text())["path"]
    sys.path.insert(0, source)
    os.environ["LU_RESEARCH_ROOT"] = str(PROJECT / "research/lu-joint-confirmation-20261028/runs/JOINT/seed-3407")
    from model_utils import load_model
    model, tokenizer = load_model(CHECKPOINT, mode="causal", device=0, trainable=False)
    model.eval()
    short = min(examples, key=lambda value: value["original_length"])
    warm, _, _ = encoder.from_ids(short["premise_ids"], short["hypothesis_ids"])
    warm = {key: value.cuda() for key, value in tokenizer.pad([warm], return_tensors="pt").items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            model(**warm)
    torch.cuda.synchronize(); del warm
    results = []
    for cap in (None, 256, 512, 1024, 2048, 4096):
        if time.monotonic() - started > 110 * 60:
            status("partial_timeout", completed_caps=[r["cap"] for r in results]); return
        name = "full" if cap is None else str(cap)
        status("length_inference", cap=name, panel_pairs=len(examples))
        inputs, predictions = [], [None] * len(examples)
        truncated = 0
        hypothesis_overflow = 0
        for i, item in enumerate(examples):
            try:
                encoded, _, cut = encoder.from_ids(item["premise_ids"], item["hypothesis_ids"], cap)
            except ValueError:
                hypothesis_overflow += 1
                continue
            truncated += cut
            inputs.append((i, encoded))
        inputs.sort(key=lambda value: len(value[1]["input_ids"]))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        clock = time.perf_counter()
        completed = real_tokens = padded_tokens = oom_singletons = 0
        while completed < len(inputs):
            end = completed
            while end < len(inputs) and end - completed < 32:
                padded = (len(inputs[end][1]["input_ids"]) + 7) // 8 * 8
                if end > completed and padded * (end - completed + 1) > 16384:
                    break
                end += 1
            while True:
                batch = tokenizer.pad([value for _, value in inputs[completed:end]], padding=True, pad_to_multiple_of=8, return_tensors="pt")
                count = int(batch["attention_mask"].sum()); padded_count = batch["attention_mask"].numel()
                try:
                    batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = model(**batch).logits.float().argmax(-1).cpu().tolist()
                    for (index, _), value in zip(inputs[completed:end], pred):
                        predictions[index] = value
                    real_tokens += count; padded_tokens += padded_count
                    del batch, pred
                    break
                except torch.cuda.OutOfMemoryError:
                    del batch; torch.cuda.empty_cache()
                    if end - completed == 1:
                        oom_singletons += 1
                        break
                    end = completed + max(1, (end - completed) // 2)
            completed = end
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - clock
        by_bucket = {}
        for bucket in BUCKETS:
            indices = [i for i, item in enumerate(examples) if item["bucket"] == bucket]
            if indices:
                by_bucket[bucket] = {"sample_pairs": len(indices), **metrics([examples[i] for i in indices], [predictions[i] for i in indices])}
        result = {"cap": name, "checkpoint": str(CHECKPOINT), "weighted": metrics(examples, predictions),
                  "unweighted": metrics(examples, predictions, weighted=False), "by_original_length_bucket": by_bucket,
                  "source_metrics": {source: metrics([item for item in examples if item["source"] == source],
                                                     [prediction for item, prediction in zip(examples, predictions) if item["source"] == source])
                                     for source in ("D0-validation", "D2-dev")},
                  "pairs": len(examples), "truncated_pairs": int(truncated), "hypothesis_overflow": hypothesis_overflow,
                  "oom_singletons": oom_singletons, "inference_wall_seconds": elapsed,
                  "samples_per_second": sum(p is not None for p in predictions) / elapsed,
                  "nonpadding_tokens_per_second": real_tokens / elapsed,
                  "padding_fraction": 1 - real_tokens / max(padded_tokens, 1),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                  "timing_scope": "collation + H2D + synchronized inference; excludes model loading and tokenization"}
        results.append(result)
        write(OUT / f"length-{name}.json", result)
        write(OUT / "length-results.json", results)
        print(json.dumps({"event": "length_case_complete", "cap": name, "weighted_metrics": result["weighted"],
                          "seconds": elapsed, "peak_allocated_gib": result["peak_allocated_gib"]}), flush=True)
    reference = results[0]["weighted"]["macro_f1"]
    choices = [r for r in results[1:] if r["weighted"]["prediction_coverage"] == 1.0
               and r["weighted"]["macro_f1"] >= reference - 0.005]
    sweet = min((int(r["cap"]) for r in choices), default=None)
    lines = ["# 固定checkpoint输入长度诊断", "", f"Checkpoint：`{CHECKPOINT}`。不更新权重。",
             f"固定诊断样本：{len(examples)} 对；完整dev框架 {manifest['population_pairs']} 对，分层样本按抽样权重还原总体。",
             f"采样中超出模型40960上限的 {manifest['sampled_over_context_limit']} 对单列，不纳入完整参考准确率。",
             "D0从原始validation文本恢复；D2使用无截断dev缓存。不是完整Language评测、不是训练长度消融。", "",
             "| 输入上限 | Accuracy | Macro-F1 | 预测覆盖 | 样本/s | 峰值allocated GiB | 额外截断对数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(results, key=lambda value: int(value["cap"]) if value["cap"] != "full" else 999999):
        lines.append(f"| {r['cap']} | {r['weighted']['accuracy']:.4%} | {r['weighted']['macro_f1']:.4%} | {r['weighted']['prediction_coverage']:.2%} | {r['samples_per_second']:.1f} | {r['peak_allocated_gib']:.2f} | {r['truncated_pairs']} |")
    lines += ["", f"暂定总体折中候选长度：{sweet if sweet else '本轮没有满足条件的有限长度'}。规则是预测覆盖完整、加权Macro-F1相对不额外截断下降不超过0.5个百分点；阈值为工作定义，不是训练最优性证明。",
              "应同时查看各原始长度桶的损失，不能用总体平均掩盖长尾；具体逐桶与逐类precision/recall/F1在length-results.json。",
              "分层抽样会增加长尾样本占比，未加权值只作诊断，不能冒充原始总体分数。dev被重复使用，不宣称新的独立测试结论。",
              "不同长度都评估相同样本、同一权重、同一标签/阈值；自然长桶与短桶难度差异不能当作长度的因果影响。",
              "正负比例的训练干预另由实验会话接续，本文件不声称改变训练配比后已获得收益。"]
    (OUT / "length-results.md").write_text("\n".join(lines) + "\n")
    write(OUT / "sweet-spot.json", {"cap": sweet, "criterion": "weighted_macro_f1_loss<=0.005 with full prediction coverage", "reference_macro_f1": reference})
    del model; torch.cuda.empty_cache()
    status("complete", report=str(OUT / "length-results.md"), wall_seconds=time.monotonic() - started, gpu0_released=True)
    print(json.dumps({"event": "length_diagnostic_complete", "report": str(OUT / "length-results.md"), "sweet_cap": sweet, "gpu0_released": True}), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
