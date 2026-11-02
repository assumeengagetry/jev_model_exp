#!/usr/bin/env python3
"""Render the active-37 data portrait from analyze.py's CSV tables, on CPU.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --root /home/dev/assumeengage/reports/benchmark-data-mining-20261031

Only numpy, pandas and matplotlib are required. No model, tokenizer, raw dataset,
network access or GPU is used. Existing tables are read once; missing inputs
produce an error rather than starting analysis or waiting for another process.
Each figure is written as both PNG and SVG, and the generated paths are printed.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors, font_manager, ft2font, ticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


AREA_ORDER = ["knowledge", "language", "retrieval", "tools", "arts"]
AREA_ZH = {
    "knowledge": "知识与推理",
    "language": "语言理解",
    "retrieval": "检索与分类",
    "tools": "工具与自动化",
    "arts": "艺术与人类偏好",
}
AREA_EN = {
    "knowledge": "Knowledge & reasoning",
    "language": "Language understanding",
    "retrieval": "Retrieval & classification",
    "tools": "Tools & automation",
    "arts": "Arts & human preferences",
}
AREA_COLORS = dict(zip(AREA_ORDER, ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]))
CAPS = [512, 1024, 2048, 4096, 8192, 16384]
BUCKETS = [
    "<128", "128–255", "256–511", "512–1023", "1024–2047",
    "2048–4095", "4096–8191", "8192–16383", "16384–32767", "≥32768",
]
LENGTH_COLORS = [colors.to_hex(plt.get_cmap("cividis")(v)) for v in np.linspace(0.06, 0.96, 10)]


def boolean(values: pd.Series) -> pd.Series:
    """Accept pandas bool columns as well as CSV string/0/1 representations."""
    return values.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def choose_font() -> tuple[bool, str]:
    """Choose an installed CJK face, including a glyph check; otherwise all-English."""
    preferred = [
        "noto sans cjk sc", "noto sans sc", "source han sans sc", "source han sans cn",
        "wenquanyi micro hei", "wenquanyi zen hei", "noto sans cjk", "source han sans",
        "noto serif cjk sc", "source han serif sc", "noto serif cjk",
    ]
    required = set(map(ord, "知识请求候选领域权重覆盖率未标注语义画像归一化"))
    candidates = []
    for entry in font_manager.fontManager.ttflist:
        name = entry.name.lower()
        rank = next((i for i, pattern in enumerate(preferred) if pattern in name), None)
        if rank is not None:
            candidates.append((rank, entry.name, entry.fname))
    for _, name, filename in sorted(set(candidates)):
        try:
            if required.issubset(ft2font.FT2Font(filename).get_charmap()):
                return True, name
        except (OSError, RuntimeError, ValueError):
            continue
    return False, "DejaVu Sans"


class Report:
    def __init__(self, root: Path, dpi: int = 180):
        self.root = root
        self.tables = root / "tables"
        self.output = root / "figures"
        self.dpi = dpi
        for font_file in (root / "assets").glob("*.otf"):
            font_manager.fontManager.addfont(str(font_file))
        self.zh, self.font = choose_font()
        self.paths: list[Path] = []
        plt.rcParams.update({
            "font.family": [self.font, "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10.5,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#6B7280",
            "axes.labelcolor": "#27303D",
            "text.color": "#27303D",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "path",
            "axes.formatter.useoffset": False,
        })
        source = self.tables / "main-37.csv"
        if not source.exists():
            source = self.tables / "benchmarks.csv"
        self.main = self.read(source)
        if "active" not in self.main:
            raise ValueError(f"{source}: missing active column; refusing to mix archived data")
        self.main = self.main.loc[boolean(self.main.active)].copy()
        self.main["benchmark_id"] = pd.to_numeric(self.main.benchmark_id).astype(int)
        if len(self.main) != 37 or self.main.benchmark_id.nunique() != 37:
            raise ValueError(f"Expected exactly 37 distinct active benchmarks, found {len(self.main)} rows")
        unexpected = set(self.main.area) - set(AREA_ORDER)
        if unexpected:
            raise ValueError(f"Unrecognized active domains: {sorted(unexpected)}")
        self.main["_area_order"] = self.main.area.map(dict(zip(AREA_ORDER, range(5))))
        self.main = self.main.sort_values(["_area_order", "benchmark_id"]).set_index("benchmark_id")
        self.ids = self.main.index.tolist()
        self.labels = [self.benchmark_label(bid, row["name"]) for bid, row in self.main.iterrows()]
        self.height = max(22.0, 4 + 37 * max(0.49, 0.19 * max(x.count("\n") + 1 for x in self.labels)))
        self.y = np.arange(len(self.main))
        self.coverage = self.active_table("coverage.csv")
        self.buckets = self.active_table("length-buckets.csv")
        self.areas = self.read(self.tables / "areas.csv").set_index("area").reindex(AREA_ORDER)
        if self.areas["requests"].isna().any():
            raise ValueError("areas.csv must contain the five active domains")
        self.semantic = self.semantic_counts()

    @staticmethod
    def read(path: Path, **kwargs) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(f"Required analysis table not found: {path}. Run analyze.py first; no waiting is performed.")
        return pd.read_csv(path, **kwargs)

    def active_table(self, filename: str) -> pd.DataFrame:
        frame = self.read(self.tables / filename)
        frame["benchmark_id"] = pd.to_numeric(frame.benchmark_id).astype(int)
        frame = frame.loc[frame.benchmark_id.isin(self.ids)].copy()
        if "active" in frame and not boolean(frame.active).all():
            raise ValueError(f"{filename}: inactive records share an active benchmark ID")
        return frame

    def t(self, chinese: str, english: str) -> str:
        return chinese if self.zh else english

    def area_name(self, area: str) -> str:
        return (AREA_ZH if self.zh else AREA_EN)[area]

    def number(self, value: float, percent: bool = False) -> str:
        if not np.isfinite(value):
            return self.t("缺失", "N/A")
        value = float(value)
        if percent:
            return f"{value:.2f}".rstrip("0").rstrip(".") + "%"
        suffix = ""
        if self.zh:
            if abs(value) >= 1e8:
                value, suffix = value / 1e8, "亿"
            elif abs(value) >= 1e4:
                value, suffix = value / 1e4, "万"
        elif abs(value) >= 1e9:
            value, suffix = value / 1e9, "B"
        elif abs(value) >= 1e6:
            value, suffix = value / 1e6, "M"
        elif abs(value) >= 1e3:
            value, suffix = value / 1e3, "k"
        return f"{value:.2f}".rstrip("0").rstrip(".") + suffix

    def formatter(self, percent: bool = False) -> ticker.FuncFormatter:
        return ticker.FuncFormatter(lambda value, _: self.number(value, percent=percent))

    def benchmark_label(self, bid: int, name: str) -> str:
        name = str(name)
        if not self.zh and not name.isascii():
            name = f"Benchmark {bid}"
        return textwrap.fill(f"{bid:02d}  {name}", width=36, break_long_words=True, break_on_hyphens=False)

    def grouped_axis(self, ax, labels: bool = True):
        ax.set_yticks(self.y)
        ax.set_yticklabels(self.labels if labels else [])
        ax.set_ylim(len(self.main) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0, pad=9)
        if labels:
            for tick, area in zip(ax.get_yticklabels(), self.main.area):
                tick.set_color(AREA_COLORS[area])
        areas = self.main.area.to_numpy()
        for i in range(1, len(areas)):
            if areas[i] != areas[i - 1]:
                ax.axhline(i - 0.5, color="#64748B", linewidth=1.2, alpha=0.7, zorder=5)
        ax.spines["left"].set_visible(False)

    def finish(self, fig, stem: str, title: str, subtitle: str, note: str, *, domains: bool = True,
               left: float = 0.29, right: float = 0.96, bottom: float = 0.075, top: float | None = None):
        fig.suptitle(title, x=0.5, y=0.992, fontsize=21, fontweight="bold")
        fig.text(0.5, 0.963, subtitle, ha="center", va="top", fontsize=11.5)
        if domains:
            fig.legend([Patch(color=AREA_COLORS[a]) for a in AREA_ORDER],
                       [self.area_name(a) for a in AREA_ORDER],
                       ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.945), fontsize=10.5)
        fig.text(0.03, 0.014, note, va="bottom", fontsize=10.5, linespacing=1.5)
        fig.subplots_adjust(left=left, right=right, bottom=bottom,
                            top=top if top is not None else (0.895 if domains else 0.925))
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            for extension in ["png", "svg"]:
                path = self.output / f"{stem}.{extension}"
                fig.savefig(path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.22)
                self.paths.append(path)
        finally:
            plt.close(fig)

    def heatmap(self, ax, values: np.ndarray, columns: list[str], *, cmap="YlGnBu", norm=None,
                percent=False, labels=True, fontsize=10.5):
        values = np.asarray(values, dtype=float)
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#E8EBEF")
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cm, norm=norm,
                          interpolation="nearest")
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, fontsize=11.5)
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=12)
        self.grouped_axis(ax, labels=labels)
        ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(self.main), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if np.isfinite(value):
                    rgba = cm(image.norm(value))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    ink = "white" if luminance < 0.47 else "#17212F"
                    text = self.number(value, percent=percent)
                else:
                    text, ink = self.t("缺失", "N/A"), "#6B7280"
                ax.text(j, i, text, color=ink, ha="center", va="center", fontsize=fontsize)
        return image

    def colorbar(self, fig, ax, image, label: str, *, percent=False, ticks=None):
        bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025, ticks=ticks)
        bar.set_label(label, labelpad=10)
        bar.ax.yaxis.set_major_formatter(self.formatter(percent))
        bar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        return bar

    def semantic_counts(self) -> pd.DataFrame:
        """Aggregate in chunks: yes/no may overlap for multi-accepted answers.

        Unknown is kept distinct from known non-yes/no labels. The five output
        categories are mutually exclusive, unlike the two raw semantic flags.
        """
        path = self.tables / "fields.csv.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Required semantic-label table not found: {path}")
        columns = ["benchmark_id", "known", "semantic_yes", "semantic_no"]
        result = pd.DataFrame(0, index=self.ids, columns=["yes", "no", "both", "other", "unknown"], dtype=np.int64)
        for chunk in pd.read_csv(path, usecols=columns, chunksize=200_000):
            chunk = chunk.loc[chunk.benchmark_id.isin(self.ids)].copy()
            if chunk.empty:
                continue
            known, yes, no = (boolean(chunk[name]) for name in columns[1:])
            flags = pd.DataFrame({
                "benchmark_id": chunk.benchmark_id,
                "yes": (known & yes & ~no).astype(np.int64),
                "no": (known & no & ~yes).astype(np.int64),
                "both": (known & yes & no).astype(np.int64),
                "other": (known & ~yes & ~no).astype(np.int64),
                "unknown": (~known).astype(np.int64),
            }).groupby("benchmark_id").sum()
            result = result.add(flags.reindex(self.ids, fill_value=0), fill_value=0)
        if not np.array_equal(result.sum(axis=1).to_numpy(), self.main.fields.to_numpy()):
            raise ValueError("fields.csv.gz counts disagree with main-37.csv; use tables from the same analysis run")
        return result

    def length_quantiles(self):
        columns = ["joint_p50", "joint_p95", "request_pair_max_p50", "request_pair_max_p95"]
        data = self.main[columns].to_numpy(dtype=float)
        positive = data[np.isfinite(data) & (data > 0)]
        if not len(positive):
            raise ValueError("Length quantiles contain no positive token counts")
        low, high = positive.min(), positive.max()
        norm = colors.LogNorm(vmin=max(1, low), vmax=max(high, max(1, low) * 1.01))
        fig, ax = plt.subplots(figsize=(17, self.height))
        image = self.heatmap(ax, data, [
            "Joint\np50", "Joint\np95", self.t("请求 Pair 最大值\np50", "Request Pair max\np50"),
            self.t("请求 Pair 最大值\np95", "Request Pair max\np95"),
        ], cmap="YlGnBu", norm=norm)
        self.colorbar(fig, ax, image, self.t("Tokens（对数色阶）", "Tokens (log color scale)"), ticks=[
            x for x in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] if low <= x <= high
        ] or [low, high])
        self.finish(fig, "01-length-quantiles-heatmap",
                    self.t("37 项长度画像：典型请求与长尾请求", "Active 37: typical and long-tail request lengths"),
                    self.t("格内为实际 token 数；颜色使用对数刻度，避免长请求压平其余数据", "Cells show actual tokens; log colors preserve detail across length scales"),
                    self.t("口径：Joint = state + 全部 questions 的联合 JSON；Request Pair max = 每请求全部实际候选 Pair 的最大长度。\np50 / p95 分别为请求级第 50 / 95 百分位；Pair 使用已有 prefix + suffix + 1 的实际分词口径，非截断长度。",
                           "Units: Joint = joint JSON of state and all questions; Request Pair max = longest actual candidate Pair in each request.\np50 / p95 are request-level percentiles. Pair lengths reuse the actual prefix + suffix + 1 tokenization, before truncation."))

    def coverage_heatmap(self, pair: bool):
        column = "request_pair_covered_pct" if pair else "joint_covered_pct"
        subset = self.coverage.loc[self.coverage.cap.isin(CAPS)]
        matrix = subset.pivot(index="benchmark_id", columns="cap", values=column).reindex(index=self.ids, columns=CAPS)
        if matrix.isna().any().any():
            raise ValueError(f"coverage.csv has missing active-37 / context-cap combinations for {column}")
        fig, ax = plt.subplots(figsize=(19, self.height))
        image = self.heatmap(ax, matrix.to_numpy(), [f"{cap:,}" for cap in CAPS],
                             norm=colors.Normalize(0, 100), percent=True, fontsize=10)
        self.colorbar(fig, ax, image, self.t("请求覆盖率", "Request coverage"), percent=True, ticks=[0, 25, 50, 75, 100])
        kind = self.t("实际候选 Pair 请求", "Actual candidate-Pair requests") if pair else self.t("Joint 联合请求", "Joint requests")
        definition = self.t("每个请求的所有候选 Pair 都不超上限，才算覆盖；不是单 Pair 覆盖率。", "A request is covered only if EVERY candidate Pair fits; this is NOT per-Pair coverage.") if pair else self.t("每个请求的联合 JSON token 数不超上限，才算覆盖；不包含答案、scoring 或 metadata。", "A request is covered when its joint JSON fits; answers, scoring and metadata are excluded.")
        self.finish(fig, "02-pair-request-coverage" if pair else "03-joint-request-coverage",
                    self.t(f"37 项上下文覆盖率：{kind}", f"Active 37 context coverage: {kind}"),
                    self.t("列为上下文 token 上限；每格为对应 benchmark 的请求覆盖百分比", "Columns are token caps; cells are within-benchmark request coverage percentages"),
                    definition + "\n" + self.t("两种请求形式分别展示，不将逐候选覆盖率误作请求覆盖率；灰格表示缺失，不表示零。", "The two input formats are shown separately; missing values are not zero."))

    def match_composition(self):
        fig, ax = plt.subplots(figsize=(18, self.height))
        counts = self.main[["positive_pairs", "negative_pairs", "unknown_pairs"]].to_numpy(dtype=float)
        values = np.divide(counts * 100, counts.sum(axis=1, keepdims=True),
                           out=np.zeros_like(counts), where=counts.sum(axis=1, keepdims=True) > 0)
        left = np.zeros(len(self.main))
        for j, (label, color) in enumerate(zip(
            [self.t("匹配正例", "Matching positive"), self.t("匹配负例", "Matching negative"), self.t("未标注", "Unlabeled")],
            ["#009E73", "#D55E00", "#BAC1CB"],
        )):
            ax.barh(self.y, values[:, j], left=left, height=0.72, color=color, label=label, edgecolor="white", linewidth=0.4)
            for i, width in enumerate(values[:, j]):
                if width >= 9:
                    ax.text(left[i] + width / 2, i, self.number(width, percent=True), va="center", ha="center", fontsize=9.5,
                            color="white" if j < 2 else "#17212F")
            left += values[:, j]
        self.grouped_axis(ax)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(self.formatter(True))
        ax.set_xlabel(self.t("占全部候选 Pair 的比例", "Share of all candidate Pairs"))
        ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
        ax.text(1.02, 1.012, self.t("已标注内正例率", "Positive / labeled"), transform=ax.transAxes, ha="left", fontsize=11)
        for i, value in enumerate(self.main.match_positive_pct):
            text = self.number(value, percent=True) if self.main.iloc[i].known_pairs > 0 else self.t("无标注", "No labels")
            ax.text(101.8, i, text, va="center", fontsize=10)
        self.finish(fig, "04-matching-label-composition",
                    self.t("匹配标签：正例、负例与未标注必须分开", "Matching labels: positive, negative and unlabeled are distinct"),
                    self.t("堆叠条分母为全部 Pair；右侧正例率分母仅为已标注 Pair", "Stack denominator: all Pairs. Right-hand positive rate: labeled Pairs only."),
                    self.t("匹配正例表示该候选属于可接受答案，不等于语义 Yes；匹配负例不等于语义 No。\n未判定检索候选保留为“未标注”，不会自动计为负例；比例与标签统计只含 active37。",
                           "Matching positive means an accepted candidate, not semantic Yes; matching negative is not semantic No.\nUnjudged retrieval candidates remain unlabeled, never automatically negative. Active 37 only."), right=0.86)

    def semantic_composition(self):
        fig, ax = plt.subplots(figsize=(19, self.height))
        values = self.semantic.to_numpy(dtype=float) * 100 / self.main.fields.to_numpy()[:, None]
        labels = [self.t("仅 Yes", "Yes only"), self.t("仅 No", "No only"), self.t("Yes / No 均可", "Both accepted"),
                  self.t("已标注非 Yes/No", "Labeled, other keys"), self.t("未标注", "Unlabeled")]
        palette = ["#0072B2", "#E69F00", "#CC79A7", "#78828F", "#CFD4DB"]
        left = np.zeros(len(self.main))
        for j, (label, color) in enumerate(zip(labels, palette)):
            ax.barh(self.y, values[:, j], left=left, height=0.72, color=color, label=label, edgecolor="white", linewidth=0.4)
            for i, width in enumerate(values[:, j]):
                if width >= 10:
                    ax.text(left[i] + width / 2, i, self.number(width, percent=True), ha="center", va="center", fontsize=9.5,
                            color="#17212F" if j in [1, 4] else "white")
            left += values[:, j]
        self.grouped_axis(ax)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(self.formatter(True))
        ax.set_xlabel(self.t("占全部题目（field）的比例", "Share of all question fields"))
        ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005), fontsize=10)
        self.finish(fig, "05-semantic-yes-no-composition",
                    self.t("语义标签：Yes / No 与匹配标签不是同一件事", "Semantic Yes / No is different from candidate matching"),
                    self.t("以题目为单位，五类互斥；正确答案同时允许 Yes 与 No 时单独列出", "Question-level, mutually exclusive categories; both-accepted answers have their own category"),
                    self.t("口径继承分析表：语义 Yes 对应可接受键 yes / true，No 对应 no / false；不额外猜测其他答案键的语义。\n“已标注非 Yes/No”不是 No，也不是未标注；未知标签独立显示。",
                           "Semantics follow the analysis table: accepted keys yes / true versus no / false; other keys are not reinterpreted.\nLabeled non-Yes/No answers are neither semantic No nor unlabeled. Unknown labels remain separate."))

    def label_rate_comparison(self):
        data = self.main[["match_positive_pct", "semantic_yes_pct"]].copy()
        data.loc[self.main.known_pairs.eq(0), "match_positive_pct"] = np.nan
        fig, ax = plt.subplots(figsize=(15, self.height))
        image = self.heatmap(ax, data.to_numpy(), [self.t("匹配正例率\n已标注 Pair 内", "Matching positive\nwithin labeled Pairs"),
                                                  self.t("语义 Yes 比例\nYes/No 题内", "Semantic Yes\nwithin Yes/No fields")],
                             norm=colors.Normalize(0, 100), percent=True)
        self.colorbar(fig, ax, image, self.t("比例", "Percentage"), percent=True, ticks=[0, 25, 50, 75, 100])
        self.finish(fig, "06-matching-vs-semantic-rates",
                    self.t("两种比例对照：明确不同分母", "Two label rates, explicitly different denominators"),
                    self.t("相同百分比不代表相同任务或相同类别分布；灰格为不适用或缺失", "Equal percentages do not imply equal tasks or class distributions; gray means not applicable / missing"),
                    self.t("左：positive_pairs ÷ known_pairs；右：含 yes/true 可接受答案的题数 ÷ 至少含 yes/true/no/false 的题数。\n右侧分母不包含已标注的非 Yes/No 题和未标注题；允许 Yes 与 No 的题在分母中只计一次。",
                           "Left: positive_pairs / known_pairs. Right: fields accepting yes/true / fields accepting any yes/true/no/false.\nThe right denominator excludes other-key and unlabeled fields; both-accepted fields enter its denominator once."))

    def domain_radars(self):
        raw = self.areas[["requests", "joint_p95", "options_mean", "fields_per_request_mean", "joint_over_4096_pct"]].copy()
        sums = self.main.groupby("area")[["pair_sum", "joint_sum"]].sum().reindex(AREA_ORDER)
        raw["pair_joint_ratio"] = sums.pair_sum / sums.joint_sum
        maxima = raw.max(axis=0).replace(0, 1)
        normalized = raw.divide(maxima, axis=1)
        short = [self.t("请求量", "Requests"), self.t("Joint p95", "Joint p95"), self.t("候选数/题", "Options/field"),
                 self.t("题数/请求", "Fields/request"), self.t("Joint >4096", "Joint >4096"), self.t("Pair/Joint", "Pair/Joint")]
        angles = np.linspace(0, 2 * np.pi, len(short), endpoint=False)
        closed_angles = np.r_[angles, angles[0]]
        fig = plt.figure(figsize=(29, 12))
        grid = fig.add_gridspec(2, 5, height_ratios=[3.2, 1.8], hspace=0.38, wspace=0.72)
        for i, area in enumerate(AREA_ORDER):
            ax = fig.add_subplot(grid[0, i], projection="polar")
            values = normalized.loc[area].to_numpy(dtype=float)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(closed_angles, np.r_[values, values[0]], color=AREA_COLORS[area], linewidth=2.3)
            ax.fill(closed_angles, np.r_[values, values[0]], color=AREA_COLORS[area], alpha=0.17)
            ax.scatter(angles, values, color=AREA_COLORS[area], s=24, zorder=4)
            ax.set_xticks(angles)
            ax.set_xticklabels(short, fontsize=10)
            ax.tick_params(axis="x", pad=10)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=8.5, color="#6B7280")
            ax.grid(color="#D2D7DE", linewidth=0.7)
            ax.spines["polar"].set_color("#CBD2DC")
            ax.set_title(self.area_name(area), color=AREA_COLORS[area], fontsize=13, pad=28, fontweight="bold")
        ax = fig.add_subplot(grid[1, :])
        ax.axis("off")
        rows = []
        for area in AREA_ORDER:
            values = raw.loc[area].to_numpy(dtype=float)
            rows.append([self.number(v, percent=(j == 4)) + ("×" if j == 5 else "") for j, v in enumerate(values)])
        headers = [self.t("请求量（条）", "Requests"), self.t("Joint p95（tokens）", "Joint p95 (tokens)"),
                   self.t("平均候选数/题", "Mean options/field"), self.t("平均题数/请求", "Mean fields/request"),
                   self.t("Joint >4096 请求占比", "Joint >4096 request share"), self.t("Pair 总 tokens / Joint", "Pair total tokens / Joint")]
        table = ax.table(cellText=rows, rowLabels=[self.area_name(a) for a in AREA_ORDER], colLabels=headers,
                         cellLoc="center", rowLoc="left", bbox=[0.12, 0.06, 0.87, 0.87])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D8DDE4")
            if row == 0:
                cell.set_facecolor("#EDF1F5")
                cell.set_text_props(fontweight="bold")
            elif col == -1:
                cell.set_text_props(color=AREA_COLORS[AREA_ORDER[row - 1]], fontweight="bold")
        self.finish(fig, "07-five-domain-data-profile-radars",
                    self.t("五领域数据画像雷达图：描述数据，不是性能或风险评分", "Five-domain data profiles: NOT performance or risk scores"),
                    self.t("每轴 = 该领域原始值 ÷ 五领域该轴最大值；所有领域使用同一组轴尺度；表格保留原始单位", "Each axis = domain raw value / maximum across the five domains; identical scales; table retains raw units"),
                    self.t("六轴依次：请求条数、联合长度 p95、每题平均候选数、每请求平均题数、Joint 超过 4096 tokens 的请求比例、候选 Pair 总 tokens / Joint 总 tokens。\n归一化仅便于比较数据形状，不求和、不排名；较大不代表更好、更差、模型能力或风险；p95 来自领域级原始请求分布。",
                           "Axes: request count; Joint p95; mean options/field; mean fields/request; Joint >4096 request share; total candidate-Pair / Joint tokens.\nNormalization only compares data shapes: no composite score, ranking, model ability or risk inference. p95 uses the domain-level request distribution."),
                    domains=False, left=0.055, right=0.98, bottom=0.13, top=0.86)

    def weight_vs_burden(self):
        measures = ["weight_pct", "requests", "pairs", "joint_sum", "pair_sum"]
        data = self.main[measures].astype(float).copy()
        for column in measures[1:]:
            data[column] = data[column] * 100 / data[column].sum()
        fig, ax = plt.subplots(figsize=(19, self.height))
        image = self.heatmap(ax, data.to_numpy(), [self.t("官方分数\n权重", "Official score\nweight"),
                                                  self.t("请求量\n占比", "Request\nshare"), self.t("候选 Pair 数\n占比", "Candidate-Pair\ncount share"),
                                                  self.t("Joint tokens\n占比", "Joint token\nshare"), self.t("候选 Pair tokens\n占比", "Candidate-Pair\ntoken share")],
                             norm=colors.Normalize(0, max(1, data.to_numpy().max())), percent=True)
        self.colorbar(fig, ax, image, self.t("占 active37 总量的比例", "Share of active-37 total"), percent=True)
        self.finish(fig, "08-official-weight-vs-sample-token-burden",
                    self.t("官方分数权重与样本 / 候选 Pair token 负担", "Official score weights versus sample and candidate-Pair token burdens"),
                    self.t("每列均以 active37 对应总量为分母；权重直接来自官方领域配额与 gold 配比", "Each column uses its active-37 total; official weights follow domain allocation and gold proportions"),
                    self.t("“请求量”是请求条数，不是唯一 state 或独立语义样本；Pair 数是逐题逐候选展开后的数量。\nToken 总量为完整未截断输入的加总，不等价于 FLOPs、训练时长、显存或模型性能；不自动据此调整官方权重。",
                           "Requests are request rows, not unique states or independent semantic samples; Pairs expand each field over its candidates.\nToken sums use full, untruncated inputs; they are not FLOPs, runtime, memory or model performance, and do not redefine official weights."))

    def candidates_and_fields(self):
        fig, axes = plt.subplots(1, 3, figsize=(23, self.height), gridspec_kw={"wspace": 0.2})
        specifications = [
            ("options", self.t("每题候选数", "Candidates per field")),
            ("fields_per_request", self.t("每请求题数", "Fields per request")),
            ("pairs_per_request", self.t("每请求候选 Pair 数", "Candidate Pairs per request")),
        ]
        for j, (ax, (prefix, title)) in enumerate(zip(axes, specifications)):
            mean = self.main[f"{prefix}_mean"].to_numpy(dtype=float)
            p95 = self.main[f"{prefix}_p95"].to_numpy(dtype=float)
            for i in self.y:
                ax.plot([mean[i], p95[i]], [i, i], color="#B4BDC9", linewidth=2.2, zorder=1)
            ax.scatter(mean, self.y, s=43, c="#0072B2", marker="o", label=self.t("均值", "Mean"), zorder=3)
            ax.scatter(p95, self.y, s=42, c="#D55E00", marker="D", label="p95", zorder=3)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(self.formatter())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_title(title, pad=19)
            ax.set_xlabel(self.t("数量（对数轴）", "Count (log axis)"))
            ax.grid(axis="x", color="#E5E9EE", linewidth=0.8)
            self.grouped_axis(ax, labels=j == 0)
        axes[1].legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.05))
        self.finish(fig, "09-candidates-fields-and-pairs-per-request",
                    self.t("候选展开结构：每题候选、每请求题目与 Pair", "Candidate expansion: options, fields and Pairs per request"),
                    self.t("圆点为均值，菱形为 p95；三个面板均用对数横轴，各自独立范围", "Circles: mean; diamonds: p95. All panels have log x-axes with independent ranges."),
                    self.t("候选数的统计单位是题目；题数与 Pair 数的统计单位是请求。\n每请求 Pair 数 = 该请求全部题目候选数之和；不同统计量的均值 / p95 不能直接相乘；数量不代表难度或性能。",
                           "Options are summarized over fields; fields and Pair counts are summarized over requests.\nPairs/request = sum of options over that request's fields; means / p95 values across different units must not be multiplied to infer totals."), left=0.235)

    def length_distribution(self, unit: str):
        frame = self.buckets.loc[self.buckets.unit.eq(unit)].pivot(index="benchmark_id", columns="bucket", values="pct")
        frame = frame.reindex(index=self.ids, columns=BUCKETS)
        if frame.isna().any().any():
            raise ValueError(f"length-buckets.csv has missing bins for {unit}")
        fig, ax = plt.subplots(figsize=(20, self.height))
        left = np.zeros(len(self.main))
        for bucket, color in zip(BUCKETS, LENGTH_COLORS):
            values = frame[bucket].to_numpy()
            # ASCII labels make even the font fallback entirely CJK-free.
            label = bucket if self.zh else bucket.replace("–", "-").replace("≥", ">=")
            ax.barh(self.y, values, left=left, height=0.73, color=color, label=label,
                    edgecolor="white", linewidth=0.3)
            left += values
        self.grouped_axis(ax)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(self.formatter(True))
        ax.set_xlabel(self.t("候选 Pair 比例", "Share of candidate Pairs") if unit == "pair" else self.t("请求比例", "Share of requests"))
        ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005), title=self.t("Tokens 长度区间", "Token-length bins"), fontsize=10)
        stem, subject = {
            "joint": ("10-joint-length-distribution", self.t("Joint 联合请求长度", "Joint request length")),
            "request_pair_max": ("11-pair-request-length-distribution", self.t("每请求 Pair 最大长度", "Request Pair maximum")),
            "pair": ("16-candidate-pair-length-distribution", self.t("全部实际候选 Pair 长度", "All actual candidate-Pair lengths")),
        }[unit]
        sampling_unit = self.t("按实际候选 Pair 统计，不是请求覆盖率。", "Sampled over actual candidate Pairs, NOT request coverage.") if unit == "pair" else self.t("按请求统计。", "Sampled over requests.")
        self.finish(fig, stem,
                    self.t(f"长度分布堆叠：{subject}", f"Length distributions: {subject}"),
                    self.t("每行在该 benchmark 内归一为 100%；区间使用相同顺序与颜色，长尾向右", "Each benchmark sums to 100%; fixed bins and colors, with the long tail to the right"),
                    sampling_unit + " " + self.t("末档 ≥32768 汇总全部更长输入，不代表它们长度相同。\n区间为左闭右开（例如 512–1023 表示 512 ≤ tokens < 1024）；CSV 四舍五入可能造成合计与 100% 有微小差异。",
                           "The >=32768 tail pools all larger lengths, not equal-length inputs.\nIntervals are lower-inclusive and upper-exclusive (512-1023 means 512 <= tokens < 1024); rounded CSV percentages may not sum exactly to 100%."))

    def answer_bias(self):
        columns = ["first_answer_pct", "last_answer_pct", "normalized_gold_position_pct", "longest_candidate_reference_pct", "shortest_candidate_reference_pct"]
        labels = [self.t("首位候选\n可接受率", "First candidate\naccepted"), self.t("末位候选\n可接受率", "Last candidate\naccepted"),
                  self.t("正确答案\n平均相对位置", "Accepted answer\nmean relative position"),
                  self.t("最长候选\n正例参考比例", "Longest candidates\npositive reference"), self.t("最短候选\n正例参考比例", "Shortest candidates\npositive reference")]
        fig, ax = plt.subplots(figsize=(20, self.height))
        image = self.heatmap(ax, self.main[columns].to_numpy(), labels, norm=colors.Normalize(0, 100), percent=True)
        self.colorbar(fig, ax, image, self.t("描述统计比例", "Descriptive percentage"), percent=True, ticks=[0, 25, 50, 75, 100])
        self.finish(fig, "12-answer-position-and-candidate-length-bias",
                    self.t("答案位置与候选长短偏向：描述标签结构", "Answer position and candidate length: label-structure descriptions"),
                    self.t("仅在已标注题目内统计；这些数字不是模型准确率，也不推断偏向产生原因", "Labeled fields only; these values are not model accuracy and do not identify causes"),
                    self.t("首/末位列 = 对应位置候选属于可接受答案的题目比例，多答案题可同时满足；相对位置 = 正确位置索引 ÷ max(候选数−1, 1)，0% 首位、100% 末位。\n长短按实际候选 suffix tokens：每题计算并列最长/最短候选中的正例比例，再对已标注题取均值；不是选择策略的真实模型结果。",
                           "First/last: share of labeled fields accepting that position (both may hold). Relative position = accepted index / max(options-1, 1); 0% first, 100% last.\nLength uses actual candidate suffix tokens: positive fraction among tied extreme-length candidates, averaged over labeled fields; not an actual model result."))

    def pair_vs_joint_tokens(self):
        fig, axes = plt.subplots(1, 2, figsize=(23, self.height), gridspec_kw={"width_ratios": [1.65, 1], "wspace": 0.2})
        ax = axes[0]
        joint = self.main.joint_sum.to_numpy(dtype=float)
        pair = self.main.pair_sum.to_numpy(dtype=float)
        ax.barh(self.y - 0.17, joint, height=0.31, color="#0072B2", label="Joint")
        ax.barh(self.y + 0.17, pair, height=0.31, color="#E69F00", label=self.t("全部候选 Pair", "All candidate Pairs"))
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(self.formatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(self.t("完整输入总 tokens（对数轴）", "Total full-input tokens (log axis)"))
        ax.set_title(self.t("同一请求集合的两种输入展开", "Two input expansions of the same requests"), pad=16)
        ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.025))
        self.grouped_axis(ax)
        ax.grid(axis="x", color="#E5E9EE", zorder=0)
        ax = axes[1]
        ratio = np.divide(pair, joint, out=np.full_like(pair, np.nan), where=joint > 0)
        ax.axvline(1, color="#8893A3", linestyle="--", linewidth=1)
        ax.scatter(ratio, self.y, s=46, c=[AREA_COLORS[a] for a in self.main.area], zorder=3)
        for i, value in enumerate(ratio):
            if np.isfinite(value):
                ax.annotate(self.number(value) + "×", (value, i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10)
        ax.set_xscale("log")
        finite = ratio[np.isfinite(ratio) & (ratio > 0)]
        if len(finite):
            ax.set_xlim(min(0.85, finite.min() / 1.3), max(1.3, finite.max() * 2.2))
        ax.xaxis.set_major_formatter(self.formatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(self.t("Pair / Joint 总 tokens（倍，对数轴）", "Pair / Joint total tokens (multiple, log axis)"))
        ax.set_title(self.t("重复展开倍数，虚线为 1×", "Expansion ratio; dashed line = 1x"), pad=16)
        self.grouped_axis(ax, labels=False)
        ax.grid(axis="x", color="#E5E9EE")
        self.finish(fig, "13-total-candidate-pair-vs-joint-tokens",
                    self.t("候选 Pair 总 tokens 与 Joint 总 tokens", "Total candidate-Pair tokens versus Joint tokens"),
                    self.t("同一 benchmark 的实际输入 token 加总；右侧为“总量之比”，不是每请求比值的平均", "Actual input-token sums; right panel is the ratio of sums, not the mean of per-request ratios"),
                    self.t("候选 Pair 展开会重复 prefix；Joint 则一次包含 state 与全部 questions，但两者的序列化与分词边界不同。\n本图衡量输入 token 负担，不宣称两种方式性能等价，也不将 token 比例直接解释为耗时 / 显存 / FLOPs 比例。",
                           "Candidate Pairs repeat prefixes; Joint includes state and all questions once, with different serialization / tokenization boundaries.\nThis is an input-token burden comparison, not equivalent performance or a direct runtime / memory / FLOPs ratio."), left=0.235)

    def domain_coverage(self):
        fig, axes = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={"wspace": 0.14})
        matrices = []
        for pair in [True, False]:
            names = [f"pair_request_over_{cap}_pct" if pair else f"joint_over_{cap}_pct" for cap in CAPS]
            matrices.append(100 - self.areas[names].to_numpy(dtype=float))
        for j, (ax, values) in enumerate(zip(axes, matrices)):
            image = ax.imshow(values, aspect="auto", cmap="YlGnBu", norm=colors.Normalize(0, 100))
            ax.set_xticks(range(6))
            ax.set_xticklabels([f"{cap:,}" for cap in CAPS])
            ax.set_yticks(range(5))
            ax.set_yticklabels([self.area_name(a) for a in AREA_ORDER] if j == 0 else [])
            ax.set_title(self.t("实际候选 Pair 请求", "Actual candidate-Pair requests") if j == 0 else self.t("Joint 请求", "Joint requests"), pad=16)
            ax.set_xlabel(self.t("上下文 token 上限", "Context token cap"))
            for i in range(5):
                for k in range(6):
                    ax.text(k, i, self.number(values[i, k], percent=True), ha="center", va="center", fontsize=11,
                            color="white" if values[i, k] > 65 else "#17212F")
        bar = fig.colorbar(image, ax=axes.tolist(), fraction=0.022, pad=0.025)
        bar.set_label(self.t("请求覆盖率", "Request coverage"))
        bar.ax.yaxis.set_major_formatter(self.formatter(True))
        # The shared colorbar uses its own position; exclude it from manual subplot adjustment.
        bar.ax.set_position([0.945, 0.19, 0.014, 0.61])
        self.finish(fig, "14-five-domain-request-coverage",
                    self.t("五领域覆盖率：按实际请求量汇总", "Five-domain coverage: pooled over actual requests"),
                    self.t("不是 37 项百分比的简单平均；分别比较 Pair 请求与 Joint 请求", "Not an unweighted average of benchmark percentages; Pair and Joint requests remain separate"),
                    self.t("领域内每个实际请求权重相同；Pair 需该请求所有候选均不超限，Joint 需联合输入不超限。\n只含 active37；覆盖率只描述输入长度适配，不等于质量、正确率或可替代性。",
                           "Every actual request has equal weight within its domain. Pair coverage requires all candidates to fit; Joint requires its joint input to fit.\nActive 37 only. Coverage describes input-length fit, not quality, correctness or interchangeability."),
                    domains=False, left=0.16, right=0.91, bottom=0.17, top=0.85)

    def feature_correlations(self):
        path = self.tables / "feature-correlations.csv"
        if not path.is_file():
            print(self.t("未找到 feature-correlations.csv；跳过可选相关性图。", "feature-correlations.csv not found; skipping optional correlation figure."), file=sys.stderr)
            return
        frame = self.read(path)
        features = ["joint_p95", "request_pair_max_p95", "options_mean", "fields_per_request_mean", "match_positive_pct", "first_answer_pct", "pair_to_joint_token_ratio"]
        labels = ["Joint p95", self.t("请求 Pair max p95", "Request Pair max p95"), self.t("每题平均候选数", "Mean options/field"),
                  self.t("每请求平均题数", "Mean fields/request"), self.t("已标注匹配正例率", "Labeled matching-positive rate"),
                  self.t("首位答案比例", "First-answer share"), self.t("Pair/Joint tokens", "Pair/Joint tokens")]
        matrix = frame.pivot(index="x", columns="y", values="spearman").reindex(index=features, columns=features)
        sample_n = frame.pivot(index="x", columns="y", values="benchmarks").reindex(index=features, columns=features)
        if sample_n.max().max() > 37:
            raise ValueError("feature-correlations.csv contains more than 37 benchmarks; refusing possible archived-data mixing")
        fig, ax = plt.subplots(figsize=(18, 14))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#E8EBEF")
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(7))
        ax.set_xticklabels([textwrap.fill(label, 23) for label in labels], rotation=30, ha="right", fontsize=10.5)
        ax.set_yticks(range(7))
        ax.set_yticklabels(labels, fontsize=11.5)
        for i in range(7):
            for j in range(7):
                value, n = matrix.iloc[i, j], sample_n.iloc[i, j]
                text = self.number(value) if np.isfinite(value) else self.t("缺失", "N/A")
                if np.isfinite(n):
                    text += f"\nn={int(n)}"
                ax.text(j, i, text, ha="center", va="center", fontsize=11,
                        color="white" if np.isfinite(value) and abs(value) > 0.65 else "#17212F")
        bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1])
        bar.set_label(self.t("Spearman 秩相关", "Spearman rank correlation"))
        bar.ax.yaxis.set_major_formatter(self.formatter())
        self.finish(fig, "15-feature-correlations-not-causation",
                    self.t("数据特征相关性：相关不代表因果", "Data-feature correlations: association is not causation"),
                    self.t("比较单位为 active37 benchmark；格内为 Spearman 系数与有效 benchmark 数 n", "Unit of comparison: active-37 benchmarks; cells show Spearman coefficients and valid benchmark count n"),
                    self.t("各格按非缺失值成对计算；常量列可能导致相关系数缺失。未做因果识别、显著性检验或多重比较校正。\n不同 benchmark 在任务与数据结构上异质；不能据此声称长度 / 候选数导致任何模型性能变化。",
                           "Pairwise non-missing observations; constant features may have undefined correlations. No causal identification or significance / multiplicity claims.\nTasks and data structures differ across benchmarks; this figure cannot establish that length or candidate count causes model-performance changes."),
                    domains=False, left=0.26, right=0.94, bottom=0.24, top=0.88)

    def generate(self):
        print(self.t(f"字体：{self.font}；语言：中文；范围：active37。", f"Font: {self.font}; language: English fallback; scope: active 37."))
        if not self.zh:
            print("No supported Chinese font with the required glyphs was found; ALL figure titles, labels and notes use English.", file=sys.stderr)
        self.length_quantiles()
        self.coverage_heatmap(pair=True)
        self.coverage_heatmap(pair=False)
        self.match_composition()
        self.semantic_composition()
        self.label_rate_comparison()
        self.domain_radars()
        self.weight_vs_burden()
        self.candidates_and_fields()
        self.length_distribution("joint")
        self.length_distribution("request_pair_max")
        self.answer_bias()
        self.pair_vs_joint_tokens()
        self.domain_coverage()
        self.feature_correlations()
        self.length_distribution("pair")
        print(self.t(f"输出目录：{self.output}", f"Output directory: {self.output}"))
        print(self.t(f"已写入 {len(self.paths) // 2} 张图，每张包含 PNG + SVG：", f"Wrote {len(self.paths) // 2} figures, each in PNG + SVG:"))
        for path in self.paths:
            print(path)


def main():
    parser = argparse.ArgumentParser(description="Render active-37 benchmark data portraits (PNG + SVG), using only existing CSV tables.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Report root containing tables/; default: the parent of scripts/.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution, default 180; SVG remains vector.")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72 for readable output")
    try:
        Report(args.root.expanduser().resolve(), args.dpi).generate()
    except (FileNotFoundError, KeyError, ValueError, pd.errors.EmptyDataError) as exc:
        parser.exit(1, f"Cannot generate benchmark figures: {exc}\n")


if __name__ == "__main__":
    main()
