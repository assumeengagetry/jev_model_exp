(function (root) {
  "use strict";
  const COLORS = ["#2c7257", "#cf6847", "#587fad", "#9271a8", "#b18a35"];
  function point(index, count, value, radius = 213, cx = 380, cy = 310) {
    const angle = index * 2 * Math.PI / count - Math.PI / 2;
    const distance = Math.max(0, Math.min(100, value)) / 100 * radius;
    return [cx + distance * Math.cos(angle), cy + distance * Math.sin(angle)];
  }
  function matches(entry, filters) {
    const query = filters.search.toLowerCase().trim();
    return (!query || `${entry.name} ${entry.code} ${entry.seed ?? ""} ${entry.round}`.toLowerCase().includes(query)) &&
      (filters.stage === "all" || entry.stage === filters.stage) &&
      (filters.rank === "all" || entry.rank === Number(filters.rank)) &&
      (filters.budget === "all" || Math.abs(entry.tokens - Number(filters.budget)) <= Math.max(1000, Number(filters.budget)*.01));
  }
  function csvCell(value) {
    let text = String(value ?? "");
    if (/^[=+@]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  }
  if (typeof module !== "undefined" && module.exports) module.exports = { point, matches, csvCell };
  if (typeof document === "undefined") return;
  const data = root.EXPERIMENT_DATA;
  if (!data) throw new Error("实验数据未加载，请先运行build_data.py");
  const $ = (id) => document.getElementById(id);
  const all = [...data.runs, ...data.groups];
  const byId = new Map(all.map((entry) => [entry.id, entry]));
  const state = { mode: "group", selected: [], reference: null };
  const fmt = (value, digits = 2) => Number(value).toFixed(digits);
  const tokenLabel = (value) => value >= 1e6 ? `${fmt(value / 1e6, value < 2e6 ? 2 : 0)}M` : value.toLocaleString();
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function svgEl(tag, attrs, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function selected() { return state.selected.map((id) => byId.get(id)).filter(Boolean); }
  function title(entry) { return `${entry.name}${entry.kind === "group" ? ` · n=${entry.n}` : ` · s${entry.seed}`}`; }
  function modelLabel(entry) {
    if (entry.deployment === "ensemble") return "五模型集成 · 新增训练0";
    if (entry.deployment === "dense_merged") return "密集融合 · 新增训练0";
    return `r${entry.rank} · ${tokenLabel(entry.tokens)}`;
  }
  function filters() {
    return { search: $("search").value, stage: $("stage").value, rank: $("rank").value, budget: $("budget").value };
  }
  function renderList() {
    const entries = data[state.mode === "group" ? "groups" : "runs"].filter((entry) => matches(entry, filters())).sort((a, b) => b.score - a.score);
    $("visible-count").textContent = `${entries.length} 个结果`;
    $("experiments").replaceChildren();
    for (const entry of entries) {
      const checked = state.selected.includes(entry.id);
      const label = el("label", `experiment-card${checked ? " checked" : ""}`);
      const checkbox = el("input"); checkbox.type = "checkbox"; checkbox.checked = checked;
      checkbox.setAttribute("aria-label", `比较 ${title(entry)}`);
      checkbox.dataset.id = entry.id;
      checkbox.addEventListener("change", () => {
        if (checkbox.checked && state.selected.length >= 5) {
          checkbox.checked = false; $("selection-message").textContent = "每次最多比较5个结果；先取消一个再添加。"; return;
        }
        state.selected = checkbox.checked ? [...state.selected, entry.id] : state.selected.filter((id) => id !== entry.id);
        $("selection-message").textContent = ""; render();
      });
      const text = el("div", "card-text");
      text.append(el("div", "card-name", entry.name), el("div", "card-meta", `${entry.code} · ${modelLabel(entry)} · ${entry.kind === "group" ? `${entry.n} seed` : `seed ${entry.seed}`}`));
      label.append(checkbox, text, el("div", "card-score", fmt(entry.score)));
      $("experiments").append(label);
    }
    if (!entries.length) $("experiments").append(el("p", "empty-state", "没有匹配的实验，试试调整筛选条件。"));
  }
  function renderRadar() {
    const entries = selected(); $("radar").replaceChildren(); $("legend").replaceChildren();
    if (!entries.length) { $("radar").append(el("div", "empty-state", "从左侧选择实验，或使用快捷对比。")); return; }
    const svg = svgEl("svg", { viewBox: "0 0 760 610", role: "img", "aria-label": "九项Language Skill雷达对比", "font-family": "sans-serif", xmlns: "http://www.w3.org/2000/svg" });
    svg.append(svgEl("title", {}, "实验九项Skill对比"), svgEl("desc", {}, entries.map((entry) => `${title(entry)}，Language ${fmt(entry.score)}`).join("；")));
    const points = (values, radius = 213) => values.map((value, i) => point(i, 9, value, radius).join(",")).join(" ");
    for (let value = 20; value <= 100; value += 20) {
      svg.append(svgEl("polygon", { points: points(Array(9).fill(value)), fill: value === 100 ? "#f8faf6" : "none", stroke: "#e0e6dc", "stroke-width": "1" }));
      svg.append(svgEl("text", { x: 388, y: 310 - value / 100 * 213 + 4, fill: "#a0a99d", "font-size": "9" }, String(value)));
    }
    // Put the outer background behind the inner rings.
    const outer = svg.querySelector('polygon[fill="#f8faf6"]'); if (outer) svg.insertBefore(outer, svg.firstChild);
    data.axes.forEach((axis, i) => {
      const [x, y] = point(i, 9, 100);
      svg.append(svgEl("line", { x1: 380, y1: 310, x2: x, y2: y, stroke: "#dfe6da", "stroke-width": ".8" }));
      const [tx, ty] = point(i, 9, 100, 255);
      const anchor = tx < 350 ? "end" : tx > 410 ? "start" : "middle";
      svg.append(svgEl("text", { x: tx, y: ty + 4, fill: "#55634e", "font-size": "12", "text-anchor": anchor }, axis.name));
    });
    entries.forEach((entry, index) => {
      const color = COLORS[index];
      svg.append(svgEl("polygon", { points: points(entry.skills), fill: color, "fill-opacity": ".055", stroke: color, "stroke-width": "2", "stroke-linejoin": "round" }));
      entry.skills.forEach((value, axis) => {
        const [cx, cy] = point(axis, 9, value);
        const dot = svgEl("circle", { cx, cy, r: "3.2", fill: "white", stroke: color, "stroke-width": "1.8", tabindex: "0" });
        dot.append(svgEl("title", {}, `${entry.name} · ${data.axes[axis].name}: ${fmt(value, 3)}`)); svg.append(dot);
      });
      const legend = el("div", "legend-item"); const line = el("span", "legend-dot"); line.style.background = color;
      legend.append(line, el("span", "", `${entry.name} ${fmt(entry.score)}${entry.kind === "group" && entry.n > 1 ? ` ± ${fmt(entry.std)}` : ""}`)); $("legend").append(legend);
    });
    $("radar").append(svg);
  }
  function renderDetails() {
    const entries = selected();
    if (!state.selected.includes(state.reference)) state.reference = state.selected[0] || null;
    $("reference").replaceChildren();
    for (const entry of entries) { const option = el("option", "", entry.name); option.value = entry.id; $("reference").append(option); }
    $("reference").value = state.reference || "";
    $("details").replaceChildren(); if (!entries.length) return;
    const reference = byId.get(state.reference) || entries[0];
    const table = el("table"); const head = el("tr");
    ["实验 / 预算", "Language", ...data.axes.map((axis) => axis.name)].forEach((name) => head.append(el("th", "", name)));
    const thead = el("thead"); thead.append(head); table.append(thead); const tbody = el("tbody");
    entries.forEach((entry, index) => {
      const row = el("tr"); const name = el("td"); const label = el("div", "", entry.name); label.style.color = COLORS[index];
      name.append(label, el("small", "", `${modelLabel(entry)} · ${entry.kind === "group" ? `n=${entry.n}` : `s${entry.seed}`}`)); row.append(name);
      [entry.score, ...entry.skills].forEach((value, axis) => {
        const cell = el("td", axis === 0 ? "total-cell" : "", fmt(value));
        if (entry.id !== reference.id) {
          const delta = value - [reference.score, ...reference.skills][axis];
          cell.append(el("small", delta > .00001 ? "positive" : delta < -.00001 ? "negative" : "", `${delta >= 0 ? "+" : ""}${fmt(delta)}`));
        }
        row.append(cell);
      });
      tbody.append(row);
    });
    table.append(tbody); $("details").append(table);
  }
  function persist() {
    try { root.history.replaceState(null, "", `#${new URLSearchParams({ mode: state.mode, selected: state.selected.join(",") })}`); } catch (_) { /* file:// and sandbox history may be unavailable */ }
  }
  function render() {
    $("selected-count").textContent = `${state.selected.length} / 5`;
    renderList(); renderRadar(); renderDetails(); persist();
  }
  function preset(name) {
    state.mode = "group"; $("mode").value = "group"; $("search").value = "";
    ["stage", "rank", "budget"].forEach((id) => { $(id).value = "all"; });
    const find = (code, stage) => data.groups.find((entry) => entry.code === code && (!stage || entry.stage === stage));
    let entries;
    if (name === "ratio") entries = ["NEG2", "NATURAL", "POS2", "POS4"].map((code) => find(code, "pair-ratio"));
    else if (name === "attention") entries = ["R1", "R2", "R3", "R4"].map((code) => find(code, "controlled"));
    else if (name === "capacity") entries = [find("JOINT", "joint"), find("LR32", "rank")];
    else if (name === "fusion") entries = [find("MIXED", "matched_mixed"), find("PROBABILITY_MEAN", "output_ensemble"), find("LINEAR", "weight_fusion"), find("TIES", "weight_fusion"), data.groups.filter((entry) => entry.stage === "specialist").sort((a,b) => b.score-a.score)[0]];
    else if (name === "distill") entries = ["HARD", "SOFT", "HYBRID"].map((code) => find(code, "student_distillation"));
    else entries = [find("R0"), find("R3", "controlled"), find("R5_DATA", "D2"), find("JOINT", "joint"), find("LR32", "rank")];
    state.selected = entries.filter(Boolean).map((entry) => entry.id); state.reference = state.selected[0];
    if (name === "fusion" && !state.selected.length) {
      render(); $("selection-message").textContent = "专家与融合评测正在进行；完成的九项结果会由数据更新程序接入。"; return;
    }
    if (name === "distill" && !state.selected.length) {
      render(); $("selection-message").textContent = "Teacher标注/Student对照正在接续，完整Language结果产出后自动接入。"; return;
    }
    $("selection-message").textContent = ""; render();
  }
  function renderLengths() {
    const inv = data.inventory; $("length-summary").replaceChildren();
    Object.entries(inv.length_quantiles).forEach(([key, value]) => { const card = el("article"); card.append(el("span", "", key === "max" ? "最长 / tokens" : `${key} / tokens`), el("strong", "", String(Math.round(value)))); $("length-summary").append(card); });
    Object.entries(inv.coverage_at_cap).forEach(([cap, share]) => {
      const row = el("div", "coverage-row"); const track = el("div", "coverage-track"); const bar = el("div", "coverage-fill"); bar.style.width = `${share * 100}%`; track.append(bar);
      row.append(el("span", "", `${cap} tokens`), track, el("span", "coverage-value", `${fmt(share * 100)}%`)); $("coverage-chart").append(row);
    });
    const table = el("table"); const head = el("tr"); ["输入上限", "加权Macro-F1", "相对full下降（百分点）", "面板耗时 / s", "峰值allocated / GiB"].forEach((name) => head.append(el("th", "", name)));
    const thead = el("thead"); thead.append(head); table.append(thead); const tbody = el("tbody");
    const full = data.length_results.find((row) => row.cap === "full");
    [...data.length_results].sort((a, b) => (a.cap === "full" ? Infinity : Number(a.cap)) - (b.cap === "full" ? Infinity : Number(b.cap))).forEach((row) => {
      const tr = el("tr"); [row.cap, `${fmt(row.macro_f1 * 100, 3)}%`, fmt((full.macro_f1 - row.macro_f1) * 100, 3), fmt(row.seconds), fmt(row.allocated_gib)].forEach((value) => tr.append(el("td", "", value))); tbody.append(tr);
    }); table.append(tbody); $("length-table").append(table);
  }
  function renderRoadmap() {
    data.roadmap.forEach((item, index) => {
      const row = el("article", `roadmap-item ${item.status}`); const text = el("div"); text.append(el("h3", "", item.title), el("p", "", item.detail));
      row.append(el("span", "roadmap-number", item.status === "done" ? "✓" : String(index + 1).padStart(2, "0")), text, el("span", "roadmap-state", { done: "已完成", next: "下一主线", running: "进行中", later: "后续" }[item.status])); $("roadmap-list").append(row);
    });
  }
  function download(name, content, mime) {
    const url = root.URL.createObjectURL(new Blob([content], { type: mime }));
    const link = el("a"); link.href = url; link.download = name; document.body.append(link); link.click(); link.remove();
    root.setTimeout(() => root.URL.revokeObjectURL(url), 1000);
  }
  $("run-count").textContent = data.counts.full;
  $("best-score").textContent = fmt(byId.get(data.best_id).score);
  $("updated").textContent = `数据更新 ${data.generated.slice(0, 16).replace("T", " ")} UTC`;
  Object.entries(data.stages).forEach(([id, name]) => { const option = el("option", "", name); option.value = id; $("stage").append(option); });
  ["search", "stage", "rank", "budget"].forEach((id) => $(id).addEventListener(id === "search" ? "input" : "change", renderList));
  $("mode").addEventListener("change", () => { state.mode = $("mode").value; state.selected = []; state.reference = null; render(); });
  $("reference").addEventListener("change", () => { state.reference = $("reference").value; renderDetails(); });
  $("clear").addEventListener("click", () => { state.selected = []; state.reference = null; render(); });
  document.querySelectorAll("[data-preset]").forEach((button) => button.addEventListener("click", () => preset(button.dataset.preset)));
  document.querySelectorAll("[data-tab]").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll("[data-tab]").forEach((other) => { const active = other === button; other.classList.toggle("active", active); other.setAttribute("aria-selected", active); });
    document.querySelectorAll(".tab-panel").forEach((panel) => { panel.hidden = panel.id !== button.dataset.tab; });
  }));
  $("export-svg").addEventListener("click", () => { const svg = $("radar").querySelector("svg"); if (svg) download("language-radar.svg", svg.outerHTML, "image/svg+xml;charset=utf-8"); });
  $("export-csv").addEventListener("click", () => {
    const rows = [["实验", "code", "seed_count", "seed", "rank", "new_tokens", "Language Skill", ...data.axes.map((axis) => axis.name)],
      ...selected().map((entry) => [entry.name, entry.code, entry.n, entry.seed, entry.rank, entry.tokens, entry.score, ...entry.skills])];
    download("language-comparison.csv", "\ufeff" + rows.map((row) => row.map(csvCell).join(",")).join("\r\n"), "text/csv;charset=utf-8");
  });
  renderLengths(); renderRoadmap();
  try {
    const query = new URLSearchParams(root.location.hash.slice(1));
    const mode = query.get("mode"); const ids = (query.get("selected") || "").split(",").filter((id) => byId.has(id));
    if (["run", "group"].includes(mode) && ids.length) {
      state.mode = mode; $("mode").value = mode; state.selected = [...new Set(ids)].filter((id) => byId.get(id).kind === mode).slice(0, 5); render();
    } else preset("journey");
  } catch (_) { preset("journey"); }
  root.LUDashboard = { state, preset, render, selected, data };
})(typeof window !== "undefined" ? window : globalThis);
