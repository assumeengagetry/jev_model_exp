const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");
const { point, matches, csvCell } = require("./app.js");
const data = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));

test("all completed runs include the95 archived endpoints and nine bounded scores", () => {
  assert.ok(data.runs.length >= 95);
  const weights = data.runs.map((row) => row.checkpoint).filter(Boolean);
  assert.equal(new Set(weights).size, weights.length);
  for (const row of data.runs) {
    assert.equal(row.skills.length, 9);
    assert.ok(row.skills.every((value) => Number.isFinite(value) && value >= 0 && value <= 100));
    assert.equal(row.coverage, 1);
    assert.equal(row.requests, 31715);
  }
});
test("radar uses one0–100 coordinate scale", () => {
  assert.deepEqual(point(0, 9, 0), [380, 310]);
  assert.deepEqual(point(0, 9, 100), [380, 97]);
  assert.deepEqual(point(0, 9, 200), point(0, 9, 100));
});
test("the20M filter includes the exact ratio19999954 budget", () => {
  const row = data.runs.find((entry) => entry.code === "POS2");
  assert.ok(matches(row, { search: "POS2", stage: "pair-ratio", rank: "32", budget: "20000000" }));
  assert.ok(!matches(row, { search: "POS2", stage: "pair-ratio", rank: "16", budget: "20000000" }));
});
test("seed means and historical winner agree with archived results", () => {
  const r32 = data.groups.find((row) => row.code === "LR32" && row.stage === "rank");
  assert.equal(r32.n, 4);
  assert.ok(Math.abs(r32.score - 35.04917625168053) < .000001);
  const historicalBest = data.runs.filter((row) => row.id.startsWith("run-")).sort((a,b) => b.score-a.score)[0];
  assert.equal(historicalBest.seed, 3409);
  assert.equal(Number(historicalBest.score.toFixed(2)), 36.33);
});
test("CSV quotes values and neutralizes leading spreadsheet formulas", () => {
  assert.equal(csvCell('a,"b"'), '"a,""b"""');
  assert.equal(csvCell("=1+1"), '"\'=1+1"');
});

const domPath = process.env.DASHBOARD_DOM_MODULE;
if (domPath) {
  const { parseHTML } = require(domPath);
  const selectValues = new WeakMap();
  function browser(hash = "") {
    const { window } = parseHTML(fs.readFileSync(path.join(__dirname, "index.html"), "utf8"));
    // linkedom implements select.value as getter-only; emulate the browser setter.
    Object.defineProperty(window.HTMLSelectElement.prototype, "value", {
      configurable: true,
      get() { return selectValues.has(this) ? selectValues.get(this) : this.querySelector("option")?.value || ""; },
      set(value) { selectValues.set(this, String(value)); }
    });
    window.location = { hash };
    window.history = { replaceState() {} };
    window.setTimeout = setTimeout;
    const context = vm.createContext({ window, document: window.document, console, URLSearchParams, Blob });
    vm.runInContext(fs.readFileSync(path.join(__dirname, "data.js"), "utf8"), context);
    vm.runInContext(fs.readFileSync(path.join(__dirname, "app.js"), "utf8"), context);
    return window;
  }
  test("offline page renders defaults, radar and actual latest-ratio scores", () => {
    const window = browser(); const document = window.document;
    assert.equal(document.getElementById("run-count").textContent, String(data.runs.length));
    assert.equal(document.getElementById("best-score").textContent, data.runs.find((row) => row.id === data.best_id).score.toFixed(2));
    assert.equal(window.LUDashboard.selected().length, 5);
    assert.equal(document.querySelectorAll("#radar circle").length, 45);
    document.querySelector('[data-preset="ratio"]').click();
    assert.equal(window.LUDashboard.selected().length, 4);
    assert.equal(document.querySelectorAll("#radar circle").length, 36);
    assert.ok(document.getElementById("details").textContent.includes("31.23"));
    assert.ok(!document.body.textContent.includes("NaN"));
  });
  test("filters, empty selection, five-item cap and tabs work", () => {
    const window = browser(); const document = window.document;
    document.querySelector('[data-preset="ratio"]').click();
    const fifth = [...document.querySelectorAll("#experiments input")].find((node) => !node.checked);
    fifth.checked = true; fifth.dispatchEvent(new window.Event("change"));
    assert.equal(window.LUDashboard.selected().length, 5);
    const sixth = [...document.querySelectorAll("#experiments input")].find((node) => !node.checked);
    sixth.checked = true; sixth.dispatchEvent(new window.Event("change"));
    assert.equal(window.LUDashboard.selected().length, 5);
    assert.ok(document.getElementById("selection-message").textContent.includes("最多"));
    document.getElementById("search").value = "no-such-run";
    document.getElementById("search").dispatchEvent(new window.Event("input"));
    assert.equal(document.querySelectorAll("#experiments input").length, 0);
    document.getElementById("clear").click();
    assert.equal(window.LUDashboard.selected().length, 0);
    assert.equal(document.querySelectorAll("#radar svg").length, 0);
    document.querySelector('[data-tab="length"]').click();
    assert.equal(document.getElementById("length").hidden, false);
    assert.equal(document.getElementById("compare").hidden, true);
    assert.ok(document.getElementById("length-table").textContent.includes("90.451"));
  });
  test("URL restoration ignores invalid and duplicate selections", () => {
    const window = browser("#mode=run&selected=run-0,invalid,run-0");
    assert.equal(window.LUDashboard.selected().length, 1);
    assert.equal(window.LUDashboard.selected()[0].code, "R0");
  });
}
