// Bench section: renders docs/bench/*.json exported by `perfpilot bench --export`.
(function () {
  "use strict";
  const BEFORE = "#C8323C";
  const AFTER = "#1F5FBF";
  const RAMP = ["#EEF3FA", "#CFDDF2", "#9DBBE6", "#5E8FD3", "#1F5FBF", "#123E80"];
  const VARIANTS = ["fallback", "single", "swarm"];
  const SVG = "http://www.w3.org/2000/svg";

  const tip = document.createElement("div");
  tip.className = "viz-tip";
  tip.setAttribute("role", "status");
  document.body.appendChild(tip);

  function showTip(evt, html) {
    tip.innerHTML = html;
    tip.style.opacity = "1";
    const pad = 14;
    const w = tip.offsetWidth;
    const x = Math.min(evt.clientX + pad, window.innerWidth - w - 8);
    tip.style.left = x + "px";
    tip.style.top = evt.clientY + pad + "px";
  }
  function hideTip() {
    tip.style.opacity = "0";
  }
  function el(tag, attrs, parent) {
    const node = document.createElementNS(SVG, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    if (parent) parent.appendChild(node);
    return node;
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  }
  function rampColor(t) {
    const i = Math.max(0, Math.min(RAMP.length - 1, Math.round(t * (RAMP.length - 1))));
    return { fill: RAMP[i], ink: i >= 3 ? "#FFFFFF" : "#292929" };
  }
  function hover(node, html) {
    node.addEventListener("mousemove", (e) => showTip(e, html));
    node.addEventListener("mouseleave", hideTip);
  }

  function heatmap(root, cells) {
    const scenarios = [...new Set(cells.map((c) => c.scenario))];
    const by = Object.fromEntries(cells.map((c) => [c.scenario + "|" + c.variant, c]));
    const probes = cells.map((c) => c.probes);
    const lo = Math.min(...probes), hi = Math.max(...probes);
    const table = document.createElement("table");
    table.className = "heat";
    const head = table.createTHead().insertRow();
    ["Scenario", ...VARIANTS].forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      head.appendChild(th);
    });
    const body = table.createTBody();
    let family = "";
    scenarios.forEach((sc) => {
      const first = by[sc + "|fallback"] || by[sc + "|swarm"];
      if (first.family !== family) {
        family = first.family;
        const r = body.insertRow();
        r.className = "fam";
        const td = r.insertCell();
        td.colSpan = 4;
        td.textContent = family === "core" ? "Core scenarios" : "Stress scenarios — real vLLM bisect hazards";
      }
      const row = body.insertRow();
      const name = row.insertCell();
      name.innerHTML = "<code>" + esc(sc) + "</code><small>" + esc(first.description) + "</small>";
      VARIANTS.forEach((v) => {
        const c = by[sc + "|" + v];
        const td = row.insertCell();
        if (!c) return;
        const { fill, ink } = rampColor(hi === lo ? 0 : (c.probes - lo) / (hi - lo));
        td.className = "cell";
        td.style.background = fill;
        td.style.color = ink;
        td.tabIndex = 0;
        td.innerHTML = "<b>" + c.probes.toFixed(1) + "</b><span>" + c.correct + "/" + c.n + "</span>";
        const html =
          "<b>" + esc(sc) + " · " + v + "</b><br>probes/run " + c.probes.toFixed(2) +
          " · $" + c.cost_usd.toFixed(2) + " sim<br>correct " + c.correct + "/" + c.n +
          " · exact " + c.exact + " · wrong " + c.wrong;
        hover(td, html);
        td.addEventListener("focus", () => {
          const r = td.getBoundingClientRect();
          showTip({ clientX: r.right, clientY: r.top }, html);
        });
        td.addEventListener("blur", hideTip);
      });
    });
    root.appendChild(table);
    const legend = document.createElement("div");
    legend.className = "ramp-legend";
    legend.innerHTML =
      "<span>" + lo.toFixed(0) + " probes</span>" +
      RAMP.map((c) => '<i style="background:' + c + '"></i>').join("") +
      "<span>" + hi.toFixed(0) + " probes</span><em>cell: mean probes per run, correct/runs below</em>";
    root.appendChild(legend);
  }

  function dumbbell(root, title, unit, rows, max) {
    const W = 520, rowH = 36, left = 190, right = 44, top = 32;
    const H = top + rows.length * rowH + 26;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    const x = (v) => left + (v / max) * (W - left - right);
    el("text", { x: 0, y: 14, class: "viz-title" }, svg).textContent = title;
    for (let t = 0; t <= max + 1e-9; t += max / 4) {
      el("line", { x1: x(t), x2: x(t), y1: top - 6, y2: H - 22, class: "grid" }, svg);
      el("text", { x: x(t), y: H - 6, class: "axis", "text-anchor": "middle" }, svg).textContent =
        Number.isInteger(t) ? t : t.toFixed(1);
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2;
      el("text", { x: 0, y: y + 4, class: "label" }, svg).textContent = r.name;
      el("line", { x1: x(r.before), x2: x(r.after), y1: y, y2: y, class: "bar-link" }, svg);
      [["before", BEFORE, r.before], ["after", AFTER, r.after]].forEach(([k, color, v]) => {
        el("circle", { cx: x(v), cy: y, r: 6, fill: color, stroke: "#FFFFFF", "stroke-width": 2, "aria-hidden": "true" }, svg);
        const hit = el("circle", { cx: x(v), cy: y, r: 14, fill: "transparent" }, svg);
        hover(hit, "<b>" + esc(r.name) + "</b><br>" + k + ": " + v + " " + unit + "<br>" + esc(r.fix));
      });
      const up = r.after >= r.before;
      el("text", { x: x(r.after) + (up ? 12 : -12), y: y + 4, class: "value", "text-anchor": up ? "start" : "end" }, svg)
        .textContent = r.after;
    });
    const fig = document.createElement("figure");
    fig.className = "viz";
    fig.appendChild(svg);
    root.appendChild(fig);
  }

  function trend(root, entries) {
    const pts = entries
      .filter((e) => e.mode === "scripted-all" && e.variants && e.variants.swarm)
      .map((e) => ({ ts: e.ts, sha: e.git_sha, score: e.variants.swarm.score, correct: e.variants.swarm.correct_rate }));
    if (pts.length < 2) {
      root.innerHTML = '<p class="report-meta">Scoreboard trend appears after two recorded runs.</p>';
      return;
    }
    const W = 640, H = 180, l = 44, r = 20, t = 28, b = 28;
    const ys = pts.map((p) => p.score);
    const lo = Math.floor(Math.min(...ys) - 1), hi = Math.ceil(Math.max(...ys) + 1);
    const x = (i) => l + (i / (pts.length - 1)) * (W - l - r);
    const y = (v) => t + (1 - (v - lo) / (hi - lo)) * (H - t - b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Swarm score per recorded bench run" });
    el("text", { x: 0, y: 14, class: "viz-title" }, svg).textContent = "Swarm score for each recorded run, 17 scenarios";
    [lo, (lo + hi) / 2, hi].forEach((v) => {
      el("line", { x1: l, x2: W - r, y1: y(v), y2: y(v), class: "grid" }, svg);
      el("text", { x: l - 8, y: y(v) + 4, class: "axis", "text-anchor": "end" }, svg).textContent = v.toFixed(1);
    });
    el("polyline", { points: pts.map((p, i) => x(i) + "," + y(p.score)).join(" "), class: "trend-line", stroke: AFTER }, svg);
    pts.forEach((p, i) => {
      el("circle", { cx: x(i), cy: y(p.score), r: 5, fill: AFTER, stroke: "#FFFFFF", "stroke-width": 2 }, svg);
      const hit = el("circle", { cx: x(i), cy: y(p.score), r: 14, fill: "transparent" }, svg);
      hover(hit, "<b>score " + p.score.toFixed(1) + "</b><br>" + esc(p.ts.slice(0, 16)) + " · " +
        esc(p.sha.slice(0, 12)) + "<br>correct " + Math.round(p.correct * 100) + "%");
    });
    const fig = document.createElement("figure");
    fig.className = "viz";
    fig.appendChild(svg);
    root.appendChild(fig);
  }

  function votes(root, trace) {
    if (!trace || !trace.picks || !trace.picks.length) return;
    const list = document.createElement("ol");
    list.className = "votes";
    trace.picks.forEach((p, i) => {
      const li = document.createElement("li");
      const chips = p.votes
        .map((v) => {
          const won = v.sha && v.sha === p.chosen;
          return '<span class="vote' + (won ? " won" : "") + '"><em>' + esc(v.role) + "</em> " +
            (v.sha ? esc(v.sha.slice(-2)) : "abstain") + " · " + esc(v.confidence) + "</span>";
        })
        .join("");
      li.innerHTML = '<span class="pick">probe ' + (i + 1) + " → " + esc((p.chosen || "–").slice(-2)) +
        (p.label ? " <small>" + esc(p.label) + "</small>" : "") + "</span>" + chips;
      list.appendChild(li);
    });
    root.appendChild(list);
    const meta = document.createElement("p");
    meta.className = "report-meta";
    meta.textContent = trace.caption;
    root.appendChild(meta);
  }

  function liveTable(root, entries) {
    const live = entries.filter((e) => e.mode === "live");
    if (!root || !live.length) return;
    const rows = live.map((e) => {
      const cells = ["single", "swarm"].map((v) => {
        const r = e.variants[v];
        return r ? r.mean_probes.toFixed(2) + " (" + Math.round(r.correct_rate * 100) + "%)" : "–";
      });
      return "<tr><td>" + esc(e.ts.slice(0, 16).replace("T", " ")) + "</td><td>" + esc(e.git_sha.slice(0, 8)) +
        "</td><td>" + cells[0] + "</td><td>" + cells[1] + "</td></tr>";
    }).join("");
    root.innerHTML = '<div class="table-scroll"><table class="data"><tr><th>Recorded</th><th>Commit</th>' +
      "<th>Single agent: measurements (correct)</th><th>Swarm: measurements (correct)</th></tr>" + rows +
      '</table></div><p class="note">Live Baseten model, 7 core scenarios, 1 seed each. The first live run showed the swarm ' +
      "losing (11.6 vs 9.4): the model gave ungrounded high-confidence votes. Confidence is now checked against the commit " +
      "message, and the statistician runs deterministically.</p>";
  }

  const FIX_CORRECT = [
    { name: "outlier (collapsed runs)", before: 17, after: 20, fix: "drift re-measure guarded; pooled median trim when both endpoint sets are contaminated" },
    { name: "flaky_worker (10% preempt)", before: 16, after: 20, fix: "run_bisect retries transient 'trial error'; build skips never retried" },
  ];
  const FIX_PROBES = [
    { name: "hint_spam, swarm", before: 20.3, after: 9.1, fix: "code_reader abstains when >25% of the open range matches" },
    { name: "code_hint, scripted swarm", before: 9.0, after: 6.0, fix: "code_reader + statistician confirm-parent combo vs plain bisect" },
    { name: "code_hint, live swarm", before: 10, after: 6, fix: "live roles now told the last probe, its label and its parent" },
  ];

  Promise.all([
    fetch("bench/bench.json").then((r) => r.json()),
    fetch("bench/swarm_trace.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ])
    .then(([data, trace]) => {
      heatmap(document.getElementById("bench-heat"), data.cells);
      const fixes = document.getElementById("bench-fixes");
      dumbbell(fixes, "Runs correct out of 20", "/20", FIX_CORRECT, 20);
      dumbbell(fixes, "Measurements per run, lower is cheaper", "probes", FIX_PROBES, 24);
      trend(document.getElementById("bench-trend"), data.scoreboard);
      liveTable(document.getElementById("bench-live"), data.scoreboard);
      votes(document.getElementById("bench-votes"), trace);
      const rows = data.variants
        .map((v) => "<tr><td>" + esc(v.variant) + "</td><td>" + Math.round(v.correct_rate * 100) + "%</td><td>" +
          Math.round(v.exact_rate * 100) + "%</td><td>" + v.wrong + "</td><td>" + v.mean_probes.toFixed(2) +
          "</td><td>$" + v.mean_cost_usd.toFixed(2) + "</td><td>" + v.score.toFixed(1) + "</td></tr>")
        .join("");
      document.getElementById("bench-variants").innerHTML =
        "<tr><th>Variant</th><th>Correct</th><th>Exact</th><th>Wrong</th><th>Probes/run</th><th>$/run (sim)</th><th>Score</th></tr>" + rows;
    })
    .catch((err) => {
      document.getElementById("bench-heat").textContent = "Bench data failed to load: " + err.message;
    });
})();
