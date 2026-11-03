// Hero: replays a recorded bisect from docs/bench/hero.json (exported from runs/<id>).
(function () {
  "use strict";
  const SVG = "http://www.w3.org/2000/svg";
  const SIGNAL = "#1F5FBF";
  const SLOW = "#C8323C";
  const H = 300, L = 48, R = 16, T = 18, B = 34;
  let W = 1100;
  const root = document.getElementById("cliff-svg");
  const caption = document.getElementById("cliff-status");
  const replay = document.getElementById("cliff-replay");
  if (!root) return;

  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let timers = [];

  function el(tag, attrs, parent) {
    const n = document.createElementNS(SVG, tag);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

  function frames(trace) {
    const n = trace.n_commits;
    const first = trace.probes.find((p) => p.idx === 0);
    const last = trace.probes.find((p) => p.idx === n - 1);
    const mid = first && last ? (mean(first.runs) + mean(last.runs)) / 2 : null;
    let good = 0, bad = n - 1;
    return trace.probes.map((p) => {
      const slow = mid !== null && mean(p.runs) < mid;
      if (p.idx !== 0 && p.idx !== n - 1) {
        if (slow) bad = Math.min(bad, p.idx);
        else good = Math.max(good, p.idx);
      }
      return { probe: p, slow, good, bad };
    });
  }

  function draw(trace) {
    timers.forEach(clearTimeout);
    timers = [];
    root.replaceChildren();
    W = root.clientWidth < 700 ? 520 : 1100;
    const n = trace.n_commits;
    const all = trace.probes.flatMap((p) => p.runs);
    const yMax = Math.ceil(Math.max(...all) / 500) * 500;
    const x = (i) => L + (i / (n - 1)) * (W - L - R);
    const y = (v) => T + (1 - v / yMax) * (H - T - B);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Recorded bisect ${trace.run_id}: per-commit throughput, bracket narrowing onto the culprit` }, root);

    for (let v = 0; v <= yMax; v += 1000) {
      el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid" }, svg);
      el("text", { x: L - 10, y: y(v) + 4, "text-anchor": "end", class: "axis" }, svg).textContent =
        v === 0 ? "0" : (v / 1000) + "k";
    }
    el("text", { x: L - 10, y: T - 4, "text-anchor": "end", class: "axis" }, svg).textContent = "tok/s";
    for (let i = 0; i < n; i++) {
      el("text", { x: x(i), y: H - 12, "text-anchor": "middle", class: "axis" }, svg).textContent = i;
    }
    el("text", { x: W - R, y: H, "text-anchor": "end", class: "axis" }, svg).textContent = "commit, oldest to newest";

    const half = (W - L - R) / (n - 1) / 2;
    const bracket = el("rect", { x: x(0), y: T, width: x(n - 1) - x(0), height: H - T - B, class: "bracket" }, svg);
    const edgeL = el("line", { x1: x(0), x2: x(0), y1: T, y2: H - B, class: "bracket-edge" }, svg);
    const edgeR = el("line", { x1: x(n - 1), x2: x(n - 1), y1: T, y2: H - B, class: "bracket-edge" }, svg);
    const band = el("g", {}, svg);
    const dots = el("g", {}, svg);
    const seq = frames(trace);

    function setBracket(good, bad) {
      bracket.setAttribute("x", x(good));
      bracket.setAttribute("width", Math.max(x(bad) - x(good), 2));
      [[edgeL, good], [edgeR, bad]].forEach(([e, i]) => { e.setAttribute("x1", x(i)); e.setAttribute("x2", x(i)); });
    }

    function step(k) {
      const f = seq[k];
      f.probe.runs.forEach((v, j) => {
        const jitter = (j - (f.probe.runs.length - 1) / 2) * 5;
        el("circle", { cx: x(f.probe.idx) + jitter, cy: y(v), r: 4.5, fill: f.slow ? SLOW : SIGNAL,
          stroke: "#FFFFFF", "stroke-width": 1.5 }, dots);
      });
      setBracket(f.good, f.bad);
      caption.textContent = `Trial ${k + 1} of ${seq.length}: commit ${f.probe.idx} measured ` +
        `${Math.round(mean(f.probe.runs)).toLocaleString()} tok/s (${f.slow ? "slow" : "fast"}). ` +
        `Suspects: commits ${f.good + 1} to ${f.bad}.`;
    }

    function finish() {
      const f = seq[seq.length - 1];
      const lo = f.good + 1, hi = f.bad;
      el("rect", { x: x(lo) - half, y: T, width: x(hi) - x(lo) + 2 * half, height: H - T - B, class: "culprit-band" }, band);
      [bracket, edgeL, edgeR].forEach((e) => e.setAttribute("visibility", "hidden"));
      const label = lo === hi ? `culprit: commit ${hi}` : `culprit in commits ${lo} to ${hi}`;
      el("text", { x: x(hi) + half + 6, y: T + 14, class: "ann ann-slow" }, svg).textContent = label;
      const planted = trace.planted_idx;
      caption.textContent = `${trace.run_id}: ${seq.length} commit trials on an A100, ${label}` +
        (planted !== undefined && planted !== null ? `; planted at commit ${planted}.` : ".");
    }

    if (reduce) {
      seq.forEach((_, k) => step(k));
      finish();
      return;
    }
    seq.forEach((_, k) => timers.push(setTimeout(() => step(k), 500 + k * 650)));
    timers.push(setTimeout(finish, 500 + seq.length * 650));
  }

  fetch("bench/hero.json")
    .then((r) => r.json())
    .then((trace) => {
      draw(trace);
      if (replay) replay.addEventListener("click", () => draw(trace));
    })
    .catch(() => { caption.textContent = "Run data failed to load. Open a recorded report below instead."; });
})();
