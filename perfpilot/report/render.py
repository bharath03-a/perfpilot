"""events.jsonl -> report.html. Pure: no GPU, no network.

Signatures (CLI wiring comes later):

    render_report(events_path: str | Path) -> str
    render_watch_page(events_path: str | Path, refresh_seconds: int = 5) -> str

`render_report` returns a self-contained HTML string (Chart.js via CDN)
with: verdict card, bisect timeline, throughput chart with intervals,
agent panel, explanation with diff links, TuneSmith winner + paste-ready
command, reproducibility block. `render_watch_page` is the same page with
an auto-refresh meta tag (``perfpilot watch`` demo view).
"""

from __future__ import annotations

import html
import json
import math
import warnings
from pathlib import Path
from typing import Any

from perfpilot.config import PerfPilotConfig
from perfpilot.modes.tune import serve_command

CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
#: $/h used only when the run_start event lacks gpu_cost_per_hour.
#: Mirrors PerfPilotConfig.gpu_cost_per_hour (measured A100-40GB rate);
#: never hardcode a competing constant.
FALLBACK_GPU_COST_PER_HOUR = float(
    PerfPilotConfig.model_fields["gpu_cost_per_hour"].default
)

EVENT_TYPES = (
    "run_start",
    "trial_start",
    "trial_result",
    "verdict",
    "agent_call",
    "boundary_update",
    "culprit",
    "verify",
    "explain",
    "tune_candidate",
    "tune_accept",
    "tune_result",
    "quality_check",
    "run_end",
)

_LABEL_CLASS = {
    "good": "good",
    "bad": "bad",
    "skip": "skipped",
    "skipped": "skipped",
    "ambiguous": "ambiguous",
    "inconclusive": "ambiguous",
}


def load_events(events_path: str | Path) -> list[dict[str, Any]]:
    """Read events.jsonl; unknown types warn-and-keep (emit side raises)."""
    events: list[dict[str, Any]] = []
    for line in Path(events_path).read_text().splitlines():
        if line.strip():
            ev = json.loads(line)
            if ev.get("type") not in EVENT_TYPES:
                warnings.warn(
                    f"unknown event type {ev.get('type')!r}: kept",
                    stacklevel=2,
                )
            events.append(ev)
    return events


def load_run_events(events_path: str | Path) -> list[dict[str, Any]]:
    """events.jsonl merged with a sibling agent-events.jsonl (if present).

    Agent calls are logged to a separate file by the CLI; the report must
    read both. Merged stream is sorted by ``ts`` (stable, so on equal
    timestamps main events come first).
    """
    events = load_events(events_path)
    sibling = Path(events_path).parent / "agent-events.jsonl"
    if sibling.exists():
        for line in sibling.read_text().splitlines():
            if line.strip():
                ev = json.loads(line)
                if ev.get("type") not in EVENT_TYPES:
                    warnings.warn(
                        f"unknown event type {ev.get('type')!r}: kept",
                        stacklevel=2,
                    )
                events.append(ev)
        events.sort(key=lambda e: str(e.get("ts", "")))
    return events


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _mean(xs: list[float]) -> float:
    return math.fsum(xs) / len(xs) if xs else 0.0


def _throughput(ev: dict[str, Any]) -> tuple[float, float | None, float | None]:
    """(mean, lo, hi) throughput for a trial_result event."""
    if isinstance(ev.get("mean"), (int, float)):
        m = float(ev["mean"])
        lo = ev.get("lo")
        hi = ev.get("hi")
        return m, lo, hi
    for key in ("output_throughput", "throughput"):
        if isinstance(ev.get(key), (int, float)):
            return float(ev[key]), None, None
    runs = ev.get("runs") or []
    vals: list[float] = []
    for r in runs:
        if isinstance(r, (int, float)):
            vals.append(float(r))
        elif isinstance(r, dict):
            for k in ("output_throughput", "throughput", "value"):
                if isinstance(r.get(k), (int, float)):
                    vals.append(float(r[k]))
                    break
    return _mean(vals), None, None


def _pct(effect: float | None) -> str:
    if effect is None:
        return "n/a"
    return f"{(math.exp(effect) - 1.0) * 100.0:+.1f}%"


def _endpoint_effect(events: list[dict[str, Any]]) -> tuple[float | None, str]:
    """Fallback regression for runs whose culprit event predates effect
    logging: log-ratio of the bad endpoint mean over the good endpoint
    mean, using the first two distinct trial_result SHAs as the
    endpoints (bisect always measures endpoints first)."""
    seen: list[str] = []
    samples: dict[str, list[float]] = {}
    for e in events:
        if e.get("type") != "trial_result":
            continue
        sha = str(e.get("sha") or e.get("commit") or "")
        if not sha:
            continue
        m, _, _ = _throughput(e)
        if sha not in samples:
            samples[sha] = []
            seen.append(sha)
        if m:
            samples[sha].append(m)
    if len(seen) < 2:
        return None, ""
    good, bad = samples[seen[0]], samples[seen[1]]
    if not good or not bad:
        return None, ""
    mg = math.fsum(good) / len(good)
    mb = math.fsum(bad) / len(bad)
    if mg <= 0 or mb <= 0:
        return None, ""
    return math.log(mb / mg), " (derived from endpoint trials)"


def _verdict_card(events: list[dict[str, Any]]) -> str:
    culprit = next((e for e in reversed(events) if e.get("type") == "culprit"), {})
    verify = next((e for e in reversed(events) if e.get("type") == "verify"), {})
    start = next((e for e in events if e.get("type") == "run_start"), {})
    rate = float(start.get("gpu_cost_per_hour") or FALLBACK_GPU_COST_PER_HOUR)
    gpu_s = math.fsum(
        float(e.get("gpu_seconds", 0.0) or 0.0)
        for e in events
        if e.get("type") == "trial_result"
    )
    gpu_min = gpu_s / 60.0
    cost = gpu_min / 60.0 * rate
    sha = culprit.get("sha") or culprit.get("culprit") or culprit.get("commit")
    culprit_range = culprit.get("culprit_range") or []
    if sha:
        culprit_txt = f"<code>{_esc(sha)}</code>"
    elif culprit_range:
        culprit_txt = (
            f"range of {len(culprit_range)} "
            f"(<code>{_esc(culprit_range[0][:12])}</code>…"
            f"<code>{_esc(culprit_range[-1][:12])}</code>)"
        )
    else:
        culprit_txt = "n/a"
    effect = culprit.get("effect")
    lo = culprit.get("lo", effect)
    hi = culprit.get("hi", effect)
    note = ""
    if effect is None:
        effect, note = _endpoint_effect(events)
        lo = hi = effect
    verified = verify.get("verified", verify.get("passed", culprit.get("verified")))
    if verified:
        vtxt = "verified"
    elif verify or "verified" in culprit:
        vtxt = "unverified"
    else:
        vtxt = "no verify step"
    return (
        "<section><h2>Verdict</h2><div class='card'>"
        f"<p>Culprit: {culprit_txt}</p>"
        f"<p>Regression: <b>{_esc(_pct(effect))}</b>"
        + (
            f" (interval [{_esc(_pct(lo))}, {_esc(_pct(hi))}]){_esc(note)}"
            if effect is not None
            else " (not logged for this run)"
        )
        + "</p>"
        f"<p>Status: <b>{_esc(vtxt)}</b></p>"
        f"<p>GPU: {gpu_min:.1f} min (${cost:.2f} @ ${rate:.2f}/h)</p>"
        "</div></section>"
    )


def _timeline(events: list[dict[str, Any]]) -> str:
    """Per-step audit trail: label plus effect/lo/hi/reason where known.

    Coloring and text use only label/verdict, never raw trial status
    (``ok``/``skip``/``error`` describe the run, not the verdict).
    """
    info: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for e in events:
        if e.get("type") not in ("trial_result", "trial_start", "verdict"):
            continue
        sha = str(e.get("sha") or e.get("commit") or "")
        if not sha:
            continue
        if sha not in info:
            info[sha] = {}
            order.append(sha)
        rec = info[sha]
        label = e.get("label") or e.get("verdict")
        if label:
            rec["label"] = str(label)
        if e.get("type") == "verdict":
            for k in ("effect", "lo", "hi", "reason"):
                if e.get(k) is not None:
                    rec[k] = e[k]
        elif e.get("reason") is not None and "reason" not in rec:
            rec["reason"] = e["reason"]

    def _item(sha: str) -> str:
        rec = info[sha]
        label = str(rec.get("label", "?"))
        cls = _LABEL_CLASS.get(label.lower(), "")
        detail = ""
        if rec.get("effect") is not None:
            detail += f" effect {_esc(_pct(rec['effect']))}"
            if rec.get("lo") is not None and rec.get("hi") is not None:
                detail += f" [{_esc(_pct(rec['lo']))}, {_esc(_pct(rec['hi']))}]"
        if rec.get("reason"):
            detail += f": {_esc(rec['reason'])}"
        return (
            f"<li class='{cls}'><code>{_esc(sha)}</code> — {_esc(label)}{detail}</li>"
        )

    bounds = [e for e in events if e.get("type") == "boundary_update"]
    items = "".join(_item(s) for s in order) or "<li>No commits tested yet.</li>"
    steps = "".join(
        f"<li>{_esc(b.get('ts', ''))}: good=<code>{_esc(b.get('good', ''))}</code> "
        f"bad=<code>{_esc(b.get('bad', ''))}</code></li>"
        for b in bounds
    )
    return (
        "<section><h2>Bisect timeline</h2>"
        f"<ul class='timeline'>{items}</ul>"
        + (f"<h3>Boundary updates</h3><ul>{steps}</ul>" if steps else "")
        + "</section>"
    )


def _chart(events: list[dict[str, Any]]) -> str:
    labels, means, los, his = [], [], [], []
    for e in events:
        if e.get("type") != "trial_result":
            continue
        m, lo, hi = _throughput(e)
        labels.append(str(e.get("sha") or e.get("commit") or "?"))
        means.append(m)
        los.append(lo if lo is not None else m)
        his.append(hi if hi is not None else m)
    if not labels:
        return "<section><h2>Throughput</h2><p>No trial results yet.</p></section>"
    return (
        "<section><h2>Throughput (tok/s) with intervals</h2>"
        "<canvas id='tput' height='120'></canvas>"
        "<script>"
        f"const labels={json.dumps(labels)};"
        f"const means={json.dumps(means)};"
        f"const los={json.dumps(los)};"
        f"const his={json.dumps(his)};"
        "new Chart(document.getElementById('tput'),{type:'line',"
        "data:{labels:labels,datasets:["
        "{label:'mean',data:means,borderColor:'#4a9',fill:false},"
        "{label:'lo',data:los,borderColor:'#999',borderDash:[4,4],fill:false},"
        "{label:'hi',data:his,borderColor:'#999',borderDash:[4,4],fill:false}]},"
        "options:{responsive:true}});"
        "</script></section>"
    )


def _agent_panel(events: list[dict[str, Any]]) -> str:
    calls = [e for e in events if e.get("type") == "agent_call"]
    if not calls:
        return (
            "<section><h2>Agent proposals</h2><p>No agent calls logged.</p></section>"
        )

    def _fallback(c: dict[str, Any]) -> str:
        fb = c.get("fallback", "")
        if isinstance(fb, (dict, list)):
            fb = json.dumps(fb, default=str)
        return _esc(fb)

    rows = "".join(
        "<tr><td>"
        + _esc(c.get("decision", ""))
        + "</td><td><code>"
        + _esc(json.dumps(c.get("proposal", {}), default=str))
        + "</code></td><td>"
        + ("accepted" if c.get("accepted", True) else "rejected")
        + "</td><td>"
        + _esc(c.get("why", c.get("reason", "")))
        + "</td><td><code>"
        + _fallback(c)
        + "</code></td></tr>"
        for c in calls
    )
    return (
        "<section><h2>Agent proposals</h2><table>"
        "<tr><th>Decision</th><th>Proposal</th><th>Status</th>"
        "<th>Why</th><th>Fallback</th></tr>"
        f"{rows}</table></section>"
    )


def _explanation(events: list[dict[str, Any]]) -> str:
    explains = [e for e in events if e.get("type") == "explain"]
    if not explains:
        return "<section><h2>Explanation</h2><p>No explanation recorded.</p></section>"
    start = next((e for e in events if e.get("type") == "run_start"), {})
    repo = str(start.get("repo", "vllm-project/vllm"))
    sha = str(explains[-1].get("sha", ""))
    blocks = ""
    for e in explains:
        files = "".join(
            f"<li><a href='https://github.com/{_esc(repo)}/blob/{_esc(sha)}/{_esc(f)}'>"
            f"{_esc(f)}</a></li>"
            for f in (e.get("files", []) or [])
        )
        blocks += (
            f"<p>{_esc(e.get('summary', ''))}</p>"
            f"<p>Mechanism: {_esc(e.get('mechanism', ''))} "
            f"(confidence: {_esc(e.get('confidence', ''))})</p>"
            f"<ul>{files}</ul>"
        )
    return f"<section><h2>Explanation</h2>{blocks}</section>"


def _tune(events: list[dict[str, Any]]) -> str:
    cands = [e for e in events if e.get("type") == "tune_candidate"]
    accepts = [e for e in events if e.get("type") == "tune_accept"]
    quals = [e for e in events if e.get("type") == "quality_check"]
    result = next((e for e in reversed(events) if e.get("type") == "tune_result"), {})
    if not cands and not accepts and not result:
        return "<section><h2>TuneSmith</h2><p>No tuning candidates tried.</p></section>"

    def _cand_cfg(c: dict[str, Any]) -> str:
        cfg = c.get("config", {c.get("knob", ""): c.get("value", "")})
        return _esc(json.dumps(cfg, default=str))

    def _interval(c: dict[str, Any]) -> str:
        if c.get("effect") is None:
            return "n/a"
        txt = _pct(c["effect"])
        if c.get("lo") is not None and c.get("hi") is not None:
            txt += f" [{_pct(c['lo'])}, {_pct(c['hi'])}]"
        return txt

    def _outcome(c: dict[str, Any]) -> str:
        if "decision" in c:
            return str(c["decision"])
        if "accepted" in c:
            return "accepted" if c["accepted"] else "rejected"
        return "tried"

    rows = "".join(
        f"<tr><td><code>{_cand_cfg(c)}</code></td>"
        f"<td>{_esc(_outcome(c))}</td>"
        f"<td>{_esc(_interval(c))}</td>"
        f"<td>{_esc(c.get('reason', ''))}</td></tr>"
        for c in cands
    )
    winner: dict[str, Any] = {}
    for a in reversed(accepts):
        if a.get("accepted", True):
            winner = a
            break
    start = next((e for e in events if e.get("type") == "run_start"), {})
    model = str(result.get("model") or start.get("model") or "")
    confirmed = bool(result.get("confirmed", bool(winner)))
    best_cfg = result.get("best") or {}
    wcfg = winner.get("config", {}) if winner else best_cfg
    uplift = f" (uplift {_esc(_interval(winner))})" if winner else ""
    if confirmed and wcfg and model:
        cmd_txt = f"<p>Run it: <code>{_esc(serve_command(model, wcfg))}</code></p>"
    elif confirmed and wcfg:
        cmd_txt = (
            f"<p>Winner: <code>{_esc(json.dumps(wcfg, default=str))}</code>{uplift}</p>"
        )
    else:
        cmd_txt = (
            "<p>Baseline kept: no candidate cleared the uplift bar, "
            "so the baseline stays.</p>"
        )

    def _qual(q: dict[str, Any]) -> str:
        drop = q.get("accuracy_drop_pts", q.get("accuracy_drop", q.get("drop_pts", "")))

        def _pctv(v: Any) -> str:
            return f"{v}%" if isinstance(v, (int, float)) else _esc(v)

        parts = [
            "pass" if q.get("passed", True) else "FAIL",
            f"accuracy {_pctv(q.get('accuracy', ''))}",
            f"baseline {_pctv(q.get('baseline_accuracy', ''))}",
            f"drop {_pctv(drop)} pts",
        ]
        if q.get("reason"):
            parts.append(_esc(q["reason"]))
        return (
            f"<li><code>{_esc(json.dumps(q.get('config', {}), default=str))}</code>: "
            + ", ".join(parts)
            + "</li>"
        )

    quals_txt = "".join(_qual(q) for q in quals)
    if wcfg:
        outcome_txt = (
            f"<p>Winner: <code>{_esc(json.dumps(wcfg, default=str))}</code>{uplift}</p>"
        )
    else:
        outcome_txt = ""
    return (
        "<section><h2>TuneSmith</h2><table>"
        "<tr><th>Candidate</th><th>Outcome</th><th>Effect + interval</th>"
        "<th>Why tried</th></tr>"
        f"{rows}</table>"
        f"{outcome_txt}{cmd_txt}"
        + (f"<h3>Quality guard</h3><ul>{quals_txt}</ul>" if quals_txt else "")
        + "</section>"
    )


def _repro(events: list[dict[str, Any]]) -> str:
    start = next((e for e in events if e.get("type") == "run_start"), {})
    end = next((e for e in reversed(events) if e.get("type") == "run_end"), {})
    keys = (
        "run_id",
        "model",
        "gpu",
        "gpu_uuid",
        "vllm_version",
        "bench_version",
        "repo",
        "good",
        "bad",
        "config",
    )

    def _cell(k: str) -> str:
        v = start.get(k, "")
        if isinstance(v, (dict, list)):
            v = json.dumps(v, default=str)
        return f"<tr><td>{_esc(k)}</td><td><code>{_esc(v)}</code></td></tr>"

    rows = "".join(_cell(k) for k in keys if start.get(k) is not None)
    status = _esc(end.get("status", end.get("result", "")) if end else "running")
    return (
        "<section><h2>Reproducibility</h2><table>"
        f"{rows}<tr><td>run status</td><td><code>{status}</code></td></tr>"
        "</table></section>"
    )


CSS = (
    "body{font-family:system-ui,sans-serif;max-width:900px}"
    "body{margin:2rem auto;padding:0 1rem}"
    ".card{border:1px solid #ccc;border-radius:8px}"
    ".card{padding:1rem;background:#fafafa}"
    ".timeline li{margin:.25rem 0}"
    ".good{color:#1a7f37}.bad{color:#d1242f}"
    ".skipped{color:#8250df}.ambiguous{color:#9a6700}"
    "table{border-collapse:collapse;width:100%}"
    "td,th{border:1px solid #ddd;padding:.4rem}"
)


def render_report(events_path: str | Path, live: bool = False) -> str:
    """Render events.jsonl to a self-contained HTML report string.

    Args:
        events_path: Path to events.jsonl (source of truth).
        live: When True, add auto-refresh (``perfpilot watch`` view).

    Returns:
        Complete HTML document as a string.
    """
    events = load_run_events(events_path)
    run_id = next(
        (str(e.get("run_id", "")) for e in events if e.get("run_id")),
        Path(events_path).parent.name,
    )
    refresh = "<meta http-equiv='refresh' content='5'>" if live else ""
    body = (
        _verdict_card(events)
        + _timeline(events)
        + _chart(events)
        + _agent_panel(events)
        + _explanation(events)
        + _tune(events)
        + _repro(events)
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>PerfPilot report — {_esc(run_id)}</title>{refresh}"
        f"<style>{CSS}</style>"
        f"<script src='{CHARTJS_CDN}'></script></head>"
        f"<body><h1>PerfPilot report — {_esc(run_id)}</h1>{body}</body></html>"
    )


def render_watch_page(events_path: str | Path, refresh_seconds: int = 5) -> str:
    """Same report page, auto-refreshing every ``refresh_seconds``."""
    html_doc = render_report(events_path, live=True)
    return html_doc.replace("content='5'", f"content='{int(refresh_seconds)}'")


def write_report(events_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Render and write report.html next to events.jsonl (or to out_path)."""
    out = Path(out_path) if out_path else Path(events_path).parent / "report.html"
    out.write_text(render_report(events_path))
    return out
