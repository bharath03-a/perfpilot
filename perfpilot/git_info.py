"""Commit ranges via the GitHub API. No GPU, no secrets in code."""

from __future__ import annotations

import json
import os
import urllib.request

from pydantic import BaseModel


class CommitInfo(BaseModel):
    sha: str
    message: str


def _api(url: str) -> object:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        return json.load(resp)


def recent_main_commits(repo: str, n: int) -> list[CommitInfo]:
    """Newest-first main commits (single branch head page walk)."""
    out: list[CommitInfo] = []
    page = 1
    while len(out) < n:
        data = _api(
            f"https://api.github.com/repos/{repo}/commits"
            f"?sha=main&per_page=100&page={page}"
        )
        if not isinstance(data, list) or not data:
            break
        for c in data:
            sha = c.get("sha", "")
            msg = (c.get("commit", {}).get("message", "") or "").splitlines()
            if len(sha) == 40:
                out.append(CommitInfo(sha=sha, message=msg[0] if msg else ""))
            if len(out) >= n:
                break
        page += 1
    return out[:n]


def newest_wheeled_sha() -> str:
    """Full SHA of the latest nightly build (its wheels exist).

    Newer main commits have no wheels yet (build lag); ranges must end here.
    """
    import re

    with urllib.request.urlopen(
        "https://wheels.vllm.ai/nightly/", timeout=30.0
    ) as resp:
        html = resp.read().decode()
    m = re.search(r"commit ([0-9a-f]{40})", html)
    if not m:
        raise RuntimeError("nightly index has no build commit marker")
    return m.group(1)


def commits_between(repo: str, good: str, bad: str) -> list[str]:
    """Oldest-first SHAs from good to bad via the compare API.

    The API caps the returned list (~250 commits); larger ranges need
    pagination the API does not offer — keep bisect ranges small.
    """
    data = _api(f"https://api.github.com/repos/{repo}/compare/{good}...{bad}")
    if not isinstance(data, dict) or data.get("status") not in (
        "ahead",
        "identical",
    ):
        raise RuntimeError(f"compare {good[:12]}...{bad[:12]}: not ahead")
    return [c["sha"] for c in data.get("commits", []) if len(c.get("sha", "")) == 40]


def oldest_first_range(repo: str, n: int, skip_newest: int = 8) -> list[str]:
    """16-commit-style range, oldest first, skipping the newest commits.

    Newest commits lack nightly wheels (build lag), so the range ends
    `skip_newest` behind main. Prefer wheeled_range(), which anchors to the
    newest nightly build instead of a fixed skip.
    """
    commits = recent_main_commits(repo, n + skip_newest)
    return [c.sha for c in reversed(commits[:n])]


def wheeled_range(repo: str, n: int) -> list[str]:
    """n-commit range, oldest first, ending at the newest wheeled commit."""
    end = newest_wheeled_sha()
    commits = recent_main_commits(repo, n + 400)
    shas = [c.sha for c in commits]
    if end not in shas:
        raise RuntimeError(
            f"newest wheeled {end[:12]} not in recent {len(shas)} main commits"
        )
    head = shas.index(end)
    window = list(reversed(shas[head : head + n]))
    if len(window) < n:
        raise RuntimeError("not enough main history above the wheeled commit")
    return window
