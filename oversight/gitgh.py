"""Read-only live derivation from git + ``gh`` — data source #1.

Pure functions returning frozen dataclasses. Everything degrades gracefully:
if ``gh`` is unauthenticated, offline, or absent, the PR readers catch the
failure and return empty lists rather than crashing the TUI. Git readers fall
back to empty when run outside a repo.

The PR-body parser is best-effort over the ``pr-writeup.md`` sections — it
pulls the declared component, the gate verdicts, and the profiling before→after
rows when present, and shrugs (empty fields) when the body doesn't follow the
template.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_GATE_KEYS = ("lint", "types", "correctness", "profiling", "evaluation")


@dataclass(frozen=True)
class ImproveBranch:
    name: str
    component: str  # parsed from improve/<component>-<slug>
    slug: str


@dataclass(frozen=True)
class WorktreeDir:
    name: str
    component: str


@dataclass(frozen=True)
class WriteupProfRow:
    stage: str
    before: str
    after: str
    delta: str


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    branch: str
    state: str
    component: str = ""
    declared_metric: str = ""
    eval_tolerance: str = ""
    gates: dict[str, bool] = field(default_factory=dict)
    prof_rows: tuple[WriteupProfRow, ...] = ()
    additions: int = 0
    deletions: int = 0


# ---------------------------------------------------------------------------
# Subprocess helpers — never raise out of this module
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout


def _component_of(branch_or_dir: str) -> tuple[str, str]:
    """``improve/portfolio-chol-cache`` → ('portfolio', 'chol-cache')."""
    tail = branch_or_dir.split("/", 1)[-1]
    tail = tail.removeprefix("improve-")  # worktree dirs use a flattened slug
    if "-" in tail:
        component, slug = tail.split("-", 1)
        return component, slug
    return tail, ""


# ---------------------------------------------------------------------------
# Git — branches + worktrees
# ---------------------------------------------------------------------------


def improve_branches(root: Path) -> list[ImproveBranch]:
    out = _run(["git", "branch", "--list", "improve/*", "--format=%(refname:short)"], root)
    if out is None:
        return []
    branches: list[ImproveBranch] = []
    for line in out.splitlines():
        name = line.strip()
        if not name:
            continue
        component, slug = _component_of(name)
        branches.append(ImproveBranch(name=name, component=component, slug=slug))
    return branches


def worktree_dirs(root: Path) -> list[WorktreeDir]:
    wt_root = root / ".worktrees"
    if not wt_root.is_dir():
        return []
    dirs: list[WorktreeDir] = []
    for entry in sorted(wt_root.iterdir()):
        if not entry.is_dir():
            continue
        component, _ = _component_of(entry.name)
        dirs.append(WorktreeDir(name=entry.name, component=component))
    return dirs


# ---------------------------------------------------------------------------
# PR-body (pr-writeup.md) parser — best-effort
# ---------------------------------------------------------------------------

_COMPONENT_RE = re.compile(r"\*\*Component:\*\*\s*(?:<)?([a-z]+)", re.IGNORECASE)
_METRIC_RE = re.compile(r"\*\*Declared metric[^:]*:\*\*\s*(.+)")
_TOL_RE = re.compile(r"\*\*Eval tolerance[^:]*:\*\*\s*(.+)")
# a profiling row: "<stage>  <before>  <after>  <delta>" with a percentage delta
_PROF_ROW_RE = re.compile(
    r"^(?P<stage>\S.*?)\s{2,}(?P<before>[\d.]+\S*)\s+(?P<after>[\d.]+\S*)\s+"
    r"(?P<delta>[-+]?\d+%|[-+−]\d+\S*)\s*$"
)


def _gate_verdict(body: str, gate: str) -> bool | None:
    """A gate counts as passed if its section quotes a clean signal."""
    lowered = body.lower()
    if gate == "lint" or gate == "types":
        return "all checks passed" in lowered
    if gate == "correctness":
        return bool(re.search(r"\d+\s+passed", lowered)) and "failed" not in lowered.replace(
            "0 failed", ""
        )
    if gate == "profiling":
        return "no other stage regressed" in lowered or "check_regressions" in lowered
    if gate == "evaluation":
        if "golden unchanged" in lowered or "within tolerance" in lowered:
            return True
        if "moved" in lowered and "tolerance" in lowered:
            return False
        return None
    return None


def parse_pr_body(body: str) -> tuple[str, str, str, dict[str, bool], tuple[WriteupProfRow, ...]]:
    """Return (component, declared_metric, eval_tolerance, gates, prof_rows)."""
    component = ""
    cm = _COMPONENT_RE.search(body)
    if cm is not None:
        component = cm.group(1).lower()

    metric = ""
    mm = _METRIC_RE.search(body)
    if mm is not None:
        metric = mm.group(1).strip().strip("`")

    tolerance = ""
    tm = _TOL_RE.search(body)
    if tm is not None:
        tolerance = tm.group(1).strip()

    gates: dict[str, bool] = {}
    for gate in _GATE_KEYS:
        verdict = _gate_verdict(body, gate)
        if verdict is not None:
            gates[gate] = verdict

    rows: list[WriteupProfRow] = []
    for line in body.splitlines():
        rm = _PROF_ROW_RE.match(line.strip())
        if rm is None:
            continue
        stage = rm.group("stage").strip()
        if stage.lower() in {"before", "stage", "stage · p50"}:
            continue
        rows.append(
            WriteupProfRow(
                stage=stage,
                before=rm.group("before"),
                after=rm.group("after"),
                delta=rm.group("delta"),
            )
        )

    return component, metric, tolerance, gates, tuple(rows)


# ---------------------------------------------------------------------------
# gh — pull requests
# ---------------------------------------------------------------------------


def _as_str_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {str(k): v for k, v in value.items()}


def _gh_json(args: list[str], root: Path) -> object | None:
    out = _run(["gh", *args], root)
    if out is None:
        return None
    stripped = out.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def open_pull_requests(root: Path) -> list[PullRequest]:
    """List open PRs and enrich each with its parsed writeup body.

    Returns an empty list (not an error) when ``gh`` is offline/unauth'd.
    """
    listing = _gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--json",
            "number,title,headRefName,state,additions,deletions",
        ],
        root,
    )
    if not isinstance(listing, list):
        return []

    prs: list[PullRequest] = []
    for item in listing:
        raw = _as_str_dict(item)
        if raw is None:
            continue
        number = raw.get("number")
        if not isinstance(number, int):
            continue
        branch = str(raw.get("headRefName", ""))
        component, _ = _component_of(branch) if branch.startswith("improve/") else ("", "")

        body = _pr_body(number, root)
        if body:
            comp2, metric, tol, gates, rows = parse_pr_body(body)
            component = comp2 or component
        else:
            metric, tol, gates, rows = "", "", {}, ()

        additions = raw.get("additions")
        deletions = raw.get("deletions")
        prs.append(
            PullRequest(
                number=number,
                title=str(raw.get("title", "")),
                branch=branch,
                state=str(raw.get("state", "")),
                component=component,
                declared_metric=metric,
                eval_tolerance=tol,
                gates=gates,
                prof_rows=rows,
                additions=additions if isinstance(additions, int) else 0,
                deletions=deletions if isinstance(deletions, int) else 0,
            )
        )
    return prs


def _pr_body(number: int, root: Path) -> str:
    raw = _as_str_dict(_gh_json(["pr", "view", str(number), "--json", "body"], root))
    if raw is not None:
        body = raw.get("body")
        if isinstance(body, str):
            return body
    return ""


__all__ = [
    "ImproveBranch",
    "PullRequest",
    "WorktreeDir",
    "WriteupProfRow",
    "improve_branches",
    "open_pull_requests",
    "parse_pr_body",
    "worktree_dirs",
]
