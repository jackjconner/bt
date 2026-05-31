"""Tests for gitgh.py — branch/worktree derivation + PR-body parse (no network)."""

from __future__ import annotations

from pathlib import Path

from oversight.gitgh import (
    improve_branches,
    open_pull_requests,
    parse_pr_body,
    worktree_dirs,
)

_PR_BODY = """# PR writeup template

## Target

- **Component:** portfolio
- **What I changed:** Cache factor-covariance Cholesky across rebalances.
- **Declared metric this round optimizes:** portfolio `elapsed_s` p50 @ n_assets=200
- **Eval tolerance for the round:** PipelineSummary fields equal to 1e-6 relative

## Gates (quote the actual output)

### Lint & types
```
$ uv run ruff check && uv run ruff format --check && uv run ty check
All checks passed!
```

### Unit tests
```
$ uv run pytest -q portfolio/
458 passed in 12.3s
```

### Profiling — before → after
```
$ uv run main.py
stage              before      after     delta
portfolio @ n=100  0.46s       0.22s     -52%
portfolio @ n=200  1.84s       0.71s     -61%
```
check_regressions vs the ratcheted baseline: no other stage regressed.

### Correctness / accuracy
Pure refactor: eval golden unchanged within tolerance.
"""


def test_parse_pr_body_component_and_metric() -> None:
    component, metric, tol, _gates, _rows = parse_pr_body(_PR_BODY)
    assert component == "portfolio"
    assert "elapsed_s" in metric
    assert "1e-6" in tol


def test_parse_pr_body_gates_pass() -> None:
    _component, _metric, _tol, gates, _rows = parse_pr_body(_PR_BODY)
    assert gates["lint"] is True
    assert gates["types"] is True
    assert gates["correctness"] is True
    assert gates["profiling"] is True
    assert gates["evaluation"] is True


def test_parse_pr_body_profiling_rows() -> None:
    _c, _m, _t, _g, rows = parse_pr_body(_PR_BODY)
    stages = {r.stage for r in rows}
    assert "portfolio @ n=100" in stages
    assert "portfolio @ n=200" in stages
    n200 = next(r for r in rows if r.stage == "portfolio @ n=200")
    assert n200.before == "1.84s"
    assert n200.after == "0.71s"
    assert n200.delta == "-61%"


def test_parse_pr_body_eval_moved_flags_fail() -> None:
    moved = _PR_BODY.replace(
        "Pure refactor: eval golden unchanged within tolerance.",
        "ic_neutralized moved 0.041 → 0.044, past the tolerance.",
    )
    _c, _m, _t, gates, _r = parse_pr_body(moved)
    assert gates["evaluation"] is False


def test_parse_empty_body_is_empty() -> None:
    component, metric, _tol, _gates, rows = parse_pr_body("nothing here")
    assert component == ""
    assert metric == ""
    assert rows == ()


def test_improve_branches_empty_outside_repo(tmp_path: Path) -> None:
    # tmp_path is not a git repo → graceful empty, no crash.
    assert improve_branches(tmp_path) == []


def test_worktree_dirs_reads_directory_names(tmp_path: Path) -> None:
    (tmp_path / ".worktrees" / "portfolio-chol-cache").mkdir(parents=True)
    (tmp_path / ".worktrees" / "signals-ic-vectorize").mkdir(parents=True)
    dirs = worktree_dirs(tmp_path)
    names = {d.name for d in dirs}
    assert names == {"portfolio-chol-cache", "signals-ic-vectorize"}
    comps = {d.component for d in dirs}
    assert "portfolio" in comps
    assert "signals" in comps


def test_worktree_dirs_empty_when_absent(tmp_path: Path) -> None:
    assert worktree_dirs(tmp_path) == []


def test_open_pull_requests_degrades_when_gh_unavailable(tmp_path: Path) -> None:
    # No gh / not a repo context that returns PRs → empty list, no crash.
    assert open_pull_requests(tmp_path) == []
