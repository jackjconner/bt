"""Tests for ledgers.py — IMPROVEMENTS.md / API_REQUESTS.md parsing."""

from __future__ import annotations

from pathlib import Path

from oversight.ledgers import (
    parse_improvements,
    parse_requests,
    read_improvements,
    read_requests,
)

# The empty-template heads, including the fenced format spec, as shipped.
_IMPROVEMENTS_TEMPLATE = """# Improvements log

Append-only record of every component-improvement round.

Format (one block per round, never edited after writing):

```
## <date> — <component>: <one-line target>  [accepted | rejected]
metric:   <name> — <before> → <after> (<delta>)
eval:     golden unchanged within <tol>
PR:       <url or #number>
note:     <why accepted, or which gate rejected it>
```

---
"""

_REQUESTS_TEMPLATE = """# API requests ledger

Format (append-only):

```
## <date> — <requester> needs <field/data> from <producer>
why:    <one line>
status: open | accepted | done
```

---
"""

_IMPROVEMENTS_POPULATED = (
    _IMPROVEMENTS_TEMPLATE
    + """
## 2026-05-20 — portfolio: Cache factor-covariance Cholesky  [accepted]
metric:   elapsed_s p50 — 1.84s → 0.71s (-61%)
eval:     golden unchanged within 1e-6
PR:       #42
note:     pure speed, no number moved; revert is one merge-commit

## 2026-05-22 — signals: Vectorize cross-sectional IC  [rejected]
metric:   elapsed_s p50 — 0.30s → 0.19s (-38%)
eval:     accuracy moved: ic_neutralized 0.041 → 0.044 — past tolerance
PR:       #43
note:     gate 5 caught a silent regression; don't re-attempt
"""
)

_REQUESTS_POPULATED = (
    _REQUESTS_TEMPLATE
    + """
## 2026-05-18 — analysis needs per-fold timestamps from models
why:    align attribution to the walk-forward split
status: open

## 2026-05-20 — portfolio needs combined-score column from signals
why:    blend into the optimizer's alpha vector
status: done
"""
)


def test_empty_template_improvements_parses_to_nothing() -> None:
    assert parse_improvements(_IMPROVEMENTS_TEMPLATE) == []


def test_empty_template_requests_parses_to_nothing() -> None:
    assert parse_requests(_REQUESTS_TEMPLATE) == []


def test_populated_improvements_count() -> None:
    entries = parse_improvements(_IMPROVEMENTS_POPULATED)
    assert len(entries) == 2


def test_populated_improvements_accepted_entry() -> None:
    entries = parse_improvements(_IMPROVEMENTS_POPULATED)
    acc = entries[0]
    assert acc.date == "2026-05-20"
    assert acc.component == "portfolio"
    assert acc.target == "Cache factor-covariance Cholesky"
    assert acc.verdict == "accepted"
    assert acc.metric == "elapsed_s p50 — 1.84s → 0.71s (-61%)"
    assert acc.pr == "#42"
    assert "no number moved" in acc.note


def test_populated_improvements_rejected_verdict() -> None:
    entries = parse_improvements(_IMPROVEMENTS_POPULATED)
    assert entries[1].verdict == "rejected"
    assert entries[1].component == "signals"


def test_populated_requests_flow_and_status() -> None:
    entries = parse_requests(_REQUESTS_POPULATED)
    assert len(entries) == 2
    first = entries[0]
    assert first.requester == "analysis"
    assert first.producer == "models"
    assert first.field == "per-fold timestamps"
    assert first.status == "open"
    assert entries[1].status == "done"


def test_read_helpers_missing_files(tmp_path: Path) -> None:
    assert read_improvements(tmp_path / "nope.md") == []
    assert read_requests(tmp_path / "nope.md") == []


def test_read_helpers_roundtrip(tmp_path: Path) -> None:
    imp = tmp_path / "IMPROVEMENTS.md"
    req = tmp_path / "API_REQUESTS.md"
    imp.write_text(_IMPROVEMENTS_POPULATED)
    req.write_text(_REQUESTS_POPULATED)
    assert len(read_improvements(imp)) == 2
    assert len(read_requests(req)) == 2
