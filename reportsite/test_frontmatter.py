"""Tests for reportsite.frontmatter — parse_frontmatter."""

from __future__ import annotations

from pathlib import Path

import pytest

from reportsite.frontmatter import Frontmatter, FrontmatterError, parse_frontmatter


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "report.md"
    p.write_text(content, encoding="utf-8")
    return p


VALID_MD = """\
---
round: 3
component: portfolio
pr: 42
date: "2026-05-31"
metric: "portfolio p50_ms"
verdict: accepted
headline_delta: "-50x p50_ms"
---

## What it addressed

Some body text.
"""


def test_parse_valid(tmp_path: Path) -> None:
    p = _write(tmp_path, VALID_MD)
    fm, body = parse_frontmatter(p)
    assert fm == Frontmatter(
        round=3,
        component="portfolio",
        pr=42,
        date="2026-05-31",
        metric="portfolio p50_ms",
        verdict="accepted",
        headline_delta="-50x p50_ms",
    )
    assert "## What it addressed" in body
    assert "Some body text." in body


def test_missing_opening_delimiter(tmp_path: Path) -> None:
    p = _write(tmp_path, "# No frontmatter\n\nBody.\n")
    with pytest.raises(FrontmatterError, match="missing opening"):
        parse_frontmatter(p)


def test_unclosed_frontmatter(tmp_path: Path) -> None:
    p = _write(tmp_path, "---\nround: 1\ncomponent: etl\n")
    with pytest.raises(FrontmatterError, match="never closed"):
        parse_frontmatter(p)


def test_missing_required_field(tmp_path: Path) -> None:
    md = "---\nround: 1\ncomponent: etl\npr: 5\ndate: 2026-01-01\nmetric: etl p50\n---\n"
    p = _write(tmp_path, md)
    with pytest.raises(FrontmatterError, match="missing required"):
        parse_frontmatter(p)


def test_bad_round_type(tmp_path: Path) -> None:
    md = VALID_MD.replace("round: 3", "round: abc")
    p = _write(tmp_path, md)
    with pytest.raises(FrontmatterError, match="'round' must be an integer"):
        parse_frontmatter(p)


def test_bad_verdict(tmp_path: Path) -> None:
    md = VALID_MD.replace("verdict: accepted", "verdict: maybe")
    p = _write(tmp_path, md)
    with pytest.raises(FrontmatterError, match="'verdict' must be one of"):
        parse_frontmatter(p)


def test_verdict_rejected(tmp_path: Path) -> None:
    md = VALID_MD.replace("verdict: accepted", "verdict: rejected")
    p = _write(tmp_path, md)
    fm, _ = parse_frontmatter(p)
    assert fm.verdict == "rejected"


def test_verdict_pending(tmp_path: Path) -> None:
    md = VALID_MD.replace("verdict: accepted", "verdict: pending")
    p = _write(tmp_path, md)
    fm, _ = parse_frontmatter(p)
    assert fm.verdict == "pending"


def test_unquoted_values(tmp_path: Path) -> None:
    md = """\
---
round: 1
component: signals
pr: 7
date: 2026-05-30
metric: signals p50_ms
verdict: accepted
headline_delta: -36% p50_ms
---
body
"""
    p = _write(tmp_path, md)
    fm, body = parse_frontmatter(p)
    assert fm.component == "signals"
    assert fm.date == "2026-05-30"
    assert "body" in body
