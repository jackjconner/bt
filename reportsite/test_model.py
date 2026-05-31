"""Tests for reportsite.model — build_index."""

from __future__ import annotations

from pathlib import Path

import pytest

from reportsite.frontmatter import FrontmatterError
from reportsite.model import build_index


def _write_report(directory: Path, filename: str, content: str) -> Path:
    p = directory / filename
    p.write_text(content, encoding="utf-8")
    return p


_PORTFOLIO_MD = """\
---
round: 0
component: portfolio
pr: 3
date: "2026-05-31"
metric: "portfolio p50_ms"
verdict: accepted
headline_delta: "-50x p50_ms"
---

## What it addressed

Replaced SLSQP with OSQP.
"""

_SIGNALS_MD = """\
---
round: 1
component: signals
pr: 17
date: "2026-05-31"
metric: "signals p50_ms"
verdict: accepted
headline_delta: "-36% p50_ms"
---

## What it addressed

Vectorized cross-sectional IC.
"""


def test_empty_reports_dir(tmp_path: Path) -> None:
    index = build_index(tmp_path)
    assert index.is_empty
    assert index.total_reports == 0
    assert index.rounds == ()


def test_single_round_single_report(tmp_path: Path) -> None:
    r0 = tmp_path / "round-000"
    r0.mkdir()
    _write_report(r0, "portfolio.md", _PORTFOLIO_MD)

    index = build_index(tmp_path)
    assert not index.is_empty
    assert index.total_reports == 1
    assert len(index.rounds) == 1

    group = index.rounds[0]
    assert group.round == 0
    assert len(group.reports) == 1
    assert group.reports[0].frontmatter.component == "portfolio"


def test_multiple_rounds_newest_first(tmp_path: Path) -> None:
    r0 = tmp_path / "round-000"
    r0.mkdir()
    _write_report(r0, "portfolio.md", _PORTFOLIO_MD)

    r1 = tmp_path / "round-001"
    r1.mkdir()
    _write_report(r1, "signals.md", _SIGNALS_MD)

    index = build_index(tmp_path)
    assert index.total_reports == 2
    assert len(index.rounds) == 2
    # Newest (round 1) first.
    assert index.rounds[0].round == 1
    assert index.rounds[1].round == 0


def test_template_and_readme_skipped(tmp_path: Path) -> None:
    r0 = tmp_path / "round-000"
    r0.mkdir()
    _write_report(r0, "portfolio.md", _PORTFOLIO_MD)
    # These should be ignored.
    _write_report(r0, "_template.md", "---\nshould: not matter\n---\n")
    _write_report(r0, "README.md", "# readme\n")

    index = build_index(tmp_path)
    assert index.total_reports == 1


def test_assets_collected(tmp_path: Path) -> None:
    r0 = tmp_path / "round-000"
    r0.mkdir()
    _write_report(r0, "portfolio.md", _PORTFOLIO_MD)
    assets_dir = r0 / "assets"
    assets_dir.mkdir()
    (assets_dir / "portfolio-flamegraph.html").write_text("<html></html>", encoding="utf-8")
    (assets_dir / "portfolio-before.cpu.calltree.txt").write_text("calltree", encoding="utf-8")

    index = build_index(tmp_path)
    report = index.rounds[0].reports[0]
    asset_names = {a.name for a in report.assets}
    assert "portfolio-flamegraph.html" in asset_names
    assert "portfolio-before.cpu.calltree.txt" in asset_names


def test_missing_frontmatter_field_raises(tmp_path: Path) -> None:
    r0 = tmp_path / "round-000"
    r0.mkdir()
    bad_md = "---\nround: 0\ncomponent: etl\n---\nbody\n"
    _write_report(r0, "etl.md", bad_md)
    with pytest.raises(FrontmatterError):
        build_index(tmp_path)
