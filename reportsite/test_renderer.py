"""Tests for reportsite.renderer — render_index and render_report_page."""

from __future__ import annotations

from pathlib import Path

from reportsite.model import build_index
from reportsite.renderer import render_index, render_report_page, render_site


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

## How it decided

Before calltree:

```
scipy.optimize.minimize  92%
```

## Pre/post profile

| metric | before | after |
|--------|--------|-------|
| p50_ms | 457 ms | 8 ms  |

![flame graph](assets/portfolio-flamegraph.html)

## System impact

Eval unchanged.

## Suggested next steps

1. ledoit_wolf_cov next.
"""


def _fixture_reports_dir(tmp_path: Path) -> Path:
    """Create a minimal fixture reports directory."""
    r0 = tmp_path / "round-000"
    r0.mkdir()
    _write_report(r0, "portfolio.md", _PORTFOLIO_MD)
    assets = r0 / "assets"
    assets.mkdir()
    (assets / "portfolio-flamegraph.html").write_text("<html><body>flame</body></html>")
    return tmp_path


def test_render_index_empty_state() -> None:
    from reportsite.model import ReportIndex

    index = ReportIndex(rounds=())
    html = render_index(index)
    assert "Awaiting the first swarm" in html
    assert "reports/README.md" in html


def test_render_index_has_round_and_component(tmp_path: Path) -> None:
    reports_dir = _fixture_reports_dir(tmp_path)
    index = build_index(reports_dir)
    html = render_index(index)
    assert "round 000" in html
    assert "portfolio" in html
    assert "-50x p50_ms" in html
    assert "accepted" in html


def test_render_report_page_has_section_headers(tmp_path: Path) -> None:
    reports_dir = _fixture_reports_dir(tmp_path)
    index = build_index(reports_dir)
    report = index.rounds[0].reports[0]
    html = render_report_page(report)
    # The markdown body is embedded as a JS template literal — check its presence.
    assert "What it addressed" in html
    assert "How it decided" in html
    assert "Pre/post profile" in html
    assert "System impact" in html
    assert "Suggested next steps" in html


def test_render_report_page_embeds_flamegraph(tmp_path: Path) -> None:
    reports_dir = _fixture_reports_dir(tmp_path)
    index = build_index(reports_dir)
    report = index.rounds[0].reports[0]
    html = render_report_page(report)
    # The .html asset reference should become an <iframe>.
    assert "<iframe" in html
    assert "flamegraph-embed" in html
    assert "portfolio-flamegraph.html" in html


def test_render_report_page_has_metadata(tmp_path: Path) -> None:
    reports_dir = _fixture_reports_dir(tmp_path)
    index = build_index(reports_dir)
    report = index.rounds[0].reports[0]
    html = render_report_page(report)
    assert "PR #3" in html
    assert "2026-05-31" in html
    assert "portfolio p50_ms" in html


def test_render_site_creates_files(tmp_path: Path) -> None:
    reports_dir = _fixture_reports_dir(tmp_path)
    index = build_index(reports_dir)
    render_site(reports_dir, index)

    site_dir = reports_dir / "_site"
    assert (site_dir / "index.html").exists()
    assert (site_dir / "round-000" / "portfolio.html").exists()
    assert (site_dir / "round-000" / "assets" / "portfolio-flamegraph.html").exists()


def test_render_site_empty_creates_index(tmp_path: Path) -> None:
    from reportsite.model import ReportIndex

    render_site(tmp_path, ReportIndex(rounds=()))
    assert (tmp_path / "_site" / "index.html").exists()
    html = (tmp_path / "_site" / "index.html").read_text()
    assert "Awaiting the first swarm" in html


def test_render_site_writes_build_id(tmp_path: Path) -> None:
    reports_dir = _fixture_reports_dir(tmp_path)
    index = build_index(reports_dir)
    render_site(reports_dir, index)
    build_id = reports_dir / "_site" / "build-id.txt"
    assert build_id.exists()
    assert build_id.read_text().strip()  # non-empty content hash


def test_build_id_changes_when_content_changes(tmp_path: Path) -> None:
    reports_dir = _fixture_reports_dir(tmp_path)
    render_site(reports_dir, build_index(reports_dir))
    first = (reports_dir / "_site" / "build-id.txt").read_text()

    # Mutate a report body → the rendered site changes → build-id must change.
    (reports_dir / "round-000" / "portfolio.md").write_text(
        _PORTFOLIO_MD + "\n## Extra\n\nNew content.\n", encoding="utf-8"
    )
    render_site(reports_dir, build_index(reports_dir))
    second = (reports_dir / "_site" / "build-id.txt").read_text()
    assert first != second


def test_build_id_stable_when_content_unchanged(tmp_path: Path) -> None:
    reports_dir = _fixture_reports_dir(tmp_path)
    render_site(reports_dir, build_index(reports_dir))
    first = (reports_dir / "_site" / "build-id.txt").read_text()
    render_site(reports_dir, build_index(reports_dir))
    second = (reports_dir / "_site" / "build-id.txt").read_text()
    assert first == second  # idempotent — no spurious reloads


def test_pages_inject_livereload_poll(tmp_path: Path) -> None:
    reports_dir = _fixture_reports_dir(tmp_path)
    index = build_index(reports_dir)

    # Index page polls build-id.txt at the site root.
    idx = render_index(index)
    assert "build-id.txt" in idx
    assert "location.reload()" in idx

    # Report page is one level down → polls ../build-id.txt.
    report = index.rounds[0].reports[0]
    page = render_report_page(report)
    assert "../build-id.txt" in page
    assert "location.reload()" in page
