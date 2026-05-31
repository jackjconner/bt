"""Index model for the report site.

``Report`` is one change report; ``ReportIndex`` groups them by round, newest
first.  Both are frozen dataclasses so the renderer cannot mutate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .frontmatter import Frontmatter, parse_frontmatter


@dataclass(frozen=True)
class AssetRef:
    """A single asset file associated with a report."""

    name: str  # filename only (no directory component)
    path: Path  # absolute path to the asset file


@dataclass(frozen=True)
class Report:
    """One change report: frontmatter + body markdown + asset paths."""

    frontmatter: Frontmatter
    body: str  # markdown body (everything after the closing ---)
    source_path: Path  # absolute path to the .md file
    assets: tuple[AssetRef, ...]  # assets under the report's assets/ directory


@dataclass(frozen=True)
class RoundGroup:
    """All reports that landed in a given round."""

    round: int
    reports: tuple[Report, ...]


@dataclass(frozen=True)
class ReportIndex:
    """The full index: rounds in descending order (newest first)."""

    rounds: tuple[RoundGroup, ...]

    @property
    def total_reports(self) -> int:
        return sum(len(g.reports) for g in self.rounds)

    @property
    def is_empty(self) -> bool:
        return self.total_reports == 0


def _collect_assets(report_dir: Path) -> tuple[AssetRef, ...]:
    """Return all files under ``<report_dir>/assets/``, sorted by name."""
    assets_dir = report_dir / "assets"
    if not assets_dir.is_dir():
        return ()
    return tuple(AssetRef(name=f.name, path=f) for f in sorted(assets_dir.iterdir()) if f.is_file())


def build_index(reports_dir: Path) -> ReportIndex:
    """Scan ``reports_dir`` for ``round-*/*.md`` files and build the index.

    Skips ``_template.md``, ``README.md``, and any file whose name starts with
    ``_``.  Rounds are sorted descending (newest round number first).  Reports
    within a round are sorted by component name.

    Raises:
        FrontmatterError: Any report has invalid or missing frontmatter.
    """
    round_dirs = sorted(
        (d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("round-")),
        key=lambda d: d.name,
        reverse=True,  # newest round first
    )

    groups: list[RoundGroup] = []
    for round_dir in round_dirs:
        _SKIP = frozenset({"README.md"})
        md_files = sorted(
            f
            for f in round_dir.iterdir()
            if f.is_file()
            and f.suffix == ".md"
            and not f.name.startswith("_")
            and f.name not in _SKIP
        )
        if not md_files:
            continue

        reports: list[Report] = []
        for md_path in md_files:
            fm, body = parse_frontmatter(md_path)
            assets = _collect_assets(round_dir)
            reports.append(Report(frontmatter=fm, body=body, source_path=md_path, assets=assets))

        if reports:
            round_num = reports[0].frontmatter.round
            groups.append(RoundGroup(round=round_num, reports=tuple(reports)))

    return ReportIndex(rounds=tuple(groups))
