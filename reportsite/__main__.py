"""CLI entry point for the report site generator.

Usage::

    uv run python -m reportsite [--reports-dir PATH]

Reads ``reports/round-*/*.md``, builds the index, and renders ``reports/_site/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import build_index
from .renderer import render_site


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m reportsite",
        description="Generate the bt change-report static site.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        metavar="PATH",
        help="Root of the reports tree (default: reports/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reports_dir = Path(args.reports_dir).resolve()

    if not reports_dir.is_dir():
        print(f"reportsite: reports directory not found: {reports_dir}", file=sys.stderr)
        return 1

    index = build_index(reports_dir)
    render_site(reports_dir, index)

    site_dir = reports_dir / "_site"
    print(f"reportsite: built {index.total_reports} reports → {site_dir}/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
