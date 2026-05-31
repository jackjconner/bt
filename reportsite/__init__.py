"""reportsite — static HTML site generator for bt change reports.

Reads ``reports/round-*/<component>.md`` files, parses their YAML frontmatter,
and renders a browsable static site to ``reports/_site/``.
"""

from .frontmatter import FrontmatterError, parse_frontmatter
from .model import Report, ReportIndex, build_index
from .renderer import render_site

__all__ = [
    "FrontmatterError",
    "Report",
    "ReportIndex",
    "build_index",
    "parse_frontmatter",
    "render_site",
]
