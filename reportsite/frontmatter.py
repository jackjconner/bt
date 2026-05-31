"""Parse YAML frontmatter from a markdown file.

Frontmatter is the YAML block delimited by ``---`` lines at the very start of
the file.  We implement a minimal parser that covers the six scalar fields used
by change reports — no PyYAML dependency.

Expected fields (all required):
    round: int
    component: str
    pr: int
    date: str   (ISO-8601)
    metric: str
    verdict: str  (accepted | rejected | pending)
    headline_delta: str
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Recognised verdict values. ``pending`` is a not-yet-merged change under review
# (the report is written when the PR is adjudicated; the verdict flips to
# ``accepted`` on merge or ``rejected`` if dropped).
VERDICTS = frozenset({"accepted", "rejected", "pending"})

# Scalar YAML value: bare string, quoted string, or integer.
_SCALAR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.+)$")


class FrontmatterError(ValueError):
    """Raised when frontmatter is missing, malformed, or has missing/bad fields."""


@dataclass(frozen=True)
class Frontmatter:
    """Parsed frontmatter from a change report."""

    round: int
    component: str
    pr: int
    date: str
    metric: str
    verdict: str
    headline_delta: str


def _strip_quotes(value: str) -> str:
    """Remove surrounding single or double quotes from a YAML scalar."""
    value = value.strip()
    if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
        return value[1:-1]
    return value


def _parse_yaml_block(lines: list[str]) -> dict[str, str]:
    """Parse a minimal flat YAML block into a string→string dict.

    Only handles ``key: value`` pairs (no nesting, no lists).  Comments and
    blank lines are ignored.  Raises ``FrontmatterError`` on duplicate keys.
    """
    result: dict[str, str] = {}
    for line in lines:
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        m = _SCALAR_RE.match(line)
        if not m:
            raise FrontmatterError(f"Unrecognised frontmatter line: {line!r}")
        key, raw_value = m.group(1), m.group(2).strip()
        if key in result:
            raise FrontmatterError(f"Duplicate frontmatter key: {key!r}")
        result[key] = _strip_quotes(raw_value)
    return result


def parse_frontmatter(path: Path) -> tuple[Frontmatter, str]:
    """Parse the YAML frontmatter from a report file.

    Returns:
        ``(Frontmatter, body)`` where ``body`` is the markdown text after the
        closing ``---`` delimiter.

    Raises:
        FrontmatterError: The file does not start with ``---``, the block is
            not closed, a required field is missing, or a field has a bad type.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if not lines or lines[0].rstrip() != "---":
        raise FrontmatterError(f"{path}: missing opening '---' frontmatter delimiter")

    # Find the closing '---'.
    close_idx: int | None = None
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip() == "---":
            close_idx = i
            break
    if close_idx is None:
        raise FrontmatterError(f"{path}: frontmatter block is never closed with '---'")

    yaml_lines = lines[1:close_idx]
    body = "\n".join(lines[close_idx + 1 :]).lstrip("\n")

    raw = _parse_yaml_block(yaml_lines)

    required = ("round", "component", "pr", "date", "metric", "verdict", "headline_delta")
    missing = [k for k in required if k not in raw]
    if missing:
        raise FrontmatterError(f"{path}: missing required frontmatter fields: {missing}")

    # Type coercions.
    try:
        round_num = int(raw["round"])
    except ValueError as exc:
        raise FrontmatterError(f"{path}: 'round' must be an integer, got {raw['round']!r}") from exc

    try:
        pr_num = int(raw["pr"])
    except ValueError as exc:
        raise FrontmatterError(f"{path}: 'pr' must be an integer, got {raw['pr']!r}") from exc

    verdict = raw["verdict"]
    if verdict not in VERDICTS:
        raise FrontmatterError(
            f"{path}: 'verdict' must be one of {sorted(VERDICTS)}, got {verdict!r}"
        )

    return (
        Frontmatter(
            round=round_num,
            component=raw["component"],
            pr=pr_num,
            date=raw["date"],
            metric=raw["metric"],
            verdict=verdict,
            headline_delta=raw["headline_delta"],
        ),
        body,
    )
