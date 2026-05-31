"""Pure, importable functions for serialize / diff / tolerance logic.

All functions are free of pipeline side-effects and can be exercised in unit
tests with synthetic dicts — no ``run_production_pipeline`` call needed.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_REL_TOL: float = 1e-6


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize(obj: Any) -> Any:
    """Recursively convert a dataclass (or nested plain value) to a JSON-safe dict.

    Handles:
    - dataclasses  → dict (recursively)
    - dict         → dict (keys cast to str for JSON)
    - float / int / bool / str / None → identity
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: serialize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(i) for i in obj]
    return obj


def save_golden(data: dict[str, Any], path: Path) -> None:
    """Write *data* to *path* as pretty-printed JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_golden(path: Path) -> dict[str, Any]:
    """Read and return the JSON golden dict from *path*."""
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Diff / tolerance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffRow:
    """Comparison result for a single field."""

    field: str
    golden: Any
    current: Any
    abs_delta: float | None  # None for non-numeric / nested dicts
    rel_delta: float | None  # None for non-numeric / nested dicts
    passed: bool


def _compare_scalars(
    field: str,
    golden: Any,
    current: Any,
    rel_tol: float,
) -> DiffRow:
    """Compare two scalar values with the appropriate tolerance rule.

    - bool / int : exact match
    - float      : relative tolerance (``rel_tol``)
    - anything else : equality
    """
    # Bools are subclasses of int — check bool first.
    if isinstance(golden, bool) or isinstance(current, bool):
        passed = bool(golden) == bool(current)
        return DiffRow(
            field=field,
            golden=golden,
            current=current,
            abs_delta=None,
            rel_delta=None,
            passed=passed,
        )

    if isinstance(golden, int) and isinstance(current, int):
        passed = golden == current
        return DiffRow(
            field=field,
            golden=golden,
            current=current,
            abs_delta=None,
            rel_delta=None,
            passed=passed,
        )

    if isinstance(golden, (int, float)) and isinstance(current, (int, float)):
        g = float(golden)
        c = float(current)
        abs_delta = abs(c - g)
        # Avoid division by zero: if golden is 0 use abs check with same tol.
        denom = abs(g) if g != 0.0 else 1.0
        rel_delta = abs_delta / denom
        # NaN/Inf: both must be in the same IEEE state.
        if not math.isfinite(g) or not math.isfinite(c):
            passed = (math.isnan(g) and math.isnan(c)) or (g == c)
            return DiffRow(
                field=field,
                golden=golden,
                current=current,
                abs_delta=None,
                rel_delta=None,
                passed=passed,
            )
        passed = rel_delta <= rel_tol
        return DiffRow(
            field=field,
            golden=g,
            current=c,
            abs_delta=abs_delta,
            rel_delta=rel_delta,
            passed=passed,
        )

    # Fallback: equality.
    passed = golden == current
    return DiffRow(
        field=field, golden=golden, current=current, abs_delta=None, rel_delta=None, passed=passed
    )


def _compare_horizon_dict(
    field: str,
    golden: dict[str, Any],
    current: dict[str, Any],
    rel_tol: float,
) -> list[DiffRow]:
    """Expand a horizon_ic-style dict into one DiffRow per key."""
    rows: list[DiffRow] = []
    all_keys = sorted(set(golden) | set(current), key=lambda k: int(k) if str(k).isdigit() else k)
    for k in all_keys:
        subfield = f"{field}[{k}]"
        if k not in golden:
            rows.append(
                DiffRow(
                    field=subfield,
                    golden=None,
                    current=current[k],
                    abs_delta=None,
                    rel_delta=None,
                    passed=False,
                )
            )
        elif k not in current:
            rows.append(
                DiffRow(
                    field=subfield,
                    golden=golden[k],
                    current=None,
                    abs_delta=None,
                    rel_delta=None,
                    passed=False,
                )
            )
        else:
            rows.append(_compare_scalars(subfield, golden[k], current[k], rel_tol))
    return rows


def diff_summaries(
    golden: dict[str, Any],
    current: dict[str, Any],
    rel_tol: float = _DEFAULT_REL_TOL,
    field_tol: dict[str, float] | None = None,
) -> list[DiffRow]:
    """Return one ``DiffRow`` per top-level field (dict fields are expanded per-key).

    Fields present in one dict but not the other produce a FAIL row.

    ``field_tol`` overrides the relative tolerance for specific fields
    (e.g. ``{"backtest_p50_s": 0.5}`` to allow ±50 % on a timing field).
    Keys must match top-level field names; sub-keys of nested dicts
    (e.g. ``horizon_ic[1]``) are not supported via this map.
    """
    _ftol = field_tol or {}
    rows: list[DiffRow] = []
    all_fields = list(golden) + [k for k in current if k not in golden]
    seen: set[str] = set()

    for field in all_fields:
        if field in seen:
            continue
        seen.add(field)

        if field not in golden:
            rows.append(
                DiffRow(
                    field=field,
                    golden=None,
                    current=current[field],
                    abs_delta=None,
                    rel_delta=None,
                    passed=False,
                )
            )
            continue
        if field not in current:
            rows.append(
                DiffRow(
                    field=field,
                    golden=golden[field],
                    current=None,
                    abs_delta=None,
                    rel_delta=None,
                    passed=False,
                )
            )
            continue

        g_val = golden[field]
        c_val = current[field]
        tol = _ftol.get(field, rel_tol)

        # Nested dict (e.g. horizon_ic) — expand per key.
        if isinstance(g_val, dict) or isinstance(c_val, dict):
            g_dict = g_val if isinstance(g_val, dict) else {}
            c_dict = c_val if isinstance(c_val, dict) else {}
            rows.extend(_compare_horizon_dict(field, g_dict, c_dict, tol))
        else:
            rows.append(_compare_scalars(field, g_val, c_val, tol))

    return rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_COL_FIELD = 28
_COL_GOLDEN = 18
_COL_CURRENT = 18
_COL_ABS = 14
_COL_REL = 12
_COL_STATUS = 6


def format_diff_table(rows: list[DiffRow]) -> str:
    """Render *rows* as a human-readable fixed-width table string."""

    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:+.6g}"
        return str(v)

    def _fmt_delta(v: float | None) -> str:
        if v is None:
            return "—"
        return f"{v:.2e}"

    header = (
        f"{'field':<{_COL_FIELD}}"
        f"{'golden':>{_COL_GOLDEN}}"
        f"{'current':>{_COL_CURRENT}}"
        f"{'abs_delta':>{_COL_ABS}}"
        f"{'rel_delta':>{_COL_REL}}"
        f"  status"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for row in rows:
        status = "PASS" if row.passed else "FAIL"
        lines.append(
            f"{row.field:<{_COL_FIELD}}"
            f"{_fmt(row.golden):>{_COL_GOLDEN}}"
            f"{_fmt(row.current):>{_COL_CURRENT}}"
            f"{_fmt_delta(row.abs_delta):>{_COL_ABS}}"
            f"{_fmt_delta(row.rel_delta):>{_COL_REL}}"
            f"  {status}"
        )

    lines.append(sep)
    n_fail = sum(1 for r in rows if not r.passed)
    n_pass = len(rows) - n_fail
    lines.append(f"  {n_pass} PASS  {n_fail} FAIL  (total {len(rows)} fields)")
    return "\n".join(lines)
