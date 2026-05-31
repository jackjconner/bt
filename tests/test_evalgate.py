"""Unit tests for evalgate pure functions.

No pipeline execution — all tests use synthetic dicts.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from evalgate._core import (
    diff_summaries,
    format_diff_table,
    load_golden,
    save_golden,
    serialize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(
    ic_raw: float = 0.05,
    ic_neutralized: float = 0.04,
    horizon_ic: dict[str, float] | None = None,
    wf_mean_ic: float = 0.03,
    wf_mean_r2: float = 0.01,
    opt_converged: bool = True,
    opt_gross: float = 1.95,
    factor_vol: float = 0.12,
    tracking_error: float = 0.08,
    gross_sharpe: float = 0.55,
    net_sharpe: float = 0.50,
    cost_drag: float = 250.0,
    n_scaling_fits: int = 9,
    backtest_p50_s: float = 0.042,
) -> dict[str, object]:
    """Return a PipelineSummary-shaped dict (no actual dataclass needed)."""
    if horizon_ic is None:
        horizon_ic = {"1": 0.051, "5": 0.040, "21": 0.025, "63": 0.010}
    return {
        "ic_raw": ic_raw,
        "ic_neutralized": ic_neutralized,
        "horizon_ic": horizon_ic,
        "wf_mean_ic": wf_mean_ic,
        "wf_mean_r2": wf_mean_r2,
        "opt_converged": opt_converged,
        "opt_gross": opt_gross,
        "factor_vol": factor_vol,
        "tracking_error": tracking_error,
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "cost_drag": cost_drag,
        "n_scaling_fits": n_scaling_fits,
        "backtest_p50_s": backtest_p50_s,
    }


# ---------------------------------------------------------------------------
# serialize
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Inner:
    x: float
    flag: bool


@dataclasses.dataclass(frozen=True)
class _Outer:
    name: str
    inner: _Inner
    mapping: dict[int, float]


def test_serialize_plain_values() -> None:
    assert serialize(3.14) == 3.14
    assert serialize(True) is True
    assert serialize(42) == 42
    assert serialize("hello") == "hello"
    assert serialize(None) is None


def test_serialize_dataclass() -> None:
    inner = _Inner(x=1.5, flag=False)
    outer = _Outer(name="test", inner=inner, mapping={1: 0.5, 2: 0.3})
    result = serialize(outer)
    assert result == {
        "name": "test",
        "inner": {"x": 1.5, "flag": False},
        "mapping": {"1": 0.5, "2": 0.3},
    }


def test_serialize_dict_keys_stringified() -> None:
    d = {1: "a", 5: "b"}
    result = serialize(d)
    assert result == {"1": "a", "5": "b"}


def test_serialize_list_and_tuple() -> None:
    assert serialize([1, 2.0, True]) == [1, 2.0, True]
    assert serialize((1, 2)) == [1, 2]


# ---------------------------------------------------------------------------
# save_golden / load_golden
# ---------------------------------------------------------------------------


def test_save_and_load_golden(tmp_path: Path) -> None:
    data = _make_summary()
    golden_path = tmp_path / "sub" / "golden.json"
    save_golden(data, golden_path)
    assert golden_path.exists()
    loaded = load_golden(golden_path)
    assert loaded == data


def test_save_golden_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c" / "golden.json"
    save_golden({"x": 1}, path)
    assert path.exists()


def test_load_golden_pretty_json(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    save_golden({"val": 42.0}, path)
    raw = path.read_text(encoding="utf-8")
    # Pretty-printed → at least one newline inside
    assert "\n" in raw
    assert json.loads(raw) == {"val": 42.0}


# ---------------------------------------------------------------------------
# diff_summaries — within tolerance → all PASS
# ---------------------------------------------------------------------------


def test_identical_summaries_all_pass() -> None:
    base = _make_summary()
    rows = diff_summaries(base, base.copy())
    assert all(r.passed for r in rows), [r for r in rows if not r.passed]


def test_within_relative_tolerance_pass() -> None:
    golden = _make_summary(ic_raw=0.050000)
    # 1 ppb relative change — well within 1e-6 default
    current = _make_summary(ic_raw=0.050000 * (1 + 1e-9))
    rows = {r.field: r for r in diff_summaries(golden, current)}
    assert rows["ic_raw"].passed


def test_at_tolerance_boundary_pass() -> None:
    golden = _make_summary(ic_raw=1.0)
    current = _make_summary(ic_raw=1.0 + 1e-6)  # exactly at 1e-6 relative
    rows = {r.field: r for r in diff_summaries(golden, current)}
    assert rows["ic_raw"].passed


# ---------------------------------------------------------------------------
# diff_summaries — past tolerance → FAIL flagged
# ---------------------------------------------------------------------------


def test_float_past_tolerance_fails() -> None:
    golden = _make_summary(ic_raw=0.05)
    current = _make_summary(ic_raw=0.06)  # 20% relative change
    rows = {r.field: r for r in diff_summaries(golden, current)}
    assert not rows["ic_raw"].passed


def test_custom_tolerance_respected() -> None:
    golden = _make_summary(gross_sharpe=0.55)
    current = _make_summary(gross_sharpe=0.56)  # ~1.8% relative
    # With default tol (1e-6): fail
    rows_strict = {r.field: r for r in diff_summaries(golden, current, rel_tol=1e-6)}
    assert not rows_strict["gross_sharpe"].passed
    # With loose tol (0.05): pass
    rows_loose = {r.field: r for r in diff_summaries(golden, current, rel_tol=0.05)}
    assert rows_loose["gross_sharpe"].passed


# ---------------------------------------------------------------------------
# diff_summaries — bool/int exact comparison
# ---------------------------------------------------------------------------


def test_bool_exact_true_vs_true_pass() -> None:
    rows = {
        r.field: r
        for r in diff_summaries(
            _make_summary(opt_converged=True), _make_summary(opt_converged=True)
        )
    }
    assert rows["opt_converged"].passed
    assert rows["opt_converged"].abs_delta is None  # no numeric delta for bool


def test_bool_exact_true_vs_false_fail() -> None:
    golden = _make_summary(opt_converged=True)
    current = _make_summary(opt_converged=False)
    rows = {r.field: r for r in diff_summaries(golden, current)}
    assert not rows["opt_converged"].passed


def test_int_exact_equal_pass() -> None:
    rows = {
        r.field: r
        for r in diff_summaries(_make_summary(n_scaling_fits=9), _make_summary(n_scaling_fits=9))
    }
    assert rows["n_scaling_fits"].passed
    assert rows["n_scaling_fits"].abs_delta is None


def test_int_exact_differ_fail() -> None:
    golden = _make_summary(n_scaling_fits=9)
    current = _make_summary(n_scaling_fits=8)
    rows = {r.field: r for r in diff_summaries(golden, current)}
    assert not rows["n_scaling_fits"].passed


# ---------------------------------------------------------------------------
# diff_summaries — horizon_ic dict compared per-key
# ---------------------------------------------------------------------------


def test_horizon_ic_all_keys_present_and_pass() -> None:
    base = _make_summary()
    rows = diff_summaries(base, base.copy())
    horizon_rows = [r for r in rows if r.field.startswith("horizon_ic[")]
    assert len(horizon_rows) == 4  # keys 1, 5, 21, 63
    assert all(r.passed for r in horizon_rows)


def test_horizon_ic_one_key_past_tolerance_fails() -> None:
    golden = _make_summary(horizon_ic={"1": 0.051, "5": 0.040, "21": 0.025, "63": 0.010})
    current = _make_summary(horizon_ic={"1": 0.051, "5": 0.040, "21": 0.999, "63": 0.010})
    rows = {r.field: r for r in diff_summaries(golden, current)}
    assert not rows["horizon_ic[21]"].passed
    assert rows["horizon_ic[1]"].passed
    assert rows["horizon_ic[5]"].passed
    assert rows["horizon_ic[63]"].passed


def test_horizon_ic_missing_key_in_current_fails() -> None:
    golden = _make_summary(horizon_ic={"1": 0.05, "5": 0.04})
    current = _make_summary(horizon_ic={"1": 0.05})  # missing key "5"
    rows = {r.field: r for r in diff_summaries(golden, current)}
    assert not rows["horizon_ic[5]"].passed
    assert rows["horizon_ic[5]"].current is None


def test_horizon_ic_extra_key_in_current_fails() -> None:
    golden = _make_summary(horizon_ic={"1": 0.05})
    current = _make_summary(horizon_ic={"1": 0.05, "99": 0.01})  # extra key
    rows = {r.field: r for r in diff_summaries(golden, current)}
    extra = rows["horizon_ic[99]"]
    assert not extra.passed
    assert extra.golden is None


# ---------------------------------------------------------------------------
# diff_summaries — missing/extra top-level fields
# ---------------------------------------------------------------------------


def test_field_missing_in_current_fails() -> None:
    golden = _make_summary()
    current = {k: v for k, v in _make_summary().items() if k != "cost_drag"}
    rows = {r.field: r for r in diff_summaries(golden, current)}
    assert not rows["cost_drag"].passed
    assert rows["cost_drag"].current is None


def test_field_extra_in_current_fails() -> None:
    golden = _make_summary()
    current = {**_make_summary(), "new_metric": 42.0}
    rows = {r.field: r for r in diff_summaries(golden, current)}
    assert not rows["new_metric"].passed
    assert rows["new_metric"].golden is None


# ---------------------------------------------------------------------------
# rel_delta math correctness
# ---------------------------------------------------------------------------


def test_rel_delta_calculation() -> None:
    golden = _make_summary(ic_raw=0.100)
    current = _make_summary(ic_raw=0.101)  # 1% relative
    rows = {r.field: r for r in diff_summaries(golden, current)}
    row = rows["ic_raw"]
    assert row.abs_delta is not None
    assert row.rel_delta is not None
    assert math.isclose(row.abs_delta, 0.001, rel_tol=1e-9)
    assert math.isclose(row.rel_delta, 0.01, rel_tol=1e-6)


def test_rel_delta_golden_zero_uses_abs() -> None:
    """When golden == 0, denom is 1 so rel_delta == abs_delta."""
    golden = _make_summary(ic_raw=0.0)
    current = _make_summary(ic_raw=1e-3)
    rows = {r.field: r for r in diff_summaries(golden, current)}
    row = rows["ic_raw"]
    assert row.abs_delta is not None
    assert row.rel_delta is not None
    assert math.isclose(row.rel_delta, row.abs_delta, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# format_diff_table
# ---------------------------------------------------------------------------


def test_format_diff_table_contains_fields() -> None:
    base = _make_summary()
    rows = diff_summaries(base, base.copy())
    table = format_diff_table(rows)
    assert "PASS" in table
    assert "FAIL" not in table.upper().replace("FAIL", "").upper() or "0 FAIL" in table
    assert "ic_raw" in table
    assert "horizon_ic" in table


def test_format_diff_table_shows_fail() -> None:
    golden = _make_summary(ic_raw=0.05)
    current = _make_summary(ic_raw=0.99)
    rows = diff_summaries(golden, current)
    table = format_diff_table(rows)
    assert "FAIL" in table


def test_format_diff_table_summary_line() -> None:
    base = _make_summary()
    rows = diff_summaries(base, base.copy())
    table = format_diff_table(rows)
    last_lines = table.strip().split("\n")
    summary_line = last_lines[-1]
    assert "PASS" in summary_line
    assert "FAIL" in summary_line


# ---------------------------------------------------------------------------
# diff_summaries — per-field tolerance override
# ---------------------------------------------------------------------------


def test_field_tol_overrides_default_for_named_field() -> None:
    """A field that fails the default tolerance passes when given a loose field_tol."""
    golden = _make_summary(backtest_p50_s=0.10)
    current = _make_summary(backtest_p50_s=0.15)  # 50% relative change — fails 1e-6
    rows_default = {r.field: r for r in diff_summaries(golden, current)}
    assert not rows_default["backtest_p50_s"].passed

    # With per-field override of 100%: passes
    rows_override = {
        r.field: r for r in diff_summaries(golden, current, field_tol={"backtest_p50_s": 1.0})
    }
    assert rows_override["backtest_p50_s"].passed


def test_field_tol_does_not_affect_other_fields() -> None:
    """field_tol for one field doesn't relax tolerance on others."""
    golden = _make_summary(backtest_p50_s=0.10, ic_raw=0.05)
    current = _make_summary(backtest_p50_s=0.15, ic_raw=0.99)
    rows = {r.field: r for r in diff_summaries(golden, current, field_tol={"backtest_p50_s": 1.0})}
    assert rows["backtest_p50_s"].passed  # exempt
    assert not rows["ic_raw"].passed  # still fails default tol
