"""Integration tests for the cross-pipeline leakage audit.

Each test verifies one of two scenarios:
  - CLEAN: correct data → all checks pass.
  - CORRUPT: deliberately broken input → the corresponding check fails and is
    reported (the other checks remain unaffected).

The three invariants tested:
  1. fwd_ret_horizon  — forward return at T uses return[T+1], not return[T].
  2. feature_target_alignment — panel.y at date D equals return[D+1].
  3. embargo_invariant — min(test_group) - max(train_group) > embargo_periods.
"""

from __future__ import annotations

import polars as pl
import pytest

from etl.datasets import GenSpec, gen_feature_panel, gen_forward_returns, gen_prices
from models.leakage import CheckResult, LeakageReport, audit_leakage
from models.panel import build_panel
from models.splitters import WalkForwardSplitter

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _make_daily_returns(spec: GenSpec) -> pl.DataFrame:
    """Single-period percent returns derived from generated prices."""
    prices = gen_prices(spec)
    return (
        prices.sort(["id", "date"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(1).over("id") - 1.0) * 100.0)
            .fill_null(0.0)
            .alias("return")
        )
        .select("date", "id", "return")
    )


@pytest.fixture(scope="module")
def clean_inputs():
    """Generate a small but statistically meaningful synthetic dataset."""
    spec = GenSpec(n_assets=15, n_dates=80, n_features=6, seed=42)
    fwd = gen_forward_returns(spec)
    feat = gen_feature_panel(spec)
    daily_ret = _make_daily_returns(spec)
    panel = build_panel(feat, fwd, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=4, embargo_periods=5)
    return {
        "spec": spec,
        "fwd": fwd,
        "feat": feat,
        "daily_ret": daily_ret,
        "panel": panel,
        "splitter": splitter,
        "embargo_periods": 5,
    }


# --------------------------------------------------------------------------- #
# Clean-run tests
# --------------------------------------------------------------------------- #


def test_clean_run_all_checks_pass(clean_inputs) -> None:
    """A correctly-generated dataset passes every leakage check."""
    inp = clean_inputs
    report = audit_leakage(
        inp["fwd"],
        inp["daily_ret"],
        inp["panel"],
        inp["splitter"],
        embargo_periods=inp["embargo_periods"],
    )
    assert isinstance(report, LeakageReport)
    assert report.all_passed, "\n".join(
        f"  {c.name}: {c.detail}" for c in report.checks if not c.passed
    )
    assert len(report.checks) == 3
    for check in report.checks:
        assert isinstance(check, CheckResult)
        assert check.passed


def test_clean_run_check_names(clean_inputs) -> None:
    """Report contains exactly the three expected check names."""
    inp = clean_inputs
    report = audit_leakage(
        inp["fwd"],
        inp["daily_ret"],
        inp["panel"],
        inp["splitter"],
        embargo_periods=inp["embargo_periods"],
    )
    names = {c.name for c in report.checks}
    assert names == {"fwd_ret_horizon", "feature_target_alignment", "embargo_invariant"}


# --------------------------------------------------------------------------- #
# Corrupt-run tests — each corrupt case trips exactly one check
# --------------------------------------------------------------------------- #


def test_corrupted_fwd_ret_horizon_fails(clean_inputs) -> None:
    """Shifting fwd_ret_1 backward by one period trips fwd_ret_horizon.

    Simulates a bug where the forward-return window starts at T instead of T+1:
    the value at date T becomes what was previously at T-1 (which covers
    (T-1, T] instead of (T, T+1]), introducing same-day data.
    """
    inp = clean_inputs
    fwd_corrupted = (
        inp["fwd"]
        .sort(["id", "date"])
        .with_columns(pl.col("fwd_ret_1").shift(1).over("id").alias("fwd_ret_1"))
    )
    report = audit_leakage(
        fwd_corrupted,
        inp["daily_ret"],
        inp["panel"],
        inp["splitter"],
        embargo_periods=inp["embargo_periods"],
    )
    assert not report.all_passed

    by_name = {c.name: c for c in report.checks}
    assert not by_name["fwd_ret_horizon"].passed, (
        "fwd_ret_horizon check should fail on shifted forward returns"
    )
    # The panel itself was built with clean forward returns, so alignment is still ok
    assert by_name["embargo_invariant"].passed


def test_corrupted_feature_target_alignment_fails(clean_inputs) -> None:
    """Using a lagged target in build_panel trips feature_target_alignment.

    Simulates pairing feature(T) with fwd_ret_1(T-1): the target's realization
    window (T-1, T] overlaps with the feature observation date T.
    """
    inp = clean_inputs
    # Shift fwd_ret_1 forward by one period so the target at date T is the
    # value that was originally at T-1 (covers window (T-1, T]).
    fwd_lagged = (
        inp["fwd"]
        .sort(["id", "date"])
        .with_columns(pl.col("fwd_ret_1").shift(1).over("id").alias("fwd_ret_1_lagged"))
    )
    panel_corrupt = build_panel(inp["feat"], fwd_lagged, "fwd_ret_1_lagged")

    report = audit_leakage(
        inp["fwd"],  # forward_returns stays clean (check 1 should pass)
        inp["daily_ret"],
        panel_corrupt,  # panel has the lagged target (check 2 should fail)
        inp["splitter"],
        embargo_periods=inp["embargo_periods"],
    )
    assert not report.all_passed

    by_name = {c.name: c for c in report.checks}
    assert not by_name["feature_target_alignment"].passed, (
        "feature_target_alignment check should fail when panel uses lagged target"
    )
    # Forward returns frame itself is clean; fwd_ret_horizon should still pass
    assert by_name["fwd_ret_horizon"].passed
    assert by_name["embargo_invariant"].passed


def test_corrupted_embargo_invariant_fails(clean_inputs) -> None:
    """Declaring embargo=0 while asserting embargo=5 trips embargo_invariant.

    Simulates a misconfiguration where the splitter was constructed with
    embargo_periods=0 but the audit is asked to verify embargo_periods=5.
    """
    inp = clean_inputs
    splitter_no_embargo = WalkForwardSplitter(n_splits=4, embargo_periods=0)

    report = audit_leakage(
        inp["fwd"],
        inp["daily_ret"],
        inp["panel"],
        splitter_no_embargo,  # embargo=0
        embargo_periods=5,  # but we assert embargo>=5
    )
    assert not report.all_passed

    by_name = {c.name: c for c in report.checks}
    assert not by_name["embargo_invariant"].passed, (
        "embargo_invariant should fail when splitter uses embargo=0 but audit checks embargo=5"
    )
    # Data and panel are clean; other checks should pass
    assert by_name["fwd_ret_horizon"].passed
    assert by_name["feature_target_alignment"].passed


# --------------------------------------------------------------------------- #
# LeakageReport.all_passed contract
# --------------------------------------------------------------------------- #


def test_all_passed_false_when_any_check_fails() -> None:
    """LeakageReport.all_passed is False if any single check failed."""
    checks = (
        CheckResult(name="fwd_ret_horizon", passed=True, detail="ok"),
        CheckResult(name="feature_target_alignment", passed=False, detail="bad"),
        CheckResult(name="embargo_invariant", passed=True, detail="ok"),
    )
    report = LeakageReport(checks=checks)
    assert not report.all_passed


def test_all_passed_true_when_all_checks_pass() -> None:
    """LeakageReport.all_passed is True when every check passed."""
    checks = (
        CheckResult(name="fwd_ret_horizon", passed=True, detail="ok"),
        CheckResult(name="feature_target_alignment", passed=True, detail="ok"),
        CheckResult(name="embargo_invariant", passed=True, detail="ok"),
    )
    report = LeakageReport(checks=checks)
    assert report.all_passed
