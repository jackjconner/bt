"""Cross-pipeline leakage audit.

Mechanically verifies that the look-ahead / leakage controls actually hold on
each run, rather than trusting the per-module controls to compose correctly.

Three invariants are checked:

1. **Forward-return horizon** — ``fwd_ret_h`` at date T is the compounded
   return over the *future* window ``(T, T+h]``.  Specifically, ``fwd_ret_1``
   at T must equal the single-period percent-return realized at T+1, not at T.

2. **Feature / target alignment** — in the ``build_panel`` output, no feature
   row at date D is paired with a forward-return target whose realization window
   starts on or before D (i.e. the target must be strictly forward of the
   feature observation date).  Concretely: ``fwd_ret_1`` paired with a feature
   at date D must equal the return at D+1, not at D or earlier.

3. **Embargo invariant** — for every walk-forward fold,
   ``min(test_group_ordinal) - max(train_group_ordinal) > embargo_periods``
   (strict; groups are integer date ordinals in calendar-day units).

Public API
----------
``CheckResult``    — frozen dataclass for a single check outcome.
``LeakageReport``  — frozen dataclass collecting all check results.
``audit_leakage``  — run all three checks and return a ``LeakageReport``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl

from .panel import PanelArrays


class _CVSplitter(Protocol):
    """Minimal interface required from a CV splitter by the leakage audit."""

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        *,
        groups: np.ndarray | None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single leakage check.

    Attributes
    ----------
    name:
        Short identifier for the check (e.g. ``"fwd_ret_horizon"``).
    passed:
        True if the invariant holds on the supplied data.
    detail:
        Human-readable explanation — what was verified and (if ``passed`` is
        False) what was found.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class LeakageReport:
    """Aggregated result from :func:`audit_leakage`.

    Attributes
    ----------
    checks:
        One :class:`CheckResult` per invariant, in the order they were run.
    all_passed:
        ``True`` iff every check passed.
    """

    checks: tuple[CheckResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


# --------------------------------------------------------------------------- #
# Check 1: forward-return horizon
# --------------------------------------------------------------------------- #


def _check_fwd_ret_horizon(
    forward_returns: pl.DataFrame,
    daily_returns: pl.DataFrame,
    *,
    horizon: int = 1,
    rtol: float = 1e-6,
) -> CheckResult:
    """Verify fwd_ret_h at date T equals realized return over (T, T+h].

    For h=1 this means: ``forward_returns["fwd_ret_1"]`` at date T must equal
    ``daily_returns["return"]`` at the *next* trading date T+1, not at T.

    Parameters
    ----------
    forward_returns:
        Long-format ``(date, id, fwd_ret_<h>, ...)`` frame.
    daily_returns:
        Long-format ``(date, id, return)`` frame where each row is the
        single-period percent return for that (date, id).
    horizon:
        Which forward horizon to verify (defaults to 1).
    rtol:
        Relative tolerance for the floating-point comparison.

    Returns
    -------
    CheckResult
    """
    target_col = f"fwd_ret_{horizon}"
    if target_col not in forward_returns.columns:
        return CheckResult(
            name="fwd_ret_horizon",
            passed=False,
            detail=f"Column {target_col!r} not found in forward_returns.",
        )
    if "return" not in daily_returns.columns:
        return CheckResult(
            name="fwd_ret_horizon",
            passed=False,
            detail="Column 'return' not found in daily_returns.",
        )

    # For h=1: fwd_ret_1[T] == return[T+1].
    # Build a shifted daily-return frame: for each (date, id) in forward_returns,
    # look up the return of the *next* date for that id.
    sorted_ret = daily_returns.select("date", "id", "return").sort(["id", "date"])
    next_ret = sorted_ret.with_columns(
        pl.col("return").shift(-1).over("id").alias("return_next_date")
    )

    # Merge with the non-null forward returns (trailing rows are null by design)
    fwd_clean = forward_returns.select("date", "id", target_col).drop_nulls()
    merged = fwd_clean.join(next_ret.select("date", "id", "return_next_date"), on=["date", "id"])

    if merged.is_empty():
        return CheckResult(
            name="fwd_ret_horizon",
            passed=False,
            detail="No overlapping rows between forward_returns and daily_returns after join.",
        )

    # For h=1 the forward return is exactly the next-day return (percent-units match).
    fwd_vals = merged[target_col].to_numpy()
    ret_vals = merged["return_next_date"].drop_nulls().to_numpy()

    # After shifting, the last row per id has a null return_next_date; drop those.
    merged_clean = merged.drop_nulls(subset=["return_next_date"])
    if merged_clean.is_empty():
        return CheckResult(
            name="fwd_ret_horizon",
            passed=False,
            detail="All next-date return values were null after shift — cannot verify.",
        )

    fwd_vals = merged_clean[target_col].to_numpy()
    ret_vals = merged_clean["return_next_date"].to_numpy()

    n = len(fwd_vals)
    abs_diff = np.abs(fwd_vals - ret_vals)
    scale = np.abs(ret_vals) + 1e-8
    max_rel_err = float(np.max(abs_diff / scale))

    if max_rel_err > rtol:
        return CheckResult(
            name="fwd_ret_horizon",
            passed=False,
            detail=(
                f"fwd_ret_{horizon} does NOT match next-period return: "
                f"max relative error = {max_rel_err:.2e} over {n} rows.  "
                "The forward return appears to include same-day data (look-ahead)."
            ),
        )

    return CheckResult(
        name="fwd_ret_horizon",
        passed=True,
        detail=(
            f"fwd_ret_{horizon} at every date T equals the realized return at T+1 "
            f"(max relative error {max_rel_err:.2e}, n={n}).  "
            "No same-day data included in forward-return labels."
        ),
    )


# --------------------------------------------------------------------------- #
# Check 2: feature / target alignment
# --------------------------------------------------------------------------- #


def _check_feature_target_alignment(
    panel: PanelArrays,
    daily_returns: pl.DataFrame,
) -> CheckResult:
    """Verify no feature row is paired with a target that starts on or before feature date.

    For ``fwd_ret_1``, the target at date D is the return over ``(D, D+1]``.
    This check verifies that the actual value in ``panel.y`` at each row (date
    D) matches the return at D+1, confirming the target's realization window
    starts strictly after D.

    If someone were to accidentally pair a feature at date D with a target from
    date D-1 (which covers ``(D-1, D]``), the values would be shifted by one
    period and this check would catch it.

    Parameters
    ----------
    panel:
        Aligned arrays from :func:`models.panel.build_panel`.
    daily_returns:
        Long-format ``(date, id, return)`` frame.

    Returns
    -------
    CheckResult
    """
    if "return" not in daily_returns.columns:
        return CheckResult(
            name="feature_target_alignment",
            passed=False,
            detail="Column 'return' not found in daily_returns.",
        )

    # Reconstruct (date, id, y) from the panel arrays
    panel_df = pl.DataFrame(
        {
            "date": list(panel.dates),
            "id": panel.ids,
            "y_panel": panel.y,
        }
    )

    # Build next-date return for each (date, id)
    sorted_ret = daily_returns.select("date", "id", "return").sort(["id", "date"])
    next_ret = sorted_ret.with_columns(
        pl.col("return").shift(-1).over("id").alias("return_next_date")
    )

    merged = panel_df.join(next_ret.select("date", "id", "return_next_date"), on=["date", "id"])
    merged_clean = merged.drop_nulls(subset=["return_next_date"])

    if merged_clean.is_empty():
        return CheckResult(
            name="feature_target_alignment",
            passed=False,
            detail="No rows with a valid next-date return after join — cannot verify alignment.",
        )

    y_vals = merged_clean["y_panel"].to_numpy()
    ret_vals = merged_clean["return_next_date"].to_numpy()
    n = len(y_vals)

    abs_diff = np.abs(y_vals - ret_vals)
    scale = np.abs(ret_vals) + 1e-8
    max_rel_err = float(np.max(abs_diff / scale))

    # Also verify the *wrong* alignment: y at date D should NOT match return at D
    same_ret = panel_df.join(daily_returns.select("date", "id", "return"), on=["date", "id"])
    if not same_ret.is_empty():
        y_wrong = same_ret["y_panel"].to_numpy()
        r_wrong = same_ret["return"].to_numpy()
        abs_diff_wrong = np.abs(y_wrong - r_wrong)
        scale_wrong = np.abs(r_wrong) + 1e-8
        max_same_day_match = float(np.max(abs_diff_wrong / scale_wrong))
        same_day_looks_correct = max_same_day_match < 1e-6
    else:
        same_day_looks_correct = False

    if max_rel_err > 1e-6:
        return CheckResult(
            name="feature_target_alignment",
            passed=False,
            detail=(
                f"Panel target (y) does NOT match the next-period return: "
                f"max relative error = {max_rel_err:.2e} over {n} rows.  "
                "Feature/target pairing is misaligned — target realization window "
                "may start on or before the feature observation date."
            ),
        )

    if same_day_looks_correct:
        return CheckResult(
            name="feature_target_alignment",
            passed=False,
            detail=(
                "Panel target (y) matches the SAME-DAY return (max rel err "
                f"{max_same_day_match:.2e}) rather than the next-day return "
                "(max rel err {max_rel_err:.2e}).  This is a look-ahead violation."
            ),
        )

    return CheckResult(
        name="feature_target_alignment",
        passed=True,
        detail=(
            f"Panel target y at each date D equals return[D+1] (max rel err "
            f"{max_rel_err:.2e}, n={n}).  No feature row is paired with a target "
            "whose realization window starts on or before the feature date."
        ),
    )


# --------------------------------------------------------------------------- #
# Check 3: embargo invariant
# --------------------------------------------------------------------------- #


def _check_embargo_invariant(
    panel: PanelArrays,
    splitter: _CVSplitter,
    embargo_periods: int,
) -> CheckResult:
    """Verify train/test group separation exceeds the embargo for every fold.

    For each fold: ``min(test_group_ordinal) - max(train_group_ordinal)
    > embargo_periods`` (strict).

    Parameters
    ----------
    panel:
        Aligned arrays whose ``.groups`` field holds integer date ordinals.
    splitter:
        Any sklearn-compatible splitter that accepts ``groups`` in ``.split``.
    embargo_periods:
        The embargo size declared at splitter construction, in units matching
        the group ordinals (calendar days when ordinals are
        ``date.toordinal()`` values).

    Returns
    -------
    CheckResult
    """
    groups = panel.groups
    violations: list[str] = []
    n_folds = 0

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(panel.X, panel.y, groups=groups)
    ):
        n_folds += 1
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        max_train = int(groups[train_idx].max())
        min_test = int(groups[test_idx].min())
        gap = min_test - max_train  # calendar days between last train date and first test date

        # The strict invariant: gap > embargo_periods
        # i.e. min_test > max_train + embargo_periods
        if gap <= embargo_periods:
            violations.append(
                f"fold {fold_idx}: max_train_ordinal={max_train}, "
                f"min_test_ordinal={min_test}, gap={gap}, "
                f"required gap > {embargo_periods}"
            )

    if violations:
        return CheckResult(
            name="embargo_invariant",
            passed=False,
            detail=(
                f"Embargo violated in {len(violations)} of {n_folds} fold(s): "
                + "; ".join(violations)
            ),
        )

    return CheckResult(
        name="embargo_invariant",
        passed=True,
        detail=(
            f"All {n_folds} fold(s) satisfy the embargo: "
            f"min(test_group) - max(train_group) > {embargo_periods} "
            "for every fold."
        ),
    )


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def audit_leakage(
    forward_returns: pl.DataFrame,
    daily_returns: pl.DataFrame,
    panel: PanelArrays,
    splitter: _CVSplitter,
    embargo_periods: int,
    *,
    fwd_horizon: int = 1,
) -> LeakageReport:
    """Run all three leakage checks and return a structured report.

    This function is **report-only** by default: failures are collected into
    the report and returned, but no exception is raised.  Whether to raise on
    failure is a human decision for the integration point (e.g.
    ``run_production_pipeline``).

    Parameters
    ----------
    forward_returns:
        Long-format ``(date, id, fwd_ret_1, ...)`` frame from
        ``etl.datasets.gen_forward_returns``.
    daily_returns:
        Long-format ``(date, id, return)`` frame (single-period percent
        returns, same session axis as forward_returns).
    panel:
        Aligned :class:`~models.panel.PanelArrays` produced by
        :func:`~models.panel.build_panel`.  Its ``.dates`` / ``.y`` fields
        are used for the alignment check.
    splitter:
        The walk-forward splitter used for CV (must expose ``.split``).
    embargo_periods:
        The embargo size passed to the splitter constructor, in calendar days
        when groups are ``date.toordinal()`` values.
    fwd_horizon:
        Which forward-return horizon to verify for checks 1 and 2 (default 1).

    Returns
    -------
    LeakageReport
        Contains one :class:`CheckResult` per invariant; ``all_passed`` is
        ``True`` iff every check passed.
    """
    check1 = _check_fwd_ret_horizon(
        forward_returns,
        daily_returns,
        horizon=fwd_horizon,
    )
    check2 = _check_feature_target_alignment(
        panel,
        daily_returns,
    )
    check3 = _check_embargo_invariant(
        panel,
        splitter,
        embargo_periods,
    )

    return LeakageReport(checks=(check1, check2, check3))


__all__ = [
    "CheckResult",
    "LeakageReport",
    "audit_leakage",
]
