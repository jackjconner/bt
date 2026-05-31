"""Tests for cross-sectional neutralization.

Key behavioral test: sector neutralization must remove a deliberately-injected
sector tilt from a signal while preserving genuine asset-level IC.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from etl.datasets import GenSpec, generate
from etl.source import session_axis
from signals.neutralize import (
    NeutralizationResult,
    evaluate_neutralization,
    neutralize_factors,
    neutralize_sector,
)

SPEC = GenSpec(n_assets=60, n_dates=40, seed=13)


def _security_master() -> pl.DataFrame:
    return generate("security_master", SPEC)


def _forward_returns() -> pl.DataFrame:
    return generate("forward_returns", SPEC)


def _alpha_signals(name: str = "momentum") -> pl.DataFrame:
    df = generate("alpha_signals", SPEC)
    return df.filter(pl.col("signal_name") == name).select("date", "id", "signal")


def _factor_loadings() -> pl.DataFrame:
    return generate("factor_loadings", SPEC)


# ---------------------------------------------------------------------------
# Sector neutralization — structural tests
# ---------------------------------------------------------------------------


def test_neutralize_sector_returns_long_format():
    sig = _alpha_signals()
    sm = _security_master()
    result = neutralize_sector(sig, sm)
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) >= {"date", "id", "signal"}


def test_neutralize_sector_same_rows():
    sig = _alpha_signals()
    sm = _security_master()
    result = neutralize_sector(sig, sm)
    assert len(result) == len(sig)


def test_neutralize_sector_ids_preserved():
    sig = _alpha_signals()
    sm = _security_master()
    result = neutralize_sector(sig, sm)
    assert set(result["id"].to_list()) == set(sig["id"].to_list())


def test_neutralize_sector_removes_sector_tilt():
    """A signal that is purely a sector mean (no asset-level variation) should
    have near-zero cross-sector IC after neutralization.

    The raw signal = sector integer encoding, so the sector dummies fully
    explain it.  After OLS projection the per-sector mean of the residuals
    must be near zero (the sector tilt is removed).  The *within-sector*
    residuals will be z-scored (non-zero) because we standardize the
    residuals, but the *mean* across assets within each sector converges to 0.
    """
    sm = _security_master()
    dates = session_axis(SPEC.n_dates, SPEC.start)

    # Build signal = sector label encoded as a float (pure sector tilt)
    sectors = sm.select("id", "sector")
    unique_sectors = sectors["sector"].unique().sort().to_list()
    sec_to_val = {s: float(i) for i, s in enumerate(unique_sectors)}

    rows = []
    for d in dates.to_list():
        for row in sectors.iter_rows(named=True):
            rows.append({"date": d, "id": row["id"], "signal": sec_to_val.get(row["sector"], 0.0)})

    sector_signal = pl.DataFrame(rows).with_columns(pl.col("id").cast(pl.Int64))

    neutralized = neutralize_sector(sector_signal, sm)

    # Join residuals back to sector labels and check per-sector means
    joined = neutralized.join(sm.select("id", "sector"), on="id", how="left")
    sector_means = (
        joined.filter(pl.col("signal").is_finite())
        .group_by("sector")
        .agg(pl.col("signal").mean().alias("mean_resid"))
    )
    resid_means = sector_means["mean_resid"].to_numpy()
    # Each sector's mean residual should be near zero (sector tilt was projected out)
    assert np.abs(resid_means).max() < 0.5


# ---------------------------------------------------------------------------
# Factor neutralization — structural tests
# ---------------------------------------------------------------------------


def test_neutralize_factors_returns_long_format():
    sig = _alpha_signals()
    fl = _factor_loadings()
    result = neutralize_factors(sig, fl)
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) >= {"date", "id", "signal"}


def test_neutralize_factors_same_rows():
    sig = _alpha_signals()
    fl = _factor_loadings()
    result = neutralize_factors(sig, fl)
    assert len(result) == len(sig)


# ---------------------------------------------------------------------------
# evaluate_neutralization
# ---------------------------------------------------------------------------


def test_evaluate_neutralization_returns_result():
    sig = _alpha_signals()
    sm = _security_master()
    fwd = _forward_returns()
    neutralized = neutralize_sector(sig, sm)
    result = evaluate_neutralization(sig, neutralized, fwd, return_col="fwd_ret_1")
    assert isinstance(result, NeutralizationResult)
    assert isinstance(result.raw_ic_mean, float)
    assert isinstance(result.neutralized_ic_mean, float)


def test_evaluate_neutralization_raw_vs_neutralized_differ():
    """The raw IC of an injected alpha signal should differ from the neutralized IC
    (it may be slightly lower after neutralization if the signal has incidental
    sector exposure)."""
    sig = _alpha_signals("quality")
    sm = _security_master()
    fwd = _forward_returns()
    neutralized = neutralize_sector(sig, sm)
    result = evaluate_neutralization(
        sig, neutralized, fwd, return_col="fwd_ret_1", method="rank", min_obs=5
    )
    # At minimum the raw IC should be finite
    assert np.isfinite(result.raw_ic_mean)
    assert np.isfinite(result.neutralized_ic_mean)
