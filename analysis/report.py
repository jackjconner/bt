"""Benchmark-relative attribution report.

Composes the existing attribution, benchmark, and turnover primitives into a
single structured ``AttributionReport`` that explains *why* a strategy beat or
lagged the benchmark — not just its Sharpe.

Architecture:
    analyse_attribution(...)
        1. reconstruct_weights          (turnover.py)       → portfolio weight panel
        2. factor_attribution           (attribution.py)    → OLS betas on factor returns
        3. _active_factor_exposures     (new, uses loadings) → per-factor active exposure
        4. _factor_return_contributions (new)               → factor contribution to active ret
        5. _brinson_decomposition       (new)               → allocation/selection/interaction
        6. active_returns + TE + IR     (benchmark.py)      → benchmark-relative scalars

Data contracts (matching the synthetic REGISTRY schemas):
    factor_loadings  : (date, id, factor_id, loading)     — fractional
    factor_returns   : (date, factor_id, return)           — in PERCENT; divided by 100 here
    benchmark_returns: (date, return_1d)                   — fractional (caller converts)
    benchmark_weights: (date, id, benchmark_weight)        — sum-to-1 per date
    asset_returns    : (date, id, return_1d)               — fractional
    security_master  : (id, sector, ...)                   — static sector map

The Brinson decomposition is computed per calendar date on which both
portfolio weights and benchmark weights are available, then aggregated
(summed) to a single cross-section. Because the synthetic data has daily
benchmark weights and portfolio weights at rebalance dates only, we merge
on the intersection of those dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from etl.source import to_float

from .attribution import FactorAttributionResult, factor_attribution
from .benchmark import active_returns, information_ratio, tracking_error
from .turnover import reconstruct_weights

if TYPE_CHECKING:
    from backtest.engine import BacktestResult

# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrinsonDecomposition:
    """Brinson-Hood-Beebower sector decomposition aggregated across the period.

    All fields are DataFrames with a ``sector`` column plus the named effect.

    Additive identity: for each date, allocation + selection + interaction
    approximately equals the active return (portfolio - benchmark) on that date,
    with small residuals from the discretisation of sector membership.
    """

    allocation: pl.DataFrame  # (sector, allocation)
    selection: pl.DataFrame  # (sector, selection)
    interaction: pl.DataFrame  # (sector, interaction)
    sector_active_return: pl.DataFrame  # (sector, active_return)  — allocation+sel+inter


@dataclass(frozen=True)
class AttributionReport:
    """Benchmark-relative attribution report.

    Produced by ``analyze_attribution``; summarises *why* the strategy
    beat or lagged the benchmark across three lenses.

    Attributes:
        active_return_series:
            Date-aligned ``(date, active_return)`` — strategy daily return
            minus benchmark daily return on the inner-join date range.
        tracking_error:
            Annualized standard deviation of active returns (fractional).
        information_ratio:
            Annualized mean active return / tracking error.
        factor_attribution:
            OLS-based ``FactorAttributionResult`` regressing strategy returns
            on factor returns; betas are portfolio-level factor exposures,
            ``alpha_annualized`` is the factor-unexplained return.
        factor_active_exposures:
            DataFrame ``(factor_id, active_exposure)`` — portfolio factor
            exposure minus the benchmark's factor exposure (benchmark-weighted
            average loading), computed from ``factor_loadings`` on the dates
            for which both portfolio weights and benchmark weights are available.
        factor_return_contributions:
            DataFrame ``(factor_id, active_contribution)`` — per-factor
            contribution to total active return over the period.
            Computed as ``active_exposure * cum_factor_return``.
        brinson:
            Brinson–Hood–Beebower decomposition by sector.  ``None`` when
            ``security_master`` or ``benchmark_weights`` are not supplied.
        benchmark_metrics:
            Dict of summary scalars: ``active_return_ann``, ``tracking_error``,
            ``information_ratio``.
    """

    active_return_series: pl.DataFrame  # (date, active_return)
    tracking_error: float
    information_ratio: float
    factor_attribution: FactorAttributionResult
    factor_active_exposures: pl.DataFrame  # (factor_id, active_exposure)
    factor_return_contributions: pl.DataFrame  # (factor_id, active_contribution)
    brinson: BrinsonDecomposition | None
    benchmark_metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _active_factor_exposures(
    portfolio_weights: pl.DataFrame,  # (date, id, weight)
    benchmark_weights: pl.DataFrame,  # (date, id, benchmark_weight)
    factor_loadings: pl.DataFrame,  # (date, id, factor_id, loading)
) -> pl.DataFrame:
    """Compute per-factor active exposure averaged over overlapping dates.

    Active exposure for factor k on date t:
        active_k_t = sum_i(w_p_it * L_ikt) - sum_i(w_b_it * L_ikt)

    We average over the dates present in both weight panels and the
    loading panel to get a single representative number per factor.

    Returns ``(factor_id, portfolio_exposure, benchmark_exposure, active_exposure)``.
    """
    # Restrict to dates present in all three panels
    port_dates = set(portfolio_weights["date"].unique().to_list())
    bmk_dates = set(benchmark_weights["date"].unique().to_list())
    loading_dates = set(factor_loadings["date"].unique().to_list())
    common_dates = list(port_dates & bmk_dates & loading_dates)

    if not common_dates:
        # Return empty frame with correct schema
        return pl.DataFrame(
            {
                "factor_id": pl.Series([], dtype=pl.Int64),
                "portfolio_exposure": pl.Series([], dtype=pl.Float64),
                "benchmark_exposure": pl.Series([], dtype=pl.Float64),
                "active_exposure": pl.Series([], dtype=pl.Float64),
            }
        )

    pw = portfolio_weights.filter(pl.col("date").is_in(common_dates))
    bw = benchmark_weights.filter(pl.col("date").is_in(common_dates))
    fl = factor_loadings.filter(pl.col("date").is_in(common_dates))

    # Portfolio exposure per (date, factor_id): sum_i(w_p * loading)
    port_exp = (
        pw.join(fl, on=["date", "id"], how="inner")
        .with_columns((pl.col("weight") * pl.col("loading")).alias("wl"))
        .group_by(["date", "factor_id"])
        .agg(pl.col("wl").sum().alias("port_exp"))
    )

    # Benchmark exposure per (date, factor_id): sum_i(w_b * loading)
    bmk_exp = (
        bw.join(fl, on=["date", "id"], how="inner")
        .with_columns((pl.col("benchmark_weight") * pl.col("loading")).alias("wl"))
        .group_by(["date", "factor_id"])
        .agg(pl.col("wl").sum().alias("bmk_exp"))
    )

    # Join and compute active; average across dates
    combined = port_exp.join(bmk_exp, on=["date", "factor_id"], how="inner")
    return (
        combined.with_columns((pl.col("port_exp") - pl.col("bmk_exp")).alias("active_exp"))
        .group_by("factor_id")
        .agg(
            pl.col("port_exp").mean().alias("portfolio_exposure"),
            pl.col("bmk_exp").mean().alias("benchmark_exposure"),
            pl.col("active_exp").mean().alias("active_exposure"),
        )
        .sort("factor_id")
    )


def _factor_return_contributions(
    active_exposures: pl.DataFrame,  # (factor_id, active_exposure, ...)
    factor_returns: pl.DataFrame,  # (date, factor_id, return) — FRACTIONAL
) -> pl.DataFrame:
    """Per-factor contribution to total active return.

    Contribution_k = active_exposure_k * sum_t(factor_return_k_t)

    Returns ``(factor_id, active_contribution)``.

    This is a first-order approximation: it attributes the active return
    that would be earned by the mean active exposure if held throughout,
    using the realized cumulative factor returns. Cross-effects between
    time-varying exposures and factor returns are captured in the residual.
    """
    cum_factor = (
        factor_returns.group_by("factor_id")
        .agg(pl.col("return").sum().alias("cum_factor_return"))
        .sort("factor_id")
    )

    return (
        active_exposures.select("factor_id", "active_exposure")
        .join(cum_factor, on="factor_id", how="inner")
        .with_columns(
            (pl.col("active_exposure") * pl.col("cum_factor_return")).alias("active_contribution")
        )
        .select("factor_id", "active_contribution")
        .sort("factor_id")
    )


def _brinson_decomposition(
    portfolio_weights: pl.DataFrame,  # (date, id, weight)
    benchmark_weights: pl.DataFrame,  # (date, id, benchmark_weight)
    asset_returns: pl.DataFrame,  # (date, id, return_1d)
    security_master: pl.DataFrame,  # (id, sector)
) -> BrinsonDecomposition:
    """Brinson-Hood-Beebower sector decomposition.

    For each date t and sector s:
        allocation_s   = (w_p_s - w_b_s) * (R_b_s - R_b)
        selection_s    = w_b_s            * (R_p_s - R_b_s)
        interaction_s  = (w_p_s - w_b_s) * (R_p_s - R_b_s)

    where:
        w_p_s  = sum of portfolio weights in sector s
        w_b_s  = sum of benchmark weights in sector s
        R_p_s  = portfolio-weight-average return in sector s
        R_b_s  = benchmark-weight-average return in sector s
        R_b    = total benchmark return (sum_s w_b_s * R_b_s)

    Results are aggregated (summed) across all overlapping dates to produce
    a period total for each sector.

    Returns a ``BrinsonDecomposition``.
    """
    sector_map = security_master.select("id", "sector")

    # Restrict to common dates
    port_dates = set(portfolio_weights["date"].unique().to_list())
    bmk_dates = set(benchmark_weights["date"].unique().to_list())
    ret_dates = set(asset_returns["date"].unique().to_list())
    common_dates = list(port_dates & bmk_dates & ret_dates)

    if not common_dates:
        empty = pl.DataFrame(
            {
                "sector": pl.Series([], dtype=pl.Categorical),
                "allocation": pl.Series([], dtype=pl.Float64),
                "selection": pl.Series([], dtype=pl.Float64),
                "interaction": pl.Series([], dtype=pl.Float64),
                "active_return": pl.Series([], dtype=pl.Float64),
            }
        )
        return BrinsonDecomposition(
            allocation=empty.select("sector", "allocation"),
            selection=empty.select("sector", "selection"),
            interaction=empty.select("sector", "interaction"),
            sector_active_return=empty.select("sector", "active_return"),
        )

    pw = portfolio_weights.filter(pl.col("date").is_in(common_dates))
    bw = benchmark_weights.filter(pl.col("date").is_in(common_dates))
    ar = asset_returns.filter(pl.col("date").is_in(common_dates)).select("date", "id", "return_1d")

    # Attach sectors
    pw_s = pw.join(sector_map, on="id", how="left").with_columns(
        pl.col("sector").fill_null(pl.lit("Unknown").cast(pl.Categorical))
    )
    bw_s = bw.join(sector_map, on="id", how="left").with_columns(
        pl.col("sector").fill_null(pl.lit("Unknown").cast(pl.Categorical))
    )

    # Per (date, id): portfolio-weighted return contribution
    pw_ret = pw_s.join(ar, on=["date", "id"], how="left").with_columns(
        pl.col("return_1d").fill_null(0.0)
    )
    bw_ret = bw_s.join(ar, on=["date", "id"], how="left").with_columns(
        pl.col("return_1d").fill_null(0.0)
    )

    # Per (date, sector): sector weights and returns
    port_sector = (
        pw_ret.group_by(["date", "sector"])
        .agg(
            pl.col("weight").sum().alias("w_p_s"),
            (pl.col("weight") * pl.col("return_1d")).sum().alias("R_p_s_raw"),
        )
        # R_p_s = weighted avg return = sum(w_i * r_i) / w_p_s
        .with_columns(
            (pl.col("R_p_s_raw") / pl.col("w_p_s").clip(lower_bound=1e-12)).alias("R_p_s")
        )
    )

    bmk_sector = (
        bw_ret.group_by(["date", "sector"])
        .agg(
            pl.col("benchmark_weight").sum().alias("w_b_s"),
            (pl.col("benchmark_weight") * pl.col("return_1d")).sum().alias("R_b_s_raw"),
        )
        .with_columns(
            (pl.col("R_b_s_raw") / pl.col("w_b_s").clip(lower_bound=1e-12)).alias("R_b_s")
        )
    )

    # Total benchmark return per date: R_b = sum_s(w_b_s * R_b_s)
    bmk_total = bmk_sector.group_by("date").agg(
        (pl.col("w_b_s") * pl.col("R_b_s")).sum().alias("R_b")
    )

    # Join everything
    sector_df = (
        port_sector.join(bmk_sector, on=["date", "sector"], how="full", coalesce=True)
        .with_columns(
            pl.col("w_p_s").fill_null(0.0),
            pl.col("w_b_s").fill_null(0.0),
            pl.col("R_p_s").fill_null(0.0),
            pl.col("R_b_s").fill_null(0.0),
        )
        .join(bmk_total, on="date", how="left")
        .with_columns(pl.col("R_b").fill_null(0.0))
    )

    # Brinson effects
    sector_df = sector_df.with_columns(
        ((pl.col("w_p_s") - pl.col("w_b_s")) * (pl.col("R_b_s") - pl.col("R_b"))).alias(
            "allocation"
        ),
        (pl.col("w_b_s") * (pl.col("R_p_s") - pl.col("R_b_s"))).alias("selection"),
        ((pl.col("w_p_s") - pl.col("w_b_s")) * (pl.col("R_p_s") - pl.col("R_b_s"))).alias(
            "interaction"
        ),
    )

    # Active return per sector per date
    sector_df = sector_df.with_columns(
        (pl.col("allocation") + pl.col("selection") + pl.col("interaction")).alias("active_return")
    )

    # Aggregate across dates
    period_totals = (
        sector_df.group_by("sector")
        .agg(
            pl.col("allocation").sum(),
            pl.col("selection").sum(),
            pl.col("interaction").sum(),
            pl.col("active_return").sum(),
        )
        .sort("sector")
    )

    return BrinsonDecomposition(
        allocation=period_totals.select("sector", "allocation"),
        selection=period_totals.select("sector", "selection"),
        interaction=period_totals.select("sector", "interaction"),
        sector_active_return=period_totals.select("sector", "active_return"),
    )


# ---------------------------------------------------------------------------
# Public aggregator
# ---------------------------------------------------------------------------


def analyze_attribution(
    backtest_result: BacktestResult,
    benchmark_returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    factor_loadings: pl.DataFrame,
    asset_returns: pl.DataFrame,
    benchmark_weights: pl.DataFrame | None = None,
    security_master: pl.DataFrame | None = None,
) -> AttributionReport:
    """Produce a benchmark-relative ``AttributionReport``.

    Composes the following primitives:
    - ``reconstruct_weights``       (analysis.turnover)
    - ``factor_attribution``        (analysis.attribution)
    - ``active_returns``            (analysis.benchmark)
    - ``tracking_error``            (analysis.benchmark)
    - ``information_ratio``         (analysis.benchmark)

    Parameters
    ----------
    backtest_result:
        A ``BacktestResult`` with at least ``trade_log`` and ``nav_history``.
    benchmark_returns:
        ``(date, return_1d)`` in fractional units. Use
        ``benchmark_returns_to_fractional`` to convert from the percent-unit
        dataset format before passing here.
    factor_returns:
        Long ``(date, factor_id, return)`` in **PERCENT** units (matching the
        ``factor_returns`` dataset schema). Divided by 100 internally before
        the OLS regression.
    factor_loadings:
        Long ``(date, id, factor_id, loading)`` in fractional units.
    asset_returns:
        Long ``(date, id, return_1d)`` in fractional units.
    benchmark_weights:
        Optional ``(date, id, benchmark_weight)``. Required for the Brinson
        decomposition and for the factor active-exposure calculation. If not
        supplied, ``brinson`` is ``None`` and ``factor_active_exposures``
        contains zeros.
    security_master:
        Optional ``(id, sector, ...)``. Required for the Brinson sector
        grouping. Ignored (and ``brinson`` is ``None``) if not supplied.

    Returns
    -------
    AttributionReport
    """
    from .metrics import returns_from_nav

    # 1. Portfolio weights from trade log
    portfolio_weights = reconstruct_weights(backtest_result.trade_log)

    # 2. Strategy returns from NAV
    strategy_returns = returns_from_nav(backtest_result.nav_history)

    # 3. Factor attribution (OLS on fractional factor returns)
    fr_fractional = factor_returns.with_columns((pl.col("return") / 100.0).alias("return"))
    fa = factor_attribution(strategy_returns, fr_fractional)

    # 4. Benchmark-relative scalars
    ar_series = active_returns(strategy_returns, benchmark_returns)
    te = tracking_error(strategy_returns, benchmark_returns)
    ir = information_ratio(strategy_returns, benchmark_returns)

    active_ret_ann = to_float(ar_series["active_return"].mean() or 0.0) * 252

    benchmark_metrics = {
        "active_return_ann": active_ret_ann,
        "tracking_error": te,
        "information_ratio": ir,
    }

    # 5. Factor active exposures
    if benchmark_weights is not None:
        fae = _active_factor_exposures(portfolio_weights, benchmark_weights, factor_loadings)
    else:
        # No benchmark weights: exposures are portfolio-only (active = portfolio exposure)
        fae = _active_factor_exposures_portfolio_only(portfolio_weights, factor_loadings)

    # 6. Factor return contributions
    frc = _factor_return_contributions(fae, fr_fractional)

    # 7. Brinson decomposition
    brinson: BrinsonDecomposition | None = None
    if benchmark_weights is not None and security_master is not None:
        brinson = _brinson_decomposition(
            portfolio_weights, benchmark_weights, asset_returns, security_master
        )

    return AttributionReport(
        active_return_series=ar_series,
        tracking_error=te,
        information_ratio=ir,
        factor_attribution=fa,
        factor_active_exposures=fae,
        factor_return_contributions=frc,
        brinson=brinson,
        benchmark_metrics=benchmark_metrics,
    )


def _active_factor_exposures_portfolio_only(
    portfolio_weights: pl.DataFrame,  # (date, id, weight)
    factor_loadings: pl.DataFrame,  # (date, id, factor_id, loading)
) -> pl.DataFrame:
    """Factor exposures when no benchmark weights are available.

    Returns the same schema as ``_active_factor_exposures`` but with
    ``benchmark_exposure = 0`` and ``active_exposure = portfolio_exposure``.
    """
    port_dates = set(portfolio_weights["date"].unique().to_list())
    loading_dates = set(factor_loadings["date"].unique().to_list())
    common_dates = list(port_dates & loading_dates)

    if not common_dates:
        return pl.DataFrame(
            {
                "factor_id": pl.Series([], dtype=pl.Int64),
                "portfolio_exposure": pl.Series([], dtype=pl.Float64),
                "benchmark_exposure": pl.Series([], dtype=pl.Float64),
                "active_exposure": pl.Series([], dtype=pl.Float64),
            }
        )

    pw = portfolio_weights.filter(pl.col("date").is_in(common_dates))
    fl = factor_loadings.filter(pl.col("date").is_in(common_dates))

    port_exp = (
        pw.join(fl, on=["date", "id"], how="inner")
        .with_columns((pl.col("weight") * pl.col("loading")).alias("wl"))
        .group_by(["date", "factor_id"])
        .agg(pl.col("wl").sum().alias("port_exp"))
        .group_by("factor_id")
        .agg(pl.col("port_exp").mean().alias("portfolio_exposure"))
        .sort("factor_id")
    )

    return port_exp.with_columns(
        pl.lit(0.0).alias("benchmark_exposure"),
        pl.col("portfolio_exposure").alias("active_exposure"),
    ).select("factor_id", "portfolio_exposure", "benchmark_exposure", "active_exposure")
