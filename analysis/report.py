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
# Internal helpers — factor exposures
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


# ---------------------------------------------------------------------------
# Internal helpers — Brinson decomposition
# ---------------------------------------------------------------------------


def _empty_brinson_decomposition() -> BrinsonDecomposition:
    """Return an empty BrinsonDecomposition for the no-overlap degenerate case."""
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


def _attach_sector_with_returns(
    weights_df: pl.DataFrame,
    sector_map: pl.DataFrame,  # (id, sector)
    asset_returns: pl.DataFrame,  # (date, id, return_1d)
    weight_col: str,
) -> pl.DataFrame:
    """Attach sector labels and daily returns to a weight frame.

    Fills missing sectors as ``"Unknown"`` and missing returns as ``0.0``.
    Returns the input frame extended with ``sector`` and ``return_1d`` columns.
    """
    with_sector = weights_df.join(sector_map, on="id", how="left").with_columns(
        pl.col("sector").fill_null(pl.lit("Unknown").cast(pl.Categorical))
    )
    return with_sector.join(asset_returns, on=["date", "id"], how="left").with_columns(
        pl.col("return_1d").fill_null(0.0)
    )


def _sector_weights_and_returns(
    weights_with_returns: pl.DataFrame,
    weight_col: str,
    return_col_prefix: str,
) -> pl.DataFrame:
    """Aggregate per-(date, sector) weight sums and weighted-average returns.

    ``weight_col`` is the name of the weight column (e.g. ``"weight"`` or
    ``"benchmark_weight"``).  ``return_col_prefix`` is used to name the
    output columns as ``w_{prefix}_s`` and ``R_{prefix}_s``.

    Returns a DataFrame with columns
    ``(date, sector, w_{prefix}_s, R_{prefix}_s)``.
    """
    w_s = f"w_{return_col_prefix}_s"
    R_raw = f"R_{return_col_prefix}_s_raw"
    R_s = f"R_{return_col_prefix}_s"

    return (
        weights_with_returns.group_by(["date", "sector"])
        .agg(
            pl.col(weight_col).sum().alias(w_s),
            (pl.col(weight_col) * pl.col("return_1d")).sum().alias(R_raw),
        )
        .with_columns((pl.col(R_raw) / pl.col(w_s).clip(lower_bound=1e-12)).alias(R_s))
        .drop(R_raw)
    )


def _compute_brinson_effects(sector_df: pl.DataFrame) -> pl.DataFrame:
    """Add allocation, selection, interaction, and active_return columns.

    Expects columns ``(date, sector, w_p_s, R_p_s, w_b_s, R_b_s, R_b)``.

    Brinson-Hood-Beebower formulae:
        allocation  = (w_p_s - w_b_s) * (R_b_s - R_b)
        selection   = w_b_s            * (R_p_s - R_b_s)
        interaction = (w_p_s - w_b_s) * (R_p_s - R_b_s)
        active_return = allocation + selection + interaction
    """
    return sector_df.with_columns(
        ((pl.col("w_p_s") - pl.col("w_b_s")) * (pl.col("R_b_s") - pl.col("R_b"))).alias(
            "allocation"
        ),
        (pl.col("w_b_s") * (pl.col("R_p_s") - pl.col("R_b_s"))).alias("selection"),
        ((pl.col("w_p_s") - pl.col("w_b_s")) * (pl.col("R_p_s") - pl.col("R_b_s"))).alias(
            "interaction"
        ),
    ).with_columns(
        (pl.col("allocation") + pl.col("selection") + pl.col("interaction")).alias("active_return")
    )


def _aggregate_sector_effects(sector_df: pl.DataFrame) -> pl.DataFrame:
    """Sum allocation/selection/interaction/active_return over all dates per sector.

    Returns ``(sector, allocation, selection, interaction, active_return)``
    sorted by sector.
    """
    return (
        sector_df.group_by("sector")
        .agg(
            pl.col("allocation").sum(),
            pl.col("selection").sum(),
            pl.col("interaction").sum(),
            pl.col("active_return").sum(),
        )
        .sort("sector")
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

    # Restrict to common dates across all three panels
    port_dates = set(portfolio_weights["date"].unique().to_list())
    bmk_dates = set(benchmark_weights["date"].unique().to_list())
    ret_dates = set(asset_returns["date"].unique().to_list())
    common_dates = list(port_dates & bmk_dates & ret_dates)

    if not common_dates:
        return _empty_brinson_decomposition()

    pw = portfolio_weights.filter(pl.col("date").is_in(common_dates))
    bw = benchmark_weights.filter(pl.col("date").is_in(common_dates))
    ar = asset_returns.filter(pl.col("date").is_in(common_dates)).select("date", "id", "return_1d")

    # Attach sectors and returns to each weight frame
    pw_ret = _attach_sector_with_returns(pw, sector_map, ar, "weight")
    bw_ret = _attach_sector_with_returns(bw, sector_map, ar, "benchmark_weight")

    # Per (date, sector): aggregated weights and weighted-average returns
    port_sector = _sector_weights_and_returns(pw_ret, "weight", "p")
    bmk_sector = _sector_weights_and_returns(bw_ret, "benchmark_weight", "b")

    # Total benchmark return per date: R_b = sum_s(w_b_s * R_b_s)
    bmk_total = bmk_sector.group_by("date").agg(
        (pl.col("w_b_s") * pl.col("R_b_s")).sum().alias("R_b")
    )

    # Merge portfolio and benchmark sector stats; fill missing sectors with zeros
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

    # Compute Brinson effects per (date, sector), then sum to period totals
    sector_df = _compute_brinson_effects(sector_df)
    period_totals = _aggregate_sector_effects(sector_df)

    return BrinsonDecomposition(
        allocation=period_totals.select("sector", "allocation"),
        selection=period_totals.select("sector", "selection"),
        interaction=period_totals.select("sector", "interaction"),
        sector_active_return=period_totals.select("sector", "active_return"),
    )


# ---------------------------------------------------------------------------
# Internal helpers — analyze_attribution sub-steps
# ---------------------------------------------------------------------------


def _portfolio_weights_and_strategy_returns(
    backtest_result: BacktestResult,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reconstruct portfolio weights and compute strategy daily returns from NAV.

    Returns ``(portfolio_weights, strategy_returns)`` where
    ``portfolio_weights`` is ``(date, id, weight)`` and
    ``strategy_returns`` is ``(date, return_1d)``.
    """
    from .metrics import returns_from_nav

    portfolio_weights = reconstruct_weights(backtest_result.trade_log)
    strategy_returns = returns_from_nav(backtest_result.nav_history)
    return portfolio_weights, strategy_returns


def _benchmark_relative_scalars(
    strategy_returns: pl.DataFrame,
    benchmark_returns: pl.DataFrame,
) -> tuple[pl.DataFrame, float, float, dict[str, float]]:
    """Compute benchmark-relative series and summary scalars.

    Returns ``(ar_series, te, ir, benchmark_metrics)`` where:
    - ``ar_series`` is ``(date, active_return)``
    - ``te`` is annualized tracking error (fractional)
    - ``ir`` is the information ratio
    - ``benchmark_metrics`` is a dict with keys
      ``active_return_ann``, ``tracking_error``, ``information_ratio``
    """
    ar_series = active_returns(strategy_returns, benchmark_returns)
    te = tracking_error(strategy_returns, benchmark_returns)
    ir = information_ratio(strategy_returns, benchmark_returns)
    active_ret_ann = to_float(ar_series["active_return"].mean() or 0.0) * 252
    benchmark_metrics: dict[str, float] = {
        "active_return_ann": active_ret_ann,
        "tracking_error": te,
        "information_ratio": ir,
    }
    return ar_series, te, ir, benchmark_metrics


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
    # 1. Portfolio weights and strategy returns from BacktestResult
    portfolio_weights, strategy_returns = _portfolio_weights_and_strategy_returns(backtest_result)

    # 2. Factor attribution (OLS on fractional factor returns)
    fr_fractional = factor_returns.with_columns((pl.col("return") / 100.0).alias("return"))
    fa = factor_attribution(strategy_returns, fr_fractional)

    # 3. Benchmark-relative scalars
    ar_series, te, ir, benchmark_metrics = _benchmark_relative_scalars(
        strategy_returns, benchmark_returns
    )

    # 4. Factor active exposures
    if benchmark_weights is not None:
        fae = _active_factor_exposures(portfolio_weights, benchmark_weights, factor_loadings)
    else:
        # No benchmark weights: exposures are portfolio-only (active = portfolio exposure)
        fae = _active_factor_exposures_portfolio_only(portfolio_weights, factor_loadings)

    # 5. Factor return contributions
    frc = _factor_return_contributions(fae, fr_fractional)

    # 6. Brinson decomposition
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
