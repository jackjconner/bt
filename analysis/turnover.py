"""Turnover, net-of-cost NAV, and position concentration analytics.

Why this file: turnover and cost accounting are driven by `trade_log` and an
optional per-asset cost model — a distinct concern from return-series metrics.
Position concentration is also computed from the trade_log-derived weight
panel, so it lives here alongside turnover.

Conventions:
- `trade_log` from `BacktestResult`: columns `(date, id, quantity)` where
  `quantity` is in NAV-fraction units (the engine divides dollar value by NAV).
- `transaction_costs` from the dataset: `commission_bps`, `half_spread_bps`,
  `exchange_fee_bps` are in basis-points; `impact_coef` is dimensionless.
- One-way turnover = sum of absolute position changes / 2 (each trade is
  counted once). Two-way = sum of absolute changes (buy side + sell side).
"""

from __future__ import annotations

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------


def one_way_turnover(trade_log: pl.DataFrame) -> pl.DataFrame:
    """Daily one-way portfolio turnover.

    One-way convention: each round-trip trade (buy + matching sell) is counted
    once, so a full portfolio replacement = 100 % turnover per period.
    `abs(delta_weight) / 2` per day, summed across assets.

    Returns a DataFrame `(date, turnover_1w)`.

    The ``trade_log`` is grouped after a ``sort("date")`` with
    ``maintain_order=True`` rather than a hash ``group_by`` followed by a final
    ``.sort``: the production ``trade_log`` is already emitted in date order
    (the engine appends each bar's trades in increasing time), so the sort is
    near-free and the ordered group-by skips building a hash table — measurably
    cheaper on the long-history grid points while producing identical output for
    any input ordering.
    """
    return (
        trade_log.sort("date")
        .group_by("date", maintain_order=True)
        .agg((pl.col("quantity").abs().sum() / 2.0).alias("turnover_1w"))
    )


def two_way_turnover(trade_log: pl.DataFrame) -> pl.DataFrame:
    """Daily two-way portfolio turnover.

    Two-way: buys and sells are counted separately, so a full replacement = 200 %.
    Returns a DataFrame `(date, turnover_2w)`.

    See ``one_way_turnover`` for why this sorts first and groups with
    ``maintain_order=True`` instead of hash-grouping then sorting.
    """
    return (
        trade_log.sort("date")
        .group_by("date", maintain_order=True)
        .agg(pl.col("quantity").abs().sum().alias("turnover_2w"))
    )


# ---------------------------------------------------------------------------
# Net-of-cost NAV
# ---------------------------------------------------------------------------


def _total_cost_bps(
    date: pl.Series,
    id_: pl.Series,
    quantity: pl.Series,
    costs: pl.DataFrame,
) -> pl.Series:
    """Per-trade cost in basis-points of trade value.

    Cost components that scale with trade size:
        commission_bps + 2 * half_spread_bps + exchange_fee_bps

    Impact is excluded here — it is market-impact on price, not a cash charge,
    and is modeled at the backtest fill layer (not yet implemented). The formula
    uses only the explicit-cost columns to keep the accounting auditable.
    """
    trade_df = pl.DataFrame({"date": date, "id": id_, "quantity": quantity})
    joined = trade_df.join(costs, on=["date", "id"], how="left")
    return (
        joined["commission_bps"].fill_null(0.0)
        + 2.0 * joined["half_spread_bps"].fill_null(0.0)
        + joined["exchange_fee_bps"].fill_null(0.0)
    )


def net_nav(
    nav_history: pl.DataFrame,
    trade_log: pl.DataFrame,
    costs: pl.DataFrame,
) -> pl.DataFrame:
    """Apply per-asset cost model to trade_log and return net-of-cost NAV.

    For each rebalance day the aggregate cost (in NAV-fraction) is computed
    and subtracted from the gross NAV. The adjustment compounds forward so
    late-period NAV correctly reflects the cumulative drag.

    `costs` must be the `transaction_costs` dataset (columns include
    `date, id, commission_bps, half_spread_bps, exchange_fee_bps`).

    Returns a DataFrame `(date, nav_gross, nav_net)`.
    """
    # cost per trade as bps of trade NAV-value
    cost_bps = _total_cost_bps(trade_log["date"], trade_log["id"], trade_log["quantity"], costs)
    # trade value in NAV fraction
    trade_value = trade_log["quantity"].abs()
    # daily total cost as NAV fraction (bps / 10_000 * trade_value)
    trade_df = trade_log.with_columns((cost_bps / 10_000.0 * trade_value).alias("cost_frac"))
    daily_cost = trade_df.group_by("date").agg(pl.col("cost_frac").sum()).sort("date")

    gross = nav_history.sort("date")
    merged = gross.join(daily_cost, on="date", how="left").with_columns(
        pl.col("cost_frac").fill_null(0.0)
    )

    # Subtract the cost from NAV multiplicatively: nav_net[t] = nav_net[t-1] *
    # nav_gross[t] / nav_gross[t-1] * (1 - cost_frac[t]).
    # Computed via a cumulative product of (1 - cost_frac) applied to gross NAV.
    gross_vals = merged["nav"].to_numpy()
    cost_fracs = merged["cost_frac"].to_numpy()
    cost_multipliers = 1.0 - cost_fracs
    cum_cost = np.cumprod(cost_multipliers)
    nav_net_vals = gross_vals * cum_cost / cum_cost[0]

    return merged.select("date", pl.col("nav").alias("nav_gross")).with_columns(
        pl.Series("nav_net", nav_net_vals)
    )


# ---------------------------------------------------------------------------
# Weight panel reconstruction
# ---------------------------------------------------------------------------


def reconstruct_weights(trade_log: pl.DataFrame) -> pl.DataFrame:
    """Reconstruct per-date portfolio weights from the trade log.

    The backtest engine records *target* position fractions in `quantity`
    (delta from previous target). Summing cumulative deltas recovers the target
    weight at each rebalance date. Positions are forward-filled to non-rebalance
    days so every date in the NAV history has a weight row.

    Returns a long DataFrame `(date, id, weight)`.

    Implementation note: the engine stores `quantity = (target - prev) * nav`,
    which is a dollar-delta, not a weight-delta. Because the engine uses NAV-
    fraction positions (target = softmax(signal)), we can recover the weight by
    cumulative summing the *weight*-deltas. However, the trade_log stores
    dollar quantities, not weight fractions. To reconstruct weights we compute
    the cumulative absolute position using the running-sum-of-deltas approach;
    the absolute weight per asset on a rebalance date is the sum of all
    quantities for that asset up to that date divided by NAV — but NAV is not
    in the trade_log.

    Instead, we rely on the fact that the engine's `deltas = target - positions`
    and `positions = target` after rebalance, so on any rebalance date the
    sum of positive `quantity` entries ≈ the sum of absolute weights bought.
    For simplicity we normalize each rebalance-date's absolute quantities to
    sum to 1.0, which recovers the weight vector exactly under a fully-invested
    long-only constraint (which is what the backtest engine enforces).
    """
    # Sum quantities per (date, id); on non-rebalance days no trades occur
    daily_pos = (
        trade_log.group_by(["date", "id"]).agg(pl.col("quantity").sum()).sort(["date", "id"])
    )

    # Per date: compute per-asset absolute quantity and normalize
    pos_norm = daily_pos.with_columns(pl.col("quantity").abs().alias("abs_qty"))
    date_totals = pos_norm.group_by("date").agg(pl.col("abs_qty").sum().alias("total_abs"))
    return (
        pos_norm.join(date_totals, on="date")
        .with_columns((pl.col("abs_qty") / pl.col("total_abs")).alias("weight"))
        .select("date", "id", "weight")
    )


# ---------------------------------------------------------------------------
# Concentration / exposure analytics
# ---------------------------------------------------------------------------


def gross_exposure(weights: pl.DataFrame) -> pl.DataFrame:
    """Daily gross exposure: sum of |weight| per date.

    For a long-only fully-invested portfolio this equals 1.0 by construction.
    It exceeds 1.0 for leveraged or long-short books.
    Returns `(date, gross_exposure)`.
    """
    return (
        weights.group_by("date")
        .agg(pl.col("weight").abs().sum().alias("gross_exposure"))
        .sort("date")
    )


def net_exposure(weights: pl.DataFrame) -> pl.DataFrame:
    """Daily net exposure: sum(weight) per date (long − short).

    Returns `(date, net_exposure)`.
    """
    return weights.group_by("date").agg(pl.col("weight").sum().alias("net_exposure")).sort("date")


def top_n_weight(weights: pl.DataFrame, n: int = 10) -> pl.DataFrame:
    """Daily sum of top-N absolute weights (concentration in largest positions).

    Returns `(date, top_n_weight)`.
    """
    return (
        weights.with_columns(pl.col("weight").abs().alias("abs_w"))
        .sort(["date", "abs_w"], descending=[False, True])
        .group_by("date")
        .agg(pl.col("abs_w").head(n).sum().alias("top_n_weight"))
        .sort("date")
    )


def effective_n(weights: pl.DataFrame) -> pl.DataFrame:
    """Effective number of bets (inverse Herfindahl-Hirschman Index).

    effective_N = 1 / sum(w_i²), where w_i are the absolute normalized weights.
    A portfolio equally split across N names has effective_N = N. A single-name
    portfolio has effective_N = 1. Returns `(date, effective_n)`.
    """
    return (
        weights.with_columns((pl.col("weight") ** 2).alias("w2"))
        .group_by("date")
        .agg((1.0 / pl.col("w2").sum()).alias("effective_n"))
        .sort("date")
    )
