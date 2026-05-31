"""Production backtest engine.

``ProductionBacktestEngine`` is an **opt-in** replacement for the simple
``BacktestEngine``.  Every feature defaults to the legacy behavior so that
callers upgrading incrementally see no change until they enable a feature.

Key differences from the legacy engine
---------------------------------------
1. **Transaction costs** — commissions, half-spread, exchange fees charged on
   each trade.
2. **Market-impact slippage** — square-root model scaled by ADV.
3. **Execution lag** — target weights decided at bar *t* execute at bar *t+1*
   (``execution_lag=1``); removes same-bar look-ahead.
4. **Universe / tradability masking** — non-tradeable assets get zero weight;
   existing positions are liquidated.
5. **Corporate actions** — splits and dividends adjust positions and cash.
6. **Price-based accounting** — share-level bookkeeping using ``prices.close``;
   NAV = ``shares @ prices + cash``.
7. **Position / portfolio constraints** — per-name caps, gross/net exposure.
8. **Borrow / short-availability costs** — daily borrow charge on shorts.
9. **Per-trade fill log** — enriched ``fill_log`` result field.
10. **Calendar-aware rebalancing** — rebalance only on scheduled sessions.
11. **Config validation** — universe alignment checked at construction.

Architecture
------------
The engine runs in **share-space** when ``prices`` is provided, falling back
to **weight-space** (legacy mode) otherwise.  All costs are deducted from
the cash ledger in share-space mode and from NAV directly in weight-space mode.

New ``BacktestResult`` fields (added with defaults so old construction sites
still work):
- ``fill_log``:    DataFrame (date, id, shares, fill_price, cost, slippage).
- ``cash_history``: DataFrame (date, cash).  Empty in weight-space mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from etl.source import to_matrix

from .accounting import (
    execute_trades,
    nav_from_shares,
    target_weights_to_share_deltas,
    weights_from_shares,
)
from .constraints import apply_all_constraints, apply_short_availability_cap
from .corporate import apply_corporate_actions, build_action_index
from .costs import (
    compute_borrow_cost,
    compute_cash_interest,
    compute_financing_cost,
    compute_transaction_costs,
)
from .engine import BacktestResult, _softmax
from .signals import SignalFrame
from .slippage import compute_slippage, fill_price_with_slippage
from .vectorized import run_weight_space_vectorized, weight_space_eligible

if TYPE_CHECKING:
    pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProductionBacktestConfig:
    """Full-featured backtest configuration.

    Fields that are ``None`` disable the corresponding feature (default path
    matches legacy behavior).

    Parameters
    ----------
    n_assets, n_dates:
        Used for validation; must match the data passed to ``run``.
    initial_cash:
        Starting NAV / cash balance.
    rebalance_every:
        Rebalance on every *n*-th bar (bar-count cadence, not calendar-aware).
        Set ``rebalance_dates`` to override with an explicit session set.
    rebalance_dates:
        Explicit set of dates on which to rebalance.  Overrides
        ``rebalance_every`` when non-empty.
    execution_lag:
        Bars between weight decision and fill.  ``0`` = same-bar (legacy);
        ``1`` = next-bar (removes look-ahead).
    enable_costs:
        Charge transaction costs when trading.
    enable_slippage:
        Apply square-root market-impact model.
    enable_universe_mask:
        Enforce tradability constraints from the ``universe_mask`` dataset.
    enable_corporate_actions:
        Apply split / dividend adjustments.
    enable_price_accounting:
        Use share-level bookkeeping with real prices.
    enable_borrow_costs:
        Charge daily borrow cost on short positions.
    enable_cash_interest:
        Credit/charge daily interest on the cash balance.
    cash_annual_rate:
        Annual interest rate applied to cash when ``enable_cash_interest=True``.
    min_weight, max_weight:
        Per-portfolio scalar weight bounds; applied uniformly to all assets.
        Use ``None`` to disable.  For per-asset bounds pass arrays via the
        ``run`` method.
    max_gross_exposure:
        Gross exposure cap (``sum(|w|)``).  ``1.0`` = fully-invested long-only.
    max_net_exposure:
        Net exposure cap (``|sum(w)|``).  ``1.0`` = no net-short constraint.
    validate_universe:
        Assert that the signal and returns frames share the same asset ids.
    enable_short_availability_gating:
        Gate short positions by the ``borrow_rates`` dataset: forbid shorts on
        names with ``shortable=False``, cap each short's market value at its
        ``loan_availability``, and charge a daily borrow cost (per-asset
        ``borrow_rate_bps``) on the surviving shorts.  Requires ``borrow_rates``
        to be passed to ``run``.  Default ``False`` — off; output is identical
        to pre-feature runs.
    enable_financing:
        Deduct per-period financing costs (borrow + leverage funding) from NAV.
        Default ``False`` — feature is off; output is identical to pre-feature runs.
    borrow_rate_annual:
        Annualized rate applied to aggregate short market value when
        ``enable_financing=True`` (fraction, e.g. ``0.005`` for 50 bps).
        Neutral default ``0.0`` ensures no cost even if flag is accidentally set.
    funding_rate_annual:
        Annualized rate applied to leveraged exposure above 1× gross when
        ``enable_financing=True`` (fraction, e.g. ``0.02`` for 200 bps).
        Neutral default ``0.0`` ensures no cost even if flag is accidentally set.
    """

    n_assets: int
    n_dates: int
    initial_cash: float = 1_000_000.0
    rebalance_every: int = 1
    rebalance_dates: frozenset[date] = field(default_factory=frozenset)
    execution_lag: int = 0
    enable_costs: bool = False
    enable_slippage: bool = False
    enable_universe_mask: bool = False
    enable_corporate_actions: bool = False
    enable_price_accounting: bool = False
    enable_borrow_costs: bool = False
    enable_cash_interest: bool = False
    cash_annual_rate: float = 0.05
    enable_short_availability_gating: bool = False
    enable_financing: bool = False
    borrow_rate_annual: float = 0.0
    funding_rate_annual: float = 0.0
    min_weight: float | None = None
    max_weight: float | None = None
    max_gross_exposure: float | None = None
    max_net_exposure: float | None = None
    validate_universe: bool = True


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProductionBacktestEngine:
    """Production backtest engine; see module docstring for feature overview."""

    config: ProductionBacktestConfig

    def run(
        self,
        returns: pl.DataFrame,
        signals: SignalFrame,
        *,
        prices: pl.DataFrame | None = None,
        transaction_costs: pl.DataFrame | None = None,
        universe_mask: pl.DataFrame | None = None,
        corporate_actions: pl.DataFrame | None = None,
        borrow_rates: pl.DataFrame | None = None,
        min_weight_per_asset: np.ndarray | None = None,
        max_weight_per_asset: np.ndarray | None = None,
    ) -> BacktestResult:
        """Run the production backtest.

        Parameters
        ----------
        returns:
            Long-format (date, id, return) in percent units.
        signals:
            Long-format (date, id, signal) SignalFrame.
        prices:
            Optional long-format (date, id, close, adv_20, ...) from
            ``gen_prices``.  Required when ``enable_price_accounting=True``,
            ``enable_slippage=True``, or ``enable_corporate_actions=True``.
        transaction_costs:
            Optional long-format (date, id, commission_bps, ...) from
            ``gen_transaction_costs``.  Required when ``enable_costs=True``.
        universe_mask:
            Optional long-format (date, id, tradable, ...) from
            ``gen_universe_mask``.  Required when ``enable_universe_mask=True``.
        corporate_actions:
            Optional wide-format (ex_date, id, action_type, ...) from
            ``gen_corporate_actions``.  Required when
            ``enable_corporate_actions=True``.
        borrow_rates:
            Optional long-format (date, id, borrow_rate_bps, ...) from
            ``gen_borrow_rates``.  Required when ``enable_borrow_costs=True``.
        min_weight_per_asset, max_weight_per_asset:
            Per-asset weight bounds (n_assets,).  Override the scalar config
            bounds.

        Returns
        -------
        BacktestResult
            Standard result plus ``fill_log`` and ``cash_history``.
        """
        cfg = self.config
        R, dates = to_matrix(returns, "return")
        S, _ = to_matrix(signals.df, "signal")
        n_dates, n_assets = R.shape

        if cfg.validate_universe:
            _validate_universe(returns, signals.df)

        mats = _preprocess_inputs(
            prices, transaction_costs, universe_mask, borrow_rates, n_dates, n_assets
        )
        close_mat = mats["close"]
        adv_mat = mats["adv_20"]
        comm_mat = mats["commission_bps"]
        spread_mat = mats["half_spread_bps"]
        fee_mat = mats["exchange_fee_bps"]
        mincomm_mat = mats["min_commission"]
        impact_mat = mats["impact_coef"]
        tradable_mat = mats["tradable"]
        borrow_mat = mats["borrow_rate_bps"]
        loan_avail_mat = mats["loan_availability"]
        shortable_mat = mats["shortable"]

        if cfg.enable_short_availability_gating and (
            shortable_mat is None or loan_avail_mat is None or borrow_mat is None
        ):
            raise ValueError(
                "enable_short_availability_gating requires borrow_rates with "
                "shortable, loan_availability, and borrow_rate_bps columns"
            )

        action_index: dict = {}
        if cfg.enable_corporate_actions and corporate_actions is not None:
            action_index = build_action_index(corporate_actions)

        lb, ub = _resolve_weight_bounds(cfg, n_assets, min_weight_per_asset, max_weight_per_asset)

        if weight_space_eligible(cfg, close_mat):
            return _run_vectorized_weight_space(
                cfg,
                dates,
                R,
                S,
                tradable_mat,
                adv_mat,
                comm_mat,
                spread_mat,
                fee_mat,
                mincomm_mat,
                impact_mat,
                lb,
                ub,
                n_dates,
                n_assets,
            )

        shares, cash, nav = _init_state(cfg, close_mat, n_assets)
        weights = np.zeros(n_assets)
        pending_target: np.ndarray | None = None

        cumulative_financing_drag: float = 0.0

        # dt per period: 1/252 for daily data (calendar-day periods between
        # consecutive dates are counted when available, capped to 5/252 to avoid
        # blowing up over weekends / holidays).
        dt_default = 1.0 / 252.0

        nav_hist: list[float] = []
        cash_hist: list[float] = []
        trade_dates: list[date] = []
        trade_ids: list[int] = []
        trade_qty: list[float] = []
        fill_dates: list[date] = []
        fill_ids: list[int] = []
        fill_shares: list[float] = []
        fill_prices_list: list[float] = []
        fill_costs_list: list[float] = []
        fill_slippage_list: list[float] = []

        asset_ids = np.arange(n_assets)

        for t in range(n_dates):
            d = dates[t]

            # -------------------------------------------------------------- #
            # 1. Corporate actions (before mark-to-market).
            # -------------------------------------------------------------- #
            if cfg.enable_corporate_actions and d in action_index:
                shares, cash, close_mat = _apply_corporate_actions_at_bar(
                    t, d, action_index, shares, cash, close_mat
                )

            # -------------------------------------------------------------- #
            # 2. Apply pending execution-lag orders (from bar t-1).
            # -------------------------------------------------------------- #
            if cfg.execution_lag > 0 and pending_target is not None:
                target = pending_target
                pending_target = None
                weights, shares, cash, nav = _execute_rebalance(
                    t,
                    d,
                    target,
                    weights,
                    shares,
                    cash,
                    nav,
                    close_mat,
                    adv_mat,
                    comm_mat,
                    spread_mat,
                    fee_mat,
                    mincomm_mat,
                    impact_mat,
                    asset_ids,
                    cfg,
                    trade_dates,
                    trade_ids,
                    trade_qty,
                    fill_dates,
                    fill_ids,
                    fill_shares,
                    fill_prices_list,
                    fill_costs_list,
                    fill_slippage_list,
                )

            # -------------------------------------------------------------- #
            # 3. Decide whether to rebalance at bar t.
            # -------------------------------------------------------------- #
            should_rebalance = _should_rebalance(t, d, cfg)
            if should_rebalance:
                tradable_t = tradable_mat[t] if tradable_mat is not None else None
                raw_target = _compute_target_weights(S[t], cfg, tradable_t, lb, ub)

                if cfg.enable_short_availability_gating:
                    raw_target = apply_short_availability_cap(
                        raw_target,
                        shortable=cast("np.ndarray", shortable_mat)[t],
                        loan_availability=cast("np.ndarray", loan_avail_mat)[t],
                        nav=nav,
                    )

                if cfg.execution_lag > 0:
                    # Store and execute next bar.
                    pending_target = raw_target
                else:
                    # Execute immediately (same-bar fill, legacy default).
                    weights, shares, cash, nav = _execute_rebalance(
                        t,
                        d,
                        raw_target,
                        weights,
                        shares,
                        cash,
                        nav,
                        close_mat,
                        adv_mat,
                        comm_mat,
                        spread_mat,
                        fee_mat,
                        mincomm_mat,
                        impact_mat,
                        asset_ids,
                        cfg,
                        trade_dates,
                        trade_ids,
                        trade_qty,
                        fill_dates,
                        fill_ids,
                        fill_shares,
                        fill_prices_list,
                        fill_costs_list,
                        fill_slippage_list,
                    )

            # -------------------------------------------------------------- #
            # 4. Mark to market: propagate NAV / positions by today's returns.
            # -------------------------------------------------------------- #
            r = R[t] / 100.0
            weights, shares, nav, close_mat = _mark_to_market(
                t, r, weights, shares, nav, cash, close_mat, n_dates
            )

            # -------------------------------------------------------------- #
            # 5. Accrue daily financing costs.
            # -------------------------------------------------------------- #
            nav, cash, cumulative_financing_drag = _accrue_daily_costs(
                t,
                dates,
                weights,
                shares,
                nav,
                cash,
                borrow_mat,
                cfg,
                cumulative_financing_drag,
                dt_default,
            )

            nav_hist.append(nav)
            cash_hist.append(cash if shares is not None else 0.0)

        return _assemble_result(
            dates,
            nav_hist,
            cash_hist,
            trade_dates,
            trade_ids,
            trade_qty,
            fill_dates,
            fill_ids,
            fill_shares,
            fill_prices_list,
            fill_costs_list,
            fill_slippage_list,
            weights,
            shares,
            close_mat,
            nav,
            n_dates,
            n_assets,
            cumulative_financing_drag,
        )


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #


def _preprocess_inputs(
    prices: pl.DataFrame | None,
    transaction_costs: pl.DataFrame | None,
    universe_mask: pl.DataFrame | None,
    borrow_rates: pl.DataFrame | None,
    n_dates: int,
    n_assets: int,
) -> dict[str, np.ndarray | None]:
    """Pivot all optional long-format DataFrames to dense (n_dates, n_assets) matrices.

    Pre-process once here so the inner loop is O(n_assets) per bar.

    Each DataFrame is sorted once and reshaped in a single NumPy call rather
    than pivoting once per column, which eliminates N-1 redundant Polars sort +
    pivot passes and is the dominant cost at large (n_assets, n_dates).
    """
    result: dict[str, np.ndarray | None] = {}

    if prices is not None:
        pm = _df_to_multi_matrix(prices, ["close", "adv_20"], n_dates, n_assets, copy=True)
        result["close"] = pm.get("close")
        result["adv_20"] = pm.get("adv_20")
    else:
        result["close"] = None
        result["adv_20"] = None

    if transaction_costs is not None:
        tc_cols = [
            "commission_bps",
            "half_spread_bps",
            "exchange_fee_bps",
            "min_commission",
            "impact_coef",
        ]
        tm = _df_to_multi_matrix(transaction_costs, tc_cols, n_dates, n_assets, copy=False)
        result["commission_bps"] = tm.get("commission_bps")
        result["half_spread_bps"] = tm.get("half_spread_bps")
        result["exchange_fee_bps"] = tm.get("exchange_fee_bps")
        result["min_commission"] = tm.get("min_commission")
        result["impact_coef"] = tm.get("impact_coef")
    else:
        result["commission_bps"] = None
        result["half_spread_bps"] = None
        result["exchange_fee_bps"] = None
        result["min_commission"] = None
        result["impact_coef"] = None

    result["tradable"] = _to_bool_matrix_or_none(universe_mask, "tradable", n_dates, n_assets)

    if borrow_rates is not None:
        bm = _df_to_multi_matrix(
            borrow_rates, ["borrow_rate_bps", "loan_availability"], n_dates, n_assets, copy=False
        )
        result["borrow_rate_bps"] = bm.get("borrow_rate_bps")
        result["loan_availability"] = bm.get("loan_availability")
        result["shortable"] = _to_bool_matrix_or_none(borrow_rates, "shortable", n_dates, n_assets)
    else:
        result["borrow_rate_bps"] = None
        result["loan_availability"] = None
        result["shortable"] = None

    return result


def _resolve_weight_bounds(
    cfg: ProductionBacktestConfig,
    n_assets: int,
    min_weight_per_asset: np.ndarray | None,
    max_weight_per_asset: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Derive per-asset lower/upper weight bound arrays from config + overrides."""
    if min_weight_per_asset is not None:
        lb = min_weight_per_asset
    elif cfg.min_weight is not None:
        lb = np.full(n_assets, cfg.min_weight)
    else:
        lb = None

    if max_weight_per_asset is not None:
        ub = max_weight_per_asset
    elif cfg.max_weight is not None:
        ub = np.full(n_assets, cfg.max_weight)
    else:
        ub = None

    return lb, ub


def _init_state(
    cfg: ProductionBacktestConfig,
    close_mat: np.ndarray | None,
    n_assets: int,
) -> tuple[np.ndarray | None, float, float]:
    """Return initial (shares, cash, nav) depending on accounting mode."""
    if cfg.enable_price_accounting and close_mat is not None:
        # Share-space mode: start fully in cash.
        return np.zeros(n_assets), float(cfg.initial_cash), float(cfg.initial_cash)
    return None, 0.0, float(cfg.initial_cash)


# --------------------------------------------------------------------------- #
# Per-bar helpers
# --------------------------------------------------------------------------- #


def _apply_corporate_actions_at_bar(
    t: int,
    d: date,
    action_index: dict,
    shares: np.ndarray | None,
    cash: float,
    close_mat: np.ndarray | None,
) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    """Apply all corporate actions scheduled for date ``d``.

    Propagates the price adjustment back into ``close_mat[t]`` for the rest
    of the simulation (point-in-time: only the current bar is patched).

    Returns updated (shares, cash, close_mat).
    """
    if shares is None or close_mat is None:
        return shares, cash, close_mat

    todays = action_index[d]
    ids_ca = [a["id"] for a in todays]
    types_ca = [a["action_type"] for a in todays]
    ratios_ca = [a["split_ratio"] for a in todays]
    amounts_ca = [a["cash_amount"] for a in todays]

    cur_prices = close_mat[t]
    shares, cur_prices, cash = apply_corporate_actions(
        shares,
        cur_prices,
        cash,
        ids_ca,
        types_ca,
        ratios_ca,
        amounts_ca,
    )
    close_mat[t] = cur_prices
    return shares, cash, close_mat


def _compute_target_weights(
    signal_row: np.ndarray,
    cfg: ProductionBacktestConfig,
    tradable_t: np.ndarray | None,
    lb: np.ndarray | None,
    ub: np.ndarray | None,
) -> np.ndarray:
    """Convert raw signal into constrained target weights."""
    raw_target = _softmax(signal_row)
    return apply_all_constraints(
        raw_target,
        tradable=tradable_t if cfg.enable_universe_mask else None,
        min_weight=lb,
        max_weight=ub,
        max_gross=cfg.max_gross_exposure,
        max_net=cfg.max_net_exposure,
    )


def _mark_to_market(
    t: int,
    r: np.ndarray,
    weights: np.ndarray,
    shares: np.ndarray | None,
    nav: float,
    cash: float,
    close_mat: np.ndarray | None,
    n_dates: int,
) -> tuple[np.ndarray, np.ndarray | None, float, np.ndarray | None]:
    """Propagate NAV and weights by today's returns.

    Share-space: new prices = close * (1 + r); NAV = shares @ new_prices + cash.
    Weight-space: NAV *= 1 + portfolio_return; weights drift with returns.

    Returns updated (weights, shares, nav, close_mat).
    """
    if shares is not None and close_mat is not None:
        # Share-space: prices move; NAV = shares @ new_prices + cash.
        new_prices = close_mat[t] * (1.0 + r)
        if t + 1 < n_dates:
            close_mat[t + 1] = new_prices  # propagate for next bar
        nav = nav_from_shares(shares, close_mat[t], cash)
        # Update weights for constraint checks on next rebalance.
        weights = weights_from_shares(shares, close_mat[t], nav)
    else:
        # Weight-space: legacy propagation.
        port_ret = float(weights @ r)
        nav *= 1.0 + port_ret
        drifted = weights * (1.0 + r)
        total = drifted.sum()
        if total > 0.0:
            weights = drifted / total

    return weights, shares, nav, close_mat


def _accrue_daily_costs(
    t: int,
    dates: list[date],
    weights: np.ndarray,
    shares: np.ndarray | None,
    nav: float,
    cash: float,
    borrow_mat: np.ndarray | None,
    cfg: ProductionBacktestConfig,
    cumulative_financing_drag: float,
    dt_default: float,
) -> tuple[float, float, float]:
    """Accrue borrow costs, cash interest, and financing drag for bar ``t``.

    Returns updated (nav, cash, cumulative_financing_drag).
    """
    if cfg.enable_borrow_costs and borrow_mat is not None:
        borrow = compute_borrow_cost(weights, nav, borrow_mat[t])
        nav -= borrow
        if shares is not None:
            cash -= borrow

    if cfg.enable_short_availability_gating and borrow_mat is not None:
        borrow = compute_borrow_cost(weights, nav, borrow_mat[t])
        nav -= borrow
        if shares is not None:
            cash -= borrow

    if cfg.enable_cash_interest and shares is not None:
        interest = compute_cash_interest(cash, cfg.cash_annual_rate)
        cash += interest
        nav += interest

    if cfg.enable_financing:
        # Compute dt: calendar days between consecutive dates, capped
        # at 5 trading days to avoid inflating costs over long gaps.
        if t + 1 < len(dates):
            dt = min(
                (dates[t + 1] - dates[t]).days / 365.0,
                5.0 / 252.0,
            )
        else:
            dt = dt_default
        financing = compute_financing_cost(
            weights,
            nav,
            cfg.borrow_rate_annual,
            cfg.funding_rate_annual,
            dt,
        )
        nav -= financing
        if shares is not None:
            cash -= financing
        cumulative_financing_drag += financing

    return nav, cash, cumulative_financing_drag


# --------------------------------------------------------------------------- #
# Vectorized weight-space fast path
# --------------------------------------------------------------------------- #


def _rebalance_schedule(
    dates: list[date],
    cfg: ProductionBacktestConfig,
) -> np.ndarray:
    """Boolean ``(n_dates,)`` rebalance mask matching ``_should_rebalance``.

    With an explicit ``rebalance_dates`` set, a bar rebalances iff its date is
    in the set; otherwise the bar-count cadence ``t % rebalance_every == 0`` is
    used.
    """
    n_dates = len(dates)
    if cfg.rebalance_dates:
        return np.array([d in cfg.rebalance_dates for d in dates], dtype=bool)
    idx = np.arange(n_dates)
    return (idx % cfg.rebalance_every) == 0


def _run_vectorized_weight_space(
    cfg: ProductionBacktestConfig,
    dates: list[date],
    R: np.ndarray,
    S: np.ndarray,
    tradable_mat: np.ndarray | None,
    adv_mat: np.ndarray | None,
    comm_mat: np.ndarray | None,
    spread_mat: np.ndarray | None,
    fee_mat: np.ndarray | None,
    mincomm_mat: np.ndarray | None,
    impact_mat: np.ndarray | None,
    lb: np.ndarray | None,
    ub: np.ndarray | None,
    n_dates: int,
    n_assets: int,
) -> BacktestResult:
    """Run the batched weight-space core and assemble the ``BacktestResult``.

    Produces output byte-identical to the event loop for the configs accepted
    by :func:`weight_space_eligible`.  The result is assembled through the same
    :func:`_assemble_result` path the loop uses, so ``cash_history`` (one zero
    row per date in weight-space mode), the empty ``fill_log``, and
    ``financing_drag`` (always ``0.0`` here — financing is an excluded feature)
    all match the loop frame-for-frame.
    """
    rebal = _rebalance_schedule(dates, cfg)
    nav_hist, final_weights, trade_dates, trade_ids, trade_qty = run_weight_space_vectorized(
        cfg,
        dates,
        R,
        S,
        rebal=rebal,
        tradable_mat=tradable_mat,
        adv_mat=adv_mat,
        comm_mat=comm_mat,
        spread_mat=spread_mat,
        fee_mat=fee_mat,
        mincomm_mat=mincomm_mat,
        impact_mat=impact_mat,
        lb=lb,
        ub=ub,
    )
    cash_hist = [0.0] * n_dates
    return _assemble_result(
        dates,
        nav_hist,
        cash_hist,
        trade_dates,
        trade_ids,
        trade_qty,
        [],
        [],
        [],
        [],
        [],
        [],
        final_weights,
        None,
        None,
        nav_hist[-1] if nav_hist else float(cfg.initial_cash),
        n_dates,
        n_assets,
        0.0,
    )


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #


def _assemble_result(
    dates: list[date],
    nav_hist: list[float],
    cash_hist: list[float],
    trade_dates: list[date],
    trade_ids: list[int],
    trade_qty: list[float],
    fill_dates: list[date],
    fill_ids: list[int],
    fill_shares: list[float],
    fill_prices_list: list[float],
    fill_costs_list: list[float],
    fill_slippage_list: list[float],
    weights: np.ndarray,
    shares: np.ndarray | None,
    close_mat: np.ndarray | None,
    nav: float,
    n_dates: int,
    n_assets: int,
    cumulative_financing_drag: float,
) -> BacktestResult:
    """Build the final BacktestResult from accumulated history lists."""
    fill_log = pl.DataFrame(
        {
            "date": fill_dates,
            "id": fill_ids,
            "shares": fill_shares,
            "fill_price": fill_prices_list,
            "cost": fill_costs_list,
            "slippage": fill_slippage_list,
        }
    )
    cash_history = pl.DataFrame({"date": dates, "cash": cash_hist})

    return BacktestResult(
        nav_history=pl.DataFrame({"date": dates, "nav": nav_hist}),
        trade_log=pl.DataFrame({"date": trade_dates, "id": trade_ids, "quantity": trade_qty}),
        final_positions=weights
        if shares is None
        else weights_from_shares(
            shares, close_mat[n_dates - 1] if close_mat is not None else np.zeros(n_assets), nav
        ),
        fill_log=fill_log,
        cash_history=cash_history,
        financing_drag=cumulative_financing_drag,
    )


# --------------------------------------------------------------------------- #
# Rebalance helpers
# --------------------------------------------------------------------------- #


def _should_rebalance(
    t: int,
    d: date,
    cfg: ProductionBacktestConfig,
) -> bool:
    if cfg.rebalance_dates:
        return d in cfg.rebalance_dates
    return t % cfg.rebalance_every == 0


def _compute_fill_prices(
    t: int,
    target: np.ndarray,
    weights: np.ndarray,
    nav: float,
    mid_prices: np.ndarray,
    adv_t: np.ndarray,
    impact_t: np.ndarray,
    cfg: ProductionBacktestConfig,
) -> np.ndarray:
    """Build per-asset fill prices, optionally applying slippage."""
    n_assets = len(target)
    if not cfg.enable_slippage:
        return mid_prices.copy()

    trade_value_est = (target - weights) * nav  # rough notional estimate
    return np.array(
        [
            fill_price_with_slippage(
                mid_prices[i],
                trade_value_est[i],
                adv_t[i],
                impact_t[i],
            )
            for i in range(n_assets)
        ]
    )


def _record_fill_log(
    d: date,
    share_deltas: np.ndarray,
    fill_prices: np.ndarray,
    trade_value: np.ndarray,
    tc_cost: float,
    slip_cost: float,
    asset_ids: np.ndarray,
    fill_dates: list,
    fill_ids: list,
    fill_shares_list: list,
    fill_prices_list: list,
    fill_costs_list: list,
    fill_slippage_list: list,
) -> None:
    """Append per-trade entries to the fill log lists (share-space only)."""
    traded_mask = share_deltas != 0.0
    abs_total = float(np.abs(trade_value[traded_mask]).sum())
    for i in asset_ids[traded_mask]:
        fill_dates.append(d)
        fill_ids.append(int(i))
        fill_shares_list.append(float(share_deltas[i]))
        fill_prices_list.append(float(fill_prices[i]))
        # Distribute total cost proportionally to trade notional.
        frac = abs(trade_value[i]) / abs_total if abs_total > 0.0 else 0.0
        fill_costs_list.append(frac * tc_cost)
        fill_slippage_list.append(frac * slip_cost)


def _rebalance_share_space(
    t: int,
    d: date,
    target: np.ndarray,
    weights: np.ndarray,
    shares: np.ndarray,
    cash: float,
    nav: float,
    close_mat: np.ndarray,
    adv_mat: np.ndarray | None,
    comm_mat: np.ndarray | None,
    spread_mat: np.ndarray | None,
    fee_mat: np.ndarray | None,
    mincomm_mat: np.ndarray | None,
    impact_mat: np.ndarray | None,
    asset_ids: np.ndarray,
    cfg: ProductionBacktestConfig,
    fill_dates: list,
    fill_ids: list,
    fill_shares_list: list,
    fill_prices_list: list,
    fill_costs_list: list,
    fill_slippage_list: list,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Execute a rebalance in share-space mode.

    Returns updated (new_weights, shares, cash, nav).
    """
    n_assets = len(target)
    mid_prices = close_mat[t]
    adv_t = adv_mat[t] if adv_mat is not None else np.zeros(n_assets)
    impact_t = impact_mat[t] if impact_mat is not None else np.zeros(n_assets)

    fill_prices = _compute_fill_prices(t, target, weights, nav, mid_prices, adv_t, impact_t, cfg)

    share_deltas = target_weights_to_share_deltas(target, shares, fill_prices, nav)
    trade_value = share_deltas * fill_prices

    tc_cost = 0.0
    slip_cost = 0.0
    if cfg.enable_costs and comm_mat is not None:
        tc_cost = compute_transaction_costs(
            trade_value,
            comm_mat[t],
            cast("np.ndarray", spread_mat)[t],
            cast("np.ndarray", fee_mat)[t],
            cast("np.ndarray", mincomm_mat)[t],
        )
    if cfg.enable_slippage:
        slip_cost = compute_slippage(trade_value, adv_t, impact_t)

    total_cost = tc_cost + slip_cost
    shares, cash = execute_trades(shares, share_deltas, fill_prices, cash, total_cost)
    nav = nav_from_shares(shares, mid_prices, cash)
    new_weights = weights_from_shares(shares, mid_prices, nav)

    _record_fill_log(
        d,
        share_deltas,
        fill_prices,
        trade_value,
        tc_cost,
        slip_cost,
        asset_ids,
        fill_dates,
        fill_ids,
        fill_shares_list,
        fill_prices_list,
        fill_costs_list,
        fill_slippage_list,
    )

    return new_weights, shares, cash, nav


def _rebalance_weight_space(
    t: int,
    target: np.ndarray,
    weights: np.ndarray,
    nav: float,
    adv_mat: np.ndarray | None,
    comm_mat: np.ndarray | None,
    spread_mat: np.ndarray | None,
    fee_mat: np.ndarray | None,
    mincomm_mat: np.ndarray | None,
    impact_mat: np.ndarray | None,
    cfg: ProductionBacktestConfig,
) -> tuple[np.ndarray, float]:
    """Execute a rebalance in weight-space mode (legacy, no prices).

    Returns updated (new_weights, nav).
    """
    n_assets = len(target)
    deltas = target - weights
    trade_value = deltas * nav
    adv_t = adv_mat[t] if adv_mat is not None else np.zeros(n_assets)
    impact_t = impact_mat[t] if impact_mat is not None else np.zeros(n_assets)

    tc_cost = 0.0
    slip_cost = 0.0
    if cfg.enable_costs and comm_mat is not None:
        tc_cost = compute_transaction_costs(
            trade_value,
            comm_mat[t],
            cast("np.ndarray", spread_mat)[t],
            cast("np.ndarray", fee_mat)[t],
            cast("np.ndarray", mincomm_mat)[t],
        )
    if cfg.enable_slippage:
        slip_cost = compute_slippage(trade_value, adv_t, impact_t)

    total_cost = tc_cost + slip_cost
    nav -= total_cost
    return target, nav


def _execute_rebalance(
    t: int,
    d: date,
    target: np.ndarray,
    weights: np.ndarray,
    shares: np.ndarray | None,
    cash: float,
    nav: float,
    close_mat: np.ndarray | None,
    adv_mat: np.ndarray | None,
    comm_mat: np.ndarray | None,
    spread_mat: np.ndarray | None,
    fee_mat: np.ndarray | None,
    mincomm_mat: np.ndarray | None,
    impact_mat: np.ndarray | None,
    asset_ids: np.ndarray,
    cfg: ProductionBacktestConfig,
    trade_dates: list,
    trade_ids: list,
    trade_qty: list,
    fill_dates: list,
    fill_ids: list,
    fill_shares_list: list,
    fill_prices_list: list,
    fill_costs_list: list,
    fill_slippage_list: list,
) -> tuple[np.ndarray, np.ndarray | None, float, float]:
    """Execute a rebalance: compute deltas, apply costs, update state.

    Dispatches to share-space or weight-space mode, then appends to the trade log.

    Returns updated (weights, shares, cash, nav).
    """
    if shares is not None and close_mat is not None:
        new_weights, shares, cash, nav = _rebalance_share_space(
            t,
            d,
            target,
            weights,
            shares,
            cash,
            nav,
            close_mat,
            adv_mat,
            comm_mat,
            spread_mat,
            fee_mat,
            mincomm_mat,
            impact_mat,
            asset_ids,
            cfg,
            fill_dates,
            fill_ids,
            fill_shares_list,
            fill_prices_list,
            fill_costs_list,
            fill_slippage_list,
        )
    else:
        new_weights, nav = _rebalance_weight_space(
            t,
            target,
            weights,
            nav,
            adv_mat,
            comm_mat,
            spread_mat,
            fee_mat,
            mincomm_mat,
            impact_mat,
            cfg,
        )

    # Record trade log (legacy format — quantity = notional traded).
    n_assets = len(target)
    trade_dates.extend([d] * n_assets)
    trade_ids.extend(asset_ids.tolist())
    deltas_for_log = target - weights
    trade_qty.extend((deltas_for_log * nav).tolist())

    return new_weights, shares, cash, nav


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def _df_to_multi_matrix(
    df: pl.DataFrame,
    cols: list[str],
    n_dates: int,
    n_assets: int,
    *,
    copy: bool,
) -> dict[str, np.ndarray]:
    """Sort once and reshape to (n_dates, n_assets) for each requested column.

    Avoids repeating the Polars sort+pivot cost once per column — the dominant
    expense in ``_preprocess_inputs`` at large grid points.  Columns absent
    from ``df`` are silently omitted from the returned dict.

    The reshape assumes the frame is a complete (date × id) grid with no
    missing cells and no duplicate (date, id) pairs.  All datasets produced by
    ``write_all`` satisfy this invariant.

    Parameters
    ----------
    df:
        Long-format frame with at least ``date``, ``id``, and the requested
        value columns.
    cols:
        Value columns to extract; subset intersection with ``df.columns`` is
        used so callers do not need to filter beforehand.
    n_dates, n_assets:
        Expected grid dimensions (used for the reshape).
    copy:
        When ``True``, return writable copies so callers can mutate in-place
        (e.g. corporate-action patches).  When ``False``, the slices may share
        the underlying buffer — safe for read-only consumers.
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        return {}
    sorted_df = df.select(["date", "id", *present]).sort(["date", "id"])
    block = sorted_df.select(present).to_numpy()  # (n_dates * n_assets, n_cols)
    block_3d = block.reshape(n_dates, n_assets, len(present))
    if copy:
        return {col: block_3d[:, :, i].copy() for i, col in enumerate(present)}
    return {col: block_3d[:, :, i] for i, col in enumerate(present)}


def _to_matrix_or_none(
    df: pl.DataFrame | None,
    col: str,
    n_dates: int,
    n_assets: int,
) -> np.ndarray | None:
    """Pivot a long-format frame to a dense (n_dates, n_assets) matrix.

    Returns ``None`` if ``df`` is ``None`` or doesn't have the column.
    """
    if df is None or col not in df.columns:
        return None
    mat, _ = to_matrix(df.select("date", "id", col), col)
    return mat.copy()  # mutable copy so corporate actions can update in-place


def _to_bool_matrix_or_none(
    df: pl.DataFrame | None,
    col: str,
    n_dates: int,
    n_assets: int,
) -> np.ndarray | None:
    """Sort once, reshape, and cast to bool — avoids the pivot overhead."""
    if df is None or col not in df.columns:
        return None
    sorted_col = df.select(["date", "id", col]).sort(["date", "id"])[col]
    return sorted_col.to_numpy().reshape(n_dates, n_assets).astype(bool)


def _validate_universe(returns: pl.DataFrame, signals_df: pl.DataFrame) -> None:
    """Assert that returns and signals cover the same set of asset ids.

    Alignment failures produce ``ValueError`` early so the engine doesn't
    silently produce misaligned results.
    """
    ret_ids = set(returns["id"].unique().to_list())
    sig_ids = set(signals_df["id"].unique().to_list())
    if ret_ids != sig_ids:
        only_ret = ret_ids - sig_ids
        only_sig = sig_ids - ret_ids
        msg_parts = []
        if only_ret:
            msg_parts.append(f"ids in returns but not signals: {sorted(only_ret)[:5]}...")
        if only_sig:
            msg_parts.append(f"ids in signals but not returns: {sorted(only_sig)[:5]}...")
        raise ValueError("Universe mismatch: " + "; ".join(msg_parts))
