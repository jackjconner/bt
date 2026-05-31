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
from .constraints import apply_all_constraints
from .corporate import apply_corporate_actions, build_action_index
from .costs import compute_borrow_cost, compute_cash_interest, compute_transaction_costs
from .engine import BacktestResult, _softmax
from .signals import SignalFrame
from .slippage import compute_slippage, fill_price_with_slippage

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

        # ------------------------------------------------------------------ #
        # Pre-process optional data into dense matrices indexed by (t, asset).
        # We sort and pivot once here so the inner loop is O(n_assets).
        # ------------------------------------------------------------------ #
        close_mat = _to_matrix_or_none(prices, "close", n_dates, n_assets)
        adv_mat = _to_matrix_or_none(prices, "adv_20", n_dates, n_assets)
        comm_mat = _to_matrix_or_none(transaction_costs, "commission_bps", n_dates, n_assets)
        spread_mat = _to_matrix_or_none(transaction_costs, "half_spread_bps", n_dates, n_assets)
        fee_mat = _to_matrix_or_none(transaction_costs, "exchange_fee_bps", n_dates, n_assets)
        mincomm_mat = _to_matrix_or_none(transaction_costs, "min_commission", n_dates, n_assets)
        impact_mat = _to_matrix_or_none(transaction_costs, "impact_coef", n_dates, n_assets)
        tradable_mat = _to_bool_matrix_or_none(universe_mask, "tradable", n_dates, n_assets)
        borrow_mat = _to_matrix_or_none(borrow_rates, "borrow_rate_bps", n_dates, n_assets)

        action_index: dict = {}
        if cfg.enable_corporate_actions and corporate_actions is not None:
            action_index = build_action_index(corporate_actions)

        # ------------------------------------------------------------------ #
        # Determine per-asset weight bounds for constraints.
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        if cfg.enable_price_accounting and close_mat is not None:
            # Share-space mode: start fully in cash.
            shares = np.zeros(n_assets)
            cash = float(cfg.initial_cash)
            nav = cash
        else:
            shares = None
            cash = 0.0
            nav = float(cfg.initial_cash)

        weights = np.zeros(n_assets)  # current portfolio weights (or positions)
        pending_target: np.ndarray | None = None  # for execution_lag > 0

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
                todays = action_index[d]
                ids_ca = [a["id"] for a in todays]
                types_ca = [a["action_type"] for a in todays]
                ratios_ca = [a["split_ratio"] for a in todays]
                amounts_ca = [a["cash_amount"] for a in todays]
                if shares is not None and close_mat is not None:
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
                    # Propagate price adjustments back to close_mat for the
                    # rest of the simulation.  This is a simplified adjustment:
                    # splits shift the *current* price; historical prices in
                    # close_mat are not adjusted (point-in-time).
                    close_mat[t] = cur_prices

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
                raw_target = _softmax(S[t])

                # Apply constraints.
                tradable_t = tradable_mat[t] if tradable_mat is not None else None
                raw_target = apply_all_constraints(
                    raw_target,
                    tradable=tradable_t if cfg.enable_universe_mask else None,
                    min_weight=lb,
                    max_weight=ub,
                    max_gross=cfg.max_gross_exposure,
                    max_net=cfg.max_net_exposure,
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

            # -------------------------------------------------------------- #
            # 5. Accrue daily financing costs.
            # -------------------------------------------------------------- #
            if cfg.enable_borrow_costs and borrow_mat is not None:
                borrow = compute_borrow_cost(weights, nav, borrow_mat[t])
                nav -= borrow
                if shares is not None:
                    cash -= borrow

            if cfg.enable_cash_interest and shares is not None:
                interest = compute_cash_interest(cash, cfg.cash_annual_rate)
                cash += interest
                nav += interest

            nav_hist.append(nav)
            cash_hist.append(cash if shares is not None else 0.0)

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
        )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _should_rebalance(
    t: int,
    d: date,
    cfg: ProductionBacktestConfig,
) -> bool:
    if cfg.rebalance_dates:
        return d in cfg.rebalance_dates
    return t % cfg.rebalance_every == 0


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

    Returns updated (weights, shares, cash, nav).
    """
    n_assets = len(target)

    if shares is not None and close_mat is not None:
        # Share-space mode: compute per-asset fill prices including slippage.
        mid_prices = close_mat[t]
        adv_t = adv_mat[t] if adv_mat is not None else np.zeros(n_assets)
        impact_t = impact_mat[t] if impact_mat is not None else np.zeros(n_assets)

        # Build fill prices with per-asset slippage (scalar loop is acceptable
        # because n_assets is typically O(100–10k), not a bottleneck).
        trade_value_est = (target - weights) * nav  # rough notional estimate
        fill_prices = np.array(
            [
                fill_price_with_slippage(
                    mid_prices[i],
                    trade_value_est[i],
                    adv_t[i],
                    impact_t[i],
                )
                if cfg.enable_slippage
                else mid_prices[i]
                for i in range(n_assets)
            ]
        )

        share_deltas = target_weights_to_share_deltas(target, shares, fill_prices, nav)
        trade_value = share_deltas * fill_prices

        # Compute costs.
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

        # Record fill log.
        traded_mask = share_deltas != 0.0
        for i in asset_ids[traded_mask]:
            fill_dates.append(d)
            fill_ids.append(int(i))
            fill_shares_list.append(float(share_deltas[i]))
            fill_prices_list.append(float(fill_prices[i]))
            # Distribute total cost proportionally to trade notional.
            abs_total = float(np.abs(trade_value[traded_mask]).sum())
            frac = abs(trade_value[i]) / abs_total if abs_total > 0.0 else 0.0
            fill_costs_list.append(frac * tc_cost)
            fill_slippage_list.append(frac * slip_cost)

    else:
        # Weight-space mode (no prices): legacy-style cost deduction from NAV.
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
        new_weights = target

    # Record trade log (legacy format — quantity = notional traded).
    trade_dates.extend([d] * n_assets)
    trade_ids.extend(asset_ids.tolist())
    deltas_for_log = target - weights
    trade_qty.extend((deltas_for_log * nav).tolist())

    return new_weights, shares, cash, nav


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
    """Same as ``_to_matrix_or_none`` but casts to bool."""
    if df is None or col not in df.columns:
        return None
    mat, _ = to_matrix(df.select("date", "id", pl.col(col).cast(pl.Float64).alias(col)), col)
    return mat.astype(bool)


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
