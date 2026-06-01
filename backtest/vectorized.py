"""Vectorized weight-space backtest core.

The production / golden / harness path runs ``ProductionBacktestEngine`` in
**weight-space** mode (``enable_price_accounting=False``): no share ledger, no
cash account, ``execution_lag=0``, no corporate actions, no borrow / financing /
cash-interest accruals.  In that mode the per-bar recurrence has a structure
that lets almost everything be hoisted out of the Python loop into batched
NumPy matrix ops:

* **Target weights** ``target[t] = constraints(softmax(S[t]))`` depend only on
  ``S[t]`` — independent across ``t``, so softmax + every constraint is applied
  to the whole ``(n_dates, n_assets)`` matrix in one pass.
* **The held-weight trajectory** is nav-independent: between two rebalances the
  weights are a normalized cumulative drift of the post-rebalance target by the
  realized returns.  We build the full ``weights_held`` matrix with a segmented
  cumulative-product, never touching nav.
* **Per-bar portfolio return** ``port_ret[t] = weights_held[t] @ (R[t]/100)`` is
  a single batched row-dot.

What *cannot* be vectorized is the NAV level itself: each rebalance charges a
transaction / slippage cost computed on ``deltas[t] * nav_t``, and that cost is
**nonlinear** in nav (the ``min_commission`` floor and the square-root slippage
term both break proportionality).  So NAV stays a scalar recurrence — but its
loop body is now pure scalar arithmetic plus one already-vectorized cost eval
per *rebalance* bar, with the heavy matrix work (softmax, constraints, drift,
the delta matrix, the dot products) lifted entirely out of the loop.

The output is **byte-identical** to the event-driven loop for the configs this
fast path accepts (see :func:`weight_space_eligible`); anything outside that
envelope falls back to the loop in :mod:`backtest.engine_pro`.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np

from .costs import compute_transaction_costs
from .slippage import compute_slippage

if TYPE_CHECKING:
    from .engine_pro import ProductionBacktestConfig


def weight_space_eligible(
    cfg: ProductionBacktestConfig,
    close_mat: np.ndarray | None,
) -> bool:
    """Return ``True`` when the vectorized weight-space core reproduces the loop.

    The fast path covers the production envelope: weight-space accounting (no
    share ledger), same-bar fills, and no path-dependent corporate-action /
    financing accruals.  Every excluded feature carries genuine inter-bar state
    (a cash ledger, a pending lagged order, a price patch) that the batched
    formulation does not model; those configs fall back to the event loop.
    """
    if cfg.enable_price_accounting and close_mat is not None:
        return False
    if cfg.execution_lag > 0:
        return False
    if cfg.enable_corporate_actions:
        return False
    if cfg.enable_borrow_costs:
        return False
    if cfg.enable_cash_interest:
        return False
    if cfg.enable_short_availability_gating:
        return False
    return not cfg.enable_financing


def _softmax_rows(s: np.ndarray) -> np.ndarray:
    """Row-wise softmax of an ``(n_dates, n_assets)`` matrix.

    Matches :func:`backtest.engine._softmax` applied per row: subtract the row
    max, exponentiate, divide by the row sum.
    """
    z = s - s.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _apply_constraints_batch(
    targets: np.ndarray,
    *,
    tradable: np.ndarray | None,
    lb: np.ndarray | None,
    ub: np.ndarray | None,
    max_gross: float | None,
    max_net: float | None,
) -> np.ndarray:
    """Apply the constraint stack to every row of ``targets`` at once.

    Mirrors :func:`backtest.constraints.apply_all_constraints` in canonical
    order (universe mask → per-name caps → gross cap → net cap), but operating
    on the whole ``(n_dates, n_assets)`` matrix.  Empty-skip semantics match the
    scalar path: a ``None`` argument leaves that constraint inactive.
    """
    w = targets
    if tradable is not None:
        w = np.where(tradable, w, 0.0)
    if lb is not None and ub is not None:
        w = np.clip(w, lb, ub)
    if max_gross is not None:
        gross = np.abs(w).sum(axis=1, keepdims=True)
        scale = np.where((gross > max_gross) & (gross > 0.0), max_gross / gross, 1.0)
        w = w * scale
    if max_net is not None:
        w = _apply_net_cap_batch(w, max_net)
    return w


def _apply_net_cap_batch(w: np.ndarray, max_net: float) -> np.ndarray:
    """Vectorized net-exposure trim matching ``apply_net_exposure_cap`` per row.

    For each row whose ``|sum(w)| > max_net``, the excess net is removed from
    the dominant-sign side proportionally to each holding's magnitude.  Rows
    within the cap (or with no dominant-side mass) are left unchanged.
    """
    net = w.sum(axis=1)  # (n_dates,)
    over = np.abs(net) > max_net
    if not over.any():
        return w
    result = w.copy()
    sign = np.where(net > 0.0, 1.0, -1.0)  # (n_dates,)
    excess = np.abs(net) - max_net  # (n_dates,)
    dominant = np.sign(w) == sign[:, None]  # (n_dates, n_assets)
    dom_mass = np.where(dominant, np.abs(w), 0.0).sum(axis=1)  # (n_dates,)
    adjust_rows = over & (dom_mass > 0.0)
    if not adjust_rows.any():
        return result
    # Match ``apply_net_exposure_cap`` operation order exactly so the result is
    # bit-identical to the scalar path: ``(sign * excess) * |w| / dom_sum``,
    # evaluated left-to-right, applied only to dominant-side holdings.
    safe_mass = np.where(dom_mass > 0.0, dom_mass, 1.0)[:, None]
    delta = (sign * excess)[:, None] * np.abs(w) / safe_mass
    return np.where(adjust_rows[:, None] & dominant, result - delta, result)


def _build_held_weights(
    targets: np.ndarray,
    growth: np.ndarray,
    rebal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the held-weight and pre-rebalance-weight matrices, nav-free.

    Reproduces the loop's weight bookkeeping exactly:

    * ``before[t]`` is the drifted carry from bar ``t-1`` (zeros at ``t=0``).
    * ``held[t] = target[t]`` on a rebalance bar, else ``before[t]``.
    * the carry into ``t+1`` is ``normalize(held[t] * growth[t])`` where
      ``growth[t] = 1 + R[t]/100``; a zero-sum drift leaves weights unchanged
      (matches the ``total > 0`` guard in the loop).

    Parameters
    ----------
    targets:
        Constrained target weights ``(n_dates, n_assets)``.
    growth:
        Per-asset gross return ``1 + R/100`` ``(n_dates, n_assets)``.
    rebal:
        Boolean ``(n_dates,)`` rebalance schedule.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(held, before, final)`` — the held- and pre-rebalance weight matrices
        ``(n_dates, n_assets)``, plus the post-drift carry after the last bar
        (``(n_assets,)``), which is the loop's ``final_positions``.
    """
    n_dates, n_assets = targets.shape
    held = np.empty((n_dates, n_assets))
    before = np.empty((n_dates, n_assets))
    carry = np.zeros(n_assets)
    for t in range(n_dates):
        before[t] = carry
        h = targets[t] if rebal[t] else carry
        held[t] = h
        drifted = h * growth[t]
        total = drifted.sum()
        carry = drifted / total if total > 0.0 else drifted
    return held, before, carry


def run_weight_space_vectorized(
    cfg: ProductionBacktestConfig,
    dates: list[date],
    R: np.ndarray,
    S: np.ndarray,
    *,
    rebal: np.ndarray,
    tradable_mat: np.ndarray | None,
    adv_mat: np.ndarray | None,
    comm_mat: np.ndarray | None,
    spread_mat: np.ndarray | None,
    fee_mat: np.ndarray | None,
    mincomm_mat: np.ndarray | None,
    impact_mat: np.ndarray | None,
    lb: np.ndarray | None,
    ub: np.ndarray | None,
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the batched weight-space backtest.

    Returns the same primitive history the event loop accumulates, so the
    caller assembles an identical ``BacktestResult``:

    ``(nav_hist, final_weights, trade_date_idx, trade_ids, trade_qty)`` — the
    long-format legacy trade log (one row per asset per rebalance,
    ``quantity = (target - weights_before) * nav_after_cost``) returned as three
    parallel NumPy arrays.  ``trade_date_idx`` indexes into ``dates`` (the caller
    gathers the actual ``date`` objects once, vectorized); ``trade_ids`` is the
    tiled asset index; ``trade_qty`` the concatenated per-bar quantities.  The
    values are identical to the event loop's; only the packing is vectorized.
    """
    n_dates, n_assets = R.shape
    r_frac = R / 100.0
    growth = 1.0 + r_frac

    # --- batched target weights (softmax → constraints), nav-independent --- #
    raw = _softmax_rows(S)
    targets = _apply_constraints_batch(
        raw,
        tradable=tradable_mat if cfg.enable_universe_mask else None,
        lb=lb,
        ub=ub,
        max_gross=cfg.max_gross_exposure,
        max_net=cfg.max_net_exposure,
    )

    # --- batched weight trajectory + per-bar portfolio return -------------- #
    held, before, final_weights = _build_held_weights(targets, growth, rebal)
    port_ret = np.einsum("ij,ij->i", held, r_frac)  # held[t] @ r_frac[t]

    # --- deltas charged at rebalance bars (nav-independent direction) ------ #
    deltas = targets - before  # only meaningful on rebalance bars

    costs_enabled = cfg.enable_costs and comm_mat is not None
    slip_enabled = cfg.enable_slippage
    zeros = np.zeros(n_assets)

    nav = float(cfg.initial_cash)
    nav_hist: list[float] = []
    # One quantity row per rebalance bar; stacked into the flat trade log at the
    # end.  Building the log from these dense ``(n_assets,)`` arrays (then one
    # vectorized ``repeat`` / ``tile`` / ``reshape``) avoids the per-bar Python
    # ``list.extend`` + per-element ``new_from_any_values`` type inference that
    # dominated assembly at large ``n_assets``.
    rebal_idx: list[int] = []
    trade_qty_rows: list[np.ndarray] = []

    for t in range(n_dates):
        if rebal[t]:
            d_t = deltas[t]
            trade_value = d_t * nav
            tc_cost = 0.0
            slip_cost = 0.0
            if costs_enabled:
                tc_cost = compute_transaction_costs(
                    trade_value,
                    comm_mat[t],
                    cast("np.ndarray", spread_mat)[t],
                    cast("np.ndarray", fee_mat)[t],
                    cast("np.ndarray", mincomm_mat)[t],
                )
            if slip_enabled:
                adv_t = adv_mat[t] if adv_mat is not None else zeros
                impact_t = impact_mat[t] if impact_mat is not None else zeros
                slip_cost = compute_slippage(trade_value, adv_t, impact_t)
            nav -= tc_cost + slip_cost

            # legacy trade log: quantity = (target - weights_before) * nav_after_cost
            rebal_idx.append(t)
            trade_qty_rows.append(d_t * nav)

        nav *= 1.0 + port_ret[t]
        nav_hist.append(nav)

    n_rebal = len(rebal_idx)
    if n_rebal:
        idx = np.asarray(rebal_idx)
        trade_date_idx = np.repeat(idx, n_assets)
        trade_ids = np.tile(np.arange(n_assets), n_rebal)
        trade_qty = np.concatenate(trade_qty_rows)
    else:
        trade_date_idx = np.empty(0, dtype=np.intp)
        trade_ids = np.empty(0, dtype=np.intp)
        trade_qty = np.empty(0)

    return nav_hist, final_weights, trade_date_idx, trade_ids, trade_qty
