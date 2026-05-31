# backtest — simulate a strategy over time, producing NAV, trades, and fills

## Files
- `engine_pro.py` — `ProductionBacktestEngine`, `ProductionBacktestConfig` (all production flags default off = POC behavior; execution lag, universe mask, corporate actions, constraints, financing, config validation). Largest file.
- `vectorized.py` — batched NumPy fast path (softmax/constraints/weight-drift/portfolio-return hoisted out of the Python event loop; NAV/cost recurrence stays scalar).
- `engine.py` — POC `BacktestEngine`, `BacktestConfig`, `BacktestResult`, `PortfolioState` (kept intact).
- `signals.py` — `SignalFrame` (df + `is_categorical`).
- `costs.py` — commission + half-spread + exchange fees, borrow cost, cash interest.
- `slippage.py` — square-root market impact scaled by trade size vs ADV.
- `corporate.py` — split shares×ratio/price÷ratio (NAV-invariant), dividends credit cash.
- `accounting.py` — price/share-level position accounting. `constraints.py` — position/gross/net constraints.

## Public API (additive-only contract — do not break)
`__all__`: `BacktestConfig`, `BacktestEngine`, `BacktestResult`, `BacktestRunner`, `PortfolioState`, `ProductionBacktestConfig`, `ProductionBacktestEngine`, `SignalFrame`.
Protocol (`_protocol.py`): `BacktestRunner.run(self, returns: pl.DataFrame, signals: SignalFrame) -> BacktestResult`.
Key sig (production entry): `ProductionBacktestEngine(cfg).run(returns, signals, *, prices=None, transaction_costs=None, universe_mask=None, corporate_actions=None, borrow_rates=None, min_weight_per_asset=None, max_weight_per_asset=None) -> BacktestResult`. `BacktestResult` carries `nav_history`, `trade_log`, `fill_log`, `cash_history` (all defaulted so POC construction still works).

## Harness entry / hot path
`harness/components.py::_backtest_run` (component-benchmark path; backtest IS also exercised on the pipeline golden path — `_analysis_setup` runs a `ProductionBacktestEngine` to produce its input). Timed call: `ProductionBacktestEngine(cfg).run(returns, signals, prices=, transaction_costs=, universe_mask=)` with costs+slippage+universe-mask enabled, `max_weight=0.1`. The per-date event loop (NAV/cost recurrence) dominates at large n_dates.

## Data contract
Consumes `prices` (→ session returns via `returns_from_prices`), `alpha_signals` momentum (`SignalFrame`), `transaction_costs`, `universe_mask`; optionally `corporate_actions`, `borrow_rates`, `trading_calendar`, `benchmark_returns`. Scales with `n_assets`, `n_dates`.

## Recently optimized (don't re-attempt — see IMPROVEMENTS.md)
- Vectorized weight-space fast path for the production envelope: softmax/constraints/drift/portfolio-return batched into NumPy, only NAV/cost recurrence stays scalar; incumbent loop retained for non-fast-path envelopes (1.13–1.24× on n_dates, byte-exact). PR #44.
- Earlier round: 8× `to_matrix` → batch pivot (3.4× at scale). PR #28.
