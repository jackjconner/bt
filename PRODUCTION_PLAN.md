# Production Plan: POC → Production

This document captures, per module, a feature plan to take the module from
its current proof-of-concept state to production, plus the new datasets each
module needs. The datasets are then **deduplicated into a unique schema list**
(see [Unique Data Schemas](#unique-data-schemas)) which the synthetic data
generator produces alongside the existing `returns` dataset.

Each plan was produced by a module-scoped review of the actual code as it
exists today (a toy/POC in every module).

---

## backtest

Today: softmax-weighted, long-only, fully-invested portfolio; returns divided
by 100; naive proportional drift renormalization; no costs, no constraints, no
order types; instantaneous zero-impact fill at the return-implied price.

### Feature plan
**Tier 1 — correctness blockers (the backtest is misleading without these):**
1. Transaction cost model — trades are computed but charged nothing; costs are the single largest paper-vs-live gap.
2. Slippage / market-impact model — fills happen at zero impact; large trades against finite ADV must move the price.
3. Realistic fill semantics with execution lag — decision-at-`t`, execute-at-`t+1` (or VWAP) to remove look-ahead.
4. Tradability / universe masking — mask listing/delisting, halts, missing prices so untradeable names can't be held.
5. Corporate-actions handling — splits/dividends/symbol changes must adjust prices, positions, cash.
6. Price-based accounting instead of return-only — need real prices to size shares, mark-to-market, reconcile cash.

**Tier 2 — portfolio realism:**
7. Position & portfolio constraints — configurable gross/net exposure, per-name caps, neutrality, cash/short.
8. Borrow / short-availability & financing costs — gate feasible shorts and price them.
9. Multiple order types — limit, MOC/MOO, participation (VWAP/TWAP/POV).
10. Cash, margin & interest accounting — explicit cash ledger, margin, interest on balances.
11. Calendar-aware rebalancing — real trading calendar so cadence and compounding align with sessions.

**Tier 3 — measurement & operability:**
12. Benchmark-relative & risk attribution baked into `BacktestResult`.
13. Per-trade cost/fill log — realized fill price, cost, slippage, partial-fill flags.
14. Determinism & config validation — assert returns/signals universe alignment; validate config at construction.

### Required data
`prices`, `volume`, `tradable_mask`, `corporate_actions`, `transaction_costs`,
`borrow_rates`, `trading_calendar`, `benchmark_returns`, `risk_factor_returns`,
`risk_free_rate`, `asset_static`.

---

## profiling

Today: `StageTimer` wall timer, RSS via `/proc`, `collect_stage` runs a stage
once and captures elapsed/result_mb/rss_delta, `print_report` prints a table.
Nothing is persisted; the param grid runs once per invocation.

### Feature plan
1. Persistent metrics storage (Parquet) keyed by run/commit/params — results currently only print and are lost.
2. Run/environment metadata capture — git SHA, host, CPU, RAM, lib versions — so metrics are comparable.
3. Regression detection vs baselines — flag deltas beyond a threshold; turns the report into a guardrail.
4. Repeated trials + percentile latencies — min/median/p90/p95/stddev instead of a single noisy sample.
5. Structured output formats — JSON/Parquet (+ JUnit/markdown) for dashboards and tooling.
6. CI integration — non-zero exit on regression + machine-readable artifact to gate merges.
7. Per-stage memory attribution — peak (tracemalloc/peak-RSS), not just net retained.
8. Scaling-curve fitting — log-log slope vs the param grid to surface super-linear stages.
9. CPU profiling + flamegraphs — cProfile/py-spy per flagged-slow stage.
10. Warmup + isolation controls — discard warmup; pin/record BLAS/Polars threads.
11. Stage registry / config-driven runs — declare stages and grids in config, not hand-written calls.
12. Historical trend tracking — query persisted runs over time to catch slow drift.

### Required data
`profiling_runs`, `stage_measurements`, `stage_baselines`,
`regression_thresholds`, `scaling_fits`, `cpu_profile_frames`.

---

## models

Today: ridge regression with sklearn `KFold(shuffle=False)` CV on a **random**
feature matrix `X (n_dates, n_features)` and **random** target `y (n_dates,)`;
stores train R² per fold; no scaling, no leakage controls, no panel structure.

### Feature plan
1. Purged + embargoed time-series CV — contiguous `KFold` still leaks across the boundary with lookback/forward windows.
2. Walk-forward / expanding-window evaluation — `KFold` trains on the future; a backtest must train only on the past.
3. Per-fold feature standardization fit on train only — ridge penalty is scale-sensitive; avoid leakage.
4. Panel-aware data handling (date×id) — stack the real panel into samples keyed by (date, id); split by date.
5. Cross-sectional / IC-based scoring — add per-date rank IC; R² on returns is near-zero even for good signals.
6. Hyperparameter search (alpha) inside CV — replace the fixed config alpha with an inner grid / nested CV.
7. Sample weighting — recency/inverse-vol/liquidity weights via `sample_weight`.
8. Multiple model types behind `FinancialModel` — Lasso/ElasticNet/GBM/linear variants.
9. Model persistence + metadata — serialize coefficients, scaler, config, training window for live scoring/audit.
10. NaN / missing-data handling — define masking/imputation; NaNs currently break the closed-form fit silently.
11. Prediction provenance output — return predictions tagged with (date, id).

### Required data
`feature_panel` (model features), `forward_returns` (target), `sample_weights`,
`universe_mask`, `feature_metadata`, `cv_splits_calendar`.

---

## analysis

Today: consumes a `BacktestResult` (nav_history, trade_log, final_positions),
produces a daily return series, drawdown series, Sharpe, max drawdown,
annualized return/vol. `rf` is a single float; annualizes with 252 over a
calendar-day axis.

### Feature plan
**P0 — correctness/calendar (no new data, blocking):**
1. Trading-calendar-aware annualization — calendar-day axis vs 252 is inconsistent.
2. Geometric (compound) annualized return / CAGR — current `mean * 252` is an arithmetic approximation.
3. Downside/risk suite — Sortino, Calmar, VaR/CVaR, skew/kurtosis, hit-rate, best/worst day.

**P1 — risk-free & benchmark awareness (core gap):**
4. Risk-free-rate-aware Sharpe/Sortino from a daily series.
5. Benchmark-relative metrics — alpha, beta, tracking error, information ratio, up/down capture, R².
6. Active-return & relative-drawdown series.

**P2 — cost & turnover (data already in `trade_log`):**
7. Turnover (one/two-way) series — primary driver of capacity and cost; currently unused.
8. Transaction-cost / slippage modeling → gross-vs-net NAV.
9. Position concentration / exposure analytics — leverage, gross/net, top-N, effective-N.

**P3 — attribution:**
10. Factor-based return attribution (Brinson / regression).
11. Sector/group attribution.
12. Per-asset contribution-to-return and contribution-to-risk.

**P4 — temporal & reporting:**
13. Rolling metrics (Sharpe/vol/beta/drawdown).
14. Periodic-return tables (monthly/quarterly/annual, heatmap).
15. Tear-sheet / report generation (HTML/PDF).
16. Multi-result / parameter-sweep comparison.

### Required data
`risk_free_rate`, `benchmark_returns`, `trading_calendar`,
`transaction_costs`, `factor_returns`, `factor_loadings`, `security_master`
(sector map + metadata). A per-(date,id) weight panel is reconstructable from
`trade_log`, so it is an analysis-internal artifact, not a new dataset.

---

## signals

Today: a single per-date IC series (Spearman for continuous, point-biserial for
binary) vs `return_{t+1}`, summarized with mean IC, IC-IR, Newey-West HAC
t-stat. Signals are pure noise (`random_continuous`/`random_binary`).

### Feature plan
1. Configurable forward-return horizon (k-day) — `R[t+1]` is hard-coded; real alphas predict over weeks.
2. Rank-IC vs Pearson-IC as an explicit choice — decouple from `is_categorical`.
3. IC decay / horizon curve — IC(h) over a horizon grid; sets holding period.
4. Quantile / decile spread analysis (+ monotonicity) — what a long-short book actually earns.
5. NaN / coverage-aware computation — per-date pairwise-complete masking + min-coverage threshold.
6. Cross-sectional neutralization (sector/factor/size) — separate alpha from incidental tilts.
7. Turnover-aware signal scoring — autocorrelation/rank-stability; IR net of cost drag.
8. Signal combination / orthogonalization — marginal contribution of each alpha.
9. Rolling IC-IR and regime/subsample stability.
10. Multiple-testing / breadth correction — deflate the best Newey-West t-stat for selection bias.
11. Forward-fill / lag-alignment safeguards — prevent off-by-one look-ahead.

### Required data
`alpha_signals` (signals with real predictive structure), `forward_returns`
(multi-horizon), `security_master` (sector), `factor_loadings`,
`universe_mask`, `transaction_costs`, `signal_registry`.

Note: `date_axis` uses calendar days (`interval="1d"`), so `R[t+1]` is "next
calendar day," not "next trading day." Horizon data should use a trading-day
axis (see [Cross-cutting fixes](#cross-cutting-fixes)).

---

## etl

Today: loads a single synthetic returns parquet (long `date/id/return`) via a
batch (in-memory) or streaming Polars engine; no validation, no PIT awareness,
one source type.

### Feature plan
1. Schema validation & dtype enforcement at load — fail loudly at the boundary, not downstream.
2. Source registry / typed loaders for multiple datasets — prices, volume, fundamentals, universe, corporate actions.
3. Point-in-time (PIT) correctness with as-of joins — `effective_date`/`knowledge_date`; load only what was known.
4. Survivorship-bias-free universe membership — listing/delisting intervals; include dead names.
5. Partitioned & incremental loads — Hive partition by date, predicate/projection pushdown, appends.
6. Date-range / universe filtering pushed into the scan — materialize only the requested slice.
7. Data-quality checks — duplicate keys, missing days, frozen/zero-variance series, return-spike outliers.
8. Explicit missing-data policy — declared fill/mask/drop + returned validity mask (pivot silently makes NaNs).
9. Corporate-action adjustment — continuous total-return / split-adjusted series.
10. Trading-calendar alignment — exchange calendar instead of naive daily axis.
11. Currency normalization — FX to a base currency.
12. Caching / artifact materialization — validated, adjusted intermediate keyed by source version + filter.

### Required data
`prices` (OHLCV), `corporate_actions`, `universe_mask` (membership),
`security_master`, `shares_outstanding`, `fundamentals`, `trading_calendar`,
`fx_rates`.

---

## portfolio

Today: softmax weights from a signal (`from_signals`), `W @ L` exposures
against **random** loadings, asset-level rolling vol / historical VaR /
drawdown. No optimizer, no real risk model, no constraints.

### Feature plan
1. Portfolio optimizer with objective + constraints (mean-variance / risk-parity) — replace the softmax heuristic.
2. Real factor risk model (factor covariance + specific risk) — `Σ = B Fᶜᵒᵛ Bᵀ + D`, not raw sample cov.
3. Position limits (min/max weight, gross/net exposure, leverage caps).
4. Sector / group / country constraints.
5. Transaction-cost-aware rebalancing — turnover penalty / no-trade band.
6. Ex-ante tracking error vs a benchmark — `sqrt((w-b)ᵀ Σ (w-b))`.
7. Risk decomposition — factor vs specific, marginal/component contributions.
8. Covariance estimation upgrade — EWMA / Ledoit-Wolf shrinkage (raw `np.cov` is singular when window < n_assets).
9. Parametric / factor-based VaR & CVaR — multiple horizons/confidence levels.
10. Configurable weighting schemes & rebalance calendar — equal/cap/inverse-vol/optimized; scheduled rebalances.
11. Constraint/feasibility validation at boundaries — weight sums/bounds, loading alignment, matrix conditioning.

### Required data
`factor_loadings`, `factor_returns`, `factor_covariance`, `specific_risk`,
`security_master` (sector), `position_constraints`, `group_constraints`,
`benchmark_weights`, `transaction_costs`, `prices` (+ market cap / shares).

---

## Unique Data Schemas

Overlapping datasets across modules are merged here into one canonical set. The
synthetic generator emits each of these alongside the existing `returns`
dataset. Conventions: long format sorted by `(date, id)`; `date` is Polars
`date`; asset `id` is `i64` over `0..n_assets`; all values from a seeded
`np.random.default_rng`. Returns/prices are kept in **percent units** to match
the engine's existing `R / 100.0` convention.

### A. Market-data panels (date × id)
| dataset | columns (dtype) |
|---|---|
| `prices` | `date: date`, `id: i64`, `open: f64`, `high: f64`, `low: f64`, `close: f64`, `vwap: f64`, `volume: f64`, `dollar_volume: f64`, `adv_20: f64`, `currency: cat` |
| `shares_outstanding` | `date: date`, `id: i64`, `shares_outstanding: f64`, `market_cap: f64` |
| `universe_mask` | `date: date`, `id: i64`, `in_universe: bool`, `tradable: bool`, `halted: bool`, `listed: bool` |
| `borrow_rates` | `date: date`, `id: i64`, `borrow_rate_bps: f64`, `shortable: bool`, `loan_availability: f64` |
| `transaction_costs` | `date: date`, `id: i64`, `commission_bps: f64`, `half_spread_bps: f64`, `impact_coef: f64`, `min_commission: f64`, `exchange_fee_bps: f64` |
| `specific_risk` | `date: date`, `id: i64`, `specific_var: f64` |

### B. Factor data
| dataset | columns (dtype) | shape |
|---|---|---|
| `factor_loadings` | `date: date`, `id: i64`, `factor_id: i64`, `loading: f64` | date × id × factor |
| `factor_returns` | `date: date`, `factor_id: i64`, `return: f64` | date × factor |
| `factor_covariance` | `date: date`, `factor_i: i64`, `factor_j: i64`, `cov: f64` | date × factor × factor |

### C. ML / signal training data (date × id)
| dataset | columns (dtype) |
|---|---|
| `feature_panel` | `date: date`, `id: i64`, `feat_0..feat_{F-1}: f64` |
| `forward_returns` | `date: date`, `id: i64`, `fwd_ret_1: f64`, `fwd_ret_5: f64`, `fwd_ret_21: f64`, `fwd_ret_63: f64` |
| `alpha_signals` | `date: date`, `id: i64`, `signal_name: cat`, `signal: f64` |
| `sample_weights` | `date: date`, `id: i64`, `weight: f64` |

### D. Per-date time series
| dataset | columns (dtype) |
|---|---|
| `risk_free_rate` | `date: date`, `annual_rate: f64`, `daily_rate: f64` |
| `benchmark_returns` | `date: date`, `benchmark_id: cat`, `return: f64` |
| `benchmark_weights` | `date: date`, `id: i64`, `benchmark_weight: f64` |
| `fx_rates` | `date: date`, `from_currency: cat`, `to_currency: cat`, `rate: f64` |

### E. Per-date calendar / events (sparse)
| dataset | columns (dtype) |
|---|---|
| `trading_calendar` | `date: date`, `exchange: cat`, `is_session: bool`, `is_half_day: bool`, `session_open: str`, `session_close: str` |
| `corporate_actions` | `ex_date: date`, `id: i64`, `action_type: cat`, `split_ratio: f64`, `cash_amount: f64`, `currency: cat`, `new_id: i64` |

### F. Per-asset static / reference
| dataset | columns (dtype) |
|---|---|
| `security_master` | `id: i64`, `ticker: str`, `name: str`, `sector: cat`, `industry: cat`, `country: cat`, `exchange: cat`, `currency: cat`, `lot_size: i64`, `listing_date: date`, `delisting_date: date`, `is_active: bool` |
| `fundamentals` | `report_date: date`, `knowledge_date: date`, `id: i64`, `revenue: f64`, `net_income: f64`, `total_assets: f64`, `total_equity: f64`, `total_debt: f64`, `operating_cash_flow: f64`, `currency: cat` |

### G. Lookup / config tables
| dataset | columns (dtype) |
|---|---|
| `feature_metadata` | `feature: str`, `category: cat`, `winsorize: bool`, `lookback_days: i64` |
| `signal_registry` | `signal_name: str`, `family: cat`, `is_categorical: bool`, `intended_horizon: i64` |
| `position_constraints` | `id: i64`, `min_weight: f64`, `max_weight: f64`, `tradable: bool` |
| `group_constraints` | `sector: cat`, `min_exposure: f64`, `max_exposure: f64` |
| `cv_splits_calendar` | `fold: i64`, `train_start: date`, `train_end: date`, `embargo_days: i64`, `test_start: date`, `test_end: date` |

### H. Profiling / telemetry
| dataset | columns (dtype) |
|---|---|
| `profiling_runs` | `run_id: str`, `run_ts: date`, `git_sha: str`, `git_dirty: bool`, `hostname: str`, `cpu_model: str`, `n_cores: i64`, `total_ram_mb: f64`, `python_version: str`, `polars_version: str`, `numpy_version: str`, `blas_threads: i64`, `trials: i64`, `warmup_trials: i64` |
| `stage_measurements` | `run_id: str`, `param_point_id: i64`, `n_assets: i64`, `n_dates: i64`, `n_features: i64`, `n_factors: i64`, `stage: cat`, `trial_idx: i64`, `elapsed_s: f64`, `result_mb: f64`, `rss_delta_mb: f64`, `peak_rss_mb: f64`, `peak_traced_mb: f64` |
| `stage_baselines` | `baseline_id: str`, `param_point_id: i64`, `n_assets: i64`, `n_dates: i64`, `n_features: i64`, `n_factors: i64`, `stage: cat`, `elapsed_s_p50: f64`, `elapsed_s_p90: f64`, `result_mb: f64`, `peak_rss_mb: f64`, `source_run_id: str`, `created_ts: date` |
| `regression_thresholds` | `stage: cat`, `metric: cat`, `max_pct_increase: f64`, `max_abs_increase: f64`, `min_samples: i64` |
| `scaling_fits` | `run_id: str`, `stage: cat`, `metric: cat`, `scaling_dim: cat`, `log_log_slope: f64`, `intercept: f64`, `r_squared: f64`, `n_points: i64` |
| `cpu_profile_frames` | `run_id: str`, `param_point_id: i64`, `stage: cat`, `function: str`, `filename: str`, `lineno: i64`, `cumulative_s: f64`, `self_s: f64`, `call_count: i64` |

### Cross-cutting fixes
- **Trading-day axis.** `etl.source.date_axis` uses calendar days. Several
  modules (signals horizons, analysis annualization, backtest cadence) assume
  trading days. `trading_calendar` is the canonical source; panels should be
  generated on its sessions.
- **Price/return consistency.** `prices.close` must satisfy
  `close[t]/close[t-1] - 1 ≈ returns.return / 100` so the existing engine's
  `R / 100.0` accounting stays valid.
</content>
</invoke>
