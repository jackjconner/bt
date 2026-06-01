# Feature backlog

The prioritized queue of candidate capabilities for `feature` rounds (see
`improvement-orchestrator`). Each row is scored against the `VISION.md` rubric;
`priority` is the weighted sum (higher = sooner). **Ideation appends `queued`
rows; Jack reprioritizes.** Before a worker builds an item, the orchestrator
dedups it against current code + `IMPROVEMENTS.md` — these are drawn from
`PRODUCTION_PLAN.md` tier-2/3 and some may be partly present already; the round
confirms genuine novelty and lands only the additive remainder.

Status lifecycle: `queued` → `building` → `done` (or `dropped`, with a reason).
Every `feature` round is **additive** — a new capability behind a flag, existing
golden fields hold, new `PipelineSummary` fields gated by `evalgate
--allow-new-fields`.

| id | component | capability | why (vs VISION) | status | priority |
|---|---|---|---|---|---|
| F-001 | analysis | turnover (one-/two-way) series from `trade_log` | breadth + observability; data already present, isolated, additive | done (≤r005, pre-dedup) | 30 |
| F-002 | analysis | rolling metrics (Sharpe / vol / beta / drawdown) | breadth; high-leverage for tear sheets, cleanly additive series | done (≤r005, `rolling.py`) | 29 |
| F-003 | analysis | periodic-return tables (monthly / quarterly / annual) | breadth + observability; additive, low cost | done (≤r005, `periodic.py`) | 27 |
| F-004 | portfolio | transaction-cost-aware rebalancing (turnover penalty / no-trade band) | correctness of net returns; flag-gated objective term | done (≤r005, `cost_scale`/`no_trade_band`) | 26 |
| F-005 | signals | regime / subsample IC stability | breadth of alpha research; additive metric, defends robustness | done (≤r005, `regime_conditional_ic`) | 24 |
| F-006 | portfolio | EWMA covariance variant beside Ledoit-Wolf | robustness; additive estimator choice behind a flag | queued (factor-cov level still open; `ewma_cov` exists on returns) | 23 |
| F-007 | backtest | borrow / short-availability gating + financing costs | correctness for shorts; `borrow_rates` dataset already exists | active r008 #60 (built r007 #53; now on in production) | 22 |
| F-008 | backtest | multiple order types (limit, MOC/MOO, VWAP/TWAP/POV) | breadth + realism; higher cost, flag per order type | queued | 20 |
| F-009 | signals | signal combination / orthogonalization | breadth; marginal-contribution of each alpha, additive | done (≤r005, `zscore_blend`/`gram_schmidt`) | 19 |
| F-010 | signals | pair-wise signal correlation + diversification ratio | breadth; screen alphas for redundancy without a backtest | done r007 #49 | 38 |
| F-011 | profiling | r²-confidence gating for regression detection | observability; suppress false alarms on noisy scaling fits | done r007 #50 | 35 |
| F-012 | portfolio | per-factor risk decomposition / attribution | observability; total/factor/specific variance breakdown | done r007 #51 | 37 |
| F-013 | models | per-fold IC dispersion + hit-rate diagnostics | correctness; fold-level stability auditing | done r007 #52 | 30 |
| F-014 | etl | optional data-quality flag columns (`include_quality_flags`) | robustness + observability; tradability/confidence gating | done r007 #54 | 37 |
| F-015 | analysis | drawdown duration & recovery time series | observability; tear-sheet drawdown-event analytics | done r007 #55 | 35 |

## Notes

- Priorities are the `VISION.md` rubric output (`3·fit + 2·leverage + 2·safety +
  1·cost`, max 40), not hand-rankings — re-score, don't re-vibe.
- A candidate that *requires* a contract break does not belong here — that's a
  `design-before-architecture-changes` conversation with Jack first.
- When an item lands, mark it `done` with the PR# and move on; never delete a row
  (the backlog is also the dedup record of what was considered).
