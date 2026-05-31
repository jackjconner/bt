# Improvements log

Append-only record of every component-improvement round — accepted *and*
rejected. This is the loop's memory: the dedup source for round planning (don't
re-attempt a recently-rejected target), the **explore-cadence** source (count
rounds since the last `type: explore`), and the audit trail of cumulative gains.
Recorded by the `improvement-orchestrator` skill. Newest entries at the bottom.

Every round declares a **`type`** (see `improvement-orchestrator` → Round types):

- `exploit` — minimal-diff perf/accuracy win on a hotspot (the original loop).
- `refactor` — structural cleanup, no behavior change; golden byte-identical.
- `feature` — additive new capability behind a flag; golden holds, new fields ok.
- `explore` — bold rewrite via K-way tournament; the merged winner is recorded
  here, the discarded attempts in `SPIKES.md`.

Verdicts: `accepted` (merged), `rejected` (no PR passed its type's gate), or
`spiked` (an explore round where every attempt was discarded — learning logged in
`SPIKES.md`, nothing merged).

Format (one block per round, never edited after writing):

```
## <date> — <component>: <one-line target>  [accepted | rejected | spiked]
type:     exploit | refactor | feature | explore
metric:   <name> — <before> → <after> (<delta>)   # or "n/a (feature/refactor)"
eval:     golden unchanged within <tol>  |  accuracy moved: <field before→after, why correct>  |  additive: <new fields>
PR:       <url or #number>
note:     <why accepted, or which gate rejected it and the takeaway>
```

---

## 2026-05-31 — portfolio: replace SLSQP with OSQP QP solver  [accepted]
metric:   portfolio harness p50 — 457/1989/4178 ms → 8/20/66 ms (~50–100×)
eval:     accuracy moved: factor_vol 0.7678 → 0.7714 (OSQP reaches the true constrained optimum; objective −0.1054 → −0.1011 strictly better, both converged, net-exposure exactly 1.0; all other PipelineSummary fields held)
PR:       #3
note:     accepted — Jack-approved new dep (osqp) + the justified eval shift. Super-linear scaling remains (now ledoit_wolf_cov-bound, not solver-bound).

## 2026-05-31 — signals: vectorize cross-sectional IC computation  [accepted]
metric:   signals harness p50 — 314/340/398 ms → 200/213/241 ms (−36 to −39%)
eval:     golden unchanged within 1e-6 (all 17 fields PASS; IC identical to 1e-16)
PR:       #17
note:     accepted — rankdata-vectorized the per-date Spearman loop, single horizon-grid pivot (was 4×), Polars-native sector neutralization. Pure perf, no number moved. signals was the post-osqp hotspot; models (~190ms) is next.

## 2026-05-31 — performance swarm (flame-graph-driven): 5 components  [accepted]
metric:   portfolio scaling n_assets^2.11 → ^0.93 (factor cov); signals 14–22×, etl 4.9×, backtest 3.4×, models ~1.9× at scale
eval:     held byte-identical for the 4 pure-perf (#28/#30/#31/#32); portfolio (#34) moved — factor risk model: opt_gross 1.62→1.97, factor_vol 0.7714→0.6040, tracking_error 0.5892→0.8084 (justified; IC/Sharpe/cost_drag held)
PR:       #28 #30 #31 #32 #34
note:     accepted — flame-graph hotspots removed: backtest 8× to_matrix→batch pivot; signals rankdata NaN-tail fast-path; etl per-asset filter + iter_rows→partition_by/pivot; models duplicate rank_ic; portfolio factor-covariance wired to production (the n² time+memory bottleneck is gone, harness now matches production). Mid-round we added scripts/bench (lock) + BT_GRID=small after concurrent runs oversubscribed 16 cores ~6× and corrupted timings.
