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

## 2026-05-31 — portfolio: lazy-materialize Σ + vectorize build_from_long  [accepted]
type:     explore (salvaged winner; the K=3 tournament was aborted mid-flight — see note + SPIKES.md)
metric:   risk-model build @2k assets — 22.3 ms → 0.43 ms (52×; 18–112× across 500→3000 assets); peak RSS n² growth → flat (+218 MB → +4 MB over that range)
eval:     golden held — 16/17 fields byte-identical; only backtest_p50_s (wall-clock timing) moved, and faster (exempt via --field-tol). Post-merge pytest 401 passed, evalgate within tolerance.
PR:       #36
note:     Σ = B·F·Bᵀ + D is now a lazy cached `.cov` property — the dense n×n object never lands on the build/optimizer hot path (variance/contribs go through the factored form B(F(Bᵀw))+specific_var⊙w); build_from_long vectorized (np.unique/searchsorted, no iter_rows). The explore round was dispatched via the Workflow tool, which spawned unsupervised worktree copies that exhausted tmpfs and capsized the run before the judge ran; one worker's rewrite (the wide-layout/lazy-Σ direction) was salvaged, re-reviewed, and individually gated. Workflow-tool tournaments retired in favor of supervised Agent dispatch (DECISIONS.md). Report: reports/round-004/portfolio.md.

## 2026-05-31 — round 005: multi-component explore (one bold rewrite per component)  [accepted ×4]
type:     explore (5 components dispatched in parallel — all 5 merged: analysis/etl/models/signals/backtest)
metric:   see per-component entries below
eval:     all five hold the golden — re-validated together post-merge: full suite 1227 passed/1 skipped, evalgate 17/17
PR:       #38 #39 #41 #42 #44
note:     First successful supervised multi-component explore round — 5/5 wins, no spikes. Each worker ran in its own worktree with diskguard + the bench lock; the committer worker-isolation guard (refuse component commits from the primary worktree) was added mid-round after several workers' cwd slipped into the main checkout (the hardened backtest re-dispatch then stayed perfectly isolated). Reports: reports/round-005/{analysis,etl,models,signals,backtest}.md. Baseline ratcheted on the merged 5/5 tree (post-round harness run); etl now n_dates^0.75, analysis n_dates^0.20.

## 2026-05-31 — analysis: fuse metrics suite into single-pass engine  [accepted]
type:     explore
metric:   analysis benchmark suite — 470.4 µs → 169.1 µs (2.78×, 4 joins → 1); harness analysis p50 1.82 → 1.74 ms
eval:     golden held — gross_sharpe / net_sharpe bit-identical (0.00e+00)
PR:       #38
note:     analyze_fused + benchmark_metrics_fused compute from shared moments instead of re-walking/re-joining per metric; +8 equivalence tests. The 2.78× suite win is only reachable through the new API — harness wiring is cross-lane → API_REQUESTS. Report: reports/round-005/analysis.md.

## 2026-05-31 — etl: vectorize corporate-action adjustment over the whole panel  [accepted]
type:     explore
metric:   etl p50 (adjust_prices) — 3000 assets 226 → 121 ms (−46%); 100×5040 dates 1103 → 63 ms (−94%, ~18×); all 11 grid points improve, scaling → sub-linear (~n^0.78)
eval:     golden held — 16/16 accuracy fields 0.00e+00 (adjust_prices is off the pipeline golden path)
PR:       #39
note:     one join_asof + a segment-reset reverse-cumprod in log space replaces the per-asset partition→Python-loop→concat; legacy _adjust_single_asset retained as a test oracle; +7 tests. Report: reports/round-005/etl.md.

## 2026-05-31 — models: batched numpy-core walk-forward engine  [accepted]
type:     explore
metric:   models walk_forward_cv — 500×252 135.5 → 56.1 ms (2.41×); 1.26–2.41× across the grid
eval:     golden held — wf_mean_ic byte-identical (0.00e+00); wf_mean_r2 at machine epsilon (1.11e-16)
PR:       #41
note:     each fold's standardized Gram assembled as a difference of cumulative block moments; one Cholesky solves all alphas; closed-form weighted ridge matches sklearn ~1e-16; wired additively via WalkForwardConfig.engine="auto" (non-ridge keeps the loop); +18 tests. Report: reports/round-005/models.md.

## 2026-05-31 — signals: lazy/streaming Polars Spearman-IC engine  [accepted]
type:     explore
metric:   signals p50 — median 248.6 → 128.7 ms (−48.3%); all 11 grid points improve
eval:     golden held — ic_raw / ic_neutralized / horizon_ic[*] bit-identical (0.00e+00)
PR:       #42
note:     a long-format group_by rank-IC replaces the dense pivot + scipy.rankdata path (Spearman = Pearson on within-date ranks; Polars rank "average" tie-default matches scipy); default engine="lazy", incumbent engine="matrix" retained; +19 bit-identity tests. Report: reports/round-005/signals.md.

## 2026-05-31 — backtest: vectorized weight-space fast path for the production envelope  [accepted]
type:     explore
metric:   backtest p50 — median 41.21 → 33.27 ms (1.24×); 1.13–1.24× on the n_dates axis, wash at largest n_assets (irreducible sequential cost loop)
eval:     golden held — path-dependent fields byte-exact: gross_sharpe 8.88e-16, net_sharpe 3.22e-15, cost_drag 2.33e-10 (last-ULP float reassociation)
PR:       #44
note:     hoist softmax / constraints / weight-drift / portfolio-return out of the Python event loop into batched NumPy; only the NAV/cost recurrence stays scalar; incumbent loop retained for non-fast-path envelopes; +12 byte-identity tests. The resumed draft had a blocking object-ndarray→Date cast bug, fixed by returning plain lists through _assemble_result. Report: reports/round-005/backtest.md.
