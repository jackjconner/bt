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

## 2026-05-31 — etl: remove dead _adjust_single_asset path  [accepted]
type:     consolidate
metric:   −193 net lines (etl/adjust.py + test_adjust.py); golden byte-identical; no perf change (hot path untouched)
eval:     byte-identical — direct branch-vs-main PipelineSummary diff "Files are identical"; post-merge pytest 1211 passed (−16 oracle/helper tests, +1 direct fixture test)
PR:       #47
note:     First consolidate round (the second half of two-phase add-then-consolidate). Round 005 vectorized adjust_prices but kept the per-asset loop as a test oracle under additive-only; with backwards-compat dropped (DECISIONS.md) it was dead code — no production caller. Scout cleared the other 4 round-005 shadows as genuinely live and left them: signals engine="matrix" is the non-rank IC backend, models "loop" is the non-ridge auto-dispatch fallback, backtest's scalar loop handles non-weight_space_eligible configs, analysis's scalar fns are still called by report.py. Equivalence test → direct fixture test (coverage preserved). Report: reports/round-006/etl.md.

## 2026-05-31 — round 007: multi-component feature round (one additive capability per component)  [accepted ×7]
type:     feature (7 components dispatched in parallel — all 7 merged: signals/profiling/portfolio/models/backtest/etl/analysis)
metric:   see per-component entries below — every feature is additive + flag-off/API-only, so the production golden is unchanged
eval:     golden held on the merged tree — re-validated together post-merge: full suite 1284 passed/1 skipped, evalgate 17/17 byte-identical (no PipelineSummary field added on a default run; flags ship off, so no golden re-save)
PR:       #49 #50 #51 #52 #53 #54 #55
note:     First feature round. The ideation/dedup step earned its keep: 7 of 9 seeded FEATURE_BACKLOG rows were ALREADY BUILT (analysis turnover/rolling/periodic F-001/2/3 by rounds ≤005, portfolio txn-cost F-004, signals regime-IC F-005 + combination F-009) — the scouts re-ideated genuinely-novel targets instead of re-doing landed work. All 7 PRs strictly in-lane (zero file overlap), additive, default-off or API-only; the committer isolation guard + absolute-path worker pinning held with no cwd slips (contrast round 005). Reports: reports/round-007/{signals,profiling,portfolio,models,backtest,etl,analysis}.md.

## 2026-05-31 — signals: pair-wise signal correlation + diversification ratio  [accepted]
type:     feature
metric:   new capability — signal_pair_correlation → SignalCorrelationResult (rank-corr matrix, diversification ratio, mean|corr|); +23 tests
eval:     golden held — 17 fields byte-identical (ic_raw / ic_neutralized / horizon_ic[*] 0.00e+00)
PR:       #49
note:     Screen a book of alphas for redundancy without a backtest. Spearman = Pearson on within-date ranks (Polars rank "average" tie default matches the IC engine). Pure new surface; IC engine untouched. Report: reports/round-007/signals.md.

## 2026-05-31 — profiling: r²-confidence gating for regression detection  [accepted]
type:     feature
metric:   new capability — check_regressions(min_r_squared=None) excludes low-r² (noisy) scaling fits from verdicts; +RegressionReport audit fields; +7 tests
eval:     golden held — profiling is off the number path; 17/0, n_scaling_fits=36 unchanged
PR:       #50
note:     Uses the scaling-fit r² the profiler already computed but never consumed. Default None ⇒ verdicts byte-identical to before; the harness hot loop is untouched. A (stage,metric) with no fit is never excluded. Report: reports/round-007/profiling.md.

## 2026-05-31 — portfolio: per-factor risk decomposition / attribution  [accepted]
type:     feature
metric:   new capability — FactorRiskModel.factor_risk_breakdown(w) → total/factor/specific variance + per-factor contribs; +9 tests
eval:     golden held — 17 fields byte-identical; weights/objective/constraints unchanged
PR:       #51
note:     Built on the existing factor_component_contrib (asserted equal, no duplication) and routed through the factored Σ = B·F·Bᵀ + D form — never materializes the dense cov, stays off the optimizer hot path. API-only this round (no PipelineSummary field). Report: reports/round-007/portfolio.md.

## 2026-05-31 — models: per-fold IC dispersion + hit-rate diagnostics  [accepted]
type:     feature
metric:   new capability — fold_ic_dispersion_enabled flag → FoldResult.fold_ic_std / fold_hit_rate + WFResult.fold_diagnostics; +54 model tests
eval:     golden held — flag-off wf_mean_ic byte-identical (0.00e+00), wf_mean_r2 at 1.11e-16 FP-noise
PR:       #52
note:     Derived from the existing per-date ic_values (no IC recompute). Both CV engines (generic loop + batched ridge) thread the flag through one shared assembler. Flag defaults off ⇒ flag-off hot path is the exact pre-change code. Report: reports/round-007/models.md.

## 2026-05-31 — backtest: short-availability gating + financing costs (flag-off)  [accepted]
type:     feature
metric:   new capability — enable_short_availability_gating (default off): forbid non-shortable shorts, cap short MV at loan_availability, charge daily borrow; +6 tests
eval:     golden BYTE-IDENTICAL with flag off — proven against a fresh clean-main golden at --tolerance 0 (0.00e+00 every field; the committed golden's 1e-16 noise reproduces on clean main)
PR:       #53
note:     Consumes the existing etl BORROW_RATES dataset (shortable / loan_availability / borrow_rate_bps) — no cross-lane API request. ValueError if enabled without borrow_rates (boundary assert). Excluded from the vectorized fast path (nav-dependent). Ships dormant; flip on in a later golden-moving round. Report: reports/round-007/backtest.md.

## 2026-05-31 — etl: optional data-quality flag columns (flag-off)  [accepted]
type:     feature
metric:   new capability — include_quality_flags (default off) appends is_duplicate_key / is_frozen_series / sparse_coverage / outlier_flagged / price_stale; +15 tests
eval:     golden held — consumed price panel frame_equal when flag off; residual 1e-16 reproduces on clean main
PR:       #54
note:     annotate_quality_flags REUSES the existing quality.check() logic (no reimplementation); QUALITY_FLAG_COLUMNS is the single source of column set+order. Worker correctly dropped the brief's is_halted/is_delisted (check() lacks the universe inputs; synthesizing would violate reuse-don't-reimplement) and removed an early-draft # noqa rather than suppress. Report: reports/round-007/etl.md.

## 2026-05-31 — analysis: drawdown duration & recovery time series  [accepted]
type:     feature
metric:   new capability — drawdown_recovery(nav) → per-event peak/trough/recovery dates + drawdown_days / recovery_days / peak_to_recovery_days; +8 tests
eval:     golden held — all 16 numeric fields byte-identical
PR:       #55
note:     Isolates each drawdown event off the same nav/cum_max−1 series analysis already reports; deepest event depth == existing max_drawdown (asserted). Additive, no PipelineSummary field, off the hot path. Tear-sheet material; pairs with portfolio's factor-risk breakdown this round. Report: reports/round-007/analysis.md.

## 2026-05-31 — round 008: activate backtest short-availability gating on the production path  [accepted]
type:     feature (activation — flips on a round-007 default-off capability)
metric:   activation — production backtest now runs short-gating + financing ON; golden byte-identical (long-only book → gating binds on nothing)
eval:     golden held — net_sharpe / gross_sharpe / cost_drag 0.00e+00 (NAV fp-reorder max |Δ| 3.5e-10, rel 3.7e-16); full suite 1285 passed
PR:       #60
note:     First round under the per-feature default-state policy (DECISIONS.md). Teed up as a golden-MOVING correctness round (case c); turned out case (b) golden-unchanged — the worker proved the production book is long-only by construction (softmax targets all-positive, no min_weight), so gating touches no negative weight and accrues zero. Correctness insurance: any future short book is now priced for shortability + loan availability + borrow with no config change. Activated at the production construction site (pipeline.py), not the dataclass default (which would ValueError every caller omitting borrow_rates). models fold-diagnostics + etl quality-flags left dormant by decision (no consumer / contract change). Report: reports/round-008/backtest.md.

## 2026-05-31 — round 009: parallel performance sweep across all 7 components  [accepted]
type:     exploit
metric:   each component's harness elapsed at the grid extremes — see per-component entries below
eval:     golden HELD on the merged tree — full suite 1291 passed/1 skipped, evalgate 17/17 (all accuracy fields byte-identical; only the documented 1e-16 FP-noise floor on wf_mean_ic/wf_mean_r2/factor_vol, plus the timing field backtest_p50_s under its 0.5 field-tol). No new PipelineSummary field → no golden re-save.
PR:       #62 #63 #64 #65 #66 #67 #68
note:     First all-component EXPLOIT round (cf. round 007 = first all-component feature round). 7 scouts → 7 parallel builders, each finding the biggest golden-safe perf win in its lane via in-process flamegraphs (capture_both). Clean sweep: all 7 landed real wins, all strictly in-lane (zero file overlap), all golden-safe. Headline: models ~2.8× (was the runaway dominant stage), backtest n_assets-scaling 0.93→0.46, profiling −62%, etl −35%, signals −24%, analysis −21%, portfolio −7…−27%. Batch serial-merge (rebase) + single comprehensive re-validation, exactly as round 007 (disjoint lanes). Several workers saw transient backtest_p50_s eval FAILs from concurrent-worker timing jitter; cleared on the idle serial re-validation. Reports: reports/round-009/{models,backtest,signals,etl,profiling,portfolio,analysis}.md.

## 2026-05-31 — models: vectorize per-date rank-IC (drop per-date spearmanr loop)  [accepted]
type:     exploit
metric:   models harness elapsed — ~2.8× on the n_dates-heavy path (e.g. 100a×1260d 851.5→303.4 ms; 100a×252d 178.1→64.8 ms)
eval:     golden held — wf_mean_ic / wf_mean_r2 each moved exactly 1.11e-16 (at the FP-noise floor; summation-order vs scipy's internal Pearson); other 15 fields bit-identical
PR:       #67
note:     spearmanr → rankdata(axis=1)+vectorized Pearson on the full cross-section (per-date loop was 86% of stage time). Ragged-group case keeps the per-date loop as fallback + oracle. rank_ic_series API unchanged; both WF engines benefit. models is no longer the runaway dominant stage. Report: reports/round-009/models.md.

## 2026-05-31 — backtest: vectorize trade-log assembly in the weight-space fast path  [accepted]
type:     exploit
metric:   backtest harness elapsed — 1.4–1.8× across the grid (3000a×252d 186→131 ms; 100a×5040d 178→100 ms); scaling n_assets^0.93→^0.46
eval:     golden held — accuracy fields abs_delta 0.00e+00; trade_log/nav_history/cash_history DataFrame.equals vs main = True across 5 grid points; only timing backtest_p50_s moved
PR:       #65
note:     Per-bar list.extend of date objects → NumPy repeat/tile/concatenate + typed Polars series (trade-log build 63→1 ms, 57×). A pure repacking of identical values. Deliberately did NOT touch the per-bar cost/slippage loop (nonlinear in path-dependent NAV → would move cost_drag) — remaining headroom for a future round. Report: reports/round-009/backtest.md.

## 2026-05-31 — signals: batch per-date sector OLS into one lstsq solve  [accepted]
type:     exploit
metric:   signals harness elapsed — −24% on the date-extreme (100a×5040d 349.8→265.3 ms); scaling n_dates^0.92→^0.87
eval:     golden held — ic_neutralized / ic_raw / horizon_ic[*] abs_delta 0.00e+00 (multi-RHS lstsq differs at ~1e-15 in residuals, rank-transformed into IC → no observable change)
PR:       #66
note:     Hotspot was neutralize_sector→_ols_residual (54% of stage), NOT the already-tuned IC engine. Sector dummies are time-invariant → one batched lstsq(X,Sᵀ) instead of n_dates solves. Per-date path kept as NaN-fallback + oracle. Report: reports/round-009/signals.md.

## 2026-05-31 — etl: scatter-fill masked matrix instead of pivot  [accepted]
type:     exploit
metric:   etl harness elapsed — −35% at 3000 assets (95.8→61.4 ms; to_masked_matrix alone 37→9 ms ~4×); improves at every grid point
eval:     golden held — to_masked_matrix output byte-identical to the pivot (cell-by-cell across the grid); consumed price panel unchanged
PR:       #62
note:     pivot cost scaled with asset COLUMNS; replaced with rank("dense")-driven numpy scatter scaling with observations. One documented behavior change on INVALID input only: duplicate (date,id) keys now tiebreak to input-order not sorted-order — a hard invariant violation quality.check flags, never in production/golden/bench. Not ADR-worthy (no valid-input behavior change). Report: reports/round-009/etl.md.

## 2026-05-31 — profiling: vectorize fit_scaling per-stage loops in NumPy  [accepted]
type:     exploit
metric:   profiling harness elapsed — 101.9→38.3 ms (−62%, 2.66×; fit_scaling in isolation ~7×). profiling is fixed-overhead, not data-scaling — this cut the constant.
eval:     golden held — n_scaling_fits=36 PASS, 16 accuracy fields byte-stable (only pre-existing 1e-16 BLAS noise); fit_scaling output identical 36-fit set vs clean main
PR:       #64
note:     144 inner iters (9 stages × 4 metrics × 4 dims) each rebuilt the same per-stage Polars filter + recomputed each dim's .mode(); both depend only on (stage,dim) not metric → hoisted once-per-stage into NumPy. Polars .mode() retained for byte-preserved tie-break; append order kept so the fit list is unchanged. Off the production number path. Report: reports/round-009/profiling.md.

## 2026-05-31 — portfolio: assemble OSQP constraint matrix from COO triplets  [accepted]
type:     exploit
metric:   portfolio harness elapsed — −7…−27% (n=100 3.27→2.39 ms; constraint build alone 2.58→1.03 ms ~2.5×); −91 net lines
eval:     golden byte-identical — opt_gross / tracking_error 0 delta, factor_vol at 1e-16 floor; constraint matrix A bit-identical over 32 spec configs → max|Δw|=0.0
PR:       #63
note:     The OSQP ADMM solve is intrinsic/irreducible without moving the golden (eps relaxation moves weights ~4e-6, w0 warm-start no benefit — both tested+rejected). The one golden-safe win was constraint assembly: per-block sp.eye/hstack/vstack → COO triplets into one CSC. Further portfolio perf needs a solver-level change = an explore-round target (would move numbers). Report: reports/round-009/portfolio.md.

## 2026-05-31 — analysis: ordered group-by for turnover (skip hash + redundant sort)  [accepted]
type:     exploit
metric:   analysis harness elapsed — two_way_turnover 1.44→1.13 ms (−21%) at the long-history extreme (n_dates=5040); neutral at modal point
eval:     golden held — tracking_error / gross_sharpe / net_sharpe / cost_drag 0.00 delta
PR:       #68
note:     group_by("date").agg().sort("date") → sort("date").group_by(maintain_order=True).agg() (sorted-key group-by skips the hash table; leading sort makes output order-independent, pinned by a shuffle test). Rejected the faster set_sorted variant (bakes unsafe sortedness into a public fn). Noted-not-actioned (cross-lane): alpha/beta/IR do 3 redundant _align joins — a harness call-site fix for a future round. Report: reports/round-009/analysis.md.
