# Decisions (ADR-lite)

Append-only. Dated. Never edit past entries.

## 2026-05-30 — Synthetic data layer location & shape
**Context:** Seven module reviews each demanded new datasets; deduped into a
unique schema list (`PRODUCTION_PLAN.md`). They must be synthetically
generated alongside the existing `returns` dataset.
**Decision:** Put schema definitions in `etl/schema.py` and generators + a
name→(schema, generator) registry in `etl/datasets.py`. Extends the existing
`etl.source` synthetic-data responsibility rather than adding a new top-level
package. Generators return Polars `DataFrame`s (long format, sorted by
`date, id`), seeded `np.random.default_rng`. Returns/prices in percent units.
**Rationale:** `generate_returns`/`write_parquet` already live in `etl`;
keeping generation there avoids a parallel package and a new import root.
A registry also feeds the etl "schema validation / source registry" feature.
**Consequences:** `etl` gains a data-catalog role. Loaders validate against
the registry schema at load.

## 2026-05-30 — Generated price/return consistency
**Context:** Backtest engine divides returns by 100; analysis annualizes.
**Decision:** Synthetic `prices.close` is built by compounding the existing
percent `returns`, so `close[t]/close[t-1]-1 == returns/100`. Factor returns,
benchmark, rf all in the same percent unit.
**Rationale:** Keeps the existing engine accounting valid and lets attribution
reconcile against the price panel.
**Consequences:** Generators are layered: returns → prices → volume/adv.

## 2026-05-30 — Trading-day axis
**Context:** `etl.source.date_axis` uses calendar days (`interval="1d"`).
Signals/analysis/backtest assume trading days.
**Decision:** Add a business-day axis; `trading_calendar` is canonical and
panels generate on its sessions. Keep `date_axis` for back-compat but default
new panels to the session axis.
**Rationale:** Removes the weekend look-ahead/annualization bug.
**Consequences:** A `session_axis` helper in `etl.source`.

## 2026-05-30 — Component harnesses & integration tests
**Context:** Need end-to-end integration testing and a profiling harness for
each distinct component (the 7 modules).
**Decision:** New `harness/` package. `spec.py` defines `ComponentBenchmark`
(setup → run → frames) + `BenchmarkContext`; `components.py` registers one
benchmark per module exercising its production path; `runner.py` drives a
GenSpec grid through `profiling.run_trials`, persists via `profiling.write_run`,
fits scaling curves (`fit_scaling`) and optionally checks regressions
(`check_regressions`). Integration tests live under `tests/integration/` as
per-component contract tests (upstream output → component → downstream input)
plus a full-pipeline e2e. Chosen by the user over alternatives.
**Rationale:** Reuses the profiling production features (dogfooding) and keeps
benchmark wiring out of the modules; contract tests catch interface drift the
unit tests miss.
**Consequences:** `harness/` depends on all modules + `etl.datasets`; main.py
runs the harness after the pipeline.
