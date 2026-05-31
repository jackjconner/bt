# bt

A synthetic-data-first quantitative backtesting system, built as seven
independent components that compose across stable public APIs — and designed to
be **iteratively improved by agents** behind those APIs, each change gated on
correctness, profiling, and evaluation.

## What it is

`bt` takes a strategy from raw data to a measured backtest:

```
data → signals → models → portfolio → backtest → analysis
                    ↑                                 ↑
                  etl (schema-validated ingestion)  profiling (measures it all)
```

Every panel is long-format `(date, id, value)`; returns and prices are in
percent units and reconcile (`close[t]/close[t-1]-1 == return/100`); panels live
on a business-day session axis. The substrate is **injected synthetic data** —
32 schema-validated datasets whose features and signals carry a known, modest
correlation with forward returns — so the production features (leakage controls,
IC recovery, risk attribution) have real structure to defend against. The same
datasets drive the unit tests, the integration suite, and the profiling harness.

## Components

Each is a module package with a `_protocol.py` (a `typing.Protocol` contract), an
intact proof-of-concept path, and an **additive** production path beside it
(feature flags default to the POC behavior, so old call sites run unchanged).

| component | does |
|---|---|
| **etl** | synthetic data generators + schema registry; typed loading, PIT as-of joins, survivorship-free universe, quality checks, corporate-action adjustment, calendar alignment |
| **signals** | alpha research — IC + horizon decay, quantile spreads, neutralization, turnover-aware scoring, combination, multiple-testing correction |
| **models** | leakage-safe cross-validation — purged/embargoed + walk-forward splitters, panel alignment, per-fold scaling, model zoo, persistence |
| **portfolio** | construction — factor risk model (`Σ = B·F·Bᵀ + D`), Ledoit-Wolf covariance, constrained mean-variance optimizer, tracking error, VaR/CVaR |
| **backtest** | simulation — transaction costs, square-root slippage, execution lag, universe masking, corporate actions, share-level accounting, constraints |
| **analysis** | performance & risk analytics from a `BacktestResult` — CAGR/Sortino/Calmar, benchmark-relative metrics, turnover, attribution, rolling/periodic tables |
| **profiling** | measurement — repeated-trial percentiles, tracemalloc peak, parquet persistence, regression detection vs baselines, log-log scaling-curve fitting, within-stage flame graphs (pyinstrument CPU / memray memory) indexed in parquet |

## Run it

```bash
uv run main.py        # scaling experiment → full production pipeline → component profiling harness
```

`pipeline.py::run_production_pipeline` runs the whole chain end-to-end and
returns a `PipelineSummary` (signal IC, walk-forward IC/R², optimizer
convergence, Sharpe gross/net, cost drag). The `harness/` package profiles each
component over a `GenSpec` grid by dogfooding the profiling module, and fits
scaling curves — which is how the portfolio optimizer's super-linear
`n_assets^1.5–2.8` cost was surfaced.

## Develop it

The repo is gated. Every commit goes through the atomic-commit wrapper, which
runs lint + types before committing:

```bash
scripts/committer "type(scope): subject" path/to/file ...   # ruff + ty gated; never --no-verify
scripts/worktree <branch>                                  # new .worktrees/<branch> + launch Claude Code there
uv run pytest -q                                            # 458 tests (unit + tests/integration/)
uv run ruff check && uv run ruff format --check             # lint + format (line-length 100)
uv run ty check                                             # strict types, error-on-warning
```

- **No suppression.** No `# type: ignore` / `# noqa` / per-rule ignores — fix the
  issue or widen the annotation. ty is at zero across the repo.
- **CI** (`.github/workflows/ci.yml`) re-runs pytest + ruff + ty on every PR and
  push to `main`, so the gates are enforced in the pipeline, not just locally.

## Agentic improvement loop

The repo ships two skills for iteratively improving a component *behind its
public API*, split by role:

- `.claude/skills/improvement-orchestrator/` — the **conductor**. Proposes a
  target (often a profiling-flagged hotspot), fans out per-component workers in
  isolated git worktrees, adjudicates the gates, serial-merges with post-merge
  re-validation, ratchets the baseline, and dispatches a docs agent afterward.
- `.claude/skills/component-improvement-loop/` — the **worker** a dispatched
  sub-agent follows: improve one component in your worktree, keep the change
  additive, and land a reviewed PR (`pr-writeup.md`).

Every PR is gated on **lint · types · correctness · profiling · evaluation**.
Two ledgers keep the loop honest:

- `IMPROVEMENTS.md` — append-only round log (the loop's memory + dedup source)
- `API_REQUESTS.md` — inter-agent data requests; APIs only ever grow additively

## Layout

```
etl/ signals/ models/ analysis/ portfolio/ backtest/ profiling/   # the 7 components
harness/                  # per-component profiling harness (dogfoods profiling/)
tests/integration/        # cross-component contract tests + full-pipeline e2e
pipeline.py  main.py      # production pipeline + orchestration entry point
scripts/committer         # gated atomic-commit wrapper
PRODUCTION_PLAN.md  DECISIONS.md  WORKING_NOTES.md                 # plan / ADRs / current state
```
