# Working Notes

## Where we are
DONE. Every module taken POC → production. `PRODUCTION_PLAN.md` holds the
per-module feature plans + unique schema list. Built: synthetic data layer
(`etl/schema.py`, `etl/datasets.py` — 31 datasets, schema-validated), the
production features in all 7 modules (`pipeline.py` wires them e2e), and a
component profiling harness + integration suite:
- `harness/` — `ComponentBenchmark` per module, `runner.run_harness` drives a
  GenSpec grid through `profiling.run_trials`, persists to parquet, fits
  scaling curves, optional regression-vs-baseline. Dogfoods the profiling
  module. `main.py` runs it after the pipeline.
- `tests/integration/` — per-component contract tests + full e2e + harness
  smoke; root `conftest.py` enables nested-test imports.
458 tests pass, ruff clean. `uv run main.py` runs scaling exp → pipeline →
harness end-to-end.

## Active task
None — all goal tasks complete. Two benign pytest warnings remain (ElasticNet
l1_ratio=0 convergence; spearmanr ConstantInput) from intentional edge-case
tests in models/.

## Follow-ups if resumed
- Calendar vs session axis: raw `generate_returns` still uses calendar days;
  the pipeline derives returns from the price panel to stay session-aligned.
  Could migrate `generate_returns` to `session_axis` (DECISIONS.md cross-cut).
- Infra-only plan items not built (out of code scope): flamegraphs/py-spy,
  HTML tear sheets. (CI gating now wired: `.github/workflows/ci.yml` runs
  pytest + ruff + ty on PRs and push to `main`.)

## Recent decisions (and why)
- Synthetic generators + schema registry live under `etl/` (that's where
  `generate_returns`/`write_parquet` already live). See DECISIONS.md.
- Long format `(date, id, ...)`, seeded `np.random.default_rng`, percent
  units for returns/prices to match the engine's existing `R / 100.0`.

## Build order (Phase B)
etl (validation/loaders/PIT/calendar) → signals → models → analysis →
portfolio → backtest → profiling. etl first because the others consume its
loaders; backtest late because it depends on costs/calendar/mask.

## Verification
`uv run pytest -q`, `uv run ruff check`, `uv run ruff format --check`,
`uv run ty check`. Commit via `scripts/committer "<msg>" [paths...]`, which
gates every commit on ruff (lint + format) and ty before committing — no
`--no-verify`, no suppression markers (`# type: ignore`/`# noqa`). ruff is
line-length 100 / strong select minus the ambiguous-unicode RUF rules; ty is
strict with `error-on-warning`. Both are green repo-wide.

## Gates + improvement loop (added this session)
- ruff + ty are dev deps with config in `pyproject.toml`; `scripts/committer`
  enforces them per commit. ty was driven 293→0 with no suppression (schema
  `DataTypeLike` annotation, `etl.source.to_float` for polars-aggregation
  typing, generic `ComponentBenchmark`, widened annotations, typed casts).
- `.claude/skills/component-improvement-loop/` — the skill for iteratively
  improving a component behind its API: head agent proposes a target, fans out
  per-component sub-agents in worktrees, each lands a PR (`pr-writeup.md`)
  gated on lint/types/correctness/profiling/evaluation, serial-merged with
  re-validation, then a docs agent reconciles. Ledgers: `IMPROVEMENTS.md`
  (round log + dedup) and `API_REQUESTS.md` (inter-agent data requests).

## Don't lose
- `data/` dir is stale bytecode only (old pkg removed in 5fc0dfe); ignore it.
- `date_axis` uses calendar days; production wants trading days via
  `trading_calendar`. Tracked as a cross-cutting fix.
- CLAUDE.md: consult before novel architecture; don't add fallbacks silently.
</content>
