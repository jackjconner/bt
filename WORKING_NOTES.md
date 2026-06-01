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
1291 tests pass, ruff clean. `uv run main.py` runs scaling exp → pipeline →
harness end-to-end.

**Improvement loop:** through round 009. Rounds 001–006 were perf/explore/
consolidate; **round 007 was the first `feature` round** — 7 components in
parallel, one additive flag-off/API-only capability each (signals pair-corr #49,
profiling r²-gating #50, portfolio risk-decomp #51, models fold-diagnostics #52,
backtest short-gating #53, etl quality-flags #54, analysis drawdown-recovery #55),
all merged, golden byte-identical (flags ship off). Round 008 activated backtest
short-gating on the production path (#60, golden-neutral). **Round 009 was the first
all-component `exploit` (perf) round** — 7 parallel workers, each landed a real
golden-safe win: models rank-IC vectorized ~2.8× (#67), backtest trade-log vectorized
n_assets-scaling 0.93→0.46 (#65), profiling fit_scaling −62% (#64), etl masked-pivot
−35% (#62), signals batched sector-OLS −24% (#66), analysis ordered group-by −21%
(#68), portfolio COO constraint-assembly −7…−27% (#63). Golden held 17/17 byte-identical;
clean sweep, all in-lane. Ideation/dedup remains load-bearing (round 007: 7 of 9 seeded
`FEATURE_BACKLOG.md` rows were already built).

## Active task
None — all goal tasks complete. Two benign pytest warnings remain (ElasticNet
l1_ratio=0 convergence; spearmanr ConstantInput) from intentional edge-case
tests in models/.

**Open loop follow-ups (round 007):**
- ~~`scripts/gate eval` resolves the golden at the *worktree* root~~ **FIXED:**
  `gate` now resolves the golden from `dirname $(git rev-parse --git-common-dir)`
  (the primary checkout) so a fresh worktree finds `.oversight/golden.json`
  without hand-copying it. cwd/TMPDIR stay pinned to the worktree; only the golden
  path resolves to primary. Verified from both primary and a throwaway worktree.
- ~~`consolidate-check` for `gate`~~ **DONE:** `scripts/gate consolidate-check
  [ref]` computes `ref`'s (default `main`) PipelineSummary fresh in an ephemeral
  worktree and diffs the current tree against it exactly — same-machine runs cancel
  the stored golden's ~1e-16 fp-noise. Wired into both loop skills as the
  refactor/consolidate evaluation bar (replaces `eval --tolerance 0`). Smoke-tested
  main-vs-main: every field `0.00e+00`.
- `FEATURE_BACKLOG.md` still-queued: F-006 (portfolio EWMA factor-cov), F-008
  (backtest order types). Round 008 activated **backtest** short-gating on
  the production path (#60) — turned out golden-neutral (the production book is
  long-only, so gating binds on nothing; correctness insurance for any future short
  book). **etl quality-flags** (contract change) and **models fold-diagnostics** (no
  consumer yet) remain dormant by decision — activate when a consumer/need appears.
  Per-feature default-state is now a deliberate, signed-off call (DECISIONS.md).

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
- Two skills, split by role, for iteratively improving a component behind its
  API: `.claude/skills/improvement-orchestrator/` (the conductor — proposes a
  target, fans out per-component workers in worktrees, adjudicates the gates,
  serial-merges with re-validation, ratchets, then dispatches a docs agent) and
  `.claude/skills/component-improvement-loop/` (the worker a dispatched sub-agent
  follows — additive change in its lane, lands a PR via `pr-writeup.md`). Every
  PR gated on lint/types/correctness/profiling/evaluation. Ledgers:
  `IMPROVEMENTS.md` (round log + dedup) and `API_REQUESTS.md` (data requests).

## Don't lose
- `data/` dir is stale bytecode only (old pkg removed in 5fc0dfe); ignore it.
- `date_axis` uses calendar days; production wants trading days via
  `trading_calendar`. Tracked as a cross-cutting fix.
- CLAUDE.md: consult before novel architecture; don't add fallbacks silently.
</content>
