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

## 2026-05-30 — Improvement-loop worktrees live in `.worktrees/`
**Context:** The improvement loop dispatches one worker sub-agent per component
into its own git worktree. No location was specified, so trees could land
anywhere relative to the repo.
**Decision:** Workers create their worktree at `.worktrees/<component>-<slug>`
(branch `improve/<component>-<slug>` off `main`). `.worktrees/` is gitignored.
**Rationale:** A single known, ignored directory keeps the repo root clean,
makes the trees easy to find and prune, and avoids them ever being staged.
**Consequences:** `.worktrees/` added to `.gitignore`; the
`improvement-orchestrator` and `component-improvement-loop` skills name the path.

## 2026-05-30 — Heavy run temp goes to disk, not tmpfs `/tmp`
**Context:** The harness, production pipeline, and evalgate each write GB-scale
per-run parquet panels via `tempfile.mkdtemp()`, which honors `$TMPDIR`. On this
box `$TMPDIR` defaults to `/tmp`, a 16G RAM-backed tmpfs. The first explore round
dispatched 3 worktree workers that each ran the harness into `/tmp`; tmpfs filled,
every write failed with `EDQUOT`, git could no longer write its index and aborted
(SIGABRT), and the interactive shell died — a full cascade from one full RAM disk.
It masqueraded as a disk quota; `/home` is plain ext4 with no quota and ~900G free.
**Decision:** Add `scripts/diskguard`: sweeps leaked `bt_*`/`bench_*` temp, points
heavy temp at a disk path (`$BT_TMPDIR`, default `~/.cache/bt-tmp` on the ext4 home
fs), and refuses to proceed if that fs is tmpfs or low on space. Source it before
any harness/pipeline/evalgate run. The `improvement-orchestrator` and
`component-improvement-loop` skills make it a preflight.
**Rationale:** A single `export TMPDIR=<disk>` redirects every downstream
`mkdtemp` (harness internals included) off RAM in one lever, no code change. The
sweep stops leaked temp from a killed run compounding the next.
**Consequences:** Rounds must `source scripts/diskguard` first; large temp now
lands on disk (slower than RAM, but unbreakable). The Workflow tool's worktree
isolation was abandoned for explore rounds in the same incident (it spawned the
runaway copies); the tournament now runs as supervised `Agent` dispatch.

## 2026-05-31 — No backwards-compat; removal allowed; two-phase add-then-consolidate
**Context:** The improvement loop enforced additive-only API discipline (never
remove/rename an exported symbol; new params/fields take defaults). Round 005 (a
5-way explore) showed the cost: every worker, following the rule, *shadowed* the
old implementation instead of replacing it — signals kept `engine="matrix"`,
models kept the non-ridge loop, etl kept `_adjust_single_asset`, backtest kept
the incumbent loop, analysis kept `analyze` delegating to `analyze_fused`.
Several retained paths are pure dead shadows kept only because removal was
forbidden. bt has no external API consumers, so the compat burden buys nothing.
**Decision:** Drop additive-only. Removing/renaming internal symbols, changing
signatures, and deleting superseded paths is allowed in **any** round, provided
every call site is updated and the **golden stays byte-identical**
(`scripts/gate eval --tolerance 0`) with the integration suite green — those two
are the guards that a removal didn't change behavior. The *preferred workflow* is
two-phase: an explore/feature round adds the new path additively (keeping the old
one as a correctness **oracle** while the change is under review), and a
dedicated **`consolidate`** round then deletes the superseded path + its flag and
makes the replacement sole. Removal is permitted outside consolidate rounds too;
the two-phase split is a safety preference, not a restriction. Supersedes the
additive-only API discipline previously stated in the loop skills.
**Rationale:** No external users → backwards-compat is dead weight. Keeping the
old impl as an oracle during the *innovation* PR is genuinely useful (it's how
round 005 proved bit-identity); keeping it forever is just cruft. Two phases
capture the benefit without the cost; the golden-byte-identical bar makes a
removal as safe as a refactor.
**Consequences:** New `consolidate` round type (delete the shadow, make the
replacement sole, golden byte-identical). The loop skills' "additive only"
sections are rewritten to "removal allowed, prefer two-phase." Round 005 left 5
shadowed components — the genuinely-dead paths (e.g. signals `engine="matrix"`)
are the first consolidate targets; functional fallbacks (models' non-ridge loop,
etl's oracle) stay until truly unused.
