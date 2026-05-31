# Working Notes

## Where we are
Taking every module from POC → production. Feature plans + unique data
schemas are in `PRODUCTION_PLAN.md` (produced by 7 module sub-agents).
Now building: (1) synthetic data layer for all unique schemas, then
(2) the production features per module.

## Active task
Phase A — synthetic data foundation: `etl/schema.py` (schema + validation),
`etl/datasets.py` (generator per schema + registry). Then Phase B — features.

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
`uv run pytest -q` and `uv run ruff check`. No commit hooks in this repo.

## Don't lose
- `data/` dir is stale bytecode only (old pkg removed in 5fc0dfe); ignore it.
- `date_axis` uses calendar days; production wants trading days via
  `trading_calendar`. Tracked as a cross-cutting fix.
- CLAUDE.md: consult before novel architecture; don't add fallbacks silently.
</content>
