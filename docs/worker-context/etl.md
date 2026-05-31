# etl — data substrate (synthetic generators + schema registry) and production ingestion path

## Files
- `datasets.py` — `GenSpec`, `REGISTRY` of seeded generators, `generate`/`generate_all`/`write_all`; price↔return reconciliation + injected predictive structure (largest file).
- `schema.py` — `Schema`/`Column`/`validate()` (dtype, nullability, key-uniqueness) + `SchemaError`.
- `loader.py` — `DatasetLoader`: lazy `scan_parquet`, date/id predicate + column projection pushdown, schema validation at load.
- `adjust.py` — `adjust_prices` → split/dividend back-adjusted close + audit log (vectorized: join_asof + log-space reverse-cumprod).
- `masked_pivot.py` — `to_masked_matrix`: dense matrix + validity mask (NaN ≠ zero).
- `quality.py` — `check` → `QualityReport` (dup keys, missing sessions, frozen series, return-spike z-outliers).
- `pit.py` — `as_of_slice` / `latest_as_of` (knowledge-date as-of joins, no future leakage).
- `universe.py` — `resolve_universe`, `apply_security_master` (listing/delisting overlay).
- `calendar.py` — `align_to_calendar`, `fill_sessions`, `sessions_between`.
- `source.py` — `to_matrix`, `date_axis`, `generate_returns`, `write_parquet`, `to_float` (POC + matrix/axis conventions).
- `batch.py` / `stream.py` — POC `BatchLoader`/`ETLConfig`, `StreamLoader`.

## Public API (additive-only contract — do not break)
`__all__`: `AdjustmentResult`, `BatchLoader`, `DatasetLoader`, `ETLConfig`, `Loader`, `QualityReport`, `StreamLoader`, `adjust_prices`, `align_to_calendar`, `apply_security_master`, `as_of_slice`, `check`, `date_axis`, `fill_sessions`, `generate_returns`, `latest_as_of`, `resolve_universe`, `sessions_between`, `to_float`, `to_masked_matrix`, `to_matrix`, `write_parquet`.
Protocol (`_protocol.py`): `Loader.load(self) -> pl.DataFrame`.
Key sigs: `adjust_prices(prices, corporate_actions, *, close_col="close") -> AdjustmentResult`; `to_masked_matrix(df, value_col) -> (matrix, mask, dates, ids)`; `check(df, value_col, *, expected_dates=None) -> QualityReport`.

## Harness entry / hot path
`harness/components.py::_etl_run` (component-benchmark path only — etl is NOT on the pipeline golden path). Timed call runs `adjust_prices(prices, corporate_actions)` + `to_masked_matrix(prices, "close")` + `check(prices, "close")`. `adjust_prices` dominates runtime.

## Data contract
Consumes `prices` (date,id,close,…), `corporate_actions` (split/dividend events). GenSpec fields that scale it: `n_assets`, `n_dates`. Full ingestion path also reads `fundamentals`, `shares_outstanding`, `universe_mask`, `security_master`, `trading_calendar`, `fx_rates`.

## Recently optimized (don't re-attempt — see IMPROVEMENTS.md)
- `adjust_prices` vectorized over the whole panel: one join_asof + segment-reset reverse-cumprod in log space replaced per-asset partition→Python-loop→concat (−46% to −94%, ~n^0.78). PR #39.
- Earlier etl perf round: per-asset filter + `iter_rows`→`partition_by`/`pivot` (4.9× at scale). PR #31.
