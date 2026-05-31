"""Root conftest: presence puts the repo root on sys.path so tests in nested
directories (e.g. tests/integration) can import the top-level packages
(etl, backtest, harness, pipeline, ...)."""
