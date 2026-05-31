"""Synthetic generators for every dataset in the production schema list.

Each dataset is declared once (name → Schema + generator) in ``REGISTRY``.
Generators follow the conventions of ``etl.source.generate_returns``: long
format sorted by ``(date, id)``, a seeded ``np.random.default_rng``, and a
business-day ``session_axis``. Returns/prices stay in **percent units** so the
backtest engine's existing ``R / 100.0`` accounting holds (see DECISIONS.md).

Coupling that gives the data real structure to test against:
- ``prices.close`` compounds the same percent returns as the ``returns`` set.
- ``feature_panel`` / ``alpha_signals`` carry an injected, modest correlation
  with next-day forward returns, so IC and cross-validated R² are
  positive-but-small rather than pure noise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .schema import Schema, col
from .source import session_axis

CURRENCIES = ["USD", "EUR", "JPY", "GBP"]
SECTORS = ["Tech", "Financials", "Energy", "Health", "Industrials", "Consumer"]
INDUSTRIES = ["Software", "Banks", "Oil", "Pharma", "Machinery", "Retail"]
COUNTRIES = ["US", "DE", "JP", "GB"]
EXCHANGES = ["XNYS", "XNAS", "XLON", "XTKS"]
ACTION_TYPES = ["split", "cash_dividend", "special_dividend", "spinoff", "delisting"]
SIGNAL_NAMES = ["momentum", "value", "quality"]
FEATURE_CATEGORIES = ["momentum", "value", "quality", "growth", "volatility"]
STAGES = [
    "etl.batch",
    "etl.stream",
    "backtest",
    "analysis",
    "portfolio.holdings",
    "portfolio.exposures",
    "portfolio.rolling_vol",
    "signals.ic",
    "models.cv",
]
FORWARD_HORIZONS = (1, 5, 21, 63)


@dataclass(frozen=True)
class GenSpec:
    n_assets: int
    n_dates: int
    n_features: int = 20
    n_factors: int = 5
    n_signals: int = len(SIGNAL_NAMES)
    n_benchmarks: int = 1
    start: str = "2000-01-01"
    seed: int = 0


def _rng(spec: GenSpec, salt: int) -> np.random.Generator:
    return np.random.default_rng(spec.seed + salt)


def _dates(spec: GenSpec) -> pl.Series:
    return session_axis(spec.n_dates, spec.start).alias("date")


def _panel(spec: GenSpec) -> pl.DataFrame:
    """(date, id) grid, date-major / id-minor — matches C-order matrix flatten."""
    ids = pl.int_range(0, spec.n_assets, eager=True).alias("id")
    return pl.DataFrame({"date": _dates(spec)}).join(
        pl.DataFrame({"id": ids}), how="cross"
    )


def _returns_matrix(spec: GenSpec) -> np.ndarray:
    """The canonical percent-return matrix (n_dates, n_assets), shared by the
    ``returns`` dataset and ``prices`` so they reconcile exactly."""
    return _rng(spec, 0).normal(0.0, 3.0, (spec.n_dates, spec.n_assets))


def _forward_matrix(spec: GenSpec, horizon: int) -> np.ndarray:
    """Compounded forward percent return over `horizon` sessions; trailing rows
    without a full horizon are NaN."""
    R = _returns_matrix(spec) / 100.0
    nd, na = R.shape
    out = np.full((nd, na), np.nan)
    growth = 1.0 + R
    for t in range(nd - horizon):
        out[t] = (np.prod(growth[t + 1 : t + 1 + horizon], axis=0) - 1.0) * 100.0
    return out


def _attach(grid: pl.DataFrame, name: str, mat: np.ndarray) -> pl.DataFrame:
    return grid.with_columns(pl.Series(name, mat.reshape(-1)))


# --------------------------------------------------------------------------- #
# A. market-data panels
# --------------------------------------------------------------------------- #

PRICES = Schema(
    "prices",
    (
        col("date", pl.Date),
        col("id", pl.Int64),
        col("open", pl.Float64),
        col("high", pl.Float64),
        col("low", pl.Float64),
        col("close", pl.Float64),
        col("vwap", pl.Float64),
        col("volume", pl.Float64),
        col("dollar_volume", pl.Float64),
        col("adv_20", pl.Float64),
        col("currency", pl.Categorical),
    ),
    keys=("date", "id"),
)


def gen_prices(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 1)
    R = _returns_matrix(spec) / 100.0
    nd, na = R.shape
    base = rng.uniform(10.0, 200.0, na)
    close = base * np.cumprod(1.0 + R, axis=0)
    hi_noise = np.abs(rng.normal(0.0, 0.01, (nd, na)))
    lo_noise = np.abs(rng.normal(0.0, 0.01, (nd, na)))
    high = close * (1.0 + hi_noise)
    low = close * (1.0 - lo_noise)
    prev = np.vstack([close[:1], close[:-1]])
    open_ = np.clip(prev * (1.0 + rng.normal(0.0, 0.005, (nd, na))), low, high)
    vwap = (high + low + close) / 3.0
    volume = rng.lognormal(12.0, 1.0, (nd, na))
    dollar_volume = volume * close
    adv20 = np.empty_like(dollar_volume)
    for t in range(nd):
        adv20[t] = dollar_volume[max(0, t - 19) : t + 1].mean(axis=0)

    grid = _panel(spec)
    for name, mat in [
        ("open", open_), ("high", high), ("low", low), ("close", close),
        ("vwap", vwap), ("volume", volume), ("dollar_volume", dollar_volume),
        ("adv_20", adv20),
    ]:
        grid = _attach(grid, name, mat)
    return grid.with_columns(pl.lit("USD").cast(pl.Categorical).alias("currency"))


SHARES_OUTSTANDING = Schema(
    "shares_outstanding",
    (
        col("date", pl.Date),
        col("id", pl.Int64),
        col("shares_outstanding", pl.Float64),
        col("market_cap", pl.Float64),
    ),
    keys=("date", "id"),
)


def gen_shares_outstanding(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 2)
    na = spec.n_assets
    base_shares = rng.uniform(1e7, 1e9, na)
    prices = gen_prices(spec).select("date", "id", "close")
    grid = _panel(spec).with_columns(
        pl.Series("shares_outstanding", np.tile(base_shares, spec.n_dates))
    )
    return grid.join(prices, on=["date", "id"]).with_columns(
        (pl.col("shares_outstanding") * pl.col("close")).alias("market_cap")
    ).select("date", "id", "shares_outstanding", "market_cap")


UNIVERSE_MASK = Schema(
    "universe_mask",
    (
        col("date", pl.Date),
        col("id", pl.Int64),
        col("in_universe", pl.Boolean),
        col("tradable", pl.Boolean),
        col("halted", pl.Boolean),
        col("listed", pl.Boolean),
    ),
    keys=("date", "id"),
)


def gen_universe_mask(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 3)
    n = spec.n_assets * spec.n_dates
    listed = rng.random(n) > 0.02
    halted = rng.random(n) < 0.01
    tradable = listed & ~halted
    in_universe = tradable & (rng.random(n) > 0.05)
    return _panel(spec).with_columns(
        pl.Series("in_universe", in_universe),
        pl.Series("tradable", tradable),
        pl.Series("halted", halted),
        pl.Series("listed", listed),
    )


BORROW_RATES = Schema(
    "borrow_rates",
    (
        col("date", pl.Date),
        col("id", pl.Int64),
        col("borrow_rate_bps", pl.Float64),
        col("shortable", pl.Boolean),
        col("loan_availability", pl.Float64),
    ),
    keys=("date", "id"),
)


def gen_borrow_rates(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 4)
    n = spec.n_assets * spec.n_dates
    rate = np.abs(rng.normal(50.0, 40.0, n))
    shortable = rng.random(n) > 0.1
    avail = np.where(shortable, rng.lognormal(11.0, 1.5, n), 0.0)
    return _panel(spec).with_columns(
        pl.Series("borrow_rate_bps", rate),
        pl.Series("shortable", shortable),
        pl.Series("loan_availability", avail),
    )


TRANSACTION_COSTS = Schema(
    "transaction_costs",
    (
        col("date", pl.Date),
        col("id", pl.Int64),
        col("commission_bps", pl.Float64),
        col("half_spread_bps", pl.Float64),
        col("impact_coef", pl.Float64),
        col("min_commission", pl.Float64),
        col("exchange_fee_bps", pl.Float64),
    ),
    keys=("date", "id"),
)


def gen_transaction_costs(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 5)
    n = spec.n_assets * spec.n_dates
    return _panel(spec).with_columns(
        pl.Series("commission_bps", np.full(n, 1.0)),
        pl.Series("half_spread_bps", np.abs(rng.normal(5.0, 3.0, n))),
        pl.Series("impact_coef", np.abs(rng.normal(0.1, 0.05, n))),
        pl.Series("min_commission", np.full(n, 1.0)),
        pl.Series("exchange_fee_bps", np.full(n, 0.3)),
    )


SPECIFIC_RISK = Schema(
    "specific_risk",
    (col("date", pl.Date), col("id", pl.Int64), col("specific_var", pl.Float64)),
    keys=("date", "id"),
)


def gen_specific_risk(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 6)
    n = spec.n_assets * spec.n_dates
    return _panel(spec).with_columns(
        pl.Series("specific_var", np.abs(rng.normal(4.0, 1.5, n)))
    )


# --------------------------------------------------------------------------- #
# B. factor data
# --------------------------------------------------------------------------- #

FACTOR_LOADINGS = Schema(
    "factor_loadings",
    (
        col("date", pl.Date),
        col("id", pl.Int64),
        col("factor_id", pl.Int64),
        col("loading", pl.Float64),
    ),
    keys=("date", "id", "factor_id"),
)


def gen_factor_loadings(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 7)
    nd, na, nk = spec.n_dates, spec.n_assets, spec.n_factors
    static = rng.normal(0.0, 1.0, (na, nk))
    dates = _dates(spec)
    ids = pl.int_range(0, na, eager=True)
    factors = pl.int_range(0, nk, eager=True)
    grid = (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"id": ids}), how="cross")
        .join(pl.DataFrame({"factor_id": factors}), how="cross")
    )
    drift = rng.normal(0.0, 0.05, (nd, na, nk))
    loading = static[None, :, :] + np.cumsum(drift, axis=0) * 0.0 + drift
    return grid.with_columns(pl.Series("loading", loading.reshape(-1)))


FACTOR_RETURNS = Schema(
    "factor_returns",
    (col("date", pl.Date), col("factor_id", pl.Int64), col("return", pl.Float64)),
    keys=("date", "factor_id"),
)


def gen_factor_returns(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 8)
    nd, nk = spec.n_dates, spec.n_factors
    ret = rng.normal(0.0, 1.0, (nd, nk))
    grid = pl.DataFrame({"date": _dates(spec)}).join(
        pl.DataFrame({"factor_id": pl.int_range(0, nk, eager=True)}), how="cross"
    )
    return grid.with_columns(pl.Series("return", ret.reshape(-1)))


FACTOR_COVARIANCE = Schema(
    "factor_covariance",
    (
        col("date", pl.Date),
        col("factor_i", pl.Int64),
        col("factor_j", pl.Int64),
        col("cov", pl.Float64),
    ),
    keys=("date", "factor_i", "factor_j"),
)


def gen_factor_covariance(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 9)
    nk = spec.n_factors
    a = rng.normal(0.0, 1.0, (nk, nk))
    cov = a @ a.T / nk + np.eye(nk)
    dates = _dates(spec)
    fi = pl.int_range(0, nk, eager=True)
    fj = pl.int_range(0, nk, eager=True)
    grid = (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"factor_i": fi}), how="cross")
        .join(pl.DataFrame({"factor_j": fj}), how="cross")
    )
    tiled = np.tile(cov.reshape(-1), spec.n_dates)
    return grid.with_columns(pl.Series("cov", tiled))


# --------------------------------------------------------------------------- #
# C. ML / signal training data
# --------------------------------------------------------------------------- #


def _feature_schema(spec: GenSpec) -> Schema:
    cols = [col("date", pl.Date), col("id", pl.Int64)]
    cols += [col(f"feat_{i}", pl.Float64) for i in range(spec.n_features)]
    return Schema("feature_panel", tuple(cols), keys=("date", "id"))


def gen_feature_panel(spec: GenSpec) -> pl.DataFrame:
    """Features carrying a modest injected correlation with next-day forward
    return, so cross-validated R²/IC is positive-but-small (not pure noise)."""
    rng = _rng(spec, 10)
    fwd = _forward_matrix(spec, 1)
    signal = np.nan_to_num(fwd)
    signal = (signal - signal.mean()) / (signal.std() + 1e-9)
    grid = _panel(spec)
    for i in range(spec.n_features):
        beta = 0.3 * rng.normal(0.0, 1.0)
        feat = beta * signal + rng.normal(0.0, 1.0, signal.shape)
        grid = _attach(grid, f"feat_{i}", feat)
    return grid


FORWARD_RETURNS = Schema(
    "forward_returns",
    (
        col("date", pl.Date),
        col("id", pl.Int64),
        col("fwd_ret_1", pl.Float64, nullable=True),
        col("fwd_ret_5", pl.Float64, nullable=True),
        col("fwd_ret_21", pl.Float64, nullable=True),
        col("fwd_ret_63", pl.Float64, nullable=True),
    ),
    keys=("date", "id"),
)


def gen_forward_returns(spec: GenSpec) -> pl.DataFrame:
    grid = _panel(spec)
    for h in FORWARD_HORIZONS:
        grid = _attach(grid, f"fwd_ret_{h}", _forward_matrix(spec, h))
    # trailing-horizon cells are np.nan; make them proper polars nulls so
    # drop_nulls / null_count treat them as missing (NaN != null in polars)
    return grid.with_columns(
        pl.col(f"fwd_ret_{h}").fill_nan(None) for h in FORWARD_HORIZONS
    )


ALPHA_SIGNALS = Schema(
    "alpha_signals",
    (
        col("date", pl.Date),
        col("id", pl.Int64),
        col("signal_name", pl.Categorical),
        col("signal", pl.Float64),
    ),
    keys=("date", "id", "signal_name"),
)


def gen_alpha_signals(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 11)
    fwd = np.nan_to_num(_forward_matrix(spec, 1))
    fwd = (fwd - fwd.mean()) / (fwd.std() + 1e-9)
    names = SIGNAL_NAMES[: spec.n_signals]
    frames = []
    for k, name in enumerate(names):
        ic = 0.05 + 0.03 * k
        sig = ic * fwd + np.sqrt(max(1.0 - ic * ic, 0.0)) * rng.normal(
            0.0, 1.0, fwd.shape
        )
        frame = _attach(_panel(spec), "signal", sig).with_columns(
            pl.lit(name).cast(pl.Categorical).alias("signal_name")
        )
        frames.append(frame.select("date", "id", "signal_name", "signal"))
    return pl.concat(frames)


SAMPLE_WEIGHTS = Schema(
    "sample_weights",
    (col("date", pl.Date), col("id", pl.Int64), col("weight", pl.Float64)),
    keys=("date", "id"),
)


def gen_sample_weights(spec: GenSpec) -> pl.DataFrame:
    """Recency-decayed weights normalized to mean 1.0."""
    rng = _rng(spec, 12)
    nd, na = spec.n_dates, spec.n_assets
    halflife = max(nd / 4.0, 1.0)
    age = (nd - 1) - np.arange(nd)
    decay = 0.5 ** (age / halflife)
    w = np.repeat(decay, na) * rng.uniform(0.5, 1.5, nd * na)
    w = w / w.mean()
    return _panel(spec).with_columns(pl.Series("weight", w))


# --------------------------------------------------------------------------- #
# D. per-date time series
# --------------------------------------------------------------------------- #

RISK_FREE_RATE = Schema(
    "risk_free_rate",
    (col("date", pl.Date), col("annual_rate", pl.Float64), col("daily_rate", pl.Float64)),
    keys=("date",),
)


def gen_risk_free_rate(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 13)
    steps = rng.normal(0.0, 0.0005, spec.n_dates)
    annual = np.clip(0.03 + np.cumsum(steps), 0.0, 0.10)
    return pl.DataFrame({"date": _dates(spec)}).with_columns(
        pl.Series("annual_rate", annual),
        pl.Series("daily_rate", annual / 252.0),
    )


BENCHMARK_RETURNS = Schema(
    "benchmark_returns",
    (col("date", pl.Date), col("benchmark_id", pl.Categorical), col("return", pl.Float64)),
    keys=("date", "benchmark_id"),
)


def gen_benchmark_returns(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 14)
    names = [f"BMK{i}" for i in range(spec.n_benchmarks)]
    frames = []
    for name in names:
        ret = rng.normal(0.03, 1.0, spec.n_dates)
        frames.append(
            pl.DataFrame({"date": _dates(spec)})
            .with_columns(
                pl.lit(name).cast(pl.Categorical).alias("benchmark_id"),
                pl.Series("return", ret),
            )
            .select("date", "benchmark_id", "return")
        )
    return pl.concat(frames)


BENCHMARK_WEIGHTS = Schema(
    "benchmark_weights",
    (col("date", pl.Date), col("id", pl.Int64), col("benchmark_weight", pl.Float64)),
    keys=("date", "id"),
)


def gen_benchmark_weights(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 15)
    nd, na = spec.n_dates, spec.n_assets
    raw = rng.uniform(0.5, 1.5, (nd, na))
    w = raw / raw.sum(axis=1, keepdims=True)
    return _attach(_panel(spec), "benchmark_weight", w)


FX_RATES = Schema(
    "fx_rates",
    (
        col("date", pl.Date),
        col("from_currency", pl.Categorical),
        col("to_currency", pl.Categorical),
        col("rate", pl.Float64),
    ),
    keys=("date", "from_currency", "to_currency"),
)


def gen_fx_rates(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 16)
    pairs = [(c, "USD") for c in CURRENCIES if c != "USD"]
    frames = []
    for frm, to in pairs:
        base = rng.uniform(0.5, 1.5)
        rate = base * np.cumprod(1.0 + rng.normal(0.0, 0.003, spec.n_dates))
        frames.append(
            pl.DataFrame({"date": _dates(spec)})
            .with_columns(
                pl.lit(frm).cast(pl.Categorical).alias("from_currency"),
                pl.lit(to).cast(pl.Categorical).alias("to_currency"),
                pl.Series("rate", rate),
            )
            .select("date", "from_currency", "to_currency", "rate")
        )
    return pl.concat(frames)


# --------------------------------------------------------------------------- #
# E. calendar / events
# --------------------------------------------------------------------------- #

TRADING_CALENDAR = Schema(
    "trading_calendar",
    (
        col("date", pl.Date),
        col("exchange", pl.Categorical),
        col("is_session", pl.Boolean),
        col("is_half_day", pl.Boolean),
        col("session_open", pl.String),
        col("session_close", pl.String),
    ),
    keys=("date", "exchange"),
)


def gen_trading_calendar(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 17)
    dates = _dates(spec)
    n = len(dates)
    half = rng.random(n) < 0.02
    return pl.DataFrame({"date": dates}).with_columns(
        pl.lit("XNYS").cast(pl.Categorical).alias("exchange"),
        pl.lit(True).alias("is_session"),
        pl.Series("is_half_day", half),
        pl.lit("09:30").alias("session_open"),
        pl.when(pl.Series(half)).then(pl.lit("13:00")).otherwise(pl.lit("16:00")).alias(
            "session_close"
        ),
    )


CORPORATE_ACTIONS = Schema(
    "corporate_actions",
    (
        col("ex_date", pl.Date),
        col("id", pl.Int64),
        col("action_type", pl.Categorical),
        col("split_ratio", pl.Float64, nullable=True),
        col("cash_amount", pl.Float64, nullable=True),
        col("currency", pl.Categorical),
        col("new_id", pl.Int64, nullable=True),
    ),
    keys=("ex_date", "id", "action_type"),
)


def gen_corporate_actions(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 18)
    dates = _dates(spec).to_list()
    n_events = max(int(spec.n_assets * spec.n_dates * 0.001), spec.n_assets // 10, 1)
    ex_idx = rng.integers(0, spec.n_dates, n_events)
    ids = rng.integers(0, spec.n_assets, n_events)
    types = rng.choice(ACTION_TYPES, n_events)
    split_ratio = np.where(types == "split", rng.choice([2.0, 3.0, 1.5], n_events), np.nan)
    cash = np.where(
        np.isin(types, ["cash_dividend", "special_dividend"]),
        np.abs(rng.normal(0.5, 0.3, n_events)),
        np.nan,
    )
    new_id = np.where(types == "spinoff", rng.integers(0, spec.n_assets, n_events), -1)
    new_id_list = [int(v) if v >= 0 else None for v in new_id]
    df = pl.DataFrame(
        {
            "ex_date": [dates[i] for i in ex_idx],
            "id": ids.astype(np.int64),
            "action_type": pl.Series(types, dtype=pl.Categorical),
            "split_ratio": split_ratio,
            "cash_amount": cash,
            "currency": pl.Series(["USD"] * n_events, dtype=pl.Categorical),
            "new_id": pl.Series("new_id", new_id_list, dtype=pl.Int64),
        }
    ).with_columns(
        pl.col("split_ratio").fill_nan(None),
        pl.col("cash_amount").fill_nan(None),
    )
    return df.unique(subset=["ex_date", "id", "action_type"], keep="first")


# --------------------------------------------------------------------------- #
# F. per-asset static / reference
# --------------------------------------------------------------------------- #

SECURITY_MASTER = Schema(
    "security_master",
    (
        col("id", pl.Int64),
        col("ticker", pl.String),
        col("name", pl.String),
        col("sector", pl.Categorical),
        col("industry", pl.Categorical),
        col("country", pl.Categorical),
        col("exchange", pl.Categorical),
        col("currency", pl.Categorical),
        col("lot_size", pl.Int64),
        col("listing_date", pl.Date),
        col("delisting_date", pl.Date, nullable=True),
        col("is_active", pl.Boolean),
    ),
    keys=("id",),
)


def gen_security_master(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 19)
    na = spec.n_assets
    sec_idx = rng.integers(0, len(SECTORS), na)
    dates = _dates(spec).to_list()
    start_d = dates[0]
    active = rng.random(na) > 0.05
    return pl.DataFrame(
        {
            "id": np.arange(na, dtype=np.int64),
            "ticker": [f"SYN{i:05d}" for i in range(na)],
            "name": [f"Synthetic Asset {i}" for i in range(na)],
            "sector": pl.Series([SECTORS[i] for i in sec_idx], dtype=pl.Categorical),
            "industry": pl.Series(
                [INDUSTRIES[i] for i in sec_idx], dtype=pl.Categorical
            ),
            "country": pl.Series(
                rng.choice(COUNTRIES, na), dtype=pl.Categorical
            ),
            "exchange": pl.Series(
                rng.choice(EXCHANGES, na), dtype=pl.Categorical
            ),
            "currency": pl.Series(["USD"] * na, dtype=pl.Categorical),
            "lot_size": rng.choice([1, 100], na).astype(np.int64),
            "listing_date": pl.Series("listing_date", [start_d] * na, dtype=pl.Date),
            "delisting_date": pl.Series(
                "delisting_date",
                [None if a else dates[-1] for a in active],
                dtype=pl.Date,
            ),
            "is_active": active,
        }
    )


FUNDAMENTALS = Schema(
    "fundamentals",
    (
        col("report_date", pl.Date),
        col("knowledge_date", pl.Date),
        col("id", pl.Int64),
        col("revenue", pl.Float64),
        col("net_income", pl.Float64),
        col("total_assets", pl.Float64),
        col("total_equity", pl.Float64),
        col("total_debt", pl.Float64),
        col("operating_cash_flow", pl.Float64),
        col("currency", pl.Categorical),
    ),
    keys=("report_date", "knowledge_date", "id"),
)


def gen_fundamentals(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 20)
    dates = _dates(spec).to_list()
    # quarterly report dates sampled from the session axis; knowledge_date lags ~45d
    q_idx = list(range(0, spec.n_dates, max(spec.n_dates // 4, 1)))
    rows = []
    for ri in q_idx:
        report = dates[ri]
        know_i = min(ri + 30, spec.n_dates - 1)
        know = dates[know_i]
        for aid in range(spec.n_assets):
            rev = rng.lognormal(18.0, 1.0)
            rows.append(
                {
                    "report_date": report,
                    "knowledge_date": know,
                    "id": aid,
                    "revenue": rev,
                    "net_income": rev * rng.uniform(-0.1, 0.2),
                    "total_assets": rev * rng.uniform(1.0, 3.0),
                    "total_equity": rev * rng.uniform(0.4, 1.5),
                    "total_debt": rev * rng.uniform(0.1, 1.0),
                    "operating_cash_flow": rev * rng.uniform(0.0, 0.3),
                    "currency": "USD",
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("id").cast(pl.Int64), pl.col("currency").cast(pl.Categorical)
    )


# --------------------------------------------------------------------------- #
# G. lookup / config tables
# --------------------------------------------------------------------------- #

FEATURE_METADATA = Schema(
    "feature_metadata",
    (
        col("feature", pl.String),
        col("category", pl.Categorical),
        col("winsorize", pl.Boolean),
        col("lookback_days", pl.Int64),
    ),
    keys=("feature",),
)


def gen_feature_metadata(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 21)
    nf = spec.n_features
    cats = rng.choice(FEATURE_CATEGORIES, nf)
    return pl.DataFrame(
        {
            "feature": [f"feat_{i}" for i in range(nf)],
            "category": pl.Series(cats, dtype=pl.Categorical),
            "winsorize": rng.random(nf) > 0.5,
            "lookback_days": rng.choice([5, 21, 63, 252], nf).astype(np.int64),
        }
    )


SIGNAL_REGISTRY = Schema(
    "signal_registry",
    (
        col("signal_name", pl.String),
        col("family", pl.Categorical),
        col("is_categorical", pl.Boolean),
        col("intended_horizon", pl.Int64),
    ),
    keys=("signal_name",),
)


def gen_signal_registry(spec: GenSpec) -> pl.DataFrame:
    names = SIGNAL_NAMES[: spec.n_signals]
    return pl.DataFrame(
        {
            "signal_name": names,
            "family": pl.Series(names, dtype=pl.Categorical),
            "is_categorical": [False] * len(names),
            "intended_horizon": [1 + 5 * i for i in range(len(names))],
        }
    ).with_columns(pl.col("intended_horizon").cast(pl.Int64))


POSITION_CONSTRAINTS = Schema(
    "position_constraints",
    (
        col("id", pl.Int64),
        col("min_weight", pl.Float64),
        col("max_weight", pl.Float64),
        col("tradable", pl.Boolean),
    ),
    keys=("id",),
)


def gen_position_constraints(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 22)
    na = spec.n_assets
    cap = rng.uniform(0.03, 0.10, na)
    return pl.DataFrame(
        {
            "id": np.arange(na, dtype=np.int64),
            "min_weight": -cap,
            "max_weight": cap,
            "tradable": rng.random(na) > 0.02,
        }
    )


GROUP_CONSTRAINTS = Schema(
    "group_constraints",
    (
        col("sector", pl.Categorical),
        col("min_exposure", pl.Float64),
        col("max_exposure", pl.Float64),
    ),
    keys=("sector",),
)


def gen_group_constraints(spec: GenSpec) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sector": pl.Series(SECTORS, dtype=pl.Categorical),
            "min_exposure": [-0.30] * len(SECTORS),
            "max_exposure": [0.30] * len(SECTORS),
        }
    )


CV_SPLITS_CALENDAR = Schema(
    "cv_splits_calendar",
    (
        col("fold", pl.Int64),
        col("train_start", pl.Date),
        col("train_end", pl.Date),
        col("embargo_days", pl.Int64),
        col("test_start", pl.Date),
        col("test_end", pl.Date),
    ),
    keys=("fold",),
)


def gen_cv_splits_calendar(spec: GenSpec, n_folds: int = 5, embargo: int = 5) -> pl.DataFrame:
    dates = _dates(spec).to_list()
    nd = spec.n_dates
    fold_size = nd // (n_folds + 1)
    rows = []
    for f in range(n_folds):
        train_end_i = fold_size * (f + 1) - 1
        test_start_i = min(train_end_i + embargo + 1, nd - 1)
        test_end_i = min(test_start_i + fold_size - 1, nd - 1)
        rows.append(
            {
                "fold": f,
                "train_start": dates[0],
                "train_end": dates[train_end_i],
                "embargo_days": embargo,
                "test_start": dates[test_start_i],
                "test_end": dates[test_end_i],
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("fold").cast(pl.Int64), pl.col("embargo_days").cast(pl.Int64)
    )


# --------------------------------------------------------------------------- #
# H. profiling / telemetry
# --------------------------------------------------------------------------- #

PROFILING_RUNS = Schema(
    "profiling_runs",
    (
        col("run_id", pl.String),
        col("run_ts", pl.Date),
        col("git_sha", pl.String),
        col("git_dirty", pl.Boolean),
        col("hostname", pl.String),
        col("cpu_model", pl.String),
        col("n_cores", pl.Int64),
        col("total_ram_mb", pl.Float64),
        col("python_version", pl.String),
        col("polars_version", pl.String),
        col("numpy_version", pl.String),
        col("blas_threads", pl.Int64),
        col("trials", pl.Int64),
        col("warmup_trials", pl.Int64),
    ),
    keys=("run_id",),
)


def gen_profiling_runs(spec: GenSpec, n_runs: int = 5) -> pl.DataFrame:
    dates = _dates(spec).to_list()
    rows = []
    for i in range(n_runs):
        rows.append(
            {
                "run_id": f"run_{i:04d}",
                "run_ts": dates[min(i, len(dates) - 1)],
                "git_sha": f"{i:040x}",
                "git_dirty": i % 2 == 0,
                "hostname": "synthetic-host",
                "cpu_model": "Synthetic CPU @ 3.0GHz",
                "n_cores": 16,
                "total_ram_mb": 65536.0,
                "python_version": "3.13.0",
                "polars_version": pl.__version__,
                "numpy_version": np.__version__,
                "blas_threads": 8,
                "trials": 5,
                "warmup_trials": 1,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("n_cores").cast(pl.Int64),
        pl.col("blas_threads").cast(pl.Int64),
        pl.col("trials").cast(pl.Int64),
        pl.col("warmup_trials").cast(pl.Int64),
    )


STAGE_MEASUREMENTS = Schema(
    "stage_measurements",
    (
        col("run_id", pl.String),
        col("param_point_id", pl.Int64),
        col("n_assets", pl.Int64),
        col("n_dates", pl.Int64),
        col("n_features", pl.Int64),
        col("n_factors", pl.Int64),
        col("stage", pl.Categorical),
        col("trial_idx", pl.Int64),
        col("elapsed_s", pl.Float64),
        col("result_mb", pl.Float64),
        col("rss_delta_mb", pl.Float64),
        col("peak_rss_mb", pl.Float64),
        col("peak_traced_mb", pl.Float64),
    ),
    keys=("run_id", "param_point_id", "stage", "trial_idx"),
)


def gen_stage_measurements(spec: GenSpec, n_runs: int = 5, n_trials: int = 5) -> pl.DataFrame:
    rng = _rng(spec, 23)
    # a real scaling grid: param points sweep n_assets (×1,2,4,8) so a log-log
    # fit over the grid has ≥3 points to fit and elapsed grows ~linearly.
    scales = [1, 2, 4, 8]
    rows = []
    for r in range(n_runs):
        for pp, mult in enumerate(scales):
            n_assets = spec.n_assets * mult
            for stage in STAGES:
                stage_cost = 0.01 + 0.4 * (STAGES.index(stage) / len(STAGES))
                for trial in range(n_trials):
                    elapsed = stage_cost * n_assets / spec.n_assets
                    rows.append(
                        {
                            "run_id": f"run_{r:04d}",
                            "param_point_id": pp,
                            "n_assets": n_assets,
                            "n_dates": spec.n_dates,
                            "n_features": spec.n_features,
                            "n_factors": spec.n_factors,
                            "stage": stage,
                            "trial_idx": trial,
                            "elapsed_s": elapsed * (1.0 + rng.normal(0.0, 0.03)),
                            "result_mb": 1.0 + n_assets * 0.05,
                            "rss_delta_mb": rng.uniform(0.0, 50.0),
                            "peak_rss_mb": 100.0 + n_assets * 2.0,
                            "peak_traced_mb": rng.uniform(1.0, 200.0),
                        }
                    )
    return pl.DataFrame(rows).with_columns(
        pl.col("param_point_id").cast(pl.Int64),
        pl.col("n_assets").cast(pl.Int64),
        pl.col("n_dates").cast(pl.Int64),
        pl.col("n_features").cast(pl.Int64),
        pl.col("n_factors").cast(pl.Int64),
        pl.col("stage").cast(pl.Categorical),
        pl.col("trial_idx").cast(pl.Int64),
    )


STAGE_BASELINES = Schema(
    "stage_baselines",
    (
        col("baseline_id", pl.String),
        col("param_point_id", pl.Int64),
        col("n_assets", pl.Int64),
        col("n_dates", pl.Int64),
        col("n_features", pl.Int64),
        col("n_factors", pl.Int64),
        col("stage", pl.Categorical),
        col("elapsed_s_p50", pl.Float64),
        col("elapsed_s_p90", pl.Float64),
        col("result_mb", pl.Float64),
        col("peak_rss_mb", pl.Float64),
        col("source_run_id", pl.String),
        col("created_ts", pl.Date),
    ),
    keys=("baseline_id", "param_point_id", "stage"),
)


def gen_stage_baselines(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 24)
    created = _dates(spec).to_list()[0]
    rows = []
    for pp in range(2):
        for stage in STAGES:
            p50 = rng.uniform(0.01, 0.5) * (pp + 1)
            rows.append(
                {
                    "baseline_id": "baseline_main",
                    "param_point_id": pp,
                    "n_assets": spec.n_assets,
                    "n_dates": spec.n_dates,
                    "n_features": spec.n_features,
                    "n_factors": spec.n_factors,
                    "stage": stage,
                    "elapsed_s_p50": p50,
                    "elapsed_s_p90": p50 * 1.3,
                    "result_mb": rng.uniform(1.0, 100.0),
                    "peak_rss_mb": rng.uniform(100.0, 2000.0),
                    "source_run_id": "run_0000",
                    "created_ts": created,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("param_point_id").cast(pl.Int64),
        pl.col("n_assets").cast(pl.Int64),
        pl.col("n_dates").cast(pl.Int64),
        pl.col("n_features").cast(pl.Int64),
        pl.col("n_factors").cast(pl.Int64),
        pl.col("stage").cast(pl.Categorical),
    )


REGRESSION_THRESHOLDS = Schema(
    "regression_thresholds",
    (
        col("stage", pl.Categorical),
        col("metric", pl.Categorical),
        col("max_pct_increase", pl.Float64),
        col("max_abs_increase", pl.Float64),
        col("min_samples", pl.Int64),
    ),
    keys=("stage", "metric"),
)


def gen_regression_thresholds(spec: GenSpec) -> pl.DataFrame:
    metrics = ["elapsed_s", "peak_rss_mb", "result_mb"]
    rows = []
    for stage in STAGES:
        for metric in metrics:
            rows.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "max_pct_increase": 0.20,
                    "max_abs_increase": 0.05,
                    "min_samples": 3,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("stage").cast(pl.Categorical),
        pl.col("metric").cast(pl.Categorical),
        pl.col("min_samples").cast(pl.Int64),
    )


SCALING_FITS = Schema(
    "scaling_fits",
    (
        col("run_id", pl.String),
        col("stage", pl.Categorical),
        col("metric", pl.Categorical),
        col("scaling_dim", pl.Categorical),
        col("log_log_slope", pl.Float64),
        col("intercept", pl.Float64),
        col("r_squared", pl.Float64),
        col("n_points", pl.Int64),
    ),
    keys=("run_id", "stage", "metric", "scaling_dim"),
)


def gen_scaling_fits(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 25)
    dims = ["n_assets", "n_dates", "n_features", "n_factors"]
    rows = []
    for stage in STAGES:
        for dim in dims:
            rows.append(
                {
                    "run_id": "run_0000",
                    "stage": stage,
                    "metric": "elapsed_s",
                    "scaling_dim": dim,
                    "log_log_slope": rng.uniform(0.8, 2.0),
                    "intercept": rng.normal(0.0, 1.0),
                    "r_squared": rng.uniform(0.7, 0.99),
                    "n_points": 4,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("stage").cast(pl.Categorical),
        pl.col("metric").cast(pl.Categorical),
        pl.col("scaling_dim").cast(pl.Categorical),
        pl.col("n_points").cast(pl.Int64),
    )


CPU_PROFILE_FRAMES = Schema(
    "cpu_profile_frames",
    (
        col("run_id", pl.String),
        col("param_point_id", pl.Int64),
        col("stage", pl.Categorical),
        col("function", pl.String),
        col("filename", pl.String),
        col("lineno", pl.Int64),
        col("cumulative_s", pl.Float64),
        col("self_s", pl.Float64),
        col("call_count", pl.Int64),
    ),
    keys=("run_id", "param_point_id", "stage", "function"),
)


def gen_cpu_profile_frames(spec: GenSpec) -> pl.DataFrame:
    rng = _rng(spec, 26)
    funcs = ["to_matrix", "collect", "softmax", "cov", "fit", "spearmanr"]
    rows = []
    for stage in STAGES:
        for i, fn in enumerate(funcs):
            cum = rng.uniform(0.001, 0.2)
            rows.append(
                {
                    "run_id": "run_0000",
                    "param_point_id": 0,
                    "stage": stage,
                    "function": fn,
                    "filename": f"{stage.split('.')[0]}/module.py",
                    "lineno": 10 + i * 7,
                    "cumulative_s": cum,
                    "self_s": cum * rng.uniform(0.3, 1.0),
                    "call_count": int(rng.integers(1, 1000)),
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("param_point_id").cast(pl.Int64),
        pl.col("stage").cast(pl.Categorical),
        pl.col("lineno").cast(pl.Int64),
        pl.col("call_count").cast(pl.Int64),
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dataset:
    name: str
    schema: Callable[[GenSpec], Schema] | Schema
    generate: Callable[[GenSpec], pl.DataFrame]

    def schema_for(self, spec: GenSpec) -> Schema:
        return self.schema(spec) if callable(self.schema) else self.schema


def _ds(name: str, schema: Schema | Callable[[GenSpec], Schema], gen) -> Dataset:
    return Dataset(name=name, schema=schema, generate=gen)


REGISTRY: dict[str, Dataset] = {
    d.name: d
    for d in [
        _ds("prices", PRICES, gen_prices),
        _ds("shares_outstanding", SHARES_OUTSTANDING, gen_shares_outstanding),
        _ds("universe_mask", UNIVERSE_MASK, gen_universe_mask),
        _ds("borrow_rates", BORROW_RATES, gen_borrow_rates),
        _ds("transaction_costs", TRANSACTION_COSTS, gen_transaction_costs),
        _ds("specific_risk", SPECIFIC_RISK, gen_specific_risk),
        _ds("factor_loadings", FACTOR_LOADINGS, gen_factor_loadings),
        _ds("factor_returns", FACTOR_RETURNS, gen_factor_returns),
        _ds("factor_covariance", FACTOR_COVARIANCE, gen_factor_covariance),
        _ds("feature_panel", _feature_schema, gen_feature_panel),
        _ds("forward_returns", FORWARD_RETURNS, gen_forward_returns),
        _ds("alpha_signals", ALPHA_SIGNALS, gen_alpha_signals),
        _ds("sample_weights", SAMPLE_WEIGHTS, gen_sample_weights),
        _ds("risk_free_rate", RISK_FREE_RATE, gen_risk_free_rate),
        _ds("benchmark_returns", BENCHMARK_RETURNS, gen_benchmark_returns),
        _ds("benchmark_weights", BENCHMARK_WEIGHTS, gen_benchmark_weights),
        _ds("fx_rates", FX_RATES, gen_fx_rates),
        _ds("trading_calendar", TRADING_CALENDAR, gen_trading_calendar),
        _ds("corporate_actions", CORPORATE_ACTIONS, gen_corporate_actions),
        _ds("security_master", SECURITY_MASTER, gen_security_master),
        _ds("fundamentals", FUNDAMENTALS, gen_fundamentals),
        _ds("feature_metadata", FEATURE_METADATA, gen_feature_metadata),
        _ds("signal_registry", SIGNAL_REGISTRY, gen_signal_registry),
        _ds("position_constraints", POSITION_CONSTRAINTS, gen_position_constraints),
        _ds("group_constraints", GROUP_CONSTRAINTS, gen_group_constraints),
        _ds("cv_splits_calendar", CV_SPLITS_CALENDAR, gen_cv_splits_calendar),
        _ds("profiling_runs", PROFILING_RUNS, gen_profiling_runs),
        _ds("stage_measurements", STAGE_MEASUREMENTS, gen_stage_measurements),
        _ds("stage_baselines", STAGE_BASELINES, gen_stage_baselines),
        _ds("regression_thresholds", REGRESSION_THRESHOLDS, gen_regression_thresholds),
        _ds("scaling_fits", SCALING_FITS, gen_scaling_fits),
        _ds("cpu_profile_frames", CPU_PROFILE_FRAMES, gen_cpu_profile_frames),
    ]
}


def generate(name: str, spec: GenSpec, validate: bool = True) -> pl.DataFrame:
    ds = REGISTRY[name]
    df = ds.generate(spec)
    if validate:
        ds.schema_for(spec).validate(df)
    return df


def generate_all(spec: GenSpec, validate: bool = True) -> dict[str, pl.DataFrame]:
    return {name: generate(name, spec, validate) for name in REGISTRY}


def write_all(out_dir: Path, spec: GenSpec, validate: bool = True) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in generate_all(spec, validate).items():
        path = out_dir / f"{name}.parquet"
        df.write_parquet(path)
        paths[name] = path
    return paths
