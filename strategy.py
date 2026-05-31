"""Strategy specification — a single, frozen, serializable object that owns
every tunable knob in the production pipeline.

``StrategySpec`` contains a ``GenSpec`` (data-generation parameters) plus the
pipeline-level knobs that were previously hardcoded in
``run_production_pipeline``.  Freezing and validating at construction makes a
strategy:

* **Reproducible** — pass the same spec, get the same numbers.
* **Diffable** — ``dataclasses.asdict`` exposes every knob; compare two specs
  with a plain dict diff.
* **Serializable** — ``to_json`` / ``from_json`` roundtrip via stdlib ``json``.

The defaults here are the *exact* literals that existed in
``run_production_pipeline`` before this spec was introduced, so the pipeline
behaves identically when no spec is supplied.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from etl.datasets import GenSpec


@dataclass(frozen=True)
class StrategySpec:
    """All tunable knobs for the production pipeline.

    Parameters
    ----------
    gen:
        Data-generation parameters (asset/date/feature counts, seed, etc.).
    wf_n_splits:
        Number of walk-forward CV folds.
    wf_embargo_periods:
        Embargo periods between train and test in walk-forward CV.
    opt_risk_aversion:
        Mean-variance optimizer risk-aversion coefficient λ.
    opt_max_iter:
        Maximum SLSQP iterations for the optimizer.
    bt_max_weight:
        Per-asset weight cap applied in both gross and net backtest runs.
    bt_enable_universe_mask:
        Whether to enforce tradability constraints in the backtest.
    bt_enable_costs:
        Whether to charge transaction costs in the *net* backtest run.
    bt_enable_slippage:
        Whether to apply square-root market-impact in the *net* backtest run.
    profile_trials:
        Number of timed trials used by ``capture_environment``.
    profile_warmup:
        Number of warmup trials used by ``capture_environment``.
    bt_profile_trials:
        Number of timed trials used by the backtest ``run_trials`` call.
    bt_profile_warmup:
        Number of warmup trials used by the backtest ``run_trials`` call.
    neutralize_sectors:
        Whether to compute sector-neutralized IC (in addition to raw IC).
    horizon_map:
        Mapping of integer horizons → forward-return column names used to
        build the IC horizon decay curve.
    """

    gen: GenSpec

    # walk-forward CV
    wf_n_splits: int = 4
    wf_embargo_periods: int = 5

    # optimizer
    opt_risk_aversion: float = 1.0
    opt_max_iter: int = 3000

    # backtest
    bt_max_weight: float = 0.1
    bt_enable_universe_mask: bool = True
    bt_enable_costs: bool = True
    bt_enable_slippage: bool = True

    # profiling
    profile_trials: int = 3
    profile_warmup: int = 1
    bt_profile_trials: int = 3
    bt_profile_warmup: int = 1

    # signal choices
    neutralize_sectors: bool = True
    horizon_map: dict[int, str] = field(
        default_factory=lambda: {1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21", 63: "fwd_ret_63"}
    )

    def __post_init__(self) -> None:
        # --- positive-rate guards ------------------------------------------ #
        if self.wf_n_splits < 2:
            raise ValueError(f"wf_n_splits must be >= 2, got {self.wf_n_splits}")
        if self.wf_embargo_periods < 0:
            raise ValueError(f"wf_embargo_periods must be >= 0, got {self.wf_embargo_periods}")
        if self.opt_risk_aversion <= 0.0:
            raise ValueError(f"opt_risk_aversion must be > 0, got {self.opt_risk_aversion}")
        if self.opt_max_iter < 1:
            raise ValueError(f"opt_max_iter must be >= 1, got {self.opt_max_iter}")
        if not (0.0 < self.bt_max_weight <= 1.0):
            raise ValueError(f"bt_max_weight must be in (0, 1], got {self.bt_max_weight}")
        if self.profile_trials < 1:
            raise ValueError(f"profile_trials must be >= 1, got {self.profile_trials}")
        if self.profile_warmup < 0:
            raise ValueError(f"profile_warmup must be >= 0, got {self.profile_warmup}")
        if self.bt_profile_trials < 1:
            raise ValueError(f"bt_profile_trials must be >= 1, got {self.bt_profile_trials}")
        if self.bt_profile_warmup < 0:
            raise ValueError(f"bt_profile_warmup must be >= 0, got {self.bt_profile_warmup}")
        if not self.horizon_map:
            raise ValueError("horizon_map must have at least one entry")

        # --- cross-parameter consistency ----------------------------------- #
        # The WalkForwardSplitter needs at least (min_train_periods=1) + n_splits
        # date-groups.  With embargo, each fold consumes embargo_periods extra
        # dates, so a rough lower bound is n_splits * (1 + embargo_periods) + 1.
        min_dates_needed = self.wf_n_splits * (1 + self.wf_embargo_periods) + 1
        if self.gen.n_dates < min_dates_needed:
            raise ValueError(
                f"n_dates={self.gen.n_dates} is too small for wf_n_splits="
                f"{self.wf_n_splits} with wf_embargo_periods="
                f"{self.wf_embargo_periods}; need at least {min_dates_needed} dates"
            )

        # bt_max_weight must be >= 1/n_assets to allow a feasible uniform alloc
        min_feasible_weight = 1.0 / self.gen.n_assets
        if self.bt_max_weight < min_feasible_weight:
            raise ValueError(
                f"bt_max_weight={self.bt_max_weight} < 1/n_assets="
                f"{min_feasible_weight:.6f}; no feasible equal-weight portfolio exists"
            )

    # ---------------------------------------------------------------------- #
    # JSON serialisation
    # ---------------------------------------------------------------------- #

    def to_json(self) -> str:
        """Serialise to a JSON string.  The ``gen`` sub-object is inlined."""
        d = asdict(self)
        # horizon_map keys are ints; JSON only allows string keys, so we
        # encode them as strings and decode symmetrically in from_json.
        d["horizon_map"] = {str(k): v for k, v in d["horizon_map"].items()}
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> StrategySpec:
        """Deserialise from a JSON string produced by ``to_json``."""
        d = json.loads(s)
        gen_d = d.pop("gen")
        gen = GenSpec(**gen_d)
        # Restore horizon_map int keys
        raw_hm = d.pop("horizon_map", {})
        horizon_map: dict[int, str] = {int(k): v for k, v in raw_hm.items()}
        return cls(gen=gen, horizon_map=horizon_map, **d)


__all__ = ["StrategySpec"]
