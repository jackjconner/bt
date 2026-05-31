"""Constraint builders for portfolio optimization.

Translates financial constraint specifications into scipy.optimize-compatible
bounds and linear constraint dicts. The optimizer owns objective; this module
owns the feasible set.

Design: keep all structures pure data (no scipy imports at module level) so
tests can instantiate ConstraintSpec without pulling in the full scipy stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ConstraintSpec:
    """Declarative specification of all portfolio constraints.

    Attributes:
        n_assets:    Number of assets in the universe.
        long_only:   If True, lower bound per asset is max(0, lb_i).
        min_weight:  Per-asset minimum weight, shape (n_assets,) or scalar.
                     For long-only portfolios, effective floor is
                     max(0, min_weight_i).
        max_weight:  Per-asset maximum weight, shape (n_assets,) or scalar.
        net_exposure: Target net weight sum (typically 1.0 for long-only,
                      0.0 for dollar-neutral). Applied as equality.
        min_gross:   Minimum sum of |w_i| (lower bound on leverage).
        max_gross:   Maximum sum of |w_i| (upper bound on leverage).
        sector_map:  (n_assets,) integer sector label per asset.
        sector_min:  Dict[sector_label -> min exposure] for sector groups.
        sector_max:  Dict[sector_label -> max exposure] for sector groups.
    """

    n_assets: int
    long_only: bool = True
    min_weight: float | np.ndarray = 0.0
    max_weight: float | np.ndarray = 1.0
    net_exposure: float = 1.0
    min_gross: float | None = None
    max_gross: float | None = None
    sector_map: np.ndarray | None = None  # (n_assets,) int labels
    sector_min: dict[int, float] = field(default_factory=dict)
    sector_max: dict[int, float] = field(default_factory=dict)

    def per_asset_bounds(self) -> list[tuple[float, float]]:
        """Lower/upper weight bounds as scipy Bounds-compatible list.

        Returns list of (lb, ub) tuples, one per asset, respecting long_only.
        """
        n = self.n_assets
        lbs = (
            np.full(n, self.min_weight)
            if np.isscalar(self.min_weight)
            else np.asarray(self.min_weight, dtype=float)
        )
        ubs = (
            np.full(n, self.max_weight)
            if np.isscalar(self.max_weight)
            else np.asarray(self.max_weight, dtype=float)
        )
        if self.long_only:
            lbs = np.maximum(lbs, 0.0)
        return [(float(lbs[i]), float(ubs[i])) for i in range(n)]

    def scipy_constraints(self) -> list[dict]:
        """Return a list of scipy-style constraint dicts.

        Includes:
          - Net exposure equality (Σ w_i = net_exposure).
          - Gross exposure bounds (if set), implemented as two inequalities
            since gross exposure Σ|w_i| is non-linear. For SLSQP with the
            long-only flag this reduces to a linear sum bound.
          - Per-sector exposure bounds (linear inequalities per sector group).

        Note on gross exposure for long-short: when long_only=False, Σ|w_i| is
        a non-smooth constraint. We approximate it via the sum of positive and
        negative parts, which SLSQP handles via sub-gradient. In practice,
        reformulate with auxiliary variables for exact LP/QP; here we use the
        direct form since SLSQP handles smooth approximations adequately for
        typical n_assets.
        """
        cons: list[dict] = []

        # Budget / net exposure equality
        cons.append(
            {
                "type": "eq",
                "fun": lambda w, ne=self.net_exposure: w.sum() - ne,
                "jac": lambda w: np.ones(self.n_assets),
            }
        )

        # Gross exposure inequality bounds
        if self.min_gross is not None:
            mg = float(self.min_gross)
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, mg=mg: np.abs(w).sum() - mg,
                }
            )
        if self.max_gross is not None:
            mg = float(self.max_gross)
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, mg=mg: mg - np.abs(w).sum(),
                }
            )

        # Sector constraints (linear)
        if self.sector_map is not None:
            sectors = np.asarray(self.sector_map)
            unique_sectors = np.unique(sectors)
            for s in unique_sectors:
                mask = (sectors == s).astype(float)
                s_int = int(s)
                if s_int in self.sector_min:
                    lo = float(self.sector_min[s_int])
                    cons.append(
                        {
                            "type": "ineq",
                            "fun": lambda w, m=mask, lo=lo: m @ w - lo,
                            "jac": lambda w, m=mask: m,
                        }
                    )
                if s_int in self.sector_max:
                    hi = float(self.sector_max[s_int])
                    cons.append(
                        {
                            "type": "ineq",
                            "fun": lambda w, m=mask, hi=hi: hi - m @ w,
                            "jac": lambda w, m=mask: -m,
                        }
                    )

        return cons


def from_polars(
    position_constraints: object,  # pl.DataFrame (id, min_weight, max_weight, tradable)
    group_constraints: object,  # pl.DataFrame (sector, min_exposure, max_exposure)
    security_master: object,  # pl.DataFrame (id, sector, ...)
    n_assets: int,
    long_only: bool = True,
    net_exposure: float = 1.0,
) -> ConstraintSpec:
    """Build a ConstraintSpec from the canonical Polars schema frames.

    Non-tradable assets receive zero weight bounds (min=max=0) so the
    optimizer cannot allocate to them.

    Args:
        position_constraints: Per-asset min/max weight and tradable flag.
        group_constraints:    Per-sector min/max exposure bounds.
        security_master:      Maps asset id to sector label (categorical).
        n_assets:             Total universe size (0..n_assets-1).
        long_only:            Floor all per-asset lbs at 0 if True.
        net_exposure:         Target sum of weights.
    """
    import polars as pl

    ids = list(range(n_assets))

    # per-asset bounds from position_constraints
    pc = {row["id"]: row for row in position_constraints.iter_rows(named=True)}
    min_w = np.zeros(n_assets)
    max_w = np.zeros(n_assets)
    for i in ids:
        if i in pc:
            row = pc[i]
            if not row["tradable"]:
                min_w[i] = max_w[i] = 0.0
            else:
                min_w[i] = row["min_weight"]
                max_w[i] = row["max_weight"]
        else:
            # default: unconstrained within [0, 1]
            min_w[i] = 0.0
            max_w[i] = 1.0

    # sector map: id → sector integer label
    sm = security_master.select("id", "sector")
    # cast categorical to string then map to int
    unique_sectors = sorted(sm["sector"].cast(pl.String).unique().to_list())
    sector_str_to_int = {s: i for i, s in enumerate(unique_sectors)}

    sector_map = np.zeros(n_assets, dtype=int)
    for row in sm.iter_rows(named=True):
        aid = row["id"]
        if 0 <= aid < n_assets:
            sector_map[aid] = sector_str_to_int[row["sector"]]

    # sector min/max from group_constraints (keyed by sector string)
    sector_min: dict[int, float] = {}
    sector_max: dict[int, float] = {}
    for row in group_constraints.iter_rows(named=True):
        k = sector_str_to_int.get(str(row["sector"]))
        if k is not None:
            sector_min[k] = row["min_exposure"]
            sector_max[k] = row["max_exposure"]

    return ConstraintSpec(
        n_assets=n_assets,
        long_only=long_only,
        min_weight=min_w,
        max_weight=max_w,
        net_exposure=net_exposure,
        sector_map=sector_map,
        sector_min=sector_min,
        sector_max=sector_max,
    )
