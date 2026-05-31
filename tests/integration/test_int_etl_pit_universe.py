"""Contract: etl point-in-time + universe correctness.

These are the leakage-prevention guarantees downstream research relies on:
an as-of slice must never surface data known only in the future, and the
resolved universe mask must align to the dense matrix produced by `to_matrix`.
"""

from __future__ import annotations

import numpy as np

from etl import as_of_slice, resolve_universe, to_matrix


def test_pit_as_of_excludes_future_knowledge(synth) -> None:
    fundamentals = synth.loader.load("fundamentals")
    known = sorted(fundamentals["knowledge_date"].unique().to_list())
    as_of = known[len(known) // 2]

    sliced = as_of_slice(fundamentals, as_of, knowledge_col="knowledge_date")
    assert sliced.height > 0
    assert sliced["knowledge_date"].max() <= as_of
    # nothing known only in the future leaked in
    assert sliced.filter(sliced["knowledge_date"] > as_of).height == 0


def test_resolve_universe_aligns_to_matrix(synth) -> None:
    umask = synth.loader.load("universe_mask")
    returns_like = synth.loader.load("prices").select("date", "id", "close")
    mat, dates = to_matrix(returns_like, "close")
    ids = sorted(returns_like["id"].unique().to_list())

    universe = resolve_universe(umask, dates, ids)
    assert universe.shape == mat.shape
    assert universe.dtype == np.bool_
    # at least some assets are investable on the first date
    assert universe[0].sum() > 0
