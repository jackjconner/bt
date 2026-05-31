from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingResult:
    stage: str
    elapsed_s: float


class StageTimer:
    def __init__(self, stage: str) -> None:
        self._stage = stage
        self._start: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self._elapsed = time.perf_counter() - self._start

    @property
    def result(self) -> TimingResult:
        return TimingResult(stage=self._stage, elapsed_s=self._elapsed)
