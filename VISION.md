# VISION — the north star

What "an **institutional production system built autonomously through a verified
loop**" means, concretely enough to *score against*. The improvement loop
(`improvement-orchestrator`) reads this when proposing `feature` rounds: every
candidate capability is ranked by the rubric below before it lands in
`FEATURE_BACKLOG.md`. This is a curated document — Jack edits the bar; the loop
ideates *toward* it, never rewrites it.

## The end state

A backtester an institutional quant desk would trust to make capital decisions —
not because a human hand-checked it, but because every line arrived through a
gated loop that proves correctness, holds the golden, and ratchets performance.
The substrate is synthetic-but-structured data, so the loop can build and defend
production features (leakage controls, IC recovery, risk attribution, realistic
fills) against data with known, modest signal.

## The five qualities we optimize

1. **Correctness** — no look-ahead, no leakage, no silent number drift. The
   golden `PipelineSummary` is sacred; a change moves it only with a written,
   defended reason. This is the non-negotiable; the other four never buy a
   correctness regression.
2. **Feature breadth** — the gap between toy and institutional is *capabilities*:
   order types, financing, attribution, rolling/periodic analytics, tear sheets,
   regime-conditional research. Breadth is what `feature` rounds add.
3. **Performance** — production-scale (thousands of names, decades of sessions)
   in sub-linear or near-linear cost. `exploit` and `explore` rounds defend this;
   the harness scaling curves are the scoreboard.
4. **Robustness** — graceful at boundaries (missing data, halts, delistings,
   singular covariance), validated at construction, deterministic under a seed.
   Boundaries are asserted, internal code is trusted.
5. **Observability** — every run measured, persisted, regressions detected,
   flame graphs on demand. You cannot improve what you cannot see; the profiling
   component is how the loop sees itself.

## Scoring rubric (feature ideation)

Each candidate capability is scored 1–5 on four axes; the weighted sum sets its
backlog priority. Ideation agents fill these in; Jack reprioritizes.

| axis | weight | 5 = | 1 = |
|---|---|---|---|
| **vision-fit** | ×3 | closes a named toy→institutional gap above | cosmetic / already covered |
| **leverage** | ×2 | unblocks several downstream capabilities | isolated, terminal |
| **gate-safety** | ×2 | cleanly additive (new fields, flag-gated, golden holds) | forces a contract break or golden move |
| **cost** | ×1 | small, one component, one round | sprawling, multi-component, multi-round |

`priority = 3·fit + 2·leverage + 2·safety + 1·cost` (higher = sooner). A
candidate that *requires* a contract break is not a `feature` round — it is a
[[design-before-architecture-changes]] talk with Jack first.

## What this is not

- Not a roadmap with dates — `FEATURE_BACKLOG.md` is the live queue.
- Not a license to break APIs — additive-only discipline still governs every
  round (see `component-improvement-loop` API discipline).
- Not a substitute for Jack's judgment on direction — the loop proposes; Jack
  prioritizes.
