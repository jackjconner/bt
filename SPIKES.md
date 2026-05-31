# Spikes log

Append-only record of every **explore** round's divergent attempts — especially
the ones that *didn't* merge. An explore round fans out K bold rewrites of one
target, judges them, and (hybrid reward) merges a strict win or a structurally
superior tie; the losers are **discarded as code but their learning is kept
here**. This is the loop's exploration memory: what bold directions were tried,
why they lost, and what frontier they revealed — so a later round doesn't re-walk
a dead end, and a promising-but-premature idea can be picked back up.

Recorded by the `improvement-orchestrator` skill after an explore round. The
merged winner (if any) is recorded in `IMPROVEMENTS.md` with `type: explore`; the
discarded attempts live here. Newest entries at the bottom.

Format (one block per explore round, never edited after writing):

```
## <date> — explore: <target> (K=<n> strategies)  [winner: <slug|none>]
target:    <the incumbent being challenged + the metric/quality at stake>
strategies:
  - <slug-a>: <one-line divergent approach> → <gate outcome + key numbers>
  - <slug-b>: <one-line divergent approach> → <gate outcome + key numbers>
  - <slug-c>: <one-line divergent approach> → <gate outcome + key numbers>
verdict:   merged <slug> (win | structural tie) | all spiked (no win, no merge)
frontier:  <what we learned — headroom revealed, dead ends closed, idea parked>
PR:        <url or #number, if a winner merged>
```

---

## EXAMPLE (template — delete when the first real explore round lands)

## 2026-05-31 — explore: signals IC engine — lazy vs eager Polars (K=3)  [winner: none]
target:    `signals.ic_series_v2` eager pipeline; metric = signals harness p50 (currently ~200ms), quality = readability
strategies:
  - lazy-scan: rewrite the whole IC path as one `LazyFrame` query, let Polars optimize → PASS gates, p50 198ms (tie, no win)
  - numpy-core: drop to numpy for the per-date rankdata inner, Polars only for I/O → PASS gates, p50 171ms (−14%, win)
  - arrow-compute: pyarrow.compute kernels for the cross-sectional rank → FAIL correctness (rank-tie handling moved horizon_ic[5])
verdict:   merged numpy-core (strict win) — see IMPROVEMENTS.md type: explore
frontier:  lazy rewrite ties eager here (query is already single-pass) — park it; the win is in the rankdata inner, not the frame engine. arrow kernels break Spearman tie semantics — dead end for IC until they match scipy's average-rank.
PR:        #00 (example)
