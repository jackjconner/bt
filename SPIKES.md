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

## 2026-05-31 — explore: portfolio FactorRiskModel.build (K=3)  [winner: wide-layout — salvaged, not judged]
target:    `portfolio.risk_model.build_from_long` / `FactorRiskModel.build` — ~54% of portfolio time + residual ^1.74 memory; challenge runtime AND the super-linear memory while holding Σ semantics.
strategies:
  - lazy-polars: fuse the whole factor-cov build into one `LazyFrame` query → NO RESULT (round aborted before completion)
  - numpy-sparse: assemble Σ = BFBᵀ+D with scipy.sparse, never holding the dense long frame → NO RESULT (round aborted)
  - wide-layout: pivot long→wide once, matrix-native build + lazy dense Σ → in-progress rewrite SALVAGED; later re-gated solo → 18–112× build, peak mem n²→flat, golden held. Merged as PR #36.
verdict:   no clean tournament — the run was aborted mid-flight (see frontier); the wide-layout worker's in-progress rewrite was salvaged and adjudicated as a standalone perf change, not as a judged winner. The other two strategies produced no comparable result.
frontier:  **Process, not algorithm.** The tournament was dispatched via the Workflow tool with worktree isolation; it spawned 3 unsupervised repo copies that each ran the harness into a RAM-backed /tmp (tmpfs), exhausted it, and capsized the run before the judge — cascading into git SIGABRT and a dead shell (DECISIONS.md: heavy temp off tmpfs). Lessons: (1) run explore rounds as **supervised Agent dispatch**, never the Workflow runner — `explore-tournament.js` was removed; (2) `source scripts/diskguard` before any round so heavy temp lands on disk. The lazy-Σ / factored-variance direction is a confirmed win and is now the incumbent; a *clean* re-run could still compare lazy-polars vs numpy-sparse against it, but the dense-Σ removal is the headline and is banked.
