# Improvements log

Append-only record of every component-improvement round — accepted *and*
rejected. This is the loop's memory: the dedup source for round planning (don't
re-attempt a recently-rejected target) and the audit trail of cumulative gains.
Recorded by the `improvement-orchestrator` skill. Newest entries at the bottom.

Format (one block per round, never edited after writing):

```
## <date> — <component>: <one-line target>  [accepted | rejected]
metric:   <name> — <before> → <after> (<delta>)
eval:     golden unchanged within <tol>  |  accuracy moved: <field before→after, why correct>
PR:       <url or #number>
note:     <why accepted, or which gate rejected it and the takeaway>
```

---

## 2026-05-31 — portfolio: replace SLSQP with OSQP QP solver  [accepted]
metric:   portfolio harness p50 — 457/1989/4178 ms → 8/20/66 ms (~50–100×)
eval:     accuracy moved: factor_vol 0.7678 → 0.7714 (OSQP reaches the true constrained optimum; objective −0.1054 → −0.1011 strictly better, both converged, net-exposure exactly 1.0; all other PipelineSummary fields held)
PR:       #3
note:     accepted — Jack-approved new dep (osqp) + the justified eval shift. Super-linear scaling remains (now ledoit_wolf_cov-bound, not solver-bound).
