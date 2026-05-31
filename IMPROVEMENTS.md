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
