# API requests ledger

When a component's sub-agent needs data a sibling component doesn't yet emit, it
posts a request here instead of reaching across the boundary — and works with
what it has this round. The orchestrator routes producers next round. A request
is closed only when the producer adds the field **additively** (new column/field
with a default; no signature break) and the consumer's gate confirms it lands.
Workers post here following the `component-improvement-loop` skill; the
`improvement-orchestrator` skill routes them.

Format (append-only):

```
## <date> — <requester> needs <field/data> from <producer>
why:    <one line>
status: open | accepted | done
```

---
