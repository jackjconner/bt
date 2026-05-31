---
round: 0
component: COMPONENT
pr: 0
date: "YYYY-MM-DD"
metric: "COMPONENT p50_ms"
verdict: accepted
headline_delta: "-Xx p50_ms"
---

# COMPONENT · round NNN

One-line summary of what changed and the headline improvement.

## What it addressed

<!-- The target: component, metric, the hotspot shown in the before flame graph.
     Paste the round proposal from .oversight/round_state.json here. -->

**Component:** COMPONENT
**Metric optimised:** `COMPONENT p50_ms` (or whichever scalar the round targeted)
**Hotspot:** `module.function` consumed XX% of wall time in the before capture.

## How it decided

<!-- The flame-graph reading and reasoning.
     What paths were examined; what was ruled out and why.
     Paste the relevant excerpt from the before calltree below. -->

Before calltree excerpt (`assets/COMPONENT-before.cpu.calltree.txt`):

```
COMPONENT.function  XX%
  └─ inner_call     YY%
     └─ bottleneck  ZZ%
```

Ruled out:
- Approach A: ...
- Approach B: ...

Chosen approach and rationale: ...

## Pre/post profile

| metric       | before   | after    | delta   |
|:-------------|:---------|:---------|:--------|
| p50_ms       | XXX ms   | YYY ms   | −ZZ%    |
| peak_mb      | AA MB    | BB MB    | ±CC%    |
| scaling exp  | 1.XX     | 1.YY     | −0.ZZ   |

After flame graph (rendered inline by the site generator):

![flame graph](assets/COMPONENT-flamegraph.html)

## System impact

Eval snapshot diff vs the golden (all values must be within declared tolerance,
or better with justification):

| eval metric           | golden  | after   | delta   | within tol? |
|:----------------------|:--------|:--------|:--------|:------------|
| signal IC             | 0.XXX   | 0.XXX   | +0.000  | yes         |
| walk-forward Sharpe   | X.XX    | X.XX    | +0.00   | yes         |
| cost drag             | X bp    | X bp    | 0 bp    | yes         |

`IMPROVEMENTS.md` ledger entry:

```
### round NNN · COMPONENT · METRIC
- target: HOTSPOT
- before: XXX ms  after: YYY ms  delta: −ZZ%
- PR: #NN
- verdict: accepted
```

Downstream effects: none / describe any.

## Suggested next steps

<!-- What the flame graph still shows that this change did not address. -->

1. `module.other_function` is the next hot path (XX% after this change).
2. Scaling exponent is still super-linear at 1.YY — see `assets/COMPONENT-after.speedscope.json.gz`.
3. Memory: `assets/COMPONENT-mem.summary.txt` shows top allocator is still X.
