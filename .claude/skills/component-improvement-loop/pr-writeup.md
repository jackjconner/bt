# PR writeup template — component improvement loop

A sub-agent fills this in as the PR body. Every section is required; write
"n/a" with a reason rather than deleting a heading. The head agent adjudicates
against these sections, so be concrete — numbers, not adjectives.

---

## Target

- **Component:** <etl | signals | models | analysis | portfolio | backtest | profiling>
- **What I changed:** <one sentence>
- **Declared metric this round optimizes:** <e.g. portfolio `elapsed_s` p50 @ n_assets=200>
- **Eval tolerance for the round:** <e.g. PipelineSummary fields equal to 1e-6 relative>

## Gates (quote the actual output)

### Lint & types
Enforced on every commit by `scripts/committer`; confirm clean on the branch tip.
```
$ uv run ruff check && uv run ruff format --check && uv run ty check
All checks passed!
```
No `# type: ignore` / `# noqa` / suppression markers were added (state this).

### Unit tests
```
$ uv run pytest -q <component>/
<paste tail: pass count, 0 failures>
```

### Integration tests
```
$ uv run pytest -q tests/integration/
<paste tail — these import across components; a contract break shows here>
```

### Profiling — before → after
Measured on the **same GenSpec grid + seed + machine** (state them).
```
$ uv run main.py        # or harness/run_harness over <grid>
            before      after     delta
<stage>     <p50>       <p50>     <-X%>
scaling:    <dim^slope before> → <after>
```
`check_regressions` vs the ratcheted baseline: <no other stage regressed / list>.

### Correctness / accuracy (if applicable)
If this is an accuracy change (not a pure refactor), the eval numbers *should*
move. Show the before→after PipelineSummary fields that changed and argue the
new value is correct (reference the analytic / a test that pins it). If it's a
pure refactor, state: eval golden unchanged within tolerance.

## Why it should merge

<The case in 2–4 sentences: what was wrong/slow, what's better now, and the
evidence above that proves it.>

## How it fits the existing architecture

<Which layer/contract it lives under; that the public `__all__` and
`_protocol.py` are unchanged or only extended (name new symbols/fields); that
the dependency arrows ([[architecture]]) are unchanged. If it added a dataclass
field or param, show it has a default.>

## Pros & cons of this approach

- **Pros:** <…>
- **Cons / what it trades off:** <…>
- **Alternatives considered & why rejected:** <…>

## Reproducibility

- Exact command(s) to reproduce the profiling numbers.
- Environment: git sha, host, BLAS thread count (from `capture_environment`).
- GenSpec grid + seed used for before and after (must be identical).

## Risk & rollback

- **Blast radius:** which downstream components could be affected, and why the
  integration gate covers them.
- **Rollback:** `git revert <merge sha>` — confirm the change is self-contained
  enough that a revert is clean.

## Ledger

- API requests posted this round (if any): <link to `API_REQUESTS.md` entry>.
