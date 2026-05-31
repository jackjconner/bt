---
name: component-improvement-loop
description: Use when iteratively improving one or more of bt's seven components (etl, signals, models, analysis, portfolio, backtest, profiling) behind their public APIs. A head agent proposes a target each round, fans out per-component sub-agents in isolated worktrees, and gates every change — landed as a reviewed PR — on correctness, profiling, and evaluation.
---

# Component improvement loop

bt is seven components composing across stable contracts (see [[architecture]],
[[interfaces]]). This skill is the protocol for making one component *better* —
faster, more accurate, cleaner — without breaking the contract everything else
codes against. A head agent owns the loop; cold sub-agents do the per-component
work in isolated worktrees and land each change as a reviewed PR; the gates
decide what survives. The loop keeps memory across rounds so it doesn't re-walk
ground it already covered.

## When to apply

When the goal is "improve component X" (the profiling harness flags a hotspot,
an analytic is wrong, a path is slow) rather than "add a new component" or
"change the public surface on purpose." If the contract *must* break, that's a
[[design-before-architecture-changes]] conversation with Jack, not this loop.

## The loop (head agent runs this each round)

1. **Propose the target.** Read the latest `harness/` table + `scaling_fits`
   (`uv run main.py`) and open `API_REQUESTS.md`. Propose the single thing to
   address and name the **metric it optimizes** + the **eval tolerance** up
   front. **Dedup:** check `IMPROVEMENTS.md` — if the same target was attempted
   in the last few rounds (accepted → the baseline already moved; rejected →
   don't re-attempt without a genuinely new idea), pick something else.
2. **Capture the round baseline.** Run all the gates *now*; save the
   `PipelineSummary` as the round's **golden**, and record the target metric's
   current value. Nothing is an improvement without a before.
3. **Dispatch one sub-agent per chosen component**, each in its own git
   worktree branched from `main` (brief format below). Components sharing no
   request edge this round are independent → parallel. A consumer waiting on a
   producer's new field serializes: producer this round, consumer next.
4. **Sub-agent works + opens a PR.** It improves the implementation behind the
   API (additive-only, in its lane), committing atomically with
   `scripts/committer` as it goes (each commit gated on ruff + ty), re-runs and
   **quotes all the gates** in its worktree, then opens a PR via `gh` using
   `pr-writeup.md`.
5. **Adjudicate + serial-merge.** Review each PR's writeup; confirm the gates
   were run. Merge **one** PR, re-run all the gates on the post-merge `main`,
   and diff the eval golden within tolerance. Only if green, merge the next.
   On any post-merge regression, revert the merge commit.
6. **Ratchet + record.** After a merge stands: regenerate `stage_baselines`
   from the new harness run (next round's `check_regressions` bar). Append the
   round to `IMPROVEMENTS.md` (target, before→after, PR link, accepted/rejected).
   Reconcile `API_REQUESTS.md`.
7. **Update the docs.** Dispatch a dedicated docs agent (commits to `main`) to
   reconcile in-repo docs with the merged change and flag any stale wiki page —
   see *Docs* below.

The head agent never edits component code — it proposes, gates, reviews, merges,
ratchets, records, and dispatches the docs update.

## The gates (a change survives only if all pass)

| gate | command | the bar |
|---|---|---|
| **lint** | `uv run ruff check` + `ruff format --check` | clean; warnings are errors. Enforced on *every* commit by `scripts/committer`. |
| **types** | `uv run ty check` | clean; `error-on-warning`. No `# type: ignore` / suppression — fix or widen the annotation. Also enforced per commit by `scripts/committer`. |
| **correctness / regression** | `uv run pytest -q` (unit + `tests/integration/`) | still green — same count, no new failures. The integration suite imports across components, so a broken contract fails *here*. |
| **profiling** | `harness/run_harness` via `uv run main.py`; `check_regressions` vs the ratcheted baseline | the declared target metric *improved*, and no other stage regressed past its threshold. |
| **evaluation** | `pipeline.py::run_production_pipeline` → `PipelineSummary` diffed vs the round's **golden** | eval numbers (signal IC, walk-forward IC/R², Sharpe, cost drag) **equal within the declared tolerance**, or better. |

Lint and types are *continuous* — `scripts/committer` runs them on every atomic
commit, so a PR is lint/type-clean by construction. Correctness, profiling, and
evaluation are the per-change gates the head agent adjudicates.

The evaluation gate is the subtle one: a refactor can keep the API and pass
every unit test yet *silently move the numbers*. A speedup that changes the
Sharpe is a correctness regression wearing a profiling win's clothes — the
golden diff catches it mechanically. A genuine accuracy improvement is the one
case the numbers *should* move; the PR must say so and show why it's correct.

## API discipline — additive only

The contract is each component's `__all__` plus its `_protocol.py` Protocol.
Sub-agents may **extend** it, never shrink it (this is why every POC path still
runs — see DECISIONS.md, additive-API discipline):

- Never remove or rename an exported symbol. Never change an existing signature.
- New parameters take defaults; new dataclass fields take defaults. "Version up"
  means *add* a field/function — the old call site keeps working byte-for-byte.
- A contract break is caught at the correctness gate: integration tests import
  across the boundary.

## Worktrees, PRs, and scope

- **One worktree per sub-agent**, branch `improve/<component>-<slug>` off `main`.
  Isolated trees let parallel sub-agents build and run gates without colliding.
- **Scope guard.** A PR touches only its component's directory + the shared
  ledgers (`API_REQUESTS.md`). The head agent rejects any cross-lane diff.
- **Serial-merge with re-validation** (loop step 5): green-on-branch ≠
  green-after-a-sibling-merged. Rollback is a `git revert` of the merge commit.
- **PR writeup** uses `pr-writeup.md`. It must carry: unit tests; integration
  tests; profiling before→after (same GenSpec grid + seed + env); correctness/
  accuracy delta if applicable; why it should merge; how it fits the existing
  architecture; pros & cons of the approach; the declared target + result;
  reproducibility (exact command, env, grid); risk & rollback.

## The two ledgers

- **`API_REQUESTS.md`** — a sub-agent that needs data a sibling doesn't yet emit
  posts a request instead of crossing the boundary, and works with what it has
  this round. Producers pick requests up next round; closed only when the field
  lands additively and the consumer's gate confirms it.
- **`IMPROVEMENTS.md`** — append-only round log: the loop's memory. Every round
  (accepted *and* rejected) with target, before→after, PR link. It is the dedup
  source for step 1 and the audit trail of cumulative gains.

## Docs (post-merge docs agent)

After a merge stands, a **dedicated agent — not the component sub-agent** —
reconciles documentation and commits to `main`:

- **In-repo (edit):** `README.md`, `WORKING_NOTES.md`, `PRODUCTION_PLAN.md`
  status, the touched component's docstrings / `__all__` notes, and a
  `DECISIONS.md` entry when the round made an ADR-worthy call (e.g. a changed
  architectural characteristic like "portfolio no longer scales super-linearly").
- **Wiki (flag, don't edit):** the `~/syl/wiki/projects/bt-backtester` pages
  (architecture, components, interfaces, harnesses, data-schemas) are Syl's
  compiled synthesis with their own verified dates. The docs agent does **not**
  overwrite them — it appends a "Wiki re-verify" note to `WORKING_NOTES.md`
  naming the page and what changed, so Syl re-verifies on her side.

## Briefing a sub-agent (it starts cold)

Per [[parallelism]]: one sentence of goal, the exact files
(`portfolio/optimizer.py`, its `_protocol.py`, its tests), what you've ruled
out, and the hard constraints — **work in your worktree, stay in your
component's lane, additive-only API, commit atomically with `scripts/committer`
(never `git commit` directly, never bypass its ruff/ty gates), re-run and quote
all the gates, open a PR
with the writeup, post to `API_REQUESTS.md` instead of crossing a boundary.**
Improving the *implementation* behind the API is in scope; touching the public
surface is not without flagging back. Prefer the smallest change that wins the
gate over a sweeping rewrite — small diffs review and revert cleanly.

## Anti-patterns

- Changing a signature or dropping a symbol "to clean up" — breaks downstream;
  rejected. Additive only.
- Merging a second PR without re-running the gates on the post-merge tree.
- Banking a profiling win that moved the eval golden without a correctness
  justification — silent regression.
- Re-attempting a target the log shows was rejected a round or two ago.
- A PR that edits outside its component's lane.
- Overwriting Syl's `~/syl` wiki pages or their verified dates — the docs agent
  flags stale wiki pages, it doesn't edit them.
- Declaring done without quoting all the gates
  ([[verification-before-completion]]).

## Exit condition

The round's target improved on the profiling gate, correctness and evaluation
are green on the *post-merge* tree, the baseline is ratcheted, and the round is
recorded in `IMPROVEMENTS.md`. If no PR passed all the gates, record the
rejection and stop — don't force a win.
