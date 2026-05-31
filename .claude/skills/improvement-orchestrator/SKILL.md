---
name: improvement-orchestrator
description: Use when driving one round of bt's component self-improvement loop — pick a target, dispatch worker sub-agents into worktrees, adjudicate the correctness/profiling/evaluation gates, serial-merge the reviewed PRs, then ratchet and record. You are the conductor; the workers run [[component-improvement-loop]].
---

# Improvement orchestrator

bt is seven components composing across stable contracts (see [[architecture]],
[[interfaces]]). This skill is the conductor's runbook for one round of making a
component *better* without breaking the contract everything else codes against.
You own the round — propose, gate, review, merge, ratchet, record, dispatch
docs — but **never edit component code yourself**; cold worker sub-agents do that
in isolated worktrees and land each change as a reviewed PR. The two ledgers
carry memory across rounds. One round per invocation; re-invoke for the next.

## When to apply

When the goal is "improve component X" (the harness flagged a hotspot, an
analytic is wrong, a path is slow) and you are the driver coordinating workers,
not the worker doing the change. If the contract *must* break, that's a
[[design-before-architecture-changes]] talk with Jack, not this loop.

## The round

1. **Propose the target.** Run `uv run main.py` to read the latest `harness/`
   table + `scaling_fits`; open `API_REQUESTS.md`. Propose the single thing to
   address and name the **metric it optimizes** + the **eval tolerance** up
   front. **Dedup against `IMPROVEMENTS.md`:** if the same target was accepted
   recently the baseline already moved; if rejected, don't re-attempt without a
   genuinely new idea — pick something else.
2. **Capture the golden.** Run all the gates *now*. Save the `PipelineSummary`
   from `pipeline.py::run_production_pipeline` as the round's **golden**, and
   record the target metric's current value. Nothing is an improvement without a
   before.
3. **Dispatch the workers.** One per chosen component, each via the **Agent
   tool** in its own `git worktree` on branch `improve/<component>-<slug>` off
   `main`. The brief is one sentence of goal + the exact files + what you've
   ruled out + the hard constraints, and it **points the worker at the
   `component-improvement-loop` skill**. Parallel-vs-serial rule: components
   sharing no `API_REQUESTS` edge this round are independent → dispatch in
   parallel; a consumer waiting on a producer's new field serializes (producer
   this round, consumer next).
4. **Adjudicate.** Review each PR's writeup against the `pr-writeup.md` sections;
   confirm the worker actually ran and quoted *every* gate. Don't take "passing"
   on faith — the numbers are in the writeup or the gate didn't run.
5. **Serial-merge + re-validate.** Merge **one** PR (`gh pr merge`). Re-run all
   the gates on the post-merge `main`: `uv run pytest -q`, `uv run main.py` +
   `check_regressions` vs the ratcheted baseline, and diff the new
   `PipelineSummary` against the golden within tolerance. Only if green, merge
   the next. On any post-merge regression, `git revert` the merge commit.
6. **Ratchet + record.** After a merge stands: regenerate `stage_baselines` from
   the new harness run (next round's `check_regressions` bar); append the round
   to `IMPROVEMENTS.md` (target, before→after, PR link, verdict); reconcile
   `API_REQUESTS.md`.
7. **Dispatch the docs agent.** A dedicated agent — *not* a worker — reconciles
   in-repo docs (`README.md`, `WORKING_NOTES.md`, `PRODUCTION_PLAN.md`, touched
   docstrings, a `DECISIONS.md` entry for an ADR-worthy call) and commits to
   `main`. It **flags** stale `~/syl/wiki/projects/bt-backtester` pages via a
   "Wiki re-verify" note in `WORKING_NOTES.md` — it never overwrites Syl's pages.

## The adjudication bar

A PR merges only if all gates pass. Lint and types are *continuous*
(`scripts/committer` runs them on every worker commit), so the live gates you
adjudicate are correctness, profiling, and evaluation:

| gate | command | the bar |
|---|---|---|
| **correctness** | `uv run pytest -q` (unit + `tests/integration/`) | still green — same count, no new failures. A broken contract fails in the integration suite. |
| **profiling** | `uv run main.py` → `check_regressions` vs the ratcheted baseline | the declared target metric *improved*, no other stage regressed past threshold. |
| **evaluation** | `run_production_pipeline` → `PipelineSummary` diffed vs the **golden** | eval numbers (signal IC, walk-forward IC/R², Sharpe, cost drag) equal within the declared tolerance, or better. |

Evaluation is the subtle one: a speedup that moves the Sharpe is a correctness
regression wearing a profiling win's clothes — the golden diff catches it. A
genuine accuracy improvement is the one case the numbers *should* move; the
writeup must say so and show why the new value is correct.

## Why

Serial-merge with re-validation exists because green-on-branch ≠
green-after-a-sibling-merged — two passing PRs can interact. The golden diff
exists because unit tests pass while numbers silently move; the ledgers, so each
round starts from what's known rather than from scratch.

## Anti-patterns

- Merging a second PR without re-running the gates on the post-merge tree.
- Editing component code yourself instead of dispatching a worker.
- Skipping the `IMPROVEMENTS.md` dedup, or re-attempting a logged rejection.
- Banking a profiling win that moved the eval golden without a correctness
  justification — a silent regression.
- Forcing a win when no PR passed all the gates.
- Accepting a writeup that doesn't quote the gates
  ([[verification-before-completion]]).

## Exit condition

The round's target improved on the profiling gate, correctness and evaluation
are green on the *post-merge* tree, the baseline is ratcheted, and the round is
recorded in `IMPROVEMENTS.md`. If no PR passed all the gates, record the
rejection and stop — don't force a win.
