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

When you are driving a round of making the system better and coordinating
workers (not doing the change yourself). "Better" is one of five **round types**
— a perf win (`exploit`), a structural cleanup (`refactor`), a new capability
(`feature`), a bold rewrite (`explore`), or a cleanup that removes a superseded
path (`consolidate`) — see Round types below. Internal API removal within a
component is fair game (no backwards-compat — see [[DECISIONS]]); only a change
to a *cross-component data contract* (the shapes components pass each other) is a
[[design-before-architecture-changes]] talk with Jack.

## Round types

A round is one of four **types**; you pick it (step 1) and the gates apply per
type (the adjudication bar below is type-aware). `exploit` is the original loop,
unchanged.

| type | target source | worker mandate | profiling gate | evaluation gate | extra |
|---|---|---|---|---|---|
| **exploit** | harness hotspot (`scaling_fits`) | smallest diff that wins | metric **improved** | golden holds, or justified accuracy move | — |
| **refactor** | a large/tangled fn or thin-test area | split it up, add unit tests, **no behavior change** | **no regression** (need not improve) | golden **byte-identical** (`--tolerance 0`) | complexity/length down, tests added |
| **feature** | top of `FEATURE_BACKLOG.md` | **additive** capability behind a flag + tests | no regression on existing stages | existing fields **hold**; new fields ok (`--allow-new-fields`) | capability works under test |
| **explore** | a bold-rewrite target | **be** the sweeping rewrite — divergence is the point | improved, **or** a justified tie | golden holds, or justified | K-way tournament → judge → hybrid reward |
| **consolidate** | a component carrying a superseded shadow path from a prior round | **delete** the old impl + its flag, make the replacement sole, update call sites | **no regression** | golden **byte-identical** (`--tolerance 0`) | dead code removed; internal API may shrink |

**Explore cadence.** Count rounds since the last `type: explore` in
`IMPROVEMENTS.md`; if ≥ 4 non-explore rounds have passed, the next round is
`explore`. Otherwise pick exploit / refactor / feature from the live signal +
backlog.

**Hybrid explore reward.** A strict gate-win merges. A *tie* that is structurally
clearly better / opens flagged headroom **may** merge (your call, justified in the
writeup). Pure losers are **discarded** — log their learning to `SPIKES.md` and
record the round `spiked`.

**Two-phase add-then-consolidate.** Explore/feature rounds add a new path
additively, keeping the incumbent as a correctness oracle while it is reviewed.
Once it is merged and the golden held, schedule a **`consolidate`** round on that
component to delete the now-dead shadow + its flag and make the replacement sole
(golden byte-identical, `scripts/gate eval --tolerance 0`). Removal is allowed in
any round (no backwards-compat — see [[DECISIONS]]); the two-phase split just
keeps the oracle around through review. A fallback/oracle still genuinely *used*
(a non-ridge code path, a test oracle) stays until truly unused. Target source:
scan recent `IMPROVEMENTS.md` entries for an additive replacement whose old path
is now dead.

**The explore tournament.** Run it as plain parallel `Agent` dispatch that you
supervise — no workflow runner, so worktree isolation and cleanup stay under your
control. Given a target and K (3–4) deliberately *divergent* strategy briefs
(e.g. lazy-Polars rewrite / numpy-core inner / a different algorithm):

1. Fan out K workers, one per strategy, each in its own worktree
   `git worktree add .worktrees/explore-<slug>-<k> -b explore/<slug>-<k> main`,
   dispatched in parallel via the **Agent tool** (they run `component-improvement-loop`).
2. When they return, dispatch a **judge** sub-agent that scores each PR on
   gate-outcome / magnitude / structural quality / headroom opened / risk, and
   returns a ranked list.
3. Apply the hybrid reward to the ranking: merge the winner (or a strong tie),
   spike the rest, recording discards in `SPIKES.md`.

## The round

**Preflight — `source scripts/diskguard` before any run, and have every worker do
the same.** The harness / pipeline / evalgate write GB-scale parquet to `$TMPDIR`,
which defaults to a RAM-backed `/tmp`; fill it (a few concurrent runs, or leaked
temp from a killed run) and writes fail with `EDQUOT`, git aborts, and the shell
dies. `diskguard` redirects heavy temp to disk and sweeps stale temp — running it
is the difference between a clean round and a capsized one. See [[workspace-isolation]].

1. **Propose the round — pick a type and a target.** First decide the **type**
   (see Round types): apply the explore cadence (≥ 4 non-explore rounds since the
   last `explore` → this is an `explore` round); else choose exploit / refactor /
   feature from the live signal. Then:
   - `exploit` — run `uv run main.py`, read the `harness/` table + `scaling_fits`,
     open `API_REQUESTS.md`; name the single target + the **metric it optimizes**
     + the **eval tolerance**.
   - `refactor` — pick a flagged large/tangled function or a thin-test area; the
     "metric" is structural (length/complexity down, tests added) and the eval
     gate is golden **byte-identical** (`evalgate --tolerance 0`).
   - `feature` — run a fan-out **ideation** sub-step: agents score candidate
     capabilities against `VISION.md` + `PRODUCTION_PLAN.md` tier-2/3 and append
     `queued` rows to `FEATURE_BACKLOG.md`; build the **top Jack-prioritized**
     item. The "metric" is the new capability working under test.
   - `explore` — pick the incumbent to challenge and draft K divergent strategy
     briefs (see The explore tournament).
   **Dedup against `IMPROVEMENTS.md`** (and `FEATURE_BACKLOG.md` for features): if
   the target was accepted recently the baseline already moved; if rejected, don't
   re-attempt without a genuinely new idea — pick something else.
2. **Capture the golden.** Run all the gates *now*. Save the `PipelineSummary`
   from `pipeline.py::run_production_pipeline` as the round's **golden**, and
   record the target metric's current value. Nothing is an improvement without a
   before.
3. **Dispatch the workers.** One per chosen component, each via the **Agent
   tool** in its own worktree:
   `git worktree add .worktrees/<component>-<slug> -b improve/<component>-<slug> main`.
   The brief is one sentence of goal + the exact files + what you've
   ruled out + the hard constraints + the component's **context pack**
   (`docs/worker-context/<component>.md` — inject or point to it so the worker
   skips hand-mapping), and it **points the worker at the
   `component-improvement-loop` skill** (gates run via the `scripts/gate`
   runner, which pins to the worktree + redirects temp off tmpfs).
   Parallel-vs-serial rule: components
   sharing no `API_REQUESTS` edge this round are independent → dispatch in
   parallel; a consumer waiting on a producer's new field serializes (producer
   this round, consumer next). For an **`explore`** round, dispatch the **K-way
   tournament** instead (see The explore tournament): K workers on the *same*
   target with divergent briefs, then a judge — not one worker per component.
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
   to `IMPROVEMENTS.md` with its **`type:`** (target, before→after, PR link,
   verdict). For a `feature` round that added `PipelineSummary` fields, re-save the
   golden (`python -m evalgate --save`) so the new fields become the baseline, and
   mark the built item `done` in `FEATURE_BACKLOG.md`. For an `explore` round, log
   the discarded attempts to `SPIKES.md` (verdict `spiked` if nothing merged).
   Reconcile `API_REQUESTS.md`.
7. **Dispatch the docs agent.** A dedicated agent — *not* a worker — reconciles
   in-repo docs (`README.md`, `WORKING_NOTES.md`, `PRODUCTION_PLAN.md`, touched
   docstrings, a `DECISIONS.md` entry for an ADR-worthy call) and commits to
   `main`. It **flags** stale `~/syl/wiki/projects/bt-backtester` pages in
   `WORKING_NOTES.md` — it never overwrites Syl's pages.

## Emit live state (for the oversight deck)

As the round moves, write `.oversight/round_state.json` so `scripts/oversight`
shows it live (telegraphic — one command per transition):

- round start (step 1–2): `python -m oversight.state set-target --round N --component C --target … --metric … --baseline … --tolerance …`
- dispatched (step 3): `set-phase --phase dispatched --dispatched K`; per worker `set-lane --component C --branch … --pr …`
- as gates resolve (step 4–5): `mark-gate --component C --gate {lint,types,correctness,profiling,evaluation} --verdict {running,pass,fail}`; `set-phase --phase {adjudicating,merging,revalidating,done}`

## The adjudication bar

A PR merges only if all gates pass. Lint and types are *continuous*
(`scripts/committer` runs them on every worker commit), so the live gates you
adjudicate are correctness, profiling, and evaluation:

| gate | command | the bar (by round type) |
|---|---|---|
| **correctness** | `scripts/gate test` (unit + `tests/integration/`) | **all types:** still green — same count, no new failures (a broken contract fails in the integration suite). refactor/feature additionally *add* tests. |
| **profiling** | `scripts/gate bench` → `check_regressions` vs the ratcheted baseline | **exploit:** target metric *improved*. **refactor / feature / consolidate:** *no regression* past threshold (need not improve). **explore:** *improved, or a justified tie*. |
| **evaluation** | `scripts/gate eval` → `PipelineSummary` diffed vs the **golden** | **exploit / explore:** golden holds within tolerance, or a justified accuracy move. **refactor / consolidate:** **byte-identical** (`scripts/gate eval --tolerance 0`) — a consolidate that removes a *dead* path must not move a single number. **feature:** existing fields hold + **new fields allowed** (`scripts/gate eval --allow-new-fields`); re-save the golden post-merge to absorb them. |

Evaluation is the subtle one: a speedup that moves the Sharpe is a correctness
regression wearing a profiling win's clothes — the golden diff catches it. A
genuine accuracy improvement is the one case the numbers *should* move; the
writeup must say why the new value is correct.

## Why

Serial-merge with re-validation exists because green-on-branch ≠
green-after-a-sibling-merged — two passing PRs can interact. The golden diff
exists because unit tests pass while numbers silently move.

## Anti-patterns

- Merging a second PR without re-running the gates on the post-merge tree.
- Editing component code yourself instead of dispatching a worker.
- Skipping the `IMPROVEMENTS.md` dedup, or re-attempting a logged rejection.
- Banking a profiling win that moved the eval golden without a correctness
  justification — a silent regression.
- Treating a *moved existing* golden field as additive growth on a `feature`
  round — `--allow-new-fields` passes only when existing fields hold; a moved
  number still needs the accuracy justification.
- Forcing a win when no PR passed its type's gate (an `explore` round with no
  winner is `spiked`, not forced).
- Accepting a writeup that doesn't quote the gates
  ([[verification-before-completion]]).

## Exit condition

The round met **its type's acceptance bar** on the *post-merge* tree (exploit:
metric improved; refactor: golden byte-identical + cleaner; feature: capability
under test + golden holds or grows additively; explore: a merged winner, or every
attempt `spiked`; consolidate: superseded path + flag removed, golden
byte-identical, tree cleaner), correctness is green, the baseline is ratcheted,
and the round
— **with its `type:`** — is recorded in `IMPROVEMENTS.md` (explore losers in
`SPIKES.md`). If no PR passed its type's gate, record the rejection and stop —
don't force a win.
