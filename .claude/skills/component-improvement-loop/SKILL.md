---
name: component-improvement-loop
description: Use when you've been dispatched as a worker to improve one of bt's seven components (etl, signals, models, analysis, portfolio, backtest, profiling) behind its public API — work in your worktree, keep the change additive, run and quote every gate, and open a reviewed PR. The round is driven by [[improvement-orchestrator]].
---

# Component improvement loop — worker protocol

bt is seven components composing across stable contracts (see [[architecture]],
[[interfaces]]). You are a cold sub-agent dispatched to make *one* component
better — faster, more accurate, cleaner — without breaking the contract
everything else codes against. The [[improvement-orchestrator]] owns the round,
proposed the target, and will adjudicate the gates and merge; your job is the
change itself, landed as a reviewed PR that quotes every gate. This is your
protocol.

## When to apply

When the orchestrator has dispatched you to improve component X behind its API.
Improving the *implementation* is in scope; touching the public surface is not
without flagging back. If the contract *must* break, that's a
[[design-before-architecture-changes]] conversation with Jack — stop and say so,
don't work around it.

## Your brief

The orchestrator hands you: the **round type**, one sentence of goal, the exact
files (`portfolio/optimizer.py`, its `_protocol.py`, its tests), what's been
ruled out, and the declared metric + eval tolerance. **Read your component's
context pack first** — `docs/worker-context/<component>.md` maps the files, the
public API you must not break, the harness hot path, and what's already been
optimized, so you orient in one read instead of grepping the component by hand
(re-verify anything load-bearing against the code — the pack can lag). Your
mandate — and what "done" looks like — depends on the type:

- **exploit** — the smallest change that wins the metric gate; small diffs review
  and revert cleanly. Behavior holds the golden within tolerance.
- **refactor** — split a large/tangled function, add unit tests, **change no
  behavior**: the golden must come back **byte-identical** (`scripts/gate
  consolidate-check` — a direct vs-`main` diff; see the consolidate note below for
  why this beats `eval --tolerance 0`). You win on structure (length/complexity
  down, coverage up), not speed — profiling need only show *no regression*.
- **feature** — add a new capability **additively, behind a flag/default**: a new
  function, a new `_protocol.py` method with a default, or a new
  `PipelineSummary` field. Existing numbers hold; any new fields are gated by
  `evalgate --allow-new-fields`. Ship it with tests that exercise the new path.
- **explore** — **you are the bold rewrite.** Maximize divergence from the
  incumbent (a different engine, algorithm, or data layout); a minimal diff is a
  *failed* explore. You are one of K siblings on the same target — a judge ranks
  you against them, so take the approach the others won't.
- **consolidate** — **remove a superseded path.** A prior round added a
  replacement and kept the old impl as a shadow/oracle; delete the old
  implementation + its flag, make the replacement the sole path, and update every
  call site. The golden must come back **byte-identical** — prove it with
  `scripts/gate consolidate-check`, which diffs your tree against a *freshly
  computed* `main` summary on this machine. Use this, **not** `eval --tolerance 0`:
  the committed golden carries pre-existing ~1e-16 fp-noise (a BLAS/build artifact
  present on clean `main` too), so `--tolerance 0` flags spuriously; a same-machine
  vs-`main` diff cancels that noise and any remaining delta is real. You are
  removing *dead* code, not changing behavior. Keep a path only if it is still
  genuinely used (a real fallback) or is a test oracle a test still references; say
  which in the PR.

## The gates (your PR must pass all of them — run and quote each)

Run every gate with the **`scripts/gate`** runner. It pins to your worktree,
redirects heavy temp off the RAM-backed `/tmp` (without that, concurrent workers
fill tmpfs, writes fail with `EDQUOT`, and git aborts), and holds the bench lock
for the timed run — so you never source diskguard or memorize an invocation.

| gate | command | the bar |
|---|---|---|
| **lint + types** | `scripts/gate lint` (ruff check + format --check + ty) | clean; warnings are errors, `error-on-warning`. No `# type: ignore` / suppression — fix or widen. Also enforced per commit by `scripts/committer`. |
| **correctness / regression** | `scripts/gate test <component>/ tests/integration/` (or `scripts/gate test` for the full suite) | still green — same count, no new failures. The integration suite imports across components, so a broken contract fails *here*. |
| **profiling** | `scripts/gate bench` (harness through the bench lock) + `check_regressions` vs the ratcheted baseline | per round type: **improved** (exploit), **no regression** (refactor / feature), or **improved-or-justified-tie** (explore). No other stage regresses past threshold. |
| **evaluation** | `scripts/gate eval` → `PipelineSummary` vs the round's **golden** (`--allow-new-fields` for feature); for refactor/consolidate use `scripts/gate consolidate-check` (direct vs-`main` diff, cancels the golden's fp-noise) instead of `eval --tolerance 0` | per type: holds within tolerance (exploit / explore), **byte-identical** (refactor / consolidate), or holds + **new fields** (feature). A genuine accuracy improvement is the one case a number *should* move — justify it. |

`scripts/gate all <component>` runs lint + test + eval in order for a fast check.

Lint and types are *continuous* — `scripts/committer` runs them on every atomic
commit, so your PR is lint/type-clean by construction. The evaluation gate is the
subtle one: a refactor can keep the API and pass every unit test yet *silently
move the numbers*. A speedup that changes the Sharpe is a correctness regression
wearing a profiling win's clothes. A genuine accuracy improvement is the one case
the numbers *should* move — your writeup must say so and show why it's correct.

## API discipline — removal allowed, prefer two-phase

bt has **no external API consumers**, so backwards-compat is not a goal (see
[[DECISIONS]], "no backwards-compat; two-phase add-then-consolidate"). You **may**
remove or rename internal symbols, change signatures, and delete superseded paths
— *provided you update every call site* and the **golden stays byte-identical**
(`scripts/gate consolidate-check`) with the integration suite green. Those two
are the guards that a removal didn't change behavior; the integration suite
imports across the component boundary, so a broken cross-component contract fails
there.

- **Preferred workflow is two-phase.** On explore/feature rounds, *add* the new
  path additively and keep the old one as a correctness **oracle** while your
  change is under review — it's how you prove bit-identity. A later
  **consolidate** round deletes the now-dead shadow.
- **On a `consolidate` round, removal is the job:** delete the shadow + its flag,
  make the replacement sole, update call sites; golden byte-identical.
- Still off-limits without a [[design-before-architecture-changes]] talk:
  changing a **cross-component data contract** (the frame shapes / new-or-moved
  `PipelineSummary` fields other components consume). Internal cleanup is yours.

## Worktree, PR, and scope

- **Work in your worktree** at `.worktrees/<component>-<slug>`, branch
  `improve/<component>-<slug>` off `main`. Isolated trees let parallel sub-agents
  build and run gates without colliding.
- **Pin every command to your worktree.** Your shell cwd does **not** persist
  between commands and defaults to the repo root, so a bare relative `Write` /
  `Edit` / `vim component/x.py` lands in the **primary** checkout, not your
  worktree — corrupting the main tree and risking a commit onto the wrong branch.
  Either prefix every command with `cd .worktrees/<component>-<slug> &&`, or use
  absolute paths under it. Before your first edit, confirm
  `git rev-parse --show-toplevel` ends in `.worktrees/<component>-<slug>`.
  `scripts/committer` **refuses** component-dir commits from the primary worktree
  (exit 3), so a slipped commit fails loudly rather than landing on the trunk —
  but don't rely on the net; pin the cwd.
- **Commit atomically with `scripts/committer`** (never `git commit` directly,
  never bypass its ruff/ty gates) as you go.
- **Stay in your lane.** Your PR touches only your component's directory + the
  shared ledgers (`API_REQUESTS.md`). The orchestrator rejects any cross-lane diff.
- **Open the PR via `gh`** using `pr-writeup.md`. It must carry: unit tests;
  integration tests; profiling before→after (same GenSpec grid + seed + env);
  correctness/accuracy delta if applicable; why it should merge; how it fits the
  existing architecture; pros & cons; the declared target + result;
  reproducibility (exact command, env, grid); risk & rollback.

## The ledgers

- **`API_REQUESTS.md`** — if you need data a sibling component doesn't yet emit,
  post a request here instead of crossing the boundary, and work with what you
  have this round. The producer picks it up a later round; it closes only when
  the field lands additively.
- **`IMPROVEMENTS.md`** — append-only round log, the loop's memory. The
  orchestrator records the round here; you don't write to it.

## Anti-patterns

- Dropping a symbol or changing a signature **without updating every call site**,
  or in a way that moves the golden — that's a real break. Removal itself is fine
  (golden byte-identical + integration green); an *un-updated* removal is not.
- Keeping a dead shadow path around "to be safe" past its consolidate round, or
  deleting a path that is still genuinely used / is a live test oracle.
- A PR that edits outside your component's lane.
- Crossing a component boundary instead of posting to `API_REQUESTS.md`.
- A sweeping rewrite on an `exploit`/`refactor` round when a small diff would win
  — but on an `explore` round the opposite is the anti-pattern: a timid diff
  fails the round, because divergence is the whole point.
- Declaring done without quoting all the gates
  ([[verification-before-completion]]).

## Exit condition

Your change lands as a PR off `improve/<component>-<slug>`, in your lane,
additive, with every gate run and quoted in the `pr-writeup.md` body. The
orchestrator adjudicates and merges from there.
