---
name: change-report
description: >
  Dispatch after a swarm to write one change report per landed change.
  One report-generator sub-agent per merged PR; reads the PR, transcript,
  flame-graph artifacts, and eval deltas, and writes
  reports/round-NNN/<component>.md from the template.
---

# Change-report generator

After a swarm round's PRs are serially merged and the baseline is ratcheted,
the orchestrator dispatches **one report-generator sub-agent per landed change**.

## When to invoke

After step 6 (ratchet + record) of `improvement-orchestrator` — for each PR
that received `verdict: accepted` in `IMPROVEMENTS.md`.

## What the sub-agent does

1. **Read the PR.** `gh pr view <PR>` — get the diff, writeup, gate output.
2. **Read the worker transcript.** The JSONL transcript lives at
   `.claude/transcripts/<worktree-slug>.jsonl` (path provided by orchestrator).
3. **Read the flame-graph artifacts.** Query `profile_artifacts.parquet` under
   the component's `profiles/` tree for the before and after run_ids; copy the
   calltree txt files and the HTML flamegraph to
   `reports/round-NNN/assets/<component>-{before,after}.cpu.calltree.txt` and
   `reports/round-NNN/assets/<component>-flamegraph.html`.
4. **Read the eval delta.** Diff the golden `PipelineSummary` (saved by
   orchestrator before the swarm) against the post-merge summary.
5. **Fill the template.** Copy `reports/_template.md` to
   `reports/round-NNN/<component>.md` and fill each of the five sections:
   - § What it addressed — component, metric, hotspot from PR description.
   - § How it decided — ruled-out approaches from writeup + calltree excerpt.
   - § Pre/post profile — p50_ms/peak_mb table + `![flame graph](assets/...)`.
   - § System impact — eval delta table + `IMPROVEMENTS.md` ledger entry.
   - § Suggested next steps — remaining hot paths from after calltree.
6. **Fill the frontmatter.** All seven fields required (see `reports/README.md`).
7. **Commit.** `scripts/committer "docs(reports): add round-NNN <component> report"
   reports/round-NNN/`.

## Rebuild the site

After all reports for the round are committed:

```bash
scripts/reports build       # generates reports/_site/
scripts/reports --serve     # build + serve over Tailscale HTTPS
```

## Inputs the orchestrator must provide

- `<PR>` — GitHub PR number.
- `<round>` — zero-padded round number (e.g. `003`).
- `<component>` — the component that shipped the change.
- `<before-run-id>` / `<after-run-id>` — profiling run IDs from `profile_artifacts.parquet`.
- `<golden-summary-path>` — path to the pre-swarm `PipelineSummary` JSON snapshot.
- `<post-summary-path>` — path to the post-merge `PipelineSummary` JSON snapshot.

## Anti-patterns

- Writing a report for a rejected PR — only accepted changes get reports.
- Leaving frontmatter fields empty or using placeholder values — the site
  generator raises on missing fields.
- Copying benchmark numbers from the PR writeup without checking the
  `profile_artifacts.parquet` source — use the artifacts directly.
