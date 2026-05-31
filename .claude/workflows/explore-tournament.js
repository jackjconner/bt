export const meta = {
  name: 'explore-tournament',
  description:
    'Fan out K divergent bold rewrites of one target, judge them, return a ranked list for the orchestrator to apply the hybrid reward.',
  whenToUse:
    'An explore round: challenge one incumbent with K deliberately divergent rewrites and keep the best.',
  phases: [
    { title: 'Diverge', detail: 'K workers, one per strategy, each in an isolated worktree' },
    { title: 'Judge', detail: 'score every attempt on gate-outcome / magnitude / structure / headroom / risk' },
  ],
}

// This is a convenience encoding of the explore tournament. The authoritative
// procedure lives in .claude/skills/improvement-orchestrator/SKILL.md ("The
// explore tournament"); keep the two in sync, and prefer the skill if they drift.
//
// args: { target: string, strategies: [{ slug: string, brief: string }, ...] }
//   target     — the incumbent being challenged (e.g. "signals.ic_series_v2 hot path")
//   strategies — the K deliberately-divergent approaches to try in parallel

const target = (args && args.target) || 'unspecified target'
const strategies = (args && args.strategies) || []

if (!strategies.length) {
  log('explore-tournament: no strategies in args.strategies — nothing to run')
  return { target, ranked: [], winner: 'none', note: 'no strategies provided' }
}

const RESULT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['gates_passed', 'summary'],
  properties: {
    gates_passed: { type: 'boolean', description: 'did every gate pass on your branch?' },
    pr: { type: 'string', description: 'PR number or url, if you opened one' },
    summary: { type: 'string', description: 'one paragraph: what you did and how divergent it is' },
    metric_before: { type: 'number', description: 'the round metric before (if applicable)' },
    metric_after: { type: 'number', description: 'the round metric after (if applicable)' },
    structural_note: { type: 'string', description: 'how clean / how radical the rewrite is' },
    headroom: { type: 'string', description: 'what optimization frontier this opens, if any' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['winner', 'merge_reason', 'ranking', 'rationale'],
  properties: {
    winner: { type: 'string', description: 'slug of the attempt to merge, or "none"' },
    merge_reason: { type: 'string', enum: ['strict-win', 'structural-tie', 'none'] },
    ranking: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['slug', 'score', 'note'],
        properties: {
          slug: { type: 'string' },
          score: { type: 'number', description: 'overall 0-100' },
          note: { type: 'string', description: 'why it ranked here; SPIKES.md learning if it lost' },
        },
      },
    },
    rationale: { type: 'string', description: 'the hybrid-reward call, in one paragraph' },
  },
}

phase('Diverge')
log(`explore: ${strategies.length} divergent attempts on ${target}`)

const attempts = (
  await parallel(
    strategies.map((s, i) => () =>
      agent(
        `You are explore-worker "${s.slug}" (attempt ${i + 1}/${strategies.length}) in an EXPLORE round.\n` +
          `Incumbent to challenge: ${target}\n` +
          `Your divergent strategy: ${s.brief}\n\n` +
          `Be the bold rewrite — maximize divergence from the incumbent; a minimal diff is a FAILED explore. ` +
          `Follow the component-improvement-loop worker protocol: work in your worktree, keep the public API ` +
          `additive, run and quote every gate (lint/types/correctness/profiling/evaluation), open a PR via gh. ` +
          `Then return the structured result.`,
        { label: `diverge:${s.slug}`, phase: 'Diverge', isolation: 'worktree', schema: RESULT_SCHEMA },
      ).then((r) => ({ slug: s.slug, brief: s.brief, ...r })),
    ),
  )
).filter(Boolean)

if (!attempts.length) {
  log('explore-tournament: every attempt failed to return — all spiked')
  return { target, ranked: [], winner: 'none', note: 'all attempts failed' }
}

phase('Judge')
const verdict = await agent(
  `You are the JUDGE of an explore tournament challenging: ${target}.\n` +
    `Score each of the ${attempts.length} attempts on gate-outcome, magnitude of win, structural quality, ` +
    `headroom opened, and risk. Rank them best-first and name the winner under the HYBRID reward: a strict ` +
    `gate-win, OR a tie that is structurally clearly better / opens flagged headroom. If no attempt clears ` +
    `that bar, winner = "none" (the round is spiked). For every loser, give the one-line SPIKES.md learning.\n\n` +
    `Attempts:\n` +
    attempts.map((a) => `- ${a.slug} (${a.brief}): ${JSON.stringify(a)}`).join('\n'),
  { label: 'judge', phase: 'Judge', schema: VERDICT_SCHEMA },
)

log(`explore-tournament: winner = ${verdict.winner} (${verdict.merge_reason})`)
return { target, attempts, ...verdict }
