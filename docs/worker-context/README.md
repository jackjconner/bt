# worker-context — per-component context packs for improvement-loop workers

These are tight (~25–45 line) reference packs, one per component (etl, signals,
models, analysis, portfolio, backtest, profiling). The
[[improvement-orchestrator]] injects the relevant pack into a dispatched
worker's brief so the worker doesn't burn ~15 tool calls mapping its component
by hand at the start of every round.

Each pack has a fixed shape: one-line responsibility, file→purpose list, the
public API (the `__all__` export surface + the `_protocol.py` Protocol — the
additive-only contract a worker must not break), the harness entry / hot path
(the `harness/components.py` call the profiling gate measures, and whether it's
on the pipeline golden path), the GenSpec data contract it consumes, and a
"recently optimized" list distilled from `IMPROVEMENTS.md` so workers don't
re-attempt a shipped win.

Sources: the bt wiki (`~/syl/wiki/projects/bt-backtester/{components,interfaces}.md`)
plus the actual code. Where the two disagreed, the code won.

These packs can go stale as components evolve. Treat a pack as a fast-start
map, not ground truth — re-verify any signature, hot path, or `__all__` entry
against the code before relying on it.
