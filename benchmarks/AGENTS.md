# Benchmarks — agent guidance

## LoCoMo harness

The LoCoMo harness lives in `benchmarks/locomo/`. It talks to a running BrainAPI over HTTP only (no `src/` imports). Typical flow: `download` → `ingest` → `evaluate` → `report` via `./locomo.sh` or `.venv/bin/python -m locomo`.

## Results ledger (`REPORTS.json`)

`benchmarks/REPORTS.json` is the public BrainAPI results ledger: top published scores for **all** benchmark suites under `benchmarks.<suite>`.

- LoCoMo harness upserts `benchmarks.locomo` via `write_report` → `update_reports_json` when report `status` is `ok`.
- After any completed evaluate/report that produces a successful scored run, ensure the suite entry in `REPORTS.json` reflects it.
- If you write or patch `runs/<id>/report.json` manually, update `REPORTS.json` yourself (same schema; upsert by `run_id` under the suite).
- When adding a new suite, add a `benchmarks.<suite_id>` key (`name` + `leaderboard`) and upsert the same way.
- Skip failed/junk runs (`status: failed`, 0% judge, nonsense scores) — they do not belong on the leaderboard.
- Research caveats and checkpoint narrative live in `runs/CHECKPOINT_NOTES.md` (local), not in `REPORTS.json`.

## Tracks

| Track | Profile | Intent |
| --- | --- | --- |
| **SOTA** | `BENCH_PROFILE=sota` | Beat HyperMem-class LoCoMo (≥93% on full-10). Prefer **deepseek-v4-flash** answerer/judge. SC + gap-fill live in the harness, not product latency path. |
| **Product** | `BENCH_PROFILE=product` (default) | ADR-006 sub-second `/retrieve/context` path — greedy, no online LLM loops. |

Protocol and competitor bar: `docs/research/08-sota-locomo-protocol.md`. Historical checkpoint narrative: `runs/CHECKPOINT_NOTES.md`.

## Live stack

TUI / `brainapi start` may run from `~/.brainapi/source`, not this workspace. Sync or restart from the intended checkout before measuring, or scores will reflect stale code.

## Conventions

- Brain IDs are alphanumeric (`conv-26` → `locomoconv26`). Prefer clean brains (`locomoconv26clean`) for product/SOTA claims after Phase 0.
- Compare judge accuracy with `./locomo.sh compare` (paired McNemar). Do not A/B graph EvR until identical-config session-set agreement clears the ≥95% gate.
- Do not ingest QA annotations. Keep answer prompts prompt-audit clean (`./locomo.sh prompt-audit`).
