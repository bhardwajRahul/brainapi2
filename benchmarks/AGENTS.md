# Benchmarks — agent guidance

## Suites

| Suite | Package | Wrapper | Ledger key |
| --- | --- | --- | --- |
| **LoCoMo** | `benchmarks/locomo/` | `./locomo.sh` | `benchmarks.locomo` |
| **LongMemEval** | `benchmarks/longmemeval/` | `./longmemeval.sh` | `benchmarks.longmemeval` |
| **BEAM** | `benchmarks/beam/` | `./beam.sh` | `benchmarks.beam` |

All harnesses talk to a running BrainAPI over HTTP only (no `src/` imports).

### LoCoMo

Typical flow: `download` → `ingest` → `evaluate` → `report` via `./locomo.sh` or `.venv/bin/python -m locomo`.

### LongMemEval

Typical flow: `download` → `ingest` → `evaluate` → `report` via `./longmemeval.sh` or `.venv/bin/python -m longmemeval`. Protocol: `docs/research/09-longmemeval-protocol.md`.

### BEAM

Typical flow: `download` → `ingest` → `evaluate` → `report` via `./beam.sh` or `.venv/bin/python -m beam`.

- Dataset: HuggingFace `Mohammadta/BEAM` splits `100K` / `500K` / `1M` (BEAM-10M deferred).
- Ingest turns only — never probing questions, rubrics, or ideal answers.
- Scoring: rubric LLM-judge (`judge_prompt_variant: beam-rubric-v1-question-aware`); headline = mean of 10 ability means; `event_ordering` uses `tau_norm`.
- Brain IDs: `beam{size}{convid}` → e.g. `beam100k1`.
- Protocol: [`docs/research/10-beam-protocol.md`](../docs/research/10-beam-protocol.md).
- Ingest parallelism: prefer `--concurrency 2` (≤4); resume skips permanent embed-8192 fails. See protocol “Parallel ingest” + `scripts/run_beam_1m_1_ingest_eval.sh` (`BEAM_INGEST_CONCURRENCY`). Do not wipe live eval brains.

## Results ledger (`REPORTS.json`)

`benchmarks/REPORTS.json` is the public BrainAPI results ledger: top published scores for **all** benchmark suites under `benchmarks.<suite>`.

- LoCoMo harness upserts `benchmarks.locomo` via `write_report` → `update_reports_json` when report `status` is `ok`.
- BEAM harness upserts `benchmarks.beam` the same way (field: `headline_score`, continuous `[0,1]`).
- After any completed evaluate/report that produces a successful scored run, ensure the suite entry in `REPORTS.json` reflects it.
- If you write or patch `runs/<id>/report.json` manually, update `REPORTS.json` yourself (same schema; upsert by `run_id` under the suite).
- When adding a new suite, add a `benchmarks.<suite_id>` key (`name` + `leaderboard`) and upsert the same way.
- Skip failed/junk runs (`status: failed`, 0% judge, nonsense scores) — they do not belong on the leaderboard.
- Research caveats and checkpoint narrative live in `runs/CHECKPOINT_NOTES.md` (local), not in `REPORTS.json`.

## Tracks (LoCoMo)

| Track | Profile | Intent |
| --- | --- | --- |
| **SOTA** | `BENCH_PROFILE=sota` | Beat HyperMem-class LoCoMo (≥93% on full-10). Prefer **deepseek-v4-flash** answerer/judge. SC + gap-fill live in the harness, not product latency path. |
| **Product** | `BENCH_PROFILE=product` (default) | ADR-006 sub-second `/retrieve/context` path — greedy, no online LLM loops. |

Protocol and competitor bar: `docs/research/08-sota-locomo-protocol.md`. Historical checkpoint narrative: `runs/CHECKPOINT_NOTES.md`.

## Live stack

TUI / `brainapi start` may run from `~/.brainapi/source`, not this workspace. Sync or restart from the intended checkout before measuring, or scores will reflect stale code.

## Conventions

- Brain IDs are alphanumeric (`conv-26` → `locomoconv26`; BEAM `100K/1` → `beam100k1`). Prefer clean brains for product/SOTA claims after Phase 0.
- Compare LoCoMo judge accuracy with `./locomo.sh compare` (paired McNemar). Do not A/B graph EvR until identical-config session-set agreement clears the ≥95% gate.
- Do not ingest QA annotations / probing questions. Keep answer prompts prompt-audit clean where the suite provides it (`./locomo.sh prompt-audit`).
