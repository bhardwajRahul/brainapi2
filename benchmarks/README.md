# LoCoMo benchmark for BrainAPI

Standalone harness that:

1. Downloads the [LoCoMo](https://github.com/snap-research/LoCoMo) dataset
2. Ingests conversations into BrainAPI over HTTP (`POST /ingest/`)
3. Answers annotated QA using `POST /retrieve/context` + an LLM answerer
4. Scores with an LLM judge plus F1 / BLEU-1

This package does **not** import anything from `src/`. It only talks to a running BrainAPI instance.

## Prerequisites

- BrainAPI running locally (`make start-all` or `brainapi start`)
- `BRAIN_CREATION_ALLOWED="true"` in the server env (needed so brains like `locomoconv26` are auto-created)
- A PAT token (`BRAINPAT_TOKEN`)
- A DeepSeek API key for answering and judging (or any OpenAI-compatible provider)

Brain IDs must be alphanumeric. This harness maps `conv-26` → `locomoconv26`.

## Setup

```bash
cd benchmarks
python3 -m venv .venv
source .venv/bin/activate
# If `python` is aliased to a system binary (common on macOS), prefer the wrapper:
#   ./locomo.sh ...
# or call the venv interpreter directly:
#   .venv/bin/python -m locomo ...
pip install -r requirements.txt
cp .env.example .env
# edit .env: BRAINPAT_TOKEN, DEEPSEEK_API_KEY
```

## Quickstart

```bash
# 1. Download dataset
./locomo.sh download
./locomo.sh dataset-stats

# 2. Smoke-test ingest + retrieve against your API
./locomo.sh smoke --brain locomosmoke

# 3. Ingest one conversation (session-level; recommended start)
./locomo.sh ingest --sample conv-26 --run locomo-conv26

# 4. Evaluate a small slice first
./locomo.sh evaluate --sample conv-26 --run locomo-conv26 --limit 20

# 5. Full non-adversarial QA for that conversation
./locomo.sh evaluate --sample conv-26 --run locomo-conv26

# 6. Rebuild the report anytime (no API calls)
./locomo.sh report --run locomo-conv26
```

Equivalent: `.venv/bin/python -m locomo ...` (avoid bare `python` if you have a shell alias).

Reports land in `runs/<run_id>/`:

- `ingest.jsonl` — per-session ingest outcomes
- `answers.jsonl` — per-QA predictions and scores
- `manifest.json` — run config
- `report.md` / `report.json` — aggregated metrics

## Commands

| Command | Purpose |
|---|---|
| `download` | Fetch `data/locomo10.json` |
| `dataset-stats` | Print sessions / turns / QA per sample |
| `smoke` | One-shot ingest + retrieve health check |
| `ingest` | Push sessions (or turns) into BrainAPI |
| `answer-once` | Retrieve + answer a single question |
| `selftest-metrics` | Local F1 / BLEU-1 sanity checks |
| `evaluate` | Retrieve → answer → judge for QA items |
| `report` | Aggregate `answers.jsonl` into markdown/JSON |

### Useful flags

**ingest**

- `--sample conv-26` (repeatable; default = all 10)
- `--granularity session|turn` (default `session`)
- `--concurrency 2`
- `--limit-sessions N`
- `--dry-run`
- `--no-resume`
- `--run <id>` (resume / continue a named run)

**evaluate**

- `--sample conv-26`
- `--limit 20`
- `--categories 1,2,4`
- `--include-adversarial` (category 5 is skipped by default)
- `--concurrency 2`
- `--run <id>`

## Scoring

- **Headline number**: LLM-judge accuracy excluding category 5 (adversarial), matching common LoCoMo memory-system reporting.
- Also reported: overall accuracy, per-category breakdown, mean F1, mean BLEU-1, retrieval latency p50/p95, total LLM tokens.
- Categories: `1` multi-hop, `2` temporal, `3` open-domain, `4` single-hop, `5` adversarial.

Defaults use **DeepSeek** (override in `.env`):

- Base URL: `https://api.deepseek.com` (`BENCH_LLM_BASE_URL`)
- Answerer + judge: `deepseek-v4-flash` (`BENCH_ANSWER_MODEL` / `BENCH_JUDGE_MODEL`)
- API key: `DEEPSEEK_API_KEY` (or `OPENAI_API_KEY` for other providers)

Other OpenAI-compatible providers work the same way — set `BENCH_LLM_BASE_URL`, model ids, and the key.

## Cost and time expectations

LoCoMo conversations are long (roughly 19–32 sessions and 370–690 turns each). With `PIPELINE_MODE=accurate`, ingesting even one conversation can take a long time and burn substantial LLM tokens inside BrainAPI.

Recommendations:

1. Start with `conv-26` only.
2. Prefer `--granularity session` (default).
3. Use `--limit-sessions` / `--limit` while iterating.
4. Optionally compare a `PIPELINE_MODE=lightweight` ingest run for cost/quality tradeoffs.
5. Do **not** ingest the `qa` annotations — that leaks the test set.

Ingest and evaluate are resumable: re-running the same `--run` skips completed units / QA rows.

## Adding another benchmark

Create a sibling package under `benchmarks/`:

```text
benchmarks/
  locomo/          # this package
  yourbench/       # new package with its own cli / client / metrics
  requirements.txt # shared deps, or a per-bench requirements file
```

Reuse patterns from `locomo/`:

- HTTP-only `BrainAPIClient`
- JSONL resume files under `runs/`
- Separate ingest → evaluate → report stages

Keep each benchmark self-contained and free of imports from BrainAPI's `src/`.
