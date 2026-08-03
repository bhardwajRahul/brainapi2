# LongMemEval Protocol Sheet

Goal: run BrainAPI against [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) with a documented, reproducible harness under `benchmarks/longmemeval/`.

## Corpus

| Variant | File | Role |
| --- | --- | --- |
| **S** (default) | `longmemeval_s_cleaned.json` | Paper LongMemEval_S (~40 sessions / ~115k tokens). Primary claim corpus. |
| **oracle** | `longmemeval_oracle.json` | Evidence sessions only — retrieval ablation upper bound. |
| **M** | `longmemeval_m_cleaned.json` | ~500 sessions; downloadable, not first-run scope. |

Source: Hugging Face [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).

500 questions across: `single-session-user`, `single-session-assistant`, `single-session-preference`, `temporal-reasoning`, `knowledge-update`, `multi-session`, plus abstention (`question_id` ends with `_abs`).

## Harness contract

- HTTP-only; one **brain per `question_id`** (`lme{alnum}`).
- Ingest haystack sessions only (timestamps + session ids). Never ingest answers or `has_answer` labels.
- Headline metric: official LongMemEval LLM yes/no judge accuracy over **all** questions (including abstention), with per-type breakdown.
- Secondary: session recall vs `answer_session_ids` (skip abstention).
- Answerer/judge models follow `BENCH_*` / `BENCH_PROFILE` (same as LoCoMo). Official judge **prompts** are ported from upstream `evaluate_qa.py`.

## Competitor bar (not apples-to-apples)

Published numbers use different answer/judge models. Record BrainAPI models in the run manifest / `REPORTS.json`.

| System | Reported overall | Notes |
| --- | ---: | --- |
| Mastra Observational Memory | ~94.9% | gpt-5-mini reader |
| Mem0 (2026 blog) | ~94.4% | proprietary stack |
| Mastra OM / oracle (gpt-4o) | ~84% / ~82% | closer to gpt-4o-class readers |
| Full-context gpt-4o | ~60% | long-context baseline |

BrainAPI claims should always list answer model, judge model, variant (`s` / `oracle`), and `BENCH_PROFILE`.

## Quickstart

```bash
cd benchmarks
./longmemeval.sh download --variant s
./longmemeval.sh smoke --limit 1
./longmemeval.sh ingest --run lme-s-smoke --limit 3
./longmemeval.sh evaluate --run lme-s-smoke --limit 3
./longmemeval.sh report --run lme-s-smoke
```

Ledger: successful runs with `n_questions >= 50` upsert `benchmarks.longmemeval` in `REPORTS.json`.
