# BEAM BrainAPI Benchmark Report

- Run dir: `/Users/christiannonis/Documents/Projects/brainapi2/benchmarks/runs/beam-1m-1-product-v3c-compose`
- Created: 2026-08-03T07:04:03.123615+00:00
- Size: 1M
- Answer model: deepseek-v4-flash
- Judge model: deepseek-v4-flash (provider: https://api.deepseek.com, same family as answerer: True)
- Judge prompt variant: beam-rubric-v1-question-aware
- Git SHA: 78cbbae50ac37777ce0315f85156f7812b28248c (dirty: True)
- Samples: 1M/1

## Run integrity

- Rows in answers.jsonl: 20
- Scored unique questions: 20
- Errored rows (excluded): 0
- Duplicate rows dropped: 0
- Empty predictions: 0

## Headline

- Mean ability score: **79.0%** (n_questions=20, abilities=10)

## Per-ability

| Ability | N | Mean |
|---|---:|---:|
| abstention | 2 | 50.0% |
| contradiction_resolution | 2 | 68.8% |
| event_ordering | 2 | 42.2% |
| information_extraction | 2 | 100.0% |
| instruction_following | 2 | 100.0% |
| knowledge_update | 2 | 100.0% |
| multi_session_reasoning | 2 | 93.8% |
| preference_following | 2 | 87.5% |
| summarization | 2 | 47.6% |
| temporal_reasoning | 2 | 100.0% |

## Latency & tokens

- Retrieval latency p50: 15785.345 ms
- Retrieval latency p95: 43059.156 ms
- Retrieval latency mean: 20108.512 ms
- Total LLM tokens (answer+judge): 1645600

## Ingest

- Units: 0
- Completed: 0
- Partial failed: 0
- Failed: 0
