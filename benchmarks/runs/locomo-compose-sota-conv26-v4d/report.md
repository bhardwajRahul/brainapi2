# LoCoMo BrainAPI Benchmark Report

- Run dir: `/Users/christiannonis/Documents/Projects/brainapi2/benchmarks/runs/locomo-compose-sota-conv26-v4d`
- Created: 2026-07-31T12:29:04.421641+00:00
- Granularity: n/a
- Answer model: deepseek-v4-flash
- Judge model: deepseek-v4-flash (provider: https://api.deepseek.com, same family as answerer: True)
- Git SHA: 78cbbae50ac37777ce0315f85156f7812b28248c (dirty: True)
- Answer prompt sha256: fd253f786ce84423478535a6aa95dc22b2961dd8f85c0d0a1f8bb035d56f8c37
- Judge prompt sha256: 94bc662ee22132e6d044145c395e014f0ff7787566a3088fdd781ba698e2c68e
- Dataset sha256: 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4
- Samples: conv-26

## Run integrity

- Rows in answers.jsonl: 152
- Scored unique questions: 152
- Errored rows (excluded): 0
- Duplicate rows dropped: 0
- Empty predictions: 0
- Rows with truncated context: 152

## Headline

- Judge accuracy (excl. adversarial / cat 5): **95.4%** [90.8%, 97.8%] (n=152)
- Judge accuracy (all categories): **95.4%** [90.8%, 97.8%] (n=152)
- Mean F1 (excl. adversarial): **0.461**
- Mean BLEU-1 (excl. adversarial): **0.377**
- Answerable rate (gold tokens in context): **94.1%**
- Evidence-session recall (full): **99.3%**
- Evidence-session recall (partial+full): **99.3%**
- Evidence-session recall, graph channel: **98.7%** (n=150)
- Evidence-session recall, passage channel: **97.3%** (n=150)
- Answerer gap (answerable − judge): **-1.3%**
- Abstention accuracy (cat 5): **n/a** (n=0)

Retrieval-side metrics are deterministic for a fixed brain and config; they are reported exactly and carry no significance test. Judge accuracy is not — compare runs with `python -m locomo compare`.

## Per-category

| Category | Name | N | Judge Acc | 95% CI | Answerable | EvRecall | EvRecall graph | EvRecall passages | Mean F1 | Mean BLEU-1 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | multi-hop | 32 | 96.9% | [84.3%, 99.4%] | 93.8% | 100.0% | 100.0% | 96.9% | 0.411 | 0.324 |
| 2 | temporal | 37 | 89.2% | [75.3%, 95.7%] | 89.2% | 97.3% | 97.3% | 94.6% | 0.451 | 0.369 |
| 3 | open-domain | 13 | 100.0% | [77.2%, 100.0%] | 76.9% | 100.0% | 90.9% | 90.9% | 0.157 | 0.095 |
| 4 | single-hop | 70 | 97.1% | [90.2%, 99.2%] | 100.0% | 100.0% | 100.0% | 100.0% | 0.547 | 0.457 |

## Latency & tokens

- Retrieval latency p50: 5194.209 ms
- Retrieval latency p95: 13785.859 ms
- Retrieval latency mean: 6666.379 ms
- Total LLM tokens (answer+judge): 43859841

## Ingest

- Units: 0
- Completed: 0
- Partial failed: 0
- Failed: 0

## Warnings

- 152 rows had a retrieved-context channel truncated before logging; see context_truncated in answers.jsonl.
