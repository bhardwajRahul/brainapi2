# BEAM protocol (BrainAPI harness)

Goal: evaluate BrainAPI as a long-term memory system on [BEAM](https://github.com/mohammadtavakoli78/BEAM) (ICLR 2026 / [arXiv:2510.27246](https://arxiv.org/abs/2510.27246)).

This is **not** a port of LIGHT. BrainAPI replaces RAG/LIGHT via `POST /ingest/` + `POST /retrieve/context`.

## Dataset

| Item | Value |
| --- | --- |
| Source (≤1M) | HuggingFace [`Mohammadta/BEAM`](https://huggingface.co/datasets/Mohammadta/BEAM) |
| Source (10M) | HuggingFace [`Mohammadta/BEAM-10M`](https://huggingface.co/datasets/Mohammadta/BEAM-10M) |
| License | CC BY-SA 4.0 |
| Splits | `100K`, `500K`, `1M`, `10M` (paper “128K” ≡ HF `100K`) |
| Local path | `benchmarks/data/beam/{size}/{conversation_id}/` |
| Env overrides | `BENCH_BEAM_HF_DATASET`, `BENCH_BEAM_HF_DATASET_10M` |

Do **not** ingest probing questions, rubrics, or ideal answers.

### BEAM-10M multi-plan normalize

Paper + HF schema: each 10M row is ~10 interlocking plans (`plan-1`…`plan-10` / `plans[]`), generated sequentially so the narrative is chronological.

**Harness decision:** concatenate plans in plan-number order into **one** long `chat.json`, then ingest with the same turn units as 1M (`bN_tM`, `session_N`, `time_anchor` stamps).

| Choice | Why |
| --- | --- |
| Concatenate chronologically (not parallel brains) | Matches sequential plan expansion in the paper; probing spans the full dialogue |
| Prefer main `chat` plan-* keys when present | HF already ships per-plan batches with `turns`; fall back to `plans[].chat` + `convert_chat_batches` |
| Global `batch_number` renumber | Keeps unit ids unique under existing `bN_tM` resume keys |
| `brain_id` | `beam10m1`; clean product claims use `--brain beam10m1clean` (never wipe `beam1m1clean`) |

`meta.json` records `normalize_strategy: concatenate_plans_chronological`, `n_plans`, per-plan batch ranges, and `ingest_target` (`n_turns - 5`, floor 1). **Do not invent ETA/TARGET without reading `n_turns` after download.**

```bash
./beam.sh download --size 10M
./beam.sh dataset-stats --size 10M
# then smoke / ingest with --size 10M --sample 1 --brain beam10m1clean --run beam-10m-1-clean
```

Still requires `benchmarks/.env`, a live API + celery, and an **explicit** go before long ingest (see `scripts/run_beam_10m_1_ingest_eval.sh`).

## Pipeline

```text
./beam.sh download --size 100K
./beam.sh smoke --size 100K --sample 1
./beam.sh ingest --size 100K --sample 1 --run beam-100k-1
./beam.sh evaluate --size 100K --sample 1 --run beam-100k-1
./beam.sh report --run beam-100k-1
```

Brain IDs: `beam{size}{convid}` → `beam100k1`, `beam10m1`.

Ingest unit = one batch turn (user + assistant messages), with `time_anchor` as `source_timestamp` when present.

## Parallel ingest (ops)

| Knob | Recommendation |
| --- | --- |
| Harness `--concurrency` | **2** default safe; **3** if API stays healthy; **≤4** soft max |
| `CELERY_WORKER_CONCURRENCY` | ≥ harness concurrency (product `.env` default **4**; thread pool) |
| Resume | On by default; skips `completed` / `partial_failed` / `permanent_failed` and legacy embed-8192 `failed` rows |
| Stall | Ops scripts stop after 3 no-progress tries; harness does not requeue permanent embed-8192 fails |

**ETA (1M/1, ~625 turns):** prior run ≈ **40 h** at concurrency **1**. Expect roughly **20–25 h** at **2**, **14–18 h** at **3** (contention / uvicorn restarts can erase ideal linear speedup).

**10M:** derive TARGET from `meta.json` `n_turns` / `ingest_target` after normalize — do not scale 1M hours blindly.

**Caveats:** concurrent units on one brain contend on Neo4j/Postgres and can knock over uvicorn under load. Keepalive must only restart when port **8000** is not LISTENing (never `pkill` a healthy API while evaluate-only is running). Prefer a **new brain id** for any full re-ingest; do not wipe brains mid-eval.

Example next 1M re-ingest (new brain; do not start casually):

```bash
BEAM_INGEST_CONCURRENCY=2 BEAM_INGEST_TIMEOUT=2400 \
  ./scripts/run_beam_1m_1_ingest_eval.sh
# or:
./beam.sh ingest --size 1M --sample 1 --run beam-1m-1-clean2 \
  --brain beam1m1clean2 --concurrency 2 --timeout 2400
```

## Scoring

- Per-question: rubric LLM-judge with `unified_llm_judge_base_prompt`.
- Variant: `beam-rubric-v1-question-aware` (always substitutes `<question>`; scores as `float` so `0.5` is kept).
- Ability score for reporting:
  - `event_ordering` → `tau_norm` (Kendall τ after LLM alignment)
  - all others → mean `llm_judge_score` over rubric items
- Headline: mean of the 10 ability means (paper-style).
- Ledger field: `headline_score` (continuous `[0,1]`), not LoCoMo’s binary accuracy.

## Abilities

abstention, contradiction_resolution, event_ordering, information_extraction, instruction_following, knowledge_update, multi_session_reasoning, preference_following, summarization, temporal_reasoning.

## Cost notes

100K chats average ~100+ turns; accurate ingest is slow/expensive. Start with `--limit-turns` / `--limit` and a single sample. Rubric judging can issue several LLM calls per question. 10M is far larger — count turns first.

## Citation

```
@misc{tavakoli2025milliontokensbenchmarkingenhancing,
  title={Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs},
  author={Mohammad Tavakoli and Alireza Salemi and Carrie Ye and Mohamed Abdalla and Hamed Zamani and J Ross Mitchell},
  year={2025},
  eprint={2510.27246},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.27246},
}
```
