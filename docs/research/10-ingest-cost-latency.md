# 10 — Ingest cost and latency (without quality trade-offs)

Workstream: make `PIPELINE_MODE=accurate` ingestion cheaper and faster while preserving event-hub graph shape and LoCoMo/LongMemEval quality. Constraints in [`00-scope-and-constraints.md`](00-scope-and-constraints.md) are binding. Extraction quality work remains owned by [`01-ingestion-extraction.md`](01-ingestion-extraction.md); this document owns **token and wall-clock** levers that must not regress quality.

**Quality bar:** accurate-mode event hubs; no McNemar-significant judge drop; Phase-0 extraction metrics non-regressing. `lightweight` is **out of scope**.

---

## Cost map (accurate mode)

| Rank | Stage | Lever shipped |
| --- | --- | --- |
| 1 | Architect full-text tool loop | Architect now runs **per Scout unit** (+ optional prior-unit window ≤4), not unbounded session text |
| 2 | Atomic Janitor per create | **Batched** post-Architect Janitor + cheap grounding precheck skips |
| 3 | Scout per 6k chunk | Unchanged (instrumented) |
| 4 | Observations | Gated by `RUN_OBSERVATIONS` (default **false**) — never wrote the graph |
| 5 | Consolidation | `RUN_GRAPH_CONSOLIDATOR` default **false** until audited (`01` Phase 6) |
| 6 | Hub/topic rebuild | Entity-scoped hub refresh; topic rebuild deferred to finalize |

---

## Phases

### C0 — Measure (shipped)

- `src/core/saving/ingest_cost.py` — per-stage ledger; contextvar accumulation from custom agent invoke loop
- Task status `cost` field via `set_ingestion_task_status(..., cost=...)`
- LoCoMo / LongMemEval ingest records + report mean tokens/unit and stage breakdown

### C1 — Waste removal (shipped)

| Knob | Default | Effect |
| --- | --- | --- |
| `RUN_OBSERVATIONS` | `false` | Skip ObservationsAgent LLM |
| `RUN_GRAPH_CONSOLIDATOR` | `false` | Skip consolidator LLM mutations |
| `JANITOR_BATCH_SIZE` | `20` | Relationships per Janitor call |

### C2 — Bound and batch (shipped)

- [`auto_kg.py`](../../src/core/saving/auto_kg.py): Scout chunks → Architect per unit with prior context; `reset=False` across chunks
- [`architect_agent.py`](../../src/core/agents/architect_agent.py): `defer_janitor=True`, `run_batched_janitor`
- Hub: `refresh_hub_bridges_for_entities`; topic rebuild on finalize only

### C3 — Gate remaining LLM (shipped)

- [`grounding.py`](../../src/core/saving/grounding.py): span align precheck before Janitor
- Ambiguous entity resolution routes to `KGAgent.verify_entity_existence` instead of silent duplicate

### Distillation (deferred)

Distill-SynthKG ([2410.16597](https://arxiv.org/abs/2410.16597)) only after `01` Phases 0–3 stabilize the teacher. Premature distillation freezes defects.

---

## Literature (access 2026-07-29)

| Paper | Role |
| --- | --- |
| ATOM [2510.22590](https://arxiv.org/abs/2510.22590) | Atomic units + parallel merge → latency cut |
| Distill-SynthKG [2410.16597](https://arxiv.org/abs/2410.16597) | Deferred student extractor |
| Zep/Graphiti [2501.13956](https://arxiv.org/abs/2501.13956) | Prior-context window; uncertainty-gated ER |
| UCCI [2605.18796](https://arxiv.org/abs/2605.18796) | Cascade routing at fixed quality |
| SafePassage [2510.00276](https://arxiv.org/abs/2510.00276) | Span grounding without LLM scorer |

---

## Verification

```bash
# Unit
pytest tests/test_ingest_cost.py tests/test_grounding.py -v

# After a live ingest, task status should include cost.stages
# Benchmark report.md Ingest section should show mean tokens when cost is present
```

Measured A/B (2026-07-30): `benchmarks/runs/ingest-cost-baseline.json` — conv-26 sessions 1–5.

| Arm | mean wait | mean LLM tokens | LLM÷source | type-named | events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy hotpath (`ingest-cost-legacy`) | 630 s | 1.19M | — | 0 | 107 |
| Cheap defaults (`ingest-cost-cheap2`) | 223 s | 0.73M | **~886×** | 0 | 100 |
| CP2 batch (`architect-cp2-batch`) | 41 s | 15.4k | **18.8×** | 0 | 60 |
| CP3 batch+scratchpad (`architect-cp3-scratchpad`) | 34 s | 12.6k | **15.3×** | 0 | 65 |

C0–C3 shipping: ≈**65%** lower mean wait, ≈**39%** fewer mean tokens vs legacy. Architect redesign: old cheap **~886× → batch+scratchpad ~15.3×** (same five sessions; source denom 4,114 `cl100k_base`). Details: [`architect-cp2-NOTES.md`](../../benchmarks/runs/architect-cp2-NOTES.md), [`architect-cp3-NOTES.md`](../../benchmarks/runs/architect-cp3-NOTES.md). No McNemar in these cost measurements.

---

## What not to do

- Do not use `PIPELINE_MODE=lightweight` as the cost lever
- Do not add reflection passes for cost goals
- Do not claim accuracy wins without paired A/B (noise floor in `00`)

**Follow-on:** Architect loop efficiency workstream and ship defaults (`batch` + `auto`/scratchpad) in [`11-architect-loop-efficiency-plan.md`](11-architect-loop-efficiency-plan.md).
