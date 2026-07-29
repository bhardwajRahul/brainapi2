# ADR-006: Two-Tier Retrieval with a Sub-Second Context API

## Status

Accepted — supersedes ADR-005 on retrieval latency and on request defaults. ADR-005 remains the record of the hybrid one-shot design, which this ADR keeps.

## Date

2026-07-28

## Context

ADR-005 accepted rising latency as a consequence of larger passage budgets and optional PPR/sufficiency retry, recording "p50 retrieve ~5s on the upgraded LoCoMo run" as an acceptable cost of the accuracy gain. That trade is no longer acceptable as stated, for two reasons.

First, `/retrieve/context` is the surface that agents, recommenders, and search integrations call synchronously. A ~5 s p50 is not viable there regardless of accuracy. Measured from `benchmarks/runs/*/report.json`: p50 4053 ms at the pre-upgrade config and 4793 ms after, p95 up to 6333 ms, and a **minimum** observed latency of 2508 ms across 152 questions — so no question, however simple, is close to budget. The two best-scoring configurations are also the two slowest and the two most expensive (4.06M → 7.74M LLM tokens for the same question set).

Second, the accuracy justification for spending that latency is weaker than ADR-005 assumed, because the measurement it rested on could not see the graph channel at all. `benchmarks/locomo/evaluate.py` caps the logged `text_context` at 20,000 characters and the API writes passages into that field before graph facts; verified against `locomo-conv26-push75-d/answers.jsonl`, all 152 rows are exactly 20,000 characters, none contains a graph fact line, and no `triples` field is logged. Both `answerable_rate` and `evidence_session_recall_full` derive from that field, so every reported retrieval number describes the passage channel only. ADR-005 itself recorded the requirement to log full context; the cap defeated it.

Separately, ADR-005's statement that defaults keep `use_ppr` and `sufficiency_retry` off is no longer true of the code, and its stated reason — API compatibility — was undermined by a latent defect: the controller read these flags as `getattr(request, "use_ppr", True)` while the Pydantic model declared `False`, so the fallback was dead and the flags were unreachable for every HTTP caller. The code and the ADR disagreed with each other and with the intent.

## Decision

**1. Two tiers, one graph.**

| Surface | Budget | Admissible techniques |
| --- | --- | --- |
| `/retrieve/context` | **p50 < 1000 ms**, cheap | Index-backed lookups, precomputed structures, write-time work, feature-based ranking over already-computed signals |
| Other REST endpoints and MCP tools | Depth allowed | Iterative retrieve-then-reason, agentic traversal, LLM and cross-encoder reranking, verification loops |

No LLM call on the context path. Gains there must come from moving work to write time, from indexing, or from cheap ranking — not from spending more inference at query time.

**2. Relocate rather than optimise.** Measurement of serial depth indicates two stages cannot fit the budget at any per-operation latency: the sufficiency retry, which is by definition a second full retrieval pass, and the per-entity dossier walk, which issues one unbatched vector fetch per edge. Both move to the deep tier. Passage retrieval is flattened to a single query with batched embedding and concurrent dense/keyword search. What remains is roughly 6-8 serial I/O operations.

**3. The cheap insufficiency signal stays, as a signal.** The lexical sufficiency check remains on the context path, but it returns an explicit insufficiency indicator to the caller instead of triggering a retry. This closes the gap where the API silently returned weak context, and lets a caller escalate to the deep tier by choice.

**4. Defaults follow the budget.** `use_ppr` defaults to `True`: it is one Cypher call over the seed set plus in-process power iterations, no LLM, and it is the change credited with moving multi-hop full evidence-session recall from 62.5% to 96.9% on a deterministic retrieval metric. `sufficiency_retry` defaults to `False` because decision 2 removes it from this path; defaulting it on would institutionalise what this ADR exists to remove. The `getattr` fallbacks are deleted so the request schema is the single source of defaults and the two can no longer drift.

**5. Instrument before optimising.** The stage attribution above is derived from serial depth in code, not from measurement. Per-stage timing lands before any relocation work, and Phase 3 gates on measured p50 < 1000 ms with per-category evidence recall held.

## Alternatives Considered

### Keep one uniform retrieval path and accept ~5 s (status quo per ADR-005)
- Pros: no routing complexity; a single contract to reason about.
- Cons: unusable for synchronous agent, recommender, and search integrations, which are the stated product targets.
- Rejected: the accuracy that latency buys is real but is not worth making the primary surface unusable, particularly while the graph channel's contribution is unmeasured.

### Make agentic navigation the default and retire one-shot
- Pros: highest ceiling on hard multi-hop; handles composition across event hubs, which the current two-hop flow-key-constrained traversal cannot.
- Cons: roughly an order of magnitude more cost and latency variance; harder failure attribution; abandons the one-shot product contract.
- Rejected, consistent with ADR-005: with evidence coverage already high on the passage channel, an agentic loop would mostly re-retrieve what is already in the prompt. It remains the escalation tier.

### Optimise the existing stages in place
- Pros: no contract change; no relocation work.
- Cons: the dossier walk alone exceeds 1000 ms at an optimistic 5 ms per operation, and the sufficiency retry is a second full pass by construction. Neither is reachable by constant-factor improvement.
- Rejected: these must be deleted from the path, not tuned.

### Cross-encoder reranking on the context path
- Pros: strong precision gains reported in the literature; the current fact filter is a no-op, so ranking is pure vector distance.
- Cons: a hosted model is a network hop on a path with no headroom; in-process over ~50 candidates costs 50-200 ms, i.e. 5-20% of the entire budget, for a component that is not the bottleneck.
- Rejected for this tier; adopted as the deep-tier escalation. The context tier gets a linear or small-GBDT ranker over features already computed — vector distance, PPR score, recency, provenance overlap, variant strength.

## Consequences

- `/retrieve/context` gains an explicit insufficiency signal and loses the automatic retry. Callers relying on the retry must opt into the deep tier.
- The response may carry optional per-stage timings when instrumentation is enabled; this is additive and off by default.
- A router is required to decide which tier serves a query. Candidate signals: cross-event-hub composition, enumeration completeness, multi-entity comparison, and the insufficiency indicator. Entity resolution confidence must gate escalation — a beam-search agent on top of low-confidence resolution will pursue a wrong entity confidently and expensively.
- ADR-005's reported figures (~86% judge accuracy, ~97% evidence recall, multi-hop ~81%, temporal ~92%) are retained as history but must not be quoted as current quality. They were produced with an answer prompt containing conv-26 gold answer strings, on brains carrying wrongly-deprecated event edges, and with the graph channel absent from the logged context. A re-baseline is in progress.
- Benchmark arms must send `use_ppr` unconditionally. `benchmarks/locomo/client.py` omits the flag when false, so with the new server default an arm configured `use_ppr=false` would silently run with PPR and be unreproducible.
