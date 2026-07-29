# ADR-005: Hybrid One-Shot Context Retrieval over Agentic Navigation

## Status

Accepted — **partially superseded by ADR-006** on retrieval latency and request defaults. The hybrid one-shot design below stands; the accepted ~5s p50 and the "defaults keep `use_ppr` and `sufficiency_retry` off" statement in the Decision section no longer hold.

## Date

2026-07-28

## Context

ADR-003 keeps `/retrieve/context` as a single request/response. ADR-004 considers agentic multi-step graph navigation. On LoCoMo conv-26, early GraphRAG work reached ~75% judge accuracy, but truncated eval logs hid the real split of blame: with full instrumentation, answerable rate was already ~93% while judge accuracy lagged by ~20 points. Remaining misses were mostly incomplete lists, relative-date phrasing, open-domain abstentions, and under-budget multi-hop evidence—not a need for a general agent loop.

We needed a retrieval posture that raises evidence coverage and multi-hop recall without replacing the simple context API or re-ingesting brains for every experiment.

## Decision

Keep the one-shot `GetContext` API, and enrich a single call with hybrid retrieval:

1. **Passage budget** — raise default passage/historical limits so evidence-session recall is not budget-capped.
2. **Hybrid passage ranking** — vector + keyword retrieval fused with reciprocal rank fusion (RRF).
3. **Enumeration query decomposition** — for list/what/which questions, issue entity/relation sub-queries and union passage sets before fusion.
4. **Optional Personalized PageRank** over seed neighborhoods (`use_ppr`) when multi-hop still lags min-seed-distance ranking.
5. **Optional sufficiency retry** (`sufficiency_retry`) — one targeted follow-up retrieval when assembled context looks insufficient.
6. **Recall-preserving fact filter fallback** — without an LLM adapter, do not truncate candidates before score-based capping.
7. **Benchmark answerer prompts** — enumeration completeness, relative temporal phrasing, and hedged open-domain answers; leave `JUDGE_SYSTEM` unchanged so scores stay comparable.

Defaults keep `use_ppr` and `sufficiency_retry` off for API compatibility; the LoCoMo harness enables them when measuring the upgraded path.

## Alternatives Considered

### Full agentic navigation (ADR-004)
- Pros: Can chase missing evidence across many hops; flexible for open-ended exploration.
- Cons: Higher latency/cost variance; harder to attribute failures; changes the product contract away from one-shot context.
- Rejected for now: LoCoMo gaps closed enough with one sufficiency pass and better budgets; agent loop not required for the measured failures.

### Prompt-only answerer changes
- Pros: Cheap, no API/runtime coupling.
- Cons: Cannot fix gold that never reaches context; alone left multi-hop weak.
- Rejected as sole strategy: used as Phase B, then combined with retrieval upgrades.

### Changing the judge prompt
- Pros: Would inflate headline accuracy on near-synonyms/supersets.
- Cons: Breaks comparability with v5/v6 and hides real system quality.
- Rejected: judge stays fixed; soft metrics (answerable rate, evidence-session recall, answerer gap) diagnose loss.

### Community-level GraphRAG summarization
- Pros: Better global sense-making for open-domain themes.
- Cons: Failures were local list completeness and temporal phrasing more than global summaries.
- Rejected for this increment.

## Consequences

- `/retrieve/context` remains one-shot but accepts richer options (`max_passages`, `max_facts`, `use_ppr`, `sufficiency_retry`).
- Eval must log full `source_passages` / `historical_context` / `retrieved_session_ids`; truncated context logs misdiagnose retrieval.
- Latency and token cost rise with larger passage budgets and optional PPR/sufficiency (p50 retrieve ~5s on the upgraded LoCoMo run).
- On LoCoMo conv-26 with the upgraded path + answerer prompts: ~86% judge accuracy, ~97% full evidence-session recall, multi-hop ~81%, temporal ~92%.
- Agentic navigation (ADR-004) stays open for workloads where one sufficiency retry is not enough; it is not the default retrieval model.
