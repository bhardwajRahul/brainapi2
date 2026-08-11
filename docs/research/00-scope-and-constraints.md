# Research Scope and Confirmed Constraints

Confirmed with the maintainer on 2026-07-28. Every proposal in `01`-`05` is judged against this.

## Product goal

A memory layer that (a) answers accurately every time, (b) surfaces non-obvious connections via multi-hop, and (c) is a substrate for agents, recommenders, and search engines.

## Where quality hurts most today

1. **Multi-hop** — single-hop questions are acceptable; chained questions fail.
2. **Temporal** — superseded facts come back as current truth.

These two set the priority order. A proposal that improves single-hop precision while leaving these untouched ranks below one that moves either.

## Retrieval architecture: two tiers, one graph

The maintainer's intent for the empty ADRs 003/004 is **a router, not a winner**:

| Surface | Budget | Implication for proposals |
| --- | --- | --- |
| `/retrieve/context` (one-shot context API) | **Sub-second, fast and cheap** | No iterative LLM loops, no per-query LLM reranking, no multi-call decomposition on this path. Gains must come from indexing, precomputation, graph structure, or cheap learned rerankers. |
| Other REST endpoints + MCP tools | Deeper navigation allowed | Agents may search iteratively and in depth here. Iterative retrieve-then-reason, agentic traversal, and verification loops are admissible. |

Anything expensive must be either moved to write time or exposed only on the deep-navigation surface. "Make retrieval smarter with another LLM call" is not an acceptable answer for the context API.

## Workstreams

| Doc | Workstream |
| --- | --- |
| `01-ingestion-extraction.md` | Raw input to written graph: chunking, agent swarm, entity resolution, consolidation |
| `02-retrieval-multihop.md` | Query-time entry points, traversal, filtering, ranking, agentic navigation |
| `03-memory-substrate.md` | Data model, indexing, knowledge lifecycle, scaling |
| `04-evaluation-and-applications.md` | Measurement validity, failure attribution, recommendation/agent-surface metrics |
| `05-temporal-truth.md` | Cross-cutting: how the system knows what is currently true, write time through read time |
| `07-event-graphrag-improvement-plan.md` | Active event-hub GraphRAG execution plan (post Phase 1); see also `06` Phase 4 |
| `10-ingest-cost-latency.md` | Accurate-mode ingest token/latency reduction without quality trade-offs |
| `11-architect-loop-efficiency-plan.md` | Architect/Janitor redesign for peer-competitive ingest token multipliers |
| `12-locomo-93-at-low-multiplier.md` | Dual goal: ≥93% LoCoMo SOTA at ~15–20× ingest median (no escalate storms) |
| `13-locomo-93-research-levers.md` | Evidence-backed research levers for HyperMem-class ≥93% (harness-first) |
| `15-ecommerce-gnn-recsys-landscape.md` | External GNN/e-commerce RecSys curriculum & industry index; BrainAPI mapping |
| `16-recsys-eval-protocol.md` | Held-out next-item eval on demorecsys via structured ingest + /retrieve/recommend |

## Decisions already taken during research

- **Event-leg supersession is a bug, not a trade-off.** `_invalidate_superseded_relationships` (`src/workers/tasks/ingestion.py:465-505`) deprecates an actor's older `MADE` edges because the Architect is instructed to use generic predicate names, and `_is_currently_valid` (`src/services/api/controllers/retrieve.py:420-426`) then hides them from retrieval. The maintainer does not recall the original intent and has ruled it a bug; a minimal fix plus regression test is being shipped ahead of this roadmap. Proposals must assume the incorrect deprecation is gone, and must not assume existing brains have been repaired — no backfill was authorised.
- **The 3072-dim embedding default silently disables the pgvector HNSW index** (`src/lib/postgresql/vectors.py:62-63` returns no DDL above 2000 dims; `.env.example:43-47` ships 3072). The fix is deferred to this roadmap and must be decided by a recall-versus-latency measurement, not asserted. Candidates to evaluate: dimension truncation, `halfvec` indexing up to 4000 dims, and a loud warning when an index is skipped.

## Measurement noise floor — binding on every claimed improvement

`locomo-conv26-push75-c` and `locomo-conv26-push75-d` have **byte-identical manifests** (same brain, same config, same `deepseek-v4-flash` answer and judge models, `historical_limit=16`, `max_passages=16`, `max_facts=50`, `use_ppr=true`, `sufficiency_retry=true`) and yet score **82.9% and 86.2%**. That is a **3.3-point run-to-run spread with nothing changed.**

At `n_total=152` the 95% CI is roughly ±7 points, and the `push75-b` (75.7%, CI 68.3–81.8) and `push75-c` (82.9%, CI 76.1–88.1) intervals **overlap**. Consequences:

- The "+7.2 points from the push75-c config" is **not statistically established**. Do not quote it as a settled gain.
- The benchmark currently **cannot resolve differences below roughly 5–7 points**, which is larger than most individual improvements worth shipping. Any proposal justified by a single-run delta of a few points is unfalsifiable as measured today.
- **Evidence recall is the trustworthy signal** in these runs *for the passage channel*, because it is computed on retrieval output rather than through the judge: full session recall moved **82.7% → 97.3%** from `push75-b` to `push75-c`. Prefer retrieval-side metrics over judge accuracy when arguing for a change. **Caveat (post Phase 1):** graph-channel evidence recall is logged but **not yet trustworthy** — identical-config runs on the same brain agree on graph session recall for only ~39–42% of questions until fact selection is order-stable. Do not A/B on graph EvR until that defect is fixed. Passage recall remains deterministic.
- Latency and cost are also from `report.json` and are trustworthy: p50 retrieval **4053 ms → 4793 ms** and total LLM tokens **4.06M → 7.74M** for the same 152 questions.

## Latency reality vs the stated budget

The sub-second target for `/retrieve/context` is **not met yet**, but the dominant cost is gone. Pre-deletion measured retrieval p50 was **~4.8 s**; post-dossier-deletion live p50 is **~1.4 s** (clean-brain eval arms, after sync+restart), with no detectable accuracy loss. Remaining gap to budget ≈ **400 ms**, likely embedding fanout (Phase 3.2). Any roadmap must still treat "close the last ~400 ms" as first-class — and must state explicitly whether the sub-second target or an accuracy gain wins where they conflict.

## Measurement outcomes (Phase 1, live after sync+restart)

Clean brain `locomoconv26clean` (19/19 conv-26 sessions), de-fitted prompt, same config as `baseline-clean-a`. Full table in `06-roadmap.md`.

1. **Extraction fixes earn their keep on graph hygiene**, not on detectable judge accuracy at n=152 (0 deprecated EVENT-touching edges vs 219; 0 type-named placeholders vs 3; denser graph 537/752/410 vs 411/584/281).
2. **The graph channel does not move judge accuracy above the resolution floor** vs passages-only (McNemar ns). Graph EvR is logged (~53% point estimate) but cross-run agreement is only ~39–42% — measurable, not yet a measurement. The harness can see the channel; it is not yet converting into answers.
3. **Dossier deletion earns its keep on latency**: ~4.8 s → ~1.4 s p50, no accuracy loss. Sub-second not yet met.

**Still open in the measurement stack:** graph EvR non-determinism (Phase 0 follow-up); judge remains shared-family with the answerer (`deepseek-v4-flash`); groundedness remains unmeasured because the judge never sees retrieved context. Absolute accuracy claims stay unsupportable; paired A/B comparisons remain valid.

## Temporal semantics — decided by the maintainer

- **`valid_at` means document/ingest time.** This is correct as implemented; event time lives in `happened_at` only. Do not propose "fixing" that as a conflation.
- **`/retrieve/context` will default to current-truth-only**, with an opt-in mode for history. This is an accepted breaking change to what callers receive. Plans must account for the migration, and for the fact that the filter must be index-backed rather than LLM-mediated to fit the sub-second budget.
- **No targeted supersession benchmark for now.** The maintainer chose to prioritise fixing over measuring, with the explicit understanding that LoCoMo category 2 measures event *localization* ("when did X happen?") rather than supersession — it scores 0.86, above multi-hop — and that roughly 12 of 1,986 questions probe current truth. Consequence for the roadmap: temporal fixes are verifiable by unit test but their end-to-end quality effect is **not currently measurable**, so do not claim benchmark gains for them.

### Verified temporal defects (fixes authorised and in progress)

- **Relative-date resolution never runs.** `_DATE_INPUT_FORMATS` (`src/utils/dates.py:5-16`) has no time-bearing format, so the real `source_timestamp` from `benchmarks/data/locomo10.json` (`'1:56 pm on 8 May, 2023'`, passed through at `benchmarks/locomo/dataset.py:131`) yields `None` and `resolve_relative_date` returns every expression verbatim. Confirmed by execution. The resolver logic itself is correct against a parseable reference, so this is a format gap.
- **The recency scorer is dead.** `src/core/search/entity_info.py:230` calls `fromisoformat` on the `dd/mm/YYYY` that `src/utils/dates.py:76` writes; it always raises, a bare `except` swallows it, and `recency` is exactly `1.0` for every node — making the `recency * 0.2` term a constant across all candidates.
- **The current-truth filter guards only one of two evidence channels.** `_is_currently_valid` is called at `src/services/api/controllers/retrieve.py:787` on the triple channel only; the passage channel (`:667-725`) and `historical_context` (`:851-880`) have no temporal predicate, so an invalidated edge and the original sentence asserting it can be returned in the same payload. This is the seam that produces the reported symptom.

### Citation caveat

`2606.01435` (*Don't Ask the LLM to Track Freshness*) is verified real and correctly quoted on its headline result (+10.8 on FactConsolidation single-hop; Zep/Graphiti at 7%). **But its strong result is for `max(serial)` with explicit monotone serials.** The paper's own LongMemEval port to `max(timestamp)` — which is what BrainAPI would actually do — ties or slightly loses to LLM judgment (57.8% vs 64.4%, n=45). Cite this paper for the latency argument for a no-LLM query path, not as evidence that deterministic aggregation is more accurate.

## Maintainer decisions (2026-07-29) — Event GraphRAG track

Mirror of answers recorded in `07-event-graphrag-improvement-plan.md`. Full rationale lives there.

1. Context-path cross-event = write-time hub adjacency + cheap hub PPR; full iterative composition = deep/MCP only (ADR-006).
2. Primary metric = **stable graph EvR** (multi-hop / single-hop); judge accuracy secondary (may stay flat).
3. Graph EvR is a measurement only at **≥95%** identical graph-session sets across two identical-config runs.
4. Diversify under `max_facts` by **session + event-hub blend**; maintainer latency tax ceiling **2000 ms**, but ADR-006 still targets **p50 &lt; 1000 ms** — prefer both; escalate with numbers if tax exceeds headroom to 1 s.
5. Cross-hub path strings not in default `text_context`; structured `paths` for deep/MCP only.
6. Temporal conflict: prefer recency in ranking and return both as meta.
7. Leiden out; k-core / deterministic hierarchy deferred (open-domain later).
8. Query–edge embedding similarity allowed on context ranker (no LLM); A/B.
9. Diversification harness arms: both `max_facts=50` and `max_facts=40`.
10. Hub-bridge index owned by **memory substrate (`03`)** — Phase A is retrieval determinism only.

## Roadmap decisions (2026-07-28, after synthesis)

- **The conv-26-fitted answer prompt rules were deleted and the benchmark re-baselined** at 82.2% (`baseline-clean-a`). Old fitted numbers must stay labelled as such.
- **The passages-only ablation ran on the clean brain** (`locomoconv26clean`). Result: no LoCoMo judge gain above the resolution floor; see Measurement outcomes above. Phase 4 is gated on non-LoCoMo-accuracy goals (EvR lift, multi-hop composition, product surfaces).
- **Phase 0 (harness) and Phase 2 (cheap defects) completed** in parallel; Phase 1 measurement is complete.
- **The judge stays shared-family** (`deepseek-v4-flash`); a human labelling pass was declined for now, and the judge still never sees the retrieved context — so **groundedness remains unmeasured**, and no claim about faithfulness is currently supportable.

## Judge configuration — decided

**The judge stays `deepseek-v4-flash`, the same model as the answerer.** The harness now supports full separation (`BENCH_JUDGE_*`, with Azure `gpt-4o` auto-selected when credentials are visible) and records `judge_shares_answer_family: true` when it is not used, so the condition is at least declared in every manifest.

What this does and does not cost:

- **Paired comparisons remain valid.** When two arms are scored by the same judge on the same questions, self-preference bias applies to both and largely cancels. Since nearly all planned work is A/B — config arms, the passages-only ablation, before/after a fix — the headline is usable for the comparisons that matter.
- **Absolute claims are not supportable.** "BrainAPI achieves X%" carries unmeasured self-preference bias and no human calibration. Do not publish or compare the figure against externally reported systems.
- **Groundedness stays unmeasured** regardless of model choice, because the judge never receives the retrieved context. A prediction that is correct from the model's parametric knowledge scores identically to one derived from the graph — the exact distinction a memory product exists to make. This is the most significant remaining gap in the measurement stack.

## Harness default divergence — deferred

The harness CLI defaults `use_ppr` off while production now defaults it on, so a plain `evaluate` run does not measure production behaviour. Deliberately deferred until Phase 3 settles which features remain on the context path. Every arm must state its flags explicitly in the meantime; the manifest now records what was actually sent.

## Deliverable

A synthesized, ranked cross-workstream roadmap for review. Implementation does not start until that roadmap is approved — with the single exception of the supersession fix above, which the maintainer authorised immediately because it corrupts data on every ingest.
