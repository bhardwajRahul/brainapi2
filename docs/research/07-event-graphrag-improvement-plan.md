# 07 — Event-centric GraphRAG improvement plan

Workstream: retrieval quality via GraphRAG techniques that respect the **5-node event path**. Constraints in `00-scope-and-constraints.md` and ADR-006 are binding. Phase 1 results in `06-roadmap.md` are not re-litigated.

**Plan mode only** — no `src/` or `benchmarks/` changes in this document's delivery. Tasks below are ordered for a future implementer.

Skill: planning-and-task-breakdown (vertical slices, S/M, acceptance criteria, verification, checkpoints).

---

## What this workstream does

### Product intent

Return, under the two-tier budget, evidence that preserves the Triangle of Attribution so a consumer can answer single-hop questions from one event hub and compose multi-hop answers across hubs — without flattening events into binary triples.

### The 5-node fact (non-negotiable)

README models every action as an **Event Hub**:

```
(subject)-[:MADE-family]->(Event)-[:TARGETED-family]->(object)
                         \-[:OCCURRED_WITHIN]->(context)
```

In code this is the event-centric 5-tuple `(n, r, m, r2, b)` tied by `flow_key` on both legs.

### Traced one-shot path (`/retrieve/context`)

| Stage | What happens | Anchors |
| --- | --- | --- |
| Entry | `get_context` → `_build_context`; spaCy elements → query variants | `retrieve.py:789-822`, `:427-458` |
| Seeds | Per variant: embed → ANN nodes (`k=_SEED_K`) + ANN relationships → seed UUIDs | `retrieve.py:606-646`, `:831-845` |
| Event expand | `get_event_centric_neighbors(seed_uuids)`: Cypher `MATCH (n)-[r]-(m)-[r2]-(b) WHERE … AND r2.flow_key = r.flow_key` | `retrieve.py:849-851`; `neo4j/client.py:1654-1681` |
| Validity | Drop candidate if either leg fails `_is_currently_valid` | `retrieve.py:855-857`, `:418-424` |
| Score | Min vector distance over `{n,m,b}` among seeds; dedupe by `(r.uuid, r2.uuid)` | `retrieve.py:858-875`, `:414-415`, `:969-977` |
| Optional PPR | Adjacency from another `get_event_centric_neighbors` over seeds; clique edges among `{n,m,b}`; reorder by `-max(PPR)` | `retrieve.py:526-549`, `:981-1002`; `fact_filter.py:81-115` |
| Cap | `curated = ranked[:max_facts]` (LLM filter only if deep tier supplies adapter) | `retrieve.py:1004-1021`; docstring `:797-799` |
| Render | Path-shaped string via `" \| ".join(...)`; session tags from provenance chunks; passages then facts in `text_context` | `retrieve.py:410-411`, `:692-720`, `:1137-1139` |
| Passages | Separate hybrid RRF channel; no graph edges | `retrieve.py:723-786`, `:882-922` |

NetworkX backend mirrors the same flow-key path (`networkx_client.py:1069-1090`).

### What is configurable vs hardcoded

| Knob | Where | Notes |
| --- | --- | --- |
| `max_facts` | `requests.py:396` default **40**; harness often 50 | Binding budget on a fragmented clean graph |
| `use_ppr` | `requests.py:399` default **True** | Replaces distance order; no blend |
| `apply_fact_filter` | schema True; no-op without adapter | Context path intentionally empty |
| `_PPR_DAMPING` / `_PPR_ITERS` | `retrieve.py:51-52` | Hardcoded 0.85 / 20 |
| Flow-key equality | Cypher only | Hardcoded; no cross-event hop |

### Silently dropped / structurally impossible today

- **Cross-event composition on the one-shot path.** `flow_key` equality keeps both hops inside one hub (`neo4j/client.py:1670-1672`). Multi-hop across events is bag-of-facts composition by the answerer, not path retrieval.
- **PPR cannot leave the seed neighbourhood.** Adjacency is rebuilt from the same event-centric call over seeds only (`retrieve.py:526-549`).
- **PPR ignores edge types.** Transitions are uniform over the three event nodes; MADE vs TARGETED vs OCCURRED_WITHIN are not distinguished.
- **Validity filter skipped when building PPR adjacency** (`retrieve.py:538-549` vs filter at `:855-857`) — structural signal and returned set can disagree.
- **No session / hub diversification under `max_facts`.** Truncation is pure score order (`:1021`).
- **Sort ties have no uuid tiebreak** (`:977`, `:1000-1002`) while candidates arrive from concurrent threads (`:961-967`) → **order-unstable fact sets** and non-deterministic graph EvR (~39–42% cross-run agreement per `06`).

Deep / MCP tier: `traverse_graph` (depth ≤5) and agentic tools exist separately; they do not share ranking with `get_context` (see `02-retrieval-multihop.md`).

---

## Guarantees and where they break

**Intended guarantee:** given a question, return a compact, **reproducible** set of 5-node event facts (plus passages) that covers the sessions needed to answer — including single-hub lookups and cross-hub composition — within ADR-006 budgets.

Ranked by impact on accuracy / multi-hop recall:

| # | Break | Kind | Why it hurts |
| --- | --- | --- | --- |
| **G1** | Graph EvR is not a measurement | Bug | Concurrent candidate merge + score-only sorts → ~40% cross-run agreement. Cannot A/B graph work until fixed. |
| **G2** | Multi-hop across events impossible on context path | Deliberate trade-off (now the accuracy ceiling) | Flow-key 2-hop is intra-hub only. Product priority #1 in `00`. |
| **G3** | Fixed `max_facts` under-covers fragmented clean graph | Gap | Clean brain denser (410 events vs 281); score truncation collapses to few sessions → ~53% provisional graph EvR. |
| **G4** | Graph facts do not move LoCoMo judge accuracy above floor | Measured trade-off | Phase 1: passages-only ≈ full graph (McNemar ns). Risk: more graph facts that still do not change answers. |
| **G5** | Single-hop via hubs is incomplete under budget | Gap | Seeds may hit object/context without ranking the full triangle highly; no hub-level scoring or path completeness check. |
| **G6** | PPR adjacency is seed-local, type-blind, validity-inconsistent | Gap | Weak multi-hop signal; CatRAG-style hub drift risk on EVENT / OCCURRED_WITHIN nodes. |
| **G7** | Context p50 ~1.4 s vs &lt;1 s budget | Gap | Remaining ~400 ms likely embedding fanout (Phase 3.2). Graph work must not blow the budget; expensive composition → deep tier or write-time. |
| **G8** | Flat prompt of facts, no cross-event path objects | Gap | PathRAG diagnosis: redundancy + flat organisation; BrainAPI already linearises one hub but not chains of hubs. |

---

## Maintainer decisions (2026-07-29)

Answers to the open questions below. Binding for Phases A–D.

1. **Cross-event on context path** = write-time entity-mediated hub adjacency + cheap PPR over hubs; full iterative composition = deep/MCP only (ADR-006).
2. **Primary metric** = **stable graph EvR** (esp. multi-hop / single-hop); judge accuracy is a secondary paired check and **may stay flat**.
3. Graph EvR is a measurement only when **≥95%** of questions have identical graph-session sets across two identical-config runs.
4. Diversification under `max_facts`: implement a **blend of session + event-hub coverage**. Maintainer max latency tax on the context path: **2000 ms**. **Tension with ADR-006:** the accepted context budget remains **p50 &lt; 1000 ms**. Do not silently ignore either number — prefer meeting both if possible; if the diversification tax would exceed remaining headroom to 1 s, escalate with measured numbers rather than quietly spending the 2000 ms allowance.
5. Path-shaped **cross-hub** strings: **not** in `text_context` by default; structured `paths` field for deep/MCP only.
6. Temporal conflict: **prefer recency** in ranking **and return both** as meta information.
7. **Leiden out** for now; **k-core / deterministic hierarchy still interesting later** for open-domain.
8. Context-path ranker **may use query–edge embedding similarity** (no LLM); evaluate A/B.
9. Harness: evaluate both `max_facts=50` and `max_facts=40` arms when measuring diversification.
10. Write-time event-adjacency / hub-bridge index is owned by **memory substrate (`03`)** — Phase A does not build it; only retrieval determinism.

---

## Open questions for the maintainer

*Answered 2026-07-29 — see Maintainer decisions above.*

1. Should cross-event composition on the **context** path be limited to a write-time **entity-mediated hub adjacency** (cheap PPR over hubs), with full iterative composition reserved for the deep tier per ADR-006?
2. Is the primary success metric for this track **stable graph EvR** (especially multi-hop / single-hop categories), with judge accuracy as a secondary paired check — accepting Phase 1's finding that judge may stay flat?
3. What minimum **cross-run graph-session agreement** (e.g. ≥95% of questions identical on two identical-config runs) gates treating graph EvR as a measurement?
4. For diversification under `max_facts`, prefer **session coverage**, **distinct event-hub coverage**, or a blend — and what is the max latency tax (ms) you will accept on the context path?
5. Should path-shaped **cross-hub** strings enter `text_context` by default, or only a structured `paths` field for deep/MCP consumers?
6. When two hubs share an entity but disagree temporally, should context-path ranking prefer recency, return both, or defer to the current-truth filter only?
7. Is Leiden / community-summary GraphRAG explicitly out of scope for this brain (sparse event graph), or is a deterministic hierarchy (e.g. k-core) still interesting for open-domain sensemaking later?
8. May the context-path ranker use **query–edge embedding similarity** (no LLM) for CatRAG-style weights, or must all query-dependent weights stay at write time / deep tier?
9. For harness A/B, is `max_facts=50` the locked comparison budget, or should we also report a diversification arm at `max_facts=40` (schema default)?
10. Who owns the write-time event-adjacency / hub-bridge index — retrieval (`02`) or memory substrate (`03`)?

---

## Frontier techniques

Each technique judged against: (1) 5-node event model, (2) sub-second context budget / ADR-006, (3) Phase 1 finding that naive graph inclusion did not lift LoCoMo judge accuracy.

### A. HippoRAG / HippoRAG 2 — PPR over a memory graph

**Mechanism.** Offline open KG + passages; online seed entities; Personalized PageRank surfaces multi-hop-relevant nodes/passages in one shot.

**arXiv.** `2405.14831` (HippoRAG), `2502.14802` (HippoRAG 2).

**Reported gain.** Up to +20% on multi-hop QA vs prior SOTA; single-step matches/beats IRCoT at 10–30× lower cost and 6–13× lower latency. HippoRAG 2: +7% associative memory vs best embedding model without factual regression.

**Cost.** Offline graph build; online PPR cheap if adjacency is local/precomputed.

**Fit.** BrainAPI already runs PPR (`fact_filter.py:81-115`) but over a **clique of three nodes inside each flow_key**, not an entity–passage graph spanning hubs. Passages are a separate RRF channel.

**Verdict: adapt.** Keep PPR on the context path; rebuild adjacency as **hub-aware** (event nodes + entity bridges across `flow_key`s) at write time. Do not flatten events to OpenIE triples.

### B. CatRAG — query-aware traversal / anti-hub-drift

**Mechanism.** Attacks static transition matrices: symbolic anchoring, query-aware edge weights, key-fact passage bias so walks complete evidence chains instead of sinking into hubs.

**arXiv.** `2602.01965`.

**Reported gain.** Consistent wins on four multi-hop benchmarks; larger gains on *reasoning completeness* (full chain) than on partial recall.

**Cost.** Paper uses LLM edge scoring (deep-tier). Embedding-only reweight of outgoing legs can be context-tier.

**Fit.** EVENT hubs and OCCURRED_WITHIN contexts are natural sinks; uniform clique PPR matches the failure mode. `flow_key` is a ready symbolic anchor for *intra*-hub integrity.

**Verdict: adapt.** Context-tier: type-aware / query–embedding edge weights + symbolic seed anchors, no LLM. LLM edge scoring → deep tier only.

### C. PathRAG — relational paths + flow pruning + path prompts

**Mechanism.** Retrieve paths between query nodes; flow-based prune; prompt with path text (not flat bags).

**arXiv.** `2502.14902`.

**Reported gain.** Beats GraphRAG/LightRAG-style baselines across six datasets / five dimensions; average win rates ~58–60% vs those systems.

**Cost.** Path enumeration can be heavy online; prompting gets smaller.

**Fit.** Intra-hub path formatting already exists (`_format_event_fact`). Missing piece is **paths that chain hubs** via shared entities. Online multi-pair enumeration risks the latency budget.

**Verdict: adapt — split by tier.** Context: path-shaped rendering + cheap 1-bridge hub chains from a precomputed index. Deep: flow pruning / beam over `traverse_graph`.

### D. REMem — episodic memory graph + agentic retriever

**Mechanism.** Offline time-aware gist/fact hybrid graph; online agentic tools for iterative episodic reasoning.

**arXiv.** `2602.13530`.

**Reported gain.** +3.4 recollection / +13.4 episodic *reasoning* over Mem0 and HippoRAG 2; stronger refusal on unanswerable questions.

**Cost.** Agentic inference at query time; offline gists.

**Fit.** Closest to BrainAPI's event hubs. Phase 1 showed one-shot graph bag does not buy LoCoMo judge points — REMem's gain is on the *reasoning* half.

**Verdict: adopt as architectural reference.** Gists / indexes = write-time (context-admissible). Agentic episodic tools = deep tier only.

### E. Zep / Graphiti — temporal KG memory

**Mechanism.** Temporally-aware KG with validity; synthesises conversational + structured data.

**arXiv.** `2501.13956`.

**Reported gain.** DMR 94.8% vs 93.4%; up to +18.5% on LongMemEval with ~90% lower latency vs baselines.

**Fit.** Validity already filtered on triples; ranking still barely temporal. Crosses into `05-temporal-truth.md`.

**Verdict: adapt (ranking/filter features only).** Not a substitute for cross-event path retrieval.

### F. KAG — logical-form hybrid reasoning

**Mechanism.** LLM-friendly KG representation, chunk↔graph mutual index, logical-form-guided hybrid reasoning.

**arXiv.** `2409.13731`.

**Reported gain.** Relative F1 +19.6% (2Wiki) / +33.5% (HotpotQA) vs prior RAG in their setup.

**Cost.** Logical-form engine and alignment stack; multi-step.

**Fit.** Mutual indexing matches provenance goals; logical forms are iterative → deep tier. Must not collapse n-ary events to binary SPO.

**Verdict: adapt selectively.** Mutual index / provenance: yes. Full logical-form engine: deep tier; reject as context-path default.

### G. Leiden / community GraphRAG — reject for this graph

**Mechanism.** Community detection + hierarchical summaries for global sensemaking (`2404.16130` GraphRAG family).

**Counter-arXiv.** `2603.05207` proves modularity on sparse, low-degree graphs admits exponentially many near-optimal partitions → Leiden communities are **inherently non-reproducible**; proposes deterministic k-core hierarchies instead.

**Fit.** Clean event graph is sparse and hub-centric. Leiden fights G1 (determinism) and the 5-node model.

**Verdict: reject Leiden communities.** Optionally revisit **k-core** later for open-domain sensemaking only — out of critical path for multi-hop event QA.

### H. Budget-aware diversification (MMR / coverage)

**Mechanism.** Maximal Marginal Relevance and coverage objectives select items that are relevant *and* non-redundant under a hard budget (classic IR; widely reused in RAG pruning).

**arXiv / canon.** Carbonell & Goldstein MMR (SIGIR 1998) is the reference mechanism; recent graph-RAG work (PathRAG `2502.14902`) restates the redundancy diagnosis empirically.

**Reported gain.** Classic: better coverage at fixed k; PathRAG: quality↑ when redundancy↓.

**Cost.** O(k · n) over candidates; microseconds–low ms at `max_facts≤50`.

**Fit.** Direct answer to G3: diversify by **source session** and/or **event hub uuid** under `max_facts`.

**Verdict: adopt — context-tier.** Highest gain/$ after determinism.

### I. Deterministic ranking / stable ordering

**Mechanism.** Total orders with explicit tie-breaks; avoid concurrent unordered aggregation; optional ORDER BY in graph queries.

**Fit.** Root cause of G1 in this codebase (not a fashion paper). Concurrent `candidates.extend` + `sorted(key=score)` without secondary keys.

**Verdict: adopt immediately.** Prerequisite to every graph metric claim.

### J. Hybrid passage+graph fusion with channel attribution

**Mechanism.** Keep channels separate in logs/metrics; fuse only at answer time with explicit credit (Phase 0 already logs channels).

**Fit.** Phase 1: passages carry ~97% EvR and all detectable judge accuracy. Graph must earn keep on **stable graph EvR** and hard multi-hop misses, not by drowning the prompt.

**Verdict: adopt measurement discipline.** Prefer RRF/feature fusion that can ablate channels; reject “more facts into `text_context`” without paired McNemar + stable EvR.

### K. ToG / IRCoT-style iterative KG agents

**Mechanism.** LLM-guided beam / retrieve-then-reason loops.

**arXiv.** `2307.07697` (ToG); IRCoT family cited via HippoRAG comparisons in `2405.14831`.

**Verdict: adopt deep-tier only; reject on context path** (ADR-006).

### L. Hyper-relational / n-ary KG embeddings (HyNT, HEHRGNN, …)

**Mechanism.** Qualifier-aware or hyperedge embeddings for link prediction.

**arXiv.** e.g. `2305.18256`, `2602.18897`.

**Verdict: reject for now.** Training/serving cost; does not fix query-time hub composition or determinism; wrong layer vs ADR-006.

---

## Implementation plan

### Architecture decisions (locked for this plan)

1. **Determinism before lift.** No graph EvR A/B until identical-config agreement clears the gate (**≥95%** questions with identical graph-session sets — Q3 decided 2026-07-29).
2. **Event hubs stay first-class.** Never project to binary SPO for retrieval primary keys; diversify and traverse in hub/`flow_key` units.
3. **Context vs deep (ADR-006).** Context: indexes, precompute, cheap rank/diversify, ≤1 bridge hub composition from materialised adjacency. Deep: iterative composition, LLM/CatRAG edge scoring, ToG-style beams.
4. **Success metrics (binding).** On `locomoconv26clean`, de-fitted prompt, same flags as clean-brain eval arms:
   - **Primary:** stable graph EvR overall + per-category (multi-hop, single-hop); passage EvR must not regress.
   - **Secondary:** paired McNemar judge accuracy (≥2 runs/arm).
   - **Guardrail:** retrieve p50 (and stage timings); ADR-006 targets p50 &lt; 1000 ms. Diversification latency tax ceiling is 2000 ms per maintainer — prefer both; escalate with numbers if tax exceeds remaining headroom to 1 s.
5. **Leiden out.** Per `2603.05207`. Hub-bridge index owned by memory substrate (`03`).
6. **Phase A scope.** Retrieval determinism only (total order, concurrent merge, harness stability). No write-time hub index, no MMR/diversification, no typed PPR.

### Measurement protocol (every phase)

```bash
# Identical-config stability (after Phase A)
./locomo.sh evaluate --sample conv-26 --brain locomoconv26clean \
  --run graph-stable-a --historical-limit 16 --max-passages 16 \
  --max-facts 50 --use-ppr --no-sufficiency-retry
./locomo.sh evaluate --sample conv-26 --brain locomoconv26clean \
  --run graph-stable-b ... # same flags
./locomo.sh compare --baseline graph-stable-a --candidate graph-stable-b

# Lift A/B (after Phase B+) — always 2 runs/arm for judge; 1 enough for passage EvR
./locomo.sh compare --baseline <control> --candidate <treatment>
```

Report: per-category judge accuracy, **graph** EvR full/partial, **passage** EvR, p50 ms, McNemar flip table. Do not claim judge wins below ~5–7 pts without paired significance (`00`).

---

### Phase A — Make graph metrics trustworthy

#### Task A.1 — Total order for fact ranking

**Description:** Add deterministic secondary keys to every sort that feeds `curated`: e.g. `(score, ppr_score, flow_key, r.uuid, r2.uuid)` (direction depends on ascending/descending). Ensure PPR reorder uses the same total order.

**Acceptance criteria:**
- [ ] Two sequential in-process `get_context` calls with fixed mock seeds/neighbors return byte-identical `triples` UUIDs and order.
- [ ] Unit test covers equal primary scores.

**Verification:**
- [ ] `pytest tests/ -k "fact_order or context_max_facts or deterministic"` (new test file OK).

**Dependencies:** None  
**Files likely touched:** `src/services/api/controllers/retrieve.py`, `tests/test_context_*determinism*.py`  
**Estimated scope:** S

#### Task A.2 — Deterministic candidate aggregation under concurrency

**Description:** Stop relying on race order from `asyncio.gather` + `candidates.extend`. Collect per-variant lists, then merge with a deterministic reduce (sort variants, then facts by total order before dedupe). Optionally `ORDER BY` in Cypher for neighbor rows.

**Acceptance criteria:**
- [ ] Under thread pool stress (e.g. 20 parallel identical requests), set of `(r.uuid,r2.uuid)` in top `max_facts` is identical across requests.
- [ ] Neo4j/NetworkX backends agree on membership for the same seeds (order may still be normalised in Python).

**Verification:**
- [ ] Stress unit/integration test; two harness runs `graph-stable-a/b` show graph-session agreement ≥ gate (Q3).

**Dependencies:** A.1  
**Files likely touched:** `retrieve.py`, optionally `neo4j/client.py`  
**Estimated scope:** M

#### Task A.3 — Harness stability report for graph channel

**Description:** Compare tool (or extend `compare`) emits per-question graph-session agreement rate and lists disagreeing `qa_index`s. Document gate in `CHECKPOINT_NOTES.md` style comment in compare output.

**Acceptance criteria:**
- [ ] One command prints % questions with identical graph session-recall boolean and identical retrieved graph session sets.
- [ ] README/research note: “do not A/B graph EvR below gate.”

**Verification:**
- [ ] Run on pre-fix vs post-A.2 artifacts; agreement rises past gate after A.2.

**Dependencies:** A.2 for the fix; can implement reporter in parallel with A.1  
**Files likely touched:** `benchmarks/locomo/report.py` or `compare` path only  
**Estimated scope:** S

### Checkpoint A

- [ ] Identical-config graph agreement **≥95%** questions identical session sets (maintainer gate, 2026-07-29).
- [ ] Passage EvR still bit-identical across runs.
- [ ] **Stop:** no diversification or composition work until this passes.
- [ ] `compare` prints graph-session agreement rate and disagreeing `qa_index`s; do not A/B graph EvR below the gate.

---

### Phase B — Raise stable graph EvR (single-hop + multi-hop) without blowing latency

#### Task B.1 — Hub- and session-aware diversification under `max_facts`

**Description:** Replace pure `ranked[:max_facts]` with MMR/coverage: relevance = distance/PPR score; redundancy = shared `flow_key` / event hub uuid / source session. Keep 5-tuples intact.

**Acceptance criteria:**
- [ ] At fixed `max_facts=50`, distinct `graph_session_ids` and distinct hub ids in curated set both increase vs control on a frozen candidate list fixture.
- [ ] Stable under A.1 total order.
- [ ] Added latency &lt; 5 ms at 200 candidates (unit benchmark).

**Verification:**
- [ ] Two-run arm vs control on `locomoconv26clean`: **stable** graph EvR ↑ on multi-hop and single-hop; passage EvR within noise; p50 delta &lt; 50 ms.
- [ ] McNemar recorded (expect possibly ns — still ship if EvR↑).

**Dependencies:** Checkpoint A  
**Files likely touched:** `retrieve.py`, `fact_filter.py` (or new small helper), tests  
**Estimated scope:** M

#### Task B.2 — Intra-hub completeness scoring (single-hop via event hubs)

**Description:** When a seed hits any of `{n,m,b}`, score the **whole** 5-tuple using a hub completeness feature (both legs present, MADE/TARGETED polarity, OCCURRED_WITHIN as context bonus not hub replacement). Prefer complete triangles over partial/noisy paths.

**Acceptance criteria:**
- [ ] Fixture: seed on object alone still surfaces the actor–event–object fact when present in neighbors.
- [ ] No binary triple projection.

**Verification:**
- [ ] Unit tests on synthetic 5-tuples; category single-hop stable graph EvR non-decreasing on clean brain.

**Dependencies:** A.1  
**Files likely touched:** `retrieve.py`, tests  
**Estimated scope:** S

#### Task B.3 — Type-aware PPR transitions (CatRAG-lite, no LLM)

**Description:** Replace uniform clique edges with typed weights (e.g. higher mass along MADE/TARGETED spine than OCCURRED_WITHIN; optional query–predicate embedding cosine). Apply `_is_currently_valid` when building adjacency so PPR and facts agree.

**Acceptance criteria:**
- [ ] Adjacency builder skips invalid legs.
- [ ] Ablation: typed vs uniform on frozen graph improves multi-hop graph EvR without p50 &gt; +30 ms.

**Verification:**
- [ ] Unit tests for validity + weights; clean-brain eval arm.

**Dependencies:** A.2, ideally B.1  
**Files likely touched:** `retrieve.py`, `fact_filter.py`, tests  
**Estimated scope:** M

#### Task B.4 — Path-shaped prompt hygiene (single hub)

**Description:** Make `_format_event_fact` explicitly label roles (Actor / Event / Target / Context) without adding LLM calls; ensure answerer sees structure PathRAG argues for. Keep backward-compatible plain line or gate behind flag.

**Acceptance criteria:**
- [ ] Structured path string round-trips hub roles in unit test.
- [ ] Token delta measured; no passage EvR regression.

**Verification:**
- [ ] Optional one-run prompt A/B; decide on retrieval+judge protocol.

**Dependencies:** None (can parallelise after A)  
**Files likely touched:** `retrieve.py`, possibly harness prompt only if needed  
**Estimated scope:** S

### Checkpoint B — **FAILED primary EvR lift (2026-07-29)**

**Shipped (code):**
- Hub/session MMR diversification under `max_facts` (`_diversify_facts`)
- Intra-hub completeness scoring
- Typed / validity-consistent PPR adjacency
- Path-shaped single-hub format (`Actor: … | Event: … | Target/Context: …`)
- Optional `temporal_conflicts` meta (prefer recency in rank, keep both as meta)

**Live on `locomoconv26clean` (retrieve-only arms `phase-b-ret-mf50-a/b`, `phase-b-ret-mf40-a`):**
- Graph-session stability **99.3%** — Checkpoint A gate **held**
- Graph EvR **flat 47.3%** vs `graph-stable-a` (same overall; cat1 multi-hop **25%**, cat4 single-hop **57%**) — Checkpoint B EvR lift **NOT met**
- Concurrent retrieve p50 ~2.2 s under 4 workers; sequential/harness-comparable ~**948 ms** (−404 ms vs ~1351 ms `graph-stable-a` baseline) — latency tax of diversification itself is negligible

**Conclusion:** re-ranking / diversifying within the same intra-hub candidate pool cannot raise session coverage. Need **cross-event candidate expansion** (Phase C hub-bridge index). Proceed to Phase C without requiring judge McNemar on Phase B.

Checkpoint B checklist status:
- [x] Code shipped (MMR, completeness, typed PPR, path format, temporal_conflicts)
- [ ] Stable graph EvR materially above control — **failed** (flat 47.3%)
- [x] Multi-hop / single-hop reported (25% / 57%); no concentrated lift
- [x] Passage EvR not targeted; stability gate held
- [ ] Judge McNemar — deferred (EvR flat → composition first)

---

### Phase C — Cross-event composition (tiered)

#### Task C.1 — Write-time entity-mediated hub bridge index

**Description:** At ingest (or rebuild job), materialise edges between **event hubs** that share a non-EVENT entity (actor/object/context), with weight = co-occurrence / recency. This is the context-tier substrate for cross-`flow_key` PPR without online multi-hop Cypher.

**Acceptance criteria:**
- [ ] Index maps `event_uuid → [(neighbor_event_uuid, shared_entity, weight)]`.
- [ ] Rebuild deterministic for a fixed brain.
- [ ] Document ownership (open Q10).

**Verification:**
- [ ] Unit test on a 2-hub diamond fixture; rebuild idempotent.

**Dependencies:** Checkpoint A; substrate coordination  
**Files likely touched:** ingestion/write path (`03`/`01`), thin read API in graph adapter  
**Estimated scope:** M

#### Task C.2 — Context-path 1-bridge expansion via hub index

**Description:** After seed hubs from `get_event_centric_neighbors`, pull top-K neighboring hubs from C.1 (K small, e.g. 2–5), expand their 5-tuples, merge into candidates, then B.1 diversify. Hard cap on extra Cypher/index lookups to protect p50.

**Acceptance criteria:**
- [ ] At least one fixture question requires two hubs; curated set contains both after expansion.
- [ ] Stage timing: expansion &lt; 50 ms p50 on conv-26-scale brain.
- [ ] Config flag `cross_event_bridges: 0|K` default conservative.

**Verification:**
- [ ] Clean-brain arm: multi-hop **stable** graph EvR ↑; p50 within budget headroom; McNemar vs Checkpoint B best arm.

**Dependencies:** C.1, B.1  
**Files likely touched:** `retrieve.py`, requests schema, tests  
**Estimated scope:** M

#### Task C.3 — Deep-tier cross-event composition route

**Description:** MCP / deep REST: ToG-style bounded beam or REMem-like tools over hub bridges + `traverse_graph`, with step cap and path isolation. Context path emits insufficiency / `needs_composition` signal (ADR-006) instead of silent weak context.

**Acceptance criteria:**
- [ ] Terminates within N steps on 50-question sample.
- [ ] Returns explicit paths across ≥2 hubs.
- [ ] Not invoked from default `/retrieve/context`.

**Verification:**
- [ ] Deep-tier eval bucket (multi-hop failures of context path); paired metrics; separate latency report.

**Dependencies:** C.1; ADR-006 router signals  
**Files likely touched:** MCP tools, optional new deep retrieve controller  
**Estimated scope:** M

#### Task C.4 — Channel-attributed fusion report

**Description:** For each eval arm, report questions where gold sessions appear in graph-only, passages-only, both, neither — so graph “earns keep” is visible even when judge is flat.

**Acceptance criteria:**
- [ ] Report table in `report.md` / compare output.
- [ ] Used in Checkpoint C go/no-go.

**Verification:**
- [ ] Recompute on Phase 1 artifacts for sanity.

**Dependencies:** Phase 0 logging (done)  
**Files likely touched:** `benchmarks/locomo/report.py`, `metrics.py`  
**Estimated scope:** S

### Checkpoint C — **primary EvR lift NOT met (2026-07-29)**

**Shipped (code):**
- Write-time / rebuild entity-mediated hub-bridge index (`kg_hub_bridges` on NetworkX/Postgres; `:HUB_BRIDGE` on Neo4j)
- Context-path ≤1-bridge expansion (`cross_event_bridges`, default 3; seed-hub cap 12) then Phase B diversify
- Additive structured `paths` on `GetContextResponse` (not dumped into `text_context`)
- Backfill on `locomoconv26clean`: **13 660** bridges (no re-ingest)

**Live retrieve-only (`phase-c-ret-mf50-a/b`, sequential):**
- Stability **98.7%** — gate held
- Graph EvR **47.3–48.0%** vs phase-b / graph-stable-a **47.3%** — Δ within 1/150 questions (noise); cat1 multi-hop still **25%**, cat4 **57–58.5%**
- Sequential p50 **~1340–1391 ms** (≈ baseline 1351 ms; under 2000 ms tax ceiling; still above ADR-006 1 s target)

**Conclusion:** entity-mediated 1-bridge expansion runs and emits paths, but does **not** move session coverage. Bridges via shared speakers mostly stay inside already-covered session neighbourhoods under `max_facts`. Checkpoint C EvR lift **NOT met**.

Next (recommended): rank/filter bridges by **novel source session** (prefer hubs whose provenance sessions are absent from the seed set); optionally raise bridge priority in MMR; keep deep-tier composition for residual multi-hop. Do not enable larger K without a session-novelty prior.

Checkpoint C checklist status:
- [x] ≤1-bridge expansion on context path (flag `cross_event_bridges`)
- [ ] Multi-hop stable graph EvR improved — **failed** (flat)
- [ ] Deep-tier composition route — out of this delivery (stub later)
- [ ] Fusion report C.4 — not required for go/no-go given flat EvR
- [x] Maintainer: keep `cross_event_bridges` default-on for substrate readiness; EvR payoff needs session-aware bridge ranking

### Checkpoint C2 — novel-session ranking **null** (2026-07-29)

**Shipped:** prefer ≤1-bridge expansions whose target hubs contribute source sessions absent from the seed/selected set (`select_bridge_neighbors` + pool fetch of hub sessions). K unchanged (`cross_event_bridges=3`).

**Live (`phase-c2-novel-session-a/b`, retrieve-only):**
- Stability **98.0%** — gate held
- Graph EvR **47.3%** flat vs phase-b / graph-stable-a; cat1 **25%**, cat4 **57.1%**
- Sequential p50 **~1876–1924 ms** (~**+500 ms** vs phase-c ~1340 ms) — pool fetch + chunk→session resolve

**Verdict:** novelty ranking alone does **not** lift EvR. Cost is real; EvR benefit is zero.

### Diagnostic — 1-bridge gold-session reachability (2026-07-29)

**Question:** among QAs where the graph channel misses gold evidence sessions on `locomoconv26clean`, does *any* 1-bridge neighbor of the retrieved seed hubs (cap 12) intersect those missing sessions?

**Method:** incomplete rows from `phase-c2-novel-session-a`; fresh `/retrieve/context` with `cross_event_bridges=0` for seed hubs; full `kg_hub_bridges` neighbourhood in Postgres; neighbor hub → session via `source_chunk_ids` → chunk text. Selection check: `cross_event_bridges=3` paths (`hubs` field) + final `graph_session_ids`. Artifacts: `.agent_tmp/hub_bridge_reachability.json`, `hub_bridge_reachability_selection.json`.

| Slice | Rate |
| --- | --- |
| Incomplete QAs where ≥1 gold-missing session is in the 1-bridge neighbourhood | **72/79 = 91.1%** |
| Same, cat1 multi-hop only | **22/24 = 91.7%** |
| Among reachable: emitted `paths` include a gold-covering neighbor hub | **64%** overall / **82%** cat1 |
| Among reachable: final curated `graph_session_ids` covers a gold-missing session | **0%** |
| Mean neighbor hubs that actually carry a missing gold session | **~5** (not “the whole graph”) |

**Unreachable residual (~9%):** mostly **session_11** — its 5 EVENT hubs use non-spine predicates (`ATTENDED`, `INVOLVED`, …) and have **0** rows in `kg_hub_bridges` (202/410 events are bridge-orphans under current spine filters). That slice is a substrate-construction gap, not a ranker bug.

**Interpretation:** intersection ≫ 0 → **do not pivot off speaker-mediated 1-bridge**. The substrate *can* reach gold sessions; C/C2 fail because **bridge facts lose the `max_facts` budget** after expand (paths often right, curated set never keeps the missing session). C2 novelty is insufficient as the sole lever and adds ~500 ms — optional latency revert is safe for EvR (null), but prefer fixing post-expand retention first so path hits can become EvR hits.

**Recommended next move (single):** reserve / hard-prioritize a small number of curated slots for **bridge-expanded facts that add novel sessions** (post-expand diversify must not drop them). Secondary (parallel, not instead): widen spine predicates (or membership rules) so orphan events like session_11 enter `kg_hub_bridges`. Deep-tier only for residual after retention lands.

### Checkpoint C3 — reserved bridge slots + ANN seed stability (**dual gate PASS**, 2026-07-29)

**Shipped (code):**
- Post-expand **reserved curated slots** for bridge-expanded facts that add novel sessions (`_BRIDGE_RESERVE_HUB_CAP`, ship setting **`reserve_hub_cap=12`** matching the expand seed shortlist)
- ANN seed determinism fixes for HNSW top-k without uuid order + embed float noise:
  - over-fetch then **uuid-stable** top-k (`ann_overfetch_k` + `stable_top_k_vectors`)
  - seed-hit stabilize (`_stabilize_seed_hits`)
  - **3 decimal-place** distance quantization (`_quantize_distance`)

**Live retrieve-only (`phase-c3-ann-stable-a/b`, sequential, `locomoconv26clean`, mf=50):**

| Metric | C3+ANN | Prior |
| --- | --- | --- |
| Graph-session agreement | **99.3%** | C3 reserved-slots pre-ANN **77.6%**; Phase A/B **99.3%** / C **98.7%** |
| Graph EvR full | **56.0%** | Phase A–C2 **47.3%** |
| Cat1 multi-hop EvR | **40.6%** | **25%** |
| Sequential p50 | **~1320 ms** | ~1351 ms graph-stable / ~1381 ms C3 reserved |

**Dual gate:** stability ≥95% **and** EvR lift (full ≥56%, cat1 ≥37.5%) → **PASS** (`dual_gate_ok=true`). Artifact: `benchmarks/runs/phase-c3-ann-stable-summary.json`.

**Root cause of the C3 stability regression (77.6%):** HNSW approximate top-k without uuid tie-break, amplified by embedding float noise → unstable seed sets → unstable reserved hubs. Fixes above restore Checkpoint A agreement while keeping the EvR lift.

**Ship setting:** `reserve_hub_cap=12` (code: `_BRIDGE_RESERVE_HUB_CAP = 12`).

**Secondary metric — judge accuracy (full `evaluate`, 2026-07-29):**

Config: de-fitted prompt, `locomoconv26clean`, mf=50, `use_ppr`, historical 16, max_passages 16. Arms `phase-c3-judge-a/b`.

| Arm | Acc | Multi-hop | Single-hop | Graph EvR | Agree A↔B | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| `phase-c3-judge-a` | 77.6% | 65.6% | 87.1% | **56.0%** | — | 1456 |
| `phase-c3-judge-b` | 78.3% | 65.6% | 90.0% | **56.0%** | **99.3%** | 1341 |
| `graph-stable-a` (clean, pre-C3) | 80.3% | 75.0% | 90.0% | 47.3% | — | ~1351 |
| `baseline-clean-a` (old brain) | 82.2% | 71.9% | 91.4% | 0%\* | — | 4776 |

\*Old-brain graph EvR unmeasurable in that run's logging era.

Identical-config full-evaluate stability: graph-session agreement **99.3%** (151/152); EvR full identical **56.0%** both arms — dual gate **held** under answerer load.

McNemar (exact):

| Comparison | Acc | Right / Wrong | Overall p | Multi-hop p | Single-hop p |
| --- | --- | --- | --- | --- | --- |
| A → B (noise) | 77.6→78.3 | 8 / 7 | 1.0 | 1.0 | 0.69 |
| `baseline-clean-a` → A | 82.2→77.6 | 3 / 10 | 0.092 | 0.63 | 0.25 |
| `baseline-clean-a` → B | 82.2→78.3 | 7 / 13 | 0.26 | 0.69 | 1.0 |
| `baseline-clean-a` → A+B pooled | 82.2→78.0 | 9 / 15 | **0.31** | 0.73 | 1.0 |
| `graph-stable-a` → A+B pooled | 80.3→78.0 | 10 / 16 | **0.33** | 0.34 | 0.73 |

**Verdict:** EvR lift **did not convert to answers**. Point estimate is slightly down vs clean-brain `graph-stable-a` and old-brain `baseline-clean-a`; all McNemar ns at n=152. Answerable stays **96.7%** while judge ~78% → answerer gap **~18–19%** (G4 / Phase D territory). Primary C3 claim remains the dual-gate EvR unlock.

**Recommended next:** Phase D.1 measured (see Checkpoint D.1). **Do not auto-start D.2** — maintainer fork: D.2 feature reranker **vs** Checkpoint D product-surface write-up (graph value = EvR/provenance, not LoCoMo judge).

Checkpoint C3 checklist status:
- [x] Reserved slots for novel-session bridge facts under `max_facts`
- [x] Stable graph EvR materially above 47.3% control (**56.0%**, cat1 **40.6%**)
- [x] Identical-config agreement ≥95% (**99.3%** retrieve-only and full evaluate)
- [x] Dual gate PASS; ship `reserve_hub_cap=12`
- [x] Judge McNemar secondary: ns vs baseline / graph-stable; EvR↑ answers flat → Phase D

---

### Phase D — Only if graph EvR↑ but answers stay flat

#### Task D.1 — Answerer consumes paths, not bags — **measured (soft convert, 2026-07-29)**

**Description:** When structured paths exist, harness/product prompt prefers path blocks for multi-hop questions (no gold fitting). Measure judge with paired McNemar; keep prompt-audit.

**Acceptance criteria:**
- [x] `prompt-audit` still passes.
- [x] Multi-hop McNemar vs bag prompt recorded (vs C3 pooled; ns).

**Dependencies:** B.4, C.2  
**Estimated scope:** S

#### Task D.2 — Feature reranker in empty filter socket (no LLM) — **not started**

**Description:** Linear/GBDT over distance, PPR, hub completeness, session novelty, provenance overlap with top passages; lower `max_facts` if EvR holds (ADR-006 §Alternatives).

**Acceptance criteria:**
- [ ] &lt;5 ms / 50 candidates; no network.
- [ ] `max_facts=20` matches `max_facts=50` stable graph EvR within tolerance.

**Dependencies:** Checkpoint B  
**Estimated scope:** M

### Checkpoint D.1 — paths into answerer (**soft convert**, 2026-07-29)

Config: de-fitted prompt, `locomoconv26clean`, mf=50, `use_ppr`. Arms `phase-d1-paths-a/b`. Artifact: `benchmarks/runs/phase-d1-paths-summary.json`.

| Arm | Acc | Multi-hop | Graph EvR | Agree A↔B |
| --- | --- | --- | --- | --- |
| `phase-d1-paths-a` | **80.9%** | 71.9% | **56.0%** | — |
| `phase-d1-paths-b` | **80.3%** | 75.0% | **56.0%** | **100%** |
| pooled D.1 | **80.6%** | — | 56.0% | — |
| C3 judge pooled | 78.0% | 65.6% | 56.0% | 99.3% |
| `baseline-clean-a` (old brain) | 82.2% | 71.9% | 0%\* | — |

\*Old-brain graph EvR unmeasurable in that run's logging era.

McNemar (exact, pooled):

| Comparison | Acc | Right / Wrong | p |
| --- | --- | --- | --- |
| C3 pooled → D.1 pooled | 78.0% → 80.6% | **17↑ / 9↓** | **0.17** ns |
| `baseline-clean-a` → D.1 pooled | 82.2% → 80.6% | 10↑ / 16↓ | **0.33** ns |

**Held:** Graph EvR **56.0%** (C3 dual-gate level); A↔B agreement **100%**; `prompt-audit` **passed**. No gold-fitting claims.

**Verdict:** **soft convert** — directionally up vs C3 on judge (point estimate +2.6 pp; McNemar ns at n=152). Answerable stays high while judge ~80.6% → answerer gap **~16%** (G4 still open). Paths help the answerer somewhat but do not clear the significance floor.

**Measurement note (TUI):** D.1 evaluate used harness `enrich_paths_from_triples` because the first TUI restart was blocked (`EPERM` unlink on `~/.brainapi/source/src/.DS_Store`). Workspace already had `retrieve.py` `_paths_for_curated` attaching fact `legs` to curated bridge paths. **Follow-up (same day):** `.DS_Store` removed; `brainapi start --no-services` synced workspace → `~/.brainapi/source` and restarted API; live `/retrieve/context` returns paths with native `legs` (probe: `native_legs_present=true`). Harness enrich remains a safe fallback when legs are already present (no-op).

**Next (maintainer fork — do not auto-start D.2):**
1. **D.2** feature reranker in the empty filter socket, **or**
2. **Checkpoint D product-surface write-up** — graph value = stable EvR / provenance / product surfaces, not LoCoMo judge conversion (Phase 4 gate in `06`).

### Checkpoint D

- [ ] Either judge multi-hop moves with paired significance, or written decision that graph value is EvR/product-surface only (per Phase 4 gate in `06`).
  - D.1: soft convert only (ns); fork above still open.
---

### Explicitly out of scope / reject list

| Item | Why |
| --- | --- |
| Leiden community summaries | Non-reproducible on sparse KGs (`2603.05207`); ignores event hubs |
| Flattening to (s,p,o) for primary retrieval | Violates Triangle of Attribution |
| LLM fact filter / CatRAG LLM edge scorer on context path | ADR-006 |
| Online IRCoT/ToG on `/retrieve/context` | Latency |
| Justifying work solely by LoCoMo headline | Phase 1 gate |

---

## Risks

| Risk | Impact | Detection |
| --- | --- | --- |
| **More graph facts, same answers (G4)** | High — wasted complexity | Channel fusion report (C.4); McNemar ns + flat answerer gap → pivot to D.1 or stop |
| Treat graph EvR as signal before determinism | High | Checkpoint A gate; refuse A/B |
| Diversification helps EvR but hurts precision | Medium | Watch single-hop judge flips; tune λ in MMR |
| Cross-event bridges create false multi-hop | High | Manual audit of new paths; require shared-entity + type constraints; Janitor hygiene |
| Latency regression undoes dossier win | High | `profile_stages` p50; budget CI on arms |
| PPR typed weights unstable across embed models | Medium | Freeze embed model in manifest; unit golden vectors |
| Clean brain still one conversation | Medium | Do not generalise; optional second sample later (`00` deferred scaling) |
| Concurrent ingest / wrong server SHA | Medium | One owner per brain; server code identity in manifest (`06`) |

---

## Critical path (task titles only)

1. Total order for fact ranking  
2. Deterministic candidate aggregation under concurrency  
3. Harness stability report for graph channel  
4. **Checkpoint A — graph metrics trustworthy**  
5. Hub- and session-aware diversification under `max_facts`  
6. Intra-hub completeness scoring  
7. Type-aware PPR transitions (CatRAG-lite)  
8. Path-shaped prompt hygiene (single hub)  
9. **Checkpoint B — stable graph EvR lifted**  
10. Write-time entity-mediated hub bridge index  
11. Context-path 1-bridge expansion via hub index  
12. Deep-tier cross-event composition route  
13. Channel-attributed fusion report  
14. **Checkpoint C — composition tiered**  
15. Reserved novel-session slots + ANN seed stability  
16. **Checkpoint C3 — dual gate (stable EvR lift) PASS**  
17. **Checkpoint D.1 — paths into answerer (soft convert; ns)**  
18. Maintainer fork: D.2 feature reranker **or** Checkpoint D product-surface write-up (stop justifying on LoCoMo judge)  

---

## Relation to other docs

- Extends `02-retrieval-multihop.md` with an **event-hub-first** execution plan after Phase 1.
- Satisfies `06` Phase 4 items (1)–(2): stabilize EvR, then composition — without reopening passages-only judge flatness.
- Obeys ADR-006 two-tier split and `00` measurement noise floor.
