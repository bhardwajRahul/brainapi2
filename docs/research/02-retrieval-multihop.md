# 02 — Query-time retrieval, ranking and multi-hop reasoning

Workstream owner: research analyst (this document). Scope: `src/core/search/`, the `/retrieve`, `/entity`, `/kg` API surface, and the agentic navigation surface (`src/core/agents/kg_agent.py`, `src/core/agents/tools/kg_agent/`, `src/services/mcp/`). Out of scope: extraction/ingestion quality, storage internals, the benchmark harness itself, and the pgvector HNSW dimension question (assigned to the roadmap per `00-scope-and-constraints.md:40`).

Revision 2. Judged against `docs/research/00-scope-and-constraints.md`, which is binding. Three constraints from that document drive most of what changed since revision 1:

- **`/retrieve/context` must be sub-second, fast and cheap** (`00:22`). No iterative LLM loops, no per-query LLM reranking, no multi-call decomposition on that path. This answers what were previously my open questions 4 and 5, and it invalidates part of my earlier plan.
- **The benchmark cannot resolve judge-accuracy differences below roughly 5-7 points** (`00:44-51`). Retrieval-side metrics are the trustworthy signal.
- **Event-leg supersession was a bug, is fixed, and existing brains were not repaired** (`00:39`). Every benchmark run analysed here was measured on a graph with wrongly-deprecated actor edges. I treat that as a confound, not a proposal.

Status of design intent: **`docs/decisions/003-simple-one-shot-context-api-retrieval.md` and `docs/decisions/004-agentic-navigation-retrieval.md` are still empty stubs.** Per `00:18` the maintainer's intent is a router rather than a winner, which matches the recommendation below.

All measurements come from the maintainer's LoCoMo runs in `benchmarks/runs/` (conv-26, n=152 non-adversarial, `deepseek-v4-flash` answering and judging at `temperature=0`). I ran no new benchmark runs. I did compute new statistics over the existing `answers.jsonl` and `report.json` files; every such number is labelled and reproducible from those files.

---

## What this workstream does

### 1. There are two retrieval surfaces, and only one of them can actually multi-hop

**Surface A — one-shot context API.** `POST /retrieve/context` → `get_context` (`src/services/api/controllers/retrieve.py:749-1065`). Single round trip, no LLM in the retrieval loop, returns a flat context blob. This is what the benchmark exercises and the surface with the sub-second budget.

**Surface B — agentic MCP tools.** `src/services/mcp/main.py:168-280` exposes `search_semantically`, `traverse_graph` (`main.py:231-260`), and `search_memory` (raw read-only Cypher/SQL, `main.py:272-280`). `traverse_graph` reaches `GraphAdapter.traverse_graph` (`src/adapters/graph.py:733-778`), which clamps to `MAX_TRAVERSE_DEPTH = 5` and `MAX_TRAVERSE_HOPS = 100` (`src/core/search/traverse.py:7-8`, clamped at `graph.py:746-747`).

These surfaces share no code, ranking, or bounds. **`traverse_graph` is not reachable from the REST API** — I searched repo-wide and only `src/services/mcp/` and tests call it. The depth-5 capability exists but only for MCP clients, and the depth-2 path is the only one ever benchmarked.

`src/core/agents/chat_agent.py` and `src/core/search/context.py` are empty. `KGAgent.search_kg` is a stub. So "agentic navigation" today is a tool surface plus a prompt (`src/services/mcp/prompt.py:62-83`) that says "you may call another tool or reply to the user" — **no iteration cap, no budget, no stopping criterion** (`prompt.py:81-82`).

### 2. Entry point: vector search only, fanned out over every content word

`get_context` runs spaCy over the query (`retrieve.py:754`, `src/utils/nlp/ner.py:97-118`) and builds query *variants* (`retrieve.py:429-460`): the full question; **every token whose POS is not in the ignore set** (`ner.py:82-83`), unbounded; up to 5 noun chunks (`retrieve.py:48`, `451-457`); and up to 8 synthesized `"<Entity> <relation-word>"` probes for enumeration-looking questions (`retrieve.py:463-515`).

Each variant is processed independently by `_collect_facts_for_variant` (`retrieve.py:770-811`), and all variants are fanned out concurrently (`retrieve.py:883-888`). Per variant: one embedding call (`retrieve.py:610`), a node vector search at `k=25`, a relationship vector search at `k=25` (`retrieve.py:51`, `614-643`), and one Cypher call.

There is **no entity linking**. `src/core/search/entities.py` has `extract_str_entities_from_text` and `graph_adapter.search_entities`, but neither is called from `get_context`. Entry points are pure ANN lookups against node and relationship embeddings — no alias table, no exact-string fallback, no name-match tiebreak, no acceptance threshold.

*Correction to revision 1:* I previously wrote that this fanout makes latency scale with query length. **That is wrong and the data refutes it.** Over 152 questions the correlation between question content-word count and `retrieve_latency_ms` is **r = 0.089** (config A) and **r = 0.068** (config B), and per-bucket medians are flat from 2 to 10 content words (3795-4331 ms in config A). The fanout is concurrent, so extra variants add load and cost, not wall-clock. Bounding the fanout is still worth doing, but it is a **cost** optimisation, not a latency one.

### 3. Traversal: hard-capped at two relationship hops inside one event

`_collect_facts_for_variant` calls `graph_adapter.get_event_centric_neighbors(seed_uuids)` (`retrieve.py:782-784`). The Cypher (`src/lib/neo4j/client.py`) matches `(n)-[r]-(m)-[r2]-(b)` with `r2.flow_key = r.flow_key`. So traversal is **exactly two relationship hops**, and the `flow_key` equality constrains both hops to the *same event hub*: it returns one event's subject/event/object triangle and **cannot cross from one event to another**.

The one-shot API therefore does not do multi-hop reasoning in the graph sense. It does fact lookup — whole events, one at a time. Composition across events is left to the answering LLM over the union of retrieved events. That is bag-of-facts multi-hop, not path-based multi-hop.

| Path | Bound | Anchor |
|---|---|---|
| `get_context` facts | 2 hops, same `flow_key` | `retrieve.py:782`, `lib/neo4j/client.py` |
| `get_context` dossiers | depth 3, branch factor 3, 50 node-visit budget per entity, ≤5 entities | `retrieve.py:49-50`, `entity_info.py:28-29` |
| `/kg/hops` | always 2nd degree; the `degrees` argument is accepted and **silently ignored** | `controllers/kg.py:22-48` |
| `/retrieve/entity/context` | `context_depth` default 3 | `entity_context.py:35-37,72-74` |
| MCP `traverse_graph` | depth 1-5 (default 2), limit ≤100 | `graph.py:746-747` |
| MCP agent loop | **unbounded** | `prompt.py:81-82` |

### 4. Passages: hybrid RRF over dense + keyword, retrieved twice and prompted three times

`_retrieve_passages` (`retrieve.py:667-725`) is the only genuinely hybrid component: dense `search_data` at `k≥24` (`retrieve.py:52`) fused with `data_adapter.search` keyword hits via reciprocal rank fusion (`retrieve.py:705`, `fact_filter.py:67-78`). `_collect_passages` (`retrieve.py:813-849`) runs this for up to 3 query variants (6 for enumeration) and RRF-fuses the result lists again.

Then `_get_historical_context` (`retrieve.py:851-862`) calls `_retrieve_passages` **with the same `request.text`** and puts the top `historical_limit` results in a separate response field. Measured over `answers.jsonl`: **91.5% of `historical_context` entries are byte-identical to a `source_passage`** in config B (80.0% in config A).

The waste compounds downstream. `text_context` is built as `[passage] <chunk>` lines followed by fact lines (`retrieve.py:1057-1058`), and the answering prompt renders `text_context`, *then* `source_passages` again, *then* `historical_context` (`benchmarks/locomo/prompts.py:55-67`). Reconstructing the prompt from the logged fields:

| Copy | Origin | Anchor | Chars per question (config B) |
|---|---|---|---:|
| 1 | passages inlined into `text_context` | `retrieve.py:1057` | 61,955 |
| 2 | `## Source passages` section | `prompts.py:56-59` | 61,827 |
| 3 | `## Historical context` section (91.5% duplicate of copy 2) | `prompts.py:64-67` | 60,196 |
| | **total passage-derived** | | **183,978** |
| | if sent once | | **61,955 (−66.3%)** |

Mean answer prompt+completion is 51,397 tokens per question, and at ~3.6 chars/token the passage text alone accounts for essentially all of it. Two of the three copies originate in this workstream (copy 1 is `text_context` composition, copy 3 is the duplicate `historical_context` retrieval); copy 2 is the harness. This is the single largest cost defect in the system and it costs nothing in accuracy to fix.

### 5. Ranking: distance, then optionally PPR, then a filter that never runs

Candidates are deduplicated by `(r.uuid, r2.uuid)` keeping the best score (`retrieve.py:890-895`), scored as `min` vector distance over the event's three nodes (`retrieve.py:789-798`). Then:

1. **Semantic** — sort ascending by that min distance (`retrieve.py:897`).
2. **Structural** — if `use_ppr`, run personalized PageRank over an adjacency built from the seed set and re-sort by `-max(ppr over the event's three nodes)` (`retrieve.py:898-912`, `554-565`; `personalized_pagerank` in `fact_filter.py:79-115`). Personalization mass is `1/(1+distance)` per seed (`retrieve.py:518-525`). The PPR score **completely replaces** distance ordering rather than blending, edge transitions are uniform, and the adjacency comes from a single `get_event_centric_neighbors` call over the seeds (`retrieve.py:528-551`) so propagation cannot leave the seed neighbourhood. Note also that `_build_adjacency_from_seeds` does **not** apply `_is_currently_valid`, so PPR propagates over edges that the candidate filter at `retrieve.py:787` then hides — the structural signal and the returned fact set disagree about which edges exist.
3. **LLM-based** — `filter_relevant_facts` (`retrieve.py:916-921`). **This is a no-op.** The call site passes no `llm_adapter`, and `fact_filter.py:21-25` returns every index in that case, so `apply_fact_filter=True` is provably identical to the `else` branch `ranked[:max_facts]`. The prompt at `fact_filter.py:27-37` is dead code on this path.
4. **Temporal** — the only temporal signal is a hard filter dropping predicates with `invalid_at` or `deprecated=True` (`retrieve.py:420-426`, applied at `787`). No recency weighting, no time-scoped filtering, no `happened_at` comparison. `happened_at` is string-appended to the fact text (`retrieve.py:405-408`) and left to the LLM.

Ranking is therefore **semantic + optionally structural, with no working LLM stage and effectively no temporal stage.**

### 6. Sufficiency retry: a lexical overlap heuristic that costs a second full pass

`_context_looks_insufficient` (`retrieve.py:568-605`) fires when the context blob is under 80 characters or fewer than 35% of the question's content words appear verbatim. On firing, it re-runs up to 3 follow-up queries and merges results (`retrieve.py:967-1012`). The check is purely lexical, so it misses paraphrase hits and fires on lexically-adjacent-but-wrong context. And on the retry path **PPR and the fact filter are both skipped** — `curated = ranked[:max_facts]` directly (`retrieve.py:1000-1001`) — so triggering the retry silently downgrades ranking. Critically for the budget, the retry is a serial second retrieval round (`retrieve.py:979-988` is a `for` loop with `await` inside).

### 7. Output: triples and text are assembled independently, and grounding is not enforced

The response (`retrieve.py:1060-1065`) has four fields: `text_context` (passages, then facts, then `[dossier:...]` lines), `triples` (the same curated facts, structured), `historical_context`, and `source_passages`.

Provenance flows one way: facts contribute `source_chunk_ids` (`retrieve.py:806`, `648-664`) and those chunks are appended to `source_passages` (`retrieve.py:950-963`). Nothing checks the reverse. **No triple carries a citation, no passage is linked to the fact it supports, and the consumer gets three undifferentiated prose buckets.** The answer is grounded in the union of retrieved text, not in a retrieved path.

### 8. What the measurements actually establish

Four runs form two arms with byte-identical manifests within each arm (`manifest.json` per run dir; `ingest.n = 0` in all four, so all four queried the same pre-existing `locomoconv26` brain with no re-ingest):

- **Config A** = `push75-a`, `push75-b`: `use_ppr=false`, `sufficiency_retry=false`, `max_passages=8`, `max_facts=40`, `historical_limit=10`.
- **Config B** = `push75-c`, `push75-d`: `use_ppr=true`, `sufficiency_retry=true`, `max_passages=16`, `max_facts=50`, `historical_limit=16`.

| | a | b | c | d |
|---|---:|---:|---:|---:|
| Judge accuracy | 72.4% | 75.7% | 82.9% | **86.2%** |
| 95% CI | 64.8-78.9 | 68.3-81.8 | 76.1-88.1 | 79.8-90.8 |
| `answerable_rate` | 0.927632 | 0.927632 | 0.940789 | 0.940789 |
| `evidence_session_recall_full` | 0.826667 | 0.826667 | 0.973333 | 0.973333 |
| `evidence_session_recall_partial` | 0.913333 | 0.913333 | 0.986667 | 0.986667 |
| `answerer_gap` | 20.4% | 17.1% | 11.2% | 7.9% |
| retrieval p50 / p95 (ms) | 4081 / 5683 | 4053 / 5446 | 4793 / 6333 | 4835 / 6860 |
| min retrieval latency (ms) | — | 2508 | — | 3410 |
| total LLM tokens | 4.007M | 4.057M | 7.736M | 7.864M |

**The retrieval-side metrics are bit-identical within each arm** — `0.826667` twice, then `0.973333` twice, to every decimal, including every per-category value. Retrieval is deterministic for a fixed graph and config. **All observed run-to-run noise lives in the answer+judge layer.** Measured directly: with byte-identical retrieval, **19 of 152 questions (12.5%) flip correctness between `push75-a` and `push75-b`, and 17 of 152 (11.2%) between `push75-c` and `push75-d`** — despite `temperature=0` at `benchmarks/locomo/answer.py:45`.

That has three consequences, and they pull in different directions.

**(a) Single-run judge-accuracy comparisons are not evidence, exactly as `00:48` says.** Comparing `push75-b` to `push75-c` — the comparison I used in revision 1 to claim "+7.2 points" — gives an exact McNemar p of **0.0614**. Not significant. My revision-1 framing was wrong and I have removed it.

**(b) But the correct test for this design does establish the effect.** The same 152 questions are asked in both arms, so this is paired data and independent-proportion CIs are the wrong instrument. Pooling both runs per arm and scoring each question 0-2:

- Config A pooled **74.01%**, config B pooled **84.54%**, delta **+10.5 points**.
- Config B better on **32** questions, config A better on **10**, tied on 110. Exact sign test **p = 9.4 × 10⁻⁴**.
- Per category (pooled): multi-hop **+26.6** (15 better / 2 worse), open-domain **+11.5** (2 / 0), temporal **+9.5** (6 / 2), single-hop **+3.6** (9 / 6).

So the config change is established at p < 0.001 *as a paired effect over two runs per arm*, and the gain is concentrated in multi-hop and temporal — the two priorities in `00:11-12`. What is *not* established is any particular point estimate; +10.5 is the best estimate from four runs on one conversation with one answerer model, and external validity beyond `conv-26` is untested.

**(c) The retrieval-side case is much stronger than the accuracy case and needs no statistics at all.** Because retrieval is deterministic, these deltas are exact, not estimates:

| Retrieval metric | Config A | Config B | Δ |
|---|---:|---:|---:|
| Full evidence-session recall, overall | 82.7% | 97.3% | **+14.6** |
| … multi-hop | 62.5% | 96.9% | **+34.4** |
| … open-domain | 63.6% | 90.9% | +27.3 |
| … temporal | 86.5% | 94.6% | +8.1 |
| … single-hop | 92.9% | 100.0% | +7.1 |
| `answerable_rate`, overall | 92.8% | 94.1% | +1.3 |

**Multi-hop full evidence recall going 62.5% → 96.9% is the argument for the config change.** It is measured on retrieval output, it is reproducible to the digit, and it moves the metric the maintainer cares most about. The judge-accuracy delta is corroboration, not the case.

**On the two recall numbers, precisely.** `94.1%` is `metrics.answerable_rate`, labelled "Answerable rate (gold tokens in context)" in `report.md` — a lexical check that the gold answer's tokens appear somewhere in the retrieved context. `97.3%` is `metrics.evidence_session_recall_full`, the fraction of questions where all gold evidence *sessions* were among the retrieved sessions. Both figures are real, distinct, and were quoted correctly in revision 1; they are not two measurements of the same thing. The important caveat is about what they are computed *over*: `retrieved_session_ids` is derived from `text_context`, `source_passages` and `historical_context` (`benchmarks/locomo/evaluate.py:201-203`), and `text_context` is truncated to 20,000 characters before being stored (`evaluate.py:25`, `194`). Since `text_context` puts passages first and passages average 61,955 chars, **the stored `text_context` contains zero graph fact lines and zero dossier lines in 152 of 152 questions** (verified: zero occurrences of the `" | "` fact separator and of `[dossier:` across the file). So evidence-session recall and the answerable rate are effectively measuring the **passage channel**, not the graph channel. They are still trustworthy retrieval-side signals — they just do not credit or debit the graph. The answerer itself did see triples, because `answer_question` receives the raw API response (`evaluate.py:204`) and `prompts.py:60-63` renders up to 80 of them; but nothing in the logs lets me attribute any answer to them.

**The answerer gap is partly instability, not inability.** The gap is 11.2% (c) and 7.9% (d) with identical retrieved evidence, while the within-arm flip rate is 11-12%. So a substantial share of the "evidence was there and the answer was still wrong" gap is the answerer being non-deterministic rather than incapable. The qualitative conclusion from revision 1 survives — recall is no longer the binding constraint in aggregate, and roughly 8-11 points are lost after the right evidence is in the prompt — but the remedy splits: instability calls for self-consistency or a less noisy answerer, while inability calls for better pruning and ordering. These are different interventions and the current numbers cannot separate them.

**Cost.** Config B costs ~51.4k answer tokens per question versus ~26.4k for config A — 1.95x — for the +10.5 paired points. Per §4, roughly two-thirds of that is literal duplication.

### 9. Every run above was measured on a corrupted graph

Per `00:39`, `_invalidate_superseded_relationships` wrongly deprecated an actor's older event legs. The fix is present in the working tree (`src/workers/tasks/ingestion.py:474-506` now guards with `_is_event_entity` at `484-485` and `505-506`, and the docstring at `481-482` states event hub legs are never invalidated), but no backfill was authorised, and all four runs show `ingest.n = 0` — they reused the brain ingested before the fix.

The interaction with this workstream is direct and severe. `retrieve.py:787` drops a candidate if **either** leg fails `_is_currently_valid`, so a single wrongly-deprecated `MADE` leg removes the entire event triangle from the fact set. On an unrepaired brain, an actor's non-most-recent events are largely invisible to `get_context`'s fact channel.

Conclusions that this plausibly confounds, flagged explicitly:

- **The multi-hop numbers are depressed by an unknown amount.** Multi-hop questions need older events; those are exactly what was hidden. Pooled multi-hop accuracy of 53.1% (config A) and 79.7% (config B) are lower bounds on what the same code would score on a clean graph.
- **The measured value of PPR may be inflated or deflated.** Because `_build_adjacency_from_seeds` skips the validity filter (§5.2), PPR propagated over the deprecated topology while the fact filter hid it. On a repaired brain, PPR and the fact set will agree for the first time, and the +34.4-point evidence-recall gain may not reproduce at that magnitude.
- **Any claim about the graph channel's contribution is currently unmeasurable**, both because of this and because the logged context excludes facts entirely (§8).

This is why the passages-only ablation cannot be run first on the existing brain: on an unrepaired graph it would show the event graph contributing little, and that result would be an artifact of the bug rather than a property of the design.

---

## Guarantees and where they break

What this workstream is trying to guarantee: *given a natural-language question, return — in under a second — a compact evidence set that contains everything needed to answer it, ordered so the most load-bearing evidence comes first, with enough structure that the consumer can tell which stored event supports which claim.* Ranked by impact on the priorities in `00:11-12` (multi-hop, then temporal):

**G1. The path is ~5x over its own latency budget, and the better configuration is the slower one. (Gap; now the defining constraint.)** Measured p50 is 4053 ms (config A) and 4793-4835 ms (config B) against a sub-second target (`00:22`, `00:55`). The **fastest single question in either arm is 2508 ms / 3410 ms** — nothing in the benchmark comes within 2.5x of budget. Config B is +782 ms and 1.95x tokens for its accuracy gain, so the accuracy and budget goals are in direct conflict and the conflict cannot be split by tuning constants. §"Latency decomposition" below works out which stages can and cannot fit.

**G2. The best-measured configuration is not the shipped default. (Gap, trivially fixable.)** `GetContextRequestBody` sets `use_ppr = False`, `sufficiency_retry = False` (`requests.py:399-400`), `max_passages = 8`, `max_facts = 40` (`requests.py:396-397`). The controller reads them with `getattr(request, "use_ppr", True)` (`retrieve.py:898`, `967`, `914`), but Pydantic always materialises the field so the `True` fallback never fires for an HTTP request. A caller posting `{"text": ...}` gets config A. The `getattr` defaults also disagree with the schema (`max_facts` 50 vs 40, `_DEFAULT_MAX_PASSAGES` 16 vs 8), so the code reads as though the good config were the default. Note this is now entangled with G1: config B is the better-retrieving config *and* the more budget-violating one.

**G3. Multi-hop is structurally impossible in the one-shot path. (Deliberate trade-off, now the dominant accuracy ceiling.)** The `flow_key`-constrained two-hop query cannot leave an event hub (§3). Nothing finds a *path* between two entities, which is the product goal in `00:7`. The +34.4-point multi-hop evidence-recall gain from PPR shows graph propagation does real work here — and that more is available.

**G4. Passages are billed three times. (Bug, pure cost, measured.)** 183,978 characters of passage text per question where 61,955 would do, a 66.3% reduction available at zero information loss (§4). Two of the three copies are this workstream's.

**G5. The LLM fact filter does not exist. (Gap.)** `apply_fact_filter` is a no-op (§5.3). The pipeline has *no* mechanism that can judge a candidate fact irrelevant; the only pruning is `[:max_facts]` over a distance ordering. Under `00:22` the fix cannot be an LLM call on this path, which changes what I recommend here versus revision 1.

**G6. Grounding is asserted, not verified. (Gap; biggest risk to "answers accurately every time".)** No triple carries a citation (§7). Combined with an answering prompt that instructs hedging over abstention (`prompts.py:14-15`), the failure mode is confident, unattributable answers, and nothing in the response lets a downstream verifier check an answer against a path.

**G7. Query fanout is unbounded and noise-generating. (Gap — cost, not latency.)** Every content-word token becomes an independent seed query (§2). A variant like `read` still pulls 25 nodes + 25 relationships into `seed_hits`, which feeds both the PPR personalization vector (`retrieve.py:899`) and dossier entity selection (`retrieve.py:1014-1029`), so weak variants corrupt the structural ranking signal and the dossier choice. Correcting revision 1: this does **not** drive p50 (measured r ≈ 0.07-0.09, §2).

**G8. Time-scoped, aggregation, negation and comparison questions have no dedicated handling. (Gap; `00:12` names temporal a top priority.)**
- *Time-scoped*: only invalidity filtering; `happened_at` is a prompt string. Temporal scores 86.5-91.9% on LoCoMo because LoCoMo puts explicit dates in the dialogue and `prompts.py:13` contains a hand-tuned temporal rubric. That is benchmark prompt engineering, not a retrieval capability, and it will not transfer to a corpus without literal dates.
- *Aggregation / enumeration*: a regex-triggered probe generator (`retrieve.py:463-515`) plus "scan EVERY passage" in the prompt (`prompts.py:12`). No count, no completeness check.
- *Negation*: none. "What did X not do" retrieves the same events as "what did X do", and nothing distinguishes absence of evidence from evidence of absence.
- *Comparison*: none. Two entities' facts merge into one pool ranked by distance to the whole question.

**G9. Empty and contradictory results are handled asymmetrically. (Gap.)** Empty: `_collect_facts_for_variant` returns silently on no seeds (`retrieve.py:772-773`); `_retrieve_passages` swallows all exceptions (`702-703`, `715-716`); the chunk fetch swallows exceptions (`962-963`); the response can be empty with no error and no signal. `_get_historical_context` then falls back to **the most recent chunks in the brain regardless of query** (`864-880`) — topically random context presented as relevant. Contradictory: nothing detects contradiction; `invalid_at` is the only mechanism and it depends on the write path.

**G10. Two endpoints crash on a cache miss. (Bug.)** `EntityContext.get_context` returns a 2-tuple on "no vector hit" and "node not found" (`entity_context.py:61-62`, `69-70`) but the caller unpacks 4 (`controllers/entities.py:72-74`) → `ValueError` → 500 for any unknown entity. `controllers/entities.py:50` reads `paths.target_node` with no `None` guard while `retrieve_matches` can return `None`.

**G11. The agentic surface is unbounded and partly broken. (Gap.)** No step cap or budget (`prompt.py:81-82`); `search_memory` hands a raw query language to a model. `KGAgentSearchGraphTool._run` initialises `v_results: list[Vector] = []` (`KGAgentSearchGraphTool.py:109`), never populates it, then uses it to drive the only relationship-aware lookup (`:150-162`), so that branch is dead — and the final projection discards relationships, returning bare nodes (`:164-173`). An agent using this tool cannot see edges.

**G12. `traverse.py` is not a traversal.** `flatten_neighborhood` (`traverse.py:50-95`) formats an already-materialised neighborhood and issues no queries; `MAX_TRAVERSE_DEPTH`/`MAX_TRAVERSE_HOPS` bound serialization. Actual depth comes from `get_neighborhood(depth)` (`graph.py:724-731`, `765`).

**G13. No acceptance threshold on entity resolution. (Gap; blocks the agentic tier.)** `entity_context.py:56-59`, `entity_info.py:170-172` and `controllers/kg.py:31-34` call `search_nodes` without `k`, taking the facade default of 10 (`src/utils/vector_search.py:24-33`); the first two then take `[0]` with no similarity threshold (`entity_context.py:64`, `entity_info.py:182-183`). Resolution is "nearest of 10, always accept", with no confidence returned. `/kg/hops` compounds it with `similarity_threshold=0.0` (`kg.py:46`).

**G14. The dossier explorer is the most expensive component and its scoring is unsound. (Gap.)** `_score_neighbors` (`entity_info.py:57-86`) issues **one `get_neighbors` Cypher per visited node and one `vector_store_adapter.get_by_ids` per neighbour edge**, unbatched, in a Python loop. `_recursive_explorer` visits up to `_MAX_EXPLORATION_WORK = 50` nodes per entity (`entity_info.py:29`, `114`), and `_run_dossiers` runs up to 5 entities **serially** (`retrieve.py:1034-1055`). `retrieve_matches` also embeds twice per entity (`entity_info.py:167-168`). Scoring is `path_score = max(score for _, score in candidate)` (`entity_info.py:145`), so a long path containing one good hop beats a short uniformly-good path, with no length penalty. The file's own header comment says it is "raw, unprecise and w/ poor devx" and "not fully supported by the Event-Centric v2 kg" (`entity_info.py:32-36`). And per §8 its output never even reached the logged context.

---

## Latency decomposition and the sub-second verdict

I could not isolate per-stage milliseconds — there is no instrumentation in `get_context` and I am not permitted to add any. What I *can* do exactly is count the sequential I/O round trips each stage requires from the code, and mark what is concurrent. Serial depth, not operation count, sets wall-clock. Config B, typical non-enumeration question, V variants:

| # | Stage | Anchor | Round trips | Serial depth | Concurrency |
|---|---|---|---|---|---|
| 1 | spaCy NLP | `retrieve.py:754`; `ner.py:116-117` | 0 network, **2 spaCy passes** (`_process_doc` called twice) | 2 | **on the event loop**, blocking |
| 2a | facts per variant | `retrieve.py:770-811` | 4 per variant (embed, node search, rel search, Cypher) | 4 | V branches concurrent (`883-885`) |
| 2b | passages | `retrieve.py:813-849`, `667-725` | 4 per query × 3 queries (6 if enumeration) | **12-24** | **serial loop in one thread** (`825-830`) |
| 2c | historical | `retrieve.py:851-862` | 4 | 4 | concurrent with 2a/2b — and 91.5% duplicate output |
| 3 | PPR | `retrieve.py:898-912`, `528-551` | 1 Cypher | 1 | + 20 Python iterations **on the event loop** |
| 4 | fact filter | `retrieve.py:916-921` | 0 | 0 | no-op |
| 5 | provenance chunks | `retrieve.py:950-957` | 1 (≤80 ids) | 1 | — |
| 6 | sufficiency retry | `retrieve.py:967-1012` | 4×3 passages + 4×3 facts = 24 | **24** | **serial `for` with `await`** (`979-988`) |
| 7 | dossiers | `retrieve.py:1031-1055`; `entity_info.py:57-150` | ≤5 × (2 embeds + 1 search + ≤50 Cypher + one vector fetch **per edge**) | **hundreds** | serial per entity, serial per node, unbatched per edge |

Critical path ≈ stage 1 + max(2a, 2b, 2c) + 3 + 5 + (6 if it fires) + 7, where stage 7 alone can exceed everything else combined. Stages 6 and 7 also explain why config B's *minimum* latency is 3410 ms: even the easiest question pays for the dossier walk.

**Which stages could plausibly fit a sub-second budget**

- **Yes, as-is:** 1 (after removing the duplicate spaCy pass and moving it off the event loop), 2a (genuinely parallel, only 4 deep, and the per-variant embeds are batchable into one call), 3 (PPR arithmetic is cheap — but its adjacency Cypher should be replaced by a precomputed table, and the 20 iterations must move off the event loop), 5 (one batched fetch), 4 (free).
- **Yes, if restructured:** 2b, but only collapsed to a single query with the embed batched and the dense and keyword searches issued concurrently — serial depth 12-24 cannot fit. 2c must simply be deleted.
- **No, at any per-operation latency:** 6 (a second full retrieval round is by definition a second pass) and 7 (hundreds of serial unbatched round trips; even at an optimistic 5 ms per operation, stage 7 alone blows a 1000 ms budget).

**Verdict: sub-second is achievable, but only by deleting stages 6 and 7 from this path and flattening 2b — not by optimising them.** A path consisting of stage 1 + one parallel wave of serial depth 4 + a precomputed-adjacency PPR + one chunk fetch has a serial depth of roughly 6-8 I/O operations, which fits 1 s at any plausible per-operation latency. The measured 2508 ms floor today is consistent with the dossier walk dominating, but **I could not verify that attribution**, so the first task in the plan is instrumentation, not optimisation. If per-stage timing shows the floor is spread evenly rather than concentrated in stage 7, the sub-second target and the current architecture are irreconcilable and the honest answer is to relax one of them — the trade being roughly **+10.5 paired accuracy points and +14.6 points of evidence recall against a 4-5x budget overrun**, which on `00:11-12`'s priorities I would resolve in favour of a two-tier split rather than by abandoning either goal.

**Where the reranker runs.** Revision 1 recommended a cross-encoder on the context path. Under `00:22` and `00:25` that is inadmissible as stated: a hosted reranker is a network hop on a path with no headroom, and an in-process cross-encoder over 50 candidates costs 50-200 ms on CPU, which is 5-20% of the entire budget for a component that is not the bottleneck. **Revised: on the context path, use a cheap learned reranker over features already computed** — vector distance, PPR score, event `happened_at` recency, provenance-chunk overlap with the top passages, seed-variant strength. A linear model or small GBDT over five features is microseconds and no model server. The cross-encoder and any listwise LLM reranking move to the deep-navigation tier, where `00:23` permits them.

---

## Open questions for the maintainer

Answered since revision 1 and removed: whether the context path may make LLM calls, and what its latency budget is (both settled by `00:22` and `00:55`). Not re-proposed: the supersession fix and the pgvector dimension question (`00:39-40`).

1. Was `filter_relevant_facts` ever meant to receive an `llm_adapter`, or was the LLM path abandoned deliberately — and given `00:22`, should the dead prompt at `fact_filter.py:27-37` be deleted or moved to the deep tier?
2. Is the `flow_key` equality constraint in `get_event_centric_neighbors` a deliberate decision that cross-event chaining belongs to the agentic surface, or an artifact of the schema you would lift if a cross-event query existed?
3. Does the sub-second budget apply to p50, p95, or a hard timeout — and is a degraded-but-fast response (fewer passages, no PPR) preferable to a slower complete one when the budget is at risk?
4. Should the dossier channel (`EventSynergyRetriever`, whose own header calls it "raw, unprecise") be deleted from `/retrieve/context` outright, or preserved behind a flag for the deep tier?
5. Who is the primary consumer of `triples` versus `text_context` — does any caller reason over the structured triples, or do they all concatenate `text_context`?
6. Is `historical_context` intended to be query-relevant (it currently re-runs the same passage search) or a genuinely separate recency channel — and if the latter, should it be recency-ordered rather than relevance-ordered?
7. Should `/retrieve/context` return an explicit "insufficient evidence" signal, or is silently returning a weak context the intended contract?
8. When two events contradict and neither has `invalid_at`, should retrieval return both, prefer recency, or refuse?
9. Is the MCP `traverse_graph` surface meant to become the deep-navigation retrieval path, or is it an integration tool a future `chat_agent` would bypass?
10. Should `search_memory` (arbitrary read-only Cypher/SQL from an LLM) survive into production?
11. Are the LoCoMo-specific rubrics in `benchmarks/locomo/prompts.py:12-15` meant to migrate into the product, or are they benchmark-only tuning?
12. Is a re-ingested clean brain available for benchmarking after the supersession fix, and if not, who owns creating one — the ablation in Phase 0 is uninterpretable without it?
13. Given the 11-12% per-question answerer flip rate at `temperature=0`, is reducing answerer variance (self-consistency, a different model) in scope for anyone, or must this workstream treat it as fixed noise?
14. Is `entity_sibilings.py` / `EntitySinergyRetriever` in scope for retrieval quality work?

---

## Frontier techniques

Each verdict is now also judged against the two-tier budget in `00:22-23`. "Context-tier" means admissible on `/retrieve/context`; "deep-tier" means MCP and other REST endpoints only; "write-time" means it must be precomputed.

### A. Personalized PageRank over an entity–passage graph (HippoRAG / HippoRAG 2)

**Mechanism.** Build an open KG from the corpus, index passages alongside entity nodes, seed PPR with query-linked entities, and let one propagation step surface multi-hop-relevant passages without iterative LLM calls.

**arXiv.** 2405.14831 (HippoRAG), 2502.14802 (HippoRAG 2).

**Reported gain.** HippoRAG: up to +20% over SOTA on multi-hop QA; single-step retrieval matches or beats IRCoT at 10-30x lower cost and 6-13x lower latency. HippoRAG 2: +7% on associative memory over the best embedding model while fixing HippoRAG's regression on simple factual recall.

**Cost.** Offline graph construction (already paid). Online PPR arithmetic is cheap; the expensive part is materialising the adjacency.

**Fit.** Partially validated here already: `use_ppr` moved multi-hop full evidence recall 62.5% → 96.9% (exact, deterministic). Three divergences from the paper matter: the adjacency is built only from seeds via a live Cypher call (`retrieve.py:528-551`) so propagation cannot go further; passages are not graph nodes, so PPR cannot rank passages; and the PPR score replaces rather than blends with the semantic score (`retrieve.py:910-912`). The single-step property is exactly what the budget requires — this is the one technique in the literature whose whole selling point is getting iterative-retrieval quality without iteration.

**Verdict: adopt — context-tier, with the adjacency moved to write time.** A materialised event-adjacency table replaces stage 3's Cypher call and makes wider propagation free at query time.

### B. Query-aware edge weighting for graph propagation (CatRAG)

**Mechanism.** Attacks the "static graph fallacy": PPR with indexing-time transition probabilities lets walks drift into high-degree hubs and stop short of the final hop. Adds symbolic anchoring, query-aware dynamic edge weighting, and a key-fact passage weight bias on top of HippoRAG 2.

**arXiv.** 2602.01965.

**Reported gain.** Consistent wins over SOTA on four multi-hop benchmarks; notably, plain recall improves modestly while *reasoning completeness* — recovering the whole evidence path without gaps — improves substantially.

**Cost.** One scoring pass over candidate edges per query; no LLM calls if embedding-based.

**Fit.** High, and the diagnosis matches: event hubs and `OCCURRED_WITHIN` context nodes are natural hubs, transitions are uniform, and `flow_key` is a ready-made symbolic anchor. The measured profile — 94.1% answerable but an 8-11 point residual gap — is "partial recall, incomplete chain."

**Verdict: adopt — context-tier**, after A, with weights precomputed per edge and only the query-similarity term computed online.

### C. Path-based retrieval and flow pruning (PathRAG)

**Mechanism.** Argues graph-RAG's problem is redundancy, not insufficiency. Retrieves relational *paths* using flow-based pruning and linearises paths, not flat chunk lists, into the prompt.

**arXiv.** 2502.14902.

**Reported gain.** Beats SOTA graph-RAG baselines across six datasets and five evaluation dimensions.

**Cost.** Path enumeration between node pairs; prompts get smaller, not larger.

**Fit.** Addresses G3 and G6 at once, and the *output-format* half is nearly free: BrainAPI already has ordered candidates and flattens them with `" | "` (`retrieve.py:412-413`). Path-shaped output also gives each fact a position in a chain, which is the raw material for attribution. The event-hub schema suits it — a path is an alternating node/event/node sequence. The flow-pruning half needs path enumeration, which is not sub-second on demand.

**Verdict: adapt — split by tier.** Path-shaped *formatting* is context-tier and cheap. Flow-based path *enumeration* is deep-tier or write-time.

### D. Listwise LLM reranking (RankGPT and successors)

**Mechanism.** Prompt an LLM to reorder candidates by relevance (sliding window for long lists), optionally distilled into a small specialised reranker.

**arXiv.** 2304.09542.

**Reported gain.** Properly instructed LLMs match or beat supervised SOTA rerankers zero-shot on BEIR; a distilled 440M model beats a 3B supervised model.

**Cost.** One or more LLM calls per query, 300 ms to seconds. A distilled cross-encoder removes the LLM but adds a hosted model and 50-200 ms on CPU.

**Fit.** The socket exists and is empty (G5), and precision-after-retrieval is where accuracy is lost. But `00:22` and `00:25` explicitly rule out per-query LLM reranking on the context path, and the distilled variant still spends 5-20% of a 1 s budget on a component that is not the bottleneck.

**Verdict: reject on the context path; adopt deep-tier.** Context-tier substitute: a linear/GBDT reranker over features already computed (distance, PPR score, `happened_at` recency, provenance overlap, variant strength) — microseconds, no model server, and it can be trained on the existing `answers.jsonl` labels. This is the significant change from revision 1.

### E. Complexity-routed retrieval (Adaptive-RAG)

**Mechanism.** A small classifier predicts query complexity and routes to no-retrieval, single-step, or iterative multi-step retrieval, with labels harvested from which strategy actually succeeded.

**arXiv.** 2403.14403.

**Reported gain.** Better efficiency *and* accuracy than both fixed-strategy and prior adaptive baselines on mixed-complexity open-domain QA.

**Cost.** One small-LM forward pass (single-digit ms) — comfortably inside the budget.

**Fit.** This is the mechanism that implements `00:18`'s "a router, not a winner". The labels already exist in `benchmarks/runs/*/answers.jsonl`: per-question category, `answerable_rate` inputs, and `judge_correct`. With four runs there are now two observations per arm per question, which is exactly what is needed to label a question as "one-shot suffices" versus "one-shot unreliable" rather than "one-shot failed once."

**Verdict: adopt — context-tier (the router itself), routing to the deep tier.** Load-bearing for the recommendation.

### F. Iterative retrieve-then-reason (IRCoT, Self-Ask, StructGPT, ToG)

**Mechanism.** Interleave retrieval with reasoning so what to retrieve next depends on what has been derived. IRCoT alternates CoT sentences and retrievals; Self-Ask emits explicit follow-up questions; StructGPT formalises invoke→linearise→generate over structured data; ToG runs LLM-guided beam search over KG paths.

**arXiv.** 2212.10509 (IRCoT), 2210.03350 (Self-Ask), 2305.09645 (StructGPT), 2307.07697 (ToG).

**Reported gain.** IRCoT: up to +21 retrieval points and +15 QA points on HotpotQA / 2WikiMultihopQA / MuSiQue / IIRC with reduced hallucination. ToG: SOTA on 6 of 9 datasets, training-free, with knowledge traceability and correctability. Self-Ask quantifies the *compositionality gap* — models answer all sub-questions correctly yet fail the composition, and the gap does not shrink with scale.

**Cost.** Multiple LLM calls plus multiple retrievals. HippoRAG measured IRCoT at 10-30x cost and 6-13x latency versus single-step graph retrieval.

**Fit.** Self-Ask's finding is the strongest argument for keeping these available: an 8-11 point gap with the evidence present is a compositionality gap by definition. ToG is the best template for the deep tier because its beam search maps onto `traverse_graph` and it preserves traceability, which G6 needs. Categorically inadmissible on the context path.

**Verdict: adopt ToG-style bounded beam search — deep-tier only. Reject on the context path.**

### G. Explicit evidence-gap analysis as the stopping criterion (FAIR-RAG)

**Mechanism.** Decompose the query into a checklist of required findings, audit accumulated evidence against it, emit targeted sub-queries for the specific gaps, iterate until sufficient.

**arXiv.** 2510.22344.

**Reported gain.** +8.3 F1 over the strongest iterative baseline on HotpotQA (0.453 F1), with 2Wiki and MuSiQue also reported.

**Cost.** An LLM call per audit round.

**Fit.** `_context_looks_insufficient` (`retrieve.py:568-605`) is a lexical stand-in for this module. The *check* is cheap and worth keeping on the context path as a **signal to the caller** (G9, open question 7). The *retry* is stage 6 — a second full retrieval pass — and per the decomposition it cannot fit the budget.

**Verdict: split. Keep the cheap check context-tier as an emitted signal; move the retry and any LLM audit deep-tier.** Also fix the existing defect that the retry skips PPR and the reranker (`retrieve.py:1000-1001`).

### H. Episodic memory graph with an agentic retriever (REMem)

**Mechanism.** Offline, convert experiences into a hybrid memory graph linking *time-aware gists* to facts. Online, an agentic retriever with purpose-built tools iterates over that graph. The paper names prior failure modes as overlooking episodicity, lacking explicit event modelling, and overemphasising simple retrieval over reasoning.

**arXiv.** 2602.13530.

**Reported gain.** +3.4 on episodic recollection and +13.4 on episodic *reasoning* over Mem0 and HippoRAG 2 across four benchmarks; also more robust refusal on unanswerable questions.

**Cost.** Agentic inference at query time; offline gist construction.

**Fit.** The closest paper to what BrainAPI already is. BrainAPI has the rarer half — explicit event modelling — paired with the weaker retrieval strategy. The offline **time-aware gist layer is write-time work and therefore admissible under `00:25`**, and it is the direct answer to G8's aggregation and time-scoping problems as well as a cheap replacement for the online dossier walk (G14). Its refusal-robustness result speaks to G9.

**Verdict: adopt — architectural reference. Gist layer is write-time (crosses into the ingestion workstream); the agentic retriever is deep-tier.**

### I. Does agentic search remove the need for a graph? (RAGSearch benchmark)

**Mechanism.** A controlled benchmark holding LLM backbone, retrieval budget and inference protocol fixed, comparing dense RAG and several GraphRAG variants *as retrieval infrastructure underneath* agentic search, training-free and RL-based.

**arXiv.** 2604.09666.

**Finding.** Agentic search substantially improves dense RAG and narrows the gap to GraphRAG, but GraphRAG **remains advantageous for complex multi-hop reasoning** and yields more stable agentic behaviour once offline cost is amortised. Explicit structure and agentic search are complementary.

**Verdict: adopt the conclusion** as the framing for ADRs 003/004 — it is the empirical backing for `00:18`'s router.

### J. RL-trained search agents (Search-R1, Search-o1)

**Mechanism.** Train the model to interleave reasoning and search calls with outcome-based rewards and retrieved-token masking (Search-R1); or add an agentic search workflow plus a Reason-in-Documents module that compresses retrieved text before it enters the reasoning chain (Search-o1).

**arXiv.** 2503.09516 (Search-R1), 2501.05366 (Search-o1).

**Reported gain.** Search-R1: +41% (Qwen2.5-7B) and +20% (Qwen2.5-3B) over RAG baselines across seven QA datasets.

**Cost.** RL infrastructure and a trainable open-weights model; BrainAPI calls a hosted `deepseek-v4-flash`.

**Verdict: reject the training requirement.** **Adopt one component, write-time:** Search-o1's Reason-in-Documents idea — compress passages before they enter a prompt — attacks the 51k-token cost directly, and if the compression is computed at ingest rather than per query it is admissible under `00:25`. That is the same artifact as REMem's gists.

### K. Recursive summary trees (RAPTOR) and community summaries (GraphRAG, LightRAG)

**Mechanism.** RAPTOR recursively embeds, clusters and summarises chunks into a tree and retrieves across abstraction levels. GraphRAG pregenerates community summaries for entity clusters and map-reduces partial answers. LightRAG adds dual-level retrieval with incremental index updates.

**arXiv.** 2401.18059 (RAPTOR), 2404.16130 (GraphRAG), 2410.05779 (LightRAG).

**Reported gain.** RAPTOR: +20% absolute on QuALITY with GPT-4. GraphRAG: substantial gains in comprehensiveness and diversity for global sensemaking over ~1M-token corpora. LightRAG: retrieval accuracy and latency improvements plus incremental updates.

**Cost.** GraphRAG's community summarisation is a heavy offline LLM pass that must be redone as the graph grows — a poor match for an append-only store with continuous writes. LightRAG's incremental update is the mitigation.

**Fit.** Open-domain is the category where recall is still clearly binding (76.9% answerable, 90.9% evidence recall, pooled 61.5% accuracy) and it is exactly the global-sensemaking class these target. But it is 13 questions, and static communities fight append-only writes.

**Verdict: reject GraphRAG community summarisation. Adapt the narrow idea — per-entity gists maintained incrementally, write-time.** Note that `[dossier:...]` (`retrieve.py:1031-1055`) is already a crude *online* version of this, computed per query at depth 3 with a 50-node budget — the expensive way to do it (G14).

### L. Grounding, attribution, and citation faithfulness (ALCE, Self-RAG)

**Mechanism.** ALCE defines the benchmark and automatic metrics (fluency, correctness, citation quality) for generating text with citations. Self-RAG trains a model to emit reflection tokens deciding when to retrieve and critiquing whether its output is supported.

**arXiv.** 2305.14627 (ALCE), 2310.11511 (Self-RAG).

**Reported gain.** ALCE: even the best systems lack complete citation support 50% of the time on ELI5 — unattributed generation is the norm unless measured. Self-RAG (7B/13B) beats ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning and fact verification, with gains in factuality and citation accuracy.

**Cost.** ALCE-style metrics are nearly free if the response carries passage IDs. Self-RAG needs a trained model.

**Fit.** `get_context` already computes `provenance_ids` per candidate (`retrieve.py:806`, `942`) and throws the association away when flattening into `source_passages`. Emitting `{triple, supporting_chunk_ids}` is a dict-shape change with no extra I/O — fully admissible on the context path — and it makes citation quality measurable for the first time.

**Verdict: adopt per-triple attribution — context-tier, zero added latency. Reject Self-RAG's trained variant; its verification loop belongs deep-tier.**

### M. Generate-then-ground and tree-structured iterative retrieval

**Mechanism.** GenGround alternates emitting a simpler single-hop question answered from parametric knowledge, then grounding that Q-A pair against retrieved documents and correcting it. Tree of Reviews expands a tree of reasoning paths, handling each retrieved paragraph separately and dynamically choosing to search, reject or accept, so one bad paragraph does not poison a chain.

**arXiv.** 2406.14891 (GenGround), 2404.14464 (Tree of Reviews).

**Reported gain.** Both report SOTA-class results on the HotpotQA / 2Wiki / MuSiQue family.

**Fit.** ToR's cascade-error argument is a design constraint on any deep-tier agent built over BrainAPI, because a single-chain agent on top of "nearest of 10, always accept" resolution (G13) will cascade confidently.

**Verdict: adapt — deep-tier design constraint, not a standalone priority.**

### N. Temporal QA benchmarking (ChronoQA) and temporal KG memory (Zep)

**Mechanism.** ChronoQA is a 5,176-question benchmark over 300k news articles covering absolute, aggregate and relative temporal types with explicit and implicit time expressions. Zep/Graphiti maintains a temporally-aware KG synthesising conversational and business data while preserving historical relationships and edge validity intervals.

**arXiv.** 2508.12282 (ChronoQA), 2501.13956 (Zep).

**Reported gain.** Zep: 94.8% vs 93.4% on DMR, up to +18.5% accuracy on LongMemEval with 90% lower latency, strongest on cross-session synthesis.

**Fit.** Temporal is a stated priority (`00:12`) and 86.5-91.9% on LoCoMo is not evidence of temporal capability: LoCoMo puts dates in the dialogue and `prompts.py:13` carries a hand-tuned rubric. Zep's validity-interval model is close to what `_is_currently_valid` gestures at, but retrieval never uses time as a ranking or filtering dimension. An as-of-time *filter* is an indexed predicate and therefore context-tier admissible; it is cheaper than what is there now, not more expensive.

**Verdict: adopt as-of-time filtering and recency features — context-tier.** ChronoQA-style question types are an eval gap to hand to `04-evaluation-and-applications.md`, coordinated with `05-temporal-truth.md`.

---

## Recommendation: one-shot vs agentic navigation

**Route, do not choose** — which matches `00:18`. RAGSearch (2604.09666) holds backbone and budget fixed and finds that agentic search lifts flat dense RAG and narrows the gap to graph RAG, while explicit graph structure stays advantageous for complex multi-hop and makes agentic behaviour more stable. REMem (2602.13530) arrives at the same place from the other side: an explicit *event* graph plus an agentic retriever beats HippoRAG 2 and Mem0, with the largest margin (+13.4) on episodic *reasoning*. BrainAPI has the rarer half of that pair and is pairing it with the weaker strategy.

The sub-second budget makes the split sharper than it was in revision 1: the context path is not merely the cheaper option, it is now constitutionally incapable of hosting anything iterative. So the two tiers get different jobs.

1. **The context path's job is recall and grounding within a hard budget** — one bounded parallel wave, precomputed structure, cheap feature-based ranking, per-triple provenance, an explicit sufficiency signal. It must get *faster* while its evidence recall stays at config B levels. Everything that made config B good and slow must be re-sourced from write-time precomputation rather than query-time work.
2. **The deep tier's job is composition** — iterative retrieve-then-reason, ToG-style bounded beam search over `traverse_graph`, gap-analysis stopping, LLM or cross-encoder reranking, verification.
3. **A cheap router decides**, trained on the labels already in `benchmarks/runs/*/answers.jsonl` (Adaptive-RAG, 2403.14403).

**Route to one-shot when:** the question names entities that resolve above a confidence threshold (which must first exist — G13); the answer plausibly lives inside one or two event hubs (attribute reads, "when did X do Y"); the lexical sufficiency check passes on the first pass; or a latency SLA applies. On LoCoMo that is most of the single-hop category, already at 88.6% with nothing an agent would add.

**Route to deep navigation when:** (a) the question requires composing facts across *different* event hubs, which the `flow_key`-constrained query structurally cannot retrieve (G3) — the strongest signal, and detectable from the question rather than the results; (b) it is a set/enumeration question where completeness matters and the regex probe generator cannot know whether it found every member (G8); (c) it is a multi-entity comparison, which needs separate retrievals and a joint ranking that the shared candidate pool destroys; (d) the sufficiency check fails on the context path — note this replaces the in-path retry, which moves here wholesale; or (e) the caller asks for discovery rather than an answer, since "non-obvious connections" (`00:7`) is an agentic workload.

**Never route to deep navigation when** entity resolution is low-confidence, because a beam-search agent on "nearest of 10, always accept" will chase a wrong entity for several expensive steps. Fix G13 first.

**Where each fails.** One-shot fails when the answer requires a path rather than a set: it returns many events ranked by distance and delegates composition to the answerer — the compositionality gap Self-Ask measured (2210.03350), which does not close with a bigger model. Deep navigation fails on cost and variance: 6-13x latency and 10-30x cost versus single-step graph retrieval (2405.14831), an unbounded loop today (G11), cascading errors from one bad expansion unless paths are isolated (2404.14464), and a tail that a p50 hides.

**Honest statement of the conflict.** As measured, the higher-recall configuration is also the slower and 1.95x more expensive one, and *both* configurations are 4-5x over budget. There is no setting of the existing knobs that is simultaneously sub-second and config-B-quality. The plan below resolves this by moving config B's expensive ingredients to write time and deleting two stages from the query path; if per-stage instrumentation shows that is not enough, the target and the quality goal are irreconcilable on this path and the maintainer has to relax one.

---

## Implementation plan

Sizing: **S** = 1-2 files, under a day. **M** = 3-5 files, a few days. Baseline is config B as measured (pooled 84.5% judge accuracy, 97.3% full evidence recall, 94.1% answerable, 4793-4835 ms p50, ~51.4k answer tokens per question).

**Measurement protocol, binding on every task below** (derived from §8 and `00:44-51`):

- **Primary readout is retrieval-side** (`evidence_session_recall_full`, `answerable_rate`, per-category). These are deterministic for a fixed graph and config, so **one run per arm is exact.** Report them per category, since the overall figure hides the multi-hop movement that matters.
- **Judge accuracy is secondary and requires ≥2 runs per arm plus a paired test** (exact sign test or McNemar on per-question correctness), never a comparison of independent CIs. I verified the difference this makes: `push75-b` vs `push75-c` as single runs gives p = 0.0614, while pooling two runs per arm and testing paired gives p = 9.4 × 10⁻⁴ for the same underlying change.
- **What this benchmark can and cannot resolve.** Simulating from the observed per-question outcomes: a *concentrated* effect of the magnitude actually seen (+10.5 points, driven by ~32 questions flipping) is detected with 0.94 power at 1 run per arm and 0.99 at 2. A *diffuse* uniform +5-point shift is detected with only 0.14 power at 2 runs per arm and 0.52 even at 6; a diffuse +3-point shift never exceeds 0.32 power at 8 runs. **So: fixes that repair specific broken question classes are measurable here; broad precision polish is not, at any affordable run count, and must be justified on retrieval-side metrics instead.**
- Latency and token figures from `report.json` are trustworthy (`00:51`) but were collected at `concurrency=2` against a local server; treat cross-run latency deltas under ~200 ms as noise.

### Phase 0 — Establish what is actually true before optimising anything

Ordered first because three of my previous conclusions turned out to rest on unmeasured assumptions.

**T0.1 (S) — Instrument `get_context` per stage.** Add timing around each stage in the §"Latency decomposition" table and emit it as a debug field or log line. No behaviour change.
*Acceptance:* a single request yields per-stage wall-clock for stages 1-7 plus a count of round trips issued.
*Verify:* run 20 questions and publish the stage breakdown. **This gates every latency claim in the plan, including my own.** If the 2508 ms floor is not concentrated in stage 7, the redesign in Phase 2 is misdirected.

**T0.2 (S) — Graph-channel ablation, on a clean brain.** Re-ingest `conv-26` on the fixed supersession code (`00:39`), then run three arms: facts+passages (config B), `max_facts=0` (passages only), and — because §8 shows the logged context never contained facts — one arm with passages capped low enough that facts survive any downstream truncation.
*Acceptance:* per-category `evidence_session_recall_full` and pooled paired judge accuracy for each arm; the graph channel's contribution stated as a point estimate with the resolvable floor attached.
*Verify:* **2 runs per arm** (0.99 power for a concentrated 10-point effect, per the protocol above). One run per arm suffices for the retrieval-side metrics. If the passages-only arm is within 5 points of config B, report "no contribution detectable above the ~5-point resolution floor" — *not* "no contribution". **Do not run this on the existing unrepaired brain**: §9 explains why the result would be an artifact of the supersession bug rather than a property of the design.

**T0.3 (S) — Re-baseline config A vs config B on the clean brain.** The +34.4-point multi-hop evidence-recall gain from PPR was measured on a graph where most historical events were filtered out at `retrieve.py:787`.
*Acceptance:* config A and config B rerun on the repaired brain, retrieval metrics reported per category.
*Verify:* 1 run per arm for retrieval metrics, 2 for accuracy. If the PPR gain shrinks substantially, Phase 1's ordering changes.

**T0.4 (S) — Audit the answerer gap by hand.** Sample 20 questions that are `answerable` but judged wrong, and classify each as (i) evidence present, answerer failed; (ii) lexical token overlap only, evidence not really present; (iii) flipped between the two runs of the arm.
*Acceptance:* a 20-row classification with counts.
*Verify:* the split between (i), (ii) and (iii) decides whether Phase 2's precision work or a variance-reduction measure is the better investment. §8 shows the 11-12% flip rate is the same order as the gap itself, so this is not a formality.

> **Checkpoint C0.** Publish: per-stage latency, the graph channel's measured contribution, clean-brain baselines, and the answerer-gap split. **No optimisation work starts before this.** Three revision-1 claims were wrong or unverifiable without it.

### Phase 1 — Free wins: correctness, cost, and defaults

None of these trade accuracy for latency; all are admissible on the context path.

**T1.1 (S) — Stop billing passages three times.** Make `_get_historical_context` (`retrieve.py:851-882`) either reuse `passage_hits` or exclude chunk IDs already in `source_passages`, and stop inlining `[passage]` blocks into `text_context` (`retrieve.py:1057`) now that `source_passages` is a separate field.
*Acceptance:* `set(historical_context) ∩ set(source_passages) == ∅`; `text_context` contains no passage text.
*Verify:* answer tokens per question **below 20k** (from 51.4k — the measured single-copy figure is 61,955 of 183,978 chars, a 66.3% cut) with per-category evidence recall unchanged. Deletes one full round trip from stage 2c as a side effect. Coordinate with `04-evaluation-and-applications.md`, since `prompts.py:55-67` also renders `source_passages` separately.

**T1.2 (S) — Align defaults with the config that retrieves better.** Set `use_ppr=True`, `sufficiency_retry` per T2.3's outcome, `max_passages=16`, `max_facts=50` in `GetContextRequestBody` (`requests.py:392-400`), and delete the `getattr(..., True)` fallbacks at `retrieve.py:898`, `914`, `967` so the schema is the single source of truth.
*Acceptance:* `POST /retrieve/context {"text": "..."}` with no other fields reproduces the config-B retrieval metrics; no `getattr` default disagrees with a Pydantic default.
*Verify:* `evidence_session_recall_full` ≥ the clean-brain config-B figure from T0.3. **Justified on retrieval-side evidence** (+14.6 overall, +34.4 multi-hop, deterministic), not on the judge-accuracy delta. Note this raises p50; it is provisional until Phase 2 restores the budget, and T1.2 and T2.x should ship together.

**T1.3 (S) — Per-triple provenance.** `provenance_ids` are computed per candidate (`retrieve.py:806`, `942`) then merged away. Attach `supporting_chunk_ids` to each `GetContextTriple`.
*Acceptance:* every returned triple carries its own chunk IDs, matching the candidate it came from; zero-provenance triples are counted.
*Verify:* a script reports the fraction of triples with ≥1 supporting chunk — the ALCE-style citation-support number (2305.14627), currently unmeasurable. No added I/O. If it comes back very low, that is a finding for `01-ingestion-extraction.md` about `source_chunk_ids` coverage, not a bug here.

**T1.4 (S) — Fix the crashing endpoints and the dead agent tool.** Return a 4-tuple from both early exits in `EntityContext.get_context` (`entity_context.py:61-62`, `69-70`); guard `paths is None` at `controllers/entities.py:50`; remove the dead `v_results` branch and stop discarding relationships in `KGAgentSearchGraphTool` (`:109`, `150-173`).
*Acceptance:* `GET /retrieve/entity/context?target=zzzznotanentity` returns 200-with-nulls or 404, not 500; the agent tool returns edges.
*Verify:* `pytest tests/ -k "entity_context or entity_info or kg_agent"` plus negative-path tests.

**T1.5 (S) — Bound the query fanout.** Restrict seed variants to the full question, NER entities and noun chunks; drop bare single tokens or down-weight them in `_seed_personalization` (`retrieve.py:518-525`) and exclude them from dossier entity selection (`retrieve.py:1014-1029`).
*Acceptance:* ≤8 variants per question; no `seed_hits` entry sourced from a single non-entity token.
*Verify:* embedding calls and vector searches per request down ≥50%; per-category evidence recall unchanged. **Expect no p50 improvement** — measured r ≈ 0.07-0.09 between question length and latency (§2). This is a cost and precision task; if evidence recall *drops*, the weak variants were carrying recall and this must be reverted and re-scoped.

> **Checkpoint C1.** Target: answer tokens per question under 20k (from 51.4k), evidence recall at or above the clean-brain config-B baseline, citation support measured for the first time, no 500s on unknown entities. Latency unchanged or slightly worse — that is Phase 2's job.

### Phase 2 — Make the context path meet its budget

This phase exists because of `00:22` and `00:55`. Its success criterion is latency, not accuracy, and its constraint is that per-category evidence recall must not regress.

**T2.1 (M) — Precompute the PPR adjacency.** Replace the live `get_event_centric_neighbors` call in `_build_adjacency_from_seeds` (`retrieve.py:528-551`) with a materialised event-adjacency table maintained at write time, and apply `_is_currently_valid` when building it so the structural signal and the fact set stop disagreeing (§5.2). Move the 20 PPR iterations off the event loop. Refs: 2405.14831, 2502.14802.
*Acceptance:* stage 3 issues zero live graph queries; PPR output is identical to today's for the same seed set on a fixed graph (regression test); adjacency covers ≥2 rings, not 1.
*Verify:* stage-3 time from T0.1 under 50 ms; multi-hop `evidence_session_recall_full` at or above baseline. Crosses into `03-memory-substrate.md` for the table itself.

**T2.2 (M) — Collapse passage retrieval to one wave.** `_collect_passages` (`retrieve.py:813-849`) runs 3-6 `_retrieve_passages` calls serially in one thread, each itself 4 serial round trips. Reduce to a single query, batch the embedding, and issue the dense and keyword searches concurrently.
*Acceptance:* stage 2b serial depth ≤4 round trips, down from 12-24.
*Verify:* stage-2b time from T0.1 cut ≥60%; `answerable_rate` and per-category evidence recall unchanged. If recall drops, the multi-query RRF was load-bearing and the fallback is to keep 2 queries concurrently rather than 6 serially.

**T2.3 (M) — Move the sufficiency retry to the deep tier; keep the check as a signal.** Delete the in-path retry (`retrieve.py:967-1012`, stage 6, 24 serial round trips) from `/retrieve/context`. Keep `_context_looks_insufficient` and return its verdict in the response so the caller — or the router — can escalate. Ref: 2510.22344.
*Acceptance:* `sufficiency_retry=true` on the context path is either removed or documented as deep-tier only; the response carries a sufficiency field; the escalation path exists on the MCP surface.
*Verify:* p95 improvement on the subset where the retry currently fires (instrument via the new field); evidence recall on that subset must be recovered by the escalation, not lost. This is the task that resolves G9 and open question 7 together.

**T2.4 (M) — Replace the online dossier walk with precomputed entity gists.** Delete `_run_dossiers` (`retrieve.py:1031-1055`) from the context path. Substitute a per-entity gist maintained incrementally at write time. Refs: 2602.13530 (time-aware gists), 2410.05779 (incremental update), 2501.05366 (compress before prompting).
*Acceptance:* stage 7 issues zero live graph queries on the context path; gists are available for the top seed entities from a single indexed lookup.
*Verify:* the largest single latency win available if T0.1 confirms stage 7 dominates — track p50 and, critically, the 2508/3410 ms *minimum*, which should fall furthest. Per §8 the dossier output never reached the logged context anyway, so the accuracy risk is low; confirm with per-category evidence recall. Gist construction crosses into `01-ingestion-extraction.md`.

**T2.5 (S) — Remove the duplicate spaCy pass and get NLP off the event loop.** `extract_elements` calls `_process_doc` twice (`ner.py:116-117`) and `retrieve.py:754` calls it synchronously on the event loop.
*Acceptance:* one spaCy pass per request, executed in a thread.
*Verify:* stage-1 time halved; no change to variant output (regression test on `_collect_query_variants`).

**T2.6 (S) — As-of-time filtering and recency features.** Add an optional as-of timestamp that filters candidates by validity interval, and expose `happened_at` recency as a ranking feature rather than a prompt string. Ref: 2501.13956.
*Acceptance:* a query with an as-of time excludes facts not valid then; recency is a numeric feature on each candidate.
*Verify:* temporal-category evidence recall and accuracy. Coordinate with `05-temporal-truth.md`, which owns validity semantics; this task consumes them rather than defining them.

> **Checkpoint C2.** **The gate is p50 under 1000 ms with per-category evidence recall at or above the C1 figure.** If instrumented stage times show that is unreachable even after T2.1-T2.5, stop and escalate: report the achieved p50, the residual serial depth, and the measured accuracy cost of the remaining cuts, so the maintainer can relax either the budget or the quality target with numbers in hand. Do not proceed to Phase 3 with the budget unmet — a router that escalates from an already-over-budget path cannot help.

### Phase 3 — Ranking quality within the budget

**T3.1 (M) — Blend PPR with the semantic score.** Change `_score_candidate_with_ppr` (`retrieve.py:554-565`) to a normalised combination, α configurable, default 0.5. Refs: 2405.14831, 2502.14802.
*Acceptance:* α=1.0 reproduces today's ordering exactly and α=0.0 the pure-distance ordering, both unit-tested.
*Verify:* sweep α ∈ {0, 0.3, 0.5, 0.7, 1.0}, reporting per-category evidence recall (deterministic, 1 run each). Judge accuracy only for the best and worst α, 2 runs each. A blend that beats both endpoints on multi-hop evidence recall wins; expect the accuracy difference to be **below the benchmark's resolution**, so decide on the retrieval metric.

**T3.2 (M) — Feature-based learned reranker in the empty filter socket.** Pass a cheap ranker into `filter_relevant_facts` (`retrieve.py:916-921`) over features already computed: vector distance, PPR score, `happened_at` recency, provenance overlap with top passages, variant strength. A linear model or small GBDT, trained on `answers.jsonl` labels. Then lower `max_facts` to 15-20.
*Acceptance:* inference under 5 ms for 50 candidates, no network call, no model server; with `max_facts=15` per-category evidence recall matches `max_facts=50`.
*Verify:* tokens per question down further with recall held; ranker latency reported as its own stage. **This replaces revision 1's cross-encoder recommendation, which `00:22` and `00:25` rule out on this path** — see technique D. If the features prove insufficient, the escalation is a cross-encoder on the *deep* tier, not on this one.

**T3.3 (S) — Query-aware edge weighting, precomputed where possible.** Modulate the materialised adjacency by a query-similarity term plus a `flow_key` symbolic anchor. Ref: 2602.01965.
*Acceptance:* two different queries produce different weight vectors over the same subgraph; the per-query term is O(edges in the seed neighbourhood) with no extra round trip.
*Verify:* multi-hop `evidence_session_recall_full` above T2.1 — CatRAG's "reasoning completeness" proxy — with stage-3 time still under 50 ms.

**T3.4 (S) — Path-shaped output.** Replace `" | "` flattening (`retrieve.py:412-413`, `1057-1058`) with ordered path strings and add a `paths` field. Ref: 2502.14902.
*Acceptance:* every returned fact appears in exactly one path with an explicit position; `text_context` stays byte-compatible until a consumer opts in.
*Verify:* diff old and new `text_context` over 50 questions requiring zero change; then measure with a path-aware answering prompt. Formatting only — the flow-pruning half is deep-tier.

> **Checkpoint C3.** Target: per-category evidence recall at or above C2 with `max_facts` cut ~3x, p50 still under 1000 ms, and paths plus per-triple citations in the response.

### Phase 4 — The deep tier and the router

**T4.1 (M) — Confidence-thresholded entity resolution.** Add an explicit `k` and a similarity threshold to the `search_nodes` calls at `entity_context.py:56-59`, `entity_info.py:170-172`, `controllers/kg.py:31-34`, plus an exact-name tiebreak, and return a confidence score.
*Acceptance:* below-threshold resolution returns "unresolved" rather than the nearest of 10; confidence is in the response.
*Verify:* prerequisite for T4.3 — an agent must not chase a wrong entity (G13, R7).

**T4.2 (M) — Train the router.** Small classifier on `benchmarks/runs/*/answers.jsonl`, predicting whether the context path suffices. With two runs per arm now available, label questions as "one-shot reliable" (correct in both) versus "one-shot unreliable" (0 or 1 of 2) rather than from a single verdict. Ref: 2403.14403.
*Acceptance:* held-out AUC > 0.7; inference under 50 ms so it fits the budget.
*Verify:* offline first — report the accuracy the system would reach if the router's "hard" bucket had oracle-perfect retrieval. **That is the ceiling on everything in Phase 4; if it is under 3 points, stop and do not build the loop.**

**T4.3 (M) — Bounded agentic route on the MCP surface.** Hard step cap, token budget, and gap-analysis stopping. Reuse `traverse_graph` for expansion; keep retrieved paths isolated so one bad path cannot poison the chain. Refs: 2602.13530, 2307.07697, 2510.22344, 2404.14464.
*Acceptance:* terminates within N steps for 100% of a 50-question sample; every answer carries its paths; `search_memory` removed from the agentic path or allowlisted.
*Verify:* per-category accuracy on the router's hard bucket versus the context path on the same bucket, **2 runs per arm, paired test**; report p50/p95 and tokens for that bucket separately so a 10x tail is not hidden inside a headline.

**T4.4 (S) — Write the ADRs.** Fill `003-simple-one-shot-context-api-retrieval.md` and `004-agentic-navigation-retrieval.md` with the routing decision, the conditions in each direction, and each tier's latency and token budget. Cite 2604.09666 for complementarity and `00:22-23` for the budgets.
*Acceptance:* both files state the decision, the alternatives, the measured evidence, and the revisit trigger.

> **Checkpoint C4.** Decide whether the deep route ships on by default for the routed bucket, stays flagged, or is dropped — gated on T4.2's ceiling against T4.3's measured cost.

### Deferred and handed off

Per-entity gist construction (T2.4's write-time half) and Reason-in-Documents-style passage compression (2602.13530, 2410.05779, 2501.05366) belong to `01-ingestion-extraction.md`. The materialised adjacency table (T2.1) belongs to `03-memory-substrate.md`. Validity-interval semantics consumed by T2.6 belong to `05-temporal-truth.md`. ChronoQA-style temporal question types and the `_TEXT_CONTEXT_CAP` truncation that hides the graph channel from the logs (`benchmarks/locomo/evaluate.py:25`, `194`) belong to `04-evaluation-and-applications.md`, along with the finding that answerer non-determinism at `temperature=0` flips 11-12% of questions per run.

---

## Risks

**R1. Phase 2 buys latency by deleting recall sources, and the benchmark cannot see small losses.** T2.2, T2.3 and T2.4 each remove work that might be carrying evidence. Per the protocol, a diffuse 3-5 point accuracy loss is undetectable at any affordable run count. *Detect:* gate every Phase 2 task on per-category `evidence_session_recall_full`, which is deterministic and exact, and never on judge accuracy. *Mitigate:* keep each removal behind a config flag so it can be reverted per request, and re-measure recall after each one individually rather than as a batch.

**R2. The whole evidence base is one conversation, one brain, one answerer, and a corrupted graph.** n=152 on `conv-26`; ±7-point CIs; a 3.3-point run-to-run spread; 11-12% of questions flip with identical retrieval; and per §9 every run was measured on wrongly-deprecated actor edges. *Detect:* T0.3 re-baselines on a clean brain before any conclusion is carried forward; add a second conversation before treating any cross-config delta as general. *Mitigate:* prefer deterministic retrieval-side metrics, which are immune to the answerer noise entirely.

**R3. The trustworthy metrics measure the passage channel, not the graph.** `evidence_session_recall_full` and `answerable_rate` are computed from `retrieved_session_ids`, derived from the 20,000-char-truncated `text_context` plus the passage lists (`evaluate.py:25`, `194`, `201-203`), and the stored `text_context` contains zero fact lines in 152/152 questions. So the metrics I am telling the plan to trust cannot credit or debit the event graph. *Detect:* T0.2's ablation is the only instrument that can, and it needs the truncation raised or the field ordering changed — a harness change owned by `04-evaluation-and-applications.md`. *Mitigate:* until then, treat every claim about the graph channel's value as unverified, including the PPR gain.

**R4. `answerable_rate` is lexical, so the precision-bound diagnosis could still be wrong.** It counts gold tokens appearing in context, which coincidental overlap satisfies. If true recall is materially below 94.1%, Phase 3's precision focus is misdirected. *Detect:* T0.4's hand audit, before Phase 3. *Mitigate:* T0.4 also separates answerer instability from answerer inability, which decides whether the fix is ranking or variance reduction.

**R5. Turning on PPR by default (T1.2) exposes an untested path to all callers, and it currently raises latency.** `personalized_pagerank` is uncommitted, has a dense-fallback branch that redistributes mass to *every* node when a node has no neighbours (`fact_filter.py:104-108`), and is O(nodes × iterations) on the event loop. On a large brain with a big seed set this could be slow or memory-heavy — and T1.2 lands before T2.1 fixes the adjacency. *Detect:* T0.1's stage-3 timer, plus a test on the largest available brain rather than `conv-26`. *Mitigate:* cap adjacency size with a fallback to distance ordering, and consider shipping T1.2 and T2.1 together rather than sequentially.

**R6. Path-shaped output and provenance change the response contract.** Open question 5 is unanswered — I do not know who reads `triples` versus `text_context`. T1.1 additionally *removes* passage text from `text_context`, which is a behaviour change for any consumer relying on it. *Detect:* keep `text_context` byte-compatible in T3.4 and diff over 50 questions; for T1.1, announce the change explicitly since byte-compatibility is not preservable there. *Mitigate:* version the response or gate T1.1's `text_context` change behind a request flag for one release.

**R7. The deep route regresses easy questions if routing is imprecise.** A ToG-style agent over "nearest of 10, always accept" resolution (G13) chases wrong entities confidently, and a router at AUC 0.7 will send genuine single-hop questions there. *Detect:* report per-category accuracy for routed versus not; require the deep route to beat the context path on the routed bucket by more than the resolution floor before it ships on. *Mitigate:* T4.1 is a hard prerequisite for T4.3.

**R8. Phase 2 succeeds on p50 and fails on p95.** The measured p95/p50 ratio is 1.3-1.4, and the tail is where the sufficiency retry and long dossier walks live — exactly what Phase 2 removes, so the tail should improve most. But the router adds a second failure mode: escalations pay both tiers. *Detect:* report p95 and p99 alongside p50 for every Phase 2 task, and report end-to-end latency for escalated queries separately from context-path latency. *Mitigate:* set an explicit escalation budget in T4.4's ADR rather than letting it emerge.
