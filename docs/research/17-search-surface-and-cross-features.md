# 17 — Search surface on the shared knowledge base

Workstream: make BrainAPI a **unique source of truth** that already powers memory and recommendations, and can also power **search**, with shared write-time features and isolated query contracts. Binding: `00-scope-and-constraints.md`, ADR-005, ADR-006, **ADR-007** (Accepted). Recs isolation: `16-recsys-eval-protocol.md`. Memory SOTA: `08`, `12`, `13`. Substrate indexes: `03` §G.

**Plan mode.** This document does not change `src/` or harnesses. Tasks are sized for later implementers.

Skills applied: scientific-brainstorming (ideas labeled, no automatic winner), scientific-critical-thinking (severity, construct validity), planning-and-task-breakdown (vertical slices, gates), paper-lookup (arXiv 2026-08-18).

---

## 0. Scope of this session

**Focal question.** How can one brain serve (a) agent memory, (b) document/entity search, and (c) recommendations as *separate products* that share indexes and cheap scores, without making LoCoMo / LongMemEval / BEAM / recsys **worse**?

| | Record |
| --- | --- |
| Purpose | Architecture + literature + implementable plan |
| Audience | Maintainer; retrieval / substrate / eval owners |
| Decision owner | Maintainer (ADR-007 Accepted 2026-08-18; §15 locked below) |
| Time horizon | Core search API in weeks; plugins and LTR later |
| In scope | Read APIs, lexical+ANN indexes, fusion, cheap rankers, search eval, plugin boundary, cross-features that are indexes or scores |
| Out of scope | Replacing ingest; training a proprietary bi-encoder as a prerequisite; making `/retrieve/context` the search engine; MS MARCO as a product SLO; dual-use / biosafety (not implicated) |
| Real constraints | ADR-006: no LLM on context path; p50 < 1000 ms target unmet (~1.4 s); judge cannot resolve <5–7 LoCoMo points; 3072-d HNSW skipped; keyword path is ILIKE |
| Assumed | Event-leg supersession bug stays fixed going forward; existing unrepaired brains may still exist |
| Negotiable | Whether context *later* opts into BM25; search p95 number; Postgres FTS vs external inverted-index plugin |
| Unknown | Keyword vs paraphrase mix of future search tenants; corpus size per brain in production |
| Prohibited | LoCoMo judge as a search metric; wiping memory eval brains; silent default changes on `/retrieve/context` |

**Perspectives represented:** retrieval engineering, memory eval, recsys (existing plugin split), IR literature. **Missing:** a production search-traffic owner (no click log). Do not treat LoCoMo questions as a substitute for that mix.

---

## 1. Product model (decision, proposed)

Three **read** products, one **write** path. Analogous to recsys (`16`): BrainAPI is the KB; heavy models are plugins.

```text
                         POST /ingest*  (chunks, graph, vectors, topics)
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
         /retrieve/context   /retrieve/search    /retrieve/recommend
         memory assembler    ranked hits         ranked items
         LoCoMo/BEAM         benchmarks.search   benchmarks.recsys
         ADR-006 SLO         search SLO          recs SLO
```

**What “search” is not.** It is not “call context and parse `text_context`.” Context exists to feed an answering model. Search exists to return **addressable hits** (chunk / entity / event / structured row) with scores, filters, and snippets.

**What “cross-feature” is.** A structure computed at write time, or a cheap score at read time, that more than one product can consume without inheriting the other’s loops (PPR over event hubs, cross-encoder, LightGCN train, sufficiency retry).

---

## 2. Current state (located evidence)

Anchors are the working tree as of 2026-08-18. Older line numbers in `02` / `07` may have drifted; behaviour below was re-checked in `retrieve.py`, `data.py`, `vectors.py`.

### 2.1 Surfaces that exist

| Surface | Role today | Search-engine analogue |
| --- | --- | --- |
| `POST /retrieve/context` | Memory: spaCy variants → dense node/rel seeds → event-centric facts + PPR + bridges + topics; passages via RRF(dense, keyword) | RAG retriever, not a SERP |
| `GET /retrieve/` | `retrieve_data`: text + preferred entity types | Internal lookup |
| `GET\|POST /retrieve/recommend` | Graph recommend (+ optional LightGCN plugin) | Recs, not lexical search |
| `GET /retrieve/entity/synergies` | Sibling / synergy scores | Related-entities widget |
| MCP `search_semantically` / `traverse_graph` / `search_memory` | Deep / agentic | Escalation, unbounded |

### 2.2 The “hybrid” passage channel is not BM25∪dense

```1680:1722:src/services/api/controllers/retrieve.py
def _retrieve_passages(...):
    ...
    vector_hits = vector_search.search_data(...)
    ...
    search_result = data_adapter.search(text, brain_id)  # ILIKE
    fused = reciprocal_rank_fusion([vector_ids, keyword_ids])
```

`data.py` `search` is `WHERE text ILIKE %s` with `'%' + escaped_query + '%'`. No inverted index, no term weights, no `k1`/`b`. RRF (`fact_filter.py`, `k=60`) fuses ANN order with a **substring hit list**.

**Implication (interpretation):** ADR-005’s hybrid design is real as *architecture*; the sparse leg is not the IR method the SOTA brief and HyperMem call BM25. Keyword/navigational search is the first thing this stack cannot honestly sell.

### 2.3 Dense ANN is optional / silently off on default pgvector

`_vector_index_ddl` returns empty string when `dimension > 2000`. Deploy examples set all stores to **3072**. `search_vectors` still `ORDER BY embeddings <=> $1` and sets `hnsw.ef_search` (no-op without an index). Over-fetch + uuid tie-break (`ann_overfetch_k`, `stable_top_k_vectors`) is in place for **determinism**, not speed. Milvus uses `AUTOINDEX` + COSINE **if** that backend is selected.

On LoCoMo-sized brains this is not the latency bottleneck (`06`: passages p50 829 ms of which **682 ms embed**). On BEAM-scale or multi-million-chunk brains it becomes a sequential scan. **Two different products, two different bottlenecks.**

### 2.4 Memory path latency and quality (do not mix with search claims)

| Signal | Status | Use for search work |
| --- | --- | --- |
| Context p50 | ~1.3–1.4 s after dossier deletion; ADR-006 target 1 s | Shared embed batching helps both; do not put CE on this path |
| Passage EvR | ~97% conv-26 | LoCoMo **cannot** show BM25 recall gains in aggregate; need a keyword slice or a non-LoCoMo corpus |
| Graph vs passages-only judge acc | McNemar ns at n=152 | Do not justify search graph-channel by LoCoMo accuracy |
| Graph EvR | C3: 47.3% → 56.0% with reserved bridge slots + ANN seed stability | Graph as a **search channel** (return events/entities as hits) is still a product idea, not a LoCoMo win |
| SOTA compose arm | conv-26 95.4% judge (`13`); full-10 still open | **Do not touch default context ranking** while that claim is live unless the gate in §8 passes |
| Recsys | Isolated brain `demorecsys`; plugins for GNN / attribute prefs | Copy this isolation for search eval |

### 2.5 Ranking on context (no LTR, no CE)

Distance (min over event nodes) → optional PPR **replaces** distance → MMR diversify. LLM fact filter is a no-op without a deep-tier adapter. No LambdaMART, no linear blend of BM25+dense+PPR. Historical context still re-runs the same passage search (~91.5% duplicate; cost, not wall-clock). Passage collection still **serial** over 3–6 queries each calling `embed_text` though `embed_texts` exists.

### 2.6 Recs already show the split we want for search

`16`: core = train-free graph ranker on `/retrieve/recommend`; `plugins/features-rec` writes extra edges; `plugins/recsys-gnn` is optional LightGCN and **does not replace** ingest. Search should copy that: core BM25+dense+filters; plugins for SPLADE / ColBERT / CE.

---

## 3. Literature (located, 2026-08-18)

**Retrieval summary**

- Query: canonical IR + memory papers by arXiv ID, plus title search for hybrid/RRF/SPLADE/ANCE
- Scope: targeted lookup (not exhaustive)
- Databases: arXiv Atom API (`export.arxiv.org/api/query`) parsed with `paper-lookup/scripts/arxiv_atom.py`
- Access date: 2026-08-18
- Limits: first page of title-OR search (`total_results=59`, retrieved 8); ID lists complete (12 + 4). RRF’s original SIGIR 2009 paper is **not on arXiv** (Cormack, Clarke, Buettcher) — cited by name only.

**Warnings:** Absence from this bound is not “never studied.” UniIR (`2311.17136`) is **multimodal** instruction retrieval — weak fit for text-only BrainAPI v1. HyDE (`2212.10496`) uses an LLM at query time — incompatible with ADR-006 on context; plugin/deep only.

### 3.1 What actually supports a BrainAPI search product

| Paper | arXiv | Status for us | Takeaway (not a BrainAPI result) |
| --- | --- | --- | --- |
| BEIR | [2104.08663](https://arxiv.org/abs/2104.08663) | **support-located** (zero-shot IR) | BM25 is a strong OOD baseline; dense often loses zero-shot; hybrid / CE / late-interaction win on average at higher cost. **This is the argument for a real lexical index before a custom bi-encoder.** |
| DPR | [2004.04906](https://arxiv.org/abs/2004.04906) | support-located (mechanism) | Dual-encoder + ANN can beat BM25 **in-domain with labels**. We have hosted embeddings, not DPR training. |
| ANCE | [2007.00808](https://arxiv.org/abs/2007.00808) | support-located (mechanism) | Hard negatives from a live ANN index close the train/test gap. High engineering cost; not a v1 gate. |
| ColBERT / v2 / PLAID | [2004.12832](https://arxiv.org/abs/2004.12832), [2112.01488](https://arxiv.org/abs/2112.01488), [2205.09707](https://arxiv.org/abs/2205.09707) | mixed | Token-level matching approaches CE quality; PLAID claims tens of ms on GPU at 140M passages. **Index size and serving complexity → plugin**, not core. |
| SPLADE / v2 | [2107.05720](https://arxiv.org/abs/2107.05720), [2109.10086](https://arxiv.org/abs/2109.10086) | mixed | Learned sparse on inverted indexes; v2 reports large BEIR gains. Index growth and inference → plugin after BM25 exists. |
| M3-Embedding (BGE-M3) | [2402.03216](https://arxiv.org/abs/2402.03216) | **idea-adjacent** | One model, three modes: dense, sparse, multi-vector, long context (8k). Strong **cross-feature candidate** if we ever change the embedder; not a silent swap (would re-embed every brain). |
| Matryoshka / MRL | [2205.13147](https://arxiv.org/abs/2205.13147) | support-located | Truncatable embeddings; `03` already maps this to making HNSW legal. `text-embedding-3-large` supports dimension cut natively. |
| Fusion analysis | [2210.11934](https://arxiv.org/abs/2210.11934) | **challenge-located vs “just RRF”** | Convex combination of lexical+dense scores can beat RRF; RRF is **parameter-sensitive**. Our hardcoded `k=60` is a choice, not a law. Prefer exposing RRF **and** CC on the search API. |
| HyperMem | [2604.08256](https://arxiv.org/abs/2604.08256) | support-located (memory) | LoCoMo SOTA report uses **topic→episode→fact + BM25∪dense RRF**. We already have topic coarse-to-fine; we lack BM25. **Same hybrid is a memory cross-feature**, opt-in on context after the gate. |
| BEAM | [2510.27246](https://arxiv.org/abs/2510.27246) | support-located (memory scale) | Long dialogues; retrieval cost grows with N. HNSW legality matters here more than on conv-26. |
| Filtered ANN | [2602.11443](https://arxiv.org/abs/2602.11443) | support-located (substrate) | Metadata filters + ANN; Milvus recall more stable than pgvector plans in their study. Search **filters** (type, validity, time) are first-class, not a WHERE after LIMIT. |
| Contrastive embeddings | [2201.10005](https://arxiv.org/abs/2201.10005) | background | Off-the-shelf embeddings can be strong for semantic search. Justifies keeping a general embedder in v1. |
| HyDE | [2212.10496](https://arxiv.org/abs/2212.10496) | **veto on context** | LLM generates a hypothetical doc then embeds. Deep/search-plugin only. |

**Not treated as evidence for BrainAPI quality:** MS MARCO MRR tables in the user’s SOTA brief (analytical interpretation, different task, no reproduction here).

### 3.2 What would confirm / refute transferring these papers

- **Confirm BM25 value:** keyword-slice Recall@10 on a brain where ILIKE misses exact tokens that BM25 posts; latency not worse than ILIKE seqscan on a 100k-chunk fixture.
- **Refute BM25 value:** no Recall@k lift on that slice and on LoCoMo passage EvR (already saturated).
- **Confirm HNSW:** recall@k vs exact `ORDER BY <=>` within a declared drop (e.g. ≤2 points at k=24) **and** p95 vector stage drop on a large fixture.
- **Refute “search will lift LoCoMo judge”:** McNemar ns after BM25-on-context opt-in — expected; do not use that as a search-ship failure.

---

## 4. Independent idea register

Stage: `independent` then clustered. Origin: `AI-assisted` + `literature-inspired` (this session). These are **proposals**, not findings.

| ID | Statement | Origin | Predicted observation | Disconfirm |
| --- | --- | --- | --- | --- |
| S01 | Add `POST /retrieve/search` returning scored hits, not a prompt blob | product | Search clients stop scraping `text_context` | Nobody uses it; context remains the only caller |
| S02 | Replace ILIKE with Postgres BM25/`tsvector` (or `pg_trgm` for short codes) as **core** lexical | BEIR, HyperMem | Keyword Recall@k ↑; long-query ILIKE misses ↓ | No lift on keyword slice |
| S03 | Legalize ANN (`halfvec` HNSW or MRL truncate ≤2000) with loud skip | `03` G, MRL | BEAM/large-brain vector p95 ↓; seed stability held via overfetch | Recall@k vs exact collapses |
| S04 | Batch `embed_texts` once per request; concurrent dense+lexical; single query on context | ADR-006 leftover | Context p50 toward <1 s | Stage timings still show multi-RTT embed |
| S05 | Expose fusion `rrf` and `cc` (convex combination) on search | 2210.11934 | CC wins in-domain if a tiny labelled set exists | RRF ≈ CC on our mix |
| S06 | Cheap linear/GBDT ranker over {dense, lexical, PPR, recency, topic, validity} | ADR-006, LTR brief | Precision@10 on search set ↑; microseconds | Features collinear; no lift |
| S07 | Search channels: `passages` \| `entities` \| `events` \| `structured` | product | Entity-name queries don’t require passage hits | Graph channel noisy; users ignore non-passage |
| S08 | Metadata filters (label, validity, time range) applied **inside** retrieval | 2602.11443 | Filtered search correct; no post-LIMIT drop | Planner ignores filter; recall hole |
| S09 | Opt-in `lexical=bm25` on context after search BM25 is proven | HyperMem | Memory keyword misses ↓ without judge drop | EvR or McNemar regresses |
| S10 | Plugin `features-search` or `search-rerank`: CE on K≤10 | brief, ADR-006 | Search nDCG ↑; context untouched | p95 blows search SLO |
| S11 | Plugin late-interaction (ColBERT/PLAID) or SPLADE | ColBERT family, SPLADE | Hard queries ↑ | Index size / reindex cost dominates per-brain |
| S12 | Optional embedder swap to BGE-M3 for dense+sparse+colbert from one model | 2402.03216 | One write, three retrieval modes | Re-embed cost; LoCoMo seed shift |
| S13 | Recs cold-start: `/retrieve/search` on item text for new SKUs, then graph recs | recs×search | Cold items appear before walks exist | Duplicate ranking; worse HitRate |
| S14 | Do **not** train ANCE/DPR in v1 | negative control | — | If we later get labels and a refresh loop, reopen |
| S15 | HyDE / query LLM rewrite only on deep search | 2212.10496, ADR-006 | Hard paraphrases ↑ at high latency | Context accidentally calls it |
| S16 | Search eval harness on a **separate** brain; ledger `benchmarks.search` | `16` analog | Claims don’t contaminate LoCoMo | People still quote judge % for search |

**Minority / preserved:** S12 (embedder swap) and S11 (ColBERT core) are attractive and **wrong as v1** if they force re-ingest of memory brains. S07 graph-as-hits may look unused until a tenant searches people/events rather than documents — keep the channel even if default k focuses on passages.

**What became less obvious after evidence check:** “Ship hybrid” is already the *shape* of context; the gap is the **sparse implementation** and the **missing hit-list API**, not the absence of RRF. “Need IVF+PQ / 100M HNSW RAM” is the SOTA brief’s catalog assumption; BrainAPI shards **per brain**. “Cross-encoder on K=10 for 200 ms p99” fights ADR-006 on memory and is optional on search.

---

## 5. Clusters (by mechanism, not by slogan)

| Cluster | IDs | Shared mechanism |
| --- | --- | --- |
| C-API | S01, S07, S08 | New contract: hits, channels, filters |
| C-LEX | S02, S09 | Real inverted/lexical index |
| C-ANN | S03, S04 | Vectors actually indexed; embed RTT collapsed |
| C-FUSE | S05, S06 | How scores combine (RRF/CC/LTR) |
| C-PLUG | S10, S11, S12, S15 | Heavy IR off the core path |
| C-X | S09, S13 | Cross-product: memory opt-in BM25; recs cold-start via search |
| C-EVAL | S16 | Isolated measurement |
| C-NOT | S14 | Explicit non-goals for v1 |

Merges: S02 and “SPLADE in core” stay **split** (different index ops and failure modes). S04 is not the same as S03 (latency vs scale).

---

## 6. Criteria (declared before preference)

Direction: higher is better unless noted. Weights are decision aids, not a computed winner.

| Criterion | Weight | Anchors (1–5) | Evidence needed |
| --- | --- | --- | --- |
| Memory non-regression | **Gate** (noncompensatory) | Fail = LoCoMo/BEAM/recsys worse | Paired McNemar, passage EvR, recs HitRate, p50 |
| Search usefulness | 3 | 1 = still ILIKE+blob; 5 = hit API + BM25 + filters | Recall@k / nDCG on search set |
| Latency / SLO fit | 3 | 1 = extra LLM; 5 = index lookup + one embed | `profile_stages` |
| Cross-feature leverage | 2 | 1 = search-only fork; 5 = one index, three readers | Same DDL used by context/search/recs |
| Feasibility / reversibility | 2 | 1 = re-embed all brains; 5 = additive index + flag | Rollback to ILIKE/exact scan |
| Information gain if null | 1 | Null still teaches query mix | Pre-registered slices |

**Vetoes (not averaged):** LLM on `/retrieve/context`; default context ranker change without §8 gate; search eval writing memory brains; dual source of truth.

---

## 7. Adversarial review (short)

| Idea | Wrong if… | Alternative explanation | Hidden dependency | Harm |
| --- | --- | --- | --- | --- |
| S02 BM25 | LoCoMo-like questions are long paraphrases; BM25 adds posting I/O and zero recall | Dense already covers; ILIKE was enough on tiny N | Tokenization mismatch with embedder tokenizer | Over-rank boilerplate terms |
| S03 HNSW | Approx top-k reshuffles seeds → graph EvR jitter (C3 already fought this) | Latency win is still embed RTT | `halfvec` ops / pgvector version | Silent quality flake on memory |
| S01 new API | All “search” users were agents who wanted context anyway | Product confusion; two ways to do one thing | Client SDK + docs | Split brain of integrations |
| S06 LTR | No labels except LoCoMo sessions (biased to memory) | Overfit to session IDs | Click log does not exist | Ranking that looks good offline, worse UX |
| S12 M3 embedder | Different geometry vs current 3072-d OpenAI vectors | “One model three modes” marketing | Full re-embed | SOTA compose arm incomparable |

**Anchoring risk:** the user’s SOTA brief pushes hybrid→CE→LTR on a 200 ms p99 web SLO. We keep the **sequence of primitives** (lexical, ANN, optional CE) but **reject the SLO and the default CE**.

---

## 8. Non-regression protocol (how we refuse to make memory worse)

Copy recsys isolation (`16`, `benchmarks/AGENTS.md`).

### 8.1 What may change without a memory A/B

- New routes under `/retrieve/search*` that do not alter context defaults.
- New tables/indexes that context **does not read** yet (e.g. `tsvector` column unused by `_retrieve_passages`).
- Plugins that only register search rerank.

### 8.2 What requires the gate

- Context reading BM25 instead of ILIKE.
- Enabling HNSW / changing embedding dimensionality or metric.
- Changing RRF `k`, seed `k`, passage fanout, PPR default, fact diversify.
- Shared embed batching that changes variant set or seed order (stabilize with existing uuid/distance quantization).

### 8.3 Gate (all must hold)

1. **Memory champion arm** (document the exact `run_id` at implementation time; today the conv-26 SOTA pointer is `locomo-compose-sota-conv26-v4d` in `REPORTS.json`): paired McNemar **not** significantly worse; point estimate ≥ baseline. Prefer **product** profile for default-path changes; do not require SOTA compose (SC/gap-fill) to move.
2. Passage **full evidence-session recall** held to the digit on that brain (deterministic). Graph EvR: if ANN/seeds change, identical-config agreement ≥95% (`00`).
3. Context p50: not worse than the then-current post-dossier baseline beyond noise (record both numbers). Sub-second remains ADR-006, not a search ship blocker.
4. Recsys: `demorecsys` HitRate@K unchanged unless the PR is a recs PR.
5. Brains: search fixtures use IDs like `searchbench*`. Never wipe `locomoconv26*`, `beam*clean`, `demorecsys`.

**If BM25-on-context is null on LoCoMo:** keep ILIKE as context default; search still ships BM25. That is a successful split, not a failed hybrid.

---

## 9. Proposed search API (idea → later spec)

Vertical slice: one call that a UI or engine can use without parsing memory blobs.

```http
POST /retrieve/search
```

```json
{
  "query": "alice counseling license",
  "brain_id": "default",
  "k": 20,
  "channels": ["passages", "entities"],
  "lexical": "bm25",
  "dense": true,
  "fusion": "rrf",
  "filters": {
    "labels": ["PERSON", "EVENT"],
    "currently_valid": true,
    "happened_after": null
  },
  "rerank": "none",
  "profile_stages": false
}
```

```json
{
  "hits": [
    {
      "id": "chunk_…",
      "channel": "passages",
      "score": 0.031,
      "scores": { "bm25": 12.4, "dense": 0.22, "rrf": 0.031 },
      "snippet": "…",
      "source_session_ids": ["session_3"],
      "entity_uuid": null
    }
  ],
  "insufficient": false,
  "stage_timings": null
}
```

Optional later: `GET /retrieve/search/suggest` (prefix / trgm). Out of v1 if it expands scope.

`rerank`: `none` | `linear` | `plugin:<name>`. Unknown plugin → 400, not a silent fallback that looks like core quality.

---

## 10. Cross-feature map

| Feature | Memory | Search | Recs | When to share |
| --- | --- | --- | --- | --- |
| BM25 / tsvector | Opt-in passage channel (HyperMem-like) | **Default lexical** | Item-text cold start (S13) | Write once; read per product flag |
| HNSW / halfvec | Seed + passage ANN at scale | Same | Node similarity if used | Shared; gate on seed stability |
| Batched embed | Closes ~400 ms gap (`06`) | One RTT per search query | If recs embed queries | Shared helper, no rank change if vectors identical |
| Topic memberships | Coarse-to-fine already on context | Facet / boost | Weak | Already written |
| Validity / `happened_at` | Current-truth vs history (`05`) | Filters | Recency decay (already on recs) | Shared properties |
| PPR / hub bridges | Multi-hop coverage | `events` channel ranking | Not default | **Do not** turn search into a second context assembler |
| Linear LTR features | ADR-006 context ranker | Search precision | Could consume overlap scores | Train **per product** or freeze a tiny global blend with separate weights |
| Cross-encoder | Deep tier only | Search plugin, K≤10 | No | Plugin |
| LightGCN | No | No | Plugin | Unchanged |

**Prediction:** the highest cross-leverage items are **S02+S03+S04** (indexes + embed batch). They improve search immediately and *can* improve memory latency/keyword recall without touching judge prompts.

---

## 11. Architecture decisions for implementers

- **Lexical engine v1:** Postgres FTS (`tsvector` + `ts_rank_cd` or a BM25 extension available in the deployed image). Avoid requiring Elasticsearch as system of record (ADR-007). A plugin may *project* to an external engine later.
- **Do not change `EMBEDDING_*_DIMENSION` as a drive-by** in the search PR. HNSW legality is a dedicated slice with a recall-vs-latency measurement (`00`, `03`).
- **Context keeps ILIKE** until S09 gate. Implementation may write tsvector in the same migration.
- **RRF stays** as default fusion (already in code, HyperMem-like). Expose `fusion=cc` with `alpha` for search; do not retune context `k=60` in the same PR.
- **No ANCE/DPR training, no ColBERT core, no HyDE on context** in this workstream’s v1.

---

## 12. Search evaluation protocol

Mirror `16` so search cannot hide inside LoCoMo. Binding write-up: [`18-search-eval-protocol.md`](18-search-eval-protocol.md). Harness: `benchmarks/search/` + `./search.sh`.

| Item | Rule |
| --- | --- |
| Brain | `searchbench*` only |
| Harness | `benchmarks/search/` + `./search.sh`; HTTP only |
| Ledger | `benchmarks.search` in `REPORTS.json` only |
| Metrics | Recall@{5,10,20}, nDCG@10, MRR; p50/p95 of `/retrieve/search` **excluding** `embed.query`; optional keyword/paraphrase slice |
| Query mix | At least: (a) **keyword / name**, (b) **paraphrase / semantic**. Filter/time deferred until the search API accepts filters. Gold = chunk IDs via `DOCID` markers, not LLM judge |
| Cheap start | Reuse a **held-out** subset of memory questions as *retrieval* labels (gold session → chunks via provenance) **without** running the answerer. Report as `search-from-locomo-qrels`, never as LoCoMo accuracy. Not in harness v1 |
| Forbidden | Ingest into memory eval brains; quoting judge % as search quality |

Until a scored `evaluate` run is on the `benchmarks.search` leaderboard, do not claim “search is better than BM25-in-the-brief.” Toy `search_toy` rows are isolation checks, not BEIR.

---

## 13. Task list

Vertical slices. Each leaves the system working. No task is XL.

### Phase 0 — Contract and isolation

## Task 1: Freeze the three-surface contract

**Description:** ADR-007 accepted or explicitly deferred by the maintainer; this doc linked from `00`. No ranking code.

**Acceptance criteria:**
- [ ] Maintainer records accept / amend / reject on ADR-007
- [ ] `00` workstream row for `17` remains accurate

**Verification:** Human review of ADR-007.

**Dependencies:** None  
**Files:** `docs/decisions/007-three-product-surfaces-one-kb.md`, `docs/research/00-scope-and-constraints.md`  
**Scope:** S

## Task 2: Search eval skeleton

**Description:** Package `benchmarks/search/` with smoke: ingest N chunks into `searchbenchsmoke`, call a **stub** or interim `POST /retrieve/search` (may 501 until Task 4), ledger key documented in `benchmarks/AGENTS.md`.

**Acceptance criteria:**
- [ ] `./search.sh smoke` fails loudly if it would use a memory brain id
- [ ] `REPORTS.json` schema comment or empty `benchmarks.search` leaderboard allowed

**Verification:** `./search.sh smoke` documented; unit test on brain-id guard.

**Dependencies:** Task 1 (or proceed as Proposed)  
**Files:** `benchmarks/search/**`, `benchmarks/AGENTS.md`  
**Scope:** M

### Checkpoint A
- [ ] Product split is written; search eval cannot wipe LoCoMo/BEAM brains

---

### Phase 1 — Core search first stage (does not change context defaults)

## Task 3: Lexical index at write time

**Description:** Persist `tsvector` (or equivalent) on `data_text_chunks` (and entity name field if cheap). GIN index. Backfill existing brains lazily or via explicit job. Context **does not** query it yet.

**Acceptance criteria:**
- [ ] New chunks get lexical vectors on ingest
- [ ] ILIKE path still used by `_retrieve_passages`
- [ ] Unit test: token query matches a chunk ILIKE would miss or vice versa, documented

**Verification:** `pytest` on FTS helpers; explain/analyze on a fixture shows index use.

**Dependencies:** None (can parallel Task 2)  
**Files:** `src/lib/postgresql/data.py`, ingest write path, tests  
**Scope:** M

## Task 4: `POST /retrieve/search` core hybrid

**Description:** New controller: embed once, dense `search_data` + BM25/tsvector, fusion RRF default, return hits. No PPR, no fact assembler, no historical duplicate.

**Acceptance criteria:**
- [ ] OpenAPI/schema: query, k, channels⊇passages, lexical, dense, fusion
- [ ] Does not import dossier/synergy retrievers
- [ ] `profile_stages` works like context
- [ ] Context tests unchanged (no default behavior change)

**Verification:** `pytest` API tests; `./search.sh smoke` green against live API.

**Dependencies:** Task 3  
**Files:** `src/services/api/routes/retrieve.py`, new controller module, `requests.py`  
**Scope:** M

## Task 5: Filters + entity channel (optional second slice)

**Description:** `currently_valid`, label filter; optional `entities` channel via node ANN + name FTS.

**Acceptance criteria:**
- [ ] Filter is applied so that invalid entities are not in top-k when `currently_valid=true`
- [ ] Passage-only default remains if `channels` omitted

**Verification:** Tests with a superseded edge / deprecated node.

**Dependencies:** Task 4  
**Files:** search controller, graph/data queries  
**Scope:** M

### Checkpoint B
- [ ] Search API returns BM25∪dense hits
- [ ] `/retrieve/context` byte-level ranking defaults unchanged (diff + LoCoMo smoke optional)

---

### Phase 2 — Shared speed/scale (gated)

## Task 6: Batch embeddings on context **and** search

**Description:** One `embed_texts` per request; concurrent dense+lexical; collapse context passage variant loop toward ADR-006 (single query or batched variants sharing the batch). Preserve seed uuid/distance stabilization.

**Acceptance criteria:**
- [ ] Stage timings: a single `embed.query` (or one batched stage) on a profiled request
- [ ] Passage EvR held on a recorded memory arm **if** context code changed
- [ ] Search p50 excluding network policy: one embed RTT

**Verification:** `profile_stages` JSON; paired EvR if context touched.

**Dependencies:** Task 4 if sharing helper; else can start from context-only  
**Files:** `retrieve.py`, embeddings adapter call sites  
**Scope:** M

## Task 7: Legalize HNSW

**Description:** Measurement-first: `halfvec` vs truncate-to-2000 vs Milvus-only. Loud log/metric when index DDL skipped. Keep overfetch+uuid order.

**Acceptance criteria:**
- [ ] Default 3072 install either has an ANN index **or** fails closed with a visible error at store init
- [ ] Recall@k vs exact on a ≥10k vector fixture recorded
- [ ] Memory gate §8 if context seeds change

**Verification:** Fixture script + notes in run dir; no silent seqscan.

**Dependencies:** None vs search API; **gate** before flipping production default  
**Files:** `src/lib/postgresql/vectors.py`, config, tests  
**Scope:** M

### Checkpoint C
- [ ] Context p50 moved toward 1 s **or** numbers escalate per ADR-006
- [ ] ANN skip cannot be silent

---

### Phase 3 — Fusion / cheap LTR / memory opt-in

## Task 8: Search fusion knobs + linear rerank

**Description:** `fusion=rrf|cc` with documented `alpha`; optional `rerank=linear` using dense, lexical, recency, validity — **no LLM**.

**Acceptance criteria:**
- [ ] Default remains RRF k=60-equivalent
- [ ] Linear weights are config/request, not a trained black box in v1
- [ ] Search nDCG measured on `searchbench*` vs RRF-only

**Verification:** `./search.sh evaluate` two arms.

**Dependencies:** Task 2, Task 4  
**Files:** search controller, `fact_filter.py` or shared fusion  
**Scope:** S–M

## Task 9: Opt-in BM25 on context (cross-feature)

**Description:** `GetContextRequestBody.lexical: ilike | bm25` default **`ilike`**. Document HyperMem-style union as the reason to try `bm25`.

**Acceptance criteria:**
- [ ] Default path = today’s ILIKE RRF
- [ ] `lexical=bm25` uses the Task 3 index
- [ ] Gate §8 on a product LoCoMo arm before considering default flip

**Verification:** McNemar + passage EvR; do not use SOTA compose as the only arm.

**Dependencies:** Task 3, Task 6 recommended  
**Files:** `retrieve.py` `_retrieve_passages`, `requests.py`  
**Scope:** S

### Checkpoint D
- [ ] Search has fusion/linear knobs
- [ ] Memory default still ILIKE unless gate passed and maintainer flips

---

### Phase 4 — Plugins (after core is real)

## Task 10: Search rerank plugin hook

**Description:** Same pattern as `recsys-gnn`: optional package registers `rerank=plugin:cross-encoder` (or SPLADE document expansion). Core works if plugin absent.

**Acceptance criteria:**
- [x] Plugin yaml + route or entrypoint
- [x] Context path cannot call the plugin
- [x] K≤10 documented for CE

**Verification:** Load/unload plugin; 400 on unknown name.

**Dependencies:** Task 4  
**Files:** `plugins/search-rerank/` or `features-search/`, plugin loader  
**Scope:** M

## Task 11: Recs cold-start via search (optional)

**Description:** When graph walks return < k items, fill from `/retrieve/search` on item text / attributes. Flag on recommend request, default off.

**Acceptance criteria:**
- [ ] Default HitRate on `demorecsys` unchanged
- [ ] Flag-on arm recorded separately in `benchmarks.recsys`

**Verification:** `./recsys.sh evaluate` both flags.

**Dependencies:** Task 4  
**Files:** `recommend.py`, recsys harness  
**Scope:** S–M

### Checkpoint E
- [ ] Heavy IR is optional
- [ ] Recs default unchanged

---

## 14. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| HNSW changes memory seeds | High | Overfetch+uuid; EvR agreement gate; exact scan fallback flag |
| BM25 on context looks “free SOTA” | Med | Opt-in; HyperMem is not our judge protocol (`13`) |
| Search API unused; context still scraped | Med | SDKs/docs; chatbot plugin uses search for “find” and context for “answer” |
| FTS tokenization ≠ embed tokenizer | Med | Document analyzer; don’t expect identical hits |
| Scope bleed into ColBERT/ANCE | High | C-NOT / Task 14; plugins only |
| Judge-noise used to “prove” search | High | Separate metrics and brains |

---

## 15. Open questions (maintainer)

Locked 2026-08-18 (see ADR-007 Status):

1. **Routes.** Keep search under `/retrieve` as `GET|POST /retrieve/search`. Do not replace `GET /retrieve/`.
2. **Lexical engine.** Env-gated. When on, Okapi BM25 over Postgres `tsvector` + GIN on `pgvector/pgvector:pg16`. Do not require `pg_search` / `rum`. Do not ship `ILIKE` or `ts_rank_cd` as the search lexical ranker.
3. **3072-d ANN.** Keep float32 vectors. `halfvec` HNSW (≤4000-d) + exact float32 cosine rerank of overfetch. Do not truncate / MRL-cut dimensions.
4. **Chatbot plugin.** Out of scope for v1. No `plugins/chatbot-memory` client.
5. **SLO.** p50 **< 200 ms excluding embed RTT**. `profile_stages` must expose `embed.query` separately from retrieve.
6. **Context vs search ranking.** Search default is both channels fused (`SEARCH_FUSION=rrf`). Context stays today’s dense ∪ ILIKE unless `SEARCH_ENABLED=true` and `CONTEXT_PASSAGE_MODE` is left at `hybrid` (or set to `bm25` / `dense`). `ilike` freezes the memory eval path.

---

## 16. Overall assessment

BrainAPI is already a **knowledge base** with a strong memory assembler and a recs surface that knows how to keep plugins off the default path. It is **not** yet a search engine: there is no hit-list API, no BM25, and default vectors may be unindexed.

The SOTA search brief is useful as a **checklist of primitives** (lexical, dense ANN, optional CE, LTR). It is the wrong **product recipe** if pasted onto `/retrieve/context` (wrong SLO, wrong response shape, wrong eval).

Highest-leverage path that can help search *and* memory without betting LoCoMo: **write a real lexical index and a legal ANN index, batch embeddings, ship `/retrieve/search` that reads them, keep context defaults frozen until a paired gate.** Cross-encoders, SPLADE, ColBERT, HyDE, and ANCE stay plugins or deep-tier — the same split recsys already uses.

**Next action:** maintainer accept/amend ADR-007, then Task 2+3 in parallel (eval skeleton + tsvector), then Task 4 (search API). Label: **protocol development**, not a validated quality claim.

**Status note (2026-08-21):** ADR-007 accepted; `/retrieve/search` shipped. Subsequent workstreams: `18`–`24` (eval, ESCI, WANDS, catalog graph), `25` + [ADR-008](../decisions/008-query-gated-search-personalization.md) (query-gated personalize). §16 above is the 2026-08-18 assessment that motivated the ship. Default omitted `channels=["passages"]` and the 200 ms SLO are unchanged.
