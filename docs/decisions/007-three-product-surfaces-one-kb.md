# ADR-007: Three Product Surfaces on One Knowledge Base

## Status

Accepted — 2026-08-18. Does not supersede ADR-005 or ADR-006. Those remain the memory-path contract. This ADR adds **search** as a first-class product beside memory and recommendations, and states how they share the same brain without sharing request semantics.

Extended by [ADR-008](008-query-gated-search-personalization.md) (Accepted 2026-08-21): optional search `target` reranks retrieved hits. The omitted-`target` path and this ADR’s SLO are unchanged.

Locked §15 resolutions:

1. Routes stay under `/retrieve` as `GET|POST /retrieve/search` (not a new top-level prefix). Existing `GET /retrieve/` is unchanged.
2. Search is env-gated (`SEARCH_ENABLED`, also in TUI init/config). v1 lexical is Okapi BM25 over Postgres `tsvector` + GIN on the stock `pgvector/pgvector:pg16` image — not `ILIKE`, not `ts_rank_cd`, not `rum` / ParadeDB `pg_search`.
3. Dense quality at 3072-d: keep float32 storage; build HNSW on a `halfvec` expression (legal to 4000-d); exact float32 cosine rerank of the overfetch window. Do not truncate embedding dimensions.
4. `plugins/chatbot-memory` is out of scope for v1.
5. Search SLO: p50 **< 200 ms excluding embed RTT**. `profile_stages` splits `embed.query` from retrieve so the budget is measurable.
6. Operators switch dense / BM25 / both via `SEARCH_USE_DENSE` / `SEARCH_USE_BM25`; default is both fused with `SEARCH_FUSION=rrf`. `/retrieve/context` stays ILIKE unless `SEARCH_ENABLED=true` and `CONTEXT_PASSAGE_MODE` is `hybrid` / `bm25` / `dense`.

## Date

2026-08-18

## Context

`docs/research/00-scope-and-constraints.md` already names the product as a memory layer that is also “a substrate for agents, recommenders, and search engines.” In code those are not three products yet:

| Intended product | Shipped surface | What the caller actually gets |
| --- | --- | --- |
| Agent / conversational memory | `POST /retrieve/context` | Prompt-shaped blob: passages + event facts + topics. Budget **p50 < 1000 ms**, no LLM (ADR-006). Evaluated on LoCoMo / LongMemEval / BEAM. |
| Recommendations | `GET|POST /retrieve/recommend` | Ranked **nodes** (items) from graph walks, synergies, optional attribute prefs. Evaluated on `demorecsys` (`16-recsys-eval-protocol.md`). Heavy CF is a plugin (`plugins/recsys-gnn`). |
| Search | **Missing** | `GET /retrieve/` is entity-biased graph+data lookup. Passage “hybrid” in context is dense ANN ∪ **`ILIKE '%query%'`**, fused with RRF — not BM25, not a hit list API, not a search SLO. |

Default pgvector installs ship **3072-d** embeddings and skip HNSW (`vectors.py` returns no DDL above 2000 dims). Context p50 after dossier deletion is **~1.3–1.4 s**, residual dominated by **serial embedding HTTP**, not ANN. Passage evidence-session recall on LoCoMo conv-26 is already **~97%**; graph vs passages-only does not move judge accuracy above the 5–7 point noise floor.

If search is implemented by stuffing more ranking into `/retrieve/context`, two failures follow: (1) memory benchmarks and the sub-second budget become the search SLO, which they cannot be; (2) search users receive a concatenated context blob instead of scored hits with filters, facets, and stable document IDs.

The literature that actually transfers is first-stage IR (BM25, dense dual-encoders, hybrid fusion) and BEIR’s finding that **lexical and dense are complementary off-distribution** — not MS MARCO leaderboard recipes that assume a 200 ms p99 web index and a cross-encoder on every query. Analysis and task breakdown: `docs/research/17-search-surface-and-cross-features.md`.

## Decision

**1. One write path, three read products.**

A brain remains the unique source of truth. Ingest (`POST /ingest/`, structured ingest, observations) writes chunks, embeddings, event graph, topics, and (after this work) a real lexical index **once**. Read products are separate HTTP contracts:

| Product | Canonical API | Response shape | Default SLO | Benchmarks |
| --- | --- | --- | --- | --- |
| Memory | `POST /retrieve/context` | `text_context`, `triples`, passages, topics, paths | p50 < 1000 ms; no LLM | LoCoMo, LongMemEval, BEAM |
| Search | **`POST /retrieve/search`** (new) | Ranked **hits** (id, channel, score, snippet, extras) | p50 < 200 ms excluding embed RTT; p95 budget TBD from telemetry | New `benchmarks.search` (not LoCoMo judge) |
| Recommendations | `GET|POST /retrieve/recommend` | Ranked **nodes** + channel | Unchanged | `benchmarks.recsys` on `demorecsys` only |

MCP / deep REST remains the expensive escalation lane (ADR-006): agentic traversal, cross-encoder, sufficiency retry, HyDE-style query expansion.

**2. Do not overload `/retrieve/context` into a search engine.**

Context stays a **memory assembler** (evidence for an answering model). Search stays a **ranker of addressable objects** (chunks, entities, events, optional structured records). A search client may *call* context afterwards; the APIs do not merge.

**3. Share write-time indexes; isolate query-time defaults.**

Shared (one index, many readers):

- Vector tables + a **legal ANN index** (HNSW or `halfvec` HNSW at 3072-d; loud failure if skipped).
- Lexical postings (`tsvector` / BM25 or equivalent) over chunk text and entity names.
- Provenance, validity (`invalid_at`, `deprecated`), `happened_at`, topic memberships, hub-bridge table.

Isolated (per product, flags on the request, defaults frozen per surface):

- Fusion method and weights (RRF vs convex combination).
- Candidate fanout, channels enabled, graph PPR, fact diversification.
- Rerankers. Core: linear / small GBDT over already-computed scores. Plugins: SPLADE, ColBERT/PLAID, cross-encoder.

`/retrieve/context` **must not** change its default ranking because search shipped. Opt-in only (`lexical=bm25` on context, or a one-way cutover after the non-regression gate).

**4. Core vs plugin, same pattern as recsys.**

| Layer | Owns | Analog |
| --- | --- | --- |
| Core | BM25 (or `tsvector` rank), dense ANN, RRF/CC fusion, filters, cheap feature ranker, search API | `/retrieve/recommend` train-free graph ranker |
| Plugin (optional) | Learned sparse (SPLADE family), late interaction (ColBERT/PLAID), cross-encoder rerank, instruction-tuned embedders | `plugins/recsys-gnn`, `plugins/features-rec` |

A missing plugin degrades search to core hybrid, never 500s the memory path.

**5. Non-regression is a gate, not a hope.**

Any change that touches a **shared** index or the default context path:

1. Paired LoCoMo (or the current memory champion arm) McNemar vs the last accepted product/SOTA run: **no statistically significant accuracy drop**; point estimate not lower. Prefer retrieval-side **passage EvR held** (deterministic) over judge accuracy.
2. Context p50 **not worse** than the post-dossier baseline (~1.4 s) by more than measurement noise; the sub-second target remains ADR-006’s, not this ADR’s.
3. Graph EvR agreement gate from `00` (≥95% identical session sets) still applies if seeds/ANN change.
4. Recsys HitRate on `demorecsys` unchanged unless the change is explicitly a recs feature.
5. Search and recs eval **never** ingest into or wipe `locomoconv26*`, `beam*clean`, or other memory brains.

Search quality is **not** judged by LoCoMo LLM-as-judge. It uses ranking metrics (Recall@k, nDCG@k, MRR) on a dedicated query set.

**6. Cross-features are allowed when they are indexes or cheap scores, not when they are product-specific loops.**

Allowed to help more than one product: BM25, HNSW, batched embeddings, metadata filters, topic coarse-to-fine, recency/validity features, a linear blend of dense+lexical+PPR.

Forbidden on the memory hot path, allowed on search-deep or plugins: per-query LLM (HyDE), cross-encoder over large K, ANCE training loops, ColBERT serving.

## Alternatives considered

### Implement search by adding flags to `/retrieve/context`
- Pros: one endpoint; less API surface.
- Cons: prompt blob ≠ hit list; memory SLO and search SLO conflict; LoCoMo harness would pick up search ranking noise.
- Rejected.

### Separate search database (Elastic/Meilisearch) as source of truth
- Pros: mature BM25.
- Cons: two write paths, drift from the event graph, contradicts “unique source of truth.”
- Rejected as the system of record. An **adapter** that *projects* brain indexes into an external engine is a plugin, not a second brain.

### Put a distilled cross-encoder on every search and context query
- Pros: largest precision jump in the MS MARCO literature.
- Cons: ADR-006 already rejected this on context (50–200 ms, 5–20% of budget). Search can add it as an **optional** second stage with small K.
- Rejected as default; accepted as plugin / `rerank=cross_encoder` on search only.

### Train a BrainAPI-specific bi-encoder (DPR/ANCE) before shipping BM25
- Pros: best dense quality in-domain.
- Cons: needs relevance labels and an index-refresh loop; LoCoMo N cannot support it; BEIR shows off-the-shelf dense is weak zero-shot without lexical backup.
- Rejected as a prerequisite. Off-the-shelf embeddings + real BM25 first.

## Consequences

- New route family under `/retrieve/search*`. Existing `/retrieve/context` and `/retrieve/recommend` keep their contracts.
- Lexical storage and ANN legality become substrate work (`03`), not a search-plugin secret.
- `benchmarks/search/` and `benchmarks.search` in `REPORTS.json` are required before claiming search quality, mirroring recsys isolation.
- Memory SOTA work (`12`, `13`) may **opt into** BM25∪dense RRF on context after the gate — HyperMem (`2604.08256`) already uses that hybrid — but shipping search does not force that opt-in.
- Documentation in `00` gains workstream `17`. Implementation does not start from this ADR alone; tasks live in `docs/research/17-search-surface-and-cross-features.md`.
- Optional query-gated personalize on search (`target`) is [ADR-008](008-query-gated-search-personalization.md). It must not change omitted-`target` ranking, default `channels=["passages"]`, or this ADR’s 200 ms SLO.
