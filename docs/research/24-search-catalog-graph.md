# 24 — Opt-in catalog graph + hybrid search

Workstream: structured catalog ingest + `channels=["communities"]` product retrieve. Ledger: `benchmarks.search` only, **architecture demo**. This is not a quality-default flip and not a claim against frozen WANDS passages nDCG@10 **0.823** or ESCI n=74 **0.500**.

Every claim is labeled **idea**, **assumption**, **prediction**, **located evidence**, or **decision**.

---

## Focal question

Can BrainAPI ship an **opt-in** catalog path: labeled product/hub nodes with `search_text` embeddings + node BM25, hub intersection (CLASS ∩ ATTR), then hybrid-rank **product nodes** inside the communities channel — without changing the live omitted-`channels` default (`["passages"]`)?

**Claim this run can support (located evidence, 2026-08-21):** on brain `searchbenchwandsgraph`, isolated communities nDCG@10 **0.409** / Recall@10 **0.148** / Recall@50 **0.287** / p50 retrieve **5609 ms** (ex-embed); paired passages nDCG@10 **0.823** / Recall@10 **0.269** / Recall@50 **0.837** / p50 retrieve **99 ms**. Architecture demo only.

**Claim the 2026-08-21 join round can support (decision + prediction):** architecture-demo that (a) paired passages nDCG@10 on `searchbenchwandsgraph` stays ~0.823, (b) catalog passage hits expose `node_id` = `doc_id` = ENTITY uuid, (c) that uuid round-trips to `GET /retrieve/entities/neighbors`.

**Claim this run cannot support:** we beat frozen `searchbenchwands` 0.823, ESCI 0.500, Reddy 0.857, or Chen et al. Communities beat passages. We should change the product default. Mixing chunk ids and node ids in default RRF.

---

## Architecture (decision)

One KB, three contracts (ADR-007). Search stays `POST /retrieve/search`. Recs stay `/retrieve/recommend`. Interactions may be ingested onto the same graph; they are not scored as HitRate and `user_id` is not a `SearchRequestBody` field. Optional search `target` (query-gated rerank of retrieved hits) is [ADR-008](../decisions/008-query-gated-search-personalization.md) / [25-personalized-search.md](25-personalized-search.md), not this workstream.

```text
catalog JSONL
  → skip_enrichment chunk ingest (passages)
  → POST /ingest/structured (HAS triples; entity uuid = doc_id)
  → CLASS / ATTR / TYPE hubs + ENTITY products (search_text on nodes)
  → channels=["communities"]: match hubs → intersect if ≥2 hub kinds else union
  → RRF(node BM25 ∪ node dense) on the product set (inside graph_channels)
```

- **Passages stay the live default (decision, 2026-08-21).** Omitted `channels` remains `["passages"]`. First-stage is query → passage. The catalog graph is not a competing candidate generator. `rerank=none`. `search.py` fusion does not mix chunk ids and node ids on the default path.
- **`node_id` is a sidecar, not fusion (decision).** After ranking, catalog passage hits attach `SearchHit.node_id` (and unique `SearchResponse.node_ids` in hit order) by parsing `DOCID <id>` from fetched chunk text. `entity_uuid(doc_id)` is `str(doc_id)`, so the join is deterministic and does not extra-round-trip the graph. Graph-channel item hits set `node_id = id`. Missing marker → `node_id` is null. The field is **`node_id`**, not `product_id` / `sku` / `brand` / `user_id`.
- **Clients use existing graph reads.** `node_id` is the product ENTITY uuid for `/retrieve/entities/neighbors`, `/retrieve/entity/synergies`, `/retrieve/recommend`, entity info. Recsys and memory already consume the graph. This round does not change `/retrieve/context` (ADR-006).
- **Product ENTITY `search_text` is the catalog blob (decision).** Prefer `doc["text"]` when present, else title + class + features + description (plus hierarchy/brand/color when composed). Hub `search_text` stays short name/key text.
- **Static catalog attributes are a direct edge (decision, 2026-08-21).** Mapper `has_triple` writes `ENTITY -HAS-> CLASS|TYPE|ATTR` with no EVENT mid-node. Structured ingest collapses the same shape if an old 5-tuple `HAS` wrapper arrives with no `happened_at`. Keep EVENT for real occurrences (`USER -MADE-> View/Purchase -TARGETED-> PRODUCT`, with `happened_at`). Polarity is not a reason to keep HAS wrappers. **Located evidence:** `searchbenchwandsgraph` was persisted on the 2-edge wrapper (≈32k HAS events); that brain is not rewritten this change.
- **Catalog product path is explicit:** `channels=["communities"]` (optionally `entities`) remains an opt-in architecture demo. Do not re-eval communities nDCG as a quality gate this round.
- **Hubs, not Louvain.** Communities already means TYPE/CLASS/TOPIC/ATTR hubs.
- **Continuous attributes are not hubs.** Price / measures / rating stay node properties when present. Style / color / material / class become ATTR/CLASS hubs.
- **Eval isolation:** brain `searchbenchwandsgraph` only. Never wipe `searchbenchwands`, `searchbenchesci74`, `searchbenchescies`, `searchbenchesciltr2`, `locomoconv*`, `beam*`, `demorecsys`. Frozen JSONL `benchmarks/data/search_wands.jsonl` is reuse-only; do not wipe frozen structured brains. ENTITY text/embed refresh, if run, refuses `FROZEN_STRUCTURED_BRAINS` and does not resubmit Celery architect.

Mapper lives in product code [`src/core/search/catalog_graph.py`](../../src/core/search/catalog_graph.py). The harness re-exports it from [`benchmarks/search/mapping.py`](../../benchmarks/search/mapping.py) (repo root on `sys.path`). `--ingest-graph` / `--interactions` targeting frozen structured brains raises `SystemExit`.

---

## Literature (located evidence, 2026-08-20)

**Support-located for query–document first-stage (not hub-walk):**

- Nigam et al., *Semantic Product Search*, arXiv `1907.00937` (KDD 2019) — dense semantic matching complements lexical; query → document, not query → hub → walk.
- Choi et al., *Semantic Product Search for Matching Structured Product Catalogs*, arXiv `2008.08180` — fielded catalog text plus lexical features.
- Balog / Garigliotti entity-oriented search, arXiv `1802.08010` — query → typed entities → fulfill (hubs then products) is a graph-read pattern, not the live search default.

**Challenge-located / do not overclaim (PKG as the only candidate generator):**

- Xu et al., *Product Knowledge Graph Embedding for E-commerce*, arXiv `1911.12481`, DOI `10.1145/3336191.3371778` — product KG used in ranking **and** recs; does not validate symbolic hub intersection as first-stage IR.
- Ai et al. DREM, arXiv `1909.07212` — user–item KG embeddings for search; learned relations, not HAS expansion.
- Zhang et al. PKGM, arXiv `2105.00388` — PKG for recs/classification, not first-stage nDCG.
- No located paper that community-detection retrieval beats BM25+dense on WANDS/ESCI. Absence from this bound is not “never studied.”

Prior BrainAPI **located evidence:** peer RRF of graph lists with passages **hurt** an ESCI slice (nDCG@10 0.682 vs 0.758 passages). Do not treat extra graph RRF peers as a quality win.

---

## Protocol (decision)

| Knob | Value |
| --- | --- |
| Dataset | Frozen [`benchmarks/data/search_wands.jsonl`](../../benchmarks/data/search_wands.jsonl) (reuse only) |
| Actual n | **66** queries / **2000** docs |
| Brain | `searchbenchwandsgraph` (**new**; never `searchbenchwands`) |
| Ingest | `skip_enrichment` chunks, then `--ingest-graph`. After first eval, `--skip-ingest` |
| Communities | `--channels communities --k 50 --ks 5,10,20,50`, fusion RRF, `rerank=none` |
| Paired passages | same brain, `--channels passages`, `--skip-ingest` |
| Runs | `search-wandsgraph-communities-k50`, `search-wandsgraph-passages-k50` |
| Ledger claim | `architecture-demo` |

```bash
./search.sh --brain searchbenchwandsgraph evaluate \
  --dataset data/search_wands.jsonl \
  --run search-wandsgraph-communities-k50 \
  --channels communities --k 50 --ks 5,10,20,50 \
  --ingest-graph --skip-ingest --timeout 10800
./search.sh --brain searchbenchwandsgraph evaluate \
  --dataset data/search_wands.jsonl \
  --run search-wandsgraph-passages-k50 \
  --channels passages --k 50 --ks 5,10,20,50 \
  --skip-ingest
./search.sh --brain searchbenchwandsgraph smoke \
  --dataset data/search_wands.jsonl \
  --interactions data/search_wands_interactions_smoke.jsonl \
  --channels events --skip-ingest --timeout 1800
```

**Go/no-go for a later default (not executed this round):** communities nDCG@10 ≥ paired passages **on this brain**. **Located evidence:** 0.409 < 0.823 — gate fails. Still must not regress frozen ESCI 0.500 / WANDS 0.823 protocols. Do not quote either row as beating those frozen numbers.

---

## Mapper / graph write (located evidence, 2026-08-20)

Offline count on the frozen JSONL via `docs_to_triples` (not a quality claim):

| Count | Value |
| --- | --- |
| HAS triples | 32208 |
| Product ENTITY nodes | 2000 |
| Hub nodes | 3641 |
| CLASS hubs (unique names) | 157 |
| ATTR hubs (unique names) | 3484 |
| TYPE hubs | 0 (frozen JSONL has no `hierarchy` / `category_hierarchy` field) |
| Docs with `class` | 1863 |
| Docs with `description` field | **0** |
| ATTR `modern` present | yes (`dsprimaryproductstyle : modern` and similar) |

Those 32208 HAS triples were written as **two edges + EVENT mid-node** on `searchbenchwandsgraph`. **Decision (2026-08-21):** new mapper/ingest writes them as one `HAS` edge. Do not treat the live 37383/64416 counts as the post-change shape. Do not wipe that brain to apply this.

**Located limitation (ingest already on disk, 2026-08-21):** the first `searchbenchwandsgraph` persist wrote title-only ENTITY `search_text` (mapper used title + missing `description` field). Frozen WANDS JSONL rows expose `title` / `class` / `features` and a full `text` blob, but not a separate `description` / `hierarchy` field. **Decision:** new mapper writes prefer `doc["text"]` (the same catalog blob passages embed). Already-persisted nodes do not rewrite until an ENTITY-only `search_text` + `nodes` vector refresh (2000 products, not 32k HAS events, not Celery architect). Overwrite of `search_wands.jsonl` remains blocked. Node BM25 after that refresh is **architecture-demo**, not a quality claim vs passages 0.823.

**Assumption:** many furniture queries are class-like; CLASS∩ATTR intersection may rarely fire on this slice. Fixture tests guarantee the operator; WANDS eval is still valid on union+hybrid. Feature keys are capped (`FEATURE_KEY_CAP=16`) and measure/price tokens are skipped; unique ATTR values across 2000 products still numbered in the thousands.

Structured ingest embeds `properties.search_text` into store `"nodes"` (cache key `uuid:{uuid}`). Nodes without `search_text` still embed **name only** (memory path). Node BM25 (`kg_nodes.search_tsv`, english) is gated on `SEARCH_ENABLED` like chunk DDL. Italian node FTS is not added this round.

---

## Isolated architecture eval

**Located evidence (2026-08-21).** Brain `searchbenchwandsgraph`. `rerank=none`, k=50, n=66. Ledger `claim: architecture-demo`. Do **not** quote vs frozen WANDS passages 0.823 or ESCI 0.500.

Live graph after Celery persist (exact counts): **37383** `kg_nodes`, **64416** `kg_relationships` (32208 HAS triples × 2 EVENT-mid edges). `vectors_relationships` leftover HAS from a killed first run is not a success signal.

Ingest harness timed out twice (`1800s` on batch 1; `10800s` on the remaining-triple parent while Celery was still writing). Persist finished in the worker; communities metrics below are **skip-ingest** search on that completed graph. Do not treat harness timeout as a failed graph write.

| Run | Channels | nDCG@10 | Recall@10 | Recall@50 | p50 retrieve ms (ex-embed) |
| --- | --- | --- | --- | --- | --- |
| `search-wandsgraph-communities-k50` | communities | 0.409 | 0.148 | 0.287 | 5609 |
| `search-wandsgraph-passages-k50` | passages | 0.823 | 0.269 | 0.837 | 99 |

ADR-007 architecture SLO: p50 < 200 ms excluding embed. Passages **99 ms** meets it. Communities **5609 ms** misses it (p50 embed 205 ms recorded separately). SLO check, not a quality win.

Interactions smoke (`channels=["events"]`, `--skip-ingest`, synthetic view/cart/purchase on WANDS `doc_id`s `0` / `10` / `13`) **passed** as pipeline-only (`retrieve/search(wands-0) -> 0 hits`). Not mixed into the communities nDCG ledger row. Not recsys HitRate. No `user_id` on the search body.

---

## Passages first-stage + node_id (located evidence, 2026-08-21)

Skip-ingest only on `searchbenchwandsgraph`. No new architect persist. Ledger `claim: architecture-demo`. Do **not** mix with frozen ESCI 0.500.

- Smoke (`channels=["passages"]`, 2 queries): catalog hits expose `node_id` = `doc_id` = ENTITY uuid (example `7468`); `GET /retrieve/entities/neighbors?uuid=7468` returned **200**. Neighbor `count` may be 0 depending on the neighbors filter; 200 is the round-trip gate.
- Skip-ingest k=50 eval `search-wandsgraph-passages-nodeid-k50`: nDCG@10 **0.823**, Recall@50 **0.837**, p50 retrieve **84 ms** (ex-embed). Same nDCG as paired passages `search-wandsgraph-passages-k50`. All 50 hits on `wands-0` had `node_id` matching DOCID. Join is string parse only.

Graph shape at this check: **37393** `kg_nodes` / **64429** `kg_relationships` (~1996 ENTITY / ~31756 EVENT). Close to the persist snapshot; not a rewrite.

ENTITY-only `search_text` + `nodes` vector refresh (`backfill-entity-text` on `searchbenchwandsgraph`, 2026-08-21): **1996** ENTITY nodes updated, **4** missing, **0** skipped. EVENT count stayed **31756**; total nodes stayed **37393**. Postgres node BM25 for `rubberwood` / `modern` hits product ENTITY `0` (`solid wood platform bed`). `channels=["entities"]` query `rubberwood` returned 200 with `node_id == id`. Refuses `FROZEN_STRUCTURED_BRAINS` (`searchbenchwands` exits 1). **Architecture-demo only** — not a quality claim vs passages 0.823.

---

## Passages first-stage + node_id (decision, 2026-08-21)

Recall@50 0.287 vs 0.837 was **candidate generation**, not ranking. Hub-walk cannot recover gold that sits outside the matched CLASS. Fanout/intersection/HAS-ANN remain valid **graph-channel** hygiene, not the search default. Keep `channels=["communities"]` as an opt-in demo; do not treat 0.409 as a quality gate.

**Out of scope this round:** raising communities nDCG@10 from 0.409; flipping the live default; mixed-ID RRF; `product_id` on `SearchRequestBody` / `SearchHit`; wiping frozen brains; editing `retrieve.py` / `entities.py`; a new 20k-triple architect persist.

---

## Explicitly later

- Fusion of passages + communities in `search.py` (needs this `node_id` join, then rank **products**, not mixed chunk/node RRF).
- Communities fanout/union/HAS-ANN hygiene as opt-in channel quality.
- `/retrieve/context` using catalog node text.
- Quality-default flip.
- Range filters for price/measures; SKU/configurable-option child nodes; mood taxonomies beyond feature strings.
- Learned PKG embeddings (Xu/DREM/PKGM) or GNN recs on `/retrieve/search`.
- Query-gated personalized search is a **separate** workstream, already shipped: [25-personalized-search.md](25-personalized-search.md), [ADR-008](../decisions/008-query-gated-search-personalization.md). Not a first-stage change and not a quality-default flip.
