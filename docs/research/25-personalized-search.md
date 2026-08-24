# 25 — Query-gated personalized search

Workstream: optional `target` on `/retrieve/search` reranks **already retrieved** hits. Ledger, if used: `benchmarks.search` only, **architecture demo**. This is not a quality-default flip, not recsys HitRate, and not a claim against frozen WANDS passages nDCG@10 **0.823** or ESCI n=74 **0.500**.

Every claim is labeled **idea**, **assumption**, **prediction**, **located evidence**, or **decision**. Binding contract: [ADR-008](../decisions/008-query-gated-search-personalization.md). Catalog graph / `node_id` join: [24-search-catalog-graph.md](24-search-catalog-graph.md). Recs isolation: [16-recsys-eval-protocol.md](16-recsys-eval-protocol.md).

---

## Focal question

Can BrainAPI personalize product search **without** (a) changing first-stage candidate generation, (b) sharing `/retrieve/recommend` request semantics, or (c) turning inferred taste into a hard `extras` filter?

**Claim this ship can support (decision, 2026-08-21):** omit `target` → anonymous ranking unchanged. Set `target` → query-gated blend of retrieve score vs user ATTR prefs over hits that already have `node_id`. Unknown user and all-zero prefs preserve order. `extras` AND stays user-asserted.

**Claim this ship cannot support:** we beat frozen WANDS 0.823 or ESCI 0.500. Recsys HitRate on `demorecsys` is search nDCG. HEM/TEM-class query–user–item training. Flipping live default `channels=["passages"]`. Using `user_id` on `SearchRequestBody`.

---

## Architecture (decision)

```text
query
  → first-stage (default: passages BM25∪dense)
  → extras AND (user-asserted only)
  → node_id attach (DOCID parse / graph hit id)
  → if target set: score given node_ids vs user prefs, blend with λ(query)
  → cut to k
```

- **Identity is `target`, not `user_id` (decision).** Same field name as `/retrieve/recommend`. GET `/retrieve/search?target=` matches the body. Schema tests ban `user_id` / `product_id` / `sku` / `brand` on the search body.
- **Rerank, not candidate generation (decision).** Do not call `EntityRecommendRetriever.recommend()`. Communities as first-stage already failed on `searchbenchwandsgraph` (Recall@50 **0.287** vs passages **0.837**; [24](24-search-catalog-graph.md)). Personalize scores only the retrieved `node_id`s.
- **Query λ (decision).** Digits or SKU-like tokens → 0 (navigational / specific). One content token ≈ 0.85; two ≈ 0.5; three ≈ 0.25; four+ ≈ 0.1. Blend is min-max retrieve vs pref; λ=0 or all-zero prefs preserve retrieve order and never drop ids.
- **Pref vector (decision).** `user_pref_weights` sums write-time `USER -PREFERS-> ATTR` (long-term) **and** recency-decayed `USER → EVENT → ITEM → ATTR` (short-term, 14-day half-life). Search does **not** skip events when PREFERS exist — unlike recommend’s `_attribute_pref_targets`, which still does. Treat ENTITY and PRODUCT as items.
- **Write path (decision).** Dated view / cart / purchase / favorite stay EVENT wrappers. Selected `options` / `attributes` emit `USER -PREFERS-> ATTR` with the **same hub uuid** as catalog HAS (`hub:attr:70s`). Options are not merged into the product `catalog_doc` (would stamp the SKU). Catalog `brand` / `color` / `class` on the row still PRODUCT-HAS. Architect persists any no-event direct triple (`HAS`, `PREFERS`, later `AVOIDS`), not only static HAS.
- **I4 weights (decision).** Favorite / wishlist **0.7**; unknown behaviors **0.2** (was 1.0). Shared by `src/core/search/recommend.py` and `plugins/features-rec/models/mapping.py`.
- **I7 rejected (decision).** Inferred style is blend-only. Do not write `extras={style: 70s}` from history. User-asserted `extras` remains a hard AND filter, applied **before** personalize.
- **SLO (decision).** Omitted-`target` path is the ADR-007 200 ms default. Personalized path is opt-in; do not cite its latency as that SLO.
- **Hits without `node_id` stay at pref 0.** Catalog mode personalizes `k_ret` then cuts to request `k`.

Code: [`src/core/search/personalize.py`](../../src/core/search/personalize.py), wired in [`src/services/api/controllers/search.py`](../../src/services/api/controllers/search.py) after extras / `node_id`. Mapper: [`src/core/search/catalog_graph.py`](../../src/core/search/catalog_graph.py) `prefers_triple` / `interaction_to_triples`.

---

## Literature (located evidence, 2026-08-21)

**Support-located for query-dependent personalization (not always-on):**

- Ai, Hill, Vishwanathan, Croft, *A Zero Attention Model for Personalized Product Search*, arXiv `1908.11322` (CIKM 2019) — commercial logs: personalization **depends on query characteristics**; a zero vector lets the model pay **no** attention to history. Maps to λ=0 on specific / digit queries, not to training ZAM.
- Bi, Ai, Croft, *A Transformer-based Embedding Model for Personalized Product Search*, arXiv `2005.08936` (SIGIR 2020) — TEM: “personalization does not always improve product search quality”; undifferentiated personalization is the failure mode. Maps to query-gated blend. **Does not** authorize training TEM here.

**Challenge-located / deferred (learned query–user–item first-stage):**

- Ai, Zhang, Bi, Chen, Croft, *Learning a Hierarchical Embedding Model for Personalized Product Search*, SIGIR 2017, DOI `10.1145/3077136.3080813` — HEM jointly embeds query, user, item from purchase language. Needs query×user logs. **I6 deferred.**
- Ai et al. DREM, arXiv `1909.07212` — learned user–item KG for search; not ATTR-hub overlap on a retrieved list. Already bounded in [24](24-search-catalog-graph.md).
- Nigam et al., *Semantic Product Search*, arXiv `1907.00937` — first-stage remains query → document. Personalize does not replace it.

**Wrong metric if cited as search quality:**

- Recsys next-item HitRate on `demorecsys` via `/retrieve/recommend` ([16](16-recsys-eval-protocol.md)) is a different API, task, brain, and label. WANDS / ESCI have no users.

---

## Protocol (decision)

| Knob | Value |
| --- | --- |
| Identity | Optional `target`; omit = control |
| First-stage | Unchanged; default omitted `channels=["passages"]` |
| Filter | User-asserted `extras` AND, then personalize |
| Unit tests | `tests/test_search_personalize.py` (λ, blend, PREFERS+recency, extras AND, schema ban) |
| Optional live smoke | skip-ingest on `searchbenchwandsgraph` only; fixture `benchmarks/data/search_personalize_smoke.jsonl` (`wands-u70`, favorite + `style=70s` options). Mapping unit-tested; live ingest is maintainer-gated |
| Ledger claim | If ledgered: `architecture-demo` only |

```bash
# Mapping-only fixture (gitignored under benchmarks/data/*). Compare omitted
# target vs target=wands-u70 on a generic query (lamp) vs a specific query
# (oak dining table 180cm). Skip-ingest on searchbenchwandsgraph after
# maintainer OK; do not wipe or re-architect.
# Fixture: benchmarks/data/search_personalize_smoke.jsonl
```

**Go/no-go for a later default (not executed):** personalized nDCG on a query×user labeled set beats omitted-`target` without regressing specific queries. WANDS/ESCI cannot run that gate. Do not quote vs frozen 0.823 / 0.500.

---

## Isolation

Never wipe `searchbenchwands`, `searchbenchesci74`, `searchbenchescies`, `searchbenchesciltr2`, `locomoconv*`, `beam*`, `demorecsys`. Optional smoke only on `searchbenchwandsgraph`. Do not score recsys HitRate as search nDCG. Do not mix architecture-demo numbers with frozen WANDS / ESCI quality rows.

---

## Explicitly later

- **I6 deferred:** HEM / TEM / ZAM training. Needs real query+user logs, not WANDS. Not in this ship.
- Learned query-conditioned attention (ZAM/TEM) instead of the token-count λ heuristic.
- Recommend `_attribute_pref_targets` combining PREFERS + EVENT (search already does; recs still skip EVENT when PREFERS exist).
- `AVOIDS` polarity as a direct edge.
- Using personalized search latency as the ADR-007 200 ms product claim.
