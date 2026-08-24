# ADR-008: Query-Gated Personalized Search on Retrieved Hits

## Status

Accepted — 2026-08-21. Does not supersede ADR-007. Search remains a separate product from recommendations. This ADR locks how identity and taste may touch `/retrieve/search` without sharing recs request semantics or becoming the default ranker.

Locked resolutions:

1. Identity on search is optional `target` (same name as `/retrieve/recommend`). The field is never `user_id`.
2. First-stage stays query → passage (omitted `channels=["passages"]`). Personalize reorders **already retrieved** hits; it does not call `EntityRecommendRetriever.recommend()`.
3. User-asserted `extras` remains a hard AND filter. Inferred taste is blend-only (I7 rejected).
4. HEM / TEM / ZAM training is deferred (I6). WANDS and ESCI cannot supply query×user labels.
5. Omitted-`target` ranking and latency stay the ADR-007 product path (p50 < 200 ms excluding embed). Personalized search is opt-in.

## Date

2026-08-21

## Context

ADR-007 shipped search as a ranked hit list with its own SLO and eval (`benchmarks.search`). Catalog work ([24-search-catalog-graph.md](../research/24-search-catalog-graph.md)) attached `SearchHit.node_id` so a client can walk the same KB. Recommendations already take `target` on `/retrieve/recommend` and are scored as next-item HitRate on `demorecsys` ([16-recsys-eval-protocol.md](../research/16-recsys-eval-protocol.md)).

Three failures follow if personalize is implemented as “recs on the search path”:

1. **Wrong first-stage.** Hub-walk / recommend candidate generation on `searchbenchwandsgraph` recovered Recall@50 **0.287** vs passages **0.837**. That is missing gold, not a ranking defect.
2. **Wrong training data.** HEM (SIGIR 2017, DOI `10.1145/3077136.3080813`) and TEM (arXiv `2005.08936`) need query×user×item logs. Frozen WANDS and ESCI have no users.
3. **Wrong filter.** Writing `extras` from inferred style AND-drops relevant hits the user did not exclude.

A fourth failure is **wrong identity**: a `user_id` field on `SearchRequestBody` forks the recommend contract and invites mixing HitRate with nDCG.

Literature that actually transfers is **query-dependent** personalization — ZAM (arXiv `1908.11322`) and TEM both record that personalization does not always help — not always-on user vectors and not replacing BM25∪dense.

Analysis: [25-personalized-search.md](../research/25-personalized-search.md).

## Decision

**1. Optional `target` on search; omit means anonymous.**

`SearchRequestBody.target` and `GET /retrieve/search?target=` accept a USER uuid or id. Empty / omitted → no personalize call; hit order equals the pre-ship retrieve order. Unknown user → no-op. Schema forbids `user_id`, `product_id`, `sku`, and `brand` on the search body.

**2. Rerank retrieved hits; do not generate candidates.**

Personalize runs after extras AND and `node_id` attach, before the final `k` cut. Catalog `mode` personalizes `k_ret`. Hits without `node_id` get pref 0 and stay in the list. All-zero prefs preserve order. Never drop an id because taste did not match.

Do not call `EntityRecommendRetriever.recommend()`. Score only the given node ids by ATTR-hub overlap with `user_pref_weights`.

**3. Query-gated blend, not always-on taste.**

λ is 0 when the query has digits / SKU-like tokens; otherwise it falls with token count (one token high, four+ low). Blend is min-max retrieve vs pref. This is a cheap stand-in for ZAM/TEM’s “personalize only when the query is under-specified,” not a trained attention model.

**4. Write taste as PREFERS; keep EVENT for dated behavior.**

Interaction `options` / `attributes` emit `USER -PREFERS-> ATTR` with the same hub uuid as catalog HAS. Options are not copied onto the product catalog blob. View / cart / purchase / favorite stay EVENT wrappers with `happened_at`. Architect persists any no-event direct triple, not only static HAS.

Favorite / wishlist weight **0.7**; unknown behaviors **0.2**. Search pref vectors combine write-time PREFERS with recency-decayed EVENT→ITEM→ATTR (14-day half-life) even when PREFERS exist.

**5. Inferred taste is not a filter.**

User-asserted `extras` stays AND. The ranker must not inject `extras={style: …}` from history.

**6. Eval isolation.**

Unit tests in `tests/test_search_personalize.py`. Optional skip-ingest smoke only on `searchbenchwandsgraph`. Never wipe frozen `searchbench*` quality brains, `locomoconv*`, `beam*`, or `demorecsys`. Do not quote vs frozen WANDS **0.823** / ESCI **0.500**. Do not score recsys HitRate as search nDCG. If ledgered: `claim: architecture-demo` only.

## Alternatives considered

### Add `user_id` on `SearchRequestBody`
- Pros: matches ingest JSONL and recsys datasets.
- Cons: two identity names across retrieve; invites mixing HitRate with nDCG.
- Rejected. Reuse `target`.

### Call `/retrieve/recommend` (or `EntityRecommendRetriever`) as search first-stage
- Pros: one ranking implementation.
- Cons: different task (next-item vs query); communities/graph first-stage already lost Recall@50 on the architecture-demo brain.
- Rejected.

### Infer `extras` from PREFERS / recent events
- Pros: “personalize” looks like faceted search.
- Cons: AND-drops relevant hits the caller did not filter; confuses asserted constraints with taste.
- Rejected (I7).

### Train HEM / TEM / ZAM before shipping a personalize knob
- Pros: literature SOTA for personalized product search.
- Cons: no query×user logs on WANDS/ESCI; would not be the 200 ms default; blocks the opt-in rerank that the graph already supports.
- Deferred (I6). Not a prerequisite.

### Always-on personalize whenever `target` is present
- Pros: simpler than λ.
- Cons: ZAM/TEM evidence that specific queries get worse; navigational / SKU queries must stay retrieve-order.
- Rejected.

## Consequences

- `SearchHitScores.personalize` is additive; omitted-`target` responses keep `personalize` unset.
- Core library: `src/core/search/personalize.py`. Recommend candidate generation is unchanged aside from I4 weights.
- Recommend `_attribute_pref_targets` still skips EVENT walks when PREFERS exist; search personalize does not. That split is intentional until recs explicitly adopt the combine.
- Research record: `docs/research/25-personalized-search.md`. ADR-007 SLO and default channels stay the omitted-`target` path.
