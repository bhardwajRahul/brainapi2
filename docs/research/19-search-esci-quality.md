# 19 — ESCI slice search quality: retrieval, fusion, and eval

Workstream: `/retrieve/search` ranking quality on `searchbench*` only. Ledger: `benchmarks.search` only. This document is a **proposal and experiment plan**, not a published finding.

**Focal question:** Which interventions could raise nDCG@10, Recall@10, and/or MRR on BrainAPI search for the ESCI product-search slice, and which of those are actually testable given the current eval?

Session purpose: decide the next search-quality experiments. Audience: product owner. Decision owner: the user. Horizon: days.

Every claim is labeled **idea**, **assumption**, **prediction**, **located evidence**, or **decision**.

---

## What this workstream does

Search quality here is labeled product ranking, not LoCoMo judge accuracy and not `/retrieve/recommend`.

### Write path (this slice)

1. Catalog JSONL docs are ingested as chunks with a `DOCID <doc_id>` marker (`benchmarks/search/catalog.py:234-256`, `dataset.py:9-12`). Default ingest is `skip_enrichment=true` (chunk + embed only; `evaluate.py:45-46`, `cli.py:46-47`).
2. Optional `--ingest-graph` writes deterministic HAS triples (`evaluate.py:315-320`). Entity uuid equals catalog `doc_id` (`mapping.py:39-40`). Hubs are `hub:{kind}:{value}` (`mapping.py:43-44`). Class/brand/color/locale/feature values become CLASS / ATTR / TYPE nodes (`mapping.py:180-251`). If no fields exist, a TYPE hub from `dataset` is used (`mapping.py:241-250`).
3. **Located evidence:** this slice’s 200 docs have brand/color/locale in the JSONL and **0/200 `class` fields**. CLASS hubs from mapping are therefore mostly absent; ATTR/TYPE hubs still exist.

### Read path

1. Default omitted `channels` is `["passages"]` (`src/services/api/constants/requests.py:478`; `hooks.py:81-89` treats core names as non-plugins).
2. Passages: dense ANN + BM25, fused with RRF or convex combination (`search.py:176-177, 257-265`; `hybrid.py:109-135`; RRF at `fact_filter.py:67-78`).
3. Graph channels (`graph_channels.py:18-28`):
   - `entities`: name CONTAINS via `graph.search_entities` plus optional node ANN (`graph_channels.py:280-392`). No default `node_labels` filter unless the request sets one (`search.py:111`).
   - `events`: same as entities with `EVENT` labels plus recency (`graph_channels.py:395-433`). Empty-ok on catalog-only brains.
   - `communities`: search TYPE/CLASS/TOPIC hubs (default `SEARCH_COMMUNITY_LABELS`, `config.py:701-709`; `graph_channels.py:30, 564-615`), then expand neighbors with degree-IDF. Member hits can be ATTR hubs because skip labels are only the community labels (`graph_channels.py:588-614`).
4. Each populated graph channel is appended as its own `extra_id_lists` entry and RRF-fused with passage id lists (`search.py:238-265`). IDs are mixed chunk ids and node uuids. A unit test locks this mixing in (`tests/test_search_graph.py:409-416`).
5. Optional `rerank=plugin:<name>` reranks at most `RERANK_MAX_K = 10` on `mode=default` (`hooks.py`; `search.py`). Opt-in `mode=catalog` retrieves `min(200, max(k, 50))` and reranks at most `CATALOG_RERANK_MAX_K = 50`, then cuts to request `k`. `rerank=none` on both ESCI arms unless stated.

### Eval path

1. After ingest (or `--skip-ingest` reuse), chunks are listed and mapped by DOCID substring (`evaluate.py:332-335`, `dataset.py:61-76`).
2. Gold is doc_id space: `gold_hit_ids` unions `gold_doc_ids` / positive `gold_grades` and maps `gold_chunk_ids` through `chunk_to_doc` (`evaluate.py:201-222`).
3. Hits are canonicalized chunk to doc and deduped; unknown ids (including `hub:*`) stay as themselves (`evaluate.py:149-161, 361-362`).
4. Metrics: Recall@{5,10,20}, graded nDCG@10, MRR (`metrics.py:7-67`). nDCG uses `gold_grades` when present (`metrics.py:42-56`).
5. Ledger upserts `benchmarks.search` and now records `channels` (`report.py:61, 157`).

### Located evidence — two arms on the same brain

Same dataset `benchmarks/data/search_esci_slice.jsonl` (200 docs, 11 queries, all product columns), brain `searchbenchesci20`, fusion `rrf`, rerank `none`, skip_enrichment true. Graph arm ingested triples; passages arm used `--skip-ingest` (chunks reused; `eval.json` `ingest.reused=true`).

| Arm | Channels | nDCG@10 | Recall@10 | Recall@20 | MRR | p50 retrieve |
| --- | --- | --- | --- | --- | --- | --- |
| Graph | passages+entities+communities | 0.6815 | 0.4657 | 0.5818 | 0.8485 | 108 ms |
| Passages | passages | 0.7579 | 0.5106 | 0.8470 | 0.8485 | 29 ms |

Sources: `benchmarks/runs/search-esci-slice-allcols/{report,eval}.json`, `benchmarks/runs/search-esci-slice-passages/{report,eval}.json`.

Do **not** claim graph-channel lift. Graph dropped nDCG@10 (−0.076) and Recall@10 (−0.045). MRR was identical on every query. Recall@20 dropped far more (−0.265).

Per-query first-relevant rank never moved. Five graph queries put `hub:attr:*` in the fused top-10. Graph unique hit counts after canonicalize were 12–18 vs 20 for passages. Offline drop of `hub:*` from the already fused graph lists only moved nDCG 0.682 to 0.691 and Recall@10 0.466 to 0.476 (Recall@20 unchanged). **Prediction that failed:** removing visible hubs from the scored list recovers passages. Residual gap is ASIN substitution and a shorter unique product pool created at fusion time.

One query (esci-18) had graph nDCG +0.057 with unchanged Recall@10 because rank-10 swapped a Substitute (gain 0.1) for an Exact (gain 1.0). That is a single-query graded accident, not a mechanism claim.

Both arms score 0 on esci-72 (`$100 things that are not electronics`; 6 Exact golds never retrieved).

Gold sizes: min 1 / median 20 / max 20. Gain histogram on the slice: E=1.0 ×127, S=0.1 ×37, C=0.01 ×2. Recall@10 is therefore capped near 0.5 on the typical query even if top-10 is all gold.

---

## Maintainer decision (2026-08-18)

**Decision (located):** make graph channels useful even if passages stays the default. Headline nDCG vs passages-only is **not** the go/no-go. Isolated-channel gold retrieval is: Recall@10 of gold ASINs and presence of product uuids in hits. Follow-up evals reuse brain `searchbenchesci20` with `--skip-ingest`. Default omitted `channels` remains `["passages"]`.

---

## Isolated channels then ranking change (same brain, n=11)

All arms: `search_esci_slice.jsonl`, `--skip-ingest`, fusion `rrf`, rerank `none`. No re-ingest. `eval.json` now stores per-hit `id`, `channel`, and canonical `doc_id` (`evaluate.py`).

### Before ranking change (measurement)

| Arm | Channels | nDCG@10 | Recall@10 | Recall@20 | MRR | p50 ms | Queries with gold in top-10 | `hub:*` in top-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Passages | passages | 0.758 | 0.511 | 0.847 | 0.848 | 29 | 10/11 | 0 |
| Entities | entities | 0.532 | 0.415 | 0.709 | 0.667 | 53 | 10/11 | 6 queries (13 slots) |
| Communities | communities | 0.000 | 0.000 | 0.000 | 0.000 | 52 | 0/11 | empty lists |
| Passages+entities | passages,entities | 0.682 | 0.466 | 0.582 | 0.848 | 74 | 10/11 | 5 queries |

Sources: `runs/search-esci-slice-{passages,entities,communities,pe}/eval.json`.

**Call (located evidence):** entities-only already retrieved gold ASINs (Recall@10 0.415, 10/11 queries). ATTR hubs stole rank on 6 queries. Communities-only never fired (TYPE/CLASS/TOPIC hubs do not match these queries; this slice has 0/200 `class` fields). Passages+entities matched the original fused graph arm — communities added nothing. Peer RRF of entity lists with passages still shortened the product pool (Recall@20 0.582 vs 0.847).

Offline compacting of `hub:*` from the original fused `hit_ids` (promote later ranks into @10) yields nDCG@10 0.764 / Recall@10 0.476 / Recall@20 0.582 / MRR 0.848 (`python -m search.replay_fusion`). An earlier 0.691 nDCG figure did not refill @10 from ranks 11+. Compact nDCG matching passages is **not** a claim that live fusion is fixed; Recall@20 stays 0.582.

### After ranking change (same frozen corpus)

Code: `graph_channels.py` defaults the entities channel to `ENTITY` nodes, ranks by name overlap, drops `hub:*` / ATTR/TYPE/CLASS members from entity lists. Communities also seed ATTR hubs (brand/color/feature), expand at most 8 hubs, and emit only item ENTITY neighbors. `search.py` will not RRF hub ids into fused lists. Default omitted channels still `["passages"]`.

| Arm | Channels | nDCG@10 | Recall@10 | Recall@20 | MRR | p50 ms | Queries with gold in top-10 | `hub:*` in top-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Passages smoke | passages | 0.758 | 0.511 | 0.847 | 0.848 | 28 | 10/11 | 0 |
| Entities | entities | 0.655 | 0.482 | 0.822 | 0.757 | 51 | 10/11 | 0 |
| Communities | communities | 0.478 | 0.350 | 0.391 | 0.848 | 129 | 10/11 | 0 |
| Fused | passages,entities,communities | 0.653 | 0.497 | 0.653 | 0.816 | 184 | — | 0 |

Sources: `runs/search-esci-slice-{passages-smoke,entities-after,communities-after,pec-after}/eval.json`.

**Useful, in numbers (go/no-go):** isolated entities Recall@10 0.415 → 0.482; zero hub ids in top-10; product uuids on all 11 queries; Recall@20 0.709 → 0.822. Isolated communities Recall@10 0 → 0.350 with gold in top-10 on 10/11 queries (esci-19 still 0). Passages-only smoke is unchanged at 0.758 / 0.511 / 0.848.

**Not a quality win for fused search.** p+e+c nDCG@10 0.653 is still below passages 0.758 (and below the old fused 0.682). MRR fell from 0.848 to 0.816. Do not put entities/communities in the default mix. Do not claim ESCI-wide lift. n=11.

Communities retrieve p50 129 ms is under the ~200 ms SLO; an uncapped hub expansion on this brain was 239 ms p50 and was capped to 8 hubs.

### Still not useful / unmeasured

- **events:** no interaction EVENT ingest on this catalog; still empty-ok.
- **expand=neighbors:** unit-tested; no isolated ESCI arm.
- **esci-72:** entities-only still 0 gold in top-10; passages-only still 0; communities-after had 1/6 gold in top-10 — n=1, not a negation fix.
- **Fused graph+passages:** still crowds Recall@20 vs passages (0.653 vs 0.847).

Reproduce (no re-ingest):

```bash
cd benchmarks
./search.sh --brain searchbenchesci20 evaluate \
  --dataset data/search_esci_slice.jsonl --skip-ingest --fusion rrf \
  --channels entities --run search-esci-slice-entities-after
./search.sh --brain searchbenchesci20 evaluate \
  --dataset data/search_esci_slice.jsonl --skip-ingest --fusion rrf \
  --channels communities --run search-esci-slice-communities-after
./search.sh --brain searchbenchesci20 evaluate \
  --dataset data/search_esci_slice.jsonl --skip-ingest --fusion rrf \
  --channels passages --run search-esci-slice-passages-smoke
```

---

## Passages CE, esci-72, ranking-in-pool (2026-08-18)

**Decision:** default `rerank=none` stays. Isolated graph channels stay off the default mix.

### Passages MiniLM (Task 1)

`runs/search-esci-slice-passages-ce` — `--channels passages --rerank plugin:cross-encoder`, `--skip-ingest`, n=11.

| Arm | nDCG@10 | Recall@10 | Recall@20 | MRR | p50 retrieve |
| --- | --- | --- | --- | --- | --- |
| Passages (control) | 0.758 | 0.511 | 0.847 | 0.848 | 28–29 ms |
| Passages + MiniLM k≤10 | **0.738** | 0.511 | 0.847 | 0.803 | 30 ms |

**Located evidence:** nDCG@10 −0.020 vs control (win rule was +0.02). Recall@20 unchanged. MRR down. p50 retrieve 30 ms passes the ~200 ms labeled gate. p95 client wall ~7.5 s is first-query model load, not retrieve. **Null result.** Do not change the product default.

### esci-72 dump (Tasks 2–3)

Query: `$100 things that are not electronics`. All 6 Exact golds are **in the JSONL**. None appear in passages top-20. Helper: `benchmarks/search/miss.py`.

| ASIN | Title (short) | Marker |
| --- | --- | --- |
| B00TREI0JI | Apple iPad Air 2 (Renewed) | electronics |
| B073R68TSH | Beam Electronics car phone holder | electronics |
| B07NSX2ZBS | FindKey RF key/phone tracker | electronics |
| B07QYLTT7X | Galaxy Watch Active TPU case | electronics |
| B07XG6Y847 | AirPods silicone case | electronics |
| B083S3ZXDF | Potaroma electric flopping fish toy | electronics (electric toy) |

Hits are mostly price tags, gift cards, Pokémon lots, batteries — lexical `$100` / “tags”, not a negation failure. **Decision: qrel mismatch.** Do not rewrite `not electronics` on this slice. No eval-only intervention.

### Ranking-in-pool (Tasks 4–7)

Catalog query rows now store `candidate_doc_ids` / `candidate_grades` (I=0 kept off `gold_grades`). Slice JSONL grew 200 → 365 docs so CE-on-pool has texts for I-labels. Those extra docs were **not** ingested into `searchbenchesci20`.

| Arm | Protocol | nDCG@10 | nDCG@20 | nDCG full | R@10 | MRR | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BrainAPI in-pool | search then keep pool ids | 0.788 | **0.811** | 0.814 | 0.535 | 0.909 | coverage 0.47; ~15.5 pool ids/query missing from brain |
| MiniLM CE-on-pool | harness CE on all labeled pairs | 0.598 | **0.669** | 0.786 | 0.399 | 0.833 | `cross-encoder/ms-marco-MiniLM-L-6-v2`; missing_text=0 |

**Cite Reddy EN CE 0.857 / KDD 0.9043 only against CE-on-pool.** That arm is n=11, zero-shot MS MARCO, not ESCI-finetuned, not the 4,477-query public test. Do not average these rows with slice first-stage 0.758.

Reproduce:

```bash
cd benchmarks
./search.sh --brain searchbenchesci20 evaluate \
  --dataset data/search_esci_slice.jsonl --skip-ingest \
  --channels passages --rerank plugin:cross-encoder \
  --run search-esci-slice-passages-ce
./search.sh --brain searchbenchesci20 evaluate \
  --dataset data/search_esci_slice.jsonl --skip-ingest \
  --channels passages --rank-pool \
  --run search-esci-slice-passages-pool
./search.sh --brain searchbenchesci20 rank-pool-ce \
  --dataset data/search_esci_slice.jsonl \
  --run search-esci-slice-ce-pool
```

`rank-pool-ce` uses the repo `.venv` (sentence-transformers). 74-query ingest is a later human checkpoint on a **new** `searchbench*` brain.

---

## Guarantees and where they break

**Stated guarantee (this workstream):** hybrid search returns a ranked list of catalog items such that graded nDCG@10, Recall@10, and MRR measure product relevance on ESCI-style qrels, with p50 retrieve (ex-embed) under ~200 ms (`18-search-eval-protocol.md` SLO).

Where that guarantee cannot be delivered today:

1. **Fusion pollution (high impact on nDCG and recall).** RRF treats graph node lists as peer ranked lists (`search.py:246-265`). Incompatible ids occupy slots; more importantly, extra lists reorder and crowd out passage ASINs. **Gap**, not a documented trade-off. Tests currently require mixed ids (`test_search_graph.py:409-416`).
2. **Community/entity channels unfiltered (high impact on recall@20).** Entities search all labels (`search.py:111`). Communities emit neighbors of TYPE/CLASS/TOPIC, including ATTR hubs (`graph_channels.py:588-614`). Those ids cannot match gold doc_ids unless they are product entity uuids.
3. **Eval cannot attribute a hit to a channel (high impact on diagnosis).** `eval.json` stores canonicalized `hit_ids` only (`evaluate.py:364-372`). We cannot yet falsify “graph never fired useful hits” vs “graph fired gold ASINs that RRF buried” without a new dump or isolated arms.
4. **n=11, no pre-registered control until this ablation (critical for inference).** Means have no interval. One query (esci-67) has a single gold; one (esci-72) is a total miss. Deliberate slice size for ingest cost; **not** a full-ESCI claim.
5. **MRR is saturated and insensitive here (important).** Mean 0.848 is “first Exact usually at rank 1.” Interventions that only clean ranks 2–10 will not move MRR. Eval artifact, not a ranking win.
6. **skip_enrichment + generic HAS triples (medium).** Graph is attribute hubs, not LLM entity resolution. Deliberate cost trade-off. Do not interpret this graph as GraphRAG.
7. **Title/class BM25 confound (medium, weaker than assumed on this file).** Session prior: class words in titles can inflate BM25. Located evidence: 0/200 docs have a `class` field; titles still contain type words (envelopes, pads, fence). Allcols text also includes brand and bullets. Do not attribute passages quality to graph CLASS hubs.
8. **k=10 vs gold size ~20 (important).** Recall@10 cannot exceed ~0.5–0.67 on most queries. Recall@20 is the more sensitive recall readout on this slice (located evidence: passages 0.847 vs graph 0.582).

---

## Open questions for the maintainer

1. Is the decision criterion “beat passages-only 0.758 / 0.511 / 0.848 on this slice,” or “make graph channels eventually useful even if passages stay the default”? **Answered:** useful isolated channels; passages stays default. Fused nDCG is not go/no-go.
2. May all follow-up arms reuse `searchbenchesci20` with `--skip-ingest` (no re-ingest, no wipe)? **Answered:** yes.
3. When isolated `entities` / `communities` arms show no gold ASINs, should those channels stay off the default search path (`channels=["passages"]`)? **Answered:** they stay off the default path even after they retrieve gold ASINs.
4. Is p50 retrieve ~200 ms ex-embed still binding for `rerank=plugin:cross-encoder` on this brain?
5. Should headline decisions use nDCG@10, or also treat Recall@20 as primary given gold size ≈20?
6. Is `benchmarks/data/search_esci.jsonl` (title-only, 2000 docs) in scope only as a later confirmation condition, not a substitute for this allcols slice?
7. Should `entities` default to `node_labels=["ENTITY"]` in product search, or remain unfiltered so ATTR/TYPE name matches can enter RRF?

---

## Independent ideas (generated before literature)

Origin: AI-assisted, stage `independent` unless marked `post-check`.

| ID | Statement | Cluster | Predicted metric move vs passages | Disconfirming observation |
| --- | --- | --- | --- | --- |
| I01 | Treat passages-only as the quality baseline; do not add graph fusion until pollution is diagnosed | eval / fusion | nDCG/recall stay; graph arm must not be the default | Isolated graph arms beat passages on nDCG and recall without hub ids |
| I02 | Dump per-hit `channel` + run isolated `entities`, `communities`, `passages+entities` on this brain | eval | none (measurement) | Isolated arms empty and fused lists contain no graph-origin gold ASINs |
| I03 | Never emit `hub:*` into `extra_id_lists`; only product ENTITY uuids | fusion pollution | +nDCG, +recall@10/@20; MRR flat | Offline/online filter leaves Recall@20 much less than 0.85 (already partly located) |
| I04 | Force `node_labels=["ENTITY"]` on the entities channel | ingest/graph | +recall if entities retrieve gold ASINs; else null | Entities-only with ENTITY labels has Recall@10 ≈ 0 |
| I05 | Downweight or gate graph lists in RRF (not peer k=60 lists) | fusion | +nDCG if entities have some gold | Weighted RRF still loses Recall@20 to passages |
| I06 | Cross-encoder rerank on passages-only head (k≤10) | precision-at-top | +nDCG; MRR maybe; recall@10 unchanged unless head misses gold | nDCG flat after rerank; or p50 retrieve >200 ms |
| I07 | Handle exclusionary queries (`without` / `not`) | recall | +nDCG/recall on esci-72 and similar | Rewrites still miss the 6 gold ASINs |
| I08 | Fielded / title-weighted BM25 | precision | +nDCG if body tokens pollute | Title-only BM25 ≤ allcols passages |
| I09 | SPLADE / ColBERT first-stage plugins | recall | +recall@10/@20 vs passages | Plugin index + latency cost, no metric lift |
| I10 | Enlarge n + paired tests; keep this slice as a smoke | eval artifact | none (inference) | Larger ESCI still shows the same fusion drop |
| I11 | Passages+entities without communities | fusion | smaller drop than full graph if communities are the pollutant | p+e equals or worse than p+e+c |
| I12 | Raise retrieve `k` then cut @10 | recall (gaming) | +recall@10 by stuffing | nDCG@10 falls; SLO missed — do not use as a quality claim |
| I13 | LLM enrichment / richer triples | ingest | unknown | Cost/latency; skip_enrichment was deliberate |
| I14 | Use graph as expansion after passages, not as RRF peers | fusion | recover passages nDCG; optional recall if expansion adds gold | Expansion still injects non-gold ASINs into @10 |
| I31 | Offline RRF replay from stored per-channel lists (`post-check`) | fusion | distinguishes pollution vs never-fired | Cannot replay until I02 dumps raw lists |

### Criteria (declared before scoring)

Scale 1–5, higher better. Weights (decision aid only; **decision stays null**):

| Criterion | Weight | 1 | 3 | 5 |
| --- | --- | --- | --- | --- |
| information_gain | 3 | No mechanism split | Splits one pair of stories | Splits pollution vs never-fired vs passages headroom |
| relevance_metrics | 3 | Will not move the three metrics vs 0.758/0.511/0.848 | Moves one metric on this slice | Directly targets a measured gap |
| originality | 1 | Standard IR | Known method, new here | No nearby literature located |
| feasibility | 2 | New models / re-ingest | 3–5 files + one eval | Eval-only on `searchbenchesci20` |
| rigor | 2 | Easy to game | Needs care | Clean ablation, hard to HARKing |
| value_if_null | 2 | Null is noise | Null slightly useful | Null decides the next fork |
| latency_slo | 2 | Likely blow ~200 ms p50 retrieve | Borderline | No extra retrieve cost |

Weighted score = 100 × Σ p_j (x−1)/4, p_j = w_j / Σ w. Not a winner picker.

| ID | IG | Rel | Orig | Feas | Rig | Null | Lat | Score | Uncertainty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I02 | 5 | 4 | 3 | 5 | 5 | 5 | 5 | 92 | Low |
| I01 | 5 | 4 | 2 | 5 | 5 | 5 | 5 | 90 | Low |
| I04 | 4 | 4 | 2 | 5 | 4 | 5 | 5 | 82 | Med: depends on search_entities quality |
| I03 | 4 | 5 | 3 | 4 | 4 | 4 | 5 | 82 | Low–med: post-hoc hub drop already weak |
| I10 | 4 | 3 | 2 | 4 | 5 | 5 | 5 | 77 | Low |
| I14 | 4 | 4 | 3 | 3 | 4 | 4 | 3 | 67 | Med |
| I05 | 3 | 4 | 2 | 3 | 3 | 4 | 5 | 63 | Med |
| I07 | 4 | 4 | 3 | 2 | 3 | 4 | 3 | 60 | High transfer |
| I08 | 3 | 3 | 2 | 3 | 3 | 3 | 5 | 55 | Med |
| I06 | 3 | 4 | 2 | 3 | 3 | 3 | 2 | 50 | Lat veto possible |
| I09 | 3 | 4 | 2 | 2 | 3 | 3 | 2 | 47 | Fashionable here |
| I12 | 2 | 3 | 1 | 5 | 1 | 2 | 4 | 42 | Rigor veto |
| I13 | 2 | 2 | 2 | 2 | 2 | 3 | 4 | 40 | Out of first wave |

---

## Adversarial review (shortlist)

**I02 isolated arms.** Falsifier: all isolated graph arms return gold ASINs in top-10 and fused p+e+c still loses — then the bug is specifically three-list RRF, not empty channels. Alternative: `--skip-ingest` passages arm hit a warmer cache (esci-67 passages retrieve 266 ms vs graph 108 ms). Residual: n=11.

**I03 hub filter.** Falsifier: already located — dropping `hub:*` after canonicalize does not recover Recall@20. Alternative: hubs in the scored list are a symptom; the cause is extra RRF lists changing passage ranks. Revise I03 to “do not append non-product lists,” not “strip hubs after the fact.”

**I04 ENTITY-only.** Falsifier: entities-only Recall@10 ≈ 0. Alternative: `search_entities` CONTAINS on long titles already returns products; label filter changes little. Selection: we only observed fused lists, not raw entity lists.

**I06 rerank.** Falsifier: nDCG flat because first-stage already puts Exact at rank 1 (MRR 0.85). Alternative: any nDCG lift is E vs S reordering on n=11. Latency: CE on 10 docs can blow the retrieve budget; call it out. Metric gaming: rerank cannot raise Recall@10 if gold is outside the head.

**I07 negation.** Falsifier: gold products for esci-72 are not matchable in this 200-doc pool. Alternative: conceptual query, not syntactic negation.

**I12 raise k.** Reject as a quality intervention. It games Recall@k without fixing ranking.

---

## Frontier techniques

Search date: **2026-08-18**. Tools: arXiv MCP `search_papers` / `get_abstract`. Queries: `"Shopping Queries Dataset" OR ESCI "product search"`; `"reciprocal rank fusion"`; `ti:"reciprocal rank fusion"`; `"knowledge graph" retrieval product search OR "entity-oriented search"`; `ti:"passage re-ranking" BERT OR ti:SPLADE OR ti:ColBERT`; `query negation retrieval OR exclusionary queries`; plus direct ids `2206.06588`, `1805.07591`, `2404.16130`, `1901.04085`, `2004.12832`, `2107.05720`. Limits: one session, cs.IR-heavy; absence from search is not novelty. Cormack et al. SIGIR 2009 RRF was not located on arXiv in this pass (`search-incomplete` for that specific paper).

### Reciprocal rank fusion of heterogeneous lists

- **Mechanism:** Unweighted RRF sums 1/(k+rank) across lists (`fact_filter.py:67-78`, k=60). BrainAPI appends each graph channel as an extra list (`search.py:246-265`).
- **arXiv:** Canonical 2009 paper not retrieved here. **2503.20698** (Samuel et al., MMMORRF) reports weighted, modality-aware RRF and +37% nDCG@20 vs single-modality on video benchmarks — different domain.
- **Cost:** Unweighted extra lists are cheap; they still change the top-k composition.
- **Fit:** We already fuse incompatible id spaces. Weighted/gated RRF is an adapt if isolated arms show useful product ids. Peer unweighted fusion of hubs is challenge-located by our ablation (graph worse than passages).
- **Verdict:** **adapt** gating/weighting after I02; do not add more unweighted lists.
- **Literature status:** `mixed` (method exists; our setting violated the “compatible lists” assumption).

### ESCI / Shopping Queries Dataset

- **Mechanism:** Difficult Amazon queries; E/S/C/I labels; Task 1 is ranking with gains typically E=1, S=0.1, C=0.01, I=0 (`2206.06588`, Reddy et al.).
- **Reported gain:** Paper is a dataset plus baselines, not a BrainAPI method. About 130k queries / 2.6M judgments, multilingual.
- **Cost:** Full ESCI ingest is large; this slice is 11 queries / 200 docs by design (`catalog.py` caps).
- **Fit:** Our grades match Task 1. Our n does not. Title-only 2000-doc file is a different condition.
- **Verdict:** **adopt** as the label protocol; **reject** treating this slice as a full-ESCI result.
- **Literature status:** `support-located` for metric definitions; `challenge-located` for generalizing n=11.

### Entity-oriented / KG-in-the-ranker (not hub-in-RRF)

- **Mechanism:** EDRM (`1805.07591`, Liu et al.) represents queries/docs with words and entity annotations; KG semantics live in embeddings; ranking is interaction-based, end-to-end.
- **Reported gain:** Commercial search log; paper claims better generalization. Not ESCI.
- **Cost:** Needs entity linking plus a learned ranker, not `hub:attr` uuids in RRF.
- **Fit:** Current mapping emits generic HAS hubs (`mapping.py:180-251`) and fuses raw node ids. That is not EDRM.
- **Verdict:** **reject** as a next experiment; **adapt** later only if entities-only retrieves gold products and we add a ranker that consumes annotations.
- **Literature status:** `mixed`.

### GraphRAG community summaries

- **Mechanism:** `2404.16130` (Edge et al.): LLM-extracted entity graph, community summaries, map-reduce answers for global questions.
- **Reported gain:** Comprehensiveness/diversity on query-focused summarization, ~1M-token corpora — not nDCG of product ASINs.
- **Cost:** LLM index plus per-query summary tokens. Violates skip_enrichment and the search retrieve SLO.
- **Fit:** BrainAPI `communities` are typed catalog hubs, explicitly not Leiden / `kg_topic_sessions` (`18-search-eval-protocol.md`).
- **Verdict:** **reject**. Assumption BrainAPI violates: search is item ranking, not corpus QFS; skip_enrichment forbids the LLM graph.
- **Literature status:** `challenge-located` (wrong task).

### Cross-encoder passage rerank

- **Mechanism:** `1901.04085` (Nogueira and Cho): BERT scores query–passage pairs and reorders a first-stage head. Protocol already has `plugins/search-rerank` via `rerank=plugin:cross-encoder` (`18-search-eval-protocol.md`).
- **Reported gain:** +27% relative MRR@10 on MS MARCO passage vs prior SOTA (2019). Not ESCI; our MRR is already 0.85.
- **Cost:** Second-stage model on ≤10 candidates (`hooks.py:7`). Likely material retrieve latency. Call out if p50 approaches 200 ms.
- **Fit:** Best tested on **passages-only**. Reranking a polluted graph list first is the wrong order.
- **Verdict:** **adapt** as a passages-only arm after I02, with an explicit latency gate.
- **Literature status:** `support-located` for method; `mixed` for expected MRR lift here.

### Learned sparse / late interaction

- **Mechanism:** SPLADE (`2107.05720`, Formal et al.) learns sparse term weights for first-stage ranking. ColBERT (`2004.12832`, Khattab and Zaharia) late-interacts token vectors (MaxSim). Both exist as optional plugins in the protocol.
- **Reported gain:** Competitive with dense/sparse SOTA on MS MARCO-class passage tasks. Extra index and FLOPs.
- **Cost:** Own index (`POST /search-splade/index` / ColBERT index). Can blow the retrieve budget. Fashionable relative to this repo’s measured gap.
- **Fit:** Worth a later passages-only bake-off, not a graph-fusion fix.
- **Verdict:** **reject** for the next few days; **adapt** after passages-only headroom is measured on a larger n.
- **Literature status:** `support-located` as IR methods; low expected gain per implementation cost here.

### Exclusionary / negation queries

- **Mechanism:** ExcluIR (`2404.17288`, Zhang et al.) shows first-stage retrievers fail on “what I do not want” queries; a dedicated train set helps but stays below humans; generative retrieval is claimed stronger.
- **Reported gain:** Architecture-wide struggle on 3,452 annotated exclusionary queries; training helps, gap remains.
- **Cost:** New query rewrite or trained retriever. Latency depends on implementation.
- **Fit:** Several slice queries are `without` / `not` (esci-12/13/18/19/2/34/42/60/67/72). Only esci-72 is a total miss. Do not treat the whole slice as an ExcluIR replica.
- **Verdict:** **adapt** as a targeted error analysis plus optional rewrite **after** fusion diagnosis. Status: `support-located` for the esci-72 failure mode.

### Instruction-following retrievers / metadata-as-text

- **Mechanism:** InF-IR (`2505.21439`) trains instruction-aware embeddings. Metadata-aware RAG (`2601.11863`) finds prefixing metadata in the chunk helps.
- **Fit:** Allcols ingest already prefixes Title/Brand/bullets (`catalog.py:234-256`). InF-style models are a new index.
- **Verdict:** **reject** for this sprint (`mixed` / not needed to explain graph vs passages). Reopen after I02.

### Post-check ideation (after literature)

- I31 (offline RRF replay) added. Weighted RRF (I05) upgraded from “fashionable” to “adapt if I02 shows useful entity ASINs” because **2503.20698** is support-located for *weighted* fusion, not for unweighted extra lists.
- GraphRAG and EDRM were not added as winners. Absence of a “graph always helps product nDCG” paper in this search is not a gap we should fill with more channels.

---

## Critique of the current eval as a study

### Summary

Two arms on one frozen 200-doc / 11-query ESCI-US slice, same brain, same qrels, fusion RRF, no rerank, skip_enrichment. The graph arm added entities+communities and structured HAS triples. The passages arm reused chunks (`--skip-ingest`). Headline: passages 0.758 / 0.511 / 0.848 vs graph 0.682 / 0.466 / 0.848. This is a paired within-corpus contrast, not an RCT and not full ESCI.

### Strengths

- Same brain and corpus; passages arm did not re-ingest (reduces write-path drift).
- Graded nDCG uses ESCI-style gains; gold collapse to doc_id is implemented (`evaluate.py:149-161, 201-222`).
- Latency split retrieve vs embed is recorded; graph p50 108 ms and passages p50 29 ms both sit under the ~200 ms retrieve SLO.
- Ledger now records `channels`, so the two arms cannot be silently compared as if they were the same system.
- Per-query lists exist in `eval.json` even though `report.json` is aggregate-only.

### Concerns

**Critical**

- n=11. No confidence interval, no multiplicity control, no pre-registered primary metric among {nDCG, Recall@10, Recall@20, MRR}. One total-miss query and one single-gold query dominate means.
- No channel-attributed hits. The study cannot yet say whether graph lists contained gold ASINs.
- Graph vs passages is confounded with extra RRF lists, mixed id spaces, structured ingest (graph arm only), and different retrieve-path work. The passages ablation removes the “no control” problem but does **not** isolate a single mechanism.

**Important**

- Recall@10 is a poor recall target when |gold|≈20; Recall@20 tells the real crowding story (0.847 vs 0.582).
- MRR identical on all 11 queries: the study is silent on precision-at-1 differences.
- Title-class BM25 confound remains a plausible passages explanation; this file has no `class` field, so the confound is title/body type words, not mapped CLASS hubs.
- skip_enrichment plus generic HAS triples means a graph loss is not a loss for “KG retrieval” in the EDRM/GraphRAG sense.
- esci-67 passages retrieve 266 ms vs graph 108 ms: cache or load difference; do not over-read p95.

**Minor**

- `report.json` omits per-query rows (they live in `eval.json`).
- Default evaluate `k=20` then metrics cut at 10 (`cli.py:321`, `evaluate.py:277-279`).
- Mixed chunk/node ids are tested as desired behavior (`test_search_graph.py:409-416`).

### Recommendations

1. Keep passages-only as the baseline for quality claims.
2. Next measurements: isolated channels + per-hit channel dump + Recall@20 in the headline table.
3. Do not ship more graph fusion until I02/I11 falsify or confirm pollution.
4. Enlarge n before any “ESCI improved” sentence. Keep title-only 2000-doc as a separate condition.
5. Preregister: primary = nDCG@10 vs passages; secondary = Recall@20; MRR expected flat; latency gate p50 retrieve <200 ms.

### Overall assessment

The numbers **can** support: on this 11-query allcols slice, adding entities+communities via unweighted RRF **hurt** nDCG@10 and recall relative to passages-only, while MRR did not change, and retrieve got slower (108 vs 29 ms p50). They **cannot** support: graph-aware search is worse on ESCI; passages is SOTA; class-word BM25 is proven; GraphRAG failed; or any intervention is validated. Offline hub-drop **can** support that visible `hub:*` ids are not the whole gap. It **cannot** support that graph never produced a useful product id.

---

## Implementation plan

**Architecture decisions**

- Measurement and ablations before ranking-code changes.
- Reuse `searchbenchesci20` with `--skip-ingest`. Never wipe or score `locomoconv*`, `beam*`, `demorecsys`.
- Quality claims compare to **passages-only** 0.758 / 0.511 / 0.848, not to the graph arm.
- Core channel names stay `passages|entities|events|communities`. Default omitted channels stay `["passages"]`.
- No edit to `~/.cursor/plans/graph-aware_search_channels_e0ddd720.plan.md`.

Passages-only on this slice/brain is **done** (located evidence above). It is Task 0, not Task 1.

### Phase 1: Measurement (fail fast)

### Task 1: Persist per-hit channel and raw id before canonicalize

**Description:** Write each hit’s `id`, `channel`, and pre-canonical id into `eval.json` queries so later arms can replay fusion and count graph-origin gold ASINs.

**Acceptance criteria:**

- [x] `eval.json` `queries[i].hits` includes `id`, `channel`, and canonical `doc_id`
- [x] Existing metrics still match a recompute from those hits
- [x] Ledger still upserts only `benchmarks.search`

**Verification:**

- [ ] `cd benchmarks && .venv/bin/python -c "import json; json.load(open('runs/search-esci-slice-passages/eval.json'))"`
- [ ] `pytest tests/test_search_graph.py -q` (no harness import of `src/` from `benchmarks/search`)

**Dependencies:** None

**Files likely touched:** `benchmarks/search/evaluate.py`, `benchmarks/search/report.py`

**Estimated scope:** S

### Task 2: Isolated channel arms on the frozen brain

**Description:** Three `--skip-ingest` evaluates on `searchbenchesci20` / `search_esci_slice.jsonl`: `entities`; `communities`; `passages,entities`. Do **not** pass `--ingest-graph`. Compare to passages-only.

**Acceptance criteria:**

- [x] Three new `runs/search-esci-slice-*` reports with `channels` recorded
- [x] Table of nDCG@10, Recall@10, Recall@20, MRR, p50 retrieve vs passages
- [x] Written call: useful-hits (gold ASINs in isolated lists) vs pollution (fused p+e or p+e+c worse than passages)

**Verification:**

```bash
cd benchmarks
./search.sh --brain searchbenchesci20 evaluate \
  --dataset data/search_esci_slice.jsonl --skip-ingest \
  --channels entities --run search-esci-slice-entities
./search.sh --brain searchbenchesci20 evaluate \
  --dataset data/search_esci_slice.jsonl --skip-ingest \
  --channels communities --run search-esci-slice-communities
./search.sh --brain searchbenchesci20 evaluate \
  --dataset data/search_esci_slice.jsonl --skip-ingest \
  --channels passages,entities --run search-esci-slice-pe
```

**Dependencies:** Task 1 preferred (channel dump); can run without it

**Files likely touched:** none in core (eval only); new run dirs

**Estimated scope:** S

### Task 3: Offline fusion-replay / hub-drop notebook in-repo

**Description:** Script that reads two `eval.json` files (or Task 1 dumps) and recomputes metrics after (a) dropping `hub:*`, (b) intersecting with product ASINs, (c) if raw lists exist, RRF replay without graph lists. Check in under `benchmarks/search/` or a research helper, not repo root.

**Acceptance criteria:**

- [x] Reproduces compact hub-drop Recall@10 0.476 / MRR 0.848 from `search-esci-slice-allcols` (nDCG 0.764 when ranks 11+ refill @10; earlier 0.691 did not)
- [x] Prints per-query deltas vs passages
- [x] Exits 0 on the two existing eval files

**Verification:**

```bash
cd benchmarks && .venv/bin/python -m search.replay_fusion \
  --graph runs/search-esci-slice-allcols/eval.json \
  --passages runs/search-esci-slice-passages/eval.json
```

**Dependencies:** None for (a)(b); Task 1 for (c)

**Files likely touched:** `benchmarks/search/replay_fusion.py` (new), maybe `tests` under `benchmarks/` if present

**Estimated scope:** S

### Checkpoint: After Tasks 1–3

- [x] We can say, in one sentence, whether entities-only retrieved gold ASINs
- [x] We can say whether communities-only retrieved gold ASINs
- [x] We can say whether passages+entities is closer to passages or to p+e+c
- [x] Human reviews before any fusion code change
- [x] If both isolated graph arms have Recall@10 = 0 and no gold ASINs: **stop graph fusion work**; go to Phase 3 (passages headroom)

Entities-only Recall@10 was 0.415 **before** the ranking change (gold ASINs present; hubs stealing slots). Communities-only was 0. Passages+entities equaled p+e+c. Ranking/filter work proceeded; default channels were not changed.

### Phase 2: Fusion hygiene (only if Task 2 shows useful product ids)

### Task 4: Fuse only product ENTITY uuids

**Description:** Before `extra_id_lists.append`, keep ids that are product entities (uuid equals a catalog doc_id / label ENTITY). Do not append ATTR/TYPE/CLASS hub ids. Default omitted channels remain passages.

**Acceptance criteria:**

- [x] Unit test: hub id does not appear in fused output when passages+entities+communities run on a fixture
- [x] Mixed chunk+ENTITY uuid still allowed (existing test can stay for ENTITY ids)
- [x] `--skip-ingest` p+e+c eval: Recall@10 moved toward passages (0.466 → 0.497); nDCG@10 did not (0.682 → 0.653). Default path unchanged.

**Verification:**

- [ ] `pytest tests/test_search_graph.py -q`
- [ ] Repeat Task 2 p+e+c command after the change; compare to passages 0.758 / 0.511 / 0.847

**Dependencies:** Task 2 (go/no-go), Task 1

**Files likely touched:** `src/services/api/controllers/search.py`, `src/core/search/graph_channels.py`, `tests/test_search_graph.py`

**Estimated scope:** M

### Task 5: Optional ENTITY label default on entities channel

**Description:** If Task 2 entities-only is dominated by ATTR hubs, pass or default `node_labels=["ENTITY"]` for catalog search without adding recsys field names to the API.

**Acceptance criteria:**

- [ ] Entities-only eval with `--node-labels ENTITY` recorded
- [ ] If Recall@10 still ≈ 0, document I04 as null and do not change product default

**Verification:** same `./search.sh --brain searchbenchesci20 evaluate --skip-ingest --channels entities --node-labels ENTITY --run search-esci-slice-entities-label`

**Dependencies:** Task 2

**Files likely touched:** none, or `search.py` only if default changes

**Estimated scope:** S

### Checkpoint: After Tasks 4–5

- [ ] Graph fusion is either gated to product ids or explicitly left off the default path
- [ ] No claim of graph lift unless nDCG and Recall@20 both beat passages on this slice
- [ ] p50 retrieve still <200 ms

### Phase 3: Passages headroom (higher leverage than more graph)

### Task 6: Passages-only cross-encoder arm (latency-gated)

**Description:** `--channels passages --rerank plugin:cross-encoder` on the frozen brain. Primary: nDCG@10 vs 0.758. Hard gate: p50 retrieve. If plugin missing, expect HTTP 400, not a fake ranking miss (`hooks.py:103-113`).

**Acceptance criteria:**

- [x] Report written even if status failed (plugin absent)
- [x] If ok: nDCG@10, MRR, Recall@10, p50/p95 retrieve vs passages
- [x] No graph channels on this arm

**Verification:**

```bash
cd benchmarks
./search.sh --brain searchbenchesci20 evaluate \
  --dataset data/search_esci_slice.jsonl --skip-ingest \
  --channels passages --rerank plugin:cross-encoder \
  --run search-esci-slice-passages-ce
```

**Dependencies:** Checkpoint after Phase 1; open question 4

**Files likely touched:** none if plugin already wired

**Estimated scope:** S

### Task 7: Exclusionary miss analysis (esci-72)

**Description:** For esci-72 (and optionally other `without`/`not` queries), dump gold titles vs top-20 passages. Decide whether the miss is pool coverage, lexical negation, or conceptual intent. No production rewrite until the dump exists.

**Acceptance criteria:**

- [x] One page or eval artifact listing the 6 gold ASINs and why they missed
- [x] Binary decision: rewrite experiment vs “not testable on this pool”

**Verification:** python over `eval.json` + `search_esci_slice.jsonl` (no new ingest)

**Dependencies:** None

**Files likely touched:** optional helper under `benchmarks/search/`

**Estimated scope:** S

### Task 8: Protocol — Recall@20 in report + paired note

**Description:** Surface Recall@20 in `report.json` / printed table; add a one-line paired-delta note (graph minus passages) in `report.md`. Do not invent p-values on n=11.

**Acceptance criteria:**

- [ ] `report.json` includes `recall@20`
- [ ] Markdown table shows channels

**Verification:** `./search.sh report --run search-esci-slice-passages`

**Dependencies:** None

**Files likely touched:** `benchmarks/search/report.py`

**Estimated scope:** S

### Checkpoint: Complete

- [ ] Passages-only remains the cited baseline
- [ ] I02 falsification path executed
- [ ] No ranking change shipped without Task 2 go/no-go
- [ ] Human approved before any larger ESCI ingest

---

## Risks

| Risk | Impact | Detection |
| --- | --- | --- |
| HARKing a graph win from esci-18 nDCG | High | Require Recall@20 and ≥2 queries |
| Treating n=11 as ESCI | High | Dataset label + n_queries in every sentence |
| Re-ingest drift on searchbenchesci20 | High | `--skip-ingest` only; never wipe |
| Ledger pollution of other suites | High | Upsert `benchmarks.search` only |
| CE / SPLADE blow 200 ms SLO | Med | Record p50 retrieve; abort claim if ≥200 ms |
| Metric gaming via larger k | Med | Reject I12 as quality |
| Cache confound on latency | Low | esci-67 already odd; compare medians |
| Changing default channels in core | High | Default stays `["passages"]`; test `test_search_graph.py:108` |

---

## Decision log (proposal only)

- Candidates considered: I01–I14, I31, plus rejected GraphRAG/EDRM-now/I12/I13.
- Criteria/weights: table above, set before scoring.
- Literature and review date: 2026-08-18.
- **Decision: useful isolated graph channels; passages remains default.** Accountable human: isolated Recall@10 / gold-ASIN presence is go/no-go; fused nDCG is not.
- Revisit trigger: larger-n ESCI, or an events/expand arm.

---

## Graph fusion hygiene (2026-08-18) — G01–G08

Workstream: optional graph channels in `/retrieve/search` without shrinking unique-product recall or nDCG vs passages-only. Brain `searchbenchesci20`, dataset `benchmarks/data/search_esci_slice.jsonl`, n=11, `--skip-ingest`, `--fusion rrf`, `--rerank none`. Default omitted `channels` stayed `["passages"]`. No production `RERANK_MAX_K` change. Ledger: `benchmarks.search` only.

Predeclared primary: Recall@20 then nDCG@10. Live fusion change only if an arm is not worse than passages on both. Null → G08 (graph as side channel).

### Task 1 — per-channel lists and unique-doc counts (located evidence)

`eval.json` now stores `dense_ids`, `bm25_ids`, `entity_ids`, `community_ids`, `n_unique_docs_raw`, `n_unique_docs_canonical`, and `gold_grades`. Search responses expose `channel_lists` when the API from this checkout is running. Ranking of hits is unchanged.

| Arm | Run | unique-doc@20 raw | unique-doc@20 canonical |
| --- | --- | --- | --- |
| Passages | `search-esci-slice-passages-gate` | 20.0 | **20.0** |
| Fused p+e+c | `search-esci-slice-pec-lists` | 20.0 | **15.3** |

**Located evidence:** passages-only unique-doc@20 stays 20 on every query. Fused p+e+c still drops unique products (mean 15.3) because chunk UUIDs and catalog `doc_id`s occupy separate RRF slots (P1).

### Tasks 2–3 — offline replay (located evidence)

Replay: `python -m search.replay_fusion --mode all` on the Task 1 dumps (`pec-lists` + `passages-gate`), stitching isolated `entities-after` / `communities-after` when a list is missing. Collapse maps every id to `doc_id` **before** RRF. Gated arms used the same dumps; expansion N=10 and graph weights `{0.1, 0.25, 0.5}` were predeclared.

| Arm | nDCG@10 | Recall@10 | Recall@20 | MRR | unique@20 |
| --- | --- | --- | --- | --- | --- |
| Passages | 0.758 | 0.511 | 0.847 | 0.848 | 20.00 |
| G01 collapse-rrf | 0.723 | 0.498 | 0.839 | 0.847 | 20.00 |
| G04 weighted-0.1 | 0.740 | 0.511 | 0.847 | 0.852 | 20.00 |
| G04 weighted-0.25 | 0.725 | 0.498 | 0.847 | 0.852 | 20.00 |
| G04 weighted-0.5 | 0.726 | 0.498 | 0.852 | 0.852 | 20.00 |
| G02 expansion N=10 | 0.758 | 0.511 | 0.650 | 0.848 | 20.00 |
| G05 confirmation | 0.704 | 0.502 | 0.847 | 0.795 | 20.00 |

Same protocol on the earlier isolated-channel dumps (no live `channel_lists`) gave collapse R@20 0.827 / nDCG 0.686 and expansion R@20 0.815 / nDCG 0.758. Winner rule was identical.

**P1 survived as the unique-doc / Recall@20 hole.** Collapse restores unique@20 to 20 and Recall@20 to 0.839 (not ≪ 0.847). **P2/P3 survived as the nDCG hole.** Collapse and every weighted/confirmation mix stay below passages nDCG 0.758. Expansion keeps nDCG but loses Recall@20 (novel graph ASINs displace gold in ranks 11–20).

No arm is not-worse than passages on **both** nDCG@10 and Recall@20.

### Checkpoint — G08 (decision)

**Decision:** G08. Keep graph off the ranking mix; expose it as a side channel. Do not implement G01/G02/G04/G05 in `fuse_passage_lists` / live RRF. Default omitted channels remain `["passages"]`.

This is not a claim that graph lists contain no gold ASINs (isolated entities Recall@10 0.482 still holds). It is a claim that **unweighted or gated fusion does not beat passages on this n=11 slice**.

### Task 5 — live passages gate (located evidence)

`runs/search-esci-slice-passages-gate` — `--channels passages`, `--skip-ingest`.

| Arm | nDCG@10 | Recall@10 | Recall@20 | MRR | p50 retrieve |
| --- | --- | --- | --- | --- | --- |
| Passages (frozen control) | 0.758 | 0.511 | 0.847 | 0.848 | 28–29 ms |
| Passages gate (this run) | 0.758 | 0.511 | 0.847 | 0.848 | 26 ms |

Within noise of the frozen control. p50 retrieve 26 ms passes the 200 ms gate. Fused p+e+c `pec-lists` is unchanged at 0.653 / 0.497 / 0.653, p50 160 ms (still under 200 ms, still worse than passages). Do not treat 0.857 as a number from this workstream (that figure is CE-on-pool only).

### Task 6 — `expand=neighbors` (located evidence)

`runs/search-esci-slice-entities-neighbors` — `--channels entities --expand neighbors`.

| Arm | nDCG@10 | Recall@10 | Recall@20 | MRR | unique@20 | p50 retrieve |
| --- | --- | --- | --- | --- | --- | --- |
| Entities-only (prior) | 0.655 | 0.482 | 0.822 | 0.757 | 20 | 51 ms |
| Entities + neighbors | 0.655 | 0.482 | 0.822 | 0.757 | 20 | **223 ms** |

Recall@10 matches entities-only 0.482. `neighbor_ids` were empty on all 11 queries (adjacent items already in the entity seed set, or dropped by the item filter). p50 223 ms **fails** the 200 ms gate. Stop. Do not fuse neighbors into default search.

### What this does not support

- Product claim of “graph precision” or ESCI-wide lift.
- Changing default search to graph.
- Shipping ID-collapse, weighted RRF, expansion-after, or confirmation as a ranking change.
- n=11 as a published ESCI result.

---

## CE-on-pool / first-stage n=74 (2026-08-18, separate from fusion)

Fusion files were not edited here. Protocol: CE-on-pool vs Reddy **0.857**; first-stage vs **0.495** nDCG@10. n=74 ≠ ~4477.

**Located evidence**

- Live `--skip-ingest --channels passages` on `searchbenchesci74`: nDCG@10 **0.495**, nDCG@20 0.549, R@10 0.377, R@20 0.592, MRR 0.758, p50 62 ms (matches the prior first-stage row).
- Local fielded BM25 / RM3 / SPLADE on the same 2043-doc JSONL: best nDCG@10 **0.465** (SPLADE) / **0.458** (title-boost BM25). All **below** 0.495. First-stage win rule (≥0.515) **failed**.
- 4-class MiniLM-L-12 CE-on-pool (40k US Task-1 train, test qids held out, catalog text, weighted ESCI gains): nDCG@20 **0.678**, full-list nDCG **0.747**. Exceeds our FT **0.669**. Does **not** beat Reddy **0.857**.

Do not mix these rows with slice n=11 0.758. Do not change product default rerank.

---

## Deeper k, then harness CE on retrieved hits (2026-08-18)

**Decision:** Goal 1 is more gold in the retrieved list (Recall@k, pool coverage). Goal 2 is nDCG of that same list. Graph stays off the ranking mix (G08). Production default `k` and `RERANK_MAX_K = 10` were not changed. Do **not** cite Reddy **0.857** on these arms (that figure is CE-on-pool only).

**Control (located evidence):** `searchbenchesci74`, `search_esci_74.jsonl`, `--skip-ingest --channels passages --fusion rrf --k 20`: nDCG@10 **0.495**, nDCG@20 0.549, Recall@10 0.377, Recall@20 0.592 (`search-esci-74-passages-control`). With `--rank-pool`: nDCG@20 0.579, **pool_coverage 0.517**, `missing_from_brain` 0 (`search-esci-74-passages-pool`).

**Predeclared Phase 1 win:** pool_coverage **≥ 0.57** or Recall@50 **≥ 0.65**. nDCG@10 may stay flat. **Phase 2** only after that win: nDCG@10 **≥ 0.515** on the same retrieve-k list, Recall@50 not down vs Phase 1.

### Phase 1 — deeper first-stage retrieve (located evidence)

Same brain, `--skip-ingest`, passages only, `rerank=none`. Mean `n_hits` at k=50 is 50; unique-doc@50 is 50. k=100 unique-doc@100 is 100. p50 retrieve 58–60 ms (under 200 ms).

| Arm | k | Recall@10 | Recall@20 | Recall@50 | Recall@100 | nDCG@10 | nDCG@20 | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control | 20 | 0.377 | 0.592 | — | — | **0.495** | 0.549 | 62 |
| `search-esci-74-passages-k50` | 50 | 0.379 | 0.590 | **0.834** | — | 0.500 | 0.552 | 60 |
| `search-esci-74-passages-k100` | 100 | 0.379 | 0.591 | 0.831 | **0.906** | 0.500 | 0.552 | 58 |

`--rank-pool` (coverage of the labeled candidate list, not Reddy Task 1):

| Arm | k | pool_coverage | missing_from_brain | Recall@20 | Recall@50 | nDCG@10 | nDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `search-esci-74-passages-pool` | 20 | **0.517** | 0 | 0.592 | — | 0.546 | 0.579 |
| `search-esci-74-passages-k50-pool` | 50 | **0.778** | 0 | 0.669 | 0.832 | 0.567 | 0.624 |
| `search-esci-74-passages-k100-pool` | 100 | **0.846** | 0 | 0.696 | 0.906 | 0.574 | 0.635 |

**Phase 1 call:** win. Recall@50 **0.834 ≥ 0.65**; pool_coverage **0.778 ≥ 0.57**. Extra golds sit in ranks 21–50 (Recall@20 is flat vs k=20). nDCG@10 is unchanged (~0.500). k=100 does not raise Recall@50 vs k=50; Recall@100 is 0.906. Do not mix with CE-on-pool **0.695** or Reddy **0.857**.

### Phase 2 — harness CE over stored `hit_ids` (located evidence)

Sibling of `rank_pool.py`: `benchmarks/search/rerank_retrieved.py` reorders the k=50 retrieved list. It does not call production rerank and does not score the full labeled pool. Default API `rerank=none` and `RERANK_MAX_K = 10` unchanged.

| Arm | model | nDCG@10 | nDCG@20 | Recall@10 | Recall@50 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| k=50 first-stage | none | **0.500** | 0.552 | 0.379 | **0.834** | 0.765 |
| `search-esci-74-passages-k50-ce` | MS MARCO MiniLM-L-6 | 0.448 | 0.515 | 0.331 | 0.834 | 0.720 |
| `search-esci-74-passages-k50-ce-l12` | 4-class MiniLM-L-12 e2 | 0.467 | 0.508 | 0.341 | 0.834 | 0.706 |

**Phase 2 call:** null. nDCG@10 did not reach **0.515** (needed +0.02 vs the 0.495 k=20 control). Both CEs **hurt** nDCG@10 vs first-stage k=50. Recall@50 held at 0.834 (same hit set). These rows are first-stage retrieve-then-rerank, not CE-on-pool 0.695 / Reddy 0.857.

**Decision:** keep production search at default k / `RERANK_MAX_K=10`. Deeper retrieve finds more gold; zero-shot and local 4-class CE over that retrieved list do not promote it into the nDCG@10 head.

---

## Gold in small and large k (2026-08-18)

**Focal question:** raise gold in both Recall@10 and Recall@50 on the same 74 queries. Graph off. Production `k` / `RERANK_MAX_K` unchanged. Do **not** cite Reddy **0.857** or CE-on-pool **0.695**.

**Predeclared dual win:** Recall@10 **≥ 0.397** and Recall@50 **≥ 0.854** (both +0.02 vs k=20 / k=50 passages). Secondary nDCG@10 ≥ 0.515.

### Task 1 — miss taxonomy (located evidence)

`miss-strata` on `search-esci-74-passages-k50` / `search_esci_74.jsonl`. Reconciles **74** queries / **1270** golds.

| Stratum | Queries | Gold items |
| --- | --- | --- |
| head-ok (gold in top-10) | 67 | 424 in top-10 |
| rank-too-low (gold in 11–50 only) | 2 | 728 in ranks 11–50 |
| total-miss | 5 | 118 missed |

Total-miss qids: `esci-113`, `esci-177`, `esci-267`, `esci-393`, `esci-72`. Dual-k gap is mostly **rank-too-low golds** (728), not the five total misses. Macro Recall@50 0.834 vs micro miss 118/1270 (9%).

### Task 2 — query-side on pathological total-miss only (located evidence)

Rewrote **only** `esci-113` (HTML unescape) and `esci-267` (strip leading `- *`). Did **not** rewrite `esci-72` (prior qrel mismatch). SQL-ish `esci-177` left unchanged. Live `--skip-ingest` `search-esci-74-passages-k50-qrewrite`:

| Arm | Recall@10 | Recall@50 | nDCG@10 |
| --- | --- | --- | --- |
| k=50 passages | 0.379 | 0.834 | 0.500 |
| qrewrite (2 qids) | 0.379 | 0.834 | 0.500 |

Per-qid Recall@10/@50 for `esci-113` and `esci-267` stayed **0**. Stratum-only null. Global dual win not moved.

### Task 3 — local ANCE dual encoder (located evidence)

Harness MiniLM-L-6 dual encoder, US Task-1 train, test qids held out, hard negatives from stored k=50 non-gold `hit_ids` (bank 1387), 6557 triples, 1 epoch. Encode 2043 JSONL docs locally. Not BrainAPI embeddings. Run `search-esci-74-dense-ance-k50`.

| Arm | Recall@10 | Recall@50 | nDCG@10 |
| --- | --- | --- | --- |
| Passages k=20 control | **0.377** | — | **0.495** |
| Passages k=50 | 0.379 | **0.834** | 0.500 |
| Local ANCE MiniLM | 0.302 | 0.750 | 0.396 |

**Dual-k call:** null. Both recalls moved **down**. Do not fuse this retriever into passages. Do not raise `RERANK_MAX_K`. ColBERT and BGE sidecars were run next (below); both also null.

**Decision:** BrainAPI hybrid passages remain the first-stage control. Matching interventions tried here (query rewrite on 2 qids; MiniLM ANCE on the 2043-doc JSONL) did not put more gold in both small and large k.

---

## Retrieved-neg CE, then ColBERT, then BGE (2026-08-18)

**Focal question:** (1) promote Exact already in hybrid ranks 11–50 via a 4-class CE trained on BM25 retrieved negatives; (2–3) try two first-stage sidecars that are not MiniLM dual-encoder. Graph off. Frozen `searchbenchesci74` / `--skip-ingest`. `RERANK_MAX_K` stayed 10. Do **not** mix these rows with CE-on-pool **0.695** or Reddy **0.857**. n=74, no p-values. All three arms were run even though arm 1 was null.

**Predeclared wins:** arm 1 nDCG@10 **≥ 0.515** and Recall@50 not below **0.834**. Arms 2–3 Recall@10 **≥ 0.397** **and** Recall@50 **≥ 0.854**. Secondary nDCG@10 ≥ 0.515.

### Arm 1 — retrieved-negative 4-class CE (located evidence)

Train lists: US Task-1 small-train, test qids held out (`n_holdout_qids=148`), BM25 top-50 over the train product catalog, unlabeled hits = class **I**. 6000 queries, 300000 pairs, all lists had ≥1 I. Checkpoint `benchmarks/data/models/esci-minilm-l12-retrieved` (`source=retrieved-bm25`, 80k pairs, 1 epoch, unweighted). Eval: harness `rerank-retrieved` on stored k=50 `hit_ids` (`search-esci-74-passages-k50-ce-retrieved`). Not production rerank.

| Arm | nDCG@10 | Recall@10 | Recall@50 | MRR |
| --- | --- | --- | --- | --- |
| k=50 passages | **0.500** | 0.379 | **0.834** | 0.765 |
| retrieved-neg 4-class L-12 | 0.465 | 0.351 | 0.834 | 0.712 |

**Arm 1 call:** null. nDCG@10 **0.465** did not reach **0.515** and is below first-stage 0.500 (same direction as pool-trained / MS MARCO CE on these hits). Recall@50 held at **0.834** (same hit set).

### Arm 2 — ColBERT MaxSim sidecar (located evidence)

In-process `plugins/search-colbert` (`colbert-ir/colbertv2.0`) over the 2043 JSONL docs. Brain id `harness-local-colbert`. Did **not** `POST /search-colbert/index` onto `searchbenchesci74`. Run `search-esci-74-colbert-k50`. Channels `harness-colbert`. Not fused with passages.

| Arm | Recall@10 | Recall@50 | nDCG@10 | p50 encode+retrieve |
| --- | --- | --- | --- | --- |
| Passages k=20 / k=50 | **0.377** / 0.379 | **0.834** | **0.495** / 0.500 | ~60 ms retrieve |
| ColBERT MaxSim | 0.311 | 0.714 | 0.434 | 10063 ms |

**Arm 2 call:** dual-k null. Recall@10 **0.311** < 0.397; Recall@50 **0.714** < 0.854. p50 ~10 s is **labeled**, not a 200 ms product claim, not a stop.

### Arm 3 — zero-shot BGE-base (located evidence)

`BAAI/bge-base-en-v1.5` via `local-dense` on the same 2043 JSONL. **Not MiniLM.** No `query:` prefixes. Run `search-esci-74-bge-base-k50`. No new production brain.

| Arm | Recall@10 | Recall@50 | nDCG@10 |
| --- | --- | --- | --- |
| Passages k=20 / k=50 | **0.377** / 0.379 | **0.834** | **0.495** / 0.500 |
| MiniLM ANCE (prior, not this arm) | 0.302 | 0.750 | 0.396 |
| BGE-base (this arm) | 0.322 | 0.799 | 0.441 |

**Arm 3 call:** dual-k null. Recall@10 **0.322** < 0.397; Recall@50 **0.799** < 0.854. Better than MiniLM ANCE 0.302 / 0.750, still below hybrid passages. p50 encode+retrieve 408 ms.

**Three-arm call:** no arm beat the passages control on its predeclared gate. Keep default omitted `channels` = passages. Do not raise `RERANK_MAX_K`. Do not fuse ColBERT or BGE into passages. Do not wipe `searchbenchesci74`.

---

## Exhaustive catalog rank then two-stage (2026-08-19)

**Focal question:** (A) does scoring **every** JSONL doc with the unweighted 4-class L-12 e2 checkpoint beat hybrid k=50; (B) do complementary sidecar lists raise dual-k recall without diluting Recall@10; (C) ship opt-in `mode=catalog` on `/retrieve/search` without changing the default path. Graph off. Frozen `searchbenchesci74` / `--skip-ingest`. Default `RERANK_MAX_K` stayed **10**. Do **not** mix these rows with CE-on-pool **0.695** or Reddy **0.857**. n=74, no p-values. B and C ran even though A was null.

**Predeclared gates:** A nDCG@10 **≥ 0.515** **and** Recall@10 **≥ 0.397**. B Recall@10 **≥ 0.397** **and** Recall@50 **≥ 0.854**, and Recall@10 not below **0.379**. C (same k=50 hits, deeper rerank): A’s nDCG@10 gate with Recall@50 held at **0.834**.

### A — exhaustive 4-class on 2043 docs (located evidence)

Harness `rank-corpus` scored all 2043 JSONL passages per query with `benchmarks/data/models/esci-minilm-l12-4class-nowt-e2` (74 × 2043 ≈ 151k pairs). Protocol `exhaustive-catalog`. Did **not** POST onto `searchbenchesci74`. Run `search-esci-74-exhaustive-ce`. Channels `exhaustive-4class`. Cut @ 5,10,20,50.

| Arm | nDCG@10 | Recall@10 | Recall@50 | p50 |
| --- | --- | --- | --- | --- |
| k=50 passages | **0.500** | **0.379** | **0.834** | 60 ms retrieve |
| Exhaustive 4-class L-12 e2 | 0.416 | 0.266 | 0.603 | 94 s labeled |

**A call:** null. nDCG@10 **0.416** < **0.515** and below hybrid 0.500. Recall@10 **0.266** < **0.397**. p50 is **labeled**, not a 200 ms product claim. MiniLM 4-class over the full file does not beat hybrid RRF on this slice.

### B1 — U01 unique-gold overlap (located evidence)

`list-overlap` on stored `hit_ids` (`search-esci-74-passages-k50` vs `search-esci-74-bge-base-k50` / `search-esci-74-colbert-k50`). Gold ASINs in sidecar top-50 absent from passages k=50:

| Sidecar | unique gold hits | queries with unique golds |
| --- | --- | --- |
| BGE-base k=50 | **40** | 19 / 74 |
| ColBERT k=50 | **21** | 13 / 74 |

Unique golds were nonzero, so B2 included ColBERT.

### B2 — harness RRF union (located evidence)

RRF of passages k=50 + BGE k=50 + ColBERT k=50. Channels `harness-union`. Not fused into live graph. Run `search-esci-74-union-bge-k50`.

| Arm | nDCG@10 | Recall@10 | Recall@50 |
| --- | --- | --- | --- |
| k=50 passages | **0.500** | **0.379** | **0.834** |
| Union passages+BGE+ColBERT | 0.479 | 0.363 | 0.827 |

**B call:** null. Recall@10 **0.363** is below control **0.379** (graph-style dilution). Dual-k **≥0.397 and ≥0.854** failed. Extra lists steal top-10 slots; unique golds in the tail do not raise Recall@50 here (0.827 vs 0.834). Do not fuse these lists into production ranking.

### C — live `mode=catalog` (located evidence)

Shipped: `SearchRequestBody.mode` default `"default"` (retrieve `k`, rerank `min(10, len)`). `mode=catalog` retrieves `k_ret = min(200, max(k, 50))`, reranks `min(len, CATALOG_RERANK_MAX_K=50)`, cuts to request `k`. Live `--skip-ingest` on frozen `searchbenchesci74`, `SEARCH_RERANK_MODEL` = nowt-e2 (same checkpoint as A, not retrieved-neg). Run `search-esci-74-catalog-ce-k50`. Request `k=50` so first-stage depth matches the k=50 control; this tests **plugin rerank of 50** (not 10).

| Arm | nDCG@10 | Recall@10 | Recall@50 | p50 retrieve | p50 client wall |
| --- | --- | --- | --- | --- | --- |
| k=50 passages (`rerank=none`) | **0.500** | **0.379** | **0.834** | 60 ms | — |
| `mode=catalog` + plugin CE | 0.467 | 0.341 | **0.834** | 59 ms | 877 ms |

**C call:** architecture **shipped**; quality gate **null**. Recall@50 held at **0.834** (same 50-hit retrieve). nDCG@10 **0.467** < **0.515** and below first-stage 0.500 — same direction as harness CE-on-retrieved 4-class L-12 e2 (0.467). Catalog+CE is **not** the ADR-007 200 ms default. Omitted `mode` and `RERANK_MAX_K=10` stay the product path. Do not wipe `searchbenchesci74`.

**A/B/C call:** exhaustive MiniLM cannot beat hybrid on this 2043-doc slice; list union dilutes Recall@10; catalog mode is the two-stage hook. Keep default omitted `channels` = passages. Do not cite Reddy **0.857** against these rows.

---

## Frozen-head cascade (2026-08-19)

**Focal question:** can sidecar unique golds raise Recall@50 if hybrid top-10 is frozen? Graph off. Harness only. Not live fusion. `RERANK_MAX_K` unchanged. Do **not** mix with CE-on-pool **0.695** or Reddy **0.857**. n=74, no p-values.

**Predeclared gate:** Recall@10 **≥ 0.379** (no dilution) **and** Recall@50 **≥ 0.854**.

**Located evidence.** `cascade-lists`: passages ranks 1–10 copied; unique golds from BGE+ColBERT top-50 that were absent from passages k=50 inserted at 11–50; remaining passages tail golds kept before non-golds. Run `search-esci-74-cascade-tail-k50`. Channels `harness-cascade`. Protocol `frozen-head-cascade`. Injected **51** unique gold ASINs (BGE 40 ∪ ColBERT 21).

| Arm | nDCG@10 | Recall@10 | Recall@20 | Recall@50 | nDCG@20 |
| --- | --- | --- | --- | --- | --- |
| k=50 passages | **0.500** | **0.379** | 0.590 | 0.834 | 0.552 |
| RRF union (prior, not this arm) | 0.479 | 0.363 | 0.577 | 0.827 | 0.538 |
| Frozen-head cascade | **0.500** | **0.379** | **0.750** | **0.889** | **0.619** |

**Call:** win. Recall@10 **0.379** matches the passages control to floating point (head frozen). Recall@50 **0.889 ≥ 0.854**. nDCG@10 held; nDCG@20 rose because extras land in 11–20. RRF union failed by stealing the head; cascade uses the same unique golds without that. Not fused into live `/retrieve/search`. Do not raise default `RERANK_MAX_K`. Do not wipe `searchbenchesci74`.

---

## DeBERTa-v3-base 4-class on catalog k=50 (2026-08-19)

**Focal question:** does replacing MiniLM with DeBERTa-v3-base (same 4-class weighted-gain recipe) promote golds already in hybrid ranks 11–50 into the top-10? Graph off. Frozen `searchbenchesci74` / no wipe. Default `RERANK_MAX_K` stayed **10**. Live `mode=catalog` only if harness gates fired. Do **not** mix first-stage rows with CE-on-pool **0.695** / **0.710** or Reddy **0.857**. n=74, no p-values.

**Predeclared first-stage win:** Recall@10 **≥ 0.397** **and** nDCG@10 **≥ 0.515**, Recall@50 held at **0.834**.

**Train (located).** `finetune-4class` `--base microsoft/deberta-v3-base`, US Task-1 train, test qids held out (`n_holdout_qids=148`), 80k pairs, 2 epochs, batch 8, unweighted, max_length 192, label smoothing 0.1. Checkpoint `benchmarks/data/models/esci-deberta-v3-base-4class`. Parent venv needed `sentencepiece` (installed 0.2.2). HuggingFace DeBERTa-v3 tokenizer shipped `model_max_length` as an unbounded sentinel; it was set to **192** to match train. Not v3-large. Not the www ensemble.

**Pool discriminator (ranking-in-pool only; not the R@10 gate).** `rank_pool_4class` run `search-esci-74-ce-pool-deberta-base`. nDCG@20 **0.710** vs MiniLM 4-class e2 **0.695**; full-list nDCG **0.777** vs **0.765**. n=74 ≠ ~4477. Do not cite Reddy **0.857** against this row.

**First-stage harness (primary).** `rerank-retrieved` on stored `search-esci-74-passages-k50` `hit_ids`. Run `search-esci-74-passages-k50-ce-deberta`. Plugin 4-logit softmax gains. No live API.

| Arm | nDCG@10 | Recall@10 | Recall@50 |
| --- | --- | --- | --- |
| k=50 passages | **0.500** | **0.379** | **0.834** |
| MiniLM 4-class e2 on those hits | 0.467 | 0.341 | **0.834** |
| DeBERTa-v3-base 4-class on those hits | 0.510 | 0.363 | **0.834** |

**Call:** first-stage **null**. Recall@10 **0.363 < 0.397** (also below passages 0.379). nDCG@10 **0.510 < 0.515** (above MiniLM 0.467 and slightly above passages 0.500). Recall@50 **0.834** held (same hit set). Live `mode=catalog` was **not** run: MiniLM catalog already matched harness CE-on-retrieved, and both quality gates missed. Cascade remains the Recall@50 win. Do not raise `RERANK_MAX_K`. Do not wipe `searchbenchesci74`.

**Interpretation (not a finding beyond this slice).** Pool nDCG@20 moved up a little while first-stage Recall@10 moved down vs hybrid: extra backbone capacity helped ranking-in-pool more than promoting golds through the hybrid head. Compatible with the alternative that labeled pools (coverage 1.0) are easier than the k=50 retrieved list (coverage 0.778). Zhang **0.90** is still ranking-in-pool + large + ensemble, not this y-axis.

---

## Multilingual first-stage (2026-08-19)

Not a US n=74 quality rerun. Not Reddy 0.857. Details in [22-multilingual-ecommerce-search.md](22-multilingual-ecommerce-search.md).

**Located evidence.** ESCI locales are US/ES/JP (`product_locale` in Reddy HTML). No Italian in ESCI. Amazon-M2 has locale IT but rec/title-generation tasks (`2307.09688`). mMARCO has Italian as MT passages (`2108.13897`), not catalog qrels. arXiv `product search` + Italian + `cs.IR` returned 0.

**Decision.** ES download must not clobber US JSONL. Italian smoke fixture is not ESCI. Product default stays passages + `rerank=none`. `searchbenchesci74` not wiped.

**ES first-stage (located evidence).** Run `search-esci-es-passages-k50`, brain `searchbenchescies`, n=62, 2000 docs, k=50 passages, `rerank=none`. nDCG@10 **0.577**, Recall@10 **0.353**, Recall@50 **0.914**, p50 84 ms. Not US n=74 0.500/0.379. Not Reddy ES 0.849 ranking-in-pool.

---

## Production first-stage arms on US n=74 (2026-08-19)

**Focal question:** can cheap first-stage changes raise Recall@50 and/or nDCG@10 on skip-ingest `searchbenchesci74` without CE-on-retrieved or graph-as-default? Graph off. Frozen brain / no wipe. `RERANK_MAX_K` stayed **10**. Do **not** mix with CE-on-pool **0.710** or Reddy **0.857**. n=74, no p-values.

**Predeclared gates.** Hold: nDCG@10 **≥ 0.500**, Recall@10 **≥ 0.379**, p50 retrieve **< 200 ms** (ex-embed). Recall@50 win: **≥ 0.854** and hold head. nDCG@10 win: **≥ 0.520** and Recall not down. Null → leave hybrid RRF, `rerank=none`.

### Frozen-head cascade (opt-in lists)

**Located evidence.** Helper `frozen_head_merge` (hybrid top-10 frozen; unique extra ids fill 11–50). Replay unit test on stored `search-esci-74-passages-k50` + BGE + ColBERT reproduces Recall@50 **0.889**, nDCG@10 **0.500**, Recall@10 **0.379**. Live `/retrieve/search` uses this merge **only** when plugin retriever lists (or `SEARCH_LITERAL_FILL`) are requested — not default RRF of plugins into the head. Default omitted `channels=["passages"]` unchanged.

**Live n=74 sidecar:** not run. `search-colbert` / `search-splade` load, but the ColBERT in-memory index for `searchbenchesci74` is empty (retrieve returns `[]`). Indexing 2043 chunks at query time was not started. **Decision:** cascade stays harness-only. Honest production Recall@50 on the default path stays **0.834**.

### C1 — `fusion=cc` alpha sweep (null)

Skip-ingest, passages, k=50, `rerank=none`. Request `fusion=cc` + `fusion_alpha`. p50 61–63 ms (hold).

| Arm | nDCG@10 | Recall@10 | Recall@50 | p50 retrieve |
| --- | --- | --- | --- | --- |
| k=50 passages RRF | **0.500** | **0.379** | **0.834** | ~60 ms |
| CC α=0.3 (`search-esci-74-cc-a03`) | 0.483 | 0.376 | 0.832 | 63 ms |
| CC α=0.5 (`search-esci-74-cc-a05`) | 0.486 | 0.377 | 0.832 | 61 ms |
| CC α=0.7 (`search-esci-74-cc-a07`) | 0.493 | 0.378 | 0.834 | 61 ms |

**Call:** null. Best nDCG@10 **0.493 < 0.520**. Recall@10 slightly below hold. Product default stays RRF.

### C2 — title-token literal residual (null)

**Predeclared fusion:** frozen-head (not RRF) so the top-10 cannot be stolen. Env `SEARCH_LITERAL_FILL` default **false**. Live skip-ingest `search-esci-74-literal-fill-k50`.

| Arm | nDCG@10 | Recall@10 | Recall@50 | p50 retrieve |
| --- | --- | --- | --- | --- |
| k=50 passages | **0.500** | **0.379** | **0.834** | ~60 ms |
| Literal fill + frozen head | **0.500** | **0.379** | **0.655** | 49 ms (`search.retrieve` only on this run) |

**Call:** null. Recall@10 held (frozen head). Recall@50 **0.655** dropped because non-gold ILIKE token hits filled 11–50 and evicted hybrid tail golds. Flag stays false. Do not treat this as a Recall@50 win.

### C3 — pairwise LTR on the stored k=50 head (null)

**Focal question:** can a cheap linear RankNet on features already in the hybrid list promote the 728 golds sitting in ranks 11–50 into the top-10, without changing the retrieved set? Graph off. Frozen `searchbenchesci74` / no wipe / skip-ingest. Not live `/retrieve/search`. Do **not** mix with CE-on-pool **0.710** or Reddy **0.857**. n=74, no p-values.

**Protocol (located evidence).** `./search.sh ltr-head --from-run search-esci-74-passages-k50 --dataset data/search_esci_74.jsonl --run search-esci-74-ltr-head-k50`. Query-grouped 5-fold CV (OOF predictions; seed=0, 40 epochs, lr=0.05, l2=1e-3, max 400 pairs/query). Features: `rrf_inv`, `bm25_inv`, `dense_inv`, `title_overlap`, `brand_hit`, `query_in_title`. Unlabeled docs in the 50 scored as 0. Overlap-only sort stored as `overlap_only_metrics`, not the gated arm.

| Arm | nDCG@10 | Recall@10 | Recall@20 | Recall@50 | MRR |
| --- | --- | --- | --- | --- | --- |
| k=50 passages RRF | **0.500** | **0.379** | 0.590 | **0.834** | 0.765 |
| Overlap-only (diagnostic) | 0.476 | 0.339 | 0.580 | 0.834 | 0.732 |
| LTR OOF (`search-esci-74-ltr-head-k50`) | 0.515 | 0.384 | 0.583 | **0.834** | 0.753 |
| Predeclared win | **≥ 0.520** | **≥ 0.397** | — | hold | — |

Mean CV weights (not a quality claim): `title_overlap` 3.35, `dense_inv` 2.06, `rrf_inv` 1.57, `brand_hit` 0.51, `bm25_inv` 0.02, `query_in_title` ~0.

**Call (decision):** null. nDCG@10 **0.515 < 0.520**. Recall@10 **0.384 < 0.397**. Recall@50 held. Directional nDCG lift is small and is **not** promotion of the 728 (Recall@10 +0.005; Recall@20 and MRR slightly down vs RRF). Overlap-only is worse than RRF — naive title ranking is not the product. Default stays hybrid RRF, `rerank=none`. Do not wire this ranker into `search.py`.

### C4 — LTR + 4-class CE as a feature (harness win; not live)

**Focal question:** can blending the already-trained 4-class MiniLM score with hybrid RRF via query-grouped LTR raise nDCG@10 on the frozen k=50 list, with Recall@10 held? Graph off. Frozen `searchbenchesci74` / no wipe. Not live `/retrieve/search`. Do **not** mix with CE-on-pool **0.710**, Reddy **0.857**, or CE-as-sole-ranker (located evidence: Recall@10 0.331–0.351). n=74, no p-values.

**Protocol (located evidence).** `./search.sh ltr-head --from-run search-esci-74-passages-k50 --ce-model data/models/esci-minilm-l12-4class-nowt-e2 --pair-policy other_query_neg --run search-esci-74-ltr-cefeat-k50`. Same hypers as C3 (seed=0, 40 epochs, lr=0.05, l2=1e-3, max 400 pairs/query, 5-fold OOF). Feature `ce_gain` = 1·P(E)+0.1·P(S)+0.01·P(C); lists are not sorted by CE. Pair policy: this-query graded golds vs other-query golds in the 50; unlabeled skipped. Predeclared this round: nDCG@10 ≥ **0.520**, Recall@10 ≥ **0.379**, Recall@50 held. Recall@10 ≥ 0.397 is not required to call the harness win.

| Arm | nDCG@10 | Recall@10 | Recall@20 | Recall@50 | MRR |
| --- | --- | --- | --- | --- | --- |
| k=50 passages RRF | 0.500 | 0.379 | 0.590 | **0.834** | 0.765 |
| CE-as-sole-ranker (prior, MiniLM on retrieved) | 0.448 | 0.331 | — | 0.834 | 0.720 |
| LTR OOF no CE (`search-esci-74-ltr-head-k50`) | 0.515 | 0.384 | 0.583 | **0.834** | 0.753 |
| LTR + `ce_gain` (`search-esci-74-ltr-cefeat-k50`) | **0.524** | 0.383 | 0.569 | **0.834** | **0.779** |
| Predeclared this round | **≥ 0.520** | **≥ 0.379** | — | hold | — |

Mean CV weights (not a quality claim): `ce_gain` 3.68, `title_overlap` 2.56, `dense_inv` 1.23, `rrf_inv` 0.66, `brand_hit` 0.45, `bm25_inv` −0.13, `query_in_title` ~0.

**Call (located evidence + decision):** harness win on the predeclared nDCG/hold. nDCG@10 **0.524 ≥ 0.520**. Recall@10 **0.383 ≥ 0.379**. Recall@50 held. MRR 0.765 → 0.779. LightGBM not run (gated only if nDCG close-but-short). **Do not wire into `search.py` without human review:** live CE on 50 hits is extra latency vs the ~60 ms retrieve path; n=74 has no CI; Recall@20 fell 0.590 → 0.569. CE-alone remains a null. Default stays hybrid RRF, `rerank=none` until review.

**A/B/C call:** Italian OR-FTS is a pipeline feature on an allowlisted searchbench. US n=74 live default is unchanged: hybrid BM25+dense, passages, `rerank=none`. Harness nDCG@10 0.524 is not Reddy **0.857**. Later DeBERTa RankNet blend (C5) is the current harness best at **0.542**; still not live.

### C5 — DeBERTa `ce_gain` LTR blend (harness win vs MiniLM; not live)

**Focal question:** on frozen [`search-esci-74-passages-k50`](../../benchmarks/runs/search-esci-74-passages-k50), does replacing MiniLM `ce_gain` with the existing DeBERTa-v3-base 4-class checkpoint raise first-stage nDCG@10 **above 0.524**, with Recall@10 ≥ **0.379** and Recall@50 **0.834** held? Graph off. Frozen `searchbenchesci74` / no wipe. Not live `/retrieve/search`. Do **not** mix with Reddy **0.857**, pool nDCG@20 **0.710**, or DeBERTa-as-sole-sorter (located: nDCG@10 **0.510**, Recall@10 **0.363**). n=74, no p-values. Horizon nDCG@10 ≥ **0.70** is not a fail of this arm.

**Protocol (located evidence).** `./search.sh ltr-head --from-run search-esci-74-passages-k50 --ce-model data/models/esci-deberta-v3-base-4class --pair-policy other_query_neg --run search-esci-74-ltr-deberta-k50`. Same RankNet hypers as C3/C4 (seed=0, 40 epochs, lr=0.05, l2=1e-3, max 400 pairs/query, 5-fold OOF). Lists are not sorted by DeBERTa. Scores cached at `runs/search-esci-74-passages-k50/ce_gain_esci-deberta-v3-base-4class.json` (MiniLM cache left in place). Tokenizer already had `model_max_length=192`; `load_4class_predict` was not edited. Predeclared this round: nDCG@10 **> 0.524**, Recall@10 ≥ **0.379**, Recall@50 held.

Gated I03 (beat 0.524, far from 0.70, `|ce_gain|` not ~0): LightGBM lambdarank, frozen `n_estimators=100`, `max_depth=3`, `learning_rate=0.05`, same CV/features/pair policy, run `search-esci-74-ltr-deberta-lgbm-k50`. No hypersearch.

| Arm | nDCG@10 | Recall@10 | Recall@20 | Recall@50 | MRR |
| --- | --- | --- | --- | --- | --- |
| k=50 passages RRF | 0.500 | 0.379 | 0.590 | **0.834** | 0.765 |
| DeBERTa as sole sorter (prior) | 0.510 | 0.363 | — | 0.834 | — |
| LTR + MiniLM `ce_gain` (`search-esci-74-ltr-cefeat-k50`) | 0.524 | 0.383 | 0.569 | **0.834** | 0.779 |
| LTR + DeBERTa `ce_gain` RankNet (`search-esci-74-ltr-deberta-k50`) | **0.542** | **0.387** | 0.586 | **0.834** | 0.774 |
| LightGBM + DeBERTa `ce_gain` (`search-esci-74-ltr-deberta-lgbm-k50`) | 0.516 | 0.379 | 0.592 | **0.834** | 0.710 |
| Must-beat this round | **> 0.524** | **≥ 0.379** | — | hold | — |
| Horizon (not a fail) | **≥ 0.70** | — | — | — | — |
| Oracle on these 50 | 0.876 | 0.548 | — | 0.834 | — |

Mean RankNet CV weights (not a quality claim): `ce_gain` 3.65, `title_overlap` 2.43, `dense_inv` 1.26, `brand_hit` 0.51, `rrf_inv` 0.45, `query_in_title` 0.02, `bm25_inv` −0.08.

Mean LightGBM gain importances (not a quality claim): `ce_gain` 646, `dense_inv` 561, `rrf_inv` 337, `title_overlap` 162, `brand_hit` 82, `bm25_inv` 19, `query_in_title` 0. LightGBM Recall@10 unrounded **0.3788** (quoted 0.379 in three decimals; hold is strict ≥ 0.379).

**Call (located evidence + decision):** I01 **win vs MiniLM**. nDCG@10 **0.542 > 0.524**. Recall@10 **0.387 ≥ 0.379**. Recall@50 held. Horizon 0.70 missed (~0.16 short; k=50 oracle 0.876). I03 LightGBM **null**: nDCG@10 **0.516** loses to RankNet DeBERTa and to MiniLM 0.524; Recall@10 misses the hold unrounded; `ce_gain` is live, so the miss is the GBDT head, not a dead feature. **Do not wire into `search.py`:** live DeBERTa on 50 hits is extra latency vs ~60 ms retrieve; n=74 has no CI; 0.542 vs 0.524 is a point-estimate gate, not a significance test. Default stays hybrid RRF, `rerank=none` until human review. Do not replace the MiniLM cache.

**I04 (this session):** both heads missed 0.70 with large `ce_gain`. Train queries on matched hybrid k=50 were run on a new `searchbench*` (C6).

**A/B/C call:** US n=74 live default unchanged. Harness best after C5 was DeBERTa RankNet CV **0.542**; C6 apply-from-train is **0.544**. Still not Reddy **0.857** and still not product ranking.

### C6 — LTR trained on 170 matched hybrid lists, applied to frozen n=74 (harness win vs 0.542; not live)

**Focal question:** does fitting RankNet+DeBERTa `ce_gain` on **matched hybrid k=50** from extra US train queries, then applying those weights to frozen [`search-esci-74-passages-k50`](../../benchmarks/runs/search-esci-74-passages-k50), raise nDCG@10 **above 0.542**, with Recall@10 ≥ **0.379** and Recall@50 **0.834** held? Graph off. Frozen `searchbenchesci74` / no wipe. New brain only. Not live `/retrieve/search`. Do **not** mix with Reddy **0.857**, pool nDCG@20 **0.710**, or official-pool LTR. n=74, no p-values. Horizon nDCG@10 ≥ **0.70** is not a fail of 170q.

**Protocol (located evidence).** Download `--split train --max-queries 200 --max-docs 4000 --out data/search_esci_ltr200.jsonl --holdout-dataset data/search_esci_74.jsonl` (did not clobber `search_esci.jsonl` / `_74`). Doc budget bound the slice to **170** queries / **4000** docs; qid overlap with the 74 is **0**. Ingest+evaluate passages k=50, `fusion=rrf`, `rerank=none`, skip_enrichment, brain **`searchbenchesciltr2`** (first `searchbenchesciltr` attempt timed out queued before the Celery worker was up; did not retry on `searchbenchesci74`). Train first-stage diagnostic (different catalog, not the gate): nDCG@10 **0.387**, Recall@10 **0.367**, Recall@50 **0.744**, p50 retrieve 76 ms (`search-esci-ltr200-passages-k50`).

Then `./search.sh ltr-head --from-run search-esci-74-passages-k50 --train-from-run search-esci-ltr200-passages-k50 --train-dataset data/search_esci_ltr200.jsonl --ce-model data/models/esci-deberta-v3-base-4class --pair-policy other_query_neg --run search-esci-74-ltr-deberta-train200`. Fit **once** on 170 train lists (no CV on the 74). Reused frozen-74 DeBERTa cache; scored train lists to `runs/search-esci-ltr200-passages-k50/ce_gain_esci-deberta-v3-base-4class.json`. MiniLM cache left in place. RankNet hypers unchanged. LightGBM not re-run.

**Assumption (catalog shift):** train lists come from a 4000-doc train-qrel index; test lists stay on the 2043-doc `searchbenchesci74` index. Feature space matches; IDF/dense geometry may not.

| Arm | nDCG@10 | Recall@10 | Recall@20 | Recall@50 | MRR |
| --- | --- | --- | --- | --- | --- |
| k=50 passages RRF | 0.500 | 0.379 | 0.590 | **0.834** | 0.765 |
| LTR + DeBERTa RankNet CV (`search-esci-74-ltr-deberta-k50`) | 0.542 | 0.387 | 0.586 | **0.834** | 0.774 |
| LTR + DeBERTa applied from 170 train lists (`search-esci-74-ltr-deberta-train200`) | **0.544** | **0.391** | 0.585 | **0.834** | 0.762 |
| Must-beat this round | **> 0.542** | **≥ 0.379** | — | hold | — |
| Horizon (not a fail) | **≥ 0.70** | — | — | — | — |

Mean applied RankNet weights (not a quality claim): `ce_gain` 4.27, `title_overlap` 3.12, `rrf_inv` 1.40, `dense_inv` 1.31, `bm25_inv` 0.49, `brand_hit` −0.23, `query_in_title` −0.22.

**Call (located evidence + decision):** I04 **win vs I01** on the predeclared point-estimate gate. nDCG@10 **0.544 > 0.542**. Recall@10 **0.391 ≥ 0.379**. Recall@50 held. Horizon 0.70 missed (~0.16 short). Lift vs CV DeBERTa is **+0.002** nDCG / **+0.004** Recall@10; n=74 has no CI, so this can be noise. Train Recall@50 **0.744** shows the 170q lists are not empty of golds, but the catalog is not the 74-index. **Do not wire into `search.py`.** Default stays hybrid RRF, `rerank=none`. Next gated step if chasing 0.70 is 400q matched lists (new `searchbench*` only), not LightGBM on these 170.

**A/B/C call:** US n=74 live default unchanged. Harness best is **0.544**, still not Reddy **0.857**.

### C7 — LightGBM applied from 170 matched hybrid lists (null vs RankNet 0.544; not live)

**Focal question:** on frozen [`search-esci-74-passages-k50`](../../benchmarks/runs/search-esci-74-passages-k50), does LightGBM lambdarank fit on the 170 train hybrid lists beat RankNet apply nDCG@10 **0.544**, with Recall@10 ≥ **0.379** and Recall@50 **0.834** held? Graph off. Frozen `searchbenchesci74` / no wipe. Not live `/retrieve/search`. Do **not** mix with Reddy **0.857** or pool nDCG@20 **0.710**. n=74, no p-values. Horizon nDCG@10 ≥ **0.70** is not a fail of this arm. Predeclared stop: nDCG@10 ≥ **0.58** would halt CE retrain.

**Protocol (located evidence).** `./search.sh ltr-head --from-run search-esci-74-passages-k50 --dataset data/search_esci_74.jsonl --train-from-run search-esci-ltr200-passages-k50 --train-dataset data/search_esci_ltr200.jsonl --ce-model data/models/esci-deberta-v3-base-4class --pair-policy other_query_neg --ltr-model lightgbm --run search-esci-74-ltr-deberta-lgbm-train200`. Same DeBERTa caches as C6. Frozen LightGBM hypers. No hypersearch.

| Arm | nDCG@10 | Recall@10 | Recall@20 | Recall@50 | MRR |
| --- | --- | --- | --- | --- | --- |
| k=50 passages RRF | 0.500 | 0.379 | 0.590 | **0.834** | 0.765 |
| LightGBM 5-fold on the 74 (`search-esci-74-ltr-deberta-lgbm-k50`) | 0.516 | 0.379 | 0.592 | **0.834** | 0.710 |
| RankNet applied from 170 (`search-esci-74-ltr-deberta-train200`) | **0.544** | **0.391** | 0.585 | **0.834** | 0.762 |
| LightGBM applied from 170 (`search-esci-74-ltr-deberta-lgbm-train200`) | 0.533 | 0.389 | 0.591 | **0.834** | 0.752 |
| Must-beat this round | **> 0.544** | **≥ 0.379** | — | hold | — |
| Horizon (not a fail) | **≥ 0.70** | — | — | — | — |

Mean LightGBM gain importances (not a quality claim): `ce_gain` 1684, `dense_inv` 1151, `rrf_inv` 998, `title_overlap` 273, `bm25_inv` 41, `brand_hit` 7, `query_in_title` 0.

**Call (located evidence + decision):** I-LGBM-APPLY **null vs C6 RankNet**. nDCG@10 **0.533 < 0.544**. Recall@10 **0.389 ≥ 0.379**. Recall@50 held. Stop bar 0.58 not reached. Horizon 0.70 missed. Linear vs tree blender on these 170 lists is not the unused 0.33. **Do not wire into `search.py`.** Continue to hybrid-list CE as `ce_gain` (C8).

**A/B/C call:** US n=74 live default unchanged. Harness best remains RankNet apply **0.544**, still not Reddy **0.857**.

### C8 — hybrid-list DeBERTa as `ce_gain` (null vs C6 0.544; not live)

**Focal question:** does continuing the 4-class DeBERTa on **matched hybrid k=50** lists (170 queries, unlabeled=`I`), then using the new scores as `ce_gain` in RankNet apply, beat nDCG@10 **0.544**, with Recall@10 ≥ **0.379** and Recall@50 **0.834** held? Graph off. Frozen `searchbenchesci74` / no wipe. Not live `/retrieve/search`. Do **not** mix with Reddy **0.857**, pool nDCG@20 **0.710**, or CE-as-sorter. n=74, no p-values. Horizon nDCG@10 ≥ **0.70** is not a fail of 8500 pairs.

**Protocol (located evidence).** Export `./search.sh export-hybrid-lists --from-run search-esci-ltr200-passages-k50 --dataset data/search_esci_ltr200.jsonl --holdout-dataset data/search_esci_74.jsonl --out data/esci_hybrid_lists_ltr200.jsonl` (did not clobber `esci_retrieved_lists.jsonl`). 170 queries / 8500 pairs; 0 holdout-qid rows; labels E 984 / S 953 / C 117 / I 6446. Then `./search.sh finetune-4class --dataset data/search_esci_74.jsonl --from-lists data/esci_hybrid_lists_ltr200.jsonl --lists-source hybrid-k50 --base data/models/esci-deberta-v3-base-4class --out data/models/esci-deberta-v3-base-4class-hybrid170 --epochs 1 --max-length 192` (266 steps, MPS; pool checkpoint left in place; `source=hybrid-k50`). Then `./search.sh ltr-head --from-run search-esci-74-passages-k50 --train-from-run search-esci-ltr200-passages-k50 --train-dataset data/search_esci_ltr200.jsonl --ce-model data/models/esci-deberta-v3-base-4class-hybrid170 --pair-policy other_query_neg --run search-esci-74-ltr-deberta-hybrid170`. New caches only (`ce_gain_esci-deberta-v3-base-4class-hybrid170.json`). Lists not sorted by CE.

**Assumption:** 8500 pairs / 170 lists may be too few to retune DeBERTa; catalog shift 4k vs 2043 remains.

| Arm | nDCG@10 | Recall@10 | Recall@20 | Recall@50 | MRR |
| --- | --- | --- | --- | --- | --- |
| k=50 passages RRF | 0.500 | 0.379 | 0.590 | **0.834** | 0.765 |
| RankNet + pool DeBERTa from 170 (`search-esci-74-ltr-deberta-train200`) | **0.544** | **0.391** | 0.585 | **0.834** | 0.762 |
| LightGBM from 170 (`search-esci-74-ltr-deberta-lgbm-train200`) | 0.533 | 0.389 | 0.591 | **0.834** | 0.752 |
| RankNet + hybrid DeBERTa (`search-esci-74-ltr-deberta-hybrid170`) | 0.530 | 0.385 | 0.588 | **0.834** | 0.765 |
| Must-beat this round | **> 0.544** | **≥ 0.379** | — | hold | — |
| Horizon (not a fail) | **≥ 0.70** | — | — | — | — |
| Oracle on these 50 | 0.876 | 0.548 | — | 0.834 | — |

Mean applied RankNet weights (not a quality claim): `ce_gain` 6.92, `title_overlap` 2.29, `rrf_inv` 1.14, `dense_inv` 0.84, `bm25_inv` 0.36, `brand_hit` −0.17, `query_in_title` −0.27.

**Call (located evidence + decision):** I-CE-HYB **null vs C6**. nDCG@10 **0.530 < 0.544**. Recall@10 **0.385 ≥ 0.379**. Recall@50 held. Horizon 0.70 missed. One epoch of hybrid hard-negatives on 170 lists did not unlock the unused 0.33; a null here can mean too little data, not that first-stage negatives never matter. Do **not** relabel a 6k BM25-list run as this construct. **Do not wire into `search.py`.** Default stays hybrid RRF, `rerank=none`.

**A/B/C call:** US n=74 live default unchanged. Harness best remains **0.544**, still not Reddy **0.857**.

