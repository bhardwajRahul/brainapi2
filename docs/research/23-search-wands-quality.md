# 23 — WANDS first-stage search quality

Workstream: `/retrieve/search` first-stage quality on a frozen Wayfair WANDS slice. Ledger: `benchmarks.search` only. This document records the **control protocol** and **located evidence**. It is not Chen et al. SOTA and not an ESCI number.

Every claim is labeled **idea**, **assumption**, **prediction**, **located evidence**, or **decision**.

---

## Focal question

What is BrainAPI’s live hybrid first-stage quality on a frozen WANDS catalog slice (passages, RRF, `rerank=none`, k=50), and what can that number actually support?

**Claim this run can support:** on this slice, with this product path, nDCG@10 / Recall@k / p50 were X.

**Claim this run cannot support:** we match Chen / WANDS SOTA / ESCI 0.500 / Reddy 0.857 / C6 0.544.

---

## Protocol (decision)

Mirror the ESCI n=74 first-stage control on a **different** brain, catalog, and gain table.

| Knob | Value |
| --- | --- |
| Dataset | Frozen [`benchmarks/data/search_wands.jsonl`](../../benchmarks/data/search_wands.jsonl) |
| Caps used to build it | `--max-queries 80 --max-docs 2000 --candidates-per-query 40` |
| Actual n | **66** queries / **2000** docs (`_select_catalog` dropped queries with no remaining gold once `max_docs` filled) |
| Slice label | `wands-80-2000` (actual n=66) |
| Brain | `searchbenchwands` (new; never wipe `searchbenchesci*`, `searchbenchesciltr2`, `locomoconv*`, `beam*`, `demorecsys`) |
| Ingest | `skip_enrichment` (default). No `--ingest-graph`. No `--enrich`. |
| Retrieve | `--channels passages --k 50 --ks 5,10,20,50`, fusion RRF, `rerank=none`. No `--rank-pool`. |
| Run | `search-wands-passages-k50` |
| After first eval | `--skip-ingest` only |
| Download freeze | `catalog_overwrite_blocked`: `--force` cannot clobber `search_wands.jsonl` once it exists |

```bash
./search.sh --brain searchbenchwands evaluate \
  --dataset data/search_wands.jsonl \
  --run search-wands-passages-k50 \
  --channels passages --k 50 --ks 5,10,20,50
# after first eval:
./search.sh --brain searchbenchwands evaluate \
  --dataset data/search_wands.jsonl \
  --run search-wands-passages-k50 \
  --channels passages --k 50 --ks 5,10,20,50 --skip-ingest
```

**Metrics (decision):**

- **nDCG@10:** linear DCG `rel / log2(rank+1)` with Exact=1, Partial=0.5, Irrelevant=0. Unlabeled retrieved docs gain 0. Affine-equivalent to linear 2/1/0; **not** TREC `2^rel-1`.
- **Recall@10 / @50:** binary over **Exact + Partial** (`gold_doc_ids` = gain > 0). Not ESCI E+S+C.
- **p50 retrieve ms** excluding embed.query.
- n=66, no CIs. Do not mix with ESCI 0.500 / C6 0.544.

**Subset caveat:** full WANDS is 480 queries / 42,994 products / 233,448 labels. This control searches the ingested 2000-doc brain, not the 40-candidate judged pool and not the 43k catalog.

Do **not** edit `src/services/api/retrieve.py`, `entities.py`, or `search.py` for this measurement.

---

## Literature (located evidence, 2026-08-20)

| Item | Status |
| --- | --- |
| Chen, Liu, Liu, Sun, Baltrunas, Schroeder. *WANDS: Dataset for Product Search Relevance Assessment*. ECIR 2022. DOI [10.1007/978-3-030-99736-6_9](https://doi.org/10.1007/978-3-030-99736-6_9). OpenAlex `W4225565720`. | **support-located** (identity). `is_oa=false`, `abstract_inverted_index=null`. Springer LNCS, not OA. |
| arXiv title search `ti:WANDS AND (Wayfair OR "product search relevance")` | **no Chen paper**. Hits were unrelated IR papers. |
| Dataset size ([wayfair/WANDS](https://github.com/wayfair/WANDS/)) | **located evidence:** 480 queries, 42,994 products, 233,448 labels. Exact / Partial / Irrelevant. |
| Published BM25 / hybrid nDCG in Chen et al. | **not located**. Do not invent. Medium notebooks are not primary. |

---

## First-stage control numbers

**Located evidence (2026-08-20).** Live `search-wands-passages-k50` on `searchbenchwands`. `eval.json`: `skip_ingest=false`, `ingest.status=completed`, `n_docs_mapped=2000`, `ingest_graph=false`, channels `passages`, fusion `rrf`, `rerank=none`, k=50. Ingest wall ~14 min. Ledger: `benchmarks.search` run `search-wands-passages-k50`.

| Metric | Value |
| --- | --- |
| n | 66 queries / 2000 docs |
| nDCG@10 | **0.823** |
| nDCG@20 | 0.812 |
| Recall@10 | **0.269** |
| Recall@20 | 0.465 |
| Recall@50 | **0.837** |
| MRR | **0.925** |
| p50 retrieve | **87 ms** |
| p95 retrieve | 97 ms |

Gold (Exact+Partial) min/median/max = 1 / 40 / 40; mean 34.1. Structural Recall@10 ceiling = mean `min(10,|gold|)/|gold|` = **0.365**. Observed 0.269 is below that ceiling, not below ESCI 0.379 (different gold density and gain table). 60/66 queries have a relevant at rank 1. Two queries have nDCG@10 = 0: `Bath Rugs & Mats` (Recall@50 also 0) and `Vanities` (first relevant at rank 27; Recall@50 = 1.0).

**Do not mix** with US ESCI n=74 nDCG@10 0.500 / C6 0.544, Reddy 0.857, or unpublished Medium WANDS notebooks. Chen et al. nDCG remains **not located**.

After this run, freeze the index: `--skip-ingest` only. Do not re-ingest `searchbenchwands`. Do not wipe ESCI brains.

---

## Isolation

- Brains: `searchbenchwands` only for this catalog. Skip-ingest on `searchbenchesci74`.
- Product default stays hybrid RRF, `rerank=none`, omitted `channels=["passages"]`.
- Ranking / LTR / CE / graph on WANDS are out of scope until a new human decision.
