# 20 — ESCI search-quality SOTA numbers to aim for

Workstream: evaluation protocol for `/retrieve/search` on `searchbench*` only. Ledger: `benchmarks.search` only. This document is **located evidence plus protocol targets**, not a ranking-code change and not a published BrainAPI finding.

**Focal question:** Which published numbers are valid aim points for BrainAPI search quality, given the current 11-query / 200-doc ESCI slice?

Every numeric claim below is labeled **located evidence**, **slice-internal**, or **not-checked**. Do **not** claim the slice is above or below full-ESCI SOTA.

Search date: **2026-08-18**. Tools: harness read; arXiv MCP `search_papers` / `get_abstract` / `download_paper`; AIcrowd/Amazon Science web pages for leaderboard confirmation.

---

## What this workstream does

The eval scores a ranked list from `POST /retrieve/search` against catalog qrels, not LoCoMo judge accuracy and not `/retrieve/recommend`.

### Scoring path (located evidence)

1. Catalog JSONL carries `gold_grades` from ESCI letters via `ESCI_GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}` (`benchmarks/search/catalog.py:31`, applied at `catalog.py:298-302`). Complement **is** in the gain table. Irrelevant is dropped (`gain > 0` filter).
2. After ingest, hits are collapsed to `doc_id` (chunk map or node uuid) and deduped (`evaluate.py:172-184`, `267-285`).
3. nDCG@10 is **graded** when `gold_grades` is present (`metrics.py:36-56`; wired at `evaluate.py:285`). Discount is `rel / log2(rank + 1)` (`metrics.py:29-33`) — the classic Järvelin DCG, not the `2^rel − 1` variant.
4. Recall@{5,10,20} is **binary** over the positive-gain set (Exact + Substitute + Complement), not Exact-only (`metrics.py:7-16`; gold union at `evaluate.py:224-245`).
5. MRR is the reciprocal rank of the **first positive-gain** hit (`metrics.py:19-26`). It is not MRR@10 as a cutoff in code; the list is whatever `k` the search returned (default evaluate `k=20`, `evaluate.py:304`).
6. Headline ledger fields are `ndcg@10`, `recall@10`, `mrr`, retrieve p50 (`report.py:65-68`).

### Slice under test (located evidence)

`benchmarks/data/search_esci_slice.jsonl`: 200 docs, 11 queries, US locale, all product columns, brain `searchbenchesci20`. Gold sizes: min 1 / median 20 / max 20. Gain histogram: E=127, S=37, C=2. Passages-only: nDCG@10 **0.758**, Recall@10 **0.511**, MRR **0.848**, p50 retrieve 28–29 ms (`docs/research/19-search-esci-quality.md`).

Later condition, not a substitute: `benchmarks/data/search_esci.jsonl` is 2000 docs / 74 queries (title-heavy; 43/2000 docs lack Brand/Color in text; 0 `class` fields). Gold min/median/max 1/15/38, mean 17.2.

Catalog builder defaults (`catalog.py:34-36, 64-71, 396-397`): locale `us`, split `test`, `small_version == 1` (Task 1 small), caps 80 queries / 2000 docs / 40 candidates per query. The JSONL is then searched as a **shared corpus**, not as a per-query candidate pool.

WANDS download exists in the same harness (`catalog.py:13-32`, gains Exact=1 / Partial=0.5 / Irrelevant=0). No BEIR or TREC Product Search harness in `benchmarks/search/`.

---

## Guarantees and where they break

**Stated guarantee:** graded nDCG@10 / Recall@10 / MRR on this slice measure product-search quality in a way that can be compared to published ESCI numbers.

Where that guarantee cannot be delivered:

1. **Pool size and task (critical).** Published Task 1 is *ranking-in-pool*: reorder ≤40 already-retrieved, fully labeled products per query (`2206.06588` §3.1). We retrieve from a 200-doc (or 2000-doc) shared index. Different candidate generation, different label density, different IDCG. **Gap**, not a bug in the scorer.
2. **Cutoff k (important).** Official Task 1 is nDCG on the provided list. Reddy does not state `@10`. Jeronymo et al. report the KDD Cup metric as **nDCG@20** (`2208.06264`). We always cut at 10 (`metrics.py:121`). Same gain table, different statistic.
3. **Recall@10 / MRR are not the Task 1 metrics (important).** Task 1 publishes nDCG. Tasks 2/3 publish micro-F1. Nobody in the primary ESCI papers reports Recall@10 or MRR as we compute them.
4. **n=11 (critical for inference).** Means have no interval. One total miss (`esci-72`) and one single-gold query (`esci-67`) dominate averages.
5. **Recall@10 ceiling (important).** Mean of `min(10, |gold|) / |gold|` on this slice is **0.667**. Six of eleven queries have `|gold|=20`, so Recall@10 cannot exceed 0.5 on those queries even if every top-10 hit is gold. Current 0.511 is already near that structural cap.
6. **MRR saturation (important).** Mean 0.848 leaves little room except the total miss. Rank-2–10 work will not show up.

Deliberate trade-off: slice size and `skip_enrichment` for ingest cost. Not a claim about full ESCI.

---

## Open questions for the maintainer

1. For a confirmatory ESCI run, is the target **ranking-in-pool nDCG** on the labeled candidate list (comparable to `2206.06588` / KDD Cup), or **first-stage nDCG@10** from a shared ingested corpus?
2. Should the headline cutoff stay nDCG@10, or switch to nDCG@20 / full-list nDCG when the goal is published-number comparability?
3. Should Recall@10 gold stay E+S+C, or Exact-only, given Task 1’s ranking order is E then S then C then I?
4. Is `search_esci.jsonl` (2000 docs, 74 queries) the first larger condition, or should the next run restrict scoring to each query’s original ≤40 labeled candidates?
5. Is p50 retrieve <200 ms still binding if a confirmatory arm uses a cross-encoder the way the published 0.85–0.90 numbers do?

---

## Frontier techniques

### ESCI / Shopping Queries Task 1 ranking-in-pool (canonical)

- **Mechanism:** For each query, reorder a provided list of up to 40 products. Graded nDCG with gains E=1, S=0.1, C=0.01, I=0. Complement is in the gain table. Locales US/ES/JP. Small version (Task 1) filters easier queries; US public test is 4,477 queries, avg depth 20.3 (`2206.06588` Tables 1–4, §3.1).
- **arXiv:** `2206.06588` (Reddy et al., 2022). Amazon Science: [dataset page](https://www.amazon.science/code-and-datasets/shopping-queries-dataset-a-large-scale-esci-benchmark-for-improving-product-search), [KDD Cup recap](https://www.amazon.science/blog/amazon-product-query-competition-draws-more-than-9-200-submissions).
- **Reported numbers (located evidence):**

| Method | Year | Metric | Split | Number | Source |
| --- | --- | --- | --- | --- | --- |
| Terrier BM25 (title, all locales together) | 2022 | nDCG (Task 1, no `@10` in paper) | Public test overall / EN / ES / JP | 0.563 / **0.675** / 0.697 / 0.136 | `2206.06588` Table 4 |
| Fine-tuned MS MARCO MiniLM cross-encoder (EN) + MPNet (ES, JP); Exact→1 else 0 at train time | 2022 | nDCG | Public test overall / EN / ES / JP | **0.852** / **0.857** / 0.849 / 0.840 | `2206.06588` Table 4 |
| Organizer “baseline” cited in the KDD recap | 2022 | nDCG | KDD private/public (unspecified cutoff in blog) | **0.8503** | [Amazon Science blog](https://www.amazon.science/blog/amazon-product-query-competition-draws-more-than-9-200-submissions) |
| Team www (DeBERTa/XLM/RemBERT ensemble; 4-class probs → weighted gain) | 2022 | NDCG (AIcrowd Task 1) | Public / private, multilingual Task 1 | **0.9057 / 0.9043** | `2208.02958`; [AIcrowd winners](https://discourse.aicrowd.com/t/final-winners-announcement/7974) |
| Team qinpersevere / day-day-up | 2022 | NDCG | Private Task 1 | 0.9036 / 0.9035 | AIcrowd winners table |
| mMonoT5-3.7B (NeuralMind) | 2022 | nDCG@20 (their label for the same leaderboard) | Public / private | 0.9012 / 0.9007 | `2208.06264` Table 1 |
| 20th place | 2022 | nDCG@20 | Private | 0.8929 | `2208.06264` Table 1 |
| SQID reproduction of ESCI_baseline on US Task 1 test (8,956 queries, ~20 judgments/query) | 2024 | NDCG (Terrier, corrected S/C mapping) | US small test | **0.8562** | `2405.15190` Table 2 |
| SQID random rank of the same pools | 2024 | NDCG | US small test | 0.7483 | `2405.15190` Table 2 |
| SQID SBERT title cosine (no fine-tune) | 2024 | NDCG | US small test | 0.8292 | `2405.15190` Table 2 |

- **Cost:** Cross-encoder or LLM ensembles over ~20–40 pairs per query. Not a first-stage ANN over a catalog.
- **Fit:** Gain table matches ours. Task, n, pool, and k do not.
- **Verdict:** **adopt** as the label protocol and as the **horizon-B ranking-in-pool target**. **reject** as a numeric bar for the 11-query / 200-doc slice.

### KDD Cup 2022 as “SOTA on ESCI”

- **Mechanism:** Same Task 1, private multilingual test. Top 10 private scores sit in **0.8998–0.9043** (AIcrowd). Jeronymo argues top-20 teams cluster near 0.90 nDCG@20 and that the dataset is too easy to discriminate retrievers (`2208.06264` abstract).
- **Cost:** Heavy fine-tune + ensemble. Wu et al. (`2208.00108`) rank 6th on Task 1 but do not publish a Task 1 nDCG in the paper body (leaderboard 6th is ETS-Lab 0.9014 private; Wu’s team name is not in the top-10 table — **not-checked** which public name maps to their 6th-place claim vs AIcrowd’s ETS-Lab).
- **Verdict:** **adopt** 0.9043 private NDCG as the published ceiling for *ranking-in-pool, multilingual Task 1*. Fashionable only if BrainAPI starts scoring labeled pools with a second-stage ranker.

### Full-catalog ESCI retrieval (BLaIR)

- **Mechanism:** Treat ESCI as product search over **1,367,729 items / 27,643 test queries**; report **NDCG@100** (`2403.03952` Table 3, Table 10). This is first-stage retrieval, not Task 1 reranking.
- **Reported (located evidence):** SFR-Embedding-Mistral **0.2560**; GritLM-7B 0.2537; e5-mistral-7b-instruct 0.2437; Qwen3-Embedding-8B 0.2328; text-embedding-3-large 0.2366; gemini-embedding-001 0.2233 (`2403.03952` Table 10).
- **Cost:** Embed the catalog; ANN. No cross-encoder over 40 candidates.
- **Fit:** Closer to “search a brain” than Task 1, but k=100, corpus ~1.4M, and nDCG@100 ≠ our nDCG@10. Recall@10 / MRR **not reported**.
- **Verdict:** **adapt** as the published bar if a later run is true first-stage over a large catalog. **reject** as a target for the 200-doc slice (our 0.758 lives on a different y-axis).

### Papers that use ESCI but are not SOTA bars here

- **ARR / SToICaL (`2601.05588`, 2026):** Custom in-context ranking, Gecko-mined lists, log-scale gains, n=310, nDCG ~95–97 and a nonstandard R@k (`|π[:k] ∩ π̂[:k]| / k`). **reject** — not ESCI Task 1 gains, not our recall.
- **GraphRAG / EDRM / LoCoMo:** Wrong task. Out of scope.
- **WANDS (ECIR 2022 Chen et al.):** Dataset paper identity **support-located** (2026-08-20): DOI [10.1007/978-3-030-99736-6_9](https://doi.org/10.1007/978-3-030-99736-6_9), OpenAlex `W4225565720`, Springer LNCS, `is_oa=false`, abstract index empty. arXiv title search did not return the chapter. Dataset size **located** on GitHub `wayfair/WANDS`: 480 queries / 42,994 products / 233,448 labels. Published BM25/hybrid **nDCG still not located**. Do not invent a paper nDCG. Medium notebooks are not primary. Local first-stage control is a different y-axis (≤80q / ≤2000 docs / linear Exact=1, Partial=0.5). See [23-search-wands-quality.md](23-search-wands-quality.md).
- **TREC 2023 Product Search (`2311.07861`):** Converts SQD toward end-to-end retrieval. **No harness in this repo.** Leftover.
- **BEIR (`2104.08663`):** No search harness here. Leftover.

---

## Implementation plan

No ranking-code change. Protocol only. Reuse `searchbenchesci20` with `--skip-ingest`. Never wipe brains.

### Phase 1 — Slice-internal aims (this week)

Treat passages-only **0.758 / 0.511 / 0.848** as the control. “Beating the slice” means:

| Metric | Now | Structural ceiling (slice-internal) | What a real win looks like | What is not a win |
| --- | --- | --- | --- | --- |
| nDCG@10 | 0.758 | Not 1.0: several queries have mixed E/S gold; `esci-72` is 0/11 of the mean | +0.02–0.05 with Recall@20 not down; or `esci-72` leaving 0 (alone +~0.09 if that query went to 1.0) | Graph fusion that drops nDCG while MRR stays flat |
| Recall@10 | 0.511 | **0.667** mean of `min(10,\|gold\|)/\|gold\|`; **0.5** on the six `|gold|=20` queries | Move `esci-72` (0 vs ceiling 1.0), `esci-19` (0.333 vs 0.667), `esci-37` (0.667 vs 1.0) | Raising retrieve `k` then cutting @10 |
| MRR | 0.848 | ~0.94 if only `esci-72` is 0 and is fixed to rank-1 | Almost only the total-miss query | Any rank-2–10 cleanup |

**Task 1 — Publish slice ceilings next to every report**

- **Acceptance:** Report or canvas shows Recall@10 ceiling 0.667 and per-query `|gold|`.
- **Verification:** `python` over `search_esci_slice.jsonl` gold sizes (already computed 2026-08-18).
- **Scope:** S. Files: report helper or canvas only.

**Checkpoint:** Humans agree the 0.758/0.511/0.848 triple is a **control**, not a SOTA claim.

### Phase 2 — Comparable confirmatory protocol

Pick one of two designs. Do not mix their numbers.

**Design B-rank (comparable to published 0.85–0.90):** For each query, score **only the original labeled candidates** (≤40), same E/S/C/I gains, report nDCG@20 **and** full-list nDCG, US `small_version` test, n in the thousands if ingest allows, else the 74-query 2000-doc file **restricted to each query’s gold pool**. Cite Reddy EN CE **0.857** as the organizer-style bar and www **0.9043** as the 2022 ceiling.

**Design B-retrieve (comparable to BLaIR, not to 0.90):** Search a shared corpus. Report nDCG@10 plus nDCG@100 if k allows. Cite SFR-Embedding-Mistral **0.256 nDCG@100** on 1.37M items only if the corpus is actually large. The 2000-doc file is still not that condition.

**Task 2 — Ranking-in-pool eval mode (if B-rank is chosen)**

- **Acceptance:** Harness can restrict hits to `gold_grades` keys plus labeled irrelevants for that `qid`; metrics include nDCG@20.
- **Verification:** One `--skip-ingest` run on a toy query with a 40-id pool; nDCG matches a manual Terrier-style recompute.
- **Scope:** M. Files: `benchmarks/search/evaluate.py`, `metrics.py`, `report.py`.
- **Depends:** Maintainer answer to open question 1.

**Task 3 — 2000-doc condition as a separate row**

- **Acceptance:** `search_esci.jsonl` run is labeled `dataset=search_esci.jsonl`, `n_queries=74`, never averaged with the 11-query slice.
- **Verification:** `./search.sh evaluate --dataset data/search_esci.jsonl --brain searchbenchesci --skip-ingest` (or a new `searchbench*` brain — no wipe of `searchbenchesci20`).
- **Scope:** S. Eval only.

**Checkpoint:** A table with three rows that are never compared as one number: slice n=11, 2000-doc retrieve, ranking-in-pool (if implemented).

### Measured n=11 ranking-in-pool smoke (2026-08-18)

Not a published-number claim. Rows must not be averaged.

| Row | nDCG@10 | nDCG@20 | full nDCG | n | Cite 0.857? |
| --- | --- | --- | --- | --- | --- |
| Slice first-stage passages | 0.758 | — | — | 11 | no |
| BrainAPI post-filter in-pool | 0.788 | 0.811 | 0.814 | 11 | no |
| MiniLM CE-on-pool (zero-shot MS MARCO) | 0.598 | 0.669 | 0.786 | 11 | **only this arm’s protocol**, still not the Reddy split |

Harness: `--rank-pool` on evaluate; `rank-pool-ce` subcommand. Catalog persists `candidate_grades` including I=0. `searchbenchesci20` was not re-ingested.

### 74-query brain `searchbenchesci74` (2026-08-18)

New brain; dataset `search_esci_74.jsonl` (74 queries, 2043 pool products, mean 28.8 labeled candidates). Passages ingest onto this brain is in progress and is **not** required for CE-on-pool.

Cite Reddy **0.857** only against CE-on-pool. n=74 is still not the 4,477-query public test.

| Arm | nDCG@20 | full nDCG | n | vs 0.857 |
| --- | --- | --- | --- | --- |
| Zero-shot MS MARCO MiniLM-L-6 | 0.631 | 0.717 | 74 | below |
| Fine-tuned 1 epoch, 30k US train pairs, Exact→1 else 0, test qids held out | 0.669 | 0.745 | 74 | below |

**Located evidence:** one-epoch Exact-vs-rest fine-tune lifted nDCG@20 +0.038 vs zero-shot and did not beat Reddy 0.857 / KDD 0.9043 (those use full Task 1 and, for the winner, ensembles).

### 74-query CE-on-pool and first-stage (2026-08-18, this workstream)

Do not average with n=11 or with Reddy n≈4477. Cite **0.857 only against CE-on-pool**.

| Arm | n | nDCG@10 | nDCG@20 | full nDCG | notes |
| --- | --- | --- | --- | --- | --- |
| Live passages skip-ingest (control) | 74 | **0.495** | 0.549 | 0.549 | R@10 0.377, R@20 0.592, MRR 0.758, p50 62 ms; `search-esci-74-passages-control` |
| Local BM25 all-text | 74 | 0.424 | 0.472 | — | harness JSONL; not BrainAPI |
| Local BM25 title-boost | 74 | 0.458 | 0.508 | — | best BM25 variant; **below** 0.495 |
| Local SPLADE-cocondenser | 74 | 0.465 | 0.520 | — | `naver/splade-cocondenser-ensembledistil`; **below** 0.495 |
| CE-on-pool FT Exact→1 MiniLM-L-6 30k | 74 | — | 0.669 | 0.745 | prior row |
| CE-on-pool 4-class MiniLM-L-12, 40k US train, catalog fields, score 1·P(E)+0.1·P(S)+0.01·P(C) | 74 | 0.597 | **0.678** | **0.747** | `esci-minilm-l12-4class`; **> 0.669 progress**; **not** 0.857 |

**Calls:** Reddy EN CE **0.857 was not beaten**. First-stage nDCG@10 did **not** beat 0.495 by ≥0.02 (live control matched 0.495; local BM25/SPLADE were lower). n=74 is still not the public test.

### 74-query 4-class weighted CE (2026-08-18, isolated workstream)

Ranking-in-pool only (I=0 stays in the list). Dataset `search_esci_74.jsonl`. Brain `searchbenchesci74` was **not** wiped or re-ingested. Did **not** repeat Exact→1 MiniLM-L-6. Cite **0.857 only against this protocol**; n=74 is not Reddy’s ~4,477-query EN public test.

**Chosen path:** team www (`2208.02958`) 4-class softmax → score `1.0 P(E)+0.1 P(S)+0.01 P(C)+0.0 P(I)`, backbone `cross-encoder/ms-marco-MiniLM-L-12-v2` (Reddy footnote 6, different **loss** from sibling Exact→1). New files: `benchmarks/search/finetune_esci_4class.py`, `benchmarks/search/rank_pool_4class.py`. Ledger runs under `benchmarks.search` only.

**Rejected (same lookup):** Reddy Exact→1 MiniLM (sibling already running that replica). Jeronymo mMonoT5-3.7B (`2208.06264`) 72 h TPU. www ensemble / DeBERTa-v3-large / RemBERT (0.90 is ensemble). SQID unfine-tuned title SBERT 0.8292 (`2405.15190` Table 2) already below 0.857. Wu (`2208.00108`) DeBERTaV3 + LightGBM group features. Pool-only ColBERT: extra infra, not a published 0.85–0.90 recipe we could finish in hours. `sentencepiece` is now in the parent venv (0.2.2); a **v3-base** 4-class slice was run 2026-08-19 (pool nDCG@20 **0.710**, still below 0.857; first-stage catalog R@10 **null**). v3-large / ensemble remain out of local MPS this pass.

**Critique (why 0.857/0.90 may be unreachable with this small CE on n=74):** construct-valid ranking-in-pool + ESCI gains. External validity fails: n=74 vs ~4477 (no interval). Training used 80k/419,653 US small-train pairs, not full KDD train+aug+ensemble. Inverse-freq class weights (C weight 5.53) **hurt** nDCG: on Exact golds mean P(I)>P(E). Unweighted 4-class (www-like) restored E>S>C>I in expected gain. Second epoch +0.006 only. 0.90 remains ensemble/3.7B.

| Arm | nDCG@20 | full nDCG | n | vs 0.669 / 0.857 |
| --- | --- | --- | --- | --- |
| Inverse-freq 4-class, 80k, 1 ep (`search-esci-74-ce-pool-4class`, `esci-minilm-l12-4class`) | 0.670 | 0.742 | 74 | ~tied / below |
| Unweighted 4-class, 80k, 1 ep (`search-esci-74-ce-pool-4class-nowt`) | 0.688 | 0.759 | 74 | **+0.019** / below |
| Unweighted 4-class, 80k, 2 ep (`search-esci-74-ce-pool-4class-e2`) | 0.695 | 0.765 | 74 | **+0.026** / below |
| Unweighted 4-class DeBERTa-v3-base, 80k, 2 ep (`search-esci-74-ce-pool-deberta-base`) | **0.710** | **0.777** | 74 | **+0.041** / below |

**Call:** Reddy EN CE **0.857 was not beaten**. Progress vs our Exact→1 FT 0.669: yes (best MiniLM 0.695; DeBERTa-v3-base **0.710** on the same 74 pools). On-disk `esci-minilm-l12-4class` is this workstream’s 80k inverse-freq checkpoint, not the 40k row above. Do not average pool **0.710** with first-stage nDCG@10.

### Deeper k then harness CE (2026-08-18, not 0.857)

Separate y-axis from the CE-on-pool rows above. First-stage retrieve from the 2043-doc shared corpus, then (Phase 2) reorder **retrieved** `hit_ids` only. Graph off. Default API `k` / `RERANK_MAX_K=10` unchanged. Cite **0.857 only against CE-on-pool**; do not average with 0.695.

**Phase 1 (located evidence).** Predeclared win: pool_coverage ≥ 0.57 **or** Recall@50 ≥ 0.65.

| Arm | k | Recall@50 | pool_coverage | nDCG@10 | p50 ms |
| --- | --- | --- | --- | --- | --- |
| Passages control | 20 | — | 0.517 (`--rank-pool`) | **0.495** | 62 |
| `search-esci-74-passages-k50` | 50 | **0.834** | — | 0.500 | 60 |
| `search-esci-74-passages-k50-pool` | 50 | 0.832 | **0.778** | 0.567 | 60 |
| `search-esci-74-passages-k100` | 100 | 0.831 | — | 0.500 | 58 |
| `search-esci-74-passages-k100-pool` | 100 | 0.906 | **0.846** | 0.574 | 59 |

**Phase 1 call:** win (Recall@50 0.834; coverage 0.778). nDCG@10 stayed flat, as predicted: extra golds are below rank 10. `missing_from_brain` stayed 0. k=100 does not raise Recall@50 vs k=50.

**Phase 2 (located evidence).** Harness `rerank-retrieved` on `search-esci-74-passages-k50`. Win needed nDCG@10 ≥ 0.515 with Recall@50 not down.

| Arm | nDCG@10 | Recall@50 |
| --- | --- | --- |
| k=50 first-stage | 0.500 | 0.834 |
| MiniLM-L-6 on retrieved hits | 0.448 | 0.834 |
| 4-class L-12 e2 on retrieved hits | 0.467 | 0.834 |

**Phase 2 call:** null. Both CEs moved nDCG@10 **down**. Same hit set, so Recall@50 is unchanged. Do not treat this as a Reddy comparison.

### Dual-k matching (2026-08-18, not 0.857)

Goal: more gold in **both** Recall@10 and Recall@50. Predeclared win: ≥0.397 and ≥0.854. Sidecar / skip-ingest only. `searchbenchesci74` not wiped.

**Task 1 located evidence.** k=50 golds: 424 in top-10, 728 in ranks 11–50, 118 missed (5 total-miss queries). Counts from `miss_strata.json` on `search-esci-74-passages-k50`.

**Task 2 located evidence.** Rewrote `esci-113` and `esci-267` only; skipped `esci-72`. Live k=50 qrewrite Recall@10 **0.379** / Recall@50 **0.834** (unchanged). Those two qids stayed at 0.

**Task 3 located evidence.** Local ANCE MiniLM (`search-esci-74-dense-ance-k50`): Recall@10 **0.302**, Recall@50 **0.750**, nDCG@10 **0.396**. Dual win **failed**. Not BrainAPI embeddings; not 0.857.

**Call:** no first-stage matching arm beat passages on both cutoffs. Stop encoder fishing on this isolation set. Do not fuse the sidecar. Do not mix with CE-on-pool 0.695.

### Retrieved-neg CE, ColBERT, BGE (2026-08-18, not 0.857)

Separate y-axis from CE-on-pool **0.695** / Reddy **0.857**. Frozen `searchbenchesci74`. Graph off. `RERANK_MAX_K=10` unchanged. n=74.

| Arm | protocol | nDCG@10 | Recall@10 | Recall@50 | vs gate |
| --- | --- | --- | --- | --- | --- |
| k=50 passages | hybrid RRF | **0.500** | 0.379 | **0.834** | control |
| retrieved-neg 4-class L-12 | CE on stored k=50 hits | 0.465 | 0.351 | 0.834 | nDCG@10 ≥0.515 **failed**; R@50 held |
| ColBERT MaxSim sidecar | local JSONL, not fused | 0.434 | 0.311 | 0.714 | dual-k **failed** |
| BGE-base (not MiniLM) | local JSONL, no `query:` | 0.441 | 0.322 | 0.799 | dual-k **failed** |

ColBERT p50 encode+retrieve **10063 ms** (200 ms labeled, not a stop). BGE p50 **408 ms**. MiniLM ANCE was **0.302 / 0.750**, not this BGE row.

**Call:** all three null vs their predeclared gates. Passages hybrid remains the first-stage control. Do not average with 0.695.

### Exhaustive catalog then two-stage (2026-08-19, not 0.857)

Separate y-axis from CE-on-pool **0.695** / Reddy **0.857**. Frozen `searchbenchesci74`. Graph off. Default `RERANK_MAX_K=10` unchanged. n=74. Cite **0.857 only against CE-on-pool**.

| Arm | protocol | nDCG@10 | Recall@10 | Recall@50 | vs gate |
| --- | --- | --- | --- | --- | --- |
| k=50 passages | hybrid RRF | **0.500** | 0.379 | **0.834** | control |
| Exhaustive 4-class L-12 e2 | exhaustive-catalog, 2043 docs | 0.416 | 0.266 | 0.603 | nDCG@10 ≥0.515 **and** R@10 ≥0.397 **failed** |
| Union passages+BGE+ColBERT | harness-union RRF | 0.479 | 0.363 | 0.827 | dual-k **failed**; R@10 diluted below 0.379 |
| `mode=catalog` + plugin CE | live two-stage, rerank 50 | 0.467 | 0.341 | **0.834** | nDCG@10 ≥0.515 **failed**; R@50 held |
| Frozen-head cascade | harness; hybrid top-10 frozen | **0.500** | **0.379** | **0.889** | R@10 held **and** R@50 ≥0.854 **win** |

U01: BGE top-50 held **40** gold ASINs absent from passages k=50 (19 qids); ColBERT **21** (13 qids). Unique golds were nonzero, so the union included ColBERT. Cascade injected **51** unique golds (set union) into ranks 11–50. Exhaustive p50 **94 s** is labeled, not a 200 ms claim. Catalog p50 retrieve **59 ms**; p50 client wall **877 ms** (not the ADR-007 default).

**Call:** A/B quality null; C ships the opt-in hook; frozen-head cascade is the first dual-k **win** on this isolation set. Do not average with 0.695. Do not treat C’s 0.467 as a Reddy comparison. Cascade is harness-only until a live fill-tail path is an explicit product decision.

### DeBERTa-v3-base 4-class catalog slice (2026-08-19, not 0.857)

C06 was previously “out of compute this pass.” This pass ran a **single** `microsoft/deberta-v3-base` 4-class CE (not v3-large, not www ensemble). Same 80k/2ep/unweighted recipe as MiniLM nowt-e2. Frozen `searchbenchesci74` not wiped. Default `RERANK_MAX_K=10` unchanged. n=74, no p-values.

**Ranking-in-pool (separate y-axis).** `search-esci-74-ce-pool-deberta-base` nDCG@20 **0.710** vs MiniLM e2 **0.695**. Still below Reddy **0.857**. n=74 ≠ ~4477.

**First-stage (R@10 gate).** Harness `search-esci-74-passages-k50-ce-deberta` on stored k=50 hits:

| Arm | protocol | nDCG@10 | Recall@10 | Recall@50 | vs gate |
| --- | --- | --- | --- | --- | --- |
| k=50 passages | hybrid RRF | **0.500** | **0.379** | **0.834** | control |
| MiniLM 4-class e2 on k=50 hits | CE on stored hits | 0.467 | 0.341 | **0.834** | nDCG@10 ≥0.515 **failed** |
| DeBERTa-v3-base 4-class on k=50 hits | CE on stored hits | 0.510 | 0.363 | **0.834** | R@10 ≥0.397 **and** nDCG@10 ≥0.515 **failed** |

**Call:** first-stage **null**. Recall@10 **0.363** missed **0.397** (below passages). nDCG@10 **0.510** missed **0.515**. Recall@50 held. Live `mode=catalog` skipped (gates miss; MiniLM catalog already matched harness). Cascade remains the Recall@50 result. Do not average with pool **0.710** / MiniLM **0.695** / Reddy **0.857**. v3-large / ensemble still out.

### Phase 3 — Optional adjacent harness

WANDS first-stage control is a **separate y-axis** from ESCI n=74. Chen et al. identity located (DOI `10.1007/978-3-030-99736-6_9`); published nDCG **not located**. Do not set a Chen-paper nDCG aim. Do not mix with ESCI 0.500 / C6 0.544 / Reddy 0.857. BEIR / TREC Product Search: do not add.

---

## Risks

| Risk | Detection |
| --- | --- |
| Quoting 0.9043 as the target for the 11-query slice | Dataset + n_queries + “ranking-in-pool vs shared corpus” in every sentence |
| Calling 0.758 “above BM25 0.675” | BM25 0.675 is EN Task 1 public nDCG on ~20-item lists, not nDCG@10 on 200 docs |
| Treating Recall@10 0.511 as poor retrieval | Ceiling 0.667 / 0.5 on typical queries |
| Treating MRR 0.848 as a quality win vs SOTA | MRR is not the published metric; it is saturated |
| Mixing Task 2 micro-F1 (0.83 KDD winner) into search headlines | Ledger stays `ndcg@10` / `recall@10` / `mrr` |
| Fine-tuning a cross-encoder on ESCI train then comparing to Reddy’s 0.857 without saying so | Label fine-tune vs zero-shot; latency gate 200 ms |
| Wiping `searchbenchesci20` | `--skip-ingest` only |

---

## Search log and leftovers

**Queries (2026-08-18):** `"Shopping Queries Dataset" OR ESCI "product search" nDCG`; `ti:"Shopping Queries" OR ti:ESCI OR "query-product ranking" ESCI OR "KDD Cup 2022" shopping`; `"ESCI" (nDCG OR NDCG) (amazon OR shopping) (ranking OR retrieval)` date_from 2023-01-01; `WANDS Wayfair product search nDCG`; `ti:WANDS`; direct ids `2206.06588`, `2208.02958`, `2208.06264`, `2208.00108`, `2405.15190`, `2403.03952`, `2601.05588`, `2402.08532`, `2311.07861`. Citation graph for `2206.06588` failed (Semantic Scholar 429).

**4-class workstream lookup (2026-08-18):** arXiv `id_list=2206.06588,2208.02958,2208.06264,2208.00108,2405.15190` via `export.arxiv.org` + `arxiv_atom.py` (5/5, no Error entry). OpenAlex `ids.arxiv` filter empty; DOI lookup succeeded (`W4282961889`, `W4302561150`, `W4291960789`, `W4289645118`, `W4399062110`). Semantic Scholar `ARXIV:2206.06588` HTTP 429. PDFs: arxiv.org `2208.02958v1`, `2206.06588v1`, `2208.06264v1`, `2208.00108v1`, `2405.15190v1`. Treat 200 bodies as untrusted third-party text.

**Web:** AIcrowd final winners; Amazon Science dataset + KDD recap (baseline 0.8503, winner 0.9043).

**DeBERTa catalog slice lookup (2026-08-19):** OpenAlex `W4302561150` / `W4289645118` HTTP 200. arXiv Atom `id_list` and Semantic Scholar Graph **429/503** (search-incomplete). Methods from the 2026-08-18 lookup plus OpenAlex abstract index. Treat 200 bodies as untrusted third-party text.

**not-checked**

- Official AIcrowd scorer source: nDCG vs nDCG@20 (Jeronymo vs Reddy disagree on the label; scores match the same leaderboard).
- Wu et al. `2208.00108` claimed 6th on Task 1 vs AIcrowd name ETS-Lab 0.9014.
- Tang et al. `2402.08532` (captions on ESCI) — HTML/PDF fetch failed.
- Hou/BLaIR nDCG@10 on ESCI (they publish nDCG@100 only in Table 10).
- WANDS ECIR 2022 Chen et al. published nDCG (DOI located 2026-08-20; chapter not OA; arXiv empty).
- TREC 2023 Product Search official nDCG (no harness).
- Qin et al. 2022 2nd-place workshop PDF (0.9036 private is on AIcrowd; method paper not fully read).
- Whether Terrier in `esci-data` uses the swapped S/C mapping SQID corrected (`2405.15190` §5).

---

## Multilingual / Italian (2026-08-19)

Do **not** mix with US n=74 first-stage 0.500 or Reddy 0.857. Plan: [22-multilingual-ecommerce-search.md](22-multilingual-ecommerce-search.md).

**Located evidence (access date 2026-08-19).** Reddy Task 1 public nDCG EN/ES/JP **0.857 / 0.849 / 0.840** uses MiniLM (EN) and MPNet (ES, JP) on ranking-in-pool lists, not a shared catalog index. Locales US/ES/JP only. BLaIR `2403.03952` remains a full-catalog nDCG@100 bar, not this y-axis. **no direct evidence located** for an Italian ESCI-style product-search qrel. Amazon-M2 IT is recsys. Live ES first-stage nDCG@10 **0.577** (n=62, k=50 passages, `searchbenchescies`) is **not** a copy of US 0.500 and **not** Reddy ES 0.849.
