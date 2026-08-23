# 21 — ESCI CE-on-pool vs Reddy 0.857, and first-stage nDCG on n=74

Workstream: **separate from graph-fusion testing**. Harness CE / rank-pool-ce / local first-stage index only. Ledger: `benchmarks.search` only. Brains: `searchbench*` only; no wipe of `searchbenchesci20` / `searchbenchesci74`.

**Focal questions**

1. Can a MiniLM-scale CE, trained and scored on labeled pools, beat **our FT 0.669 nDCG@20** and, on a comparable protocol only, Reddy EN CE **0.857**?
2. Can a first-stage intervention raise nDCG@10 on the shared 2043-doc `search_esci_74.jsonl` corpus by **≥0.02** vs **0.495**, without mixing y-axes with 0.857?

n=74 is **not** Reddy’s ~4477-query US public test. Do not claim full-ESCI SOTA.

Search date: **2026-08-18**. Origin of ideas: AI-assisted. Stage labels follow the scientific-brainstorming skill.

---

## Scope

- Purpose: pick 1–2 harness interventions; measure them.
- In: `finetune_esci_ce.py`, `rank_pool.py`, `plugins/search-rerank/`, new harness scripts, this note plus append-only claims in `19` / `20`.
- Out: `search.py`, `hybrid.py`, `fact_filter.py`, `graph_channels.py`, `replay_fusion.py`, `evaluate.py` instrumentation, fusion unit tests, production `RERANK_MAX_K`, default `channels=["passages"]`, default rerank.
- Constraints: `--brain` on parent `./search.sh`; `BRAINPAT_TOKEN` never printed; this checkout not `~/.brainapi/source`.

---

## Independent ideas (before literature)

Stage `independent`. Contributor: this workstream. Origin: AI-assisted.

### CE-on-pool (ranking-in-pool)

| ID | Statement | Assumption | Prediction | Disconfirm |
| --- | --- | --- | --- | --- |
| C01 | Train with nDCG gains (E=1, S=0.1, C=0.01, I=0) instead of Exact→1 else 0 | Binary labels collapse S and I, which nDCG still separates | nDCG@20 > 0.669 on the same 74 pools | Graded MSE ≤ 0.669 |
| C02 | 4-class CE, score = 1·P(E)+0.1·P(S)+0.01·P(C) | Softmax over ESCI matches Task 1 gains better than a single logit | nDCG@20 > 0.669; still < 0.857 on MiniLM / n=74 | 4-class ≤ graded or ≤ binary FT |
| C03 | More US Task-1 train pairs (hold out test qids) | 30k is under-capacity vs ~420k US train judgements | Modest lift vs 30k, diminishing after tens of thousands | Extra pairs do not move nDCG@20 |
| C04 | MiniLM-L-12-v2 (Reddy’s EN backbone) | L-6 is the main capacity gap vs 0.857 | Lift vs L-6; still short of 0.857 | L-12 ≈ L-6 on n=74 |
| C05 | Concatenate title+brand+color+bullets+description | Reddy used titles; KDD-style fields add attributes | Helps attribute queries; can add noise | Catalog text ≤ title-only |
| C06 | Ensemble DeBERTa / XLM / RemBERT | Winner margin is ensemble + size | Could approach 0.90 on full Task 1 | v3-base catalog slice run 2026-08-19; large/ensemble still out |
| C07 | monoT5 / mT5 generative ranker | Jeronymo: 580M mMARCO z.s. 0.864, 3.7B FT ~0.90 | Beats MiniLM | Too large for this harness |
| C08 | Distill a large CE into MiniLM | Teacher signal > binary ESCI labels | Lift at MiniLM cost | Distillation data/time not available |
| C09 | Listwise / LambdaRank on the pool | Metric-aware listwise fits nDCG | Lift vs pairwise CE | Needs list construction; easy to overfit n=74 |
| C10 | Query expansion before CE | CE already sees the full pool | Null on CE-on-pool (coverage 1.0) | Any CE lift from expansion |

### First-stage (shared 2043-doc index; not 0.857)

| ID | Statement | Assumption | Prediction | Disconfirm |
| --- | --- | --- | --- | --- |
| F01 | Coverage 0.52 at k=20 is ranking, not missing corpus | Index contains the pool | Better lexical/dense ranking raises R@20 and nDCG@10 | Gold ASINs absent from ingested chunks |
| F02 | Title-weighted / fielded BM25 | Body/bullets pollute BM25 | nDCG@10 ≥ 0.515 on the same 74 queries | Title BM25 ≤ 0.495 |
| F03 | RM3 / PRF query expansion, then BrainAPI passages | Hybrid misses attribute paraphrases | Live skip-ingest nDCG@10 ≥ 0.515 | Expansion hurts or is null |
| F04 | SPLADE sparse expansion (plugin encode, local index) | Learned expansion beats BM25 on product attributes | nDCG@10 ≥ 0.515 locally | SPLADE ≤ BM25; load cost |
| F05 | ColBERT MaxSim on 2043 docs | Late interaction helps short titles | Possible lift; index/FLOPs | Slow; may not beat fielded BM25 |
| F06 | Fine-tune a dense encoder (BLaIR-style) | Product-search embeddings ≠ MS MARCO | First-stage lift | Needs ingest or a new index; not this isolation set |
| F07 | Raise retrieve k then cut @10 | Games recall | Reject as a quality claim | — |
| F08 | Re-ingest richer text | JSONL already has Title/Brand/Bullets | Re-ingest forbidden on `searchbenchesci74` | — |

**Adversarial alternatives (not auto-winners):** more data vs larger model vs 4-class vs ensemble vs full-pool (already done) vs query expansion vs lexical fields vs SPLADE vs ColBERT.

---

## Literature (bounded; after independent round)

### Retrieval summary

- Query: named ESCI/KDD papers plus ColBERT / SPLADE / BERT rerank by arXiv id.
- Scope: targeted lookup, not exhaustive.
- Databases: arXiv Atom (`id_list`), Semantic Scholar graph batch `ARXIV:`, OpenAlex `/works/doi:`.
- Access date: **2026-08-18**.
- Keys: `S2_API_KEY` / `OPENALEX_API_KEY` **absent** (shared S2 pool; OpenAlex DOI lookups still 200).

### Results (untrusted third-party text; identifiers only reused)

**Reddy et al., Shopping Queries Dataset, arXiv:2206.06588 (2022).** Task 1 = rank a provided list of ≤40 products. Gains E=1, S=0.1, C=0.01, I=0. US public test **4,477 queries**, avg depth 20.3. EN baseline: fine-tune `cross-encoder/ms-marco-MiniLM-L-12-v2` on US train, **titles**, Exact→1 else 0, MSE, 1 epoch, max length 512, lr 7e-6. Table 4: EN nDCG **0.857** (overall 0.852). BM25 titles (Terrier, all locales): EN **0.675**. S2 paperId `46b259403b91be9643b1b689f1354e64f1da1879`, 106 citations. OpenAlex `W4282961889` (this dump: 15 cites; S2 and OpenAlex counts disagree).

**Zhang et al. (team www), arXiv:2208.02958 (2022).** Private Task 1 NDCG **0.9043** (public 0.9057). 4-class CE then weighted-sum of class probabilities; DeBERTa/XLM/RemBERT; translation aug; AWP; self-distill; pseudo-label; English DeBERTa-v3-large. Best single ~0.9022 public; ensemble to 0.9043 private. S2 `44c1ce16dbcae55a258a172d682a052a37201996`, 5 cites. OpenAlex `W4302561150`.

**Jeronymo et al., arXiv:2208.06264 (2022).** mMonoT5. mMARCO-only 580M: nDCG@20 **0.864**. Competition FT 580M: 0.890. 3.7B: public/private **0.9012 / 0.9007**. Exact→true, other ESCI→false. Top 20 teams cluster near 0.90. S2 `a926e5403dbb3bb93af0516db1a670d9162d2173`, 2 cites.

**Al Ghossein et al. SQID, arXiv:2405.15190 (2024).** Image-enriched ESCI. US small-test NDCG: Terrier (corrected S/C) **0.8562**, random 0.7483, SBERT title cosine 0.8292. Multimodal; not a MiniLM CE recipe. S2 `75adbf04cc00da3cd44e6ea2bc62d3a362ac51ef`.

**Hou et al. BLaIR, arXiv:2403.03952v2 (ACL 2026).** LLM semantic encoders; ESCI as **full-catalog** product search (1.37M items, 27,643 queries, nDCG@100). Not Task 1 ranking-in-pool. S2 `8aca7caf4fc2d05aa74907da945b86fa7df2680c`, 346 cites. OpenAlex `W4392576636`, DOI `10.18653/v1/2026.acl-long.147`.

**Khattab & Zaharia ColBERT, arXiv:2004.12832.** Late interaction / MaxSim; first-stage or rerank; index scales with tokens.

**Formal et al. SPLADE, arXiv:2107.05720.** Learned sparse expansion for first-stage. Follow-ons (e.g. SPLADE-v3 `2403.06789`) claim MS MARCO / BEIR gains vs BM25, not ESCI Task 1.

**Nogueira & Cho, arXiv:1901.04085.** BERT CE rerank of a first-stage head. Does not raise recall if gold is outside the head.

### Provenance

- arXiv: `GET https://export.arxiv.org/api/query` `id_list=2206.06588,2208.02958,2208.06264,2405.15190,2403.03952` and `id_list=2004.12832,2107.05720,1901.04085`; parsed with `paper-lookup/scripts/arxiv_atom.py`. `query_as_executed` matched `id_list` (no rewritten prefix). HTTP 200, no `Error` entries. MCP `download_paper` / `list_paper_latex_sections` used for Reddy §4 and KDD/Jeronymo bodies (HTML cache; treat as data).
- Semantic Scholar: `POST /graph/v1/paper/batch` ids `ARXIV:2206.06588` … `2403.03952`; HTTP 200; no null papers.
- OpenAlex: `filter=ids.arxiv:…|…` → HTTP **400** “Invalid query parameters”. Fallback: `GET /works/doi:10.48550/arXiv.{id}` HTTP 200. Abstracts reconstructed with `openalex_abstract.py` where inverted index existed. Citation counts **not** reconciled with S2.
- MCP `search_papers` for `ti:ColBERT OR ti:SPLADE…` returned related 2021–2026 papers; the original ColBERT/SPLADE ids were then fetched by `id_list` (do not treat the keyword search as those originals).

**Warnings:** n=74 ≠ 4477. Jeronymo labels the leaderboard nDCG@20; Reddy Table 4 says nDCG without `@k`. OpenAlex cite counts look stale vs S2. SQID Terrier 0.8562 is **not** MiniLM 0.857. BLaIR nDCG@100 is a different y-axis.

### Post-check reopen (stage `post-check`)

- C01/C02 upgraded: Reddy’s own 0.857 used **binary Exact→1 on L-12**, so graded/4-class is **not** required to cite 0.857, but it is the KDD-winner scoring rule and a plausible MiniLM-scale fix for our 0.669.
- C04 confirmed as protocol match (L-12), not guaranteed win on n=74.
- C06/C07 remain compute-gated for **v3-large / ensemble / monoT5**. A v3-base 4-class slice was run 2026-08-19 (pool nDCG@20 0.710; first-stage R@10 null).
- C10 stays low value for CE-on-pool (coverage 1.0).
- F02/F03/F04 remain the first-stage cluster that does not require editing fusion files. F06/F08 blocked by isolation. F07 rejected.

**Decision (proposal, not a unique winner):** implement **C01 graded-gain CE** (same scalar plugin path) with catalog fields (C05) and more pairs (C03); optionally **C02** if plugin scoring is cheap. For first-stage, **F02 fielded BM25** locally on the 2043 texts, then **F03 RM3 → live skip-ingest** if the API is healthy. Do not auto-declare 0.857 reachable.

---

## Critique of 0.857 comparability and this plan

### Summary

The published 0.857 is organizer EN Task 1 ranking-in-pool nDCG on ~4477 public-test queries, MiniLM-L-12, titles, Exact→1 MSE, 1 epoch. Our comparable arm is CE-on-pool on **74** US Task-1-style pools (2043 products, mean ~29 candidates). First-stage 0.495/0.549 is a **different estimand**.

### Strengths

- Gain table already matches Reddy §3.1 (`ESCI_GAINS`).
- CE-on-pool coverage 1.0; test qids held out of FT.
- Slice n=11 and n=74 are labeled separately in `20-esci-sota-targets.md`.

### Concerns

**Critical**

- **n=74 vs 4477.** No interval. A 0.857 miss (or hit) on 74 queries is not a replication. Small-n mean is sensitive to a few qrels (statistical-pitfalls: underpowered comparison; do not treat a 0.02 delta as confirmatory).
- **Wrong y-axis if 0.857 is cited vs first-stage or `--rank-pool` BrainAPI (coverage 0.517).** That would be a construct-validity failure, not a ranking win.
- **Selection of “hard” Task 1 small queries.** Reddy’s small split filters easier NDCG queries. Our 74 is a cap from the catalog builder, not the official public test. Even a CE above 0.857 here would not license “we beat Reddy.”

**Important**

- **Architecture mismatch:** our FT is MiniLM-**L-6**, 30k pairs, titles. Reddy is L-12, full US train (~419k judgements). Gap 0.669 vs 0.857 is over-determined (model, data, n, split).
- **Label function mismatch vs KDD 0.9043:** winner used 4-class probabilities; Reddy used binary. Our Exact→1 FT copied Reddy’s label map, not the winner’s.
- **Multiple arms without a predeclared primary** inflate false wins (multiplicity). Predeclare: CE progress = nDCG@20 **> 0.669**; Reddy bar = **0.857** on CE-on-pool only; first-stage win = nDCG@10 **≥ 0.515** on skip-ingest passages or an explicitly labeled local first-stage on the same 74 queries.
- **HARKing risk** if we train until n=74 looks good. One graded run + optional 4-class; stop if neither beats 0.669.

**Minor**

- Full-list nDCG vs nDCG@20: report both; cite 0.857 only next to nDCG@20 / full-list pool nDCG, never @10 first-stage.
- SQID 0.8562 Terrier is a different system than Reddy CE 0.857.

### Overall

It is **reasonable** to try to beat 0.857 on this 74-query pool task as an **optimistic smoke**. It is **not** reasonable to treat success or failure as a statement about the public test. MiniLM-L-6 + n=74 makes 0.857 **unlikely**; say so unless evidence appears.

---

## Tasks (win / stop)

### Task 1: Graded-gain CE trainer (C01, C03, C05)

**Acceptance:** `finetune_esci_ce.py` can emit Exact→1 **or** ESCI-gain labels; catalog passage text; hold out test qids; save under `benchmarks/data/models/`.

**Win:** code path runs; pair counts logged. **Stop:** parquet missing.

### Task 2: Rank-pool-ce scores scalar FT (and 4-class if present)

**Acceptance:** `SEARCH_RERANK_MODEL=…` + `rank-pool-ce` on `search_esci_74.jsonl` writes nDCG@20 and full nDCG. Plugin maps 4-logit models to gain-weighted scores without changing production default model.

**Win for CE:** nDCG@20 **> 0.669**. **Reddy bar:** **> 0.857** (may fail). **Stop:** ≤ 0.669 after one graded run (optional one 4-class retry, then stop).

### Task 3: Fielded BM25 first-stage on the 2043-doc JSONL (F02)

**Acceptance:** Local BM25 (all-text / title / title-boost / optional RM3) reports nDCG@10/@20. **Not** cited as Reddy. **Not** a brain wipe.

**Win:** nDCG@10 **≥ 0.515**. **Stop:** all variants < 0.495+0.02.

### Task 4: RM3 expanded queries → live skip-ingest passages (F03)

**Acceptance:** Copy JSONL with expanded `query` text; `./search.sh --brain searchbenchesci74 evaluate --skip-ingest --channels passages`. No re-ingest.

**Win:** same as Task 3 on the **BrainAPI** list. **Stop:** API down or nDCG@10 < 0.515.

### Checkpoint

- CE-on-pool row vs 0.669 / 0.857, n=74 stated.
- First-stage row vs 0.495, protocol labeled (local BM25 vs live passages).
- No mix with slice n=11 0.758.

---

## Located numbers (do not average)

| Arm | n | nDCG@20 | notes |
| --- | --- | --- | --- |
| CE-on-pool z.s. MiniLM-L-6 MS MARCO | 74 | 0.631 | coverage 1.0 |
| CE-on-pool FT 1 ep, 30k, Exact→1 | 74 | 0.669 | did not beat 0.857 |
| Reddy EN CE | ~4477 | **0.857** | ranking-in-pool, FT L-12 |
| KDD www | private | 0.9043 | ensembles |
| BrainAPI `--rank-pool` k=20 | 74 | 0.579 | coverage 0.517 |
| First-stage passages 2043-doc | 74 | 0.549 (@20), **0.495 (@10)** | not 0.857 |
| Slice passages | 11 | — | nDCG@10 0.758; frozen `searchbenchesci20` |

---

## Outcomes (located evidence, 2026-08-18)

**CE-on-pool.** 4-class MiniLM-L-12 (40k US Task-1 pairs, test qids held out, catalog fields, score = 1·P(E)+0.1·P(S)+0.01·P(C)): nDCG@20 **0.678**, full-list nDCG **0.747**, nDCG@10 0.597, missing_text=0 (`runs/search-esci-74-ce-pool-l12-4class`). **Progress vs 0.669: yes (+0.009).** **Reddy 0.857: not beaten.** n=74 ≠ ~4477. Confounded vs prior FT (L-12 vs L-6, 4-class vs Exact→1, 40k vs 30k, catalog vs title).

**First-stage.** Live skip-ingest passages: nDCG@10 **0.495**, nDCG@20 0.549, R@10 0.377, R@20 0.592, MRR 0.758, p50 62 ms (`search-esci-74-passages-control`). Local BM25 title-boost 0.458; SPLADE 0.465; RM3 hurt. **Win ≥0.515: failed.** BrainAPI hybrid already beats these harness first-stage indexes on this 2043-doc set. Task 4 RM3 live expand was **not** run after local RM3 lost.

**Decision log:** implemented C02+C04 (4-class L-12) and F02/F04 (fielded BM25 + local SPLADE). C06 v3-base catalog slice run 2026-08-19 (first-stage null; pool 0.710 ≠ 0.857). C06 large/ensemble and C07 not run. F03 stopped. Product default rerank unchanged. `searchbenchesci74` not wiped.

### Deeper retrieve k, then CE on retrieved hits (2026-08-18)

Not CE-on-pool. Not Reddy 0.857. Details in `19-search-esci-quality.md` and `20-esci-sota-targets.md`.

**Phase 1 located evidence:** raising search `k` to 50 on frozen `searchbenchesci74` passages raised Recall@50 to **0.834** and `--rank-pool` coverage to **0.778** (from 0.517 at k=20). nDCG@10 stayed **~0.500** vs control **0.495**. k=100 Recall@100 0.906; Recall@50 did not beat k=50.

**Phase 2 located evidence:** harness CE over stored k=50 `hit_ids` (MiniLM-L-6 and 4-class L-12 e2) left Recall@50 at 0.834 and dropped nDCG@10 to **0.448** / **0.467**. Win ≥0.515 failed. `RERANK_MAX_K` stayed 10.

### Dual-k first-stage matching (2026-08-18)

Not CE-on-pool. Not Reddy 0.857. Dual win needed Recall@10 ≥ 0.397 **and** Recall@50 ≥ 0.854.

**Located evidence:** miss taxonomy 67 head-ok / 2 rank-too-low / 5 total-miss (1270 golds: 424 / 728 / 118). Query rewrite of `esci-113` and `esci-267` did not move those qids or the mean. Local ANCE MiniLM sidecar Recall@10 **0.302** / Recall@50 **0.750**. **Dual win failed.** Passages hybrid remains the control.

### Retrieved-neg CE, ColBERT, BGE (2026-08-18)

Not CE-on-pool. Not Reddy 0.857. Details in `19-search-esci-quality.md` and `20-esci-sota-targets.md`.

**Located evidence:** retrieved-neg 4-class L-12 on stored k=50 hits nDCG@10 **0.465** (gate ≥0.515 failed; Recall@50 held 0.834). ColBERT MaxSim sidecar Recall@10 **0.311** / Recall@50 **0.714**. BGE-base (not MiniLM; no `query:` prefix) Recall@10 **0.322** / Recall@50 **0.799**. Dual-k gates ≥0.397 **and** ≥0.854 failed. `RERANK_MAX_K` stayed 10. `searchbenchesci74` not wiped.

### DeBERTa-v3-base 4-class on retrieved k=50 (2026-08-19)

Not Reddy 0.857 as a first-stage target. Details in `19-search-esci-quality.md` and `20-esci-sota-targets.md`.

**Located evidence:** unweighted 4-class `microsoft/deberta-v3-base`, 80k pairs, 2 epochs, test qids held out. Ranking-in-pool nDCG@20 **0.710** vs MiniLM e2 **0.695** (`search-esci-74-ce-pool-deberta-base`; n=74 ≠ ~4477). Harness CE on stored k=50 hits (`search-esci-74-passages-k50-ce-deberta`): nDCG@10 **0.510**, Recall@10 **0.363**, Recall@50 **0.834**. Predeclared first-stage win Recall@10 ≥0.397 **and** nDCG@10 ≥0.515 **failed**. Live `mode=catalog` skipped. `RERANK_MAX_K` stayed 10. `searchbenchesci74` not wiped.

### Multilingual first-stage foundation (2026-08-19)

Not CE-on-pool. Not Reddy 0.857. Not a US n=74 rerun. Details in `22-multilingual-ecommerce-search.md`.

**Located evidence:** Reddy locales US/ES/JP only; ES/JP organizer neural baseline is MPNet, EN is MiniLM. No Italian product-search qrel located. Live first-stage ES (`search-esci-es-passages-k50`, `searchbenchescies`, n=62, k=50 passages): nDCG@10 **0.577**, Recall@10 **0.353**, Recall@50 **0.914**. Not Reddy ES 0.849. `searchbenchesci74` not wiped.
