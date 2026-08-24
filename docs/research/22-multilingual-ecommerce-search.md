# 22 — Multilingual first-stage and production e-commerce search

Workstream: `/retrieve/search` only on `searchbench*`. Ledger: `benchmarks.search` only. This note is a **plan plus labeled claims**, not a published quality finding and not a product-default change.

**Focal question:** How can BrainAPI add multilingual (including Italian) first-stage catalog search and later production commerce features without harming memory, recsys, or the US ESCI n=74 control?

Session: 2026-08-19. Origin of ideas: AI-assisted, independent round before literature. Decision owner: the user.

---

## Isolation (decision)

- Brains: `searchbench*` only. Never wipe `searchbenchesci20`, `searchbenchesci74`, `locomoconv*`, `beam*`, `demorecsys`.
- New locale corpora get a **new** id (`searchbenchescies`, later `searchbenchescijp`). Skip-ingest on frozen brains.
- Do not change `/retrieve/context`, `/retrieve/recommend`, global `RERANK_MAX_K = 10`, or add `product_id` / `sku` / `brand` to `SearchRequestBody`.
- Default omitted `channels` stays `["passages"]`. Graph off ranking unless a predeclared isolated arm.
- n=74 rows stay first-stage; do not mix with CE-on-pool 0.695/0.710 or Reddy 0.857.
- Never print `BRAINPAT_TOKEN`. Benchmarks do not import `src/`.

---

## Independent ideation (before literature)

Stage `independent`. Contributor: this workstream.

### Multilingual first-stage

| ID | Statement | Kind | Assumption | Prediction | Disconfirm |
| --- | --- | --- | --- | --- | --- |
| M01 | Build an ES ESCI slice parallel to US n=74 on a **new** brain | idea | Catalog `--locale es` already filters parquet | Harness can emit `search_esci_es.jsonl` without touching `search_esci.jsonl` | ES download overwrites US JSONL |
| M02 | JP is a separate, costlier arm | idea | English `to_tsvector` will tokenize JP poorly | JP BM25 will look like Reddy’s JP BM25 collapse; dense may carry more | JP BM25 ≈ US BM25 |
| M03 | Do not MT the US 74 and call it Italian quality | idea | ESCI has no IT locale | An MT “IT” nDCG is not an Italian product-search estimand | A real Italian product qrel appears |
| M04 | Default OpenAI/Azure `text-embedding-3-large` is already multilingual enough for a first ES ingest | assumption | Provider docs; not re-checked on this corpus | ES ingest on a new brain does not require a new encoder; do **not** re-ingest `searchbenchesci74` | ES lexical overlap with EN embeddings is unusable |
| M05 | English Snowball on `search_tsv` hurts Italian morphology | idea | PostgreSQL `english` config stems EN, not IT | Italian smoke may still match exact tokens; inflected queries will miss | Italian analyzer on an isolated arm lifts smoke recall without US regression |
| M06 | Per-brain or per-request FTS language must not flip the global generated column | idea | `_SEARCH_DDL` is shared | Changing `'english'` globally would retokenize memory brains | Isolated search-only language column |
| M07 | Dual-language work must not regress US skip-ingest on `searchbenchesci74` | idea | Frozen brain + skip-ingest | US passages k=50 stays the control protocol | Any US re-ingest |

### Production commerce (Phase 2, plan only)

| ID | Statement | Kind | Notes |
| --- | --- | --- | --- |
| P01 | Spelling / query rewrite as a **harness** arm before API | idea | Same isolation as RM3; live only after a win |
| P02 | Locale and attribute filters via existing `node_labels` / `extras` / query params — **not** new catalog enums on `SearchRequestBody` | idea | Locked by `test_contract_has_graph_fields_not_catalog_enums` |
| P03 | Facets / merchandising only if they do not touch `/retrieve/recommend` or `/retrieve/context` | idea | Out of this pass unless Phase 1 wins |
| P04 | Keep product default: hybrid BM25+dense, passages only, `rerank=none` | prediction | Matches located US n=74 survival |

Adversarial alternatives (not auto-winners): multilingual encoder swap vs keep `text-embedding-3-large`; Italian analyzer vs `simple` FTS; ES ESCI vs Amazon-M2 titles-as-smoke; mMARCO-IT as a language smoke vs product qrels.

---

## Literature (bounded; after independent round)

### Retrieval summary

- Query: Reddy `2206.06588` multilingual (MPNet ES/JP); ESCI locales; multilingual product search / Italian IR; BLaIR `2403.03952` as full-catalog bar.
- Scope: targeted lookup, not exhaustive.
- Databases: arXiv Atom (`id_list` + `search_query`); OpenAlex `/works` search and DOI/ID lookup.
- Access date: **2026-08-19**.
- Keys: `S2_API_KEY` / `OPENALEX_API_KEY` **absent**. Semantic Scholar not called this pass.

### Results (untrusted third-party text; identifiers only reused)

**Reddy et al., Shopping Queries Dataset, arXiv:2206.06588 (2022).** OpenAlex `W4282961889`. Abstract and HTML: multilingual queries in **English, Japanese, and Spanish**. `product_locale` is US, Spain, or Japan. **No `Italian` substring in the HTML body** (located). Task 1 ranking-in-pool: EN MiniLM-L-12-v2 titles nDCG **0.857**; ES/JP multilingual **MPNet** 0.849 / 0.840; BM25 all-locales 0.675 / 0.697 / **0.136** (JP collapse from non-JP preprocessing). US public test **4,477** queries, avg depth 20.3. **Not** first-stage shared-index catalog search.

**Zhang et al. (www), arXiv:2208.02958.** Query-product pairs in English, Japanese and Spanish. Private Task 1 NDCG 0.9043. Ranking-in-pool. Translation aug is MT **among ESCI locales**, not Italian labels.

**Hou et al. BLaIR, arXiv:2403.03952v2 (ACL 2026).** OpenAlex `W4392576636`, DOI `10.18653/v1/2026.acl-long.147`. Full-catalog product search + rec; Amazon Reviews 2023. nDCG@100 bar (prior note: Table 10). Different y-axis from Reddy 0.857 and from our n=74 nDCG@10.

**Amazon-M2, arXiv:2307.09688 (NeurIPS 2023 Datasets).** Locales **UK, JP, DE, ES, IT, FR**. Major product languages include **Italian**. Tasks: next-product recommendation, domain-shift rec, title generation. OpenAlex `W4384918874`. **Not** ESCI-style search qrels. Do not score M2 MRR@100 as `/retrieve/search` quality.

**mMARCO, arXiv:2108.13897.** Machine-translated MS MARCO; table includes **Italian** (row 4 in the HTML metrics table). Passage IR, not product search. Same warning as M03: MT is not Italian catalog quality.

**MIRACL, arXiv:2210.09984.** 18-language Wikipedia ad hoc. Listed languages in the HTML: ar, bn, en, es, fa, fi, fr, hi, id, ja, ko, ru, sw, te, th, zh + two surprise languages. **No `Italian` substring in the HTML.** Not a product-search set.

**arXiv `all:"product search" AND (Italian OR italiano) AND cat:cs.IR`:** totalResults **0** (genuine no-match). **`all:ESCI AND (Italian OR italiano)`:** totalResults **0**.

**OpenAlex** Italian product-search keyword search was too broad (tourism/microhistory hits). Tighter ESCI query recovered Reddy / Zhang / KDD papers (EN/ES/JP). XMarket (Bonab et al., DOI `10.1145/3459637.3482493`) is **cross-market recommendation**, not search qrels.

### Provenance

- arXiv: `GET https://export.arxiv.org/api/query` `id_list=2206.06588,2403.03952` then (after ≥3s) `id_list=2307.09688,2108.13897,2210.09984`; parsed with `paper-lookup/scripts/arxiv_atom.py`. `query_as_executed` matched `id_list`. HTTP 200, no `Error` entries.
- arXiv keyword: `all:"product search" AND (Italian|italiano) AND cat:cs.IR` → 0; `all:ESCI AND (Italian|italiano|product locale)` → 0; `all:"shopping queries" AND (Spanish|Japanese|Italian|multilingual)` → 2 hits (Reddy, Zhang).
- OpenAlex: `/works?search=…` (ESCI multilingual; XMarket; mMARCO; MIRACL; Amazon-M2); `/works/doi:10.48550/arxiv.2206.06588`; `/works/W4392576636`; `/works/W4384918874`. Abstracts via `openalex_abstract.py` where inverted index existed. BLaIR `doi:10.48550/arxiv.2403.03952` was **not valid JSON** this pass; ID lookup 200.
- Reddy / MIRACL / mMARCO / Amazon-M2 HTML: `https://arxiv.org/html/{id}` on 2026-08-19.

**Warnings:** 429s not observed on arXiv/OpenAlex this pass. Do not invent Italian-in-ESCI. mMARCO-IT and Amazon-M2-IT are **not** ESCI Task 1. BLaIR nDCG@100 ≠ first-stage nDCG@10.

### Post-check reopen

- M03 strengthened: arXiv Italian product-search query empty; Reddy HTML has no Italian; Amazon-M2 has Italian **sessions/titles** but rec tasks.
- M02 aligned with Reddy JP BM25 0.136 — our `'english'` tsvector is the same class of mismatch.
- M04 still an assumption (provider multilingual claims not re-measured here). Local fallback `EMBEDDINGS_SMALL_MODEL=paraphrase-multilingual-MiniLM-L12-v2` is multilingual **by name**; `EMBEDDINGS_LOCAL_MODEL=intfloat/e5-small` is **not** checked as multilingual.
- P02 unchanged: no catalog enums on the search body.

**Italian product-search qrels:** **no direct evidence located.** Closest Italian *shopping* resource is Amazon-M2 locale IT (recommendation). Closest Italian *IR* resource in this lookup is mMARCO-IT (MT passages). Neither is an ESCI Task 1 stand-in.

### Phase 2 Italian qrel re-query (2026-08-19)

**Kind:** located evidence (empty). **Do not** treat Amazon-M2 or mMARCO-IT as `/retrieve/search` gold. Do not machine-translate US/ES ESCI.

| Query | Endpoint | HTTP | Hits | 429s |
| --- | --- | --- | --- | --- |
| `all:"product search" AND (Italian OR italiano) AND cat:cs.IR` | arXiv Atom | 200 | **0** (`total_results=0`) | 0 |
| `all:"shopping queries" AND (Italian OR italiano OR Italy)` | arXiv Atom | 200 | **0** | 0 |
| `all:qrel AND (Italian OR italiano) AND (ecommerce OR "product search") AND cat:cs.IR` | arXiv Atom | 200 | **0** | 0 |
| `search="product search" Italian qrel` | OpenAlex `/works` | 200 | 4 incidental (not IT catalog qrels) | 0 |
| `search=ecommerce search italiano nDCG` | OpenAlex `/works` | 200 | **0** | 0 |
| `W4282961889` (ESCI Reddy) | OpenAlex work | 200 | exists; locales EN/ES/JP only | 0 |

OpenAlex incidental titles (not used as qrels): W4401043175 synthetic query generation; W4411549462 GRIT e-commerce graph recall; W4414971087 hashing/RAG survey; W7201914832 structure-aware Boolean retrieval. None are Italian product-search qrels.

**Decision:** `--locale it` remains an ESCI error. No new Italian ESCI locale. Italian work stays the n=3 pipeline smoke fixture.

---

## Codebase map (located evidence)

| Area | Finding |
| --- | --- |
| Catalog | `prepare_esci_rows` already filters `product_locale`; CLI `--locale` existed but wrote the **same** `search_esci.jsonl` as US |
| Embeddings | Default `text-embedding-3-large` (OpenAI/Azure `.env.example`); TUI may run from `~/.brainapi/source` |
| BM25 | `src/lib/postgresql/data.py` `_SEARCH_DDL`: `to_tsvector('english', …)` generated column. Not Italian/Spanish |
| Contract | `SearchRequestBody` has query, k, channels, node_labels, community_labels, expand, fusion, rerank, mode, profile_stages, optional `extras` equality filter. Hits have `labels` / `extras`. Response may include `facets`. No sku/brand/product_id **fields** |
| Eval | `evaluate.py` already records `slice`; `metrics.py` `by_slice`. Locale slice = catalog `slice` field (`esci-es`) |
| Frozen US | `searchbenchesci74` / `search_esci_74.jsonl` — skip-ingest only |

---

## Phased plan

### Phase 1 — multilingual first-stage (this pass: foundation only)

Predeclared gates (**before** any live ES eval):

1. US default JSONL path remains `search_esci.jsonl`. `--locale es` writes `search_esci_es.jsonl`. `--locale it` errors.
2. `to_tsvector('english'` stays the product FTS config. Dual-language must not regress US skip-ingest on `searchbenchesci74`.
3. ES passages nDCG@10 is **not** a copy of US 0.500. Report ES separately. No p-values. n may differ from 74.
4. Italian smoke (`search_italian_smoke.jsonl`) is a **pipeline** check: ingest+search returns hits. It is **not** ESCI Task 1 and not an Italian quality claim.
5. Product default remains hybrid BM25+dense, passages, `rerank=none`. Do not re-ingest `searchbenchesci74`.

**Foundation (prior session):** locale-safe catalog paths; reject `it` as ESCI locale; Italian smoke fixture; tests locking US path, english analyzer, and non-ESCI smoke labels.

**Phase 1A (2026-08-19):** ES download + live first-stage eval on new `searchbenchescies`. JP still out. Product FTS stays english in `_SEARCH_DDL`.

---

## Located ES first-stage (2026-08-19)

**Kind:** located evidence. **Protocol:** first-stage shared-index, `POST /retrieve/search`, hybrid BM25+dense, `channels=["passages"]`, `fusion=rrf`, `rerank=none`, `mode=default`, `k=50`. Brain `searchbenchescies` (new; did not wipe `searchbenchesci74`). Dataset `search_esci_es.jsonl` (`slice=esci-es`). n=**62** queries, **2000** docs. English `to_tsvector` (same as US). No p-values.

| Metric | ES n=62 k=50 |
| --- | --- |
| nDCG@10 | **0.577** |
| Recall@10 | **0.353** |
| Recall@50 | **0.914** |
| MRR | 0.870 |
| p50 retrieve | 84 ms |

**Not** a copy of US n=74 nDCG@10 0.500 / Recall@10 0.379. Different n, different catalog, not a paired test. Do not average with US. **Not** Reddy ES Task 1 MPNet **0.849** (ranking-in-pool, ~20-item lists). **Not** CE-on-pool 0.695/0.710.

**Finding:** Recall@50 **0.914** is not a Reddy-JP-BM25-collapse class outcome (Reddy JP BM25 0.136 on Task 1 pools). Product default unchanged.

**Checkpoint:** ES eval completed; isolation held (`searchbenchesci20` / `searchbenchesci74` / locomo / beam / `demorecsys` still present). Human review of these numbers is the gate before any FTS schema work. JP arm stays out.

**Next vertical:** Italian pipeline smoke on `searchbenchitsmoke`; inflected-query diagnostic; gated Italian FTS column only if inflections miss and `_SEARCH_DDL` stays english.

**Command:** `./search.sh --brain searchbenchescies evaluate --dataset data/search_esci_es.jsonl --run search-esci-es-passages-k50 --channels passages --k 50 --ks 5,10,20,50`

---

## Located Italian pipeline (2026-08-19)

**Kind:** located evidence. **Not** an Italian product-search quality estimand (n=3; not ESCI).

Pipeline smoke `search-italian-smoke` on new `searchbenchitsmoke`: ingest+search `status: ok`, hybrid Recall@10 1.0. Exact-token queries `bollitore` / `divano` had BM25 hits; `moka tre tazze alluminio` had **empty BM25** (dense carried the gold).

Inflected diagnostic `data/search_italian_smoke_inflect.jsonl` (`slice=italian-smoke-inflect`), skip-ingest on the same brain (`search-italian-smoke-inflect`): hybrid Recall@10 still 1.0 because **dense** retrieved the three docs. **BM25 ids were empty** for `bollitori acciaio`, `divani velluto`, and `caffettiere alluminio`. Idea M05: english Snowball missed Italian morphology. Not a quality claim.

**Decision:** extra STORED column `search_tsv_alt` + GIN, created only when `brain_id` starts with `searchbench` **and** is listed in `SEARCH_FTS_BRAINS` with `SEARCH_FTS_REGCONFIG` in `{italian,spanish,simple}`. `_SEARCH_DDL` stays `to_tsvector('english'`. Memory/recsys brains never get the column.

Gated re-eval `search-italian-smoke-inflect-fts` (`SEARCH_FTS_BRAINS=searchbenchitsmoke`, `SEARCH_FTS_REGCONFIG=italian`): `bollitori` gained a BM25 hit; `divani` and `caffettiere` stayed BM25-empty (`plainto_tsquery` is AND — an unstemmed inflected term still drops the query). `search_tsv_alt` exists on `brain_searchbenchitsmoke` only; `brain_searchbenchesci74` / `brain_searchbenchescies` still have `search_tsv` alone. `RERANK_MAX_K` stayed 10. Product default unchanged.

**FTS OR on alt BM25 (2026-08-19, located evidence).** Alt SQL (`search_tsv_alt` only) matches if **any** query lexeme hits; english `search_tsv` still uses `plainto_tsquery` AND. Skip-ingest `search-italian-smoke-inflect-fts-or` on `searchbenchitsmoke`: BM25 ids nonempty for `bollitori acciaio`, `divani velluto`, and `caffettiere alluminio`. Hybrid Recall@10 1.0 is still dense+BM25 on n=3 — **not** an Italian nDCG claim. Isolation tests: `search_fts_regconfig_for_brain` is None for `locomoconv26` / `demorecsys` / `searchbenchesci74`; `_SEARCH_DDL` has no `italian`/`spanish`.

**Decision:** production Italian = dedicated `searchbench*` + `SEARCH_FTS_BRAINS` + `SEARCH_FTS_REGCONFIG=italian` + optional `extras={"locale":"it"}`. Do not enable alt FTS on memory/recsys brains. Do not MT US/ES ESCI.

### Phase 2 — thin commerce on `/retrieve/search` (2026-08-19)

**Kind:** located evidence + decision. Isolation held: no edits to `retrieve.py` / `entities.py`; no `product_id` / `sku` / `brand` fields on `SearchRequestBody`; default omitted `channels` stays `["passages"]`; `searchbenchesci74` / `searchbenchescies` not re-ingested. Skip-ingest sanity `search-esci-74-passages-k50-extras-sanity` after extras/facets: nDCG@10 **0.500**, Recall@10 **0.379**, Recall@50 **0.834** (bit-identical to the frozen control). p50 63 ms vs ~60 ms. Product default unchanged.

#### Italian qrels

Re-query (arXiv Atom + OpenAlex, access date 2026-08-19): **no direct evidence located.** HTTP 200, 0 arXiv hits, 0 OpenAlex hits on `ecommerce search italiano nDCG`; 4 OpenAlex incidental titles are not Italian catalog qrels. Amazon-M2 / mMARCO-IT still not `/retrieve/search` gold. `--locale it` still errors.

#### 2.1 Spelling harness (null → not live)

Miss-strata on `search-esci-es-passages-k50` / `search_esci_es.jsonl`: n=62, **0 total-miss** (`stratum_counts={'head-ok': 62}`). Query-level head-ok is not Recall@10 0.353 (item-level). Harness `spell-normalize` wrote `data/search_esci_es_spell.jsonl` (23 queries changed: NFKC, junk punctuation, conservative accent fold). No live LLM.

Skip-ingest eval `search-esci-es-spell-k50` on `searchbenchescies`, passages k=50:

| Metric | ES baseline | Spell arm | Gate |
| --- | --- | --- | --- |
| nDCG@10 | 0.577 | **0.595** | win if ≥ 0.597 — **miss** (+0.018) |
| Recall@10 | 0.353 | **0.362** | hold ≥ 0.353 — pass |
| Recall@50 | 0.914 | **0.914** | hold ≥ 0.914 — pass |

No previously total-miss qid to lift. **Decision:** spelling stays harness-only. Production `search.py` query text is unchanged. Do not mix with Reddy ES 0.849 or US n=74 0.500.

#### 2.2–2.3 Extras filter + hit-list facets

`SearchRequestBody.extras: dict[str, str] | None` is an equality filter after fusion, before cutting `k`. Catalog keys live inside that object (and on `SearchHit.extras`), not as body fields named `brand`. Search ingest now copies JSONL `brand` / `color` / `locale` via `meta_keys`. `SearchResponse.facets` counts extras values on the **current** hit list. No merchandising, no `/retrieve/recommend`.

Live on `searchbenchitsmoke` (3-doc re-ingest allowed): `extras={"locale":"it"}` keeps the three IT docs (Recall@10 1.0); facets `{locale: {it: 3}, brand: {CasaLuce, AtelierNord, FornoBasso}, color: {argento, blu navy, alluminio}}`. `extras={"color":"nope"}` returns empty hits / empty facets. p50 retrieve ~17 ms on this 3-doc brain — **not** a 200 ms SLO claim. Contract test still bans catalog enum field names.

#### Encoder smoke (M04 Italian-only)

Harness `local-dense` `paraphrase-multilingual-MiniLM-L12-v2` on `data/search_italian_smoke.jsonl`, labeled brain `searchbenchitmmini`, run `search-italian-minilm-pipeline`. **Did not** reuse `searchbenchitsmoke` 3-large vectors; **did not** re-ingest 74 or ES. n=3, `status: ok`, all three golds retrieved at k=10 (same pipeline recall as `search-italian-smoke` 3-large). **Not** a claim that MiniLM beats 3-large. **Not** Reddy. **Not** ES n=62. Product embedder unchanged.

### Out of scope

Wiping brains; raising default `RERANK_MAX_K`; CE as default; graph fusion as default; recsys; LoCoMo/BEAM; claiming Reddy 0.857; MT-US-as-Italian; Amazon-M2 MRR as search quality; merchandising that needs recommend/context.

---

## Located US control (do not redo)

Passages k=50 n=74: nDCG@10 **0.500**, Recall@10 **0.379**, Recall@50 **0.834**, p50 ~60ms. MiniLM/DeBERTa CE on those hits hurt Recall@10. Live `mode=catalog` MiniLM 0.467/0.341 null. Cascade R@50 0.889 harness-only (live sidecar not indexed). `fusion=cc` alpha sweep best nDCG@10 **0.493** (null vs 0.520). Literal residual Recall@50 **0.655** (null). Head LTR 5-fold CV on stored k=50: nDCG@10 **0.515**, Recall@10 **0.384** (null vs 0.520 / 0.397); overlap-only 0.476 / 0.339. LTR + MiniLM `ce_gain`: nDCG@10 **0.524**, Recall@10 **0.383**, Recall@50 **0.834** (harness win vs 0.520 / 0.379 hold; not live). LTR + DeBERTa-v3-base `ce_gain` RankNet CV: nDCG@10 **0.542**, Recall@10 **0.387**, Recall@50 **0.834** (win vs MiniLM 0.524; horizon 0.70 missed; not live). Gated LightGBM on the same DeBERTa matrix: nDCG@10 **0.516** (null vs RankNet). LTR + DeBERTa applied from 170 train hybrid lists (`searchbenchesciltr2`): nDCG@10 **0.544**, Recall@10 **0.391**, Recall@50 **0.834** (win vs 0.542; horizon missed; not live). LightGBM apply on the same 170 lists: nDCG@10 **0.533** (null vs RankNet 0.544; R@10 0.389 / R@50 0.834 held; not live). RankNet + hybrid-list DeBERTa `ce_gain` (`esci-deberta-v3-base-4class-hybrid170`): nDCG@10 **0.530** (null vs 0.544; R@10 0.385 / R@50 0.834 held; not live). Default path that survived: hybrid BM25+dense, passages, rerank=none, Italian via gated FTS+OR.
