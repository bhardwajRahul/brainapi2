# Search eval protocol — hybrid BM25 + dense (`/retrieve/search`)

How to ingest a labeled toy corpus into BrainAPI and score ranked hits without contaminating memory eval.

**Role split (locked):**

| Component | Responsibility |
| --- | --- |
| BrainAPI | KB + **core hybrid search**: `POST /ingest/` writes chunks (tsvector + dense); `POST /retrieve/search` returns ranked hits (`fusion=rrf` default, `cc` override). `rerank=none` (default). Unknown `plugin:<name>` is **400**, never a silent fallback |
| `plugins/search-rerank` | Optional second-stage cross-encoder via `rerank=plugin:cross-encoder`. Default `mode=default` reranks at most `RERANK_MAX_K=10`. Opt-in `mode=catalog` retrieves `k_ret=min(200, max(k, 50))` and reranks at most `CATALOG_RERANK_MAX_K=50`, then cuts to request `k`. Health: `GET /search-rerank/health`. Must not run on `/retrieve/context`. Catalog+CE is **not** the ADR-007 200 ms default path |
| `plugins/search-splade` | Learned-sparse first stage. Own index: `POST /search-splade/index`. Retrieve via `channels=["plugin:splade"]` (optionally fused with `passages`) |
| `plugins/search-colbert` | Late-interaction first stage (MaxSim). Own index: `POST /search-colbert/index`. Retrieve via `channels=["plugin:colbert"]` |

**Not** LoCoMo / LongMemEval / BEAM / RecSys. Never write to or wipe `beam*`, `locomoconv*`, `demorecsys`, or LongMemEval brains. Ledger upserts go only to `benchmarks.search` in [`benchmarks/REPORTS.json`](../../benchmarks/REPORTS.json). Never quote LoCoMo judge % as a search score.

Related: [ADR-007](../decisions/007-three-product-surfaces-one-kb.md), [ADR-008](../decisions/008-query-gated-search-personalization.md), [17-search-surface-and-cross-features.md](17-search-surface-and-cross-features.md) §12.

---

## Server prerequisites

The API under test must have:

- `SEARCH_ENABLED=true`
- `DATA_DB=postgresql`
- `BRAIN_CREATION_ALLOWED=true`
- At least one of `SEARCH_USE_DENSE` / `SEARCH_USE_BM25` true (product default: both)

Harness auth: `BRAINPAT_TOKEN` and `BRAINAPI_URL` in `benchmarks/.env`.

A 404 on `/retrieve/search` is **search disabled**, not a ranking miss. The harness must fail loudly (`Set SEARCH_ENABLED=true`).

---

## Pipeline

```text
docs JSONL → POST /ingest/ (skip_enrichment=true → chunk + embed only)
          → optional POST /ingest/structured (`--ingest-graph`; entity uuid = catalog `doc_id`)
          → optional interaction JSONL (`--interactions`; EVENT + `happened_at` on `searchbench*`)
          → GET /retrieve/text-chunks  (map DOCID markers → chunk ids)
          → POST /retrieve/search  (profile_stages=true, fusion=rrf|cc;
               channels: passages | entities | events | communities;
               expand: none | neighbors;
               mode: default | catalog)
          → Recall@{5,10,20} / nDCG@10 / MRR
          → p50/p95 retrieve excluding embed.query
```

Gold matches **chunk ids** (via DOCID mapping) **or** `hit.id == doc_id` (graph entity uuid). Binary relevance is enough for the toy fixture. Catalog datasets carry graded `gold_grades` and nDCG@10 uses those gains.

Each ingested document embeds a unique marker `DOCID <doc_id>`. After ingest, paginate `GET /retrieve/text-chunks` and map `doc_id → {chunk_id, ...}` by substring. A hit matches if `hit.id` is in that gold set **or** equals the catalog `doc_id`.

Search ingest defaults to `skip_enrichment=true` (chunk + embedding, no Scout/Architect). Pass `--enrich` to restore full KG ingest. Catalog evals should stay on the skip path so OpenAI/DeepSeek extract quota is not in the loop. Embeddings still run. `--ingest-graph` writes generic HAS triples (entity + CLASS/TYPE/ATTR hubs) via product `catalog_graph`. Core search has no PRODUCT/CATEGORY/BRAND field names; those strings may appear only as ordinary node types in harness JSONL.

---

## Query mix

Bundled fixture: [`benchmarks/data/search_toy.jsonl`](../../benchmarks/data/search_toy.jsonl).

| Slice | Intent |
| --- | --- |
| `keyword` | Name / token overlap (BM25 should help) |
| `paraphrase` | Semantic rewrite (dense should help) |

Product-search corpora (download, not committed):

| Dataset | Command | Default slice | Gains |
| --- | --- | --- | --- |
| Amazon ESCI / Shopping Queries (KDD Cup 2022) | `./search.sh download --name esci` | US, Task 1 `small_version`, `test` | E=1, S=0.1, C=0.01, I=0 |
| WANDS (Wayfair) | `./search.sh download --name wands` | `query_class` | Exact=1, Partial=0.5, Irrelevant=0 |

Defaults cap the local JSONL at 80 queries / 2000 products / 40 candidates per query so ingest is feasible. Frozen WANDS quality slice is `data/search_wands.jsonl` (caps 80/2000/40; actual n may be lower because `_select_catalog` drops queries with no remaining gold once `max_docs` fills). `download --name wands` refuses to overwrite that file once it exists. Use `--brain searchbenchesci` / `searchbenchwands`. Do **not** reuse `demorecsys`. Structured recsys JSONL is not a search qrel. WANDS first-stage control is passages-only, k=50, `rerank=none` — see [23-search-wands-quality.md](23-search-wands-quality.md). Graph `--ingest-graph --channels passages,entities,communities` is not the quality control. Opt-in catalog graph (hub intersection + node hybrid) uses a **new** brain `searchbenchwandsgraph` and is an **architecture demo** — see [24-search-catalog-graph.md](24-search-catalog-graph.md). Do **not** mix those numbers with frozen WANDS 0.823 or ESCI 0.500. `--ingest-graph` / `--interactions` on `searchbenchwands`, `searchbenchesci74`, `searchbenchescies`, or `searchbenchesciltr2` is refused.

Filter / time slices are **deferred** until `/retrieve/search` accepts filters. Do not fake them with ILIKE on the harness side.

Graph channels (core, use-case agnostic): `entities` (node ANN ∪ name CONTAINS), `events` (EVENT nodes; empty-ok on catalog-only brains; recency from `happened_at` when present), `communities` (typed hub nodes, default labels `TYPE,CLASS,TOPIC` via `SEARCH_COMMUNITY_LABELS`), `expand=neighbors` (depth-1, fanout cap + degree IDF). Default omitted `channels` remains `passages`. Hits are a mixed list of chunk ids and node uuids until a join table exists. Communities are **not** Leiden / `kg_topic_sessions`. Do not score `/retrieve/recommend` HitRate as search.

Optional later arm: reuse held-out memory questions as *retrieval* labels (`search-from-locomo-qrels`) **without** running the answerer, and never as LoCoMo accuracy. Not in v1 of this harness.

---

## Metrics and SLO

| Metric | Definition |
| --- | --- |
| Recall@k | \|retrieved ∩ gold\| / \|gold\| at k ∈ {5,10,20} |
| nDCG@10 | Binary on toy gold; graded `gold_grades` on ESCI/WANDS. Hits collapse to `doc_id` (chunk map **or** node uuid) |
| MRR | Reciprocal rank of the first gold hit |
| `p50_retrieve_ms` / `p95_retrieve_ms` | Prefer `stage_timings` `search.retrieve.wall_ms`; else `client_wall − embed.query.wall_ms` |

ADR-007 SLO: **p50 < 200 ms excluding embed RTT**. Record `embed.query` separately so the ledger cannot mix the two. That SLO is the **default** path (`mode=default`, `rerank=none`). `mode=catalog` plus `rerank=plugin:cross-encoder` is an opt-in two-stage; do not cite its client wall as the 200 ms product claim.

Headline for the ledger: `ndcg@10`, `recall@10`, `p50_retrieve_ms`. Label `dataset` (`search_toy.jsonl` vs `search_esci.jsonl` / `search_wands.jsonl`) so a toy run is not mistaken for product search.

A passing **smoke** is a health check, not a published quality claim. Only `evaluate` with `status: ok` upserts `benchmarks.search`.

---

## Harness

HTTP only (`benchmarks/search/` must not import `src/`, except `search/mapping.py` which re-exports `src.core.search.catalog_graph` after putting the repo root on `sys.path`). Default brain: `searchbenchsmoke`. Any `--brain` **must** start with `searchbench`.

```bash
cd benchmarks
./search.sh dataset-stats
./search.sh smoke
./search.sh evaluate --fusion rrf
./search.sh evaluate --fusion cc --run search-toy-cc
./search.sh download --name esci
./search.sh download --name esci --locale es
./search.sh download --name wands
./search.sh --brain searchbenchitsmoke evaluate --dataset data/search_italian_smoke.jsonl --run search-italian-smoke
./search.sh evaluate --dataset data/search_esci.jsonl --brain searchbenchesci --run search-esci
./search.sh --brain searchbenchwands evaluate --dataset data/search_wands.jsonl --run search-wands-passages-k50 --channels passages --k 50 --ks 5,10,20,50
./search.sh --brain searchbenchwands evaluate --dataset data/search_wands.jsonl --run search-wands-passages-k50 --channels passages --k 50 --ks 5,10,20,50 --skip-ingest
./search.sh --brain searchbenchwandsgraph evaluate --dataset data/search_wands.jsonl --run search-wandsgraph-communities-k50 --channels communities --k 50 --ks 5,10,20,50 --ingest-graph
./search.sh evaluate --rerank plugin:cross-encoder --run search-toy-ce
./search.sh evaluate --mode catalog --rerank plugin:cross-encoder --k 50 --brain searchbenchesci74 --skip-ingest --dataset data/search_esci_74.jsonl --run search-esci-74-catalog-ce-k50
./search.sh rank-corpus --dataset data/search_esci_74.jsonl --run search-esci-74-exhaustive-ce
./search.sh list-overlap --passages-run search-esci-74-passages-k50 --against-runs search-esci-74-bge-base-k50,search-esci-74-colbert-k50
./search.sh union-lists --from-runs search-esci-74-passages-k50,search-esci-74-bge-base-k50,search-esci-74-colbert-k50 --run search-esci-74-union-bge-k50
./search.sh cascade-lists --passages-run search-esci-74-passages-k50 --from-runs search-esci-74-bge-base-k50,search-esci-74-colbert-k50 --run search-esci-74-cascade-tail-k50
./search.sh finetune-4class --dataset data/search_esci_74.jsonl --base microsoft/deberta-v3-base --epochs 2 --max-pairs 80000 --batch-size 8 --out data/models/esci-deberta-v3-base-4class
SEARCH_RERANK_MODEL=data/models/esci-deberta-v3-base-4class ./search.sh rerank-retrieved --from-run search-esci-74-passages-k50 --dataset data/search_esci_74.jsonl --run search-esci-74-passages-k50-ce-deberta --ks 5,10,20,50
./search.sh evaluate --channels passages,plugin:splade --run search-toy-splade
./search.sh smoke --interactions data/recsys_toy.jsonl --channels events --brain searchbenchevents
./search.sh report --run <run_id>
```

Requires `BRAINPAT_TOKEN` in `benchmarks/.env`.

**Live stack note:** TUI / `brainapi start` may run from `~/.brainapi/source`. Sync or restart from the intended checkout before measuring, or scores will reflect stale code (`SEARCH_ENABLED` must be on in **that** process).

---

## Guardrails

- Score only via `POST /retrieve/search`. Do **not** call `/retrieve/context` as the metric path.
- Do **not** call `/retrieve/recommend`. Optional search `target` is query-gated rerank of retrieved hits (ADR-008), not this recommend protocol.
- Do **not** mutate `benchmarks.locomo` / `beam` / `longmemeval` / `recsys` ledger rows.
- Do **not** point `--brain` at memory-eval or recsys brains (`searchbench*` only; never `demorecsys`).
- Failed / empty runs stay off the leaderboard.
- Cross-encoder, SPLADE, and ColBERT are plugins. Core hybrid works if they are absent. `--rerank plugin:<missing>` and `--channels plugin:<missing>` must 400, never look like a ranking miss.

---

## Multilingual catalog paths (2026-08-19)

**Located evidence:** `download --locale` already filtered ESCI parquet (`us` / `es` / `jp`) but defaulted to the same `search_esci.jsonl` as US. **Decision:** US stays `search_esci.jsonl`; ES/JP write `search_esci_{locale}.jsonl`; `--locale it` errors. ESCI has no Italian split (Reddy HTML; arXiv Italian product-search query empty). Fixture `data/search_italian_smoke.jsonl` is a pipeline smoke (`slice=italian-smoke`), not ESCI Task 1 and not an Italian quality claim. FTS analyzer stays `to_tsvector('english'`. Do not re-ingest `searchbenchesci74`. Plan: [22-multilingual-ecommerce-search.md](22-multilingual-ecommerce-search.md).

**ES first-stage (2026-08-19, located evidence):** `./search.sh download --name esci --locale es` wrote `data/search_esci_es.jsonl` (n=62 queries, 2000 docs, `slice=esci-es`) without changing `search_esci.jsonl` / `search_esci_74.jsonl`. Live `./search.sh --brain searchbenchescies evaluate --dataset data/search_esci_es.jsonl --run search-esci-es-passages-k50 --channels passages --k 50 --ks 5,10,20,50` → nDCG@10 **0.577**, Recall@10 **0.353**, Recall@50 **0.914**, p50 84 ms. Protocol: first-stage shared-index, passages, `rerank=none`. Do not mix with US n=74 0.500/0.379 or Reddy ES 0.849 (ranking-in-pool). Product default unchanged.

Italian smoke is pipeline-only (`./search.sh --brain searchbenchitsmoke evaluate --dataset data/search_italian_smoke.jsonl`). Inflected BM25 miss on english FTS; optional `SEARCH_FTS_BRAINS=searchbenchitsmoke` + `SEARCH_FTS_REGCONFIG=italian` adds `search_tsv_alt` on that brain only. `_SEARCH_DDL` stays english.

**Phase 2 (2026-08-19).** Italian product-search qrels: still **no direct evidence located** (arXiv 0; OpenAlex incidental hits are not qrels). ES spelling harness `search-esci-es-spell-k50` skip-ingest: nDCG@10 0.595 (win bar 0.597) / Recall@10 0.362 / Recall@50 0.914 — **null**, not live. Optional `SearchRequestBody.extras` equality filter + `SearchResponse.facets` on hit extras; live `searchbenchitsmoke` `locale=it` keeps 3 docs, `color=nope` empty. MiniLM local-dense `search-italian-minilm-pipeline` on labeled `searchbenchitmmini` retrieved all 3 golds at k=10 (pipeline only; not product embedder). Do not mix with Reddy 0.849 / US 0.500 / ES n=62 first-stage. Plan: [22-multilingual-ecommerce-search.md](22-multilingual-ecommerce-search.md).

**Production Italian + US n=74 arms (2026-08-19).** Isolation held: `searchbench*` only; `_SEARCH_DDL` stays `to_tsvector('english'`; locomo/beam/`demorecsys` never get `search_tsv_alt`. **Decision:** production Italian is a dedicated `searchbench*` brain + `SEARCH_FTS_BRAINS` + `SEARCH_FTS_REGCONFIG=italian` + optional `extras={"locale":"it"}`. Alt BM25 matches **any** query lexeme (OR); english `search_tsv` keeps `plainto_tsquery` AND. Inflect skip-ingest `search-italian-smoke-inflect-fts-or`: BM25 nonempty for `bollitori` / `divani` / `caffettiere`. Do not quote nDCG as Italian quality.

US n=74 skip-ingest, passages, k=50, `rerank=none` (control nDCG@10 **0.500** / Recall@10 **0.379** / Recall@50 **0.834** / p50 ~60 ms):

- Frozen-head cascade is in `hybrid.frozen_head_merge` and is wired only when extra plugin (or `SEARCH_LITERAL_FILL`) lists are present. Default omitted `channels=["passages"]` stays hybrid RRF. Replay of stored passages+BGE+ColBERT lists still Recall@50 **0.889** / nDCG@10 **0.500**. Live n=74 sidecar not run: ColBERT/SPLADE plugin indexes are empty for `searchbenchesci74`. Honest default Recall@50 stays **0.834**.
- `fusion=cc` alpha `{0.3,0.5,0.7}`: best nDCG@10 **0.493** (alpha 0.7) vs win **0.520**. **Null.** Leave RRF.
- Title-token literal residual (`SEARCH_LITERAL_FILL`, frozen-head): Recall@10 **0.379** held; Recall@50 **0.655** (below 0.834). **Null.** Flag stays false.
- Head LTR (`./search.sh ltr-head`, run `search-esci-74-ltr-head-k50`): query-grouped 5-fold CV reorders stored `search-esci-74-passages-k50` `hit_ids` (same 50; Recall@50 **0.834** held by construction). nDCG@10 **0.515**, Recall@10 **0.384** vs win **0.520** / **0.397**. Overlap-only diagnostic nDCG@10 **0.476** / Recall@10 **0.339**. **Null.** Not live. Do not train-on-all-74 as the quality number.
- Head LTR + 4-class `ce_gain` feature (`search-esci-74-ltr-cefeat-k50`): same CV, pair policy `other_query_neg`. nDCG@10 **0.524** (gate ≥ 0.520), Recall@10 **0.383** (hold ≥ 0.379), Recall@50 **0.834**. **Harness win on predeclared nDCG/hold.** Not live. Human review before `search.py`. Do not mix with Reddy 0.857 or CE-as-sole-ranker (those hurt Recall@10).
- Head LTR + DeBERTa-v3-base `ce_gain` (`search-esci-74-ltr-deberta-k50`): same lists/CV/`other_query_neg`; checkpoint `data/models/esci-deberta-v3-base-4class`. nDCG@10 **0.542** (must-beat this round: > **0.524**), Recall@10 **0.387** (≥ 0.379), Recall@50 **0.834**. **Harness win vs MiniLM blend.** Horizon nDCG@10 ≥ **0.70** missed (oracle on these 50 is 0.876). Gated LightGBM lambdarank (`search-esci-74-ltr-deberta-lgbm-k50`): nDCG@10 **0.516**, Recall@10 **0.379** (unrounded 0.3788, below hold), `ce_gain` gain-importance largest. **Null vs RankNet DeBERTa.** Not live. Do not mix with Reddy 0.857, pool nDCG@20 0.710, or DeBERTa-as-sole-sorter (0.510 / 0.363).
- I04 train-on-matched-hybrid (`search-esci-74-ltr-deberta-train200`): RankNet+DeBERTa fit on 170 US train queries' hybrid k=50 from new brain `searchbenchesciltr2` (JSONL cap 200q/4k docs, doc budget bound to **170** queries / **4000** docs; 0 qid overlap with the 74). Applied to frozen `search-esci-74-passages-k50`. nDCG@10 **0.544** (> **0.542**), Recall@10 **0.391** (≥ 0.379), Recall@50 **0.834**. **Harness win vs I01 CV.** Horizon 0.70 missed. Train-set first-stage diagnostic (different catalog): nDCG@10 **0.387** / Recall@50 **0.744**. Not live. Do not wipe `searchbenchesci74`.
- I-LGBM-APPLY (`search-esci-74-ltr-deberta-lgbm-train200`): LightGBM lambdarank on the same 170 hybrid lists / DeBERTa `ce_gain` / `other_query_neg`, applied to frozen 74. Frozen hypers (`n_estimators=100`, `max_depth=3`, `lr=0.05`). nDCG@10 **0.533**, Recall@10 **0.389**, Recall@50 **0.834**. **Null vs RankNet apply 0.544.** Horizon 0.70 missed. Stop-for-human bar 0.58 not reached. Not live.
- I-CE-HYB (`search-esci-74-ltr-deberta-hybrid170`): continue pool DeBERTa 4-class on 8500 hybrid-k50 rows from the 170 lists (`data/models/esci-deberta-v3-base-4class-hybrid170`; did not clobber the pool checkpoint). RankNet apply with new `ce_gain`. nDCG@10 **0.530**, Recall@10 **0.385**, Recall@50 **0.834**. **Null vs C6 0.544.** Horizon 0.70 missed. Not live.

Product default that survived: hybrid BM25+dense, passages, `rerank=none`, extras/facets as shipped, Italian via gated FTS+OR. Harness nDCG@10 **0.544** is not a live default. Do not mix with Reddy 0.857 or CE-on-pool 0.710.

## WANDS first-stage control (2026-08-20)

**Decision:** frozen `data/search_wands.jsonl` (caps 80q/2000d/40 cand; **actual 66 queries / 2000 docs**), brain `searchbenchwands`, passages, k=50, `rerank=none`, skip_enrichment, no graph. Run `search-wands-passages-k50`. Gains Exact=1 / Partial=0.5 / Irrelevant=0. Recall is binary Exact+Partial. Linear DCG. After first eval, `--skip-ingest` only. `download --name wands` refuses overwrite.

**Located evidence:** nDCG@10 **0.823**, Recall@10 **0.269**, Recall@50 **0.837**, MRR **0.925**, p50 **87 ms**. Gold median 40; Recall@10 ceiling **0.365**. Do **not** mix with ESCI 0.500 / C6 0.544, Reddy 0.857, or Chen et al. (DOI located; nDCG **not located**). Details: [23-search-wands-quality.md](23-search-wands-quality.md).

## Opt-in catalog graph (2026-08-20)

**Decision:** architecture-opt-in on new brain `searchbenchwandsgraph` only. Product mapper is [`src/core/search/catalog_graph.py`](../../src/core/search/catalog_graph.py) (harness re-exports it). Communities match hubs then intersect item neighbors when ≥2 hub kinds match, else union; hybrid-rank product nodes inside `graph_channels`. Live omitted `channels` stays `["passages"]`. Node BM25 is `SEARCH_ENABLED` english `kg_nodes.search_tsv`. Do not edit `search.py` fusion this round.

**Isolation:** reuse frozen `data/search_wands.jsonl`; never wipe `searchbenchwands` / `searchbenchesci74`. Ledger runs `search-wandsgraph-communities-k50` / `search-wandsgraph-passages-k50` are labeled **architecture-demo**. Do not quote them as beating 0.823 / 0.500. Details: [24-search-catalog-graph.md](24-search-catalog-graph.md).

## Query-gated personalize (2026-08-21)

**Decision:** optional `SearchRequestBody.target` reranks retrieved hits after extras AND. Omit `target` → anonymous ranking (this protocol’s default). Do not call `/retrieve/recommend`. Do not score HitRate as nDCG. If ledgered: `architecture-demo` only. Details: [25-personalized-search.md](25-personalized-search.md), [ADR-008](../decisions/008-query-gated-search-personalization.md).
