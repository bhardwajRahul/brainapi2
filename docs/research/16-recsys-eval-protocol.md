# RecSys eval protocol — train-free graph recommend (+ optional LightGCN)

How to store interactions in BrainAPI and score held-out next-item HitRate/Recall@K.

**Role split (locked):**

| Component | Responsibility |
| --- | --- |
| BrainAPI | KB + **train-free ranker**: `POST /ingest/structured` (`mode=deterministic`) and `GET /retrieve/recommend` on brain `demorecsys` |
| `plugins/features-rec` | Multi-behavior + attribute ingest; write-time `USER-PREFERS-ATTR` upserts via `POST /features-rec/interactions` |
| `plugins/recsys-gnn` | Optional offline baseline: export edges, train LightGCN, `GET /recsys/recommend` |

Freshness: each structured ingest changes graph walks / attribute prefs immediately — **no** `/recsys/train` on the live path.

**Not** LoCoMo / LongMemEval / BEAM. Never write to or wipe `beam1m1clean`, `locomoconv26*`, or other memory-eval brains. Ledger upserts go only to `benchmarks.recsys` in [`benchmarks/REPORTS.json`](../../benchmarks/REPORTS.json).

Related: [ADR-002](../decisions/002-structured-ingestion-specific-processing.md), [15-ecommerce-gnn-recsys-landscape.md](15-ecommerce-gnn-recsys-landscape.md).

---

## Pipelines

### A — Train-free graph (default)

```text
interactions → POST /ingest/structured (or POST /features-rec/interactions)
           → GET /retrieve/recommend?target=<user>&labels=PRODUCT&exclude_seen=true
           → HitRate@K / Recall@K (+ popularity baseline)
```

Channels: synergies, asymmetric event walks, **collaborative** 2-hop (user→item→user→item), optional `attribute_pref` (PREFERS edges and/or PRODUCT→ATTR soft prefs).

```bash
curl -sS "$BRAINAPI_URL/retrieve/recommend?target=u01&top_k=20&exclude_seen=true&labels=PRODUCT&include_attribute_pref=true&dampen_degree=true&recency_half_life_days=90" \
  -H "Authorization: Bearer $BRAINPAT_TOKEN" \
  -H "X-Brain-ID: demorecsys"
```

### B — Optional LightGCN

```text
interactions → POST /ingest/structured → POST /recsys/train → GET /recsys/recommend
```

---

## Features-rec (multi-behavior + attributes)

```bash
curl -sS -X POST "$BRAINAPI_URL/features-rec/interactions" \
  -H "Authorization: Bearer $BRAINPAT_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Brain-ID: demorecsys" \
  -d '{
    "user_id":"u01","item_id":"chair-1","behavior":"purchase",
    "timestamp":"2024-02-01T12:00:00Z",
    "attributes":{"color":"black","material":"wood","category":"furniture"},
    "brain_id":"demorecsys","wait":true
  }'
```

Then rank with core `/retrieve/recommend` (`include_attribute_pref=true`). Do **not** use a plugin-local recommend endpoint.

Behavior weights (config, not trained): View/Click 0.2, AddToCart 0.5, Purchase 1.0.

---

## Harness

```bash
cd benchmarks
# toy smoke (graph, no train)
./recsys.sh smoke --backend graph

# attributed toy
./recsys.sh evaluate --backend graph --dataset data/recsys_toy_attrs.jsonl --min-interactions 2 --max-users 20

# MovieLens subsample
./recsys.sh download --name ml-100k
./recsys.sh evaluate \
  --backend graph \
  --dataset data/movielens_100k.jsonl \
  --max-users 50 \
  --min-interactions 5 \
  --timeout 7200

# optional LightGCN comparator
./recsys.sh evaluate --backend lightgcn --dataset data/recsys_toy.jsonl --epochs 20
```

Requires `BRAINPAT_TOKEN` in `benchmarks/.env`. Default brain: `demorecsys`. Report `model=graph-recommend` or `lightgcn`.

**Live stack note:** after core or plugin changes, sync + restart (`brainapi start` / `brainapi update`) so OpenAPI lists `/retrieve/recommend` knobs and `/features-rec/interactions`.

---

## Guardrails

- Do **not** call free-text `POST /ingest/` for this suite.
- Default harness backend is **graph** (`GET /retrieve/recommend`); use `--backend lightgcn` only for the offline comparator.
- Do **not** mutate `benchmarks.locomo` / `beam` / `longmemeval` ledger rows.
- Do **not** point `--brain` at memory-eval brains.
