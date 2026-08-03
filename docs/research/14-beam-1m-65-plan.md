# 14 — BEAM 1M ≥65% quality plan

**Date:** 2026-07-31  
**Owner:** beam-quality-iterator  
**Baseline:** clean product `beam-100k-1-clean-product` / brain `beam100k1clean` / headline **74.0%** (n=20, one chat).  
**Primary gap:** `event_ordering` τ ≈ **13.6%** (f1=0 on both items). Abstention ≈50%.  
**Ingest cost (same brain):** median `llm_source_multiplier` ≈ **72×**, mean ≈ **91×** (n=78 completed units) — above 15–20× ship band; must not explode on 1M.

## Critical-thinking caveats (binding)

| Claim class | Rule |
| --- | --- |
| Data | Scores, τ, f1, rubrics, session refs, multipliers from runs/REPORTS |
| Interpretation | Failure class labels; “retrieval miss” vs “present-but-unused” |
| Overclaim ban | Do not treat 100K/1 lift as suite lift; n=20; DeepSeek answer+judge same family |
| Stop | Written diminishing returns or blocked ops before infinite iteration |

## Diagnosed gap (data)

**event_ordering 100K/1**

| qa | Gold (rubric order) | Gold sessions | Clean product prediction theme | τ | f1 |
| --- | --- | --- | --- | ---: | ---: |
| 0 | Core functionality → Transaction error handling → Security and deployment | 4, 60, 116 | Early planning / init / core only | 0.125 | 0 |
| 1 | Initial setup → Transaction CRUD → Deployment → Integration tests → Deploy/test improvements | 6, 62, 118, 120 | Early planning / MVP / env / UI | 0.146 | 0 |

**Interpretation (not data):** Predictions respect “ONLY N” count but pick **wrong milestones** — consistent with retrieval/coverage bias toward early sessions, not only format exhaustion. f1=0 means LLM alignment never equated predicted lines to gold aspects.

**abstention qa1:** Invented “Craig” biography from current budget-tracker context (under_abstain).

## Independent brainstorm (pre-literature)

| ID | Type | Statement |
| --- | --- | --- |
| I1 | idea | Exact-N + short milestone labels ordered by first `session_*` mention |
| I2 | idea | Deterministic ordering aspect query variants at retrieve (no LLM) for mid/late themes |
| I3 | idea | Harness multi-retrieve merge for EO (early/mid/late aspect queries) |
| I4 | idea | Hard-abstain prompt when Q asks background/previous projects and context is current-app only |
| I5 | idea | Raise session diversify / historical budget only when ordering regex fires |
| A1 | assumption | Mid/late session text for gold aspects exists in `beam100k1clean` |
| P1 | prediction | Aspect variants + prompt → EO f1 > 0 and τ ≥ 0.35 on ≥1 of 2 items without re-ingest |
| P2 | prediction | Hard-abstain prompt → abstention mean ≥ 0.75 without KU regression |

**Adversarial review:** I2/I3 may retrieve noisy late sessions and hurt KU/IE if budgets steal slots from current-truth facts — mitigate by triggering only on ordering/history regex. I4 may over-abstain on answerable preference/summary Qs — scope to biography/background wording.

## Literature provenance (2026-07-31)

| Query (abbrev) | Tool | IDs used |
| --- | --- | --- |
| long-term / conversational memory, ordering, abstention, contradiction (cs.CL/AI/IR, ≥2024) | arXiv `search_papers` | 2606.17328 MemTrace; 2607.16211 MOSAIC; 2605.20926 MemConflict; 2606.06240 TOKI; 2510.18731 RLAAR; 2606.05182 LANTERN; 2411.00489 SALM survey |
| Known prior | — | **2510.27246** BEAM/LIGHT |

**Located-evidence → design mapping**

- MemTrace: failures often evidence-*use* not missing storage → keep answer-format + use of session-stamped context, not only store more.
- MOSAIC / TOKI: write-time conflict handling → Phase 4 conflict/scratchpad remains deferred; contradiction format stays harness/product prompt for now.
- RLAAR: calibrated abstention reduces premature answering → hard-abstain wording for biography Qs.
- LANTERN: episodic restore after compaction → session diversify + aspect queries approximate multi-episode recall without LLM-on-read.

## Architecture decisions

1. Dual track: product retrieve stays ADR-006 (deterministic variants/diversify); harness may multi-retrieve + richer prompts.
2. Evaluate-only on `beam100k1clean` before any re-ingest.
3. Escalate to BEAM 1M after EO/abstention move **or** written diminishing returns.
4. LoCoMo regression on best prior conv (ledger top compose-sota / product for that scope) after BEAM slice(s).

## Task list (vertical slices)

### Task 1: Ordering coverage + format (S)

**Description:** Fix EO prompts (exact N, short aspect labels, session-order). Add product ordering aspect query variants. Optional harness multi-retrieve merge for `event_ordering`.

**Acceptance criteria:**
- [ ] Unit tests for prompt / aspect helpers
- [ ] Evaluate-only EO (+ full product) on `beam100k1clean`
- [ ] EO τ mean improves vs 0.136 **or** taxonomy shows residual is write-time / missing ingest (documented)
- [ ] KU remains ≥ 0.9 on same run

**Verification:** `./beam.sh evaluate … --abilities event_ordering` then full product report; compare REPORTS.

**Dependencies:** None  
**Files:** `benchmarks/beam/prompts.py`, `sota.py`, `evaluate.py`, `test_beam_helpers.py`; `src/services/api/controllers/retrieve.py`  
**Scope:** M

### Task 2: Abstention hard-abstain (S)

**Description:** Strengthen abstention system/ability hints; avoid inventing biography from current project.

**Acceptance criteria:**
- [ ] Abstention mean ≥ 0.75 on clean brain product remeasure **or** taxonomy proves present-but-unused only
- [ ] No KU drop below 0.9

**Dependencies:** Task 1 preferred (same evaluate pass OK)  
**Scope:** S

### Checkpoint A (after Tasks 1–2) — **met 2026-07-31**

- [x] 100K/1 product headline **78.9%** (`beam-100k-1-clean-ordv1`) vs prior **74.0%**
- [x] EO τ **78.8%** (was 13.6%); abstention **100%**; KU **100%**
- [x] Taxonomy: `benchmarks/runs/beam-100k-1-ordv1-taxonomy.md`
- [x] REPORTS.json upserted

### Checkpoint B / Task 3 — **met 2026-08-03**

- [x] BEAM 1M product evaluate on `beam1m1clean`
- [x] Baseline `beam-1m-1-clean-product` **55.2%** → **`beam-1m-1-product-v2` 70.14%** → compose **`beam-1m-1-product-v3c-compose` 78.97%** (n=20; ≥74% band)
- [x] EO gold present in retrieve (no re-ingest); see `benchmarks/runs/beam-1m-1-product-v3-DONE.md`
- [x] Ingest 622/625; multiplier median 58.3× / mean 103.4×

### Checkpoint C / Task 4 — **met 2026-08-01**

- [x] LoCoMo conv-26 product regression on `locomoconv26`: **90.1%** (n=152) vs prior product best **85.5%** (`oldbrain-recheck-a`) — **no regression** (+4.6 pp; same-family judge caveat)
- [x] Run: `locomo-conv26-beamreg-product`

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| uvicorn dies mid-ingest | High | ensure_api + flock; concurrency **2–3** (not 1 forever); resume ingest; longer client transport backoff |
| Permanent embed-8192 units | Med | Harness marks/skips `permanent_failed`; stall guard; document 622/625 |
| Same-family judge bias | Med | Report caveat; don’t claim absolute SOTA |
| Aspect variants hurt non-EO | Med | Gate on ordering regex only |
| 1M cost/time | High | One sample first; multiplier watch; parallel ingest **2–3** (~20–25h / ~14–18h vs ~40h @1) |

## Next 1M re-ingest (when needed)

Do **not** wipe `beam1m1clean` while evaluate-only iteration is live. For a future full re-ingest use a **new brain** and:

```bash
# Prefer harness concurrency 2 (try 3 only if keepalive stays quiet).
# Requires CELERY_WORKER_CONCURRENCY≥2 (default 4 in product .env).
BEAM_INGEST_CONCURRENCY=2 BEAM_INGEST_TIMEOUT=2400 \
  bash benchmarks/scripts/run_beam_1m_1_ingest_eval.sh
```

Keepalive: `benchmarks/scripts/beam_1m_keepalive.sh` (flock; never pkill healthy LISTEN).

## Open questions

- Is gold EO text present in graph/passages for sessions 60+ on clean brain? (probe retrieve before re-ingest)
- Whether SC should stay off for product on BEAM (clean SOTA already regressed abstention)
