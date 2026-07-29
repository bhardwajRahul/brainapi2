# 05 — Temporal Truth (cross-cutting)

**Concern:** how BrainAPI knows what is *currently* true, traced from write time to read time.
**Confirmed problem being investigated:** superseded facts are returned as if they were current truth (`docs/research/00-scope-and-constraints.md:12`).
**Binding constraint honoured throughout:** `/retrieve/context` must stay sub-second and cheap, so any temporal filtering on that path must be answerable by index or precomputed structure — never by an LLM call at query time (`00-scope-and-constraints.md:20-25`).

---

## Answer up front

Temporal correctness breaks at **five stages simultaneously**, but there is one seam where responsibility is visibly *dropped* and one bug that makes every other stage inert.

**The seam (produces the reported symptom):** `get_context` returns two independent evidence channels in the same response — graph triples and raw text passages. A current-truth filter exists and is applied to the first channel only:

```420:426:src/services/api/controllers/retrieve.py
def _is_currently_valid(predicate: Predicate) -> bool:
    props = getattr(predicate, "properties", None) or {}
    if props.get("invalid_at"):
        return False
    if getattr(predicate, "deprecated", False):
        return False
    return True
```

It is called in exactly one place in the entire repository, guarding the graph-fact path (`src/services/api/controllers/retrieve.py:787`). The passage path (`src/services/api/controllers/retrieve.py:667-725`) and the `historical_context` path (`src/services/api/controllers/retrieve.py:851-880`) are pure vector + keyword fusion with **no temporal predicate at all**. Those passages are the verbatim sentences that stated the now-superseded fact. So the system can correctly invalidate the edge and still hand the caller the original sentence asserting it, in the same payload, unlabelled. That is precisely "superseded facts come back as current truth."

**The bug that makes everything else inert:** `src/utils/dates.py:59-68` cannot parse the timestamp format the system actually receives, so no comparable time value ever enters the graph. Verified empirically against the real benchmark data:

| input | `parse_date_string` | `resolve_relative_date('yesterday', ref)` |
| --- | --- | --- |
| `'1:56 pm on 8 May, 2023'` (real LoCoMo `session_1_date_time`) | `None` | `'yesterday'` (unresolved) |

`_DATE_INPUT_FORMATS` (`src/utils/dates.py:5-16`) contains ten formats and **not one includes a time component**, so every LoCoMo session timestamp fails to parse. `resolve_relative_date` then bails at `src/utils/dates.py:91-98` (`ref is None → return cleaned`) and returns the relative expression verbatim. All of `yesterday`, `last week`, `last Tuesday`, `next quarter`, `last summer`, `two weeks ago`, `a few months ago`, `last month` pass through unresolved for the entire benchmark.

**One bug or a missing capability?** A **missing capability wrapped around three real bugs**. Fixing `dates.py` alone changes nothing observable, because there is still no validity-interval representation (Stage 3), no notion of which claims are single-valued (Stage 2), and the read filter covers one of roughly six read paths (Stage 4). The three concrete bugs are listed in *Guarantees and where they break* (G1, G3, G6).

---

## Stage-by-stage assessment

| # | Stage | Status | Evidence |
| --- | --- | --- | --- |
| 1 | **Write time** — event vs ingestion vs document time | **Partial, broken in practice** | `source_timestamp` is the only temporal input (`src/constants/tasks/ingestion.py:81-87`) and is *document time*. It is injected as a reference date into the Scout prompt (`src/core/agents/scout_agent.py:412-414`, `src/constants/prompts/scout_agent.py:144,154`) and used for post-hoc resolution (`src/workers/tasks/ingestion.py:192-205`). Both mechanisms are defeated by `src/utils/dates.py:59-68` for the format actually supplied. `src/core/agents/temporal_agent.py` is a 9-line stub — the named component does not exist. |
| 2 | **Conflict detection** — does anything notice supersession? | **Absent** (one unsound heuristic) | `_invalidate_superseded_relationships` (`src/workers/tasks/ingestion.py:465-505`) is the only mechanism; it is a syntactic predicate-name match with no recency comparison. `src/core/agents/validator_agent.py` and all three `src/services/triggers/*.py` are 9-line stubs. The Janitor has zero temporal awareness (no `happened_at`/`valid_at`/date logic in `src/core/agents/janitor_agent.py`), and its prompts cover direction, dedup and schema only — no prompt in `src/constants/prompts/` mentions contradiction, supersession or "no longer true". |
| 3 | **Representation** — what the graph physically stores | **Partial, ad hoc** | Edges may carry `valid_at`, `invalid_at`, `deprecated`, `source_timestamp` (`src/workers/tasks/ingestion.py:418-422,494-503`; `src/core/saving/identity.py:104-108`). Nodes carry only free-text `happened_at`. Format is `dd/mm/YYYY` (`src/utils/dates.py:76`) — not sortable, not a temporal type. No `valid_to`, no invalidation edges, no bitemporal pair. **No `CREATE INDEX` / `CREATE CONSTRAINT` anywhere in `src/lib/neo4j/client.py`**, so no indexed temporal filter is possible today. Postgres has `inserted_at`/`created_at TIMESTAMPTZ` (`src/lib/postgresql/data.py:60,71,84`) never used for retrieval. |
| 4 | **Read time** — filtering, ranking, current vs historical | **Partial — 1 of ~6 paths** | Filtered: graph facts in `get_context` (`src/services/api/controllers/retrieve.py:787`). Unfiltered: passages (`:667-725`), `historical_context` (`:851-880`), `/retrieve/entity/status`, `/retrieve/entity/synergies`, and all of `src/core/search/` (zero temporal keywords across all 8 modules; `src/core/search/context.py` is a 9-line stub). No `as_of` parameter exists anywhere in `src/` and the response exposes no validity fields, so a caller cannot distinguish current from historical. |
| 5 | **Measurement** — is supersession scored? | **Partial — measures the wrong thing** | Per-category reporting exists and category 2 is `temporal` (`benchmarks/locomo/config.py:21-27`, `benchmarks/locomo/metrics.py:155-158,236-238`). But cat-2 scores **0.86** in the best run — *above* multi-hop's 0.78 (`benchmarks/runs/locomo-conv26-push75-c/report.json`). Every cat-2 question is "When did X happen?" (event localization), and across all 1,986 LoCoMo questions only ~12 contain a current-truth marker (`current` 3, `used to` 6, `still` 1, `no longer` 1, `latest` 1, `now` 0, `anymore` 0). Supersession is ~0.6% of the question set: **the maintainer's #1 temporal complaint is invisible to CI.** |

---

## What this workstream does (traced flow)

### Write path

1. **Ingress.** `source_timestamp` arrives as a free-text string described as "Reference timestamp for the source content (e.g. session date). Used to resolve relative dates like 'yesterday' into absolute dates" (`src/constants/tasks/ingestion.py:81-87`). This is the *only* temporal input to the system. There is no separate event-time or valid-from input.

2. **Prompt-side resolution.** It is passed as `reference_time` into the Scout (`src/core/saving/auto_kg.py:121` → `src/core/agents/scout_agent.py:412-414`), which renders `Reference date for resolving relative dates: {reference_time}` into the prompt. The prompts mandate `DD/MM/YYYY` output stored as `happened_at` (`src/constants/prompts/scout_agent.py:153-154,174-175`; `src/constants/prompts/architect_agent.py:108`).

3. **Code-side resolution.** In the worker, `_normalize_relationship_dates` re-resolves `DATE`-typed entity names and `happened_at` values against `reference_time` (`src/workers/tasks/ingestion.py:192-205`) using `resolve_relative_date` (`src/utils/dates.py:80-123`).

4. **Stamping.** `stamp_provenance` attaches `source_chunk_ids` and `source_timestamp` to relationships and both endpoints (`src/core/saving/identity.py:85-109`). Relationships additionally get `valid_at = source_timestamp` when unset (`src/workers/tasks/ingestion.py:418-422`).

5. **Invalidation.** After each relationship is written, `_invalidate_superseded_relationships` fetches the subject's neighbours and stamps `invalid_at` + `deprecated: True` on any same-predicate-name edge pointing at a different object (`src/workers/tasks/ingestion.py:465-505`, called at `:972-975`).

6. **Consolidation.** `consolidate_graph` runs the Janitor then the KG agent for name/connection normalisation and dedup (`src/core/layers/graph_consolidation/graph_consolidation.py:31-94`). Nothing in this layer reads or writes time.

### Read path

`get_context` fans out over query variants, collecting two independent channels:

- **Graph facts** — `get_event_centric_neighbors` returns 5-tuples `(n, r, m, r2, b)` joined on `flow_key` (`src/lib/neo4j/client.py:1669-1681`); candidates are dropped unless *both* predicates pass `_is_currently_valid` (`src/services/api/controllers/retrieve.py:787-788`), then ranked by seed distance, optionally re-ranked by personalized PageRank, then optionally LLM-filtered.
- **Text passages** — `_retrieve_passages` fuses vector and keyword hits via RRF (`src/services/api/controllers/retrieve.py:667-725`), plus `historical_context` (`:851-880`). Neither consults time.

Both channels are concatenated into the response. Only one is temporally guarded.

---

## Guarantees and where they break

The guarantee this workstream should provide: *given a query and an optional as-of time, BrainAPI returns the facts that were true at that time, labels them as current or historical, and never presents a superseded assertion as current — while retaining full provenance of how the fact evolved.* None of those four clauses holds today.

Ranked by impact on answer accuracy.

### G1 — Reference time is unparseable, so no comparable time value ever enters the graph *(bug)*

`_DATE_INPUT_FORMATS` (`src/utils/dates.py:5-16`) has no time-bearing format. The real LoCoMo `session_N_date_time` value `'1:56 pm on 8 May, 2023'` therefore yields `parse_date_string → None`, `resolve_relative_date` returns the input unchanged (`src/utils/dates.py:91-98`), and `valid_at` is stamped with that raw string (`src/workers/tasks/ingestion.py:418-422`). Downstream, `invalid_at` is set from that same unorderable string (`src/workers/tasks/ingestion.py:499-500`).

Secondary gap: `_RELATIVE_PATTERNS` (`src/utils/dates.py:18-38`) is anchored `^…$` and covers nine literal patterns. Even with a valid reference date, `next quarter`, `last summer`, `two weeks ago` (word numerals), `a few months ago` and any expression with trailing words are unsupported. This is the "relative expressions resolved against a reference date" question in the brief: the intent is present in two places, the capability is not.

### G2 — Event time and document time are conflated *(design gap)*

`valid_at` on an edge is set to `source_timestamp`, i.e. *when the system read the document*, while the event's own date lives on the node as `happened_at`. Two different clocks, never reconciled. A LoCoMo session dated May 2023 discussing an event from 2019 produces an edge whose "validity" is 2023. Zep's model keeps these strictly separate as `t_valid`/`t_invalid ∈ T` (event timeline) versus `t'_created`/`t'_expired ∈ T'` (transaction timeline) — see *Frontier techniques* F1.

### G3 — The invalidation heuristic is unsound in both directions *(bug)*

`_invalidate_superseded_relationships` (`src/workers/tasks/ingestion.py:481-503`) invalidates an existing edge when: same tail, same predicate name, different tip. Three independent defects:

- **No recency comparison.** It never checks whether the existing edge is *older* than the incoming one. Out-of-order ingestion (Celery concurrency, chunk reordering, backfill) will invalidate the *newer* fact and leave the older one live. There is no `if existing.valid_at < new.valid_at` guard anywhere in the function.
- **Predicate names are deliberately generic, so the match condition is not a contradiction test.** The prompts explicitly instruct general labels — `"TARGET_PRODUCT_OBJECT_CROISSANTS"=wrong, "TARGETED"=correct` (`src/constants/prompts/architect_agent.py:382,424`) — and the schema is `MADE` / `TARGETED` / `OCCURRED_WITHIN` (`src/constants/prompts/architect_agent.py:16-20`). "Same subject, same predicate, different object" is therefore the *normal shape of accumulated history*, not a functional-dependency violation. Every second event by the same actor triggers invalidation of the first. Because `_is_currently_valid` requires **both** `r` and `r2` to be valid (`src/services/api/controllers/retrieve.py:787`), invalidating the actor-side `MADE` edge silently deletes the whole triple from `get_context`.
- **Asymmetric coverage.** Event hubs are fresh nodes per event, so the object-side (`TARGETED`) edge of a new hub has no prior siblings and is never invalidated; only actor-side edges are ever touched.

Net effect: the mechanism both over-fires (destroying true history) and under-fires (missing genuine supersession that changes predicate label or object type). It is a coin flip, not a rule.

### G4 — The read-time filter covers one of ~six paths *(the seam)*

Detailed in *Answer up front*. The ambiguity is structural: nothing in the codebase assigns ownership of temporal validity to a layer. `src/core/search/` — the module that ought to own it — contains no temporal logic whatsoever, and its `context.py` is an empty stub. The one filter lives in the API controller, so every other consumer (`/retrieve/entity/status`, `/retrieve/entity/synergies`, `/retrieve/hops`, neighbours, MCP tools) silently inherits no filtering.

### G5 — Callers cannot express or observe time *(design gap)*

`GetContextRequestBody` (`src/services/api/constants/requests.py:396-400`, including the uncommitted `use_ppr` / `sufficiency_retry` additions) has no `as_of` or `temporal_mode`. `grep` for `as_of|point_in_time|valid_from|valid_to` across `src/` returns nothing. `GetContextTriple` carries no validity fields. So a caller can neither ask "what was true in March 2024" nor tell whether what came back is current.

### G6 — The only recency scorer is a permanent no-op *(bug)*

```226:234:src/core/search/entity_info.py
            if node.happened_at:
                try:
                    happened_at = node.happened_at
                    if isinstance(happened_at, str):
                        happened_at = datetime.fromisoformat(happened_at)
                    days_ago = max(0, (datetime.now() - happened_at).days)
                except Exception:
                    days_ago = 0
            recency = 1 / (1 + np.log1p(days_ago)) if days_ago > 0 else 1.0
```

`normalize_date_string` writes `dd/mm/YYYY` (`src/utils/dates.py:76`); `datetime.fromisoformat` cannot parse that and raises; the bare `except` sets `days_ago = 0`; `recency` becomes the constant `1.0` for every node. The `* 0.2` recency term at `src/core/search/entity_info.py:235` contributes an identical constant to every candidate and therefore has zero effect on ranking. Note the normalizer actively *destroys* the one format the consumer can read — an ISO input `2024-03-08` is matched by `"%Y-%m-%d"` in `_DATE_INPUT_FORMATS` and rewritten to `08/03/2024`.

### G7 — Batch-level time smearing *(bug, minor but real)*

```754:766:src/workers/tasks/ingestion.py
        reference_time = None
        source_chunk_id = None
        for rel in relationships:
            props = rel.properties or {}
            if not reference_time:
                reference_time = props.get("source_timestamp") or props.get("valid_at")
```

`reference_time` is taken from the *first* relationship in the batch that has one, then applied to every relationship in the batch. When a batch spans chunks from different sessions, all facts inherit one session's date.

### G8 — Measurement hides the failure, and the reported score is prompt-compensated *(methodology gap)*

Beyond the ~0.6% coverage noted in the stage table, the cat-2 score is not evidence that the *pipeline* handles time. The LoCoMo answer prompt carries hand-written, benchmark-specific temporal rules:

> `- TEMPORAL: Prefer the relative phrasing used in dialogue when available… For "last weekend" / charity race timing relative to a Thursday/Friday session, prefer the preceding Sunday over Saturday when choosing one day.` (`benchmarks/locomo/prompts.py:13`)

The answer LLM is re-deriving temporal facts at read time from the session date embedded in the raw passage text, guided by rules tuned to this dataset. That produces 0.86 on cat 2 while the pipeline never normalises a single relative expression — and none of it transfers to a real `/retrieve/context` consumer, who has neither those rules nor a reason to invent them. The healthy signal already present is `answerable_rate = gold_in_context` (`benchmarks/locomo/metrics.py:176`), which decouples retrieval from answering; that is the hook a supersession metric should attach to.

### Deliberate trade-offs (not bugs)

- Running invalidation at write time rather than query time is **correct** given the sub-second budget, and should be kept.
- Generic predicate labels are a deliberate, defensible schema choice for multi-hop traversal. The fix is not to specialise labels but to add an explicit claim identity (see below).
- Doing entity/event resolution before invalidation is the right order.

---

## The append-only tension, stated directly

`README.md:198-200` says nothing is deleted, events are never merged away, and the provenance trail always survives. **The current implementation already violates this.** `_invalidate_superseded_relationships` mutates existing edges in place via `update_properties` (`src/workers/tasks/ingestion.py:494-503`), and it does so in the least auditable way available: it records *that* an edge became invalid but not *why*, or *by what*. The philosophy and the code have already diverged; the question is only how to resolve it deliberately.

There is no fundamental tension between append-only provenance and current-truth queries, provided you separate two things the codebase currently conflates:

- **Assertions** — "at ingestion time T', source S claimed P held from t_valid". These are immutable and append-only. This is the provenance record.
- **Validity** — "P is currently true". This is a *derived projection* over assertions, not a fact in its own right.

Append-only should be a property of assertions. Validity is a materialised view. The only reason to store validity on the edge at all is the sub-second budget: a projection you must compute at query time is a projection you cannot afford. So the resolution is to store validity as a **write-once, monotone materialisation** of an appended invalidation fact — a cache with a source of truth behind it.

---

## Proposed minimal representation change

Four fields on every relationship, plus one new edge type, plus one classification. Nothing is deleted; nothing is rewritten twice.

### 1. Bitemporal quad on the edge — all epoch-seconds integers

| Field | Meaning | Timeline |
| --- | --- | --- |
| `t_valid_from` | event time the fact began holding | valid time |
| `t_valid_to` | event time it stopped holding; `NULL` = currently true | valid time |
| `t_ingested` | when BrainAPI learned it | transaction time |
| `t_expired` | when BrainAPI marked it superseded | transaction time |

Integers, not `dd/mm/YYYY` strings. This is the single change that makes range scans, ordering and indexes possible, and it retires G1/G6 at the storage layer. Keep the existing `valid_at`/`invalid_at`/`happened_at` properties untouched for backwards compatibility; write the new fields alongside.

### 2. `statement_key` — the identity of the *claim*, not the event

This is the part with no equivalent in the codebase today, and it is what makes conflict detection sound. Supersession is not "same subject, same predicate, different object" (G3); it is "same claim slot, different filler". Derive:

```
statement_key = hash(subject_uuid, claim_type, object_role)
```

where `claim_type` is a normalised semantic slot taken from the **event hub's type** (`EMPLOYMENT`, `RESIDENCE`, `RELATIONSHIP_STATUS`, `HOBBY`), not from the generic edge label. Two assertions conflict only if they share a `statement_key`.

### 3. `cardinality` — single-valued vs multi-valued

The honest core of the problem: **supersession requires knowing which claims are functional.** "Employer" is single-valued (a new one supersedes the old); "hobby" and "books read" are multi-valued (a new one accumulates). Nothing in BrainAPI knows this today, which is exactly why G3 cannot be repaired by tightening the match condition. This is one boolean per `claim_type`, obtainable from the LLM at write time and cacheable per claim type — never at query time.

Invalidation then becomes sound and cheap: *on write, close the interval of the most recent live assertion sharing this `statement_key` if and only if `cardinality = single` and its `t_valid_from < new.t_valid_from`.* The recency guard fixes the out-of-order defect; `statement_key` + `cardinality` fixes the over-firing.

### 4. `SUPERSEDES` edge — append-only provenance

```
(new_event_hub)-[:SUPERSEDES {at, reason, confidence}]->(old_event_hub)
```

An appended fact, never a mutation. This makes provenance **strictly stronger than today**: you can reconstruct who superseded what, when, and why, which the current in-place stamp cannot express. `t_valid_to` on the superseded edge becomes a write-once monotone cache of this edge's existence, present only so the read path stays index-answerable.

### 5. Indexes — currently zero exist

```cypher
CREATE INDEX rel_valid_to  IF NOT EXISTS FOR ()-[r:MADE]-()      ON (r.t_valid_to);
CREATE INDEX rel_valid_from IF NOT EXISTS FOR ()-[r:MADE]-()     ON (r.t_valid_from);
CREATE INDEX stmt_key      IF NOT EXISTS FOR ()-[r:MADE]-()      ON (r.statement_key);
```

repeated per relationship type in use. Without these the read-time filter degrades to a scan and the sub-second budget is at risk — this is not optional.

### 6. Chunk-level supersession rollup (fixes the actual reported symptom)

Edges already link to their source chunks via `source_chunk_ids` (`src/core/saving/identity.py:64-82`). At write time, roll that up per chunk into `supports_live_claims: bool` and `latest_claim_valid_to: int|NULL`. `_retrieve_passages` can then deprioritise and — more importantly — **annotate** passages that only support superseded claims.

Annotate rather than filter: a chunk typically contains many statements and only some are superseded, so dropping it would cost recall. Zep does exactly this, surfacing validity ranges into the context string rather than hiding facts (F1).

### Read-time query pattern this enables

```cypher
MATCH (n)-[r]-(m)-[r2]-(b)
WHERE n.uuid IN $seeds
  AND r2.flow_key = r.flow_key
  AND r.t_valid_from <= $as_of
  AND (r.t_valid_to IS NULL OR r.t_valid_to > $as_of)
  AND r2.t_valid_from <= $as_of
  AND (r2.t_valid_to IS NULL OR r2.t_valid_to > $as_of)
RETURN ...
```

This is a two-predicate index-backed extension of the existing query at `src/lib/neo4j/client.py:1669-1681`. Cost is an index lookup, not an LLM call — it satisfies the binding constraint by construction.

The API surface it unlocks, all on the cheap path:

- `temporal_mode: "current"` (default, `$as_of = now`) — the fix for the reported bug.
- `temporal_mode: "as_of", as_of: <ts>` — point-in-time truth, a genuinely new capability.
- `temporal_mode: "all"` — today's behaviour, for callers who want the full history.
- `GetContextTriple` gains `valid_from` / `valid_to` / `is_current`, and passages gain a supersession annotation, so a caller can finally *tell* (closes G5).

Deeper temporal reasoning — interval algebra, "what changed between X and Y", contradiction explanation — belongs on the MCP/deep surface where iteration is permitted, per `00-scope-and-constraints.md:20-25`.

---

## Open questions for the maintainer

1. In what format(s) will `source_timestamp` arrive from real (non-LoCoMo) callers — ISO-8601, epoch, or free text — and may we require ISO-8601 and reject unparseable values loudly instead of silently passing them through?
2. Should `valid_at` mean *event time* (when the fact began holding in the world) or *document time* (when we read it), given the code currently sets it to the latter while `happened_at` holds the former?
3. Is `_invalidate_superseded_relationships` (`src/workers/tasks/ingestion.py:465-505`) currently believed to be working, or already suspected of over-firing on generic `MADE` predicates?
4. Is it acceptable for a write-time LLM call to classify each claim type as single-valued or multi-valued, given that the result is cached per claim type and never invoked at query time?
5. Should `/retrieve/context` default to current-truth-only (a breaking change in returned facts) or keep today's behaviour and require callers to opt in via `temporal_mode`?
6. Does "append-only" forbid write-once monotone fields like `t_valid_to`, or only forbid deletion and destructive overwrite of assertions?
7. Should superseded *text chunks* be filtered out of passages, or returned with a "no longer current" annotation and left for the consumer to weigh?
8. Is a full re-ingest of the LoCoMo brain acceptable to validate the change, given `benchmarks/runs/CHECKPOINT_NOTES.md:13-20` already establishes that pattern for provenance work?
9. Should we add a temporal/supersession sub-benchmark (conflict-heavy, LTP-style) rather than relying on LoCoMo category 2, which we show measures event localization rather than supersession?
10. Are the recency term at `src/core/search/entity_info.py:225-235` and the `deprecated` flag on `Predicate` (`src/constants/kg.py:77-78`) intended to be load-bearing, or vestigial and safe to re-specify?

---

## Frontier techniques

### F1 — Bitemporal edge model with LLM-driven edge invalidation (Zep / Graphiti)

**arXiv:** 2501.13956 — *Zep: A Temporal Knowledge Graph Architecture for Agent Memory* (Rasmussen et al., Jan 2025).

**Mechanism (verified from the paper text, §2.1 and §2.2.3, not from recollection).** Zep implements an explicitly bi-temporal model: timeline `T` is the chronological ordering of events, timeline `T'` the transactional order of ingestion. Four timestamps are stored **on edges**: `t'_created` and `t'_expired ∈ T'` track when a fact was created or invalidated *in the system*; `t_valid` and `t_invalid ∈ T` track the range during which the fact *held true*. Each episode carries a reference timestamp `t_ref` which is what lets the extractor resolve relative and partial dates — the paper names exactly the cases BrainAPI drops: *"next Thursday," "in two weeks," "last summer"*. Invalidation uses an LLM to compare a new edge against semantically related existing edges; on a temporally overlapping contradiction it sets the affected edge's `t_invalid` to the invalidating edge's `t_valid`, consistently prioritising newer information along `T'`. Two further details matter here: edge deduplication is **constrained to edges between the same entity pair**, which is the scoping discipline BrainAPI's global predicate-name match lacks; and the retrieval constructor returns `t_valid, t_invalid` alongside each fact, with the context template stating *"These are the most relevant facts and their valid date ranges."*

**Reported gain.** 94.8% vs MemGPT's 93.4% on DMR; up to +18.5% accuracy on LongMemEval with 90% lower response latency, strongest on cross-session synthesis and long-term context maintenance.

**Cost.** One LLM call per candidate contradiction at **write** time. Four extra edge properties. No query-time model.

**Fit.** Very high, and it maps almost field-for-field onto the representation proposed above. The write-time LLM placement is explicitly compatible with the binding constraint. The gap Zep does not close for BrainAPI is that "semantically related existing edges" is scoped by entity pair, which is weaker than a `statement_key` + `cardinality` test when predicates are generic — and Zep is independently reported to score only 7% on MemoryAgentBench FactConsolidation (see F2), so its invalidation is *not* sufficient on its own.

**Verdict: adapt.** Take the four-timestamp bitemporal quad, `t_ref`-anchored extraction, and surfacing validity ranges into the returned context. Do not take LLM-only contradiction detection as the sole gate; combine it with the deterministic `statement_key` test.

### F2 — Deterministic version-aware aggregation instead of LLM freshness judgement

**arXiv:** 2606.01435 — *Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution*.

**Mechanism.** On MemoryAgentBench's FactConsolidation task, the authors argue the bottleneck is the **assembly step**, not storage: baselines delegate conflict resolution to LLM-mediated retrieval or generation instead of version-aware aggregation. Replacing the LLM-judgement answer pipeline with candidate extraction plus a deterministic `max(serial)` (matched backbone, retrieval, chunking, `TOP_K`) yields +10.8 points on single-hop, widening from +8 at 6K context to +21 at 262K. The mechanism ports from `max(serial)` to `max(timestamp)`.

**Reported gain.** 78.0% FC-SH with gpt-4o-mini (94.8% with gpt-4o) vs HippoRAG-v2 54%, BM25 48%, Mem0 18%, **Zep/Graphiti 7%**. Multi-hop 30.2% → 51.5%. At matched 262K, +28 points over HippoRAG-v2.

**Cost.** Essentially zero — it *removes* an LLM call. The authors are candid that this is a whole-pipeline effect and isolating the resolver is future work, and that on LongMemEval's knowledge-update slice deterministic aggregation only ties LLM judgement (57.8% vs 64.4%, n=45).

**Fit.** Directly on point for the binding constraint: this is the strongest published evidence that current-value conflicts should be resolved by deterministic aggregation over a version field, which is exactly what `t_valid_from` + `statement_key` + `cardinality` provides. The Zep/Graphiti 7% figure is also the clearest warning that adopting F1's shape without a deterministic resolver is not enough.

**Verdict: adopt.** This is the cheapest high-leverage change available and it is the intellectual justification for the whole representation proposal.

### F3 — State-aware overlay: keep superseded records, label current / historical / transition

**arXiv:** 2607.01935 — *A-TMA: Decoupling State-Aware Memory Failures in Long-Term Agent Memory*.

**Mechanism.** Names BrainAPI's exact failure: **"ghost memory"** — old, current, and transition facts coexist in the memory bank, remain mixed during retrieval, and mislead the answer model. ATMA is an overlay that *keeps* superseded and transition records (compatible with append-only), builds evidence packets for the query's requested state view, and exposes explicit `current` / `historical` / `transition` labels to QA. The authors argue for **decoupled evaluation** of bank-, retrieval-, and answer-level failures, "since final QA accuracy can hide where ghost memory occurs."

**Reported gain.** On LTP (LoCoMo Temporal Plus, their conflict-heavy benchmark) Graphiti+ATMA improves conflict accuracy by **+0.240 absolute** over Graphiti. On LoCoMo, temporal F1 rises from **0.0295 to 0.1705**. Gains are host-dependent.

**Cost.** An overlay layer plus state labels on records; retrieval must be state-view aware.

**Fit.** Two things transfer directly. First, the label triple maps onto the proposed `is_current` / `valid_from` / `valid_to` response fields, and "keep superseded records" is precisely how append-only and current-truth are reconciled. Second — and this is the more important point for BrainAPI — their finding that *final QA accuracy hides where ghost memory occurs* is an independent confirmation of G8: LoCoMo cat-2 at 0.86 is exactly the kind of aggregate that conceals the problem. Note their LoCoMo temporal F1 numbers are not comparable to this repo's judge-accuracy metric.

**Verdict: adopt** the state-labelling and decoupled-evaluation discipline; treat LTP as the model for the missing conflict-heavy benchmark.

### F4 — Bitemporal property graphs as a storage model

**arXiv:** 2111.13499 — *Bitemporal Property Graphs to Organize Evolving Systems* (Rost et al., Oracle / University of Leipzig).

**Mechanism.** A one-year industrial collaboration producing four artefacts: a bitemporal property graph model, a temporal graph query language, a conception of continuous event detection, and a prototype bitemporal graph database. Establishes that property graphs with valid-time and transaction-time extensions are a prime candidate for organising evolving, multi-dimensional time-series-like relationships.

**Cost.** Conceptual; adopting the full model would mean a query-language layer BrainAPI does not need.

**Fit.** Useful as the canonical grounding that valid-time/transaction-time separation on *property graph edges* is a solved, industrially validated design rather than an invention — which matters because BrainAPI is a property graph and the proposal above adds exactly these two dimensions. The event-detection and dedicated-query-language portions are out of scope.

**Verdict: adapt** the model, reject the query-language and event-detection machinery as over-scoped.

### F5 — Forgetting-aware accuracy as a metric

**arXiv:** 2604.20006 — *From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents* (Memora).

**Mechanism.** Argues existing benchmarks frame long-term memory as fact retrieval from past conversations, giving little insight into consolidation or frequent knowledge updates — the precise criticism that applies to LoCoMo cat 2. Introduces **FAMA (Forgetting-Aware Memory Accuracy)**, a metric that *penalises reliance on obsolete or invalidated memory*, over weeks-to-months conversations across remembering / reasoning / recommending.

**Reported finding.** Across four LLMs and six memory agents: frequent reuse of invalid memories and failure to reconcile evolving memories; memory agents offer only marginal improvements.

**Cost.** Requires labelled supersession pairs. Cheap to approximate in this repo: the graph already knows which assertions were invalidated, so a superseded-fact-leakage rate can be computed directly from returned context without new human labels.

**Fit.** This is the missing Stage-5 instrument. A FAMA-style metric attaches naturally to the existing `gold_in_context` / `answerable_rate` hook (`benchmarks/locomo/metrics.py:176`), which already separates retrieval quality from answer quality.

**Verdict: adopt** as the primary regression guard, because without it no temporal fix here is verifiable.

### F6 — Reference-time temporal expression normalization and timeline extraction

**arXiv:** 2406.05265 — *TLEX: An Efficient Method for Extracting Exact Timelines from TimeML Temporal Graphs*; supporting context 2503.18085 (*GRAPHTREX*, clinical temporal relation extraction, +5.5% tempeval F1 and up to +8.9% on long-range relations).

**Mechanism.** TLEX converts TimeML annotations into exact timelines in a trunk-and-branch structure, checks temporal-graph consistency, identifies the specific relations involved in an inconsistency, and — the distinctive contribution — explicitly identifies **sections whose order is indeterminate**. Evaluated on 385 TimeML texts across four corpora: 123 are inconsistent, 181 have more than one main timeline, 2,541 indeterminate sections; sampling accuracy 98–100% at 95% confidence.

**Cost.** Full TimeML annotation is far heavier than BrainAPI needs.

**Fit.** Reject the pipeline; adopt two ideas. (a) **Indeterminate order is a first-class outcome.** BrainAPI currently coerces unresolvable expressions into free-text `happened_at` that silently poisons every consumer (G1/G6); representing "unknown/indeterminate" explicitly is strictly better than a string that fails to parse. (b) TIMEX-style normalization against an explicit reference time is a well-established task with known failure modes — worth treating as a bounded, testable unit (a fixture table of expression × reference date → expected ISO output) rather than a prompt instruction, which is what `src/constants/prompts/scout_agent.py:154` currently relies on.

**Verdict: adapt** narrowly — normalization contract plus an explicit indeterminate state.

### F7 — Knowledge-conflict taxonomy as shared vocabulary

**arXiv:** 2403.08319 — *Knowledge Conflicts for LLMs: A Survey*.

**Mechanism.** Categorises conflicts as context-memory, **inter-context**, and intra-memory, reviewing causes, LLM behaviour under each, and available mitigations.

**Fit.** Names BrainAPI's specific pathology precisely: returning a filtered graph triple *and* an unfiltered contradicting passage in one payload is an **inter-context conflict that the memory layer manufactures itself** (G4). Useful because it reframes the seam as a first-class known failure mode rather than an implementation slip, and because the survey's finding that LLMs resolve such conflicts unpredictably is the argument for why the memory layer must not delegate this to the consumer.

**Verdict: adopt** as framing/vocabulary. No implementation.

### F8 — Cheap NLI-based contradiction detection at write time

**arXiv:** 2410.04068 — *ECon: On the Detection and Resolution of Evidence Conflicts*.

**Mechanism.** Generates diverse validated evidence conflicts and evaluates detectors. Findings: NLI and LLM models show **high precision** in detecting answer conflicts, though weaker models suffer low recall; factual-consistency models struggle with lexically similar answer conflicts where NLI and LLMs do better; for resolution, LLMs often favour one side without justification and fall back on internal knowledge when they hold prior beliefs.

**Cost.** A small NLI model at write time — materially cheaper than an LLM call, and off the query path entirely.

**Fit.** Good as the *second* gate behind the deterministic `statement_key` test: use the cheap structural test to find candidate pairs, then NLI to confirm genuine contradiction before closing an interval. High precision is the property that matters, since a false invalidation destroys true history (the G3 over-firing failure). The resolution findings reinforce F2: do not let a model arbitrate freshness.

**Verdict: adapt** as an optional precision gate, only after the deterministic path is in place.

### F9 — Point-in-time primitives on the deep-navigation surface

**arXiv:** 2510.06002 — *Deterministic Legal Agents: A Canonical Primitive API for Auditable Reasoning over Temporal Knowledge Graphs*.

**Mechanism.** Argues that standard RAG over text fragments cannot preserve hierarchy, temporality, or causal provenance. Specifies a typed, atomic, composable primitive API mediating between a probabilistic LLM and a deterministic symbolic substrate, under a principle of **"Probability Isolation"**: uncertainty is confined to intent translation, semantic anchoring, and final synthesis, while structural, temporal, and causal traversals execute as deterministic operations. Shifts interaction from single-shot retrieve-then-generate to Reason–Act–Observe with primitives for point-in-time retrieval, context reconstruction, provenance tracing, and impact analysis. This is a formal architectural specification, not an empirical benchmark.

**Fit.** This is essentially a formalisation of the maintainer's own two-tier router intent (`00-scope-and-constraints.md:16-25`): deterministic, index-answerable temporal primitives on the cheap path; iterative reasoning confined to the deep surface. "Point-in-time retrieval" is the `as_of` primitive proposed above, and "impact analysis" is the natural MCP-tier companion ("what changed about X between T1 and T2"). No reported accuracy numbers, so cite it as design justification only.

**Verdict: adapt** for the MCP/deep tier once the cheap path is correct.

### F10 — Rejected: temporal KG completion and forecasting

**arXiv:** 2405.18106 (*TPAR*, unified interpolation/extrapolation TKG reasoning), 2607.14886 (*RAPTOR*, reachability-aware pretraining for RL-based multi-hop TKG forecasting on ICEWS).

**Why rejected.** These predict *missing or future* timestamped links. BrainAPI's problem is the opposite: it has the facts and cannot tell which are current. They also assume a dense corpus of quadruples with clean, orderable timestamps (ICEWS14/05-15/18) — the precise asset BrainAPI lacks, per G1. RL-based multi-hop path exploration is additionally incompatible with a sub-second budget. Revisit only if BrainAPI ever needs to *infer* unstated validity intervals, and only on the deep surface.

**Verdict: reject** for this workstream.

### F11 — Rejected for now: decay and strategic forgetting

**arXiv:** 2607.22562 (*SF-AMS*, strategic forgetting with utility-driven survival; reports +9.65 F1 multi-hop on Qwen2.5-7B and +6.91 F1 temporal on GPT-4o-mini over LightMem/Mem0/A-Mem), 2603.15642 (*CraniMem*, gated bounded memory with consolidation and pruning).

**Why rejected now.** Both *prune* low-utility memory. That is in direct tension with `README.md:198-200`, and more importantly it solves a different problem: BrainAPI's failure is not that the store is too large, it is that stored facts are unordered in time and unlabelled as to currency. Introducing forgetting before validity intervals exist would delete history while leaving the supersession bug intact — the worst of both. The reported temporal gains are also plausibly attributable to reducing distractor volume rather than to correct state resolution.

**Verdict: reject** for this workstream; reconsider under `03-memory-substrate` once scaling, not correctness, is the binding constraint.

### Context on what time-sensitive QA systems still fail at

Supporting evidence that this is a hard, unsolved area rather than a local implementation slip: **2311.08002** (*TempTabQA*, 11,454 QA pairs over 1,208 Infobox tables) finds top LLMs trail humans by **>13.5 F1**. **2310.19292** shows that fusing externally extracted temporal graphs into the input substantially improves temporal reasoning and sets SOTA on SituatedQA and three TimeQA splits — i.e. explicit temporal structure beats asking the model to infer time from raw text, which is the current de facto strategy at `benchmarks/locomo/prompts.py:13`. **2203.00255** (TSQA over temporal KGs) reports a 32% absolute error reduction on complex multi-step temporal questions by injecting timestamp ordering into KG embeddings, having identified that off-the-shelf temporal KG embeddings ignore temporal order — the same ordering deficiency G1 creates here. **2601.07978** independently measures Graphiti at 55–56% on LoCoMo versus 77–81% for mem0/RAG/full-context, attributing the gap to retrieval incompleteness rather than reasoning failure — a caution that adding temporal filters can cost recall if they over-fire, which is the central risk in R1 below.

---

## Implementation plan

Ordering principle: **measurement first.** The reported bug is currently invisible to CI (G8), and three of the defects (G1, G3, G6) fail silently inside bare `except` blocks. Any fix shipped before instrumentation is unfalsifiable.

Sizes per the task-breakdown discipline: S = 1–2 files, M = 3–5 files.

### Phase 0 — Make the failure visible

#### Task 0.1: Temporal-format regression fixtures
**Description.** Pin the observed behaviour of `parse_date_string` / `normalize_date_string` / `resolve_relative_date` against the timestamp formats actually supplied, including the real LoCoMo `session_N_date_time` shape, as failing-then-passing tests. Pure test addition; no `src/` change.
**Acceptance criteria.**
- [ ] A fixture table covers `'1:56 pm on 8 May, 2023'`, ISO-8601 with and without timezone, epoch, and `dd/mm/YYYY`.
- [ ] Relative expressions `yesterday`, `last week`, `last Tuesday`, `next quarter`, `last summer`, `two weeks ago`, `a few months ago` are asserted against a fixed reference date.
- [ ] Currently-broken cases are marked `xfail` with the G-number, so Task 1.1 flips them without touching assertions.

**Verification.** `poetry run pytest tests/test_temporal_normalization.py -q`
**Dependencies.** None.
**Files.** `tests/test_temporal_normalization.py`
**Scope.** S

#### Task 0.2: Superseded-fact leakage metric
**Description.** Add a FAMA-style (F5) metric to the LoCoMo harness measuring how often returned context contains an assertion the graph itself has marked invalid. Attaches to the existing `gold_in_context` hook (`benchmarks/locomo/metrics.py:176`) so it scores *retrieval*, not the answer LLM — the decoupling F3 argues for.
**Acceptance criteria.**
- [ ] `metrics.py` emits `superseded_leakage_rate` overall and per category.
- [ ] The metric counts both channels separately: leakage via graph triples and leakage via text passages (this is what makes G4 visible).
- [ ] `report.py` prints it alongside judge accuracy.

**Verification.** `cd benchmarks && ./locomo.sh evaluate --sample conv-26 --run temporal-baseline --no-resume`; confirm a non-null rate, and that passage-channel leakage is materially higher than triple-channel leakage.
**Dependencies.** None.
**Files.** `benchmarks/locomo/metrics.py`, `benchmarks/locomo/report.py`
**Scope.** S

#### Task 0.3: Conflict-heavy temporal probe set
**Description.** Add a small LTP-style (F3) question set targeting supersession specifically — "what is X's current …", "does X still …", "what did X used to …" — since only ~12 of 1,986 LoCoMo questions probe current truth (G8). Derive from existing conversations so no new ingest is needed.
**Acceptance criteria.**
- [ ] ≥40 questions, each with a gold *current* answer and the superseded value it must not return.
- [ ] Scored separately from LoCoMo categories, reusing the existing judge.
- [ ] Baseline recorded before any `src/` change.

**Verification.** `cd benchmarks && ./locomo.sh evaluate --sample conv-26 --run temporal-probe-baseline --no-resume`
**Dependencies.** 0.2
**Files.** `benchmarks/locomo/` (probe set + wiring)
**Scope.** M

### Checkpoint A — the bug is measurable
- [ ] `superseded_leakage_rate` is non-zero and attributed per channel.
- [ ] Probe-set baseline recorded; expected to be far below the 0.86 cat-2 figure, demonstrating cat 2 was measuring event localization.
- [ ] **Review with maintainer before any `src/` change.** Open questions 1, 2, 5 and 9 should be answered here.

### Phase 1 — Make time comparable

#### Task 1.1: Robust reference-time parsing with explicit indeterminate state
**Description.** Fix G1. Extend `parse_date_string` to accept ISO-8601 (with/without time and timezone), epoch, and the observed `'H:MM am/pm on D Month, YYYY'` shape; return an explicit indeterminate marker rather than passing free text through (F6). Add epoch-int emission alongside the existing `dd/mm/YYYY` string so nothing downstream breaks yet.
**Acceptance criteria.**
- [ ] All Task 0.1 `xfail` cases pass with no assertion edits.
- [ ] A new `to_epoch_seconds` helper returns `int | None`, never a string.
- [ ] Unparseable input yields an explicit indeterminate result that callers can branch on; no bare pass-through.
- [ ] `normalize_date_string`'s existing string output is unchanged for already-working inputs.

**Verification.** `poetry run pytest tests/test_temporal_normalization.py tests/test_ingestion_identity.py -q`
**Dependencies.** Checkpoint A.
**Files.** `src/utils/dates.py`, `tests/test_temporal_normalization.py`
**Scope.** S

#### Task 1.2: Broaden relative-expression coverage
**Description.** Extend `_RELATIVE_PATTERNS` beyond the nine anchored literals: word numerals, quarters, seasons, and non-anchored matching within a longer phrase.
**Acceptance criteria.**
- [ ] `next quarter`, `last summer`, `two weeks ago`, `a few months ago` resolve or return indeterminate — never silent pass-through.
- [ ] Ambiguous expressions (`last summer` without a hemisphere) return indeterminate rather than guessing.
- [ ] No regression in Task 0.1 fixtures.

**Verification.** `poetry run pytest tests/test_temporal_normalization.py -q`
**Dependencies.** 1.1
**Files.** `src/utils/dates.py`, `tests/test_temporal_normalization.py`
**Scope.** S

#### Task 1.3: Fix batch time smearing
**Description.** Fix G7. Resolve `reference_time` per relationship from its own `source_timestamp` instead of hoisting the batch's first value (`src/workers/tasks/ingestion.py:754-766`).
**Acceptance criteria.**
- [ ] A batch spanning two source timestamps produces two distinct `valid_from` values.
- [ ] Relationships with no `source_timestamp` are marked indeterminate, not silently given a sibling's date.

**Verification.** `poetry run pytest tests/test_ingestion_standard.py tests/test_ingestion_orchestration_collection.py -q`
**Dependencies.** 1.1
**Files.** `src/workers/tasks/ingestion.py`, `tests/test_ingestion_standard.py`
**Scope.** S

### Checkpoint B — timestamps are orderable
- [ ] Every new edge carries an epoch-int event time or an explicit indeterminate marker.
- [ ] Ingest a two-session fixture and confirm in-graph timestamps sort correctly.
- [ ] `superseded_leakage_rate` re-measured — expected roughly flat, since nothing reads the new fields yet. A change here means an unintended coupling.

### Phase 2 — Represent validity

#### Task 2.1: Bitemporal quad on edges + indexes
**Description.** Write `t_valid_from`, `t_valid_to`, `t_ingested`, `t_expired` as epoch ints on every relationship, and create the temporal indexes — `src/lib/neo4j/client.py` currently has none.
**Acceptance criteria.**
- [ ] All four fields written on every new edge; existing `valid_at`/`invalid_at` retained unchanged.
- [ ] Index creation is idempotent (`IF NOT EXISTS`) and runs on `ensure_database`.
- [ ] `t_valid_to` is `NULL` on creation, never `0` or `""`.
- [ ] Parity between the Neo4j and NetworkX/Postgres backends.

**Verification.** `poetry run pytest tests/test_graph_upsert_contract.py tests/test_networkx_event_centric.py -q`
**Dependencies.** Checkpoint B.
**Files.** `src/lib/neo4j/client.py`, `src/lib/postgresql/networkx_client.py`, `src/workers/tasks/ingestion.py`, `tests/test_graph_upsert_contract.py`
**Scope.** M

#### Task 2.2: `statement_key` and `cardinality`
**Description.** Derive `statement_key = hash(subject_uuid, claim_type, object_role)` from the event hub's semantic type rather than the generic edge label, and attach a `cardinality` marker per claim type (write-time LLM, cached per type — subject to open question 4).
**Acceptance criteria.**
- [ ] Two assertions about the same claim slot share a `statement_key`; two unrelated events by the same actor do not.
- [ ] `cardinality` is resolved once per claim type and cached; no per-edge LLM call.
- [ ] `single` vs `multi` classification verified on a fixture set covering employer/residence (single) and hobbies/books (multi).

**Verification.** `poetry run pytest tests/test_ingestion_identity.py -q`
**Dependencies.** 2.1
**Files.** `src/core/saving/identity.py`, `src/workers/tasks/ingestion.py`, `tests/test_ingestion_identity.py`
**Scope.** M

### Checkpoint C — representation is sound
- [ ] Ingest a fixture where a single-valued fact changes twice and a multi-valued fact accumulates three times; confirm three distinct `statement_key`s for the multi-valued fact and one shared key for the single-valued one.
- [ ] Confirm the temporal indexes are used (`EXPLAIN` on the extended neighbour query).
- [ ] **Review with maintainer** — open questions 3, 4, 6.

### Phase 3 — Detect supersession soundly

#### Task 3.1: Replace the invalidation heuristic
**Description.** Fix G3. Rewrite `_invalidate_superseded_relationships` to close intervals only when `statement_key` matches, `cardinality = single`, and the existing assertion's `t_valid_from` is strictly earlier. Emit the `SUPERSEDES` edge.
**Acceptance criteria.**
- [ ] Two unrelated events by the same actor sharing predicate `MADE` no longer invalidate each other (the over-firing regression test).
- [ ] Out-of-order ingestion leaves the newer fact live (the recency regression test).
- [ ] A genuine single-valued change closes exactly one interval and appends exactly one `SUPERSEDES` edge.
- [ ] `t_valid_to` is written at most once per edge; a second attempt is a no-op.

**Verification.** `poetry run pytest tests/test_ingestion_standard.py tests/test_graphrag_upgrade_helpers.py -q`
**Dependencies.** Checkpoint C.
**Files.** `src/workers/tasks/ingestion.py`, `src/lib/neo4j/client.py`, `tests/test_ingestion_standard.py`
**Scope.** M

#### Task 3.2: Chunk-level supersession rollup
**Description.** Precompute per text chunk whether it supports any currently-live claim, using the existing edge→chunk link (`src/core/saving/identity.py:64-82`). This is what makes the passage channel fixable within budget.
**Acceptance criteria.**
- [ ] Each chunk carries `supports_live_claims` and `latest_claim_valid_to`.
- [ ] Recomputed when an interval closes, not only at first write.
- [ ] Chunks with no extracted claims default to `supports_live_claims = true` (never penalise unclassified text).

**Verification.** `poetry run pytest tests/test_ingestion_standard.py -q`
**Dependencies.** 3.1
**Files.** `src/workers/tasks/ingestion.py`, `src/lib/postgresql/data.py`, `tests/test_ingestion_standard.py`
**Scope.** M

### Checkpoint D — write side correct
- [ ] Full re-ingest of the LoCoMo brain (per `benchmarks/runs/CHECKPOINT_NOTES.md:13-20`, subject to open question 8).
- [ ] Count closed intervals; manually audit a sample of 20 for correctness in both directions (false closure and missed closure).
- [ ] Re-measure leakage — expected still high, because read paths are unchanged.

### Phase 4 — Read current truth everywhere

#### Task 4.1: Index-backed temporal predicate in the graph query
**Description.** Extend `get_event_centric_neighbors` with the `as_of` predicate from the query pattern above, defaulting to now.
**Acceptance criteria.**
- [ ] Superseded triples are absent for `as_of = now` and present for an earlier `as_of`.
- [ ] Latency on the existing context benchmark does not regress beyond the sub-second budget.
- [ ] `NULL` `t_valid_to` is treated as live; indeterminate `t_valid_from` never silently excludes a fact.

**Verification.** `poetry run pytest tests/test_context_retrieval.py tests/test_postgresql_graph_read_query.py -q`, plus a timing check on `/retrieve/context`.
**Dependencies.** Checkpoint D.
**Files.** `src/lib/neo4j/client.py`, `src/lib/postgresql/networkx_client.py`, `tests/test_context_retrieval.py`
**Scope.** M

#### Task 4.2: `temporal_mode` / `as_of` on the request, validity on the response
**Description.** Fix G5. Add `temporal_mode` (`current` default | `as_of` | `all`) and `as_of`; surface `valid_from` / `valid_to` / `is_current` per triple and a supersession annotation per passage — the labelling F1 and F3 both converge on.
**Acceptance criteria.**
- [ ] `temporal_mode: "all"` reproduces pre-change behaviour exactly.
- [ ] `current` is the default (subject to open question 5).
- [ ] Every returned triple carries validity fields; passages carry the annotation.

**Verification.** `poetry run pytest tests/test_context_retrieval.py -q`
**Dependencies.** 4.1
**Files.** `src/services/api/constants/requests.py`, `src/services/api/controllers/retrieve.py`, `tests/test_context_retrieval.py`
**Scope.** M

#### Task 4.3: Close the seam — passages and the remaining read paths
**Description.** Fix G4 and G6. Apply the supersession annotation in `_retrieve_passages` and `historical_context`; move `_is_currently_valid` out of the controller into `src/core/search/` so `/retrieve/entity/status`, `/retrieve/entity/synergies`, hops and the MCP tools inherit it; repair or explicitly retire the dead recency term at `src/core/search/entity_info.py:225-235`.
**Acceptance criteria.**
- [ ] Passage-channel leakage drops materially in the Task 0.2 metric.
- [ ] The validity check lives in exactly one place and is reachable from every read path.
- [ ] The recency term either uses epoch ints and demonstrably changes ranking, or is removed — no silent constant.

**Verification.** `poetry run pytest tests/test_context_retrieval.py tests/test_mcp_traverse_graph.py -q`; then `cd benchmarks && ./locomo.sh evaluate --sample conv-26 --run temporal-phase4 --no-resume`
**Dependencies.** 4.2
**Files.** `src/core/search/` (validity module), `src/services/api/controllers/retrieve.py`, `src/core/search/entity_info.py`, `tests/test_context_retrieval.py`
**Scope.** M

### Checkpoint E — the reported bug is fixed
- [ ] `superseded_leakage_rate` materially reduced on **both** channels.
- [ ] Probe-set accuracy (0.3) materially improved.
- [ ] LoCoMo cat 1/3/4 **not regressed** — the key guard against over-invalidation (see R1).
- [ ] `/retrieve/context` p95 still sub-second.

### Phase 5 — Deep temporal navigation (optional, after E)

#### Task 5.1: Temporal primitives on the MCP surface
**Description.** Expose point-in-time and change-analysis primitives on the deep tier per F9 — "what changed about X between T1 and T2", "why is this fact no longer current" (answerable by walking `SUPERSEDES`).
**Acceptance criteria.**
- [ ] Primitives are deterministic graph operations; no LLM in the traversal.
- [ ] Supersession chains are walkable end to end, demonstrating provenance is stronger than before.

**Verification.** `poetry run pytest tests/test_mcp_traverse_graph.py -q`
**Dependencies.** Checkpoint E.
**Files.** MCP tool registration + `src/core/search/`
**Scope.** M

---

## Risks

| Risk | Impact | Detection | Mitigation |
| --- | --- | --- | --- |
| **R1 — Over-invalidation destroys true history.** The most likely way this work makes things worse; it is the current failure mode (G3) and 2601.07978 measures Graphiti losing 20+ points to retrieval incompleteness rather than reasoning failure. | **High** | LoCoMo cat 1/3/4 accuracy and `answerable_rate` at every checkpoint; count of closed intervals per ingest. | `cardinality = single` gate; strict recency guard; treat closure as opt-in per claim type; F8's high-precision NLI as an optional second gate. |
| **R2 — Latency regression on the cheap path.** Two extra predicates per edge across a 5-tuple join, on a graph with **zero indexes today**. | High | Time `/retrieve/context` at Checkpoints B, D, E; `EXPLAIN` the extended query. | Ship indexes in the same task as the predicate (2.1); epoch ints not strings; fail the checkpoint on p95 regression. |
| **R3 — Default flip breaks existing callers.** `temporal_mode: current` changes returned facts for everyone. | Medium | Contract tests asserting `all` reproduces pre-change output byte-for-byte. | Answer open question 5 before Task 4.2; ship `all` first, flip the default behind a separate decision. |
| **R4 — Backfill gap.** Existing brains have no `t_valid_from`/`statement_key`; `NULL` handling could hide every legacy fact or expose every superseded one. | Medium | Query legacy brain for edges missing the new fields; assert count before/after. | Treat missing `t_valid_from` as live-and-indeterminate (fail open, preserving recall); require re-ingest for full benefit, as already precedented for provenance. |
| **R5 — Wrong `cardinality` classification.** A claim type misclassified `single` silently deletes history across the whole graph. | Medium | Audit the per-type cache; assert on a fixture of known single/multi types. | Cache is per-type and inspectable; start with a small allowlist of high-confidence single-valued types and expand. |
| **R6 — Metric captures the answer LLM, not retrieval.** `benchmarks/locomo/prompts.py:13` already compensates for pipeline gaps (G8), so improvements could be masked or faked. | Medium | Score leakage on returned context independently of the judge (Task 0.2 does this by construction). | Keep the retrieval-level metric primary, per F3's decoupled-evaluation argument; do not add further dataset-specific prompt rules while this work is in flight. |
| **R7 — Event time vs document time resolved wrongly.** If open question 2 is answered differently than assumed, `t_valid_from` semantics invert and every interval is wrong. | High | Fixture where a 2023 session discusses a 2019 event; assert `t_valid_from` is 2019. | Blocking: get open question 2 answered at Checkpoint A, before Phase 2. |
| **R8 — `t_valid_to` mutation is judged to violate append-only.** Would invalidate the materialisation strategy and force query-time projection, which the latency budget cannot absorb. | High | N/A — a design decision, not a detectable defect. | Open question 6 at Checkpoint C. Fallback: keep `SUPERSEDES` as the sole truth and materialise into a separate index structure rather than an edge property. |
