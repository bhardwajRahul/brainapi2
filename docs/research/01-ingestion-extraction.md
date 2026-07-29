# 01 — Ingestion and Knowledge Extraction

Workstream: everything from raw input to written graph. Judged against `00-scope-and-constraints.md`, which fixes the priority order as **multi-hop first, temporal second**, and states that expensive work must move to write time. This workstream *is* write time, so most of the burden of both named pain points lands here.

Read-time consumption of temporal fields is deferred to `05-temporal-truth.md`; this document owns what gets written and how it is marked.

All `file:line` anchors below reflect the tree as read on 2026-07-28. One file, `src/workers/tasks/ingestion.py`, was modified by concurrent work while this analysis was in progress: roughly 24 lines were inserted around line 462, so anchors in that file beyond that point are shifted by about that much relative to the current working tree. Gap 4 is where it matters substantively.

---

## What this workstream does

### Entry points

Three, all landing in `src/workers/tasks/ingestion.py`:

| Entry | Task | Notes |
| --- | --- | --- |
| Free text / textual payload | `ingest_data` (`src/workers/tasks/ingestion.py:527`) | The main path. |
| Structured triples | `ingest_structured_data` (`src/workers/tasks/ingestion.py:1054`) | Caller supplies subject/event/object triples; Scout backfills missing entities. |
| Files | `ingest_file` (`src/workers/tasks/ingestion.py:1379`) | Docling → markdown per page → one `ingest_data` per page. Only when `OCR_MODE=docling`. |
| Files, remote OCR | `src/services/api/routes/ingest.py:138-179` | When `OCR_MODE=docparser` (`src/config.py:571-572`) the route instead posts the file to an external `DOCPARSER_ENDPOINT/ingest` with a bearer token, bypassing `ingest_file` entirely. |

`src/core/files/` contains only an `__init__.py`; `src/services/input/` and `src/services/data/` are thin adapter re-export modules (392 and 79 bytes). There is no separate file-handling or input-normalization layer — file handling is inline in `ingest_file`.

### The free-text path, traced

1. **Raw text is persisted losslessly first.** `ingest_data` builds a `TextChunk` from the whole payload and saves it (`src/workers/tasks/ingestion.py:574-581`), embeds it, and adds it to the `"data"` vector store (`:582-592`). This is a genuine strength and is the same "non-lossy episode store" that Zep (2501.13956, §2.1) and Engram (2606.09900) argue is essential. Graph elements point back to it via `source_chunk_ids`.

2. **Observations** run only in `accurate` mode (`:597-614`), and are called with `context=None` (`:601`).

3. **Enrichment** — `enrich_kg_from_input` → `_enrich_kg_impl` (`src/core/saving/auto_kg.py:81`):
   - Mode select: `mode = "coarse" if config.pipeline_mode == "lightweight" else "granular"` (`auto_kg.py:112`).
   - **Scout** `scout_agent.run(...)` (`auto_kg.py:115-123`) with `reference_time=source_timestamp`.
   - **Architect** `architect_agent.run_tooler(...)` (`auto_kg.py:125-133`) — receives the **entire** input text, not chunks.
   - Relationships are drained with `take_pending_relationships()` (`auto_kg.py:135`) and stamped with provenance (`:139-151`).
   - `should_consolidate = pipeline_mode == "accurate" and config.run_graph_consolidator and enrichment_relationships` (`auto_kg.py:153-157`).

4. **Scout** (`src/core/agents/scout_agent.py:333`):
   - Chunks with `_chunk_text(text, max_chars=6000)` (`:344`, `:346`), implemented in `src/utils/text_chunking.py:1-27`: greedy paragraph packing, hard character slice for oversized paragraphs (`text_chunking.py:20-23`), **zero overlap, no context carry-over**.
   - Each chunk goes through an independent `_run_chunk` call (`:360-371`). Chunks are merged on the key `(name.lower(), type.lower(), happened_at or "")` with **first-wins** semantics (`:372-379`).
   - If the LLM returns nothing parseable, returns an empty entity list silently (`:511-512`).
   - `src/core/chunking/text_chunker.py` is unreachable — the only reference to `text_chunker` anywhere in the repo is its own file header. It is worth noting what is being left unused: it wraps a compiled extension, `chunker_cpp.chunk_text_semantically`, with `max_chunk_size=2000`, `min_chunk_size=500`, and `min_coherence_threshold=0.3` (`text_chunker.py:21-51`), falling back to fixed slices on failure (`:65-74`). So the project already contains a coherence-aware semantic chunker, and the live path instead uses 6000-character paragraph packing with no overlap. Whatever its quality, the boundary problem in gap 6 is being solved by the worse of two implementations already present.

5. **Architect** in tooler mode drives `ArchitectAgentCreateRelationshipTool` (`src/core/agents/architect_agent.py:350-359`). Inside the tool (`src/core/agents/tools/architect_agent/ArchitectAgentCreateRelationshipTool.py`):
   - Entities are resolved by UUID or `TYPE:NAME`; unknown ones become new `ScoutEntity` objects.
   - **Janitor** is invoked only when `self.mode == "granular"` (`:383`), via `run_atomic_janitor` (`:400-409`).
   - Janitor-supplied `fixed_relationships` suppress the corresponding input relationship if the endpoints match or embeddings agree above `0.90` (`:478-513`, threshold carries a literal `TODO` at `:512`).
   - Everything surviving is queued (`:550-551`), and *then* `wrong_relationships` is checked and returned as `status: "ERROR"` to the LLM (`:553-573`).

6. **Persistence** — `process_architect_relationships` (`src/workers/tasks/ingestion.py:707`):
   - `_normalize_relationship_dates` (`:192-205`) resolves `DATE`-typed entity names and `happened_at`.
   - `_resolve_relationship_entities` (`:208-446`) does entity resolution (detailed below).
   - Relationship vectors are computed, then a **global** relationship vector search (k=10) looks for near-duplicates (`:818-831`); if the nearest is within `RELATIONSHIP_DEDUP_MAX_DISTANCE = 0.1` (`:86`, `:165-170`) **and** endpoints match in either direction (`:842-848`), the new relationship is dropped with a bare `continue` (`:854`).
   - Nodes and the edge are written (`:911-971`).
   - `_invalidate_superseded_relationships` runs unconditionally for every persisted edge (`:972-975`).
   - Item-level failures accumulate in `item_errors` and produce `partial_failed`/`failed` status plus a raised `RuntimeError` (`:990-1013`).

7. **Consolidation** — `consolidate_graph` (`src/core/layers/graph_consolidation/graph_consolidation.py:31`): batches of 20, Janitor produces free-form natural-language tasks (`:64-69`), `KGAgent.run_graph_consolidator_operator` executes each (`:84-88`), failures printed and skipped (`:89-94`).

### Identity and provenance

`src/core/saving/identity.py`:
- `stable_node_id(name, type, happened_at, supplied_uuid)` (`:13-25`) — a supplied UUID short-circuits (`:19-20`); otherwise `sha256(lower(name) | lower(type) [| normalized_date if type == event])`.
- `stable_relationship_id(tail, predicate, tip, flow_key)` (`:28-42`).
- `stamp_provenance` (`:85-109`) merges a **list** of `source_chunk_ids` and a single `source_timestamp`.

### Entity resolution, in detail

`_resolve_relationship_entities` (`src/workers/tasks/ingestion.py:208-446`), three cascading strategies per non-event entity (`:298-307`):

1. `_uuid_match` (`:241-248`).
2. `_exact_match` on name + type (`:250-258`).
3. `_vector_match` (`:260-295`): embed the name, search `k=5` in the `nodes` store, **require the entity type string to appear verbatim (lowercased) in the candidate's labels** (`:278`), require cosine ≥ `NODE_RESOLUTION_SIMILARITY = 0.9` (`:84`, `:286`), and **abstain entirely if the top two candidates are within 0.02** (`:290-291`).

Events are excluded from that path (`:303-304`) and get a second, neighbour-anchored pass (`:341-408`): candidate events must be shared by at least half the already-resolved anchors (`:384`, `:387`), must not have a conflicting normalized date (`:394`), and must clear `EVENT_RESOLUTION_NAME_SIMILARITY = 0.7` (`:85`, `:400`), abstaining on ambiguity (`:402-404`). This is a reasonable cross-document event-coreference heuristic and deserves credit.

Anything unresolved gets `stable_node_id(...)` (`:431-436`).

### The agent swarm, as it actually exists

| Agent | File | Status in the live path |
| --- | --- | --- |
| Scout | `scout_agent.py` (518 lines) | Active. |
| Architect | `architect_agent.py` (1518 lines) | Active via `run_tooler`. |
| Janitor | `janitor_agent.py` (735 lines) | Active in `granular` only. |
| Observations | `observations_agent.py` (47 lines) | Called at `ingestion.py:598`, always with `context=None`. Writes to the observation store, **not to the graph**. |
| Temporal | `temporal_agent.py` (9 lines) | Header comment only. Referenced nowhere. |
| Validator | `validator_agent.py` (9 lines) | Header comment only. Referenced nowhere. |

`ArchitectAgent.run_structured` (`architect_agent.py:932`) contains the Janitor retry loop but is used by the structured-triples path, not free text.

### ADR status

`docs/decisions/001-ingestion-structure.md` and `docs/decisions/002-structured-ingestion-specific-processing.md` are **empty stubs**: 001 has `Status: Accepted`, `Date: 2026-07-27`, and `--` under Context, Decision, Alternatives Considered, and Consequences. 002 is the same shape and shorter. I am not inferring their intent. `AGENTS.md`, which `CLAUDE.md` points at, is a zero-byte file.

---

## Guarantees and where they break

The stated guarantee: every fact in the source text becomes a correct, deduplicated, temporally-marked element of an append-only event-centric graph, and nothing enters the graph that the source does not support.

Ranked by impact on the two confirmed pain points (multi-hop first, temporal second).

### 1. Node identity is decided by exact surface string, which both fragments and catastrophically over-merges — BUG (multi-hop)

Identity is `sha256(lower(name) | lower(type))`. Everything about multi-hop depends on the middle node of a chain being *one* node. Reproduced against the real code:

```
Money/MONEY         -> 8f6db1fc-a23b-400a-2d5b-423758246dfb
money/money         -> 8f6db1fc-a23b-400a-2d5b-423758246dfb
Acme Inc./ORGANIZATION -> e224b2c3-80c0-08e5-f017-165758f7806d
Acme Inc/ORGANIZATION  -> 47ac63a4-3a19-7210-20ac-96ff9d0d9b92
Acme Inc./ORG          -> 37229514-298f-6027-53c5-bcd3bdabc774
EVENT WENT_TO 08/05/2023  -> eb91dafe-84d5-868d-d3e4-d9339241d609
EVENT Went    08/05/2023  -> ef241913-3b5d-e544-d468-cda44d9c9f0c
EVENT WENT_TO 8 May, 2023 -> 45ace7ed-4cb5-a54b-6be4-36f408bb5a35
```

Two distinct failures, both severe:

**Over-merge.** The Architect prompt's own worked examples instruct the model to emit placeholder nodes named after their type: `{"type": "UNIT", "name": "Friends"}` (`src/constants/prompts/architect_agent.py:31`) and `{"type": "MONEY", "name": "Money"}` (`:111`). Every monetary amount ever ingested therefore lands on **one shared node**. The value survives on the edge's `amount` field (`ingestion.py:950`), but the object node becomes a hub joining every funding round, salary, and price in the graph to every other. For a system whose second goal is surfacing non-obvious connections, this manufactures an unbounded supply of obvious-and-wrong ones. Any 2-hop traversal through `MONEY` is garbage.

**Fragmentation.** A trailing period, an abbreviation, or a type-label variation forks the node. `_vector_match` is the only defence, and it is gated on `entity_type not in labels` (`:278`) — an exact lowercase string comparison against a type vocabulary the LLM invents freely. `PERSON` vs `PEOPLE` vs `HUMAN` can never resolve to each other no matter how similar the names are. The type string is simultaneously load-bearing for identity and completely uncontrolled.

Compounding this, the alias mechanism intended to mitigate it is dead code. At `:315` the resolved entity's name is overwritten with the graph node's name; at `:330` the guard is `entity.name != node.name`, which is now always false. `aliases` is never populated.

### 1b. Ingestion never populates the `triplets` vector store that triple-level retrieval reads — BUG (multi-hop)

Ranked second overall, behind gap 1 only because gap 1 is upstream of it. Found late, while cross-checking a parallel trace.

Every `add_vectors` call in the repository, with its target store:

| Writer | Store |
| --- | --- |
| `src/workers/tasks/ingestion.py:611` (post-shift) | `"data"` |
| `src/core/saving/ingestion_manager.py:83-84` | `"nodes"` |
| `src/core/saving/ingestion_manager.py:129-130` | `"relationships"` |
| `src/core/agents/tools/kg_agent/KGAgentCreateNodeTool.py:105-106` | `"nodes"` |
| `src/core/agents/tools/kg_agent/KGAgentCreateRelationshipTool.py:163-164` | `"relationships"` |
| `src/core/agents/tools/kg_agent/KGAgentAddTripletsTool.py:137-138`, `:160-161` | `"nodes"` |
| **`src/core/agents/tools/kg_agent/KGAgentAddTripletsTool.py:180`** | **`"triplets"`** |

That last row is the only writer to `"triplets"` anywhere in `src/`, and `KGAgentAddTripletsTool` is an agent tool that is not on the ingestion path. Meanwhile the retrieval controller's triple-level entry point calls `vector_search.search_triplets(...)` (`src/services/api/controllers/retrieve.py:79`), which searches `store="triplets"` (`src/utils/vector_search.py:33-40`). The store is declared with its own dimension (`src/constants/embeddings.py:23`) and is searched by `KGAgentSearchGraphTool.py:115` as well.

**So no fact ingested through `/ingest/`, `/ingest/structured`, or `/ingest/file` is ever retrievable by triple-level vector search.** Single-hop lookups that enter through `"nodes"` still work, which is consistent with the maintainer's report in `00-scope-and-constraints.md` that single-hop is acceptable while chained questions fail. This is an ingestion defect with a retrieval-side symptom, and it may account for a meaningful share of the multi-hop gap on its own.

Two things I did not determine: whether `"triplets"` was intended to be populated by ingestion and the wiring was lost, or whether `retrieve.py:79` is reading the wrong store; and how much of the measured multi-hop deficit this accounts for. Both need the maintainer and `02-retrieval-multihop.md`. It is cheap to test either way — populate the store for one brain and re-run the multi-hop slice.

### 2. The Janitor cannot reject anything, and fails open silently — BUG (accuracy)

`README.md:163` draws the Janitor as the gate into the graph — `Janitor -->|"OK"| Graph` — and `README.md:170` promises it loops "until it's clean." The code does neither. Two independent defects compose badly.

**No veto.** In the live path, relationships are queued for persistence at `ArchitectAgentCreateRelationshipTool.py:550-551`, and only afterwards is `wrong_relationships` inspected and returned as an error to the LLM (`:553-573`). The LLM then produces a corrected relationship, which is *also* queued. Net result: **both the rejected edge and its replacement are written.** The only edges the Janitor can actually suppress are ones it rewrote itself into `fixed_relationships` with matching endpoints (`:478-513`). "This fact is wrong, do not write it" is not expressible — the output schema (`src/constants/prompts/janitor_agent.py:226-246`) has `required_new_nodes`, `fixed_relationships`, and `wrong_relationships` (which carries fix `instructions`), and no delete verdict.

The same inversion exists in `run_structured`: `relationships_to_persist.extend(pending_batch)` executes at `architect_agent.py:1254`, *before* `_persist_relationships` invokes the Janitor at `:1255`.

**Fail-open.** `run_atomic_janitor` returns the string `"OK"` on success (`janitor_agent.py:732-733`) and `structured_response` otherwise (`:735`) — which is `None` when the LLM produced nothing parseable. Callers use `getattr(janitor_response, "wrong_relationships", [])` (`architect_agent.py:927`, tool `:554`). `getattr` on `"OK"` and on `None` both yield `[]`. **A Janitor that failed to produce output is indistinguishable from a Janitor that approved everything**, and nothing records which happened.

**Convergence.** The loop terminates — `max_janitor_iterations = 3` (`architect_agent.py:1250`, `:1263`) with a `(tail, tip, name)` dedup guard at `:1305-1307`. But it does not converge in any useful sense: it accumulates edges, and on exhausting its budget it breaks and returns everything with no flag, no error, no marker (`:1263-1264`). "Looping until it's clean" (`README.md:170`) is three tries and then whatever we have.

Downstream, `retrieve.py:422-424` filters on `invalid_at` and `deprecated`, so these are the only two levers that can ever remove a written fact from an answer — and neither is under the Janitor's control.

### 3. Temporal marking is broken end to end, and demonstrably no-ops on the benchmark in use — BUG (temporal)

`src/utils/dates.py` accepts ten formats (`:5-16`), all requiring full day-month-year. Unparseable values pass through verbatim (`:77`). The canonical stored form is `%d/%m/%Y` (`:76`) — day-first, **not ISO 8601**, so no lexicographic ordering or range comparison is possible downstream. `resolve_relative_date` requires the reference timestamp itself to parse with those same ten formats (`:96-98`), returning the input unchanged otherwise.

LOCOMO supplies `source_timestamp` verbatim from the dataset (`benchmarks/locomo/dataset.py:131`, `:148`). The actual values look like `'1:56 pm on 8 May, 2023'`. Executed against the real module:

```
parse('1:56 pm on 8 May, 2023')     -> None
'yesterday'          -> 'yesterday'
'two weeks ago'      -> 'two weeks ago'
'3 months ago'       -> '3 months ago'
'2023-05-08T13:56:00Z' -> '2023-05-08T13:56:00Z'
```

**Every relative date in the LOCOMO run is left unresolved**, because the reference timestamp never parses. Even with a parseable reference, `3 months ago`, `March 2026`, `2019`, and ISO-8601 all fail. `valid_at` is then set to that same unparseable string (`ingestion.py:418-422`), and `invalid_at` likewise (`:499`), with a `%d/%m/%Y` fallback (`:500`) — so the graph carries a mixture of formats, none of them sortable.

Separately, there is **one timeline, not two.** `valid_at` is populated from `source_timestamp` — when the content was observed — with no extraction of when the fact became true. `happened_at` on events is the only fact-time signal and it is a free-form string that participates in the identity hash. The `temporal_agent.py` that would presumably own this is a 9-line stub.

### 4. Supersession blindly deprecates the event graph it is built on — BUG (temporal)

`_invalidate_superseded_relationships` (`ingestion.py:465-505`) matches existing edges from the same subject on **predicate name equality alone** (`:484`), skips same-tip (`:486`), and marks everything else `invalid_at` + `deprecated: True` (`:494-503`). It runs unconditionally for every persisted edge (`:972`).

There is no model of predicate cardinality. For a genuinely functional predicate this is the right rule. For BrainAPI's own event-hub vocabulary — where a subject has many `:MADE` edges to many event hubs by design — **every new event a subject participates in deprecates all of their prior events**. Since `retrieve.py` filters on `deprecated`/`invalid_at`, those events become invisible. This directly produces the maintainer's stated symptom in reverse: history vanishes, and it contradicts the append-only design in `README.md`.

Note the shape is right and the guard is missing: MemStrata (2606.26511) uses exactly a deterministic `(subject, relation, object)` supersession rule, but only for facts it has established are contradictory.

**Partially fixed while this document was being written.** An uncommitted change now in the working tree adds `_is_event_entity` and returns early when either endpoint is an event, and additionally requires the existing edge's direction to be outgoing. That closes the event-hub case described above — independent confirmation that this defect is real — and it shifts the line anchors in this section by roughly 19 lines. Two parts remain open: there is still no cardinality model, so a legitimately one-to-many predicate between two *non-event* nodes is still blindly deprecated; and `invalid_at` is still written in the unparseable form described in gap 3.

### 5. Duplicate detection cannot distinguish a duplicate from a contradiction — BUG (temporal + accuracy)

Relationship dedup is embedding-similarity plus an endpoint check: nearest neighbour within distance `0.1` and matching endpoints → `continue` (`ingestion.py:835-854`).

MemStrata (2606.26511) measured this directly: cosine similarity separates a *contradicted* fact from a *duplicated* one at **AUROC 0.59, near chance**, because contradictions are often more embedding-similar to the original than a rephrasing is. So this code path preferentially swallows the updates it most needs to act on. And the drop is a bare `continue` — no log, no `item_errors` entry, no record that a fact was discarded.

Two further problems in the same block: the endpoint check accepts **either direction** (`:842-848`), silently collapsing `A→B` and `B→A` for asymmetric predicates; and the candidate search is **global** (`:826-831`, `k=10`) rather than constrained to the subject-object pair. Graphiti constrains edge dedup to edges between the same entity pair specifically to avoid both problems and to shrink the search space (2501.13956, §2.2.2). Here, for a common predicate, all ten neighbours can belong to other entity pairs, so a genuine duplicate for *this* pair is never examined.

### 6. Extraction is schema-free, and the prompts disagree with each other and with the README — BUG (multi-hop)

`README.md` specifies the triangle of attribution with fixed predicates: `:MADE`, `:TARGETED`, `:OCCURRED_WITHIN`. The prompts do not implement that.

`src/constants/prompts/architect_agent.py` contains three incompatible policies:
- `:15-20` — the fixed three.
- `:226-231` — `:(MADE|COVERED_ROLE|EXPERIENCED|etc..)`, `:(TARGETED|RESULTED_IN|etc..)`, `:(OCCURRED_WITHIN|etc..)`. The `etc..` licenses open invention, and the worked example immediately below uses `MOVED`, `INTO_LOCATION`, `ACCOMPLISHED_ACTION`, `HAPPENED_WITHIN`, `EXPERIENCED` (`:261-310`) — `HAPPENED_WITHIN` for `OCCURRED_WITHIN`, `INTO_LOCATION` for `TARGETED`.
- `:365-370`, `:407-410` — fully open `--(predicate)-->`, constrained only by a style note at `:382`.

Event-hub naming is likewise inconsistent, and because event identity hashes the name (`identity.py:22-25`), naming convention *is* identity. The Architect prompt uses `WENT_TO`, `KNEW`, `PARTICIPATED_IN` (`:28-36`); the Scout prompt uses `Went`, `Was in`, `Partecipated in`, `Covered role` (`src/constants/prompts/scout_agent.py:49-74`, typo included). Two agents in the same pipeline are told to name the same event two different ways, and `EVENT_RESOLUTION_NAME_SIMILARITY = 0.7` has to bridge that on embeddings of two- and three-token strings.

Entity types are equally free (`MONEY`, `UNIT`, `ROLE`, `CITY`, `ORGANIZATION` appear in examples with no registry), while `_vector_match` requires exact type equality. Unconstrained vocabulary plus exact-match gating is the worst of both.

The consequence for multi-hop is direct: reliable Cypher patterns cannot be written against a predicate vocabulary that is invented per-ingestion.

### 7. The Janitor never checks whether a fact is in the source text — GAP (accuracy)

Every item in the Janitor's revision protocol (`src/constants/prompts/janitor_agent.py:211-218`) is structural: identity resolution, direction audit, stripping numbers from names, never merging events, label normalization, UUID formatting. `:224` instructs it to "Maintain original intent." Nothing asks whether the relationship is supported by `CONTEXT_TEXT`.

So a fluent, well-formed, correctly-directed hallucination passes validation cleanly. For a product whose first goal is "always-accurate answers," the validation layer has no faithfulness check at all.

### 8. `lightweight` mode produces a structurally different graph, not a cheaper one — TRADE-OFF, mispriced

`PIPELINE_MODE` defaults to `accurate` (`src/config.py:568-570`) and is process-global. In `lightweight`:

- Scout uses the coarse prompt, which puts facts into **properties of ordinary entities instead of event hubs**: `{"type": "CITY", "name": "New York City", "properties": {"friends_count": 12}}` (`scout_agent.py` prompt `:124`) and `{"type": "ORGANIZATION", "name": "Acme Inc.", "properties": {"funding_amount": 100000000, "funding_date": "19/01/2026"}}` (`:134`). No `EVENT` nodes.
- The Janitor never runs (`ArchitectAgentCreateRelationshipTool.py:383`).
- Observations never run (`ingestion.py:597`).
- Consolidation never runs (`auto_kg.py:153-154`).

This is not a speed/quality dial. Facts buried in property maps are invisible to relationship-level retrieval and to the entire triangle traversal. Because the setting is global and no node or edge records which mode produced it (only `TextChunk` carries `BRAIN_VERSION`, `ingestion.py:578`), flipping the env var permanently heterogenizes the graph with no way to tell the halves apart.

### 9. File ingestion discards all provenance and splits on page boundaries — BUG

`ingest_file` builds each page's payload as `{"data": {"data_type": "text", "text_data": markdown}, "brain_id": ...}` (`:1435-1440`). No filename, no page number, no `source_timestamp`, no `meta_keys`. Consequences:

- Nothing in the graph records which file or page a fact came from; provenance terminates at an opaque chunk id.
- `source_timestamp` is `None`, so `valid_at` is never set (`:418`) and Scout's `reference_time` is `None` (`auto_kg.py:121`) — **no relative date in any ingested document can ever be resolved.**
- Each page is an independent Celery task with its own Scout/Architect/Janitor run and no shared context (`:1420-1423`, `:1434-1446`). A sentence or fact spanning a page break is split across two isolated extractions.
- Pages run concurrently. Identical surface forms still converge because `stable_node_id` is deterministic, so this is not a duplicate-explosion risk — but *variant* surface forms fragment, and each page independently re-derives entity context from nothing.

### 10. Chunk isolation defeats cross-chunk coreference — BUG (multi-hop)

Scout's chunks are processed by independent stateless calls (`scout_agent.py:360-371`) against a chunker with zero overlap (`src/utils/text_chunking.py`). A pronoun or shortened reference in chunk 2 has no access to its antecedent in chunk 1. Merging is `(name, type, happened_at)` first-wins (`:372-379`), so a later chunk's richer description and properties are silently discarded rather than merged.

There is also an asymmetry worth noting: **Scout sees 6000-character chunks; the Architect and Janitor see the entire text** (`auto_kg.py:126`, tool `:403`). For long inputs the Architect prompt is unbounded, and the two agents are reasoning over different views of the same document.

### 11. Failure surfacing is inconsistent by layer — BUG

Persistence-level failures are handled well: collected into `item_errors`, reported as `partial_failed`/`failed` with counts, and raised (`ingestion.py:990-1013`). Everything above and around it is not:

| Silent path | Location |
| --- | --- |
| Scout produces nothing parseable → empty entity list | `scout_agent.py:511-512` |
| Janitor produces nothing parseable → treated as approval | `janitor_agent.py:735` + `getattr` callers |
| Janitor loop exhausts its budget with edges still wrong | `architect_agent.py:1263-1264` |
| Relationship dropped as a "duplicate" | `ingestion.py:854` |
| Self-relationship skipped | `ingestion.py:789-793` (print only) |
| Node embedding fails, node still written via the edge | `ingestion.py:900-909` |
| Consolidation task fails | `graph_consolidation.py:89-94` (print only) |

The node-embedding case deserves emphasis: on failure the node is dropped from `graph_nodes` (`:904`, `:909`) but `add_relationship` still runs at `:913` with that node as an endpoint. The node exists in the graph and **not** in the vector store, so it is permanently invisible to `_vector_match` — guaranteeing future duplicates of exactly the entity that already had trouble.

### 12. The graph is not append-only: consolidation can physically delete, driven by an LLM from prose — BUG

`consolidate_graph` has the Janitor emit free-form natural-language tasks (`graph_consolidation.py:64-69`) which `KGAgent.run_graph_consolidator_operator` executes (`:86-88`). The tool set that agent is handed (`src/core/agents/kg_agent.py:507-539`) is:

| Tool | `kg_agent.py` |
| --- | --- |
| `KGAgentExecuteGraphOperationTool` | `:507` |
| `KGAgentCreateNodeTool` | `:514` |
| `KGAgentCreateRelationshipTool` | `:521` |
| **`KGAgentRemoveNodeTool`** | `:527` |
| **`KGAgentRemoveRelationshipTool`** | `:533` |

So an LLM, acting on natural-language instructions produced by another LLM, holds arbitrary graph-operation execution plus **physical delete** on nodes and relationships. `README.md:163` labels the graph "append-only"; it is not. This runs by default — `accurate` is the default mode (`src/config.py:568-570`) and `RUN_GRAPH_CONSOLIDATOR` defaults to true (`:562-564`) — and failures are printed and skipped (`graph_consolidation.py:89-94`) with no record of which mutations were applied. There is no audit log, so a deletion is unattributable and unrecoverable.

This is the sharpest disagreement with the literature in this document. Zep explicitly rejected the weaker version of it: "We chose this approach [predefined Cypher queries] over LLM-generated database queries to ensure consistent schema formats and reduce the potential for hallucinations" (2501.13956, §2.2.1). Both Zep and Engram (2606.09900) invalidate and never delete, precisely so provenance and supersession chains survive.

Scope, to be fair to the design: the Janitor builds a 2-hop neighbourhood snapshot around the new relationships' endpoints, vector-filtered at `0.35` (`janitor_agent.py:268`, `:280`), and processes relationships in batches of 20 (`graph_consolidation.py:28`, `:45`). It is a local normalizer over recent writes, not an all-pairs pass — the blast radius is bounded, but within that radius it is unconstrained.

### 13. Documented behaviour that does not exist — GAP

`README.md:168` states the Observations agent writes notes "taking the previously known context into account." It is called with `context=None` (`ingestion.py:601`) and writes to the observation store, never to the graph. The Temporal and Validator agents are 9-line stubs referenced nowhere. Together with gap 2, three of the five documented swarm behaviours do not hold: the Janitor does not gate writes (`README.md:163`), it does not loop until clean (`:170`), and Observations has no prior context (`:168`).

### 14. Retry safety and session bookkeeping are backend-dependent and partly broken — BUG

Two smaller defects that share a cause — orchestration state that assumes a single clean pass:

**Source-chunk persistence is idempotent on one backend and not the other.** PostgreSQL writes the chunk with `ON CONFLICT (id) DO UPDATE` (`src/lib/postgresql/data.py:183`); Mongo uses a bare `insert_one` (`src/lib/mongo/client.py:48`). Celery is configured with `task_acks_late=True` and `task_reject_on_worker_lost=True`, so redelivery is a designed-for event, not an edge case. On Mongo, a worker lost mid-task therefore duplicates the source chunk, and since `source_chunk_ids` provenance is list-unioned at the graph layer (`src/lib/neo4j/client.py:271-276`), the duplicate propagates into provenance rather than being absorbed.

**The session pending-task counter only counts down.** `{brain_id}:session:{session_id}:pending_tasks` is read (`ingestion.py:1042`), decremented (`:1072`), and deleted (`:1388`) — and there is no `incr` or `incrby` anywhere in `src/`. The counter is never initialized or raised, so it descends into negative values from the first decrement. Any logic gating on it reaching zero is either dead or firing at the wrong time. I did not determine which, because it depends on Redis's behaviour for the read path when the key is absent, but no reachable code raises it.

### Assets already in the codebase and unused where they matter

Three capabilities the project has already built and does not apply on the ingestion path. This matters for sizing the recommendations below: several are less "build" than "wire up."

1. **LLM adjudication of entity identity.** The structured path resolves its anchor node with vector candidate generation followed by `verify_entity_existence(entity_name, entity_types, entity_meta_description, pool_nodes, brain_id)` (`src/core/agents/kg_agent.py:219`), called at `ingestion.py:1134-1142`. That is precisely the Graphiti entity-resolution pattern (2501.13956, §2.2.1). It is used for one node in one path, and never in the free-text path where nearly all data flows — including at the two abstention branches that most need it (gap 1, task 2.4).

2. **Linguistic analysis.** `src/utils/nlp/` provides NER, lemmas, POS tags, and noun chunks across 12 languages (`src/constants/spacy_models.py`, `src/utils/nlp/ner.py:39-146`). The only importer in the repository is the retrieval controller (`retrieve.py:44`, `:754`). So query strings get linguistic normalization and ingested entity names — the ones that must match them — get only `strip().lower()`. Any lemmatization or singularization in the canonicalization work of task 2.1 can reuse this rather than add a dependency.

3. **Semantic chunking.** The unreachable `chunker_cpp` wrapper described above, versus 6000-character packing in the live path (task 1.8).

### Deliberate trade-offs, correctly made

Not everything here is a defect, and these should not be "fixed":

- **Abstaining on ambiguity** (`ingestion.py:290-291`, `:402-404`) — failing toward duplication rather than a wrong merge is the right default for an accuracy-first system. It is the *absence of a follow-up* that is the gap, not the abstention.
- **Never merging EVENT nodes** (`janitor_agent.py:216`) — preserving distinct historical instances is correct for an event-centric model.
- **Lossless raw-text store** (`ingestion.py:574-592`) — matches Zep and Engram, and Engram measured that facts alone lose recall while facts plus retrieved chunks recover it (2606.09900).
- **Deterministic content-addressed IDs** (`identity.py`) — makes ingestion idempotent and concurrent page ingestion safe. The problem is the *inputs* to the hash, not the hashing.
- **Neighbour-anchored event resolution** (`ingestion.py:341-408`) — a genuinely thoughtful cross-document event coreference heuristic.
- **Bounded Janitor iterations** (`architect_agent.py:1250`) — guarantees termination.

---

## Open questions for the maintainer

1. Is the predicate vocabulary meant to be the fixed `:MADE` / `:TARGETED` / `:OCCURRED_WITHIN` triangle from `README.md`, or an open set that the Architect invents per ingestion?
2. Should an EVENT node's name be a normalized verb phrase (`WENT_TO`) or a sentence-case label (`Went`) — given that the choice determines the node's identity hash?
3. Is `_invalidate_superseded_relationships` intended to apply to event-hub predicates like `:MADE`, or only to functional attribute predicates?
4. What is the intended format of the `source_timestamp` API field, given that the current parser rejects both ISO-8601 and the LOCOMO dataset's own format?
5. Should `valid_at` mean "when we observed this" or "when this became true in the world" — and do you want both tracked separately?
6. When the Janitor concludes a relationship is simply wrong rather than fixable, should it be able to prevent the write entirely?
7. When the Janitor exhausts its three iterations with relationships still marked wrong, should the ingestion be reported as degraded rather than completed?
8. Is `lightweight` mode intended to produce a graph without event hubs, or is that an unintended consequence of the coarse Scout prompt?
9. Are `lightweight` and `accurate` ever mixed within one brain, and if so should each node and edge record which pipeline wrote it?
10. Should the placeholder nodes the prompts demonstrate (`MONEY`/"Money", `UNIT`/"Friends") be per-instance nodes, or are they deliberately intended as shared type-level hubs?
11. Should file ingestion preserve filename and page number as queryable provenance, or is the opaque chunk id considered sufficient?
12. Are `temporal_agent.py` and `validator_agent.py` planned work, or abandoned ideas that should be deleted?
13. Should the Observations agent's output ever reach the graph, or is the observation store a separate retrieval surface by design?
14. `src/core/chunking/text_chunker.py` wraps a compiled semantic chunker (`chunker_cpp`, coherence-thresholded) and is unreachable, while the live path uses fixed 6000-character paragraph packing. Was the C++ chunker abandoned for a reason — quality, build friction, the threading lock — or did the wiring simply get lost?
15. Was the `triplets` vector store meant to be populated by ingestion (gap 1b), or is `retrieve.py:79` reading the wrong store? Either way, was triple-level vector retrieval ever observed working on ingested data?
16. Consolidation currently holds physical delete on nodes and relationships (gap 12). Is destructive consolidation intended, or should it be restricted to invalidation like the rest of the write path?
17. What is the acceptable added write latency and token cost per ingested unit for accuracy improvements — is doubling write cost acceptable if it removes a class of error?

---

## Frontier techniques

### A. Ontology-grounded post-extraction correction

**Mechanism.** Keep extraction open-domain, then apply embedding-based canonicalization of types and predicates, then targeted LLM correction of only the ontology violations that remain. Deferring correction to a post-extraction stage avoids repeated LLM calls during extraction.

**arXiv.** 2605.29168 — *Better Later Than Sooner: Neuro-Symbolic Knowledge Graph Construction via Ontology-grounded Post-extraction Correction*.

**Reported gain.** Improved KG consistency and substantially reduced token usage versus constrain-at-extraction pipelines, while preserving downstream QA quality; validated by measuring SPARQL graph-pattern occurrence, i.e. by whether the resulting graph is actually queryable.

**Cost.** One canonicalization pass over new types/predicates plus a small number of correction calls. Cheaper than schema-constrained extraction, and it is write-time work, which `00-scope-and-constraints.md` explicitly prefers.

**Fit.** Very high, and it maps onto a component that already exists. The Janitor is already a post-extraction corrector; it is already told to "use `get_schema` (target: `relationship_types`) to normalize relationship labels" (`janitor_agent.py:217`). What is missing is a persisted canonical registry and an embedding-based merge of near-synonym labels, rather than ad-hoc per-batch alignment against whatever noise is already in the graph.

**Verdict. Adopt.** This is the single highest-leverage item, because it attacks the top-priority pain point (multi-hop) at its root: gap 1 and gap 6 are the same problem seen from two sides.

### B. Simultaneous schema induction with entity-and-event modelling

**Mechanism.** Extract triples and induce the schema in the same pass, modelling both entities and events, and use conceptualization to organize instances into semantic categories.

**arXiv.** 2505.23628 — *AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora*.

**Reported gain.** Schema induction reaches **92% semantic alignment with human-crafted schemas with zero manual intervention**; outperforms baselines on multi-hop QA. Demonstrated at 50M+ documents, 900M+ nodes.

**Cost.** Schema induction over a corpus; the resulting type system then needs storage and versioning.

**Fit.** High, and unusually good: AutoSchemaKG models events as first-class alongside entities, which most schema work does not, and BrainAPI is event-centric. It answers question 1 and 2 empirically rather than by decree — you do not have to choose between schema-free and hand-authored ontology.

**Verdict. Adapt.** Do not adopt the full web-scale pipeline. Use it to induce a type/predicate registry from the graph BrainAPI has already built, then feed that registry to technique A as the canonicalization target.

Supporting references for the same problem: 2607.21610 (*SCOPE and SCION*) is a benchmark and auditable reference pipeline for schema induction whose core target is **event types and within-event argument roles** — structurally the closest published work to the triangle of attribution, and useful for evaluating an induced registry. 2412.20942 (*Ontology-grounded Automatic KG Construction under Wikidata schema*) authors an ontology from generated competency questions and maps relations onto Wikidata, which is a cheaper route if interoperability matters more than fit.

### C. Entity resolution as a first-class stage, with triple reflection

**Mechanism.** Treat LLM-generated KGs as noisy by default and denoise explicitly: entity resolution to eliminate redundant entities, plus "triple reflection" to remove erroneous relations.

**arXiv.** 2510.14271 — *Less is More: Denoising Knowledge Graphs For Retrieval Augmented Generation* (DEG-RAG).

**Reported gain.** Drastically reduced graph size and consistently improved QA across diverse Graph-RAG variants. The paper also contributes what it describes as the first systematic evaluation of entity resolution for LLM-generated KGs, sweeping **blocking strategies, embedding choices, similarity metrics, and merging techniques**.

**Cost.** A resolution pass over the graph; the empirical sweep means the parameters need tuning rather than guessing.

**Fit.** Directly applicable, and the surface is larger than it first appeared. A full sweep of the tree turns up at least twelve independent hardcoded similarity constants governing identity and dedup decisions:

| Value | Location | Governs |
| --- | --- | --- |
| `0.9` | `ingestion.py:84`, `:286` | node resolution by vector |
| `0.7` | `ingestion.py:85`, `:400` | event resolution by name |
| `0.02` | `ingestion.py:290`, `:402` | ambiguity abstention band |
| `0.1` | `ingestion.py:86`, `:165` | relationship dedup max distance |
| `0.90` | `architect_agent.py:886`, `ArchitectAgentCreateRelationshipTool.py:511` | near-duplicate descriptions (carries the `TODO`) |
| `0.92` | `list_reduction.py:89` | `reduce_list` pairwise dedup default |
| `0.35` | `janitor_agent.py:280` | consolidation 2-hop snapshot filter |
| `0.75` | `names.py:66` | Levenshtein+label acceptance |
| `0.6` / `0.4` | `names.py:49` | Levenshtein vs label-Jaccard blend weights |
| `0.8` | `adapters/graph.py:71` | neighbour vector reduction |
| `0.5` | `retrieve.py:199`, `entity_sibilings.py:35` | retrieval-side neighbour filters |

Nothing ties these to each other and none has a recorded justification. Two additional details compound it: `src/utils/nlp/names.py` implements a *different* matching function (Levenshtein blended with label Jaccard, accepted at `0.75`) reached from `KGAgentAddTripletsTool.py:108-114`, so the codebase has two unrelated entity-matching algorithms with unrelated thresholds; and `list_reduction.py:57-65` falls back, when no embeddings client resolves, to an **8-dimensional character-sum hash** used as an embedding — under which nearly any two strings of similar character composition will exceed `0.92`. That fallback is currently only reachable from retrieval-side neighbour reduction, but it is a silent correctness cliff rather than a degradation.

This paper is the evidence base these numbers lack.

**Verdict. Adopt** the methodology for choosing thresholds and merge strategy. Do not adopt wholesale graph compaction — "less is more" pruning risks removing exactly the low-degree bridge nodes that multi-hop depends on, which is the opposite of this project's priority.

### D. LLM adjudication of ambiguous resolution candidates

**Mechanism.** Cheap candidate generation (embeddings + full-text), then an LLM decides duplicate/not-duplicate over the candidate set, and on a merge emits a canonical name and summary. Graphiti runs this per entity with the episode as context (2501.13956, §2.2.1). Cost-efficient variants only spend LLM calls where the decision is uncertain: 2401.03426 initializes candidate partitions, defines uncertainty, and selects the most valuable pairs to query, with error-tolerant handling of LLM mistakes. 2506.02509 goes further and has the LLM cluster records directly rather than comparing pairs.

**arXiv.** 2501.13956, 2401.03426, 2506.02509.

**Reported gain.** LLM-CER (2506.02509) reports up to **150% higher accuracy, +10% F-measure, and 5× fewer API calls** than pairwise baselines across nine real-world datasets. Zep attributes part of its DMR/LongMemEval results to this resolution design.

**Cost.** LLM calls at write time, bounded if gated on uncertainty.

**Fit.** Excellent, and nearly free to wire up: `KGAgent.verify_entity_existence` (`kg_agent.py:219`) already implements exactly this and is already called with a vector-search candidate pool at `ingestion.py:1116-1142`. The abstain-on-ambiguity branches (`:290-291`, `:402-404`) are precisely the uncertainty signal that 2401.03426 says should trigger the LLM — and today they instead silently create a duplicate.

**Verdict. Adopt.** Highest ratio of gain to implementation cost in this document: route the existing abstention branches into the existing adjudicator.

### E. Bi-temporal model with contradiction-triggered supersession

**Mechanism.** Track four timestamps: transaction-time created/expired, and valid-time valid/invalid. On a new edge, compare against semantically related existing edges; on a genuine temporally-overlapping contradiction, set the old edge's `t_invalid` to the new edge's `t_valid`. Invalidate, never delete, so provenance and the supersession chain survive (2501.13956, §2.2.3). Engram (2606.09900) runs the same model with **no LLM call per fact**. MemStrata (2606.26511) reduces it further to a deterministic `(subject, relation, object)` supersession rule in a bi-temporal ledger with no similarity threshold and no LLM call.

**arXiv.** 2606.26511, 2501.13956, 2606.09900.

**Reported gain.** MemStrata: ties RAG on static knowledge, reaches **0.95–1.00 accuracy on evolving knowledge where RAG reaches 0.20–0.47**, and drives the **stale-fact-error rate from 15–40% to ~0%**, at ~2.1s retrieval latency versus ~16–18s for LLM-reranking baselines. Engram: 83.6% vs 73.2% full-context on the full 500-question LongMemEval_S at ~8× fewer tokens. Zep: up to 18.5% accuracy improvement on LongMemEval with 90% lower latency.

**Crucially,** MemStrata quantifies why BrainAPI's current approach cannot work: cosine similarity distinguishes a contradicted fact from a duplicated one at **AUROC 0.59**, near chance, because contradictions are often *more* embedding-similar to the original than rephrasings are.

**Fit.** Exact. This is gaps 3, 4, and 5 with a published, cheap, benchmarked answer, and it targets the maintainer's confirmed #2 pain point ("superseded facts come back as current truth"). The stale-fact-error rate is also the metric this project currently lacks.

**Verdict. Adopt.** Note that the mechanisms are deliberately *not* LLM-heavy, which suits the write-time budget.

### F. Grounding verification via required source spans

**Mechanism.** Require the extractor to emit, alongside each structured item, the context it came from. Align that context against the document with a string-based global aligner, then score the alignment. Anything unaligned or low-scoring is flagged as unsafe.

**arXiv.** 2510.00276 — *SafePassage: High-Fidelity Information Extraction with Black Box LLMs*.

**Reported gain.** **Up to 85% reduction in hallucinations** with minimal risk of flagging non-hallucinations, and high agreement with human judgments of extraction quality — so the same pipeline doubles as an evaluator. Notably, a **transformer encoder fine-tuned on a small number of task-specific examples outperformed an LLM scoring model** at flagging unsafe passages, with annotations collectible in 1–2 hours.

**Cost.** Extra output tokens per extraction (the span), a cheap string alignment, and optionally a small encoder. No extra LLM call on the critical path if the encoder is used.

**Fit.** Very high. It fills gap 7 exactly: the Janitor validates structure and never grounding. BrainAPI already has the raw text stored (`ingestion.py:574-581`) and `source_chunk_ids` on every element, so the document side of the alignment is already available. And the "small encoder beats LLM scorer" finding matters given the write-time cost constraint.

**Verdict. Adopt.** This is the mechanism that makes gap 2's veto meaningful — a veto with no faithfulness signal has nothing to act on.

### G. Atomic decontextualized facts as the extraction unit

**Mechanism.** Split input into minimal, **self-contained** atomic facts before graph construction, build per-fact temporal graphs with dual-time modelling, then merge in parallel (2510.22590). SynthKG (2410.16597) reaches the same conclusion from the data-generation side: systematic chunking plus **decontextualization** plus structured extraction, then distilled into a single-step small model. Dense X Retrieval (2312.06648) independently establishes propositions — atomic, self-contained natural-language expressions of a single factoid — as a superior retrieval unit.

**arXiv.** 2510.22590 (ATOM), 2410.16597 (SynthKG / Distill-SynthKG), 2312.06648 (Dense X Retrieval).

**Reported gain.** ATOM: **~18% higher exhaustivity, ~33% better stability across runs, >90% latency reduction** versus baselines. Distill-SynthKG: a fine-tuned smaller model surpasses baselines up to eight times larger on KG quality, and improves downstream retrieval and QA. Dense X: proposition-level indexing significantly outperforms passage-level.

**Cost.** An extra decomposition pass per input, offset by parallelism; ATOM reports a large net latency *win*.

**Fit.** High. Decontextualization is the correct fix for gaps 9 and 10: instead of trying to preserve context across chunk and page boundaries, rewrite each unit to stand alone before extraction. ATOM's stability result is directly relevant — non-determinism across runs is invisible in the current design because nothing measures it.

**Verdict. Adapt.** Adopt decontextualization and dual-time extraction. Defer the distillation half (2410.16597) until the pipeline is stable enough to be worth distilling; it is a cost optimization, not an accuracy one, and fine-tuning now would freeze in the current defects.

### H. Multi-round extraction with reflection

**Mechanism.** After an initial extraction pass, run a reflection step to catch omissions and suppress hallucinations. Zep applies a Reflexion-inspired pass after entity extraction specifically "to minimize hallucinations and enhance extraction coverage," and supplies the previous `n = 4` messages as context for named entity recognition (2501.13956, §2.2.1).

**arXiv.** 2501.13956; GraphRAG's two-stage graph index (2404.16130) is the canonical reference for LLM-built graph indexes.

**Reported gain.** Not separately ablated in the Zep paper — I could not determine the isolated contribution of the reflection pass.

**Cost.** Roughly doubles extraction calls.

**Fit.** Moderate. Recall is a real problem (Scout is single-pass and silently returns empty on parse failure, `scout_agent.py:511-512`), and the `n = 4` prior-context window is a cheap partial answer to cross-chunk coreference that requires no decontextualization machinery. But without a measurement of omission rate, adding a pass is unfalsifiable.

**Verdict. Adapt, after measurement.** Establish an omission baseline first (technique J), then decide. Ranked below the others because the gain is unquantified in the source.

### I. Late chunking

**Mechanism.** Embed all tokens of the long text with a long-context embedding model, then apply chunk boundaries after the transformer and before mean pooling, so each chunk embedding carries whole-document context. No additional training required.

**arXiv.** 2409.04701 — *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*.

**Reported gain.** Superior results across retrieval tasks versus independently-encoded chunks.

**Cost.** Requires a long-context embedding model; changes the embedding pipeline.

**Fit.** Low **for this workstream**. Late chunking improves chunk *embeddings*; BrainAPI's chunking problem is that the *LLM extraction call* lacks cross-chunk context (`scout_agent.py:360-371`), which a better pooled vector does not address. It is potentially relevant to the `"data"` text-chunk store used at retrieval time.

**Verdict. Reject for ingestion; refer to `02-retrieval-multihop.md`.** Rejected because the assumption it addresses — that the retrieval unit is an embedding — does not hold for the extraction path, where the unit is an LLM prompt.

### J. Extraction evaluation that does not depend on exact match

**Mechanism.** Evaluate generative extraction on multiple dimensions — topic similarity, uniqueness, granularity, factualness, completeness — rather than precision/recall against human-annotated reference triples (2402.10744). Complementarily, evaluate KG construction with explicit hallucination and omission metrics plus BERTScore-based graph similarity (2502.05239).

**arXiv.** 2402.10744 (GenRES), 2502.05239.

**Reported gain.** GenRES establishes that precision/recall fails to justify generative extraction performance, that human-annotated reference relations are frequently incomplete, and — importantly for technique A — that **prompting LLMs with a fixed set of relations or entities can itself cause hallucinations**. Its scores align with human preference judgments.

**Cost.** Harness work only.

**Fit.** Essential and unavoidable. Every other proposal here needs a measurement to be a task rather than a hope, and BrainAPI's current schema-free extraction has no reference triples to match against by construction.

**Verdict. Adopt first.** This is a prerequisite, not an improvement.

The GenRES finding is also the most important tension in this literature and should temper technique A: naive schema constraining at extraction time *increases* hallucination. That is precisely why 2605.29168's deferred post-extraction correction is the right architecture — canonicalize after open extraction, not before.

### K. Rejected, with reasons

- **Hyper-relational / n-ary KG embedding** (2305.18256, 2308.06512, 2306.02199, 2404.09848, 2411.06191, 2508.03280, 2505.11803, 2307.10219, 2305.06588, 2411.07019, 2312.09219). I searched this area specifically because BrainAPI's event hub is a reification of an n-ary fact. Almost all of it is representation learning for link prediction over a *given* graph — it assumes the facts already exist and asks how to embed them. It says nothing about how to extract or canonicalize, so it is out of scope for ingestion. **Rejected: assumes the graph as input.**
- **Event-centric attribution / viewpoint modelling** (2503.03563). Parameterized predicates for facts valid only within a stated viewpoint, for controversial events with multiple contested accounts. Genuinely relevant to a problem BrainAPI has no answer for — two sources asserting incompatible facts — but it presumes a conflict-detection layer that does not exist yet, and BrainAPI's current conflict handling (technique E) must land first. **Defer, revisit after E.**
- **Fine-tuned / distilled extraction models** (2410.16597 distillation half, and the fine-tuning result in 2502.05239). 2502.05239 found fine-tuned models improved accuracy and reduced hallucination and omission **but generalized worse** on a held-out dataset. Distilling the current pipeline would bake in the defects catalogued above. **Rejected for now: premature.**
- **Community detection / summarization at write time** (2404.16130, 2410.05779, and Zep's label-propagation variant). Real write-time work with real retrieval benefit, but it is graph-structure enrichment rather than extraction quality, and it operates on whatever the extractor produced. Building communities over a graph with a `MONEY` super-node and a fragmented type system would summarize the noise. **Defer: fix identity first; owned by `03-memory-substrate.md`.**
- **Memory-architecture comparators** — MemGPT (2310.08560), HippoRAG (2405.14831), A-MEM (2502.12110), Mem0 (2504.19413). Read for comparison, not adoption. HippoRAG's contribution is Personalized-PageRank retrieval, not writing. A-MEM's "memory evolution," where a new memory triggers updates to the attributes of existing ones, is the closest conceptual match to what BrainAPI's append-only model forbids, and is worth revisiting for `05-temporal-truth.md`. Mem0 is the most directly comparable published number, since it evaluates on **LOCOMO** — the same benchmark in `benchmarks/locomo/` — across single-hop, temporal, multi-hop, and open-domain categories, and reports a graph-memory variant scoring ~2% above its base configuration. **Use as baselines, not as designs.**

---

## Implementation plan

Sizing: **S** = 1–2 files, **M** = 3–5 files. Every task has a verification command. Phases are ordered by dependency; each checkpoint is a go/no-go.

### Phase 0 — Make the pipeline measurable

Nothing below can be judged without this. No production code changes.

| # | Task | Size | Acceptance criteria | Verification |
| --- | --- | --- | --- | --- |
| 0.1 | Extraction-quality harness: for a fixed corpus, report per-input hallucination rate (triples with no supporting span), omission rate, and predicate/type cardinality, following 2502.05239 and 2402.10744 | M | Runs on ≥50 LOCOMO units; emits JSON with all four metrics; two runs on identical input report a stability score | `python -m benchmarks.extraction_quality --units 50 --out runs/eq-baseline.json` |
| 0.2 | Graph-health report: count of nodes sharing a name with their own type, top-20 nodes by degree, type-vocabulary size, predicate-vocabulary size, count of predicate pairs with cosine > 0.9 | S | Identifies the `MONEY`/"Money" class of super-node without being told to look for it | `python -m scripts.graph_health --brain-id <id>` |
| 0.3 | Stale-fact-error probe, per 2606.26511: ingest a fact, ingest its update, query, and record how often the superseded value is returned | S | Produces a single stale-fact-error-rate number | `python -m benchmarks.stale_facts --brain-id <id>` |
| 0.4 | Date-coverage report over every distinct `happened_at`, `valid_at`, and `invalid_at` string in the graph, bucketed into parsed / passed-through-verbatim | S | Reports the percentage of temporal values that are unparseable | `python -m scripts.date_coverage --brain-id <id>` |

**Checkpoint 0.** Baselines recorded and reviewed. Expected from the code read: high type/predicate cardinality, at least one type-named super-node, ~100% unparseable temporal values on LOCOMO. If those do not appear, the analysis above is wrong and the plan must be revisited before any change ships.

### Phase 1 — Stop the bleeding (small, independent, high-confidence)

Each of these is a localized fix for a defect established above. None depends on the others.

| # | Task | Size | Acceptance criteria | Verification |
| --- | --- | --- | --- | --- |
| 1.1 | Extend `src/utils/dates.py`: accept ISO-8601 with time, year-only, month-year, and the LOCOMO `'H:MM am/pm on D Month, YYYY'` form; store canonically as ISO-8601; extend relative patterns to word numerals and months | S | `resolve_relative_date('two weeks ago', '1:56 pm on 8 May, 2023')` returns an absolute date; existing `%d/%m/%Y` inputs still parse; 0.4 shows unparseable share below 5% | `pytest tests/test_dates.py -v && python -m scripts.date_coverage --brain-id <id>` |
| 1.2 | Gate `_invalidate_superseded_relationships` on an explicit allowlist of functional predicates. The event-hub exemption already landed in the working tree; what remains is the cardinality model for non-event predicates | S | A subject with three `:MADE` edges to three event hubs retains all three as valid (already satisfied); a legitimately one-to-many non-event predicate also retains all edges; a functional attribute predicate still supersedes | `pytest tests/test_event_hub_invalidation.py -v` |
| 1.3 | Distinguish Janitor failure from Janitor approval: return an explicit outcome type instead of `"OK"` / `None`, and record unvalidated batches in `item_errors` | S | A Janitor whose response fails to parse yields a `partial_failed` task status; a genuine approval does not | `pytest tests/test_janitor_outcomes.py -v` |
| 1.4 | Make silent drops observable: log and record `item_errors` entries for dedup drops (`ingestion.py:854`), self-relationship skips (`:789`), node-embedding failures (`:900-909`), and consolidation task failures (`graph_consolidation.py:89`) | S | Every discarded fact appears in the task record with a reason; counts reconcile against relationships submitted | `pytest tests/test_ingestion_accounting.py -v` |
| 1.5 | Do not write a node whose embedding failed as a bare relationship endpoint; either retry the embedding or fail the item | S | No node exists in the graph without a corresponding vector-store entry | `python -m scripts.graph_health --check orphan-vectors --brain-id <id>` |
| 1.6 | Thread `filename`, page number, and `source_timestamp` through `ingest_file` into each page's `ingest_data` payload | S | A fact extracted from page 7 of `report.pdf` reports both; relative dates in documents resolve | `pytest tests/test_ingest_file_provenance.py -v` |
| 1.7 | Fix the dead alias branch at `ingestion.py:329-336` by capturing the incoming surface form before `:315` overwrites it | S | Resolving "Acme Inc" onto "Acme Inc." leaves "Acme Inc" in `aliases` | `pytest tests/test_entity_resolution.py -k alias -v` |
| 1.8 | Resolve the two-chunker split: either delete `src/core/chunking/text_chunker.py` or bench its C++ semantic chunker against the live 6000-char packer and keep the winner — pending open question 14 | S | Exactly one chunking implementation is reachable from the ingestion path, and the choice is recorded with a number behind it | `rg -n "chunk_text\|TextChunker" src/` |
| 1.9 | Resolve the `triplets` store mismatch (gap 1b): either populate it during relationship persistence alongside `"relationships"`, or point `retrieve.py:79` at the store ingestion actually writes — pending open question 15. **Do 1.9 first and measure it alone**, since it may move multi-hop more than the rest of Phase 1 combined | S | A fact ingested via `/ingest/` is returned by triple-level vector search; the multi-hop slice from 0.1 is re-measured against this change in isolation | `pytest tests/test_triplet_store_population.py -v && python -m benchmarks.locomo.cli run --slice multihop` |
| 1.10 | Give Mongo's `save_text_chunk` (`src/lib/mongo/client.py:48`) upsert semantics matching PostgreSQL's `ON CONFLICT (id) DO UPDATE`, so late-acked task redelivery does not duplicate source chunks | S | Ingesting the same chunk id twice yields one stored chunk on both backends; a simulated worker loss and redelivery produces no duplicate | `pytest tests/test_chunk_idempotency.py -v` |
| 1.11 | Fix or remove the session pending-task counter (gap 14): it is decremented and deleted but never incremented anywhere in `src/` | S | The counter reflects outstanding tasks and reaches zero exactly once per session, or the dead gate is removed outright | `rg -n "pending_tasks" src/ && pytest tests/test_session_lifecycle.py -v` |

**Checkpoint 1.** Re-run all of Phase 0. Stale-fact-error rate and unparseable-temporal share should both drop sharply from 1.1 and 1.2 alone, and multi-hop should move on 1.9 — attribute it before proceeding, because a large jump there changes the priority of Phase 2. Hallucination and cardinality metrics should be unchanged; if they move, something in Phase 1 had unintended reach.

### Phase 2 — Canonicalization (the multi-hop fix)

Depends on Checkpoint 1 and on 0.2 for a target to move.

| # | Task | Size | Acceptance criteria | Verification |
| --- | --- | --- | --- | --- |
| 2.1 | Induce a type and predicate registry from the existing graph per 2505.23628, with embedding-clustered canonical forms and recorded surface variants | M | Registry covers ≥95% of existing type and predicate occurrences; every cluster is human-reviewable | `python -m scripts.induce_registry --brain-id <id> --out registry.json` |
| 2.2 | Canonicalize types and predicates at write time against the registry, after open extraction, per 2605.29168 | M | Predicate cardinality drops by ≥50% with no loss in 0.1 completeness; `PERSON`/`PEOPLE` collapse to one type | `python -m benchmarks.extraction_quality --units 50 --out runs/eq-canon.json && python -m scripts.graph_health --brain-id <id>` |
| 2.3 | Forbid type-named placeholder nodes: reject or per-instance-qualify any node whose name matches its type, and correct the prompt examples at `architect_agent.py:31` and `:111` | S | 0.2 reports zero type-named nodes; monetary amounts no longer share a node | `python -m scripts.graph_health --check type-named --brain-id <id>` |
| 2.4 | Route the abstain-on-ambiguity branches (`ingestion.py:290-291`, `:402-404`) into the existing `KGAgent.verify_entity_existence` adjudicator, per 2401.03426 and 2501.13956 | M | Ambiguous pairs are decided rather than duplicated; duplicate-node count falls; LLM calls fire only on abstention | `pytest tests/test_entity_resolution.py -v && python -m scripts.graph_health --brain-id <id>` |
| 2.5 | Constrain relationship dedup search to the subject-object pair and stop treating direction-reversed edges as duplicates, per 2501.13956 §2.2.2 | S | A duplicate for a given pair is found regardless of how many unrelated similar edges exist; `A→B` and `B→A` remain distinct | `pytest tests/test_relationship_dedup.py -v` |
| 2.6 | Choose thresholds by sweep rather than assertion, following 2510.14271: blocking strategy, embedding, similarity metric, merge rule. Collect the twelve constants inventoried in technique C into one configuration surface first | M | Every identity/dedup threshold is swept with a recorded precision/recall curve and read from one place; the `TODO` at `ArchitectAgentCreateRelationshipTool.py:512` is resolved | `python -m benchmarks.er_sweep --out runs/er-sweep.json` |
| 2.8 | Collapse the two entity-matching algorithms into one: either route `KGAgentAddTripletsTool`'s Levenshtein+Jaccard path (`names.py:47-66`) through the vector resolver, or state why triplet-tool matching differs | S | One matching function governs entity identity; `most_similar_name_with_labels_or_none` is either the single path or removed | `rg -n "most_similar_name_with_labels_or_none" src/` |
| 2.9 | Make `list_reduction.py`'s missing-embeddings path fail loudly instead of substituting the 8-dimensional character-sum hash (`:57-65`) | S | Absent an embeddings client, reduction raises rather than returning near-random similarity | `pytest tests/test_list_reduction.py -k fallback -v` |
| 2.7 | Reconcile the three predicate policies and two event-naming conventions in `src/constants/prompts/` into one, per answers to open questions 1 and 2 | M | One vocabulary statement across Scout, Architect, and Janitor prompts; the triangle in `README.md` matches the prompts | `pytest tests/test_prompt_consistency.py -v` |

**Checkpoint 2.** Multi-hop is the top-priority pain point, so this is the decisive gate. Requires: measurable drop in type and predicate cardinality, zero type-named nodes, reduced duplicate-node count, and **no regression in 0.1 completeness** — canonicalization must not be silently discarding facts. Hand off to `02-retrieval-multihop.md` for the read-side multi-hop measurement.

### Phase 3 — Grounding and the Janitor veto

Depends on Checkpoint 2, because a veto over a fragmented graph would reject correct facts for the wrong reason.

| # | Task | Size | Acceptance criteria | Verification |
| --- | --- | --- | --- | --- |
| 3.1 | Require every extracted relationship to carry the source span it came from | M | Every new edge has a non-empty span; 0.1 hallucination rate becomes directly computable rather than estimated | `pytest tests/test_source_spans.py -v` |
| 3.2 | Add a string-based global aligner scoring each span against the stored `TextChunk`, per 2510.00276 | M | Fabricated spans score below unaligned threshold; genuine paraphrases are not flagged; false-positive rate on a labelled sample under 5% | `python -m benchmarks.grounding --units 50 --out runs/grounding.json` |
| 3.3 | Give the Janitor a delete verdict and honour it: move the queue-for-persistence call in `ArchitectAgentCreateRelationshipTool.py` to **after** the `wrong_relationships` check, and fix the same inversion at `architect_agent.py:1254` | M | A relationship the Janitor rejects never appears in the graph; the corrected version does; both-versions-written no longer occurs | `pytest tests/test_janitor_veto.py -v` |
| 3.4 | Report non-convergence: when the Janitor loop exhausts `max_janitor_iterations` with edges still wrong, mark the ingestion degraded and name the edges | S | Task status distinguishes clean completion from budget exhaustion | `pytest tests/test_janitor_convergence.py -v` |
| 3.5 | Optional, per 2510.00276: replace the LLM span scorer with a small fine-tuned encoder on 1–2 hours of annotations | S | Matches or beats the LLM scorer at lower latency; skip if 3.2 already meets the write-latency budget | `python -m benchmarks.grounding --scorer encoder --out runs/grounding-enc.json` |

**Checkpoint 3.** Hallucination rate from 0.1 must fall substantially — SafePassage reports up to 85% reduction as the achievable ceiling. Completeness must not regress: a veto that improves precision by discarding facts has failed.

### Phase 4 — Bi-temporal writes

Depends on Checkpoint 1 (1.1 and 1.2 specifically). Can proceed in parallel with Phases 2–3. Coordinate the read side with `05-temporal-truth.md`.

| # | Task | Size | Acceptance criteria | Verification |
| --- | --- | --- | --- | --- |
| 4.1 | Separate transaction time from valid time on edges: `created_at`/`expired_at` and `valid_at`/`invalid_at`, per 2501.13956 §2.2.3 and 2606.09900 | M | Every new edge carries all four where determinable; observation time is never conflated with fact time | `pytest tests/test_bitemporal.py -v` |
| 4.2 | Extract fact-time explicitly against a reference timestamp, in ISO-8601, only when the text establishes or changes the relationship — the Graphiti temporal-extraction contract | M | "started my job two weeks ago" yields an absolute `valid_at` distinct from `created_at`; dates unrelated to the relationship are not attached | `pytest tests/test_temporal_extraction.py -v` |
| 4.3 | Replace similarity-based supersession with a deterministic `(subject, predicate, object)` rule over the canonical predicate registry, per 2606.26511 | M | Stale-fact-error rate from 0.3 approaches zero; no similarity threshold on the supersession path | `python -m benchmarks.stale_facts --brain-id <id>` |
| 4.4 | Set the superseded edge's `invalid_at` to the superseding edge's `valid_at` and record a supersession chain, so history is queryable rather than merely hidden | S | An "as-of" query returns the value valid at that time; the full chain is traversable | `pytest tests/test_supersession_chain.py -v` |
| 4.5 | Decide whether the 9-line `temporal_agent.py` becomes the home for 4.2 or is deleted, per open question 12 | S | No stub files referenced nowhere remain under `src/core/agents/` | `rg -n "temporal_agent\|validator_agent" src/` |

**Checkpoint 4.** Stale-fact-error rate near zero, and Phase 0 completeness and hallucination metrics unmoved. This closes the maintainer's confirmed #2 pain point on the write side.

### Phase 5 — Chunk-boundary integrity

Depends on Checkpoint 3, since 0.1's completeness metric is the only way to detect whether decontextualization is adding facts or inventing them.

| # | Task | Size | Acceptance criteria | Verification |
| --- | --- | --- | --- | --- |
| 5.1 | Decontextualize each unit before extraction so it stands alone, per 2510.22590 and 2410.16597 | M | Pronouns and shortened references are resolved before Scout sees them; 0.1 completeness rises on multi-chunk inputs; stability score improves | `python -m benchmarks.extraction_quality --units 50 --multichunk --out runs/eq-decon.json` |
| 5.2 | Supply prior-unit context to Scout (Graphiti uses `n = 4`) and merge rather than discard richer duplicates at `scout_agent.py:372-379` | S | An entity first named in chunk 1 and referenced in chunk 2 lands on one node with merged description | `pytest tests/test_scout_chunking.py -v` |
| 5.3 | Resolve the Scout-chunks-versus-Architect-whole-text asymmetry (`auto_kg.py:126`) so both agents see the same view, with a bounded Architect prompt | M | A 100k-character input completes without prompt overflow; extraction is stable across runs | `python -m benchmarks.extraction_quality --units 5 --long-input --out runs/eq-long.json` |
| 5.4 | Give file ingestion overlap or shared context across page boundaries so a fact spanning pages 3–4 is extracted once | M | A deliberately page-split fact is captured; no duplicate hub for it | `pytest tests/test_ingest_file_boundaries.py -v` |

**Checkpoint 5.** Completeness up on multi-chunk and multi-page inputs, hallucination rate flat, stability improved toward ATOM's reported ~33%.

### Phase 6 — Make consolidation and mode-splitting safe

Lowest priority, but both are latent hazards.

| # | Task | Size | Acceptance criteria | Verification |
| --- | --- | --- | --- | --- |
| 6.1 | Replace free-form LLM graph mutations in `graph_consolidation.py:84-88` with parameterized operations from a fixed set, per 2501.13956 §2.2.1, and remove `KGAgentRemoveNodeTool` / `KGAgentRemoveRelationshipTool` from the consolidator's tool list (`kg_agent.py:527`, `:533`) in favour of invalidation | M | Every consolidation action is one of a known, audited set; nothing on the consolidation path issues a physical delete; each application is logged and reversible; failures surface rather than print | `pytest tests/test_consolidation_ops.py -v` |
| 6.1a | Interim mitigation, do this in Phase 1 if 6.1 will not land soon: log every consolidation mutation with its originating task string before applying it, so the deletions described in gap 12 are at least attributable | S | Every applied consolidation mutation is recoverable from logs, including deletes | `pytest tests/test_consolidation_audit.py -v` |
| 6.2 | Record on every node and edge which pipeline mode wrote it, pending open question 9 | S | A mixed-mode brain can be partitioned by writer; `graph_health` reports the split | `python -m scripts.graph_health --check pipeline-mode --brain-id <id>` |
| 6.3 | Resolve the `lightweight` structural divergence, pending open question 8: either make coarse mode emit event hubs, or document it as a separate incompatible product | M | Either both modes produce traversable event hubs, or the incompatibility is explicit in an ADR and enforced per brain | `pytest tests/test_pipeline_modes.py -v` |
| 6.4 | Fill ADRs 001 and 002, and decide the fate of the empty `AGENTS.md` | S | Neither ADR contains `--` placeholders | `rg -n '^\-\-$' docs/decisions/` |

**Checkpoint 6.** Full Phase 0 suite plus the retrieval-side metrics from `02-retrieval-multihop.md`, as input to the cross-workstream roadmap.

---

## Risks

**Canonicalization over-merges and destroys multi-hop precision.** Collapsing `PERSON`/`PEOPLE` is right; collapsing two genuinely distinct predicates is not, and it silently fuses unrelated subgraphs. *Detect:* 0.2's top-degree node list should not grow a new super-node after Phase 2; 0.1's completeness must hold; require human review of every merge cluster above a size threshold before it is applied.

**The Janitor veto suppresses correct facts.** Once the Janitor can delete, its false-positive rate becomes a data-loss rate, and there is no record of what was rejected. *Detect:* completeness at Checkpoint 3 must not regress; log every veto with its reason and span score, and sample them manually before enabling enforcement. Note that unaudited LLM-driven deletion is not a risk this plan introduces — per gap 12 it exists today on the consolidation path, and task 6.1a is the cheap way to make it visible in the meantime.

**Grounding verification rejects valid inference.** Not every true fact is a verbatim span — "John moved to New York" may be legitimately inferred across two sentences. A strict aligner flags it. *Detect:* 3.2's acceptance criterion is explicitly a false-positive rate on a labelled sample, not just a hallucination reduction. SafePassage reports "minimal risk of flagging non-hallucinations," which is a claim to verify locally, not assume.

**Fixing supersession un-hides facts that were wrong to begin with.** Task 1.2 will resurrect edges currently marked `deprecated`. Some were deprecated by the blind rule, but some may be genuinely superseded. *Detect:* count and sample the resurrected set before and after; do not backfill blindly, and treat the existing `deprecated` flag as untrustworthy rather than as ground truth.

**Date-format migration corrupts existing values.** Moving from `%d/%m/%Y` to ISO-8601 means reinterpreting stored strings, and `01/02/2026` is ambiguous without knowing which writer produced it. *Detect:* 0.4 before and after; migrate to a new field rather than rewriting in place, and keep the original string.

**Decontextualization introduces hallucinations.** Rewriting a chunk to be self-contained means an LLM inventing the missing context — exactly the failure mode being fixed elsewhere. *Detect:* 5.1's acceptance pairs completeness gain with a flat hallucination rate; if hallucination rises, the rewriting is fabricating antecedents.

**Write latency and cost exceed what the product can bear.** Phases 2–5 add LLM calls to the write path. Open question 15 exists because I could not determine the budget. *Detect:* record tokens and wall-clock per ingested unit at every checkpoint; ATOM (2510.22590) and Engram (2606.09900) both report large *reductions*, so a large increase is a signal that the design has drifted from the papers.

**Phase 0 metrics are themselves wrong.** Every gate here depends on the harness. A completeness metric that misses a class of fact will approve a change that discards it. *Detect:* validate 0.1 against a small hand-annotated set before trusting it as a gate, and follow 2402.10744's finding that human-annotated reference relations are themselves frequently incomplete.

**Concurrency assumptions break under the new resolution path.** Today, deterministic `stable_node_id` makes concurrent page ingestion safe for identical surface forms. Adding LLM adjudication (2.4) introduces a non-deterministic resolution decision into a path where multiple workers race on the same entity. *Detect:* ingest a document whose pages repeatedly mention the same entity under varying surface forms, concurrently, and assert a single resulting node.

**These changes are invisible to the current benchmark.** LOCOMO's temporal and multi-hop categories will move, but nothing in `benchmarks/` currently measures write-side quality, so a regression in extraction can be masked by a retrieval gain. *Detect:* report Phase 0 metrics alongside every benchmark run, and treat `04-evaluation-and-applications.md` failure-attribution work as a hard dependency for interpreting any end-to-end number.
