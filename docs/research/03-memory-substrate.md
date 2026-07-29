# 03 — Memory Substrate: data model, indexing, and knowledge lifecycle

Scope per `docs/research/00-scope-and-constraints.md`: the physical model underneath the graph, the vector stores, the text/observation stores, and the lifecycle machinery that is supposed to keep them true and bounded over time. Extraction prompt quality (`01`), query-time ranking (`02`), the benchmark harness (`04`), and temporal *policy* (`05`) are owned elsewhere; this document owns whether the storage layer can physically **express and index** what those workstreams need.

Every implementation claim below carries a `file:line` anchor. Every research claim carries an arXiv ID that was verified against the arXiv metadata API during this review.

---

## What this workstream does

### 1. Backend selection is a runtime switch with three independent axes

`src/config.py:550-552` reads three env vars:

```550:552:src/config.py
        self.vector_db = os.getenv("VECTOR_DB", "milvus")
        self.data_db = os.getenv("DATA_DB", "mongo")
        self.graph_db = os.getenv("GRAPH_DB", "neo4j")
```

`src/core/instances.py:108-153` binds adapters from those values: graph → Neo4j (`instances.py:109-112`) or a NetworkX-over-Postgres client (`instances.py:113-116`); data → Mongo (`instances.py:118-122`) or Postgres (`instances.py:123-127`); vectors → Milvus (`instances.py:144-148`) or pgvector (`instances.py:149-153`).

Two facts follow immediately:

- **The code defaults and the shipped defaults disagree.** `.env.example:144-146` sets `GRAPH_DB="networkx"`, `DATA_DB="postgresql"`, `VECTOR_DB="postgresql"`. So the configuration a new user actually runs exercises an entirely different set of storage code paths from the in-code defaults. Anything measured on one says little about the other.
- **Qdrant is dead code.** `src/lib/qdrant/` contains only `__init__.py`, and no branch in `instances.py` references it. `src/lib/embeddings/local.py` is likewise unwired. Both should be deleted or they will keep implying capabilities that do not exist.

There is also no `else` on those branches. Setting `GRAPH_DB=foo` yields a `GraphAdapter` with `self.graph is None` and the first traversal fails with an `AttributeError`; setting `VECTOR_DB=foo` leaves `vector_store_adapter` undefined and the failure is an import-time `NameError`. Neither is a config error message.

There is also no schema migration framework anywhere — no Alembic, no `migrations/` directory. All DDL is lazy `CREATE TABLE IF NOT EXISTS` executed on first use (`src/lib/postgresql/data.py:148-172`). Any Phase 2 schema change has to bring its own migration mechanism with it.

### 2. The graph model: labels, relationship types, and properties

Nodes are written by `Neo4jClient.add_nodes` (`src/lib/neo4j/client.py:204-306`). Labels come straight from the extractor's `type` string (`src/workers/tasks/ingestion.py:883`, `labels=[node_data.type]`), cleaned by `_clean_labels`. The identity key is `uuid`:

```284:288:src/lib/neo4j/client.py
            cypher_query = f"""
    MERGE (n:{labels_expression} {{uuid: {self._format_value(node.uuid)}}})
    SET {properties_set}
    RETURN n
            """
```

The uuid is content-derived, not random — `stable_node_id` (`src/core/saving/identity.py:13-25`) hashes `(name.lower(), type.lower())`, plus a normalized `happened_at` when the type is `EVENT`. That is the deduplication key: same name + same type = same node. Relationship identity is `stable_relationship_id` over `(tail_uuid, PREDICATE, tip_uuid, flow_key)` (`identity.py:28-42`).

Node properties actually written (`client.py:243-250`): `description`, `happened_at`, `last_updated`, `metadata`, `observations`, `polarity`, plus `name`, `uuid`, and whatever free-form `properties` dict the extractor produced. `description` and the list-valued `source_chunk_ids`/`aliases` get merge semantics rather than overwrite (`client.py:262-277`), which is a genuinely good append-only touch.

Relationships are written by `add_relationship` (`client.py:308-366`), which MERGEs on `uuid` and sets `description`, `v_id`, `flow_key`, and any of `properties`/`happened_at`/`last_updated`/`amount` found on the subject, predicate, **or** object (`client.py:330-350` — note the loop copies attributes from all three objects onto the edge, so a subject's `happened_at` can end up as the edge's `happened_at`).

The event-hub triangle from the README is not encoded in the schema. It is an emergent convention: the extractor is prompted to emit generic relation names (`ArchitectAgentCreateRelationshipTool.py:102` — *"'TARGETED' = correct"*), and the two legs of one hub are tied together by a shared `flow_key` (`identity.py:45-61`, consumed by `get_nexts_by_flow_key` at `client.py:1752-1810` and the two-hop query at `client.py:1670-1672`, which joins on `r2['flow_key'] = r['flow_key']`). There is no `:EventHub` label, no constraint that a hub has exactly one `MADE` in-edge, and nothing prevents the LLM from emitting a relation type that breaks the convention.

Provenance is real but thin. `stamp_provenance` (`identity.py:85-109`) writes `source_chunk_ids` (a merged list) and `source_timestamp` onto both entity and relationship property bags. There is **no confidence, no extractor/model version, and no agent attribution**.

### 3. Tenancy: database-per-brain, on all three backends

This is the strongest part of the substrate. Neo4j uses `database_=brain_id` on every call plus `ensure_database` (`client.py:57-77`), Milvus uses `db_name=brain_id` (`src/lib/milvus/client.py:141-155`), and Postgres provisions `brain_<sanitized>` databases via `brain_db_name` (`src/lib/postgresql/_naming.py:30-49`), with a deterministic SHA-256 suffix when the name would exceed the 63-byte identifier limit. Isolation is physical, not a `WHERE brain_id = ...` predicate, so there is no class of query that can leak across tenants by forgetting a filter.

The cost is operational: every brain is a database. Neo4j Community Edition supports exactly one user database — `ensure_database` swallows `"not supported in community edition"` (`client.py:71`) and silently continues, meaning **on Community Edition every brain shares the `neo4j` database with no `brain_id` predicate anywhere**. That is a real multi-tenant leak on the most likely self-hosted configuration. Milvus similarly caps databases (64 by default), and Postgres pays a connection pool per brain (`_provisioning.get_brain_pool`).

The deeper cost is that isolation now depends entirely on one argument being threaded correctly through every call, with no second line of defence — and it is not. `GraphAdapter.add_nodes` and `execute_operation` both default `brain_id="default"` (`src/adapters/graph.py:145`, `graph.py:122`), and two call sites rely on that default by accident:

```67:67:src/core/agents/tools/kg_agent/KGAgentAddNodesTool.py
        self.kg.add_nodes(nodes, self.identification_params, self.metadata)
```

The signature is `add_nodes(nodes, brain_id, identification_params, metadata)`, so `identification_params` is being passed as the database name. And `src/core/agents/kg_agent.py:121` calls `self.kg.execute_operation(operation)` with no `brain_id` at all, so arbitrary agent-authored Cypher executes against a shared `default` database. Database-per-tenant is the right design, but a silent `"default"` fallback converts every threading mistake into a cross-tenant write instead of an error. See "Guarantees and where they break" §5.

### 4. Vector stores

**Collections.** Five logical stores with independently configured dimensions: `nodes`, `triplets`, `observations`, `data`, `relationships` (`src/constants/embeddings.py:21-27`, fed from `EMBEDDING_*_DIMENSION` env vars at `config.py:321-345`). `.env.example:43-47` sets all five to 3072 (`text-embedding-3-large`).

Four of the five are actually written. **Nothing anywhere calls `add_vectors(..., "observations", ...)`** — the store is declared and provisioned but stays empty, so the agent-written notes in `data_observations` are reachable only by `ILIKE` substring match and by `resource_id` join. Observations are the one part of the substrate designed to hold synthesized insight, and they are the one part with no semantic index.

**Milvus** (`src/lib/milvus/client.py:158-193`): quick-setup `create_collection(store, dimension=..., vector_field_name="embeddings")`, then `AUTOINDEX` with `COSINE`. No partition key, no scalar index on any metadata field, and `search_vectors` (`client.py:218-255`) exposes **no filter parameter at all** — filtered ANN is not possible through this interface.

**pgvector** (`src/lib/postgresql/vectors.py:124-160`): a table per store with `id BIGINT PRIMARY KEY, uuid TEXT, embeddings vector(d), metadata JSONB`, a btree index on `uuid`, and an HNSW index — conditionally:

```60:67:src/lib/postgresql/vectors.py
def _vector_index_ddl(table: str, dimension: int) -> str:
    dim = int(dimension)
    if dim > 2000:
        return ""
    return f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_embeddings
                ON {table} USING hnsw (embeddings vector_cosine_ops);
            """
```

pgvector's HNSW caps at 2000 dimensions, so at the shipped 3072 this returns the empty string and **no vector index is created**. Every `search_vectors` (`vectors.py:205-232`) then runs `ORDER BY embeddings <=> %s LIMIT k` as a full sequential scan. There is no log line, no warning, no metric. See "Guarantees and where they break" §2.

**Embeddings.** `EmbeddingsAdapter` (`src/adapters/embeddings.py:74-140`) retries five times with exponential backoff and then falls back to `ReturnEmptyVectorStrategy` (default, `embeddings.py:77`), which returns `Vector(id=uuid4(), embeddings=[], metadata={})` (`embeddings.py:56-63`). Nothing calls `set_failure_strategy`, so `RaiseEmbeddingFailureStrategy` (`embeddings.py:66-71`) is unreachable in production. The vector id is a fresh `uuid4()` (`embeddings.py:123`), unrelated to the graph node uuid; the graph→vector link is the integer `v_id` stored in node properties (`src/core/saving/ingestion_manager.py:86-89`) and the vector→graph link is `metadata["uuid"]` (`ingestion_manager.py:77-81`). **No embedding model name or version is stored anywhere on the vector.**

### 5. Relational and text stores

`src/lib/postgresql/data.py:55-101` defines four per-brain tables: `data_text_chunks`, `data_observations`, `data_structured_data`, and `data_kg_changes`. Indexes exist on `inserted_at`, `resource_id`, a GIN on `metadata->'labels'`, a GIN on `types`, and `timestamp`. There is **no tsvector/GIN full-text index**; text search is `ILIKE '%...%'` (`data.py:104-106`). `data_kg_changes` is a real append-only change log and is the only durable audit trail in the system.

Redis is always wired (`instances.py:105-106`) and keys are brain-prefixed (`src/lib/redis/client.py:39-43`). Task status is deliberately TTL'd to seven days, but the brain-registry and PAT entries (`src/services/api/middlewares/brains.py:132-136`, `middlewares/auth.py:77-79`) are written with **no TTL and no invalidation on write** — nothing calls `cache_adapter.delete` when a brain or its PAT changes. A revoked token stays valid in cache indefinitely.

### 6. Knowledge lifecycle: what exists, and what is an empty file

Live:

- **Graph consolidation** (`src/core/layers/graph_consolidation/graph_consolidation.py:31-94`): batches of 20 relationships handed to the Janitor agent, which emits normalization/dedup tasks executed by the KG agent. Failures per task are caught and `continue`d (`graph_consolidation.py:89-94`).
- **Supersession** (`src/workers/tasks/ingestion.py:465-505`): marks older same-type edges from the same subject with `invalid_at` and `deprecated=True`.
- **Deprecation via agent tool** (`client.py:1147-1150`, `SET r['deprecated'] = true`).
- **Observations**: an `ObservationsAgent` singleton (`src/services/observations/main.py:11-15`) writing into `data_observations`.

Not implemented — these are files containing only a docstring header:

| File | State |
| --- | --- |
| `src/core/agents/temporal_agent.py` | 9 lines, docstring only. No temporal agent exists. |
| `src/core/backups/backup_creator.py` | `create_backup` is `pass` with a TODO comment list (`backup_creator.py:24-37`); its `-> Backup` annotation references an undefined name, so importing the module raises `NameError`. |
| `src/core/backups/scheduler.py` | 10 lines, docstring only. |
| `src/services/triggers/neighbourhood.py` | 9 lines, docstring only. |
| `src/services/triggers/synthetic_kg.py` | 9 lines, docstring only. |

There is no community detection, no hierarchical summarization, no forgetting or decay, no compaction, and no backup. The append-only graph currently has **no counterweight of any kind**.

---

## Guarantees and where they break

What this layer is trying to guarantee, in my words: *a fact written once is retrievable forever, attributable to its source, distinguishable from the facts that superseded it, isolated from other tenants, and findable in bounded time as the graph grows.* Ranked by damage to answer accuracy and multi-hop recall:

### 1. Supersession invalidates the wrong edges, and in an event-centric graph it is actively destructive — **bug, highest impact**

`_invalidate_superseded_relationships` (`ingestion.py:465-505`) is the entire mechanism by which BrainAPI knows a fact stopped being true. It works like this: fetch the subject's neighbours, and for every edge whose relation name matches the new edge's name but whose other endpoint differs, stamp `invalid_at` and `deprecated=True`.

```481:503:src/workers/tasks/ingestion.py
    for predicate, neighbor in pairs:
        if not predicate or not neighbor:
            continue
        if (predicate.name or "").strip().upper() != (relationship.name or "").strip().upper():
            continue
        if neighbor.uuid == relationship.tip.uuid:
            continue
        ...
            graph_adapter.update_properties(
                predicate.uuid,
                "relationship",
                brain_id=brain_id,
                new_properties={
                    "invalid_at": valid_at
                    or datetime.datetime.utcnow().strftime("%d/%m/%Y"),
                    "deprecated": True,
                },
            )
```

The assumption is functional: *"for a given subject and relation, only one object can be current."* That holds for `(Emily)-[:LIVES_IN]->(city)`. It is exactly wrong for the event-hub model, where relation names are deliberately generic — `MADE`, `TARGETED`, `OCCURRED_WITHIN`. Emily has one `MADE` edge per event she participates in. When she does a second thing, `relationship.name == "MADE"` matches her first `MADE` edge, the neighbour differs (it is a different hub), and **the first event's `MADE` edge is marked deprecated**. Ingest N events by the same actor and N−1 of them lose their actor attribution.

Three amplifiers:

- `get_neighbors` is undirected (`client.py:697`, `MATCH (n)-[r]-(c)`), so incoming edges of the same name are caught too.
- `valid_at` is a `"%d/%m/%Y"` string (`ingestion.py:500`) — no time zone, no ordering, no range query, no Neo4j temporal type.
- There is no `valid_from` on the *new* edge, only `invalid_at` on the old one, so you cannot reconstruct an interval and cannot answer "what was true on date X."

This is the single biggest data-model limitation and it is the direct mechanical cause of the maintainer's stated symptom #2 in `00-scope-and-constraints.md` ("superseded facts come back as current truth") — with the twist that the failure runs in both directions: valid facts get invalidated, and readers mostly ignore the flag anyway (see §3).

### 2. At the shipped configuration there is no vector index — **gap, second highest impact**

`_vector_index_ddl` returns `""` for any dimension above 2000 (`vectors.py:60-63`) and `.env.example:43-47` ships 3072. Every one of the five stores is therefore an unindexed sequential scan under the default install, and the failure is silent. `search_similar_by_ids` (`vectors.py:265-336`) runs one such scan **per source vector in a Python loop** (`vectors.py:306-317`), so a dedup pass over a batch of 20 relationships is 20 full table scans.

The Milvus path does build an index (`milvus/client.py:175-186`) but `AUTOINDEX` with no `m`/`ef_construct`/`ef` tuning and no quantization choice means recall/latency is whatever the server defaults to, unmeasured. Neither backend can filter — `search_vectors` takes no predicate on either path — so any "search only current facts" or "search only nodes of type PERSON" has to be post-filtered after top-k, which silently truncates recall exactly when the filter is selective.

### 3. Readers cannot tell current truth from history

`deprecated`/`invalid_at` is consulted in exactly one place in the entire codebase:

```420:426:src/services/api/controllers/retrieve.py
def _is_currently_valid(predicate: Predicate) -> bool:
    props = getattr(predicate, "properties", None) or {}
    if props.get("invalid_at"):
        return False
    if getattr(predicate, "deprecated", False):
        return False
    return True
```

`get_neighbors`, `list_triples`, `get_event_centric_neighbors`, `get_nexts_by_flow_key`, the hops endpoints, and every MCP tool return deprecated edges undifferentiated. A validity flag that only one of a dozen read paths honours is not a validity model.

### 4. The dual write is vector-first, graph-second, with no compensation

`process_node_vectors` (`ingestion_manager.py:57-94`) commits the vector, then `graph_adapter.add_nodes` (`ingestion.py:911`) commits the graph. If the graph write throws, the vector is already durable and orphaned. There is no outbox, no saga, no reconciliation job, and — because `backup_creator.py` is a stub — no snapshot to diff against.

The reverse leak is worse because it is silent. When an embedding future fails or times out, the loop `continue`s without adding the node to `graph_nodes` (`ingestion.py:900-909`), and then `add_relationship` is called anyway at `ingestion.py:913` referencing that node's uuid. The Cypher is `MATCH (a) WHERE a['uuid'] = ...` (`client.py:354`); no match means zero rows, means the `MERGE` never executes, means **the relationship is dropped with no exception and no error record**. And when the embedding itself fails, `ReturnEmptyVectorStrategy` yields `embeddings=[]`, `process_node_vectors` takes the `else` branch (`ingestion_manager.py:92-93`), prints, and returns the uuid as if nothing happened — so the graph node exists with no vector, invisible to every similarity query, forever, with no repair path.

### 5. Tenant isolation has a silent `"default"` fallback, and two call sites hit it — **bug**

Database-per-brain has no second layer: no query anywhere carries a `brain_id` predicate, so if the wrong database name is threaded in, the write lands in the wrong tenant with no error. `GraphAdapter` defaults that argument to the string `"default"` (`src/adapters/graph.py:122`, `graph.py:145`), which turns a missing argument into a shared database rather than an exception. Two call sites depend on it:

- `src/core/agents/kg_agent.py:121` — `self.kg.execute_operation(operation)`. Agent-authored Cypher, unscoped, against `default`.
- `src/core/agents/tools/kg_agent/KGAgentAddNodesTool.py:67` — `self.kg.add_nodes(nodes, self.identification_params, self.metadata)` against the signature `add_nodes(nodes, brain_id, identification_params, metadata)`. The identification dict is being passed as the database name, so this write is simply broken.

Combined with the Community Edition case in "What this workstream does" §3, isolation is currently a convention enforced by argument order.

### 6. Milvus primary keys are not stable across processes — **bug**

```205:205:src/lib/milvus/client.py
                "id": hash(vector.id) % (2**63),
```

Python salts string hashing per process by default; `PYTHONHASHSEED` is not set anywhere in this repo. The same logical vector therefore gets a different Milvus primary key in each Celery worker and after every restart, so `get_by_ids` and `remove_vectors` miss, and re-inserting a logical vector appends a duplicate row instead of upserting. The deterministic helper is right there in the same file and unused:

```30:32:src/lib/milvus/client.py
def string_to_int64(s: str) -> int:
    sha256 = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(sha256[:8], byteorder="big") % (2**63)
```

The Postgres store uses it correctly (`vectors.py:172`). This is a one-line fix with a backfill caveat.

### 7. No Neo4j index or constraint exists anywhere in the repository

A repo-wide search for `CREATE INDEX` / `CREATE CONSTRAINT` / `db.index.*` returns only the Postgres DDL in `data.py`, `vectors.py`, and `graph_store.py`. Neo4j gets nothing — no uniqueness constraint on `uuid`, no lookup index, no full-text index. Meanwhile the hot queries are label-free scans:

- `client.py:354-355` — `MATCH (a) WHERE a['uuid'] = ...`, twice per relationship write.
- `client.py:530`, `client.py:562`, `client.py:698`, `client.py:1877` — same pattern for reads.
- `client.py:391-395` — `MATCH (n) WHERE toLower(n['name']) CONTAINS toLower(...)`, a full scan with a per-node `toLower` on every text search.

Also note `n['uuid']` (bracket syntax) rather than `n.uuid`: Neo4j's planner will not use a property index for dynamic-key access even if one existed, so adding indexes requires also rewriting the accessors. Every write is currently O(|V|) and every text search is O(|V|) with string work per node.

### 8. Unbounded growth with no compaction and hub-node blowup

Append-only is stated policy (README "Append-only by design") but it is not what the code does — `DELETE n` (`client.py:1890`) and `DELETE r` (`client.py:1929`, `1948`) are both reachable from agent tools, and `add_nodes`/`add_relationship` overwrite properties in place via `SET`. So the graph carries the costs of append-only (unbounded growth) without the guarantee (history is recoverable). The only durable audit trail is `data_kg_changes`.

Meanwhile there is no counterweight to growth: no TTL, no decay, no summarization, no community layer, no archival tier. Two specific pressures:

- **Hub blowup.** A frequently-mentioned entity accumulates one edge per event. `get_neighbors` (`client.py:696-709`) has an optional `limit` but callers such as `_invalidate_superseded_relationships` (`ingestion.py:472-473`) pass none, so every write touching a popular actor materializes that actor's entire neighbourhood. This is quadratic in the number of events per entity. `get_connected_nodes` is worse: it accepts a `limit` parameter (`client.py:846`) that never appears in the Cypher it builds (`client.py:862-871`), so callers believe they are bounded and are not.
- **Vector store growth.** Five stores, one vector per node/relationship/triplet/chunk, no deletion path except the dedup `remove_vectors` at `ingestion.py:849-853`.

There is also no write atomicity to lean on. Every operation is a bare `driver.execute_query`, one auto-commit transaction each; `add_nodes` issues one query per node in a Python loop (`client.py:226-290`). A batch of nodes can therefore be half-written, and `_execute_query_with_retry` is wired into only two methods (`client.py:290`, `1102`) — `add_relationship`, the deletes, and most reads have no retry at all.

### 9. Re-embedding is impossible; changing the model is destructive

Nothing records which model produced a vector, so you cannot identify stale vectors. And if you change `EMBEDDING_NODES_DIMENSION`, `_ensure_store` notices the mismatch and drops the table:

```148:158:src/lib/postgresql/vectors.py
                    existing_dim = _table_vector_dimension(cur, table)
                    if existing_dim is not None and existing_dim != dimension:
                        logger.warning(
                            "Recreating %s for brain %s: vector dimension %s -> %s", ...
                        )
                        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                    cur.execute(ddl)
```

The graph keeps its now-dangling `v_id` properties. There is no re-embed job in `scripts/` or `src/workers/`. Today, a "full re-index" means re-ingesting every source document through the full LLM swarm — the entire extraction cost, again.

### 10. Expressiveness the model does not have

Judged against what `02`/`05` will need:

| Needed | Present? | Evidence |
| --- | --- | --- |
| Validity interval `[valid_from, valid_to)` | No — only `invalid_at` on the superseded edge | `ingestion.py:499` |
| Ingestion time vs. event time (bitemporal) | Partially — `source_timestamp` and `happened_at` exist but are unordered strings on a free-form property bag | `identity.py:104-108`, `client.py:243-250` |
| Provenance to source span | Chunk-level only (`source_chunk_ids`), no offsets | `identity.py:94-102` |
| Confidence / extraction certainty | No | no such field anywhere in `add_nodes`/`add_relationship` |
| Logical negation ("Emily did *not* attend") | No. `polarity` is sentiment good/bad/neutral, not truth value | `src/constants/prompts/scout_agent.py:25-41`, `scout_agent.py:59-61` |
| n-ary facts beyond the 3-slot triangle | Only via the `flow_key` convention; a 4th role has nowhere to go | `identity.py:45-61` |
| Statement-level metadata (who said it, when, how sure) | No reified statement node exists | — |

The `flow_key` convention is a *reification by naming*: the event hub node is the reified statement, and `flow_key` is the correlation id. That is architecturally the right instinct (it is what StarE-style qualifier models formalize) but it is enforced by prompt text, not by schema, and it is invisible to the storage layer.

### Deliberate trade-offs, not bugs

- **Database-per-brain over a `brain_id` column** — costs operational scale, buys leak-proof isolation. Right call.
- **Content-derived uuids** — makes dedup idempotent and re-ingestion safe.
- **Free-form labels and relation types from the LLM** — buys domain generality (the plugin story), costs the ability to index or constrain anything. Reasonable, but it means quality controls have to live in properties, not labels.
- **Merge-on-write for `description`/`aliases`/`source_chunk_ids`** (`client.py:262-277`) — good append-only hygiene, already correct.

---

## Open questions for the maintainer

1. Is the `.env.example` triple (`GRAPH_DB=networkx`, `DATA_DB=postgresql`, `VECTOR_DB=postgresql`) the configuration you consider production, or is the Neo4j/Milvus/Mongo path the one that matters?
2. Is `_invalidate_superseded_relationships` intended to fire on event-hub legs like `MADE`/`TARGETED`, or was it written for functional attribute edges only?
3. Should a superseded fact be excluded from retrieval by default, or returned with a "no longer current" marker so the caller can reason about change?
4. Is Neo4j Community Edition a supported deployment target, given that `CREATE DATABASE` is unavailable there and all brains would then share one database?
5. What is the largest single brain you need to support in the next twelve months, measured in entities and in events per entity?
6. Is `polarity` meant to carry sentiment only, or did you also intend it to express factual negation?
7. Are you willing to add a small fixed vocabulary of reserved relationship types (`MADE`/`TARGETED`/`OCCURRED_WITHIN`) that the extractor must use, so the substrate can index the event-hub shape?
8. Should re-embedding after a model change preserve old vectors under a version tag, or is a destructive rebuild acceptable if the source text is still in `data_text_chunks`?
9. Is `src/lib/qdrant/` intended to become a real backend, or can it be deleted?
10. Do you want confidence to be a stored number per fact, or should uncertainty stay implicit in the Janitor's accept/reject decision?
11. The `observations` vector store is configured but never written to, so agent-written notes are only reachable by substring match. Should observations be embedded and semantically searchable, or are they intentionally a write-only annotation layer?
12. Hard `DELETE` of nodes and relationships is reachable from the agent tools despite the README's "append-only by design". Is deletion a capability you want agents to have, or should it be soft-delete only?

---

## Frontier techniques

### A. Bitemporal edge invalidation with explicit validity intervals

**Mechanism.** Zep/Graphiti stores each edge with four timestamps: when the fact became true and stopped being true in the world (`valid_at`/`invalid_at`), and when the system learned and unlearned it (`created_at`/`expired_at`). New facts are checked against existing edges for contradiction; contradicted edges get `invalid_at` set rather than deleted, so point-in-time queries stay answerable. ATOM sharpens the same idea by splitting documents into minimal self-contained "atomic" facts before building the temporal graph, explicitly separating observation time from validity time.

**arXiv.** Zep: `2501.13956`. ATOM: `2510.22590`. Ontology-driven legal variant with versioned expressions and reified legislative events: `2505.00039`.

**Reported gain.** Zep: 94.8% vs 93.4% on DMR, and up to +18.5 points accuracy on LongMemEval with ~90% latency reduction versus the baseline. ATOM: ~18% higher extraction exhaustivity, ~33% better stability, >90% latency reduction versus baselines for dynamic TKG construction.

**Cost.** Contradiction detection is an LLM call per candidate-conflict at write time. Storage cost is four timestamp properties per edge plus the indexes to range-scan them.

**Fit.** BrainAPI already has the two halves in the wrong shape: `source_timestamp` is an ingestion time and `valid_at` is an event time, but both are unordered strings in a property bag, and the invalidation rule keys on relation-name collision rather than semantic contradiction. Moving to typed temporal properties on the *hub node* rather than on the legs sidesteps the `MADE`-collision bug entirely, because a hub represents one event and is never in competition with another hub.

**Verdict: adopt**, restricted to the physical shape (typed `valid_from`/`valid_to`/`ingested_at`/`retracted_at`, range-indexed). The contradiction *policy* — when to set `valid_to` — belongs to `05-temporal-truth.md`; this workstream's job is to make sure there is somewhere correct to write it.

### B. Continuous-phase temporal rotation instead of hard invalidation

**Mechanism.** RoMem learns a per-relation "volatility" score from the relation's text embedding (fast for `president of`, slow for `born in`) and applies continuous phase rotation in complex vector space, so obsolete facts rotate out of phase and rank below current ones without ever being deleted or flagged.

**arXiv.** `2604.11544`.

**Reported gain.** 72.6 MRR on ICEWS05-15 (state of the art for temporal KG completion); 2–3× MRR and answer accuracy on MultiTQ temporal reasoning; reported to dominate on LoCoMo with zero degradation on static memory (DMR-MSC).

**Cost.** Requires training or importing a Semantic Speed Gate, plus complex-valued embeddings — a second embedding space alongside the existing five stores.

**Fit.** Attractive precisely because it needs *no* LLM call at ingestion, which suits the sub-second context API budget in `00-scope-and-constraints.md`. But it presumes a learned KG embedding space that BrainAPI does not have: today the only vectors are text embeddings of names and descriptions, with no structural training signal.

**Verdict: reject for now, revisit after A.** The volatility idea is separable and cheap, though: a static per-relation-type decay prior, stored as a property, would give most of the ranking benefit for a fraction of the work. Note the paper is very recent (April 2026) with no independent replication.

### C. Hyper-relational / qualifier representation instead of ad-hoc property bags

**Mechanism.** StarE represents a fact as a main triple plus an arbitrary set of `(qualifier_relation, qualifier_entity)` pairs, and message-passes over that structure so qualifiers modulate the main triple's representation without collapsing into it. GRAN models each n-ary fact as a small heterogeneous graph with edge-biased fully-connected attention. VITA extends this to temporal validity as a first-class qualifier type covering all four cases (since / until / period / time-invariant).

**arXiv.** StarE: `2009.10847`. GRAN: `2105.08476`. VITA: `2505.11803`. Inductive extension THOR: `2602.05424`. (HINGE, mentioned in the brief, is a WWW'19 paper — I could not find it on arXiv and am not citing it.)

**Reported gain.** StarE: up to +25 MRR points over triple-only representations on WD50K. VITA: up to 75.3% improvement over the best baselines across link-prediction variants. Ali et al. `2107.04894`: +6 absolute Hits@10 for qualifier-aware inductive link prediction.

**Cost.** These are *embedding* models with training pipelines. Adopting the learning machinery is a large project.

**Fit.** The representational insight transfers without the models. BrainAPI's event hub *is* a reified n-ary fact; what is missing is that the qualifier slots are unnamed. Today a fourth role (instrument, beneficiary, quantity) has to be smuggled into a free-form `properties` dict on one of the legs, where nothing can index or reason over it.

**Verdict: adapt.** Take the data shape — a hub node with typed, named role edges and typed qualifier properties — and skip the embedding models. This is the schema change that makes n-ary facts expressible.

### D. Which reification style to use for statement metadata

**Mechanism.** Standard RDF reification, singleton properties, RDF-star quoted triples, and named graphs are four ways to attach metadata to a statement. Egami et al. compare them head-to-head under a fair evaluation protocol for link prediction.

**arXiv.** `2503.21804`. Foundations of RDF-star: `1406.3399`. Singleton-property reasoning: `1509.04513`. RDF-star → property-graph transformation: `2210.05781`.

**Reported finding.** Reification performs well on simple hyper-relational graphs while singleton property is less effective; on complex hyper-relational graphs the differences between representation models are minimal.

**Cost.** Free — this is a design-choice input, not an implementation.

**Fit.** BrainAPI is a labelled property graph, not RDF, and it *already* reifies (the event hub node). The literature's conclusion — that on complex graphs the choice barely matters, so pick the one your engine indexes best — is a direct argument for keeping the hub-node design and investing in indexing it, rather than migrating to edge properties or a nested representation.

**Verdict: adopt the conclusion, reject a migration.** Keep hub-node reification. Explicitly do not adopt RDF-star.

### E. Community detection and hierarchical summarization as a bounded-growth mechanism

**Mechanism.** GraphRAG runs Leiden community detection over the entity graph and pre-generates a summary per community at each hierarchy level, so global questions are answered from summaries rather than from the raw graph. This is also the only structural answer in the literature to unbounded growth: the raw layer keeps growing, but the queried layer is bounded by the number of communities.

**arXiv.** GraphRAG: `2404.16130`. Core-based deterministic alternative: `2603.05207`. Dual-perception community detection: `2508.19855`. Design-space survey: `2411.05844`. Skeptical counterpoint: `2603.29875`, and the when-does-GraphRAG-help analysis `2506.05690`.

**Reported gain.** GraphRAG: substantial improvements in comprehensiveness and diversity over conventional RAG for global sensemaking on ~1M-token corpora. `2603.05207` proves that on sparse graphs (constant average degree) modularity optimization admits exponentially many near-optimal partitions, making **Leiden communities non-reproducible run to run**, and proposes k-core decomposition for a deterministic, linear-time hierarchy instead.

**Cost.** One LLM summarization call per community per level at index time, and re-clustering on drift. Substantial token cost, but entirely at write time, which is exactly where `00-scope-and-constraints.md` says expensive work belongs.

**Fit.** BrainAPI's graph is sparse and event-centric — precisely the regime where `2603.05207` says Leiden is unstable. A hub node has degree 2–3 by construction.

**Verdict: adapt — use k-core, not Leiden.** Non-reproducible communities are unacceptable for a memory layer that must give the same answer twice. The `flow_key` grouping also gives a free deterministic pre-clustering (one hub = one event), so a hierarchy could be built bottom-up from hubs without any clustering randomness at the leaf level.

### F. Mega-hub mitigation via concept-mediated nodes

**Mechanism.** GAAMA argues that entity-centric graphs over conversational data develop "mega-hub" nodes that dilute relevance propagation, and inserts a concept node layer (episode / fact / reflection / concept, five edge types) so traversal crosses via concepts instead of piling onto hot entities. CatRAG attacks the same failure from the retrieval side, showing that PPR random walks get diverted into high-degree hubs before reaching downstream evidence.

**arXiv.** GAAMA: `2603.27910`. CatRAG: `2602.01965`. HippoRAG (the PPR baseline both build on): `2405.14831`; HippoRAG 2: `2502.14802`.

**Reported gain.** GAAMA: 79.1% mean reward on LoCoMo-10, +4.2 points over a tuned RAG baseline. HippoRAG: up to 20% over prior RAG on multi-hop QA at 10–30× lower cost than iterative retrieval. HippoRAG 2: +7% on associative memory over the state-of-the-art embedding model.

**Cost.** An extra node layer and extra LLM calls to mint concepts at write time.

**Fit.** Direct. BrainAPI *will* develop mega-hubs — the append-only policy plus one edge per event per participant guarantees it, and §8 above shows the write path already scans full neighbourhoods.

**Verdict: adopt the diagnosis; defer the concept layer.** The immediate substrate action is to bound neighbourhood expansion (degree caps, ordered adjacency, indexed `flow_key`) and to measure the degree distribution. Whether the graph needs a concept layer is a question for `02`, and should be answered with a measured degree histogram rather than assumed.

### G. Vector index and quantization choices

**Mechanism.** HNSW gives the best recall/latency balance in memory; DiskANN/Vamana trades to SSD for billion-scale; product and binary quantization compress vectors at a measurable recall cost; Matryoshka Representation Learning trains embeddings so that a truncated prefix is itself a valid embedding, letting you store 512 dimensions for search and 3072 for reranking from one model.

**arXiv.** Matryoshka: `2205.13147`. In-place streaming DiskANN updates: `2502.13826`. Filtered ANN across FAISS/Milvus/pgvector: `2602.11443`. Low-selectivity filtered search: `2607.00768`. Binary-quantization-native topology and its limits: `2605.02171`. Declarative recall targets: `2505.19001`. ColBERT late interaction: `2004.12832`, ColBERTv2: `2112.01488`, PLAID: `2205.09707`.

**Reported gain.** Matryoshka: up to 14× smaller embeddings at equal accuracy and up to 14× real-world retrieval speedup. `2602.11443` finds Milvus achieves superior recall stability through hybrid approximate/exact execution while pgvector's cost-based optimizer frequently picks suboptimal plans, and that IVFFlat beats HNSW at low selectivity. `2605.02171` finds binary-quantization-native topology works well on cosine-native contrastive embeddings (≥88% Recall@10) but collapses (<15%) on Euclidean-native data — a falsifiable applicability boundary. `2502.13826` (IP-DiskANN) keeps recall stable under long insert/delete streams where batch-consolidation approaches degrade.

**Cost.** Matryoshka requires an MRL-trained model; `text-embedding-3-large` supports dimension truncation natively, so this is a config change for the current default provider and a model change for Ollama/local users.

**Fit.** Immediate and cheap. Truncating to ≤2000 dimensions makes the pgvector HNSW index legal *today* and directly fixes §2. `2602.11443` is worth reading before choosing between Milvus and pgvector at scale.

**Verdict: adopt — this is the highest gain-per-hour item in the document.** Reject binary quantization for now: `2605.02171`'s boundary condition is exactly the kind of thing that fails silently on a mixed corpus.

### H. Governed evolving memory: making lifecycle a first-class workload

**Mechanism.** Orogat & Mansour argue that agent memory is a distinct data-management workload whose correctness is a property of the *state trajectory*, not of individual records, and formalize four state-level operators (ingestion, revision, forgetting, retrieval) under six correctness conditions. They name four recurring failure modes: unregulated growth, missing semantic revision, capacity-driven forgetting, and read-only retrieval. Uddin et al. contribute Forgetting-Aware Memory Accuracy (FAMA), a metric that penalizes reliance on obsolete or invalidated memory.

**arXiv.** GEM/MemState: `2605.26252`. Memora/FAMA: `2604.20006`. Storage-time provenance and versioning as a security requirement: `2604.16548`. Placement of the control plane: `2606.15903`.

**Reported finding.** `2604.20006` evaluates four LLMs and six memory agents and finds *frequent reuse of invalid memories and failures to reconcile evolving memories*, with memory agents offering only marginal improvements. `2606.15903` finds production failures are predominantly forgetting failures rather than recall failures, yet existing benchmarks measure only recall.

**Cost.** Conceptual adoption is free. FAMA needs labelled obsolescence in the eval set, which is a `04` dependency.

**Fit.** BrainAPI exhibits three of the four named failure modes verbatim: unregulated growth (§8), missing semantic revision (§1), and read-only retrieval (§3). The one it avoids is capacity-driven forgetting — because it never forgets.

**Verdict: adopt the framing and the metric.** This is the argument for why substrate work ranks above ranking work: a validity flag nothing reads and a growth curve nothing bounds are not ranking problems.

### I. Confidence and provenance modelling

**Mechanism.** ProVe automatically verifies that a KG triple is supported by the text at its documented provenance URL. Adobe's KG-RAG pipeline assigns confidence scores to entity-relation pairs and filters to high-confidence pairs, and links every fact to its source document.

**arXiv.** ProVe: `2210.14846`. Adobe KG-RAG: `2502.15237`. Provenance semirings for query-result probabilities: `2108.07758`. Epistemic stance over attributed claims: `2606.15246`. LLM confidence miscalibration and grouping loss: `2402.04957`.

**Reported gain.** ProVe: 87.5% accuracy / 82.9% macro-F1 detecting provenance support on text-rich sources. Adobe: >50% reduction in irrelevant answers and +88% fully-relevant answers versus their production system, from a combination of incremental entity resolution, similarity dedup, confidence filtering, and source linking.

**Cost.** A confidence float per node and edge is nearly free to store. Populating it well is not — and `2402.04957` shows LLM self-reported confidence is systematically overconfident and unevenly so across subpopulations, so a raw LLM score would be actively misleading.

**Fit.** BrainAPI already has the provenance half (`source_chunk_ids`, `source_timestamp`) and can therefore afford the cheap, well-calibrated version: count distinct supporting chunks. A fact seen in five independent chunks is more reliable than one seen once, and that number is already computable from `source_chunk_ids`.

**Verdict: adapt.** Store `support_count` (derived, calibrated by construction) rather than an LLM-reported confidence. Reject asking the LLM for a probability.

### Explicitly rejected

| Technique | Why not here |
| --- | --- |
| **RDF-star / named-graph migration** (`1406.3399`, `2211.16195`) | BrainAPI is an LPG. `2503.21804` finds representation choice barely matters on complex hyper-relational graphs. All cost, no measured gain. |
| **Learned hyper-relational KG embeddings** (`2602.05424`, `2605.24064`, `2306.02199`) | Transductive or requiring a training pipeline over a vocabulary that changes on every ingest. BrainAPI's entity vocabulary is open and per-tenant. |
| **Binary-quantization-native ANN** (`2605.02171`) | The paper's own applicability boundary excludes Euclidean-native and "structureless" distributions, and BrainAPI cannot characterize its embedding distribution today. Revisit after recall is measured. |
| **Parametric / KV-cache knowledge injection** (`2510.17934`) | Requires model-side integration and precludes per-tenant isolation. Incompatible with database-per-brain. |
| **Optical / resolution-based forgetting** (`2605.03804`) | Multimodal, on-device framing. BrainAPI's pressure is graph edges, not image bytes. |
| **Leiden-based communities** (`2404.16130` as-implemented) | `2603.05207` proves non-reproducibility on sparse graphs. Use k-core (technique E). |

---

## Implementation plan

Ordered by dependency and by gain per unit of cost. Phases 1 and 2 are correctness and are worth more than everything after them. Nothing here changes retrieval ranking — that is `02`.

**Baseline needed before Task 1.** Every acceptance criterion below that mentions latency or recall assumes a captured baseline. If `benchmarks/` cannot produce one, Task 0 becomes a blocker.

### Phase 1 — Stop the bleeding

#### Task 1: Make Milvus primary keys deterministic

**Description:** Replace `hash()` with the existing `string_to_int64` in `MilvusClient.add_vectors`, and add a one-off script that rewrites existing rows to their deterministic ids.

**Acceptance criteria:**
- [ ] `add_vectors` uses `string_to_int64(vector.id)`; `hash()` no longer appears in `src/lib/milvus/`.
- [ ] Inserting the same `Vector` twice from two separate Python processes yields one row, not two.
- [ ] A migration script maps existing rows to new ids or documents that a rebuild is required.

**Verification:** `poetry run pytest tests/ -k milvus`; manual — insert a vector, restart the worker, `get_by_ids` with the same logical id returns it.

**Dependencies:** None. **Files:** `src/lib/milvus/client.py`, `scripts/`. **Scope:** S.

#### Task 2: Give the default configuration a working vector index

**Description:** The 3072-dimension default silently disables HNSW on pgvector. Make the store request a dimension the index supports, and make the unindexed case loud.

**Acceptance criteria:**
- [ ] `_vector_index_ddl` logs an explicit error (not silence) when it declines to build an index.
- [ ] `.env.example` ships a dimension ≤ 2000 for all five stores, using Matryoshka truncation (`2205.13147`) rather than a different model.
- [ ] `EXPLAIN ANALYZE` on `search_vectors` shows an index scan on a fresh default install.
- [ ] A recall@10 comparison of truncated vs. full-dimension embeddings on the existing benchmark set is recorded in the PR.

**Verification:** `EXPLAIN ANALYZE SELECT ... ORDER BY embeddings <=> ...` shows `Index Scan using idx_vectors_nodes_embeddings`; benchmark latency for `/retrieve/context` recorded before and after.

**Dependencies:** None. **Files:** `src/lib/postgresql/vectors.py`, `.env.example`, `src/lib/embeddings/client*.py`. **Scope:** M.

#### Task 3: Scope supersession so it cannot invalidate event-hub legs

**Description:** `_invalidate_superseded_relationships` treats `MADE` as a functional relation. Restrict it to an explicit allow-list of functional relation types, defaulting to empty, and never let it fire on an edge whose endpoint carries an `EVENT` label or a `flow_key`.

**Acceptance criteria:**
- [ ] Ingesting three events by the same actor leaves all three `MADE` edges with `deprecated` unset.
- [ ] A functional-attribute change (same subject, same relation, new object, relation in the allow-list) still marks the old edge `invalid_at`.
- [ ] `get_neighbors` is called with a `limit`, or the function is rewritten to a targeted single query rather than a full neighbourhood fetch.

**Verification:** New test in `tests/` asserting both branches; `poetry run pytest tests/ -k supersede`.

**Dependencies:** None. **Files:** `src/workers/tasks/ingestion.py`, `tests/`. **Scope:** S.

#### Task 4: Fail loudly when a node is written without its vector

**Description:** Three silent-loss paths converge: `ReturnEmptyVectorStrategy` swallowing embedding failures, `process_node_vectors` returning success with no vector, and `add_relationship` silently no-op'ing when its endpoint MATCH finds nothing.

**Acceptance criteria:**
- [ ] An embedding failure produces an entry in the task's `item_errors`, not just a `print`.
- [ ] `add_relationship` checks the driver result and raises when zero rows were returned.
- [ ] A node written without a vector is recorded in `data_kg_changes` so a repair job can find it.

**Verification:** Inject an embedding-provider failure in a test and assert the ingestion task reports `partial_failed`; `poetry run pytest tests/ -k ingestion`.

**Dependencies:** None. **Files:** `src/adapters/embeddings.py`, `src/core/saving/ingestion_manager.py`, `src/lib/neo4j/client.py`, `src/workers/tasks/ingestion.py`. **Scope:** M.

#### Task 5: Remove the `"default"` brain_id fallback

**Description:** With database-per-tenant and no `brain_id` predicate in any query, a defaulted `brain_id` is a cross-tenant write. Make it a required argument and fix the two call sites that currently rely on the default.

**Acceptance criteria:**
- [ ] `brain_id` has no default value on any `GraphAdapter`, `DataAdapter`, or `VectorStoreAdapter` method; omitting it is a `TypeError` at call time.
- [ ] `KGAgentAddNodesTool` passes `brain_id` in the correct position.
- [ ] `KGAgent.execute_operation` threads the caller's `brain_id`.
- [ ] A grep for `brain_id: str = "default"` in `src/` returns nothing.

**Verification:** `poetry run pytest tests/`; a test asserting that calling `add_nodes` without `brain_id` raises rather than writing.

**Dependencies:** None. **Files:** `src/adapters/graph.py`, `src/adapters/data.py`, `src/adapters/embeddings.py`, `src/core/agents/kg_agent.py`, `src/core/agents/tools/kg_agent/`. **Scope:** S.

### Checkpoint: Phase 1

- [ ] Full test suite passes.
- [ ] A default-config install shows an index scan for vector search.
- [ ] The three-events-one-actor scenario preserves all attribution.
- [ ] No code path can write to a brain database without an explicit `brain_id`.
- [ ] `/retrieve/context` p95 measured and recorded against the pre-Phase-1 baseline.
- [ ] **Review with the maintainer.** Open questions 1, 2, and 4 must be answered before Phase 2, because the target schema depends on which backend is canonical.

### Phase 2 — Target schema

The proposed diff. Additive; nothing is removed, so old data stays readable.

```
EventHub node (currently: an ordinary node whose legs share a flow_key)
+ :EventHub                    label, reserved, set by the writer not the LLM
+ flow_key         string      promoted from edge property to node property, indexed
+ valid_from       datetime    Neo4j temporal type; when the fact became true in the world
+ valid_to         datetime?   null = still current; set on supersession
+ ingested_at      datetime    when BrainAPI learned it
+ retracted_at     datetime?   when BrainAPI stopped believing it (audit, not world-time)
+ support_count    int         count of distinct source_chunk_ids; derived, recomputed on merge
+ negated          boolean     default false; true = "this event did NOT happen"
+ embedding_model  string      model id that produced the linked vectors

Role edges (currently: free-form LLM-chosen names)
+ reserved types   MADE | TARGETED | OCCURRED_WITHIN | QUALIFIED_BY
  LLM-chosen names remain legal but are no longer the mechanism for the triangle
+ role             string      on QUALIFIED_BY: instrument | beneficiary | quantity | ...
                               this is the n-ary slot the model currently lacks (technique C)

Entity node
+ degree_hint      int         maintained edge count, for hub detection (technique F)

Neo4j indexes (currently: none exist anywhere in the repo)
+ CONSTRAINT       uuid uniqueness per label
+ INDEX            :EventHub(flow_key)
+ RANGE INDEX      :EventHub(valid_from), :EventHub(valid_to)
+ FULLTEXT INDEX   name, description   -- replaces the toLower(...) CONTAINS scan

pgvector
+ GIN index        metadata jsonb_path_ops     -- makes vector→graph lookup by node uuid indexable
+ column           embedding_model text        -- makes selective re-embedding possible
```

What this unlocks, concretely: point-in-time queries (`valid_from <= t < valid_to`) become an index range scan instead of impossible; the `MADE` collision bug becomes structurally impossible because validity lives on the hub, and two hubs are never in competition; a fourth participant in an event has a typed home; text search stops being a full scan; and re-embedding becomes a filtered update instead of a table drop.

#### Task 6: Reserved labels and relationship types

**Description:** The writer, not the LLM, stamps `:EventHub` and the three reserved role types when the extractor's output matches the triangle shape. Existing free-form names keep working.

**Acceptance criteria:**
- [ ] Newly ingested hub nodes carry `:EventHub` and a node-level `flow_key`.
- [ ] The two legs of one hub use reserved types; a mismatch is logged, not silently accepted.
- [ ] Existing data without `:EventHub` still returns from all read paths.

**Verification:** `poetry run pytest tests/ -k event_hub`; Cypher spot-check that `MATCH (h:EventHub) RETURN count(h)` is non-zero after an ingest.

**Dependencies:** Task 3, open question 7. **Files:** `src/workers/tasks/ingestion.py`, `src/core/saving/identity.py`, `src/lib/neo4j/client.py`. **Scope:** M.

#### Task 7: Neo4j indexes and constraints, with accessor rewrite

**Description:** Add a startup migration creating the constraints and indexes above. This requires rewriting `n['uuid']` to `n.uuid` in the hot queries, because the planner will not use a property index for dynamic-key access.

**Acceptance criteria:**
- [ ] A startup migration is idempotent and runs per brain database.
- [ ] `PROFILE` on the uuid lookup at `client.py:354` shows `NodeUniqueIndexSeek`, not `AllNodesScan`.
- [ ] `node_text_search` uses the fulltext index instead of `toLower(...) CONTAINS`.
- [ ] Community Edition, where `CREATE DATABASE` is unavailable, is either supported explicitly or refused at startup with a clear message.

**Verification:** `PROFILE MATCH (a) WHERE a.uuid = '...' RETURN a` in the Neo4j browser; ingest-throughput benchmark before and after on a graph of ≥50k nodes.

**Dependencies:** Task 6, open question 4. **Files:** `src/lib/neo4j/client.py`, a new migration module — note that no migration framework exists today, so this task establishes one. **Scope:** M.

#### Task 8: Typed temporal properties on the hub

**Description:** Write `valid_from`/`valid_to`/`ingested_at`/`retracted_at` as Neo4j temporal types on `:EventHub`, replacing the `"%d/%m/%Y"` string. Backfill by parsing existing `happened_at`/`source_timestamp` where parseable.

**Acceptance criteria:**
- [ ] A point-in-time query returns only hubs valid at that instant and uses the range index.
- [ ] Unparseable legacy dates are left null and counted, not guessed.
- [ ] Setting `valid_to` on supersession writes to the hub, not to a leg.

**Verification:** `poetry run pytest tests/ -k temporal`; a fixture asserting that a superseded and a current fact are distinguishable by query alone.

**Dependencies:** Tasks 6, 7. **Files:** `src/lib/neo4j/client.py`, `src/workers/tasks/ingestion.py`, `src/core/saving/identity.py`. **Scope:** M.

#### Task 9: Make validity visible to every reader

**Description:** `_is_currently_valid` exists in one controller. Push the filter into the graph client as an opt-out parameter so `get_neighbors`, `list_triples`, `get_event_centric_neighbors`, and the hops queries all honour it by default.

**Acceptance criteria:**
- [ ] Every read method accepts `include_superseded: bool = False`.
- [ ] With the default, a superseded fact does not appear in any read path.
- [ ] With `include_superseded=True`, the fact returns with its `valid_to` populated so a caller can render history.

**Verification:** `poetry run pytest tests/ -k validity`; a test that ingests a fact, supersedes it, and asserts every read method's default excludes it.

**Dependencies:** Task 8. **Files:** `src/lib/neo4j/client.py`, `src/lib/postgresql/networkx_client.py`, `src/adapters/graph.py`, `src/services/api/controllers/retrieve.py`. **Scope:** M.

### Checkpoint: Phase 2

- [ ] Point-in-time query works and is index-backed.
- [ ] Superseded facts are invisible by default and retrievable on request from every read path.
- [ ] Ingest throughput has not regressed more than 10% (indexes cost write time).
- [ ] `/retrieve/context` still meets its sub-second budget.
- [ ] **Review with the maintainer** before Phase 3.

### Phase 3 — Bounded growth and reversibility

#### Task 10: Record the embedding model with every vector

**Description:** Add `embedding_model` to vector metadata and to the pgvector schema, populated from config at write time.

**Acceptance criteria:**
- [ ] Every newly written vector carries the model id.
- [ ] A query can count vectors per model per store.
- [ ] `_ensure_store` no longer drops a table on dimension change; it refuses and points at the re-embed job.

**Verification:** `SELECT embedding_model, count(*) FROM vectors_nodes GROUP BY 1`; a test that a dimension change raises rather than deletes.

**Dependencies:** Task 2. **Files:** `src/lib/postgresql/vectors.py`, `src/lib/milvus/client.py`, `src/core/saving/ingestion_manager.py`. **Scope:** M.

#### Task 11: Re-embed and reconcile job

**Description:** A Celery task that (a) re-embeds vectors whose `embedding_model` differs from current config, sourcing text from the graph and `data_text_chunks`, and (b) reports graph nodes with no vector and vectors with no graph node.

**Acceptance criteria:**
- [ ] Re-embedding a brain of 10k nodes completes without dropping a table and without a full LLM re-extraction.
- [ ] The reconcile report lists orphan counts in both directions.
- [ ] The job is resumable and idempotent.

**Verification:** Change the model in a test brain, run the job, assert `SELECT count(*) WHERE embedding_model = <old>` is zero and node count is unchanged.

**Dependencies:** Task 10. **Files:** `src/workers/tasks/`, `scripts/`. **Scope:** M.

#### Task 12: Degree instrumentation and hub bounds

**Description:** Before designing a concept layer (technique F), measure. Emit the entity degree distribution and put a hard `limit` on every unbounded `get_neighbors` call site.

**Acceptance criteria:**
- [ ] A script prints p50/p95/p99/max entity degree for a brain.
- [ ] No `get_neighbors` call site in `src/` passes `limit=None`.
- [ ] The degree histogram for the LoCoMo benchmark brain is recorded in this document.

**Verification:** Run the script against the benchmark brain; grep confirms no unbounded call sites.

**Dependencies:** Task 7. **Files:** `scripts/`, `src/workers/tasks/ingestion.py`, `src/core/search/`. **Scope:** S.

#### Task 13: Deterministic k-core hierarchy over event hubs

**Description:** A write-time job building a k-core hierarchy (`2603.05207`) over the entity graph, with one summary per group. Deterministic by construction — do not use Leiden.

**Acceptance criteria:**
- [ ] Running the job twice on the same graph produces byte-identical group membership.
- [ ] Group summaries are stored as nodes with edges to their members.
- [ ] The job is incremental — a new ingest does not require a full rebuild.

**Verification:** Run twice, diff the membership table; assert equality.

**Dependencies:** Tasks 7, 12, and a decision from `02` on whether retrieval will use it. **Scope:** M.

#### Task 14: Backup and restore

**Description:** Implement `create_backup` — currently `pass` — across graph, vectors, text/structured data, and observations, as a resumable Celery task with progress in Redis.

**Acceptance criteria:**
- [ ] A backup of one brain restores into an empty instance with identical node, edge, and vector counts.
- [ ] The module imports without `NameError`.
- [ ] Progress is observable while running.

**Verification:** Backup then restore a benchmark brain; assert counts and spot-check ten facts.

**Dependencies:** Task 10. **Files:** `src/core/backups/`, `src/workers/tasks/`. **Scope:** M.

### Checkpoint: Phase 3

- [ ] A model change is a job, not a data-loss event.
- [ ] Degree distribution is measured and recorded.
- [ ] Backup/restore round-trips.
- [ ] **Review with the maintainer.**

### Parallelization

Tasks 1 through 5 are independent and can run concurrently. Tasks 6→7→8→9 are a strict chain. Tasks 10 and 12 can run alongside the Phase 2 chain. Task 13 must wait on `02`'s decision about whether retrieval consumes a hierarchy.

---

## Risks

| Risk | Impact | How it shows up | Mitigation |
| --- | --- | --- | --- |
| Truncating embeddings to ≤2000 dims loses recall | High | Benchmark recall@k drops after Task 2 | Measure recall before and after in the same PR; Matryoshka models are designed for this (`2205.13147`) but the local/Ollama models in `.env.example` are not MRL-trained and must be checked separately |
| Fixing the Milvus key breaks existing deployments | High | Post-deploy, `get_by_ids` misses every pre-existing vector | Ship the migration script with the fix; if a deployment cannot migrate, document a rebuild as the supported path |
| Neo4j indexes slow ingestion | Medium | Ingest throughput regresses after Task 7 | Benchmark ingest before/after; a uniqueness constraint on `uuid` is cheap, a fulltext index is not free |
| Restricting supersession leaves genuinely stale facts current | Medium | Temporal benchmark accuracy drops after Task 3 | Task 3 narrows an over-firing rule; the replacement policy is `05`'s deliverable and must land before Phase 2 closes, or the system trades one temporal failure for another |
| Reserved labels conflict with existing extractor output | Medium | Nodes get both `:EventHub` and an LLM-chosen `:EVENT` label | Additive only; never remove an LLM label; assert old read paths still work in Task 6's tests |
| Community Edition users silently lose tenant isolation | High | Two brains' data interleaves in one database | Tasks 5 and 7 together must either refuse startup or add a `brain_id` predicate everywhere; this is a correctness and possibly a compliance issue, not a performance one (see open question 4) |
| Making `brain_id` required breaks callers found only at runtime | Medium | An MCP tool or agent path raises `TypeError` in production after Task 5 | Grep every call site before landing; the alternative — leaving the default in place — is a silent cross-tenant write, so a loud break is the better failure mode |
| The whole plan optimizes the wrong backend | High | Work lands on Neo4j while users run NetworkX-over-Postgres | Open question 1 gates Phase 2; every Phase 2 task must be mirrored in `src/lib/postgresql/networkx_client.py` or explicitly scoped out |
| Degree instrumentation reveals the real problem is elsewhere | Low | Hub degree turns out to be small | Good outcome — Task 12 is cheap precisely so that Task 13 and technique F are not built on assumption |

---

## What I could not determine

- Whether the Mongo data client is exercised by anyone; it is wired (`instances.py:118-122`) and 501 lines long, but `.env.example` selects Postgres.
- Actual production graph sizes. Every scale claim here is analytical (query shape × index absence), not measured, because no profiling data exists in the repo. Task 12 exists to replace that gap with numbers.
- Whether the Milvus `AUTOINDEX` resolves to HNSW or IVF in the maintainer's deployment — it depends on the Milvus version and deployment mode, and nothing pins it.
