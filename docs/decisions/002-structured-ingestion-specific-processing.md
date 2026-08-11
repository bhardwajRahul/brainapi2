# ADR-002: Structured Ingestion Dual-Mode Contract

## Status

Accepted

## Date

2026-08-05

## Context

BrainAPI exposes two write surfaces:

- `POST /ingest/` — free-text Scout → Architect → optional Janitor (LoCoMo / LongMemEval / BEAM path).
- `POST /ingest/structured` — caller-supplied event triples, with optional free-text enrichment.

Callers building recommenders and catalogs need a **deterministic, LLM-free** write path for known facts (purchases, catalog edges), while still allowing LLM enrichment when text requires indirect intelligence. Before this ADR, “no LLM” was implied only by omitting `text`, and name+type anchors could still call `KGAgent.verify_entity_existence` (LLM). That made deterministic guarantees untestable.

Benchmarks never call `/ingest/structured`. Changing default `/ingest/` or global `PIPELINE_MODE` would invalidate published LoCoMo/BEAM brains. Dual-mode must therefore stay request-scoped on the structured endpoint only.

See also: `docs/research/01-ingestion-extraction.md`, `docs/research/15-ecommerce-gnn-recsys-landscape.md` §7.

## Decision

**1. Explicit `mode` on `IngestionStructuredRequestBody`:** `deterministic` | `hybrid` | `enrich`.

When `mode` is omitted:

- `text` present → `hybrid`
- else → `deterministic`

**2. Mode semantics**

| Mode | Submitted triples | Text / LLM | Anchor resolution |
| --- | --- | --- | --- |
| `deterministic` | Persist via `ingestion_triples_to_relationships` → `process_architect_relationships` | Never run Scout/Architect/Janitor/KGAgent verify. `text` may be stored as provenance only. | UUID lookup, or exact name+type match. Missing name+type anchor → hard fail (no LLM verify, no create-via-verify). |
| `hybrid` | Persist first; authoritative | If `text` set: Scout/Architect `run_structured` with `persist_submitted=False`; enrichment edges tagged `properties.source=llm_enrichment` | Existing behavior (exact match, else vector + LLM verify / create) |
| `enrich` | Same as hybrid | Same as hybrid (LLM backfill when text present) | Same as hybrid |

Submitted edges are tagged `properties.source=structured_deterministic`.

**3. Probability Isolation**

LLM uncertainty is confined to the intelligence lane (`hybrid`/`enrich` with text, or free-text `/ingest/`). Known facts stay on the deterministic lane. Do **not** introduce a process-global “skip LLM” flag that alters `/ingest/`.

**4. Benchmark isolation**

- Do not change default free-text `/ingest/` semantics or `PIPELINE_MODE` defaults for this contract.
- Do not use `PIPELINE_MODE=lightweight` as a structured/deterministic substitute.
- Recsys / interaction structured data should use dedicated brains (e.g. `*recsys*`), never published eval brains (`beam1m1clean`, `locomoconv26*`).

**5. Embeddings**

Deterministic mode still runs embedding encode + vector write in `process_architect_relationships`. “No LLM” ≠ “no model calls.”

## Alternatives Considered

- Global `INGEST_SKIP_LLM` env — rejected; would risk LoCoMo/BEAM.
- Relying on omitted `text` alone — rejected; silent LLM on ambiguous anchors.
- `PIPELINE_MODE=lightweight` as cheap structured path — rejected; different graph shape.

## Consequences

- Callers can assert zero Scout/Architect/KGAgent constructions under `mode=deterministic`.
- Hybrid enrichment is additive and provenance-tagged for recommend ranking.
- ADR-001 remains the broader ingestion-structure decision; this ADR owns structured-only dual-mode.
