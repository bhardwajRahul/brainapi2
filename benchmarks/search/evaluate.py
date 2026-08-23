from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from search.client import BrainAPIClient, SearchDisabledError, TimedResult
from search.config import FROZEN_STRUCTURED_BRAINS, Settings
from search.dataset import map_doc_ids_to_chunks, split_corpus
from search.mapping import docs_to_triples, interactions_to_triples
from search.metrics import (
    aggregate_query_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
    retrieve_latency_ms,
)


def ensure_run_dir(settings: Settings, run_id: str | None = None) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rid = run_id or f"search-{stamp}"
    run_dir = settings.runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return rid, run_dir


def _hit_ids(payload: dict[str, Any] | None) -> list[str]:
    hits = (payload or {}).get("hits") or []
    ids: list[str] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        hid = hit.get("id")
        if hid:
            ids.append(str(hid))
    return ids


def _hit_dump(
    payload: dict[str, Any] | None,
    chunk_to_doc: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    mapping = chunk_to_doc or {}
    rows: list[dict[str, Any]] = []
    for hit in (payload or {}).get("hits") or []:
        if not isinstance(hit, dict):
            continue
        hid = hit.get("id")
        if not hid:
            continue
        raw_id = str(hid)
        node_id = str(hit.get("node_id") or "").strip() or None
        rows.append(
            {
                "id": raw_id,
                "channel": str(hit.get("channel") or ""),
                "doc_id": mapping.get(raw_id, raw_id),
                "node_id": node_id,
            }
        )
    return rows


_DOC_EXTRAS_KEYS = ("brand", "color", "locale")


def doc_meta_keys(doc: dict[str, Any]) -> dict[str, str] | None:
    meta: dict[str, str] = {}
    extras = doc.get("extras")
    if isinstance(extras, dict):
        for key, value in extras.items():
            if value is None:
                continue
            meta[str(key)] = str(value)
    for key in _DOC_EXTRAS_KEYS:
        value = doc.get(key)
        if value is None:
            continue
        meta[str(key)] = str(value)
    return meta or None


def ingest_docs(
    client: BrainAPIClient,
    docs: list[dict[str, Any]],
    *,
    timeout_s: float,
    skip_enrichment: bool = True,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for doc in docs:
        text = str(doc.get("text") or "")
        accepted = client.ingest_text(
            text,
            skip_enrichment=skip_enrichment,
            meta_keys=doc_meta_keys(doc),
        )
        task_id = str((accepted.data or {}).get("task_id") or "")
        if not task_id:
            tasks.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "status": "failed",
                    "error": "ingest response missing task_id",
                }
            )
            continue
        waited = client.wait_for_task(
            task_id,
            timeout_s=timeout_s,
            poll_interval_s=0.25 if skip_enrichment else 2.0,
        )
        status = str((waited.data or {}).get("status") or "unknown")
        tasks.append(
            {
                "doc_id": doc.get("doc_id"),
                "task_id": task_id,
                "status": status,
            }
        )
    statuses = [str(item.get("status")) for item in tasks]
    if not tasks:
        overall = "failed"
    elif all(status == "completed" for status in statuses):
        overall = "completed"
    elif any(status == "completed" for status in statuses):
        overall = "partial_failed"
    else:
        overall = "failed"
    return {"status": overall, "n_docs": len(docs), "tasks": tasks}


def ingest_triples(
    client: BrainAPIClient,
    triples: list[dict[str, Any]],
    *,
    timeout_s: float,
    chunk_size: int = 20000,
) -> dict[str, Any]:
    if not triples:
        return {"status": "skipped", "n_triples": 0, "tasks": []}
    tasks: list[dict[str, Any]] = []
    chunk = max(1, int(chunk_size))
    for start in range(0, len(triples), chunk):
        batch = triples[start : start + chunk]
        submitted = client.ingest_structured(batch)
        task_id = str((submitted.data or {}).get("task_id") or "")
        if not task_id:
            tasks.append(
                {
                    "status": "failed",
                    "error": "structured ingest response missing task_id",
                }
            )
            continue
        waited = client.wait_for_task(task_id, timeout_s=timeout_s)
        status = str((waited.data or {}).get("status") or "unknown")
        tasks.append({"task_id": task_id, "status": status})
    statuses = [str(item.get("status")) for item in tasks]
    if not tasks:
        overall = "failed"
    elif all(status == "completed" for status in statuses):
        overall = "completed"
    elif any(status == "completed" for status in statuses):
        overall = "partial_failed"
    else:
        overall = "failed"
    return {"status": overall, "n_triples": len(triples), "tasks": tasks}


def list_all_text_chunks(client: BrainAPIClient) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    skip = 0
    page = 100
    total = None
    while True:
        result = client.list_text_chunks(limit=page, skip=skip)
        payload = result.data or {}
        batch = payload.get("data") or []
        if not isinstance(batch, list):
            break
        chunks.extend([item for item in batch if isinstance(item, dict)])
        total = payload.get("total")
        skip += len(batch)
        if not batch:
            break
        if total is not None and skip >= int(total):
            break
    return chunks


def invert_doc_chunks(doc_to_chunks: dict[str, set[str]]) -> dict[str, str]:
    chunk_to_doc: dict[str, str] = {}
    for doc_id, chunks in (doc_to_chunks or {}).items():
        for chunk_id in chunks or set():
            chunk_to_doc[str(chunk_id)] = str(doc_id)
    return chunk_to_doc


def canonicalize_hit_ids(
    ranked: list[str],
    chunk_to_doc: dict[str, str],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for hid in ranked:
        key = chunk_to_doc.get(str(hid), str(hid))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def unique_doc_counts(
    raw_ids: list[str],
    chunk_to_doc: dict[str, str] | None = None,
    *,
    k: int = 20,
) -> tuple[int, int]:
    head = [str(item) for item in raw_ids[:k] if item]
    mapping = chunk_to_doc or {}
    canonical = [mapping.get(item, item) for item in head]
    return len(set(head)), len({item for item in canonical if item})


def channel_id_lists(
    payload: dict[str, Any] | None,
    hits: list[dict[str, str]] | None = None,
) -> dict[str, list[str]]:
    raw = (payload or {}).get("channel_lists") or {}
    if not isinstance(raw, dict):
        raw = {}
    out: dict[str, list[str]] = {
        "dense_ids": [str(item) for item in (raw.get("dense") or []) if item],
        "bm25_ids": [str(item) for item in (raw.get("bm25") or []) if item],
        "entity_ids": [str(item) for item in (raw.get("entities") or []) if item],
        "community_ids": [str(item) for item in (raw.get("communities") or []) if item],
        "event_ids": [str(item) for item in (raw.get("events") or []) if item],
        "neighbor_ids": [str(item) for item in (raw.get("neighbors") or []) if item],
        "passage_ids": [],
    }
    by_channel: dict[str, list[str]] = {}
    for hit in hits or []:
        channel = str(hit.get("channel") or "")
        hid = str(hit.get("id") or "")
        if not channel or not hid:
            continue
        by_channel.setdefault(channel, []).append(hid)
    out["passage_ids"] = list(by_channel.get("passages") or [])
    if not out["entity_ids"]:
        out["entity_ids"] = list(by_channel.get("entities") or [])
    if not out["community_ids"]:
        out["community_ids"] = list(by_channel.get("communities") or [])
    if not out["event_ids"]:
        out["event_ids"] = list(by_channel.get("events") or [])
    if not out["neighbor_ids"]:
        out["neighbor_ids"] = list(by_channel.get("neighbors") or [])
    return out


def gold_chunk_ids(
    query: dict[str, Any],
    doc_to_chunks: dict[str, set[str]],
) -> set[str]:
    gold: set[str] = set()
    for doc_id in query.get("gold_doc_ids") or []:
        gold.update(doc_to_chunks.get(str(doc_id)) or set())
    for chunk_id in query.get("gold_chunk_ids") or []:
        gold.add(str(chunk_id))
    grades = query.get("gold_grades") or {}
    if isinstance(grades, dict):
        for doc_id, gain in grades.items():
            if float(gain or 0) <= 0:
                continue
            gold.update(doc_to_chunks.get(str(doc_id)) or set())
    return gold


def gold_chunk_grades(
    query: dict[str, Any],
    doc_to_chunks: dict[str, set[str]],
) -> dict[str, float]:
    grades: dict[str, float] = {}
    raw = query.get("gold_grades") or {}
    if isinstance(raw, dict) and raw:
        for doc_id, gain in raw.items():
            value = float(gain or 0)
            if value <= 0:
                continue
            for chunk_id in doc_to_chunks.get(str(doc_id)) or set():
                grades[chunk_id] = max(grades.get(chunk_id, 0.0), value)
        return grades
    for chunk_id in gold_chunk_ids(query, doc_to_chunks):
        grades[chunk_id] = 1.0
    return grades


def gold_hit_ids(
    query: dict[str, Any],
    doc_to_chunks: dict[str, set[str]],
) -> set[str]:
    gold: set[str] = set()
    for doc_id in query.get("gold_doc_ids") or []:
        gold.add(str(doc_id))
    grades = query.get("gold_grades") or {}
    if isinstance(grades, dict):
        for doc_id, gain in grades.items():
            if float(gain or 0) <= 0:
                continue
            gold.add(str(doc_id))
    chunk_to_doc = invert_doc_chunks(doc_to_chunks)
    for chunk_id in query.get("gold_chunk_ids") or []:
        gold.add(chunk_to_doc.get(str(chunk_id), str(chunk_id)))
    if gold:
        return gold
    return {
        chunk_to_doc.get(chunk_id, chunk_id)
        for chunk_id in gold_chunk_ids(query, doc_to_chunks)
    }


def gold_hit_grades(
    query: dict[str, Any],
    doc_to_chunks: dict[str, set[str]],
) -> dict[str, float]:
    grades: dict[str, float] = {}
    raw = query.get("gold_grades") or {}
    if isinstance(raw, dict) and raw:
        for doc_id, gain in raw.items():
            value = float(gain or 0)
            if value <= 0:
                continue
            key = str(doc_id)
            grades[key] = max(grades.get(key, 0.0), value)
        return grades
    for doc_id in gold_hit_ids(query, doc_to_chunks):
        grades[str(doc_id)] = 1.0
    return grades


def candidate_pool_ids(query: dict[str, Any]) -> list[str]:
    raw = query.get("candidate_doc_ids") or []
    if raw:
        return [str(item) for item in raw]
    return [str(item) for item in (query.get("gold_doc_ids") or [])]


def candidate_pool_grades(query: dict[str, Any]) -> dict[str, float]:
    raw = query.get("candidate_grades") or {}
    if isinstance(raw, dict) and raw:
        return {str(doc_id): float(gain) for doc_id, gain in raw.items()}
    grades = {
        str(doc_id): float(gain)
        for doc_id, gain in (query.get("gold_grades") or {}).items()
    }
    for doc_id in query.get("gold_doc_ids") or []:
        grades.setdefault(str(doc_id), 1.0)
    return grades


def ingested_doc_ids(doc_to_chunks: dict[str, set[str]]) -> set[str]:
    return {
        str(doc_id)
        for doc_id, chunks in (doc_to_chunks or {}).items()
        if chunks
    }


def filter_ranked_to_pool(
    ranked: list[str],
    pool: list[str],
) -> list[str]:
    allowed = {str(item) for item in pool}
    return [item for item in ranked if str(item) in allowed]


def score_search_result(
    result: TimedResult,
    *,
    gold: set[str],
    ks: tuple[int, ...],
    grades: dict[str, float] | None = None,
    chunk_to_doc: dict[str, str] | None = None,
    pool_ids: list[str] | None = None,
) -> dict[str, Any]:
    payload = result.data if isinstance(result.data, dict) else {}
    ranked = _hit_ids(payload)
    hits = _hit_dump(payload, chunk_to_doc)
    n_unique_raw, n_unique_canonical = unique_doc_counts(
        ranked, chunk_to_doc, k=20
    )
    unique_k = max(ks) if ks else 20
    n_retrieve_raw, n_retrieve_canonical = unique_doc_counts(
        ranked, chunk_to_doc, k=unique_k
    )
    lists = channel_id_lists(payload, hits)
    if chunk_to_doc:
        ranked = canonicalize_hit_ids(ranked, chunk_to_doc)
    if pool_ids:
        ranked = filter_ranked_to_pool(ranked, pool_ids)
    retrieve_ms, embed_ms = retrieve_latency_ms(
        payload.get("stage_timings") if isinstance(payload, dict) else None,
        result.latency_ms,
    )
    ndcg_ks = tuple(dict.fromkeys((*ks, 10, 20)))
    metrics = {f"recall@{cut}": recall_at_k(ranked, gold, cut) for cut in ks}
    for cut in ndcg_ks:
        metrics[f"ndcg@{cut}"] = ndcg_at_k(ranked, gold, cut, grades=grades)
    full_k = max(len(ranked), 1)
    metrics["ndcg"] = ndcg_at_k(ranked, gold, full_k, grades=grades)
    metrics["mrr"] = mrr(ranked, gold)
    return {
        "hit_ids": ranked,
        "hits": hits,
        "metrics": metrics,
        "retrieve_ms": retrieve_ms,
        "embed_ms": embed_ms,
        "client_wall_ms": result.latency_ms,
        "n_hits": len(ranked),
        "n_unique_docs_raw": n_unique_raw,
        "n_unique_docs_canonical": n_unique_canonical,
        "n_unique_docs_retrieve_raw": n_retrieve_raw,
        "n_unique_docs_retrieve_canonical": n_retrieve_canonical,
        "unique_docs_k": unique_k,
        "facets": payload.get("facets") if isinstance(payload, dict) else None,
        "response_node_ids": (
            list(payload.get("node_ids") or []) if isinstance(payload, dict) else []
        ),
        **lists,
    }


WANDSGRAPH_BRAIN = "searchbenchwandsgraph"


def assert_wandsgraph_node_join(
    client: BrainAPIClient,
    result: dict[str, Any],
    *,
    neighbor_limit: int = 5,
) -> dict[str, Any]:
    if client.settings.brain_id != WANDSGRAPH_BRAIN:
        return {"skipped": True, "brain_id": client.settings.brain_id}
    queries = result.get("queries") or []
    if not queries:
        raise SystemExit("searchbenchwandsgraph node_id join needs at least one query")
    first = queries[0]
    hits = first.get("hits") or []
    matched = [
        hit
        for hit in hits
        if isinstance(hit, dict)
        and hit.get("node_id")
        and str(hit.get("node_id")) == str(hit.get("doc_id") or "")
    ]
    if not matched:
        raise SystemExit(
            "searchbenchwandsgraph passage hits missing node_id matching DOCID/doc_id. "
            f"sample={hits[:3]!r}"
        )
    node_id = str(matched[0]["node_id"])
    timed = client.get_neighbors(node_id, limit=neighbor_limit)
    if timed.status_code != 200:
        raise SystemExit(
            f"GET /retrieve/entities/neighbors?uuid={node_id} returned {timed.status_code}"
        )
    payload = timed.data if isinstance(timed.data, dict) else {}
    return {
        "skipped": False,
        "node_id": node_id,
        "status_code": timed.status_code,
        "neighbor_count": payload.get("count"),
        "qid": first.get("qid"),
    }


def evaluate_search(
    client: BrainAPIClient,
    rows: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = (5, 10, 20),
    timeout_s: float = 600.0,
    k: int = 20,
    fusion: str | None = None,
    fusion_alpha: float | None = None,
    rerank: str | None = None,
    mode: str | None = None,
    channels: list[str] | None = None,
    node_labels: list[str] | None = None,
    community_labels: list[str] | None = None,
    expand: str | None = None,
    dataset_name: str = "search_toy.jsonl",
    limit_docs: int | None = None,
    limit_queries: int | None = None,
    skip_enrichment: bool = True,
    ingest_graph: bool = False,
    skip_ingest: bool = False,
    rank_pool: bool = False,
    personalize: bool = False,
    interactions: list[dict[str, Any]] | None = None,
    extras: dict[str, str] | None = None,
) -> dict[str, Any]:
    brain_id = client.settings.brain_id
    if (ingest_graph or interactions) and brain_id in FROZEN_STRUCTURED_BRAINS:
        raise SystemExit(
            f"Refusing structured graph/interaction ingest on frozen brain {brain_id}. "
            "Use searchbenchwandsgraph (or another non-frozen searchbench*)."
        )
    docs, queries = split_corpus(rows)
    if limit_docs is not None:
        docs = docs[:limit_docs]
    if limit_queries is not None:
        queries = queries[:limit_queries]

    if skip_ingest:
        ingest = {
            "status": "completed",
            "n_docs": len(docs),
            "tasks": [],
            "reused": True,
        }
    else:
        ingest = ingest_docs(
            client,
            docs,
            timeout_s=timeout_s,
            skip_enrichment=skip_enrichment,
        )
    graph_ingest: dict[str, Any] = {"status": "skipped", "n_triples": 0, "tasks": []}
    if ingest_graph and docs:
        graph_ingest = ingest_triples(
            client,
            docs_to_triples(docs),
            timeout_s=timeout_s,
        )
    interaction_ingest: dict[str, Any] = {
        "status": "skipped",
        "n_triples": 0,
        "tasks": [],
    }
    if interactions:
        interaction_ingest = ingest_triples(
            client,
            interactions_to_triples(interactions),
            timeout_s=timeout_s,
        )
    chunks = list_all_text_chunks(client) if ingest["status"] != "failed" else []
    doc_to_chunks = map_doc_ids_to_chunks(docs, chunks)
    mapped = sum(1 for ids in doc_to_chunks.values() if ids)
    chunk_to_doc = invert_doc_chunks(doc_to_chunks)

    per_query: list[dict[str, Any]] = []
    search_error: str | None = None
    event_probe: dict[str, Any] | None = None
    for query in queries:
        gold = gold_hit_ids(query, doc_to_chunks)
        pool = candidate_pool_ids(query) if rank_pool else []
        grades = (
            candidate_pool_grades(query)
            if rank_pool
            else gold_hit_grades(query, doc_to_chunks)
        )
        target = None
        if personalize:
            raw_target = str(query.get("target") or "").strip()
            if raw_target:
                target = raw_target
        try:
            result = client.search(
                str(query.get("query") or ""),
                k=k,
                fusion=fusion,
                fusion_alpha=fusion_alpha,
                rerank=rerank,
                mode=mode,
                channels=channels,
                node_labels=node_labels,
                community_labels=community_labels,
                expand=expand,
                extras=extras,
                target=target,
                profile_stages=True,
            )
        except SearchDisabledError as exc:
            search_error = str(exc)
            break
        except RuntimeError as exc:
            search_error = str(exc)
            break
        scored = score_search_result(
            result,
            gold=gold,
            ks=ks,
            grades=grades,
            chunk_to_doc=chunk_to_doc,
            pool_ids=pool or None,
        )
        ingested = ingested_doc_ids(doc_to_chunks)
        missing_brain = [pid for pid in pool if pid not in ingested] if pool else []
        retrieved_in_pool = list(scored.get("hit_ids") or [])
        coverage = (
            (len(retrieved_in_pool) / len(pool)) if pool else None
        )
        per_query.append(
            {
                "qid": query.get("qid"),
                "query": query.get("query"),
                "slice": query.get("slice") or "unspecified",
                "gold_doc_ids": list(query.get("gold_doc_ids") or []),
                "gold_grades": {
                    str(doc_id): float(gain)
                    for doc_id, gain in grades.items()
                    if float(gain or 0) > 0
                },
                "gold_chunk_ids": sorted(gold),
                "candidate_doc_ids": pool,
                "pool_size": len(pool) if rank_pool else None,
                "pool_coverage": coverage,
                "missing_from_brain": missing_brain,
                **scored,
            }
        )

    if not queries and interactions and search_error is None:
        names: list[str] = []
        seen: set[str] = set()
        for row in interactions:
            behavior = str(row.get("behavior") or "").strip()
            if behavior and behavior not in seen:
                seen.add(behavior)
                names.append(behavior)
        hit_counts: list[dict[str, Any]] = []
        for name in names[:5]:
            try:
                result = client.search(
                    name,
                    k=k,
                    fusion=fusion,
                    rerank=rerank,
                    mode=mode,
                    channels=channels or ["events"],
                    expand=expand,
                    profile_stages=True,
                )
            except SearchDisabledError as exc:
                search_error = str(exc)
                break
            except RuntimeError as exc:
                search_error = str(exc)
                break
            payload = result.data if isinstance(result.data, dict) else {}
            hits = payload.get("hits") or []
            hit_counts.append({"query": name, "n_hits": len(hits)})
        event_probe = {
            "n_event_queries": len(hit_counts),
            "queries": hit_counts,
            "n_hits": sum(int(item.get("n_hits") or 0) for item in hit_counts),
        }

    metrics = aggregate_query_metrics(per_query, ks=ks) if per_query else {}
    if per_query and any(
        row.get("n_unique_docs_canonical") is not None for row in per_query
    ):
        n_raw = [
            float(row.get("n_unique_docs_raw") or 0) for row in per_query
        ]
        n_canon = [
            float(row.get("n_unique_docs_canonical") or 0) for row in per_query
        ]
        metrics["unique_docs@20_raw"] = sum(n_raw) / len(n_raw)
        metrics["unique_docs@20_canonical"] = sum(n_canon) / len(n_canon)
        retrieve_k = int(per_query[0].get("unique_docs_k") or 20)
        n_ret = [
            float(row.get("n_unique_docs_retrieve_canonical") or 0)
            for row in per_query
        ]
        metrics[f"unique_docs@{retrieve_k}_canonical"] = sum(n_ret) / len(n_ret)
    if rank_pool and per_query:
        coverages = [
            float(row["pool_coverage"])
            for row in per_query
            if row.get("pool_coverage") is not None
        ]
        if coverages:
            metrics["pool_coverage"] = sum(coverages) / len(coverages)
        metrics["missing_from_brain"] = sum(
            len(row.get("missing_from_brain") or []) for row in per_query
        ) / len(per_query)
    status = "ok"
    if docs and ingest["status"] not in {"completed", "partial_failed"}:
        status = "failed"
    if docs and mapped < 1:
        status = "failed"
    if queries and not per_query:
        status = "failed"
    if not queries and interactions:
        if event_probe is None or int(event_probe.get("n_hits") or 0) < 1:
            status = "failed"
        elif search_error is None:
            status = "ok"
    if search_error:
        status = "failed"

    protocol = None
    if brain_id == "searchbenchwandsgraph":
        protocol = (
            "architecture demo: isolated catalog graph on searchbenchwandsgraph; "
            "not a quality-default claim vs frozen WANDS passages 0.823 or ESCI 0.500"
        )
    if brain_id == "searchbenchjdslice":
        protocol = (
            "architecture demo: JDsearch rank-pool slice on searchbenchjdslice; "
            "not TEM 0.219; not frozen WANDS 0.823 or ESCI 0.500"
        )

    return {
        "status": status,
        "brain_id": client.settings.brain_id,
        "dataset": dataset_name,
        "protocol": protocol,
        "fusion": fusion or "rrf",
        "fusion_alpha": fusion_alpha,
        "rerank": rerank or "none",
        "mode": mode or "default",
        "channels": channels or ["passages"],
        "expand": expand or "none",
        "k": k,
        "ks": list(ks),
        "n_docs": len(docs),
        "n_queries": len(per_query),
        "n_docs_mapped": mapped,
        "skip_enrichment": skip_enrichment,
        "ingest_graph": ingest_graph,
        "skip_ingest": skip_ingest,
        "rank_pool": rank_pool,
        "personalize": personalize,
        "extras": extras,
        "ingest": ingest,
        "graph_ingest": graph_ingest,
        "interaction_ingest": interaction_ingest,
        "event_probe": event_probe,
        "search_error": search_error,
        "metrics": metrics,
        "queries": per_query,
    }
