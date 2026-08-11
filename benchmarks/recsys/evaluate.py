from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recsys.client import BrainAPIClient
from recsys.config import Settings
from recsys.mapping import (
    interactions_to_triples,
    item_uuid,
    leave_one_out_splits,
)


def load_interactions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def ensure_run_dir(settings: Settings, run_id: str | None = None) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rid = run_id or f"recsys-{stamp}"
    run_dir = settings.runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return rid, run_dir


def recommendation_hits_holdout(
    items: list[dict[str, Any]],
    *,
    holdout_item_id: str,
    holdout_item_uuid: str,
    k: int,
) -> bool:
    targets = {
        holdout_item_id,
        holdout_item_uuid,
        item_uuid(holdout_item_id),
        holdout_item_id.removeprefix("item:") if holdout_item_id.startswith("item:") else holdout_item_id,
    }
    for item in items[:k]:
        if not isinstance(item, dict):
            continue
        node = item.get("node") if isinstance(item.get("node"), dict) else {}
        cand = {
            str(item.get("item_id") or ""),
            str(item.get("uuid") or ""),
            str(item.get("id") or ""),
            str(node.get("uuid") or ""),
            str(node.get("name") or ""),
        }
        if cand & targets:
            return True
    return False


def _normalize_recommend_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    items = payload.get("items") or payload.get("recommendations") or []
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def popularity_baseline_hits(
    train_rows: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    *,
    ks: tuple[int, ...],
) -> dict[str, float]:
    freq = Counter(str(r["item_id"]) for r in train_rows)
    ranked = [item for item, _c in freq.most_common()]
    hit_counts = {k: 0 for k in ks}
    for split in splits:
        holdout = str(split["holdout_item_id"])
        for k in ks:
            if holdout in ranked[:k]:
                hit_counts[k] += 1
    n = max(1, len(splits))
    return {f"popularity_hit_rate@{k}": hit_counts[k] / n for k in ks}


def ingest_train_interactions(
    client: BrainAPIClient,
    interactions: list[dict[str, Any]],
    *,
    timeout_s: float = 600.0,
    chunk_size: int = 200,
) -> dict[str, Any]:
    triples = interactions_to_triples(interactions, include_catalog=True)
    if not triples:
        raise RuntimeError("No triples to ingest")

    task_ids: list[str] = []
    last_waited: dict[str, Any] | None = None
    total_latency = 0.0
    chunk = max(1, int(chunk_size))
    for start in range(0, len(triples), chunk):
        batch = triples[start : start + chunk]
        submitted = client.ingest_structured(batch)
        task_id = (submitted.data or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id from structured ingest: {submitted.data}")
        waited = client.wait_for_task(task_id, timeout_s=timeout_s)
        status = (waited.data or {}).get("status")
        if status not in {"completed", "partial_failed"}:
            raise RuntimeError(
                f"Structured ingest chunk failed: status={status} task={task_id}"
            )
        task_ids.append(task_id)
        last_waited = waited.data
        total_latency += waited.latency_ms

    return {
        "task_id": task_ids[-1],
        "task_ids": task_ids,
        "status": "completed",
        "n_triples": len(triples),
        "n_interactions": len(interactions),
        "n_chunks": len(task_ids),
        "latency_ms": total_latency,
        "task": last_waited,
    }


def evaluate_leave_one_out(
    client: BrainAPIClient,
    interactions: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = (10, 20),
    timeout_s: float = 600.0,
    epochs: int = 20,
    ingest_chunk_size: int = 200,
    dataset_name: str | None = None,
    backend: str = "graph",
    include_attribute_pref: bool = True,
) -> dict[str, Any]:
    backend = (backend or "graph").strip().lower()
    if backend not in {"graph", "lightgcn"}:
        raise ValueError(f"Unknown backend {backend!r}; use graph or lightgcn")

    splits = leave_one_out_splits(interactions)
    if not splits:
        raise RuntimeError("No users with ≥2 interactions for leave-one-out eval")

    train_rows = [row for split in splits for row in split["train"]]
    ingest_info = ingest_train_interactions(
        client,
        train_rows,
        timeout_s=timeout_s,
        chunk_size=ingest_chunk_size,
    )
    if ingest_info["status"] not in {"completed", "partial_failed"}:
        raise RuntimeError(
            f"Structured ingest failed: status={ingest_info['status']} "
            f"task={ingest_info['task_id']}"
        )

    train_payload: dict[str, Any] = {"status": "skipped", "backend": backend}
    if backend == "lightgcn":
        train_result = client.train_lightgcn(
            epochs=epochs, wait=True, timeout_s=timeout_s
        )
        train_payload = train_result.data or {}
        train_status = train_payload.get("status")
        nested = (train_payload.get("task") or {}).get("status")
        if train_status == "timeout" or (
            train_status not in {"completed"} and nested != "completed"
        ):
            hint = ""
            if train_status == "timeout" or nested in {None, "queued"}:
                hint = (
                    " Train stayed queued — Celery likely missing queue recsys_gnn. "
                    "Harness uses wait=true (in-process train) after plugin sync/restart; "
                    "re-sync plugins/recsys-gnn and restart the API."
                )
            raise RuntimeError(f"LightGCN train failed: {train_payload}.{hint}")

    max_k = max(ks)
    per_user: list[dict[str, Any]] = []
    hit_counts = {k: 0 for k in ks}
    model_name = "graph-recommend" if backend == "graph" else "lightgcn"

    for split in splits:
        user_id = split["user_id"]
        holdout_id = split["holdout_item_id"]
        holdout_uuid = split["holdout_item_uuid"]
        try:
            if backend == "graph":
                result = client.recommend_graph(
                    user_id,
                    top_k=max_k,
                    exclude_seen=True,
                    include_attribute_pref=include_attribute_pref,
                )
            else:
                result = client.recommend(
                    user_id, top_k=max_k, exclude_seen=True
                )
            payload = result.data or {}
            items = _normalize_recommend_items(payload)
            error = None
            if payload.get("status") not in (None, "ok") and backend == "lightgcn":
                error = str(payload.get("message") or payload.get("status"))
        except Exception as exc:  # noqa: BLE001
            items = []
            error = str(exc)
            result = None

        hits = {
            k: recommendation_hits_holdout(
                items,
                holdout_item_id=holdout_id,
                holdout_item_uuid=holdout_uuid,
                k=k,
            )
            for k in ks
        }
        for k, hit in hits.items():
            if hit:
                hit_counts[k] += 1

        per_user.append(
            {
                "user_id": user_id,
                "holdout_item_id": holdout_id,
                "holdout_item_uuid": holdout_uuid,
                "n_train": len(split["train"]),
                "n_recs": len(items),
                "hits": hits,
                "latency_ms": getattr(result, "latency_ms", None),
                "error": error,
            }
        )

    n_users = len(per_user)
    metrics = {f"hit_rate@{k}": hit_counts[k] / n_users for k in ks}
    metrics.update({f"recall@{k}": hit_counts[k] / n_users for k in ks})
    metrics.update(popularity_baseline_hits(train_rows, splits, ks=ks))

    return {
        "brain_id": client.settings.brain_id,
        "model": model_name,
        "backend": backend,
        "dataset": dataset_name,
        "n_users": n_users,
        "ks": list(ks),
        "ingest": ingest_info,
        "train": train_payload,
        "metrics": metrics,
        "per_user": per_user,
    }
