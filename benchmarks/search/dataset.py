from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

DOC_MARKER_PREFIX = "DOCID "


def doc_marker(doc_id: str) -> str:
    return f"{DOC_MARKER_PREFIX}{doc_id}"


def write_records(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def split_corpus(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    docs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows:
        kind = str(row.get("type") or "").strip().lower()
        if kind == "doc":
            docs.append(row)
        elif kind == "query":
            queries.append(row)
        else:
            raise ValueError(f"Unknown record type {kind!r} in {row!r}")
    return docs, queries


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    slices = Counter(str(q.get("slice") or "unspecified") for q in queries)
    return {
        "n_docs": len(docs),
        "n_queries": len(queries),
        "slices": dict(slices),
        "doc_ids": [str(d.get("doc_id") or "") for d in docs],
        "graded": any(bool(q.get("gold_grades")) for q in queries),
    }


def map_doc_ids_to_chunks(
    docs: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {str(doc["doc_id"]): set() for doc in docs}
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        chunk_id = str(chunk.get("id") or "")
        if not chunk_id:
            continue
        for doc in docs:
            doc_id = str(doc.get("doc_id") or "")
            marker = str(doc.get("marker") or doc_marker(doc_id))
            if marker and marker in text:
                mapping.setdefault(doc_id, set()).add(chunk_id)
    return mapping
