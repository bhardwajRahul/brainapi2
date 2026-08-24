from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from search.dataset import load_records, split_corpus

ELECTRONICS_MARKERS = (
    "electronic",
    "electronics",
    "ipad",
    "iphone",
    "phone",
    "tablet",
    "laptop",
    "computer",
    "charger",
    "usb",
    "hdmi",
    "bluetooth",
    "wifi",
    "wi-fi",
    "router",
    "led",
    "battery",
    "samsung",
    "apple",
)


def looks_like_electronics(text: str) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in ELECTRONICS_MARKERS)


def dump_query_miss(
    *,
    dataset_path: Path,
    eval_path: Path,
    qid: str,
) -> dict[str, Any]:
    docs, queries = split_corpus(load_records(dataset_path))
    by_id = {str(doc.get("doc_id") or ""): doc for doc in docs}
    query = next((row for row in queries if str(row.get("qid")) == qid), None)
    if query is None:
        raise SystemExit(f"Missing query {qid} in {dataset_path}")
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    scored = next(
        (row for row in eval_data.get("queries") or [] if str(row.get("qid")) == qid),
        None,
    )
    if scored is None:
        raise SystemExit(f"Missing query {qid} in {eval_path}")
    gold_rows = []
    in_pool = 0
    electronics_gold = 0
    for doc_id in query.get("gold_doc_ids") or []:
        doc = by_id.get(str(doc_id)) or {}
        title = str(doc.get("title") or "")
        text = str(doc.get("text") or "")
        present = bool(doc)
        in_pool += int(present)
        electronic = looks_like_electronics(f"{title}\n{text}")
        electronics_gold += int(electronic)
        gold_rows.append(
            {
                "doc_id": doc_id,
                "title": title,
                "brand": doc.get("brand") or "",
                "in_jsonl": present,
                "electronics_marker": electronic,
            }
        )
    hit_rows = []
    for doc_id in scored.get("hit_ids") or []:
        doc = by_id.get(str(doc_id)) or {}
        title = str(doc.get("title") or "")
        hit_rows.append(
            {
                "doc_id": doc_id,
                "title": title,
                "electronics_marker": looks_like_electronics(title),
            }
        )
    gold_set = {str(item) for item in (query.get("gold_doc_ids") or [])}
    overlap = gold_set & {str(item["doc_id"]) for item in hit_rows}
    if in_pool < len(gold_rows):
        decision = "not_testable"
        reason = "one or more Exact golds are missing from the JSONL pool"
    elif electronics_gold >= max(1, len(gold_rows) // 2):
        decision = "qrel_mismatch"
        reason = (
            "majority of Exact golds look like electronics or electronics accessories; "
            "do not rewrite 'not electronics'"
        )
    elif overlap:
        decision = "not_testable"
        reason = "golds already appear in the scored list"
    else:
        decision = "rewrite_experiment"
        reason = "golds are in-pool, non-electronics, and missing from top hits"
    return {
        "qid": qid,
        "query": query.get("query"),
        "metrics": scored.get("metrics"),
        "gold": gold_rows,
        "hits": hit_rows,
        "n_gold": len(gold_rows),
        "n_gold_in_jsonl": in_pool,
        "n_electronics_gold": electronics_gold,
        "overlap": sorted(overlap),
        "decision": decision,
        "reason": reason,
    }
