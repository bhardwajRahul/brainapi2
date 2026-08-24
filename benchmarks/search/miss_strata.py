from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from search.dataset import load_records, split_corpus, write_records
from search.metrics import recall_at_k

TOKEN = re.compile(r"[a-z0-9]+", re.I)
JUNK_PUNCT = re.compile(r"[^\w\s\-]", re.UNICODE)
COMBINING = re.compile(r"[\u0300-\u036f]")
SQL_MARKERS = (
    "not null",
    "integer",
    "varchar",
    "p_num",
    "i_num",
    "primary key",
)
SKIP_REWRITE_QIDS = frozenset({"esci-72", "72"})


def canonical_qid(qid: str) -> str:
    raw = str(qid or "").strip()
    if raw.lower().startswith("esci-"):
        return raw.lower()
    if raw.isdigit():
        return f"esci-{raw}"
    return raw.lower()


def gold_ids(query: dict[str, Any]) -> list[str]:
    grades = query.get("gold_grades") or {}
    ordered: list[str] = []
    seen: set[str] = set()
    for doc_id in query.get("gold_doc_ids") or []:
        key = str(doc_id)
        if key in seen:
            continue
        if isinstance(grades, dict) and key in grades and float(grades.get(key) or 0) <= 0:
            continue
        seen.add(key)
        ordered.append(key)
    if isinstance(grades, dict):
        for doc_id, gain in grades.items():
            if float(gain or 0) <= 0:
                continue
            key = str(doc_id)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def letter_for_gain(gain: float) -> str:
    value = float(gain or 0)
    if value >= 0.99:
        return "E"
    if value >= 0.09:
        return "S"
    if value > 0:
        return "C"
    return "I"


def query_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN.findall(text or "")}


def token_overlap(query: str, doc_text: str) -> float:
    qtok = query_tokens(query)
    if not qtok:
        return 0.0
    dtok = query_tokens(doc_text)
    return len(qtok & dtok) / len(qtok)


def looks_sql(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in SQL_MARKERS)


def looks_html_entity(text: str) -> bool:
    raw = text or ""
    return "&#" in raw or "&amp;" in raw or "&quot;" in raw or "&lt;" in raw or "&gt;" in raw


def normalize_spelling(text: str) -> str:
    raw = unicodedata.normalize("NFKC", str(text or ""))
    raw = html.unescape(raw)
    raw = unicodedata.normalize("NFKD", raw)
    raw = COMBINING.sub("", raw)
    raw = re.sub(r"^[\s\-*._]+", "", raw)
    raw = re.sub(r"[\s\-*._]+$", "", raw)
    raw = JUNK_PUNCT.sub(" ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def rewrite_query(qid: str, text: str) -> str | None:
    if canonical_qid(qid) in SKIP_REWRITE_QIDS:
        return None
    original = str(text or "").strip()
    if not original:
        return None
    if looks_sql(original):
        return None
    unescaped = html.unescape(original).strip()
    unescaped = re.sub(r"^[\s\-*._]+", "", unescaped)
    unescaped = re.sub(r"[\s\-*._]+$", "", unescaped)
    unescaped = re.sub(r"\s+", " ", unescaped).strip()
    if len(unescaped) < 3:
        return None
    if unescaped == original and not looks_html_entity(original):
        return None
    return unescaped


def classify_query(
    eval_row: dict[str, Any],
    query_row: dict[str, Any],
    docs: dict[str, dict[str, Any]],
    *,
    k: int = 50,
) -> dict[str, Any]:
    hits = [str(item) for item in (eval_row.get("hit_ids") or []) if item]
    gold = gold_ids(query_row if query_row else eval_row)
    grades = (query_row.get("gold_grades") if query_row else None) or eval_row.get(
        "gold_grades"
    ) or {}
    hitset = set(hits)
    in_head = [doc_id for doc_id in gold if doc_id in hits[:10]]
    in_tail = [doc_id for doc_id in gold if doc_id in hitset and doc_id not in set(hits[:10])]
    missed = [doc_id for doc_id in gold if doc_id not in hitset]
    if in_head:
        stratum = "head-ok"
    elif in_tail:
        stratum = "rank-too-low"
    else:
        stratum = "total-miss"
    qtext = str(eval_row.get("query") or query_row.get("query") or "")
    miss_letters = Counter()
    gold_letters = Counter()
    overlap_hit: list[float] = []
    overlap_miss: list[float] = []
    for doc_id in gold:
        letter = letter_for_gain(float((grades or {}).get(doc_id) or 1.0))
        gold_letters[letter] += 1
        text = str((docs.get(doc_id) or {}).get("text") or "")
        overlap = token_overlap(qtext, text)
        if doc_id in hitset:
            overlap_hit.append(overlap)
        else:
            miss_letters[letter] += 1
            overlap_miss.append(overlap)
    metrics = eval_row.get("metrics") or {}
    rewritten = rewrite_query(str(eval_row.get("qid") or query_row.get("qid") or ""), qtext)
    pathological = bool(
        rewritten
        or looks_html_entity(qtext)
        or looks_sql(qtext)
    ) and canonical_qid(str(eval_row.get("qid") or "")) not in SKIP_REWRITE_QIDS
    return {
        "qid": eval_row.get("qid") or query_row.get("qid"),
        "query": qtext,
        "stratum": stratum,
        "n_gold": len(gold),
        "n_in_top10": len(in_head),
        "n_in_11_to_k": len(in_tail),
        "n_missed": len(missed),
        "recall@10": float(metrics.get("recall@10") or recall_at_k(hits, gold, 10)),
        "recall@50": float(metrics.get("recall@50") or recall_at_k(hits, gold, k)),
        "gold_letters": dict(gold_letters),
        "miss_letters": dict(miss_letters),
        "mean_overlap_hit": (sum(overlap_hit) / len(overlap_hit)) if overlap_hit else None,
        "mean_overlap_miss": (sum(overlap_miss) / len(overlap_miss)) if overlap_miss else None,
        "pathological": pathological,
        "rewrite": rewritten,
        "skip_rewrite": canonical_qid(str(eval_row.get("qid") or "")) in SKIP_REWRITE_QIDS,
        "k": k,
    }


def classify_eval(
    eval_result: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    k: int | None = None,
) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    by_id = {str(doc.get("doc_id") or ""): doc for doc in docs}
    by_qid = {str(query.get("qid") or ""): query for query in queries}
    cut = int(k or eval_result.get("k") or 50)
    classified: list[dict[str, Any]] = []
    gold_total = 0
    in_head = 0
    in_tail = 0
    missed = 0
    for row in eval_result.get("queries") or []:
        qid = str(row.get("qid") or "")
        item = classify_query(row, by_qid.get(qid) or {}, by_id, k=cut)
        classified.append(item)
        gold_total += int(item["n_gold"])
        in_head += int(item["n_in_top10"])
        in_tail += int(item["n_in_11_to_k"])
        missed += int(item["n_missed"])
    counts = Counter(str(item["stratum"]) for item in classified)
    total_miss = [item for item in classified if item["stratum"] == "total-miss"]
    rewritable = [
        item
        for item in total_miss
        if item.get("rewrite") and not item.get("skip_rewrite")
    ]
    return {
        "n_queries": len(classified),
        "n_gold": gold_total,
        "n_in_top10": in_head,
        "n_in_11_to_k": in_tail,
        "n_missed": missed,
        "k": cut,
        "stratum_counts": dict(counts),
        "n_total_miss": len(total_miss),
        "n_rewritable_total_miss": len(rewritable),
        "run_query_side": bool(rewritable),
        "queries": classified,
        "rewritable_qids": [item["qid"] for item in rewritable],
        "total_miss_qids": [item["qid"] for item in total_miss],
    }


def write_rewritten_jsonl(
    rows: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    dest: Path,
) -> Path:
    rewrites = {
        str(item["qid"]): str(item["rewrite"])
        for item in taxonomy.get("queries") or []
        if item.get("rewrite")
        and item.get("stratum") == "total-miss"
        and not item.get("skip_rewrite")
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("type") or "") != "query":
            out.append(row)
            continue
        qid = str(row.get("qid") or "")
        rewritten = rewrites.get(qid)
        if not rewritten:
            out.append(row)
            continue
        cloned = dict(row)
        cloned["query_original"] = row.get("query")
        cloned["query"] = rewritten
        out.append(cloned)
    return write_records(out, dest)


def write_spell_jsonl(rows: list[dict[str, Any]], dest: Path) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    n_changed = 0
    for row in rows:
        if str(row.get("type") or "") != "query":
            out.append(row)
            continue
        original = str(row.get("query") or "")
        rewritten = normalize_spelling(original)
        cloned = dict(row)
        if rewritten != original:
            cloned["query_original"] = original
            cloned["query"] = rewritten
            n_changed += 1
        out.append(cloned)
    write_records(out, dest)
    return {
        "out_path": str(dest),
        "n_rows": len(out),
        "n_queries_changed": n_changed,
    }


def load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main_classify(
    *,
    eval_path: Path,
    dataset_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    eval_result = load_eval(eval_path)
    rows = load_records(dataset_path)
    taxonomy = classify_eval(eval_result, rows)
    taxonomy["eval_path"] = str(eval_path)
    taxonomy["dataset"] = dataset_path.name
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(taxonomy, indent=2) + "\n", encoding="utf-8")
        taxonomy["out_path"] = str(out_path)
    return taxonomy
