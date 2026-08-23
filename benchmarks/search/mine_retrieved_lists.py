from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from search.catalog import (
    ESCI_CACHE_DIR,
    ESCI_EXAMPLES_FILE,
    ESCI_PRODUCTS_FILE,
    extra_fields_from_catalog,
    _esci_products,
    _iter_parquet,
)
from search.finetune_esci_4class import (
    LABELS,
    LABEL_TO_ID,
    held_out_query_ids,
    product_passage,
)
from search.pool_first_stage import Bm25Index, rank_docs, tokenize

DEFAULT_OUT = Path("data/esci_retrieved_lists.jsonl")


def is_held_out(qid: str, holdout: set[str]) -> bool:
    text = str(qid or "").strip()
    if not text:
        return True
    if text in holdout:
        return True
    if text.lower().startswith("esci-") and text[5:] in holdout:
        return True
    if f"esci-{text}" in holdout:
        return True
    return False


def load_train_groups(
    *,
    holdout: set[str],
    locale: str = "us",
) -> dict[str, dict[str, Any]]:
    examples_path = ESCI_CACHE_DIR / ESCI_EXAMPLES_FILE
    wanted_locale = locale.strip().lower()
    groups: dict[str, dict[str, Any]] = {}
    for item in _iter_parquet(examples_path):
        if int(item.get("small_version") or 0) != 1:
            continue
        if str(item.get("product_locale") or "").strip().lower() != wanted_locale:
            continue
        if str(item.get("split") or "").strip().lower() != "train":
            continue
        qid = str(item.get("query_id") or "").strip()
        if is_held_out(qid, holdout):
            continue
        pid = str(item.get("product_id") or "").strip()
        query = str(item.get("query") or "").strip()
        label = str(item.get("esci_label") or "").strip().upper()
        if not pid or not query or label not in LABEL_TO_ID:
            continue
        group = groups.setdefault(qid, {"query": query, "qrels": {}})
        group["query"] = query
        group["qrels"][pid] = label
    return groups


def select_groups(
    groups: dict[str, dict[str, Any]],
    *,
    holdout: set[str],
    max_queries: int,
    seed: int = 11,
) -> dict[str, dict[str, Any]]:
    kept = {
        qid: group
        for qid, group in groups.items()
        if not is_held_out(qid, holdout)
    }
    ids = list(kept)
    rng = random.Random(seed)
    rng.shuffle(ids)
    cap = max(1, int(max_queries))
    chosen = ids[:cap]
    return {qid: kept[qid] for qid in chosen}


def labeled_hits(
    ranked: Sequence[str],
    qrels: dict[str, str],
    *,
    k: int = 50,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for pid in ranked:
        key = str(pid)
        label = str(qrels.get(key) or "I").strip().upper()
        if label not in LABEL_TO_ID:
            label = "I"
        pairs.append((key, label))
    cut = max(1, int(k))
    top = pairs[:cut]
    if any(label == "I" for _, label in top):
        return top
    for pid, label in pairs[cut:]:
        if label == "I":
            if top:
                top[-1] = (pid, label)
            else:
                top.append((pid, label))
            break
    return top


def passages_for_pids(
    pids: Iterable[str],
    *,
    locale: str = "us",
) -> dict[str, str]:
    needed = {str(pid) for pid in pids if str(pid)}
    products = _esci_products(
        ESCI_CACHE_DIR / ESCI_PRODUCTS_FILE,
        needed,
        locale=locale,
    )
    extra_fields = extra_fields_from_catalog(
        products,
        title_key="product_title",
        description_key="product_description",
    )
    out: dict[str, str] = {}
    for pid in needed:
        catalog = products.get(pid) or {"product_id": pid}
        out[pid] = product_passage(catalog, extra_fields)
    return out


def mine_from_groups(
    groups: dict[str, dict[str, Any]],
    passages: dict[str, str],
    *,
    k: int = 50,
) -> list[dict[str, Any]]:
    docs: list[tuple[str, list[str]]] = []
    for pid, text in passages.items():
        tokens = tokenize(text)
        if not tokens:
            continue
        docs.append((str(pid), tokens))
    if not docs:
        return []
    index = Bm25Index(docs)
    doc_ids = [pid for pid, _ in docs]
    rows: list[dict[str, Any]] = []
    n_groups = len(groups)
    for offset, (qid, group) in enumerate(groups.items(), start=1):
        query = str(group.get("query") or "")
        qrels = {
            str(pid): str(label).upper()
            for pid, label in (group.get("qrels") or {}).items()
        }
        scores = index.scores(tokenize(query))
        ranked = rank_docs(scores, doc_ids)
        hits = labeled_hits(ranked, qrels, k=k)
        for pid, label in hits:
            rows.append(
                {
                    "query_id": str(qid),
                    "query": query,
                    "product_id": pid,
                    "label": label,
                    "passage": passages.get(pid) or pid,
                }
            )
        if offset % 200 == 0 or offset == n_groups:
            print(f"mine retrieve {offset}/{n_groups}", flush=True)
    return rows


def write_lists(rows: Sequence[dict[str, Any]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dest


def mine(
    *,
    jsonl_path: Path,
    out_path: Path,
    max_queries: int = 6000,
    k: int = 50,
    seed: int = 11,
    locale: str = "us",
) -> dict[str, Any]:
    holdout = held_out_query_ids(jsonl_path)
    groups = load_train_groups(holdout=holdout, locale=locale)
    selected = select_groups(
        groups,
        holdout=holdout,
        max_queries=max_queries,
        seed=seed,
    )
    needed: set[str] = set()
    for group in selected.values():
        needed.update(str(pid) for pid in (group.get("qrels") or {}))
    print(
        f"mine-retrieved-lists queries={len(selected)} products={len(needed)} "
        f"holdout={len(holdout)}",
        flush=True,
    )
    passages = passages_for_pids(needed, locale=locale)
    rows = mine_from_groups(selected, passages, k=k)
    write_lists(rows, out_path)
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[str(row["label"])] = counts.get(str(row["label"]), 0) + 1
    n_with_i = 0
    by_q: dict[str, list[str]] = {}
    for row in rows:
        by_q.setdefault(str(row["query_id"]), []).append(str(row["label"]))
    for labels in by_q.values():
        if "I" in labels:
            n_with_i += 1
    meta = {
        "n_queries": len(selected),
        "n_pairs": len(rows),
        "n_holdout_qids": len(holdout),
        "k": k,
        "label_counts": counts,
        "n_lists_with_i": n_with_i,
        "source": "retrieved-bm25",
        "out": str(out_path),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    if out_path.suffix == ".jsonl":
        meta_path = out_path.with_name(out_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine BM25 top-k train lists; unlabeled hits are class I"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-queries", type=int, default=6000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--locale", default="us")
    args = parser.parse_args(argv)
    meta = mine(
        jsonl_path=args.dataset,
        out_path=args.out,
        max_queries=int(args.max_queries),
        k=int(args.k),
        seed=int(args.seed),
        locale=str(args.locale),
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
