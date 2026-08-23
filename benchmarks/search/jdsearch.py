from __future__ import annotations

import argparse
import shutil
import tarfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from search.catalog import (
    DEFAULT_MAX_DOCS,
    DEFAULT_MAX_QUERIES,
    format_product_text,
)
from search.config import DATA_DIR
from search.dataset import write_records

JDSEARCH_NAME = "jdsearch"
JDSEARCH_TAR = DATA_DIR / "JDsearch.tar.gz"
JDSEARCH_CACHE_DIR = DATA_DIR / "jdsearch"
JDSEARCH_JSONL = DATA_DIR / "search_jdsearch.jsonl"
JDSEARCH_INTERACTIONS_JSONL = DATA_DIR / "search_jdsearch_interactions.jsonl"
PRODUCT_MEMBER = "JDsearch/product_meta_data.txt"
BEHAVIOR_MEMBER = "JDsearch/user_behavior_data.txt"
PRODUCT_FILENAME = "product_meta_data.txt"
BEHAVIOR_FILENAME = "user_behavior_data.txt"
TERM_SEP = "\030"
LIST_SEP = "_"
TEST_EPOCH = datetime(2022, 10, 17, tzinfo=timezone.utc)
MAX_DISPLAYED_POOL = 200
GRADED_GAINS = {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.0}
HISTORY_BEHAVIOR = {
    "CLICK": "click",
    "CART": "cart",
    "ORD": "purchase",
    "FLW": "follow",
}
PAPER_N_PRODUCTS = 12_900_000
PAPER_N_QUERIES = 171_728


def join_terms(raw: str | None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in text.split(TERM_SEP) if part.strip()]
    return " ".join(parts)


def split_list(raw: str | None) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [part for part in text.split(LIST_SEP) if part]


def parse_label(raw: str | None) -> float:
    try:
        return float(str(raw or "").strip())
    except ValueError:
        return 0.0


def label_bucket(value: float) -> int:
    if value <= 0:
        return 0
    return int(round(value))


def label_hist_key(value: float) -> str:
    bucket = label_bucket(value)
    if abs(value - bucket) < 1e-9:
        return str(bucket)
    return str(value)


def label_scheme(keys: Iterable[str]) -> str:
    buckets: set[int] = set()
    for key in keys:
        try:
            buckets.add(label_bucket(float(key)))
        except ValueError:
            continue
    if buckets <= {0, 1}:
        return "binary"
    return "graded"


def label_gain(value: float, *, scheme: str) -> float:
    if value <= 0:
        return 0.0
    if scheme == "binary":
        return 1.0
    return GRADED_GAINS.get(label_bucket(value), 1.0)


def user_id_for_row(index: int) -> str:
    return f"jd-u{index}"


def qid_for_row(index: int) -> str:
    return f"jdsearch-{index}"


def map_history_behavior(raw: str | None) -> str:
    key = str(raw or "").strip().upper()
    return HISTORY_BEHAVIOR.get(key, str(raw or "").strip().lower() or "click")


def history_event_times(gaps: list[int], n_hist: int) -> list[datetime]:
    if n_hist <= 0:
        return []
    values = [max(0, int(item)) for item in gaps]
    if not values:
        values = [0]
    while len(values) < n_hist + 1:
        values.append(0)
    cumul = []
    acc = 0
    for gap in values:
        acc += gap
        cumul.append(acc)
    test_offset = cumul[n_hist] if len(cumul) > n_hist else cumul[-1]
    start = TEST_EPOCH - timedelta(seconds=test_offset)
    return [start + timedelta(seconds=cumul[i]) for i in range(n_hist)]


def iso_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jdsearch_interactions_path(jsonl: Path) -> Path:
    return jsonl.with_name(f"{jsonl.stem}_interactions.jsonl")


def cached_product_path(cache_dir: Path | None = None) -> Path:
    return (cache_dir or JDSEARCH_CACHE_DIR) / PRODUCT_FILENAME


def cached_behavior_path(cache_dir: Path | None = None) -> Path:
    return (cache_dir or JDSEARCH_CACHE_DIR) / BEHAVIOR_FILENAME


def _copy_tar_member(tf: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = tf.extractfile(member)
    if src is None:
        raise FileNotFoundError(member.name)
    with dest.open("wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)


def ensure_jdsearch_extracted(
    *,
    tar_path: Path | None = None,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    tar = tar_path or JDSEARCH_TAR
    cache = cache_dir or JDSEARCH_CACHE_DIR
    product = cached_product_path(cache)
    behavior = cached_behavior_path(cache)
    have_product = product.exists() and product.stat().st_size > 0
    have_behavior = behavior.exists() and behavior.stat().st_size > 0
    if have_product and have_behavior:
        return product, behavior
    if not tar.exists():
        raise FileNotFoundError(
            f"Missing JDsearch archive {tar}. Place JDsearch.tar.gz under benchmarks/data/."
        )
    cache.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "r:gz") as tf:
        for member in tf:
            name = Path(member.name).name
            if name == PRODUCT_FILENAME and not have_product:
                _copy_tar_member(tf, member, product)
                have_product = True
            elif name == BEHAVIOR_FILENAME and not have_behavior:
                _copy_tar_member(tf, member, behavior)
                have_behavior = True
            if have_product and have_behavior:
                break
    if not (product.exists() and product.stat().st_size > 0):
        raise FileNotFoundError(f"Missing {PRODUCT_MEMBER} in {tar}")
    if not (behavior.exists() and behavior.stat().st_size > 0):
        raise FileNotFoundError(f"Missing {BEHAVIOR_MEMBER} in {tar}")
    return product, behavior


def resolve_jdsearch_files(
    *,
    product_path: Path | None = None,
    behavior_path: Path | None = None,
    tar_path: Path | None = None,
    cache_dir: Path | None = None,
    extract: bool = True,
) -> tuple[Path, Path]:
    if product_path and behavior_path:
        return Path(product_path), Path(behavior_path)
    cache = cache_dir or JDSEARCH_CACHE_DIR
    product = Path(product_path) if product_path else cached_product_path(cache)
    behavior = Path(behavior_path) if behavior_path else cached_behavior_path(cache)
    if product.exists() and behavior.exists():
        return product, behavior
    if extract:
        return ensure_jdsearch_extracted(tar_path=tar_path, cache_dir=cache)
    missing = [str(path) for path in (product, behavior) if not path.exists()]
    raise FileNotFoundError(f"Missing JDsearch files: {missing}")


def iter_tsv_rows(path: Path, *, max_rows: int | None = None) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from _iter_tsv_stream(handle, max_rows=max_rows)


def _iter_tsv_stream(handle: TextIO, *, max_rows: int | None = None) -> Iterator[dict[str, str]]:
    header: list[str] | None = None
    yielded = 0
    for raw in handle:
        line = raw.rstrip("\n\r")
        if not line:
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        if len(parts) < len(header):
            parts = parts + [""] * (len(header) - len(parts))
        yield dict(zip(header, parts[: len(header)]))
        yielded += 1
        if max_rows is not None and yielded >= max_rows:
            return


def parse_behavior_row(row: dict[str, str], *, index: int) -> dict[str, Any]:
    candidates = split_list(row.get("candidate_wid_list"))
    labels = [parse_label(item) for item in split_list(row.get("candidate_label_list"))]
    while len(labels) < len(candidates):
        labels.append(0.0)
    labels = labels[: len(candidates)]
    history_wids = split_list(row.get("history_wid_list"))
    history_types = split_list(row.get("history_type_list"))
    while len(history_types) < len(history_wids):
        history_types.append("CLICK")
    history_types = history_types[: len(history_wids)]
    gaps = []
    for item in split_list(row.get("history_time_list")):
        try:
            gaps.append(int(float(item)))
        except ValueError:
            gaps.append(0)
    times = history_event_times(gaps, len(history_wids))
    return {
        "index": index,
        "qid": qid_for_row(index),
        "target": user_id_for_row(index),
        "query": join_terms(row.get("query")),
        "candidates": candidates[:MAX_DISPLAYED_POOL],
        "labels": labels[:MAX_DISPLAYED_POOL],
        "history_wids": history_wids,
        "history_types": history_types,
        "history_times": times,
    }


def iter_behavior_rows(
    path: Path, *, max_rows: int | None = None
) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(iter_tsv_rows(path, max_rows=max_rows)):
        yield parse_behavior_row(row, index=index)


def parse_product_row(row: dict[str, str]) -> dict[str, str] | None:
    wid = str(row.get("wid") or "").strip()
    if not wid:
        return None
    cate1 = join_terms(row.get("cate_name_1"))
    cate2 = join_terms(row.get("cate_name_2"))
    cate3 = join_terms(row.get("cate_name_3"))
    cate4 = join_terms(row.get("cate_name_4"))
    hierarchy = " > ".join(part for part in (cate1, cate2, cate3, cate4) if part)
    return {
        "wid": wid,
        "name": join_terms(row.get("name")),
        "brand_name": join_terms(row.get("brand_name")),
        "cate_name_1": cate1,
        "cate_name_2": cate2,
        "cate_name_3": cate3,
        "cate_name_4": cate4,
        "hierarchy": hierarchy,
        "shop_id": str(row.get("shop_id") or "").strip(),
    }


def load_products(
    path: Path,
    needed: set[str],
    *,
    max_rows: int | None = None,
    stop_when_found: bool = True,
) -> tuple[dict[str, dict[str, str]], int, bool]:
    found: dict[str, dict[str, str]] = {}
    seen = 0
    remaining = set(needed)
    for row in iter_tsv_rows(path, max_rows=max_rows):
        seen += 1
        parsed = parse_product_row(row)
        if parsed is None:
            continue
        wid = parsed["wid"]
        if wid in remaining:
            found[wid] = parsed
            remaining.remove(wid)
            if stop_when_found and not remaining:
                return found, seen, False
    truncated = max_rows is not None and seen >= max_rows
    return found, seen, truncated


def collect_jdsearch_stats(
    *,
    product_path: Path | None = None,
    behavior_path: Path | None = None,
    tar_path: Path | None = None,
    cache_dir: Path | None = None,
    extract: bool = True,
    max_behavior_rows: int | None = None,
    max_product_rows: int | None = None,
) -> dict[str, Any]:
    product, behavior = resolve_jdsearch_files(
        product_path=product_path,
        behavior_path=behavior_path,
        tar_path=tar_path,
        cache_dir=cache_dir,
        extract=extract,
    )
    labels = Counter()
    history_types = Counter()
    pool_sizes: list[int] = []
    wids: set[str] = set()
    n_behavior = 0
    truncated = False
    for parsed in iter_behavior_rows(behavior, max_rows=max_behavior_rows):
        n_behavior += 1
        pool_sizes.append(len(parsed["candidates"]))
        for gain in parsed["labels"]:
            labels[label_hist_key(gain)] += 1
        for kind in parsed["history_types"]:
            history_types[kind] += 1
        wids.update(parsed["candidates"])
        wids.update(parsed["history_wids"])
    if max_behavior_rows is not None and n_behavior >= max_behavior_rows:
        truncated = True
    products, n_products, product_truncated = load_products(
        product,
        wids,
        max_rows=max_product_rows,
        stop_when_found=False,
    )
    truncated = truncated or product_truncated
    missing = len(wids) - len(products)
    missing_frac = (missing / len(wids)) if wids else 0.0
    mean_pool = (sum(pool_sizes) / len(pool_sizes)) if pool_sizes else 0.0
    return {
        "n_behavior_rows": n_behavior,
        "n_products_seen": n_products,
        "n_wids_referenced": len(wids),
        "n_wids_in_meta": len(products),
        "n_wids_missing_meta": missing,
        "missing_wid_fraction": missing_frac,
        "label_histogram": dict(labels),
        "label_scheme": label_scheme(labels),
        "mean_pool_size": mean_pool,
        "history_type_counts": dict(history_types),
        "truncated": truncated,
        "product_path": str(product),
        "behavior_path": str(behavior),
    }


def _select_behavior(
    rows: Iterable[dict[str, Any]],
    *,
    max_queries: int,
    max_docs: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    selected: list[dict[str, Any]] = []
    needed: set[str] = set()
    for parsed in rows:
        if not any(gain > 0 for gain in parsed["labels"]):
            continue
        new_ids = [wid for wid in parsed["candidates"] if wid not in needed]
        if needed and len(needed) + len(new_ids) > max_docs:
            continue
        if not needed and len(new_ids) > max_docs:
            parsed = dict(parsed)
            parsed["candidates"] = parsed["candidates"][:max_docs]
            parsed["labels"] = parsed["labels"][:max_docs]
            new_ids = parsed["candidates"]
        needed.update(new_ids)
        selected.append(parsed)
        if len(selected) >= max_queries:
            break
    for parsed in selected:
        for wid in parsed["history_wids"]:
            if len(needed) >= max_docs:
                break
            needed.add(wid)
        if len(needed) >= max_docs:
            break
    return selected, needed


def _query_record(
    parsed: dict[str, Any],
    products: dict[str, dict[str, str]],
    *,
    scheme: str,
) -> dict[str, Any] | None:
    gold_grades: dict[str, float] = {}
    gold_doc_ids: list[str] = []
    candidate_doc_ids: list[str] = []
    candidate_grades: dict[str, float] = {}
    for wid, raw in zip(parsed["candidates"], parsed["labels"]):
        candidate_doc_ids.append(wid)
        gain = label_gain(raw, scheme=scheme)
        candidate_grades[wid] = gain
        if gain <= 0:
            continue
        if wid not in products:
            continue
        gold_grades[wid] = gain
        gold_doc_ids.append(wid)
    if not gold_doc_ids or not parsed["query"]:
        return None
    return {
        "type": "query",
        "qid": parsed["qid"],
        "query": parsed["query"],
        "target": parsed["target"],
        "gold_doc_ids": gold_doc_ids,
        "gold_grades": gold_grades,
        "candidate_doc_ids": candidate_doc_ids,
        "candidate_grades": candidate_grades,
        "slice": JDSEARCH_NAME,
    }


def _doc_record(product: dict[str, str]) -> dict[str, Any]:
    wid = product["wid"]
    title = product["name"] or wid
    extras = (
        ("Brand", product["brand_name"]),
        ("Class", product["cate_name_1"]),
        ("Hierarchy", product["hierarchy"]),
        ("Cate2", product["cate_name_2"]),
        ("Cate3", product["cate_name_3"]),
        ("Cate4", product["cate_name_4"]),
    )
    row: dict[str, Any] = {
        "type": "doc",
        "doc_id": wid,
        "text": format_product_text(
            wid,
            title=title,
            extras=extras,
        ),
        "title": title,
        "class": product["cate_name_1"],
        "hierarchy": product["hierarchy"],
        "brand": product["brand_name"],
        "ingest": "chunks",
        "dataset": JDSEARCH_NAME,
    }
    return row


def _interaction_rows(
    parsed: dict[str, Any],
    products: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wid, kind, moment in zip(
        parsed["history_wids"], parsed["history_types"], parsed["history_times"]
    ):
        if not wid:
            continue
        item = products.get(wid) or {}
        row: dict[str, Any] = {
            "user_id": parsed["target"],
            "item_id": wid,
            "behavior": map_history_behavior(kind),
            "timestamp": iso_utc(moment),
        }
        if item.get("cate_name_1"):
            row["category"] = item["cate_name_1"]
            row["class"] = item["cate_name_1"]
        if item.get("brand_name"):
            row["brand"] = item["brand_name"]
        if item.get("name"):
            row["title"] = item["name"]
        rows.append(row)
    return rows


def prepare_jdsearch_rows(
    *,
    max_queries: int = DEFAULT_MAX_QUERIES,
    max_docs: int = DEFAULT_MAX_DOCS,
    candidates_per_query: int = 0,
    force: bool = False,
    product_path: Path | None = None,
    behavior_path: Path | None = None,
    tar_path: Path | None = None,
    cache_dir: Path | None = None,
    extract: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    del force, candidates_per_query
    product, behavior = resolve_jdsearch_files(
        product_path=product_path,
        behavior_path=behavior_path,
        tar_path=tar_path,
        cache_dir=cache_dir,
        extract=extract,
    )
    selected, needed = _select_behavior(
        iter_behavior_rows(behavior),
        max_queries=max(1, max_queries),
        max_docs=max(1, max_docs),
    )
    products, n_products_seen, truncated = load_products(product, needed)
    hist_keys = [
        label_hist_key(gain) for parsed in selected for gain in parsed["labels"]
    ]
    scheme = label_scheme(hist_keys)
    queries: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    kept_ids: set[str] = set()
    for parsed in selected:
        record = _query_record(parsed, products, scheme=scheme)
        if record is None:
            continue
        queries.append(record)
        kept_ids.update(record["candidate_doc_ids"])
        kept_ids.update(wid for wid in parsed["history_wids"] if wid in products)
        interactions.extend(_interaction_rows(parsed, products))
    kept_ids = {wid for wid in kept_ids if wid in products}
    if len(kept_ids) > max_docs:
        keep_order = [wid for parsed in selected for wid in parsed["candidates"] if wid in kept_ids]
        extra = [wid for wid in kept_ids if wid not in keep_order]
        kept_ids = set((keep_order + extra)[:max_docs])
        queries = [
            query
            for query in queries
            if any(doc_id in kept_ids for doc_id in query["gold_doc_ids"])
        ]
        for query in queries:
            query["gold_doc_ids"] = [
                doc_id for doc_id in query["gold_doc_ids"] if doc_id in kept_ids
            ]
            query["gold_grades"] = {
                doc_id: gain
                for doc_id, gain in query["gold_grades"].items()
                if doc_id in kept_ids
            }
        queries = [query for query in queries if query["gold_doc_ids"]]
        interactions = [
            row for row in interactions if str(row.get("item_id")) in kept_ids
        ]
    docs = [_doc_record(products[wid]) for wid in sorted(kept_ids, key=lambda item: int(item) if item.isdigit() else item)]
    stats = {
        "n_behavior_selected": len(selected),
        "n_queries": len(queries),
        "n_docs": len(docs),
        "n_interactions": len(interactions),
        "n_products_seen": n_products_seen,
        "label_scheme": scheme,
        "truncated": truncated,
        "product_path": str(product),
        "behavior_path": str(behavior),
    }
    return docs + queries, interactions, stats


def prepare_jdsearch_bundle(
    dest: Path,
    *,
    max_queries: int = DEFAULT_MAX_QUERIES,
    max_docs: int = DEFAULT_MAX_DOCS,
    candidates_per_query: int = 0,
    force: bool = False,
    product_path: Path | None = None,
    behavior_path: Path | None = None,
    tar_path: Path | None = None,
    cache_dir: Path | None = None,
    extract: bool = True,
) -> Path:
    rows, interactions, _stats = prepare_jdsearch_rows(
        max_queries=max_queries,
        max_docs=max_docs,
        candidates_per_query=candidates_per_query,
        force=force,
        product_path=product_path,
        behavior_path=behavior_path,
        tar_path=tar_path,
        cache_dir=cache_dir,
        extract=extract,
    )
    path = write_records(rows, dest)
    write_records(interactions, jdsearch_interactions_path(dest))
    return path


def print_jdsearch_stats(stats: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="JDsearch dataset-stats")
    table.add_column("field")
    table.add_column("value")
    for key in (
        "n_behavior_rows",
        "n_products_seen",
        "n_wids_referenced",
        "n_wids_in_meta",
        "n_wids_missing_meta",
        "missing_wid_fraction",
        "label_histogram",
        "label_scheme",
        "mean_pool_size",
        "history_type_counts",
        "truncated",
        "product_path",
        "behavior_path",
    ):
        if key in stats:
            table.add_row(key, str(stats[key]))
    Console().print(table)


def main_stats(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="search.jdsearch_stats")
    parser.add_argument("--product", type=Path, default=None)
    parser.add_argument("--behavior", type=Path, default=None)
    parser.add_argument("--tar", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-behavior-rows", type=int, default=None)
    parser.add_argument("--max-product-rows", type=int, default=None)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args(argv)
    stats = collect_jdsearch_stats(
        product_path=args.product,
        behavior_path=args.behavior,
        tar_path=args.tar,
        cache_dir=args.cache_dir,
        extract=not args.no_extract,
        max_behavior_rows=args.max_behavior_rows,
        max_product_rows=args.max_product_rows,
    )
    if stats.get("truncated") and stats.get("n_products_seen") in {
        PAPER_N_PRODUCTS,
        PAPER_N_QUERIES,
    }:
        raise SystemExit("Refusing invented JDsearch paper counts on a truncated stream")
    print_jdsearch_stats(stats)
    return 0
