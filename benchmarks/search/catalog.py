from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.request import urlretrieve

from search.config import DATA_DIR
from search.dataset import DOC_MARKER_PREFIX, write_records

ESCI_NAME = "esci"
WANDS_NAME = "wands"
JDSEARCH_NAME = "jdsearch"
SUPPORTED_CATALOGS = (ESCI_NAME, WANDS_NAME, JDSEARCH_NAME)
DOWNLOAD_ALL_CATALOGS = (ESCI_NAME, WANDS_NAME)
ESCI_LOCALES = ("us", "es", "jp")

ESCI_JSONL = DATA_DIR / "search_esci.jsonl"
WANDS_JSONL = DATA_DIR / "search_wands.jsonl"
JDSEARCH_JSONL = DATA_DIR / "search_jdsearch.jsonl"
ESCI_CACHE_DIR = DATA_DIR / "esci"
WANDS_CACHE_DIR = DATA_DIR / "wands"

ESCI_GITHUB_BASE = (
    "https://github.com/amazon-science/esci-data/raw/main/"
    "shopping_queries_dataset"
)
ESCI_HF_REPO = "khanmu2003/amazon-shopping-queries-dataset"
ESCI_EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
ESCI_PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"

WANDS_GITHUB_BASE = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"

ESCI_GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
WANDS_GAINS = {"Exact": 1.0, "Partial": 0.5, "Irrelevant": 0.0}

DEFAULT_MAX_QUERIES = 80
DEFAULT_MAX_DOCS = 2000
DEFAULT_CANDIDATES_PER_QUERY = 40
PRODUCT_TEXT_MAX_CHARS = 4000
FROZEN_JSONL_ALWAYS = ("search_esci_74.jsonl",)
FROZEN_JSONL_IF_EXISTS = ("search_wands.jsonl", "search_jdsearch.jsonl")


def normalize_esci_locale(locale: str) -> str:
    loc = (locale or "us").strip().lower()
    if loc not in ESCI_LOCALES:
        raise ValueError(
            f"Unknown ESCI locale {locale!r}. Supported: {ESCI_LOCALES}. "
            "ESCI has no Italian (it) split."
        )
    return loc


def catalog_overwrite_blocked(path: Path) -> bool:
    resolved = path.resolve()
    for name in FROZEN_JSONL_ALWAYS:
        if resolved == (DATA_DIR / name).resolve():
            return True
    for name in FROZEN_JSONL_IF_EXISTS:
        frozen = (DATA_DIR / name).resolve()
        if resolved == frozen and frozen.exists() and frozen.stat().st_size > 0:
            return True
    return False


def catalog_jsonl_path(name: str, *, locale: str = "us") -> Path:
    key = name.strip().lower()
    if key == ESCI_NAME:
        loc = normalize_esci_locale(locale)
        if loc == "us":
            return ESCI_JSONL
        return DATA_DIR / f"search_esci_{loc}.jsonl"
    if key == WANDS_NAME:
        return WANDS_JSONL
    if key == JDSEARCH_NAME:
        return JDSEARCH_JSONL
    raise ValueError(f"Unknown catalog {name!r}. Supported: {SUPPORTED_CATALOGS}")


def prepare_catalog(
    name: str,
    *,
    out_path: Path | None = None,
    force: bool = False,
    max_queries: int = DEFAULT_MAX_QUERIES,
    max_docs: int = DEFAULT_MAX_DOCS,
    candidates_per_query: int = DEFAULT_CANDIDATES_PER_QUERY,
    locale: str = "us",
    split: str = "test",
    holdout_qids: set[str] | None = None,
) -> Path:
    key = name.strip().lower()
    dest = out_path or catalog_jsonl_path(key, locale=locale)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    if key == ESCI_NAME:
        rows = prepare_esci_rows(
            max_queries=max_queries,
            max_docs=max_docs,
            candidates_per_query=candidates_per_query,
            locale=locale,
            split=split,
            force=force,
            holdout_qids=holdout_qids,
        )
    elif key == WANDS_NAME:
        rows = prepare_wands_rows(
            max_queries=max_queries,
            max_docs=max_docs,
            candidates_per_query=candidates_per_query,
            force=force,
        )
    elif key == JDSEARCH_NAME:
        from search.jdsearch import prepare_jdsearch_bundle

        return prepare_jdsearch_bundle(
            dest,
            max_queries=max_queries,
            max_docs=max_docs,
            candidates_per_query=candidates_per_query,
            force=force,
        )
    else:
        raise ValueError(f"Unknown catalog {name!r}. Supported: {SUPPORTED_CATALOGS}")
    return write_records(rows, dest)


def prepare_esci_rows(
    *,
    max_queries: int = DEFAULT_MAX_QUERIES,
    max_docs: int = DEFAULT_MAX_DOCS,
    candidates_per_query: int = DEFAULT_CANDIDATES_PER_QUERY,
    locale: str = "us",
    split: str = "test",
    force: bool = False,
    holdout_qids: set[str] | None = None,
) -> list[dict[str, Any]]:
    locale = normalize_esci_locale(locale)
    examples_path, products_path = download_esci(force=force)
    selected, needed_ids = _select_catalog(
        _esci_judgments(examples_path, locale=locale, split=split),
        max_queries=max_queries,
        max_docs=max_docs,
        candidates_per_query=candidates_per_query,
        slice_name=f"esci-{locale}",
        dataset=ESCI_NAME,
        gains=ESCI_GAINS,
        label_key="esci_label",
        query_text_key="query",
        holdout_qids=holdout_qids,
    )
    products = _esci_products(products_path, needed_ids, locale=locale)
    return _docs_and_queries(
        selected,
        products=products,
        doc_ids=needed_ids,
        dataset=ESCI_NAME,
        title_key="product_title",
        description_key="product_description",
        extra_fields=extra_fields_from_catalog(
            products,
            title_key="product_title",
            description_key="product_description",
        ),
    )


def prepare_wands_rows(
    *,
    max_queries: int = DEFAULT_MAX_QUERIES,
    max_docs: int = DEFAULT_MAX_DOCS,
    candidates_per_query: int = DEFAULT_CANDIDATES_PER_QUERY,
    force: bool = False,
) -> list[dict[str, Any]]:
    product_path, query_path, label_path = download_wands(force=force)
    queries = _wands_queries(query_path)
    selected, needed_ids = _select_catalog(
        _wands_judgments(label_path, queries),
        max_queries=max_queries,
        max_docs=max_docs,
        candidates_per_query=candidates_per_query,
        slice_name="wands",
        dataset=WANDS_NAME,
        gains=WANDS_GAINS,
        label_key="label",
        query_text_key="query",
        query_slice_key="query_class",
    )
    products = _wands_products(product_path, needed_ids)
    return _docs_and_queries(
        selected,
        products=products,
        doc_ids=needed_ids,
        dataset=WANDS_NAME,
        title_key="product_name",
        description_key="product_description",
        extra_fields=(
            ("Class", "product_class"),
            ("Category", "category_hierarchy"),
            ("Hierarchy", "category_hierarchy"),
            ("Features", "product_features"),
        ),
    )


def download_esci(*, force: bool = False) -> tuple[Path, Path]:
    ESCI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    examples = ESCI_CACHE_DIR / ESCI_EXAMPLES_FILE
    products = ESCI_CACHE_DIR / ESCI_PRODUCTS_FILE
    _ensure_file(
        examples,
        url=f"{ESCI_GITHUB_BASE}/{ESCI_EXAMPLES_FILE}",
        hf_filename=ESCI_EXAMPLES_FILE,
        force=force,
    )
    _ensure_file(
        products,
        url=f"{ESCI_GITHUB_BASE}/{ESCI_PRODUCTS_FILE}",
        hf_filename=ESCI_PRODUCTS_FILE,
        force=force,
    )
    return examples, products


def download_wands(*, force: bool = False) -> tuple[Path, Path, Path]:
    WANDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    product = WANDS_CACHE_DIR / "product.csv"
    query = WANDS_CACHE_DIR / "query.csv"
    label = WANDS_CACHE_DIR / "label.csv"
    for path, name in (
        (product, "product.csv"),
        (query, "query.csv"),
        (label, "label.csv"),
    ):
        _download_url(
            f"{WANDS_GITHUB_BASE}/{name}",
            path,
            force=force,
        )
    return product, query, label


def _stringify_catalog_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_stringify_catalog_value(item) for item in value]
        return " ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _stringify_catalog_value(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    return str(value).strip()


def extra_fields_from_catalog(
    products: dict[str, dict[str, Any]],
    *,
    title_key: str,
    description_key: str,
    skip: tuple[str, ...] = ("product_id",),
) -> tuple[tuple[str, str], ...]:
    keys: set[str] = set()
    for row in products.values():
        keys.update(str(key) for key in row.keys())
    extras: list[tuple[str, str]] = []
    skipped = {title_key, description_key, *skip}
    for key in sorted(keys):
        if key in skipped:
            continue
        label = key.replace("product_", "").replace("_", " ").title()
        extras.append((label, key))
    return tuple(extras)


def format_product_text(
    doc_id: str,
    *,
    title: str = "",
    description: str = "",
    extras: Iterable[tuple[str, str]] = (),
    max_chars: int = PRODUCT_TEXT_MAX_CHARS,
) -> str:
    lines = [f"{DOC_MARKER_PREFIX}{doc_id}."]
    title = (title or "").strip()
    if title:
        lines.append(f"Title: {title}")
    for label, value in extras:
        text = (value or "").strip()
        if text:
            lines.append(f"{label}: {text}")
    description = (description or "").strip()
    if description:
        lines.append(f"Description: {description}")
    body = "\n".join(lines).strip()
    if len(body) > max_chars:
        return body[: max_chars - 1].rstrip() + "…"
    return body


def _holdout_ids(holdout_qids: Iterable[str] | None) -> set[str]:
    out: set[str] = set()
    for raw in holdout_qids or []:
        qid = str(raw or "").strip()
        if not qid:
            continue
        out.add(qid)
        lower = qid.lower()
        if lower.startswith("esci-"):
            out.add(qid[5:])
        else:
            out.add(f"esci-{qid}")
    return out


def _select_catalog(
    judgments: list[dict[str, Any]],
    *,
    max_queries: int,
    max_docs: int,
    candidates_per_query: int,
    slice_name: str,
    dataset: str,
    gains: dict[str, float],
    label_key: str,
    query_text_key: str,
    query_slice_key: str | None = None,
    holdout_qids: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_meta: dict[str, dict[str, Any]] = {}
    for row in judgments:
        qid = str(row.get("query_id") or "").strip()
        pid = str(row.get("product_id") or "").strip()
        if not qid or not pid:
            continue
        by_query[qid].append(row)
        if qid not in query_meta:
            query_meta[qid] = row

    blocked = _holdout_ids(holdout_qids)
    eligible = [
        qid
        for qid in sorted(by_query, key=_sort_key)
        if qid not in blocked and f"esci-{qid}" not in blocked
    ]
    selected_qids = eligible[: max(1, max_queries)]
    needed_ids: set[str] = set()
    queries: list[dict[str, Any]] = []

    for qid in selected_qids:
        ranked = _rank_candidates(by_query[qid], gains, label_key)
        if candidates_per_query > 0:
            ranked = ranked[:candidates_per_query]
        gold_grades: dict[str, float] = {}
        gold_doc_ids: list[str] = []
        candidate_doc_ids: list[str] = []
        candidate_grades: dict[str, float] = {}
        for item in ranked:
            pid = str(item["product_id"])
            label = str(item.get(label_key) or "")
            gain = float(gains.get(label, 0.0))
            candidate_doc_ids.append(pid)
            candidate_grades[pid] = gain
            if pid not in needed_ids and len(needed_ids) >= max_docs:
                continue
            needed_ids.add(pid)
            if gain > 0:
                gold_grades[pid] = gain
                gold_doc_ids.append(pid)
        if not gold_doc_ids:
            continue
        meta = query_meta[qid]
        slice_value = slice_name
        if query_slice_key:
            slice_value = str(meta.get(query_slice_key) or slice_name)
        queries.append(
            {
                "type": "query",
                "qid": f"{dataset}-{qid}",
                "query": str(meta.get(query_text_key) or "").strip(),
                "gold_doc_ids": gold_doc_ids,
                "gold_grades": gold_grades,
                "candidate_doc_ids": candidate_doc_ids,
                "candidate_grades": candidate_grades,
                "slice": slice_value,
            }
        )
    return queries, needed_ids


def _qid_to_source_id(qid: str, dataset: str) -> str:
    text = str(qid or "").strip()
    prefix = f"{dataset}-"
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix) :]
    return text


def attach_esci_candidate_pools(
    rows: list[dict[str, Any]],
    *,
    locale: str = "us",
    split: str = "test",
    candidates_per_query: int = DEFAULT_CANDIDATES_PER_QUERY,
) -> list[dict[str, Any]]:
    from search.dataset import split_corpus

    locale = normalize_esci_locale(locale)
    docs, queries = split_corpus(rows)
    if not queries:
        return rows
    wanted = {
        _qid_to_source_id(str(query.get("qid") or ""), ESCI_NAME)
        for query in queries
    }
    wanted.discard("")
    examples_path, products_path = download_esci(force=False)
    judgments = [
        row
        for row in _esci_judgments(examples_path, locale=locale, split=split)
        if str(row.get("query_id") or "").strip() in wanted
    ]
    selected, needed_ids = _select_catalog(
        judgments,
        max_queries=max(1, len(wanted)),
        max_docs=max(len(needed_ids_from_docs(docs)) + 4000, 1),
        candidates_per_query=candidates_per_query,
        slice_name=f"esci-{locale}",
        dataset=ESCI_NAME,
        gains=ESCI_GAINS,
        label_key="esci_label",
        query_text_key="query",
    )
    by_source = {
        _qid_to_source_id(str(row.get("qid") or ""), ESCI_NAME): row
        for row in selected
    }
    existing_ids = {str(doc.get("doc_id") or "") for doc in docs}
    missing = needed_ids - existing_ids
    for query in queries:
        source = _qid_to_source_id(str(query.get("qid") or ""), ESCI_NAME)
        attached = by_source.get(source)
        if not attached:
            query.setdefault("candidate_doc_ids", list(query.get("gold_doc_ids") or []))
            query.setdefault(
                "candidate_grades",
                dict(query.get("gold_grades") or {}),
            )
            continue
        query["candidate_doc_ids"] = list(attached.get("candidate_doc_ids") or [])
        query["candidate_grades"] = dict(attached.get("candidate_grades") or {})
        missing.update(set(query["candidate_doc_ids"]) - existing_ids)
    if not missing:
        return docs + queries
    products = _esci_products(products_path, missing, locale=locale)
    extra = _docs_and_queries(
        [],
        products=products,
        doc_ids=missing,
        dataset=ESCI_NAME,
        title_key="product_title",
        description_key="product_description",
        extra_fields=extra_fields_from_catalog(
            products,
            title_key="product_title",
            description_key="product_description",
        ),
    )
    extra_docs = [row for row in extra if row.get("type") == "doc"]
    return docs + extra_docs + queries


def needed_ids_from_docs(docs: list[dict[str, Any]]) -> set[str]:
    return {str(doc.get("doc_id") or "") for doc in docs if doc.get("doc_id")}


def _docs_and_queries(
    queries: list[dict[str, Any]],
    *,
    products: dict[str, dict[str, Any]],
    doc_ids: set[str],
    dataset: str,
    title_key: str,
    description_key: str,
    extra_fields: tuple[tuple[str, str], ...] = (),
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    field_aliases = {
        "brand": "brand",
        "color": "color",
        "class": "class",
        "category": "class",
        "hierarchy": "hierarchy",
        "category hierarchy": "hierarchy",
        "features": "features",
        "locale": "locale",
        "description": "description",
        "price": "price",
        "rating": "rating",
    }
    for pid in sorted(doc_ids, key=_sort_key):
        catalog = products.get(pid) or {}
        extras = [
            (label, _stringify_catalog_value(catalog.get(field)))
            for label, field in extra_fields
        ]
        title = _stringify_catalog_value(catalog.get(title_key)) or pid
        description = _stringify_catalog_value(catalog.get(description_key))
        row: dict[str, Any] = {
            "type": "doc",
            "doc_id": pid,
            "text": format_product_text(
                pid,
                title=title,
                description=description,
                extras=extras,
            ),
            "title": title,
            "description": description,
            "ingest": "chunks",
            "dataset": dataset,
        }
        for label, field in extra_fields:
            mapped = field_aliases.get(label.strip().lower())
            if not mapped:
                continue
            value = _stringify_catalog_value(catalog.get(field))
            if value and mapped not in row:
                row[mapped] = value
        docs.append(row)
    queries = [q for q in queries if q.get("query") and q.get("gold_doc_ids")]
    queries.sort(key=lambda row: str(row["qid"]))
    return docs + queries


def _rank_candidates(
    rows: list[dict[str, Any]],
    gains: dict[str, float],
    label_key: str,
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, str]:
        label = str(row.get(label_key) or "")
        return (-float(gains.get(label, 0.0)), str(row.get("product_id") or ""))

    return sorted(rows, key=key)


def _esci_judgments(
    path: Path,
    *,
    locale: str,
    split: str,
) -> list[dict[str, Any]]:
    wanted_locale = locale.strip().lower()
    wanted_split = split.strip().lower()
    rows: list[dict[str, Any]] = []
    for item in _iter_parquet(path):
        if int(item.get("small_version") or 0) != 1:
            continue
        if str(item.get("product_locale") or "").strip().lower() != wanted_locale:
            continue
        if str(item.get("split") or "").strip().lower() != wanted_split:
            continue
        rows.append(
            {
                "query_id": item.get("query_id"),
                "query": item.get("query"),
                "product_id": item.get("product_id"),
                "esci_label": item.get("esci_label"),
            }
        )
    return rows


def _esci_products(
    path: Path,
    needed_ids: set[str],
    *,
    locale: str = "us",
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    if not needed_ids:
        return catalog
    wanted_locale = locale.strip().lower()
    for item in _iter_parquet(path):
        pid = str(item.get("product_id") or "").strip()
        if pid not in needed_ids or pid in catalog:
            continue
        item_locale = str(item.get("product_locale") or "").strip().lower()
        if item_locale and item_locale != wanted_locale:
            continue
        catalog[pid] = item
        if len(catalog) >= len(needed_ids):
            break
    return catalog


def _wands_queries(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in _iter_tsv(path):
        qid = str(row.get("query_id") or "").strip()
        if not qid:
            continue
        out[qid] = {
            "query_id": qid,
            "query": str(row.get("query") or "").strip(),
            "query_class": str(row.get("query_class") or "wands").strip()
            or "wands",
        }
    return out


def _wands_products(path: Path, needed_ids: set[str]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    if not needed_ids:
        return catalog
    for row in _iter_tsv(path):
        pid = str(row.get("product_id") or "").strip()
        if pid not in needed_ids:
            continue
        catalog[pid] = row
        if len(catalog) >= len(needed_ids):
            break
    return catalog


def _wands_judgments(
    path: Path,
    queries: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _iter_tsv(path):
        qid = str(row.get("query_id") or "").strip()
        pid = str(row.get("product_id") or "").strip()
        if not qid or not pid:
            continue
        meta = queries.get(qid) or {}
        rows.append(
            {
                "query_id": qid,
                "query": meta.get("query") or "",
                "query_class": meta.get("query_class") or "wands",
                "product_id": pid,
                "label": str(row.get("label") or "").strip(),
            }
        )
    return rows


def _iter_tsv(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            yield {str(k): ("" if v is None else str(v)) for k, v in row.items()}


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read ESCI parquet files. "
            "Install benchmarks/requirements.txt."
        ) from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            yield row


def _ensure_file(
    dest: Path,
    *,
    url: str,
    hf_filename: str,
    force: bool,
) -> Path:
    if dest.exists() and dest.stat().st_size > 64 and not force:
        if not _is_git_lfs_pointer(dest):
            return dest
    try:
        _download_url(url, dest, force=True)
        if dest.exists() and dest.stat().st_size > 64 and not _is_git_lfs_pointer(dest):
            return dest
    except Exception:
        pass
    return _download_hf(hf_filename, dest, force=True)


def _download_hf(filename: str, dest: Path, *, force: bool) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required to download ESCI. "
            "Install benchmarks/requirements.txt."
        ) from exc
    cached = hf_hub_download(
        repo_id=ESCI_HF_REPO,
        filename=filename,
        repo_type="dataset",
        force_download=force,
    )
    src = Path(cached)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        dest.write_bytes(src.read_bytes())
    return dest


def _download_url(url: str, dest: Path, *, force: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:80]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _sort_key(value: str) -> tuple[int, str | int]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)
