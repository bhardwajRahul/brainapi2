from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from search.catalog import (
    ESCI_CACHE_DIR,
    ESCI_EXAMPLES_FILE,
    ESCI_GAINS,
    ESCI_PRODUCTS_FILE,
    PRODUCT_TEXT_MAX_CHARS,
    _esci_products,
    _iter_parquet,
    _stringify_catalog_value,
)
from search.dataset import load_records, split_corpus

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REDDY_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
CLASS_ORDER = ("E", "S", "C", "I")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}


def held_out_query_ids(jsonl_path: Path) -> set[str]:
    _, queries = split_corpus(load_records(jsonl_path))
    out: set[str] = set()
    for row in queries:
        qid = str(row.get("qid") or "")
        if qid.lower().startswith("esci-"):
            out.add(qid[5:])
        out.add(qid)
    return out


def product_passage(catalog: dict[str, Any], *, fields: str) -> str:
    title = _stringify_catalog_value(catalog.get("product_title")) or ""
    if fields == "title":
        return title
    parts: list[str] = []
    if title:
        parts.append(title)
    brand = _stringify_catalog_value(catalog.get("product_brand"))
    if brand:
        parts.append(f"Brand: {brand}")
    color = _stringify_catalog_value(catalog.get("product_color"))
    if color:
        parts.append(f"Color: {color}")
    bullets = _stringify_catalog_value(catalog.get("product_bullet_point"))
    if bullets:
        parts.append(f"Bullets: {bullets}")
    description = _stringify_catalog_value(catalog.get("product_description"))
    if description:
        parts.append(f"Description: {description}")
    body = "\n".join(parts).strip()
    if len(body) > PRODUCT_TEXT_MAX_CHARS:
        return body[: PRODUCT_TEXT_MAX_CHARS - 1].rstrip() + "…"
    return body


def load_train_rows(
    *,
    holdout: set[str],
    locale: str = "us",
    max_pairs: int = 80000,
    seed: int = 7,
) -> list[dict[str, Any]]:
    examples_path = ESCI_CACHE_DIR / ESCI_EXAMPLES_FILE
    wanted_locale = locale.strip().lower()
    rows: list[dict[str, Any]] = []
    for item in _iter_parquet(examples_path):
        if int(item.get("small_version") or 0) != 1:
            continue
        if str(item.get("product_locale") or "").strip().lower() != wanted_locale:
            continue
        if str(item.get("split") or "").strip().lower() != "train":
            continue
        qid = str(item.get("query_id") or "").strip()
        if not qid or qid in holdout:
            continue
        pid = str(item.get("product_id") or "").strip()
        query = str(item.get("query") or "").strip()
        if not pid or not query:
            continue
        label = str(item.get("esci_label") or "").strip().upper()
        if label not in ESCI_GAINS:
            continue
        rows.append(
            {
                "query": query,
                "product_id": pid,
                "label": label,
                "gain": float(ESCI_GAINS[label]),
                "binary": 1.0 if label == "E" else 0.0,
                "class_index": CLASS_TO_INDEX[label],
            }
        )
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: max(1, max_pairs)]


def load_train_pairs(
    *,
    holdout: set[str],
    locale: str = "us",
    max_pairs: int = 30000,
    seed: int = 7,
    fields: str = "title",
    label_mode: str = "binary",
) -> list[tuple[str, str, float]]:
    rows = load_train_rows(
        holdout=holdout, locale=locale, max_pairs=max_pairs, seed=seed
    )
    products_path = ESCI_CACHE_DIR / ESCI_PRODUCTS_FILE
    needed = {str(row["product_id"]) for row in rows}
    products = _esci_products(products_path, needed, locale=locale)
    pairs: list[tuple[str, str, float]] = []
    for row in rows:
        catalog = products.get(str(row["product_id"])) or {}
        passage = product_passage(catalog, fields=fields) or str(row["product_id"])
        if label_mode == "binary":
            target = float(row["binary"])
        elif label_mode == "graded":
            target = float(row["gain"])
        else:
            target = float(row["class_index"])
        pairs.append((row["query"], passage, target))
    return pairs


def finetune(
    *,
    jsonl_path: Path,
    out_dir: Path,
    max_pairs: int = 80000,
    epochs: int = 1,
    batch_size: int = 32,
    label_mode: str = "graded",
    fields: str = "catalog",
    base: str = DEFAULT_MODEL,
    lr: float = 7e-6,
    max_length: int = 256,
) -> dict[str, Any]:
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    if label_mode not in {"binary", "graded", "multiclass"}:
        raise ValueError(f"Unknown label_mode {label_mode!r}")
    if fields not in {"title", "catalog"}:
        raise ValueError(f"Unknown fields {fields!r}")
    holdout = held_out_query_ids(jsonl_path)
    pairs = load_train_pairs(
        holdout=holdout,
        max_pairs=max_pairs,
        fields=fields,
        label_mode=label_mode,
    )
    if label_mode == "multiclass":
        examples = [
            InputExample(texts=[query, passage], label=int(target))
            for query, passage, target in pairs
        ]
        model = CrossEncoder(base, num_labels=len(CLASS_ORDER), max_length=max_length)
    else:
        examples = [
            InputExample(texts=[query, passage], label=float(target))
            for query, passage, target in pairs
        ]
        model = CrossEncoder(
            base,
            num_labels=1,
            max_length=max_length,
            default_activation_function=None,
        )
    n_pos = sum(
        1
        for _, _, target in pairs
        if (
            int(target) == CLASS_TO_INDEX["E"]
            if label_mode == "multiclass"
            else float(target) >= 1.0
        )
    )
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    warmup = max(10, len(loader) // 10)
    model.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    counts: dict[str, int] = {}
    if label_mode == "multiclass":
        for _, _, target in pairs:
            label = CLASS_ORDER[int(target)]
            counts[label] = counts.get(label, 0) + 1
    meta = {
        "n_pairs": len(pairs),
        "n_exact": n_pos,
        "n_holdout_qids": len(holdout),
        "epochs": epochs,
        "model": str(out_dir),
        "base": base,
        "label_mode": label_mode,
        "fields": fields,
        "lr": lr,
        "max_length": max_length,
        "class_counts": counts,
        "class_order": list(CLASS_ORDER) if label_mode == "multiclass" else None,
    }
    (out_dir / "finetune_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finetune_esci_ce")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/search_esci_74.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/models/esci-minilm-graded"),
    )
    parser.add_argument("--max-pairs", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--label-mode",
        choices=["binary", "graded", "multiclass"],
        default="graded",
    )
    parser.add_argument("--fields", choices=["title", "catalog"], default="catalog")
    parser.add_argument("--base", default=DEFAULT_MODEL)
    parser.add_argument("--lr", type=float, default=7e-6)
    parser.add_argument("--max-length", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    meta = finetune(
        jsonl_path=args.dataset,
        out_dir=args.out,
        max_pairs=args.max_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label_mode=args.label_mode,
        fields=args.fields,
        base=args.base,
        lr=args.lr,
        max_length=args.max_length,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
