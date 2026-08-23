from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from search.catalog import (
    ESCI_CACHE_DIR,
    ESCI_EXAMPLES_FILE,
    ESCI_PRODUCTS_FILE,
    extra_fields_from_catalog,
    format_product_text,
    _esci_products,
    _iter_parquet,
    _stringify_catalog_value,
)
from search.dataset import load_records, split_corpus

LABELS = ("E", "S", "C", "I")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
CLASS_GAINS = (1.0, 0.1, 0.01, 0.0)
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_OUT = Path("data/models/esci-minilm-l12-4class")


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def held_out_query_ids(jsonl_path: Path) -> set[str]:
    _, queries = split_corpus(load_records(jsonl_path))
    out: set[str] = set()
    for row in queries:
        qid = str(row.get("qid") or "")
        if qid.lower().startswith("esci-"):
            out.add(qid[5:])
        out.add(qid)
    return out


def product_passage(catalog: dict[str, Any], extra_fields: Sequence[tuple[str, str]]) -> str:
    pid = str(catalog.get("product_id") or "unknown")
    title = _stringify_catalog_value(catalog.get("product_title")) or pid
    extras = [
        (label, _stringify_catalog_value(catalog.get(field)))
        for label, field in extra_fields
    ]
    return format_product_text(
        pid,
        title=title,
        description=_stringify_catalog_value(catalog.get("product_description")),
        extras=extras,
    )


def weighted_scores(
    class_probs: Sequence[Sequence[float]] | Sequence[float],
    gains: Sequence[float] = CLASS_GAINS,
) -> list[float]:
    rows = list(class_probs)
    if not rows:
        return []
    first = rows[0]
    if isinstance(first, (int, float)):
        rows = [rows]
    gain = [float(value) for value in gains]
    scores: list[float] = []
    for row in rows:
        total = 0.0
        for index, value in enumerate(row):
            if index >= len(gain):
                break
            total += float(value) * gain[index]
        scores.append(total)
    return scores


def rank_doc_ids(doc_ids: Sequence[str], scores: Sequence[float]) -> list[str]:
    indexed = list(range(len(doc_ids)))
    indexed.sort(key=lambda index: (-float(scores[index]), str(doc_ids[index])))
    return [str(doc_ids[index]) for index in indexed]


def persist_tokenizer_max_length(model: Any, out_dir: Path) -> None:
    max_length = getattr(model, "max_length", None)
    tokenizer = getattr(model, "tokenizer", None)
    if not max_length or tokenizer is None:
        return
    tokenizer.model_max_length = int(max_length)
    tokenizer.save_pretrained(str(out_dir))


def class_weights(counts: Sequence[int]) -> list[float]:
    values = [max(1, int(count)) for count in counts]
    total = float(sum(values))
    n_labels = float(len(values))
    return [total / (n_labels * float(count)) for count in values]


def load_train_rows(
    *,
    holdout: set[str],
    locale: str = "us",
    max_pairs: int = 180000,
    seed: int = 11,
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
        label = str(item.get("esci_label") or "").strip().upper()
        if not pid or not query or label not in LABEL_TO_ID:
            continue
        rows.append(
            {
                "query": query,
                "product_id": pid,
                "label_id": LABEL_TO_ID[label],
                "label": label,
            }
        )
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: max(1, max_pairs)]


def load_retrieved_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            label = str(item.get("label") or "").strip().upper()
            query = str(item.get("query") or "").strip()
            pid = str(item.get("product_id") or "").strip()
            if not query or not pid or label not in LABEL_TO_ID:
                continue
            row = {
                "query": query,
                "product_id": pid,
                "label": label,
                "label_id": LABEL_TO_ID[label],
            }
            passage = str(item.get("passage") or "").strip()
            if passage:
                row["passage"] = passage
            rows.append(row)
    return rows


def attach_passages(
    rows: list[dict[str, Any]],
    *,
    locale: str = "us",
) -> list[tuple[str, str, int]]:
    products_path = ESCI_CACHE_DIR / ESCI_PRODUCTS_FILE
    needed = {str(row["product_id"]) for row in rows}
    products = _esci_products(products_path, needed, locale=locale)
    extra_fields = extra_fields_from_catalog(
        products,
        title_key="product_title",
        description_key="product_description",
    )
    pairs: list[tuple[str, str, int]] = []
    for row in rows:
        catalog = products.get(str(row["product_id"])) or {"product_id": row["product_id"]}
        passage = product_passage(catalog, extra_fields)
        pairs.append((str(row["query"]), passage, int(row["label_id"])))
    return pairs


def train_loop(
    model,
    loader,
    *,
    epochs: int,
    loss_fct,
    warmup_steps: int,
    lr: float,
    out_dir: Path,
    ckpt_every: int = 200,
) -> int:
    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    loader.collate_fn = model.smart_batching_collate
    model.model.to(model._target_device)
    optimizer = AdamW(model.model.parameters(), lr=lr)
    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    model.model.train()
    for _epoch in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model.model(**features).logits
            loss = loss_fct(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % 50 == 0 or global_step == total_steps:
                print(
                    f"step {global_step}/{total_steps} loss={float(loss.detach().cpu()):.4f}",
                    flush=True,
                )
            if global_step % ckpt_every == 0:
                model.save(str(out_dir))
                persist_tokenizer_max_length(model, out_dir)
                (out_dir / "train_step.json").write_text(
                    json.dumps(
                        {
                            "global_step": global_step,
                            "loss": float(loss.detach().cpu()),
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"checkpoint {out_dir} step={global_step}", flush=True)
    model.save(str(out_dir))
    persist_tokenizer_max_length(model, out_dir)
    return global_step


def finetune(
    *,
    jsonl_path: Path,
    out_dir: Path,
    max_pairs: int = 80000,
    epochs: int = 1,
    batch_size: int = 32,
    max_length: int = 192,
    base_model: str = DEFAULT_MODEL,
    seed: int = 11,
    ckpt_every: int = 200,
    use_class_weights: bool = False,
    lists_path: Path | None = None,
    lists_source: str | None = None,
) -> dict[str, Any]:
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    holdout = held_out_query_ids(jsonl_path)
    source = "pool"
    if lists_path is not None:
        source = str(lists_source or "").strip() or "retrieved-bm25"
        rows = load_retrieved_rows(lists_path)
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[: max(1, max_pairs)]
        if rows and all(str(row.get("passage") or "").strip() for row in rows):
            pairs = [
                (str(row["query"]), str(row["passage"]), int(row["label_id"]))
                for row in rows
            ]
        else:
            pairs = attach_passages(rows)
    else:
        rows = load_train_rows(holdout=holdout, max_pairs=max_pairs, seed=seed)
        pairs = attach_passages(rows)
    label_counts = Counter(label_id for _, _, label_id in pairs)
    counts = [int(label_counts.get(index, 0)) for index in range(len(LABELS))]
    weights = class_weights(counts) if use_class_weights else [1.0] * len(LABELS)
    examples = [
        InputExample(texts=[query, passage], label=label_id)
        for query, passage, label_id in pairs
    ]
    device = pick_device()
    model = CrossEncoder(
        base_model,
        num_labels=len(LABELS),
        max_length=max_length,
        device=device,
        automodel_args={"ignore_mismatched_sizes": True},
        default_activation_function=torch.nn.Identity(),
    )
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    warmup = max(10, len(loader) // 10)
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=model._target_device),
        label_smoothing=0.1,
    )
    steps = train_loop(
        model,
        loader,
        epochs=epochs,
        loss_fct=loss_fct,
        warmup_steps=warmup,
        lr=2e-5,
        out_dir=out_dir,
        ckpt_every=ckpt_every,
    )
    meta = {
        "n_pairs": len(pairs),
        "n_steps": steps,
        "label_counts": {LABELS[index]: counts[index] for index in range(len(LABELS))},
        "class_weights": {LABELS[index]: weights[index] for index in range(len(LABELS))},
        "class_gains": {LABELS[index]: CLASS_GAINS[index] for index in range(len(LABELS))},
        "n_holdout_qids": len(holdout),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "model": str(out_dir),
        "base": base_model,
        "device": device,
        "loss": "4-class-ce-label-smoothing-0.1",
        "use_class_weights": use_class_weights,
        "score": "1.0*P(E)+0.1*P(S)+0.01*P(C)+0.0*P(I)",
        "source": source,
        "lists_path": str(lists_path) if lists_path is not None else None,
    }
    payload = json.dumps(meta, indent=2) + "\n"
    (out_dir / "train_meta.json").write_text(payload, encoding="utf-8")
    (out_dir / "finetune_meta.json").write_text(payload, encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune a 4-class ESCI cross-encoder (not Exact→1 MiniLM-L-6)"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-pairs", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--base", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--from-lists", type=Path, default=None)
    parser.add_argument("--lists-source", default="")
    args = parser.parse_args(argv)
    meta = finetune(
        jsonl_path=args.dataset,
        out_dir=args.out,
        max_pairs=args.max_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        base_model=args.base,
        seed=args.seed,
        ckpt_every=args.ckpt_every,
        use_class_weights=bool(args.class_weights),
        lists_path=args.from_lists,
        lists_source=(
            str(args.lists_source).strip() or "retrieved-bm25"
            if args.from_lists is not None
            else None
        ),
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
