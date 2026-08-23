from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from search.catalog import ESCI_CACHE_DIR, ESCI_PRODUCTS_FILE, _esci_products
from search.dataset import load_records, split_corpus
from search.finetune_esci_ce import held_out_query_ids, load_train_rows, product_passage
from search.local_dense import DEFAULT_BASE, hard_negative_texts_from_eval

DEFAULT_OUT = Path("data/models/esci-minilm-dense-ance")


def build_triples(
    *,
    jsonl_path: Path,
    eval_path: Path | None,
    max_pairs: int,
    fields: str,
    seed: int,
) -> tuple[list[tuple[str, str, str]], dict[str, Any]]:
    holdout = held_out_query_ids(jsonl_path)
    rows = load_train_rows(holdout=holdout, max_pairs=max_pairs, seed=seed)
    needed = {str(row["product_id"]) for row in rows}
    products = _esci_products(ESCI_CACHE_DIR / ESCI_PRODUCTS_FILE, needed, locale="us")
    passages: dict[str, str] = {}
    for pid in needed:
        catalog = products.get(pid) or {}
        passages[pid] = product_passage(catalog, fields=fields) or pid
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row["query"]), []).append(row)
    docs, queries = split_corpus(load_records(jsonl_path))
    bank: list[str] = []
    if eval_path and eval_path.exists():
        eval_result = json.loads(eval_path.read_text(encoding="utf-8"))
        from search.rank_pool import doc_texts

        bank = hard_negative_texts_from_eval(eval_result, doc_texts(docs), queries)
    rng = random.Random(seed)
    all_passages = [text for text in passages.values() if text]
    triples: list[tuple[str, str, str]] = []
    n_from_bank = 0
    for query, group in by_query.items():
        positives = [
            passages.get(str(row["product_id"])) or ""
            for row in group
            if str(row.get("label") or "").upper() == "E"
        ]
        positives = [text for text in positives if text]
        if not positives:
            continue
        pos = positives[0]
        neg = ""
        if bank:
            neg = rng.choice(bank)
            n_from_bank += 1
        if not neg or neg == pos:
            candidates = [text for text in all_passages if text != pos]
            if candidates:
                neg = rng.choice(candidates)
        if not neg or neg == pos:
            continue
        triples.append((query, pos, neg))
    meta = {
        "n_triples": len(triples),
        "n_holdout_qids": len(holdout),
        "n_hardneg_from_k50_bank": n_from_bank,
        "n_bank": len(bank),
        "fields": fields,
        "max_pairs": max_pairs,
    }
    return triples, meta


def finetune_dense(
    *,
    jsonl_path: Path,
    out_dir: Path,
    eval_path: Path | None = None,
    max_pairs: int = 20000,
    epochs: int = 1,
    batch_size: int = 32,
    fields: str = "catalog",
    base: str = DEFAULT_BASE,
    lr: float = 2e-5,
    max_length: int = 256,
    seed: int = 7,
) -> dict[str, Any]:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    triples, counts = build_triples(
        jsonl_path=jsonl_path,
        eval_path=eval_path,
        max_pairs=max_pairs,
        fields=fields,
        seed=seed,
    )
    if not triples:
        raise SystemExit("No training triples after holdout (check ESCI cache).")
    examples = [
        InputExample(texts=[query, positive, negative])
        for query, positive, negative in triples
    ]
    model = SentenceTransformer(base)
    try:
        model.max_seq_length = max_length
    except Exception:
        pass
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = max(10, len(loader) // 10)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    meta = {
        **counts,
        "epochs": epochs,
        "model": str(out_dir),
        "base": base,
        "lr": lr,
        "max_length": max_length,
        "batch_size": batch_size,
        "holdout_leak": False,
    }
    (out_dir / "finetune_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finetune_esci_dense")
    parser.add_argument("--dataset", type=Path, default=Path("data/search_esci_74.jsonl"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--from-eval",
        type=Path,
        default=Path("runs/search-esci-74-passages-k50/eval.json"),
    )
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fields", choices=["title", "catalog"], default="catalog")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    meta = finetune_dense(
        jsonl_path=args.dataset,
        out_dir=args.out,
        eval_path=args.from_eval,
        max_pairs=int(args.max_pairs),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        fields=str(args.fields),
        base=str(args.base),
        lr=float(args.lr),
        max_length=int(args.max_length),
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
