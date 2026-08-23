from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from search.config import Settings, validate_brain_id
from search.dataset import load_records, split_corpus
from search.evaluate import candidate_pool_grades, candidate_pool_ids, ensure_run_dir
from search.finetune_esci_4class import (
    CLASS_GAINS,
    DEFAULT_MODEL,
    DEFAULT_OUT,
    pick_device,
    rank_doc_ids,
    weighted_scores,
)
from search.metrics import aggregate_query_metrics, mrr, ndcg_at_k, recall_at_k
from search.report import print_report_table, write_report

PredictFn = Callable[[list[tuple[str, str]]], Any]


def doc_texts(docs: list[dict[str, Any]]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        if not doc_id:
            continue
        texts[doc_id] = str(doc.get("text") or doc.get("title") or "")
    return texts


def score_ranked_pool(
    ranked: list[str],
    query: dict[str, Any],
    *,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    gold = {
        str(doc_id)
        for doc_id, gain in candidate_pool_grades(query).items()
        if float(gain) > 0
    }
    if not gold:
        gold = {str(item) for item in (query.get("gold_doc_ids") or [])}
    grades = candidate_pool_grades(query)
    metrics = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in ks}
    metrics["ndcg@10"] = ndcg_at_k(ranked, gold, 10, grades=grades)
    metrics["ndcg@20"] = ndcg_at_k(ranked, gold, 20, grades=grades)
    metrics["ndcg"] = ndcg_at_k(ranked, gold, max(len(ranked), 1), grades=grades)
    metrics["mrr"] = mrr(ranked, gold)
    return metrics


def load_4class_predict(model_dir: Path, *, max_length: int = 192) -> tuple[PredictFn, str]:
    from sentence_transformers import CrossEncoder

    device = pick_device()
    model = CrossEncoder(
        str(model_dir),
        num_labels=4,
        max_length=max_length,
        device=device,
        default_activation_function=None,
    )

    def predict(pairs: list[tuple[str, str]]):
        if not pairs:
            return []
        return model.predict(
            [list(pair) for pair in pairs],
            batch_size=32,
            apply_softmax=True,
            show_progress_bar=False,
        )

    return predict, str(model_dir)


def run_4class_on_pool(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    predict: PredictFn,
    model_name: str,
    ks: tuple[int, ...] = (5, 10, 20),
    brain_id: str | None = None,
) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    texts = doc_texts(docs)
    per_query: list[dict[str, Any]] = []
    missing_text = 0
    for query in queries:
        pool = candidate_pool_ids(query)
        pairs: list[tuple[str, str]] = []
        for doc_id in pool:
            text = texts.get(doc_id) or ""
            if not text:
                missing_text += 1
            pairs.append((str(query.get("query") or ""), text))
        probs = predict(pairs) if pairs else []
        scores = weighted_scores(probs, CLASS_GAINS) if len(pairs) else []
        if len(scores) < len(pool):
            scores = list(scores) + [0.0] * (len(pool) - len(scores))
        ranked = rank_doc_ids(pool, scores)
        metrics = score_ranked_pool(ranked, query, ks=ks)
        per_query.append(
            {
                "qid": query.get("qid"),
                "query": query.get("query"),
                "slice": query.get("slice") or "unspecified",
                "gold_doc_ids": list(query.get("gold_doc_ids") or []),
                "candidate_doc_ids": pool,
                "pool_size": len(pool),
                "pool_coverage": 1.0 if pool else None,
                "missing_from_brain": [],
                "hit_ids": ranked,
                "metrics": metrics,
                "retrieve_ms": 0.0,
                "embed_ms": None,
                "client_wall_ms": 0.0,
                "n_hits": len(ranked),
            }
        )
    metrics = aggregate_query_metrics(per_query, ks=ks) if per_query else {}
    return {
        "status": "ok" if per_query else "failed",
        "brain_id": brain_id,
        "dataset": dataset_name,
        "fusion": "none",
        "rerank": "4class-weighted-ce",
        "channels": ["rank-pool-4class"],
        "expand": "none",
        "k": max((len(row.get("hit_ids") or []) for row in per_query), default=0),
        "ks": list(ks),
        "n_docs": len(docs),
        "n_queries": len(per_query),
        "n_docs_mapped": len(texts),
        "skip_enrichment": True,
        "ingest_graph": False,
        "skip_ingest": True,
        "rank_pool": True,
        "rank_pool_ce": True,
        "ce_model": model_name,
        "ce_missing_text": missing_text,
        "ingest": {
            "status": "completed",
            "n_docs": len(docs),
            "tasks": [],
            "reused": True,
        },
        "graph_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "interaction_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "event_probe": None,
        "search_error": None,
        "protocol": (
            "ranking-in-pool 4-class CE over labeled candidates including I=0; "
            "score=1.0*P(E)+0.1*P(S)+0.01*P(C); not MiniLM Exact→1; "
            "n=74 is not Reddy public test n≈4477"
        ),
        "metrics": metrics,
        "queries": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ranking-in-pool eval with a 4-class weighted cross-encoder"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run", default="search-esci-74-ce-pool-4class")
    parser.add_argument("--brain", default="searchbenchesci74")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args(argv)
    if not args.dataset.exists():
        print(f"Missing {args.dataset}")
        return 1
    if not args.model.exists():
        print(f"Missing model {args.model}")
        return 1
    settings = Settings.load(args.env_file)
    settings.brain_id = validate_brain_id(args.brain)
    rows = load_records(args.dataset)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    predict, model_name = load_4class_predict(args.model, max_length=args.max_length)
    result = run_4class_on_pool(
        rows,
        dataset_name=args.dataset.name,
        predict=predict,
        model_name=model_name,
        brain_id=settings.brain_id,
    )
    report = write_report(run_dir, result)
    print_report_table(report)
    print(f"Wrote {run_dir / 'report.json'}")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "ce_model": result.get("ce_model"),
                "ndcg@20": report.get("ndcg@20"),
                "ndcg": report.get("ndcg"),
                "n_queries": report.get("n_queries"),
                "missing_text": result.get("ce_missing_text"),
                "base_default": DEFAULT_MODEL,
            },
            indent=2,
        )
    )
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
