from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from search.dataset import load_records, split_corpus, write_records
from search.evaluate import ensure_run_dir
from search.metrics import aggregate_query_metrics
from search.rank_pool import score_ranked_pool

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


def field_value(text: str, name: str) -> str:
    prefix = f"{name}:"
    lines = []
    capture = False
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if ":" in stripped and not stripped.lower().startswith("docid "):
            label = stripped.split(":", 1)[0].strip().lower()
            capture = label == name.lower()
            if capture:
                lines.append(stripped.split(":", 1)[1].strip())
            continue
        if capture:
            lines.append(stripped)
    return " ".join(part for part in lines if part)


def title_text(text: str) -> str:
    title = field_value(text, "Title")
    return title or (text or "")


def bm25_idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


class Bm25Index:
    def __init__(self, docs: list[tuple[str, list[str]]], *, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = [doc_id for doc_id, _ in docs]
        self.tfs: list[Counter[str]] = [Counter(tokens) for _, tokens in docs]
        self.dl = [max(1, len(tokens)) for _, tokens in docs]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 1.0
        df: Counter[str] = Counter()
        postings: dict[str, list[int]] = defaultdict(list)
        for index, tokens in enumerate(self.tfs):
            df.update(tokens.keys())
            for term in tokens:
                postings[term].append(index)
        n = max(1, len(self.doc_ids))
        self.idf = {term: bm25_idf(n, count) for term, count in df.items()}
        self.postings = postings

    def scores(self, query_tokens: list[str]) -> dict[str, float]:
        qtf = Counter(query_tokens)
        out: dict[str, float] = defaultdict(float)
        for term, qf in qtf.items():
            idf = self.idf.get(term, 0.0)
            if not idf:
                continue
            for index in self.postings.get(term, ()):
                tf = self.tfs[index].get(term, 0)
                if tf <= 0:
                    continue
                dl = self.dl[index]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                out[self.doc_ids[index]] += float(qf) * idf * (tf * (self.k1 + 1.0)) / denom
        return out


def rank_docs(scores: dict[str, float], doc_ids: Iterable[str]) -> list[str]:
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    seen = {doc_id for doc_id, _ in ranked}
    ordered = [doc_id for doc_id, _ in ranked]
    for doc_id in doc_ids:
        if doc_id not in seen:
            ordered.append(doc_id)
    return ordered


def rm3_terms(
    query_tokens: list[str],
    ranked: list[str],
    token_map: dict[str, list[str]],
    *,
    top_docs: int = 8,
    extra_terms: int = 8,
) -> list[str]:
    counts: Counter[str] = Counter()
    for doc_id in ranked[:top_docs]:
        counts.update(token_map.get(doc_id) or [])
    query_set = set(query_tokens)
    added: list[str] = []
    for term, _ in counts.most_common():
        if term in STOP or term in query_set or term.isdigit():
            continue
        added.append(term)
        if len(added) >= extra_terms:
            break
    return added


def expand_query(query: str, extra: list[str]) -> str:
    if not extra:
        return query
    return f"{query} {' '.join(extra)}".strip()


def evaluate_variant(
    queries: list[dict[str, Any]],
    ranked_by_qid: dict[str, list[str]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    for query in queries:
        qid = str(query.get("qid") or "")
        ranked = ranked_by_qid.get(qid) or []
        metrics = score_ranked_pool(ranked, query)
        per_query.append(
            {
                "qid": qid,
                "query": query.get("query"),
                "hit_ids": ranked[:20],
                "metrics": metrics,
                "n_hits": len(ranked),
                "retrieve_ms": 0.0,
                "embed_ms": None,
                "client_wall_ms": 0.0,
            }
        )
    metrics = aggregate_query_metrics(per_query, ks=(5, 10, 20)) if per_query else {}
    return metrics, per_query


def _splade_rank(
    docs: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    k: int = 20,
) -> dict[str, list[str]]:
    plugin_dir = Path(__file__).resolve().parents[2] / "plugins" / "search-splade"
    import sys

    if str(plugin_dir) not in sys.path:
        sys.path.insert(0, str(plugin_dir))
    from index import index_chunks, reset, retrieve

    brain = "harness-local-splade"
    reset(brain)
    chunks = []
    doc_ids: list[str] = []
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        if not doc_id:
            continue
        doc_ids.append(doc_id)
        chunks.append(
            {"id": doc_id, "text": str(doc.get("text") or doc.get("title") or "")}
        )
    index_chunks(brain, chunks)
    ranked_by_qid: dict[str, list[str]] = {}
    for query in queries:
        qid = str(query.get("qid") or "")
        ids, scores, _ = retrieve(str(query.get("query") or ""), brain, max(k, len(doc_ids)))
        ranked_by_qid[qid] = rank_docs(
            {str(doc_id): float(scores.get(doc_id, 0.0)) for doc_id in ids},
            doc_ids,
        )
    return ranked_by_qid


def run_first_stage(
    rows: list[dict[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    if variant == "splade":
        ranked_by_qid = _splade_rank(docs, queries)
        metrics, per_query = evaluate_variant(queries, ranked_by_qid)
        return {
            "variant": variant,
            "metrics": metrics,
            "queries": per_query,
            "expansions": {},
            "n_docs": len(docs),
            "n_queries": len(queries),
        }
    all_tokens: dict[str, list[str]] = {}
    title_tokens: dict[str, list[str]] = {}
    doc_ids: list[str] = []
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        if not doc_id:
            continue
        text = str(doc.get("text") or doc.get("title") or "")
        doc_ids.append(doc_id)
        all_tokens[doc_id] = tokenize(text)
        title_tokens[doc_id] = tokenize(title_text(text))
    all_index = Bm25Index([(doc_id, all_tokens[doc_id]) for doc_id in doc_ids])
    title_index = Bm25Index([(doc_id, title_tokens[doc_id]) for doc_id in doc_ids])
    ranked_by_qid: dict[str, list[str]] = {}
    expansions: dict[str, str] = {}
    for query in queries:
        qid = str(query.get("qid") or "")
        qtext = str(query.get("query") or "")
        q_tokens = tokenize(qtext)
        all_scores = all_index.scores(q_tokens)
        title_scores = title_index.scores(q_tokens)
        if variant == "title":
            scores = title_scores
        elif variant == "title-boost":
            scores = dict(all_scores)
            for doc_id, value in title_scores.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + 2.0 * value
        elif variant in {"rm3", "rm3-title-boost"}:
            seed_scores = dict(all_scores)
            if variant == "rm3-title-boost":
                for doc_id, value in title_scores.items():
                    seed_scores[doc_id] = seed_scores.get(doc_id, 0.0) + 2.0 * value
            seed_ranked = rank_docs(seed_scores, doc_ids)
            extra = rm3_terms(q_tokens, seed_ranked, title_tokens)
            expansions[qid] = expand_query(qtext, extra)
            expanded_tokens = tokenize(expansions[qid])
            scores = all_index.scores(expanded_tokens)
            if variant == "rm3-title-boost":
                title_exp = title_index.scores(expanded_tokens)
                for doc_id, value in title_exp.items():
                    scores[doc_id] = scores.get(doc_id, 0.0) + 2.0 * value
        else:
            scores = all_scores
        ranked_by_qid[qid] = rank_docs(scores, doc_ids)
    metrics, per_query = evaluate_variant(queries, ranked_by_qid)
    return {
        "variant": variant,
        "metrics": metrics,
        "queries": per_query,
        "expansions": expansions,
        "n_docs": len(doc_ids),
        "n_queries": len(queries),
    }


def write_expanded_jsonl(
    rows: list[dict[str, Any]],
    expansions: dict[str, str],
    dest: Path,
) -> Path:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("type") or "") != "query":
            out.append(row)
            continue
        cloned = dict(row)
        qid = str(cloned.get("qid") or "")
        expanded = expansions.get(qid)
        if expanded:
            cloned["query"] = expanded
            cloned["query_original"] = row.get("query")
        out.append(cloned)
    return write_records(out, dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pool_first_stage")
    parser.add_argument("--dataset", type=Path, default=Path("data/search_esci_74.jsonl"))
    parser.add_argument("--run", default="search-esci-74-fielded-bm25")
    parser.add_argument(
        "--variant",
        default="all",
        help="Comma-separated: all,title,title-boost,rm3,rm3-title-boost",
    )
    parser.add_argument(
        "--expand-out",
        type=Path,
        default=Path("data/search_esci_74_rm3.jsonl"),
    )
    args = parser.parse_args(argv)
    from search.config import Settings

    settings = Settings.load()
    rows = load_records(args.dataset)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    variants = [item.strip() for item in str(args.variant).split(",") if item.strip()]
    summary: dict[str, Any] = {"run_id": run_id, "dataset": args.dataset.name, "variants": {}}
    expand_source = None
    for variant in variants:
        result = run_first_stage(rows, variant=variant)
        metrics = result["metrics"]
        summary["variants"][variant] = {
            "ndcg@10": metrics.get("ndcg@10"),
            "ndcg@20": metrics.get("ndcg@20"),
            "recall@10": metrics.get("recall@10"),
            "recall@20": metrics.get("recall@20"),
            "mrr": metrics.get("mrr"),
        }
        if variant.startswith("rm3"):
            expand_source = result
    if expand_source and expand_source.get("expansions"):
        write_expanded_jsonl(rows, expand_source["expansions"], args.expand_out)
        summary["expand_out"] = str(args.expand_out)
    docs, queries = split_corpus(rows)
    payload = {
        "status": "ok",
        "brain_id": "harness-local",
        "dataset": args.dataset.name,
        "fusion": "none",
        "rerank": "none",
        "channels": ["harness-bm25"],
        "protocol": (
            "local fielded BM25 over search_esci_74.jsonl 2043-doc corpus; "
            "not BrainAPI passages; not Reddy 0.857"
        ),
        "n_docs": len(docs),
        "n_queries": len(queries),
        "metrics": (summary["variants"].get(variants[0]) or {}),
        "summary": summary,
        "ingest": {"status": "completed", "n_docs": 0, "tasks": [], "reused": True},
    }
    (run_dir / "eval.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
