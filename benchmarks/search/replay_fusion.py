from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from search.metrics import mrr, ndcg_at_k, recall_at_k

RRF_K = 60
EXPANSION_N = 10
GRAPH_WEIGHTS = (0.1, 0.25, 0.5)
FUSED_NDCG = 0.653
RECALL20_TARGET = 0.80
_SLICE_JSONL = Path(__file__).resolve().parent.parent / "data" / "search_esci_slice.jsonl"


def _load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gold(query: dict[str, Any]) -> set[str]:
    gold = {str(item) for item in (query.get("gold_doc_ids") or []) if item}
    grades = query.get("gold_grades") or {}
    if isinstance(grades, dict):
        for doc_id, gain in grades.items():
            if float(gain or 0) <= 0:
                continue
            gold.add(str(doc_id))
    if gold:
        return gold
    return {str(item) for item in (query.get("gold_chunk_ids") or []) if item}


def dataset_gold_grades(path: Path | None = None) -> dict[str, dict[str, float]]:
    jsonl = path or _SLICE_JSONL
    if not jsonl.exists():
        return {}
    grades: dict[str, dict[str, float]] = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") != "query":
            continue
        qid = str(row.get("qid") or "")
        raw = row.get("gold_grades") or {}
        if not qid or not isinstance(raw, dict):
            continue
        grades[qid] = {
            str(doc_id): float(gain)
            for doc_id, gain in raw.items()
            if float(gain or 0) > 0
        }
    return grades


def _grades(
    query: dict[str, Any],
    gold: set[str],
    dataset_grades: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    raw = query.get("gold_grades") or {}
    if isinstance(raw, dict) and raw:
        return {
            str(doc_id): float(gain)
            for doc_id, gain in raw.items()
            if float(gain or 0) > 0
        }
    qid = str(query.get("qid") or "")
    from_dataset = (dataset_grades or {}).get(qid) or {}
    if from_dataset:
        return from_dataset
    return {doc_id: 1.0 for doc_id in gold}


def _ranked(query: dict[str, Any]) -> list[str]:
    return [str(item) for item in (query.get("hit_ids") or []) if item]


def drop_hubs(ranked: list[str]) -> list[str]:
    return [item for item in ranked if not item.startswith("hub:")]


def keep_product_asins(ranked: list[str]) -> list[str]:
    return [
        item
        for item in ranked
        if item and not item.startswith("hub:") and not item.startswith("evt:")
    ]


def score_ranked(
    ranked: list[str],
    gold: set[str],
    grades: dict[str, float],
) -> dict[str, float]:
    return {
        "ndcg@10": ndcg_at_k(ranked, gold, 10, grades=grades),
        "recall@10": recall_at_k(ranked, gold, 10),
        "recall@20": recall_at_k(ranked, gold, 20),
        "mrr": mrr(ranked, gold),
        "unique_docs@20": float(len({str(item) for item in ranked[:20] if item})),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ("ndcg@10", "recall@10", "recall@20", "mrr", "unique_docs@20")
    if not rows:
        return {key: 0.0 for key in keys}
    present = [key for key in keys if all(key in row for row in rows)]
    return {key: sum(row[key] for row in rows) / len(rows) for key in present}


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[str]:
    scores: dict[str, float] = {}
    for index, ranked in enumerate(ranked_lists):
        weight = 1.0 if not weights else float(weights[index])
        for rank, item in enumerate(ranked):
            if not item:
                continue
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
    return [
        item
        for item, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    ]


def chunk_to_doc_map(*queries: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for query in queries:
        for hit in query.get("hits") or []:
            if not isinstance(hit, dict):
                continue
            hid = str(hit.get("id") or "")
            doc_id = str(hit.get("doc_id") or hid)
            if hid:
                mapping[hid] = doc_id
    return mapping


def collapse_to_doc(
    ids: list[str],
    chunk_to_doc: dict[str, str],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in ids:
        key = chunk_to_doc.get(str(item), str(item))
        if (
            not key
            or key.startswith("hub:")
            or key.startswith("evt:")
            or key in seen
        ):
            continue
        seen.add(key)
        out.append(key)
    return out


def _id_list(query: dict[str, Any], key: str, channel: str | None = None) -> list[str]:
    dumped = [str(item) for item in (query.get(key) or []) if item]
    if dumped:
        return dumped
    if not channel:
        return []
    out: list[str] = []
    for hit in query.get("hits") or []:
        if not isinstance(hit, dict):
            continue
        hid = str(hit.get("id") or "")
        if hid and str(hit.get("channel") or "") == channel:
            out.append(hid)
    return out


def query_channel_lists(query: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "dense": _id_list(query, "dense_ids"),
        "bm25": _id_list(query, "bm25_ids"),
        "passages": _id_list(query, "passage_ids", "passages"),
        "entities": _id_list(query, "entity_ids", "entities"),
        "communities": _id_list(query, "community_ids", "communities"),
    }


def merge_query_lists(
    graph_query: dict[str, Any],
    passages_query: dict[str, Any] | None = None,
    entities_query: dict[str, Any] | None = None,
    communities_query: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    graph = query_channel_lists(graph_query)
    passages = query_channel_lists(passages_query or {})
    entities = query_channel_lists(entities_query or {})
    communities = query_channel_lists(communities_query or {})
    dense = graph["dense"] or passages["dense"]
    bm25 = graph["bm25"] or passages["bm25"]
    passage_ids = passages["passages"] or graph["passages"]
    if not passage_ids and passages_query:
        passage_ids = [
            str(hit.get("id") or "")
            for hit in (passages_query.get("hits") or [])
            if hit.get("id")
        ]
        if not passage_ids:
            passage_ids = _ranked(passages_query)
    entity_ids = graph["entities"] or entities["entities"]
    if not entity_ids and entities_query:
        entity_ids = [
            str(hit.get("id") or "")
            for hit in (entities_query.get("hits") or [])
            if hit.get("id")
        ]
        if not entity_ids:
            entity_ids = _ranked(entities_query)
    community_ids = graph["communities"] or communities["communities"]
    if not community_ids and communities_query:
        community_ids = [
            str(hit.get("id") or "")
            for hit in (communities_query.get("hits") or [])
            if hit.get("id")
        ]
        if not community_ids:
            community_ids = _ranked(communities_query)
    return {
        "dense": dense,
        "bm25": bm25,
        "passages": passage_ids,
        "entities": entity_ids,
        "communities": community_ids,
    }


def collapsed_lists(
    lists: dict[str, list[str]],
    chunk_to_doc: dict[str, str],
) -> dict[str, list[str]]:
    collapsed = {
        name: collapse_to_doc(ids, chunk_to_doc) for name, ids in lists.items()
    }
    if not collapsed["dense"] and not collapsed["bm25"] and collapsed["passages"]:
        collapsed["passages_core"] = collapsed["passages"]
    else:
        collapsed["passages_core"] = reciprocal_rank_fusion(
            [ids for ids in (collapsed["dense"], collapsed["bm25"]) if ids]
        ) or collapsed["passages"]
    return collapsed


def _by_qid(eval_result: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not eval_result:
        return {}
    return {
        str(row.get("qid")): row for row in (eval_result.get("queries") or [])
    }


def _pair_queries(
    graph_eval: dict[str, Any],
    passages_eval: dict[str, Any],
    entities_eval: dict[str, Any] | None = None,
    communities_eval: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    passages_by = _by_qid(passages_eval)
    entities_by = _by_qid(entities_eval)
    communities_by = _by_qid(communities_eval)
    slice_grades = dataset_gold_grades()
    rows: list[dict[str, Any]] = []
    for graph_query in graph_eval.get("queries") or []:
        qid = str(graph_query.get("qid") or "")
        passages_query = passages_by.get(qid) or {}
        entities_query = entities_by.get(qid)
        communities_query = communities_by.get(qid)
        gold = _gold(graph_query) or _gold(passages_query)
        grades = _grades(graph_query, gold, slice_grades)
        if not (graph_query.get("gold_grades") or {}):
            grades = _grades(passages_query, gold, slice_grades)
        mapping = chunk_to_doc_map(
            graph_query, passages_query, entities_query or {}, communities_query or {}
        )
        lists = merge_query_lists(
            graph_query, passages_query, entities_query, communities_query
        )
        collapsed = collapsed_lists(lists, mapping)
        rows.append(
            {
                "qid": qid,
                "gold": gold,
                "grades": grades,
                "lists": lists,
                "collapsed": collapsed,
                "passages_ranked": collapse_to_doc(
                    lists["passages"] or _ranked(passages_query), mapping
                ),
                "graph_ranked": _ranked(graph_query),
                "passage_metrics": passages_query.get("metrics") or {},
            }
        )
    return rows


def replay(graph_eval: dict[str, Any], passages_eval: dict[str, Any]) -> dict[str, Any]:
    passages_by_qid = _by_qid(passages_eval)
    hub_rows: list[dict[str, float]] = []
    asin_rows: list[dict[str, float]] = []
    per_query: list[dict[str, Any]] = []
    for query in graph_eval.get("queries") or []:
        qid = str(query.get("qid") or "")
        gold = _gold(query)
        grades = _grades(query, gold)
        ranked = _ranked(query)
        hub = score_ranked(drop_hubs(ranked), gold, grades)
        asin = score_ranked(keep_product_asins(ranked), gold, grades)
        hub_rows.append(hub)
        asin_rows.append(asin)
        passage_row = passages_by_qid.get(qid) or {}
        passage_metrics = passage_row.get("metrics") or {}
        per_query.append(
            {
                "qid": qid,
                "n_hits": len(ranked),
                "n_hits_no_hub": len(drop_hubs(ranked)),
                "n_product_ids": len(keep_product_asins(ranked)),
                "hub_drop": hub,
                "product_ids": asin,
                "passages": {
                    "ndcg@10": float(passage_metrics.get("ndcg@10") or 0.0),
                    "recall@10": float(passage_metrics.get("recall@10") or 0.0),
                    "recall@20": float(passage_metrics.get("recall@20") or 0.0),
                    "mrr": float(passage_metrics.get("mrr") or 0.0),
                },
            }
        )
    return {
        "n_queries": len(per_query),
        "hub_drop": mean_metrics(hub_rows),
        "product_ids": mean_metrics(asin_rows),
        "queries": per_query,
    }


def _core_and_graph(collapsed: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]:
    core = [ids for ids in (collapsed["dense"], collapsed["bm25"]) if ids]
    if not core and collapsed.get("passages_core"):
        core = [collapsed["passages_core"]]
    entities = collapsed.get("entities") or []
    communities = collapsed.get("communities") or []
    return core, entities, communities


def rank_collapse_rrf(collapsed: dict[str, list[str]]) -> list[str]:
    core, entities, communities = _core_and_graph(collapsed)
    lists = [*core]
    if entities:
        lists.append(entities)
    if communities:
        lists.append(communities)
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0])
    return reciprocal_rank_fusion(lists)


def rank_weighted_rrf(collapsed: dict[str, list[str]], graph_weight: float) -> list[str]:
    core, entities, communities = _core_and_graph(collapsed)
    lists: list[list[str]] = [*core]
    weights: list[float] = [1.0] * len(core)
    if entities:
        lists.append(entities)
        weights.append(graph_weight)
    if communities:
        lists.append(communities)
        weights.append(graph_weight)
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0])
    return reciprocal_rank_fusion(lists, weights=weights)


def rank_expansion(
    passage_docs: list[str],
    collapsed: dict[str, list[str]],
    *,
    n: int = EXPANSION_N,
) -> list[str]:
    head = list(passage_docs[:10])
    passage_set = set(passage_docs)
    graph_docs: list[str] = []
    seen: set[str] = set()
    for item in (collapsed.get("entities") or []) + (collapsed.get("communities") or []):
        if item in seen:
            continue
        seen.add(item)
        graph_docs.append(item)
    novel = [item for item in graph_docs if item not in passage_set][:n]
    novel_set = set(novel)
    rest = [item for item in passage_docs[10:] if item not in novel_set]
    return head + novel + rest


def rank_confirmation(
    passage_docs: list[str],
    collapsed: dict[str, list[str]],
) -> list[str]:
    passage_set = set(passage_docs)
    confirmed: list[str] = []
    seen: set[str] = set()
    for item in (collapsed.get("entities") or []) + (collapsed.get("communities") or []):
        if item not in passage_set or item in seen:
            continue
        seen.add(item)
        confirmed.append(item)
    if not confirmed:
        return list(passage_docs)
    return reciprocal_rank_fusion([passage_docs, confirmed])


def replay_offline(
    graph_eval: dict[str, Any],
    passages_eval: dict[str, Any],
    *,
    entities_eval: dict[str, Any] | None = None,
    communities_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paired = _pair_queries(
        graph_eval, passages_eval, entities_eval, communities_eval
    )
    arms: dict[str, list[dict[str, float]]] = {
        "passages": [],
        "collapse-rrf": [],
        "weighted-0.1": [],
        "weighted-0.25": [],
        "weighted-0.5": [],
        "expansion-n10": [],
        "confirmation": [],
    }
    per_query: list[dict[str, Any]] = []
    for row in paired:
        gold = row["gold"]
        grades = row["grades"]
        collapsed = row["collapsed"]
        passage_docs = row["passages_ranked"]
        scored = {
            "passages": score_ranked(passage_docs, gold, grades),
            "collapse-rrf": score_ranked(
                rank_collapse_rrf(collapsed), gold, grades
            ),
            "expansion-n10": score_ranked(
                rank_expansion(passage_docs, collapsed, n=EXPANSION_N),
                gold,
                grades,
            ),
            "confirmation": score_ranked(
                rank_confirmation(passage_docs, collapsed), gold, grades
            ),
        }
        for weight in GRAPH_WEIGHTS:
            name = f"weighted-{weight}"
            scored[name] = score_ranked(
                rank_weighted_rrf(collapsed, weight), gold, grades
            )
        for name, metrics in scored.items():
            arms[name].append(metrics)
        per_query.append({"qid": row["qid"], **scored})
    summary = {name: mean_metrics(rows) for name, rows in arms.items()}
    return {
        "n_queries": len(per_query),
        "expansion_n": EXPANSION_N,
        "graph_weights": list(GRAPH_WEIGHTS),
        "primary": ["recall@20", "ndcg@10"],
        "arms": summary,
        "queries": per_query,
    }


def pick_offline_winner(summary: dict[str, dict[str, float]]) -> str:
    passages = summary.get("passages") or {}
    passages_ndcg = float(passages.get("ndcg@10") or 0.0)
    passages_r20 = float(passages.get("recall@20") or 0.0)
    candidates: list[tuple[str, dict[str, float]]] = []
    for name, metrics in summary.items():
        if name == "passages":
            continue
        recall20 = float(metrics.get("recall@20") or 0.0)
        ndcg = float(metrics.get("ndcg@10") or 0.0)
        if recall20 + 1e-9 < max(RECALL20_TARGET, passages_r20):
            continue
        if ndcg + 1e-9 < max(FUSED_NDCG, passages_ndcg):
            continue
        candidates.append((name, metrics))
    if not candidates:
        return "G08"

    def _key(item: tuple[str, dict[str, float]]) -> tuple[float, float]:
        metrics = item[1]
        return (
            float(metrics.get("recall@20") or 0.0),
            float(metrics.get("ndcg@10") or 0.0),
        )

    name, _metrics = max(candidates, key=_key)
    return name


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    unique = metrics.get("unique_docs@20")
    unique_s = f" unique@20={unique:.2f}" if unique is not None else ""
    print(
        f"{label:18} "
        f"nDCG@10={metrics.get('ndcg@10', 0.0):.3f} "
        f"R@10={metrics.get('recall@10', 0.0):.3f} "
        f"R@20={metrics.get('recall@20', 0.0):.3f} "
        f"MRR={metrics.get('mrr', 0.0):.3f}"
        f"{unique_s}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline rescore of fused search eval.json lists."
    )
    parser.add_argument("--graph", required=True, help="Graph-arm eval.json")
    parser.add_argument("--passages", required=True, help="Passages-arm eval.json")
    parser.add_argument(
        "--entities",
        default=None,
        help="Optional isolated entities eval.json",
    )
    parser.add_argument(
        "--communities",
        default=None,
        help="Optional isolated communities eval.json",
    )
    parser.add_argument(
        "--mode",
        default="hub-drop",
        choices=("hub-drop", "collapse-rrf", "gated", "all"),
    )
    args = parser.parse_args(argv)
    graph_eval = _load_eval(Path(args.graph))
    passages_eval = _load_eval(Path(args.passages))
    entities_eval = _load_eval(Path(args.entities)) if args.entities else None
    communities_eval = (
        _load_eval(Path(args.communities)) if args.communities else None
    )
    if args.mode in {"hub-drop", "all"}:
        result = replay(graph_eval, passages_eval)
        hub = result["hub_drop"]
        print(
            "hub-drop "
            f"nDCG@10={hub['ndcg@10']:.3f} "
            f"R@10={hub['recall@10']:.3f} "
            f"R@20={hub['recall@20']:.3f} "
            f"MRR={hub['mrr']:.3f}"
        )
        asin = result["product_ids"]
        print(
            "product-ids "
            f"nDCG@10={asin['ndcg@10']:.3f} "
            f"R@10={asin['recall@10']:.3f} "
            f"R@20={asin['recall@20']:.3f} "
            f"MRR={asin['mrr']:.3f}"
        )
        for row in result["queries"]:
            delta = row["hub_drop"]["ndcg@10"] - row["passages"]["ndcg@10"]
            print(
                f"{row['qid']}: hub-drop nDCG {row['hub_drop']['ndcg@10']:.3f} "
                f"(passages {row['passages']['ndcg@10']:.3f}, delta {delta:+.3f})"
            )
    if args.mode in {"collapse-rrf", "gated", "all"}:
        offline = replay_offline(
            graph_eval,
            passages_eval,
            entities_eval=entities_eval,
            communities_eval=communities_eval,
        )
        names = ["passages", "collapse-rrf"]
        if args.mode in {"gated", "all"}:
            names.extend(
                [
                    "weighted-0.1",
                    "weighted-0.25",
                    "weighted-0.5",
                    "expansion-n10",
                    "confirmation",
                ]
            )
        print(
            "primary recall@20 then nDCG@10; "
            f"expansion_n={offline['expansion_n']}; "
            f"graph_weights={offline['graph_weights']}"
        )
        for name in names:
            _print_metrics(name, offline["arms"][name])
        if args.mode in {"gated", "all"}:
            winner = pick_offline_winner(offline["arms"])
            print(f"winner {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
