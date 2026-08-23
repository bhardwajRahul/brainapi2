from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from search.dataset import split_corpus
from search.local_dense import evaluate_dense
from search.pool_first_stage import field_value, tokenize

PROTOCOL = "ltr-head-cv"
PROTOCOL_APPLY = "ltr-head-apply"
PAIR_UNLABELED_ZERO = "unlabeled_zero"
PAIR_OTHER_QUERY_NEG = "other_query_neg"
PAIR_POLICIES = (PAIR_UNLABELED_ZERO, PAIR_OTHER_QUERY_NEG)
BASE_FEATURE_NAMES = (
    "rrf_inv",
    "bm25_inv",
    "dense_inv",
    "title_overlap",
    "brand_hit",
    "query_in_title",
)
CE_FEATURE = "ce_gain"
FEATURE_NAMES = BASE_FEATURE_NAMES
HEAD_RANKNET = "ranknet"
HEAD_LIGHTGBM = "lightgbm"
HEADS = (HEAD_RANKNET, HEAD_LIGHTGBM)
N_FOLDS = 5
SEED = 0
EPOCHS = 40
LR = 0.05
L2 = 1e-3
MAX_PAIRS = 400
LGBM_N_ESTIMATORS = 100
LGBM_MAX_DEPTH = 3
LGBM_LEARNING_RATE = 0.05


def feature_names(*, with_ce: bool = False) -> tuple[str, ...]:
    if with_ce:
        return (*BASE_FEATURE_NAMES, CE_FEATURE)
    return BASE_FEATURE_NAMES


def _gain(row: dict[str, Any], doc_id: str) -> float:
    grades = row.get("gold_grades") or {}
    if isinstance(grades, dict) and doc_id in grades:
        return float(grades.get(doc_id) or 0.0)
    gold = {str(item) for item in (row.get("gold_doc_ids") or []) if item}
    return 1.0 if doc_id in gold else 0.0


def _chunk_to_doc(row: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for hit in row.get("hits") or []:
        chunk_id = str(hit.get("id") or "")
        doc_id = str(hit.get("doc_id") or "")
        if chunk_id and doc_id:
            mapping[chunk_id] = doc_id
    return mapping


def _as_docs(ids: list[str], mapping: dict[str, str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        key = mapping.get(str(raw), str(raw))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _rank_inv(ids: list[str], doc_id: str) -> float:
    try:
        return 1.0 / (ids.index(doc_id) + 1)
    except ValueError:
        return 0.0


def _title(doc: dict[str, Any]) -> str:
    titled = str(doc.get("title") or "").strip()
    if titled:
        return titled
    return field_value(str(doc.get("text") or ""), "Title")


def _brand(doc: dict[str, Any]) -> str:
    branded = str(doc.get("brand") or "").strip()
    if branded:
        return branded
    return field_value(str(doc.get("text") or ""), "Brand")


def overlap(query: str, text: str) -> float:
    qtoks = [tok for tok in tokenize(query) if len(tok) >= 3]
    if not qtoks:
        return 0.0
    ttoks = set(tokenize(text))
    return len(set(qtoks) & ttoks) / float(len(set(qtoks)))


def ce_cache_name(model_dir: Path) -> str:
    stem = model_dir.name.strip() or "ce"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in stem)
    return f"ce_gain_{safe}.json"


def load_ce_cache(path: Path) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    out: dict[str, dict[str, float]] = {}
    for qid, scores in raw.items():
        if not isinstance(scores, dict):
            continue
        out[str(qid)] = {str(doc_id): float(value) for doc_id, value in scores.items()}
    return out


def write_ce_cache(path: Path, scores: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2), encoding="utf-8")


def score_ce_gains(
    eval_result: dict[str, Any],
    docs_by_id: dict[str, dict[str, Any]],
    *,
    model_dir: Path,
    k: int = 50,
    cache_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    if cache_path is not None:
        cached = load_ce_cache(cache_path)
        if cached:
            return cached
    from search.finetune_esci_4class import CLASS_GAINS, weighted_scores
    from search.rank_pool import doc_texts
    from search.rank_pool_4class import load_4class_predict

    predict, _model = load_4class_predict(model_dir)
    texts = doc_texts(list(docs_by_id.values()))
    scored: dict[str, dict[str, float]] = {}
    rows = list(eval_result.get("queries") or [])
    for index, row in enumerate(rows):
        qid = str(row.get("qid") or "")
        query = str(row.get("query") or "")
        mapping = _chunk_to_doc(row)
        hit_ids = [str(item) for item in (row.get("hit_ids") or []) if item][:k]
        doc_ids = _as_docs(hit_ids, mapping)
        if not doc_ids:
            doc_ids = hit_ids
        pairs: list[tuple[str, str]] = []
        for doc_id in doc_ids:
            text = texts.get(doc_id) or _title(docs_by_id.get(doc_id) or {})
            pairs.append((query, text))
        probs = predict(pairs) if pairs else []
        gains = weighted_scores(probs, CLASS_GAINS) if pairs else []
        if len(gains) < len(doc_ids):
            gains = list(gains) + [0.0] * (len(doc_ids) - len(gains))
        scored[qid] = {
            doc_id: float(gains[pos]) for pos, doc_id in enumerate(doc_ids)
        }
        print(f"ltr-ce {index + 1}/{len(rows)} {qid}", flush=True)
    if cache_path is not None:
        write_ce_cache(cache_path, scored)
    return scored


def features_for_doc(
    query: str,
    doc_id: str,
    *,
    rrf_ids: list[str],
    bm25_ids: list[str],
    dense_ids: list[str],
    doc: dict[str, Any] | None,
    ce_gain: float | None = None,
) -> np.ndarray:
    title = _title(doc or {})
    brand = _brand(doc or {})
    q = (query or "").strip().lower()
    in_title = 1.0 if q and len(q) >= 3 and q in title.lower() else 0.0
    qtoks = [tok for tok in tokenize(query) if len(tok) >= 3]
    brand_toks = set(tokenize(brand))
    brand_hit = 1.0 if qtoks and set(qtoks) & brand_toks else 0.0
    values = [
        _rank_inv(rrf_ids, doc_id),
        _rank_inv(bm25_ids, doc_id),
        _rank_inv(dense_ids, doc_id),
        overlap(query, title),
        brand_hit,
        in_title,
    ]
    if ce_gain is not None:
        values.append(float(ce_gain))
    return np.array(values, dtype=np.float64)


def example_from_eval_row(
    row: dict[str, Any],
    docs_by_id: dict[str, dict[str, Any]],
    *,
    k: int = 50,
    ce_scores: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    mapping = _chunk_to_doc(row)
    rrf_ids = [str(item) for item in (row.get("hit_ids") or []) if item][:k]
    if not rrf_ids:
        return None
    bm25_ids = _as_docs(list(row.get("bm25_ids") or []), mapping)
    dense_ids = _as_docs(list(row.get("dense_ids") or []), mapping)
    query = str(row.get("query") or "")
    feats: list[np.ndarray] = []
    gains: list[float] = []
    for doc_id in rrf_ids:
        ce_gain = None
        if ce_scores is not None:
            ce_gain = float(ce_scores.get(doc_id) or 0.0)
        feats.append(
            features_for_doc(
                query,
                doc_id,
                rrf_ids=rrf_ids,
                bm25_ids=bm25_ids,
                dense_ids=dense_ids,
                doc=docs_by_id.get(doc_id),
                ce_gain=ce_gain,
            )
        )
        gains.append(_gain(row, doc_id))
    return {
        "qid": str(row.get("qid") or ""),
        "query": query,
        "ids": rrf_ids,
        "features": np.stack(feats, axis=0),
        "gains": np.array(gains, dtype=np.float64),
        "row": row,
    }


def other_query_gold_ids(examples: list[dict[str, Any]], skip_qid: str) -> set[str]:
    gold: set[str] = set()
    for example in examples:
        if example["qid"] == skip_qid:
            continue
        for doc_id, gain in zip(example["ids"], example["gains"]):
            if float(gain) > 0:
                gold.add(str(doc_id))
    return gold


def training_gains(
    example: dict[str, Any],
    *,
    pair_policy: str,
    other_gold: set[str],
) -> list[float | None]:
    policy = (pair_policy or PAIR_UNLABELED_ZERO).strip()
    out: list[float | None] = []
    for doc_id, gain in zip(example["ids"], example["gains"]):
        value = float(gain)
        if policy == PAIR_OTHER_QUERY_NEG:
            if value > 0:
                out.append(value)
            elif str(doc_id) in other_gold:
                out.append(0.0)
            else:
                out.append(None)
        else:
            out.append(value)
    return out


def collect_pairs(
    example: dict[str, Any],
    *,
    pair_policy: str = PAIR_UNLABELED_ZERO,
    other_gold: set[str] | None = None,
) -> list[tuple[int, int]]:
    labeled = training_gains(
        example,
        pair_policy=pair_policy,
        other_gold=other_gold or set(),
    )
    eligible = [
        (index, gain)
        for index, gain in enumerate(labeled)
        if gain is not None
    ]
    pairs: list[tuple[int, int]] = []
    for left in range(len(eligible)):
        for right in range(left):
            left_i, left_g = eligible[left]
            right_i, right_g = eligible[right]
            if left_g == right_g:
                continue
            hi, lo = (
                (left_i, right_i) if left_g > right_g else (right_i, left_i)
            )
            pairs.append((hi, lo))
    return pairs


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def fit_pairwise(
    examples: list[dict[str, Any]],
    *,
    epochs: int = EPOCHS,
    lr: float = LR,
    l2: float = L2,
    seed: int = SEED,
    pair_policy: str = PAIR_UNLABELED_ZERO,
) -> np.ndarray:
    n_features = int(examples[0]["features"].shape[1]) if examples else len(BASE_FEATURE_NAMES)
    rng = np.random.default_rng(seed)
    weights = np.zeros(n_features, dtype=np.float64)
    if not examples:
        return weights
    for _ in range(max(1, int(epochs))):
        order = rng.permutation(len(examples))
        for index in order:
            example = examples[int(index)]
            feats = example["features"]
            other_gold = other_query_gold_ids(examples, example["qid"])
            pairs = collect_pairs(
                example,
                pair_policy=pair_policy,
                other_gold=other_gold,
            )
            if not pairs:
                continue
            if len(pairs) > MAX_PAIRS:
                chosen = rng.choice(len(pairs), size=MAX_PAIRS, replace=False)
                pairs = [pairs[int(pos)] for pos in chosen]
            for hi, lo in pairs:
                diff = feats[hi] - feats[lo]
                pred = _sigmoid(float(weights.dot(diff)))
                grad = (pred - 1.0) * diff + l2 * weights
                weights -= lr * grad
    return weights


def score_docs(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return features.dot(weights)


def rerank_ids(example: dict[str, Any], weights: np.ndarray) -> list[str]:
    scores = score_docs(example["features"], weights)
    order = np.argsort(-scores, kind="stable")
    return [example["ids"][int(pos)] for pos in order]


def gain_to_rank_label(gain: float) -> int:
    if gain >= 0.99:
        return 3
    if gain >= 0.09:
        return 2
    if gain >= 0.005:
        return 1
    return 0


def rank_train_group(
    example: dict[str, Any],
    *,
    pair_policy: str,
    other_gold: set[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    labeled = training_gains(
        example,
        pair_policy=pair_policy,
        other_gold=other_gold,
    )
    idxs = [index for index, gain in enumerate(labeled) if gain is not None]
    if len(idxs) < 2:
        return None
    y = np.array([gain_to_rank_label(float(labeled[index] or 0.0)) for index in idxs])
    if int(np.unique(y).size) < 2:
        return None
    return example["features"][idxs], y


class _ZeroRanker:
    def __init__(self, n_features: int) -> None:
        self.feature_importances_ = np.zeros(max(1, int(n_features)), dtype=np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=np.float64)


class _LgbmRanker:
    def __init__(self, booster: Any, n_features: int) -> None:
        self.booster = booster
        raw = booster.feature_importance(importance_type="gain")
        importances = np.asarray(raw, dtype=np.float64)
        if importances.size < n_features:
            importances = np.pad(importances, (0, n_features - int(importances.size)))
        self.feature_importances_ = importances[:n_features]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.booster.predict(features), dtype=np.float64)


def fit_lightgbm(
    examples: list[dict[str, Any]],
    *,
    pair_policy: str = PAIR_UNLABELED_ZERO,
    seed: int = SEED,
    min_data_in_leaf: int | None = None,
) -> Any:
    n_features = (
        int(examples[0]["features"].shape[1]) if examples else len(BASE_FEATURE_NAMES)
    )
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[int] = []
    for example in examples:
        group = rank_train_group(
            example,
            pair_policy=pair_policy,
            other_gold=other_query_gold_ids(examples, example["qid"]),
        )
        if group is None:
            continue
        feats, labels = group
        xs.append(feats)
        ys.append(labels)
        groups.append(int(len(labels)))
    if not groups:
        return _ZeroRanker(n_features)
    import lightgbm as lgb

    dataset = lgb.Dataset(
        np.vstack(xs),
        label=np.concatenate(ys),
        group=groups,
        free_raw_data=False,
    )
    params: dict[str, Any] = {
        "objective": "lambdarank",
        "max_depth": LGBM_MAX_DEPTH,
        "learning_rate": LGBM_LEARNING_RATE,
        "num_leaves": 8,
        "verbosity": -1,
        "seed": int(seed),
    }
    if min_data_in_leaf is not None:
        params["min_data_in_leaf"] = int(min_data_in_leaf)
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=LGBM_N_ESTIMATORS,
    )
    return _LgbmRanker(booster, n_features)


def rerank_ids_model(example: dict[str, Any], model: Any) -> list[str]:
    scores = np.asarray(model.predict(example["features"]), dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    return [example["ids"][int(pos)] for pos in order]


def overlap_only_ids(example: dict[str, Any]) -> list[str]:
    title_scores = example["features"][:, 3]
    order = np.argsort(-title_scores, kind="stable")
    return [example["ids"][int(pos)] for pos in order]


def grouped_folds(qids: list[str], n_folds: int = N_FOLDS) -> list[set[str]]:
    folds: list[set[str]] = [set() for _ in range(max(2, int(n_folds)))]
    for index, qid in enumerate(sorted(qids)):
        folds[index % len(folds)].add(qid)
    return folds


def build_ltr_examples(
    eval_result: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    k: int = 50,
    ce_model: Path | None = None,
    ce_cache_path: Path | None = None,
    ce_scores: dict[str, dict[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    docs, queries = split_corpus(rows)
    docs_by_id = {str(doc.get("doc_id") or ""): doc for doc in docs if doc.get("doc_id")}
    resolved_ce = ce_scores
    ce_model_path = Path(ce_model) if ce_model else None
    if resolved_ce is None and ce_model_path is not None:
        resolved_ce = score_ce_gains(
            eval_result,
            docs_by_id,
            model_dir=ce_model_path,
            k=k,
            cache_path=ce_cache_path,
        )
    with_ce = resolved_ce is not None
    examples: list[dict[str, Any]] = []
    for row in eval_result.get("queries") or []:
        qid = str(row.get("qid") or "")
        example = example_from_eval_row(
            row,
            docs_by_id,
            k=k,
            ce_scores=(resolved_ce or {}).get(qid) if with_ce else None,
        )
        if example is None:
            continue
        examples.append(example)
    return examples, queries, docs_by_id


def run_ltr_head(
    eval_result: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    k: int = 50,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    n_folds: int = N_FOLDS,
    brain_id: str | None = None,
    source_run: str | None = None,
    pair_policy: str = PAIR_UNLABELED_ZERO,
    ce_model: Path | None = None,
    ce_cache_path: Path | None = None,
    ce_scores: dict[str, dict[str, float]] | None = None,
    ltr_head: str = HEAD_RANKNET,
    train_eval_result: dict[str, Any] | None = None,
    train_rows: list[dict[str, Any]] | None = None,
    train_source_run: str | None = None,
    train_ce_cache_path: Path | None = None,
    train_ce_scores: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    policy = (pair_policy or PAIR_UNLABELED_ZERO).strip()
    if policy not in PAIR_POLICIES:
        raise ValueError(f"Unknown pair policy {pair_policy!r}")
    head = (ltr_head or HEAD_RANKNET).strip()
    if head not in HEADS:
        raise ValueError(f"Unknown LTR head {ltr_head!r}")
    if train_eval_result is not None and train_rows is None:
        raise ValueError("train_rows is required when train_eval_result is set")
    apply_mode = train_eval_result is not None
    ce_model_path = Path(ce_model) if ce_model else None
    eval_examples, queries, docs_by_id = build_ltr_examples(
        eval_result,
        rows,
        k=k,
        ce_model=ce_model_path,
        ce_cache_path=ce_cache_path,
        ce_scores=ce_scores,
    )
    names = feature_names(with_ce=ce_model_path is not None or ce_scores is not None)
    if apply_mode:
        train_examples, _, _ = build_ltr_examples(
            train_eval_result or {},
            train_rows or [],
            k=k,
            ce_model=ce_model_path,
            ce_cache_path=train_ce_cache_path,
            ce_scores=train_ce_scores,
        )
        names = feature_names(
            with_ce=bool(train_examples)
            and int(train_examples[0]["features"].shape[1]) > len(BASE_FEATURE_NAMES)
        )
        ranked_cv: dict[str, list[str]] = {}
        ranked_overlap: dict[str, list[str]] = {}
        fold_weights: list[list[float]] = []
        if head == HEAD_LIGHTGBM:
            model = fit_lightgbm(train_examples, pair_policy=policy)
            fold_weights.append([float(item) for item in model.feature_importances_])
            for example in eval_examples:
                ranked_cv[example["qid"]] = rerank_ids_model(example, model)[:k]
                ranked_overlap[example["qid"]] = overlap_only_ids(example)[:k]
        else:
            weights = fit_pairwise(train_examples, pair_policy=policy)
            fold_weights.append([float(item) for item in weights])
            for example in eval_examples:
                ranked_cv[example["qid"]] = rerank_ids(example, weights)[:k]
                ranked_overlap[example["qid"]] = overlap_only_ids(example)[:k]
        n_train = len(train_examples)
    else:
        examples = eval_examples
        by_qid = {example["qid"]: example for example in examples}
        qids = [example["qid"] for example in examples]
        ranked_cv = {}
        ranked_overlap = {}
        fold_weights = []
        for test_qids in grouped_folds(qids, n_folds=n_folds):
            train_examples = [
                example for example in examples if example["qid"] not in test_qids
            ]
            if head == HEAD_LIGHTGBM:
                model = fit_lightgbm(train_examples, pair_policy=policy)
                fold_weights.append([float(item) for item in model.feature_importances_])
                for qid in test_qids:
                    example = by_qid.get(qid)
                    if example is None:
                        continue
                    ranked_cv[qid] = rerank_ids_model(example, model)[:k]
                    ranked_overlap[qid] = overlap_only_ids(example)[:k]
                continue
            weights = fit_pairwise(train_examples, pair_policy=policy)
            fold_weights.append([float(item) for item in weights])
            for qid in test_qids:
                example = by_qid.get(qid)
                if example is None:
                    continue
                ranked_cv[qid] = rerank_ids(example, weights)[:k]
                ranked_overlap[qid] = overlap_only_ids(example)[:k]
        n_train = None
        train_examples = []
    metrics, per_query = evaluate_dense(
        ranked_cv,
        queries,
        ks=ks,
        encode_ms=0.0,
    )
    overlap_metrics, _ = evaluate_dense(
        ranked_overlap,
        queries,
        ks=ks,
        encode_ms=0.0,
    )
    mean_weights = (
        np.mean(np.array(fold_weights, dtype=np.float64), axis=0)
        if fold_weights
        else np.zeros(len(names))
    )
    return {
        "status": "ok" if per_query else "failed",
        "brain_id": brain_id or "harness-local-ltr",
        "dataset": dataset_name,
        "fusion": "none",
        "rerank": "ltr-lightgbm" if head == HEAD_LIGHTGBM else "ltr-pairwise",
        "channels": ["harness-ltr-head"],
        "expand": "none",
        "k": k,
        "ks": list(ks),
        "n_docs": len(docs_by_id),
        "n_queries": len(per_query),
        "n_docs_mapped": 0,
        "skip_enrichment": True,
        "ingest_graph": False,
        "skip_ingest": True,
        "rank_pool": False,
        "ltr_from_run": source_run or eval_result.get("run_id"),
        "ltr_train_run": train_source_run if apply_mode else None,
        "ltr_n_train_queries": n_train if apply_mode else None,
        "protocol": PROTOCOL_APPLY if apply_mode else PROTOCOL,
        "ltr_features": list(names),
        "ltr_pair_policy": policy,
        "ltr_head": head,
        "ltr_ce_model": str(ce_model_path) if ce_model_path else None,
        "ltr_lgbm": (
            {
                "n_estimators": LGBM_N_ESTIMATORS,
                "max_depth": LGBM_MAX_DEPTH,
                "learning_rate": LGBM_LEARNING_RATE,
            }
            if head == HEAD_LIGHTGBM
            else None
        ),
        "ltr_n_folds": 1 if apply_mode else n_folds,
        "ltr_seed": SEED,
        "ltr_epochs": EPOCHS,
        "ltr_mean_weights": {
            name: float(weight)
            for name, weight in zip(names, mean_weights)
        },
        "overlap_only_metrics": overlap_metrics,
        "ingest": {
            "status": "completed",
            "n_docs": len(docs_by_id),
            "tasks": [],
            "reused": True,
        },
        "graph_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "interaction_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "event_probe": None,
        "search_error": None,
        "metrics": metrics,
        "queries": per_query,
    }
