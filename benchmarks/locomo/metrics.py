from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Iterable


_ARTICLES = {"a", "an", "the"}
_DIALOG_EVIDENCE_RE = re.compile(r"^D(\d+)", re.IGNORECASE)
_SESSION_RE = re.compile(r"session_(\d+)", re.IGNORECASE)


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    tokens = [t for t in text.split() if t and t not in _ARTICLES]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return normalize_answer(text).split()


def f1_score(prediction: str, gold: str) -> float | None:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not gold_tokens:
        return None
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction: str, gold: str) -> float | None:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not gold_tokens:
        return None
    if not pred_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    match = 0
    for token in pred_tokens:
        if gold_counts[token] > 0:
            match += 1
            gold_counts[token] -= 1
    precision = match / len(pred_tokens)
    brevity = min(1.0, math.exp(1 - len(gold_tokens) / len(pred_tokens)))
    return precision * brevity


def overlap_scores(
    prediction: str, gold: str, *, adversarial: bool = False
) -> tuple[float | None, float | None]:
    if adversarial:
        return None, None
    return f1_score(prediction, gold), bleu1_score(prediction, gold)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def evidence_session_ids(evidence: Any) -> set[str]:
    sessions: set[str] = set()
    for item in evidence or []:
        text = str(item or "")
        match = _DIALOG_EVIDENCE_RE.match(text)
        if match:
            sessions.add(f"session_{match.group(1)}")
            continue
        for session_match in _SESSION_RE.finditer(text):
            sessions.add(f"session_{session_match.group(1)}")
    return sessions


CHANNELS = ("graph", "passages")

_CHANNEL_ID_FIELD = {
    "combined": "retrieved_session_ids",
    "graph": "retrieved_session_ids_graph",
    "passages": "retrieved_session_ids_passages",
}
_CHANNEL_TEXT_FIELDS = {
    "combined": (
        "text_context",
        "graph_context",
        "triples",
        "paths",
        "source_passages",
        "historical_context",
    ),
    "graph": ("graph_context", "triples", "paths"),
    "passages": ("source_passages", "historical_context"),
}


def _channel_blobs(row: dict[str, Any], channel: str) -> list[str]:
    blobs: list[str] = []
    for field in _CHANNEL_TEXT_FIELDS[channel]:
        value = row.get(field)
        if isinstance(value, (list, tuple)):
            blobs.append("\n".join(str(item) for item in value))
        elif value:
            blobs.append(str(value))
    return blobs


def _scrape_session_ids(blobs: list[str]) -> set[str]:
    sessions: set[str] = set()
    for blob in blobs:
        for match in _SESSION_RE.finditer(blob):
            sessions.add(f"session_{match.group(1)}")
        for match in re.finditer(r"\bD(\d+):", blob, re.IGNORECASE):
            sessions.add(f"session_{match.group(1)}")
    return sessions


def retrieved_session_ids(
    row: dict[str, Any], channel: str = "combined"
) -> set[str] | None:
    field = _CHANNEL_ID_FIELD[channel]
    explicit = row.get(field)
    if explicit is not None:
        sessions = {str(s) for s in explicit if s}
        if sessions or channel != "combined":
            return sessions
    if channel != "combined" and not any(
        f in row for f in _CHANNEL_TEXT_FIELDS[channel]
    ):
        return None
    return _scrape_session_ids(_channel_blobs(row, channel))


def full_context_blob(row: dict[str, Any]) -> str:
    return "\n".join(_channel_blobs(row, "combined")).lower()


def gold_in_context(row: dict[str, Any], threshold: float = 0.5) -> bool:
    gold_tokens = [t for t in tokenize(str(row.get("gold") or "")) if len(t) > 2]
    if not gold_tokens:
        return False
    blob = full_context_blob(row)
    hits = sum(1 for token in gold_tokens if token in blob)
    return hits / len(gold_tokens) >= threshold


def evidence_coverage(row: dict[str, Any], channel: str = "combined") -> str | None:
    needed = evidence_session_ids(row.get("evidence"))
    if not needed:
        return None
    have = retrieved_session_ids(row, channel)
    if have is None:
        return None
    if needed <= have:
        return "full"
    if needed & have:
        return "partial"
    return "none"


def _recall_bucket(items: list[dict[str, Any]], channel: str) -> dict[str, Any]:
    coverages = [evidence_coverage(r, channel) for r in items]
    with_evidence = [c for c in coverages if c is not None]
    n_ev = len(with_evidence)
    full = sum(1 for c in with_evidence if c == "full")
    partial = sum(1 for c in with_evidence if c in {"full", "partial"})
    return {
        "evidence_session_recall_full": (full / n_ev) if n_ev else None,
        "evidence_session_recall_partial": (partial / n_ev) if n_ev else None,
        "n_with_evidence": n_ev,
    }


def aggregate_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        cat = int(row.get("category") or 0)
        by_category.setdefault(cat, []).append(row)

    def _bucket(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            empty = {
                "n": 0,
                "judge_accuracy": None,
                "judge_accuracy_ci95": None,
                "mean_f1": None,
                "mean_bleu1": None,
                "n_scored_f1": 0,
                "answerable_rate": None,
                "evidence_session_recall_full": None,
                "evidence_session_recall_partial": None,
                "n_with_evidence": 0,
            }
            for channel in CHANNELS:
                empty[f"evidence_session_recall_full_{channel}"] = None
                empty[f"evidence_session_recall_partial_{channel}"] = None
                empty[f"n_with_evidence_{channel}"] = 0
            return empty
        correct = [1.0 if r.get("judge_correct") else 0.0 for r in items]
        successes = int(sum(correct))
        lo, hi = wilson_ci(successes, len(items))
        answerable = [1.0 if gold_in_context(r) else 0.0 for r in items]
        f1s = [float(r["f1"]) for r in items if r.get("f1") is not None]
        bleus = [float(r["bleu1"]) for r in items if r.get("bleu1") is not None]
        bucket = {
            "n": len(items),
            "judge_accuracy": mean(correct),
            "judge_accuracy_ci95": (
                {"low": lo, "high": hi} if lo is not None and hi is not None else None
            ),
            "mean_f1": mean(f1s),
            "mean_bleu1": mean(bleus),
            "n_scored_f1": len(f1s),
            "answerable_rate": mean(answerable),
            **_recall_bucket(items, "combined"),
        }
        for channel in CHANNELS:
            channel_bucket = _recall_bucket(items, channel)
            for key, value in channel_bucket.items():
                bucket[f"{key}_{channel}"] = value
        return bucket

    non_adv = [r for r in rows if int(r.get("category") or 0) != 5]
    adversarial = [r for r in rows if int(r.get("category") or 0) == 5]
    all_rows = rows
    non_adv_bucket = _bucket(non_adv)
    all_bucket = _bucket(all_rows)
    retrieval_latencies = [
        float(r["retrieve_latency_ms"])
        for r in rows
        if r.get("retrieve_latency_ms") is not None
    ]
    total_tokens = 0
    for r in rows:
        for key in ("answer_total_tokens", "judge_total_tokens"):
            if r.get(key) is not None:
                total_tokens += int(r[key])

    answerer_gap = None
    if (
        non_adv_bucket.get("answerable_rate") is not None
        and non_adv_bucket.get("judge_accuracy") is not None
    ):
        answerer_gap = (
            float(non_adv_bucket["answerable_rate"])
            - float(non_adv_bucket["judge_accuracy"])
        )

    return {
        "n_total": len(all_rows),
        "n_non_adversarial": len(non_adv),
        "headline_judge_accuracy": non_adv_bucket["judge_accuracy"],
        "headline_judge_accuracy_ci95": non_adv_bucket["judge_accuracy_ci95"],
        "overall_judge_accuracy": all_bucket["judge_accuracy"],
        "overall_judge_accuracy_ci95": all_bucket["judge_accuracy_ci95"],
        "mean_f1_non_adversarial": non_adv_bucket["mean_f1"],
        "mean_bleu1_non_adversarial": non_adv_bucket["mean_bleu1"],
        "answerable_rate": non_adv_bucket["answerable_rate"],
        "evidence_session_recall_full": non_adv_bucket["evidence_session_recall_full"],
        "evidence_session_recall_partial": non_adv_bucket[
            "evidence_session_recall_partial"
        ],
        "evidence_session_recall_by_channel": {
            channel: {
                "full": non_adv_bucket[f"evidence_session_recall_full_{channel}"],
                "partial": non_adv_bucket[
                    f"evidence_session_recall_partial_{channel}"
                ],
                "n_with_evidence": non_adv_bucket[f"n_with_evidence_{channel}"],
            }
            for channel in CHANNELS
        },
        "answerer_gap": answerer_gap,
        "abstention": {
            "n": len(adversarial),
            "accuracy": _bucket(adversarial)["judge_accuracy"],
        },
        "by_category": {
            str(cat): _bucket(items) for cat, items in sorted(by_category.items())
        },
        "retrieval_latency_ms": {
            "p50": percentile(retrieval_latencies, 50),
            "p95": percentile(retrieval_latencies, 95),
            "mean": mean(retrieval_latencies),
        },
        "total_llm_tokens": total_tokens,
    }


def mcnemar_exact(flipped_wrong: int, flipped_right: int) -> float:
    n = flipped_wrong + flipped_right
    if n <= 0:
        return 1.0
    k = min(flipped_wrong, flipped_right)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _correct_by_question(rows: list[dict[str, Any]]) -> dict[tuple[str, int], bool]:
    scores: dict[tuple[str, int], bool] = {}
    for row in rows:
        if row.get("error"):
            continue
        key = (str(row.get("sample_id")), int(row.get("qa_index") or 0))
        scores[key] = bool(row.get("judge_correct"))
    return scores


def _arm_scores(arm: list[list[dict[str, Any]]]) -> dict[tuple[str, int], float]:
    per_run = [_correct_by_question(rows) for rows in arm if rows]
    if not per_run:
        return {}
    keys = set(per_run[0])
    for run in per_run[1:]:
        keys &= set(run)
    return {
        key: sum(1.0 for run in per_run if run[key]) / len(per_run) for key in keys
    }


def _category_by_question(
    arm: list[list[dict[str, Any]]],
) -> dict[tuple[str, int], int]:
    categories: dict[tuple[str, int], int] = {}
    for rows in arm:
        for row in rows:
            key = (str(row.get("sample_id")), int(row.get("qa_index") or 0))
            categories.setdefault(key, int(row.get("category") or 0))
    return categories


def _flip_table(
    baseline: dict[tuple[str, int], float],
    candidate: dict[tuple[str, int], float],
    keys: Iterable[tuple[str, int]],
) -> dict[str, Any]:
    flipped_right: list[tuple[str, int]] = []
    flipped_wrong: list[tuple[str, int]] = []
    agreed = 0
    for key in keys:
        base = baseline[key]
        cand = candidate[key]
        if cand > base:
            flipped_right.append(key)
        elif cand < base:
            flipped_wrong.append(key)
        else:
            agreed += 1
    p_value = mcnemar_exact(len(flipped_wrong), len(flipped_right))
    return {
        "n_paired": agreed + len(flipped_right) + len(flipped_wrong),
        "agreed": agreed,
        "flipped_right": len(flipped_right),
        "flipped_wrong": len(flipped_wrong),
        "mcnemar_exact_p": p_value,
        "significant_at_05": p_value < 0.05,
        "flipped_right_questions": [f"{s}::{i}" for s, i in sorted(flipped_right)],
        "flipped_wrong_questions": [f"{s}::{i}" for s, i in sorted(flipped_wrong)],
    }


def compare_arms(
    baseline_runs: list[list[dict[str, Any]]],
    candidate_runs: list[list[dict[str, Any]]],
    *,
    skip_adversarial: bool = True,
) -> dict[str, Any]:
    baseline = _arm_scores(baseline_runs)
    candidate = _arm_scores(candidate_runs)
    categories = _category_by_question(baseline_runs + candidate_runs)
    keys = sorted(set(baseline) & set(candidate))
    if skip_adversarial:
        keys = [k for k in keys if categories.get(k, 0) != 5]

    overall = _flip_table(baseline, candidate, keys)
    by_category: dict[str, Any] = {}
    for cat in sorted({categories.get(k, 0) for k in keys}):
        cat_keys = [k for k in keys if categories.get(k, 0) == cat]
        by_category[str(cat)] = _flip_table(baseline, candidate, cat_keys)

    graph_stability = None
    if len(baseline_runs) == 1 and len(candidate_runs) == 1:
        graph_stability = graph_session_stability(
            baseline_runs[0],
            candidate_runs[0],
            skip_adversarial=skip_adversarial,
        )

    return {
        "baseline_runs": len(baseline_runs),
        "candidate_runs": len(candidate_runs),
        "baseline_accuracy": mean(baseline[k] for k in keys),
        "candidate_accuracy": mean(candidate[k] for k in keys),
        "overall": overall,
        "by_category": by_category,
        "graph_session_stability": graph_stability,
    }


_GRAPH_STABILITY_GATE = 0.95


def _graph_session_set(row: dict[str, Any]) -> frozenset[str]:
    raw = row.get("retrieved_session_ids_graph")
    if raw is None:
        raw = row.get("graph_session_ids")
    if not raw:
        return frozenset()
    return frozenset(str(s) for s in raw if s)


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("sample_id") or ""), int(row.get("qa_index") or 0))


def graph_session_stability(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    skip_adversarial: bool = True,
    gate: float = _GRAPH_STABILITY_GATE,
) -> dict[str, Any]:
    baseline_map = {_row_key(r): r for r in baseline_rows if not r.get("error")}
    candidate_map = {_row_key(r): r for r in candidate_rows if not r.get("error")}
    keys = sorted(set(baseline_map) & set(candidate_map))
    if skip_adversarial:
        keys = [
            k
            for k in keys
            if int(baseline_map[k].get("category") or 0) != 5
            and int(candidate_map[k].get("category") or 0) != 5
        ]

    identical = 0
    identical_coverage = 0
    coverage_comparable = 0
    disagreeing: list[str] = []
    for key in keys:
        base = baseline_map[key]
        cand = candidate_map[key]
        base_set = _graph_session_set(base)
        cand_set = _graph_session_set(cand)
        if base_set == cand_set:
            identical += 1
        else:
            disagreeing.append(f"{key[0]}::{key[1]}")
        base_cov = evidence_coverage(base, "graph")
        cand_cov = evidence_coverage(cand, "graph")
        if base_cov is not None and cand_cov is not None:
            coverage_comparable += 1
            if base_cov == cand_cov:
                identical_coverage += 1

    n = len(keys)
    agreement_rate = (identical / n) if n else None
    coverage_agreement_rate = (
        (identical_coverage / coverage_comparable) if coverage_comparable else None
    )
    return {
        "n_paired": n,
        "identical_session_sets": identical,
        "agreement_rate": agreement_rate,
        "identical_graph_coverage": identical_coverage,
        "n_coverage_comparable": coverage_comparable,
        "coverage_agreement_rate": coverage_agreement_rate,
        "gate": gate,
        "passes_gate": (
            agreement_rate is not None and agreement_rate >= gate
        ),
        "disagreeing_questions": disagreeing,
        "note": (
            "Do not A/B graph EvR below the gate: graph EvR is a measurement only "
            f"when ≥{gate:.0%} of questions have identical graph-session sets "
            "across two identical-config runs."
        ),
    }


def retrieval_arm_summary(arm: list[list[dict[str, Any]]]) -> dict[str, Any]:
    per_run = []
    for rows in arm:
        scored = [r for r in rows if not r.get("error")]
        metrics = aggregate_answers(scored)
        per_run.append(
            {
                "evidence_session_recall_full": metrics["evidence_session_recall_full"],
                "evidence_session_recall_by_channel": metrics[
                    "evidence_session_recall_by_channel"
                ],
                "answerable_rate": metrics["answerable_rate"],
            }
        )
    identical = all(run == per_run[0] for run in per_run) if per_run else True
    return {"per_run": per_run, "identical_across_runs": identical}


def selftest_metrics() -> list[str]:
    errors: list[str] = []
    if abs(f1_score("7 May 2023", "7 May 2023") - 1.0) > 1e-9:
        errors.append("identical answers should have F1=1")
    if f1_score("Paris", "London") != 0.0:
        errors.append("unrelated answers should have F1=0")
    if abs(bleu1_score("the cat", "cat") - 1.0) > 1e-9:
        errors.append("article stripping should allow BLEU-1=1")
    if normalize_answer("The Quick, Brown!") != "quick brown":
        errors.append("normalize_answer failed")
    lo, hi = wilson_ci(5, 10)
    if lo is None or hi is None or not (0.0 <= lo <= 0.5 <= hi <= 1.0):
        errors.append("wilson_ci(5,10) should contain 0.5")
    if evidence_session_ids(["D7:3", "D12:1"]) != {"session_7", "session_12"}:
        errors.append("evidence_session_ids failed")
    sample = {
        "gold": "Charlotte's Web",
        "text_context": "Melanie loves Charlotte's Web",
        "source_passages": [],
        "historical_context": [],
        "evidence": ["D7:1"],
        "retrieved_session_ids": ["session_7"],
    }
    if not gold_in_context(sample):
        errors.append("gold_in_context should detect gold tokens")
    if evidence_coverage(sample) != "full":
        errors.append("evidence_coverage should be full")

    if f1_score("", "") is not None:
        errors.append("empty prediction vs empty gold must not score F1")
    if bleu1_score("", "") is not None:
        errors.append("empty prediction vs empty gold must not score BLEU-1")
    if f1_score("Not mentioned", "") is not None:
        errors.append("empty gold must not be scorable by F1")
    if f1_score("", "Paris") != 0.0:
        errors.append("empty prediction vs non-empty gold should have F1=0")
    if overlap_scores("", "", adversarial=True) != (None, None):
        errors.append("adversarial items must not be scored by n-gram overlap")
    if overlap_scores("LGBTQ+ folks", "LGBTQ+ individuals", adversarial=True) != (
        None,
        None,
    ):
        errors.append("matching an adversarial trap answer must not score as overlap")

    channels = {
        "gold": "slipper",
        "graph_context": "Melanie | OWNS | slipper (session_3)",
        "triples": ["Melanie: Melanie | PUT | keys | IN | slipper"],
        "source_passages": ["[passage] Session id: session_9. Caroline said hi."],
        "historical_context": [],
        "evidence": ["D3:2"],
    }
    if retrieved_session_ids(channels, "graph") != {"session_3"}:
        errors.append("graph channel session ids failed")
    if retrieved_session_ids(channels, "passages") != {"session_9"}:
        errors.append("passage channel session ids failed")
    if evidence_coverage(channels, "graph") != "full":
        errors.append("graph channel evidence coverage should be full")
    if evidence_coverage(channels, "passages") != "none":
        errors.append("passage channel evidence coverage should be none")
    legacy = {"evidence": ["D3:2"], "retrieved_session_ids": ["session_3"]}
    if evidence_coverage(legacy, "graph") is not None:
        errors.append("graph coverage must be unknown for rows without a graph channel")

    if abs(mcnemar_exact(9, 20) - 0.06142835) > 1e-6:
        errors.append("mcnemar_exact(9, 20) should be ~0.0614")
    if abs(mcnemar_exact(10, 32) - 0.00094067) > 1e-7:
        errors.append("mcnemar_exact(10, 32) should be ~0.00094")
    if mcnemar_exact(0, 0) != 1.0:
        errors.append("mcnemar_exact with no discordant pairs should be 1.0")

    base_rows = [
        {"sample_id": "s", "qa_index": 0, "category": 1, "judge_correct": False},
        {"sample_id": "s", "qa_index": 1, "category": 1, "judge_correct": True},
    ]
    cand_rows = [
        {"sample_id": "s", "qa_index": 0, "category": 1, "judge_correct": True},
        {"sample_id": "s", "qa_index": 1, "category": 1, "judge_correct": True},
    ]
    comparison = compare_arms([base_rows], [cand_rows])
    if comparison["overall"]["flipped_right"] != 1:
        errors.append("compare_arms should count one flip to correct")
    if comparison["overall"]["flipped_wrong"] != 0:
        errors.append("compare_arms should count no flip to wrong")

    stable_a = [
        {
            "sample_id": "s",
            "qa_index": 0,
            "category": 1,
            "retrieved_session_ids_graph": ["session_1", "session_2"],
            "evidence": ["D1:1", "D2:1"],
            "judge_correct": True,
        },
        {
            "sample_id": "s",
            "qa_index": 1,
            "category": 4,
            "retrieved_session_ids_graph": ["session_3"],
            "evidence": ["D3:1"],
            "judge_correct": True,
        },
    ]
    stable_b = [
        {
            "sample_id": "s",
            "qa_index": 0,
            "category": 1,
            "graph_session_ids": ["session_2", "session_1"],
            "evidence": ["D1:1", "D2:1"],
            "judge_correct": True,
        },
        {
            "sample_id": "s",
            "qa_index": 1,
            "category": 4,
            "retrieved_session_ids_graph": ["session_3"],
            "evidence": ["D3:1"],
            "judge_correct": True,
        },
    ]
    stability = graph_session_stability(stable_a, stable_b)
    if stability["agreement_rate"] != 1.0 or not stability["passes_gate"]:
        errors.append("graph_session_stability should pass on identical sets")
    unstable_b = [
        dict(stable_b[0], retrieved_session_ids_graph=["session_9"]),
        stable_b[1],
    ]
    unstable = graph_session_stability(stable_a, unstable_b)
    if unstable["agreement_rate"] != 0.5:
        errors.append("graph_session_stability should report 50% on one disagreement")
    if "s::0" not in unstable["disagreeing_questions"]:
        errors.append("graph_session_stability should list disagreeing qa keys")
    return errors
