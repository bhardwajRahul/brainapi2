from __future__ import annotations

import math
import string
from collections import Counter
from typing import Any, Iterable


_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    tokens = [t for t in text.split() if t and t not in _ARTICLES]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return normalize_answer(text).split()


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 1.0 if not pred_tokens and not gold_tokens else 0.0
    gold_counts = Counter(gold_tokens)
    match = 0
    for token in pred_tokens:
        if gold_counts[token] > 0:
            match += 1
            gold_counts[token] -= 1
    precision = match / len(pred_tokens)
    brevity = min(1.0, math.exp(1 - len(gold_tokens) / len(pred_tokens)))
    return precision * brevity


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


def aggregate_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        cat = int(row.get("category") or 0)
        by_category.setdefault(cat, []).append(row)

    def _bucket(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {
                "n": 0,
                "judge_accuracy": None,
                "judge_accuracy_ci95": None,
                "mean_f1": None,
                "mean_bleu1": None,
            }
        correct = [1.0 if r.get("judge_correct") else 0.0 for r in items]
        successes = int(sum(correct))
        lo, hi = wilson_ci(successes, len(items))
        return {
            "n": len(items),
            "judge_accuracy": mean(correct),
            "judge_accuracy_ci95": (
                {"low": lo, "high": hi} if lo is not None and hi is not None else None
            ),
            "mean_f1": mean(float(r.get("f1") or 0.0) for r in items),
            "mean_bleu1": mean(float(r.get("bleu1") or 0.0) for r in items),
        }

    non_adv = [r for r in rows if int(r.get("category") or 0) != 5]
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

    return {
        "n_total": len(all_rows),
        "n_non_adversarial": len(non_adv),
        "headline_judge_accuracy": non_adv_bucket["judge_accuracy"],
        "headline_judge_accuracy_ci95": non_adv_bucket["judge_accuracy_ci95"],
        "overall_judge_accuracy": all_bucket["judge_accuracy"],
        "overall_judge_accuracy_ci95": all_bucket["judge_accuracy_ci95"],
        "mean_f1_non_adversarial": non_adv_bucket["mean_f1"],
        "mean_bleu1_non_adversarial": non_adv_bucket["mean_bleu1"],
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
    return errors
