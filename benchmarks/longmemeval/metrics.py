from __future__ import annotations

import math
import string
from typing import Any, Iterable


_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    tokens = [t for t in text.split() if t and t not in _ARTICLES]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return normalize_answer(text).split()


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


def session_recall_coverage(row: dict[str, Any]) -> str | None:
    if row.get("is_abstention"):
        return None
    needed = {str(s) for s in (row.get("answer_session_ids") or []) if s}
    if not needed:
        return None
    have = {str(s) for s in (row.get("retrieved_session_ids") or []) if s}
    if needed <= have:
        return "full"
    if needed & have:
        return "partial"
    return "none"


def _bucket(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "n": 0,
            "judge_accuracy": None,
            "judge_accuracy_ci95": None,
            "session_recall_full": None,
            "session_recall_partial": None,
            "n_with_evidence": 0,
        }
    correct = [1.0 if r.get("judge_correct") else 0.0 for r in items]
    successes = int(sum(correct))
    lo, hi = wilson_ci(successes, len(items))
    coverages = [session_recall_coverage(r) for r in items]
    with_evidence = [c for c in coverages if c is not None]
    n_ev = len(with_evidence)
    full = sum(1 for c in with_evidence if c == "full")
    partial = sum(1 for c in with_evidence if c in {"full", "partial"})
    return {
        "n": len(items),
        "judge_accuracy": mean(correct),
        "judge_accuracy_ci95": (
            {"low": lo, "high": hi} if lo is not None and hi is not None else None
        ),
        "session_recall_full": (full / n_ev) if n_ev else None,
        "session_recall_partial": (partial / n_ev) if n_ev else None,
        "n_with_evidence": n_ev,
    }


def aggregate_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qtype = str(row.get("question_type") or "unknown")
        by_type.setdefault(qtype, []).append(row)

    all_bucket = _bucket(rows)
    abstention_rows = [r for r in rows if r.get("is_abstention")]
    non_abs = [r for r in rows if not r.get("is_abstention")]

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
        "n_questions": len(rows),
        "n_abstention": len(abstention_rows),
        "n_non_abstention": len(non_abs),
        "headline_judge_accuracy": all_bucket["judge_accuracy"],
        "headline_judge_accuracy_ci95": all_bucket["judge_accuracy_ci95"],
        "session_recall_full": _bucket(non_abs)["session_recall_full"],
        "session_recall_partial": _bucket(non_abs)["session_recall_partial"],
        "n_with_evidence": _bucket(non_abs)["n_with_evidence"],
        "abstention": {
            "n": len(abstention_rows),
            "accuracy": _bucket(abstention_rows)["judge_accuracy"],
        },
        "by_type": {
            qtype: _bucket(items) for qtype, items in sorted(by_type.items())
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
    from longmemeval.dataset import brain_id_for, format_session, is_abstention
    from longmemeval.judge import parse_yes_no
    from longmemeval.prompts import get_anscheck_prompt

    if brain_id_for("e47b4ab3aa_abs") != "lmee47b4ab3aaabs":
        errors.append("brain_id_for should strip non-alnum and prefix lme")
    if not is_abstention("abc_abs"):
        errors.append("is_abstention should detect _abs suffix")
    if is_abstention("abc"):
        errors.append("non-abs question_id should not be abstention")

    text = format_session(
        "sess-1",
        "2023/05/07",
        [{"role": "user", "content": "hi", "has_answer": True}],
    )
    if "has_answer" in text:
        errors.append("format_session must strip has_answer labels")
    if "Session id: sess-1" not in text:
        errors.append("format_session must include session id")

    abs_prompt = get_anscheck_prompt(
        "multi-session", "q", "expl", "resp", abstention=True
    )
    if "unanswerable" not in abs_prompt.lower():
        errors.append("abstention judge prompt missing")
    temp_prompt = get_anscheck_prompt(
        "temporal-reasoning", "q", "a", "r", abstention=False
    )
    if "off-by-one" not in temp_prompt:
        errors.append("temporal judge prompt missing off-by-one clause")
    ku_prompt = get_anscheck_prompt(
        "knowledge-update", "q", "a", "r", abstention=False
    )
    if "updated answer" not in ku_prompt:
        errors.append("knowledge-update judge prompt missing")
    pref_prompt = get_anscheck_prompt(
        "single-session-preference", "q", "rubric", "r", abstention=False
    )
    if "Rubric:" not in pref_prompt:
        errors.append("preference judge prompt missing Rubric")

    if not parse_yes_no("Yes"):
        errors.append("parse_yes_no Yes failed")
    if parse_yes_no("No"):
        errors.append("parse_yes_no No failed")

    full = session_recall_coverage(
        {
            "is_abstention": False,
            "answer_session_ids": ["a", "b"],
            "retrieved_session_ids": ["a", "b", "c"],
        }
    )
    if full != "full":
        errors.append(f"expected full recall, got {full}")
    partial = session_recall_coverage(
        {
            "is_abstention": False,
            "answer_session_ids": ["a", "b"],
            "retrieved_session_ids": ["a"],
        }
    )
    if partial != "partial":
        errors.append(f"expected partial recall, got {partial}")
    none = session_recall_coverage(
        {
            "is_abstention": False,
            "answer_session_ids": ["a"],
            "retrieved_session_ids": ["z"],
        }
    )
    if none != "none":
        errors.append(f"expected none recall, got {none}")
    skipped = session_recall_coverage(
        {"is_abstention": True, "answer_session_ids": ["a"], "retrieved_session_ids": []}
    )
    if skipped is not None:
        errors.append("abstention rows should skip session recall")

    agg = aggregate_answers(
        [
            {
                "question_type": "multi-session",
                "judge_correct": True,
                "is_abstention": False,
                "answer_session_ids": ["a"],
                "retrieved_session_ids": ["a"],
                "retrieve_latency_ms": 10,
                "answer_total_tokens": 5,
                "judge_total_tokens": 2,
            },
            {
                "question_type": "multi-session",
                "judge_correct": False,
                "is_abstention": False,
                "answer_session_ids": ["a"],
                "retrieved_session_ids": [],
                "retrieve_latency_ms": 20,
                "answer_total_tokens": 5,
                "judge_total_tokens": 2,
            },
            {
                "question_type": "multi-session",
                "judge_correct": True,
                "is_abstention": True,
                "answer_session_ids": [],
                "retrieved_session_ids": [],
            },
        ]
    )
    if agg["n_questions"] != 3:
        errors.append("aggregate n_questions wrong")
    if abs(float(agg["headline_judge_accuracy"]) - (2 / 3)) > 1e-9:
        errors.append("headline accuracy should include abstention")
    if abs(float(agg["session_recall_full"]) - 0.5) > 1e-9:
        errors.append("session recall should skip abstention and average non-abs")
    if agg["by_type"]["multi-session"]["n"] != 3:
        errors.append("by_type count wrong")

    return errors
