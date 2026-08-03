from __future__ import annotations

from typing import Any

from beam.config import ABILITY_NAMES


def ability_score(row: dict[str, Any]) -> float | None:
    ability = str(row.get("ability") or "")
    if ability == "event_ordering":
        tau = row.get("tau_norm")
        if tau is not None:
            return float(tau)
    score = row.get("llm_judge_score")
    if score is None:
        return None
    return float(score)


def aggregate_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ability: dict[str, list[float]] = {name: [] for name in ABILITY_NAMES}
    latencies: list[float] = []
    total_tokens = 0
    for row in rows:
        ability = str(row.get("ability") or "")
        score = ability_score(row)
        if ability in by_ability and score is not None:
            by_ability[ability].append(score)
        if row.get("retrieve_latency_ms") is not None:
            latencies.append(float(row["retrieve_latency_ms"]))
        for key in ("answer_total_tokens", "judge_total_tokens"):
            if row.get(key) is not None:
                total_tokens += int(row[key])

    per_ability: dict[str, dict[str, Any]] = {}
    ability_means: list[float] = []
    for name in ABILITY_NAMES:
        values = by_ability[name]
        mean = sum(values) / len(values) if values else None
        per_ability[name] = {
            "n": len(values),
            "mean": mean,
        }
        if mean is not None:
            ability_means.append(mean)

    headline = (
        sum(ability_means) / len(ability_means) if ability_means else None
    )
    lat_sorted = sorted(latencies)
    mid = len(lat_sorted) // 2
    p50 = None
    p95 = None
    mean_lat = None
    if lat_sorted:
        p50 = (
            lat_sorted[mid]
            if len(lat_sorted) % 2
            else (lat_sorted[mid - 1] + lat_sorted[mid]) / 2
        )
        p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * (len(lat_sorted) - 1)))]
        mean_lat = sum(lat_sorted) / len(lat_sorted)

    return {
        "n_questions": len(rows),
        "headline_score": headline,
        "per_ability": per_ability,
        "n_abilities_scored": len(ability_means),
        "retrieval_latency_ms": {
            "p50": p50,
            "p95": p95,
            "mean": mean_lat,
            "n": len(lat_sorted),
        },
        "total_llm_tokens": total_tokens,
    }


def selftest_metrics() -> list[str]:
    errors: list[str] = []
    rows = [
        {
            "ability": "information_extraction",
            "llm_judge_score": 1.0,
            "retrieve_latency_ms": 10,
            "answer_total_tokens": 5,
            "judge_total_tokens": 5,
        },
        {
            "ability": "information_extraction",
            "llm_judge_score": 0.5,
            "retrieve_latency_ms": 20,
            "answer_total_tokens": 5,
            "judge_total_tokens": 5,
        },
        {
            "ability": "event_ordering",
            "llm_judge_score": 0.0,
            "tau_norm": 0.8,
            "retrieve_latency_ms": 30,
        },
        {
            "ability": "abstention",
            "llm_judge_score": 1.0,
        },
    ]
    metrics = aggregate_answers(rows)
    ie = metrics["per_ability"]["information_extraction"]["mean"]
    if ie != 0.75:
        errors.append(f"IE mean expected 0.75, got {ie}")
    eo = metrics["per_ability"]["event_ordering"]["mean"]
    if eo != 0.8:
        errors.append(f"event_ordering should use tau_norm=0.8, got {eo}")
    # headline = mean of scored ability means present in rows
    scored = [
        metrics["per_ability"][a]["mean"]
        for a in ABILITY_NAMES
        if metrics["per_ability"][a]["mean"] is not None
    ]
    expected = sum(scored) / len(scored)
    if metrics["headline_score"] != expected:
        errors.append(
            f"headline_score expected {expected}, got {metrics['headline_score']}"
        )
    if metrics["n_questions"] != 4:
        errors.append(f"n_questions expected 4, got {metrics['n_questions']}")
    return errors
