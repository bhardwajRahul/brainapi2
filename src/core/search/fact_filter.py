from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class FactFilterDecision(BaseModel):
    keep_indices: list[int] = Field(default_factory=list)


def filter_relevant_facts(
    question: str,
    candidates: list[str],
    *,
    llm_adapter: Any = None,
    max_keep: int = 40,
) -> list[int]:
    if not candidates:
        return []
    if len(candidates) <= max_keep and llm_adapter is None:
        return list(range(len(candidates)))

    indexed = "\n".join(f"{i}: {text}" for i, text in enumerate(candidates))
    prompt = (
        "You are a recognition-memory filter for a conversational memory system.\n"
        "Given a question and numbered candidate facts/passages, return the indices "
        "of items that are relevant evidence for answering the question.\n"
        "Prefer recall of useful evidence over aggressive pruning, but drop clear noise.\n"
        f"Keep at most {max_keep} indices.\n\n"
        f"Question: {question}\n\n"
        f"Candidates:\n{indexed}\n\n"
        'Respond with JSON: {"keep_indices": [0, 2, ...]}'
    )

    if llm_adapter is None:
        return list(range(min(max_keep, len(candidates))))

    try:
        response = llm_adapter.generate_structured(
            prompt,
            FactFilterDecision,
        )
        if isinstance(response, FactFilterDecision):
            indices = response.keep_indices
        elif isinstance(response, dict):
            indices = response.get("keep_indices") or []
        else:
            indices = getattr(response, "keep_indices", []) or []
        cleaned = sorted(
            {
                int(i)
                for i in indices
                if isinstance(i, (int, float, str))
                and str(i).isdigit()
                and 0 <= int(i) < len(candidates)
            }
        )
        if cleaned:
            return cleaned[:max_keep]
    except Exception:
        pass
    return list(range(min(max_keep, len(candidates))))


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            if not item:
                continue
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
