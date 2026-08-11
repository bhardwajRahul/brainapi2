"""
Structural recommendation / synergies metrics (deterministic, no LLM judge).

Does not upsert LoCoMo/BEAM REPORTS.json. See docs/research/04 §4.9 and
docs/research/15 §6 (topology-aware eval).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def intra_list_diversity(label_sets: Sequence[Sequence[str]]) -> float:
    """Fraction of pairwise label-set Jaccard distances (1 - intersection/union)."""
    n = len(label_sets)
    if n < 2:
        return 0.0
    sets = [set(l.upper() for l in labels) for labels in label_sets]
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            union = sets[i] | sets[j]
            if not union:
                dist = 0.0
            else:
                dist = 1.0 - (len(sets[i] & sets[j]) / len(union))
            total += dist
            pairs += 1
    return total / pairs if pairs else 0.0


def type_coverage(label_sets: Sequence[Sequence[str]]) -> int:
    """Number of distinct labels appearing in the recommendation list."""
    covered = set()
    for labels in label_sets:
        covered.update(l.upper() for l in labels)
    return len(covered)


def unexpectedness_by_graph_distance(
    distances: Iterable[int],
    *,
    max_distance: int = 4,
) -> float:
    """
    Mean normalized hop distance from seed (higher => more unexpected).
    Missing / unknown distances contribute 0.
    """
    vals = list(distances)
    if not vals:
        return 0.0
    capped = [min(max(0, d), max_distance) / max_distance for d in vals]
    return sum(capped) / len(capped)


def popularity_stratum(degree: int, *, head_threshold: int = 20) -> str:
    if degree >= head_threshold:
        return "head"
    if degree <= 2:
        return "tail"
    return "torso"


def summarize_recommendation_list(
    label_sets: Sequence[Sequence[str]],
    distances: Sequence[int],
) -> dict:
    return {
        "n": len(label_sets),
        "intra_list_diversity": intra_list_diversity(label_sets),
        "type_coverage": type_coverage(label_sets),
        "unexpectedness": unexpectedness_by_graph_distance(distances),
    }
