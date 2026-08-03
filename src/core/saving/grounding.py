from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence


_WHITESPACE_RE = re.compile(r"\s+")

GroundingVerdict = Literal["accept", "reject", "ambiguous"]


def normalize_span(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip().lower())


def extract_source_span(
    rel: Any,
    source_text: str,
    *,
    allow_synthetic: bool = True,
) -> str:
    props = getattr(rel, "properties", None) or {}
    if isinstance(props, dict):
        for key in ("source_span", "span", "evidence", "quote"):
            value = props.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    explicit = getattr(rel, "source_span", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    description = getattr(rel, "description", None) or ""
    if description and source_text:
        score, aligned = align_span(description, source_text)
        if score >= 0.55 and aligned:
            return aligned
    if description.strip():
        return description.strip()
    if not allow_synthetic:
        return ""
    name = getattr(rel, "name", None) or ""
    tail = getattr(getattr(rel, "tail", None), "name", "") or ""
    tip = getattr(getattr(rel, "tip", None), "name", "") or ""
    return " ".join(p for p in (tail, name, tip) if p).strip()


def extract_explicit_evidence(rel: Any, source_text: str) -> str:
    """Evidence quote/description only — never invent from endpoint names."""
    return extract_source_span(rel, source_text, allow_synthetic=False)


def align_span(span: str, source_text: str) -> tuple[float, str]:
    """
    Cheap string alignment: exact / containment / token Jaccard against windows.
    Returns (score in [0,1], best matching source window or "").
    """
    needle = normalize_span(span)
    hay = normalize_span(source_text)
    if not needle or not hay:
        return 0.0, ""
    if needle in hay:
        return 1.0, span.strip()
    needle_tokens = [t for t in needle.split(" ") if t]
    if not needle_tokens:
        return 0.0, ""
    window = max(len(needle_tokens) + 4, 8)
    hay_tokens = hay.split(" ")
    best_score = 0.0
    best_window = ""
    needle_set = set(needle_tokens)
    for i in range(0, max(1, len(hay_tokens) - window + 1)):
        chunk = hay_tokens[i : i + window]
        chunk_set = set(chunk)
        if not chunk_set:
            continue
        score = len(needle_set & chunk_set) / len(needle_set | chunk_set)
        if score > best_score:
            best_score = score
            best_window = " ".join(chunk)
    return best_score, best_window


def find_span_offsets(
    span: str,
    source_text: str,
) -> Optional[tuple[int, int]]:
    """
    Locate `span` inside `source_text` with exact then whitespace-tolerant match.
    Returns (start, end) character offsets into the original source, or None.
    """
    text = source_text or ""
    needle = (span or "").strip()
    if not needle or not text:
        return None
    exact = text.find(needle)
    if exact >= 0:
        return exact, exact + len(needle)

    # Whitespace-tolerant: map normalized needle back onto original offsets.
    norm_needle = normalize_span(needle)
    if not norm_needle:
        return None
    lower = text.lower()
    # Build map from normalized index -> original index
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    prev_space = False
    for i, ch in enumerate(lower):
        if ch.isspace():
            if norm_chars and not prev_space:
                norm_chars.append(" ")
                norm_to_orig.append(i)
            prev_space = True
            continue
        if not norm_chars and ch.isspace():
            continue
        norm_chars.append(ch)
        norm_to_orig.append(i)
        prev_space = False
    norm_hay = "".join(norm_chars).strip()
    # Trim leading spaces recorded above
    while norm_to_orig and norm_hay and norm_hay[0] == " ":
        norm_hay = norm_hay[1:]
        norm_to_orig = norm_to_orig[1:]
    idx = norm_hay.find(norm_needle)
    if idx < 0:
        return None
    start_orig = norm_to_orig[idx]
    end_idx = idx + len(norm_needle) - 1
    if end_idx >= len(norm_to_orig):
        return None
    end_orig = norm_to_orig[end_idx] + 1
    return start_orig, end_orig


def realign_span_offsets(
    *,
    span: str,
    source_text: str,
    span_start: Optional[int] = None,
    span_end: Optional[int] = None,
    min_score: float = 0.45,
) -> tuple[bool, Optional[int], Optional[int], str]:
    """
    Validate or repair character offsets for a source span.

    Returns (ok, new_start, new_end, reason). On success with repaired or
    cleared offsets, reason is "". On failure, reason is a machine-readable code.
    """
    text = source_text or ""
    needle = (span or "").strip()
    if not needle:
        return False, span_start, span_end, "missing_source_span"

    score, _ = align_span(needle, text)
    if score < min_score:
        return False, span_start, span_end, "source_span_not_grounded"

    if span_start is None or span_end is None:
        found = find_span_offsets(needle, text)
        if found:
            return True, found[0], found[1], ""
        return True, None, None, ""

    if span_start < 0 or span_end > len(text) or span_start >= span_end:
        found = find_span_offsets(needle, text)
        if found:
            return True, found[0], found[1], ""
        # Offsets unusable but span text is grounded — clear offsets.
        return True, None, None, ""

    sliced = text[span_start:span_end]
    if sliced.strip() and normalize_span(sliced) == normalize_span(needle):
        return True, span_start, span_end, ""

    slice_score, _ = align_span(needle, sliced)
    if slice_score >= min_score:
        return True, span_start, span_end, ""

    found = find_span_offsets(needle, text)
    if found:
        return True, found[0], found[1], ""

    # Soften: keep the grounded quote, drop bad offsets instead of rejecting.
    return True, None, None, ""


def relationship_looks_grounded(
    rel: Any,
    source_text: str,
    *,
    min_score: float = 0.45,
) -> tuple[bool, float, str]:
    span = extract_explicit_evidence(rel, source_text)
    score, aligned = align_span(span, source_text)
    if not span:
        return False, 0.0, ""
    return score >= min_score, score, aligned or span


def endpoints_well_formed(rel: Any) -> bool:
    for endpoint in (getattr(rel, "tail", None), getattr(rel, "tip", None)):
        if endpoint is None:
            return False
        name = (getattr(endpoint, "name", None) or "").strip()
        typ = (getattr(endpoint, "type", None) or "").strip()
        if not name or not typ:
            return False
        if name.lower() == typ.lower():
            return False
    predicate = (getattr(rel, "name", None) or "").strip()
    return bool(predicate)


def _endpoint_is_type_named(endpoint: Any) -> bool:
    if endpoint is None:
        return True
    name = (getattr(endpoint, "name", None) or "").strip()
    typ = (getattr(endpoint, "type", None) or "").strip()
    if not name or not typ:
        return True
    return name.lower() == typ.lower()


@dataclass
class GroundingDecision:
    decision: GroundingVerdict
    reason: str
    score: float = 0.0
    span: str = ""
    relationship: Any = None


@dataclass
class JanitorTriageResult:
    accept: list[Any] = field(default_factory=list)
    reject: list[GroundingDecision] = field(default_factory=list)
    ambiguous: list[Any] = field(default_factory=list)


def decide_relationship_grounding(
    rel: Any,
    source_text: str,
    *,
    min_score: float = 0.45,
    reject_below: float = 0.15,
) -> GroundingDecision:
    """
    Classify one relationship as accept / reject / ambiguous for Janitor triage.
    """
    if _endpoint_is_type_named(getattr(rel, "tail", None)) or _endpoint_is_type_named(
        getattr(rel, "tip", None)
    ):
        return GroundingDecision(
            decision="reject",
            reason="type_named_placeholder_endpoint",
            relationship=rel,
        )
    if not endpoints_well_formed(rel):
        return GroundingDecision(
            decision="reject",
            reason="malformed_endpoints",
            relationship=rel,
        )
    predicate = (getattr(rel, "name", None) or "").strip()
    if not predicate:
        return GroundingDecision(
            decision="reject",
            reason="missing_predicate",
            relationship=rel,
        )

    span = extract_explicit_evidence(rel, source_text)
    score, aligned = align_span(span, source_text) if span else (0.0, "")
    if not span:
        return GroundingDecision(
            decision="ambiguous",
            reason="missing_source_span",
            score=0.0,
            relationship=rel,
        )
    if score >= min_score:
        if isinstance(getattr(rel, "properties", None), dict) and aligned:
            props = rel.properties or {}
            if not props.get("source_span"):
                try:
                    rel.properties = {
                        **props,
                        "source_span": aligned,
                        "grounding_score": score,
                    }
                except Exception:
                    pass
        return GroundingDecision(
            decision="accept",
            reason="grounded_endpoints_ok",
            score=score,
            span=aligned or span,
            relationship=rel,
        )
    if score < reject_below:
        return GroundingDecision(
            decision="reject",
            reason="source_span_not_grounded",
            score=score,
            span=span,
            relationship=rel,
        )
    return GroundingDecision(
        decision="ambiguous",
        reason="weak_grounding",
        score=score,
        span=span,
        relationship=rel,
    )


def triage_relationships_for_janitor(
    relationships: Sequence[Any],
    source_text: str,
    *,
    min_grounding: float = 0.45,
    reject_below: float = 0.15,
) -> JanitorTriageResult:
    """
    Split relationships into accept (skip LLM), reject (drop), ambiguous (LLM).
    """
    result = JanitorTriageResult()
    for rel in relationships:
        decision = decide_relationship_grounding(
            rel,
            source_text,
            min_score=min_grounding,
            reject_below=reject_below,
        )
        if decision.decision == "accept":
            result.accept.append(rel)
        elif decision.decision == "reject":
            result.reject.append(decision)
        else:
            result.ambiguous.append(rel)
    return result


def cheap_janitor_precheck(
    relationships: Sequence[Any],
    source_text: str,
    *,
    min_grounding: float = 0.45,
) -> tuple[list[Any], list[Any]]:
    """
    Split relationships into (skip_janitor, need_janitor).

    Rejects from triage are folded into need_janitor for backward compatibility;
    prefer triage_relationships_for_janitor for explicit drop auditing.
    """
    triage = triage_relationships_for_janitor(
        relationships, source_text, min_grounding=min_grounding
    )
    need = list(triage.ambiguous)
    need.extend(d.relationship for d in triage.reject if d.relationship is not None)
    return list(triage.accept), need
