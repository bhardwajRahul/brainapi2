"""Compact prior-state scratchpad for Architect batch extract.

Replaces replaying raw prior unit text with a token-capped summary of
entities, open event hubs, dates, and recent predicate/span pointers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from src.core.saving.ingest_cost import count_source_tokens

DEFAULT_SCRATCHPAD_TOKEN_CAP = 500
DEFAULT_PRIOR_UNIT_WINDOW = 4
_SPAN_POINTER_CHARS = 72
_DESC_CHARS = 60
_MAX_ENTITIES = 40
_MAX_HUBS = 16
_MAX_PREDICATES = 24
_MAX_DATES = 12


@dataclass
class ScratchpadEntity:
    uuid: str
    name: str
    type: str
    happened_at: Optional[str] = None


@dataclass
class ScratchpadEventHub:
    uuid: str
    name: str
    happened_at: Optional[str] = None
    description: Optional[str] = None
    actors: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)


@dataclass
class ScratchpadPredicate:
    predicate: str
    tail: str
    tip: str
    span_pointer: Optional[str] = None


@dataclass
class ArchitectScratchpad:
    entities: list[ScratchpadEntity] = field(default_factory=list)
    event_hubs: list[ScratchpadEventHub] = field(default_factory=list)
    recent_predicates: list[ScratchpadPredicate] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    raw_span_fetches: int = 0


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _truncate(text: Optional[str], limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _is_event(typ: Optional[str]) -> bool:
    return (typ or "").strip().lower() == "event"


def _entity_from_any(obj: Any) -> Optional[ScratchpadEntity]:
    name = (_attr(obj, "name") or "").strip()
    typ = (_attr(obj, "type") or "").strip()
    if not name or not typ:
        return None
    if name.lower() == typ.lower():
        return None
    uuid_ = str(_attr(obj, "uuid") or "").strip()
    happened = _attr(obj, "happened_at")
    return ScratchpadEntity(
        uuid=uuid_,
        name=name,
        type=typ,
        happened_at=(str(happened).strip() if happened else None) or None,
    )


def _span_pointer_from_rel(rel: Any) -> Optional[str]:
    props = _attr(rel, "properties") or {}
    if not isinstance(props, dict):
        props = {}
    span = props.get("source_span") or _attr(rel, "source_span") or ""
    span = str(span).strip()
    if not span:
        return None
    return _truncate(span, _SPAN_POINTER_CHARS)


def build_scratchpad(
    prior_entities: Sequence[Any],
    prior_relationships: Sequence[Any] = (),
    *,
    max_entities: int = _MAX_ENTITIES,
    max_hubs: int = _MAX_HUBS,
    max_predicates: int = _MAX_PREDICATES,
) -> ArchitectScratchpad:
    """Build a compact prior-state scratchpad from Scout entities + Architect rels."""
    entities: list[ScratchpadEntity] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_uuids: set[str] = set()
    dates: list[str] = []
    seen_dates: set[str] = set()

    for raw in prior_entities:
        ent = _entity_from_any(raw)
        if ent is None:
            continue
        key = (ent.name.lower(), ent.type.lower())
        if ent.uuid and ent.uuid in seen_uuids:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if ent.uuid:
            seen_uuids.add(ent.uuid)
        entities.append(ent)
        if ent.happened_at and ent.happened_at not in seen_dates:
            seen_dates.add(ent.happened_at)
            dates.append(ent.happened_at)

    entities = entities[-max_entities:]

    hubs_by_key: dict[str, ScratchpadEventHub] = {}
    predicates: list[ScratchpadPredicate] = []

    for rel in prior_relationships:
        tail = _entity_from_any(_attr(rel, "tail"))
        tip = _entity_from_any(_attr(rel, "tip"))
        pred = (_attr(rel, "name") or "").strip()
        if not pred or tail is None or tip is None:
            continue

        for endpoint in (tail, tip):
            if endpoint.happened_at and endpoint.happened_at not in seen_dates:
                seen_dates.add(endpoint.happened_at)
                dates.append(endpoint.happened_at)
            props = _attr(_attr(rel, "tail" if endpoint is tail else "tip"), "properties")
            if isinstance(props, dict):
                ha = props.get("happened_at")
                if ha and str(ha) not in seen_dates:
                    seen_dates.add(str(ha))
                    dates.append(str(ha))

        span_ptr = _span_pointer_from_rel(rel)
        predicates.append(
            ScratchpadPredicate(
                predicate=pred,
                tail=tail.name,
                tip=tip.name,
                span_pointer=span_ptr,
            )
        )

        for endpoint, other, role in (
            (tip, tail, "actor"),
            (tail, tip, "object"),
        ):
            if not _is_event(endpoint.type):
                continue
            hub_key = endpoint.uuid or f"{endpoint.name.lower()}|{endpoint.type.lower()}"
            hub = hubs_by_key.get(hub_key)
            if hub is None:
                desc = _attr(
                    _attr(rel, "tip" if endpoint is tip else "tail"),
                    "description",
                )
                hub = ScratchpadEventHub(
                    uuid=endpoint.uuid,
                    name=endpoint.name,
                    happened_at=endpoint.happened_at,
                    description=_truncate(desc, _DESC_CHARS) or None,
                )
                hubs_by_key[hub_key] = hub
            if role == "actor":
                if other.name not in hub.actors:
                    hub.actors.append(other.name)
            else:
                if other.name not in hub.objects:
                    hub.objects.append(other.name)

    predicates = predicates[-max_predicates:]
    hubs = list(hubs_by_key.values())[-max_hubs:]
    dates = dates[-_MAX_DATES:]

    return ArchitectScratchpad(
        entities=entities,
        event_hubs=hubs,
        recent_predicates=predicates,
        dates=dates,
    )


def serialize_scratchpad(
    pad: ArchitectScratchpad,
    *,
    token_cap: int = DEFAULT_SCRATCHPAD_TOKEN_CAP,
) -> tuple[str, int]:
    """Serialize scratchpad under a reference-token cap. Returns (text, tokens)."""
    cap = max(32, int(token_cap or DEFAULT_SCRATCHPAD_TOKEN_CAP))
    sections: list[str] = []

    if pad.entities:
        lines = ["Entities:"]
        for ent in pad.entities:
            bits = [ent.name, ent.type]
            if ent.uuid:
                bits.append(ent.uuid[:8])
            if ent.happened_at:
                bits.append(f"@{ent.happened_at}")
            lines.append("- " + " | ".join(bits))
        sections.append("\n".join(lines))

    if pad.event_hubs:
        lines = ["Open event hubs:"]
        for hub in pad.event_hubs:
            head = hub.name
            if hub.happened_at:
                head += f" @{hub.happened_at}"
            detail_parts = []
            if hub.actors:
                detail_parts.append("actors=" + ", ".join(hub.actors[:4]))
            if hub.objects:
                detail_parts.append("objects=" + ", ".join(hub.objects[:4]))
            if hub.description:
                detail_parts.append(hub.description)
            line = f"- {head}"
            if detail_parts:
                line += " (" + "; ".join(detail_parts) + ")"
            lines.append(line)
        sections.append("\n".join(lines))

    if pad.dates:
        sections.append("Dates: " + ", ".join(pad.dates))

    if pad.recent_predicates:
        lines = ["Recent predicates:"]
        for pred in pad.recent_predicates:
            line = f"- {pred.tail} --{pred.predicate}--> {pred.tip}"
            if pred.span_pointer:
                line += f' [{pred.span_pointer}]'
            lines.append(line)
        sections.append("\n".join(lines))

    if not sections:
        return "", 0

    # Drop trailing sections until under cap (prefer keep entities + hubs).
    while sections:
        text = "\n".join(sections)
        tokens, _, _ = count_source_tokens(text)
        if tokens <= cap:
            return text, tokens
        if len(sections) == 1:
            # Hard-trim the last remaining section by lines.
            lines = sections[0].split("\n")
            while len(lines) > 1:
                lines.pop()
                text = "\n".join(lines)
                tokens, _, _ = count_source_tokens(text)
                if tokens <= cap:
                    return text, tokens
            return "", 0
        sections.pop()

    return "", 0


def fetch_prior_span(
    prior_chunks: Sequence[str],
    needle: str,
    *,
    window_chars: int = 200,
) -> Optional[str]:
    """On-demand: locate `needle` in prior unit text and return a small window."""
    query = (needle or "").strip()
    if not query or not prior_chunks:
        return None
    window = max(40, int(window_chars or 200))
    for chunk in reversed(list(prior_chunks)):
        text = chunk or ""
        idx = text.find(query)
        if idx < 0:
            idx = text.lower().find(query.lower())
        if idx < 0:
            continue
        start = max(0, idx - window // 4)
        end = min(len(text), idx + len(query) + window // 2)
        return text[start:end].strip()
    return None


def format_unit_with_scratchpad(current_chunk: str, scratchpad_text: str) -> str:
    """Compose Architect unit prompt text: scratchpad preamble + current unit."""
    if not (scratchpad_text or "").strip():
        return current_chunk
    return (
        f"[Prior scratchpad]\n{scratchpad_text.strip()}\n\n"
        f"[Current unit]\n{current_chunk}"
    )


def resolve_prior_context_mode(
    configured: str | None,
    architect_mode: str,
) -> str:
    """
    Resolve prior-context strategy.

    auto (default): scratchpad for batch/schema, raw for tooler.
    """
    value = (configured or "auto").strip().lower()
    if value in ("scratchpad", "raw"):
        return value
    if value in ("auto", ""):
        if (architect_mode or "").strip().lower() in ("batch", "schema"):
            return "scratchpad"
        return "raw"
    raise ValueError(
        f"Invalid INGEST_ARCHITECT_PRIOR_CONTEXT: {configured!r}. "
        "Expected 'auto', 'scratchpad', or 'raw'."
    )
