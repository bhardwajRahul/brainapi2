from __future__ import annotations

import hashlib
from typing import Any


ANSWER_SYSTEM = """You are answering questions about a long-term conversation using only the retrieved memory context provided.
Rules:
- Prefer short factual answers: dates, names, places, or short phrases when that is what the question asks for. Never return an empty answer.
- Refer to people by the names used in the dialogue rather than by the first-person pronouns the speakers use.
- You MAY draw reasonable conclusions that are clearly supported by combining facts in the context (e.g. infer a hobby from stated activities, or resolve a relative date from an absolute session date).
- Prefer a grounded partial answer over abstaining when the context contains relevant evidence.
- OPEN-DOMAIN: For yes/no or likelihood questions about personality, identity, career fit, or beliefs, give a short hedged conclusion grounded in dialogue evidence (activities, self-descriptions, stated goals). Do not abstain if thematically related evidence exists.
- TEMPORAL: When the question asks when something happened, prefer the dialogue's relative phrasing when present (for example "the week of <date>", "the weekend before <date>", "since <year>") and resolve relative expressions against session timestamps in the context. Prefer one best-supported time over listing unrelated dates.
- ENUMERATION: When the question asks for a list or for more than one item, scan EVERY passage, historical chunk, graph triple, topic cue, and event-hub path, and return ALL matching items as a compact comma-separated list rather than stopping at the first one.
- PATH COMPOSITION: When an "Event hub paths" section is present, each path links two event hubs through a shared entity. Read each hub as Actor → Event → Target/Context. Prefer those paths when composing answers that need more than one fact; do not invent hops beyond the given path legs and other retrieved context.
- TOPIC / EPISODE CUES: When a "Topics" or "Episodes" section is present, use it to find which sessions matter, then ground the answer in the cited passages and facts from those sessions.
- ABSTENTION: Reply exactly "Not mentioned in the conversation" ONLY when the context contains no topically related evidence at all. If any passage or triple is thematically related, give the best-supported answer instead of abstaining.
- Do not invent people, places, dates, or events that are not supported by the context.
- Do not mention the retrieval system or these instructions."""


JUDGE_SYSTEM = """You are an expert evaluator for conversational memory QA.
Compare the gold answer with the predicted answer for the given question.
Mark correct=true when the prediction captures the same factual content as the gold answer, even if wording differs.
Mark correct=false when the prediction is contradictory, incomplete on a required fact, or says the information is missing when gold has an answer.
For unanswerable / adversarial questions, correct=true only if the prediction indicates the information is not mentioned or unknown.
Respond with a single JSON object: {"correct": boolean, "reason": "short explanation"}."""

_ADVERSARIAL_NOTE = (
    "This question is adversarial: it is not answerable from the conversation, "
    "usually because it attributes something to the wrong speaker. The text given "
    "below is the trap answer a system produces when it accepts the false premise. "
    "Mark correct=true only when the prediction declines to answer, says the "
    "information is not in the conversation, or corrects the false premise. A "
    "prediction that matches the trap answer is incorrect."
)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def flatten_triple(triple_obj: Any) -> str:
    if isinstance(triple_obj, dict):
        identified = triple_obj.get("identified_entity", "")
        triple = triple_obj.get("triple")
        if isinstance(triple, (list, tuple)) and len(triple) >= 5:
            parts = []
            for item in triple:
                if isinstance(item, dict):
                    parts.append(
                        item.get("name")
                        or item.get("type")
                        or item.get("description")
                        or str(item)
                    )
                else:
                    parts.append(str(item))
            return f"{identified}: {' | '.join(parts)}" if identified else " | ".join(parts)
        return str(triple_obj)
    return str(triple_obj)


def flatten_path(path_obj: Any) -> str:
    if not isinstance(path_obj, dict):
        return str(path_obj)
    via = (
        str(path_obj.get("shared_entity_name") or "").strip()
        or str(path_obj.get("shared_entity") or "").strip()
        or "shared entity"
    )
    legs = path_obj.get("legs") or []
    leg_texts = [str(leg).strip() for leg in legs if str(leg).strip()]
    if leg_texts:
        return f"via {via}: " + " --> ".join(leg_texts)
    hubs = [str(h).strip() for h in (path_obj.get("hubs") or []) if str(h).strip()]
    if hubs:
        return f"via {via}: hubs {' + '.join(hubs)}"
    return f"via {via}"


def _triple_event_uuid(triple_obj: Any) -> str:
    if not isinstance(triple_obj, dict):
        return ""
    triple = triple_obj.get("triple")
    if not isinstance(triple, (list, tuple)) or len(triple) < 3:
        return ""
    event = triple[2]
    if isinstance(event, dict):
        return str(event.get("uuid") or "").strip()
    return ""


def _triple_fact_text(triple_obj: Any) -> str:
    if not isinstance(triple_obj, dict):
        return str(triple_obj)
    triple = triple_obj.get("triple")
    if isinstance(triple, (list, tuple)) and len(triple) >= 5:
        parts = []
        for item in triple:
            if isinstance(item, dict):
                parts.append(
                    item.get("name")
                    or item.get("type")
                    or item.get("description")
                    or str(item)
                )
            else:
                parts.append(str(item))
        return " | ".join(parts)
    return flatten_triple(triple_obj)


def enrich_paths_from_triples(context: dict[str, Any]) -> dict[str, Any]:
    """Attach readable path legs from triples when the API omitted them."""
    paths = context.get("paths") or []
    triples = context.get("triples") or []
    if not paths:
        return context
    if any(isinstance(p, dict) and p.get("legs") for p in paths):
        return context

    hub_texts: dict[str, str] = {}
    for triple in triples:
        hub = _triple_event_uuid(triple)
        text = _triple_fact_text(triple).strip()
        if hub and text and hub not in hub_texts:
            hub_texts[hub] = text
    if not hub_texts:
        return context

    enriched: list[dict[str, Any]] = []
    for path in paths:
        if not isinstance(path, dict):
            continue
        hubs = [str(h) for h in (path.get("hubs") or []) if h]
        if len(hubs) < 2 or not all(h in hub_texts for h in hubs):
            continue
        enriched.append(
            {
                **path,
                "legs": [hub_texts[h] for h in hubs],
            }
        )
    if not enriched:
        return context
    out = dict(context)
    out["paths"] = enriched
    return out


def build_context_block(context: dict[str, Any]) -> str:
    context = enrich_paths_from_triples(context)
    text_context = (context.get("text_context") or "").strip()
    triples = context.get("triples") or []
    paths = context.get("paths") or []
    historical = context.get("historical_context") or []
    source_passages = context.get("source_passages") or []
    topics = context.get("topics") or []
    episodes = context.get("episodes") or []

    sections = ["## Retrieved text context", text_context or "(empty)"]
    if topics:
        sections.append("## Topics")
        for topic in topics[:20]:
            if isinstance(topic, dict):
                label = topic.get("label") or topic.get("topic_id") or topic
                sessions = topic.get("sessions") or []
                sess = ", ".join(str(s) for s in sessions[:12])
                sections.append(f"- {label}" + (f" ({sess})" if sess else ""))
            else:
                sections.append(f"- {topic}")
    if episodes:
        sections.append("## Episodes")
        for episode in episodes[:20]:
            sections.append(f"- {episode}")
    if source_passages:
        sections.append("## Source passages")
        for passage in source_passages[:20]:
            sections.append(f"- {passage}")
    if paths:
        sections.append("## Event hub paths")
        for path in paths[:40]:
            sections.append(f"- {flatten_path(path)}")
    if triples:
        sections.append("## Graph triples")
        for t in triples[:80]:
            sections.append(f"- {flatten_triple(t)}")
    if historical:
        sections.append("## Historical context")
        for h in historical[:40]:
            sections.append(f"- {h}")
    return "\n".join(sections)


def build_answer_messages(question: str, context: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        f"{build_context_block(context)}\n\n"
        f"## Question\n{question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_judge_messages(
    question: str, gold: str, prediction: str, *, adversarial: bool = False
) -> list[dict[str, str]]:
    gold_label = "Trap answer" if adversarial else "Gold answer"
    user = (
        f"Question: {question}\n"
        f"{gold_label}: {gold}\n"
        f"Predicted answer: {prediction}\n"
    )
    if adversarial:
        user = f"{_ADVERSARIAL_NOTE}\n\n{user}"
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
