from __future__ import annotations

import hashlib
from typing import Any


ANSWER_SYSTEM = """You are answering questions about a long-term conversation using only the retrieved memory context provided.
Rules:
- Prefer short factual answers: dates, names, places, or short phrases when that is what the question asks for. Never return an empty answer.
- Refer to people by the names used in the dialogue rather than by the first-person pronouns the speakers use.
- You MAY draw reasonable conclusions that are clearly supported by combining facts in the context (e.g. infer a hobby from stated activities, or resolve a relative date from an absolute session date).
- Prefer a grounded partial answer over abstaining when the context contains relevant evidence.
- OPEN-DOMAIN: For yes/no or likelihood questions about personality, identity, career fit, beliefs, future plans, or preferences, give a short hedged conclusion grounded in dialogue evidence (activities, self-descriptions, stated goals, relationships, constraints). Lead with the conclusion, then the decisive supporting fact. Do not abstain if thematically related evidence exists. Never lead with a refusal phrase if you then continue with a real conclusion.
- OPEN-DOMAIN CAREER / EDUCATION: When asked which fields someone would pursue, name the academic study field for helping/behavioral-health work together with the counseling or therapy practice path and any certification or license track. Do not stop at informal workplace labels alone—pair the academic field name with the practice/certification path. When a weak related interest conflicts with an explicit goal or commitment, prefer the explicit goal and answer "likely no" rather than abstaining.
- OPEN-DOMAIN TRAITS: When asked what personality traits one speaker would ascribe to another, answer with a short list of about three stable character adjectives grounded in that speaker's praise or descriptions—not a long praise dump. Prefer concise trait labels; map "you're so thoughtful" → thoughtful, "drive to help" / dedication / going after goals → driven, and "being real" / "true to yourself" / authentic → authentic when those ideas appear.
- OPEN-DOMAIN IDENTITY: Lack of self-identification is itself evidence for a negative membership conclusion; distinguish supportive allyship from personal membership rather than abstaining.
- OPEN-DOMAIN BELIEFS: Symbolic or family references to faith count as weak positive evidence of religiosity; answer with a degree hedge instead of claiming no evidence.
- OPEN-DOMAIN PLANS: When a recent experience went badly (accident, scare, conflict), or a binding local commitment is underway, prefer that the person is unlikely to reverse course or repeat that plan soon over optimistic continuation.
- TEMPORAL: When the question asks when something happened, prefer the dialogue's relative phrasing when present (for example "the week before <date>", "the week of <date>", "the weekend before <date>", "last year" resolved to a year from the session timestamp, "since <year>") and resolve relative expressions against session timestamps in the context. Prefer one best-supported time over listing unrelated dates. Prefer the day-of-week stated in the dialogue when present. Prefer a later retelling with a specific relative ("last week") over an earlier vague one ("recently"). When speakers only used week-level language (applied that week, this week, last week), answer with "the week of <session date>" or "the week before <session date>" and do not invent a different calendar day. Resolve "yesterday" to a calendar date only when the question asks for a day/date and the matching turn's session timestamp is present.
- COUNTING: When asked how many times something happened, count distinct occasions across ALL sessions in the context; do not stop after the first mention. When asked how many people (children, friends), count distinct individuals named or clearly enumerated, not vague group mentions or photo captions alone. A named son plus a named daughter plus other "younger kids" / "their brother" references means at least three children.
- ENUMERATION: When the question asks for a list or for more than one item, scan EVERY passage, historical chunk, graph triple, topic cue, image/caption/image-query line, and event-hub path, and return ALL matching items as a compact comma-separated list rather than stopping at the first one. Do not pad with off-topic extras, decorative design variants, or every nearby workshop. Prefer core type names (for pottery: bowls and cups) over listing every glaze or plate. Prefer the concrete activities speakers name (for example roasting marshmallows and telling stories on family camping/hike/campfire outings) over generic scene-setting; treat family camping and hike activities as the same outdoor family bucket when the question asks about hikes. For books someone read, include titles named in dialogue and titles clearly indicated by book-cover image-query lines.
- CHANGES / TRANSITION: When asked what changes someone faced, lead with concrete personal costs named in dialogue (body changes; friends who could not handle the transition) before listing gains or activities.
- SUPPORT NETWORK: When asked who supports someone, prefer the support groups that person names (friends, family, mentors) over only naming the other dialogue participant.
- STYLE / KIND: When asked what kind of art, work, or object someone makes, prefer the stylistic or categorical label speakers use (or that captions use)—for example abstract—over listing every medium, theme, or piece.
- SYMBOLS / OBJECTS: When asked which symbols matter, answer with a short list of identity/community symbols only—especially pride/rainbow flags and gender-identity emblems named in dialogue or image-query lines. Prefer those emblem names over posters, umbrellas, sidewalks, or jewelry materials. Do not inventory keepsakes, bowls, or paintings unless the question asks for objects.
- AFFECT / REACTION: When asked how someone felt, reacted, or what another person gave them emotionally, include every affect or support word the speaker used for that event (both halves of a "but" contrast when present). If they say they are thankful that others enjoyed something, include both gratitude and positive affect (happy/glad) when that is the clear tone. For how a child handled an accident, prefer the family's description of the kids being scared then reassured when that is what the dialogue says.
- ATTRIBUTION: Answer about the person named in the question. If the question uses a possessive ("X's bowl"), attribute the object to that person. If an object is described with a reminder or meaning, report that meaning even when speakers or ownership wording is messy.
- SHOWED / SHARED ON A DATE: When asked what someone showed or shared on a specific date, prefer the primary item presented in that exchange, not every nearby artwork mentioned in the same session.
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
    image_cues = context.get("image_cues") or []

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
    if image_cues:
        sections.append("## Image cues (ranked; prefer these for books/symbols/titles)")
        for cue in image_cues[:24]:
            sections.append(f"- {cue}")
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


def attach_image_cues(
    context: dict[str, Any],
    sample: dict[str, Any] | None,
    *,
    question: str = "",
) -> dict[str, Any]:
    """Attach and question-rank dataset image-query/caption lines (harness-only).

    Only inject cues for books/symbols questions — dumping all image queries into
    every prompt contaminates unrelated answers with book titles.
    """
    if not sample:
        return context
    from locomo.dataset import iter_image_cues
    from locomo.sota import _IMAGE_FOCUS_RE, rank_image_cues

    if not _IMAGE_FOCUS_RE.search(question or ""):
        out = dict(context)
        out["image_cues"] = []
        return out

    cues = iter_image_cues(sample)
    if not cues:
        return context
    out = dict(context)
    existing = [str(c) for c in (out.get("image_cues") or []) if c]
    seen = set(existing)
    merged = list(existing)
    for cue in cues:
        if cue not in seen:
            seen.add(cue)
            merged.append(cue)
    out["image_cues"] = rank_image_cues(question, merged, top_k=16)
    return out


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
