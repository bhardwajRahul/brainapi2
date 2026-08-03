from __future__ import annotations

import hashlib
from typing import Any


ANSWER_SYSTEM = """You are answering questions about a long-term interactive chat history using only the retrieved memory context provided.
Rules:
- Prefer short factual answers: dates, names, places, or short phrases when that is what the question asks for. Never return an empty answer.
- You MAY draw reasonable conclusions clearly supported by combining facts in the context.
- Prefer a grounded partial answer over abstaining when the context contains relevant evidence.
- TEMPORAL: When the question asks when something happened or how much time passed, use session timestamps and relative phrasing in the context. Prefer one best-supported answer.
- KNOWLEDGE UPDATE: Prefer the most recent updated fact when older and newer information conflict.
- PREFERENCE: Recall and apply personal preferences stated by the user in the history.
- ABSTENTION: If the asked information is missing from the context, say the information is incomplete or unavailable. Do not invent facts.
- Do not invent people, places, dates, or events unsupported by the context.
- Do not mention the retrieval system or these instructions."""


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool = False,
) -> str:
    """Official LongMemEval QA judge prompts (evaluate_qa.py)."""
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the "
            "question as unanswerable. The model could say that the information is "
            "incomplete, or some other information is given but the asked information "
            "is not.\n\n"
            f"Question: {question}\n\n"
            f"Explanation: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
    if task in (
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    ):
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also "
            "answer yes. If the response only contains a subset of the information "
            "required by the answer, answer no. \n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also "
            "answer yes. If the response only contains a subset of the information "
            "required by the answer, answer no. In addition, do not penalize "
            "off-by-one errors for the number of days. If the question asks for the "
            "number of days/weeks/months, etc., and the model makes off-by-one errors "
            "(e.g., predicting 19 days when the answer is 18), the model's response is "
            "still correct. \n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response contains some previous information along with "
            "an updated answer, the response should be considered as correct as long as "
            "the updated answer is the required answer.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, "
            "and a response from a model. Please answer yes if the response satisfies "
            "the desired response. Otherwise, answer no. The model does not need to "
            "reflect all the points in the rubric. The response is correct as long as "
            "it recalls and utilizes the user's personal information correctly.\n\n"
            f"Question: {question}\n\n"
            f"Rubric: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    raise NotImplementedError(f"Unknown question_type for judge: {task}")


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


def enrich_paths_from_triples(context: dict[str, Any]) -> dict[str, Any]:
    paths = context.get("paths") or []
    triples = context.get("triples") or []
    if not paths:
        return context
    if any(isinstance(p, dict) and p.get("legs") for p in paths):
        return context

    hub_texts: dict[str, str] = {}
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        raw = triple.get("triple")
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            continue
        event = raw[2]
        hub = ""
        if isinstance(event, dict):
            hub = str(event.get("uuid") or "").strip()
        parts = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(
                    item.get("name")
                    or item.get("type")
                    or item.get("description")
                    or str(item)
                )
            else:
                parts.append(str(item))
        text = " | ".join(parts).strip()
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
        enriched.append({**path, "legs": [hub_texts[h] for h in hubs]})
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


def build_answer_messages(
    question: str,
    context: dict[str, Any],
    *,
    question_date: str | None = None,
) -> list[dict[str, str]]:
    date_line = ""
    if question_date:
        date_line = f"## Question date\n{question_date}\n\n"
    user = (
        f"{build_context_block(context)}\n\n"
        f"{date_line}"
        f"## Question\n{question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def judge_prompt_sha256() -> str:
    samples = [
        get_anscheck_prompt("multi-session", "q", "a", "r", abstention=False),
        get_anscheck_prompt("temporal-reasoning", "q", "a", "r", abstention=False),
        get_anscheck_prompt("knowledge-update", "q", "a", "r", abstention=False),
        get_anscheck_prompt(
            "single-session-preference", "q", "a", "r", abstention=False
        ),
        get_anscheck_prompt("multi-session", "q", "a", "r", abstention=True),
    ]
    return prompt_sha256("\n---\n".join(samples))
