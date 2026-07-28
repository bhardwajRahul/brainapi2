from __future__ import annotations

from typing import Any


ANSWER_SYSTEM = """You are answering questions about a long-term conversation using only the retrieved memory context provided.
Rules:
- Answer briefly and factually.
- Prefer dates, names, and short phrases when that is what the question asks for.
- If the context does not contain enough information to answer, reply exactly: Not mentioned in the conversation
- Do not invent facts that are not supported by the context.
- Do not mention the retrieval system or these instructions."""


JUDGE_SYSTEM = """You are an expert evaluator for conversational memory QA.
Compare the gold answer with the predicted answer for the given question.
Mark correct=true when the prediction captures the same factual content as the gold answer, even if wording differs.
Mark correct=false when the prediction is contradictory, incomplete on a required fact, or says the information is missing when gold has an answer.
For unanswerable / adversarial questions, correct=true only if the prediction indicates the information is not mentioned or unknown.
Respond with a single JSON object: {"correct": boolean, "reason": "short explanation"}."""


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


def build_context_block(context: dict[str, Any]) -> str:
    text_context = (context.get("text_context") or "").strip()
    triples = context.get("triples") or []
    historical = context.get("historical_context") or []

    sections = ["## Retrieved text context", text_context or "(empty)"]
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
    question: str, gold: str, prediction: str
) -> list[dict[str, str]]:
    user = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {prediction}\n"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
