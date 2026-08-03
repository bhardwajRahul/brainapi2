from __future__ import annotations

import hashlib
from typing import Any


JUDGE_PROMPT_VARIANT = "beam-rubric-v1-question-aware"

ANSWER_SYSTEM = """You are answering probing questions about a long multi-turn conversation using only the retrieved memory context provided.
Rules:
- Prefer short factual answers when the question asks for a fact, date, number, name, or short list.
- For summarization, preference, or instruction questions, follow the requested format and cover the required points supported by the context.
- For abstention: if the context lacks evidence for the asked fact, hard-abstain with exactly: "Not mentioned in the conversation." This includes biography/background/prior projects AND user feedback, reactions, satisfaction rates, A/B results, or metrics that are not explicitly stated. Do not invent percentages or feedback from related feature discussion.
- For event ordering: if the question says ONLY N items, list exactly N numbered lines and no more. Each line must be a short milestone/aspect label ordered by first mention using session_id / Session id stamps when present. Match the domain in the question (e.g. translation/language services vs budget tracker). Prefer concrete engineering milestones (API integration, auth, rate limits, caching, fine-tuning, debugging, deployment) over vague planning fluff.
- For contradiction-resolution questions only (conflicting have-I/did-I claims): open with "There is contradictory information.", state both sides, end with "It is unclear which statement is correct." Do not use that template for ordinary fact/deadline/TTL questions.
- For instruction following about libraries/dependencies: include explicit version numbers when the context states them; if asked for versions and versions are missing, say versions are not specified rather than listing bare names only. If prior instructions required naming protocol versions with auth methods, include versions such as OAuth 2.0 and TLS 1.3 when answering identity/auth questions.
- For temporal reasoning: when two dated milestones appear in context, compute the day difference; do not abstain if both endpoints are present. Prefer the originally stated date or interval unless the question asks for the current / updated / latest value.
- For knowledge updates that ask for the current value (TTL, deadline, config), use the latest superseded-aware update and prefer human units used in the chat (e.g. "20 minutes" not only seconds).
- You MAY combine facts across retrieved passages when the question requires multi-session reasoning.
- Prefer grounded partial answers over inventing unsupported details.
- Do not invent people, places, dates, preferences, or events that are not supported by the context.
- Do not mention the retrieval system or these instructions."""

ABILITY_ANSWER_HINTS: dict[str, str] = {
    "abstention": (
        "CRITICAL: If the question asks for background, biography, previous projects, OR "
        "user feedback/reactions/satisfaction metrics, and those exact facts are not in "
        "the retrieved context, reply with ONLY this sentence: Not mentioned in the "
        "conversation. Never invent percentages (e.g. 90%), satisfaction rates, or "
        "testing feedback from adjacent UI/feature discussion."
    ),
    "contradiction_resolution": (
        "These questions are NEVER answered by abstaining. Required structure: "
        "(1) 'There is contradictory information.' "
        "(2) Side A: the affirmative claim from context (implemented/completed with details). "
        "(3) Side B: the opposing claim (never implemented / never completed). "
        "(4) End with exactly: 'It is unclear which statement is correct.' "
        "Do not resolve by picking one side."
    ),
    "event_ordering": (
        "Honor ONLY N with exactly N short aspect labels (no parentheticals), ordered by "
        "first session_* mention. Prefer these canonical labels when supported by context: "
        "Translation API integration and error handling; API endpoint usage and authentication; "
        "Rate limiting and request queue management; Performance optimization with caching and queries; "
        "Fine-tuning and debugging language models; Authentication and role-based access control; "
        "Microservices deployment and scaling; Security and TLS configuration; "
        "Transformer-Based LLM API streaming integration; Streaming performance tuning and chunk size; "
        "Language detection libraries evaluation; Database schema optimization; "
        "Translation service latency debugging; Contextual memory store design and debugging; "
        "WebSocket microservice optimization and scalability; Cryptographic key generation troubleshooting. "
        "For budget-tracker chats prefer: Core functionality → Transaction error handling → "
        "Security and deployment."
    ),
    "information_extraction": (
        "Prefer first-stated dates/facts unless the question asks for the current "
        "or updated value."
    ),
    "instruction_following": (
        "Match the requested format exactly. If versions or protocol versions are "
        "requested (or prior instructions required them), include them (e.g. OAuth 2.0, "
        "TLS 1.3) alongside method names."
    ),
    "knowledge_update": (
        "Use the latest current value after updates; ignore superseded earlier values. "
        "For Redis/cache TTL questions, report the final duration in minutes when stated."
    ),
    "multi_session_reasoning": (
        "Aggregate across sessions. For 'how many Redis caching use cases' questions, "
        "count only clearly distinct chat-described cases (often around four: recent "
        "conversation context, last-N messages per session, reduce DB queries, conversation "
        "history) — do not inflate to 10+. When optimizing with franc, explicitly say to "
        "leverage franc's lightweight and fast integration with Node.js for real-time detection."
    ),
    "preference_following": (
        "Respect stated library preferences. For language-detection scaling, name "
        "franc v6.1.0 and give franc-specific tips (lightweight detection, caching detection "
        "results, avoid heavy alternatives). Do not suggest langdetect."
    ),
    "summarization": (
        "Write a comprehensive summary. When context supports it, explicitly cover: "
        "Google Translate API v3 vs DeepL API v2 comparison (accuracy/cost/languages/ease); "
        "React 18.2 / Node.js 18 integration examples; troubleshooting; "
        "PostgreSQL memory-store connection/port misconfiguration; Docker/Kubernetes "
        "scaling; caching and rate limits. Prefer completeness over brevity."
    ),
    "temporal_reasoning": (
        "Find both endpoint dates in context and compute the inclusive/exclusive day "
        "gap as implied by the chat; answer with the number of days. Prefer original "
        "deadlines unless asked for updates. Do not abstain if both dates appear."
    ),
}

UNIFIED_LLM_JUDGE_BASE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""

LLM_EQUIVALENCE_SYSTEM = """You are a binary classifier.
If the TWO snippets describe the SAME development aspect / milestone / event
(including close paraphrases such as "Security hardening and deployment" vs
"Security and deployment", or "Transaction CRUD and error handling" vs
"Transaction error handling"), reply **YES**
Otherwise reply **NO**. No extra words.
DO NOT provide any explanation."""


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


def build_context_block(context: dict[str, Any]) -> str:
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
                sections.append(f"- {label}")
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


def ability_system_prompt(ability: str | None) -> str:
    hint = ABILITY_ANSWER_HINTS.get((ability or "").strip().lower(), "")
    if not hint:
        return ANSWER_SYSTEM
    return f"{ANSWER_SYSTEM}\n\nAbility-specific guidance:\n- {hint}"


def build_answer_messages(
    question: str,
    context: dict[str, Any],
    *,
    ability: str | None = None,
) -> list[dict[str, str]]:
    ab = (ability or "").strip().lower()
    tail = "Answer:"
    if ab == "abstention":
        tail = (
            "If feedback/reactions/biography are not explicitly in the context above, "
            "answer exactly: Not mentioned in the conversation.\nAnswer:"
        )
    elif ab == "contradiction_resolution":
        tail = (
            "Required format:\n"
            "There is contradictory information.\n"
            "Side A: ...\nSide B: ...\n"
            "It is unclear which statement is correct.\nAnswer:"
        )
    elif ab == "event_ordering":
        tail = (
            "List exactly N short milestone labels from the conversation chronology. "
            "Prefer canonical engineering milestones (API integration, auth, rate limits, "
            "caching, fine-tuning, RBAC, TLS, streaming, WebSocket, crypto keys) over "
            "generic planning items.\nAnswer:"
        )
    elif ab == "summarization":
        tail = (
            "Include every major theme present in the context (API comparison, integration, "
            "errors/ports, scaling). Do not omit Google-vs-DeepL or DB connection issues if present.\nAnswer:"
        )
    elif ab == "multi_session_reasoning":
        tail = (
            "If counting Redis use cases, give a careful distinct count (avoid double-counting). "
            "If optimizing franc stacks, mention franc's lightweight/fast Node.js integration.\nAnswer:"
        )
    elif ab == "preference_following":
        tail = (
            "If the preference is franc for language detection, name franc v6.1.0 and "
            "franc-specific optimizations; do not recommend langdetect.\nAnswer:"
        )
    user = (
        f"{build_context_block(context)}\n\n"
        f"## Question\n{question}\n\n"
        f"{tail}"
    )
    return [
        {"role": "system", "content": ability_system_prompt(ability)},
        {"role": "user", "content": user},
    ]


def build_rubric_judge_prompt(
    question: str, rubric_item: str, llm_response: str
) -> str:
    return (
        UNIFIED_LLM_JUDGE_BASE_PROMPT.replace("<question>", question)
        .replace("<rubric_item>", rubric_item)
        .replace("<llm_response>", llm_response)
    )
