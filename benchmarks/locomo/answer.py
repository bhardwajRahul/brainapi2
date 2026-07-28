from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from locomo.config import Settings
from locomo.prompts import build_answer_messages


@dataclass
class AnswerResult:
    answer: str
    model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def _make_openai_client(settings: Settings) -> OpenAI:
    settings.require_llm()
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return OpenAI(**kwargs)


def answer_question(
    settings: Settings,
    question: str,
    context: dict[str, Any],
    *,
    model: str | None = None,
) -> AnswerResult:
    client = _make_openai_client(settings)
    model_name = model or settings.answer_model
    messages = build_answer_messages(question, context)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    content = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return AnswerResult(
        answer=content,
        model=model_name,
        latency_ms=latency_ms,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
    )
