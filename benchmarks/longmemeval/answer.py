from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import AzureOpenAI, OpenAI

from longmemeval.config import Settings
from longmemeval.prompts import build_answer_messages
from longmemeval.sota import majority_vote


@dataclass
class AnswerResult:
    answer: str
    model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    sc_samples_used: int = 1


def _make_openai_client(settings: Settings) -> OpenAI | AzureOpenAI:
    settings.require_llm()
    if settings.answer_azure_endpoint:
        return AzureOpenAI(
            api_key=settings.openai_api_key,
            azure_endpoint=settings.answer_azure_endpoint,
            api_version=settings.answer_azure_api_version,
        )
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return OpenAI(**kwargs)


def _one_completion(
    client: OpenAI | AzureOpenAI,
    *,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, Any]:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    content = (response.choices[0].message.content or "").strip()
    return content, response


def answer_question(
    settings: Settings,
    question: str,
    context: dict[str, Any],
    *,
    question_date: str | None = None,
    model: str | None = None,
    sc_samples: int | None = None,
    sc_temperature: float | None = None,
) -> AnswerResult:
    client = _make_openai_client(settings)
    model_name = model or settings.answer_model
    messages = build_answer_messages(
        question, context, question_date=question_date
    )
    n = max(1, int(sc_samples if sc_samples is not None else settings.sc_samples))
    temperature = (
        float(sc_temperature)
        if sc_temperature is not None
        else (settings.sc_temperature if n > 1 else 0.0)
    )
    started = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    samples: list[str] = []
    last_response: Any = None

    for _ in range(n):
        content, response = _one_completion(
            client,
            model_name=model_name,
            messages=messages,
            temperature=0.0 if n == 1 else temperature,
        )
        last_response = response
        if response.usage:
            prompt_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(response.usage, "completion_tokens", 0) or 0
            total_tokens += getattr(response.usage, "total_tokens", 0) or 0
        if content:
            samples.append(content)

    content = majority_vote(samples) if samples else ""

    if not content:
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous reply was empty. Answer the question now using the "
                    "retrieved context. Prefer a short factual list or phrase."
                ),
            }
        ]
        content, response = _one_completion(
            client,
            model_name=model_name,
            messages=retry_messages,
            temperature=0,
        )
        last_response = response
        if response.usage:
            prompt_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(response.usage, "completion_tokens", 0) or 0
            total_tokens += getattr(response.usage, "total_tokens", 0) or 0

    latency_ms = (time.perf_counter() - started) * 1000
    return AnswerResult(
        answer=content,
        model=getattr(last_response, "model", None) or model_name,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens or None,
        completion_tokens=completion_tokens or None,
        total_tokens=total_tokens or None,
        sc_samples_used=n,
    )
