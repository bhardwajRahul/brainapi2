from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import AzureOpenAI, OpenAI

from longmemeval.config import Settings
from longmemeval.prompts import get_anscheck_prompt


@dataclass
class JudgeResult:
    correct: bool
    reason: str
    raw: str
    model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def _make_openai_client(settings: Settings) -> OpenAI | AzureOpenAI:
    settings.require_judge_llm()
    if settings.judge_azure_endpoint:
        return AzureOpenAI(
            api_key=settings.judge_api_key,
            azure_endpoint=settings.judge_azure_endpoint,
            api_version=settings.judge_azure_api_version,
        )
    kwargs: dict[str, Any] = {"api_key": settings.judge_api_key}
    if settings.judge_base_url:
        kwargs["base_url"] = settings.judge_base_url
    return OpenAI(**kwargs)


def parse_yes_no(raw: str) -> bool:
    return "yes" in (raw or "").strip().lower()


def judge_answer(
    settings: Settings,
    question: str,
    gold: str,
    prediction: str,
    *,
    question_type: str,
    abstention: bool = False,
    model: str | None = None,
) -> JudgeResult:
    client = _make_openai_client(settings)
    model_name = model or settings.judge_model
    prompt = get_anscheck_prompt(
        question_type,
        question,
        gold,
        prediction,
        abstention=abstention,
    )
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    raw = (response.choices[0].message.content or "").strip()
    correct = parse_yes_no(raw)
    usage = response.usage
    return JudgeResult(
        correct=correct,
        reason=raw,
        raw=raw,
        model=getattr(response, "model", None) or model_name,
        latency_ms=latency_ms,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
    )
