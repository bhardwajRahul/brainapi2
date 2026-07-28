from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from locomo.config import Settings
from locomo.prompts import build_judge_messages


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


def _make_openai_client(settings: Settings) -> OpenAI:
    settings.require_llm()
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return OpenAI(**kwargs)


def _parse_judge_payload(raw: str) -> tuple[bool, str]:
    text = raw.strip()
    try:
        data = json.loads(text)
        return bool(data.get("correct")), str(data.get("reason", ""))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return bool(data.get("correct")), str(data.get("reason", ""))
            except json.JSONDecodeError:
                pass
    lowered = text.lower()
    if '"correct": true' in lowered or "correct: true" in lowered:
        return True, text
    if '"correct": false' in lowered or "correct: false" in lowered:
        return False, text
    return False, f"unparseable judge output: {text[:200]}"


def judge_answer(
    settings: Settings,
    question: str,
    gold: str,
    prediction: str,
    *,
    model: str | None = None,
) -> JudgeResult:
    client = _make_openai_client(settings)
    model_name = model or settings.judge_model
    messages = build_judge_messages(question, gold, prediction)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    raw = (response.choices[0].message.content or "").strip()
    correct, reason = _parse_judge_payload(raw)
    usage = response.usage
    return JudgeResult(
        correct=correct,
        reason=reason,
        raw=raw,
        model=model_name,
        latency_ms=latency_ms,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
    )
