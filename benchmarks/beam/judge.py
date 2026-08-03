from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from json_repair import repair_json
from openai import AzureOpenAI, OpenAI
from scipy.stats import kendalltau

from beam.config import Settings
from beam.prompts import LLM_EQUIVALENCE_SYSTEM, build_rubric_judge_prompt


@dataclass
class RubricItemResult:
    score: float
    reason: str
    raw: str


@dataclass
class JudgeResult:
    llm_judge_score: float
    rubric_results: list[RubricItemResult] = field(default_factory=list)
    tau_norm: float | None = None
    event_ordering_f1: float | None = None
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


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


def parse_json_response(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        match = re.search(
            r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", text, re.DOTALL
        )
        if match:
            text = match.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    try:
        data = json.loads(repair_json(text))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        data = json.loads(repair_json(match.group(1)))
        if isinstance(data, dict):
            return data
    raise ValueError(f"No valid JSON found in judge response: {text[:200]}")


def _chat(
    client: OpenAI | AzureOpenAI,
    *,
    model_name: str,
    messages: list[dict[str, str]],
) -> tuple[str, Any]:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    return content, response


def _score_from_payload(payload: dict[str, Any]) -> float:
    raw = payload.get("score", 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value not in (0.0, 0.5, 1.0):
        value = max(0.0, min(1.0, value))
    return value


def _strip_list_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:\d+[\).\]]\s*|[-*•]\s+)", "", (text or "").strip())


def _token_set(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in {"and", "the", "with", "for"}
    }


def lexical_aspect_match(first: str, second: str) -> bool:
    a = _token_set(_strip_list_prefix(first))
    b = _token_set(_strip_list_prefix(second))
    if not a or not b:
        return False
    overlap = len(a & b)
    if overlap == 0:
        return False
    # Near-paraphrase gate: shared content words cover most of the shorter label.
    return overlap >= max(1, min(len(a), len(b)) - 1)


def llm_equivalence(
    client: OpenAI | AzureOpenAI,
    *,
    model_name: str,
    first: str,
    second: str,
) -> bool:
    left = _strip_list_prefix(first)
    right = _strip_list_prefix(second)
    if lexical_aspect_match(left, right):
        return True
    messages = [
        {"role": "system", "content": LLM_EQUIVALENCE_SYSTEM},
        {
            "role": "user",
            "content": f"First snippet: {left}\nSecond snippet: {right}",
        },
    ]
    content, _ = _chat(client, model_name=model_name, messages=messages)
    return "yes" in content.lower()


def align_with_llm(
    client: OpenAI | AzureOpenAI,
    *,
    model_name: str,
    reference: list[str],
    system: list[str],
) -> tuple[list[str], list[str]]:
    used: set[int] = set()
    system_out: list[str] = []
    for s in system:
        matched_index = None
        for index, r in enumerate(reference):
            if index in used:
                continue
            if llm_equivalence(
                client, model_name=model_name, first=r, second=s
            ):
                matched_index = index
                break
        if matched_index is not None:
            system_out.append(reference[matched_index])
            used.add(matched_index)
        else:
            system_out.append(_strip_list_prefix(s))
    return reference, system_out


def event_ordering_score(
    client: OpenAI | AzureOpenAI,
    *,
    model_name: str,
    reference_list: list[str],
    system_list: list[str],
) -> dict[str, float]:
    reference_canon, system_canon = align_with_llm(
        client,
        model_name=model_name,
        reference=reference_list,
        system=system_list,
    )
    tp = len(set(reference_canon) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference_canon])
    fn = len([x for x in reference_canon if x not in system_canon])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq: list[str]) -> list[int]:
        ranks = {item: i + 1 for i, item in enumerate(seq)}
        return [ranks.get(u, tie_rank) for u in union]

    if len(union) < 2:
        tau_b_norm = 1.0 if reference_canon == system_canon and reference_canon else 0.0
    else:
        tau_b, _ = kendalltau(
            to_rank(reference_canon),
            to_rank(system_canon),
            variant="b",
            method="auto",
        )
        tau_b_norm = (float(tau_b) + 1) / 2 if tau_b is not None else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tau_norm": tau_b_norm,
        "final_score": tau_b_norm * f1,
    }


def judge_rubric_response(
    settings: Settings,
    *,
    ability: str,
    question: str,
    prediction: str,
    rubric: list[str],
    model: str | None = None,
) -> JudgeResult:
    client = _make_openai_client(settings)
    model_name = model or settings.judge_model
    started = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    rubric_results: list[RubricItemResult] = []
    score_sum = 0.0
    last_model = model_name
    tau_norm = None
    event_f1 = None

    if not rubric:
        return JudgeResult(
            llm_judge_score=0.0,
            rubric_results=[],
            model=model_name,
            latency_ms=0.0,
        )

    if ability == "event_ordering":
        system_list = [
            line.strip()
            for line in prediction.splitlines()
            if line.strip()
        ] or [prediction]
        ordering = event_ordering_score(
            client,
            model_name=model_name,
            reference_list=[str(item) for item in rubric],
            system_list=system_list,
        )
        tau_norm = float(ordering["tau_norm"])
        event_f1 = float(ordering["f1"])

    for item in rubric:
        prompt = build_rubric_judge_prompt(question, str(item), prediction)
        raw, response = _chat(
            client,
            model_name=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        last_model = getattr(response, "model", None) or model_name
        if response.usage:
            prompt_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(response.usage, "completion_tokens", 0) or 0
            total_tokens += getattr(response.usage, "total_tokens", 0) or 0
        try:
            payload = parse_json_response(raw)
        except Exception:
            payload = {"score": 0.0, "reason": f"unparseable: {raw[:200]}"}
        item_score = _score_from_payload(payload)
        score_sum += item_score
        rubric_results.append(
            RubricItemResult(
                score=item_score,
                reason=str(payload.get("reason", "")),
                raw=raw,
            )
        )

    latency_ms = (time.perf_counter() - started) * 1000
    return JudgeResult(
        llm_judge_score=score_sum / len(rubric),
        rubric_results=rubric_results,
        tau_norm=tau_norm,
        event_ordering_f1=event_f1,
        model=last_model,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens or None,
        completion_tokens=completion_tokens or None,
        total_tokens=total_tokens or None,
    )
