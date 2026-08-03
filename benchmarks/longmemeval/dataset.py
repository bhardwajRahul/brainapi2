from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from longmemeval.config import (
    DEFAULT_VARIANT,
    VARIANT_FILES,
    Settings,
    variant_path,
    variant_url,
)


def brain_id_for(question_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]", "", f"lme{question_id}")
    if not slug.isalnum():
        raise ValueError(f"Could not derive alphanumeric brain_id from {question_id!r}")
    return slug.lower()


def is_abstention(question_id: str) -> bool:
    return str(question_id).endswith("_abs")


def download_dataset(
    dest: Path | None = None,
    url: str | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
) -> Path:
    dest = dest or variant_path(variant)
    url = url or variant_url(variant)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urlretrieve(url, dest)
    return dest


def load_dataset(path: Path | None = None, *, variant: str = DEFAULT_VARIANT) -> list[dict[str, Any]]:
    path = path or variant_path(variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m longmemeval download --variant {variant}"
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path.name} to be a JSON array")
    return data


def get_question(dataset: list[dict[str, Any]], question_id: str) -> dict[str, Any]:
    for entry in dataset:
        if entry.get("question_id") == question_id:
            return entry
    raise KeyError(f"question_id {question_id!r} not found")


def format_turn(turn: dict[str, Any]) -> str:
    role = str(turn.get("role") or "unknown").strip()
    content = str(turn.get("content") or "").strip()
    return f"{role}: {content}"


def format_session(
    session_id: str,
    session_date: str,
    turns: list[Any],
) -> str:
    lines = [
        f"Session time: {session_date or 'unknown time'}.",
        f"Session id: {session_id}.",
        "",
    ]
    for turn in turns or []:
        if not isinstance(turn, dict):
            continue
        clean = {
            "role": turn.get("role"),
            "content": turn.get("content"),
        }
        lines.append(format_turn(clean))
    return "\n".join(lines)


def iter_ingest_units(
    entry: dict[str, Any],
    *,
    limit_sessions: int | None = None,
) -> list[dict[str, Any]]:
    session_ids = list(entry.get("haystack_session_ids") or [])
    dates = list(entry.get("haystack_dates") or [])
    sessions = list(entry.get("haystack_sessions") or [])
    n = min(len(session_ids), len(dates), len(sessions))
    if limit_sessions is not None:
        n = min(n, limit_sessions)

    units: list[dict[str, Any]] = []
    for i in range(n):
        sid = str(session_ids[i])
        units.append(
            {
                "unit_id": sid,
                "session_id": sid,
                "text": format_session(sid, str(dates[i]), sessions[i]),
                "source_timestamp": str(dates[i]) if dates[i] else None,
            }
        )
    return units


def dataset_stats(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(str(e.get("question_type") or "unknown") for e in dataset)
    abstention = sum(1 for e in dataset if is_abstention(str(e.get("question_id") or "")))
    sessions = [
        len(e.get("haystack_session_ids") or [])
        for e in dataset
    ]
    turns = []
    for e in dataset:
        for sess in e.get("haystack_sessions") or []:
            if isinstance(sess, list):
                turns.append(len(sess))
    return {
        "n_questions": len(dataset),
        "n_abstention": abstention,
        "by_type": dict(sorted(type_counts.items())),
        "sessions_per_question": {
            "min": min(sessions) if sessions else 0,
            "max": max(sessions) if sessions else 0,
            "mean": (sum(sessions) / len(sessions)) if sessions else 0.0,
        },
        "turns_per_session": {
            "min": min(turns) if turns else 0,
            "max": max(turns) if turns else 0,
            "mean": (sum(turns) / len(turns)) if turns else 0.0,
        },
    }


def resolve_questions(
    settings: Settings,
    *,
    question_ids: list[str] | None = None,
    limit: int | None = None,
    question_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(settings.dataset_path, variant=settings.variant)
    if question_ids:
        wanted = set(question_ids)
        dataset = [e for e in dataset if e.get("question_id") in wanted]
        missing = wanted - {e.get("question_id") for e in dataset}
        if missing:
            raise KeyError(f"question_id(s) not found: {sorted(missing)}")
    if question_types:
        dataset = [
            e for e in dataset if str(e.get("question_type") or "") in question_types
        ]
    if limit is not None:
        dataset = dataset[: max(0, limit)]
    return dataset


def resolve_variant_settings(
    settings: Settings, variant: str | None
) -> Settings:
    if not variant or variant == settings.variant:
        return settings
    key = variant.strip().lower()
    if key not in VARIANT_FILES:
        raise SystemExit(f"Unknown variant {variant!r}; choose from {sorted(VARIANT_FILES)}")
    return Settings(
        brainapi_url=settings.brainapi_url,
        brainpat_token=settings.brainpat_token,
        openai_api_key=settings.openai_api_key,
        llm_base_url=settings.llm_base_url,
        answer_model=settings.answer_model,
        judge_model=settings.judge_model,
        judge_api_key=settings.judge_api_key,
        judge_base_url=settings.judge_base_url,
        judge_azure_endpoint=settings.judge_azure_endpoint,
        judge_azure_api_version=settings.judge_azure_api_version,
        dataset_path=variant_path(key),
        dataset_url=variant_url(key),
        variant=key,
        runs_dir=settings.runs_dir,
        bench_profile=settings.bench_profile,
        sc_samples=settings.sc_samples,
        sc_temperature=settings.sc_temperature,
        gap_fill=settings.gap_fill,
        answer_azure_endpoint=settings.answer_azure_endpoint,
        answer_azure_api_version=settings.answer_azure_api_version,
    )
