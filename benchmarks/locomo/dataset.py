from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from locomo.config import DEFAULT_DATASET_PATH, DEFAULT_DATASET_URL, Settings


def brain_id_for(sample_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]", "", f"locomo{sample_id}")
    if not slug.isalnum():
        raise ValueError(f"Could not derive alphanumeric brain_id from {sample_id!r}")
    return slug.lower()


def download_dataset(
    dest: Path | None = None,
    url: str = DEFAULT_DATASET_URL,
) -> Path:
    dest = dest or DEFAULT_DATASET_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urlretrieve(url, dest)
    return dest


def load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m locomo download"
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected locomo10.json to be a JSON array")
    return data


def get_sample(dataset: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    for sample in dataset:
        if sample.get("sample_id") == sample_id:
            return sample
    raise KeyError(f"sample_id {sample_id!r} not found")


def session_keys(conversation: dict[str, Any]) -> list[str]:
    keys = [
        k
        for k, v in conversation.items()
        if k.startswith("session_")
        and not k.endswith("date_time")
        and isinstance(v, list)
    ]

    def sort_key(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    return sorted(keys, key=sort_key)


def format_turn(turn: dict[str, Any]) -> str:
    text = (turn.get("text") or "").strip()
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text} [image: {caption}]".strip()
    speaker = turn.get("speaker") or "Unknown"
    dia_id = turn.get("dia_id") or "?"
    return f"{speaker} ({dia_id}): {text}"


def format_session(conversation: dict[str, Any], session_key: str) -> str:
    when = conversation.get(f"{session_key}_date_time", "unknown time")
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    turns = conversation.get(session_key) or []
    lines = [
        f"Conversation between {speaker_a} and {speaker_b}.",
        f"Session time: {when}.",
        f"Session id: {session_key}.",
        "",
    ]
    for turn in turns:
        if isinstance(turn, dict):
            lines.append(format_turn(turn))
    return "\n".join(lines)


def format_turn_unit(
    conversation: dict[str, Any], session_key: str, turn: dict[str, Any]
) -> str:
    when = conversation.get(f"{session_key}_date_time", "unknown time")
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    return "\n".join(
        [
            f"Conversation between {speaker_a} and {speaker_b}.",
            f"Session time: {when}.",
            f"Session id: {session_key}.",
            "",
            format_turn(turn),
        ]
    )


def iter_ingest_units(
    sample: dict[str, Any],
    granularity: str = "session",
    limit_sessions: int | None = None,
) -> list[dict[str, Any]]:
    conversation = sample["conversation"]
    keys = session_keys(conversation)
    if limit_sessions is not None:
        keys = keys[:limit_sessions]

    units: list[dict[str, Any]] = []
    if granularity == "session":
        for key in keys:
            units.append(
                {
                    "unit_id": key,
                    "session_key": key,
                    "text": format_session(conversation, key),
                }
            )
        return units

    if granularity == "turn":
        for key in keys:
            for idx, turn in enumerate(conversation.get(key) or []):
                if not isinstance(turn, dict):
                    continue
                dia_id = turn.get("dia_id") or f"{key}:{idx}"
                units.append(
                    {
                        "unit_id": f"{key}:{dia_id}",
                        "session_key": key,
                        "dia_id": dia_id,
                        "text": format_turn_unit(conversation, key, turn),
                    }
                )
        return units

    raise ValueError("granularity must be 'session' or 'turn'")


def dataset_stats(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in dataset:
        conversation = sample.get("conversation") or {}
        keys = session_keys(conversation)
        turns = sum(len(conversation.get(k) or []) for k in keys)
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "brain_id": brain_id_for(str(sample.get("sample_id"))),
                "speaker_a": conversation.get("speaker_a"),
                "speaker_b": conversation.get("speaker_b"),
                "sessions": len(keys),
                "turns": turns,
                "qa": len(sample.get("qa") or []),
            }
        )
    return rows


def resolve_samples(
    settings: Settings,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(settings.dataset_path)
    if not sample_ids:
        return dataset
    return [get_sample(dataset, sid) for sid in sample_ids]
