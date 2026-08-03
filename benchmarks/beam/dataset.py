from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from beam.config import ABILITY_NAMES, CHAT_SIZES, Settings

_PLAN_KEY_RE = re.compile(r"^plan[-_]?(\d+)$", re.IGNORECASE)
NORMALIZE_STRATEGY_SINGLE = "single_chat"
NORMALIZE_STRATEGY_PLANS = "concatenate_plans_chronological"


def brain_id_for(size: str, conversation_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]", "", f"beam{size}{conversation_id}")
    if not slug.isalnum():
        raise ValueError(
            f"Could not derive alphanumeric brain_id from size={size!r} "
            f"conversation_id={conversation_id!r}"
        )
    return slug.lower()


def sample_id_for(size: str, conversation_id: str) -> str:
    return f"{size}/{conversation_id}"


def parse_sample_id(sample_id: str) -> tuple[str, str]:
    parts = sample_id.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"sample_id must look like '100K/1', got {sample_id!r}"
        )
    return parts[0], parts[1]


def _normalize_time_anchor(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # BEAM often uses March-15-2024; convert to a parseable date-ish string.
    m = re.match(r"^([A-Za-z]+)-(\d{1,2})-(\d{4})$", text)
    if m:
        return f"{m.group(1)} {m.group(2)}, {m.group(3)}"
    return text


def convert_chat_batches(chat: list[Any]) -> list[dict[str, Any]]:
    """Normalize HF chat (list of message lists) into batch/turn JSON."""
    json_object: list[dict[str, Any]] = []
    for index, batch in enumerate(chat):
        if not isinstance(batch, list):
            continue
        batch_number = index + 1
        turns: list[list[dict[str, Any]]] = []
        single_turn: list[dict[str, Any]] = []
        time_anchor = None
        for message in batch:
            if not isinstance(message, dict):
                continue
            if (
                message.get("question_type") == "main_question"
                and single_turn
            ):
                if message.get("time_anchor"):
                    time_anchor = message.get("time_anchor")
                turns.append(single_turn)
                single_turn = []
            single_turn.append(message)
        if single_turn:
            turns.append(single_turn)
        json_object.append(
            {
                "batch_number": batch_number,
                "turns": turns,
                "time_anchor": time_anchor,
            }
        )
    return json_object


def plan_sort_key(label: Any) -> tuple[int, str]:
    text = str(label or "").strip()
    match = _PLAN_KEY_RE.match(text)
    if match:
        return (int(match.group(1)), text)
    if text.isdigit():
        return (int(text), text)
    return (10**9, text)


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    keys = getattr(value, "keys", None)
    if callable(keys):
        try:
            return {str(k): value[k] for k in keys()}
        except Exception:
            return None
    return None


def _iter_plan_keyed_entries(chat: Any) -> list[tuple[str, Any]]:
    """Yield (plan_id, payload) from BEAM-10M main chat (plan-1..plan-10)."""
    mapping = _as_mapping(chat)
    if mapping is not None:
        entries = [
            (key, mapping[key])
            for key in mapping
            if _PLAN_KEY_RE.match(str(key).strip())
        ]
        entries.sort(key=lambda item: plan_sort_key(item[0]))
        return entries
    if isinstance(chat, list):
        collected: list[tuple[str, Any]] = []
        for item in chat:
            nested = _as_mapping(item)
            if nested is None:
                continue
            for key, payload in nested.items():
                if _PLAN_KEY_RE.match(str(key).strip()):
                    collected.append((str(key), payload))
        collected.sort(key=lambda item: plan_sort_key(item[0]))
        return collected
    return []


def _batches_from_chat_payload(value: Any) -> list[dict[str, Any]]:
    """Accept already-normalized batches or raw HF message-list chat."""
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    if isinstance(first, dict) and "turns" in first:
        return [row for row in value if isinstance(row, dict) and row.get("turns")]
    if isinstance(first, list):
        return [row for row in convert_chat_batches(value) if row.get("turns")]
    return []


def concatenate_plan_batches(
    plan_batches: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Merge plan chats in chronological order into one chat.json-compatible list.

    Reassigns batch_number globally so unit ids remain unique under the existing
    bN_tM / session_N conventions used by the 1M ingest path.
    """
    out: list[dict[str, Any]] = []
    plan_meta: list[dict[str, Any]] = []
    next_batch = 1
    for plan_id, batches in plan_batches:
        start = next_batch
        n_turns = 0
        for batch in batches:
            turns = batch.get("turns") or []
            if not isinstance(turns, list) or not turns:
                continue
            n_turns += len(turns)
            out.append(
                {
                    "batch_number": next_batch,
                    "turns": turns,
                    "time_anchor": batch.get("time_anchor"),
                    "plan_id": plan_id,
                }
            )
            next_batch += 1
        if next_batch > start:
            plan_meta.append(
                {
                    "plan_id": plan_id,
                    "batch_number_start": start,
                    "batch_number_end": next_batch - 1,
                    "n_batches": next_batch - start,
                    "n_turns": n_turns,
                }
            )
    return out, plan_meta


def extract_normalized_chat(
    row: dict[str, Any],
    *,
    size: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build normalized chat batches for a HF row.

    BEAM-10M: concatenate plan-1..plan-10 (or plans[]) chronologically into one
    long chat. Smaller sizes: single-chat convert_chat_batches path.
    """
    chat = row.get("chat")
    plans = row.get("plans")
    plan_entries = _iter_plan_keyed_entries(chat)
    use_multi_plan = size == "10M" or bool(plan_entries) or (
        isinstance(plans, list) and bool(plans)
    )

    if use_multi_plan:
        ordered: list[tuple[str, list[dict[str, Any]]]] = []
        if plan_entries:
            for plan_id, payload in plan_entries:
                batches = _batches_from_chat_payload(payload)
                if batches:
                    ordered.append((str(plan_id), batches))
            source = "chat_plan_keys"
        else:
            source = "plans_array"
            plan_rows = [p for p in (plans or []) if isinstance(p, dict)]
            plan_rows.sort(
                key=lambda p: plan_sort_key(p.get("plan_id") or p.get("id") or "")
            )
            for index, plan in enumerate(plan_rows):
                plan_id = str(
                    plan.get("plan_id") or plan.get("id") or f"plan-{index + 1}"
                )
                batches = _batches_from_chat_payload(plan.get("chat"))
                if batches:
                    ordered.append((plan_id, batches))

        if ordered:
            chat_json, plan_meta = concatenate_plan_batches(ordered)
            return chat_json, {
                "normalize_strategy": NORMALIZE_STRATEGY_PLANS,
                "n_plans": len(plan_meta),
                "plans": plan_meta,
                "plan_source": source,
            }

    chat_list = chat if isinstance(chat, list) else []
    return convert_chat_batches(chat_list), {
        "normalize_strategy": NORMALIZE_STRATEGY_SINGLE,
    }


def ingest_target_turns(n_turns: int, *, slack: int = 5) -> int:
    """Ops TARGET from normalized turn count (never invent without n_turns)."""
    if n_turns <= 0:
        return 0
    return max(1, n_turns - max(0, slack))


def _parse_probing_questions(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        data = ast.literal_eval(raw)
    else:
        raise ValueError(f"Unexpected probing_questions type: {type(raw)}")
    if not isinstance(data, dict):
        raise ValueError("probing_questions must parse to a dict")
    out: dict[str, list[dict[str, Any]]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            continue
        out[str(key)] = [item for item in value if isinstance(item, dict)]
    return out


def conversation_dir(data_dir: Path, size: str, conversation_id: str) -> Path:
    return data_dir / size / str(conversation_id)


def write_conversation(
    dest: Path,
    *,
    size: str,
    conversation_id: str,
    topic: Any,
    chat: list[Any] | None = None,
    probing_questions: Any,
    plans: Any = None,
    row: dict[str, Any] | None = None,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if row is not None:
        chat_json, extra_meta = extract_normalized_chat(row, size=size)
        if topic is None:
            topic = row.get("conversation_seed")
        if probing_questions is None:
            probing_questions = row.get("probing_questions")
    else:
        chat_json, extra_meta = extract_normalized_chat(
            {"chat": chat or [], "plans": plans},
            size=size,
        )
    probing = _parse_probing_questions(probing_questions)
    (dest / "chat.json").write_text(
        json.dumps(chat_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dest / "probing_questions.json").write_text(
        json.dumps(probing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if isinstance(topic, dict):
        (dest / "topic.json").write_text(
            json.dumps(topic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    n_turns = sum(len(b.get("turns") or []) for b in chat_json)
    meta = {
        "conversation_id": str(conversation_id),
        "size": size,
        "sample_id": sample_id_for(size, str(conversation_id)),
        "brain_id": brain_id_for(size, str(conversation_id)),
        "n_batches": len(chat_json),
        "n_turns": n_turns,
        "n_questions": sum(len(v) for v in probing.values()),
        "ingest_target": ingest_target_turns(n_turns),
        **extra_meta,
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest


def download_and_normalize(
    settings: Settings,
    *,
    sizes: list[str] | None = None,
    force: bool = False,
) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required for BEAM download. "
            "Run: pip install -r benchmarks/requirements.txt"
        ) from exc

    selected = sizes or list(CHAT_SIZES)
    for size in selected:
        if size not in CHAT_SIZES:
            raise SystemExit(f"Unsupported size {size!r}; choose from {CHAT_SIZES}")

    settings.beam_data_dir.mkdir(parents=True, exist_ok=True)

    for size in selected:
        dataset_name = settings.hf_dataset_for(size)
        conversations = load_dataset(dataset_name, split=size)
        for conversation in conversations:
            conversation_id = str(conversation["conversation_id"])
            dest = conversation_dir(
                settings.beam_data_dir, size, conversation_id
            )
            if (
                not force
                and (dest / "chat.json").exists()
                and (dest / "probing_questions.json").exists()
            ):
                continue
            row = dict(conversation)
            write_conversation(
                dest,
                size=size,
                conversation_id=conversation_id,
                topic=row.get("conversation_seed"),
                probing_questions=row.get("probing_questions"),
                row=row,
            )
    return settings.beam_data_dir


def list_local_samples(
    settings: Settings, *, size: str | None = None
) -> list[dict[str, Any]]:
    root = settings.beam_data_dir
    if not root.exists():
        return []
    sizes = [size] if size else [
        p.name for p in sorted(root.iterdir()) if p.is_dir() and p.name in CHAT_SIZES
    ]
    samples: list[dict[str, Any]] = []
    for sz in sizes:
        size_dir = root / sz
        if not size_dir.exists():
            continue
        for conv_dir in sorted(
            [p for p in size_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name) if p.name.isdigit() else p.name,
        ):
            meta_path = conv_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = {
                    "conversation_id": conv_dir.name,
                    "size": sz,
                    "sample_id": sample_id_for(sz, conv_dir.name),
                    "brain_id": brain_id_for(sz, conv_dir.name),
                }
            meta["path"] = str(conv_dir)
            samples.append(meta)
    return samples


def load_sample(settings: Settings, sample_id: str) -> dict[str, Any]:
    size, conversation_id = parse_sample_id(sample_id)
    dest = conversation_dir(settings.beam_data_dir, size, conversation_id)
    chat_path = dest / "chat.json"
    probing_path = dest / "probing_questions.json"
    if not chat_path.exists() or not probing_path.exists():
        raise FileNotFoundError(
            f"Sample {sample_id!r} not found under {dest}. "
            "Run: python -m beam download"
        )
    chat = json.loads(chat_path.read_text(encoding="utf-8"))
    probing = json.loads(probing_path.read_text(encoding="utf-8"))
    meta_path = dest / "meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else {
            "conversation_id": conversation_id,
            "size": size,
            "sample_id": sample_id,
            "brain_id": brain_id_for(size, conversation_id),
        }
    )
    topic = None
    topic_path = dest / "topic.json"
    if topic_path.exists():
        topic = json.loads(topic_path.read_text(encoding="utf-8"))
    return {
        **meta,
        "chat": chat,
        "probing_questions": probing,
        "topic": topic,
        "path": str(dest),
    }


def resolve_samples(
    settings: Settings,
    *,
    size: str,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if size not in CHAT_SIZES:
        raise SystemExit(f"Unsupported size {size!r}; choose from {CHAT_SIZES}")
    if sample_ids:
        return [load_sample(settings, sid if "/" in sid else f"{size}/{sid}") for sid in sample_ids]
    local = list_local_samples(settings, size=size)
    if not local:
        raise SystemExit(
            f"No local BEAM samples for size {size}. Run: python -m beam download --size {size}"
        )
    return [load_sample(settings, str(row["sample_id"])) for row in local]


def format_turn_text(
    messages: list[dict[str, Any]],
    *,
    batch_number: int,
    turn_index: int,
    batch_time_anchor: Any = None,
    global_turn: int | None = None,
) -> tuple[str, str | None]:
    session_n = global_turn if global_turn is not None else turn_index + 1
    unit_id = f"b{batch_number}_t{turn_index + 1}"
    lines = [
        f"Session id: session_{session_n}.",
        f"BEAM dialogue batch {batch_number}, turn {turn_index + 1}.",
        f"Unit id: {unit_id}.",
    ]
    turn_anchor = None
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = str(message.get("content") or "").strip()
        msg_anchor = message.get("time_anchor")
        if msg_anchor and not turn_anchor:
            turn_anchor = msg_anchor
        qtype = message.get("question_type")
        prefix = f"{role}"
        if qtype:
            prefix = f"{role} [{qtype}]"
        if msg_anchor:
            prefix = f"{prefix} (time_anchor={msg_anchor})"
        lines.append(f"{prefix}: {content}")
    source_ts = _normalize_time_anchor(turn_anchor or batch_time_anchor)
    return "\n".join(lines), source_ts


def iter_ingest_units(
    sample: dict[str, Any],
    *,
    limit_turns: int | None = None,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    global_turn = 0
    for batch in sample.get("chat") or []:
        batch_number = int(batch.get("batch_number") or 0)
        batch_anchor = batch.get("time_anchor")
        for turn_index, messages in enumerate(batch.get("turns") or []):
            if not isinstance(messages, list) or not messages:
                continue
            global_turn += 1
            text, source_ts = format_turn_text(
                messages,
                batch_number=batch_number,
                turn_index=turn_index,
                batch_time_anchor=batch_anchor,
                global_turn=global_turn,
            )
            unit_id = f"b{batch_number}_t{turn_index + 1}"
            units.append(
                {
                    "unit_id": unit_id,
                    "session_key": f"session_{global_turn}",
                    "batch_number": batch_number,
                    "turn_index": turn_index,
                    "global_turn": global_turn,
                    "text": text,
                    "source_timestamp": source_ts,
                }
            )
            if limit_turns is not None and len(units) >= limit_turns:
                return units
    return units


def iter_probing_jobs(
    sample: dict[str, Any],
    *,
    abilities: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    probing = sample.get("probing_questions") or {}
    jobs: list[dict[str, Any]] = []
    for ability in ABILITY_NAMES:
        if abilities is not None and ability not in abilities:
            continue
        questions = probing.get(ability) or []
        for qa_index, qa in enumerate(questions):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            jobs.append(
                {
                    "ability": ability,
                    "qa_index": qa_index,
                    "question": question,
                    "rubric": list(qa.get("rubric") or []),
                    "qa": qa,
                }
            )
            if limit is not None and len(jobs) >= limit:
                return jobs
    return jobs


def dataset_stats(settings: Settings) -> list[dict[str, Any]]:
    rows = []
    for sample_meta in list_local_samples(settings):
        sample = load_sample(settings, str(sample_meta["sample_id"]))
        units = iter_ingest_units(sample)
        jobs = iter_probing_jobs(sample)
        by_ability = {
            ability: len((sample.get("probing_questions") or {}).get(ability) or [])
            for ability in ABILITY_NAMES
        }
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "brain_id": sample["brain_id"],
                "size": sample["size"],
                "batches": len(sample.get("chat") or []),
                "turns": len(units),
                "questions": len(jobs),
                "by_ability": by_ability,
            }
        )
    return rows
