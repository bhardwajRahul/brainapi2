from __future__ import annotations

from pathlib import Path

from beam.dataset import (
    brain_id_for,
    concatenate_plan_batches,
    convert_chat_batches,
    extract_normalized_chat,
    format_turn_text,
    ingest_target_turns,
    iter_ingest_units,
    write_conversation,
)
from beam.ingest import (
    completed_unit_ids,
    is_permanent_ingest_error,
    normalize_ingest_status,
)
from beam.judge import parse_json_response
from beam.metrics import aggregate_answers, selftest_metrics
from beam.prompts import ability_system_prompt, build_rubric_judge_prompt
from beam.sota import majority_vote, plan_gap_fill, profile_defaults


def test_brain_id_for() -> None:
    assert brain_id_for("100K", "1") == "beam100k1"
    assert brain_id_for("500K", "3") == "beam500k3"
    assert brain_id_for("1M", "12") == "beam1m12"
    assert brain_id_for("10M", "1") == "beam10m1"
    assert brain_id_for("10M", "1clean") == "beam10m1clean"


def test_convert_chat_batches_splits_on_main_question() -> None:
    chat = [
        [
            {
                "role": "user",
                "content": "Q1",
                "question_type": "main_question",
                "time_anchor": "March-15-2024",
            },
            {"role": "assistant", "content": "A1"},
            {
                "role": "user",
                "content": "Q2",
                "question_type": "main_question",
            },
            {"role": "assistant", "content": "A2"},
        ]
    ]
    batches = convert_chat_batches(chat)
    assert len(batches) == 1
    assert len(batches[0]["turns"]) == 2
    assert batches[0]["turns"][0][0]["content"] == "Q1"
    assert batches[0]["turns"][1][0]["content"] == "Q2"


def _raw_turn(content: str, *, anchor: str = "March-15-2024") -> list[dict]:
    return [
        {
            "role": "user",
            "content": content,
            "question_type": "main_question",
            "time_anchor": anchor,
        },
        {"role": "assistant", "content": f"re:{content}"},
    ]


def test_extract_normalized_chat_10m_plan_keys() -> None:
    row = {
        "chat": {
            "plan-2": [
                {
                    "batch_number": 1,
                    "time_anchor": None,
                    "turns": [_raw_turn("p2")],
                }
            ],
            "plan-1": [
                {
                    "batch_number": 1,
                    "time_anchor": None,
                    "turns": [_raw_turn("p1a"), _raw_turn("p1b")],
                }
            ],
        },
        "plans": [],
        "probing_questions": {"summarization": [{"question": "Sum?", "rubric": ["x"]}]},
    }
    chat_json, meta = extract_normalized_chat(row, size="10M")
    assert meta["normalize_strategy"] == "concatenate_plans_chronological"
    assert meta["n_plans"] == 2
    assert meta["plan_source"] == "chat_plan_keys"
    assert [b["plan_id"] for b in chat_json] == ["plan-1", "plan-2"]
    assert [b["batch_number"] for b in chat_json] == [1, 2]
    assert len(chat_json[0]["turns"]) == 2
    assert chat_json[0]["turns"][0][0]["content"] == "p1a"


def test_extract_normalized_chat_10m_plans_array_raw() -> None:
    row = {
        "chat": None,
        "plans": [
            {
                "plan_id": "plan-10",
                "chat": [
                    [
                        {
                            "role": "user",
                            "content": "late",
                            "question_type": "main_question",
                            "time_anchor": "April-01-2024",
                        },
                        {"role": "assistant", "content": "re:late"},
                    ]
                ],
            },
            {
                "plan_id": "plan-1",
                "chat": [
                    [
                        {
                            "role": "user",
                            "content": "early-a",
                            "question_type": "main_question",
                            "time_anchor": "March-15-2024",
                        },
                        {"role": "assistant", "content": "a"},
                        {
                            "role": "user",
                            "content": "early-b",
                            "question_type": "main_question",
                            "time_anchor": "March-16-2024",
                        },
                        {"role": "assistant", "content": "b"},
                    ]
                ],
            },
        ],
    }
    chat_json, meta = extract_normalized_chat(row, size="10M")
    assert meta["normalize_strategy"] == "concatenate_plans_chronological"
    assert meta["plan_source"] == "plans_array"
    assert [b["plan_id"] for b in chat_json] == ["plan-1", "plan-10"]
    assert [b["batch_number"] for b in chat_json] == [1, 2]
    sample = {
        "sample_id": "10M/1",
        "size": "10M",
        "conversation_id": "1",
        "chat": chat_json,
        "probing_questions": {},
    }
    units = iter_ingest_units(sample)
    assert len(units) == 3
    assert [u["unit_id"] for u in units] == ["b1_t1", "b1_t2", "b2_t1"]
    assert [u["session_key"] for u in units] == [
        "session_1",
        "session_2",
        "session_3",
    ]
    assert "early-a" in units[0]["text"]
    assert "late" in units[2]["text"]


def test_concatenate_plan_batches_renumbers_globally() -> None:
    chat_json, plan_meta = concatenate_plan_batches(
        [
            (
                "plan-1",
                [
                    {"turns": [_raw_turn("a")], "time_anchor": None},
                    {"turns": [_raw_turn("b")], "time_anchor": None},
                ],
            ),
            ("plan-2", [{"turns": [_raw_turn("c")], "time_anchor": None}]),
        ]
    )
    assert [b["batch_number"] for b in chat_json] == [1, 2, 3]
    assert plan_meta[0]["batch_number_start"] == 1
    assert plan_meta[0]["batch_number_end"] == 2
    assert plan_meta[1]["batch_number_start"] == 3
    assert ingest_target_turns(100) == 95
    assert ingest_target_turns(0) == 0


def test_write_conversation_10m_fixture(tmp_path: Path) -> None:
    row = {
        "conversation_id": "1",
        "conversation_seed": {"title": "Fixture 10M"},
        "chat": {
            "plan-1": [
                {
                    "batch_number": 1,
                    "turns": [_raw_turn("hello")],
                    "time_anchor": None,
                }
            ]
        },
        "plans": [],
        "probing_questions": {
            "information_extraction": [{"question": "What?", "rubric": ["hello"]}]
        },
    }
    dest = tmp_path / "10M" / "1"
    write_conversation(
        dest,
        size="10M",
        conversation_id="1",
        topic=row["conversation_seed"],
        probing_questions=row["probing_questions"],
        row=row,
    )
    meta = __import__("json").loads((dest / "meta.json").read_text(encoding="utf-8"))
    assert meta["brain_id"] == "beam10m1"
    assert meta["n_turns"] == 1
    assert meta["ingest_target"] == 1
    assert meta["normalize_strategy"] == "concatenate_plans_chronological"
    chat = __import__("json").loads((dest / "chat.json").read_text(encoding="utf-8"))
    units = iter_ingest_units(
        {
            "sample_id": "10M/1",
            "chat": chat,
            "probing_questions": {},
        }
    )
    assert len(units) == 1
    assert units[0]["unit_id"] == "b1_t1"


def test_format_turn_text_normalizes_time_anchor() -> None:
    text, ts = format_turn_text(
        [
            {
                "role": "user",
                "content": "Hello",
                "time_anchor": "March-15-2024",
                "question_type": "main_question",
            },
            {"role": "assistant", "content": "Hi"},
        ],
        batch_number=1,
        turn_index=0,
        global_turn=7,
    )
    assert "Session id: session_7." in text
    assert "Unit id: b1_t1." in text
    assert "user [main_question]" in text
    assert ts == "March 15, 2024"


def test_rubric_prompt_substitutes_placeholders() -> None:
    prompt = build_rubric_judge_prompt("Q?", "must say X", "X")
    assert "<question>" not in prompt
    assert "<rubric_item>" not in prompt
    assert "<llm_response>" not in prompt
    assert "Q?" in prompt
    assert "must say X" in prompt


def test_parse_json_response_keeps_half_scores() -> None:
    payload = parse_json_response('{"score": 0.5, "reason": "partial"}')
    assert float(payload["score"]) == 0.5


def test_aggregate_answers_uses_tau_for_event_ordering() -> None:
    errors = selftest_metrics()
    assert errors == []
    metrics = aggregate_answers(
        [
            {"ability": "event_ordering", "llm_judge_score": 0.1, "tau_norm": 0.9},
            {"ability": "summarization", "llm_judge_score": 0.5},
        ]
    )
    assert metrics["per_ability"]["event_ordering"]["mean"] == 0.9
    assert metrics["per_ability"]["summarization"]["mean"] == 0.5


def test_profile_defaults_sota() -> None:
    product = profile_defaults("product")
    sota = profile_defaults("sota")
    assert product["bench_profile"] == "product"
    assert product["sc_samples"] == 1
    assert sota["bench_profile"] == "sota"
    assert sota["sc_samples"] >= 3
    assert sota["gap_fill"] is True


def test_gap_fill_on_thin_ordering() -> None:
    plan = plan_gap_fill(
        "Can you list the order in which I brought up aspects?",
        "1. Auth 2. Tracking",
        {"text_context": "auth tracking analytics transactions"},
    )
    assert plan.needs_retry is True


def test_majority_vote_prefers_substantive() -> None:
    assert "March 29" in majority_vote(
        ["Not mentioned in the conversation", "March 29", "March 29"]
    )


def test_ability_system_prompt_includes_ordering_hint() -> None:
    text = ability_system_prompt("event_ordering")
    assert "ONLY N" in text or "milestone" in text.lower() or "aspect" in text.lower()


def test_ordering_aspect_queries_for_only_three() -> None:
    from beam.sota import ordering_aspect_queries, parse_only_n

    q = (
        "Can you list the order in which I brought up different aspects "
        "of developing my personal budget tracker throughout our conversations, "
        "in order? Mention ONLY and ONLY three items."
    )
    assert parse_only_n(q) == 3
    aspects = ordering_aspect_queries(q)
    assert len(aspects) >= 3
    blob = " ".join(aspects).lower()
    assert "transaction" in blob or "security" in blob


def test_ordering_aspect_queries_translation_domain() -> None:
    from beam.sota import ordering_aspect_queries

    q = (
        "How did my discussions about integrating and optimizing language and "
        "translation services progress in order? Mention ONLY and ONLY ten items."
    )
    aspects = ordering_aspect_queries(q)
    blob = " ".join(aspects).lower()
    assert "translation" in blob or "deepl" in blob or "language detection" in blob


def test_biography_abstain_skips_gap_fill() -> None:
    plan = plan_gap_fill(
        "Can you tell me about my background and previous development projects?",
        "Not mentioned in the conversation.",
        {"text_context": "Flask budget tracker authentication"},
    )
    assert plan.needs_retry is False


def test_lexical_aspect_match_accepts_paraphrase() -> None:
    from beam.judge import lexical_aspect_match, _strip_list_prefix

    assert _strip_list_prefix("1. Security hardening and deployment") == (
        "Security hardening and deployment"
    )
    assert lexical_aspect_match(
        "Security and deployment",
        "1. Security hardening and deployment",
    )
    assert lexical_aspect_match(
        "Transaction error handling",
        "Transaction CRUD and error handling",
    )


def test_permanent_embed_8192_detection() -> None:
    err = (
        "Error code: 400 - {'error': {'message': \"Invalid 'input': "
        "maximum context length is 8192 tokens.\"}}"
    )
    assert is_permanent_ingest_error(err)
    assert normalize_ingest_status("failed", err) == "permanent_failed"
    assert not is_permanent_ingest_error("Connection refused")
    assert normalize_ingest_status("failed", "Connection refused") == "failed"


def test_resume_skips_permanent_and_legacy_failed(tmp_path: Path) -> None:
    path = tmp_path / "ingest.jsonl"
    rows = [
        {
            "sample_id": "1M/1",
            "unit_id": "b1_t1",
            "status": "completed",
            "error": None,
        },
        {
            "sample_id": "1M/1",
            "unit_id": "b5_t25",
            "status": "failed",
            "error": "Invalid 'input': maximum context length is 8192 tokens.",
        },
        {
            "sample_id": "1M/1",
            "unit_id": "b2_t1",
            "status": "failed",
            "error": "Connection refused",
        },
        {
            "sample_id": "1M/1",
            "unit_id": "b6_t2",
            "status": "permanent_failed",
            "error": "maximum context length is 8192 tokens.",
        },
    ]
    path.write_text(
        "\n".join(__import__("json").dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    done = completed_unit_ids(path)
    assert "1M/1::b1_t1" in done
    assert "1M/1::b5_t25" in done
    assert "1M/1::b6_t2" in done
    assert "1M/1::b2_t1" not in done
