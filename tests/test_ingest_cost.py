from src.core.saving.ingest_cost import (
    SOURCE_TOKENIZER_ID,
    IngestCostLedger,
    StageCost,
    count_source_tokens,
    merge_cost_into_status_payload,
    record_usage_from_response,
    track_stage,
)


def test_stage_cost_accumulates():
    stage = StageCost()
    stage.add_usage(input_tokens=10, output_tokens=5, cached_tokens=2, calls=1)
    stage.add_usage(input_tokens=3, output_tokens=1, calls=1)
    assert stage.input_tokens == 13
    assert stage.output_tokens == 6
    assert stage.cached_tokens == 2
    assert stage.calls == 2
    assert stage.total_tokens == 19


def test_ledger_roundtrip_and_merge():
    a = IngestCostLedger()
    a.scout.add_usage(input_tokens=100, output_tokens=20, calls=2)
    a.janitor_skipped = 3
    payload = a.to_dict()
    assert payload["total_llm_tokens"] == 120
    assert payload["stages"]["scout"]["calls"] == 2

    b = IngestCostLedger.from_dict(payload)
    b.architect.add_usage(input_tokens=50, output_tokens=10, calls=1)
    a.merge(b)
    assert a.scout.input_tokens == 200
    assert a.architect.input_tokens == 50
    assert a.janitor_skipped == 6


def test_track_stage_records_usage_from_response():
    ledger = IngestCostLedger()
    with track_stage(ledger, "architect"):
        record_usage_from_response(
            {"usage_metadata": {"input_tokens": 40, "output_tokens": 8}}
        )
    assert ledger.architect.input_tokens == 40
    assert ledger.architect.output_tokens == 8
    assert ledger.architect.calls == 1
    assert ledger.architect.latency_ms >= 0


def test_merge_cost_into_status_payload():
    existing = {"status": "started"}
    ledger = IngestCostLedger()
    ledger.scout.add_usage(input_tokens=1, output_tokens=1, calls=1)
    merged = merge_cost_into_status_payload(existing, ledger)
    assert merged["cost"]["stages"]["scout"]["input_tokens"] == 1
    again = merge_cost_into_status_payload(
        merged, IngestCostLedger.from_dict(merged["cost"])
    )
    assert again["cost"]["stages"]["scout"]["input_tokens"] == 2


def test_source_token_multiplier_uses_pinned_tokenizer():
    ledger = IngestCostLedger()
    text = "Alice hired Bob as a contractor in May."
    ledger.set_source_text(text)
    tokens, tokenizer_id, estimated = count_source_tokens(text)
    assert tokens == ledger.source_tokens
    assert tokenizer_id == ledger.source_tokenizer
    assert not estimated
    assert ledger.source_tokenizer == SOURCE_TOKENIZER_ID
    assert ledger.source_chars == len(text)

    ledger.architect.add_usage(input_tokens=90, output_tokens=10, calls=2)
    payload = ledger.to_dict()
    assert payload["llm_source_multiplier"] == round(100 / float(tokens), 3)
    assert payload["source_tokenizer"] == SOURCE_TOKENIZER_ID
    assert payload["source_tokens"] == tokens


def test_escalate_and_janitor_rates():
    ledger = IngestCostLedger()
    ledger.record_architect_unit(escalated=False, schema_calls=1)
    ledger.record_architect_unit(
        escalated=True, reason="schema_empty_or_all_rejected", schema_calls=2, repair_calls=1
    )
    ledger.janitor_skipped = 7
    ledger.janitor_ran = 3
    payload = ledger.to_dict()
    assert payload["architect_units"] == 2
    assert payload["architect_escalations"] == 1
    assert payload["escalate_rate"] == 0.5
    assert payload["janitor_skip_rate"] == 0.7
    assert payload["architect_repair_calls"] == 1
    assert "schema_empty_or_all_rejected" in payload["escalate_reasons"]
