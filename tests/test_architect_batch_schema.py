from src.core.saving.architect_batch import (
    BatchEndpoint,
    BatchExtractResponse,
    BatchRelationship,
    event_leg_incomplete,
    validate_batch_extract,
)


def _entity(uuid: str, name: str, type_: str):
    return type("E", (), {"uuid": uuid, "name": name, "type": type_})()


def test_validate_accepts_grounded_event_hub():
    source = "Alice hired Bob as a contractor in May 2024 for $5000."
    entities = [
        _entity("u1", "Alice", "PERSON"),
        _entity("u2", "Hiring", "EVENT"),
        _entity("u3", "Bob", "PERSON"),
    ]
    payload = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="Alice", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                name="MADE",
                description="Alice hired Bob",
                amount=5000,
                source_span="Alice hired Bob as a contractor",
                happened_at="2024-05",
            ),
            BatchRelationship(
                tail=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                tip=BatchEndpoint(uuid="u3", name="Bob", type="PERSON"),
                name="TARGETED",
                description="Alice hired Bob",
                amount=5000,
                source_span="Alice hired Bob as a contractor in May 2024",
            ),
        ]
    )
    result = validate_batch_extract(
        payload, source_text=source, scout_entities=entities
    )
    assert len(result.accepted) == 2
    assert result.rejected == []
    assert not event_leg_incomplete(result.accepted, entities)


def test_validate_rejects_type_named_and_ungrounded():
    source = "Alice met Bob at the park."
    entities = [
        _entity("u1", "Alice", "PERSON"),
        _entity("u2", "Bob", "PERSON"),
    ]
    payload = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="PERSON", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Bob", type="PERSON"),
                name="MET",
                source_span="Alice met Bob",
            ),
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="Alice", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Bob", type="PERSON"),
                name="MET",
                source_span="totally fabricated zebra claim",
            ),
            BatchRelationship(
                # well-formed unknown endpoint is soft-admitted (no escalate)
                tail=BatchEndpoint(uuid="missing", name="Carol", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Bob", type="PERSON"),
                name="MET",
                source_span="Alice met Bob at the park",
            ),
        ]
    )
    result = validate_batch_extract(
        payload, source_text=source, scout_entities=entities
    )
    reasons = {issue.reason for issue in result.rejected}
    assert "type_named_placeholder_endpoint" in reasons
    assert "source_span_not_grounded" in reasons
    assert "unknown_tail_endpoint" not in reasons
    assert len(result.accepted) == 1
    assert result.accepted[0].tail.name == "Carol"
    assert any(n.name == "Carol" for n in result.new_nodes)


def test_validate_admits_unknown_well_formed_endpoints():
    source = "Alice introduced Carol to Bob at the park."
    entities = [
        _entity("u1", "Alice", "PERSON"),
        _entity("u2", "Bob", "PERSON"),
    ]
    payload = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(name="Carol", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Bob", type="PERSON"),
                name="MET",
                source_span="Alice introduced Carol to Bob at the park",
            ),
        ]
    )
    result = validate_batch_extract(
        payload, source_text=source, scout_entities=entities
    )
    assert result.usable
    assert len(result.accepted) == 1
    assert result.rejected == []


def test_validate_span_offsets_and_scout_uuid_reuse():
    source = "On Tuesday Alice paid Bob 12 dollars."
    entities = [_entity("a", "Alice", "PERSON"), _entity("b", "Bob", "PERSON")]
    span = "Alice paid Bob 12 dollars"
    start = source.index(span)
    payload = {
        "new_nodes": [],
        "relationships": [
            {
                "tail": {"uuid": "a", "name": "Alice", "type": "PERSON"},
                "tip": {"uuid": "b", "name": "Bob", "type": "PERSON"},
                "name": "PAID",
                "amount": 12,
                "source_span": span,
                "span_start": start,
                "span_end": start + len(span),
            }
        ],
    }
    result = validate_batch_extract(
        payload, source_text=source, scout_entities=entities
    )
    assert len(result.accepted) == 1
    assert result.accepted[0].amount == 12


def test_event_leg_incomplete_detects_untouched_event():
    entities = [
        _entity("e1", "Trip", "EVENT"),
        _entity("p1", "Alice", "PERSON"),
    ]
    rels = [
        BatchRelationship(
            tail=BatchEndpoint(uuid="p1", name="Alice", type="PERSON"),
            tip=BatchEndpoint(uuid="p1", name="Alice", type="PERSON"),
            name="SELF",
            source_span="Alice",
        )
    ]
    assert event_leg_incomplete(rels, entities)


def test_validate_partial_keeps_usable_and_drops_bad_new_nodes():
    source = "Alice hired Bob as a contractor in May 2024."
    entities = [
        _entity("u1", "Alice", "PERSON"),
        _entity("u2", "Hiring", "EVENT"),
        _entity("u3", "Bob", "PERSON"),
    ]
    payload = BatchExtractResponse(
        new_nodes=[
            BatchEndpoint(name="PERSON", type="PERSON"),
            BatchEndpoint(name="Contract", type="DOCUMENT"),
        ],
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="Alice", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                name="MADE",
                source_span="Alice hired Bob as a contractor",
            ),
            BatchRelationship(
                tail=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                tip=BatchEndpoint(uuid="missing", name="Carol", type="PERSON"),
                name="TARGETED",
                source_span="Alice hired Bob as a contractor in May 2024",
            ),
        ],
    )
    result = validate_batch_extract(
        payload, source_text=source, scout_entities=entities
    )
    assert result.usable
    assert not result.ok
    assert len(result.accepted) == 2
    assert {r.name for r in result.accepted} == {"MADE", "TARGETED"}
    assert len(result.new_nodes) >= 2
    assert any(n.name == "Contract" for n in result.new_nodes)
    assert any(n.name == "Carol" for n in result.new_nodes)
    reasons = {issue.reason for issue in result.rejected}
    assert "unknown_tip_endpoint" not in reasons
    assert any(r.startswith("new_node_type_named_placeholder:") for r in reasons)
