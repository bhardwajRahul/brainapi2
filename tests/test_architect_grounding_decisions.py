from types import SimpleNamespace

from src.core.saving.grounding import (
    decide_relationship_grounding,
    triage_relationships_for_janitor,
)
from src.core.saving.architect_batch import (
    BatchEndpoint,
    BatchExtractResponse,
    BatchRelationship,
    validate_batch_extract,
)


def _entity(uuid: str, name: str, type_: str, description: str = ""):
    return type(
        "E",
        (),
        {"uuid": uuid, "name": name, "type": type_, "description": description},
    )()


def test_validate_realigns_mismatched_span_offsets():
    source = "On Tuesday Alice paid Bob 12 dollars."
    entities = [_entity("a", "Alice", "PERSON"), _entity("b", "Bob", "PERSON")]
    span = "Alice paid Bob 12 dollars"
    payload = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="a", name="Alice", type="PERSON"),
                tip=BatchEndpoint(uuid="b", name="Bob", type="PERSON"),
                name="PAID",
                source_span=span,
                span_start=0,
                span_end=3,
            )
        ]
    )
    result = validate_batch_extract(
        payload, source_text=source, scout_entities=entities
    )
    assert len(result.accepted) == 1
    accepted = result.accepted[0]
    assert accepted.span_start is not None
    assert source[accepted.span_start : accepted.span_end] == span


def test_validate_fills_event_description_from_scout():
    source = "Alice hired Bob as a contractor in May 2024."
    entities = [
        _entity("u1", "Alice", "PERSON"),
        _entity(
            "u2",
            "Hiring",
            "EVENT",
            description="Alice hired Bob as a contractor in May 2024",
        ),
        _entity("u3", "Bob", "PERSON"),
    ]
    payload = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="Alice", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                name="MADE",
                source_span="Alice hired Bob as a contractor",
            )
        ]
    )
    result = validate_batch_extract(
        payload, source_text=source, scout_entities=entities
    )
    assert len(result.accepted) == 1
    event = result.accepted[0].tip
    assert (event.description or "").strip()
    assert "hired" in event.description.lower()


def test_validate_fills_event_description_from_source_span():
    source = "Alice hired Bob as a contractor in May 2024."
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
                source_span="Alice hired Bob as a contractor",
            )
        ]
    )
    result = validate_batch_extract(
        payload, source_text=source, scout_entities=entities
    )
    assert len(result.accepted) == 1
    assert result.accepted[0].tip.description == "Alice hired Bob as a contractor"


def test_validate_rejects_new_event_node_without_description():
    source = "Alice hired Bob as a contractor."
    entities = [
        _entity("u1", "Alice", "PERSON"),
        _entity("u3", "Bob", "PERSON"),
    ]
    payload = BatchExtractResponse(
        new_nodes=[BatchEndpoint(name="Hiring", type="EVENT")],
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="Alice", type="PERSON"),
                tip=BatchEndpoint(name="Hiring", type="EVENT"),
                name="MADE",
                source_span="Alice hired Bob as a contractor",
            )
        ],
    )
    result = validate_batch_extract(
        payload, source_text=source, scout_entities=entities
    )
    reasons = {issue.reason for issue in result.rejected}
    assert any(r.startswith("new_node_missing_event_description:") for r in reasons)


def test_grounding_decision_machine_readable():
    source = "Alice met Bob at the park."
    rel = SimpleNamespace(
        name="MET",
        description="Alice met Bob at the park",
        properties={"source_span": "Alice met Bob at the park"},
        tip=SimpleNamespace(name="Bob", type="PERSON", uuid="1"),
        tail=SimpleNamespace(name="Alice", type="PERSON", uuid="2"),
    )
    decision = decide_relationship_grounding(rel, source)
    assert decision.decision == "accept"
    assert decision.reason == "grounded_endpoints_ok"
    triage = triage_relationships_for_janitor([rel], source)
    assert len(triage.accept) == 1
    assert triage.reject == []
    assert triage.ambiguous == []
