from types import SimpleNamespace

from src.core.saving.grounding import (
    align_span,
    cheap_janitor_precheck,
    decide_relationship_grounding,
    endpoints_well_formed,
    find_span_offsets,
    realign_span_offsets,
    relationship_looks_grounded,
    triage_relationships_for_janitor,
)


def test_align_span_exact_and_fuzzy():
    source = "Alice met Bob at the park on Tuesday afternoon."
    score, _ = align_span("Alice met Bob at the park", source)
    assert score == 1.0
    score2, window = align_span("Bob at the park on Tuesday", source)
    assert score2 >= 0.45
    assert "bob" in window or "park" in window


def test_find_span_offsets_exact_and_whitespace():
    source = "On Tuesday Alice paid Bob 12 dollars."
    span = "Alice paid Bob 12 dollars"
    start, end = find_span_offsets(span, source)
    assert source[start:end] == span

    spaced = "Alice   paid Bob 12 dollars"
    found = find_span_offsets(spaced, source)
    assert found is not None
    assert "alice" in source[found[0] : found[1]].lower()


def test_realign_span_offsets_repairs_mismatch():
    source = "On Tuesday Alice paid Bob 12 dollars."
    span = "Alice paid Bob 12 dollars"
    ok, start, end, reason = realign_span_offsets(
        span=span,
        source_text=source,
        span_start=0,
        span_end=5,
    )
    assert ok and reason == ""
    assert start is not None and end is not None
    assert source[start:end] == span


def test_realign_clears_bad_offsets_when_span_grounded():
    source = "Alice hired Bob as a contractor in May."
    span = "Alice hired Bob as a contractor"
    # Offsets point nowhere useful and span cannot be exact-found if we
    # mutate — use a grounded span with impossible offsets that find can fix.
    ok, start, end, reason = realign_span_offsets(
        span=span,
        source_text=source,
        span_start=-1,
        span_end=9999,
    )
    assert ok and reason == ""
    assert start is not None
    assert source[start:end] == span


def test_endpoints_reject_type_named_placeholders():
    good = SimpleNamespace(
        name="MET",
        tip=SimpleNamespace(name="Bob", type="PERSON"),
        tail=SimpleNamespace(name="Alice", type="PERSON"),
    )
    bad = SimpleNamespace(
        name="PAID",
        tip=SimpleNamespace(name="Money", type="MONEY"),
        tail=SimpleNamespace(name="Alice", type="PERSON"),
    )
    assert endpoints_well_formed(good)
    assert not endpoints_well_formed(bad)


def test_cheap_janitor_precheck_splits():
    source = "Alice hired Bob as a contractor in May."
    grounded = SimpleNamespace(
        name="HIRED",
        description="Alice hired Bob as a contractor",
        properties={},
        tip=SimpleNamespace(name="Bob", type="PERSON", uuid="1"),
        tail=SimpleNamespace(name="Alice", type="PERSON", uuid="2"),
    )
    ungrounded = SimpleNamespace(
        name="HIRED",
        description="totally fabricated claim about zebras",
        properties={},
        tip=SimpleNamespace(name="Bob", type="PERSON", uuid="1"),
        tail=SimpleNamespace(name="Alice", type="PERSON", uuid="2"),
    )
    skip, need = cheap_janitor_precheck([grounded, ungrounded], source)
    assert grounded in skip
    assert ungrounded in need
    ok, score, _ = relationship_looks_grounded(grounded, source)
    assert ok and score >= 0.45


def test_triage_accept_reject_ambiguous():
    source = "Alice hired Bob as a contractor in May."
    accept_rel = SimpleNamespace(
        name="HIRED",
        description="Alice hired Bob as a contractor",
        properties={"source_span": "Alice hired Bob as a contractor"},
        tip=SimpleNamespace(name="Bob", type="PERSON", uuid="1"),
        tail=SimpleNamespace(name="Alice", type="PERSON", uuid="2"),
    )
    reject_placeholder = SimpleNamespace(
        name="HIRED",
        description="Alice hired Bob",
        properties={},
        tip=SimpleNamespace(name="PERSON", type="PERSON", uuid="1"),
        tail=SimpleNamespace(name="Alice", type="PERSON", uuid="2"),
    )
    reject_ungrounded = SimpleNamespace(
        name="HIRED",
        description="zebras invented a spaceship yesterday",
        properties={"source_span": "zebras invented a spaceship yesterday"},
        tip=SimpleNamespace(name="Bob", type="PERSON", uuid="1"),
        tail=SimpleNamespace(name="Alice", type="PERSON", uuid="2"),
    )
    ambiguous = SimpleNamespace(
        name="HIRED",
        description="",
        properties={},
        tip=SimpleNamespace(name="Bob", type="PERSON", uuid="1"),
        tail=SimpleNamespace(name="Alice", type="PERSON", uuid="2"),
    )
    triage = triage_relationships_for_janitor(
        [accept_rel, reject_placeholder, reject_ungrounded, ambiguous],
        source,
    )
    assert accept_rel in triage.accept
    reject_reasons = {d.reason for d in triage.reject}
    assert "type_named_placeholder_endpoint" in reject_reasons
    assert "source_span_not_grounded" in reject_reasons
    assert ambiguous in triage.ambiguous


def test_decide_relationship_grounding_accept_sets_span():
    source = "Alice hired Bob as a contractor in May."
    rel = SimpleNamespace(
        name="HIRED",
        description="Alice hired Bob as a contractor",
        properties={},
        tip=SimpleNamespace(name="Bob", type="PERSON", uuid="1"),
        tail=SimpleNamespace(name="Alice", type="PERSON", uuid="2"),
    )
    decision = decide_relationship_grounding(rel, source)
    assert decision.decision == "accept"
    assert decision.score >= 0.45
    assert rel.properties.get("source_span")
    assert "grounding_score" in rel.properties
