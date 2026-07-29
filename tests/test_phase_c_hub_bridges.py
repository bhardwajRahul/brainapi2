import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.kg import Node, Predicate
from src.core.saving.hub_bridges import (
    HubBridge,
    bridges_from_memberships,
    canonical_event_pair,
    entity_event_memberships,
    select_bridge_neighbors,
)
from src.services.api.controllers import retrieve as retrieve_mod


def _node(uuid: str, name: str, *, labels=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        properties={},
    )


def _pred(uuid: str, name: str, *, flow_key: str = ""):
    return Predicate(
        uuid=uuid,
        name=name,
        description=f"desc-{name}",
        flow_key=flow_key or None,
        properties={},
    )


def _candidate(
    *,
    score: float,
    r_uuid: str,
    r2_uuid: str,
    flow_key: str,
    m_uuid: str,
    n_uuid: str = "n",
    b_uuid: str = "b",
):
    n = _node(n_uuid, n_uuid)
    r = _pred(r_uuid, "MADE", flow_key=flow_key)
    m = _node(m_uuid, m_uuid, labels=["EVENT"])
    r2 = _pred(r2_uuid, "TARGETED", flow_key=flow_key)
    b = _node(b_uuid, b_uuid)
    return {
        "identified_entity": "q",
        "triple": (n, r, m, r2, b),
        "score": score,
        "key": (r_uuid, r2_uuid),
        "text": f"fact-{m_uuid}",
        "chunk_ids": [],
    }


class HubBridgeIndexTests(unittest.TestCase):
    def test_canonical_pair_orders_lexicographically(self):
        self.assertEqual(canonical_event_pair("b", "a"), ("a", "b"))

    def test_build_bridges_from_shared_actor(self):
        rows = [
            ("event-1", "melanie", "Melanie", "MADE"),
            ("event-2", "melanie", "Melanie", "MADE"),
            ("event-1", "park", "Park", "OCCURRED_WITHIN"),
            ("event-2", "park", "Park", "OCCURRED_WITHIN"),
            ("event-3", "jon", "Jon", "TARGETED"),
        ]
        membership = entity_event_memberships(rows)
        bridges = bridges_from_memberships(membership)
        self.assertEqual(len(bridges), 1)
        self.assertEqual(bridges[0].event_a, "event-1")
        self.assertEqual(bridges[0].event_b, "event-2")
        self.assertEqual(bridges[0].shared_entity, "melanie")

    def test_rebuild_idempotent_membership(self):
        rows = [
            ("e2", "ent", "Ent", "MADE"),
            ("e1", "ent", "Ent", "TARGETED"),
            ("e2", "ent", "Ent", "MADE"),
        ]
        first = bridges_from_memberships(entity_event_memberships(rows))
        second = bridges_from_memberships(entity_event_memberships(rows))
        self.assertEqual(first, second)

    def test_select_bridge_neighbors_respects_max_and_determinism(self):
        bridges = [
            HubBridge("e1", "e2", "a", "A", 1.0),
            HubBridge("e1", "e3", "b", "B", 2.0),
            HubBridge("e1", "e4", "c", "C", 1.5),
        ]
        a = select_bridge_neighbors(["e1"], bridges, max_per_hub=2)
        b = select_bridge_neighbors(["e1"], bridges, max_per_hub=2)
        self.assertEqual(a, b)
        self.assertEqual([n for n, _ in a], ["e3", "e4"])

    def test_prefer_bridge_to_novel_source_session(self):
        bridges = [
            HubBridge("e1", "e_same", "shared", "Shared", 2.0),
            HubBridge("e1", "e_novel", "shared", "Shared", 1.0),
        ]
        chosen = select_bridge_neighbors(
            ["e1"],
            bridges,
            max_per_hub=1,
            seed_sessions={"session_1"},
            hub_sessions={
                "e_same": {"session_1"},
                "e_novel": {"session_5"},
            },
        )
        self.assertEqual([n for n, _ in chosen], ["e_novel"])

    def test_novel_session_ranking_is_deterministic(self):
        bridges = [
            HubBridge("e1", "e_a", "ent", "Ent", 1.0),
            HubBridge("e1", "e_b", "ent", "Ent", 1.0),
            HubBridge("e1", "e_c", "ent", "Ent", 1.0),
        ]
        kwargs = dict(
            seed_sessions={"session_1"},
            hub_sessions={
                "e_a": {"session_1"},
                "e_b": {"session_5"},
                "e_c": {"session_1", "session_7"},
            },
            max_per_hub=2,
        )
        a = select_bridge_neighbors(["e1"], bridges, **kwargs)
        b = select_bridge_neighbors(["e1"], bridges, **kwargs)
        self.assertEqual(a, b)
        self.assertEqual([n for n, _ in a], ["e_b", "e_c"])


class BridgeExpandTests(unittest.TestCase):
    def test_one_bridge_expand_adds_cross_event_candidates(self):
        seed = _candidate(
            score=0.1,
            r_uuid="r1",
            r2_uuid="r2",
            flow_key="f1",
            m_uuid="event-seed",
            n_uuid="actor",
            b_uuid="obj1",
        )
        bridged_n = _node("actor2", "actor2")
        bridged_r = _pred("r3", "MADE", flow_key="f2")
        bridged_m = _node("event-bridge", "event-bridge", labels=["EVENT"])
        bridged_r2 = _pred("r4", "TARGETED", flow_key="f2")
        bridged_b = _node("obj2", "obj2")
        bridge = HubBridge(
            event_a="event-bridge",
            event_b="event-seed",
            shared_entity="actor",
            shared_entity_name="actor",
            weight=1.0,
        )

        mock_adapter = MagicMock()
        mock_adapter.get_hub_bridges.return_value = [bridge]
        mock_adapter.get_event_hub_facts.return_value = [
            (bridged_n, bridged_r, bridged_m, bridged_r2, bridged_b)
        ]

        with patch.object(retrieve_mod, "graph_adapter", mock_adapter):
            expanded, paths = retrieve_mod._expand_cross_event_bridges(
                [seed],
                "brain",
                max_per_hub=3,
            )

        self.assertEqual(len(expanded), 2)
        hubs = {retrieve_mod._event_hub_id(c) for c in expanded}
        self.assertEqual(hubs, {"event-seed", "event-bridge"})
        self.assertEqual(len(paths), 1)
        self.assertEqual(sorted(paths[0]["hubs"]), ["event-bridge", "event-seed"])

    def test_paths_for_curated_attaches_legs(self):
        seed = _candidate(
            score=0.1,
            r_uuid="r1",
            r2_uuid="r2",
            flow_key="f1",
            m_uuid="event-seed",
        )
        bridged = _candidate(
            score=0.2,
            r_uuid="r3",
            r2_uuid="r4",
            flow_key="f2",
            m_uuid="event-bridge",
            n_uuid="actor2",
            b_uuid="obj2",
        )
        path_meta = [
            {
                "hubs": ["event-bridge", "event-seed"],
                "shared_entity": "actor",
                "shared_entity_name": "actor",
                "weight": 1.0,
            }
        ]
        kept = retrieve_mod._paths_for_curated(path_meta, [seed, bridged])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["legs"], ["fact-event-bridge", "fact-event-seed"])
        dropped = retrieve_mod._paths_for_curated(path_meta, [seed])
        self.assertEqual(dropped, [])

    def test_bridge_expand_disabled_when_zero(self):
        seed = _candidate(
            score=0.1,
            r_uuid="r1",
            r2_uuid="r2",
            flow_key="f1",
            m_uuid="event-seed",
        )
        expanded, paths = retrieve_mod._expand_cross_event_bridges(
            [seed],
            "brain",
            max_per_hub=0,
        )
        self.assertEqual(expanded, [seed])
        self.assertEqual(paths, [])

    def test_diversify_after_expand_respects_max_facts(self):
        candidates = [
            _candidate(
                score=0.01 * i,
                r_uuid=f"r{i}",
                r2_uuid=f"rr{i}",
                flow_key=f"f{i}",
                m_uuid=f"event-{i}",
                n_uuid=f"n{i}",
                b_uuid=f"b{i}",
            )
            for i in range(8)
        ]
        curated = retrieve_mod._diversify_facts(candidates, max_facts=3)
        self.assertEqual(len(curated), 3)

    def test_expand_then_diversify_deterministic(self):
        seed = _candidate(
            score=0.2,
            r_uuid="r1",
            r2_uuid="r2",
            flow_key="f1",
            m_uuid="event-seed",
        )
        bridged = []
        for i in range(3):
            bridged.append(
                (
                    _node(f"n{i}", f"n{i}"),
                    _pred(f"br{i}", "MADE", flow_key=f"bf{i}"),
                    _node(f"event-b{i}", f"event-b{i}", labels=["EVENT"]),
                    _pred(f"brr{i}", "TARGETED", flow_key=f"bf{i}"),
                    _node(f"bb{i}", f"bb{i}"),
                )
            )
        bridges = [
            HubBridge("event-b0", "event-seed", "shared", "Shared", 1.0),
            HubBridge("event-b1", "event-seed", "shared", "Shared", 1.0),
            HubBridge("event-b2", "event-seed", "shared", "Shared", 1.0),
        ]
        mock_adapter = MagicMock()
        mock_adapter.get_hub_bridges.return_value = bridges
        mock_adapter.get_event_hub_facts.return_value = bridged

        with patch.object(retrieve_mod, "graph_adapter", mock_adapter):
            a, _ = retrieve_mod._expand_cross_event_bridges(
                [seed], "brain", max_per_hub=2
            )
            b, _ = retrieve_mod._expand_cross_event_bridges(
                [seed], "brain", max_per_hub=2
            )
        self.assertEqual(
            [c["key"] for c in a],
            [c["key"] for c in b],
        )
        curated_a = retrieve_mod._diversify_facts(
            retrieve_mod._rank_facts_with_completeness(a), max_facts=2
        )
        curated_b = retrieve_mod._diversify_facts(
            retrieve_mod._rank_facts_with_completeness(b), max_facts=2
        )
        self.assertEqual(
            [c["key"] for c in curated_a],
            [c["key"] for c in curated_b],
        )
        self.assertEqual(len(curated_a), 2)


    def test_expand_selects_by_weight_without_session_pool(self):
        seed = _candidate(
            score=0.1,
            r_uuid="r1",
            r2_uuid="r2",
            flow_key="f1",
            m_uuid="event-seed",
        )
        seed["session_ids"] = ["session_1"]
        seed["chunk_ids"] = ["chunk-seed"]

        same_n = _node("n-same", "n-same")
        same_r = _pred("rs", "MADE", flow_key="fs")
        same_m = _node("event-same", "event-same", labels=["EVENT"])
        same_r2 = _pred("rrs", "TARGETED", flow_key="fs")
        same_b = _node("b-same", "b-same")
        for item in (same_n, same_r, same_m, same_r2, same_b):
            item.properties = {"source_chunk_ids": ["chunk-same"]}

        novel_n = _node("n-novel", "n-novel")
        novel_r = _pred("rn", "MADE", flow_key="fn")
        novel_m = _node("event-novel", "event-novel", labels=["EVENT"])
        novel_r2 = _pred("rrn", "TARGETED", flow_key="fn")
        novel_b = _node("b-novel", "b-novel")
        for item in (novel_n, novel_r, novel_m, novel_r2, novel_b):
            item.properties = {"source_chunk_ids": ["chunk-novel"]}

        bridges = [
            HubBridge("event-same", "event-seed", "shared", "Shared", 2.0),
            HubBridge("event-novel", "event-seed", "shared", "Shared", 1.0),
        ]
        mock_adapter = MagicMock()
        mock_adapter.get_hub_bridges.return_value = bridges
        mock_adapter.get_event_hub_facts.return_value = [
            (same_n, same_r, same_m, same_r2, same_b),
            (novel_n, novel_r, novel_m, novel_r2, novel_b),
        ]

        chunk_sessions = {
            "chunk-seed": ["session_1"],
            "chunk-same": ["session_1"],
            "chunk-novel": ["session_5"],
        }

        with patch.object(retrieve_mod, "graph_adapter", mock_adapter):
            with patch.object(
                retrieve_mod,
                "_resolve_chunk_sessions",
                side_effect=lambda ids, _brain: {
                    cid: chunk_sessions[cid]
                    for cid in ids
                    if cid in chunk_sessions
                },
            ):
                expanded, _paths = retrieve_mod._expand_cross_event_bridges(
                    [seed],
                    "brain",
                    max_per_hub=1,
                )

        hubs = {retrieve_mod._event_hub_id(c) for c in expanded}
        self.assertIn("event-seed", hubs)
        self.assertIn("event-same", hubs)
        self.assertNotIn("event-novel", hubs)
        mock_adapter.get_event_hub_facts.assert_called_once()
        fetched = mock_adapter.get_event_hub_facts.call_args[0][0]
        self.assertEqual(fetched, ["event-same"])

    def test_reserved_slots_keep_novel_bridge_session_under_max_facts(self):
        seeds = [
            _candidate(
                score=0.01 * i,
                r_uuid=f"rs{i}",
                r2_uuid=f"rrs{i}",
                flow_key=f"fs{i}",
                m_uuid=f"event-seed-{i}",
                n_uuid=f"ns{i}",
                b_uuid=f"bs{i}",
            )
            for i in range(8)
        ]
        for seed in seeds:
            seed["session_ids"] = ["session_1"]

        bridges = [
            _candidate(
                score=5.0 + 0.01 * i,
                r_uuid=f"rb{i}",
                r2_uuid=f"rrb{i}",
                flow_key=f"fb{i}",
                m_uuid=f"event-bridge-{i}",
                n_uuid=f"nb{i}",
                b_uuid=f"bb{i}",
            )
            for i in range(3)
        ]
        for bridge in bridges:
            bridge["session_ids"] = ["session_5"]
            bridge["bridge"] = {
                "from_hub": "event-seed-0",
                "to_hub": bridge["triple"][2].uuid,
                "shared_entity": "shared",
                "shared_entity_name": "Shared",
            }

        ranked = seeds + bridges
        max_facts = 5

        pool = [retrieve_mod._prepare_diversify_item(c) for c in ranked]
        pool.sort(key=lambda item: (-item["relevance"], item["tie"]))
        without_reserve: list[dict] = []
        selected_meta: list[dict] = []
        selected_hubs: set[str] = set()
        selected_sessions: set[str] = set()
        while len(without_reserve) < max_facts and pool:
            idx = retrieve_mod._mmr_pick_index(
                pool, selected_meta, selected_hubs, selected_sessions
            )
            chosen = pool.pop(idx)
            without_reserve.append(chosen["candidate"])
            selected_meta.append(chosen)
            selected_hubs.add(chosen["hub"])
            selected_sessions.update(chosen["sessions"])
        without_sessions = {
            s for c in without_reserve for s in (c.get("session_ids") or [])
        }
        self.assertNotIn("session_5", without_sessions)

        curated = retrieve_mod._diversify_facts(ranked, max_facts=max_facts)
        self.assertEqual(len(curated), max_facts)
        curated_sessions = {
            s for c in curated for s in (c.get("session_ids") or [])
        }
        self.assertIn("session_5", curated_sessions)
        self.assertTrue(any(c.get("bridge") for c in curated))

        again = retrieve_mod._diversify_facts(
            list(reversed(ranked)), max_facts=max_facts
        )
        self.assertEqual(
            [c["key"] for c in curated],
            [c["key"] for c in again],
        )

    def test_build_fact_channel_keeps_candidate_session_ids(self):
        seed = _candidate(
            score=0.1,
            r_uuid="r1",
            r2_uuid="r2",
            flow_key="f1",
            m_uuid="event-seed",
        )
        seed["session_ids"] = ["session_1"]
        seed["chunk_ids"] = ["chunk-missing-from-map"]
        bridge = _candidate(
            score=5.0,
            r_uuid="rb",
            r2_uuid="rrb",
            flow_key="fb",
            m_uuid="event-bridge",
            n_uuid="nb",
            b_uuid="bb",
        )
        bridge["session_ids"] = ["session_5"]
        bridge["chunk_ids"] = ["chunk-bridge-missing"]
        bridge["bridge"] = {
            "from_hub": "event-seed",
            "to_hub": "event-bridge",
            "shared_entity": "shared",
            "shared_entity_name": "Shared",
        }
        curated = retrieve_mod._diversify_facts([seed, bridge], max_facts=2)
        text_lines, triples, graph_sessions = retrieve_mod._build_fact_channel(
            curated, chunk_sessions={}
        )
        self.assertIn("session_5", graph_sessions)
        self.assertTrue(any("session_5" in line for line in text_lines))
        emitted = {
            sid
            for t in triples
            for sid in (t.source_session_ids or [])
        }
        self.assertIn("session_5", emitted)

    def test_rank_bridge_seed_hubs_total_order_ignores_arrival(self):
        candidates = [
            _candidate(
                score=0.4,
                r_uuid="r1",
                r2_uuid="x1",
                flow_key="f1",
                m_uuid="hub-b",
            ),
            _candidate(
                score=0.1,
                r_uuid="r2",
                r2_uuid="x2",
                flow_key="f2",
                m_uuid="hub-a",
            ),
            _candidate(
                score=0.1,
                r_uuid="r3",
                r2_uuid="x3",
                flow_key="f3",
                m_uuid="hub-c",
            ),
            _candidate(
                score=0.2,
                r_uuid="r4",
                r2_uuid="x4",
                flow_key="f4",
                m_uuid="hub-b",
            ),
        ]
        a = retrieve_mod._rank_bridge_seed_hubs(candidates, cap=3)
        b = retrieve_mod._rank_bridge_seed_hubs(list(reversed(candidates)), cap=3)
        self.assertEqual(a, b)
        self.assertEqual(a, ["hub-a", "hub-c", "hub-b"])

    def test_expand_marks_reserve_ok_only_for_top_reserve_hubs(self):
        seeds = [
            _candidate(
                score=0.01 * i,
                r_uuid=f"rs{i}",
                r2_uuid=f"rrs{i}",
                flow_key=f"fs{i}",
                m_uuid=f"event-seed-{i:02d}",
                n_uuid=f"ns{i}",
                b_uuid=f"bs{i}",
            )
            for i in range(10)
        ]
        bridges = [
            HubBridge(
                f"event-bridge-{i}",
                f"event-seed-{i:02d}",
                f"ent-{i}",
                f"Ent{i}",
                1.0,
            )
            for i in range(10)
        ]
        facts = []
        for i in range(10):
            facts.append(
                (
                    _node(f"nb{i}", f"nb{i}"),
                    _pred(f"rb{i}", "MADE", flow_key=f"fb{i}"),
                    _node(
                        f"event-bridge-{i}",
                        f"event-bridge-{i}",
                        labels=["EVENT"],
                    ),
                    _pred(f"rrb{i}", "TARGETED", flow_key=f"fb{i}"),
                    _node(f"bb{i}", f"bb{i}"),
                )
            )
        mock_adapter = MagicMock()
        mock_adapter.get_hub_bridges.return_value = bridges
        mock_adapter.get_event_hub_facts.return_value = facts
        with patch.object(retrieve_mod, "graph_adapter", mock_adapter):
            expanded, _ = retrieve_mod._expand_cross_event_bridges(
                seeds, "brain", max_per_hub=1
            )
        by_to = {
            c["bridge"]["to_hub"]: c["bridge"]
            for c in expanded
            if c.get("bridge")
        }
        reserve_cap = retrieve_mod._BRIDGE_RESERVE_HUB_CAP
        for i in range(10):
            meta = by_to[f"event-bridge-{i}"]
            self.assertEqual(meta["reserve_ok"], i < reserve_cap)

    def test_reserved_slots_ignore_unstable_tail_hub_bridges(self):
        seeds = [
            _candidate(
                score=0.01 * i,
                r_uuid=f"rs{i}",
                r2_uuid=f"rrs{i}",
                flow_key=f"fs{i}",
                m_uuid=f"event-seed-{i}",
                n_uuid=f"ns{i}",
                b_uuid=f"bs{i}",
            )
            for i in range(8)
        ]
        for seed in seeds:
            seed["session_ids"] = ["session_1"]

        stable = _candidate(
            score=5.0,
            r_uuid="rb0",
            r2_uuid="rrb0",
            flow_key="fb0",
            m_uuid="event-bridge-stable",
            n_uuid="nb0",
            b_uuid="bb0",
        )
        stable["session_ids"] = ["session_5"]
        stable["bridge"] = {
            "from_hub": "event-seed-0",
            "to_hub": "event-bridge-stable",
            "shared_entity": "shared",
            "shared_entity_name": "Shared",
            "reserve_ok": True,
        }

        jitter_a = _candidate(
            score=5.1,
            r_uuid="rb1",
            r2_uuid="rrb1",
            flow_key="fb1",
            m_uuid="event-bridge-jitter-a",
            n_uuid="nb1",
            b_uuid="bb1",
        )
        jitter_a["session_ids"] = ["session_9"]
        jitter_a["bridge"] = {
            "from_hub": "event-seed-9",
            "to_hub": "event-bridge-jitter-a",
            "shared_entity": "shared",
            "shared_entity_name": "Shared",
            "reserve_ok": False,
        }

        jitter_b = _candidate(
            score=5.1,
            r_uuid="rb2",
            r2_uuid="rrb2",
            flow_key="fb2",
            m_uuid="event-bridge-jitter-b",
            n_uuid="nb2",
            b_uuid="bb2",
        )
        jitter_b["session_ids"] = ["session_13"]
        jitter_b["bridge"] = {
            "from_hub": "event-seed-9",
            "to_hub": "event-bridge-jitter-b",
            "shared_entity": "shared",
            "shared_entity_name": "Shared",
            "reserve_ok": False,
        }

        curated_a = retrieve_mod._diversify_facts(
            seeds + [stable, jitter_a], max_facts=5
        )
        curated_b = retrieve_mod._diversify_facts(
            seeds + [stable, jitter_b], max_facts=5
        )
        sessions_a = {
            s for c in curated_a for s in (c.get("session_ids") or [])
        }
        sessions_b = {
            s for c in curated_b for s in (c.get("session_ids") or [])
        }
        self.assertEqual(sessions_a, sessions_b)
        self.assertIn("session_5", sessions_a)
        self.assertNotIn("session_9", sessions_a)
        self.assertNotIn("session_13", sessions_b)

    def test_expand_then_diversify_stable_under_tail_seed_jitter(self):
        core = [
            _candidate(
                score=0.01 * i,
                r_uuid=f"rs{i}",
                r2_uuid=f"rrs{i}",
                flow_key=f"fs{i}",
                m_uuid=f"event-seed-{i:02d}",
                n_uuid=f"ns{i}",
                b_uuid=f"bs{i}",
            )
            for i in range(retrieve_mod._BRIDGE_RESERVE_HUB_CAP)
        ]
        for seed in core:
            seed["session_ids"] = ["session_1"]
            seed["chunk_ids"] = ["chunk-seed"]

        def run_with_tail(tail_hub: str, novel_hub: str, novel_session: str):
            tail = _candidate(
                score=0.9,
                r_uuid=f"rt-{tail_hub}",
                r2_uuid=f"rrt-{tail_hub}",
                flow_key=f"ft-{tail_hub}",
                m_uuid=tail_hub,
                n_uuid=f"nt-{tail_hub}",
                b_uuid=f"bt-{tail_hub}",
            )
            tail["session_ids"] = ["session_1"]
            tail["chunk_ids"] = ["chunk-seed"]
            seeds = core + [tail]
            bridges = [
                HubBridge("event-bridge-core", "event-seed-00", "ent-c", "C", 1.0),
                HubBridge(novel_hub, tail_hub, "ent-t", "T", 1.0),
            ]
            core_fact = (
                _node("nbc", "nbc"),
                _pred("rbc", "MADE", flow_key="fbc"),
                _node("event-bridge-core", "event-bridge-core", labels=["EVENT"]),
                _pred("rrbc", "TARGETED", flow_key="fbc"),
                _node("bbc", "bbc"),
            )
            for item in core_fact:
                item.properties = {"source_chunk_ids": ["chunk-core"]}
            novel_fact = (
                _node("nbn", "nbn"),
                _pred("rbn", "MADE", flow_key="fbn"),
                _node(novel_hub, novel_hub, labels=["EVENT"]),
                _pred("rrbn", "TARGETED", flow_key="fbn"),
                _node("bbn", "bbn"),
            )
            for item in novel_fact:
                item.properties = {"source_chunk_ids": ["chunk-novel"]}
            mock_adapter = MagicMock()
            mock_adapter.get_hub_bridges.return_value = bridges
            mock_adapter.get_event_hub_facts.return_value = [core_fact, novel_fact]
            chunk_sessions = {
                "chunk-seed": ["session_1"],
                "chunk-core": ["session_5"],
                "chunk-novel": [novel_session],
            }
            with patch.object(retrieve_mod, "graph_adapter", mock_adapter):
                with patch.object(
                    retrieve_mod,
                    "_resolve_chunk_sessions",
                    side_effect=lambda ids, _brain: {
                        cid: chunk_sessions[cid]
                        for cid in ids
                        if cid in chunk_sessions
                    },
                ):
                    expanded, _ = retrieve_mod._expand_cross_event_bridges(
                        seeds, "brain", max_per_hub=1
                    )
            return retrieve_mod._diversify_facts(expanded, max_facts=5)

        a = run_with_tail("event-seed-90", "event-bridge-a", "session_9")
        b = run_with_tail("event-seed-91", "event-bridge-b", "session_13")
        sessions_a = {s for c in a for s in (c.get("session_ids") or [])}
        sessions_b = {s for c in b for s in (c.get("session_ids") or [])}
        self.assertEqual(sessions_a, sessions_b)
        self.assertIn("session_5", sessions_a)
        self.assertNotIn("session_9", sessions_a)
        self.assertNotIn("session_13", sessions_b)


if __name__ == "__main__":
    unittest.main()
