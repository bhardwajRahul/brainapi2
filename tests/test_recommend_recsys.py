import os
import sys
import unittest
from datetime import datetime, timedelta
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
from src.core.search.recommend import (
    DEFAULT_BEHAVIOR_WEIGHTS,
    EntityRecommendRetriever,
    behavior_weight,
    degree_dampen,
    recency_decay,
    attr_idf,
)


def _node(uuid: str, name: str, labels: list[str], happened_at=None) -> Node:
    return Node.model_construct(
        uuid=uuid,
        name=name,
        labels=labels,
        happened_at=happened_at,
        properties={},
    )


def _pred(name: str = "MADE", weight=None) -> Predicate:
    props = {"weight": weight} if weight is not None else {}
    return Predicate.model_construct(
        uuid=f"pred-{name}-{weight}",
        name=name,
        description=name,
        properties=props,
        deprecated=False,
    )


class BehaviorWeightTests(unittest.TestCase):
    def test_purchase_higher_than_view(self):
        self.assertEqual(behavior_weight("Purchase"), 1.0)
        self.assertEqual(behavior_weight("View"), 0.2)
        self.assertEqual(behavior_weight("AddToCart"), 0.5)

    def test_unknown_defaults_to_one(self):
        self.assertEqual(behavior_weight("WeirdEvent"), 1.0)

    def test_custom_table(self):
        self.assertEqual(behavior_weight("View", {"view": 0.9}), 0.9)


class RecencyDecayTests(unittest.TestCase):
    def test_off_when_half_life_none(self):
        self.assertEqual(recency_decay("01/01/2020", None), 1.0)

    def test_older_decays_more(self):
        now = datetime(2026, 8, 6)
        older = recency_decay("01/01/2020", 30.0, now=now)
        newer = recency_decay("01/08/2026", 30.0, now=now)
        self.assertLess(older, newer)
        self.assertLessEqual(newer, 1.0)

    def test_half_life_halves(self):
        now = datetime(2026, 8, 6)
        past = (now - timedelta(days=30)).strftime("%d/%m/%Y")
        score = recency_decay(past, 30.0, now=now)
        self.assertAlmostEqual(score, 0.5, places=2)


class DegreeDampenTests(unittest.TestCase):
    def test_disabled(self):
        self.assertEqual(degree_dampen(100, False), 1.0)

    def test_hub_lower_than_rare(self):
        self.assertLess(degree_dampen(100, True), degree_dampen(1, True))

    def test_attr_idf_monotonic(self):
        self.assertLess(attr_idf(50), attr_idf(2))


class RecommendExcludeSeenTests(unittest.TestCase):
    @patch("src.core.search.recommend.graph_adapter")
    @patch("src.core.search.recommend.EntitySinergyRetriever")
    def test_exclude_seen_keeps_collaborative_unseen(self, mock_syn_cls, mock_graph):
        user = _node("user:u1", "u1", ["USER"])
        peer = _node("user:u2", "u2", ["USER"])
        shared = _node("item:a", "a", ["PRODUCT"])
        fresh = _node("item:b", "b", ["PRODUCT"])
        evt_u1 = _node("evt:1", "Purchase", ["EVENT"], happened_at="01/08/2026")
        evt_u2_shared = _node("evt:2", "Purchase", ["EVENT"], happened_at="01/08/2026")
        evt_u2_fresh = _node("evt:3", "Purchase", ["EVENT"], happened_at="01/08/2026")

        mock_syn = MagicMock()
        mock_syn.retrieve_sibilings.return_value = (user, [], [], [])
        mock_syn_cls.return_value = mock_syn

        def get_neighbors(uuids, of_types=None, brain_id="default"):
            uid = uuids[0]
            if uid == user.uuid:
                return {user.uuid: [(_pred("MADE"), evt_u1)]}
            if uid == evt_u1.uuid:
                return {
                    evt_u1.uuid: [
                        (_pred("TARGETED"), shared),
                        (_pred("MADE"), user),
                    ]
                }
            if uid == shared.uuid:
                return {
                    shared.uuid: [
                        (_pred("TARGETED"), evt_u1),
                        (_pred("TARGETED"), evt_u2_shared),
                    ]
                }
            if uid == evt_u2_shared.uuid:
                return {
                    evt_u2_shared.uuid: [
                        (_pred("MADE"), peer),
                        (_pred("TARGETED"), shared),
                    ]
                }
            if uid == peer.uuid:
                return {
                    peer.uuid: [
                        (_pred("MADE"), evt_u2_shared),
                        (_pred("MADE"), evt_u2_fresh),
                    ]
                }
            if uid == evt_u2_fresh.uuid:
                return {
                    evt_u2_fresh.uuid: [
                        (_pred("MADE"), peer),
                        (_pred("TARGETED"), fresh),
                    ]
                }
            return {uid: []}

        mock_graph.get_neighbors.side_effect = get_neighbors

        retriever = EntityRecommendRetriever("demorecsys")
        _, recs = retriever.recommend(
            "u1",
            include_asymmetric=True,
            include_multi_interest=False,
            include_attribute_pref=False,
            exclude_seen=True,
            diversify=False,
            labels=["PRODUCT"],
            top_k=10,
        )
        uuids = {r["node"].uuid for r in recs}
        self.assertNotIn(shared.uuid, uuids)
        self.assertIn(fresh.uuid, uuids)

    @patch("src.core.search.recommend.graph_adapter")
    @patch("src.core.search.recommend.EntitySinergyRetriever")
    def test_behavior_weights_rank_purchase_above_view(
        self, mock_syn_cls, mock_graph
    ):
        user = _node("user:u1", "u1", ["USER"])
        p_view = _node("item:v", "v", ["PRODUCT"])
        p_buy = _node("item:b", "b", ["PRODUCT"])
        evt_view = _node("evt:v", "View", ["EVENT"], happened_at="01/08/2026")
        evt_buy = _node("evt:b", "Purchase", ["EVENT"], happened_at="01/08/2026")

        mock_syn = MagicMock()
        mock_syn.retrieve_sibilings.return_value = (user, [], [], [])
        mock_syn_cls.return_value = mock_syn

        def get_neighbors(uuids, of_types=None, brain_id="default"):
            uid = uuids[0]
            if uid == user.uuid:
                return {
                    user.uuid: [
                        (_pred("MADE"), evt_view),
                        (_pred("MADE"), evt_buy),
                    ]
                }
            if uid == evt_view.uuid:
                return {evt_view.uuid: [(_pred("TARGETED"), p_view)]}
            if uid == evt_buy.uuid:
                return {evt_buy.uuid: [(_pred("TARGETED"), p_buy)]}
            return {uid: []}

        mock_graph.get_neighbors.side_effect = get_neighbors

        retriever = EntityRecommendRetriever("demorecsys")
        _, recs = retriever.recommend(
            "u1",
            include_asymmetric=True,
            include_multi_interest=False,
            include_attribute_pref=False,
            exclude_seen=False,
            diversify=False,
            labels=["PRODUCT"],
            top_k=10,
            behavior_weights=DEFAULT_BEHAVIOR_WEIGHTS,
        )
        self.assertGreaterEqual(len(recs), 2)
        by_uuid = {r["node"].uuid: r["score"] for r in recs}
        self.assertGreater(by_uuid[p_buy.uuid], by_uuid[p_view.uuid])


if __name__ == "__main__":
    unittest.main()
