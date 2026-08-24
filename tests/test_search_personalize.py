import os
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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.constants.data import TextChunk
from src.constants.kg import Node, Predicate
from src.core.search.personalize import (
    blend_ranked,
    personalize_ranked_ids,
    query_personalize_lambda,
    score_nodes_for_user,
    user_pref_weights,
)
from src.services.api.constants.requests import SearchRequestBody


def _node(uuid: str, name: str, labels: list[str], happened_at: str | None = None) -> Node:
    return Node(
        uuid=uuid,
        name=name,
        labels=labels,
        happened_at=happened_at,
        properties={},
    )


def _pred(name: str, amount: float | None = None, weight: float | None = None) -> Predicate:
    props = {}
    if weight is not None:
        props["weight"] = weight
    return Predicate(
        uuid=f"rel:{name}:{id(name)}",
        name=name,
        description="",
        amount=amount,
        properties=props,
    )


class FakeGraph:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, list[tuple[Predicate, Node]]] = {}

    def add(self, node: Node) -> Node:
        self.nodes[node.uuid] = node
        self.edges.setdefault(node.uuid, [])
        return node

    def link(self, tail: str, pred: Predicate, tip: str) -> None:
        self.edges.setdefault(tail, []).append((pred, self.nodes[tip]))

    def get_by_uuids(self, uuids, brain_id="default"):
        return [self.nodes[uid] for uid in uuids if uid in self.nodes]

    def get_by_uuid(self, uuid, brain_id="default"):
        return self.nodes.get(uuid)

    def get_neighbors(self, nodes, brain_id="default", of_types=None, **kwargs):
        out = {}
        for item in nodes:
            uid = item if isinstance(item, str) else item.uuid
            neigh = list(self.edges.get(uid, []))
            if of_types:
                wanted = {label.upper() for label in of_types}
                neigh = [
                    (pred, node)
                    for pred, node in neigh
                    if wanted & {label.upper() for label in (node.labels or [])}
                ]
            out[uid] = neigh
        return out


def _style_graph() -> tuple[FakeGraph, Node]:
    graph = FakeGraph()
    user = graph.add(_node("user:u01", "u01", ["USER"]))
    recent = graph.add(
        _node("evt-recent", "View", ["EVENT"], happened_at="08/20/2026")
    )
    old = graph.add(_node("evt-old", "View", ["EVENT"], happened_at="08/21/2025"))
    seventies = graph.add(_node("70s-item", "lamp 70s", ["ENTITY"]))
    modern = graph.add(_node("modern-item", "lamp modern", ["ENTITY"]))
    attr_70s = graph.add(_node("hub:attr:70s", "70s", ["ATTR", "STYLE"]))
    attr_modern = graph.add(_node("hub:attr:modern", "modern", ["ATTR", "STYLE"]))
    graph.link("user:u01", _pred("MADE"), "evt-recent")
    graph.link("evt-recent", _pred("TARGETED"), "70s-item")
    graph.link("70s-item", _pred("HAS"), "hub:attr:70s")
    graph.link("user:u01", _pred("MADE"), "evt-old")
    graph.link("evt-old", _pred("TARGETED"), "modern-item")
    graph.link("modern-item", _pred("HAS"), "hub:attr:modern")
    graph.link(
        "user:u01",
        _pred("PREFERS", amount=0.4, weight=0.4),
        "hub:attr:70s",
    )
    return graph, user


class QueryLambdaTests(unittest.TestCase):
    def test_generic_one_token(self):
        self.assertAlmostEqual(query_personalize_lambda("lamp"), 0.85)

    def test_two_and_three_tokens(self):
        self.assertAlmostEqual(query_personalize_lambda("oak table"), 0.5)
        self.assertAlmostEqual(query_personalize_lambda("oak dining table"), 0.25)

    def test_four_plus_tokens(self):
        self.assertAlmostEqual(query_personalize_lambda("oak dining table set"), 0.1)

    def test_digits_are_specific(self):
        self.assertEqual(query_personalize_lambda("oak dining table 180cm"), 0.0)
        self.assertEqual(query_personalize_lambda("sku 7468"), 0.0)


class BlendTests(unittest.TestCase):
    def test_zero_lambda_preserves_order(self):
        ids = ["a", "b", "c"]
        ordered, _ = blend_ranked(ids, {"a": 3, "b": 2, "c": 1}, {"c": 9}, 0.0)
        self.assertEqual(ordered, ids)

    def test_all_zero_prefs_preserve_order(self):
        ids = ["a", "b", "c"]
        ordered, _ = blend_ranked(ids, {"a": 3, "b": 2, "c": 1}, {}, 0.9)
        self.assertEqual(ordered, ids)

    def test_high_lambda_promotes_overlap_without_dropping(self):
        ids = ["a", "b", "c"]
        ordered, _ = blend_ranked(
            ids, {"a": 3, "b": 2, "c": 1}, {"a": 0, "b": 0, "c": 10}, 1.0
        )
        self.assertEqual(ordered[0], "c")
        self.assertEqual(set(ordered), set(ids))


class PrefVectorTests(unittest.TestCase):
    def test_recent_70s_outranks_old_modern_and_combines_prefers(self):
        graph, user = _style_graph()
        with patch("src.core.search.personalize.graph_adapter", graph):
            weights = user_pref_weights(user, "brain")
        self.assertGreater(weights["hub:attr:70s"], weights["hub:attr:modern"])
        self.assertGreater(weights["hub:attr:70s"], 0.4)

    def test_missing_user_is_empty(self):
        graph = FakeGraph()
        with patch("src.core.search.personalize.graph_adapter", graph):
            self.assertEqual(user_pref_weights(None, "brain"), {})
            self.assertEqual(
                user_pref_weights(_node("missing", "x", ["USER"]), "brain"),
                {},
            )

    def test_score_nodes_overlap_only(self):
        graph, user = _style_graph()
        with patch("src.core.search.personalize.graph_adapter", graph):
            prefs = user_pref_weights(user, "brain")
            scores = score_nodes_for_user(
                ["70s-item", "modern-item", "ghost"], prefs, "brain"
            )
        self.assertGreater(scores["70s-item"], scores["modern-item"])
        self.assertEqual(scores["ghost"], 0.0)


class SearchPersonalizeTests(unittest.IsolatedAsyncioTestCase):
    async def _search(self, body: SearchRequestBody, chunks: list):
        from src.services.api.controllers import search as search_mod

        mock_data = MagicMock()
        mock_data.search_bm25.return_value = chunks
        mock_data.get_text_chunks_by_ids.return_value = (
            [item[0] for item in chunks],
            None,
        )
        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod.config, "search_literal_fill", False),
            patch.object(search_mod, "embeddings_adapter", MagicMock()),
            patch.object(search_mod, "vector_search", MagicMock()),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            return await search_mod.search(body)

    def test_schema_allows_target_not_user_id(self):
        body = SearchRequestBody(query="lamp", target="u01")
        self.assertEqual(body.target, "u01")
        self.assertEqual(body.channels, ["passages"])
        fields = set(SearchRequestBody.model_fields)
        self.assertIn("target", fields)
        for banned in ("product_id", "sku", "brand", "user_id"):
            self.assertNotIn(banned, fields)

    async def test_omitted_target_keeps_retrieve_order(self):
        chunks = [
            (TextChunk(id="first", text="DOCID 70s-item. lamp"), 9.0),
            (TextChunk(id="second", text="DOCID modern-item. lamp"), 8.0),
            (TextChunk(id="third", text="no join"), 7.0),
        ]
        response = await self._search(SearchRequestBody(query="lamp", k=3), chunks)
        self.assertEqual([hit.id for hit in response.hits], ["first", "second", "third"])
        self.assertTrue(all(hit.scores.personalize is None for hit in response.hits))

    async def test_target_promotes_70s_on_generic_query(self):
        graph, _ = _style_graph()
        chunks = [
            (TextChunk(id="first", text="DOCID modern-item. lamp"), 9.0),
            (TextChunk(id="second", text="DOCID 70s-item. lamp"), 8.0),
            (TextChunk(id="third", text="no join lamp"), 7.0),
        ]
        with patch("src.core.search.personalize.graph_adapter", graph):
            response = await self._search(
                SearchRequestBody(query="lamp", k=3, target="u01"),
                chunks,
            )
        ids = [hit.id for hit in response.hits]
        self.assertEqual(ids[0], "second")
        self.assertIn("third", ids)
        self.assertEqual(response.hits[0].node_id, "70s-item")
        self.assertIsNotNone(response.hits[0].scores.personalize)

    async def test_specific_query_does_not_reorder(self):
        graph, _ = _style_graph()
        chunks = [
            (TextChunk(id="first", text="DOCID modern-item. oak"), 9.0),
            (TextChunk(id="second", text="DOCID 70s-item. oak"), 8.0),
        ]
        with patch("src.core.search.personalize.graph_adapter", graph):
            response = await self._search(
                SearchRequestBody(
                    query="oak dining table 180cm", k=2, target="u01"
                ),
                chunks,
            )
        self.assertEqual([hit.id for hit in response.hits], ["first", "second"])

    async def test_extras_and_survives_personalize(self):
        graph, _ = _style_graph()
        chunks = [
            (
                TextChunk(
                    id="first",
                    text="DOCID modern-item. lamp",
                    metadata={"color": "navy"},
                ),
                9.0,
            ),
            (
                TextChunk(
                    id="second",
                    text="DOCID 70s-item. lamp",
                    metadata={"color": "olive"},
                ),
                8.0,
            ),
            (
                TextChunk(
                    id="third",
                    text="DOCID other-item. lamp",
                    metadata={"color": "navy"},
                ),
                7.0,
            ),
        ]
        with patch("src.core.search.personalize.graph_adapter", graph):
            response = await self._search(
                SearchRequestBody(
                    query="lamp",
                    k=3,
                    target="u01",
                    extras={"color": "navy"},
                ),
                chunks,
            )
        ids = [hit.id for hit in response.hits]
        self.assertNotIn("second", ids)
        self.assertEqual(set(ids), {"first", "third"})
        self.assertTrue(
            all((hit.extras or {}).get("color") == "navy" for hit in response.hits)
        )
        self.assertTrue(
            all((hit.extras or {}).get("style") != "70s" for hit in response.hits)
        )

    def test_personalize_never_filters_ids(self):
        graph, _ = _style_graph()
        with patch("src.core.search.personalize.graph_adapter", graph):
            ordered, _ = personalize_ranked_ids(
                query="lamp",
                ranked_ids=["a", "b"],
                retrieve_scores={"a": 2, "b": 1},
                node_id_by_hit={"a": "modern-item", "b": "70s-item"},
                target="u01",
                brain_id="brain",
            )
        self.assertEqual(set(ordered), {"a", "b"})
