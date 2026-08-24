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

from fastapi import HTTPException

from src.constants.embeddings import Vector
from src.constants.kg import Node, SearchEntitiesResult
from src.core.search.graph_channels import (
    DEFAULT_COMMUNITY_LABELS,
    collect_community_hits,
    collect_entity_hits,
    collect_event_hits,
    expand_neighbor_hits,
)
from src.core.search.hybrid import fuse_passage_lists
from src.services.api.constants.requests import SearchHit, SearchRequestBody, SearchResponse


def _node(
    uuid: str,
    name: str,
    labels: list[str],
    happened_at: str | None = None,
    description: str | None = None,
    properties: dict | None = None,
) -> Node:
    props = dict(properties or {})
    if happened_at:
        props["happened_at"] = happened_at
    return Node(
        uuid=uuid,
        name=name,
        labels=labels,
        happened_at=happened_at,
        description=description,
        properties=props,
    )


class FakeGraph:
    def __init__(self, nodes: list[Node], neighbors: dict[str, list[Node]] | None = None):
        self.nodes = {node.uuid: node for node in nodes}
        self.neighbors = neighbors or {}
        self.neighbor_calls = 0

    def search_entities(
        self,
        brain_id: str = "default",
        limit: int = 10,
        skip: int = 0,
        node_labels: list[str] | None = None,
        query_text: str | None = None,
    ) -> SearchEntitiesResult:
        matched: list[Node] = []
        want = {str(item).upper() for item in (node_labels or [])}
        needle = (query_text or "").lower()
        for node in self.nodes.values():
            have = {str(item).upper() for item in node.labels}
            if want and not (have & want):
                continue
            if needle and needle not in node.name.lower():
                continue
            matched.append(node)
        page = matched[skip : skip + limit]
        return SearchEntitiesResult(results=page, total=len(matched))

    def search_nodes_bm25(
        self,
        query_text: str,
        brain_id: str = "default",
        limit: int = 10,
        node_labels: list[str] | None = None,
        node_uuids: list[str] | None = None,
    ):
        needle = (query_text or "").lower()
        want = {str(item).upper() for item in (node_labels or [])}
        allowed = {str(item) for item in (node_uuids or [])} or None
        scored: list[tuple[Node, float]] = []
        for node in self.nodes.values():
            if allowed is not None and node.uuid not in allowed:
                continue
            have = {str(item).upper() for item in node.labels}
            if want and not (have & want):
                continue
            props = node.properties or {}
            text = " ".join(
                [
                    node.name or "",
                    node.description or "",
                    str(props.get("search_text") or ""),
                ]
            ).lower()
            score = 0.0
            if needle and needle in text:
                score += 5.0
            for token in needle.split():
                if token and token in text:
                    score += 1.0
            if score:
                scored.append((node, score))
        scored.sort(key=lambda item: (-item[1], item[0].uuid))
        return scored[:limit]

    def get_by_uuids(self, uuids: list[str], brain_id: str = "default") -> list[Node]:
        return [self.nodes[item] for item in uuids if item in self.nodes]

    def get_by_uuid(self, uuid: str, brain_id: str = "default") -> Node:
        return self.nodes[uuid]

    def get_neighbors(self, nodes, brain_id: str = "default", **kwargs):
        self.neighbor_calls += 1
        keys = []
        for item in nodes:
            keys.append(item if isinstance(item, str) else item.uuid)
        out = {}
        for key in keys:
            out[key] = [(None, node) for node in self.neighbors.get(key, [])]
        return out

    def get_event_centric_neighbors(self, nodes, brain_id: str = "default"):
        return []


class SearchGraphChannelTests(unittest.IsolatedAsyncioTestCase):
    def test_contract_has_graph_fields_not_catalog_enums(self):
        body = SearchRequestBody(query="sofa")
        self.assertEqual(body.channels, ["passages"])
        self.assertEqual(body.expand, "none")
        self.assertEqual(body.mode, "default")
        self.assertIsNone(body.node_labels)
        fields = set(SearchRequestBody.model_fields)
        hit_fields = set(SearchHit.model_fields)
        for banned in (
            "product_id",
            "product_features",
            "sku",
            "category",
            "brand",
            "user_id",
        ):
            self.assertNotIn(banned, fields)
            self.assertNotIn(banned, hit_fields)
        self.assertIn("labels", hit_fields)
        self.assertIn("extras", hit_fields)
        self.assertIn("node_id", hit_fields)
        self.assertIn("node_ids", set(SearchResponse.model_fields))
        self.assertIn("extras", fields)
        self.assertEqual(list(DEFAULT_COMMUNITY_LABELS), ["TYPE", "CLASS", "TOPIC"])

    async def test_unknown_channel_is_400(self):
        from src.services.api.controllers import search as search_mod

        with patch.object(search_mod.config, "search_enabled", True):
            with self.assertRaises(HTTPException) as ctx:
                await search_mod.search(
                    SearchRequestBody(query="sofa", channels=["foo"])
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Unknown search channel", str(ctx.exception.detail))

    async def test_entities_empty_graph_is_200(self):
        from src.services.api.controllers import search as search_mod

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []
        graph = FakeGraph([])

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "graph_adapter", graph),
            patch.object(search_mod, "data_adapter", MagicMock()),
        ):
            response = await search_mod.search(
                SearchRequestBody(query="sofa", channels=["entities"], k=5)
            )

        self.assertEqual(response.hits, [])

    async def test_seeded_entity_name_in_top_k(self):
        from src.services.api.controllers import search as search_mod

        graph = FakeGraph([_node("sku-1", "navy velvet sofa", ["ENTITY"])])
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.2], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "graph_adapter", graph),
            patch.object(search_mod, "data_adapter", MagicMock()),
        ):
            response = await search_mod.search(
                SearchRequestBody(query="velvet sofa", channels=["entities"], k=5)
            )

        self.assertEqual([hit.id for hit in response.hits], ["sku-1"])
        self.assertEqual(response.hits[0].channel, "entities")
        self.assertIn("ENTITY", response.hits[0].labels)
        self.assertEqual(response.hits[0].node_id, "sku-1")
        self.assertEqual(response.node_ids, ["sku-1"])

    async def test_node_labels_mismatch_is_empty_not_400(self):
        from src.services.api.controllers import search as search_mod

        graph = FakeGraph([_node("sku-1", "navy velvet sofa", ["ENTITY"])])
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.2], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "graph_adapter", graph),
            patch.object(search_mod, "data_adapter", MagicMock()),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="velvet sofa",
                    channels=["entities"],
                    node_labels=["EVENT"],
                    k=5,
                )
            )

        self.assertEqual(response.hits, [])

    async def test_default_passages_does_not_touch_graph(self):
        from src.services.api.controllers import search as search_mod
        from src.constants.data import TextChunk

        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id="lex", text="license board"), 3.1)
        ]
        mock_graph = MagicMock()
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "graph_adapter", mock_graph),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(SearchRequestBody(query="license", k=5))

        self.assertEqual([hit.id for hit in response.hits], ["lex"])
        self.assertEqual(response.hits[0].channel, "passages")
        mock_graph.search_entities.assert_not_called()
        mock_embeddings.embed_text.assert_not_called()

    async def test_events_empty_ok(self):
        from src.services.api.controllers import search as search_mod

        graph = FakeGraph([_node("sku-1", "navy velvet sofa", ["ENTITY"])])
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.2], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "graph_adapter", graph),
            patch.object(search_mod, "data_adapter", MagicMock()),
        ):
            response = await search_mod.search(
                SearchRequestBody(query="view", channels=["events"], k=5)
            )

        self.assertEqual(response.hits, [])

    async def test_events_with_happened_at(self):
        from src.services.api.controllers import search as search_mod

        graph = FakeGraph(
            [
                _node(
                    "evt-1",
                    "View",
                    ["EVENT"],
                    happened_at="2024-01-05T12:00:00Z",
                )
            ]
        )
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.2], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "graph_adapter", graph),
            patch.object(search_mod, "data_adapter", MagicMock()),
        ):
            response = await search_mod.search(
                SearchRequestBody(query="View", channels=["events"], k=5)
            )

        self.assertEqual([hit.id for hit in response.hits], ["evt-1"])
        self.assertEqual(response.hits[0].channel, "events")
        self.assertEqual(response.hits[0].extras["happened_at"], "2024-01-05T12:00:00Z")

    def test_community_hub_returns_members_with_fanout_cap(self):
        hub = _node("class-sofas", "sofas", ["CLASS"])
        members = [
            _node("e1", "velvet sofa", ["ENTITY"]),
            _node("e2", "leather sofa", ["ENTITY"]),
            _node("e3", "loveseat", ["ENTITY"]),
        ]
        unrelated = _node("e4", "lamp", ["ENTITY"])
        graph = FakeGraph(
            [hub, *members, unrelated],
            neighbors={"class-sofas": members},
        )
        hits = collect_community_hits(
            query="sofas",
            brain_id="searchbenchsmoke",
            k=10,
            graph=graph,
            fanout=2,
        )
        ids = [hit.id for hit in hits]
        self.assertTrue(set(ids).issubset({"e1", "e2", "e3"}))
        self.assertNotIn("e4", ids)
        self.assertNotIn("class-sofas", ids)
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(hit.channel == "communities" for hit in hits))

    def test_expand_neighbors_default_off(self):
        seed = _node("sku-1", "velvet sofa", ["ENTITY"])
        neighbor = _node("class-sofas", "sofas", ["CLASS"])
        graph = FakeGraph([seed, neighbor], neighbors={"sku-1": [neighbor]})
        seeds = collect_entity_hits(
            query="velvet sofa",
            brain_id="searchbenchsmoke",
            k=5,
            graph=graph,
        )
        self.assertEqual(graph.neighbor_calls, 0)
        expanded = expand_neighbor_hits(
            seeds,
            brain_id="searchbenchsmoke",
            k=5,
            graph=graph,
            fanout=10,
        )
        self.assertEqual([hit.id for hit in expanded], ["class-sofas"])
        self.assertEqual(expanded[0].channel, "neighbors")

    async def test_expand_omitted_skips_hops(self):
        from src.services.api.controllers import search as search_mod

        seed = _node("sku-1", "velvet sofa", ["ENTITY"])
        neighbor = _node("n1", "pillow", ["ENTITY"])
        graph = FakeGraph([seed, neighbor], neighbors={"sku-1": [neighbor]})
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.2], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "graph_adapter", graph),
            patch.object(search_mod, "data_adapter", MagicMock()),
        ):
            response = await search_mod.search(
                SearchRequestBody(query="velvet sofa", channels=["entities"], k=5)
            )

        self.assertEqual([hit.id for hit in response.hits], ["sku-1"])
        self.assertEqual(graph.neighbor_calls, 0)

    async def test_expand_neighbors_adds_hop(self):
        from src.services.api.controllers import search as search_mod

        seed = _node("sku-1", "velvet sofa", ["ENTITY"])
        neighbor = _node("n1", "pillow", ["ENTITY"])
        graph = FakeGraph([seed, neighbor], neighbors={"sku-1": [neighbor]})
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.2], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "graph_adapter", graph),
            patch.object(search_mod, "data_adapter", MagicMock()),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="velvet sofa",
                    channels=["entities"],
                    expand="neighbors",
                    k=5,
                )
            )

        ids = [hit.id for hit in response.hits]
        self.assertIn("sku-1", ids)
        self.assertIn("n1", ids)
        self.assertTrue(any(hit.channel == "neighbors" for hit in response.hits))

    def test_fuse_mixed_chunk_and_node_ids(self):
        fused = fuse_passage_lists(
            ["chunk-1"],
            [],
            extra_id_lists=[["sku-1"]],
        )
        ids = [item for item, _ in fused]
        self.assertEqual(set(ids), {"chunk-1", "sku-1"})

    async def test_fused_graph_lists_drop_hub_ids(self):
        from src.constants.data import TextChunk
        from src.services.api.controllers import search as search_mod

        hub = _node("hub:attr:red", "red", ["ATTR"])
        product = _node("sku-1", "red velvet sofa", ["ENTITY"])
        graph = FakeGraph([hub, product], neighbors={"hub:attr:red": [product]})
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id="chunk-1", text="red velvet sofa"), 2.0)
        ]
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.2], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "graph_adapter", graph),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="red sofa",
                    channels=["passages", "entities", "communities"],
                    k=10,
                )
            )

        ids = [hit.id for hit in response.hits]
        self.assertNotIn("hub:attr:red", ids)
        self.assertIn("sku-1", ids)
        self.assertIn("chunk-1", ids)
        self.assertEqual(response.channel_lists["bm25"], ["chunk-1"])
        self.assertIn("sku-1", response.channel_lists["entities"])
        self.assertNotIn("hub:attr:red", response.channel_lists["entities"])
        self.assertNotIn("hub:attr:red", response.channel_lists["communities"])

    def test_entities_drop_attr_hubs(self):
        hub = _node("hub:attr:red", "red", ["ATTR"])
        product = _node("sku-1", "red velvet sofa", ["ENTITY"])
        graph = FakeGraph([hub, product])
        hits = collect_entity_hits(
            query="red sofa",
            brain_id="searchbenchsmoke",
            k=5,
            graph=graph,
        )
        ids = [hit.id for hit in hits]
        self.assertEqual(ids, ["sku-1"])
        self.assertTrue(all(hit.channel == "entities" for hit in hits))

    def test_community_skips_attr_member_hubs(self):
        hub = _node("class-sofas", "sofas", ["CLASS"])
        product = _node("e1", "velvet sofa", ["ENTITY"])
        attr = _node("hub:attr:red", "red", ["ATTR"])
        graph = FakeGraph(
            [hub, product, attr],
            neighbors={"class-sofas": [product, attr]},
        )
        hits = collect_community_hits(
            query="sofas",
            brain_id="searchbenchsmoke",
            k=10,
            graph=graph,
            fanout=5,
        )
        ids = [hit.id for hit in hits]
        self.assertEqual(ids, ["e1"])
        self.assertNotIn("hub:attr:red", ids)

    def test_community_attr_hub_returns_products(self):
        hub = _node("hub:attr:red", "red", ["ATTR"])
        product = _node("e1", "red sofa", ["ENTITY"])
        graph = FakeGraph([hub, product], neighbors={"hub:attr:red": [product]})
        hits = collect_community_hits(
            query="red sofa",
            brain_id="searchbenchsmoke",
            k=10,
            graph=graph,
            fanout=5,
        )
        self.assertEqual([hit.id for hit in hits], ["e1"])
        self.assertEqual(hits[0].channel, "communities")

    def test_search_controller_does_not_import_retrieve_assembler(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        search_text = (
            root / "src" / "services" / "api" / "controllers" / "search.py"
        ).read_text(encoding="utf-8")
        retrieve_text = (
            root / "src" / "services" / "api" / "controllers" / "retrieve.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("controllers.retrieve", search_text)
        self.assertNotIn("kg_topic_sessions", search_text)
        self.assertNotIn("graph_channels", retrieve_text)
        schema = (
            root / "src" / "services" / "api" / "constants" / "requests.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("product_id", schema.split("class SearchRequestBody")[1].split("class SearchHitScores")[0])


class SearchGraphHelperTests(unittest.TestCase):
    def test_entity_ann_and_name_match(self):
        named = _node("sku-1", "navy velvet sofa", ["ENTITY"])
        vector_only = _node("sku-2", "other item", ["ENTITY"])
        graph = FakeGraph([named, vector_only])
        vs = MagicMock()
        vs.search_nodes.return_value = [
            Vector(id="v2", metadata={"uuid": "sku-2", "name": "other item"}, distance=0.1)
        ]
        hits = collect_entity_hits(
            query="velvet sofa",
            brain_id="searchbenchsmoke",
            k=5,
            graph=graph,
            vector_search=vs,
            query_vector=[0.1, 0.2],
        )
        ids = [hit.id for hit in hits]
        self.assertIn("sku-1", ids)
        self.assertIn("sku-2", ids)

    def test_events_rank_recent_higher(self):
        older = _node("evt-old", "View", ["EVENT"], happened_at="2020-01-01T00:00:00Z")
        newer = _node("evt-new", "View", ["EVENT"], happened_at="2026-08-01T00:00:00Z")
        graph = FakeGraph([older, newer])
        hits = collect_event_hits(
            query="View",
            brain_id="searchbenchsmoke",
            k=5,
            graph=graph,
        )
        self.assertEqual(hits[0].id, "evt-new")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_entity_bm25_matches_description_only(self):
        named = _node("sku-1", "catalog item", ["ENTITY"])
        described = _node(
            "sku-2",
            "item two",
            ["ENTITY"],
            description="navy velvet ottoman for small rooms",
            properties={"search_text": "item two navy velvet ottoman for small rooms"},
        )
        graph = FakeGraph([named, described])
        hits = collect_entity_hits(
            query="ottoman",
            brain_id="searchbenchsmoke",
            k=5,
            graph=graph,
        )
        ids = [hit.id for hit in hits]
        self.assertIn("sku-2", ids)
        self.assertNotIn("sku-1", ids)

    def test_community_intersects_class_and_attr_hubs(self):
        class_hub = _node("hub:class:sofas", "sofas", ["CLASS"])
        attr_hub = _node("hub:attr:modern", "modern", ["ATTR"])
        both = _node("e1", "velvet sofa", ["ENTITY"])
        class_only = _node("e2", "leather sofa", ["ENTITY"])
        graph = FakeGraph(
            [class_hub, attr_hub, both, class_only],
            neighbors={
                "hub:class:sofas": [both, class_only],
                "hub:attr:modern": [both],
            },
        )
        hits = collect_community_hits(
            query="modern sofas",
            brain_id="searchbenchsmoke",
            k=10,
            graph=graph,
            fanout=5,
        )
        ids = [hit.id for hit in hits]
        self.assertEqual(ids, ["e1"])
        self.assertNotIn("e2", ids)
        self.assertNotIn("hub:class:sofas", ids)

    def test_community_union_when_single_hub_kind(self):
        class_hub = _node("hub:class:sofas", "sofas", ["CLASS"])
        a = _node("e1", "velvet sofa", ["ENTITY"])
        b = _node("e2", "leather sofa", ["ENTITY"])
        graph = FakeGraph(
            [class_hub, a, b],
            neighbors={"hub:class:sofas": [a, b]},
        )
        hits = collect_community_hits(
            query="sofas",
            brain_id="searchbenchsmoke",
            k=10,
            graph=graph,
            fanout=5,
        )
        self.assertEqual(set(hit.id for hit in hits), {"e1", "e2"})

    def test_community_hybrid_ranks_by_node_bm25_not_degree(self):
        class_hub = _node("hub:class:sofas", "sofas", ["CLASS"])
        attr_hub = _node("hub:attr:modern", "modern", ["ATTR"])
        weak = _node("e1", "sofa", ["ENTITY"], description="basic")
        strong = _node(
            "e2",
            "plush seating",
            ["ENTITY"],
            description="modern sofas extra plush",
            properties={"search_text": "plush seating modern sofas extra plush"},
        )
        graph = FakeGraph(
            [class_hub, attr_hub, weak, strong],
            neighbors={
                "hub:class:sofas": [weak, strong],
                "hub:attr:modern": [weak, strong],
            },
        )
        hits = collect_community_hits(
            query="modern sofas",
            brain_id="searchbenchsmoke",
            k=10,
            graph=graph,
            fanout=5,
        )
        self.assertEqual([hit.id for hit in hits], ["e2", "e1"])


if __name__ == "__main__":
    unittest.main()
