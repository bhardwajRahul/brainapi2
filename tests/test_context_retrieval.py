import asyncio
import os
import sys
import types
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

from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.search.entity_info import EventSynergyRetriever, MatchPath
from src.services.api.constants.requests import GetContextRequestBody
from src.utils.nlp.ner import ExtractElementsResponse


def _node(uuid: str, name: str, *, happened_at: str | None = None, labels=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        happened_at=happened_at,
    )


def _pred(uuid: str, name: str):
    return Predicate(uuid=uuid, name=name, description=f"desc-{name}")


class CollectQueryVariantsTests(unittest.TestCase):
    def test_includes_noun_chunks_deduped_and_capped(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        elements = ExtractElementsResponse(
            tokens=[{"text": "Alice", "lemma": "Alice", "pos": "PROPN"}],
            noun_chunks=[
                "closest friend",
                "Alice",
                "charity event",
                "weekend trip",
                "old school",
                "extra chunk",
                "another one",
            ],
        )
        variants = retrieve_mod._collect_query_variants(
            "What did Alice do with her closest friend?", elements
        )
        self.assertEqual(variants[0], "What did Alice do with her closest friend?")
        self.assertIn("Alice", variants)
        self.assertIn("closest friend", variants)
        self.assertIn("charity event", variants)
        noun_only = [
            v
            for v in variants
            if v
            not in {
                "What did Alice do with her closest friend?",
                "Alice",
            }
        ]
        self.assertLessEqual(len(noun_only), retrieve_mod._MAX_NOUN_CHUNKS)


class FormatHelpersTests(unittest.TestCase):
    def test_format_includes_happened_at(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        n = _node("a", "Alice")
        r = _pred("r1", "ATTENDED")
        m = _node("e", "Wedding", happened_at="2024-05-01", labels=["EVENT"])
        r2 = _pred("r2", "INVOLVED")
        b = _node("b", "Bob")
        line = retrieve_mod._format_event_fact(n, r, m, r2, b)
        self.assertIn("@2024-05-01", line)
        self.assertIn("Wedding", line)


class GetContextRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_retains_all_event_facts_not_just_first(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        facts = [
            (
                _node("a", "Alice"),
                _pred("r1", "ATTENDED"),
                _node("e1", "Event1", happened_at="2024-01-01", labels=["EVENT"]),
                _pred("r2", "WITH"),
                _node("b", "Bob"),
            ),
            (
                _node("a", "Alice"),
                _pred("r3", "VISITED"),
                _node("e2", "Event2", happened_at="2024-02-01", labels=["EVENT"]),
                _pred("r4", "IN"),
                _node("c", "Paris", labels=["LOCATION"]),
            ),
            (
                _node("a", "Alice"),
                _pred("r5", "BOUGHT"),
                _node("e3", "Event3", labels=["EVENT"]),
                _pred("r6", "ITEM"),
                _node("d", "Book", labels=["OBJECT"]),
            ),
            (
                _node("a", "Alice"),
                _pred("r7", "MET"),
                _node("e4", "Event4", labels=["EVENT"]),
                _pred("r8", "WITH"),
                _node("f", "Carol"),
            ),
            (
                _node("a", "Alice"),
                _pred("r9", "MOVED"),
                _node("e5", "Event5", labels=["EVENT"]),
                _pred("r10", "TO"),
                _node("g", "London", labels=["LOCATION"]),
            ),
        ]

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )

        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.1,
            )
        ]
        mock_vs.search_relationships.return_value = []

        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = facts

        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []

        elements = ExtractElementsResponse(tokens=[], noun_chunks=[])

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
            patch.object(
                retrieve_mod,
                "EventSynergyRetriever",
                return_value=MagicMock(
                    retrieve_matches=MagicMock(
                        return_value=MatchPath(
                            target_node=None,
                            path=(_pred("", ""), None),
                            similarity=0.0,
                        )
                    )
                ),
            ),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(text="Alice", brain_id="brain-a", max_facts=40)
            )

        self.assertEqual(len(response.triples), 5)
        self.assertIn("@2024-01-01", response.text_context)
        self.assertIn("Paris", response.text_context)

    async def test_relationship_vector_search_seeds_facts(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        fact = (
            _node("tail", "Alice"),
            _pred("r1", "ORGANIZED"),
            _node("mid", "Fundraiser", happened_at="2023-11-11", labels=["EVENT"]),
            _pred("r2", "FOR"),
            _node("tip", "Charity", labels=["ORG"]),
        )

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )

        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []
        mock_vs.search_relationships.return_value = [
            Vector(
                id="rel1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "r1", "node_ids": ["tail", "mid"]},
                distance=0.05,
            )
        ]

        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [fact]

        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []

        elements = ExtractElementsResponse(tokens=[], noun_chunks=[])

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
            patch.object(
                retrieve_mod,
                "EventSynergyRetriever",
                return_value=MagicMock(
                    retrieve_matches=MagicMock(
                        return_value=MatchPath(
                            target_node=None,
                            path=(_pred("", ""), None),
                            similarity=0.0,
                        )
                    )
                ),
            ),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="who organized the fundraiser",
                    brain_id="brain-a",
                )
            )

        mock_graph.get_event_centric_neighbors.assert_called()
        seed_uuids = mock_graph.get_event_centric_neighbors.call_args[0][0]
        self.assertIn("tail", seed_uuids)
        self.assertIn("mid", seed_uuids)
        self.assertEqual(len(response.triples), 1)
        self.assertIn("Fundraiser", response.text_context)

    async def test_max_facts_cap_and_dedup(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        shared = (
            _node("a", "Alice"),
            _pred("r1", "ATTENDED"),
            _node("e1", "Event1", labels=["EVENT"]),
            _pred("r2", "WITH"),
            _node("b", "Bob"),
        )
        other = (
            _node("a", "Alice"),
            _pred("r3", "VISITED"),
            _node("e2", "Event2", labels=["EVENT"]),
            _pred("r4", "IN"),
            _node("c", "Paris", labels=["LOCATION"]),
        )

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.2,
            )
        ]
        mock_vs.search_relationships.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [shared, shared, other]
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        elements = ExtractElementsResponse(tokens=[], noun_chunks=[])

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
            patch.object(
                retrieve_mod,
                "EventSynergyRetriever",
                return_value=MagicMock(
                    retrieve_matches=MagicMock(
                        return_value=MatchPath(
                            target_node=None,
                            path=(_pred("", ""), None),
                            similarity=0.0,
                        )
                    )
                ),
            ),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(text="Alice", brain_id="brain-a", max_facts=1)
            )

        self.assertEqual(len(response.triples), 1)

    async def test_dossier_lines_appended_to_text_context(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        fact = (
            _node("a", "Alice"),
            _pred("r1", "KNOWS"),
            _node("e1", "Meeting", labels=["EVENT"]),
            _pred("r2", "WITH"),
            _node("b", "Bob"),
        )
        dossier = MatchPath(
            target_node=_node("a", "Alice"),
            path=(_pred("d1", "FRIEND_OF"), _node("b", "Bob")),
            similarity=0.9,
            children=[
                MatchPath(
                    target_node=_node("a", "Alice"),
                    path=(
                        _pred("d2", "LIVES_IN"),
                        _node("c", "Chicago", labels=["LOCATION"]),
                    ),
                    similarity=0.8,
                )
            ],
        )

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.1,
            )
        ]
        mock_vs.search_relationships.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [fact]
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        elements = ExtractElementsResponse(
            tokens=[{"text": "Alice", "lemma": "Alice", "pos": "PROPN"}],
            noun_chunks=[],
        )
        retriever = MagicMock()
        retriever.retrieve_matches.return_value = dossier

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
            patch.object(retrieve_mod, "EventSynergyRetriever", return_value=retriever),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Where does Alice's friend live?",
                    brain_id="brain-a",
                )
            )

        self.assertIn("[dossier:Alice]", response.text_context)
        self.assertIn("Chicago", response.text_context)
        self.assertEqual(len(response.triples), 1)


class EventSynergyHardeningTests(unittest.TestCase):
    def test_explores_top_k_branches_not_only_greedy_best(self):
        alice = _node("a", "Alice")
        wrong = _node("w", "Wrong")
        mid = _node("m", "Bridge")
        goal = _node("g", "Goal")
        rel_wrong = _pred("rw", "DISTRACTOR")
        rel_wrong.properties = {"v_id": "v_wrong"}
        rel_bridge = _pred("rb", "VIA")
        rel_bridge.properties = {"v_id": "v_bridge"}
        rel_goal = _pred("rg", "REACHES")
        rel_goal.properties = {"v_id": "v_goal"}

        neighbors = {
            "a": [(rel_wrong, wrong), (rel_bridge, mid)],
            "m": [(rel_goal, goal)],
            "w": [],
            "g": [],
        }

        embeddings = {
            "v_wrong": [1.0, 0.0, 0.0],
            "v_bridge": [0.8, 0.2, 0.0],
            "v_goal": [0.0, 1.0, 0.0],
        }
        query_embedding = [0.0, 1.0, 0.0]

        def fake_get_by_ids(ids, store="relationships", brain_id="default"):
            vid = ids[0]
            return [
                Vector(id=vid, embeddings=embeddings[vid], metadata={})
            ]

        with (
            patch(
                "src.core.search.entity_info.graph_adapter.get_neighbors",
                side_effect=lambda nodes, brain_id="default", **kwargs: {
                    nodes[0]: neighbors.get(nodes[0], [])
                },
            ),
            patch(
                "src.core.search.entity_info.vector_store_adapter.get_by_ids",
                side_effect=fake_get_by_ids,
            ),
            patch(
                "src.core.search.entity_info.cosine_similarity",
                side_effect=lambda a, b: sum(x * y for x, y in zip(a, b)),
            ),
        ):
            retriever = EventSynergyRetriever("brain-a")
            path = retriever._recursive_explorer(
                "a",
                query_embedding,
                depth=3,
                visited_ids=set(),
                work_counter=[0],
                branch_factor=3,
            )

        self.assertGreaterEqual(len(path), 2)
        hop_names = [hop[0][0].name for hop in path]
        self.assertIn("VIA", hop_names)
        self.assertIn("REACHES", hop_names)

    def test_work_cap_stops_exploration(self):
        nodes = {f"n{i}": _node(f"n{i}", f"N{i}") for i in range(20)}
        rels = {}
        for i in range(19):
            pred = _pred(f"r{i}", f"REL{i}")
            pred.properties = {"v_id": f"v{i}"}
            rels[f"n{i}"] = [(pred, nodes[f"n{i+1}"])]
        rels["n19"] = []

        with (
            patch(
                "src.core.search.entity_info.graph_adapter.get_neighbors",
                side_effect=lambda nodes_arg, brain_id="default", **kwargs: {
                    nodes_arg[0]: rels.get(nodes_arg[0], [])
                },
            ),
            patch(
                "src.core.search.entity_info.vector_store_adapter.get_by_ids",
                side_effect=lambda ids, store="relationships", brain_id="default": [
                    Vector(id=ids[0], embeddings=[1.0, 0.0], metadata={})
                ],
            ),
            patch(
                "src.core.search.entity_info.cosine_similarity",
                return_value=0.5,
            ),
            patch("src.core.search.entity_info._MAX_EXPLORATION_WORK", 3),
        ):
            retriever = EventSynergyRetriever("brain-a")
            work = [0]
            retriever._recursive_explorer(
                "n0",
                [1.0, 0.0],
                depth=20,
                visited_ids=set(),
                work_counter=work,
                branch_factor=1,
            )
            self.assertLessEqual(work[0], 3)


if __name__ == "__main__":
    unittest.main()
